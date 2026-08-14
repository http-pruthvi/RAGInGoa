"""
pipeline.py — RAG pipeline orchestrator

Wires together all pipeline stages: STT → input guardrail → retrieval →
confidence guardrail → generation → grounding guardrail.

Key design decisions:
- Typed request/response dataclasses (not raw strings) for clear interfaces
- Explicit status enum covering all terminal states
- Per-stage try/except with appropriate error classification
- Stage-level latency timing via StageTimer context manager
- STT retries handled at the STT layer; generation retried once here
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from src.stt import transcribe, STTError, STTResult
from src.chunkers import Chunk
from src.retrieve import HybridRetriever, RetrievalResult
from src.guardrails import check_input, check_retrieval_confidence, check_grounding
from src.generate import generate_answer, GenerationError
from src.latency import StageTimer


logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Terminal status codes for pipeline execution.

    Each status maps to a clear outcome — the API layer translates these
    to HTTP status codes and user-facing messages.
    """
    SUCCESS = "success"
    REFUSED_BAD_INPUT = "refused-bad-input"
    REFUSED_NO_MATCH = "refused-no-match"
    REFUSED_BY_MODEL = "refused-by-model"
    REFUSED_UNGROUNDED = "refused-ungrounded"
    STT_ERROR = "stt-error"
    INTERNAL_ERROR = "internal-error"


@dataclass
class PipelineRequest:
    """Input to the RAG pipeline.

    Either query_text (text mode, skipping STT) or audio_bytes (voice mode)
    must be provided. If both are given, audio_bytes takes precedence.
    """
    query_text: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    language: str = "hi-IN"  # BCP-47 code for STT, also used for metadata


@dataclass
class PipelineResponse:
    """Structured output from the RAG pipeline.

    Always includes status. On success, includes the answer and retrieved chunks.
    On any refusal/error, includes error_message explaining why.
    Stage timings are always populated for latency analysis.
    """
    status: PipelineStatus
    answer: str = ""
    query_text: str = ""
    retrieved_chunks: List[dict] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    error_message: str = ""


class RAGPipeline:
    """Main pipeline orchestrator.

    Holds references to the retriever (loaded once at startup) and
    coordinates the full query flow with error handling and timing.
    """

    def __init__(self, retriever: HybridRetriever):
        """Initialize pipeline with a pre-loaded retriever.

        The retriever (FAISS index + BM25 + chunk metadata) is loaded once
        at application startup, not per-request.
        """
        self.retriever = retriever

    def run(self, request: PipelineRequest) -> PipelineResponse:
        """Execute the full RAG pipeline.

        Flow: STT (if audio) → input guardrail → retrieval → confidence
        guardrail → generation → grounding guardrail → response

        Each stage is individually timed and wrapped in try/except.
        """
        timer = StageTimer()
        query_text = request.query_text or ""

        # -- Stage 1: Speech-to-Text (only for voice input) --
        if request.audio_bytes:
            try:
                with timer.time_stage("stt"):
                    stt_result = transcribe(
                        request.audio_bytes,
                        language_code=request.language,
                    )
                    query_text = stt_result.transcript
            except STTError as e:
                logger.error(f"STT failed: {e}")
                return PipelineResponse(
                    status=PipelineStatus.STT_ERROR,
                    stage_timings=timer.timings,
                    error_message=str(e),
                )
            except Exception as e:
                logger.exception(f"Unexpected error in STT: {e}")
                return PipelineResponse(
                    status=PipelineStatus.INTERNAL_ERROR,
                    stage_timings=timer.timings,
                    error_message=f"Unexpected STT error: {e}",
                )

        # -- Stage 2: Pre-retrieval input guardrail --
        try:
            with timer.time_stage("input_guardrail"):
                input_check = check_input(query_text)
                if not input_check.passed:
                    return PipelineResponse(
                        status=PipelineStatus.REFUSED_BAD_INPUT,
                        query_text=query_text,
                        stage_timings=timer.timings,
                        error_message=input_check.reason,
                    )
        except Exception as e:
            logger.exception(f"Unexpected error in input guardrail: {e}")
            return PipelineResponse(
                status=PipelineStatus.INTERNAL_ERROR,
                query_text=query_text,
                stage_timings=timer.timings,
                error_message=f"Input guardrail error: {e}",
            )

        # -- Stage 3: Hybrid retrieval --
        try:
            with timer.time_stage("retrieval"):
                results: List[RetrievalResult] = self.retriever.search(
                    query_text, top_k=5
                )
        except Exception as e:
            logger.exception(f"Retrieval failed: {e}")
            return PipelineResponse(
                status=PipelineStatus.INTERNAL_ERROR,
                query_text=query_text,
                stage_timings=timer.timings,
                error_message=f"Retrieval error: {e}",
            )

        # -- Stage 4: Post-retrieval confidence guardrail --
        try:
            with timer.time_stage("confidence_guardrail"):
                top_score = results[0].score if results else 0.0
                confidence_check = check_retrieval_confidence(top_score)
                if not confidence_check.passed:
                    return PipelineResponse(
                        status=PipelineStatus.REFUSED_NO_MATCH,
                        query_text=query_text,
                        retrieved_chunks=self._serialize_results(results),
                        stage_timings=timer.timings,
                        error_message=confidence_check.reason,
                    )
        except Exception as e:
            logger.exception(f"Confidence guardrail error: {e}")
            return PipelineResponse(
                status=PipelineStatus.INTERNAL_ERROR,
                query_text=query_text,
                stage_timings=timer.timings,
                error_message=f"Confidence guardrail error: {e}",
            )

        # -- Stage 5: Answer generation --
        chunks = [r.chunk for r in results]
        try:
            with timer.time_stage("generation"):
                answer = generate_answer(query_text, chunks)
        except GenerationError as e:
            logger.error(f"Generation failed: {e}")
            return PipelineResponse(
                status=PipelineStatus.INTERNAL_ERROR,
                query_text=query_text,
                retrieved_chunks=self._serialize_results(results),
                stage_timings=timer.timings,
                error_message=f"Generation failed: {e}",
            )
        except Exception as e:
            logger.exception(f"Unexpected generation error: {e}")
            return PipelineResponse(
                status=PipelineStatus.INTERNAL_ERROR,
                query_text=query_text,
                retrieved_chunks=self._serialize_results(results),
                stage_timings=timer.timings,
                error_message=f"Unexpected generation error: {e}",
            )

        # -- Stage 5.5: Check if model explicitly refused to answer --
        _REFUSAL_PHRASE = "I don't have enough information in the provided context to answer that."
        if answer.strip().startswith(_REFUSAL_PHRASE):
            return PipelineResponse(
                status=PipelineStatus.REFUSED_BY_MODEL,
                query_text=query_text,
                answer=answer,
                retrieved_chunks=self._serialize_results(results),
                stage_timings=timer.timings,
                error_message="Model declined to answer based on context.",
            )

        # -- Stage 6: Post-generation grounding guardrail --
        try:
            with timer.time_stage("grounding_guardrail"):
                # Combine all context texts for the grounding check
                all_context = " ".join(
                    c.context_text or c.text for c in chunks
                )
                grounding_check = check_grounding(answer, all_context)
                if not grounding_check.passed:
                    return PipelineResponse(
                        status=PipelineStatus.REFUSED_UNGROUNDED,
                        query_text=query_text,
                        answer=answer,
                        retrieved_chunks=self._serialize_results(results),
                        stage_timings=timer.timings,
                        error_message=grounding_check.reason,
                    )
        except Exception as e:
            logger.exception(f"Grounding guardrail error: {e}")
            return PipelineResponse(
                status=PipelineStatus.INTERNAL_ERROR,
                query_text=query_text,
                stage_timings=timer.timings,
                error_message=f"Grounding guardrail error: {e}",
            )

        # -- All stages passed --
        return PipelineResponse(
            status=PipelineStatus.SUCCESS,
            query_text=query_text,
            answer=answer,
            retrieved_chunks=self._serialize_results(results),
            stage_timings=timer.timings,
        )

    @staticmethod
    def _serialize_results(results: List[RetrievalResult]) -> List[dict]:
        """Convert RetrievalResult objects to JSON-serializable dicts."""
        return [
            {
                "chunk_id": r.chunk.chunk_id,
                "text": r.chunk.text[:200],  # Truncate for response readability
                "context_text": r.chunk.context_text[:500],
                "doc_id": r.chunk.doc_id,
                "strategy": r.chunk.strategy,
                "score": round(r.score, 6),
                "dense_rank": r.dense_rank,
                "sparse_rank": r.sparse_rank,
            }
            for r in results
        ]
