"""
app.py — FastAPI application for the voice-enabled RAG system

Provides three endpoints:
- POST /query/voice: Full pipeline including STT (multipart audio upload)
- POST /query/text: Text-only pipeline (skips STT, for testing/eval)
- GET /health: Health check

The embedding model, Qdrant vector database, and BM25 index are loaded once at startup
via the lifespan context manager, not per-request. This avoids the model
load penalty on every query.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env file for API keys
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrieve import HybridRetriever
from src.pipeline import RAGPipeline, PipelineRequest, PipelineStatus
from src.embed import get_model  # Pre-load embedding model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Module-level pipeline reference, initialized at startup
_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and indices once at startup, release on shutdown.

    This avoids the model load and index load on every request.
    The Qdrant index, BM25 index, and embedding model stay in memory
    for the lifetime of the server process.
    """
    global _pipeline

    logger.info("Loading embedding model...")
    get_model()  # Trigger lazy load of sentence-transformers model

    logger.info("Loading retrieval indices...")
    data_dir = os.environ.get("INDEX_DIR", "data")
    retriever = HybridRetriever(index_dir=data_dir)
    collection_info = retriever.qdrant.get_collection(retriever.collection_name)
    logger.info(
        f"Loaded Qdrant collection '{retriever.collection_name}' with "
        f"{collection_info.points_count} vectors"
    )

    _pipeline = RAGPipeline(retriever=retriever)
    logger.info("Pipeline ready!")

    yield

    # Cleanup (Python GC handles most of it)
    _pipeline = None
    logger.info("Pipeline shut down.")


app = FastAPI(
    title="Voice-Enabled RAG System",
    description=(
        "Retrieval-Augmented Generation for Indic languages using "
        "MSMARCO-XI dataset, Sarvam STT, and Groq LLaMA generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# -- Request/Response models for the API --

class TextQueryRequest(BaseModel):
    """JSON body for text-based queries."""
    query: str
    language: str = "hi-IN"


class QueryResponse(BaseModel):
    """Structured API response."""
    status: str
    answer: str = ""
    query_text: str = ""
    retrieved_chunks: list = []
    stage_timings: dict = {}
    error_message: str = ""


# -- Endpoints --

@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns 200 if the pipeline is loaded and ready to serve queries.
    Returns 503 if the pipeline hasn't been initialized yet.
    """
    if _pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "message": "Pipeline not initialized"}
        )
    collection_info = _pipeline.retriever.qdrant.get_collection(
        _pipeline.retriever.collection_name
    )
    return {
        "status": "healthy",
        "vector_count": collection_info.points_count,
        "collection_name": _pipeline.retriever.collection_name,
    }


@app.post("/query/text", response_model=QueryResponse)
async def query_text(request: TextQueryRequest):
    """Text-based query endpoint — skips STT.

    Use this for testing and evaluation. Accepts a JSON body with
    a "query" field and optional "language" field.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    pipeline_request = PipelineRequest(
        query_text=request.query,
        language=request.language,
    )

    response = _pipeline.run(pipeline_request)

    # Map pipeline status to HTTP status codes
    http_status = _status_to_http(response.status)

    return JSONResponse(
        status_code=http_status,
        content={
            "status": response.status.value,
            "answer": response.answer,
            "query_text": response.query_text,
            "retrieved_chunks": response.retrieved_chunks,
            "stage_timings": {
                k: round(v * 1000, 2)  # Convert to milliseconds
                for k, v in response.stage_timings.items()
            },
            "error_message": response.error_message,
        },
    )


@app.post("/query/voice", response_model=QueryResponse)
async def query_voice(
    audio: UploadFile = File(...),
    language: str = "hi-IN",
):
    """Voice-based query endpoint — full pipeline including STT.

    Accepts multipart audio upload. The audio file is sent to Sarvam's
    STT API for transcription before entering the retrieval pipeline.

    Args:
        audio: Audio file (WAV, MP3, etc.) as multipart upload.
        language: BCP-47 language code (default "hi-IN").
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Read audio bytes from the upload
    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    pipeline_request = PipelineRequest(
        audio_bytes=audio_bytes,
        language=language,
    )

    response = _pipeline.run(pipeline_request)

    http_status = _status_to_http(response.status)

    return JSONResponse(
        status_code=http_status,
        content={
            "status": response.status.value,
            "answer": response.answer,
            "query_text": response.query_text,
            "retrieved_chunks": response.retrieved_chunks,
            "stage_timings": {
                k: round(v * 1000, 2)
                for k, v in response.stage_timings.items()
            },
            "error_message": response.error_message,
        },
    )


def _status_to_http(status: PipelineStatus) -> int:
    """Map PipelineStatus enum to HTTP status codes.

    Successful queries and safe refusals return 200.
    Actual errors (STT failure, internal errors) return 500/502.
    """
    if status in (
        PipelineStatus.SUCCESS,
        PipelineStatus.REFUSED_BAD_INPUT,
        PipelineStatus.REFUSED_NO_MATCH,
        PipelineStatus.REFUSED_BY_MODEL,
        PipelineStatus.REFUSED_UNGROUNDED,
    ):
        return 200
    elif status == PipelineStatus.STT_ERROR:
        return 502  # Bad gateway — upstream STT service failed
    else:
        return 500


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
