"""
generate.py — Answer generation using Groq (LLaMA 3.3 70B)

Uses Groq's free-tier API for answer generation. Groq provides low-latency
LLM inference via custom LPU hardware. We use their OpenAI-compatible SDK.

Why Groq instead of Anthropic?
- Groq's free tier requires no credit card and has generous rate limits
- Anthropic has no permanent free API tier (only a small one-time trial credit)
- Groq's LPU inference is significantly faster than most providers (~200-500ms)
- The task doesn't mandate a specific generation provider — only STT (Sarvam) is fixed
- This is a budget-driven engineering decision: $0 operational cost for development/eval

The system prompt is carefully crafted to:
1. Restrict answers to ONLY the provided context (no parametric knowledge)
2. Require explicit citation of which context snippet was used
3. Instruct the model to say "I don't have enough information" when unsure

Includes 1 retry on transient API failures (rate limits, server errors).
"""

import os
import time
import logging
from groq import Groq
from typing import List, Optional

from src.chunkers import Chunk

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Raised when LLM generation fails after retries."""
    pass


# System prompt that strictly constrains the model to answer from context only
_SYSTEM_PROMPT = """You are a helpful assistant that answers questions ONLY based on the provided context passages. You must follow these rules strictly:

1. ONLY use information from the provided context passages to answer the question. NEVER use your own knowledge, training data, or information from outside the context.

2. When you answer, ALWAYS cite which context snippet(s) you used by referencing their index number in square brackets, e.g., [1], [2], [1,3].

3. If the provided context does not contain enough information to answer the question fully and accurately, you MUST respond with: "I don't have enough information in the provided context to answer that."

4. Do NOT speculate, infer beyond what the context explicitly states, or fill in gaps with your own knowledge.

5. If the context partially answers the question, provide what you can from the context and clearly state what information is missing.

6. Keep your answers concise (1-3 sentences) and directly relevant to the question asked."""


# Groq-hosted model — LLaMA 3.3 70B offers the best quality/speed tradeoff
# on Groq's free tier. 70B params gives strong instruction-following, and
# Groq's LPU hardware keeps inference latency low (~200-500ms).
_MODEL = "llama-3.3-70b-versatile"


def _get_api_key() -> str:
    """Fetch GROQ_API_KEY from environment, fail fast if missing."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise GenerationError(
            "GROQ_API_KEY environment variable is not set. "
            "Get your free key at https://console.groq.com/"
        )
    return key


# Module-level Groq client cache to reuse HTTP connection pooling across requests
_groq_client: Optional[Groq] = None

def _get_client() -> Groq:
    """Return a cached Groq client with a 4.0s timeout."""
    global _groq_client
    if _groq_client is None:
        api_key = _get_api_key()
        # 4.0s timeout bounds the P100 latency outlier
        _groq_client = Groq(api_key=api_key, timeout=4.0)
    return _groq_client


def _format_context(chunks: List[Chunk]) -> str:
    """Format retrieved chunks as numbered context snippets for the prompt.

    Uses context_text (the richer version) rather than the embedding text,
    so the model sees full parent chunks or sentence-window context.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        # Use context_text for generation — it has the broader context
        text = chunk.context_text if chunk.context_text else chunk.text
        parts.append(f"[{i}] {text}")
    return "\n\n".join(parts)


def generate_answer(
    query: str,
    chunks: List[Chunk],
    max_retries: int = 1,
) -> str:
    """Generate an answer from retrieved context using Groq.

    Args:
        query: The user's question.
        chunks: Retrieved Chunk objects (top-k from hybrid retrieval).
        max_retries: Number of retries on transient API errors (default 1).

    Returns:
        Generated answer string with citations.

    Raises:
        GenerationError: If generation fails after all retries.
    """
    client = _get_client()

    context = _format_context(chunks)

    user_message = f"""Context passages:
{context}

Question: {query}

Answer the question using ONLY the context passages above. Cite the passage number(s) you used."""

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            # Groq uses the OpenAI-compatible chat.completions.create interface
            response = client.chat.completions.create(
                model=_MODEL,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )

            # OpenAI-compatible response: text at choices[0].message.content
            answer = response.choices[0].message.content
            return answer.strip() if answer else ""

        except Exception as e:
            error_str = str(e)

            # Rate limit errors (429) are retryable
            if "429" in error_str or "rate" in error_str.lower():
                last_error = f"Rate limited: {e}"
            # Server errors (5xx) are retryable
            elif "500" in error_str or "502" in error_str or "503" in error_str:
                last_error = f"Server error: {e}"
            # Connection errors are retryable
            elif "connection" in error_str.lower() or "timeout" in error_str.lower():
                last_error = f"Connection error: {e}"
            else:
                # Non-retryable errors (auth, bad request, etc.)
                raise GenerationError(f"Groq API error: {e}")

        logger.warning(f"Groq API attempt {attempt + 1} failed: {e}. Retrying...")
        # Backoff before retry
        if attempt < max_retries:
            time.sleep(1.0)

    raise GenerationError(
        f"Generation failed after {max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
