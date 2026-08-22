import os
import json
import time
import logging
from groq import Groq
from typing import List, Optional, Dict, Any

from src.chunkers import Chunk

logger = logging.getLogger(__name__)

class GenerationError(Exception):
    pass

_SYSTEM_PROMPT = """You are a helpful assistant that answers questions ONLY based on the provided context passages. You must follow these rules strictly:

1. ONLY use information from the provided context passages to answer the question.
2. If the provided context does not contain enough information to answer the question, you MUST set has_sufficient_context to false.
3. You MUST output your response in valid JSON format.

Your JSON MUST strictly adhere to this schema:
{
  "answer": "Your concise answer (1-3 sentences) directly relevant to the question. If you don't have enough information, say 'I don't have enough information'.",
  "citations": [list of integers representing the context passage indices you used, e.g. [1, 2]. Empty list if none used.],
  "has_sufficient_context": true or false
}"""

_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

def _get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise GenerationError("GROQ_API_KEY environment variable is not set.")
    return key

_groq_client: Optional[Groq] = None
_cached_key: Optional[str] = None

def _get_client() -> Groq:
    global _groq_client, _cached_key
    api_key = _get_api_key()
    if _groq_client is None or _cached_key != api_key:
        _groq_client = Groq(api_key=api_key, timeout=4.0)
        _cached_key = api_key
    return _groq_client

def _format_context(chunks: List[Chunk], max_chunks: int = 5) -> str:
    parts = []
    for i, chunk in enumerate(chunks[:max_chunks], 1):
        text = chunk.context_text if chunk.context_text else chunk.text
        words = text.split()
        if len(words) > 150:
            text = " ".join(words[:150]) + "..."
        parts.append(f"[{i}] {text}")
    return "\n\n".join(parts)

def generate_answer(
    query: str,
    chunks: List[Chunk],
    max_retries: int = 1,
) -> str:
    client = _get_client()
    context = _format_context(chunks, max_chunks=3)

    user_message = f"""Context passages:\n{context}\n\nQuestion: {query}\n\nAnswer the question using ONLY the context passages above. Respond in JSON."""

    last_error = "Unknown error"

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=_MODEL,
                max_tokens=256,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            raw_text = response.choices[0].message.content
            if not raw_text:
                return "Error: Empty response"
            
            try:
                parsed = json.loads(raw_text)
                answer = parsed.get("answer", "")
                citations = parsed.get("citations", [])
                
                if citations:
                    answer += f" \n\nUsed passage numbers: {citations}"
                return answer
            except json.JSONDecodeError:
                return raw_text.strip()

        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries:
                time.sleep(1.0)
                continue
            elif "timeout" in error_str.lower() and attempt < max_retries:
                time.sleep(0.5)
                continue
            last_error = error_str
            break

    raise GenerationError(f"Failed to generate answer from Groq: {last_error}")
