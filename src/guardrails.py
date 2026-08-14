"""
guardrails.py — Three independent safety/quality gates for the RAG pipeline

All guardrails are local and cheap (no LLM calls) — they use regex patterns,
numeric thresholds, and lexical overlap checks. This design choice avoids
adding latency from extra API calls and keeps the guardrail evaluation
well under 1ms.

The three gates:
1. Pre-retrieval (input validation): Rejects empty/garbled/unsafe input before
   wasting retrieval compute.
2. Post-retrieval (confidence gate): Refuses to answer when retrieval scores
   are too low — this is the primary "knows when not to answer" mechanism.
3. Post-generation (grounding check): Verifies the LLM's answer is grounded
   in the retrieved context via lexical overlap, catching hallucinations.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class GuardrailResult:
    """Result of a guardrail check.

    Attributes:
        passed: Whether the check passed (True = proceed, False = block).
        reason: Human-readable explanation if blocked.
    """
    passed: bool
    reason: str = ""


# -- Common stop words to exclude from grounding overlap checks --
# These are too common to be meaningful evidence of grounding
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must", "need",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "and", "or", "but", "if", "then", "else", "when", "where", "how", "why",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "not", "no", "nor", "so", "too", "very", "just", "also", "than",
    "about", "up", "out", "into", "over", "after", "before", "between",
    "under", "again", "further", "once", "here", "there", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "only", "own", "same",
    # Common Hindi stop words (transliterated and Devanagari)
    "का", "के", "की", "है", "में", "से", "को", "पर", "इस", "ने",
    "और", "एक", "यह", "नहीं", "कि", "हैं", "तो", "या", "भी",
})

# Patterns that indicate obviously unsafe/adversarial input
_UNSAFE_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?above", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"pretend\s+you", re.IGNORECASE),
    re.compile(r"act\s+as\s+if", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Gate 1: Pre-retrieval input validation
# ---------------------------------------------------------------------------

def check_input(text: str) -> GuardrailResult:
    """Validate query text before retrieval.

    Rejects:
    - Empty or whitespace-only input
    - Very short input (< 2 chars after stripping) likely from STT noise
    - Input that's mostly non-alphanumeric (garbled transcription)
    - Known prompt injection patterns

    Args:
        text: The query text (from STT or direct input).

    Returns:
        GuardrailResult — passed=True if input is acceptable.
    """
    if not text or not text.strip():
        return GuardrailResult(False, "Empty or whitespace-only input")

    stripped = text.strip()

    # Reject very short transcripts — likely STT noise
    if len(stripped) < 3:
        return GuardrailResult(False, "Input too short (< 3 characters), likely garbled")

    # Check for mostly non-alphanumeric content (garbled STT output)
    # Allow Indic scripts by checking for Unicode letters, not just ASCII
    alnum_chars = sum(1 for c in stripped if c.isalnum())
    if len(stripped) > 5 and alnum_chars / len(stripped) < 0.3:
        return GuardrailResult(
            False,
            "Input appears garbled (< 30% alphanumeric characters)"
        )

    # Check for prompt injection patterns
    for pattern in _UNSAFE_PATTERNS:
        if pattern.search(stripped):
            return GuardrailResult(
                False,
                f"Input contains potentially unsafe pattern: {pattern.pattern}"
            )

    return GuardrailResult(True)


# ---------------------------------------------------------------------------
# Gate 2: Post-retrieval confidence check
# ---------------------------------------------------------------------------

def check_retrieval_confidence(
    top_score: float,
    threshold: float = 0.015,
) -> GuardrailResult:
    """Check if retrieval results are confident enough to answer.

    This is the primary "knows when not to answer" mechanism. If the best
    retrieval score (RRF fused score) is below the threshold, it means
    no chunk in the corpus is a good match — better to refuse than let
    the LLM hallucinate from weak context.

    The threshold of 0.015 was chosen empirically: with RRF using k=60
    and fusing top-20 from each retriever, a score below 0.015 typically
    means the query is out-of-domain or the corpus lacks relevant content.

    Args:
        top_score: The highest RRF fused score from retrieval.
        threshold: Minimum acceptable score (default 0.015).

    Returns:
        GuardrailResult — passed=True if confidence is sufficient.
    """
    if top_score < threshold:
        return GuardrailResult(
            False,
            f"Retrieval confidence too low ({top_score:.4f} < {threshold}). "
            "The query doesn't match any content in the corpus well enough "
            "to provide a reliable answer."
        )
    return GuardrailResult(True)


# ---------------------------------------------------------------------------
# Gate 3: Post-generation grounding check
# ---------------------------------------------------------------------------

def _extract_content_words(text: str) -> set:
    """Extract meaningful content words from text, excluding stop words.

    Lowercases everything and filters out stop words and very short tokens.
    Works for both English and Indic text (Indic stop words are included).
    """
    # Split on whitespace and punctuation
    words = re.findall(r'\w+', text.lower())
    # Filter stop words and very short words (< 3 chars)
    return {w for w in words if w not in _STOP_WORDS and len(w) >= 3}


def check_grounding(
    generated_answer: str,
    context_text: str,
    threshold: float = 0.15,
) -> GuardrailResult:
    """Verify the generated answer is grounded in the retrieved context.

    Computes Jaccard similarity between content words in the answer and
    content words in the context. If overlap is too low, the answer likely
    contains hallucinated content not supported by the retrieved passages.

    Why Jaccard and not something fancier? Because:
    1. It's instantaneous (< 0.1ms) — no model inference needed
    2. For factoid QA (MSMARCO-style), correct answers almost always share
       key terms with their source passage
    3. It's a conservative check — we'd rather flag borderline cases than
       miss obvious hallucinations

    Args:
        generated_answer: The LLM-generated answer text.
        context_text: The concatenated retrieved context that was fed to the LLM.
        threshold: Minimum Jaccard overlap (default 0.15).

    Returns:
        GuardrailResult — passed=True if answer appears grounded.
    """
    answer_words = _extract_content_words(generated_answer)
    context_words = _extract_content_words(context_text)

    if not answer_words:
        # If the answer has no content words (e.g., "I don't know"),
        # that's fine — it's a valid refusal
        return GuardrailResult(True)

    if not context_words:
        # No context to check against — can't verify grounding
        return GuardrailResult(
            False,
            "No context content words to verify grounding against"
        )

    # Jaccard similarity: |A ∩ B| / |A ∪ B|
    intersection = answer_words & context_words
    union = answer_words | context_words
    jaccard = len(intersection) / len(union) if union else 0.0

    if jaccard < threshold:
        return GuardrailResult(
            False,
            f"Generated answer appears insufficiently grounded in context "
            f"(Jaccard overlap: {jaccard:.3f} < {threshold}). "
            f"Answer words not in context: {answer_words - context_words}"
        )

    return GuardrailResult(True)
