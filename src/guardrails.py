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
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple, Set


@dataclass
class GuardrailResult:
    """Result of a guardrail check.

    Attributes:
        passed: Whether the check passed (True = proceed, False = block).
        reason: Human-readable explanation if blocked.
    """
    passed: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# Language-agnostic stopword list
# ---------------------------------------------------------------------------
# Function words too common to signal grounding. Covers English + 13 Indic
# languages in MSMARCO-XI. For Indic languages, these are high-frequency
# postpositions, auxiliaries, pronouns, and conjunctions.
_STOP_WORDS = frozenset({
    # --- English ---
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
    # --- Hindi / Marathi / Nepali / Sanskrit (Devanagari) ---
    "का", "के", "की", "है", "हैं", "में", "से", "को", "पर", "इस",
    "ने", "और", "एक", "यह", "वह", "नहीं", "कि", "तो", "या", "भी",
    "था", "थी", "थे", "हो", "होता", "जो", "इसका", "उसका", "अपने",
    "कर", "जा", "आ", "ये", "वो", "मैं", "तुम", "हम", "उन", "इन",
    "पहले", "बाद", "लिए", "साथ", "सब", "कुछ", "बहुत", "अब", "जब",
    # --- Bengali (Bangla) ---
    "এই", "এটি", "একটি", "এবং", "তা", "তার", "যে", "হয়", "করে",
    "থেকে", "জন্য", "সাথে", "বা", "না", "তবে", "আর", "নয়", "ও",
    # --- Tamil ---
    "இது", "அது", "ஒரு", "என்று", "மற்றும்", "இல்லை", "உள்ள",
    "என்ற", "பல", "அல்லது", "போன்ற", "கொண்ட", "செய்",
    # --- Telugu ---
    "ఈ", "ఆ", "ఒక", "మరియు", "లేదా", "కాదు", "అది", "ఇది",
    "వారు", "అతను", "ఆమె", "మేము",
    # --- Gujarati ---
    "આ", "એ", "એક", "અને", "છે", "હતી", "હતું", "માટે",
    "પર", "થી", "સાથે", "કે", "પણ", "નથી",
    # --- Kannada ---
    "ಈ", "ಅದು", "ಒಂದು", "ಮತ್ತು", "ಅಥವಾ", "ಅಲ್ಲ", "ಇದು",
    # --- Malayalam ---
    "ഈ", "ഒരു", "അത്", "എന്ന", "ആണ്", "അല്ല", "മറ്റ്",
    # --- Punjabi (Gurmukhi) ---
    "ਇਹ", "ਇੱਕ", "ਅਤੇ", "ਦੀ", "ਦਾ", "ਦੇ", "ਨੂੰ", "ਵਿੱਚ",
    "ਤੋਂ", "ਨਾਲ", "ਜਾਂ", "ਨਹੀਂ", "ਹੈ", "ਸੀ",
    # --- Odia ---
    "ଏହା", "ଏକ", "ଏବଂ", "କିମ୍ବା", "ନାହିଁ", "ଅଛି",
    # --- Assamese ---
    "এই", "এটা", "আৰু", "বা", "নহয়", "হয়",
    # --- Urdu (Arabic script, shares grammar with Hindi) ---
    "اور", "ہے", "کا", "کے", "کی", "میں", "سے", "کو",
    "نہیں", "یہ", "وہ", "پر", "تھا", "ہیں",
    # --- Numerals and common tokens ---
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
})

# Punctuation characters to strip from tokens (covers Latin, Devanagari, etc.)
_PUNCT_RE = re.compile(
    r'[।॥,.\-!?;:"\'\(\)\[\]\{\}…—–/\\@#$%^&*+=<>~`|।॥,]'
)

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
    # Use 2 chars, not 3, because single-word Indic queries can be short
    if len(stripped) < 2:
        return GuardrailResult(False, "Input too short (< 2 characters), likely garbled")

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

def _extract_content_words(text: str) -> Set[str]:
    r"""Extract meaningful content words from text, excluding stop words.

    Language-agnostic approach:
    1. Lowercase the text
    2. Strip punctuation characters (not \w+ regex, which breaks Indic scripts
       by splitting at matra/vowel-sign boundaries)
    3. Split on whitespace to get whole words
    4. Remove stopwords

    This correctly preserves Devanagari, Bengali, Tamil, Telugu, Kannada,
    Malayalam, Gujarati, Odia, Gurmukhi, and Arabic-script words intact.
    """
    # Strip punctuation without breaking Unicode word boundaries
    cleaned = _PUNCT_RE.sub(' ', text.lower())
    # Split on whitespace — preserves Indic words with matras intact
    words = cleaned.split()
    # Filter stopwords only (no length filter — short Indic words are valid)
    return {w for w in words if w and w not in _STOP_WORDS}


def check_grounding(
    generated_answer: str,
    context_text: str,
    threshold: float = 0.45,
) -> GuardrailResult:
    """Verify the generated answer is grounded in the retrieved context.

    Uses recall-based overlap (what fraction of answer words appear in the
    context) rather than Jaccard. Jaccard unfairly penalizes short answers
    against large contexts because the denominator (union) is dominated by
    context words. Recall measures: "of the words the model used in its
    answer, how many came from the context?" — which is the actual
    grounding question.

    Args:
        generated_answer: The LLM-generated answer text.
        context_text: The concatenated retrieved context that was fed to the LLM.
        threshold: Minimum recall overlap (default 0.3).

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

    # Recall: what fraction of answer words appear in the context?
    overlap = answer_words & context_words
    recall = len(overlap) / len(answer_words)

    if recall < threshold:
        return GuardrailResult(
            False,
            f"Generated answer appears insufficiently grounded in context "
            f"(recall overlap: {recall:.3f} < {threshold}). "
            f"Answer words not in context: {answer_words - context_words}"
        )

    return GuardrailResult(True)
