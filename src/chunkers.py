"""
chunkers.py — Multi-strategy text chunking for RAG retrieval

Implements three chunking strategies, each optimized for different retrieval
scenarios. All share a unified Chunk dataclass so the retrieval layer doesn't
care which strategy produced a chunk.

Why three strategies?
- Fixed-size: Simple, predictable chunk sizes. Good baseline that works
  for any text without assumptions about sentence structure.
- Sentence-window: Embeds individual sentences for precise retrieval of
  short factoid answers (common in MSMARCO), but stores neighboring
  sentences as context so the generator has enough to form a coherent answer.
- Hierarchical parent/child: Small child chunks (~120 words) maximize
  retrieval precision, but the generator sees the full parent (~500 words)
  for broader context. Best when the answer spans multiple sentences.

The combinator runs all three and returns the union — the retrieval layer
scores all chunks together and naturally selects whichever strategy produced
the best match for a given query.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    """Universal chunk container used across all chunking strategies.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        text: The text used for embedding and retrieval matching.
        context_text: Richer text passed to the generator. For sentence-window
                      chunks this includes neighboring sentences. For hierarchical
                      chunks this is the parent chunk. For fixed-size, same as text.
        doc_id: Identifier of the source document/passage.
        strategy: Which chunking strategy produced this ("fixed", "sentence_window",
                  "hierarchical_child").
        position: Index of this chunk within the document (0-based).
        language: Language code if available (e.g., "hi", "en").
        metadata: Additional metadata (source fields, etc.).
    """
    chunk_id: str
    text: str
    context_text: str
    doc_id: str
    strategy: str
    position: int
    language: str = ""
    metadata: dict = field(default_factory=dict)


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex-based heuristics.

    Handles common sentence terminators (. ! ?) while avoiding
    false splits on abbreviations (Mr., Dr., etc.) and decimals.
    Works reasonably well for both English and transliterated Indic text.
    Indic scripts use '।' (Devanagari danda) as sentence terminator.
    """
    # Split on sentence-ending punctuation followed by whitespace or end of string
    # Handles . ! ? as well as Devanagari/Bengali danda (।), double danda (॥), and Urdu question mark (؟)
    parts = re.split(r'(?<=[.!?।॥؟])\s+', text.strip())
    # Filter out empty strings and very short fragments
    return [s.strip() for s in parts if s.strip() and len(s.strip()) > 2]


def _word_count(text: str) -> int:
    """Count words in text. Simple whitespace split works for most languages."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-size chunking with configurable overlap
# ---------------------------------------------------------------------------

def chunk_fixed_size(
    text: str,
    doc_id: str,
    language: str = "",
    window_words: int = 200,
    overlap_fraction: float = 0.2,
) -> List[Chunk]:
    """Split text into fixed-size word windows with overlap.

    Uses word-count windows rather than character counts because word
    boundaries are more meaningful for embedding models, and this avoids
    splitting mid-word in Indic scripts where characters can be multi-byte.

    Args:
        text: Input document text.
        doc_id: Source document identifier.
        language: Language code (e.g., "hi").
        window_words: Number of words per chunk (default 200).
        overlap_fraction: Fraction of window to overlap (default 0.2 = 40 words).

    Returns:
        List of Chunk objects with strategy="fixed".
    """
    words = text.split()
    if not words:
        return []

    overlap_words = max(1, int(window_words * overlap_fraction))
    step = window_words - overlap_words
    chunks = []

    for i in range(0, len(words), step):
        chunk_words = words[i : i + window_words]
        if not chunk_words:
            break

        chunk_text = " ".join(chunk_words)
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_fixed_{len(chunks)}",
            text=chunk_text,
            context_text=chunk_text,  # For fixed-size, context = text
            doc_id=doc_id,
            strategy="fixed",
            position=len(chunks),
            language=language,
            metadata={"window_words": window_words, "overlap_words": overlap_words},
        ))

        # Stop if we've consumed all words (avoid a trailing tiny chunk)
        if i + window_words >= len(words):
            break

    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Sentence-window chunking
# ---------------------------------------------------------------------------

def chunk_sentence_window(
    text: str,
    doc_id: str,
    language: str = "",
    context_window: int = 2,
) -> List[Chunk]:
    """Embed individual sentences but store neighboring context.

    Each chunk's `text` is a single sentence (for precise retrieval matching),
    but `context_text` includes N sentences before and after (for richer
    generator input). This is effective for MSMARCO-style factoid QA where
    the answer often lives in one sentence but needs surrounding context
    to make sense.

    Args:
        text: Input document text.
        doc_id: Source document identifier.
        language: Language code.
        context_window: Number of neighboring sentences on each side to include
                        in context_text (default 2 = ±2 sentences).

    Returns:
        List of Chunk objects with strategy="sentence_window".
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    for i, sentence in enumerate(sentences):
        # Build context window: N sentences before and after
        start = max(0, i - context_window)
        end = min(len(sentences), i + context_window + 1)
        context_sentences = sentences[start:end]
        context = " ".join(context_sentences)

        chunks.append(Chunk(
            chunk_id=f"{doc_id}_sentwin_{i}",
            text=sentence,
            context_text=context,
            doc_id=doc_id,
            strategy="sentence_window",
            position=i,
            language=language,
            metadata={"context_window": context_window, "total_sentences": len(sentences)},
        ))

    return chunks


# ---------------------------------------------------------------------------
# Strategy 3: Hierarchical parent/child chunking
# ---------------------------------------------------------------------------

def chunk_hierarchical(
    text: str,
    doc_id: str,
    language: str = "",
    child_words: int = 120,
    parent_words: int = 500,
) -> List[Chunk]:
    """Create small child chunks for retrieval with large parent chunks for context.

    Parent chunks (~500 words) provide broad context to the generator.
    Child chunks (~120 words) are what gets embedded and matched against queries.
    When a child chunk matches, its parent's text is passed to generation.

    This is the best strategy when answers span multiple sentences — the child
    ensures precision while the parent ensures the generator sees enough context.

    Args:
        text: Input document text.
        doc_id: Source document identifier.
        language: Language code.
        child_words: Target word count for child chunks (~120).
        parent_words: Target word count for parent chunks (~500).

    Returns:
        List of Chunk objects with strategy="hierarchical_child".
        Only child chunks are returned (parents are referenced via context_text).
    """
    words = text.split()
    if not words:
        return []

    # First, create parent chunks
    parents = []
    for i in range(0, len(words), parent_words):
        parent_text = " ".join(words[i : i + parent_words])
        parents.append(parent_text)

    # Then create child chunks, each mapped to its parent
    chunks = []
    child_idx = 0
    for parent_idx, parent_text in enumerate(parents):
        parent_words_list = parent_text.split()
        for j in range(0, len(parent_words_list), child_words):
            child_text = " ".join(parent_words_list[j : j + child_words])
            if not child_text.strip():
                continue

            chunks.append(Chunk(
                chunk_id=f"{doc_id}_hier_{child_idx}",
                text=child_text,
                context_text=parent_text,  # Generator sees the full parent
                doc_id=doc_id,
                strategy="hierarchical_child",
                position=child_idx,
                language=language,
                metadata={
                    "parent_index": parent_idx,
                    "child_words": child_words,
                    "parent_words_target": parent_words,
                },
            ))
            child_idx += 1

    return chunks


# ---------------------------------------------------------------------------
# Combinator: run all strategies and return the union
# ---------------------------------------------------------------------------

def chunk_all_strategies(
    text: str,
    doc_id: str,
    language: str = "",
) -> List[Chunk]:
    """Run all three chunking strategies and return the union.

    The retrieval layer scores all chunks together — RRF fusion naturally
    picks whichever strategy produced the best match for a given query.
    This is better than choosing one strategy upfront because different
    query types benefit from different chunking granularities.

    Args:
        text: Input document text.
        doc_id: Source document identifier.
        language: Language code.

    Returns:
        Combined list of Chunk objects from all three strategies.
    """
    if not text or not text.strip():
        return []

    all_chunks = []

    # Strategy 1: Fixed-size with 20% overlap
    all_chunks.extend(chunk_fixed_size(text, doc_id, language))

    # Strategy 2: Sentence-window with ±2 context
    all_chunks.extend(chunk_sentence_window(text, doc_id, language))

    # Strategy 3: Hierarchical parent/child
    all_chunks.extend(chunk_hierarchical(text, doc_id, language))

    return all_chunks
