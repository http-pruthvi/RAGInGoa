"""
embed.py — Multilingual embedding using intfloat/multilingual-e5-small

Uses the E5 model family because it has strong performance on multilingual
benchmarks including Indic languages, and the "small" variant (118M params)
is fast enough for real-time retrieval (~5ms per query on CPU).

E5 models require specific prefixes:
  - "query: " for search queries
  - "passage: " for document passages
This asymmetric encoding lets the model distinguish between short questions
and longer passages during contrastive training, improving retrieval quality.

Vectors are L2-normalized so that dot product = cosine similarity,
which lets us use FAISS IndexFlatIP for exact cosine search.
"""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer


# Module-level model cache — loaded once, reused across calls
_model: Optional[SentenceTransformer] = None
_MODEL_NAME = "intfloat/multilingual-e5-small"


def get_model() -> SentenceTransformer:
    """Load or return cached embedding model.

    Lazy-loaded to avoid paying the model load cost at import time.
    The model is ~500MB and takes ~2-3s to load on first call.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def get_embedding_dimension() -> int:
    """Return the dimensionality of the embedding vectors (384 for e5-small)."""
    model = get_model()
    return model.get_sentence_embedding_dimension()


def encode_passages(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Encode document passages into normalized embedding vectors.

    Adds the "passage: " prefix required by E5 models. Returns L2-normalized
    vectors so that FAISS IndexFlatIP computes cosine similarity.

    Args:
        texts: List of passage texts to embed.
        batch_size: Batch size for encoding (default 64).

    Returns:
        numpy array of shape (len(texts), embedding_dim), L2-normalized.
    """
    model = get_model()

    # E5 requires "passage: " prefix for document encoding
    prefixed = [f"passage: {t}" for t in texts]

    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,  # Only show progress for large batches
        normalize_embeddings=True,  # L2 normalize so IP = cosine similarity
    )

    return np.array(embeddings, dtype=np.float32)


def encode_query(text: str) -> np.ndarray:
    """Encode a single search query into a normalized embedding vector.

    Adds the "query: " prefix required by E5 models.

    Args:
        text: Query text to embed.

    Returns:
        numpy array of shape (1, embedding_dim), L2-normalized.
    """
    model = get_model()

    # E5 requires "query: " prefix for query encoding
    prefixed = f"query: {text}"

    embedding = model.encode(
        [prefixed],
        normalize_embeddings=True,
    )

    return np.array(embedding, dtype=np.float32)
