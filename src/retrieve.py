"""
retrieve.py — Hybrid dense+sparse retrieval with Reciprocal Rank Fusion

Combines two complementary retrieval strategies:

1. Dense retrieval (FAISS): Semantic similarity via multilingual-e5-small
   embeddings. Good at matching paraphrases and conceptual similarity,
   even across languages. Uses IndexFlatIP on L2-normalized vectors
   for exact cosine similarity — no approximation.

2. Sparse retrieval (BM25): Lexical matching via term frequency statistics.
   Good at matching specific keywords, names, and numbers that embedding
   models sometimes miss. Especially useful for Indic languages where the
   embedding model may be less precise on rare terms.

Reciprocal Rank Fusion (RRF) combines rankings without needing to calibrate
scores across different retrieval methods. The formula:
    score(d) = Σ 1/(k + rank_i(d))
where k=60 (standard constant from Cormack et al. 2009).

RRF is preferred over linear score combination because FAISS and BM25 scores
are on completely different scales — RRF only uses rank positions, making
it robust to score distribution differences.
"""

import pickle
import os
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

from src.chunkers import Chunk
from src.embed import encode_query


@dataclass
class RetrievalResult:
    """A single retrieval result with its fused score.

    Attributes:
        chunk: The matched Chunk object.
        score: RRF fused score (higher = better match).
        dense_rank: Rank in dense retrieval (0-based, None if not in top-k).
        sparse_rank: Rank in sparse retrieval (0-based, None if not in top-k).
    """
    chunk: Chunk
    score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None


class HybridRetriever:
    """Hybrid retriever combining FAISS dense search with BM25 sparse search.

    Loads pre-built indices from disk (created by build_index.py).
    All indices are held in memory for zero-latency access.
    """

    def __init__(
        self,
        index_dir: str = "data",
        rrf_k: int = 60,
    ):
        """Load Qdrant client and BM25 index from disk.

        Args:
            index_dir: Directory containing qdrant_db and bm25.pkl.
            rrf_k: RRF constant (default 60, standard from literature).
        """
        self.rrf_k = rrf_k
        self.collection_name = "msmarco_chunks"

        # Initialize Qdrant Client (replaces FAISS and chunks.pkl)
        qdrant_path = os.path.join(index_dir, "qdrant_db")
        if not os.path.exists(qdrant_path):
            raise FileNotFoundError(
                f"Qdrant DB not found at {qdrant_path}. "
                "Run build_index.py first to create the index."
            )
        self.qdrant = QdrantClient(path=qdrant_path)

        # Load BM25 index
        bm25_path = os.path.join(index_dir, "bm25.pkl")
        with open(bm25_path, "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)

        # Warm up the embedding model and Qdrant connection to eliminate cold-start latency
        # We discard the result. The first query forces model JIT and connection pool creation.
        _ = self.search("warmup query", top_k=1, dense_candidates=5, sparse_candidates=5)

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_candidates: int = 20,
        sparse_candidates: int = 20,
    ) -> List[RetrievalResult]:
        """Run hybrid retrieval: dense + sparse + RRF fusion.

        Args:
            query: Search query text.
            top_k: Number of final results to return after fusion.
            dense_candidates: How many candidates to pull from FAISS.
            sparse_candidates: How many candidates to pull from BM25.

        Returns:
            Top-k RetrievalResult objects sorted by RRF score (descending).
        """
        # 1. Dense retrieval via FAISS
        dense_results = self._dense_search(query, dense_candidates)

        # 2. Sparse retrieval via BM25
        sparse_results = self._sparse_search(query, sparse_candidates)

        # 3. Fuse with RRF
        fused = self._rrf_fuse(dense_results, sparse_results, top_k)

        return fused

    def _dense_search(
        self, query: str, k: int
    ) -> List[Tuple[int, float]]:
        """Search Qdrant index for nearest neighbors.

        Returns list of (chunk_index, similarity_score) tuples.
        """
        query_vec = encode_query(query)  # shape: (1, dim)
        
        # Qdrant returns a QueryResponse object
        search_result = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vec[0].tolist(),
            limit=k,
        )

        results = []
        for point in search_result.points:
            results.append((int(point.id), float(point.score)))
        return results

    def _sparse_search(
        self, query: str, k: int
    ) -> List[Tuple[int, float]]:
        """Search BM25 index for best lexical matches.

        Tokenizes query by whitespace (simple but effective for multilingual).
        Returns list of (chunk_index, bm25_score) tuples.
        """
        # Simple whitespace tokenization — works for both English and Indic
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices by score
        top_indices = np.argsort(scores)[::-1][:k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        return results

    def _rrf_fuse(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Combine dense and sparse rankings using Reciprocal Rank Fusion.

        RRF score for document d = Σ 1/(k + rank_i(d))
        where k is the RRF constant (60) and rank_i is the 1-based rank
        in retrieval method i.

        The k=60 constant dampens the contribution of lower-ranked results.
        A document ranked #1 in both methods gets: 1/61 + 1/61 ≈ 0.033
        A document ranked #1 in one and #10 in the other: 1/61 + 1/70 ≈ 0.031
        """
        # Build rank maps (1-based ranks)
        dense_ranks = {}
        for rank, (idx, _score) in enumerate(dense_results):
            dense_ranks[idx] = rank + 1

        sparse_ranks = {}
        for rank, (idx, _score) in enumerate(sparse_results):
            sparse_ranks[idx] = rank + 1

        # Compute RRF scores for all unique candidate indices
        all_indices = list(set(dense_ranks.keys()) | set(sparse_ranks.keys()))
        
        # Fetch chunk payloads from Qdrant
        points = self.qdrant.retrieve(
            collection_name=self.collection_name,
            ids=all_indices
        )
        # Map payloads back to Chunk objects
        chunk_map = {}
        for p in points:
            chunk_map[p.id] = Chunk(**p.payload)
            
        scored = []

        for idx in all_indices:
            if idx not in chunk_map:
                continue
                
            rrf_score = 0.0
            d_rank = None
            s_rank = None

            if idx in dense_ranks:
                d_rank = dense_ranks[idx]
                rrf_score += 1.0 / (self.rrf_k + d_rank)

            if idx in sparse_ranks:
                s_rank = sparse_ranks[idx]
                rrf_score += 1.0 / (self.rrf_k + s_rank)

            scored.append(RetrievalResult(
                chunk=chunk_map[idx],
                score=rrf_score,
                dense_rank=d_rank - 1 if d_rank else None,  # Convert back to 0-based
                sparse_rank=s_rank - 1 if s_rank else None,
            ))

        # Sort by RRF score descending, take top-k
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
