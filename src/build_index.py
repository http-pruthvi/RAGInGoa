"""
build_index.py — Offline index builder for the RAG pipeline

Loads passages from the ai4bharat/MSMARCO-XI dataset, runs all three
chunking strategies, embeds the chunks, and builds the Qdrant + BM25
indices. Results are persisted to disk so the query-time API doesn't
need to rebuild indices on startup.

Dataset structure:
  Per-language parquet files on HuggingFace, each ~3.7 GB:
    train/asmtrain.parquet  (Assamese,  asm_Beng)
    train/bentrain.parquet  (Bengali,   ben_Beng)
    train/gujtrain.parquet  (Gujarati,  guj_Gujr)
    train/hintrain.parquet  (Hindi,     hin_Deva)
    train/kantrain.parquet  (Kannada,   kan_Knda)
    train/maltrain.parquet  (Malayalam, mal_Mlym)
    train/martrain.parquet  (Marathi,   mar_Deva)
    train/neptrain.parquet  (Nepali,    nep_Deva)
    train/oritrain.parquet  (Odia,      ori_Orya)
    train/pantrain.parquet  (Punjabi,   pan_Guru)
    train/santrain.parquet  (Sanskrit,  san_Deva)
    train/tamtrain.parquet  (Tamil,     tam_Taml)
    train/urdtrain.parquet  (Urdu,      urd_Arab)

  Each row has:
    query (str), Answer (str), query_id (int),
    passages: struct with Translated_passages (list[str]),
              English_passages (list[str]), is_selected (list[int])

  Uses pyarrow.iter_batches for memory-efficient reads from multi-GB files.

Usage:
    python -m src.build_index [--max-per-lang 50] [--data-dir data]
    python -m src.build_index --languages hi bn ta --max-per-lang 100
"""

import os
import sys
import pickle
import argparse
import logging
import urllib.request
from typing import List
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunkers import Chunk, chunk_all_strategies
from src.embed import encode_passages, get_embedding_dimension

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Per-language parquet files in the HuggingFace repository.
# Keys are short ISO-style codes we use internally; values are HF filenames.
_LANG_FILES = {
    "as": "asmtrain.parquet",   # Assamese
    "bn": "bentrain.parquet",   # Bengali
    "gu": "gujtrain.parquet",   # Gujarati
    "hi": "hintrain.parquet",   # Hindi
    "kn": "kantrain.parquet",   # Kannada
    "ml": "maltrain.parquet",   # Malayalam
    "mr": "martrain.parquet",   # Marathi
    "ne": "neptrain.parquet",   # Nepali
    "or": "oritrain.parquet",   # Odia
    "pa": "pantrain.parquet",   # Punjabi
    "sa": "santrain.parquet",   # Sanskrit
    "ta": "tamtrain.parquet",   # Tamil
    "ur": "urdtrain.parquet",   # Urdu
}

_BASE_URL = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/train"


def _download_lang_file(lang_code: str, filename: str, cache_dir: str) -> str:
    """Download a language parquet file if not already cached."""
    local_path = os.path.join(cache_dir, filename)
    if os.path.exists(local_path):
        size_gb = os.path.getsize(local_path) / 1e9
        logger.info(f"  [{lang_code}] Using cached {filename} ({size_gb:.2f} GB)")
        return local_path

    url = f"{_BASE_URL}/{filename}"
    logger.info(f"  [{lang_code}] Downloading {filename} (~3.7 GB)...")
    urllib.request.urlretrieve(url, local_path)
    size_gb = os.path.getsize(local_path) / 1e9
    logger.info(f"  [{lang_code}] Downloaded {filename} ({size_gb:.2f} GB)")
    return local_path


def _extract_passages_from_file(
    lang_code: str,
    local_path: str,
    max_passages: int,
) -> List[dict]:
    """Extract translated passages from a parquet file using iter_batches.

    Reads small batches (50 rows at a time) to avoid loading multi-GB
    row groups into memory. Extracts Translated_passages from the nested
    passages struct, deduplicates, and returns up to max_passages.
    """
    pf = pq.ParquetFile(local_path)
    passages = []
    seen_texts = set()

    batch_iter = pf.iter_batches(
        batch_size=50,
        columns=["query_id", "passages"],
    )

    for batch in batch_iter:
        if len(passages) >= max_passages:
            break

        passages_col = batch.column("passages")

        for i in range(batch.num_rows):
            if len(passages) >= max_passages:
                break

            row = passages_col[i].as_py()
            translated = row.get("Translated_passages", [])
            is_selected = row.get("is_selected", [])
            query_id = batch.column("query_id")[i].as_py()

            for idx, passage_text in enumerate(translated):
                if len(passages) >= max_passages:
                    break

                if not passage_text or not passage_text.strip():
                    continue

                text = passage_text.strip()
                if len(text) < 50:
                    continue

                # Deduplicate by first 200 chars
                text_key = text[:200]
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)

                passages.append({
                    "doc_id": f"{lang_code}_{query_id}_{idx}",
                    "text": text,
                    "language": lang_code,
                })

    return passages


def load_passages_from_dataset(
    max_per_lang: int = 50,
    cache_dir: str = "data/parquet_cache",
    languages: List[str] = None,
) -> List[dict]:
    """Load real passages from ai4bharat/MSMARCO-XI.

    Downloads each per-language parquet file (~3.7 GB each), caches it,
    and extracts up to max_per_lang passages per language.

    Args:
        max_per_lang: Max passages per language.
        cache_dir: Directory to cache downloaded parquet files.
        languages: Which language codes to load (default: all 13).

    Returns:
        List of dicts with keys: doc_id, text, language.
    """
    os.makedirs(cache_dir, exist_ok=True)

    if languages is None:
        languages = list(_LANG_FILES.keys())

    all_passages = []
    lang_counts = Counter()

    for lang_code in languages:
        filename = _LANG_FILES.get(lang_code)
        if not filename:
            logger.warning(f"Unknown language code: {lang_code}, skipping")
            continue

        try:
            local_path = _download_lang_file(lang_code, filename, cache_dir)
            passages = _extract_passages_from_file(lang_code, local_path, max_per_lang)
            all_passages.extend(passages)
            lang_counts[lang_code] = len(passages)
            logger.info(f"  [{lang_code}] Extracted {len(passages)} passages")
        except Exception as e:
            logger.error(f"  [{lang_code}] Failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info(f"Total passages: {len(all_passages)}")
    logger.info(f"Language distribution: {dict(sorted(lang_counts.items()))}")
    return all_passages


def build_chunks(passages: List[dict]) -> List[Chunk]:
    """Run all chunking strategies on each passage."""
    logger.info(f"Chunking {len(passages)} passages with all 3 strategies...")

    all_chunks = []
    for passage in passages:
        chunks = chunk_all_strategies(
            text=passage["text"],
            doc_id=passage["doc_id"],
            language=passage["language"],
        )
        all_chunks.extend(chunks)

    logger.info(f"Produced {len(all_chunks)} total chunks")

    strategy_counts = Counter()
    for c in all_chunks:
        strategy_counts[c.strategy] += 1
    for strategy, count in sorted(strategy_counts.items()):
        logger.info(f"  {strategy}: {count} chunks")

    lang_counts = Counter()
    for c in all_chunks:
        lang_counts[c.language] += 1
    logger.info(f"Chunk language distribution: {dict(sorted(lang_counts.items()))}")

    return all_chunks


def build_qdrant_index(chunks: List[Chunk], data_dir: str, batch_size: int = 64):
    """Embed all chunks and build a Qdrant index locally on disk."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct

    logger.info(f"Embedding {len(chunks)} chunks...")
    texts = [c.text for c in chunks]
    embeddings = encode_passages(texts, batch_size=batch_size)

    qdrant_path = os.path.join(data_dir, "qdrant_db")
    logger.info(f"Building Qdrant index at {qdrant_path}")
    client = QdrantClient(path=qdrant_path)

    collection_name = "msmarco_chunks"
    dim = embeddings.shape[1]

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = []
    for i, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "context_text": chunk.context_text,
                    "doc_id": chunk.doc_id,
                    "strategy": chunk.strategy,
                    "position": chunk.position,
                    "language": chunk.language,
                }
            )
        )

    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=100,
    )
    logger.info(f"Uploaded {len(points)} points to Qdrant '{collection_name}'")


def build_bm25_index(chunks: List[Chunk]) -> BM25Okapi:
    """Build BM25 index from chunk texts."""
    logger.info("Building BM25 index...")
    tokenized = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    logger.info(f"BM25 index built over {len(tokenized)} documents")
    return bm25


def main():
    parser = argparse.ArgumentParser(
        description="Build Qdrant + BM25 indices from ai4bharat/MSMARCO-XI"
    )
    parser.add_argument(
        "--max-per-lang", type=int, default=50,
        help="Max passages per language (default: 50)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Index output directory (default: data/)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Embedding batch size (default: 64)"
    )
    parser.add_argument(
        "--languages", type=str, nargs="*", default=None,
        help="Language codes to include, e.g. hi bn ta (default: all 13)"
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # Step 1: Load real passages
    passages = load_passages_from_dataset(
        max_per_lang=args.max_per_lang,
        cache_dir=os.path.join(args.data_dir, "parquet_cache"),
        languages=args.languages,
    )
    if not passages:
        logger.error("No passages loaded!")
        sys.exit(1)

    # Step 2: Chunk
    chunks = build_chunks(passages)
    if not chunks:
        logger.error("No chunks produced!")
        sys.exit(1)

    # Step 3: Build Qdrant index
    build_qdrant_index(chunks, data_dir=args.data_dir, batch_size=args.batch_size)

    # Step 4: Build BM25 index
    bm25_index = build_bm25_index(chunks)

    # Step 5: Persist BM25
    bm25_path = os.path.join(args.data_dir, "bm25.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_index, f)
    logger.info(f"BM25 index saved to {bm25_path}")

    logger.info("=" * 50)
    logger.info("INDEX BUILD COMPLETE")
    logger.info(f"  Total chunks: {len(chunks)}")
    logger.info(f"  Output: {os.path.abspath(args.data_dir)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
