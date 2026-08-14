"""
run_latency_test.py — Batch latency evaluation for the RAG pipeline

Runs a set of test queries through the pipeline using text input (bypassing
STT) to isolate chunking + retrieval + guardrail + generation latency.

Computes P50/P70/P100 percentiles per stage and overall, and writes a
JSON report. The report separates local stages (sub-200ms target) from
external API stages (STT, generation) for honest latency accounting.

Usage:
    python -m eval.run_latency_test [--queries eval/test_queries.txt] [--output eval/latency_report.json]
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.retrieve import HybridRetriever
from src.pipeline import RAGPipeline, PipelineRequest, PipelineStatus
from src.embed import get_model
from src.latency import compute_latency_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_test_queries(filepath: str) -> List[str]:
    """Load test queries from a text file (one query per line)."""
    with open(filepath, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(queries)} test queries from {filepath}")
    return queries


def run_evaluation(
    pipeline: RAGPipeline,
    queries: List[str],
) -> List[Dict]:
    """Run all queries through the pipeline, collect timings.

    Returns list of timing dicts (one per query). Queries that fail
    with INTERNAL_ERROR are still included in timing data — errors
    are part of the latency distribution.
    """
    all_timings = []
    results_summary = []

    for i, query in enumerate(queries):
        logger.info(f"[{i+1}/{len(queries)}] Query: {query[:60]}...")

        request = PipelineRequest(query_text=query)
        response = pipeline.run(request)

        all_timings.append(response.stage_timings)

        # Log per-query summary
        total_ms = sum(response.stage_timings.values()) * 1000
        results_summary.append({
            "query": query,
            "status": response.status.value,
            "total_ms": round(total_ms, 2),
            "stage_timings_ms": {
                k: round(v * 1000, 2)
                for k, v in response.stage_timings.items()
            },
            "answer_preview": response.answer[:100] if response.answer else "",
            "error": response.error_message if response.error_message else None,
        })

        status_icon = "✓" if response.status == PipelineStatus.SUCCESS else "✗"
        logger.info(f"  {status_icon} {response.status.value} ({total_ms:.1f}ms)")

        # Rate-limiting delay between queries so the benchmark harness doesn't trigger Groq limits
        if i < len(queries) - 1:
            import time
            time.sleep(1.5)

    return all_timings, results_summary


def main():
    parser = argparse.ArgumentParser(
        description="Run latency evaluation on the RAG pipeline"
    )
    parser.add_argument(
        "--queries", type=str, default="eval/test_queries.txt",
        help="Path to test queries file (default: eval/test_queries.txt)"
    )
    parser.add_argument(
        "--output", type=str, default="eval/latency_report.json",
        help="Path to write JSON report (default: eval/latency_report.json)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Directory with index files (default: data/)"
    )
    args = parser.parse_args()

    # Load pipeline
    logger.info("Loading embedding model...")
    get_model()

    logger.info("Loading retrieval indices...")
    retriever = HybridRetriever(index_dir=args.data_dir)
    pipeline = RAGPipeline(retriever=retriever)
    logger.info("Pipeline ready. Qdrant index loaded successfully.")

    # Load queries
    queries = load_test_queries(args.queries)

    # Run evaluation
    logger.info(f"Running {len(queries)} queries...")
    all_timings, results_summary = run_evaluation(pipeline, queries)

    # Compute latency report
    report = compute_latency_report(all_timings)
    report["individual_results"] = results_summary
    report["notes"] = {
        "latency_target": "200ms for local stages (chunking + retrieval + guardrails)",
        "external_api_stages": [
            "stt (Sarvam API, typically 500-2000ms per call)",
            "generation (Groq API, typically 200-1000ms per call)",
        ],
        "honest_assessment": (
            "The 200ms target realistically applies ONLY to the local "
            "chunking + retrieval + guardrail-evaluation stages. STT and LLM "
            "generation are external API calls with inherent network latency "
            "that cannot be reduced below ~200ms individually. The "
            "'local_stages_only' field in this report shows the latency for "
            "just the local computation, which is the number that should be "
            "compared against the 200ms target."
        ),
    }

    # Write report
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Count status occurrences
    from collections import Counter
    status_counts = Counter(r["status"] for r in results_summary)

    logger.info(f"\nLatency report written to {args.output}")
    logger.info(f"\n=== LATENCY SUMMARY ===")
    logger.info(f"Queries: {report['num_queries']}")
    logger.info(f"Statuses:")
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count}")
    logger.info(f"\nOverall: Mean={report['overall']['mean_ms']}ms, P50={report['overall']['p50_ms']}ms, "
                f"P70={report['overall']['p70_ms']}ms, "
                f"P100={report['overall']['p100_ms']}ms (n={report['overall']['n']})")
    logger.info(f"Local stages only (chunking + retrieval + guardrails): Mean={report['local_stages_only']['mean_ms']}ms, P50={report['local_stages_only']['p50_ms']}ms, "
                f"P70={report['local_stages_only']['p70_ms']}ms, "
                f"P100={report['local_stages_only']['p100_ms']}ms (n={report['local_stages_only']['n']})")

    logger.info(f"\nPer-stage Breakdown:")
    for stage, stats in report["per_stage"].items():
        logger.info(f"  {stage}: Mean={stats['mean_ms']}ms, P50={stats['p50_ms']}ms, "
                    f"P70={stats['p70_ms']}ms, P100={stats['p100_ms']}ms (n={stats['n']})")


if __name__ == "__main__":
    main()
