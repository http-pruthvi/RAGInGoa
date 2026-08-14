"""
run_latency_test_with_stt.py — End-to-end latency evaluation including STT

Runs a set of test audio clips through the full pipeline (voice-to-answer)
to measure end-to-end latency including the Sarvam STT API.

Usage:
    python -m eval.run_latency_test_with_stt [--audio-dir eval/audio] [--output eval/latency_report_with_stt.json]

Prerequisites:
    Please drop sample .wav files (with spoken questions in Hindi or English) 
    into the eval/audio/ directory before running this script.
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Tuple

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


def load_test_audio(audio_dir: str) -> List[Tuple[str, bytes]]:
    """Load test audio clips from a directory."""
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir, exist_ok=True)
        
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    if not wav_files:
        logger.warning(
            f"No .wav files found in {audio_dir}. "
            "Please drop sample .wav files into this directory to run the STT latency benchmark."
        )
        return []

    audio_data = []
    for f in wav_files:
        filepath = os.path.join(audio_dir, f)
        with open(filepath, "rb") as audio_file:
            audio_data.append((f, audio_file.read()))
            
    logger.info(f"Loaded {len(audio_data)} audio clips from {audio_dir}")
    return audio_data


def run_evaluation(
    pipeline: RAGPipeline,
    audio_clips: List[Tuple[str, bytes]],
) -> List[Dict]:
    all_timings = []
    results_summary = []

    for i, (filename, audio_bytes) in enumerate(audio_clips):
        logger.info(f"[{i+1}/{len(audio_clips)}] Audio: {filename}...")

        request = PipelineRequest(audio_bytes=audio_bytes)
        response = pipeline.run(request)

        all_timings.append(response.stage_timings)

        # Log per-query summary
        total_ms = sum(response.stage_timings.values()) * 1000
        results_summary.append({
            "filename": filename,
            "transcription": response.query_text,
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
        logger.info(f"  {status_icon} {response.status.value} (Transcript: '{response.query_text}') ({total_ms:.1f}ms)")

    return all_timings, results_summary


def main():
    parser = argparse.ArgumentParser(
        description="Run STT-included latency evaluation on the RAG pipeline"
    )
    parser.add_argument(
        "--audio-dir", type=str, default="eval/audio",
        help="Path to directory containing test .wav files (default: eval/audio)"
    )
    parser.add_argument(
        "--output", type=str, default="eval/latency_report_with_stt.json",
        help="Path to write JSON report (default: eval/latency_report_with_stt.json)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Directory with index files (default: data/)"
    )
    args = parser.parse_args()

    # Load queries
    audio_clips = load_test_audio(args.audio_dir)
    if not audio_clips:
        return

    # Load pipeline
    logger.info("Loading embedding model...")
    get_model()

    logger.info("Loading retrieval indices...")
    retriever = HybridRetriever(index_dir=args.data_dir)
    pipeline = RAGPipeline(retriever=retriever)
    logger.info("Pipeline ready. Qdrant index loaded successfully.")

    # Run evaluation
    logger.info(f"Running {len(audio_clips)} queries...")
    all_timings, results_summary = run_evaluation(pipeline, audio_clips)

    # Compute latency report
    report = compute_latency_report(all_timings)
    report["individual_results"] = results_summary
    report["notes"] = {
        "end_to_end": "This report includes STT latency and measures true voice-to-answer time."
    }

    # Write report
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Count status occurrences
    from collections import Counter
    status_counts = Counter(r["status"] for r in results_summary)

    logger.info(f"\nLatency report written to {args.output}")
    logger.info(f"\n=== LATENCY SUMMARY (WITH STT) ===")
    logger.info(f"Queries: {report['num_queries']}")
    logger.info(f"Statuses:")
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count}")
    logger.info(f"\nOverall (End-to-End): Mean={report['overall']['mean_ms']}ms, P50={report['overall']['p50_ms']}ms, "
                f"P70={report['overall']['p70_ms']}ms, "
                f"P100={report['overall']['p100_ms']}ms (n={report['overall']['n']})")

    logger.info(f"\nPer-stage Breakdown:")
    for stage, stats in report["per_stage"].items():
        logger.info(f"  {stage}: Mean={stats['mean_ms']}ms, P50={stats['p50_ms']}ms, "
                    f"P70={stats['p70_ms']}ms, P100={stats['p100_ms']}ms (n={stats['n']})")


if __name__ == "__main__":
    main()
