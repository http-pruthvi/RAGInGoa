"""
latency.py — Stage-level latency measurement utilities

Provides a context manager for timing individual pipeline stages and
percentile computation for latency reporting. Used by the pipeline
orchestrator to attach per-stage timing to every response, and by
the eval harness to compute P50/P70/P100 latency breakdowns.

All timing is done with time.perf_counter() for monotonic, high-resolution
measurements that aren't affected by system clock adjustments.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List
from contextlib import contextmanager


@dataclass
class StageTimer:
    """Accumulates timing data for pipeline stages.

    Each stage records its elapsed time in seconds. The orchestrator
    attaches one StageTimer per request to capture all stage latencies.
    """
    timings: Dict[str, float] = field(default_factory=dict)

    @contextmanager
    def time_stage(self, stage_name: str):
        """Context manager that records wall-clock time for a named stage.

        Usage:
            timer = StageTimer()
            with timer.time_stage("retrieval"):
                results = retriever.search(query)
            print(timer.timings["retrieval"])  # e.g. 0.023 seconds
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[stage_name] = elapsed

    @property
    def total_seconds(self) -> float:
        """Sum of all recorded stage timings."""
        return sum(self.timings.values())


def percentile(values: List[float], p: int) -> float:
    """Compute the p-th percentile using the nearest-rank method.

    This is a simple implementation that doesn't require numpy.
    Nearest-rank: the p-th percentile is the smallest value in the sorted
    data such that at least p% of the data is <= that value.

    Args:
        values: List of numeric values (must be non-empty).
        p: Percentile to compute (0-100).

    Returns:
        The p-th percentile value.

    Raises:
        ValueError: If values is empty or p is out of range.
    """
    if not values:
        raise ValueError("Cannot compute percentile of empty list")
    if not (0 <= p <= 100):
        raise ValueError(f"Percentile must be 0-100, got {p}")

    sorted_values = sorted(values)
    n = len(sorted_values)

    if p == 0:
        return sorted_values[0]

    # Nearest-rank formula: rank = ceil(p/100 * n)
    # Using integer arithmetic to avoid float precision issues
    rank = (p * n + 99) // 100  # This is ceil(p*n/100)
    # Clamp to valid index range
    rank = min(rank, n)
    return sorted_values[rank - 1]


def compute_latency_report(
    all_timings: List[Dict[str, float]],
) -> Dict:
    """Compute P50/P70/P100 latency stats from a batch of pipeline runs.

    Args:
        all_timings: List of stage-timing dicts, one per pipeline run.
                     Each dict maps stage name -> elapsed seconds.

    Returns:
        Dict with structure:
        {
            "per_stage": {
                "retrieval": {"p50": 0.02, "p70": 0.03, "p100": 0.05},
                ...
            },
            "overall": {"p50": 0.3, "p70": 0.5, "p100": 1.2},
            "local_stages": {"p50": ..., "p70": ..., "p100": ...},
            "num_queries": 20
        }
    """
    if not all_timings:
        return {"per_stage": {}, "overall": {}, "local_stages": {}, "num_queries": 0}

    # Collect all stage names across runs
    stage_names = set()
    for t in all_timings:
        stage_names.update(t.keys())

    # Per-stage percentiles
    per_stage = {}
    for stage in sorted(stage_names):
        values = [t[stage] for t in all_timings if stage in t]
        if values:
            per_stage[stage] = {
                "p50_ms": round(percentile(values, 50) * 1000, 2),
                "p70_ms": round(percentile(values, 70) * 1000, 2),
                "p100_ms": round(percentile(values, 100) * 1000, 2),
                "mean_ms": round((sum(values) / len(values)) * 1000, 2),
                "n": len(values),
            }

    # Overall (sum of all stages per run)
    totals = [sum(t.values()) for t in all_timings]
    overall = {
        "p50_ms": round(percentile(totals, 50) * 1000, 2),
        "p70_ms": round(percentile(totals, 70) * 1000, 2),
        "p100_ms": round(percentile(totals, 100) * 1000, 2),
        "mean_ms": round((sum(totals) / len(totals)) * 1000, 2),
        "n": len(totals),
    }

    # Local-only stages: everything except stt and generation (the external API calls)
    # This is the number that should be compared against the 200ms target
    external_stages = {"stt", "generation"}
    local_totals = [
        sum(v for k, v in t.items() if k not in external_stages)
        for t in all_timings
    ]
    local_stages = {
        "p50_ms": round(percentile(local_totals, 50) * 1000, 2),
        "p70_ms": round(percentile(local_totals, 70) * 1000, 2),
        "p100_ms": round(percentile(local_totals, 100) * 1000, 2),
        "mean_ms": round((sum(local_totals) / len(local_totals)) * 1000, 2),
        "n": len(local_totals),
    }

    return {
        "per_stage": per_stage,
        "overall": overall,
        "local_stages_only": local_stages,
        "num_queries": len(all_timings),
    }
