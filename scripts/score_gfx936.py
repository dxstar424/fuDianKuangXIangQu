#!/usr/bin/env python3
"""Calculate the reproduced competition score from saved SCNet A/B runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence


TIERS = ("4-8K", "8-16K", "16-32K")
WEIGHTS = {"4-8K": 0.2, "8-16K": 0.5, "16-32K": 0.3}


def _number(document: dict[str, Any], *names: str) -> float:
    for name in names:
        value = document.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    raise ValueError(f"missing numeric result key; tried {names!r}")


def read_run(results_root: Path, run: str, tier: str) -> dict[str, float]:
    document = json.loads((results_root / "throughput" / run / f"{tier}.json").read_text())
    return {
        "throughput": _number(
            document, "output_throughput", "throughput_tok_s", "throughput"
        ),
        "ttft_p99_ms": _number(document, "p99_ttft_ms", "ttft_p99_ms"),
        "tpot_p99_ms": _number(document, "p99_tpot_ms", "tpot_p99_ms"),
    }


def score_results(
    *,
    results_root: Path,
    control_runs: Sequence[str],
    candidate_runs: Sequence[str],
    accuracy_coefficient: float,
    max_score: float = 100.0,
) -> dict[str, Any]:
    if not control_runs or not candidate_runs:
        raise ValueError("at least one control and candidate run are required")
    if not 0.0 <= accuracy_coefficient <= 1.0:
        raise ValueError("accuracy coefficient must be in [0, 1]")

    tiers: dict[str, Any] = {}
    weighted_raw = 0.0
    for tier in TIERS:
        controls = [read_run(results_root, run, tier) for run in control_runs]
        candidates = [read_run(results_root, run, tier) for run in candidate_runs]
        control_throughput = statistics.median(row["throughput"] for row in controls)
        candidate_throughput = statistics.median(
            row["throughput"] for row in candidates
        )
        control_ttft = max(row["ttft_p99_ms"] for row in controls)
        candidate_ttft = max(row["ttft_p99_ms"] for row in candidates)
        control_tpot = max(row["tpot_p99_ms"] for row in controls)
        candidate_tpot = max(row["tpot_p99_ms"] for row in candidates)
        relative_gain = (
            (candidate_throughput - control_throughput) / control_throughput
            if control_throughput > 0.0
            else 0.0
        )
        ttft_sla = candidate_ttft <= control_ttft * 1.5
        tpot_sla = candidate_tpot <= control_tpot * 1.5
        sla_pass = ttft_sla and tpot_sla
        curve_score = max_score * (
            0.6 + 0.4 * (1.0 - math.exp(-1.3 * relative_gain))
        )
        curve_score = max(0.0, min(max_score, curve_score)) if sla_pass else 0.0
        weighted_raw += WEIGHTS[tier] * curve_score
        tiers[tier] = {
            "weight": WEIGHTS[tier],
            "control_median_throughput": control_throughput,
            "candidate_median_throughput": candidate_throughput,
            "relative_gain": relative_gain,
            "control_worst_ttft_p99_ms": control_ttft,
            "candidate_worst_ttft_p99_ms": candidate_ttft,
            "control_worst_tpot_p99_ms": control_tpot,
            "candidate_worst_tpot_p99_ms": candidate_tpot,
            "ttft_sla_pass": ttft_sla,
            "tpot_sla_pass": tpot_sla,
            "sla_pass": sla_pass,
            "raw_tier_score": curve_score,
        }
    return {
        "control_runs": list(control_runs),
        "candidate_runs": list(candidate_runs),
        "tiers": tiers,
        "weighted_raw_score": weighted_raw,
        "accuracy_coefficient": accuracy_coefficient,
        "final_score": weighted_raw * accuracy_coefficient,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--control-run", action="append", required=True)
    parser.add_argument("--candidate-run", action="append", required=True)
    parser.add_argument("--accuracy-coefficient", type=float, required=True)
    parser.add_argument("--max-score", type=float, default=100.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = score_results(
        results_root=args.results_root,
        control_runs=args.control_run,
        candidate_runs=args.candidate_run,
        accuracy_coefficient=args.accuracy_coefficient,
        max_score=args.max_score,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
