"""
Mini-sweep for provenance fidelity threshold.

Runs the Phase 4 pipeline on a small fixed sample for each threshold
and prints divergence rate and mean iteration count.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import evaluate_provenance_batch as epb
from src.common.api_abstraction import get_philosopher_api
from src.apg.phase4_synthesis import build_synthesis_subgraph


def _parse_list(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_thresholds(value: str) -> List[float]:
    return [float(item) for item in _parse_list(value)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mini-sweep for Phase 4 provenance fidelity threshold"
    )
    parser.add_argument(
        "--archive-dir",
        default="logs/archive/article2",
        help="Root directory containing archived JSON logs",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of sessions to evaluate per threshold",
    )
    parser.add_argument(
        "--conditions",
        default="kg_cfr_full",
        help="Comma-separated list of conditions to include",
    )
    parser.add_argument(
        "--prefixes",
        default="exp_e2_20260314_052549",
        help="Comma-separated list of session_id prefixes to include",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--thresholds",
        default="0.80,0.95,0.99",
        help="Comma-separated list of thresholds to sweep",
    )
    parser.add_argument(
        "--proposer-model",
        default=None,
        help="Override proposer model name",
    )
    parser.add_argument(
        "--validator-model",
        default=None,
        help="Override validator model name",
    )
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    if not archive_dir.exists():
        print(f"[ERROR] Archive directory not found: {archive_dir}")
        return 1

    prefixes = _parse_list(args.prefixes) or None
    conditions = [epb._normalize_condition(c) for c in _parse_list(args.conditions)]
    thresholds = _parse_thresholds(args.thresholds)

    log_paths = epb._pick_random_logs(
        archive_dir,
        args.sample_size,
        args.seed,
        conditions,
        prefixes,
    )
    if not log_paths:
        print(f"[ERROR] No JSON logs found under: {archive_dir}")
        return 1

    api = get_philosopher_api()
    if args.proposer_model:
        proposer_llm = api.get_llm(model_override=args.proposer_model)
    else:
        proposer_llm = api.get_llm()

    if args.validator_model:
        validator_llm = api.get_llm(model_override=args.validator_model)
    else:
        validator_llm = None

    print(f"Selected sessions: {len(log_paths)}")
    for threshold in thresholds:
        print(f"\n--- Threshold {threshold:.2f} ---")
        synthesis_graph = build_synthesis_subgraph(
            llm=proposer_llm,
            validator_llm=validator_llm,
            validator_threshold=threshold,
        )

        rows = []
        failures = 0
        for idx, (condition, path) in enumerate(log_paths, start=1):
            print(f"[{idx}/{len(log_paths)}] Evaluating: {path}")
            try:
                result = epb._run_phase4_for_log(path, synthesis_graph)
                result["condition"] = condition
                result["prefix"] = epb._extract_prefix(path, prefixes)
                metrics_row = epb._extract_metrics(result)
                rows.append(metrics_row)
            except Exception as exc:
                failures += 1
                print(f"[WARN] Failed on {path}: {exc}")

        if not rows:
            print("[ERROR] No results collected for this threshold.")
            continue

        divergence_count = sum(1 for row in rows if row.get("was_divergence_report"))
        iterations = [
            row.get("iterations")
            for row in rows
            if isinstance(row.get("iterations"), (int, float))
        ]
        mean_iterations = sum(iterations) / len(iterations) if iterations else 0.0
        divergence_rate = (divergence_count / len(rows)) * 100.0

        print(
            "Summary: "
            f"threshold={threshold:.2f} "
            f"divergence_rate={divergence_rate:.1f}% "
            f"mean_iterations={mean_iterations:.2f} "
            f"n={len(rows)} failures={failures}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
