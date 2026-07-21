"""
Recalculate PRR activation counts with a stricter drop threshold.

Inputs:
- CSV with columns: run_id, condition, turn, is_shock_turn, judge_score
Outputs:
- Printed table of unique run_id activations per condition
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def recalc_prr_activation(
    df: pd.DataFrame,
    drop_threshold: float = -0.20,
) -> pd.DataFrame:
    """Return activation counts per condition for shock turns.

    Activation means at least one shock turn in a run has delta <= drop_threshold.
    """
    required = {"run_id", "condition", "turn", "is_shock_turn", "judge_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.sort_values(["run_id", "turn"]).copy()
    df["delta_judge"] = df.groupby("run_id")["judge_score"].diff()

    shock_df = df[df["is_shock_turn"]].copy()

    activation = (
        shock_df.loc[shock_df["delta_judge"] <= drop_threshold, ["condition", "run_id"]]
        .drop_duplicates()
        .groupby("condition")["run_id"]
        .count()
        .reset_index(name="prr_activation_count")
        .sort_values("prr_activation_count", ascending=False)
    )

    return activation


def main() -> int:
    parser = argparse.ArgumentParser(description="Recalculate PRR activation counts")
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument(
        "--threshold",
        type=float,
        default=-0.20,
        help="Critical drop threshold (default: -0.20)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    summary = recalc_prr_activation(df, drop_threshold=args.threshold)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
