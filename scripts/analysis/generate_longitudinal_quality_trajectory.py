import argparse
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _compute_ci95(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["sem_overall"] = summary["std_overall"] / summary["n"].replace(0, pd.NA) ** 0.5
    summary["ci95_halfwidth"] = 1.96 * summary["sem_overall"]
    summary["ci95_low"] = summary["mean_overall"] - summary["ci95_halfwidth"]
    summary["ci95_high"] = summary["mean_overall"] + summary["ci95_halfwidth"]
    summary[["sem_overall", "ci95_halfwidth", "ci95_low", "ci95_high"]] = summary[[
        "sem_overall",
        "ci95_halfwidth",
        "ci95_low",
        "ci95_high",
    ]].fillna(0.0)
    return summary


def _extract_shock_turns(df: pd.DataFrame) -> list[tuple[int, str]]:
    shock_rows = df.loc[
        df["shock_type_turn"].notna() & (df["shock_type_turn"].astype(str).str.upper() != "NA")
    ]
    if shock_rows.empty:
        return []

    shock_rows = shock_rows[["turn_number", "shock_type_turn"]].drop_duplicates()
    shock_rows["turn_number"] = shock_rows["turn_number"].astype(int)
    shock_rows["shock_type_turn"] = shock_rows["shock_type_turn"].astype(str)
    return sorted({(row.turn_number, row.shock_type_turn) for row in shock_rows.itertuples(index=False)})


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["turn_number"] = _coerce_numeric(df["turn_number"]).astype("Int64")
    df["overall"] = _coerce_numeric(df["overall"])

    if "condition_label" not in df.columns:
        df["condition_label"] = df["condition"]

    summary = (
        df.groupby(["condition", "condition_label", "turn_number"], dropna=False)["overall"]
        .agg(mean_overall="mean", std_overall="std", n="count")
        .reset_index()
    )

    summary = _compute_ci95(summary)
    summary = summary.sort_values(["condition", "turn_number"]).reset_index(drop=True)
    return summary


def plot_trajectory(summary: pd.DataFrame, shock_turns: list[tuple[int, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4.8))

    condition_order = ["no_cfr_baseline", "cfr_no_kg", "kg_cfr_full"]
    palette = sns.color_palette("colorblind", n_colors=3)
    palette_map = {cond: palette[idx] for idx, cond in enumerate(condition_order)}

    for condition in summary["condition"].unique():
        cond_data = summary[summary["condition"] == condition]
        label = cond_data["condition_label"].iloc[0]
        color = palette_map.get(condition)

        ax.plot(
            cond_data["turn_number"].astype(int),
            cond_data["mean_overall"],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
        ax.fill_between(
            cond_data["turn_number"].astype(int),
            cond_data["ci95_low"],
            cond_data["ci95_high"],
            alpha=0.18,
            color=color,
        )

    y_max = summary["ci95_high"].max()
    y_min = summary["ci95_low"].min()
    y_range = y_max - y_min

    # Place vertical shock lines and a single descriptive label per affected turn.
    base_offset = 0.03 * y_range
    # get unique turn numbers (deduplicate different shock types at same turn)
    unique_turns = sorted({int(tn) for tn, _ in shock_turns})
    for idx, turn_number in enumerate(unique_turns):
        ax.axvline(turn_number, linestyle="--", color="#444444", linewidth=1, alpha=0.6)

        # alternate vertical placement to reduce collisions between nearby turns
        parity = idx % 2
        offset_mult = 1 + (parity * 0.8)
        y_text = y_max + (base_offset * offset_mult)

        display_label = "Shock injection"
        ax.text(
            turn_number + 0.05,
            y_text,
            display_label,
            fontsize=9,
            color="#444444",
            bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 1},
        )

    ax.set_title("Longitudinal quality trajectory (overall score)")
    ax.set_xlabel("Turn number")
    ax.set_ylabel("Mean overall score")
    ax.set_xticks(range(1, 12))
    ax.set_ylim(y_min - (0.05 * y_range), y_max + (0.14 * y_range))

    # Place legend outside to the right to avoid overlapping annotations
    ax.legend(title="Condition", frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()

    png_path = output_dir / "longitudinal_quality_trajectory_e2.png"
    pdf_path = output_dir / "longitudinal_quality_trajectory_e2.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate turn-level mean overall scores and CI95.")
    parser.add_argument(
        "--input",
        default="logs/corrected_e2_batch_20260314_multiscenario/E2_Turns_judge_fixed.csv",
        help="Path to the E2 turns CSV file.",
    )
    parser.add_argument(
        "--summary-out",
        default="analysis/outputs/e2/turn_condition_overall_ci95.csv",
        help="Path to write the turn-level summary CSV.",
    )
    parser.add_argument(
        "--plot-dir",
        default="analysis/plots/outputs",
        help="Directory for the longitudinal trajectory figure.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    summary_path = Path(args.summary_out)
    plot_dir = Path(args.plot_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    summary = build_summary(df)
    shock_turns = _extract_shock_turns(df)

    summary.to_csv(summary_path, index=False)
    plot_trajectory(summary, shock_turns, plot_dir)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote plot files to: {plot_dir}")


if __name__ == "__main__":
    main()
