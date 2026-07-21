import json
import math
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


SCENARIO_PRR_CSV = "analysis/outputs/e2/prr_by_scenario_threshold_-0.20.csv"
AGGREGATE_JSON = "logs/corrected_e2_batch_20260308_225734/summary/corrected_e2_aggregate.json"

CONDITION_ORDER = ["no_cfr_baseline", "cfr_no_kg", "kg_cfr_full"]
CONDITION_LABELS = {
    "no_cfr_baseline": "No-CFR baseline",
    "cfr_no_kg": "CFR (no KG)",
    "kg_cfr_full": "KG-CFR (full)",
}
CONDITION_COLORS = {
    "no_cfr_baseline": "#888888",
    "cfr_no_kg": "#4C9BE8",
    "kg_cfr_full": "#2ECC71",
}

METRICS = [
    ("prr", "PRR"),
    ("aca", "ACA"),
    ("dis", "DIS"),
    ("cc", "CC"),
]

OUTPUT_PDF = "analysis/outputs/figures/figure4_resilience_summary.pdf"
OUTPUT_PNG = "analysis/outputs/figures/figure4_resilience_summary.png"


def load_aggregate(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    # data may have structure {"aggregate": [ ... ]}
    agg_map = {}
    if isinstance(data, dict) and "aggregate" in data:
        for entry in data["aggregate"]:
            agg_map[entry["condition"]] = entry
    elif isinstance(data, list):
        for entry in data:
            agg_map[entry["condition"]] = entry
    else:
        raise ValueError("Unrecognized aggregate JSON structure")

    return agg_map


def load_prr_csv(path: Path) -> dict:
    import csv

    prr = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    # rows: scenario, cfr_no_kg, kg_cfr_full, no_cfr_baseline
    for row in rows:
        scenario = row["scenario"]
        prr[scenario] = {
            "no_cfr_baseline": int(row.get("no_cfr_baseline", 0)),
            "cfr_no_kg": int(row.get("cfr_no_kg", 0)),
            "kg_cfr_full": int(row.get("kg_cfr_full", 0)),
        }

    return prr


def build_pooled(agg_map: dict, prr_map: dict, n_per_cell: int = 30) -> dict:
    pooled = {cond: {} for cond in CONDITION_ORDER}

    # compute per-scenario prr proportions
    scenarios = list(prr_map.keys())
    prr_props = {scenario: {cond: prr_map[scenario].get(cond, 0) / n_per_cell for cond in CONDITION_ORDER} for scenario in scenarios}

    for cond in CONDITION_ORDER:
        # PRR: mean of per-scenario proportions
        values = [prr_props[s][cond] for s in scenarios]
        pooled[cond]["prr_mean"] = statistics.mean(values)
        # 95% CI across scenarios
        if len(values) > 1:
            std = statistics.stdev(values)
            ci95 = 1.96 * std / math.sqrt(len(values))
        else:
            ci95 = 0.0
        pooled[cond]["prr_ci95"] = ci95

        # other metrics: take from agg_map if present
        entry = agg_map.get(cond, {})
        for metric_key in ("aca", "dis", "cc"):
            mean_key = f"{metric_key}_mean"
            ci_key = f"{metric_key}_ci95"
            # fallback to v5_cc naming for CC if present
            if metric_key == "cc" and mean_key not in entry and "v5_cc_mean" in entry:
                val_mean = entry.get("v5_cc_mean")
                val_ci = entry.get("v5_cc_ci95")
            else:
                val_mean = entry.get(mean_key)
                val_ci = entry.get(ci_key)

            pooled[cond][mean_key] = float(val_mean) if val_mean is not None else float('nan')
            pooled[cond][ci_key] = float(val_ci) if val_ci is not None else float('nan')

    return pooled


def plot_grouped_bars(pooled: dict, output_pdf: Path, output_png: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
    })

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

    group_positions = list(range(len(METRICS)))
    bar_width = 0.22
    offsets = [-bar_width, 0, bar_width]

    for idx, condition in enumerate(CONDITION_ORDER):
        means = []
        errors = []
        for metric, _ in METRICS:
            means.append(pooled[condition].get(f"{metric}_mean", float('nan')))
            errors.append(pooled[condition].get(f"{metric}_ci95", float('nan')))

        x_positions = [pos + offsets[idx] for pos in group_positions]
        ax.bar(
            x_positions,
            means,
            width=bar_width,
            label=CONDITION_LABELS[condition],
            color=CONDITION_COLORS[condition],
            yerr=errors,
            capsize=3,
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_xticks(group_positions, [label for _, label in METRICS])
    ax.set_ylabel("Mean score")
    ax.set_ylim(0.0, 1.0 if any(m[0]=="prr" for m in METRICS) else 2.0)
    ax.yaxis.grid(True, color="#dddddd")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png)


def main() -> None:
    agg_map = load_aggregate(Path(AGGREGATE_JSON))
    prr_map = load_prr_csv(Path(SCENARIO_PRR_CSV))

    pooled = build_pooled(agg_map, prr_map, n_per_cell=30)
    plot_grouped_bars(pooled, Path(OUTPUT_PDF), Path(OUTPUT_PNG))


if __name__ == "__main__":
    main()
