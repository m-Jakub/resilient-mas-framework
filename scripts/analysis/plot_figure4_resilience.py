import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


SCENARIO_FILES = [
    "analysis/outputs/e2/exp_e2_20260313_223929_aegis_blackout/summary/e2_aggregate_aegis_blackout.json",
    "analysis/outputs/e2/exp_e2_20260314_015807_cerberus_biocontainment/summary/e2_aggregate_cerberus_biocontainment.json",
    "analysis/outputs/e2/exp_e2_20260314_052549_synapse_orbital_strike/summary/e2_aggregate_synapse_orbital_strike.json",
]

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


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def load_scenario(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    scenario = {}
    for entry in data:
        scenario[entry["condition"]] = entry
    return scenario


def pool_scenarios(paths: list[Path]) -> dict:
    pooled = {condition: {} for condition in CONDITION_ORDER}

    scenario_data = [load_scenario(path) for path in paths]

    for condition in CONDITION_ORDER:
        for metric, _ in METRICS:
            mean_key = f"{metric}_mean"
            ci_key = f"{metric}_ci95"

            means = []
            cis = []

            for scenario in scenario_data:
                value = scenario.get(condition, {}).get(mean_key)
                ci_value = scenario.get(condition, {}).get(ci_key)

                if value is not None:
                    means.append(float(value))
                if ci_value is not None:
                    cis.append(float(ci_value))

            if means:
                pooled[condition][mean_key] = _mean(means)
            else:
                pooled[condition][mean_key] = math.nan

            if cis:
                pooled[condition][ci_key] = _mean(cis)
            else:
                pooled[condition][ci_key] = math.nan

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
            means.append(pooled[condition][f"{metric}_mean"])
            errors.append(pooled[condition][f"{metric}_ci95"])

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
    ax.set_ylim(0.0, 2.0)
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
    paths = [Path(path) for path in SCENARIO_FILES]
    pooled = pool_scenarios(paths)

    plot_grouped_bars(pooled, Path(OUTPUT_PDF), Path(OUTPUT_PNG))


if __name__ == "__main__":
    main()

