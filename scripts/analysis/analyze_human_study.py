import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def _resolve_input_path(file_name: str) -> Path:
    candidate = Path(file_name)
    if candidate.exists():
        return candidate

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    for base in (script_dir, repo_root, Path.cwd()):
        path = base / file_name
        if path.exists():
            return path

    # Fallback: try to locate a likely CSV match in the repo
    matches = list(repo_root.rglob("*Responses*.csv"))
    if matches:
        match_list = "\n".join(f"- {m}" for m in matches)
        raise FileNotFoundError(
            f"Input CSV not found: {file_name}\n"
            f"Found similar files:\n{match_list}"
        )

    raise FileNotFoundError(f"Input CSV not found: {file_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze human study responses")
    parser.add_argument(
        "--input",
        default="Ocena systemów syntezy decyzji AI w sytuacjach kryzysowych (Responses) - Form Responses 1.csv",
        help="Path to the Google Forms CSV export",
    )
    parser.add_argument(
        "--output",
        default="human_study_visualization.png",
        help="Output figure path",
    )
    args = parser.parse_args()

    # 1. Wczytanie i czyszczenie danych
    input_path = _resolve_input_path(args.input)
    df = pd.read_csv(input_path)

    # Filtrowanie tylko osób, które wyraziły zgodę (pominie puste lub odmowne)
    df = df[df.iloc[:, 1].str.contains("Tak", na=False)]

    # 2. Mapowanie danych do scenariuszy (indeksy kolumn na bazie Twojego pliku)
    # Scenariusz 1 (S009 - Low Shock): Alokacja energii
    s1_groundedness = df.iloc[:, 5]
    s1_trust_a = df.iloc[:, 7].astype(float)
    s1_trust_b = df.iloc[:, 8].astype(float)

    # Scenariusz 2 (S026 - High Shock): Pat zero-sum (Safe Failure)
    s2_groundedness = df.iloc[:, 9]
    s2_trust_a = df.iloc[:, 11].astype(float)
    s2_trust_b = df.iloc[:, 12].astype(float)

    # 3. Obliczenia statystyczne
    def get_stats(a, b):
        return {
            "mean_a": a.mean(),
            "std_a": a.std(),
            "mean_b": b.mean(),
            "std_b": b.std(),
            "p_val": stats.ttest_rel(a, b).pvalue,
        }

    stats_s1 = get_stats(s1_trust_a, s1_trust_b)
    stats_s2 = get_stats(s2_trust_a, s2_trust_b)

    # Procentowe wskazania rzetelności (Pytanie 1)
    def get_accuracy_pct(series):
        counts = series.value_counts(normalize=True) * 100
        return counts.to_dict()

    acc_s1 = get_accuracy_pct(s1_groundedness)
    acc_s2 = get_accuracy_pct(s2_groundedness)

    # 4. WIZUALIZACJA
    labels = ["Scenario 1 (S009)", "Scenario 2 (S026)"]
    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))

    # Panel LEWY: Zaufanie (Decision Utility)
    trust_a = [stats_s1["mean_a"], stats_s2["mean_a"]]
    trust_b = [stats_s1["mean_b"], stats_s2["mean_b"]]
    err_a = [stats_s1["std_a"], stats_s2["std_a"]]
    err_b = [stats_s1["std_b"], stats_s2["std_b"]]

    ax1.bar(
        x - width / 2,
        trust_a,
        width,
        label="Baseline (Raport A)",
        color="#ff9999",
        yerr=err_a,
        capsize=5,
        alpha=0.8,
    )
    ax1.bar(
        x + width / 2,
        trust_b,
        width,
        label="APG (Raport B)",
        color="#66b3ff",
        yerr=err_b,
        capsize=5,
        alpha=0.8,
    )

    ax1.set_ylabel("Mean Trust Score (1-5)", fontsize=12)
    ax1.set_title(
        "A: Decision Utility & Calibrated Trust\n(Mean ± SD)",
        fontweight="bold",
        fontsize=13,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 5.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    ax1.tick_params(axis="both", labelsize=11)
    ax1.legend(fontsize=10)

    # Panel PRAWY: Percepcja Rzetelności (Groundedness)
    # Wyciągamy procenty dla Raportu B, Raportu A i "Innych"
    b_vals = [acc_s1.get("Raport B", 0), acc_s2.get("Raport B", 0)]
    a_vals = [acc_s1.get("Raport A", 0), acc_s2.get("Raport A", 0)]
    other_vals = [100 - b_vals[0] - a_vals[0], 100 - b_vals[1] - a_vals[1]]

    ax2.bar(labels, b_vals, label="Voted B (Accurate)", color="#1f77b4")
    ax2.bar(labels, a_vals, bottom=b_vals, label="Voted A (Inaccurate)", color="#d62728")
    ax2.bar(
        labels,
        other_vals,
        bottom=np.array(b_vals) + np.array(a_vals),
        label="Other/Both",
        color="#9467bd",
    )

    ax2.set_ylabel("Percentage of Respondents (%)", fontsize=12)
    ax2.set_title(
        "B: Groundedness Perception\n(Who was more accurate?)",
        fontweight="bold",
        fontsize=13,
    )
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis="both", labelsize=11)
    ax2.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Wykres został zapisany jako {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())