import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# Ustawienia stylów pod standardy konferencji IEEE
# ---------------------------------------------------------
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,             # Lekko mniejszy bazowy font dla czytelności
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'lines.linewidth': 1.2,
})

COLOR_INIT_PF = '#B0C4DE'
COLOR_FINAL_PF = '#4682B4'
COLOR_DIVERGENCE = '#8B0000'
COLOR_BASELINE = '#A9A9A9'
COLOR_APG = '#2E8B57'

# =========================================================
# FIGURE 3: Shock Sensitivity
# =========================================================
def generate_fig3():
    # Dodajemy \n do etykiet, aby wymusić zawijanie i uniknąć nachodzenia
    labels = ['Low Conflict\n(Non-Zero Sum)', 'High Conflict\n(Zero-Sum)']
    initial_pf = [0.183, 0.288]
    final_pf = [0.586, 0.617]
    divergence_rates = [60.0, 75.0]

    x = np.arange(len(labels))
    width = 0.32

    fig, ax1 = plt.subplots(figsize=(3.4, 2.6))

    rects1 = ax1.bar(x - width/2, initial_pf, width, label='Initial PF', color=COLOR_INIT_PF, edgecolor='black', linewidth=0.5)
    rects2 = ax1.bar(x + width/2, final_pf, width, label='Final PF', color=COLOR_FINAL_PF, edgecolor='black', linewidth=0.5)

    ax1.set_ylabel('Provenance Fidelity (PF)')
    ax1.set_ylim(0, 1.05) # Wyższy limit, by legenda nie zasłaniała słupków
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    
    ax2 = ax1.twinx()
    ax2.plot(x, divergence_rates, color=COLOR_DIVERGENCE, marker='o', markersize=4, linestyle='--', label='Div. Rate (%)')
    ax2.set_ylabel('Divergence Rate (%)', color=COLOR_DIVERGENCE)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelcolor=COLOR_DIVERGENCE)

    # Legenda w jednym wierszu na górze, aby zaoszczędzić miejsce
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=6.5)

    plt.tight_layout()
    plt.savefig('figures/fig3_shock_pf.png', bbox_inches='tight')
    plt.close()
    print("Zaktualizowano: figures/fig3_shock_pf.png")

# =========================================================
# FIGURE 4: Decision Safety & Validation Rigor
# =========================================================
def generate_fig4():
    # Zwiększamy hspace/wspace, żeby wykresy na siebie nie wchodziły
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.2))
    plt.subplots_adjust(wspace=0.5) # Większy odstęp między lewym a prawym wykresem

    # --- Subplot 1: Calibrated Trust ---
    trust_labels = ['Baseline', 'APG\n(Safe Fail)']
    trust_scores = [2.64, 2.91] 
    
    ax1.bar(trust_labels, trust_scores, color=[COLOR_BASELINE, COLOR_APG], edgecolor='black', linewidth=0.5, width=0.5)
    ax1.set_ylim(0, 4.5) 
    ax1.set_ylabel('Mean Safety Score (1-5)')
    ax1.set_title('Calibrated Trust', fontweight='bold', fontsize=8)
    
    for i, v in enumerate(trust_scores):
        ax1.text(i, v + 0.1, f"{v:.2f}", ha='center', fontsize=7)

    # --- Subplot 2: Model Ablation ---
    # Skracamy etykiety, żeby się nie nakładały
    ablation_labels = ['Homogen.\n(Flash)', 'Heterogen.\n(Pro)']
    ablation_divergence = [14.4, 63.3] 
    
    ax2.bar(ablation_labels, ablation_divergence, color=[COLOR_BASELINE, COLOR_APG], edgecolor='black', linewidth=0.5, width=0.5)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Divergence Rate (%)')
    ax2.set_title('Validation Rigor', fontweight='bold', fontsize=8)

    for i, v in enumerate(ablation_divergence):
        ax2.text(i, v + 2, f"{v}%", ha='center', fontsize=7)

    # Dodatkowe czyszczenie layoutu
    plt.tight_layout()
    plt.savefig('figures/fig4_trust_cost.png', bbox_inches='tight')
    plt.close()
    print("Zaktualizowano: figures/fig4_trust_cost.png")

if __name__ == "__main__":
    generate_fig3()
    generate_fig4()
    print("Gotowe. Wykresy są teraz czytelne i nie nachodzą na siebie.")