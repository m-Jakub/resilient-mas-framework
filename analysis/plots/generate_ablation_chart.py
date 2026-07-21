import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_ablation_barchart():
    """
    Generuje i zapisuje grupowy wykres słupkowy dla wyników Ablation Study (RQ3),
    używając pełnych, akademickich nazw metryk.
    """
    
    # 1. Twoje dane (z Tabeli 4.4.1)
    # Używamy PEŁNYCH NAZW w słowniku, aby legenda była automatyczna i poprawna
    data_wide = {
        'Condition': [
            'Vanilla RAG Only', 
            'Vanilla + ID-RAG', 
            'Vanilla + ToM', 
            'Full System'
        ],
        'Doctrinal Accuracy': [  # Poprawna nazwa metryki
            0.51,
            0.90,
            0.79,
            1.00
        ],
        'Cross-Referencing': [   # Poprawna nazwa metryki
            0.05,
            0.25,
            0.40,
            0.45
        ]
    }
    
    df_wide = pd.DataFrame(data_wide)
    
    # 2. Przekształć dane z formatu "wide" do "long" (wymagane przez Seaborn)
    df_long = df_wide.melt('Condition', var_name='Metric', value_name='Score')
    
    # 3. Ustawienia wykresu (styl akademicki)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5)) # Nieco szerszy, aby zmieścić grupy
    
    # 4. Stwórz grupowy wykres słupkowy
    barplot = sns.barplot(
        data=df_long,
        x='Condition',
        y='Score',
        hue='Metric', # 'Metric' zawiera teraz "Doctrinal Accuracy" i "Cross-Referencing"
        palette="colorblind" 
    )
    
    # 5. Ustaw Tytuły (po angielsku)
    plt.title('Complementary Contributions of ID-RAG and ToM-Lite', fontsize=14)
    plt.xlabel('Experimental Condition', fontsize=10)
    plt.ylabel('Metric Score (Proportion)', fontsize=10)
    
    # 6. Popraw czytelność osi X (etykiety są długie)
    plt.xticks(rotation=15, ha='right') # Lekka rotacja etykiet
    
    # 7. Ustaw limity osi Y (od 0 do 1)
    plt.ylim(0, 1.1) 
    
    # 8. Popraw legendę (teraz automatycznie pobierze pełne nazwy)
    plt.legend(title='Metric', loc='upper left')
    
    # 9. Zapisz plik (najważniejszy krok)
    output_filename = 'charts/ablation_chart.pdf' # Nowa, czysta nazwa
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    
    print(f"Wykres ablacyjny pomyślnie zapisano jako: {output_filename}")


# --- Uruchomienie skryptu ---
if __name__ == "__main__":
    create_ablation_barchart()