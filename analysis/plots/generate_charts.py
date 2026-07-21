import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_acs_barchart():
    """
    Generuje i zapisuje wykres słupkowy dla wyników Delta_ACS (RQ1 i H2.2).
    """
    
    # 1. Twoje dane (z Twojej tabeli)
    data = {
        'Condition': [
            'B Chat', 
            'B SingleRAG', 
            'Homo', 
            'Hetero'
        ],
        'Mean_Delta_ACS': [
            0.20,
            0.40,
            -0.29,
            2.20
        ]
    }
    
    df = pd.DataFrame(data)
    
    # 2. Ustawienia wykresu (styl akademicki)
    sns.set_theme(style="whitegrid") # Czysty, profesjonalny styl tła
    
    # Ustaw rozmiar figury (ważne dla LaTeX-a)
    # (Szerokość, Wysokość) w calach. (6, 4) jest dobre dla półkolumny.
    plt.figure(figsize=(6, 4))
    
    # 3. Stwórz wykres
    # Używamy `palette="colorblind"` dla dostępności (dobra praktyka naukowa)
    barplot = sns.barplot(
        x='Condition',
        y='Mean_Delta_ACS',
        data=df,
        palette="colorblind"
    )
    
    # 4. Dodaj etykiety danych (liczby nad słupkami)
    for p in barplot.patches:
        barplot.annotate(
            format(p.get_height(), '.2f'), # Formatuj do 2 miejsc po przecinku
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha = 'center', 
            va = 'center',
            xytext = (0, 9), # Odsuń etykietę 9 punktów w górę
            textcoords = 'offset points'
        )
    
    # 5. Ustaw Tytuły (po angielsku, tak jak w LaTeX-u)
    # Używamy $...$ aby LaTeX (w Matplotlib) renderował $\Delta$
    plt.title('Mean Argument Complexity Score Gain ($\Delta$ACS) by Condition', fontsize=12)
    plt.xlabel('Experimental Condition', fontsize=10)
    plt.ylabel('Mean $\Delta$ACS Gain (Post-Test - Pre-Test)', fontsize=10)
    
    # 6. Ustaw limity osi Y (krytyczne, aby pokazać ujemny słupek!)
    plt.ylim(-0.5, df['Mean_Delta_ACS'].max() * 1.2) # np. od -0.5 do ~2.6
    
    # Dodaj linię zerową dla klarowności
    plt.axhline(0, color='grey', linewidth=0.8)
    
    # 7. Zapisz plik (najważniejszy krok)
    # Zapisujemy jako PDF dla idealnej jakości wektorowej w LaTeX-u
    output_filename = 'charts/outputs/student_acs_barchart.pdf'
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    
    print(f"Wykres pomyślnie zapisano jako: {output_filename}")
    
    # Opcjonalnie: pokaż wykres (jeśli uruchamiasz lokalnie)
    # plt.show()

# --- Uruchomienie skryptu ---
if __name__ == "__main__":
    create_acs_barchart()