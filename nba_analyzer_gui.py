import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
import os

warnings.filterwarnings('ignore')

# --- Konfiguracja ---
st.set_page_config(page_title="Analiza Rzutów NBA", layout="wide", initial_sidebar_state="expanded")

# --- Kolory NBA ---
NBA_BLUE = '#1D428A'
NBA_RED = '#C8102E'
NBA_WHITE = '#FFFFFF'
# Można dodać kolory specyficzne dla drużyn, jeśli analiza miałaby to uwzględniać

# --- Ścieżka do pliku (Użytkownik musi go tu umieścić) ---
# Należy dostosować nazwę pliku do faktycznej nazwy pobranej z Kaggle
CSV_FILENAME = "NBA Player Shooting Data 2023-24 Regular Season & Playoffs.csv" # <--- ZMIEŃ NA WŁAŚCIWĄ NAZWĘ

# --- Funkcje pomocnicze (zmodyfikowane z oryginalnego skryptu) ---

@st.cache_data # Cache'owanie wczytywania danych
def load_shooting_data(filepath):
    if not os.path.exists(filepath):
        st.error(f"Plik '{filepath}' nie został znaleziony. Pobierz dane z Kaggle i umieść w tym samym katalogu co skrypt.")
        return pd.DataFrame()
    try:
        data = pd.read_csv(filepath)
        # Proste sprawdzenie/mapowanie kolumn (jak w oryginale) - przykład
        required_columns = ['PLAYER_NAME', 'SHOT_TYPE', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE',
                            'SHOT_DISTANCE', 'LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']
        column_mapping = {}
        # ... (logika mapowania jak w oryginale, np. sprawdzanie lower case) ...
        data = data.rename(columns=column_mapping)

        # Sprawdzenie czy kluczowe kolumny istnieją po mapowaniu
        missing_after_map = [col for col in required_columns if col not in data.columns]
        if missing_after_map:
             st.warning(f"Po próbie mapowania nadal brakuje kolumn: {missing_after_map}. Niektóre analizy mogą nie działać.")

        # Konwersja SHOT_MADE_FLAG na int, jeśli jest jako obiekt/float
        if 'SHOT_MADE_FLAG' in data.columns:
             data['SHOT_MADE_FLAG'] = pd.to_numeric(data['SHOT_MADE_FLAG'], errors='coerce').fillna(0).astype(int)

        return data
    except Exception as e:
        st.error(f"Błąd podczas wczytywania danych: {e}")
        return pd.DataFrame()

# Przykład zmodyfikowanej funkcji do tworzenia mapy rzutów (Plotly)
# Ta funkcja powinna teraz ZWRACAĆ figurę Plotly
def create_player_shot_map_plotly(player_shots_df, player_name):
    if 'LOC_X' not in player_shots_df.columns or 'LOC_Y' not in player_shots_df.columns:
         return None # Zwróć None jeśli brakuje danych

    # Mapowanie kolorów
    if 'SHOT_MADE_FLAG' in player_shots_df.columns:
        player_shots_df['result_color'] = player_shots_df['SHOT_MADE_FLAG'].map({1: NBA_BLUE, 0: NBA_RED}) # Używamy kolorów NBA
        color_discrete_map = {1: NBA_BLUE, 0: NBA_RED}
        symbol_map = {1: 'circle', 0: 'x'}
        hover_name = 'SHOT_MADE_FLAG' # Nazwa dla legendy/tooltipa
    else:
        player_shots_df['result_color'] = NBA_BLUE # Domyślny kolor jeśli brak flagi
        color_discrete_map = None
        symbol_map = None
        hover_name = None


    fig = px.scatter(player_shots_df, x='LOC_X', y='LOC_Y',
                     color='result_color' if 'SHOT_MADE_FLAG' in player_shots_df.columns else None, # Kolor punktów
                     #symbol=player_shots_df['SHOT_MADE_FLAG'].map(symbol_map) if 'SHOT_MADE_FLAG' in player_shots_df.columns else None, # Symbol punktów
                     title=f'Mapa rzutów - {player_name}',
                     opacity=0.7,
                     hover_data=['SHOT_DISTANCE', 'SHOT_TYPE'] if 'SHOT_DISTANCE' in player_shots_df.columns and 'SHOT_TYPE' in player_shots_df.columns else None) # Dodatkowe info w tooltipie

    fig.update_traces(marker=dict(size=5))

    # Dodanie kształtu boiska (uproszczone)
    fig.add_shape(type="circle", xref="x", yref="y", x0=-237.5, y0=-237.5, x1=237.5, y1=237.5, line_color=NBA_BLUE, line_width=1) # Linia 3pkt (przybliżenie w stopach)
    fig.add_shape(type="circle", xref="x", yref="y", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, line_color=NBA_RED, fillcolor=NBA_RED, opacity=0.8) # Obręcz
    fig.add_shape(type="line", xref="x", yref="y", x0=-30, y0=-47.5, x1=30, y1=-47.5, line_color=NBA_BLUE, line_width=2) # Tablica

    # Stylizacja layoutu
    fig.update_layout(
        xaxis_title='Pozycja X (stopy)',
        yaxis_title='Pozycja Y (stopy)',
        xaxis=dict(range=[-300, 300], showgrid=False, zeroline=False),
        yaxis=dict(range=[-100, 400], showgrid=False, zeroline=False),
        width=700, height=600,
        plot_bgcolor=NBA_WHITE, # Białe tło
        paper_bgcolor=NBA_WHITE,
        font_color=NBA_BLUE, # Niebieskie napisy
        showlegend=False # Ukrywamy legendę, bo kolory mówią same za siebie
    )
    return fig

# Funkcja do generowania raportu musi zwracać string lub dict
def generate_player_report_string(player_name, data):
    # ... (cała logika z oryginalnej funkcji) ...
    report_lines = []
    report_lines.append(f"\n--- RAPORT DLA GRACZA: {player_name} ---\n")
    # ... (append kolejnych linii zamiast print()) ...
    # Przykład: report_lines.append(f"Całkowita liczba rzutów: {total_shots}")
    # ...
    return "\n".join(report_lines)


# --- Główna część aplikacji ---

st.title("🏀 Analiza Preferencji Rzutowych w NBA")
st.markdown(f"Dane: Sezon 2023-24 | Kolory: NBA Style ({NBA_RED}, {NBA_BLUE}, {NBA_WHITE})")

# Wczytanie danych
shooting_data = load_shooting_data(CSV_FILENAME)

if not shooting_data.empty:
    if 'PLAYER_NAME' in shooting_data.columns:
        player_list = sorted(shooting_data['PLAYER_NAME'].unique())

        # --- Pasek Boczny ---
        st.sidebar.header("Panel Sterowania")
        selected_player = st.sidebar.selectbox("Wybierz Gracza:", player_list, index=player_list.index('Luka Doncic') if 'Luka Doncic' in player_list else 0)

        players_to_compare = st.sidebar.multiselect(
            "Wybierz Graczy do Porównania:",
            player_list,
            # Domyślne wybory - pierwszych 4 lub konkretni jeśli istnieją
            default = [p for p in ['Luka Doncic', 'Stephen Curry', 'Nikola Jokic', 'LeBron James'] if p in player_list][:4]
        )

        st.sidebar.info("Wybierz gracza i przeglądaj analizy w zakładkach.")

        # --- Zakładki ---
        tab_player, tab_compare, tab_model, tab_report = st.tabs([
            f"👤 Analiza Gracza ({selected_player})",
            "🆚 Porównanie Graczy",
            "🤖 Model Skuteczności (KNN)",
            "📄 Raport Tekstowy"
        ])

        # Dane dla wybranego gracza
        player_data = shooting_data[shooting_data['PLAYER_NAME'] == selected_player].copy()

        with tab_player:
            if player_data.empty:
                st.warning("Brak danych dla wybranego gracza.")
            else:
                st.header(f"Analiza dla: {selected_player}")

                # Wyświetlanie podstawowych statystyk (np. st.metric)
                total_shots = len(player_data)
                made_shots = player_data['SHOT_MADE_FLAG'].sum() if 'SHOT_MADE_FLAG' in player_data else 'N/A'
                fg_pct = (made_shots / total_shots * 100) if isinstance(made_shots, (int, float)) and total_shots > 0 else 'N/A'

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Liczba Rzutów", total_shots)
                with col2:
                     st.metric("Skuteczność (FG%)", f"{fg_pct:.1f}%" if isinstance(fg_pct, float) else fg_pct)

                # Wyświetlanie mapy rzutów
                st.subheader("Mapa Rzutów")
                shot_map_fig = create_player_shot_map_plotly(player_data, selected_player)
                if shot_map_fig:
                    st.plotly_chart(shot_map_fig, use_container_width=True)
                else:
                    st.write("Brak danych lokalizacji (LOC_X, LOC_Y) do stworzenia mapy.")

                # TODO: Dodać inne analizy dla gracza (histogram dystansu, hot zones, etc.)
                # Pamiętaj o modyfikacji funkcji, by zwracały figury i używały st.pyplot/st.plotly_chart

        with tab_compare:
            st.header("Porównanie Wybranych Graczy")
            if len(players_to_compare) < 2:
                st.warning("Wybierz co najmniej dwóch graczy w panelu bocznym.")
            else:
                st.write(f"Porównanie dla: {', '.join(players_to_compare)}")
                # TODO: Dodać wykresy porównawcze (KDE dystansu, mapy obok siebie, skuteczność vs dystans)
                # Użyj st.columns do umieszczenia wykresów obok siebie

        with tab_model:
            st.header(f"Modelowanie Skuteczności (KNN) dla: {selected_player}")
            if len(player_data) < 50: # Przykładowy próg
                st.warning("Zbyt mało danych (<50 rzutów) dla wybranego gracza, aby zbudować sensowny model.")
            else:
                # TODO: Dodać logikę trenowania modelu KNN (przycisk, spinner)
                # Wyświetlić metryki (st.metric, st.dataframe dla classification_report)
                # Wyświetlić macierz pomyłek (st.pyplot z ConfusionMatrixDisplay)
                # Wyświetlić mapę prawdopodobieństwa trafienia (st.plotly_chart)
                st.info("Funkcjonalność modelowania KNN do implementacji.")


        with tab_report:
            st.header(f"Raport Tekstowy dla: {selected_player}")
            if player_data.empty:
                st.warning("Brak danych dla wybranego gracza.")
            else:
                # Generowanie i wyświetlanie raportu
                with st.spinner("Generowanie raportu..."):
                    # Należy zmodyfikować funkcję generate_player_report, aby zwracała string
                    # report_text = generate_player_report_string(selected_player, shooting_data)
                    report_text = f"Funkcja generowania raportu (generate_player_report_string) musi zostać zaimplementowana i zmodyfikowana, aby zwracała tekst.\n\nPrzykładowe dane:\nGracz: {selected_player}\nLiczba rzutów: {total_shots}\nSkuteczność: {fg_pct}" # Placeholder
                st.text_area("Raport:", report_text, height=400)

    else:
         st.error("Kolumna 'PLAYER_NAME' nie została znaleziona w danych. Sprawdź plik CSV.")
else:
    st.warning("Nie udało się wczytać danych. Aplikacja nie może kontynuować.")