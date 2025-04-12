# nba_app_v3.py
# Final version incorporating all requested modifications as of 2025-04-11
# Using st.radio and st.session_state instead of st.tabs
# KNN tab includes Cross-Validation, Single Split Eval, and Configurable Test Size
# ADDED: XGBoost model prediction tab with similar evaluation structure

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# Usunięto mylące komentarze z poniższych importów sklearn, ponieważ model KNN JEST zaimplementowany
from sklearn.neighbors import KNeighborsClassifier
# Imports needed for evaluation
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import os
from datetime import datetime
import pytz # For timezone handling in footer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb # <--- Dodano import XGBoost

warnings.filterwarnings('ignore')

# --- Konfiguracja Początkowa ---
st.set_page_config(
    layout="wide",
    page_title="Analiza Rzutów NBA 2023-24"
    # UPEWNIJ SIĘ, ŻE TUTAJ NIE MA ŻADNYCH LINII Z primaryColor, backgroundColor itp.
)

# === Inicjalizacja Session State dla aktywnego widoku ===
if 'active_view' not in st.session_state:
    # Ustaw domyślną zakładkę/widok przy pierwszym uruchomieniu
    st.session_state.active_view = "📊 Rankingi Skuteczności"

# --- Ścieżka do pliku CSV ---
# !!! WAŻNE: Zmień tę ścieżkę, jeśli Twój plik CSV ma inną nazwę lub lokalizację !!!
CSV_FILE_PATH = 'nba_player_shooting_data_2023_24.csv' # Przykład nazwy - dostosuj!

# --- Funkcje Pomocnicze ---

@st.cache_data
def load_shooting_data(file_path):
    """Wczytuje i wstępnie przetwarza dane o rzutach graczy NBA."""
    load_status = {"success": False} # Domyślny status
    try:
        data = pd.read_csv(file_path, parse_dates=['GAME_DATE'])
        # Komunikat o sukcesie będzie wyświetlony później

        required_columns_map = {
            'PLAYER': 'PLAYER_NAME', 'TEAM': 'TEAM_NAME',
            'SHOT RESULT': 'SHOT_MADE_FLAG', 'SHOT TYPE': 'SHOT_TYPE',
            'ACTION TYPE': 'ACTION_TYPE',
            'SHOT ZONE BASIC': 'SHOT_ZONE_BASIC', 'SHOT ZONE AREA': 'SHOT_ZONE_AREA',
            'SHOT ZONE RANGE': 'SHOT_ZONE_RANGE', 'SHOT DISTANCE': 'SHOT_DISTANCE',
            'LOC_X': 'LOC_X', 'LOC_Y': 'LOC_Y', 'GAME DATE': 'GAME_DATE',
            'PERIOD': 'PERIOD', 'MINUTES REMAINING': 'MINUTES_REMAINING',
            'SECONDS REMAINING': 'SECONDS_REMAINING',
            'Season Type': 'SEASON_TYPE'
        }
        rename_map = {old: new for old, new in required_columns_map.items() if old in data.columns and new not in data.columns}
        if rename_map:
            data = data.rename(columns=rename_map)

        nan_in_made_flag = False
        if 'SHOT_MADE_FLAG' in data.columns:
            if data['SHOT_MADE_FLAG'].dtype == 'object':
                made_values = ['Made', 'made', '1', 1]
                # Sprawdź czy są inne wartości przed konwersją
                original_unique = data['SHOT_MADE_FLAG'].unique()
                data['SHOT_MADE_FLAG'] = data['SHOT_MADE_FLAG'].apply(lambda x: 1 if str(x).strip() in made_values else (0 if pd.notna(x) else np.nan))
                if data['SHOT_MADE_FLAG'].isnull().any():
                    nan_in_made_flag = True
                data['SHOT_MADE_FLAG'] = pd.to_numeric(data['SHOT_MADE_FLAG'], errors='coerce')

            elif pd.api.types.is_numeric_dtype(data['SHOT_MADE_FLAG']):
                 if data['SHOT_MADE_FLAG'].isnull().any():
                     nan_in_made_flag = True
                 # Konwertuj na int, jeśli nie ma NaN, inaczej zostaw float i obsłuż NaN później
                 if not data['SHOT_MADE_FLAG'].isnull().any():
                     data['SHOT_MADE_FLAG'] = data['SHOT_MADE_FLAG'].astype(int)
            else: # Inny nieobsługiwany typ
                nan_in_made_flag = True # Zakładamy, że mogą być problemy
                data['SHOT_MADE_FLAG'] = pd.to_numeric(data['SHOT_MADE_FLAG'], errors='coerce')

        time_cols = ['PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING']
        missing_time_cols_warning = False
        nan_in_time_cols_warning = False
        for col in time_cols:
            if col in data.columns:
                original_type = data[col].dtype
                # Sprawdź czy konwersja jest potrzebna i możliwa
                if not pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    if data[col].isnull().any():
                        nan_in_time_cols_warning = True
            else:
                missing_time_cols_warning = True

        # Tworzenie kolumn czasowych
        can_create_time_cols = not missing_time_cols_warning and not nan_in_time_cols_warning
        if can_create_time_cols and all(col in data.columns for col in ['PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING']):
            if all(pd.api.types.is_numeric_dtype(data[col]) for col in time_cols):
                if 'MINUTES_REMAINING' in data.columns and 'SECONDS_REMAINING' in data.columns:
                    data['GAME_TIME_SEC'] = data['MINUTES_REMAINING'] * 60 + data['SECONDS_REMAINING']
                if 'PERIOD' in data.columns:
                    # Upewnij się, że PERIOD jest int przed porównaniem > 4
                    period_int = pd.to_numeric(data['PERIOD'], errors='coerce').fillna(-1).astype(int)
                    data['QUARTER_TYPE'] = period_int.apply(lambda x: 'Dogrywka' if x > 4 else 'Regularna')
            else:
                # Jeśli kolumny istnieją, ale nie są numeryczne po próbie konwersji
                nan_in_time_cols_warning = True # Ustaw flagę ostrzeżenia


        # Sprawdzenie kluczowych kolumn
        key_cols_to_check = ['PLAYER_NAME', 'TEAM_NAME', 'LOC_X', 'LOC_Y', 'SHOT_DISTANCE', 'SEASON_TYPE',
                             'ACTION_TYPE', 'SHOT_TYPE', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE', 'SHOT_MADE_FLAG']
        missing_key_cols = [col for col in key_cols_to_check if col not in data.columns]

        if 'SHOT_MADE_FLAG' in data.columns:
             # Upewnij się, że NaN są obsługiwane przed analizą - często usuwane później w funkcjach
             pass # Już skonwertowane wyżej

        load_status.update({
            "success": True,
            "shape": data.shape,
            "missing_time_cols": missing_time_cols_warning,
            "nan_in_time_cols": nan_in_time_cols_warning,
            "nan_in_made_flag": nan_in_made_flag,
            "missing_key_cols": missing_key_cols
        })
        return data, load_status

    except FileNotFoundError:
        error_msg = f"Błąd: Nie znaleziono pliku {file_path}."
        st.session_state.load_error_message = error_msg
        load_status["error"] = error_msg
        return pd.DataFrame(), load_status
    except Exception as e:
        error_msg = f"Błąd podczas wczytywania lub przetwarzania danych: {e}"
        st.session_state.load_error_message = error_msg
        load_status["error"] = error_msg
        return pd.DataFrame(), load_status

# --- Funkcje Wizualizacyjne i Analityczne ---

def add_court_shapes(fig):
    """Dodaje kształty boiska do figury Plotly."""
    # Obręcz
    fig.add_shape(type="circle", x0=-7.5, y0=-7.5+52.5, x1=7.5, y1=7.5+52.5, line_color="orange", line_width=2)
    # Deska
    fig.add_shape(type="rect", x0=-30, y0=-7.5+40, x1=30, y1=-4.5+40, line_color="black", line_width=1, fillcolor="#e8e8e8") # Jasnoszary
    # Linia pola 3 sekund ("trumna")
    fig.add_shape(type="rect", x0=-80, y0=-47.5, x1=80, y1=142.5, line_color="black", line_width=1)
    # Strefa ograniczona (półkole pod koszem)
    fig.add_shape(type="path", path=f"M -40 -7.5 A 40 40 0 0 1 40 -7.5", line_color="black", line_width=1, y0=-47.5) # Poprawka y0
    # Koło na linii rzutów wolnych
    fig.add_shape(type="circle", x0=-60, y0=142.5-60, x1=60, y1=142.5+60, line_color="black", line_width=1)
    # Linie boczne dla rzutów za 3 punkty (proste)
    fig.add_shape(type="line", x0=-220, y0=-47.5, x1=-220, y1=92.5, line_color="black", line_width=1)
    fig.add_shape(type="line", x0=220, y0=-47.5, x1=220, y1=92.5, line_color="black", line_width=1)
    # Łuk rzutów za 3 punkty
    fig.add_shape(type="path", path=f"M -220 92.5 C -135 300, 135 300, 220 92.5", line_color="black", line_width=1)
    # Koło środkowe
    fig.add_shape(type="circle", x0=-60, y0=470-60, x1=60, y1=470+60, line_color="black", line_width=1)
    # Linia środkowa
    fig.add_shape(type="line", x0=-250, y0=422.5, x1=250, y1=422.5, line_color="black", line_width=1)
    # Linie ograniczające boisko (prostokąt)
    fig.add_shape(type="rect", x0=-250, y0=-47.5, x1=250, y1=422.5, line_color="black", line_width=1)

    # Ustawienie zakresów osi dla standardowego widoku połowy boiska NBA
    fig.update_xaxes(range=[-260, 260])
    fig.update_yaxes(range=[-60, 480]) # Dostosowano zakres Y

    return fig

@st.cache_data
def filter_data_by_player(player_name, data):
    """Filtruje dane dla wybranego gracza."""
    if 'PLAYER_NAME' not in data.columns or data.empty: return pd.DataFrame()
    return data[data['PLAYER_NAME'] == player_name].copy()

@st.cache_data
def filter_data_by_team(team_name, data):
    """Filtruje dane dla wybranej drużyny."""
    if 'TEAM_NAME' not in data.columns or data.empty: return pd.DataFrame()
    return data[data['TEAM_NAME'] == team_name].copy()

#@st.cache_data # Usunięto cache, bo dane wejściowe się często zmieniają (player/team data)
def get_basic_stats(entity_data, entity_name, entity_type="Gracz"):
    """Oblicza podstawowe statystyki (ogółem) dla gracza lub drużyny."""
    stats = {'total_shots': 0, 'made_shots': "N/A", 'shooting_pct': "N/A"}
    if entity_data is None or entity_data.empty:
        return stats

    stats['total_shots'] = len(entity_data)

    if 'SHOT_MADE_FLAG' in entity_data.columns and stats['total_shots'] > 0:
        # Pracuj na kopii, aby uniknąć SettingWithCopyWarning
        data_copy = entity_data[['SHOT_MADE_FLAG']].copy()
        # Konwersja na numeryczny i usunięcie NaN *lokalnie* dla tej kalkulacji
        data_copy['SHOT_MADE_FLAG_NUM'] = pd.to_numeric(data_copy['SHOT_MADE_FLAG'], errors='coerce')
        numeric_shots = data_copy['SHOT_MADE_FLAG_NUM'].dropna()

        if not numeric_shots.empty:
            stats['made_shots'] = int(numeric_shots.sum())
            total_valid_shots = len(numeric_shots) # Użyj liczby ważnych (nie-NaN) rzutów
            if total_valid_shots > 0:
                stats['shooting_pct'] = (stats['made_shots'] / total_valid_shots) * 100
            else:
                stats['made_shots'], stats['shooting_pct'] = 0, 0.0
        else:
            # Jeśli po usunięciu NaN nic nie zostało
            stats['made_shots'], stats['shooting_pct'] = 0, 0.0
    elif stats['total_shots'] == 0:
        stats['made_shots'], stats['shooting_pct'] = 0, 0.0

    return stats


@st.cache_data
def plot_player_eff_vs_distance(player_data, player_name, bin_width=1, min_attempts_per_bin=5):
    """Tworzy wykres liniowy skuteczności gracza (FG%) w zależności od odległości rzutu (binowanie)."""
    required_cols = ['SHOT_DISTANCE', 'SHOT_MADE_FLAG']
    if not all(col in player_data.columns for col in required_cols):
        #st.warning("Brak wymaganych kolumn (SHOT_DISTANCE, SHOT_MADE_FLAG) do analizy skuteczności wg odległości.")
        return None

    # Pracuj na kopii, aby uniknąć ostrzeżeń
    distance_data = player_data[required_cols].copy()
    distance_data['SHOT_MADE_FLAG'] = pd.to_numeric(distance_data['SHOT_MADE_FLAG'], errors='coerce')
    distance_data['SHOT_DISTANCE'] = pd.to_numeric(distance_data['SHOT_DISTANCE'], errors='coerce')

    distance_data = distance_data.dropna(subset=required_cols)

    if distance_data.empty:
        #st.info("Brak kompletnych danych (Odległość, Wynik) do analizy skuteczności wg odległości.")
        return None

    # Sprawdzenie typu danych po konwersji
    if not pd.api.types.is_numeric_dtype(distance_data['SHOT_DISTANCE']):
        #st.warning("Kolumna SHOT_DISTANCE nie jest typu numerycznego po konwersji.")
        return None

    max_dist = distance_data['SHOT_DISTANCE'].max()
    if pd.isna(max_dist) or max_dist <= 0:
        #st.info("Nieprawidłowe lub brakujące dane odległości.")
        return None

    # Tworzenie binów
    try:
        # Zaokrąglij max_dist w górę dla pewności
        max_dist_rounded = np.ceil(max_dist)
        distance_bins = np.arange(0, max_dist_rounded + bin_width, bin_width)
        # Użyj etykiet numerycznych (środków przedziałów) dla łatwiejszego sortowania
        bin_labels_numeric = [(distance_bins[i] + distance_bins[i+1]) / 2 for i in range(len(distance_bins)-1)]

        if not bin_labels_numeric: # Jeśli max_dist było bardzo małe
             return None

        distance_data['distance_bin_mid'] = pd.cut(distance_data['SHOT_DISTANCE'], bins=distance_bins, labels=bin_labels_numeric, right=False, include_lowest=True)

        # Konwertuj etykiety binów na numeryczne, jeśli pd.cut zwrócił kategorie
        if pd.api.types.is_categorical_dtype(distance_data['distance_bin_mid']):
             distance_data['distance_bin_mid'] = pd.to_numeric(distance_data['distance_bin_mid'].astype(str), errors='coerce')


        effectiveness = distance_data.groupby('distance_bin_mid', observed=False)['SHOT_MADE_FLAG'].agg(Made='sum', Attempts='count').reset_index()
        effectiveness = effectiveness[effectiveness['Attempts'] >= min_attempts_per_bin]

        if effectiveness.empty:
            #st.info(f"Brak wystarczających danych (min. {min_attempts_per_bin} prób na {bin_width}-stopowy przedział odległości).")
            return None

        effectiveness['FG%'] = (effectiveness['Made'] / effectiveness['Attempts']) * 100
        # Sortowanie według środka przedziału
        effectiveness = effectiveness.sort_values(by='distance_bin_mid')

        if effectiveness.empty:
            #st.info("Problem z danymi po obliczeniu skuteczności i środka przedziału.")
            return None

        fig = px.line(effectiveness, x='distance_bin_mid', y='FG%', title=f'Wpływ odległości rzutu na skuteczność - {player_name}',
                      labels={'distance_bin_mid': 'Środek przedziału odległości (stopy)', 'FG%': 'Skuteczność (%)'}, markers=True, hover_data=['Attempts', 'Made'])
        fig.update_layout(yaxis_range=[-5, 105], xaxis_title='Odległość rzutu (stopy)', yaxis_title='Skuteczność (%)', hovermode="x unified")
        fig.update_traces(connectgaps=False)
        return fig

    except ValueError as e:
        #st.error(f"Błąd podczas tworzenia binów odległości: {e}. Sprawdź zakres danych.")
        return None
    except Exception as e_general:
        #st.error(f"Nieoczekiwany błąd w plot_player_eff_vs_distance: {e_general}")
         return None


@st.cache_data
def plot_shot_chart(entity_data, entity_name, entity_type="Gracz"):
    """Tworzy interaktywną mapę rzutów."""
    required_cols = ['LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']
    if not all(col in entity_data.columns for col in required_cols): return None

    # Pracuj na kopii
    plot_data = entity_data[required_cols + [col for col in ['SHOT_DISTANCE', 'SHOT_TYPE', 'ACTION_TYPE', 'SHOT_ZONE_BASIC', 'PERIOD'] if col in entity_data.columns]].copy()
    # Konwersje i usunięcie NaN
    plot_data['LOC_X'] = pd.to_numeric(plot_data['LOC_X'], errors='coerce')
    plot_data['LOC_Y'] = pd.to_numeric(plot_data['LOC_Y'], errors='coerce')
    plot_data['SHOT_MADE_FLAG'] = pd.to_numeric(plot_data['SHOT_MADE_FLAG'], errors='coerce')
    plot_data = plot_data.dropna(subset=required_cols)

    if plot_data.empty: return None

    plot_data['Wynik Rzutu'] = plot_data['SHOT_MADE_FLAG'].map({0: 'Niecelny', 1: 'Celny'})
    color_col, color_map, cat_orders = 'Wynik Rzutu', {'Niecelny': 'red', 'Celny': 'green'}, {"Wynik Rzutu": ['Niecelny', 'Celny']}

    # Przygotuj hover_data - upewnij się, że kolumny istnieją
    hover_cols_present = [col for col in ['SHOT_DISTANCE', 'SHOT_TYPE', 'ACTION_TYPE', 'SHOT_ZONE_BASIC', 'PERIOD'] if col in plot_data.columns]
    hover_data_config = {col: True for col in hover_cols_present} # Użyj słownika dla pewności

    fig = px.scatter(plot_data, x='LOC_X', y='LOC_Y', color=color_col, title=f'Mapa rzutów - {entity_name} ({entity_type})',
                     labels={'LOC_X': 'Pozycja X', 'LOC_Y': 'Pozycja Y', 'Wynik Rzutu': 'Wynik'},
                     hover_data=hover_data_config if hover_data_config else None,
                     category_orders=cat_orders, color_discrete_map=color_map, opacity=0.7)

    fig = add_court_shapes(fig) # Dodaj boisko
    fig.update_layout(
        height=600,
        xaxis_showgrid=False, # Ukryj siatkę X
        yaxis_showgrid=False, # Ukryj siatkę Y
        plot_bgcolor='rgba(255, 255, 255, 1)' # Białe tło
    )
    return fig


@st.cache_data
def calculate_hot_zones(entity_data, min_shots_in_zone=5, n_bins=10):
    """Oblicza statystyki dla stref rzutowych."""
    required_cols = ['LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']
    if not all(col in entity_data.columns for col in required_cols): return pd.DataFrame()

    # Pracuj na kopii
    zone_data = entity_data[required_cols].copy()
    zone_data['LOC_X'] = pd.to_numeric(zone_data['LOC_X'], errors='coerce')
    zone_data['LOC_Y'] = pd.to_numeric(zone_data['LOC_Y'], errors='coerce')
    zone_data['SHOT_MADE_FLAG'] = pd.to_numeric(zone_data['SHOT_MADE_FLAG'], errors='coerce')
    zone_data = zone_data.dropna(subset=required_cols)

    if zone_data.empty: return pd.DataFrame()

    # Definicja binów - użyj zakresów z add_court_shapes
    x_min, x_max = -250, 250
    y_min, y_max = -47.5, 422.5 # Użyj granic boiska

    # Upewnij się, że n_bins jest co najmniej 2
    n_bins = max(2, n_bins)

    try:
        # Użyj pd.cut do stworzenia stref
        zone_data['zone_x'] = pd.cut(zone_data['LOC_X'], bins=np.linspace(x_min, x_max, n_bins + 1), include_lowest=True, right=True)
        zone_data['zone_y'] = pd.cut(zone_data['LOC_Y'], bins=np.linspace(y_min, y_max, n_bins + 1), include_lowest=True, right=True)

        # Agregacja danych
        zones = zone_data.groupby(['zone_x', 'zone_y'], observed=False).agg(
            total_shots=('SHOT_MADE_FLAG', 'count'),
            made_shots=('SHOT_MADE_FLAG', 'sum'),
            # mean() obsłuży NaN poprawnie (zignoruje)
            percentage_raw=('SHOT_MADE_FLAG', 'mean')
        ).reset_index()

        # Filtrowanie stref z małą liczbą rzutów
        zones = zones[zones['total_shots'] >= min_shots_in_zone].copy()

        if zones.empty: return pd.DataFrame()

        # Obliczanie procentów i środków stref
        zones['percentage'] = zones['percentage_raw'] * 100
        zones['x_center'] = zones['zone_x'].apply(lambda x: x.mid if isinstance(x, pd.Interval) else None)
        zones['y_center'] = zones['zone_y'].apply(lambda x: x.mid if isinstance(x, pd.Interval) else None)

        # Usunięcie wierszy, gdzie nie udało się obliczyć środka (choć nie powinno się zdarzyć)
        return zones.dropna(subset=['x_center', 'y_center'])

    except Exception as e:
         # st.error(f"Błąd w calculate_hot_zones: {e}") # Logowanie błędu może być pomocne
         return pd.DataFrame()


@st.cache_data
def plot_hot_zones_heatmap(hot_zones_df, entity_name, entity_type="Gracz", min_shots_in_zone=5):
    """Tworzy interaktywną mapę ciepła stref rzutowych (skuteczność)."""
    required_cols = ['x_center', 'y_center', 'total_shots', 'percentage', 'made_shots']
    if hot_zones_df is None or hot_zones_df.empty or not all(col in hot_zones_df.columns for col in required_cols):
        return None

    # Upewnij się, że dane są numeryczne
    plot_df = hot_zones_df[required_cols].copy()
    for col in required_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    plot_df = plot_df.dropna()

    if plot_df.empty: return None

    min_pct, max_pct = plot_df['percentage'].min(), plot_df['percentage'].max()
    color_range = [max(0, min_pct - 5), min(100, max_pct + 5)] if pd.notna(min_pct) and pd.notna(max_pct) else [0, 100]
    if color_range[0] >= color_range[1]: color_range = [0, 100] # Poprawka dla przypadku jednej wartości

    # Maksymalny rozmiar punktu zależny od liczby rzutów
    max_bubble_size = plot_df["total_shots"].max()
    size_ref = max(1, max_bubble_size / 50.0) # Dostosuj dzielnik wg potrzeb

    fig = px.scatter(plot_df, x='x_center', y='y_center', size='total_shots', color='percentage',
                     color_continuous_scale=px.colors.diverging.RdYlGn, # Skala czerwony-żółty-zielony
                     size_max=60, # Maksymalny rozmiar punktu na wykresie
                     range_color=color_range,
                     title=f'Skuteczność stref rzutowych ({entity_type}: {entity_name}, min. {min_shots_in_zone} rzutów)',
                     labels={'x_center': 'Pozycja X', 'y_center': 'Pozycja Y', 'total_shots': 'Liczba rzutów', 'percentage': 'Skuteczność (%)'},
                     custom_data=['made_shots', 'total_shots']) # Dodaj dane do hovera

    fig.update_traces(
        hovertemplate="<b>Strefa X:</b> %{x:.1f}, <b>Y:</b> %{y:.1f}<br>" +
                      "<b>Liczba rzutów:</b> %{customdata[1]}<br>" +
                      "<b>Trafione:</b> %{customdata[0]}<br>" +
                      "<b>Skuteczność:</b> %{marker.color:.1f}%<extra></extra>",
        marker=dict(sizeref=size_ref, sizemin=4) # Dostosuj sizeref i sizemin
    )

    fig = add_court_shapes(fig) # Dodaj boisko
    fig.update_layout(
        height=600,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        plot_bgcolor='rgba(255, 255, 255, 1)'
    )
    return fig


@st.cache_data
def plot_shot_frequency_heatmap(data, season_name, nbins_x=50, nbins_y=50):
    """Tworzy heatmapę częstotliwości rzutów na boisku (Histogram2d)."""
    required_cols = ['LOC_X', 'LOC_Y']
    if not all(col in data.columns for col in required_cols):
        # st.warning("Brak wymaganych kolumn (LOC_X, LOC_Y) do mapy częstotliwości.")
        return None

    # Pracuj na kopii
    plot_data = data[required_cols].copy()
    plot_data['LOC_X'] = pd.to_numeric(plot_data['LOC_X'], errors='coerce')
    plot_data['LOC_Y'] = pd.to_numeric(plot_data['LOC_Y'], errors='coerce')
    plot_data = plot_data.dropna(subset=required_cols)

    if plot_data.empty:
        # st.info("Brak danych lokalizacji (LOC_X, LOC_Y) do stworzenia mapy częstotliwości.")
        return None

    fig = go.Figure()
    fig.add_trace(go.Histogram2d(
        x=plot_data['LOC_X'],
        y=plot_data['LOC_Y'],
        colorscale='YlOrRd', # Skala kolorów od żółtego do czerwonego
        nbinsx=nbins_x,
        nbinsy=nbins_y,
        zauto=True, # Automatyczne skalowanie osi Z (kolor)
        hovertemplate='<b>Zakres X:</b> %{x}<br><b>Zakres Y:</b> %{y}<br><b>Liczba rzutów:</b> %{z}<extra></extra>',
        colorbar=dict(title='Liczba rzutów') # Tytuł dla paska kolorów
    ))

    fig = add_court_shapes(fig) # Dodaj boisko

    fig.update_layout(
        title=f'Mapa Częstotliwości Rzutów ({season_name})',
        xaxis_title="Pozycja X",
        yaxis_title="Pozycja Y",
        height=650,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        plot_bgcolor='rgba(255, 255, 255, 1)' # Białe tło
    )
    # Zakresy osi ustawione w add_court_shapes
    # fig.update_xaxes(range=[-300, 300])
    # fig.update_yaxes(range=[-100, 500])
    return fig


@st.cache_data
def plot_player_quarter_eff(entity_data, entity_name, entity_type="Gracz", min_attempts=5):
    """Wykres skuteczności w poszczególnych kwartach/dogrywkach."""
    if 'PERIOD' not in entity_data.columns or 'SHOT_MADE_FLAG' not in entity_data.columns: return None

    # Pracuj na kopii
    quarter_data = entity_data[['PERIOD', 'SHOT_MADE_FLAG']].copy()
    quarter_data['PERIOD'] = pd.to_numeric(quarter_data['PERIOD'], errors='coerce')
    quarter_data['SHOT_MADE_FLAG'] = pd.to_numeric(quarter_data['SHOT_MADE_FLAG'], errors='coerce')
    quarter_data = quarter_data.dropna()

    if quarter_data.empty: return None

    # Konwersja PERIOD na int dla grupowania
    quarter_data['PERIOD'] = quarter_data['PERIOD'].astype(int)

    quarter_eff = quarter_data.groupby('PERIOD')['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
    quarter_eff['mean'] *= 100
    quarter_eff = quarter_eff[quarter_eff['count'] >= min_attempts] # Filtracja wg minimalnej liczby prób

    if quarter_eff.empty: return None

    def map_period(p):
        if p <= 4: return f"Kwarta {int(p)}"
        elif p == 5: return "OT 1"
        else: return f"OT {int(p-4)}"
    quarter_eff['Okres Gry'] = quarter_eff['PERIOD'].apply(map_period)
    quarter_eff = quarter_eff.sort_values(by='PERIOD') # Sortuj wg numeru kwarty/OT

    fig = px.bar(quarter_eff, x='Okres Gry', y='mean', text='mean', title=f'Skuteczność w kwartach/dogrywkach - {entity_name} ({entity_type}, min. {min_attempts} prób)',
                 labels={'Okres Gry': 'Okres Gry', 'mean': 'Skuteczność (%)'}, hover_data=['count'])
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(yaxis_range=[0, 105], uniformtext_minsize=8, uniformtext_mode='hide') # Zwiększono zakres Y
    return fig


@st.cache_data
def plot_player_season_trend(entity_data, entity_name, entity_type="Gracz", min_monthly_attempts=10):
    """Wykres trendu skuteczności w trakcie sezonu (miesięcznie)."""
    if 'GAME_DATE' not in entity_data.columns or 'SHOT_MADE_FLAG' not in entity_data.columns: return None

    # Pracuj na kopii
    trend_data = entity_data[['GAME_DATE', 'SHOT_MADE_FLAG']].copy()
    trend_data['GAME_DATE'] = pd.to_datetime(trend_data['GAME_DATE'], errors='coerce')
    trend_data['SHOT_MADE_FLAG'] = pd.to_numeric(trend_data['SHOT_MADE_FLAG'], errors='coerce')
    trend_data = trend_data.dropna()

    if trend_data.empty or len(trend_data) < min_monthly_attempts: return None # Potrzebujemy wystarczająco danych

    trend_data = trend_data.set_index('GAME_DATE')
    # 'ME' oznacza Month End frequency
    monthly_eff = trend_data.resample('ME')['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
    monthly_eff['mean'] *= 100
    monthly_eff = monthly_eff[monthly_eff['count'] >= min_monthly_attempts] # Filtracja wg minimalnej liczby prób miesięcznie

    if monthly_eff.empty or len(monthly_eff) < 2: return None # Potrzebujemy co najmniej 2 miesięcy z danymi

    monthly_eff['Miesiąc'] = monthly_eff['GAME_DATE'].dt.strftime('%Y-%m') # Format BBBB-MM dla osi X

    fig = px.line(monthly_eff, x='Miesiąc', y='mean', markers=True, title=f'Miesięczny trend skuteczności - {entity_name} ({entity_type}, min. {min_monthly_attempts} prób/miesiąc)',
                  labels={'Miesiąc': 'Miesiąc', 'mean': 'Skuteczność (%)'}, hover_data=['count'])

    # Ustalenie zakresu Y, aby nie ucinać wartości
    max_y = 105
    min_y = -5
    if not pd.isna(monthly_eff['mean'].max()):
        max_y = min(105, monthly_eff['mean'].max() + 10)
    if not pd.isna(monthly_eff['mean'].min()):
         min_y = max(-5, monthly_eff['mean'].min() - 10)

    fig.update_layout(yaxis_range=[min_y, max_y])
    return fig


@st.cache_data
def plot_grouped_effectiveness(entity_data, group_col, entity_name, entity_type="Gracz", top_n=10, min_attempts=5):
    """Tworzy wykres skuteczności pogrupowany wg wybranej kolumny."""
    if group_col not in entity_data.columns or 'SHOT_MADE_FLAG' not in entity_data.columns: return None

    # Pracuj na kopii
    grouped_data = entity_data[[group_col, 'SHOT_MADE_FLAG']].copy()
    grouped_data[group_col] = grouped_data[group_col].astype(str) # Konwertuj na string dla pewności grupowania
    grouped_data['SHOT_MADE_FLAG'] = pd.to_numeric(grouped_data['SHOT_MADE_FLAG'], errors='coerce')
    grouped_data = grouped_data.dropna()

    if grouped_data.empty: return None

    grouped_eff = grouped_data.groupby(group_col)['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
    grouped_eff['mean'] *= 100
    grouped_eff = grouped_eff[grouped_eff['count'] >= min_attempts] # Filtracja wg minimalnej liczby prób

    # Sortuj wg liczby prób (count) malejąco, aby wybrać najczęstsze
    grouped_eff = grouped_eff.sort_values(by='count', ascending=False).head(top_n)
    # Następnie sortuj wg nazwy kategorii dla lepszej czytelności osi X
    grouped_eff = grouped_eff.sort_values(by=group_col, ascending=True)


    if grouped_eff.empty: return None

    axis_label = group_col.replace('_',' ').title()
    fig = px.bar(grouped_eff, x=group_col, y='mean', text='mean',
                 title=f'Skuteczność wg {axis_label} - {entity_name} ({entity_type}) (Top {top_n} najczęstszych, min. {min_attempts} prób)',
                 labels={group_col: axis_label, 'mean': 'Skuteczność (%)'}, hover_data=['count'])
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    # Usunięto sortowanie osi X, bo sortujemy DataFrame powyżej
    fig.update_layout(yaxis_range=[0, 105], uniformtext_minsize=8, uniformtext_mode='hide', xaxis_title=axis_label)
    return fig


@st.cache_data
def plot_comparison_eff_distance(compare_data, selected_players, bin_width=3, min_attempts_per_bin=5):
    """Porównuje skuteczność graczy względem odległości (linowy)."""
    required_cols = ['SHOT_DISTANCE', 'SHOT_MADE_FLAG', 'PLAYER_NAME']
    if not all(col in compare_data.columns for col in required_cols): return None

    # Pracuj na kopii
    compare_data_eff = compare_data[required_cols].copy()
    compare_data_eff['SHOT_MADE_FLAG'] = pd.to_numeric(compare_data_eff['SHOT_MADE_FLAG'], errors='coerce')
    compare_data_eff['SHOT_DISTANCE'] = pd.to_numeric(compare_data_eff['SHOT_DISTANCE'], errors='coerce')
    compare_data_eff = compare_data_eff.dropna(subset=required_cols)

    if compare_data_eff.empty: return None

    # Sprawdzenie typu danych po konwersji
    if not pd.api.types.is_numeric_dtype(compare_data_eff['SHOT_DISTANCE']):
        #st.warning("SHOT_DISTANCE nie jest numeryczny w danych porównawczych.")
        return None

    max_dist = compare_data_eff['SHOT_DISTANCE'].max()
    if pd.isna(max_dist) or max_dist <= 0: return None

    # Tworzenie binów (jak w plot_player_eff_vs_distance)
    try:
        max_dist_rounded = np.ceil(max_dist)
        distance_bins = np.arange(0, max_dist_rounded + bin_width, bin_width)
        bin_labels_numeric = [(distance_bins[i] + distance_bins[i+1]) / 2 for i in range(len(distance_bins)-1)]

        if not bin_labels_numeric: return None

        compare_data_eff['distance_bin_mid'] = pd.cut(compare_data_eff['SHOT_DISTANCE'], bins=distance_bins, labels=bin_labels_numeric, right=False, include_lowest=True)

        if pd.api.types.is_categorical_dtype(compare_data_eff['distance_bin_mid']):
            compare_data_eff['distance_bin_mid'] = pd.to_numeric(compare_data_eff['distance_bin_mid'].astype(str), errors='coerce')

        effectiveness = compare_data_eff.groupby(['PLAYER_NAME', 'distance_bin_mid'], observed=False)['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
        effectiveness['mean'] *= 100

        effectiveness = effectiveness[effectiveness['count'] >= min_attempts_per_bin]
        effectiveness = effectiveness.dropna(subset=['distance_bin_mid']) # Usuń NaN jeśli powstały

        if effectiveness.empty: return None # Jeśli żaden gracz/bin nie spełnia warunku

        # Sortowanie dla lepszego wykresu
        effectiveness = effectiveness.sort_values(by=['PLAYER_NAME', 'distance_bin_mid'])

        # Zakres Y
        max_eff_val = effectiveness['mean'].max()
        yaxis_range = [0, min(105, max_eff_val + 10 if not pd.isna(max_eff_val) else 105)]

        fig = px.line(effectiveness, x='distance_bin_mid', y='mean', color='PLAYER_NAME',
                      title=f'Porównanie skuteczności vs odległości (min. {min_attempts_per_bin} prób w przedziale {bin_width} stóp)',
                      labels={'distance_bin_mid': 'Odległość (stopy)', 'mean': 'Skuteczność (%)', 'PLAYER_NAME': 'Gracz'},
                      markers=True, hover_data=['count'])
        fig.update_layout(yaxis_range=yaxis_range, hovermode="x unified")
        return fig

    except Exception as e:
         # st.error(f"Błąd w plot_comparison_eff_distance: {e}")
         return None


@st.cache_data
def plot_comparison_eff_by_zone(compare_data, selected_players, min_shots_per_zone=5):
    """Tworzy grupowany wykres słupkowy porównujący skuteczność graczy wg SHOT_ZONE_BASIC."""
    required_cols = ['PLAYER_NAME', 'SHOT_MADE_FLAG', 'SHOT_ZONE_BASIC']
    if not all(col in compare_data.columns for col in required_cols):
        #st.warning(f"Brak kolumn do porównania wg stref: {required_cols}")
        return None

    # Pracuj na kopii
    zone_eff_data = compare_data[required_cols].copy()
    zone_eff_data['SHOT_MADE_FLAG'] = pd.to_numeric(zone_eff_data['SHOT_MADE_FLAG'], errors='coerce')
    # Konwertuj strefy na string i usuń potencjalne białe znaki
    zone_eff_data['SHOT_ZONE_BASIC'] = zone_eff_data['SHOT_ZONE_BASIC'].astype(str).str.strip()
    zone_eff_data = zone_eff_data.dropna(subset=required_cols)

    if zone_eff_data.empty:
        #st.info("Brak kompletnych danych do analizy skuteczności wg stref.")
        return None

    # Agregacja
    zone_stats = zone_eff_data.groupby(['PLAYER_NAME', 'SHOT_ZONE_BASIC'], observed=False)['SHOT_MADE_FLAG'].agg(Made='sum', Attempts='count').reset_index()

    # Filtracja wg minimalnej liczby prób
    zone_stats_filtered = zone_stats[zone_stats['Attempts'] >= min_shots_per_zone]

    if zone_stats_filtered.empty:
        #st.info(f"Brak wystarczających danych (min. {min_shots_per_zone} prób na strefę) do porównania skuteczności wg stref.")
        return None

    zone_stats_filtered['FG%'] = (zone_stats_filtered['Made'] / zone_stats_filtered['Attempts']) * 100

    # !!! DOSTOSUJ TĘ LISTĘ DO NAZW STREF W TWOIM PLIKU !!!
    # Użyj nazw, które faktycznie występują w danych po przetworzeniu
    # Można dynamicznie pobrać unikalne strefy z zone_stats_filtered['SHOT_ZONE_BASIC'].unique()
    # ale ustalona kolejność jest lepsza dla porównań
    zone_order_ideal = ['Restricted Area', 'In The Paint (Non-RA)', 'Mid-Range', 'Left Corner 3', 'Right Corner 3', 'Above the Break 3', 'Backcourt']
    # Użyj tylko tych stref z idealnej kolejności, które istnieją w danych
    actual_zones_in_data = zone_stats_filtered['SHOT_ZONE_BASIC'].unique()
    zone_order = [zone for zone in zone_order_ideal if zone in actual_zones_in_data]
    # Dodaj pozostałe strefy (jeśli są), aby niczego nie pominąć
    zone_order += [zone for zone in actual_zones_in_data if zone not in zone_order_ideal]


    if not zone_order: # Jeśli żadne ze stref nie istnieją
        #st.info("Brak danych dla predefiniowanych stref po filtracji.")
        return None

    # Filtruj DataFrame, aby zawierał tylko strefy z `zone_order`
    zone_stats_plot = zone_stats_filtered[zone_stats_filtered['SHOT_ZONE_BASIC'].isin(zone_order)].copy()

    if zone_stats_plot.empty:
        #st.info("Brak danych do wyświetlenia po ostatecznej filtracji stref.")
        return None

    fig = px.bar(zone_stats_plot, x='SHOT_ZONE_BASIC', y='FG%', color='PLAYER_NAME', barmode='group',
                 title=f'Porównanie skuteczności (FG%) wg Strefy Rzutowej (min. {min_shots_per_zone} prób)',
                 labels={'SHOT_ZONE_BASIC': 'Strefa Rzutowa', 'FG%': 'Skuteczność (%)', 'PLAYER_NAME': 'Gracz'},
                 hover_data=['Attempts', 'Made'],
                 category_orders={'SHOT_ZONE_BASIC': zone_order}, # Użyj dynamicznej kolejności
                 text='FG%') # Dodaj wartość na słupku

    fig.update_layout(yaxis_range=[0, 105], xaxis={'categoryorder':'array', 'categoryarray':zone_order}, legend_title_text='Gracze')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    return fig


@st.cache_data
def calculate_top_performers(data, group_by_col, min_total_shots, min_2pt_shots, min_3pt_shots, top_n=10):
    """Oblicza rankingi Top N graczy/zespołów wg skuteczności."""
    if group_by_col not in data.columns or 'SHOT_MADE_FLAG' not in data.columns: return None, None, None

    # Pracuj na kopii i usuń NaN kluczowych kolumn
    valid_data = data[[group_by_col, 'SHOT_MADE_FLAG', 'SHOT_TYPE'] if 'SHOT_TYPE' in data.columns else [group_by_col, 'SHOT_MADE_FLAG']].copy()
    valid_data['SHOT_MADE_FLAG'] = pd.to_numeric(valid_data['SHOT_MADE_FLAG'], errors='coerce')
    # Grupowanie po stringach - upewnij się, że kolumna grupująca jest stringiem
    valid_data[group_by_col] = valid_data[group_by_col].astype(str)
    valid_data = valid_data.dropna(subset=[group_by_col, 'SHOT_MADE_FLAG'])


    if valid_data.empty: return None, None, None

    # Skuteczność ogólna
    overall_stats = valid_data.groupby(group_by_col)['SHOT_MADE_FLAG'].agg(Made='sum', Attempts='count').reset_index()
    overall_stats = overall_stats[overall_stats['Attempts'] >= min_total_shots]
    top_overall = pd.DataFrame()
    if not overall_stats.empty:
        overall_stats['FG%'] = (overall_stats['Made'] / overall_stats['Attempts']) * 100
        top_overall = overall_stats.sort_values(by=['FG%', 'Attempts'], ascending=[False, False]).head(top_n) # Sortuj też wg Attempts
        col_name = group_by_col.replace('_',' ').title()
        top_overall = top_overall.rename(columns={group_by_col: col_name, 'Attempts': 'Próby'})
        top_overall = top_overall[[col_name, 'FG%', 'Próby']]

    # Skuteczność za 2 punkty
    shot_type_2pt = '2PT Field Goal' # Dostosuj, jeśli nazwa w danych jest inna
    top_2pt = pd.DataFrame()
    if 'SHOT_TYPE' in valid_data.columns:
         # Upewnij się, że SHOT_TYPE jest stringiem
         valid_data['SHOT_TYPE'] = valid_data['SHOT_TYPE'].astype(str).str.strip()
         if shot_type_2pt in valid_data['SHOT_TYPE'].unique():
             data_2pt = valid_data[valid_data['SHOT_TYPE'] == shot_type_2pt]
             stats_2pt = data_2pt.groupby(group_by_col)['SHOT_MADE_FLAG'].agg(Made_2PT='sum', Attempts_2PT='count').reset_index()
             stats_2pt = stats_2pt[stats_2pt['Attempts_2PT'] >= min_2pt_shots]
             if not stats_2pt.empty:
                 stats_2pt['2PT FG%'] = (stats_2pt['Made_2PT'] / stats_2pt['Attempts_2PT']) * 100
                 top_2pt = stats_2pt.sort_values(by=['2PT FG%', 'Attempts_2PT'], ascending=[False, False]).head(top_n)
                 col_name = group_by_col.replace('_',' ').title()
                 top_2pt = top_2pt.rename(columns={group_by_col: col_name, 'Attempts_2PT': 'Próby 2PT'})
                 top_2pt = top_2pt[[col_name, '2PT FG%', 'Próby 2PT']]
         # else: st.caption(f"Nie znaleziono '{shot_type_2pt}' w SHOT_TYPE dla rankingu 2PT.") # Komunikaty lepiej poza funkcją cache
    # else: st.caption("Brak 'SHOT_TYPE' dla rankingu 2PT.")

    # Skuteczność za 3 punkty
    shot_type_3pt = '3PT Field Goal' # Dostosuj, jeśli nazwa w danych jest inna
    top_3pt = pd.DataFrame()
    if 'SHOT_TYPE' in valid_data.columns:
       # SHOT_TYPE już powinien być stringiem z poprzedniego kroku
        if shot_type_3pt in valid_data['SHOT_TYPE'].unique():
             data_3pt = valid_data[valid_data['SHOT_TYPE'] == shot_type_3pt]
             stats_3pt = data_3pt.groupby(group_by_col)['SHOT_MADE_FLAG'].agg(Made_3PT='sum', Attempts_3PT='count').reset_index()
             stats_3pt = stats_3pt[stats_3pt['Attempts_3PT'] >= min_3pt_shots]
             if not stats_3pt.empty:
                 stats_3pt['3PT FG%'] = (stats_3pt['Made_3PT'] / stats_3pt['Attempts_3PT']) * 100
                 top_3pt = stats_3pt.sort_values(by=['3PT FG%', 'Attempts_3PT'], ascending=[False, False]).head(top_n)
                 col_name = group_by_col.replace('_',' ').title()
                 top_3pt = top_3pt.rename(columns={group_by_col: col_name, 'Attempts_3PT': 'Próby 3PT'})
                 top_3pt = top_3pt[[col_name, '3PT FG%', 'Próby 3PT']]
         # else: st.caption(f"Nie znaleziono '{shot_type_3pt}' w SHOT_TYPE dla rankingu 3PT.")
    # else: st.caption("Brak 'SHOT_TYPE' dla rankingu 3PT.")

    # Zwróć DataFrame'y, nawet jeśli są puste
    return top_overall, top_2pt, top_3pt


# --- Główna część aplikacji Streamlit ---
st.title("🏀 Interaktywna Analiza Rzutów Graczy NBA (Sezon 2023-24)")

# Wywołanie funkcji ładowania danych
shooting_data, load_status = load_shooting_data(CSV_FILE_PATH)

# --- Sidebar ---
st.sidebar.header("Opcje Filtrowania i Analizy")

# Główna część aplikacji renderowana tylko jeśli dane są wczytane
if load_status.get("success", False) and not shooting_data.empty:

    # Wyświetl komunikaty o ładowaniu DANE po inicjalizacji UI
    st.success(f"Wczytano dane z {CSV_FILE_PATH}. Wymiary: {load_status.get('shape')}")
    if load_status.get("missing_time_cols"):
          st.warning("Brak jednej lub więcej kolumn czasowych (PERIOD, MINUTES_REMAINING, SECONDS_REMAINING). Nie można było utworzyć 'GAME_TIME_SEC' ani 'QUARTER_TYPE'.")
    if load_status.get("nan_in_time_cols"):
          st.warning("W kolumnach czasowych znaleziono wartości nienumeryczne, które zostały zignorowane lub zastąpione NaN.")
    if load_status.get("nan_in_made_flag"):
          st.warning("W kolumnie SHOT_MADE_FLAG znaleziono wartości, których nie można było jednoznacznie zinterpretować jako 0 lub 1 - zostały one zignorowane w obliczeniach skuteczności.")
    if load_status.get("missing_key_cols"):
          st.warning(f"Brakujące kluczowe kolumny: {', '.join(load_status['missing_key_cols'])}. Niektóre analizy mogą być niedostępne lub niepoprawne.")


    # Sidebar Filters (kod jak poprzednio, ale wewnątrz if'a)
    available_season_types = ['Wszystko'] + shooting_data['SEASON_TYPE'].dropna().unique().tolist() if 'SEASON_TYPE' in shooting_data.columns else ['Wszystko']
    selected_season_type = st.sidebar.selectbox(
        "Wybierz typ sezonu:", options=available_season_types, index=0, key='season_select'
    )
    # Filtruj dane na podstawie sezonu
    if selected_season_type != 'Wszystko' and 'SEASON_TYPE' in shooting_data.columns:
        filtered_data = shooting_data[shooting_data['SEASON_TYPE'] == selected_season_type].copy()
    else:
        filtered_data = shooting_data.copy() # Użyj wszystkich danych, jeśli 'Wszystko' lub brak kolumny
    st.sidebar.write(f"Wybrano: {selected_season_type} ({len(filtered_data)} rzutów)")

    # Player Selection
    available_players = sorted(filtered_data['PLAYER_NAME'].dropna().unique()) if 'PLAYER_NAME' in filtered_data.columns else []
    default_player = "LeBron James"
    default_player_index = None
    if available_players:
        try:
            default_player_index = available_players.index(default_player)
        except ValueError:
            # st.sidebar.warning(f"Domyślny gracz '{default_player}' nie znaleziony. Wybieram pierwszego.") # Mniej komunikatów
            default_player_index = 0
    else:
        default_player_index = 0

    selected_player = st.sidebar.selectbox(
        "Gracz do analizy:", options=available_players, index=default_player_index, # Usunięto warunek if available_players else 0
        key='player_select', disabled=not available_players,
        help="Wybierz gracza do szczegółowej analizy."
    )

    # Team Selection
    available_teams = sorted(filtered_data['TEAM_NAME'].dropna().unique()) if 'TEAM_NAME' in filtered_data.columns else []
    default_team = "Los Angeles Lakers"
    default_team_index = None
    if available_teams:
        try:
            default_team_index = available_teams.index(default_team)
        except ValueError:
            # st.sidebar.warning(f"Domyślna drużyna '{default_team}' nie znaleziona. Wybieram pierwszą.")
            default_team_index = 0
    else:
        default_team_index = 0

    selected_team = st.sidebar.selectbox(
        "Drużyna do analizy:", options=available_teams, index=default_team_index, # Usunięto warunek if available_teams else 0
        key='team_select', disabled=not available_teams,
        help="Wybierz drużynę do analizy zespołowej."
    )

    # Player Comparison Selection
    default_compare_players_req = ["LeBron James", "Stephen Curry"] # Przykładowi gracze
    # Sprawdź, którzy z domyślnych graczy są faktycznie dostępni w *przefiltrowanych* danych
    default_compare_players_available = [p for p in default_compare_players_req if p in available_players]
    # if len(default_compare_players_available) < len(default_compare_players_req) and available_players:
        # missing_defaults = set(default_compare_players_req) - set(default_compare_players_available)
        # st.sidebar.caption(f"Brak domyślnych graczy do porównania: {', '.join(missing_defaults)}")

    selected_players_compare = st.sidebar.multiselect(
        "Gracze do porównania (2-5):", options=available_players, default=default_compare_players_available,
        max_selections=5, key='player_multi_select', disabled=not available_players,
        help="Wybierz od 2 do 5 graczy, aby porównać ich statystyki i mapy rzutów."
    )

    # === Zastąpienie st.tabs widgetem st.radio ===
    tab_options = [
        "📊 Rankingi Skuteczności",
        "⛹️ Analiza Gracza",
        "🆚 Porównanie Graczy",
        "🏀 Analiza Zespołowa",
        "🎯 Model Predykcji (KNN)",
        "🎯 Model Predykcji (XGBoost)"  # <--- DODANO XGBoost
    ]

    # Użyj st.radio do wyboru widoku, ustawiając domyślną wartość z session_state
    st.session_state.active_view = st.radio(
        "Wybierz widok:",
        options=tab_options,
        index=tab_options.index(st.session_state.active_view), # Ustawia zaznaczenie na podstawie stanu
        key='view_selector', # Klucz dla widgetu radio
        horizontal=True,
        label_visibility="collapsed" # Ukryj etykietę "Wybierz widok:"
    )
    st.markdown("---") # Linia oddzielająca

    # === Użycie if/elif zamiast bloków 'with tabX:' ===

    if st.session_state.active_view == "📊 Rankingi Skuteczności":
        # === Zawartość Widoku 1: Rankingi Skuteczności (Przearanżowane + Efektywność Akcji) ===
        # (Kod bez zmian)
        st.header(f"Analiza Ogólna Sezonu: {selected_season_type}")

        # --- 1. Sekcja: Ogólne Rozkłady ---
        st.subheader("Ogólne Rozkłady Rzutów")
        st.caption(f"Dane dla wybranego typu sezonu: {selected_season_type}")
        c1_dist, c2_dist = st.columns(2)

        # Kolumna 1: Rozkład Typów Rzutów (bez zmian)
        with c1_dist:
            if 'SHOT_TYPE' in filtered_data.columns and not filtered_data['SHOT_TYPE'].isnull().all():
                shot_type_counts = filtered_data['SHOT_TYPE'].dropna().value_counts().reset_index()
                if not shot_type_counts.empty:
                     fig_type = px.pie(shot_type_counts, names='SHOT_TYPE', values='count', title='Rozkład Typów Rzutów')
                     fig_type.update_layout(legend_title_text='Typ Rzutu')
                     st.plotly_chart(fig_type, use_container_width=True)
                else: st.caption("Brak danych dla typów rzutów po usunięciu NaN.")
            else: st.caption("Brak kolumny 'SHOT_TYPE'.")

        # Kolumna 2: Najczęstsze ORAZ Najefektywniejsze Typy Akcji
        with c2_dist:
            # -- Najczęstsze Typy Akcji (jak poprzednio) --
            if 'ACTION_TYPE' in filtered_data.columns and not filtered_data['ACTION_TYPE'].isnull().all():
                action_type_counts = filtered_data['ACTION_TYPE'].dropna().value_counts().head(15).reset_index()
                if not action_type_counts.empty:
                    fig_action_freq = px.bar(action_type_counts, y='ACTION_TYPE', x='count', orientation='h', title='Najczęstsze Typy Akcji (Top 15)', labels={'count':'Liczba Rzutów', 'ACTION_TYPE':''})
                    fig_action_freq.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                    st.plotly_chart(fig_action_freq, use_container_width=True)
                else: st.caption("Brak danych dla typów akcji po usunięciu NaN.")
            else: st.caption("Brak kolumny 'ACTION_TYPE' do analizy częstotliwości.")

            st.markdown("---") # Separator wewnątrz kolumny

            # -- NOWA CZĘŚĆ: Najefektywniejsze Typy Akcji --
            st.markdown("###### Najefektywniejsze Typy Akcji")
            if 'ACTION_TYPE' in filtered_data.columns and 'SHOT_MADE_FLAG' in filtered_data.columns:
                # Ustawienie minimalnej liczby prób dla rankingu efektywności
                min_attempts_eff_action = st.number_input(
                    "Min. prób dla rankingu skuteczności akcji:",
                    min_value=5, value=10, step=1, key='min_attempts_eff_action_dist', # Unikalny klucz
                    help="Minimalna liczba rzutów danego typu akcji, aby została uwzględniona w rankingu skuteczności."
                )

                # Przygotowanie danych do obliczenia efektywności
                action_eff_data = filtered_data[['ACTION_TYPE', 'SHOT_MADE_FLAG']].copy()
                action_eff_data['ACTION_TYPE'] = action_eff_data['ACTION_TYPE'].astype(str).str.strip()
                action_eff_data['SHOT_MADE_FLAG'] = pd.to_numeric(action_eff_data['SHOT_MADE_FLAG'], errors='coerce')
                action_eff_data = action_eff_data.dropna()

                if not action_eff_data.empty:
                    # Grupowanie i obliczanie statystyk
                    action_stats = action_eff_data.groupby('ACTION_TYPE')['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
                    # Filtracja wg minimalnej liczby prób
                    action_stats_filtered = action_stats[action_stats['count'] >= min_attempts_eff_action].copy()

                    if not action_stats_filtered.empty:
                        # Obliczanie procentów i sortowanie
                        action_stats_filtered['FG%'] = action_stats_filtered['mean'] * 100
                        action_stats_filtered = action_stats_filtered.sort_values(by='FG%', ascending=False).head(15) # Top 15 najefektywniejszych

                        # Wykres słupkowy efektywności
                        fig_action_eff = px.bar(
                            action_stats_filtered,
                            y='ACTION_TYPE',
                            x='FG%',
                            orientation='h',
                            title=f'Najefektywniejsze Typy Akcji (Top 15, min. {min_attempts_eff_action} prób)',
                            labels={'FG%': 'Skuteczność (%)', 'ACTION_TYPE': ''},
                            text='FG%', # Pokaż wartość % na słupku
                            hover_data=['count'] # Pokaż liczbę prób w tooltipie
                        )
                        fig_action_eff.update_layout(
                            yaxis={'categoryorder': 'total ascending'}, # Sortuj słupki wg wartości (od najwyższej FG%)
                            xaxis_range=[0, 105], # Ustaw zakres osi X dla %
                            height=400 # Dostosuj wysokość
                        )
                        fig_action_eff.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        st.plotly_chart(fig_action_eff, use_container_width=True)
                    else:
                        st.caption(f"Brak typów akcji spełniających kryterium min. {min_attempts_eff_action} prób.")
                else:
                    st.caption("Brak wystarczających danych do obliczenia skuteczności akcji.")
            else:
                st.caption("Brak kolumn 'ACTION_TYPE' lub 'SHOT_MADE_FLAG' do analizy skuteczności.")


        st.markdown("---") # Separator przed kolejną główną sekcją

        # --- 2. Sekcja: Mapa Częstotliwości Rzutów ---
        # (Bez zmian w stosunku do poprzedniej wersji)
        st.subheader(f"Mapa Częstotliwości Rzutów")
        st.caption(f"Pokazuje gęstość rzutów na boisku dla: {selected_season_type}")
        st.markdown("Ciemniejszy/cieplejszy kolor oznacza więcej rzutów z danego obszaru.")
        num_bins_freq = st.slider("Dokładność mapy (liczba kwadratów na oś X):", 20, 80, 50, 5, key='frequency_map_bins_slider_reorder')
        fig_freq_map = plot_shot_frequency_heatmap(filtered_data, selected_season_type, nbins_x=num_bins_freq, nbins_y=int(num_bins_freq * 1.9))
        if fig_freq_map: st.plotly_chart(fig_freq_map, use_container_width=True)
        else: st.caption("Nie udało się wygenerować mapy częstotliwości rzutów (być może brak danych lokalizacji).")
        st.markdown("---") # Separator

        # --- 3. Sekcja: Rankingi Skuteczności ---
        # (Bez zmian w stosunku do poprzedniej wersji)
        st.subheader(f"Rankingi Skuteczności")
        st.caption(f"Top 10 graczy/zespołów dla typu sezonu: {selected_season_type}")
        st.markdown("##### Ustaw minimalną liczbę prób")
        col_att1, col_att2, col_att3 = st.columns(3)
        with col_att1: min_total = st.number_input("Min. rzutów ogółem:", 10, 1000, 100, 10, key="min_total_r_reorder")
        with col_att2: min_2pt = st.number_input("Min. rzutów za 2 pkt:", 5, 500, 50, 5, key="min_2pt_r_reorder")
        with col_att3: min_3pt = st.number_input("Min. rzutów za 3 pkt:", 5, 500, 30, 5, key="min_3pt_r_reorder")
        tp_ov, tp_2, tp_3 = calculate_top_performers(filtered_data, 'PLAYER_NAME', min_total, min_2pt, min_3pt)
        tt_ov, tt_2, tt_3 = calculate_top_performers(filtered_data, 'TEAM_NAME', min_total*5, min_2pt*5, min_3pt*5)
        st.markdown("###### Skuteczność Ogółem (FG%)")
        c1_rank, c2_rank = st.columns(2)
        with c1_rank:
            st.markdown(f"**Top 10 Graczy (min. {min_total} prób)**")
            if tp_ov is not None and not tp_ov.empty: st.dataframe(tp_ov, use_container_width=True, hide_index=True, column_config={"FG%": st.column_config.ProgressColumn("FG%", format="%.1f%%", min_value=0, max_value=100)})
            else: st.caption("Brak graczy spełniających kryteria.")
        with c2_rank:
            st.markdown(f"**Top 10 Zespołów (min. {min_total*5} prób)**")
            if tt_ov is not None and not tt_ov.empty: st.dataframe(tt_ov, use_container_width=True, hide_index=True, column_config={"FG%": st.column_config.ProgressColumn("FG%", format="%.1f%%", min_value=0, max_value=100)})
            else: st.caption("Brak zespołów spełniających kryteria.")
        st.markdown("###### Skuteczność za 2 Punkty (2PT FG%)")
        c1_rank_2, c2_rank_2 = st.columns(2)
        with c1_rank_2:
            st.markdown(f"**Top 10 Graczy (min. {min_2pt} prób)**")
            if tp_2 is not None and not tp_2.empty: st.dataframe(tp_2, use_container_width=True, hide_index=True, column_config={"2PT FG%": st.column_config.ProgressColumn("2PT FG%", format="%.1f%%", min_value=0, max_value=100)})
            else: st.caption("Brak graczy spełniających kryteria.")
        with c2_rank_2:
            st.markdown(f"**Top 10 Zespołów (min. {min_2pt*5} prób)**")
            if tt_2 is not None and not tt_2.empty: st.dataframe(tt_2, use_container_width=True, hide_index=True, column_config={"2PT FG%": st.column_config.ProgressColumn("2PT FG%", format="%.1f%%", min_value=0, max_value=100)})
            else: st.caption("Brak zespołów spełniających kryteria.")
        st.markdown("###### Skuteczność za 3 Punkty (3PT FG%)")
        c1_rank_3, c2_rank_3 = st.columns(2)
        with c1_rank_3:
            st.markdown(f"**Top 10 Graczy (min. {min_3pt} prób)**")
            if tp_3 is not None and not tp_3.empty: st.dataframe(tp_3, use_container_width=True, hide_index=True, column_config={"3PT FG%": st.column_config.ProgressColumn("3PT FG%", format="%.1f%%", min_value=0, max_value=100)})
            else: st.caption("Brak graczy spełniających kryteria.")
        with c2_rank_3:
            st.markdown(f"**Top 10 Zespołów (min. {min_3pt*5} prób)**")
            if tt_3 is not None and not tt_3.empty: st.dataframe(tt_3, use_container_width=True, hide_index=True, column_config={"3PT FG%": st.column_config.ProgressColumn("3PT FG%", format="%.1f%%", min_value=0, max_value=100)})
            else: st.caption("Brak zespołów spełniających kryteria.")
        # === Koniec Widoku 1 ===

    elif st.session_state.active_view == "⛹️ Analiza Gracza":
        # === Zawartość Zakładki 2: Analiza Gracza ===
        # (Kod bez zmian)
        st.header(f"Analiza Gracza: {selected_player}")
        if selected_player and 'PLAYER_NAME' in filtered_data.columns:
            player_data = filter_data_by_player(selected_player, filtered_data)
            if not player_data.empty:
                # Stats Section (Top)
                st.subheader("Statystyki Podstawowe")
                total_shots = len(player_data)
                made_shots, shooting_pct = "N/A", "N/A"
                if 'SHOT_MADE_FLAG' in player_data.columns:
                     made_flag_numeric = pd.to_numeric(player_data['SHOT_MADE_FLAG'], errors='coerce').dropna()
                     if not made_flag_numeric.empty:
                         made_shots = int(made_flag_numeric.sum())
                         if len(made_flag_numeric) > 0: shooting_pct = (made_shots / len(made_flag_numeric)) * 100
                         else: shooting_pct = 0.0
                     else: made_shots, shooting_pct = 0, 0.0
                pct_2pt, attempts_2pt_str = "N/A", "(brak danych)"
                pct_3pt, attempts_3pt_str = "N/A", "(brak danych)"
                if 'SHOT_TYPE' in player_data.columns and not player_data['SHOT_TYPE'].isnull().all():
                     shot_type_2pt = '2PT Field Goal' # Adjust if needed
                     data_2pt = player_data[player_data['SHOT_TYPE'] == shot_type_2pt]
                     made_flag_2pt = pd.to_numeric(data_2pt['SHOT_MADE_FLAG'], errors='coerce').dropna()
                     attempts_2pt = len(made_flag_2pt)
                     if attempts_2pt > 0: made_2pt = int(made_flag_2pt.sum()); pct_2pt = (made_2pt / attempts_2pt) * 100; attempts_2pt_str = f"({made_2pt}/{attempts_2pt})"
                     else: attempts_2pt_str = "(0 prób)"
                     shot_type_3pt = '3PT Field Goal' # Adjust if needed
                     data_3pt = player_data[player_data['SHOT_TYPE'] == shot_type_3pt]
                     made_flag_3pt = pd.to_numeric(data_3pt['SHOT_MADE_FLAG'], errors='coerce').dropna()
                     attempts_3pt = len(made_flag_3pt)
                     if attempts_3pt > 0: made_3pt = int(made_flag_3pt.sum()); pct_3pt = (made_3pt / attempts_3pt) * 100; attempts_3pt_str = f"({made_3pt}/{attempts_3pt})"
                     else: attempts_3pt_str = "(0 prób)"
                avg_dist = "N/A"
                if 'SHOT_DISTANCE' in player_data.columns and pd.api.types.is_numeric_dtype(player_data['SHOT_DISTANCE']):
                     valid_distances = player_data['SHOT_DISTANCE'].dropna()
                     if not valid_distances.empty: avg_dist = valid_distances.mean()

                st.markdown("##### Statystyki Ogólne")
                c1, c2, c3 = st.columns(3)
                c1.metric("Całk. rzutów", total_shots)
                c2.metric("Trafione", f"{made_shots}" if isinstance(made_shots, int) else "N/A")
                c3.metric("Skut. (FG%)", f"{shooting_pct:.1f}%" if isinstance(shooting_pct, (float, int)) else "N/A")
                st.markdown("##### Statystyki Szczegółowe")
                c4, c5, c6 = st.columns(3)
                c4.metric(f"Skut. 2pkt {attempts_2pt_str}", f"{pct_2pt:.1f}%" if isinstance(pct_2pt, (float, int)) else "N/A")
                c5.metric(f"Skut. 3pkt {attempts_3pt_str}", f"{pct_3pt:.1f}%" if isinstance(pct_3pt, (float, int)) else "N/A")
                c6.metric("Śr. Odległość (stopy)", f"{avg_dist:.1f}" if isinstance(avg_dist, (float, int)) else "N/A")
                st.markdown("---")

                # Visualizations Section
                st.subheader("Wizualizacje Rzutów")
                fig_p_chart = plot_shot_chart(player_data, selected_player, "Gracz")
                if fig_p_chart: st.plotly_chart(fig_p_chart, use_container_width=True)
                else: st.warning("Nie można wygenerować mapy rzutów (być może brak danych lokalizacji).")
                st.markdown("---")

                st.subheader("Skuteczność vs Odległość")
                cc1, cc2 = st.columns([1,2]) # Daj więcej miejsca na suwak min prób
                with cc1: bin_w = st.slider("Szerokość przedziału (stopy):", 1, 5, 1, key='p_eff_dist_bin')
                with cc2: min_att = st.slider("Min. prób w przedziale:", 1, 50, 5, key='p_eff_dist_min') # Domyślnie 5 prób
                fig_p_eff_dist = plot_player_eff_vs_distance(player_data, selected_player, bin_width=bin_w, min_attempts_per_bin=min_att)
                if fig_p_eff_dist: st.plotly_chart(fig_p_eff_dist, use_container_width=True)
                else: st.caption(f"Nie udało się wygenerować wykresu skuteczności vs odległość (min. {min_att} prób / {bin_w} stóp).")
                st.markdown("---")

                st.subheader("Strefy Rzutowe ('Hot Zones')")
                hz_min_shots = st.slider("Min. rzutów w strefie:", 3, 50, 5, key='p_hotzone_min_shots')
                hz_bins = st.slider("Liczba stref na oś:", 5, 15, 10, key='p_hotzone_bins')
                p_hot_zones = calculate_hot_zones(player_data, min_shots_in_zone=hz_min_shots, n_bins=hz_bins)
                if p_hot_zones is not None and not p_hot_zones.empty:
                    fig_p_hot = plot_hot_zones_heatmap(p_hot_zones, selected_player, "Gracz", min_shots_in_zone=hz_min_shots)
                    if fig_p_hot: st.plotly_chart(fig_p_hot, use_container_width=True)
                    else: st.info("Nie można wygenerować mapy stref.")
                else: st.info(f"Brak wystarczających danych do analizy stref (min. {hz_min_shots} prób na strefę).")
                st.markdown("---")

                st.subheader("Analiza Czasowa")
                q_min_shots = st.slider("Min. prób w kwarcie/OT:", 3, 50, 5, key='p_quarter_min_shots')
                fig_p_q = plot_player_quarter_eff(player_data, selected_player, min_attempts=q_min_shots)
                if fig_p_q: st.plotly_chart(fig_p_q, use_container_width=True)
                else: st.info(f"Brak danych do analizy skuteczności w kwartach (min. {q_min_shots} prób).")

                m_min_shots = st.slider("Min. prób w miesiącu:", 5, 100, 10, key='p_month_min_shots')
                fig_p_t = plot_player_season_trend(player_data, selected_player, min_monthly_attempts=m_min_shots)
                if fig_p_t: st.plotly_chart(fig_p_t, use_container_width=True)
                else: st.info(f"Brak danych do analizy trendu sezonowego (min. {m_min_shots} prób/miesiąc, wymagane min. 2 miesiące).")
                st.markdown("---")

                st.subheader("Analiza wg Typu Akcji / Strefy")
                g_min_shots = st.slider("Min. prób w grupie (typ akcji/strefa):", 3, 50, 5, key='p_group_min_shots')
                cg1, cg2 = st.columns(2)
                with cg1:
                    fig_p_a = plot_grouped_effectiveness(player_data, 'ACTION_TYPE', selected_player, "Gracz", top_n=10, min_attempts=g_min_shots)
                    if fig_p_a: st.plotly_chart(fig_p_a, use_container_width=True)
                    else: st.caption(f"Brak danych wg typu akcji (min. {g_min_shots} prób).")
                with cg2:
                    fig_p_z = plot_grouped_effectiveness(player_data, 'SHOT_ZONE_BASIC', selected_player, "Gracz", top_n=7, min_attempts=g_min_shots)
                    if fig_p_z: st.plotly_chart(fig_p_z, use_container_width=True)
                    else: st.caption(f"Brak danych wg strefy podstawowej (min. {g_min_shots} prób).")
            else:
                st.warning(f"Brak danych dla gracza '{selected_player}' w wybranym typie sezonu.")
        else:
            st.info("Wybierz gracza z panelu bocznego.")
        # === Koniec Zakładki 2 ===


    elif st.session_state.active_view == "🆚 Porównanie Graczy":
        # === Zawartość Zakładki 3: Porównanie Graczy ===
        # (Kod bez zmian)
        st.header("Porównanie Graczy")
        if len(selected_players_compare) >= 2:
            st.write(f"Porównujesz: {', '.join(selected_players_compare)}")
            # Filtruj dane *główne* (nie tylko przefiltrowane wg sezonu), aby mieć pewność, że gracze istnieją
            compare_data_base = shooting_data[shooting_data['PLAYER_NAME'].isin(selected_players_compare)].copy()
            # Następnie zastosuj filtr sezonu, jeśli jest aktywny
            if selected_season_type != 'Wszystko' and 'SEASON_TYPE' in compare_data_base.columns:
                 compare_data_filtered = compare_data_base[compare_data_base['SEASON_TYPE'] == selected_season_type].copy()
            else:
                 compare_data_filtered = compare_data_base.copy()

            if not compare_data_filtered.empty:
                st.subheader("Skuteczność vs Odległość")
                comp_eff_dist_bin = st.slider("Szerokość przedziału odległości (stopy):", 1, 5, 3, key='comp_eff_dist_bin')
                comp_eff_dist_min = st.slider("Min. prób w przedziale odl.:", 3, 50, 5, key='comp_eff_dist_min')
                fig_comp_eff_dist = plot_comparison_eff_distance(compare_data_filtered, selected_players_compare, bin_width=comp_eff_dist_bin, min_attempts_per_bin=comp_eff_dist_min)
                if fig_comp_eff_dist: st.plotly_chart(fig_comp_eff_dist, use_container_width=True)
                else: st.caption(f"Nie można wygenerować wykresu porównania skuteczności vs odległość (min. {comp_eff_dist_min} prób / {comp_eff_dist_bin} stóp).")
                st.markdown("---")

                st.subheader("Skuteczność wg Strefy Rzutowej (SHOT_ZONE_BASIC)")
                min_attempts_zone = st.slider("Min. prób w strefie:", 3, 50, 5, key='compare_zone_min_slider')
                fig_comp_zone = plot_comparison_eff_by_zone(compare_data_filtered, selected_players_compare, min_shots_per_zone=min_attempts_zone)
                if fig_comp_zone: st.plotly_chart(fig_comp_zone, use_container_width=True)
                else: st.caption(f"Nie udało się wygenerować porównania wg stref (min. {min_attempts_zone} prób).")
                st.markdown("---")

                st.subheader("Mapy Rzutów")
                num_players_comp = len(selected_players_compare)
                cols = st.columns(num_players_comp)
                for i, player in enumerate(selected_players_compare):
                    with cols[i]:
                        st.markdown(f"**{player}**")
                        # Filtruj dane dla konkretnego gracza z już przefiltrowanych danych porównawczych
                        player_comp_data = compare_data_filtered[compare_data_filtered['PLAYER_NAME'] == player]
                        if not player_comp_data.empty:
                            fig_comp_chart = plot_shot_chart(player_comp_data, player, "Gracz")
                            if fig_comp_chart:
                                fig_comp_chart.update_layout(height=450, title="") # Usunięcie tytułu dla oszczędności miejsca
                                st.plotly_chart(fig_comp_chart, use_container_width=True, key=f"comp_ch_{player.replace(' ','_')}") # Unikalny klucz
                            else: st.caption("Błąd mapy.")
                        else: st.caption("Brak danych w wybranym sezonie.")
            else:
                st.warning(f"Brak danych dla wybranych graczy w sezonie '{selected_season_type}'.")
        else:
            st.info("Wybierz min. 2 graczy do porównania z panelu bocznego.")
        # === Koniec Zakładki 3 ===


    elif st.session_state.active_view == "🏀 Analiza Zespołowa":
        # === Zawartość Zakładki 4: Analiza Zespołowa ===
        # (Kod bez zmian)
        st.header(f"Analiza Zespołowa: {selected_team}")
        if selected_team and 'TEAM_NAME' in filtered_data.columns:
            team_data = filter_data_by_team(selected_team, filtered_data)
            if not team_data.empty:
                st.subheader("Statystyki Podstawowe Zespołu")
                t_stats = get_basic_stats(team_data, selected_team, "Zespół") # Użyj funkcji get_basic_stats
                c1, c2, c3 = st.columns(3)
                c1.metric("Rzuty Zespołu", t_stats.get('total_shots', 'N/A'))
                c2.metric("Trafione Zespołu", t_stats.get('made_shots', 'N/A'))
                pct_val = t_stats.get('shooting_pct')
                c3.metric("Skuteczność Zespołu", f"{pct_val:.1f}%" if isinstance(pct_val, (float, int)) else "N/A")
                st.markdown("---")

                st.subheader("Wizualizacje Rzutów Zespołu")
                fig_t_c = plot_shot_chart(team_data, selected_team, "Zespół")
                if fig_t_c: st.plotly_chart(fig_t_c, use_container_width=True)
                else: st.warning("Nie można wygenerować mapy rzutów zespołu.")

                st.subheader("Skuteczność Zespołu vs Odległość")
                # Użyjemy plot_player_eff_vs_distance, bo działa tak samo dla danych zespołu
                t_eff_dist_bin = st.slider("Szerokość przedziału odległości (stopy):", 1, 5, 2, key='t_eff_dist_bin')
                t_eff_dist_min = st.slider("Min. prób w przedziale odl.:", 5, 100, 10, key='t_eff_dist_min')
                fig_t_ed = plot_player_eff_vs_distance(team_data, selected_team, bin_width=t_eff_dist_bin, min_attempts_per_bin=t_eff_dist_min)
                if fig_t_ed: st.plotly_chart(fig_t_ed, use_container_width=True)
                else: st.caption(f"Nie udało się wygenerować wykresu skuteczności vs odległość dla zespołu (min. {t_eff_dist_min} prób / {t_eff_dist_bin} stóp).")
                st.markdown("---")

                st.subheader("Strefy Rzutowe ('Hot Zones') Zespołu")
                t_hz_min_shots = st.slider("Min. rzutów w strefie:", 5, 100, 10, key='t_hotzone_min_shots')
                t_hz_bins = st.slider("Liczba stref na oś:", 5, 15, 10, key='t_hotzone_bins')
                t_hz = calculate_hot_zones(team_data, min_shots_in_zone=t_hz_min_shots, n_bins=t_hz_bins)
                if t_hz is not None and not t_hz.empty:
                    fig_t_h = plot_hot_zones_heatmap(t_hz, selected_team, "Zespół", min_shots_in_zone=t_hz_min_shots)
                    if fig_t_h: st.plotly_chart(fig_t_h, use_container_width=True)
                    else: st.info("Nie można wygenerować mapy stref zespołu.")
                else: st.info(f"Brak danych do analizy stref zespołu (min. {t_hz_min_shots} prób na strefę).")
                st.markdown("---")

                st.subheader("Analiza Czasowa Zespołu")
                 # Użyjemy plot_player_quarter_eff
                t_q_min_shots = st.slider("Min. prób w kwarcie/OT:", 5, 100, 10, key='t_quarter_min_shots')
                fig_t_q = plot_player_quarter_eff(team_data, selected_team, "Zespół", min_attempts=t_q_min_shots)
                if fig_t_q: st.plotly_chart(fig_t_q, use_container_width=True)
                else: st.info(f"Brak danych do analizy skuteczności zespołu w kwartach (min. {t_q_min_shots} prób).")
                # Trend sezonowy może być mniej sensowny dla całego zespołu, pomijamy go
                st.markdown("---")

                st.subheader("Analiza Zespołu wg Typu Akcji / Strefy")
                 # Użyjemy plot_grouped_effectiveness
                t_g_min_shots = st.slider("Min. prób w grupie (typ akcji/strefa):", 5, 100, 10, key='t_group_min_shots')
                cg1, cg2 = st.columns(2)
                with cg1:
                    fig_t_a = plot_grouped_effectiveness(team_data, 'ACTION_TYPE', selected_team, "Zespół", top_n=10, min_attempts=t_g_min_shots)
                    if fig_t_a: st.plotly_chart(fig_t_a, use_container_width=True)
                    else: st.caption(f"Brak danych zespołu wg typu akcji (min. {t_g_min_shots} prób).")
                with cg2:
                    fig_t_z = plot_grouped_effectiveness(team_data, 'SHOT_ZONE_BASIC', selected_team, "Zespół", top_n=7, min_attempts=t_g_min_shots)
                    if fig_t_z: st.plotly_chart(fig_t_z, use_container_width=True)
                    else: st.caption(f"Brak danych zespołu wg strefy podstawowej (min. {t_g_min_shots} prób).")
            else:
                st.warning(f"Brak danych dla drużyny '{selected_team}' w wybranym typie sezonu.")
        else:
            st.info("Wybierz drużynę z panelu bocznego.")
        # === Koniec Zakładki 4 ===


    elif st.session_state.active_view == "🎯 Model Predykcji (KNN)":
        # === Zawartość Zakładki 5: Model Predykcji (KNN) - Rozszerzony o Cechy Kategoryczne ===
        # (Kod bez zmian)
        st.header(f"Model Predykcji Rzutów (KNN) dla: {selected_player}")

        # --- ZAKTUALIZOWANY TEKST INTERPRETACJI ---
        st.markdown(f"""
        ### Interpretacja Wyników Rozszerzonego Modelu KNN ({selected_player})

        Zakładka ta prezentuje model K-Najbliższych Sąsiadów (KNN) do przewidywania wyniku rzutu (celny/niecelny), **rozszerzony o dodatkowe cechy kategoryczne:** `ACTION_TYPE`, `SHOT_TYPE` i `PERIOD`. Ocenia jego wydajność na dwa sposoby:

        **Obsługa Cech Kategorycznych:**

        * Algorytm KNN działa na podstawie odległości między punktami danych w przestrzeni cech. Aby uwzględnić cechy nienumeryczne (jak typ akcji, typ rzutu czy kwarta), zostały one przekształcone za pomocą **One-Hot Encoding (OHE)**.
        * OHE tworzy nowe, binarne (0 lub 1) kolumny dla każdej unikalnej wartości w oryginalnej kolumnie kategorycznej. Na przykład, dla `SHOT_TYPE` mogą powstać kolumny `SHOT_TYPE_2PT Field Goal` i `SHOT_TYPE_3PT Field Goal`.
        * Cechy numeryczne (`LOC_X`, `LOC_Y`, `SHOT_DISTANCE`) są nadal **skalowane** za pomocą `StandardScaler`, aby miały podobny wpływ na obliczaną odległość.
        * Zarówno OHE, jak i skalowanie są wykonywane w ramach `Pipeline` za pomocą `ColumnTransformer`, co zapewnia spójne przetwarzanie danych treningowych i testowych.

        **1. Walidacja Krzyżowa (Stratified K-Fold) - Ocena Ogólnej Wydajności:**

        * **Cel:** Uzyskanie bardziej **niezawodnej i stabilnej** oceny, jak dobrze *rozszerzony* model prawdopodobnie będzie działał na nowych danych.
        * **Jak działa:** Dane gracza są dzielone na `n` części (folds) z zachowaniem proporcji klas. Model (teraz `ColumnTransformer` + `KNN` w `Pipeline`) jest trenowany `n` razy na `n-1` częściach i testowany na pozostałej.
        * **Wyniki:** Średnia dokładność i odchylenie standardowe pokazują oczekiwaną skuteczność i stabilność *rozszerzonego* modelu.

        **2. Ocena na Pojedynczym Podziale Trening/Test - Szczegółowa Analiza:**

        * **Cel:** Zaprezentowanie **szczegółowych metryk** (Raport Klasyfikacji) i **wizualizacji błędów** (Macierz Pomyłek) dla *jednego konkretnego* podziału danych, aby zrozumieć, jakie błędy popełnia *rozszerzony* model.
        * **Jak działa:** Dane są jednorazowo dzielone (zgodnie z wybranym procentem). Ten sam `Pipeline` (`ColumnTransformer` + `KNN`) jest trenowany na zbiorze treningowym i oceniany na testowym.
        * **Wyniki:** Dokładność, Raport Klasyfikacji (Precision, Recall, F1-Score, Support) i Macierz Pomyłek (TP, TN, FP, FN) pokazują szczegółowe działanie *rozszerzonego* modelu na tym konkretnym podziale.

        **Podsumowanie:** Walidacja krzyżowa daje lepszy obraz *ogólnej* wydajności rozszerzonego modelu, podczas gdy pojedynczy podział dostarcza *szczegółowego wglądu* w jego działanie. Wyniki z pojedynczego podziału mogą zależeć od losowego podziału (`random_state=42`). Dodanie cech kategorycznych może (ale nie musi) poprawić dokładność modelu, ale zwiększa też złożoność (więcej wymiarów po OHE).
        """) # KONIEC TEKSTU INTERPRETACJI

        # --- Reszta kodu dla Modelu KNN ---
        if selected_player:
            player_model_data = filter_data_by_player(selected_player, filtered_data)

            if not player_model_data.empty:
                # === ZMIANA: Definicja cech numerycznych i kategorycznych ===
                numerical_features = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE']
                categorical_features = ['ACTION_TYPE', 'SHOT_TYPE', 'PERIOD'] # Nowe cechy
                target_variable = 'SHOT_MADE_FLAG'
                all_features = numerical_features + categorical_features

                # Sprawdzenie kolumn
                if all(feat in player_model_data.columns for feat in all_features) and target_variable in player_model_data.columns:

                    # === ZMIANA: Przygotowanie danych - uwzględnienie wszystkich cech ===
                    # Wybierz potrzebne kolumny i usuń wiersze z NaN w *którejkolwiek* z tych kolumn
                    pmdc = player_model_data[all_features + [target_variable]].dropna().copy()

                    # Konwersja targetu i upewnienie się, że nie ma NaN po konwersji
                    pmdc[target_variable] = pd.to_numeric(pmdc[target_variable], errors='coerce')
                    pmdc = pmdc.dropna(subset=[target_variable])

                    # === NOWOŚĆ: Konwersja cech kategorycznych na string przed OHE ===
                    for col in categorical_features:
                        pmdc[col] = pmdc[col].astype(str)

                    # Sprawdzenie czy są 2 klasy PO usunięciu NaN
                    if pmdc[target_variable].nunique() != 2 and not pmdc.empty:
                             st.warning(f"Po usunięciu brakujących wartości w '{target_variable}' lub cechach, pozostała tylko jedna klasa wyniku rzutu ({int(pmdc[target_variable].unique()[0]) if not pmdc.empty else 'brak'}). Nie można zbudować modelu.")
                    elif pmdc.empty:
                             st.warning(f"Brak ważnych danych (po usunięciu NaN we wszystkich cechach: {', '.join(all_features)}, {target_variable}) dla gracza '{selected_player}'.")
                    else:
                        pmdc[target_variable] = pmdc[target_variable].astype(int)

                        # Sprawdzenie warunków ilości danych PO usunięciu NaN
                        min_samples_for_model = 50 # Można dostosować
                        if len(pmdc) >= min_samples_for_model:

                            st.subheader("Konfiguracja Modelu KNN i Oceny")
                            # Suwaki konfiguracji KNN i CV (bez zmian)
                            k = st.slider(
                                "Liczba sąsiadów (k):", min_value=3, max_value=min(25, len(pmdc)//3) if len(pmdc) >= 9 else 3,
                                value=5, step=2, key='knn_k_slider_cv_detailed_cat' # Zmieniono klucz
                            )
                            n_splits = st.slider(
                                "Liczba podziałów walidacji krzyżowej (folds):", min_value=3, max_value=10, value=5, step=1,
                                key='knn_cv_splits_detailed_cat' # Zmieniono klucz
                            )

                            # Konfiguracja Podziału Testowego (bez zmian)
                            st.markdown("---")
                            st.subheader("Konfiguracja Pojedynczego Podziału Testowego")
                            test_size_percent = st.slider(
                                "Rozmiar zbioru testowego (%):",
                                min_value=10, max_value=50,
                                value=20,
                                step=5,
                                key='test_split_slider_cat', # Zmieniono klucz
                                format="%d%%",
                                help="Procent danych użyty jako zbiór testowy. Reszta zostanie użyta do treningu."
                            )
                            train_size_percent = 100 - test_size_percent
                            test_size_float = test_size_percent / 100.0

                            # Przycisk uruchamiający obie oceny
                            if st.button(f"Uruchom Ocenę Rozszerzonego Modelu KNN dla {selected_player}", key='run_eval_knn_button_cat'): # Zmieniono klucz

                                # === ZMIANA: Definicja preprocesora i Pipeline ===
                                # Preprocessor używający ColumnTransformer
                                preprocessor = ColumnTransformer(
                                    transformers=[
                                        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
                                        ('num', StandardScaler(), numerical_features)
                                    ],
                                    remainder='passthrough' # Na wszelki wypadek, gdyby były inne kolumny (choć nie powinno być)
                                )

                                # Nowy Pipeline
                                pipeline = Pipeline([
                                    ('preprocessor', preprocessor),
                                    ('knn', KNeighborsClassifier(n_neighbors=k))
                                ])

                                X = pmdc[all_features] # Używamy teraz wszystkich zdefiniowanych cech
                                y = pmdc[target_variable]

                                # --- 1. Walidacja Krzyżowa ---
                                st.markdown("---")
                                st.subheader(f"1. Wyniki {n_splits}-krotnej Walidacji Krzyżowej (z cechami kat.)")
                                with st.spinner(f"Przeprowadzanie walidacji krzyżowej (k={k}, folds={n_splits})..."):
                                    try:
                                        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                                        # Upewnij się, że X jest przekazywane do cross_val_score
                                        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                                        st.success("Walidacja krzyżowa zakończona.")
                                        st.metric("Średnia dokładność (Accuracy):", f"{scores.mean():.2%}")
                                        st.metric("Odchylenie standardowe dokładności:", f"{scores.std():.4f} (≈ {scores.std()*100:.2f} p.p.)")
                                        st.write("Wyniki dokładności dla poszczególnych podziałów:")
                                        scores_formatted = [f"{s:.2%}" for s in scores]
                                        st.text(", ".join(scores_formatted))
                                        st.caption(f"Model oceniony na {len(pmdc)} próbkach (po usunięciu NaN).")
                                    except Exception as e_cv:
                                        st.error(f"Błąd podczas walidacji krzyżowej: {e_cv}")

                                # --- 2. Ocena na Pojedynczym Podziale Train/Test ---
                                st.markdown("---")
                                st.subheader(f"2. Szczegółowa Ocena na Podziale ({train_size_percent}%/{test_size_percent}%) (z cechami kat.)")
                                st.caption(f"Wyniki dla podziału: {train_size_percent}% trening / {test_size_percent}% test (random_state=42).")

                                with st.spinner(f"Przygotowanie i ocena na podziale {train_size_percent}/{test_size_percent} (k={k})..."):
                                    try:
                                        X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, test_size=test_size_float, random_state=42, stratify=y
                                        )
                                        if len(X_test) == 0 or len(y_test) == 0:
                                             st.warning(f"Wybrany rozmiar zbioru testowego ({test_size_percent}%) skutkuje pustym zbiorem testowym dla dostępnych danych ({len(y)} próbek). Wybierz większy rozmiar.")
                                        else:
                                            pipeline.fit(X_train, y_train)
                                            y_pred_single = pipeline.predict(X_test)
                                            accuracy_single = accuracy_score(y_test, y_pred_single)
                                            report_single_dict = classification_report(y_test, y_pred_single, target_names=['Niecelny (0)', 'Celny (1)'], output_dict=True, zero_division=0)
                                            conf_matrix_single = confusion_matrix(y_test, y_pred_single)

                                            st.success(f"Ocena na podziale {train_size_percent}/{test_size_percent} zakończona.")
                                            st.metric(f"Dokładność (Accuracy) na tym podziale testowym ({test_size_percent}%):", f"{accuracy_single:.2%}")

                                            # Wyświetlanie raportu i macierzy pomyłek (bez zmian logiki wyświetlania)
                                            st.subheader("Raport Klasyfikacji:")
                                            st.caption("Pokazuje precyzję, pełność (recall) i F1-score dla każdej klasy.")
                                            report_df = pd.DataFrame(report_single_dict).transpose()
                                            if 'support' in report_df.columns: report_df['support'] = report_df['support'].astype(int)
                                            for col in ['precision', 'recall', 'f1-score']:
                                                if col in report_df.columns: report_df[col] = report_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                                            report_df_display = report_df.loc[['Niecelny (0)', 'Celny (1)']]
                                            st.dataframe(report_df_display, use_container_width=True)
                                            st.caption(f"Support: Liczba rzeczywistych próbek w zbiorze testowym ({y_test.shape[0]}). Macro avg F1: {report_df.loc['macro avg','f1-score']}, Weighted avg F1: {report_df.loc['weighted avg','f1-score']}")

                                            st.subheader("Macierz Pomyłek:")
                                            st.caption("Pokazuje, ile rzutów zostało poprawnie (TN, TP) i niepoprawnie (FP, FN) sklasyfikowanych.")
                                            fig_cm = px.imshow(conf_matrix_single, text_auto=True, aspect="auto",
                                                               labels=dict(x="Predykowana Klasa", y="Prawdziwa Klasa", color="Liczba Próbek"),
                                                               x=['Niecelny (0)', 'Celny (1)'], y=['Niecelny (0)', 'Celny (1)'],
                                                               color_continuous_scale=px.colors.sequential.Blues,
                                                               title=f"Macierz Pomyłek (Zbiór Testowy {test_size_percent}%, k={k})")
                                            fig_cm.update_layout(coloraxis_showscale=False)
                                            st.plotly_chart(fig_cm, use_container_width=True)
                                            st.caption(f"TN={conf_matrix_single[0,0]}, FP={conf_matrix_single[0,1]}, FN={conf_matrix_single[1,0]}, TP={conf_matrix_single[1,1]}")
                                            st.caption(f"Pojedynczy podział: {len(X_train)} próbek treningowych ({train_size_percent}%), {len(X_test)} próbek testowych ({test_size_percent}%).")

                                    except ValueError as ve:
                                         if ("test_size=" in str(ve) or "train_size=" in str(ve)) and "samples" in str(ve):
                                             st.error(f"Błąd: Wybrany rozmiar zbioru testowego ({test_size_percent}%) jest prawdopodobnie zbyt mały dla liczby próbek ({len(y)} ogółem) lub liczby próbek w jednej z klas, aby wykonać podział stratyfikowany. Spróbuj wybrać inny procent.")
                                         # Dodaj obsługę błędu, jeśli OHE napotka nieznaną wartość w teście (choć handle_unknown='ignore' powinien temu zapobiec)
                                         elif "found unknown categories" in str(ve):
                                              st.error(f"Błąd: W zbiorze testowym znaleziono kategorie, które nie wystąpiły w zbiorze treningowym. Sprawdź spójność danych. Błąd: {ve}")
                                         else: st.error(f"Wystąpił błąd wartości podczas podziału danych lub oceny: {ve}")
                                    except Exception as e_single:
                                        st.error(f"Wystąpił błąd podczas generowania szczegółowych metryk na podziale testowym: {e_single}")

                        # Komunikaty o błędach ilości danych
                        else: # len(pmdc) < min_samples_for_model
                            st.warning(f"Niewystarczająca ilość danych ({len(pmdc)} próbek po usunięciu NaN we wszystkich wymaganych cechach) dla gracza '{selected_player}' do zbudowania wiarygodnego modelu. Wymagane minimum: {min_samples_for_model}.")

                # Komunikat o błędzie brakujących kolumn
                else:
                    missing_cols = [col for col in all_features + [target_variable] if col not in player_model_data.columns]
                    st.warning(f"Brak wymaganych kolumn dla '{selected_player}' w danych: {', '.join(missing_cols)}. Potrzebne: {', '.join(all_features + [target_variable])}")
            # Komunikat o braku danych dla gracza
            else:
                st.warning(f"Brak danych dla gracza: '{selected_player}'.")
        # Komunikat o konieczności wyboru gracza
        else:
            st.info("Wybierz gracza z panelu bocznego, aby ocenić model KNN.")
        # === Koniec Zakładki 5 ===

    # ==========================================================================
    # === NOWA ZAKŁADKA: XGBoost ===============================================
    # ==========================================================================
    elif st.session_state.active_view == "🎯 Model Predykcji (XGBoost)":
        st.header(f"Model Predykcji Rzutów (XGBoost) dla: {selected_player}")

        # --- TEKST INTERPRETACJI DLA XGBoost ---
        st.markdown(f"""
        ### Interpretacja Wyników Modelu XGBoost ({selected_player})

        Ta zakładka prezentuje model **XGBoost (Extreme Gradient Boosting)** do przewidywania wyniku rzutu (celny/niecelny). Podobnie jak KNN, używa on tych samych cech wejściowych:

        * **Numeryczne:** `LOC_X`, `LOC_Y`, `SHOT_DISTANCE` (skalowane za pomocą `StandardScaler`).
        * **Kategoryczne:** `ACTION_TYPE`, `SHOT_TYPE`, `PERIOD` (przekształcone za pomocą `One-Hot Encoding`).

        Wykorzystano ten sam potok (`Pipeline`) przetwarzania danych co dla KNN, aby zapewnić spójność i umożliwić porównanie wyników. Wydajność modelu XGBoost jest oceniana na dwa sposoby:

        **1. Walidacja Krzyżowa (Stratified K-Fold) - Ocena Ogólnej Wydajności:**

        * **Cel:** Uzyskanie **niezawodnej** oceny, jak dobrze model XGBoost generalizuje na nowych, niewidzianych danych.
        * **Jak działa:** Dane gracza są dzielone na `n` części (folds). Model (potok z `ColumnTransformer` + `XGBClassifier`) jest trenowany `n` razy na `n-1` częściach i testowany na pozostałej.
        * **Wyniki:** Średnia dokładność i odchylenie standardowe pokazują oczekiwaną skuteczność i stabilność modelu XGBoost.

        **2. Ocena na Pojedynczym Podziale Trening/Test - Szczegółowa Analiza:**

        * **Cel:** Zaprezentowanie **szczegółowych metryk** (Raport Klasyfikacji, Macierz Pomyłek) dla *jednego konkretnego* podziału danych, aby zrozumieć, jakie błędy popełnia model XGBoost.
        * **Jak działa:** Dane są jednorazowo dzielone (zgodnie z wybranym procentem). Ten sam potok (`ColumnTransformer` + `XGBClassifier`) jest trenowany na zbiorze treningowym i oceniany na testowym.
        * **Wyniki:** Dokładność, Raport Klasyfikacji i Macierz Pomyłek pokazują działanie modelu na tym konkretnym podziale.

        **Podsumowanie:** XGBoost jest często potężniejszym algorytmem niż KNN, szczególnie na danych tabelarycznych, ale może wymagać więcej zasobów obliczeniowych. Porównaj wyniki z zakładki KNN, aby zobaczyć, który model lepiej sprawdza się dla danego gracza i zestawu danych. Pamiętaj, że wyniki na pojedynczym podziale mogą zależeć od losowości podziału (`random_state=42`).
        """) # KONIEC TEKSTU INTERPRETACJI XGBoost

        # --- Reszta kodu dla Modelu XGBoost (skopiowana i zaadaptowana z KNN) ---
        if selected_player:
            player_model_data = filter_data_by_player(selected_player, filtered_data)

            if not player_model_data.empty:
                # Definicja cech (identyczna jak w KNN)
                numerical_features = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE']
                categorical_features = ['ACTION_TYPE', 'SHOT_TYPE', 'PERIOD']
                target_variable = 'SHOT_MADE_FLAG'
                all_features = numerical_features + categorical_features

                # Sprawdzenie kolumn (identyczne jak w KNN)
                if all(feat in player_model_data.columns for feat in all_features) and target_variable in player_model_data.columns:

                    # Przygotowanie danych (identyczne jak w KNN)
                    pmdc = player_model_data[all_features + [target_variable]].dropna().copy()
                    pmdc[target_variable] = pd.to_numeric(pmdc[target_variable], errors='coerce')
                    pmdc = pmdc.dropna(subset=[target_variable])
                    for col in categorical_features:
                        pmdc[col] = pmdc[col].astype(str)

                    # Sprawdzenie klas i ilości danych (identyczne jak w KNN)
                    if pmdc[target_variable].nunique() != 2 and not pmdc.empty:
                        st.warning(f"Po usunięciu brakujących wartości w '{target_variable}' lub cechach, pozostała tylko jedna klasa wyniku rzutu ({int(pmdc[target_variable].unique()[0]) if not pmdc.empty else 'brak'}). Nie można zbudować modelu.")
                    elif pmdc.empty:
                        st.warning(f"Brak ważnych danych (po usunięciu NaN we wszystkich cechach: {', '.join(all_features)}, {target_variable}) dla gracza '{selected_player}'.")
                    else:
                        pmdc[target_variable] = pmdc[target_variable].astype(int)

                        min_samples_for_model = 50
                        if len(pmdc) >= min_samples_for_model:

                            st.subheader("Konfiguracja Oceny Modelu XGBoost")
                            # Usunięto suwak 'k' - niepotrzebny dla XGBoost
                            # Można dodać suwaki dla hiperparametrów XGBoost (np. n_estimators, max_depth), ale na razie użyjemy domyślnych

                            n_splits_xgb = st.slider( # Zmieniono klucz
                                "Liczba podziałów walidacji krzyżowej (folds):", min_value=3, max_value=10, value=5, step=1,
                                key='xgb_cv_splits_detailed_xgb'
                            )

                            st.markdown("---")
                            st.subheader("Konfiguracja Pojedynczego Podziału Testowego")
                            test_size_percent_xgb = st.slider( # Zmieniono klucz
                                "Rozmiar zbioru testowego (%):",
                                min_value=10, max_value=50,
                                value=20,
                                step=5,
                                key='test_split_slider_xgb',
                                format="%d%%",
                                help="Procent danych użyty jako zbiór testowy. Reszta zostanie użyta do treningu."
                            )
                            train_size_percent_xgb = 100 - test_size_percent_xgb
                            test_size_float_xgb = test_size_percent_xgb / 100.0

                            # Przycisk uruchamiający obie oceny
                            if st.button(f"Uruchom Ocenę Modelu XGBoost dla {selected_player}", key='run_eval_xgb_button_xgb'): # Zmieniono klucz

                                # === ZMIANA: Definicja preprocesora i Pipeline z XGBoost ===
                                # Preprocessor (identyczny jak w KNN)
                                preprocessor = ColumnTransformer(
                                    transformers=[
                                        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
                                        ('num', StandardScaler(), numerical_features)
                                    ],
                                    remainder='passthrough'
                                )

                                # Nowy Pipeline z XGBoost
                                pipeline_xgb = Pipeline([ # Zmieniono nazwę zmiennej
                                    ('preprocessor', preprocessor),
                                    # Używamy XGBClassifier zamiast KNeighborsClassifier
                                    ('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
                                ])

                                X = pmdc[all_features]
                                y = pmdc[target_variable]

                                # --- 1. Walidacja Krzyżowa ---
                                st.markdown("---")
                                st.subheader(f"1. Wyniki {n_splits_xgb}-krotnej Walidacji Krzyżowej (XGBoost)")
                                with st.spinner(f"Przeprowadzanie walidacji krzyżowej (XGBoost, folds={n_splits_xgb})..."):
                                    try:
                                        cv_xgb = StratifiedKFold(n_splits=n_splits_xgb, shuffle=True, random_state=42)
                                        # Używamy pipeline_xgb
                                        scores_xgb = cross_val_score(pipeline_xgb, X, y, cv=cv_xgb, scoring='accuracy', n_jobs=-1)
                                        st.success("Walidacja krzyżowa XGBoost zakończona.")
                                        st.metric("Średnia dokładność (Accuracy):", f"{scores_xgb.mean():.2%}")
                                        st.metric("Odchylenie standardowe dokładności:", f"{scores_xgb.std():.4f} (≈ {scores_xgb.std()*100:.2f} p.p.)")
                                        st.write("Wyniki dokładności dla poszczególnych podziałów:")
                                        scores_formatted_xgb = [f"{s:.2%}" for s in scores_xgb]
                                        st.text(", ".join(scores_formatted_xgb))
                                        st.caption(f"Model XGBoost oceniony na {len(pmdc)} próbkach (po usunięciu NaN).")
                                    except Exception as e_cv_xgb:
                                        st.error(f"Błąd podczas walidacji krzyżowej XGBoost: {e_cv_xgb}")

                                # --- 2. Ocena na Pojedynczym Podziale Train/Test ---
                                st.markdown("---")
                                st.subheader(f"2. Szczegółowa Ocena na Podziale ({train_size_percent_xgb}%/{test_size_percent_xgb}%) (XGBoost)")
                                st.caption(f"Wyniki dla podziału: {train_size_percent_xgb}% trening / {test_size_percent_xgb}% test (random_state=42).")

                                with st.spinner(f"Przygotowanie i ocena XGBoost na podziale {train_size_percent_xgb}/{test_size_percent_xgb}..."):
                                    try:
                                        X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, test_size=test_size_float_xgb, random_state=42, stratify=y
                                        )
                                        if len(X_test) == 0 or len(y_test) == 0:
                                             st.warning(f"Wybrany rozmiar zbioru testowego ({test_size_percent_xgb}%) skutkuje pustym zbiorem testowym dla dostępnych danych ({len(y)} próbek). Wybierz większy rozmiar.")
                                        else:
                                            # Używamy pipeline_xgb
                                            pipeline_xgb.fit(X_train, y_train)
                                            y_pred_single_xgb = pipeline_xgb.predict(X_test)
                                            accuracy_single_xgb = accuracy_score(y_test, y_pred_single_xgb)
                                            report_single_dict_xgb = classification_report(y_test, y_pred_single_xgb, target_names=['Niecelny (0)', 'Celny (1)'], output_dict=True, zero_division=0)
                                            conf_matrix_single_xgb = confusion_matrix(y_test, y_pred_single_xgb)

                                            st.success(f"Ocena XGBoost na podziale {train_size_percent_xgb}/{test_size_percent_xgb} zakończona.")
                                            st.metric(f"Dokładność (Accuracy) na tym podziale testowym ({test_size_percent_xgb}%):", f"{accuracy_single_xgb:.2%}")

                                            # Wyświetlanie raportu i macierzy pomyłek
                                            st.subheader("Raport Klasyfikacji (XGBoost):")
                                            st.caption("Pokazuje precyzję, pełność (recall) i F1-score dla każdej klasy.")
                                            report_df_xgb = pd.DataFrame(report_single_dict_xgb).transpose()
                                            if 'support' in report_df_xgb.columns: report_df_xgb['support'] = report_df_xgb['support'].astype(int)
                                            for col in ['precision', 'recall', 'f1-score']:
                                                if col in report_df_xgb.columns: report_df_xgb[col] = report_df_xgb[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                                            report_df_display_xgb = report_df_xgb.loc[['Niecelny (0)', 'Celny (1)']]
                                            st.dataframe(report_df_display_xgb, use_container_width=True)
                                            st.caption(f"Support: Liczba rzeczywistych próbek w zbiorze testowym ({y_test.shape[0]}). Macro avg F1: {report_df_xgb.loc['macro avg','f1-score']}, Weighted avg F1: {report_df_xgb.loc['weighted avg','f1-score']}")


                                            st.subheader("Macierz Pomyłek (XGBoost):")
                                            st.caption("Pokazuje, ile rzutów zostało poprawnie (TN, TP) i niepoprawnie (FP, FN) sklasyfikowanych.")
                                            fig_cm_xgb = px.imshow(conf_matrix_single_xgb, text_auto=True, aspect="auto",
                                                               labels=dict(x="Predykowana Klasa", y="Prawdziwa Klasa", color="Liczba Próbek"),
                                                               x=['Niecelny (0)', 'Celny (1)'], y=['Niecelny (0)', 'Celny (1)'],
                                                               color_continuous_scale=px.colors.sequential.Greens, # Zmieniono kolor dla odróżnienia
                                                               title=f"Macierz Pomyłek XGBoost (Zbiór Testowy {test_size_percent_xgb}%)")
                                            fig_cm_xgb.update_layout(coloraxis_showscale=False)
                                            st.plotly_chart(fig_cm_xgb, use_container_width=True)
                                            st.caption(f"TN={conf_matrix_single_xgb[0,0]}, FP={conf_matrix_single_xgb[0,1]}, FN={conf_matrix_single_xgb[1,0]}, TP={conf_matrix_single_xgb[1,1]}")
                                            st.caption(f"Pojedynczy podział: {len(X_train)} próbek treningowych ({train_size_percent_xgb}%), {len(X_test)} próbek testowych ({test_size_percent_xgb}%).")

                                    except ValueError as ve_xgb:
                                         if ("test_size=" in str(ve_xgb) or "train_size=" in str(ve_xgb)) and "samples" in str(ve_xgb):
                                             st.error(f"Błąd (XGBoost): Wybrany rozmiar zbioru testowego ({test_size_percent_xgb}%) jest prawdopodobnie zbyt mały dla liczby próbek ({len(y)} ogółem) lub liczby próbek w jednej z klas, aby wykonać podział stratyfikowany. Spróbuj wybrać inny procent.")
                                         elif "found unknown categories" in str(ve_xgb):
                                              st.error(f"Błąd (XGBoost): W zbiorze testowym znaleziono kategorie, które nie wystąpiły w zbiorze treningowym. Sprawdź spójność danych. Błąd: {ve_xgb}")
                                         else: st.error(f"Wystąpił błąd wartości podczas podziału danych lub oceny XGBoost: {ve_xgb}")
                                    except Exception as e_single_xgb:
                                        st.error(f"Wystąpił błąd podczas generowania szczegółowych metryk XGBoost na podziale testowym: {e_single_xgb}")

                        else: # len(pmdc) < min_samples_for_model
                            st.warning(f"Niewystarczająca ilość danych ({len(pmdc)} próbek po usunięciu NaN we wszystkich wymaganych cechach) dla gracza '{selected_player}' do zbudowania wiarygodnego modelu XGBoost. Wymagane minimum: {min_samples_for_model}.")
                else:
                    missing_cols_xgb = [col for col in all_features + [target_variable] if col not in player_model_data.columns]
                    st.warning(f"Brak wymaganych kolumn dla '{selected_player}' w danych: {', '.join(missing_cols_xgb)}. Potrzebne: {', '.join(all_features + [target_variable])}")
            else:
                st.warning(f"Brak danych dla gracza: '{selected_player}'.")
        else:
            st.info("Wybierz gracza z panelu bocznego, aby ocenić model XGBoost.")
        # === Koniec Zakładki 6: XGBoost ===
    # ==========================================================================


# Obsługa przypadku, gdy dane nie zostały wczytane poprawnie
else:
    st.error("Nie udało się wczytać lub przetworzyć danych. Sprawdź ścieżkę do pliku CSV i jego format.")
    # Wyświetl szczegóły błędu, jeśli zostały zapisane w stanie sesji
    if 'load_error_message' in st.session_state and st.session_state.load_error_message:
        st.error(f"Szczegóły błędu: {st.session_state.load_error_message}")
        # Można usunąć komunikat po wyświetleniu, aby nie pojawiał się ciągle
        # del st.session_state.load_error_message


# --- Sidebar Footer (z poprawką czasu) ---
st.sidebar.markdown("---")
st.sidebar.info("Rozszerzona Aplikacja Streamlit - Analiza NBA")
try:
    # Poprawka dla potencjalnych błędów strefy czasowej
    try:
        tz = pytz.timezone('Europe/Warsaw')
        # Użyj .now() z tz dla aktualnego czasu
        ts = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z')
    except pytz.exceptions.UnknownTimeZoneError:
        st.sidebar.warning("Nie można znaleźć strefy czasowej 'Europe/Warsaw'. Używam UTC.")
        ts = datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
except ImportError:
     st.sidebar.warning("Biblioteka 'pytz' nie jest zainstalowana. Używam czasu lokalnego bez strefy.")
     ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S (czas lokalny)')
except Exception as e:
     st.sidebar.warning(f"Problem z pobraniem czasu: {e}. Używam czasu lokalnego.")
     ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S (czas lokalny)')
st.sidebar.markdown(f"Czas serwera: {ts}")