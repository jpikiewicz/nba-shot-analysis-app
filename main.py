# nba_app_v5_with_shap_interp.py
# Revision History:
# v1: Base functionality (Player/Team analysis, basic plots)
# v2: Added KNN model evaluation
# v3: Added XGBoost model evaluation & model comparison tab
# v4: Revisions by AI/User (Key naming, SHAP plots, League Avg, Simple Actions, Shot Dist, Layout)
# v5: Added dynamic SHAP interpretation comment below SHAP plots in XGBoost tab

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import os
from datetime import datetime
import pytz
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import shap # Import SHAP
import matplotlib.pyplot as plt # Import Matplotlib for SHAP plots

warnings.filterwarnings('ignore')

# --- Konfiguracja Początkowa ---
st.set_page_config(
    layout="wide",
    page_title="Analiza Rzutów NBA 2023-24"
)

# === Inicjalizacja Session State dla aktywnego widoku ===
if 'active_view' not in st.session_state:
    st.session_state.active_view = "📊 Rankingi Skuteczności"

# --- Ścieżka do pliku CSV ---
# UWAGA: Dostosuj ścieżkę do swojego pliku!
CSV_FILE_PATH = 'nba_player_shooting_data_2023_24.csv'

# --- Funkcje Pomocnicze ---

# --- POCZĄTEK FUNKCJI POMOCNICZYCH ---
@st.cache_data
def load_shooting_data(file_path):
    """Wczytuje i wstępnie przetwarza dane o rzutach graczy NBA."""
    load_status = {"success": False} # Domyślny status
    try:
        # Sprawdzenie czy plik istnieje
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Nie znaleziono pliku: {file_path}")

        data = pd.read_csv(file_path, parse_dates=['GAME_DATE'], low_memory=False) # Dodano low_memory=False
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
                data['SHOT_MADE_FLAG'] = data['SHOT_MADE_FLAG'].apply(lambda x: 1 if str(x).strip().lower() in [str(mv).lower() for mv in made_values] else (0 if pd.notna(x) else np.nan))
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
    # Linie ograniczające boisko (prostokąt) - UWAGA: y1 powinno być równe linii środkowej
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
        return None
    distance_data = player_data[required_cols].copy()
    distance_data['SHOT_MADE_FLAG'] = pd.to_numeric(distance_data['SHOT_MADE_FLAG'], errors='coerce')
    distance_data['SHOT_DISTANCE'] = pd.to_numeric(distance_data['SHOT_DISTANCE'], errors='coerce')
    distance_data = distance_data.dropna(subset=required_cols)
    if distance_data.empty:
        return None
    if not pd.api.types.is_numeric_dtype(distance_data['SHOT_DISTANCE']):
        return None
    max_dist = distance_data['SHOT_DISTANCE'].max()
    if pd.isna(max_dist) or max_dist <= 0:
        return None
    try:
        max_dist_rounded = np.ceil(max_dist)
        distance_bins = np.arange(0, max_dist_rounded + bin_width, bin_width)
        bin_labels_numeric = [(distance_bins[i] + distance_bins[i+1]) / 2 for i in range(len(distance_bins)-1)]
        if not bin_labels_numeric: return None
        distance_data['distance_bin_mid'] = pd.cut(distance_data['SHOT_DISTANCE'], bins=distance_bins, labels=bin_labels_numeric, right=False, include_lowest=True)
        # Konwertuj etykiety binów na numeryczne, jeśli są kategoryczne
        if pd.api.types.is_categorical_dtype(distance_data['distance_bin_mid']):
            distance_data['distance_bin_mid'] = pd.to_numeric(distance_data['distance_bin_mid'].astype(str), errors='coerce')

        effectiveness = distance_data.groupby('distance_bin_mid', observed=False)['SHOT_MADE_FLAG'].agg(Made='sum', Attempts='count').reset_index()
        effectiveness = effectiveness[effectiveness['Attempts'] >= min_attempts_per_bin]
        if effectiveness.empty: return None
        effectiveness['FG%'] = (effectiveness['Made'] / effectiveness['Attempts']) * 100
        effectiveness = effectiveness.sort_values(by='distance_bin_mid')
        if effectiveness.empty: return None
        fig = px.line(effectiveness, x='distance_bin_mid', y='FG%', title=f'Wpływ odległości rzutu na skuteczność - {player_name}',
                      labels={'distance_bin_mid': 'Środek przedziału odległości (stopy)', 'FG%': 'Skuteczność (%)'}, markers=True, hover_data=['Attempts', 'Made'])
        fig.update_layout(yaxis_range=[-5, 105], xaxis_title='Odległość rzutu (stopy)', yaxis_title='Skuteczność (%)', hovermode="x unified")
        fig.update_traces(connectgaps=False) # Nie łącz luk danych
        return fig
    except ValueError as e: return None
    except Exception as e_general: return None


@st.cache_data
def plot_shot_chart(entity_data, entity_name, entity_type="Gracz"):
    """Tworzy interaktywną mapę rzutów."""
    required_cols = ['LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']
    if not all(col in entity_data.columns for col in required_cols): return None
    # Dodajmy ACTION_TYPE_SIMPLE do hover data jeśli istnieje
    hover_base = ['SHOT_DISTANCE', 'SHOT_TYPE', 'ACTION_TYPE', 'SHOT_ZONE_BASIC', 'PERIOD']
    if 'ACTION_TYPE_SIMPLE' in entity_data.columns:
        hover_base.append('ACTION_TYPE_SIMPLE')

    plot_data = entity_data[required_cols + [col for col in hover_base if col in entity_data.columns]].copy()
    plot_data['LOC_X'] = pd.to_numeric(plot_data['LOC_X'], errors='coerce')
    plot_data['LOC_Y'] = pd.to_numeric(plot_data['LOC_Y'], errors='coerce')
    plot_data['SHOT_MADE_FLAG'] = pd.to_numeric(plot_data['SHOT_MADE_FLAG'], errors='coerce')
    plot_data = plot_data.dropna(subset=required_cols)
    if plot_data.empty: return None
    plot_data['Wynik Rzutu'] = plot_data['SHOT_MADE_FLAG'].map({0: 'Niecelny', 1: 'Celny'})
    color_col, color_map, cat_orders = 'Wynik Rzutu', {'Niecelny': 'red', 'Celny': 'green'}, {"Wynik Rzutu": ['Niecelny', 'Celny']}

    hover_cols_present = [col for col in hover_base if col in plot_data.columns] # Użyj hover_base
    hover_data_config = {col: True for col in hover_cols_present}

    fig = px.scatter(plot_data, x='LOC_X', y='LOC_Y', color=color_col, title=f'Mapa rzutów - {entity_name} ({entity_type})',
                     labels={'LOC_X': 'Pozycja X', 'LOC_Y': 'Pozycja Y', 'Wynik Rzutu': 'Wynik'},
                     hover_data=hover_data_config if hover_data_config else None,
                     category_orders=cat_orders, color_discrete_map=color_map, opacity=0.7)
    fig = add_court_shapes(fig)
    fig.update_layout(height=600, xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='rgba(255, 255, 255, 1)')
    return fig


@st.cache_data
def calculate_hot_zones(entity_data, min_shots_in_zone=5, n_bins=10):
    """Oblicza statystyki dla stref rzutowych."""
    required_cols = ['LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']
    if not all(col in entity_data.columns for col in required_cols): return pd.DataFrame()
    zone_data = entity_data[required_cols].copy()
    zone_data['LOC_X'] = pd.to_numeric(zone_data['LOC_X'], errors='coerce')
    zone_data['LOC_Y'] = pd.to_numeric(zone_data['LOC_Y'], errors='coerce')
    zone_data['SHOT_MADE_FLAG'] = pd.to_numeric(zone_data['SHOT_MADE_FLAG'], errors='coerce')
    zone_data = zone_data.dropna(subset=required_cols)
    if zone_data.empty: return pd.DataFrame()
    x_min, x_max = -250, 250
    y_min, y_max = -47.5, 422.5
    n_bins = max(2, n_bins) # Minimum 2 biny
    try:
        zone_data['zone_x'] = pd.cut(zone_data['LOC_X'], bins=np.linspace(x_min, x_max, n_bins + 1), include_lowest=True, right=True)
        zone_data['zone_y'] = pd.cut(zone_data['LOC_Y'], bins=np.linspace(y_min, y_max, n_bins + 1), include_lowest=True, right=True)
        zones = zone_data.groupby(['zone_x', 'zone_y'], observed=False).agg(
            total_shots=('SHOT_MADE_FLAG', 'count'),
            made_shots=('SHOT_MADE_FLAG', 'sum'),
            percentage_raw=('SHOT_MADE_FLAG', 'mean')
        ).reset_index()
        zones = zones[zones['total_shots'] >= min_shots_in_zone].copy()
        if zones.empty: return pd.DataFrame()
        zones['percentage'] = zones['percentage_raw'] * 100
        zones['x_center'] = zones['zone_x'].apply(lambda x: x.mid if isinstance(x, pd.Interval) else None)
        zones['y_center'] = zones['zone_y'].apply(lambda x: x.mid if isinstance(x, pd.Interval) else None)
        return zones.dropna(subset=['x_center', 'y_center'])
    except Exception as e: return pd.DataFrame()


@st.cache_data
def plot_hot_zones_heatmap(hot_zones_df, entity_name, entity_type="Gracz", min_shots_in_zone=5):
    """Tworzy interaktywną mapę ciepła stref rzutowych (skuteczność)."""
    required_cols = ['x_center', 'y_center', 'total_shots', 'percentage', 'made_shots']
    if hot_zones_df is None or hot_zones_df.empty or not all(col in hot_zones_df.columns for col in required_cols):
        return None
    plot_df = hot_zones_df[required_cols].copy()
    # Upewnij się, że kolumny są numeryczne przed operacjami
    for col in required_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    plot_df = plot_df.dropna() # Usuń wiersze z NaN po konwersji
    if plot_df.empty: return None

    min_pct, max_pct = plot_df['percentage'].min(), plot_df['percentage'].max()
    # Zakres kolorów - unikaj sytuacji min > max
    color_range = [max(0, min_pct - 5), min(100, max_pct + 5)] if pd.notna(min_pct) and pd.notna(max_pct) else [0, 100]
    if color_range[0] >= color_range[1]: color_range = [0, 100] # Fallback

    # Skalowanie rozmiaru bąbelków
    max_bubble_size = plot_df["total_shots"].max() if not plot_df["total_shots"].empty else 1
    size_ref = max(1, max_bubble_size / 50.0) # Eksperymentalnie dobrany współczynnik skalowania

    fig = px.scatter(plot_df, x='x_center', y='y_center', size='total_shots', color='percentage',
                     color_continuous_scale=px.colors.diverging.RdYlGn, # Czerwony-Żółty-Zielony
                     size_max=60, range_color=color_range,
                     title=f'Skuteczność stref rzutowych ({entity_type}: {entity_name}, min. {min_shots_in_zone} rzutów)',
                     labels={'x_center': 'Pozycja X', 'y_center': 'Pozycja Y', 'total_shots': 'Liczba rzutów', 'percentage': 'Skuteczność (%)'},
                     custom_data=['made_shots', 'total_shots']) # Dodaj dane do hovera

    # Ulepszony hovertemplate
    fig.update_traces(
        hovertemplate="<b>Strefa X:</b> %{x:.1f}, <b>Y:</b> %{y:.1f}<br>" +
                      "<b>Liczba rzutów:</b> %{customdata[1]}<br>" +
                      "<b>Trafione:</b> %{customdata[0]}<br>" +
                      "<b>Skuteczność:</b> %{marker.color:.1f}%<extra></extra>",
        marker=dict(sizeref=size_ref, sizemin=4) # Ustaw skalowanie i min. rozmiar
    )

    fig = add_court_shapes(fig)
    fig.update_layout(height=600, xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='rgba(255, 255, 255, 1)')
    return fig


@st.cache_data
def plot_shot_frequency_heatmap(data, season_name, nbins_x=50, nbins_y=50):
    """Tworzy heatmapę częstotliwości rzutów na boisku (Histogram2d)."""
    required_cols = ['LOC_X', 'LOC_Y']
    if not all(col in data.columns for col in required_cols): return None
    plot_data = data[required_cols].copy()
    plot_data['LOC_X'] = pd.to_numeric(plot_data['LOC_X'], errors='coerce')
    plot_data['LOC_Y'] = pd.to_numeric(plot_data['LOC_Y'], errors='coerce')
    plot_data = plot_data.dropna(subset=required_cols)
    if plot_data.empty: return None

    fig = go.Figure()
    fig.add_trace(go.Histogram2d(
        x = plot_data['LOC_X'],
        y = plot_data['LOC_Y'],
        colorscale = 'YlOrRd', # Skala kolorów od żółtego do czerwonego
        nbinsx = nbins_x,
        nbinsy = nbins_y,
        zauto = True, # Automatyczne skalowanie osi Z (kolor)
        hovertemplate = '<b>Zakres X:</b> %{x}<br><b>Zakres Y:</b> %{y}<br><b>Liczba rzutów:</b> %{z}<extra></extra>',
        colorbar=dict(title='Liczba rzutów')
    ))

    fig = add_court_shapes(fig) # Dodaj linie boiska

    fig.update_layout(
        title=f'Mapa Częstotliwości Rzutów ({season_name})',
        xaxis_title="Pozycja X", yaxis_title="Pozycja Y",
        height=650,
        xaxis_showgrid=False, yaxis_showgrid=False,
        plot_bgcolor='rgba(255, 255, 255, 1)' # Białe tło
    )
    return fig


@st.cache_data
def plot_player_quarter_eff(entity_data, entity_name, entity_type="Gracz", min_attempts=5):
    """Wykres skuteczności w poszczególnych kwartach/dogrywkach."""
    if 'PERIOD' not in entity_data.columns or 'SHOT_MADE_FLAG' not in entity_data.columns: return None
    quarter_data = entity_data[['PERIOD', 'SHOT_MADE_FLAG']].copy()
    quarter_data['PERIOD'] = pd.to_numeric(quarter_data['PERIOD'], errors='coerce')
    quarter_data['SHOT_MADE_FLAG'] = pd.to_numeric(quarter_data['SHOT_MADE_FLAG'], errors='coerce')
    quarter_data = quarter_data.dropna()
    if quarter_data.empty: return None
    quarter_data['PERIOD'] = quarter_data['PERIOD'].astype(int) # Konwersja na int po usunięciu NaN

    quarter_eff = quarter_data.groupby('PERIOD')['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
    quarter_eff['mean'] *= 100 # Konwersja na procenty
    quarter_eff = quarter_eff[quarter_eff['count'] >= min_attempts] # Filtrowanie wg minimalnej liczby prób
    if quarter_eff.empty: return None

    # Mapowanie numeru kwarty na czytelną etykietę
    def map_period(p):
        if p <= 4: return f"Kwarta {int(p)}"
        elif p == 5: return "OT 1"
        else: return f"OT {int(p-4)}"
    quarter_eff['Okres Gry'] = quarter_eff['PERIOD'].apply(map_period)

    # Sortowanie wg numeru kwarty/dogrywki
    quarter_eff = quarter_eff.sort_values(by='PERIOD')

    fig = px.bar(quarter_eff, x='Okres Gry', y='mean', text='mean',
                 title=f'Skuteczność w kwartach/dogrywkach - {entity_name} ({entity_type}, min. {min_attempts} prób)',
                 labels={'Okres Gry': 'Okres Gry', 'mean': 'Skuteczność (%)'},
                 hover_data=['count']) # Pokaż liczbę prób w tooltipie

    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside') # Formatowanie tekstu na słupkach
    fig.update_layout(yaxis_range=[0, 105], uniformtext_minsize=8, uniformtext_mode='hide') # Poprawki layoutu
    return fig


@st.cache_data
def plot_player_season_trend(entity_data, entity_name, entity_type="Gracz", min_monthly_attempts=10):
    """Wykres trendu skuteczności w trakcie sezonu (miesięcznie)."""
    if 'GAME_DATE' not in entity_data.columns or 'SHOT_MADE_FLAG' not in entity_data.columns: return None
    trend_data = entity_data[['GAME_DATE', 'SHOT_MADE_FLAG']].copy()
    trend_data['GAME_DATE'] = pd.to_datetime(trend_data['GAME_DATE'], errors='coerce')
    trend_data['SHOT_MADE_FLAG'] = pd.to_numeric(trend_data['SHOT_MADE_FLAG'], errors='coerce')
    trend_data = trend_data.dropna()
    if trend_data.empty or len(trend_data) < min_monthly_attempts: return None # Sprawdzenie ogólnej liczby danych

    trend_data = trend_data.set_index('GAME_DATE')
    # Resampling do miesięcznej częstotliwości ('ME' - Month End)
    monthly_eff = trend_data.resample('ME')['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
    monthly_eff['mean'] *= 100 # Skuteczność w procentach
    monthly_eff = monthly_eff[monthly_eff['count'] >= min_monthly_attempts] # Filtr min. prób miesięcznie
    if monthly_eff.empty or len(monthly_eff) < 2: return None # Potrzebujemy min. 2 punktów do linii

    # Formatowanie daty na 'YYYY-MM' dla osi X
    monthly_eff['Miesiąc'] = monthly_eff['GAME_DATE'].dt.strftime('%Y-%m')

    fig = px.line(monthly_eff, x='Miesiąc', y='mean', markers=True,
                  title=f'Miesięczny trend skuteczności - {entity_name} ({entity_type}, min. {min_monthly_attempts} prób/miesiąc)',
                  labels={'Miesiąc': 'Miesiąc', 'mean': 'Skuteczność (%)'},
                  hover_data=['count']) # Pokaż liczbę prób w tooltipie

    # Dynamiczne dostosowanie zakresu osi Y dla lepszej czytelności
    max_y = 105
    min_y = -5
    if not pd.isna(monthly_eff['mean'].max()): max_y = min(105, monthly_eff['mean'].max() + 10)
    if not pd.isna(monthly_eff['mean'].min()): min_y = max(-5, monthly_eff['mean'].min() - 10)
    fig.update_layout(yaxis_range=[min_y, max_y])
    return fig


@st.cache_data
def plot_grouped_effectiveness(entity_data, group_col, entity_name, entity_type="Gracz", top_n=10, min_attempts=5):
    """Tworzy wykres skuteczności pogrupowany wg wybranej kolumny."""
    if group_col not in entity_data.columns or 'SHOT_MADE_FLAG' not in entity_data.columns: return None
    grouped_data = entity_data[[group_col, 'SHOT_MADE_FLAG']].copy()
    grouped_data[group_col] = grouped_data[group_col].astype(str).str.strip() # Upewnij się, że to string i usuń białe znaki
    grouped_data['SHOT_MADE_FLAG'] = pd.to_numeric(grouped_data['SHOT_MADE_FLAG'], errors='coerce')
    grouped_data = grouped_data.dropna(subset=[group_col, 'SHOT_MADE_FLAG']) # Drop NaN also in group_col
    if grouped_data.empty: return None

    grouped_eff = grouped_data.groupby(group_col)['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
    grouped_eff['mean'] *= 100 # Skuteczność w procentach
    grouped_eff = grouped_eff[grouped_eff['count'] >= min_attempts] # Filtr min. prób

    # Sortuj wg liczby prób malejąco, aby wybrać top N najczęstszych, a potem sortuj wg kategorii dla spójności
    grouped_eff = grouped_eff.sort_values(by='count', ascending=False).head(top_n)

    # Specjalne sortowanie dla SHOT_ZONE_BASIC
    if group_col == 'SHOT_ZONE_BASIC':
        zone_order_basic = ['Restricted Area', 'In The Paint (Non-RA)', 'Mid-Range', 'Left Corner 3', 'Right Corner 3', 'Above the Break 3', 'Backcourt']
        # Użyj pd.Categorical do sortowania wg zdefiniowanej kolejności
        present_zones = [z for z in zone_order_basic if z in grouped_eff[group_col].unique()]
        grouped_eff[group_col] = pd.Categorical(
            grouped_eff[group_col],
            categories=present_zones,
            ordered=True
        )
        grouped_eff = grouped_eff.sort_values(by=group_col) # Sortuj wg kategorii
    elif group_col == 'ACTION_TYPE_SIMPLE':
         # Można zdefiniować logiczną kolejność dla uproszczonych typów
         action_order_simple = ['Dunk', 'Layup', 'Tip Shot', 'Hook Shot', 'Floater', 'Driving Shot', 'Jump Shot', 'Bank Shot', 'Alley Oop', 'Other']
         present_actions = [a for a in action_order_simple if a in grouped_eff[group_col].unique()]
         # Dodaj pozostałe nieprzewidziane typy posortowane alfabetycznie
         other_actions = sorted([a for a in grouped_eff[group_col].unique() if a not in present_actions])
         final_action_order = present_actions + other_actions
         grouped_eff[group_col] = pd.Categorical(
             grouped_eff[group_col],
             categories=final_action_order,
             ordered=True
         )
         grouped_eff = grouped_eff.sort_values(by=group_col)
    else:
        # Inne kategorie sortuj alfabetycznie dla spójności
        grouped_eff = grouped_eff.sort_values(by=group_col, ascending=True)

    if grouped_eff.empty: return None

    # Tworzenie tytułu i etykiet osi
    axis_label = group_col.replace('_',' ').title()
    chart_title = f'Skuteczność wg {axis_label} - {entity_name} ({entity_type})'
    if top_n < grouped_eff[group_col].nunique(): # Dostosuj tytuł jeśli pokazano tylko Top N
         chart_title += f' (Top {top_n} najczęstszych, min. {min_attempts} prób)'
    else:
         chart_title += f' (min. {min_attempts} prób)'


    fig = px.bar(grouped_eff, x=group_col, y='mean', text='mean',
                 title=chart_title,
                 labels={group_col: axis_label, 'mean': 'Skuteczność (%)'},
                 hover_data=['count']) # Pokaż liczbę prób w tooltipie

    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(yaxis_range=[0, 105], uniformtext_minsize=8, uniformtext_mode='hide', xaxis_title=axis_label)

    # Upewnij się, że kolejność na osi X odpowiada sortowaniu
    if group_col in ['SHOT_ZONE_BASIC', 'ACTION_TYPE_SIMPLE']:
         category_order = grouped_eff[group_col].cat.categories.tolist() # Pobierz kolejność z Categorical
         fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray': category_order})

    return fig


@st.cache_data
def plot_comparison_eff_distance(compare_data, selected_players, bin_width=3, min_attempts_per_bin=5):
    """Porównuje skuteczność graczy względem odległości (linowy)."""
    required_cols = ['SHOT_DISTANCE', 'SHOT_MADE_FLAG', 'PLAYER_NAME']
    if not all(col in compare_data.columns for col in required_cols): return None
    compare_data_eff = compare_data[required_cols].copy()
    compare_data_eff['SHOT_MADE_FLAG'] = pd.to_numeric(compare_data_eff['SHOT_MADE_FLAG'], errors='coerce')
    compare_data_eff['SHOT_DISTANCE'] = pd.to_numeric(compare_data_eff['SHOT_DISTANCE'], errors='coerce')
    compare_data_eff = compare_data_eff.dropna(subset=required_cols)
    if compare_data_eff.empty: return None
    if not pd.api.types.is_numeric_dtype(compare_data_eff['SHOT_DISTANCE']): return None

    max_dist = compare_data_eff['SHOT_DISTANCE'].max()
    if pd.isna(max_dist) or max_dist <= 0: return None

    try:
        max_dist_rounded = np.ceil(max_dist)
        distance_bins = np.arange(0, max_dist_rounded + bin_width, bin_width)
        bin_labels_numeric = [(distance_bins[i] + distance_bins[i+1]) / 2 for i in range(len(distance_bins)-1)]
        if not bin_labels_numeric: return None

        compare_data_eff['distance_bin_mid'] = pd.cut(compare_data_eff['SHOT_DISTANCE'], bins=distance_bins, labels=bin_labels_numeric, right=False, include_lowest=True)
        if pd.api.types.is_categorical_dtype(compare_data_eff['distance_bin_mid']):
            compare_data_eff['distance_bin_mid'] = pd.to_numeric(compare_data_eff['distance_bin_mid'].astype(str), errors='coerce')

        effectiveness = compare_data_eff.groupby(['PLAYER_NAME', 'distance_bin_mid'], observed=False)['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
        effectiveness['mean'] *= 100 # Procenty
        effectiveness = effectiveness[effectiveness['count'] >= min_attempts_per_bin] # Filtr min. prób
        effectiveness = effectiveness.dropna(subset=['distance_bin_mid']) # Usuń NaN w binach
        if effectiveness.empty: return None

        effectiveness = effectiveness.sort_values(by=['PLAYER_NAME', 'distance_bin_mid']) # Sortuj

        # Dynamiczny zakres osi Y
        max_eff_val = effectiveness['mean'].max()
        yaxis_range = [0, min(105, max_eff_val + 10 if not pd.isna(max_eff_val) else 105)]

        fig = px.line(effectiveness, x='distance_bin_mid', y='mean', color='PLAYER_NAME',
                      title=f'Porównanie skuteczności vs odległości (min. {min_attempts_per_bin} prób w przedziale {bin_width} stóp)',
                      labels={'distance_bin_mid': 'Odległość (stopy)', 'mean': 'Skuteczność (%)', 'PLAYER_NAME': 'Gracz'},
                      markers=True, hover_data=['count'])

        fig.update_layout(yaxis_range=yaxis_range, hovermode="x unified")
        return fig
    except Exception as e: return None


@st.cache_data
def plot_comparison_eff_by_zone(compare_data, selected_players, min_shots_per_zone=5):
    """Tworzy grupowany wykres słupkowy porównujący skuteczność graczy wg SHOT_ZONE_BASIC."""
    required_cols = ['PLAYER_NAME', 'SHOT_MADE_FLAG', 'SHOT_ZONE_BASIC']
    if not all(col in compare_data.columns for col in required_cols): return None
    zone_eff_data = compare_data[required_cols].copy()
    zone_eff_data['SHOT_MADE_FLAG'] = pd.to_numeric(zone_eff_data['SHOT_MADE_FLAG'], errors='coerce')
    zone_eff_data['SHOT_ZONE_BASIC'] = zone_eff_data['SHOT_ZONE_BASIC'].astype(str).str.strip()
    zone_eff_data = zone_eff_data.dropna(subset=required_cols)
    if zone_eff_data.empty: return None

    zone_stats = zone_eff_data.groupby(['PLAYER_NAME', 'SHOT_ZONE_BASIC'], observed=False)['SHOT_MADE_FLAG'].agg(Made='sum', Attempts='count').reset_index()
    zone_stats_filtered = zone_stats[zone_stats['Attempts'] >= min_shots_per_zone] # Filtr min. prób
    if zone_stats_filtered.empty: return None

    zone_stats_filtered['FG%'] = (zone_stats_filtered['Made'] / zone_stats_filtered['Attempts']) * 100 # Procenty

    # Definiowanie kolejności stref jak w analizie pojedynczego gracza
    zone_order_ideal = ['Restricted Area', 'In The Paint (Non-RA)', 'Mid-Range', 'Left Corner 3', 'Right Corner 3', 'Above the Break 3', 'Backcourt']
    actual_zones_in_data = zone_stats_filtered['SHOT_ZONE_BASIC'].unique()
    # Zachowaj tylko istniejące strefy w idealnej kolejności, dodaj resztę posortowaną
    zone_order = [zone for zone in zone_order_ideal if zone in actual_zones_in_data]
    zone_order += sorted([zone for zone in actual_zones_in_data if zone not in zone_order_ideal])
    if not zone_order: return None # Jeśli nie ma żadnych stref

    zone_stats_plot = zone_stats_filtered[zone_stats_filtered['SHOT_ZONE_BASIC'].isin(zone_order)].copy()
    if zone_stats_plot.empty: return None

    fig = px.bar(zone_stats_plot, x='SHOT_ZONE_BASIC', y='FG%', color='PLAYER_NAME', barmode='group',
                 title=f'Porównanie skuteczności (FG%) wg Strefy Rzutowej (min. {min_shots_per_zone} prób)',
                 labels={'SHOT_ZONE_BASIC': 'Strefa Rzutowa', 'FG%': 'Skuteczność (%)', 'PLAYER_NAME': 'Gracz'},
                 hover_data=['Attempts', 'Made'],
                 category_orders={'SHOT_ZONE_BASIC': zone_order}, # Użyj zdefiniowanej kolejności
                 text='FG%')

    fig.update_layout(yaxis_range=[0, 105], xaxis={'categoryorder':'array', 'categoryarray':zone_order}, legend_title_text='Gracze')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    return fig


@st.cache_data
def calculate_top_performers(data, group_by_col, min_total_shots, min_2pt_shots, min_3pt_shots, top_n=10):
    """Oblicza rankingi Top N graczy/zespołów wg skuteczności."""
    if group_by_col not in data.columns or 'SHOT_MADE_FLAG' not in data.columns: return None, None, None
    valid_data = data[[group_by_col, 'SHOT_MADE_FLAG', 'SHOT_TYPE'] if 'SHOT_TYPE' in data.columns else [group_by_col, 'SHOT_MADE_FLAG']].copy()
    valid_data['SHOT_MADE_FLAG'] = pd.to_numeric(valid_data['SHOT_MADE_FLAG'], errors='coerce')
    valid_data[group_by_col] = valid_data[group_by_col].astype(str)
    valid_data = valid_data.dropna(subset=[group_by_col, 'SHOT_MADE_FLAG'])
    if valid_data.empty: return None, None, None

    # Ranking Ogólny (FG%)
    overall_stats = valid_data.groupby(group_by_col)['SHOT_MADE_FLAG'].agg(Made='sum', Attempts='count').reset_index()
    overall_stats = overall_stats[overall_stats['Attempts'] >= min_total_shots]
    top_overall = pd.DataFrame()
    if not overall_stats.empty:
        overall_stats['FG%'] = (overall_stats['Made'] / overall_stats['Attempts']) * 100
        top_overall = overall_stats.sort_values(by=['FG%', 'Attempts'], ascending=[False, False]).head(top_n)
        col_name = group_by_col.replace('_',' ').title()
        top_overall = top_overall.rename(columns={group_by_col: col_name, 'Attempts': 'Próby'})
        top_overall = top_overall[[col_name, 'FG%', 'Próby']] # Zachowaj tylko potrzebne kolumny

    # Ranking 2PT%
    shot_type_2pt = '2PT Field Goal'
    top_2pt = pd.DataFrame()
    if 'SHOT_TYPE' in valid_data.columns:
        valid_data['SHOT_TYPE'] = valid_data['SHOT_TYPE'].astype(str).str.strip()
        if shot_type_2pt in valid_data['SHOT_TYPE'].unique():
            data_2pt = valid_data[valid_data['SHOT_TYPE'] == shot_type_2pt]
            if not data_2pt.empty:
                stats_2pt = data_2pt.groupby(group_by_col)['SHOT_MADE_FLAG'].agg(Made_2PT='sum', Attempts_2PT='count').reset_index()
                stats_2pt = stats_2pt[stats_2pt['Attempts_2PT'] >= min_2pt_shots]
                if not stats_2pt.empty:
                    stats_2pt['2PT FG%'] = (stats_2pt['Made_2PT'] / stats_2pt['Attempts_2PT']) * 100
                    top_2pt = stats_2pt.sort_values(by=['2PT FG%', 'Attempts_2PT'], ascending=[False, False]).head(top_n)
                    col_name = group_by_col.replace('_',' ').title()
                    top_2pt = top_2pt.rename(columns={group_by_col: col_name, 'Attempts_2PT': 'Próby 2PT'})
                    top_2pt = top_2pt[[col_name, '2PT FG%', 'Próby 2PT']]

    # Ranking 3PT%
    shot_type_3pt = '3PT Field Goal'
    top_3pt = pd.DataFrame()
    if 'SHOT_TYPE' in valid_data.columns:
        if shot_type_3pt in valid_data['SHOT_TYPE'].unique():
            data_3pt = valid_data[valid_data['SHOT_TYPE'] == shot_type_3pt]
            if not data_3pt.empty:
                stats_3pt = data_3pt.groupby(group_by_col)['SHOT_MADE_FLAG'].agg(Made_3PT='sum', Attempts_3PT='count').reset_index()
                stats_3pt = stats_3pt[stats_3pt['Attempts_3PT'] >= min_3pt_shots]
                if not stats_3pt.empty:
                    stats_3pt['3PT FG%'] = (stats_3pt['Made_3PT'] / stats_3pt['Attempts_3PT']) * 100
                    top_3pt = stats_3pt.sort_values(by=['3PT FG%', 'Attempts_3PT'], ascending=[False, False]).head(top_n)
                    col_name = group_by_col.replace('_',' ').title()
                    top_3pt = top_3pt.rename(columns={group_by_col: col_name, 'Attempts_3PT': 'Próby 3PT'})
                    top_3pt = top_3pt[[col_name, '3PT FG%', 'Próby 3PT']]

    return top_overall, top_2pt, top_3pt

def simplify_action_type(df):
    """
    Grupuje wartości w kolumnie ACTION_TYPE do uproszczonych kategorii.

    Argumenty:
        df (pd.DataFrame): DataFrame zawierający kolumnę 'ACTION_TYPE'.

    Zwraca:
        pd.DataFrame: DataFrame z dodaną nową kolumną 'ACTION_TYPE_SIMPLE'.
                      Zwraca oryginalny DataFrame z ostrzeżeniem, jeśli
                      kolumna 'ACTION_TYPE' nie istnieje.
    """
    if 'ACTION_TYPE' not in df.columns:
        # Można też użyć st.warning() jeśli funkcja będzie w głównym skrypcie Streamlit
        print("Ostrzeżenie: Kolumna 'ACTION_TYPE' nie znaleziona. Nie można uprościć.")
        return df

    # Upewnijmy się, że pracujemy na stringach i ignorujemy wielkość liter
    # Używamy .astype(str), aby uniknąć błędów przy potencjalnych NaN lub innych typach
    action_col = df['ACTION_TYPE'].astype(str).str.lower()

    # Definiujemy warunki i odpowiadające im kategorie
    # Kolejność ma znaczenie - od najbardziej specyficznych/ważnych
    conditions = [
        action_col.str.contains('dunk', na=False),                       # 1. Dunks
        action_col.str.contains('layup', na=False),                      # 2. Layups (po dunkach)
        # 3. Jump Shots (szeroka kategoria, łapie też pullup, step back, fadeaway)
        action_col.str.contains('jump shot|pullup|step back|fadeaway', na=False, regex=True),
        action_col.str.contains('hook shot', na=False),                  # 4. Hook Shots
        action_col.str.contains('tip|putback', na=False, regex=True),    # 5. Tip-ins / Putbacks
        action_col.str.contains('driving', na=False),                    # 6. Driving (inne niż layup/dunk/jump) - musi być po jump shot
        action_col.str.contains('floating|floater', na=False, regex=True),# 7. Floaters
        action_col.str.contains('alley oop', na=False),                  # 8. Alley Oops (jeśli nie są dunk/layup)
        action_col.str.contains('bank shot', na=False)                   # 9. Bank Shots (jeśli nie pasują wyżej)
        # Można dodać więcej warunków w razie potrzeby
    ]

    choices = [
        'Dunk',
        'Layup',
        'Jump Shot',
        'Hook Shot',
        'Tip Shot',
        'Driving Shot', # Kategoria dla "driving X shot" jeśli nie jest to layup/dunk/jump
        'Floater',
        'Alley Oop',
        'Bank Shot'
        # Odpowiadające kategorie
    ]

    # Używamy np.select do przypisania kategorii, 'Other' jako domyślna
    df['ACTION_TYPE_SIMPLE'] = np.select(conditions, choices, default='Other')

    # Opcjonalnie: Wyświetl podsumowanie nowej kolumny dla weryfikacji
    # print("Utworzono kolumnę 'ACTION_TYPE_SIMPLE'. Rozkład wartości:")
    # print(df['ACTION_TYPE_SIMPLE'].value_counts())

    return df

# --- KONIEC FUNKCJI POMOCNICZYCH ---


# --- Główna część aplikacji Streamlit ---
st.title("🏀 Interaktywna Analiza Rzutów Graczy NBA (Sezon 2023-24)")

# Wywołanie funkcji ładowania danych
shooting_data, load_status = load_shooting_data(CSV_FILE_PATH)

# --- Integracja Uproszczenia ACTION_TYPE ---
if load_status.get("success", False) and not shooting_data.empty:
    # Wywołaj funkcję upraszczającą TUTAJ, przed filtrowaniem sezonu
    shooting_data = simplify_action_type(shooting_data) # Dodaje kolumnę 'ACTION_TYPE_SIMPLE'
    # Komunikat sukcesu przeniesiony niżej, aby uwzględnić przetworzenie
# --- Koniec Integracji ---


# --- Sidebar ---
st.sidebar.header("Opcje Filtrowania i Analizy")

# Główna część aplikacji renderowana tylko jeśli dane są wczytane
if load_status.get("success", False) and not shooting_data.empty:

    # Komunikat o sukcesie po wczytaniu i potencjalnym przetworzeniu
    success_msg = f"Wczytano i przetworzono dane z {CSV_FILE_PATH}. Wymiary: {load_status.get('shape')}."
    if 'ACTION_TYPE_SIMPLE' in shooting_data.columns:
         success_msg += " Dodano uproszczone typy akcji ('ACTION_TYPE_SIMPLE')."
    st.success(success_msg)

    # Wyświetl ostrzeżenia z procesu ładowania
    if load_status.get("missing_time_cols"): st.warning("Brak kolumn czasowych. Nie można było utworzyć 'GAME_TIME_SEC' ani 'QUARTER_TYPE'.")
    if load_status.get("nan_in_time_cols"): st.warning("W kolumnach czasowych znaleziono wartości nienumeryczne (NaN) po próbie konwersji.")
    if load_status.get("nan_in_made_flag"): st.warning("W kolumnie SHOT_MADE_FLAG znaleziono wartości niejednoznaczne lub nienumeryczne (NaN) po próbie konwersji.")
    if load_status.get("missing_key_cols"): st.warning(f"Brakujące kluczowe kolumny do pełnej analizy: {', '.join(load_status['missing_key_cols'])}.")

    # Sidebar Filters
    # Filtr Sezonu
    available_season_types = ['Wszystko'] + shooting_data['SEASON_TYPE'].dropna().unique().tolist() if 'SEASON_TYPE' in shooting_data.columns else ['Wszystko']
    selected_season_type = st.sidebar.selectbox(
        "Wybierz typ sezonu:", options=available_season_types, index=0, key='sidebar_season_select'
    )
    # Filtruj dane GŁÓWNE na podstawie wybranego sezonu
    if selected_season_type != 'Wszystko' and 'SEASON_TYPE' in shooting_data.columns:
        # Filtrowanie odbywa się na danych już z ACTION_TYPE_SIMPLE
        filtered_data = shooting_data[shooting_data['SEASON_TYPE'] == selected_season_type].copy()
    else:
        filtered_data = shooting_data.copy() # Użyj wszystkich danych (z ACTION_TYPE_SIMPLE)
    st.sidebar.write(f"Wybrano: {selected_season_type} ({len(filtered_data)} rzutów)")

    # Player Selection (na podstawie przefiltrowanych danych sezonu)
    available_players = sorted(filtered_data['PLAYER_NAME'].dropna().unique()) if 'PLAYER_NAME' in filtered_data.columns else []
    default_player = "LeBron James"
    default_player_index = 0
    if available_players:
        try: default_player_index = available_players.index(default_player)
        except ValueError: default_player_index = 0 # Jeśli LeBrona nie ma, weź pierwszego
    selected_player = st.sidebar.selectbox(
        "Gracz do analizy/modelowania:", options=available_players, index=default_player_index,
        key='sidebar_player_select', disabled=not available_players,
        help="Wybierz gracza do szczegółowej analizy lub oceny modeli."
    )

    # Team Selection (na podstawie przefiltrowanych danych sezonu)
    available_teams = sorted(filtered_data['TEAM_NAME'].dropna().unique()) if 'TEAM_NAME' in filtered_data.columns else []
    default_team = "Los Angeles Lakers"
    default_team_index = 0
    if available_teams:
        try: default_team_index = available_teams.index(default_team)
        except ValueError: default_team_index = 0 # Jeśli Lakersów nie ma, weź pierwszy zespół
    selected_team = st.sidebar.selectbox(
        "Drużyna do analizy:", options=available_teams, index=default_team_index,
        key='sidebar_team_select', disabled=not available_teams,
        help="Wybierz drużynę do analizy zespołowej."
    )

    # Player Comparison Selection (na podstawie przefiltrowanych danych sezonu)
    default_compare_players_req = ["LeBron James", "Stephen Curry"] # Przykładowi gracze
    # Weź tylko tych, którzy są dostępni w przefiltrowanych danych
    default_compare_players_available = [p for p in default_compare_players_req if p in available_players]
    selected_players_compare = st.sidebar.multiselect(
        "Gracze do porównania (2-5):", options=available_players, default=default_compare_players_available,
        max_selections=5, key='sidebar_player_multi_select', disabled=not available_players,
        help="Wybierz od 2 do 5 graczy do porównania."
    )

    # === Zaktualizowana lista opcji zakładek ===
    tab_options = [
        "📊 Rankingi Skuteczności",
        "⛹️ Analiza Gracza",
        "🆚 Porównanie Graczy",
        "🏀 Analiza Zespołowa",
        "🎯 Ocena Modelu (KNN)",
        "🎯 Ocena Modelu (XGBoost)",
        "📊 Porównanie Modeli (KNN vs XGBoost)"
    ]

    # Upewnij się, że domyślny widok jest poprawny
    if st.session_state.active_view not in tab_options:
        st.session_state.active_view = tab_options[0]

    # Wybór widoku za pomocą st.radio
    current_index = tab_options.index(st.session_state.active_view)
    st.session_state.active_view = st.radio(
        "Wybierz widok:", options=tab_options, index=current_index,
        key='main_view_selector', horizontal=True, label_visibility="collapsed"
    )
    st.markdown("---") # Separator

    # === Poszczególne Widoki / Zakładki ===

    if st.session_state.active_view == "📊 Rankingi Skuteczności":
        # --- Prefiks dla kluczy widgetów w tym widoku: rank_ ---
        st.header(f"Analiza Ogólna Sezonu: {selected_season_type}")

        # --- Obliczanie i Wyświetlanie Średnich Ligowych ---
        st.markdown("---")
        st.subheader("Średnie Skuteczności w Lidze")
        league_avg_2pt = "N/A"; league_avg_3pt = "N/A"
        attempts_2pt_league = 0; attempts_3pt_league = 0
        made_2pt_league = 0; made_3pt_league = 0
        if 'SHOT_TYPE' in filtered_data.columns and 'SHOT_MADE_FLAG' in filtered_data.columns:
            calc_data = filtered_data[['SHOT_TYPE', 'SHOT_MADE_FLAG']].copy()
            calc_data['SHOT_MADE_FLAG'] = pd.to_numeric(calc_data['SHOT_MADE_FLAG'], errors='coerce')
            calc_data.dropna(subset=['SHOT_MADE_FLAG'], inplace=True)
            if not calc_data.empty:
                calc_data['SHOT_MADE_FLAG'] = calc_data['SHOT_MADE_FLAG'].astype(int)
                # Obliczenia dla 2PT
                data_2pt = calc_data[calc_data['SHOT_TYPE'] == '2PT Field Goal']
                attempts_2pt_league = len(data_2pt)
                if attempts_2pt_league > 0:
                    made_2pt_league = data_2pt['SHOT_MADE_FLAG'].sum()
                    league_avg_2pt = (made_2pt_league / attempts_2pt_league) * 100
                # Obliczenia dla 3PT
                data_3pt = calc_data[calc_data['SHOT_TYPE'] == '3PT Field Goal']
                attempts_3pt_league = len(data_3pt)
                if attempts_3pt_league > 0:
                    made_3pt_league = data_3pt['SHOT_MADE_FLAG'].sum()
                    league_avg_3pt = (made_3pt_league / attempts_3pt_league) * 100
            else: st.caption("Brak ważnych danych do obliczenia średnich ligowych.")
        # Wyświetlanie metryk
        col_avg1, col_avg2 = st.columns(2)
        with col_avg1:
            st.metric(label=f"Średnia Skuteczność 2PT (Liga)",
                      value=f"{league_avg_2pt:.1f}%" if isinstance(league_avg_2pt, (float, int)) else "Brak danych",
                      help=f"Obliczono na podstawie {attempts_2pt_league:,} rzutów ({made_2pt_league:,} trafionych) w {selected_season_type}.".replace(',', ' '))
        with col_avg2:
            st.metric(label=f"Średnia Skuteczność 3PT (Liga)",
                      value=f"{league_avg_3pt:.1f}%" if isinstance(league_avg_3pt, (float, int)) else "Brak danych",
                      help=f"Obliczono na podstawie {attempts_3pt_league:,} rzutów ({made_3pt_league:,} trafionych) w {selected_season_type}.".replace(',', ' '))
        st.markdown("---")
        # --- Koniec Sekcji Średnich Ligowych ---


        st.subheader("Ogólne Rozkłady Rzutów")
        st.caption(f"Dane dla: {selected_season_type}")

        # --- Definicja Wyboru Typu Akcji PRZED Kolumnami ---
        action_type_col_rank = 'ACTION_TYPE' # Domyślnie oryginalne
        action_choice = 'Oryginalne'
        if 'ACTION_TYPE_SIMPLE' in filtered_data.columns:
            action_choice_options = ('Oryginalne', 'Uproszczone')
            default_action_index = 1 # Domyślnie uproszczone, jeśli dostępne
            action_choice = st.radio(
                "Pokaż typy akcji:",
                action_choice_options,
                key='rank_action_type_choice', horizontal=True, index=default_action_index
            )
            if action_choice == 'Uproszczone':
                action_type_col_rank = 'ACTION_TYPE_SIMPLE'
        # --- Koniec Definicji Wyboru ---


        # --- Layout: Dwa Wiersze po Dwie Kolumny ---
        # --- Wiersz 1 ---
        c1_top, c2_top = st.columns(2)

        with c1_top: # Lewa Górna: Rozkład Typów Rzutów (Pie)
            st.markdown("###### Rozkład Typów Rzutów")
            if 'SHOT_TYPE' in filtered_data.columns and not filtered_data['SHOT_TYPE'].isnull().all():
                shot_type_counts = filtered_data['SHOT_TYPE'].dropna().value_counts().reset_index()
                if not shot_type_counts.empty:
                    fig_type = px.pie(shot_type_counts, names='SHOT_TYPE', values='count', hole=0.3)
                    fig_type.update_layout(legend_title_text='Typ Rzutu', height=350, margin=dict(t=20, b=0, l=0, r=0))
                    st.plotly_chart(fig_type, use_container_width=True)
                else: st.caption("Brak danych dla typów rzutów.")
            else: st.caption("Brak kolumny 'SHOT_TYPE'.")

        with c2_top: # Prawa Górna: Najczęstsze Typy Akcji (Bar)
            st.markdown(f"###### Najczęstsze Typy Akcji ({action_choice})")
            if action_type_col_rank in filtered_data.columns and not filtered_data[action_type_col_rank].isnull().all():
                action_type_counts = filtered_data[action_type_col_rank].dropna().value_counts().head(15).reset_index()
                if not action_type_counts.empty:
                    fig_action_freq = px.bar(action_type_counts, y=action_type_col_rank, x='count', orientation='h',
                                             labels={'count':'Liczba Rzutów', action_type_col_rank:''}, text='count')
                    fig_action_freq.update_layout(yaxis={'categoryorder':'total ascending'}, height=350, margin=dict(t=20, b=0, l=0, r=0))
                    fig_action_freq.update_traces(texttemplate='%{text:,}'.replace(',', ' '), textposition='outside')
                    st.plotly_chart(fig_action_freq, use_container_width=True)
                else: st.caption(f"Brak danych dla typów akcji ('{action_type_col_rank}').")
            else: st.caption(f"Brak kolumny '{action_type_col_rank}'.")

        st.markdown("<br>", unsafe_allow_html=True) # Mały odstęp

        # --- Wiersz 2 ---
        c1_bottom, c2_bottom = st.columns(2)

        with c1_bottom: # Lewa Dolna: Rozkład Rzutów wg Strefy (Bar)
            st.markdown("###### Rozkład Rzutów wg Strefy")
            if 'SHOT_ZONE_BASIC' in filtered_data.columns and not filtered_data['SHOT_ZONE_BASIC'].isnull().all():
                zone_basic_counts = filtered_data['SHOT_ZONE_BASIC'].dropna().value_counts().reset_index()
                if not zone_basic_counts.empty:
                    zone_order_basic = ['Restricted Area', 'In The Paint (Non-RA)', 'Mid-Range', 'Left Corner 3', 'Right Corner 3', 'Above the Break 3', 'Backcourt']
                    # Sortowanie wg zdefiniowanej kolejności
                    zone_basic_counts['SHOT_ZONE_BASIC'] = pd.Categorical(
                        zone_basic_counts['SHOT_ZONE_BASIC'],
                        categories=[z for z in zone_order_basic if z in zone_basic_counts['SHOT_ZONE_BASIC'].unique()],
                        ordered=True
                    )
                    zone_basic_counts = zone_basic_counts.sort_values('SHOT_ZONE_BASIC')

                    fig_zone_basic_dist = px.bar(
                        zone_basic_counts, x='SHOT_ZONE_BASIC', y='count',
                        labels={'SHOT_ZONE_BASIC': 'Strefa Rzutowa (Podstawowa)', 'count': 'Liczba Rzutów'},
                        text='count'
                    )
                    # Poprawka widoczności etykiety: auto + zwiększenie zakresu Y
                    fig_zone_basic_dist.update_traces(texttemplate='%{text:,}'.replace(',', ' '), textposition='auto')
                    fig_zone_basic_dist.update_layout(
                        xaxis_title=None, yaxis_title="Liczba Rzutów",
                        height=400, margin=dict(t=0, b=0, l=0, r=0)
                    )
                    # Dynamiczne ustawienie zakresu osi Y dla lepszej widoczności etykiet
                    max_y_val = zone_basic_counts['count'].max()
                    if pd.notna(max_y_val):
                        fig_zone_basic_dist.update_layout(yaxis_range=[0, max_y_val * 1.1]) # Margines 10%

                    st.plotly_chart(fig_zone_basic_dist, use_container_width=True)
                else: st.caption("Brak danych dla rozkładu stref podstawowych.")
            else: st.caption("Brak kolumny 'SHOT_ZONE_BASIC'.")


        with c2_bottom: # Prawa Dolna: Najefektywniejsze Typy Akcji (Bar)
            st.markdown(f"###### Najefektywniejsze Typy Akcji ({action_choice})")
            if action_type_col_rank in filtered_data.columns and 'SHOT_MADE_FLAG' in filtered_data.columns:
                min_attempts_eff_action_b = st.number_input(
                    f"Min. prób dla rankingu skuteczności akcji ({action_choice}):", min_value=5, value=10, step=1,
                    key=f'rank_min_attempts_eff_action_{action_type_col_rank}_bottom', # Unikalny klucz
                    help="Minimalna liczba rzutów danego typu akcji w rankingu."
                )
                action_eff_data = filtered_data[[action_type_col_rank, 'SHOT_MADE_FLAG']].copy()
                action_eff_data[action_type_col_rank] = action_eff_data[action_type_col_rank].astype(str).str.strip()
                action_eff_data['SHOT_MADE_FLAG'] = pd.to_numeric(action_eff_data['SHOT_MADE_FLAG'], errors='coerce')
                action_eff_data = action_eff_data.dropna()
                if not action_eff_data.empty:
                    action_stats = action_eff_data.groupby(action_type_col_rank)['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
                    action_stats_filtered = action_stats[action_stats['count'] >= min_attempts_eff_action_b].copy()
                    if not action_stats_filtered.empty:
                        action_stats_filtered['FG%'] = action_stats_filtered['mean'] * 100
                        action_stats_filtered = action_stats_filtered.sort_values(by='FG%', ascending=False).head(15)
                        fig_action_eff = px.bar(
                            action_stats_filtered, y=action_type_col_rank, x='FG%', orientation='h',
                            labels={'FG%': 'Skuteczność (%)', action_type_col_rank: ''}, text='FG%', hover_data=['count']
                        )
                        fig_action_eff.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_range=[0, 105], height=400, margin=dict(t=0, b=0, l=0, r=0))
                        fig_action_eff.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        st.plotly_chart(fig_action_eff, use_container_width=True)
                    else: st.caption(f"Brak typów akcji ('{action_type_col_rank}') z min. {min_attempts_eff_action_b} prób.")
                else: st.caption("Brak danych do obliczenia skuteczności akcji.")
            else: st.caption(f"Brak kolumn '{action_type_col_rank}' lub 'SHOT_MADE_FLAG'.")
        # --- Koniec Zmiany Layoutu ---


        # --- Reszta zakładki Rankingi Skuteczności ---
        st.markdown("---")
        st.subheader(f"Mapa Częstotliwości Rzutów")
        st.caption(f"Gęstość rzutów dla: {selected_season_type}")
        num_bins_freq = st.slider("Dokładność mapy (X):", 20, 80, 50, 5, key='rank_frequency_map_bins',
                                  help="Liczba 'koszyków', na które dzielona jest oś X boiska do zliczania rzutów. Oś Y dostosuje się proporcjonalnie.")
        nbins_y_freq = int(num_bins_freq * (470 / 500)) # Skalowanie liczby binów Y proporcjonalnie do wymiarów boiska
        fig_freq_map = plot_shot_frequency_heatmap(filtered_data, selected_season_type, nbins_x=num_bins_freq, nbins_y=nbins_y_freq)
        if fig_freq_map: st.plotly_chart(fig_freq_map, use_container_width=True)
        else: st.caption("Nie udało się wygenerować mapy częstotliwości.")
        st.markdown("---")

        st.subheader(f"Rankingi Skuteczności Graczy i Zespołów")
        st.caption(f"Top 10 dla: {selected_season_type}")
        st.markdown("##### Min. liczba prób")
        col_att1, col_att2, col_att3 = st.columns(3)
        with col_att1: min_total = st.number_input("Ogółem:", 10, 1000, 100, 10, key="rank_min_total")
        with col_att2: min_2pt = st.number_input("Za 2 pkt:", 5, 500, 50, 5, key="rank_min_2pt")
        with col_att3: min_3pt = st.number_input("Za 3 pkt:", 5, 500, 30, 5, key="rank_min_3pt")

        # Oblicz rankingi dla graczy i zespołów
        tp_ov, tp_2, tp_3 = calculate_top_performers(filtered_data, 'PLAYER_NAME', min_total, min_2pt, min_3pt)
        tt_ov, tt_2, tt_3 = calculate_top_performers(filtered_data, 'TEAM_NAME', min_total*5, min_2pt*5, min_3pt*5) # Wyższe progi dla zespołów

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

    elif st.session_state.active_view == "⛹️ Analiza Gracza":
        # --- Prefiks dla kluczy widgetów w tym widoku: player_ ---
        st.header(f"Analiza Gracza: {selected_player}")
        if selected_player and 'PLAYER_NAME' in filtered_data.columns:
            player_data = filter_data_by_player(selected_player, filtered_data) # Filtruj dane dla wybranego gracza
            if not player_data.empty:
                st.subheader("Statystyki Podstawowe")
                # Obliczanie statystyk podstawowych dla gracza
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
                    shot_type_2pt = '2PT Field Goal'
                    data_2pt = player_data[player_data['SHOT_TYPE'] == shot_type_2pt]
                    made_flag_2pt = pd.to_numeric(data_2pt['SHOT_MADE_FLAG'], errors='coerce').dropna()
                    attempts_2pt = len(made_flag_2pt)
                    if attempts_2pt > 0: made_2pt = int(made_flag_2pt.sum()); pct_2pt = (made_2pt / attempts_2pt) * 100; attempts_2pt_str = f"({made_2pt}/{attempts_2pt})"
                    else: attempts_2pt_str = "(0 prób)"

                    shot_type_3pt = '3PT Field Goal'
                    data_3pt = player_data[player_data['SHOT_TYPE'] == shot_type_3pt]
                    made_flag_3pt = pd.to_numeric(data_3pt['SHOT_MADE_FLAG'], errors='coerce').dropna()
                    attempts_3pt = len(made_flag_3pt)
                    if attempts_3pt > 0: made_3pt = int(made_flag_3pt.sum()); pct_3pt = (made_3pt / attempts_3pt) * 100; attempts_3pt_str = f"({made_3pt}/{attempts_3pt})"
                    else: attempts_3pt_str = "(0 prób)"

                avg_dist = "N/A"
                if 'SHOT_DISTANCE' in player_data.columns and pd.api.types.is_numeric_dtype(player_data['SHOT_DISTANCE']):
                    valid_distances = player_data['SHOT_DISTANCE'].dropna()
                    if not valid_distances.empty: avg_dist = valid_distances.mean()

                # Wyświetlanie metryk
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

                st.subheader("Wizualizacje Rzutów")
                # Mapa rzutów gracza
                fig_p_chart = plot_shot_chart(player_data, selected_player, "Gracz")
                if fig_p_chart: st.plotly_chart(fig_p_chart, use_container_width=True)
                else: st.warning("Nie można wygenerować mapy rzutów dla tego gracza (brak danych?).")
                st.markdown("---")

                st.subheader("Skuteczność vs Odległość")
                # Wykres skuteczności vs odległość
                cc1, cc2 = st.columns([1,2])
                with cc1: bin_w = st.slider("Szerokość przedziału (stopy):", 1, 5, 1, key='player_eff_dist_bin')
                with cc2: min_att = st.slider("Min. prób w przedziale:", 1, 50, 5, key='player_eff_dist_min')
                fig_p_eff_dist = plot_player_eff_vs_distance(player_data, selected_player, bin_width=bin_w, min_attempts_per_bin=min_att)
                if fig_p_eff_dist: st.plotly_chart(fig_p_eff_dist, use_container_width=True)
                else: st.caption(f"Brak wystarczających danych dla wykresu skuteczności vs odległość (min. {min_att} prób w przedziale {bin_w} stóp).")
                st.markdown("---")

                st.subheader("Strefy Rzutowe ('Hot Zones')")
                # Mapa ciepła skuteczności w strefach
                hz_min_shots = st.slider("Min. rzutów w strefie:", 3, 50, 5, key='player_hotzone_min_shots')
                hz_bins = st.slider("Liczba stref na oś:", 5, 15, 10, key='player_hotzone_bins')
                p_hot_zones = calculate_hot_zones(player_data, min_shots_in_zone=hz_min_shots, n_bins=hz_bins)
                if p_hot_zones is not None and not p_hot_zones.empty:
                    fig_p_hot = plot_hot_zones_heatmap(p_hot_zones, selected_player, "Gracz", min_shots_in_zone=hz_min_shots)
                    if fig_p_hot: st.plotly_chart(fig_p_hot, use_container_width=True)
                    else: st.info("Nie można wygenerować mapy stref.")
                else: st.info(f"Brak wystarczających danych do analizy stref rzutowych (min. {hz_min_shots} prób na strefę).")
                st.markdown("---")

                st.subheader("Analiza Czasowa")
                # Wykres skuteczności w kwartach
                q_min_shots = st.slider("Min. prób w kwarcie/OT:", 3, 50, 5, key='player_quarter_min_shots')
                fig_p_q = plot_player_quarter_eff(player_data, selected_player, min_attempts=q_min_shots)
                if fig_p_q: st.plotly_chart(fig_p_q, use_container_width=True)
                else: st.info(f"Brak wystarczających danych do analizy skuteczności w kwartach (min. {q_min_shots} prób).")

                # Wykres trendu miesięcznego
                m_min_shots = st.slider("Min. prób w miesiącu:", 5, 100, 10, key='player_month_min_shots')
                fig_p_t = plot_player_season_trend(player_data, selected_player, min_monthly_attempts=m_min_shots)
                if fig_p_t: st.plotly_chart(fig_p_t, use_container_width=True)
                else: st.info(f"Brak wystarczających danych do analizy trendu miesięcznego (min. {m_min_shots} prób miesięcznie i/lub min. 2 miesiące).")
                st.markdown("---")

                st.subheader("Analiza wg Typu Akcji / Strefy")
                # Wykresy skuteczności wg grup
                g_min_shots = st.slider("Min. prób w grupie:", 3, 50, 5, key='player_group_min_shots')

                # Wybór typu akcji (oryginalny vs uproszczony) dla wykresu grupowego
                action_type_col_player = 'ACTION_TYPE'
                action_choice_player = 'Oryginalne'
                if 'ACTION_TYPE_SIMPLE' in player_data.columns:
                     action_choice_options_player = ('Oryginalne', 'Uproszczone')
                     default_action_index_player = 1 # Domyślnie uproszczone
                     action_choice_player = st.radio("Pokaż typy akcji dla skuteczności:", action_choice_options_player,
                                                     key='player_action_type_choice', horizontal=True, index=default_action_index_player)
                     if action_choice_player == 'Uproszczone':
                         action_type_col_player = 'ACTION_TYPE_SIMPLE'


                cg1, cg2 = st.columns(2)
                with cg1:
                    st.markdown(f"###### Skuteczność wg Typu Akcji ({action_choice_player})")
                    fig_p_a = plot_grouped_effectiveness(player_data, action_type_col_player, selected_player, "Gracz", top_n=10, min_attempts=g_min_shots)
                    if fig_p_a: st.plotly_chart(fig_p_a, use_container_width=True)
                    else: st.caption(f"Brak wystarczających danych wg typu akcji '{action_choice_player}' (min. {g_min_shots} prób).")
                with cg2:
                    st.markdown("###### Skuteczność wg Strefy Podstawowej")
                    fig_p_z = plot_grouped_effectiveness(player_data, 'SHOT_ZONE_BASIC', selected_player, "Gracz", top_n=7, min_attempts=g_min_shots)
                    if fig_p_z: st.plotly_chart(fig_p_z, use_container_width=True)
                    else: st.caption(f"Brak wystarczających danych wg strefy podstawowej (min. {g_min_shots} prób).")
            else:
                st.warning(f"Brak danych dla gracza '{selected_player}' w wybranym typie sezonu '{selected_season_type}'.")
        else:
            st.info("Wybierz gracza z panelu bocznego, aby zobaczyć jego analizę.")

    elif st.session_state.active_view == "🆚 Porównanie Graczy":
        # --- Prefiks dla kluczy widgetów w tym widoku: comp_ ---
        st.header("Porównanie Graczy")
        if len(selected_players_compare) >= 2:
            st.write(f"Porównujesz: {', '.join(selected_players_compare)}")
            # Filtruj dane tylko dla wybranych graczy DO PORÓWNANIA (bazując na danych już przefiltrowanych wg sezonu)
            compare_data_filtered = filtered_data[filtered_data['PLAYER_NAME'].isin(selected_players_compare)].copy()

            if not compare_data_filtered.empty:
                st.subheader("Skuteczność vs Odległość")
                # Porównanie sk. vs odl.
                comp_eff_dist_bin = st.slider("Szerokość przedziału (stopy):", 1, 5, 3, key='comp_eff_dist_bin')
                comp_eff_dist_min = st.slider("Min. prób w przedziale:", 3, 50, 5, key='comp_eff_dist_min')
                fig_comp_eff_dist = plot_comparison_eff_distance(compare_data_filtered, selected_players_compare, bin_width=comp_eff_dist_bin, min_attempts_per_bin=comp_eff_dist_min)
                if fig_comp_eff_dist: st.plotly_chart(fig_comp_eff_dist, use_container_width=True)
                else: st.caption(f"Brak wystarczających danych dla porównania skuteczności vs odległość (min. {comp_eff_dist_min} prób w przedziale {comp_eff_dist_bin} stóp).")
                st.markdown("---")

                st.subheader("Skuteczność wg Strefy Rzutowej")
                # Porównanie wg strefy
                min_attempts_zone = st.slider("Min. prób w strefie:", 3, 50, 5, key='comp_zone_min')
                fig_comp_zone = plot_comparison_eff_by_zone(compare_data_filtered, selected_players_compare, min_shots_per_zone=min_attempts_zone)
                if fig_comp_zone: st.plotly_chart(fig_comp_zone, use_container_width=True)
                else: st.caption(f"Brak wystarczających danych dla porównania wg stref (min. {min_attempts_zone} prób w strefie).")
                st.markdown("---")

                st.subheader("Mapy Rzutów")
                # Wyświetlanie map rzutów dla porównywanych graczy
                num_players_comp = len(selected_players_compare)
                cols = st.columns(num_players_comp) # Utwórz kolumny dla każdego gracza
                for i, player in enumerate(selected_players_compare):
                    with cols[i]:
                        st.markdown(f"**{player}**")
                        player_comp_data = compare_data_filtered[compare_data_filtered['PLAYER_NAME'] == player]
                        if not player_comp_data.empty:
                            chart_key = f"comp_chart_{player.replace(' ','_').replace('.','')}" # Unikalny klucz dla wykresu
                            fig_comp_chart = plot_shot_chart(player_comp_data, player, "Gracz")
                            if fig_comp_chart:
                                fig_comp_chart.update_layout(height=450, title="") # Zmniejsz wysokość i usuń tytuł
                                st.plotly_chart(fig_comp_chart, use_container_width=True, key=chart_key)
                            else: st.caption("Błąd generowania mapy.")
                        else: st.caption("Brak danych w wybranym sezonie.")
            else:
                st.warning(f"Brak danych dla wybranych graczy do porównania w sezonie '{selected_season_type}'.")
        else:
            st.info("Wybierz min. 2 graczy do porównania z panelu bocznego.")


    elif st.session_state.active_view == "🏀 Analiza Zespołowa":
        # --- Prefiks dla kluczy widgetów w tym widoku: team_ ---
        st.header(f"Analiza Zespołowa: {selected_team}")
        if selected_team and 'TEAM_NAME' in filtered_data.columns:
            team_data = filter_data_by_team(selected_team, filtered_data) # Filtruj dane dla wybranej drużyny
            if not team_data.empty:
                # Statystyki podstawowe zespołu
                st.subheader("Statystyki Podstawowe Zespołu")
                t_stats = get_basic_stats(team_data, selected_team, "Zespół") # Użyj funkcji pomocniczej
                c1, c2, c3 = st.columns(3)
                c1.metric("Rzuty Zespołu", t_stats.get('total_shots', 'N/A'))
                c2.metric("Trafione Zespołu", t_stats.get('made_shots', 'N/A'))
                pct_val = t_stats.get('shooting_pct')
                c3.metric("Skuteczność Zespołu", f"{pct_val:.1f}%" if isinstance(pct_val, (float, int)) else "N/A")
                st.markdown("---")

                # Wizualizacje dla zespołu (analogiczne do gracza)
                st.subheader("Wizualizacje Rzutów Zespołu")
                fig_t_c = plot_shot_chart(team_data, selected_team, "Zespół")
                if fig_t_c: st.plotly_chart(fig_t_c, use_container_width=True)
                else: st.warning("Nie można wygenerować mapy rzutów zespołu.")

                st.subheader("Skuteczność Zespołu vs Odległość")
                t_eff_dist_bin = st.slider("Szerokość przedziału (stopy):", 1, 5, 2, key='team_eff_dist_bin')
                t_eff_dist_min = st.slider("Min. prób w przedziale:", 5, 100, 10, key='team_eff_dist_min')
                fig_t_ed = plot_player_eff_vs_distance(team_data, selected_team, bin_width=t_eff_dist_bin, min_attempts_per_bin=t_eff_dist_min)
                if fig_t_ed: st.plotly_chart(fig_t_ed, use_container_width=True)
                else: st.caption(f"Brak wystarczających danych dla wykresu sk. vs odl. zespołu (min. {t_eff_dist_min} prób / {t_eff_dist_bin} stóp).")
                st.markdown("---")

                st.subheader("Strefy Rzutowe ('Hot Zones') Zespołu")
                t_hz_min_shots = st.slider("Min. rzutów w strefie:", 5, 100, 10, key='team_hotzone_min_shots')
                t_hz_bins = st.slider("Liczba stref na oś:", 5, 15, 10, key='team_hotzone_bins')
                t_hz = calculate_hot_zones(team_data, min_shots_in_zone=t_hz_min_shots, n_bins=t_hz_bins)
                if t_hz is not None and not t_hz.empty:
                    fig_t_h = plot_hot_zones_heatmap(t_hz, selected_team, "Zespół", min_shots_in_zone=t_hz_min_shots)
                    if fig_t_h: st.plotly_chart(fig_t_h, use_container_width=True)
                    else: st.info("Nie można wygenerować mapy stref zespołu.")
                else: st.info(f"Brak wystarczających danych do analizy stref zespołu (min. {t_hz_min_shots} prób/strefę).")
                st.markdown("---")

                st.subheader("Analiza Czasowa Zespołu")
                t_q_min_shots = st.slider("Min. prób w kwarcie/OT:", 5, 100, 10, key='team_quarter_min_shots')
                fig_t_q = plot_player_quarter_eff(team_data, selected_team, "Zespół", min_attempts=t_q_min_shots)
                if fig_t_q: st.plotly_chart(fig_t_q, use_container_width=True)
                else: st.info(f"Brak wystarczających danych do analizy kwart zespołu (min. {t_q_min_shots} prób).")
                st.markdown("---")

                st.subheader("Analiza Zespołu wg Typu Akcji / Strefy")
                t_g_min_shots = st.slider("Min. prób w grupie:", 5, 100, 10, key='team_group_min_shots')

                 # Wybór typu akcji (oryginalny vs uproszczony) dla wykresu grupowego zespołu
                action_type_col_team = 'ACTION_TYPE'
                action_choice_team = 'Oryginalne'
                if 'ACTION_TYPE_SIMPLE' in team_data.columns:
                     action_choice_options_team = ('Oryginalne', 'Uproszczone')
                     default_action_index_team = 1 # Domyślnie uproszczone
                     action_choice_team = st.radio("Pokaż typy akcji dla skuteczności zespołu:", action_choice_options_team,
                                                     key='team_action_type_choice', horizontal=True, index=default_action_index_team)
                     if action_choice_team == 'Uproszczone':
                         action_type_col_team = 'ACTION_TYPE_SIMPLE'

                cg1, cg2 = st.columns(2)
                with cg1:
                    st.markdown(f"###### Skuteczność Zespołu wg Typu Akcji ({action_choice_team})")
                    fig_t_a = plot_grouped_effectiveness(team_data, action_type_col_team, selected_team, "Zespół", top_n=10, min_attempts=t_g_min_shots)
                    if fig_t_a: st.plotly_chart(fig_t_a, use_container_width=True)
                    else: st.caption(f"Brak wystarczających danych zespołu wg typu akcji '{action_choice_team}' (min. {t_g_min_shots} prób).")
                with cg2:
                    st.markdown("###### Skuteczność Zespołu wg Strefy Podstawowej")
                    fig_t_z = plot_grouped_effectiveness(team_data, 'SHOT_ZONE_BASIC', selected_team, "Zespół", top_n=7, min_attempts=t_g_min_shots)
                    if fig_t_z: st.plotly_chart(fig_t_z, use_container_width=True)
                    else: st.caption(f"Brak wystarczających danych zespołu wg strefy podstawowej (min. {t_g_min_shots} prób).")
            else:
                st.warning(f"Brak danych dla drużyny '{selected_team}' w wybranym typie sezonu '{selected_season_type}'.")
        else:
            st.info("Wybierz drużynę z panelu bocznego, aby zobaczyć jej analizę.")


    elif st.session_state.active_view == "🎯 Ocena Modelu (KNN)":
        # --- Prefiks dla kluczy widgetów w tym widoku: knn_eval_ ---
        st.header(f"Ocena Modelu Predykcyjnego (KNN) dla: {selected_player}")

        st.markdown(f"""
        ### Interpretacja Wyników Rozszerzonego Modelu KNN ({selected_player})

        Zakładka ta prezentuje model K-Najbliższych Sąsiadów (KNN) do przewidywania wyniku rzutu (celny/niecelny), rozszerzony o dodatkowe cechy kategoryczne: `ACTION_TYPE`, `SHOT_TYPE` i `PERIOD`. Ocenia jego wydajność na dwa sposoby:

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
        """)

        if selected_player:
            player_model_data = filter_data_by_player(selected_player, filtered_data) # Użyj danych przefiltrowanych wg sezonu
            if not player_model_data.empty:
                # Definicja cech i zmiennej docelowej
                numerical_features = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE']
                # Użyj ACTION_TYPE_SIMPLE jeśli istnieje i jest wybrane, inaczej ACTION_TYPE
                action_type_col_model = 'ACTION_TYPE' # Domyślnie
                if 'ACTION_TYPE_SIMPLE' in player_model_data.columns:
                     # Można dodać radio button, aby użytkownik wybrał, której kolumny użyć
                     use_simple_action_knn = st.checkbox("Użyj uproszczonych typów akcji (ACTION_TYPE_SIMPLE) w modelu KNN?", value=True, key='knn_eval_use_simple_action')
                     if use_simple_action_knn:
                         action_type_col_model = 'ACTION_TYPE_SIMPLE'

                categorical_features = [action_type_col_model, 'SHOT_TYPE', 'PERIOD']
                target_variable = 'SHOT_MADE_FLAG'
                all_features = numerical_features + categorical_features

                # Sprawdzenie czy wszystkie potrzebne kolumny istnieją
                missing_cols = [col for col in all_features + [target_variable] if col not in player_model_data.columns]
                if not missing_cols:
                    # Przygotowanie danych: usuwanie NaN, konwersja typów
                    pmdc = player_model_data[all_features + [target_variable]].dropna().copy()
                    pmdc[target_variable] = pd.to_numeric(pmdc[target_variable], errors='coerce')
                    pmdc = pmdc.dropna(subset=[target_variable]) # Usuń NaN w target
                    # Konwertuj kolumny kategoryczne na string
                    for col in categorical_features: pmdc[col] = pmdc[col].astype(str)

                    # Sprawdzenie, czy mamy obie klasy (0 i 1)
                    if pmdc[target_variable].nunique() < 2 and not pmdc.empty:
                        st.warning(f"Dla gracza '{selected_player}' pozostała tylko jedna klasa wyniku rzutu po przetworzeniu danych. Nie można zbudować modelu klasyfikacyjnego.")
                    elif pmdc.empty:
                        st.warning(f"Brak kompletnych danych (po usunięciu NaN) dla gracza '{selected_player}' do zbudowania modelu.")
                    else:
                        pmdc[target_variable] = pmdc[target_variable].astype(int) # Konwersja target na int
                        min_samples_for_model = 50 # Minimalna liczba próbek do budowy modelu
                        if len(pmdc) >= min_samples_for_model:
                            st.subheader("Konfiguracja Modelu KNN i Oceny")
                            # Slidery do konfiguracji
                            k = st.slider("Liczba sąsiadów (k):", 3, min(25, len(pmdc)//3), 5, 2, key='knn_eval_k')
                            n_splits = st.slider("Liczba podziałów CV:", 3, 10, 5, 1, key='knn_eval_cv_splits')
                            st.markdown("---")
                            st.subheader("Konfiguracja Pojedynczego Podziału Testowego")
                            test_size_percent = st.slider("Rozmiar zbioru testowego (%):", 10, 50, 20, 5, key='knn_eval_test_split', format="%d%%")
                            train_size_percent = 100 - test_size_percent
                            test_size_float = test_size_percent / 100.0

                            if st.button(f"Uruchom Ocenę Modelu KNN dla {selected_player}", key='knn_eval_run_button'):
                                # Definicja preprocesora i pipeline'u
                                preprocessor = ColumnTransformer(
                                    transformers=[
                                        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_features),
                                        ('num', StandardScaler(), numerical_features)],
                                    remainder='passthrough' # Zachowaj inne kolumny (chociaż nie powinno ich być)
                                )
                                pipeline = Pipeline([
                                    ('preprocessor', preprocessor),
                                    ('knn', KNeighborsClassifier(n_neighbors=k))
                                ])

                                X = pmdc[all_features]
                                y = pmdc[target_variable]

                                # 1. Walidacja Krzyżowa
                                st.markdown("---"); st.subheader(f"1. Wyniki {n_splits}-krotnej Walidacji Krzyżowej (KNN)")
                                with st.spinner(f"Uruchamianie {n_splits}-krotnej walidacji krzyżowej KNN (k={k})..."):
                                    try:
                                        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                                        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                                        st.success("Walidacja krzyżowa KNN zakończona.")
                                        st.metric("Średnia dokładność (CV):", f"{scores.mean():.2%}")
                                        st.metric("Odchylenie standardowe (CV):", f"{scores.std():.4f}")
                                        st.text("Dokładności w poszczególnych foldach: " + ", ".join([f"{s:.2%}" for s in scores]))
                                    except Exception as e_cv:
                                        st.error(f"Wystąpił błąd podczas walidacji krzyżowej KNN: {e_cv}")

                                # 2. Ocena na pojedynczym podziale
                                st.markdown("---"); st.subheader(f"2. Ocena KNN na Podziale Trening/Test ({train_size_percent}%/{test_size_percent}%)")
                                with st.spinner(f"Trenowanie i ocena KNN na pojedynczym podziale (k={k})..."):
                                     try:
                                         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_float, random_state=42, stratify=y)
                                         if X_test.empty:
                                             st.warning("Zbiór testowy jest pusty po podziale. Nie można przeprowadzić oceny.")
                                         else:
                                             # Trenowanie pipeline'u
                                             pipeline.fit(X_train, y_train)
                                             # Predykcja na zbiorze testowym
                                             y_pred_single = pipeline.predict(X_test)
                                             # Obliczanie metryk
                                             accuracy_single = accuracy_score(y_test, y_pred_single)
                                             report_dict = classification_report(y_test, y_pred_single, target_names=['Niecelny', 'Celny'], output_dict=True, zero_division=0)
                                             cm = confusion_matrix(y_test, y_pred_single)

                                             st.success("Ocena KNN na pojedynczym podziale zakończona.")
                                             st.metric("Dokładność (na podziale testowym):", f"{accuracy_single:.2%}")

                                             st.subheader("Raport Klasyfikacji (KNN):")
                                             report_df = pd.DataFrame(report_dict).transpose()
                                             st.dataframe(report_df.style.format({'precision': '{:.2%}', 'recall': '{:.2%}', 'f1-score': '{:.2f}', 'support': '{:.0f}'}))

                                             st.subheader("Macierz Pomyłek (KNN):")
                                             fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predykcja", y="Prawda"),
                                                                x=['Niecelny (0)', 'Celny (1)'], y=['Niecelny (0)', 'Celny (1)'],
                                                                title="Macierz Pomyłek KNN (na podziale testowym)")
                                             st.plotly_chart(fig_cm, use_container_width=True)
                                     except Exception as e_split:
                                         st.error(f"Wystąpił błąd podczas oceny KNN na pojedynczym podziale: {e_split}")
                        else:
                            st.warning(f"Niewystarczająca ilość danych ({len(pmdc)}) dla gracza '{selected_player}' do zbudowania modelu KNN. Minimum wymagane: {min_samples_for_model}.")
                else:
                    st.warning(f"Brak wymaganych kolumn do zbudowania modelu dla gracza '{selected_player}'. Brakujące kolumny: {', '.join(missing_cols)}")
            else:
                st.warning(f"Brak danych dla gracza '{selected_player}' w wybranym typie sezonu '{selected_season_type}'.")
        else:
            st.info("Wybierz gracza z panelu bocznego, aby uruchomić ocenę modelu KNN.")


    elif st.session_state.active_view == "🎯 Ocena Modelu (XGBoost)":
        # --- Prefiks dla kluczy widgetów w tym widoku: xgb_eval_ ---
        st.header(f"Ocena Modelu Predykcyjnego (XGBoost) dla: {selected_player}")

        st.markdown(f"""
        ### Interpretacja Wyników Modelu XGBoost ({selected_player})

        Ta zakładka prezentuje model **XGBoost (Extreme Gradient Boosting)** do przewidywania wyniku rzutu (celny/niecelny). Podobnie jak KNN, używa on tych samych cech wejściowych:

        * **Numeryczne:** `LOC_X`, `LOC_Y`, `SHOT_DISTANCE` (skalowane za pomocą `StandardScaler`).
        * **Kategoryczne:** `ACTION_TYPE`/`ACTION_TYPE_SIMPLE`, `SHOT_TYPE`, `PERIOD` (przekształcone za pomocą `One-Hot Encoding`).

        Wykorzystano ten sam potok (`Pipeline`) przetwarzania danych co dla KNN, aby zapewnić spójność i umożliwić porównanie wyników. Wydajność modelu XGBoost jest oceniana na dwa sposoby:

        **1. Walidacja Krzyżowa (Stratified K-Fold) - Ocena Ogólnej Wydajności:**

        * **Cel:** Uzyskanie **niezawodnej** oceny, jak dobrze model XGBoost generalizuje na nowych, niewidzianych danych.
        * **Jak działa:** Dane gracza są dzielone na `n` części (folds). Model (potok z `ColumnTransformer` + `XGBClassifier`) jest trenowany `n` razy na `n-1` częściach i testowany na pozostałej.
        * **Wyniki:** Średnia dokładność i odchylenie standardowe pokazują oczekiwaną skuteczność i stabilność modelu XGBoost.

        **2. Ocena na Pojedynczym Podziale Trening/Test - Szczegółowa Analiza:**

        * **Cel:** Zaprezentowanie **szczegółowych metryk** (Raport Klasyfikacji, Macierz Pomyłek) dla *jednego konkretnego* podziału danych, aby zrozumieć, jakie błędy popełnia model XGBoost.
        * **Jak działa:** Dane są jednorazowo dzielone (zgodnie z wybranym procentem). Ten sam potok (`ColumnTransformer` + `XGBClassifier`) jest trenowany na zbiorze treningowym i oceniany na testowym.
        * **Wyniki:** Dokładność, Raport Klasyfikacji i Macierz Pomyłek pokazują działanie modelu na tym konkretnym podziale.

        **3. Interpretacja SHAP (Dodano w tej wersji):**

        * **Cel:** Zrozumienie, **które cechy** miały największy wpływ na predykcje modelu i **w jaki sposób** wpływały (pozytywnie/negatywnie) na przewidywanie rzutu celnego.
        * **Jak działa:** Używa biblioteki SHAP do obliczenia wartości Shapleya dla każdej cechy w danych testowych. Wartości te reprezentują "wkład" każdej cechy w końcową predykcję.
        * **Wyniki:** Wykresy `summary_plot` (beeswarm i bar) wizualizują globalną ważność cech i kierunek ich wpływu. Dodatkowo, poniżej wykresów znajduje się **automatyczne podsumowanie** wskazujące najważniejszą cechę i jej ogólny wpływ.

        **Podsumowanie:** XGBoost jest często potężniejszym algorytmem niż KNN, szczególnie na danych tabelarycznych, ale może wymagać więcej zasobów obliczeniowych. Porównaj wyniki z zakładki KNN, aby zobaczyć, który model lepiej sprawdza się dla danego gracza i zestawu danych. Pamiętaj, że wyniki na pojedynczym podziale mogą zależeć od losowości podziału (`random_state=42`). Analiza SHAP dostarcza dodatkowego wglądu w "czarną skrzynkę" modelu.
        """)

        if selected_player:
            player_model_data = filter_data_by_player(selected_player, filtered_data) # Użyj danych przefiltrowanych wg sezonu
            if not player_model_data.empty:
                # Definicja cech i zmiennej docelowej (analogicznie jak w KNN)
                numerical_features = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE']
                action_type_col_model_xgb = 'ACTION_TYPE' # Domyślnie
                if 'ACTION_TYPE_SIMPLE' in player_model_data.columns:
                     use_simple_action_xgb = st.checkbox("Użyj uproszczonych typów akcji (ACTION_TYPE_SIMPLE) w modelu XGBoost?", value=True, key='xgb_eval_use_simple_action')
                     if use_simple_action_xgb:
                         action_type_col_model_xgb = 'ACTION_TYPE_SIMPLE'

                categorical_features = [action_type_col_model_xgb, 'SHOT_TYPE', 'PERIOD']
                target_variable = 'SHOT_MADE_FLAG'
                all_features = numerical_features + categorical_features

                missing_cols = [col for col in all_features + [target_variable] if col not in player_model_data.columns]
                if not missing_cols:
                    # Przygotowanie danych (identyczne jak dla KNN)
                    pmdc = player_model_data[all_features + [target_variable]].dropna().copy()
                    pmdc[target_variable] = pd.to_numeric(pmdc[target_variable], errors='coerce')
                    pmdc = pmdc.dropna(subset=[target_variable])
                    for col in categorical_features: pmdc[col] = pmdc[col].astype(str)

                    if pmdc[target_variable].nunique() < 2 and not pmdc.empty:
                        st.warning(f"Dla gracza '{selected_player}' pozostała tylko jedna klasa wyniku rzutu po przetworzeniu danych. Nie można zbudować modelu klasyfikacyjnego.")
                    elif pmdc.empty:
                        st.warning(f"Brak kompletnych danych (po usunięciu NaN) dla gracza '{selected_player}' do zbudowania modelu.")
                    else:
                        pmdc[target_variable] = pmdc[target_variable].astype(int)
                        min_samples_for_model = 50
                        if len(pmdc) >= min_samples_for_model:
                            st.subheader("Konfiguracja Oceny Modelu XGBoost")
                            # Slidery do konfiguracji (tylko CV i podział)
                            n_splits_xgb = st.slider("Liczba podziałów CV:", 3, 10, 5, 1, key='xgb_eval_cv_splits')
                            st.markdown("---")
                            st.subheader("Konfiguracja Pojedynczego Podziału Testowego")
                            test_size_percent_xgb = st.slider("Rozmiar zbioru testowego (%):", 10, 50, 20, 5, key='xgb_eval_test_split', format="%d%%")
                            train_size_percent_xgb = 100 - test_size_percent_xgb
                            test_size_float_xgb = test_size_percent_xgb / 100.0

                            if st.button(f"Uruchom Ocenę Modelu XGBoost dla {selected_player}", key='xgb_eval_run_button'):
                                # Definicja preprocesora i pipeline'u XGBoost (identyczny preprocesor jak w KNN)
                                preprocessor = ColumnTransformer(
                                    transformers=[
                                        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_features),
                                        ('num', StandardScaler(), numerical_features)],
                                    remainder='passthrough'
                                )
                                pipeline_xgb = Pipeline([
                                    ('preprocessor', preprocessor),
                                    ('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')) # Użyj standardowych parametrów XGBoost
                                ])

                                X = pmdc[all_features]
                                y = pmdc[target_variable]

                                # 1. Walidacja Krzyżowa XGBoost
                                st.markdown("---"); st.subheader(f"1. Wyniki {n_splits_xgb}-krotnej Walidacji Krzyżowej (XGBoost)")
                                with st.spinner(f"Uruchamianie {n_splits_xgb}-krotnej walidacji krzyżowej XGBoost..."):
                                    try:
                                        cv_xgb = StratifiedKFold(n_splits=n_splits_xgb, shuffle=True, random_state=42)
                                        scores_xgb = cross_val_score(pipeline_xgb, X, y, cv=cv_xgb, scoring='accuracy', n_jobs=-1)
                                        st.success("Walidacja krzyżowa XGBoost zakończona.")
                                        st.metric("Średnia dokładność (CV):", f"{scores_xgb.mean():.2%}")
                                        st.metric("Odchylenie standardowe (CV):", f"{scores_xgb.std():.4f}")
                                        st.text("Dokładności w poszczególnych foldach: " + ", ".join([f"{s:.2%}" for s in scores_xgb]))
                                    except Exception as e_cv_xgb:
                                        st.error(f"Wystąpił błąd podczas walidacji krzyżowej XGBoost: {e_cv_xgb}")

                                # 2. Ocena XGBoost na pojedynczym podziale
                                st.markdown("---"); st.subheader(f"2. Ocena XGBoost na Podziale Trening/Test ({train_size_percent_xgb}%/{test_size_percent_xgb}%)")
                                with st.spinner(f"Trenowanie i ocena XGBoost na pojedynczym podziale..."):
                                     try:
                                         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_float_xgb, random_state=42, stratify=y)
                                         if X_test.empty:
                                             st.warning("Zbiór testowy jest pusty po podziale. Nie można przeprowadzić oceny.")
                                         else:
                                             # Trenowanie pipeline'u XGBoost
                                             pipeline_xgb.fit(X_train, y_train)
                                             # Predykcja
                                             y_pred_xgb = pipeline_xgb.predict(X_test)
                                             # Metryki
                                             accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
                                             report_dict_xgb = classification_report(y_test, y_pred_xgb, target_names=['Niecelny', 'Celny'], output_dict=True, zero_division=0)
                                             cm_xgb = confusion_matrix(y_test, y_pred_xgb)

                                             st.success("Ocena XGBoost na pojedynczym podziale zakończona.")
                                             st.metric("Dokładność (na podziale testowym):", f"{accuracy_xgb:.2%}")

                                             st.subheader("Raport Klasyfikacji (XGBoost):")
                                             report_df_xgb = pd.DataFrame(report_dict_xgb).transpose()
                                             st.dataframe(report_df_xgb.style.format({'precision': '{:.2%}', 'recall': '{:.2%}', 'f1-score': '{:.2f}', 'support': '{:.0f}'}))

                                             st.subheader("Macierz Pomyłek (XGBoost):")
                                             fig_cm_xgb = px.imshow(cm_xgb, text_auto=True, labels=dict(x="Predykcja", y="Prawda"),
                                                                    x=['Niecelny (0)', 'Celny (1)'], y=['Niecelny (0)', 'Celny (1)'],
                                                                    title="Macierz Pomyłek XGBoost (na podziale testowym)",
                                                                    color_continuous_scale=px.colors.sequential.Greens) # Inna skala kolorów dla odróżnienia
                                             st.plotly_chart(fig_cm_xgb, use_container_width=True)

                                             # === SEKCJA SHAP ===
                                             st.markdown("---")
                                             st.subheader("3. Interpretacja Modelu XGBoost (SHAP)")
                                             st.caption("Analiza wpływu cech na predykcje (dane testowe).")
                                             with st.spinner("Obliczanie wartości SHAP..."):
                                                 try:
                                                     # Pobierz wytrenowany model i preprocesor z pipeline'u
                                                     model = pipeline_xgb.named_steps['xgb']
                                                     preprocessor_fitted = pipeline_xgb.named_steps['preprocessor']

                                                     # Utwórz explainer SHAP dla modelu drzewiastego
                                                     explainer = shap.TreeExplainer(model)

                                                     # Przetransformuj dane testowe za pomocą dopasowanego preprocesora
                                                     X_test_transformed = preprocessor_fitted.transform(X_test)
                                                     feature_names_out = preprocessor_fitted.get_feature_names_out()
                                                     feature_names_out = [str(fn) for fn in feature_names_out] # Konwersja na stringi

                                                     # Oblicz wartości SHAP
                                                     shap_values = explainer.shap_values(X_test_transformed)

                                                     # Wybierz wartości SHAP dla klasy pozytywnej (1 - 'Celny')
                                                     if isinstance(shap_values, list) and len(shap_values) == 2:
                                                         shap_values_for_plot = shap_values[1]
                                                         # st.info("Interpretacja SHAP dla klasy pozytywnej (rzut celny).") # Mniej gadatliwe
                                                     else:
                                                         shap_values_for_plot = shap_values
                                                         # st.info("Interpretacja SHAP (zakładając wpływ na wynik pozytywny).")

                                                     # Stwórz DataFrame z przekształconymi danymi (potrzebne do summary_plot)
                                                     X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names_out)

                                                     # Wykres Beeswarm
                                                     st.markdown("#### Globalna Ważność Cech (Summary Plot - Beeswarm)")
                                                     st.markdown("Pozycja X: Wpływ na predykcję 'Celny' (wyższe SHAP = większa szansa). Kolor: Wartość cechy (Czerwony=Wysoka, Niebieski=Niska).")
                                                     fig_summary, ax_summary = plt.subplots()
                                                     shap.summary_plot(shap_values_for_plot, X_test_transformed_df, feature_names=feature_names_out, show=False, plot_type="beeswarm")
                                                     st.pyplot(fig_summary, bbox_inches='tight')
                                                     plt.clf() # Wyczyść figurę

                                                     # Wykres Bar (średnia abs. wartość SHAP)
                                                     st.markdown("#### Średnia Absolutna Wartość SHAP (Ważność Cech - Bar)")
                                                     fig_bar, ax_bar = plt.subplots()
                                                     shap.summary_plot(shap_values_for_plot, X_test_transformed_df, feature_names=feature_names_out, show=False, plot_type="bar")
                                                     st.pyplot(fig_bar, bbox_inches='tight')
                                                     plt.clf() # Wyczyść figurę

                                                     # --- POCZĄTEK: Dodana sekcja interpretacji ---
                                                     try:
                                                         st.markdown("---")
                                                         st.markdown("##### Podsumowanie Interpretacji SHAP")

                                                         # 1. Oblicz średnią absolutną wartość SHAP dla rankingu ważności
                                                         mean_abs_shap = np.abs(shap_values_for_plot).mean(axis=0)

                                                         # 2. Stwórz DataFrame ważności cech
                                                         feature_importance_df = pd.DataFrame({
                                                             'feature': feature_names_out,
                                                             'importance': mean_abs_shap
                                                         }).sort_values(by='importance', ascending=False).reset_index(drop=True)

                                                         if not feature_importance_df.empty:
                                                             # 3. Pobierz najważniejszą cechę
                                                             top_feature_name = feature_importance_df.loc[0, 'feature']
                                                             top_feature_importance = feature_importance_df.loc[0, 'importance']

                                                             # 4. Spróbuj określić ogólny kierunek wpływu (heurystyka korelacji)
                                                             comment = f"Dla gracza **{selected_player}**, najważniejszą cechą wpływającą na predykcję wyniku rzutu przez model XGBoost była **`{top_feature_name}`** (średni abs. wpływ SHAP: {top_feature_importance:.3f})."
                                                             try:
                                                                 # Znajdź indeks cechy w oryginalnej liście (dla spójności)
                                                                 # Musimy użyć listy, bo feature_names_out może być numpy array
                                                                 top_feature_original_index = list(feature_names_out).index(top_feature_name)

                                                                 # Pobierz wartości tej cechy z przekształconych danych testowych
                                                                 top_feature_values = X_test_transformed_df[top_feature_name].values
                                                                 # Pobierz wartości SHAP dla tej cechy (dla klasy pozytywnej)
                                                                 top_feature_shap_values = shap_values_for_plot[:, top_feature_original_index]

                                                                 # Oblicz korelację, upewniając się, że dane mają wariancję
                                                                 correlation = np.nan # Domyślnie NaN
                                                                 # Użyj małego progu dla std dev, aby uniknąć dzielenia przez zero
                                                                 if len(top_feature_values) > 1 and len(top_feature_shap_values) > 1 and np.std(top_feature_values) > 1e-6 and np.std(top_feature_shap_values) > 1e-6:
                                                                      correlation = np.corrcoef(top_feature_values, top_feature_shap_values)[0, 1]

                                                                 if not np.isnan(correlation):
                                                                     impact_description = ""
                                                                     # Sprawdź, czy cecha jest prawdopodobnie numeryczna (prosty test)
                                                                     # UWAGA: To uproszczenie!
                                                                     is_numerical_heuristic = any(num_feat in top_feature_name for num_feat in numerical_features)

                                                                     if correlation < -0.1: # Istotna negatywna korelacja
                                                                         if is_numerical_heuristic:
                                                                             impact_description = f" Generalnie, **wyższe wartości** tej cechy (np. większa odległość) **zmniejszają** przewidywaną szansę na trafienie (niższe SHAP)."
                                                                         else: # Prawdopodobnie cecha OHE
                                                                             impact_description = f" Obecność tej cechy (wartość 1) generalnie **zmniejsza** przewidywaną szansę na trafienie (niższe SHAP)."
                                                                     elif correlation > 0.1: # Istotna pozytywna korelacja
                                                                          if is_numerical_heuristic:
                                                                              impact_description = f" Generalnie, **wyższe wartości** tej cechy **zwiększają** przewidywaną szansę na trafienie (wyższe SHAP)."
                                                                          else: # Prawdopodobnie cecha OHE
                                                                              impact_description = f" Obecność tej cechy (wartość 1) generalnie **zwiększa** przewidywaną szansę na trafienie (wyższe SHAP)."
                                                                     else: # Słaba korelacja
                                                                          impact_description = " Kierunek jej wpływu wydaje się być zróżnicowany lub słaby w ogólnym ujęciu."

                                                                     comment += impact_description
                                                                 else:
                                                                      comment += " Nie udało się jednoznacznie określić ogólnego kierunku wpływu tej cechy (np. z powodu braku wariancji lub błędu obliczeń)."

                                                             except IndexError:
                                                                  comment += f" Nie udało się znaleźć indeksu dla cechy '{top_feature_name}'."
                                                             except Exception as e_corr:
                                                                 # Błąd podczas próby analizy korelacji
                                                                 comment += f" Nie udało się określić kierunku wpływu (wystąpił błąd: {e_corr})."

                                                             st.caption(comment) # Wyświetl podsumowanie jako podpis
                                                         else:
                                                             st.caption("Nie znaleziono ważnych cech w analizie SHAP.")

                                                     except Exception as e_interp:
                                                         st.warning(f"Wystąpił błąd podczas generowania tekstowej interpretacji SHAP: {e_interp}")
                                                     # --- KONIEC: Dodana sekcja interpretacji ---

                                                 except ImportError:
                                                     st.error("Biblioteka SHAP lub Matplotlib nie jest zainstalowana. Zainstaluj `shap` i `matplotlib`, aby zobaczyć interpretację modelu.")
                                                 except Exception as e_shap:
                                                     st.error(f"Wystąpił błąd podczas obliczania lub wizualizacji SHAP: {e_shap}")
                                             # === KONIEC SEKCJI SHAP ===

                                     except Exception as e_split_xgb:
                                         st.error(f"Wystąpił błąd podczas oceny XGBoost na pojedynczym podziale: {e_split_xgb}")
                        else:
                            st.warning(f"Niewystarczająca ilość danych ({len(pmdc)}) dla gracza '{selected_player}' do zbudowania modelu XGBoost. Minimum wymagane: {min_samples_for_model}.")
                else:
                    st.warning(f"Brak wymaganych kolumn do zbudowania modelu dla gracza '{selected_player}'. Brakujące kolumny: {', '.join(missing_cols)}")
            else:
                st.warning(f"Brak danych dla gracza '{selected_player}' w wybranym typie sezonu '{selected_season_type}'.")
        else:
            st.info("Wybierz gracza z panelu bocznego, aby uruchomić ocenę modelu XGBoost.")

    elif st.session_state.active_view == "📊 Porównanie Modeli (KNN vs XGBoost)":
        # --- Prefiks dla kluczy widgetów w tym widoku: model_comp_ ---
        st.header(f"Porównanie Modeli Predykcyjnych (KNN vs XGBoost) dla: {selected_player}")

        st.markdown(f"""
        Ta zakładka pozwala na bezpośrednie porównanie wydajności modeli KNN i XGBoost dla wybranego gracza: **{selected_player}**.
        Używane są te same dane wejściowe i ten sam potok przetwarzania wstępnego (skalowanie cech numerycznych, One-Hot Encoding cech kategorycznych) dla obu modeli, aby zapewnić uczciwe porównanie.

        Porównywane są:
        * **Średnia dokładność z walidacji krzyżowej:** Daje obraz ogólnej stabilności i zdolności generalizacji obu modeli.
        * **Dokładność na pojedynczym podziale testowym:** Pokazuje wydajność na konkretnym, tym samym dla obu modeli, zbiorze testowym.
        * **Macierze pomyłek na podziale testowym:** Pozwalają wizualnie porównać typy błędów popełnianych przez oba modele.

        Pamiętaj, że wyniki na pojedynczym podziale mogą zależeć od losowego podziału danych (`random_state=42`). Walidacja krzyżowa daje bardziej wiarygodny obraz ogólnej wydajności.
        """)

        if selected_player:
            player_model_data = filter_data_by_player(selected_player, filtered_data) # Użyj danych przefiltrowanych wg sezonu
            if not player_model_data.empty:
                 # Definicja cech (identyczna jak w poprzednich zakładkach)
                numerical_features = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE']
                # Użyj tej samej logiki wyboru kolumny akcji jak w KNN/XGBoost
                action_type_col_model_comp = 'ACTION_TYPE' # Domyślnie
                if 'ACTION_TYPE_SIMPLE' in player_model_data.columns:
                     # Dodaj checkbox, aby użytkownik mógł wybrać spójnie
                     use_simple_action_comp = st.checkbox("Użyj uproszczonych typów akcji w obu modelach do porównania?", value=True, key='comp_model_use_simple_action')
                     if use_simple_action_comp:
                         action_type_col_model_comp = 'ACTION_TYPE_SIMPLE'

                categorical_features = [action_type_col_model_comp, 'SHOT_TYPE', 'PERIOD']
                target_variable = 'SHOT_MADE_FLAG'
                all_features = numerical_features + categorical_features

                missing_cols = [col for col in all_features + [target_variable] if col not in player_model_data.columns]
                if not missing_cols:
                    # Przygotowanie danych
                    pmdc = player_model_data[all_features + [target_variable]].dropna().copy()
                    pmdc[target_variable] = pd.to_numeric(pmdc[target_variable], errors='coerce')
                    pmdc = pmdc.dropna(subset=[target_variable])
                    for col in categorical_features: pmdc[col] = pmdc[col].astype(str)

                    if pmdc[target_variable].nunique() < 2 and not pmdc.empty:
                        st.warning(f"Dla gracza '{selected_player}' pozostała tylko jedna klasa wyniku rzutu. Nie można porównać modeli.")
                    elif pmdc.empty:
                        st.warning(f"Brak kompletnych danych dla gracza '{selected_player}' do porównania modeli.")
                    else:
                        pmdc[target_variable] = pmdc[target_variable].astype(int)
                        min_samples_for_model = 50
                        if len(pmdc) >= min_samples_for_model:
                            st.subheader("Konfiguracja Porównania Modeli")
                            # Wspólne parametry dla obu modeli
                            k_comp = st.slider("Liczba sąsiadów (k dla KNN):", 3, min(25, len(pmdc)//3), 5, 2, key='model_comp_knn_k')
                            n_splits_comp = st.slider("Liczba podziałów CV:", 3, 10, 5, 1, key='model_comp_cv_splits')
                            test_size_percent_comp = st.slider("Rozmiar zbioru testowego (%):", 10, 50, 20, 5, key='model_comp_test_split', format="%d%%")
                            train_size_percent_comp = 100 - test_size_percent_comp
                            test_size_float_comp = test_size_percent_comp / 100.0

                            if st.button(f"Uruchom Porównanie KNN vs XGBoost dla {selected_player}", key='model_comp_run_button'):
                                X = pmdc[all_features]; y = pmdc[target_variable]
                                # Podział na zbiór treningowy i testowy (ten sam dla obu modeli)
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_float_comp, random_state=42, stratify=y)

                                if X_test.empty:
                                    st.error("Zbiór testowy jest pusty po podziale. Nie można porównać modeli.")
                                else:
                                    # Definicja preprocesora (ten sam dla obu)
                                    preprocessor = ColumnTransformer(
                                        transformers=[
                                            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_features),
                                            ('num', StandardScaler(), numerical_features)],
                                        remainder='passthrough'
                                    )
                                    # Definicja pipeline'ów
                                    pipeline_knn = Pipeline([('preprocessor', preprocessor), ('knn', KNeighborsClassifier(n_neighbors=k_comp))])
                                    pipeline_xgb = Pipeline([('preprocessor', preprocessor), ('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))])

                                    results = {"knn": {}, "xgb": {}} # Słownik na wyniki

                                    # Porównanie Walidacji Krzyżowej
                                    st.markdown("---"); st.subheader(f"Porównanie Walidacji Krzyżowej ({n_splits_comp} folds)")
                                    col_cv1, col_cv2 = st.columns(2)
                                    with col_cv1:
                                        st.markdown(f"**KNN (k={k_comp})**")
                                        with st.spinner(f"CV KNN..."):
                                            try:
                                                cv_knn = StratifiedKFold(n_splits=n_splits_comp, shuffle=True, random_state=42)
                                                scores_knn = cross_val_score(pipeline_knn, X, y, cv=cv_knn, scoring='accuracy', n_jobs=-1)
                                                results["knn"]["cv_mean"] = scores_knn.mean(); results["knn"]["cv_std"] = scores_knn.std()
                                                st.metric("Średnia Dokł.:", f"{results['knn']['cv_mean']:.2%}"); st.metric("Odch. Std.:", f"{results['knn']['cv_std']:.4f}")
                                            except Exception as e: results["knn"]["cv_mean"] = "Błąd"; st.error(f"Błąd CV KNN: {e}")
                                    with col_cv2:
                                        st.markdown("**XGBoost**")
                                        with st.spinner(f"CV XGBoost..."):
                                            try:
                                                cv_xgb = StratifiedKFold(n_splits=n_splits_comp, shuffle=True, random_state=42)
                                                scores_xgb = cross_val_score(pipeline_xgb, X, y, cv=cv_xgb, scoring='accuracy', n_jobs=-1)
                                                results["xgb"]["cv_mean"] = scores_xgb.mean(); results["xgb"]["cv_std"] = scores_xgb.std()
                                                st.metric("Średnia Dokł.:", f"{results['xgb']['cv_mean']:.2%}"); st.metric("Odch. Std.:", f"{results['xgb']['cv_std']:.4f}")
                                            except Exception as e: results["xgb"]["cv_mean"] = "Błąd"; st.error(f"Błąd CV XGBoost: {e}")

                                    # Porównanie na Podziale Testowym
                                    st.markdown("---"); st.subheader(f"Porównanie na Podziale Testowym ({test_size_percent_comp}%)")
                                    st.caption(f"Porównanie na {len(y_test)} próbkach testowych.")
                                    col_s1, col_s2 = st.columns(2)
                                    with col_s1:
                                        st.markdown(f"**KNN (k={k_comp})**")
                                        with st.spinner("Ocena KNN..."):
                                            try:
                                                pipeline_knn.fit(X_train, y_train)
                                                y_pred_knn = pipeline_knn.predict(X_test)
                                                results["knn"]["single_acc"] = accuracy_score(y_test, y_pred_knn)
                                                results["knn"]["conf_matrix"] = confusion_matrix(y_test, y_pred_knn)
                                                st.metric("Dokładność:", f"{results['knn']['single_acc']:.2%}")
                                                fig_cm_knn = px.imshow(results["knn"]["conf_matrix"], text_auto=True, x=['N(0)', 'C(1)'], y=['N(0)', 'C(1)'], title=f"Macierz Pomyłek KNN")
                                                fig_cm_knn.update_layout(height=300, title_font_size=14); st.plotly_chart(fig_cm_knn, use_container_width=True)
                                            except Exception as e: results["knn"]["single_acc"] = "Błąd"; st.error(f"Błąd oceny KNN: {e}")
                                    with col_s2:
                                        st.markdown("**XGBoost**")
                                        with st.spinner("Ocena XGBoost..."):
                                            try:
                                                pipeline_xgb.fit(X_train, y_train)
                                                y_pred_xgb = pipeline_xgb.predict(X_test)
                                                results["xgb"]["single_acc"] = accuracy_score(y_test, y_pred_xgb)
                                                results["xgb"]["conf_matrix"] = confusion_matrix(y_test, y_pred_xgb)
                                                st.metric("Dokładność:", f"{results['xgb']['single_acc']:.2%}")
                                                fig_cm_xgb = px.imshow(results["xgb"]["conf_matrix"], text_auto=True, x=['N(0)', 'C(1)'], y=['N(0)', 'C(1)'], title="Macierz Pomyłek XGBoost", color_continuous_scale=px.colors.sequential.Greens)
                                                fig_cm_xgb.update_layout(height=300, title_font_size=14); st.plotly_chart(fig_cm_xgb, use_container_width=True)
                                            except Exception as e: results["xgb"]["single_acc"] = "Błąd"; st.error(f"Błąd oceny XGBoost: {e}")

                                    # Podsumowanie porównania
                                    st.markdown("---"); st.subheader("Podsumowanie Porównania")
                                    summary_data = {
                                        'Metryka': ['Śr. Dokładność (CV)', 'Dokładność (Test)'],
                                        'KNN': [f"{results['knn'].get('cv_mean', 'N/A'):.2%}" if isinstance(results['knn'].get('cv_mean'), float) else "N/A",
                                                f"{results['knn'].get('single_acc', 'N/A'):.2%}" if isinstance(results['knn'].get('single_acc'), float) else "N/A"],
                                        'XGBoost': [f"{results['xgb'].get('cv_mean', 'N/A'):.2%}" if isinstance(results['xgb'].get('cv_mean'), float) else "N/A",
                                                    f"{results['xgb'].get('single_acc', 'N/A'):.2%}" if isinstance(results['xgb'].get('single_acc'), float) else "N/A"]
                                    }
                                    summary_df = pd.DataFrame(summary_data)
                                    st.dataframe(summary_df, hide_index=True, use_container_width=True)

                                    winner_cv = "Remis"
                                    if isinstance(results['knn'].get('cv_mean'), float) and isinstance(results['xgb'].get('cv_mean'), float):
                                        if results['xgb']['cv_mean'] > results['knn']['cv_mean']: winner_cv = "XGBoost"
                                        elif results['knn']['cv_mean'] > results['xgb']['cv_mean']: winner_cv = "KNN"
                                    winner_test = "Remis"
                                    if isinstance(results['knn'].get('single_acc'), float) and isinstance(results['xgb'].get('single_acc'), float):
                                         if results['xgb']['single_acc'] > results['knn']['single_acc']: winner_test = "XGBoost"
                                         elif results['knn']['single_acc'] > results['xgb']['single_acc']: winner_test = "KNN"

                                    st.markdown(f"**Wnioski:**")
                                    st.markdown(f"- Pod względem **średniej dokładności CV**: **{winner_cv}**")
                                    st.markdown(f"- Pod względem **dokładności na podziale testowym**: **{winner_test}**")

                        else:
                            st.warning(f"Niewystarczająca ilość danych ({len(pmdc)}) dla gracza '{selected_player}' do porównania modeli. Minimum wymagane: {min_samples_for_model}.")
                else:
                    st.warning(f"Brak wymaganych kolumn do porównania modeli dla gracza '{selected_player}'. Brakujące kolumny: {', '.join(missing_cols)}")
            else:
                st.warning(f"Brak danych dla gracza '{selected_player}' w wybranym typie sezonu '{selected_season_type}'.")
        else:
            st.info("Wybierz gracza z panelu bocznego, aby uruchomić porównanie modeli.")

# Obsługa przypadku, gdy dane nie zostały wczytane poprawnie
else:
    st.error("Nie udało się wczytać lub przetworzyć danych. Sprawdź ścieżkę do pliku CSV i jego format.")
    if 'load_error_message' in st.session_state and st.session_state.load_error_message:
        st.error(f"Szczegóły błędu: {st.session_state.load_error_message}")

# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Rozszerzona Aplikacja Streamlit - Analiza NBA")
try:
    # Użyj strefy czasowej UTC jako fallback, jeśli 'Europe/Warsaw' nie jest dostępna
    try:
        tz = pytz.timezone('Europe/Warsaw')
    except pytz.exceptions.UnknownTimeZoneError:
        tz = pytz.utc
        print("Ostrzeżenie: Strefa czasowa 'Europe/Warsaw' niedostępna, używam UTC.")

    # Pobierz bieżącą datę i czas w odpowiedniej strefie
    ts = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z')
except ImportError:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S (czas lokalny - brak pytz)')
    print("Ostrzeżenie: Biblioteka 'pytz' nie jest zainstalowana.")
except Exception as e_time:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S (czas lokalny - błąd strefy)')
    print(f"Błąd podczas ustawiania strefy czasowej: {e_time}")
st.sidebar.markdown(f"Czas serwera: {ts}")