# nba_app_v4_with_comparison_rev2.py
# Added dedicated model comparison tab
# Revision 1: Implemented key naming convention and tab renaming
# Revision 2: Added SHAP interpretability for XGBoost model
# Revision 3 (by AI): Added League Average 2PT/3PT display in Rankings Tab
# Revision 4 (by AI): Added Simplified Action Types (feature & option)
# Revision 5 (by AI): Added Shot Distribution by Basic Zone chart
# Revision 6 (by AI): Adjusted layout and label visibility in Rankings Tab

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
import matplotlib.pyplot as plt # Import Matplotlib

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
CSV_FILE_PATH = 'nba_player_shooting_data_2023_24.csv' # Dostosuj!

# --- Funkcje Pomocnicze ---
# (Wszystkie funkcje pomocnicze z poprzedniej wersji pozostają bez zmian)
# load_shooting_data, add_court_shapes, filter_data_by_player,
# filter_data_by_team, get_basic_stats, plot_player_eff_vs_distance,
# plot_shot_chart, calculate_hot_zones, plot_hot_zones_heatmap,
# plot_shot_frequency_heatmap, plot_player_quarter_eff,
# plot_player_season_trend, plot_grouped_effectiveness,
# plot_comparison_eff_distance, plot_comparison_eff_by_zone,
# calculate_top_performers, simplify_action_type
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
        fig.update_traces(connectgaps=False)
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
    n_bins = max(2, n_bins)
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
    for col in required_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    plot_df = plot_df.dropna()
    if plot_df.empty: return None
    min_pct, max_pct = plot_df['percentage'].min(), plot_df['percentage'].max()
    color_range = [max(0, min_pct - 5), min(100, max_pct + 5)] if pd.notna(min_pct) and pd.notna(max_pct) else [0, 100]
    if color_range[0] >= color_range[1]: color_range = [0, 100]
    max_bubble_size = plot_df["total_shots"].max() if not plot_df["total_shots"].empty else 1
    size_ref = max(1, max_bubble_size / 50.0)
    fig = px.scatter(plot_df, x='x_center', y='y_center', size='total_shots', color='percentage',
                     color_continuous_scale=px.colors.diverging.RdYlGn,
                     size_max=60, range_color=color_range,
                     title=f'Skuteczność stref rzutowych ({entity_type}: {entity_name}, min. {min_shots_in_zone} rzutów)',
                     labels={'x_center': 'Pozycja X', 'y_center': 'Pozycja Y', 'total_shots': 'Liczba rzutów', 'percentage': 'Skuteczność (%)'},
                     custom_data=['made_shots', 'total_shots'])
    fig.update_traces(
        hovertemplate="<b>Strefa X:</b> %{x:.1f}, <b>Y:</b> %{y:.1f}<br>" +
                      "<b>Liczba rzutów:</b> %{customdata[1]}<br>" +
                      "<b>Trafione:</b> %{customdata[0]}<br>" +
                      "<b>Skuteczność:</b> %{marker.color:.1f}%<extra></extra>",
        marker=dict(sizeref=size_ref, sizemin=4)
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
        x=plot_data['LOC_X'], y=plot_data['LOC_Y'], colorscale='YlOrRd',
        nbinsx=nbins_x, nbinsy=nbins_y, zauto=True,
        hovertemplate='<b>Zakres X:</b> %{x}<br><b>Zakres Y:</b> %{y}<br><b>Liczba rzutów:</b> %{z}<extra></extra>',
        colorbar=dict(title='Liczba rzutów')
    ))
    fig = add_court_shapes(fig)
    fig.update_layout(
        title=f'Mapa Częstotliwości Rzutów ({season_name})', xaxis_title="Pozycja X", yaxis_title="Pozycja Y",
        height=650, xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='rgba(255, 255, 255, 1)'
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
    quarter_data['PERIOD'] = quarter_data['PERIOD'].astype(int)
    quarter_eff = quarter_data.groupby('PERIOD')['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
    quarter_eff['mean'] *= 100
    quarter_eff = quarter_eff[quarter_eff['count'] >= min_attempts]
    if quarter_eff.empty: return None
    def map_period(p):
        if p <= 4: return f"Kwarta {int(p)}"
        elif p == 5: return "OT 1"
        else: return f"OT {int(p-4)}"
    quarter_eff['Okres Gry'] = quarter_eff['PERIOD'].apply(map_period)
    quarter_eff = quarter_eff.sort_values(by='PERIOD')
    fig = px.bar(quarter_eff, x='Okres Gry', y='mean', text='mean', title=f'Skuteczność w kwartach/dogrywkach - {entity_name} ({entity_type}, min. {min_attempts} prób)',
                 labels={'Okres Gry': 'Okres Gry', 'mean': 'Skuteczność (%)'}, hover_data=['count'])
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(yaxis_range=[0, 105], uniformtext_minsize=8, uniformtext_mode='hide')
    return fig


@st.cache_data
def plot_player_season_trend(entity_data, entity_name, entity_type="Gracz", min_monthly_attempts=10):
    """Wykres trendu skuteczności w trakcie sezonu (miesięcznie)."""
    if 'GAME_DATE' not in entity_data.columns or 'SHOT_MADE_FLAG' not in entity_data.columns: return None
    trend_data = entity_data[['GAME_DATE', 'SHOT_MADE_FLAG']].copy()
    trend_data['GAME_DATE'] = pd.to_datetime(trend_data['GAME_DATE'], errors='coerce')
    trend_data['SHOT_MADE_FLAG'] = pd.to_numeric(trend_data['SHOT_MADE_FLAG'], errors='coerce')
    trend_data = trend_data.dropna()
    if trend_data.empty or len(trend_data) < min_monthly_attempts: return None
    trend_data = trend_data.set_index('GAME_DATE')
    monthly_eff = trend_data.resample('ME')['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
    monthly_eff['mean'] *= 100
    monthly_eff = monthly_eff[monthly_eff['count'] >= min_monthly_attempts]
    if monthly_eff.empty or len(monthly_eff) < 2: return None
    monthly_eff['Miesiąc'] = monthly_eff['GAME_DATE'].dt.strftime('%Y-%m')
    fig = px.line(monthly_eff, x='Miesiąc', y='mean', markers=True, title=f'Miesięczny trend skuteczności - {entity_name} ({entity_type}, min. {min_monthly_attempts} prób/miesiąc)',
                  labels={'Miesiąc': 'Miesiąc', 'mean': 'Skuteczność (%)'}, hover_data=['count'])
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
    grouped_data[group_col] = grouped_data[group_col].astype(str)
    grouped_data['SHOT_MADE_FLAG'] = pd.to_numeric(grouped_data['SHOT_MADE_FLAG'], errors='coerce')
    grouped_data = grouped_data.dropna(subset=[group_col, 'SHOT_MADE_FLAG']) # Drop NaN also in group_col
    if grouped_data.empty: return None
    grouped_eff = grouped_data.groupby(group_col)['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
    grouped_eff['mean'] *= 100
    grouped_eff = grouped_eff[grouped_eff['count'] >= min_attempts]
    # Sort by count first to get top N most frequent, then sort by group_col for consistent display
    grouped_eff = grouped_eff.sort_values(by='count', ascending=False).head(top_n)
    # Sort alphabetically or by a defined order if applicable
    if group_col == 'SHOT_ZONE_BASIC':
         zone_order_basic = ['Restricted Area', 'In The Paint (Non-RA)', 'Mid-Range', 'Left Corner 3', 'Right Corner 3', 'Above the Break 3', 'Backcourt']
         grouped_eff[group_col] = pd.Categorical(
             grouped_eff[group_col],
             categories=[z for z in zone_order_basic if z in grouped_eff[group_col].unique()],
             ordered=True
         )
         grouped_eff = grouped_eff.sort_values(by=group_col)
    else:
         # Sort other categories alphabetically for consistency
         grouped_eff = grouped_eff.sort_values(by=group_col, ascending=True)

    if grouped_eff.empty: return None
    axis_label = group_col.replace('_',' ').title()
    fig = px.bar(grouped_eff, x=group_col, y='mean', text='mean',
                 title=f'Skuteczność wg {axis_label} - {entity_name} ({entity_type}) (Top {top_n} najczęstszych, min. {min_attempts} prób)',
                 labels={group_col: axis_label, 'mean': 'Skuteczność (%)'}, hover_data=['count'])
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(yaxis_range=[0, 105], uniformtext_minsize=8, uniformtext_mode='hide', xaxis_title=axis_label)
    # If it's a specific category order, ensure layout reflects it
    if group_col == 'SHOT_ZONE_BASIC':
         fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray': [z for z in zone_order_basic if z in grouped_eff[group_col].unique()]})

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
        effectiveness['mean'] *= 100
        effectiveness = effectiveness[effectiveness['count'] >= min_attempts_per_bin]
        effectiveness = effectiveness.dropna(subset=['distance_bin_mid'])
        if effectiveness.empty: return None
        effectiveness = effectiveness.sort_values(by=['PLAYER_NAME', 'distance_bin_mid'])
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
    zone_stats_filtered = zone_stats[zone_stats['Attempts'] >= min_shots_per_zone]
    if zone_stats_filtered.empty: return None
    zone_stats_filtered['FG%'] = (zone_stats_filtered['Made'] / zone_stats_filtered['Attempts']) * 100
    zone_order_ideal = ['Restricted Area', 'In The Paint (Non-RA)', 'Mid-Range', 'Left Corner 3', 'Right Corner 3', 'Above the Break 3', 'Backcourt']
    actual_zones_in_data = zone_stats_filtered['SHOT_ZONE_BASIC'].unique()
    zone_order = [zone for zone in zone_order_ideal if zone in actual_zones_in_data]
    zone_order += sorted([zone for zone in actual_zones_in_data if zone not in zone_order_ideal])
    if not zone_order: return None
    zone_stats_plot = zone_stats_filtered[zone_stats_filtered['SHOT_ZONE_BASIC'].isin(zone_order)].copy()
    if zone_stats_plot.empty: return None
    fig = px.bar(zone_stats_plot, x='SHOT_ZONE_BASIC', y='FG%', color='PLAYER_NAME', barmode='group',
                 title=f'Porównanie skuteczności (FG%) wg Strefy Rzutowej (min. {min_shots_per_zone} prób)',
                 labels={'SHOT_ZONE_BASIC': 'Strefa Rzutowa', 'FG%': 'Skuteczność (%)', 'PLAYER_NAME': 'Gracz'},
                 hover_data=['Attempts', 'Made'], category_orders={'SHOT_ZONE_BASIC': zone_order}, text='FG%')
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

    overall_stats = valid_data.groupby(group_by_col)['SHOT_MADE_FLAG'].agg(Made='sum', Attempts='count').reset_index()
    overall_stats = overall_stats[overall_stats['Attempts'] >= min_total_shots]
    top_overall = pd.DataFrame()
    if not overall_stats.empty:
        overall_stats['FG%'] = (overall_stats['Made'] / overall_stats['Attempts']) * 100
        top_overall = overall_stats.sort_values(by=['FG%', 'Attempts'], ascending=[False, False]).head(top_n)
        col_name = group_by_col.replace('_',' ').title()
        top_overall = top_overall.rename(columns={group_by_col: col_name, 'Attempts': 'Próby'})
        top_overall = top_overall[[col_name, 'FG%', 'Próby']]

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
        action_col.str.contains('dunk', na=False),                            # 1. Dunks
        action_col.str.contains('layup', na=False),                           # 2. Layups (po dunkach)
        # 3. Jump Shots (szeroka kategoria)
        action_col.str.contains('jump shot|pullup|step back|fadeaway', na=False, regex=True),
        action_col.str.contains('hook shot', na=False),                       # 4. Hook Shots
        action_col.str.contains('tip|putback', na=False, regex=True),         # 5. Tip-ins / Putbacks
        action_col.str.contains('driving', na=False),                         # 6. Driving (inne niż layup/dunk)
        action_col.str.contains('floating|floater', na=False, regex=True),    # 7. Floaters
        action_col.str.contains('alley oop', na=False),                       # 8. Alley Oops (jeśli nie są dunk/layup)
        action_col.str.contains('bank shot', na=False)                        # 9. Bank Shots (jeśli nie pasują wyżej)
        # Można dodać więcej warunków w razie potrzeby
    ]

    choices = [
        'Dunk',
        'Layup',
        'Jump Shot',
        'Hook Shot',
        'Tip Shot',
        'Driving Shot', # Kategoria dla "driving X shot" jeśli nie jest to layup/dunk
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
    # Wywołaj funkcję upraszczającą TUTAJ
    shooting_data = simplify_action_type(shooting_data) # Dodaje kolumnę 'ACTION_TYPE_SIMPLE'
    # Komunikat sukcesu przeniesiony niżej, aby uwzględnić przetworzenie
# --- Koniec Integracji ---


# --- Sidebar ---
st.sidebar.header("Opcje Filtrowania i Analizy")

# Główna część aplikacji renderowana tylko jeśli dane są wczytane
if load_status.get("success", False) and not shooting_data.empty:

    # Komunikat o sukcesie po przetworzeniu (jeśli było)
    st.success(f"Wczytano i przetworzono dane z {CSV_FILE_PATH}. Wymiary: {load_status.get('shape')}. " +
               ("Dodano 'ACTION_TYPE_SIMPLE'." if 'ACTION_TYPE_SIMPLE' in shooting_data.columns else ""))

    if load_status.get("missing_time_cols"): st.warning("Brak kolumn czasowych. Nie można było utworzyć 'GAME_TIME_SEC' ani 'QUARTER_TYPE'.")
    if load_status.get("nan_in_time_cols"): st.warning("W kolumnach czasowych znaleziono wartości nienumeryczne (NaN).")
    if load_status.get("nan_in_made_flag"): st.warning("W kolumnie SHOT_MADE_FLAG znaleziono wartości niejednoznaczne (NaN).")
    if load_status.get("missing_key_cols"): st.warning(f"Brakujące kluczowe kolumny: {', '.join(load_status['missing_key_cols'])}.")

    # Sidebar Filters
    available_season_types = ['Wszystko'] + shooting_data['SEASON_TYPE'].dropna().unique().tolist() if 'SEASON_TYPE' in shooting_data.columns else ['Wszystko']
    selected_season_type = st.sidebar.selectbox(
        "Wybierz typ sezonu:", options=available_season_types, index=0, key='sidebar_season_select'
    )
    if selected_season_type != 'Wszystko' and 'SEASON_TYPE' in shooting_data.columns:
        # Filtrowanie odbywa się na danych już z ACTION_TYPE_SIMPLE
        filtered_data = shooting_data[shooting_data['SEASON_TYPE'] == selected_season_type].copy()
    else:
        filtered_data = shooting_data.copy()
    st.sidebar.write(f"Wybrano: {selected_season_type} ({len(filtered_data)} rzutów)")

    # Player Selection
    available_players = sorted(filtered_data['PLAYER_NAME'].dropna().unique()) if 'PLAYER_NAME' in filtered_data.columns else []
    default_player = "LeBron James"
    default_player_index = 0
    if available_players:
        try: default_player_index = available_players.index(default_player)
        except ValueError: default_player_index = 0
    selected_player = st.sidebar.selectbox(
        "Gracz do analizy/modelowania:", options=available_players, index=default_player_index,
        key='sidebar_player_select', disabled=not available_players,
        help="Wybierz gracza do szczegółowej analizy lub oceny modeli."
    )

    # Team Selection
    available_teams = sorted(filtered_data['TEAM_NAME'].dropna().unique()) if 'TEAM_NAME' in filtered_data.columns else []
    default_team = "Los Angeles Lakers"
    default_team_index = 0
    if available_teams:
        try: default_team_index = available_teams.index(default_team)
        except ValueError: default_team_index = 0
    selected_team = st.sidebar.selectbox(
        "Drużyna do analizy:", options=available_teams, index=default_team_index,
        key='sidebar_team_select', disabled=not available_teams,
        help="Wybierz drużynę do analizy zespołowej."
    )

    # Player Comparison Selection
    default_compare_players_req = ["LeBron James", "Stephen Curry"]
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

    if st.session_state.active_view not in tab_options:
        st.session_state.active_view = tab_options[0]

    current_index = tab_options.index(st.session_state.active_view)
    st.session_state.active_view = st.radio(
        "Wybierz widok:", options=tab_options, index=current_index,
        key='main_view_selector', horizontal=True, label_visibility="collapsed"
    )
    st.markdown("---")

    # === Poszczególne Widoki ===

    if st.session_state.active_view == "📊 Rankingi Skuteczności":
        # --- Prefiks dla kluczy w tym widoku: rank_ ---
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
                 data_2pt = calc_data[calc_data['SHOT_TYPE'] == '2PT Field Goal']
                 attempts_2pt_league = len(data_2pt)
                 if attempts_2pt_league > 0:
                     made_2pt_league = data_2pt['SHOT_MADE_FLAG'].sum()
                     league_avg_2pt = (made_2pt_league / attempts_2pt_league) * 100
                 data_3pt = calc_data[calc_data['SHOT_TYPE'] == '3PT Field Goal']
                 attempts_3pt_league = len(data_3pt)
                 if attempts_3pt_league > 0:
                     made_3pt_league = data_3pt['SHOT_MADE_FLAG'].sum()
                     league_avg_3pt = (made_3pt_league / attempts_3pt_league) * 100
            else: st.caption("Brak ważnych danych do obliczenia średnich ligowych.")
        col_avg1, col_avg2 = st.columns(2)
        with col_avg1:
            st.metric(label=f"Średnia Skuteczność 2PT (Liga)", value=f"{league_avg_2pt:.1f}%" if isinstance(league_avg_2pt, (float, int)) else "Brak danych", help=f"Obliczono na podstawie {attempts_2pt_league:,} rzutów ({made_2pt_league:,} trafionych) w {selected_season_type}.".replace(',', ' '))
        with col_avg2:
            st.metric(label=f"Średnia Skuteczność 3PT (Liga)", value=f"{league_avg_3pt:.1f}%" if isinstance(league_avg_3pt, (float, int)) else "Brak danych", help=f"Obliczono na podstawie {attempts_3pt_league:,} rzutów ({made_3pt_league:,} trafionych) w {selected_season_type}.".replace(',', ' '))
        st.markdown("---")
        # --- Koniec Sekcji Średnich Ligowych ---


        st.subheader("Ogólne Rozkłady Rzutów")
        st.caption(f"Dane dla: {selected_season_type}")

        # --- Definicja Wyboru Typu Akcji PRZED Kolumnami ---
        action_type_col_rank = 'ACTION_TYPE'
        action_choice = 'Oryginalne'
        if 'ACTION_TYPE_SIMPLE' in filtered_data.columns:
             action_choice_options = ('Oryginalne', 'Uproszczone')
             # Domyślnie wybierz 'Uproszczone', jeśli istnieje
             default_action_index = 1
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
                    fig_type = px.pie(shot_type_counts, names='SHOT_TYPE', values='count')
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
                                             labels={'count':'Liczba Rzutów', action_type_col_rank:''})
                    fig_action_freq.update_layout(yaxis={'categoryorder':'total ascending'}, height=350, margin=dict(t=20, b=0, l=0, r=0))
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


        # --- Reszta zakładki Rankingi Skuteczności (Mapa częstotliwości, Rankingi graczy/zespołów) ---
        st.markdown("---")
        st.subheader(f"Mapa Częstotliwości Rzutów")
        st.caption(f"Gęstość rzutów dla: {selected_season_type}")
        num_bins_freq = st.slider("Dokładność mapy (X):", 20, 80, 50, 5, key='rank_frequency_map_bins')
        nbins_y_freq = int(num_bins_freq * 1.0)
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

    # --- Pozostałe zakładki (elif ...) ---
    # Kod dla Analiza Gracza, Porównanie Graczy, Analiza Zespołowa,
    # Ocena Modelu (KNN), Ocena Modelu (XGBoost), Porównanie Modeli
    # pozostaje taki sam jak w poprzedniej pełnej odpowiedzi.
    # (Fragmenty kodu dla tych zakładek zostały pominięte tutaj dla zwięzłości,
    # ale są one obecne w kodzie z poprzedniej odpowiedzi, który zawierał już
    # integrację ACTION_TYPE_SIMPLE i poprawki w modelowaniu)

    # Przykład dla Analiza Gracza (bez zmian w stosunku do poprzedniej pełnej wersji)
    elif st.session_state.active_view == "⛹️ Analiza Gracza":
         # ... (cały kod dla Analizy Gracza - jak poprzednio) ...
         pass # Placeholder - wklej tutaj kod tej zakładki z poprzedniej odpowiedzi

    elif st.session_state.active_view == "🆚 Porównanie Graczy":
         # ... (cały kod dla Porównania Graczy - jak poprzednio) ...
         pass # Placeholder

    elif st.session_state.active_view == "🏀 Analiza Zespołowa":
        # ... (cały kod dla Analizy Zespołowej - jak poprzednio) ...
         pass # Placeholder

    elif st.session_state.active_view == "🎯 Ocena Modelu (KNN)":
         # ... (cały kod dla Oceny KNN - jak poprzednio, z ACTION_TYPE_SIMPLE) ...
         pass # Placeholder

    elif st.session_state.active_view == "🎯 Ocena Modelu (XGBoost)":
        # ... (cały kod dla Oceny XGBoost - jak poprzednio, z ACTION_TYPE_SIMPLE i SHAP) ...
         pass # Placeholder

    elif st.session_state.active_view == "📊 Porównanie Modeli (KNN vs XGBoost)":
        # ... (cały kod dla Porównania Modeli - jak poprzednio, z ACTION_TYPE_SIMPLE) ...
         pass # Placeholder


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
    # Użyj bieżącej daty (ponieważ to przyszłość względem daty kompilacji)
    ts = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z')
except ImportError:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S (czas lokalny - brak pytz)')
    print("Ostrzeżenie: Biblioteka 'pytz' nie jest zainstalowana.")
except Exception as e_time:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S (czas lokalny - błąd strefy)')
    print(f"Błąd podczas ustawiania strefy czasowej: {e_time}")
st.sidebar.markdown(f"Czas serwera: {ts}")