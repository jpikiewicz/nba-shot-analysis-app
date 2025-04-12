# nba_app_v2.py
# Final version incorporating all requested modifications as of 2025-04-07

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go # Added import for graph_objects
# Usunięto mylące komentarze z poniższych importów sklearn, ponieważ model KNN JEST zaimplementowany
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# --- Konfiguracja Początkowa ---
st.set_page_config(
    layout="wide",
    page_title="Analiza Rzutów NBA 2023-24"
    # UPEWNIJ SIĘ, ŻE TUTAJ NIE MA ŻADNYCH LINII Z primaryColor, backgroundColor itp.
)
# --- KONIEC Konfiguracji Początkowej ---

# --- Ścieżka do pliku CSV ---
# !!! WAŻNE: Zmień tę ścieżkę, jeśli Twój plik CSV ma inną nazwę lub lokalizację !!!
CSV_FILE_PATH = 'nba_player_shooting_data_2023_24.csv' # Przykład nazwy - dostosuj!

# --- Funkcje Pomocnicze ---

@st.cache_data
def load_shooting_data(file_path):
    """Wczytuje i wstępnie przetwarza dane o rzutach graczy NBA."""
    try:
        data = pd.read_csv(file_path, parse_dates=['GAME_DATE'])
        st.success(f"Wczytano dane z {file_path}. Wymiary: {data.shape}")

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
            # Komunikat o zmianie nazw kolumn został ukryty (zakomentowany)
            # st.info(f"Zmieniam nazwy kolumn: {rename_map}")
            data = data.rename(columns=rename_map)

        if 'SHOT_MADE_FLAG' in data.columns:
            if data['SHOT_MADE_FLAG'].dtype == 'object':
                made_values = ['Made', 'made', '1', 1]
                data['SHOT_MADE_FLAG'] = data['SHOT_MADE_FLAG'].apply(lambda x: 1 if x in made_values else 0).astype(int)
            elif pd.api.types.is_numeric_dtype(data['SHOT_MADE_FLAG']):
                 data['SHOT_MADE_FLAG'] = data['SHOT_MADE_FLAG'].astype(int)
            else:
                st.warning(f"Nieznany typ danych dla 'SHOT_MADE_FLAG': {data['SHOT_MADE_FLAG'].dtype}.")
                data['SHOT_MADE_FLAG'] = pd.to_numeric(data['SHOT_MADE_FLAG'], errors='coerce')

        time_cols = ['PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING']
        missing_time_cols = False
        for col in time_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                if data[col].isnull().any():
                        st.warning(f"W kolumnie '{col}' znaleziono wartości nienumeryczne. Zostały one zastąpione przez NaN.")
            else:
                    missing_time_cols = True

        if not missing_time_cols and all(pd.api.types.is_numeric_dtype(data[col]) for col in time_cols if col in data.columns):
            if 'MINUTES_REMAINING' in data.columns and 'SECONDS_REMAINING' in data.columns:
                 data['GAME_TIME_SEC'] = data['MINUTES_REMAINING'] * 60 + data['SECONDS_REMAINING']
            if 'PERIOD' in data.columns:
                 data['QUARTER_TYPE'] = data['PERIOD'].apply(lambda x: 'Dogrywka' if pd.notna(x) and x > 4 else 'Regularna')
        elif any(col not in data.columns for col in ['PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING']):
             st.warning("Brak jednej lub więcej kolumn czasowych (PERIOD, MINUTES_REMAINING, SECONDS_REMAINING). Nie można utworzyć 'GAME_TIME_SEC' ani 'QUARTER_TYPE'.")
        else:
            st.warning("Nie można utworzyć kolumn pomocniczych 'GAME_TIME_SEC' lub 'QUARTER_TYPE' z powodu błędów w kolumnach czasowych.")

        key_cols = ['PLAYER_NAME', 'TEAM_NAME', 'LOC_X', 'LOC_Y', 'SHOT_DISTANCE', 'SEASON_TYPE',
                    'ACTION_TYPE', 'SHOT_TYPE', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE']
        for col in key_cols:
            if col not in data.columns:
                st.warning(f"Brak oczekiwanej kolumny: '{col}'. Niektóre analizy mogą być niedostępne.")

        if 'SHOT_MADE_FLAG' in data.columns:
                data['SHOT_MADE_FLAG'] = pd.to_numeric(data['SHOT_MADE_FLAG'], errors='coerce')

        return data

    except FileNotFoundError:
        st.error(f"Błąd: Nie znaleziono pliku {file_path}.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Błąd podczas wczytywania lub przetwarzania danych: {e}")
        return pd.DataFrame()

# --- Funkcje Wizualizacyjne i Analityczne ---

def add_court_shapes(fig):
    """Dodaje kształty boiska do figury Plotly."""
    fig.add_shape(type="rect", x0=-250, y0=-47.5, x1=250, y1=422.5, line=dict(color="black", width=1))
    fig.add_shape(type="circle", x0=0-7.5, y0=-47.5 + 52.5 - 7.5, x1=0+7.5, y1=-47.5 + 52.5 + 7.5, line=dict(color="orange", width=2), fillcolor="orange", layer='above')
    fig.add_shape(type="line", x0=-30, y0=-47.5+5.25, x1=30, y1=-47.5+5.25, line=dict(color="black", width=1))
    fig.add_shape(type="rect", x0=-80, y0=-47.5, x1=80, y1=142.5, line=dict(color="black", width=1))
    fig.add_shape(type="path", path=f"M -40 -47.5 A 40 40 0 0 1 40 -47.5", line=dict(color="black", width=1))
    fig.add_shape(type="circle", x0=-60, y0=142.5-60, x1=60, y1=142.5+60, line=dict(color="black", width=1))
    fig.add_shape(type="line", x0=-220, y0=-47.5, x1=-220, y1=92.5, line=dict(color="black", width=1))
    fig.add_shape(type="line", x0=220, y0=-47.5, x1=220, y1=92.5, line=dict(color="black", width=1))
    fig.add_shape(type="path", path=f"M -220 92.5 C -135 300, 135 300, 220 92.5", line=dict(color="black", width=1))
    fig.update_xaxes(range=[-300, 300])
    fig.update_yaxes(range=[-100, 500])
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

@st.cache_data
def get_basic_stats(entity_data, entity_name, entity_type="Gracz"):
    """Oblicza podstawowe statystyki (ogółem) dla gracza lub drużyny."""
    stats = {'total_shots': len(entity_data), 'made_shots': "N/A", 'shooting_pct': "N/A"}
    if 'SHOT_MADE_FLAG' in entity_data.columns and stats['total_shots'] > 0:
        numeric_shots = pd.to_numeric(entity_data['SHOT_MADE_FLAG'], errors='coerce').dropna()
        if not numeric_shots.empty:
            stats['made_shots'] = int(numeric_shots.sum())
            total_valid_shots = len(numeric_shots)
            if total_valid_shots > 0: stats['shooting_pct'] = (stats['made_shots'] / total_valid_shots) * 100
            else: stats['made_shots'], stats['shooting_pct'] = 0, 0.0
        else: stats['made_shots'], stats['shooting_pct'] = 0, "N/A"
    elif stats['total_shots'] == 0: stats['made_shots'], stats['shooting_pct'] = 0, "N/A"
    return stats

@st.cache_data
def plot_player_eff_vs_distance(player_data, player_name, bin_width=1, min_attempts_per_bin=5):
    """Tworzy wykres liniowy skuteczności gracza (FG%) w zależności od odległości rzutu (binowanie)."""
    required_cols = ['SHOT_DISTANCE', 'SHOT_MADE_FLAG']
    if not all(col in player_data.columns for col in required_cols):
        st.warning("Brak wymaganych kolumn (SHOT_DISTANCE, SHOT_MADE_FLAG) do analizy skuteczności wg odległości.")
        return None
    player_data['SHOT_MADE_FLAG'] = pd.to_numeric(player_data['SHOT_MADE_FLAG'], errors='coerce')
    distance_data = player_data.dropna(subset=required_cols)
    if distance_data.empty:
        st.info("Brak kompletnych danych (Odległość, Wynik) do analizy skuteczności wg odległości.")
        return None
    if not pd.api.types.is_numeric_dtype(distance_data['SHOT_DISTANCE']):
         st.warning("Kolumna SHOT_DISTANCE nie jest typu numerycznego.")
         return None
    max_dist = distance_data['SHOT_DISTANCE'].max()
    if pd.isna(max_dist) or max_dist <= 0:
        st.info("Nieprawidłowe lub brakujące dane odległości.")
        return None
    distance_bins = np.arange(0, int(max_dist) + bin_width * 2, bin_width)
    bin_labels = [f"{b + bin_width/2:.1f}" for b in distance_bins[:-1]]
    if len(bin_labels) != len(distance_bins) - 1:
         st.error("Problem z tworzeniem etykiet dla binów odległości.")
         return None
    try:
        distance_data['distance_bin_label'] = pd.cut(distance_data['SHOT_DISTANCE'], bins=distance_bins, labels=bin_labels, right=False, include_lowest=True)
    except ValueError as e:
         st.error(f"Błąd podczas tworzenia binów odległości: {e}. Sprawdź zakres danych.")
         return None
    effectiveness = distance_data.groupby('distance_bin_label', observed=False)['SHOT_MADE_FLAG'].agg(Made='sum', Attempts='count').reset_index()
    effectiveness = effectiveness[effectiveness['Attempts'] >= min_attempts_per_bin]
    if effectiveness.empty:
        st.info(f"Brak wystarczających danych (min. {min_attempts_per_bin} prób na {bin_width}-stopowy przedział odległości).")
        return None
    effectiveness['FG%'] = (effectiveness['Made'] / effectiveness['Attempts']) * 100
    effectiveness['distance_mid'] = pd.to_numeric(effectiveness['distance_bin_label'], errors='coerce')
    effectiveness = effectiveness.dropna(subset=['distance_mid']).sort_values(by='distance_mid')
    if effectiveness.empty:
         st.info("Problem z danymi po obliczeniu skuteczności i środka przedziału.")
         return None
    fig = px.line(effectiveness, x='distance_mid', y='FG%', title=f'Wpływ odległości rzutu na skuteczność - {player_name}',
                  labels={'distance_mid': 'Odległość rzutu (stopy)', 'FG%': 'Skuteczność (%)'}, markers=True, hover_data=['Attempts', 'Made'])
    fig.update_layout(yaxis_range=[-5, 105], xaxis_title='Odległość rzutu (stopy)', yaxis_title='Skuteczność (%)', hovermode="x unified")
    fig.update_traces(connectgaps=False)
    return fig

@st.cache_data
def plot_shot_chart(entity_data, entity_name, entity_type="Gracz"):
    """Tworzy interaktywną mapę rzutów."""
    required_cols = ['LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']
    if not all(col in entity_data.columns for col in required_cols): return None
    plot_data = entity_data.dropna(subset=required_cols)
    if plot_data.empty: return None
    plot_data['Wynik Rzutu'] = plot_data['SHOT_MADE_FLAG'].map({0: 'Niecelny', 1: 'Celny'})
    color_col, color_map, cat_orders = 'Wynik Rzutu', {'Niecelny': 'red', 'Celny': 'green'}, {"Wynik Rzutu": ['Niecelny', 'Celny']}
    hover_cols = [col for col in ['SHOT_DISTANCE', 'SHOT_TYPE', 'ACTION_TYPE', 'SHOT_ZONE_BASIC', 'PERIOD'] if col in plot_data.columns]
    fig = px.scatter(plot_data, x='LOC_X', y='LOC_Y', color=color_col, title=f'Mapa rzutów - {entity_name} ({entity_type})',
                     labels={'LOC_X': 'Pozycja X', 'LOC_Y': 'Pozycja Y', 'Wynik Rzutu': 'Wynik'}, hover_data=hover_cols if hover_cols else None,
                     category_orders=cat_orders, color_discrete_map=color_map, opacity=0.7)
    fig = add_court_shapes(fig)
    fig.update_layout(height=600)
    return fig

@st.cache_data
def calculate_hot_zones(entity_data, min_shots_in_zone=5):
    """Oblicza statystyki dla stref rzutowych."""
    required_cols = ['LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']
    if not all(col in entity_data.columns for col in required_cols): return pd.DataFrame()
    zone_data = entity_data.dropna(subset=required_cols)
    if zone_data.empty: return pd.DataFrame()
    x_bins, y_bins = np.linspace(-300, 300, 11), np.linspace(-50, 450, 11)
    zone_data['zone_x'], zone_data['zone_y'] = pd.cut(zone_data['LOC_X'], bins=x_bins), pd.cut(zone_data['LOC_Y'], bins=y_bins)
    zones = zone_data.groupby(['zone_x', 'zone_y'], observed=False).agg(total_shots=('SHOT_MADE_FLAG', 'count'), made_shots=('SHOT_MADE_FLAG', 'sum'), percentage=('SHOT_MADE_FLAG', 'mean')).reset_index()
    zones = zones[zones['total_shots'] >= min_shots_in_zone].copy()
    if zones.empty: return pd.DataFrame()
    zones['percentage'] *= 100
    zones['x_center'], zones['y_center'] = zones['zone_x'].apply(lambda x: x.mid if isinstance(x, pd.Interval) else None), zones['zone_y'].apply(lambda x: x.mid if isinstance(x, pd.Interval) else None)
    return zones.dropna(subset=['x_center', 'y_center'])

@st.cache_data
def plot_hot_zones_heatmap(hot_zones_df, entity_name, entity_type="Gracz"):
    """Tworzy interaktywną mapę ciepła stref rzutowych (skuteczność)."""
    required_cols = ['x_center', 'y_center', 'total_shots', 'percentage', 'made_shots']
    if hot_zones_df.empty or not all(col in hot_zones_df.columns for col in required_cols): return None
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(hot_zones_df[col]):
            try: hot_zones_df[col] = pd.to_numeric(hot_zones_df[col])
            except ValueError: st.warning(f"Problem z konwersją '{col}' w danych hot zone."); return None
    hot_zones_df = hot_zones_df.dropna(subset=required_cols)
    if hot_zones_df.empty: return None
    min_shots, min_pct, max_pct = int(hot_zones_df["total_shots"].min()), hot_zones_df['percentage'].min(), hot_zones_df['percentage'].max()
    if pd.isna(min_pct): min_pct = 0
    if pd.isna(max_pct): max_pct = 100
    color_range = [max(0, min_pct - 5), min(100, max_pct + 5)]
    if color_range[0] >= color_range[1]: color_range = [0, 100]
    fig = px.scatter(hot_zones_df, x='x_center', y='y_center', size='total_shots', color='percentage', color_continuous_scale=px.colors.diverging.RdYlGn,
                     size_max=40, range_color=color_range, title=f'Skuteczność stref rzutowych ({entity_type}: {entity_name}, min. {min_shots} rzutów)',
                     labels={'x_center': 'Pozycja X', 'y_center': 'Pozycja Y', 'total_shots': 'Liczba rzutów', 'percentage': 'Skuteczność (%)'},
                     custom_data=['made_shots', 'total_shots'])
    fig.update_traces(hovertemplate="<b>Strefa X:</b> %{x:.1f}, <b>Y:</b> %{y:.1f}<br><b>Liczba rzutów:</b> %{customdata[1]}<br><b>Trafione:</b> %{customdata[0]}<br><b>Skuteczność:</b> %{marker.color:.1f}%<extra></extra>")
    fig = add_court_shapes(fig)
    fig.update_layout(height=600)
    return fig

@st.cache_data
def plot_shot_frequency_heatmap(data, season_name, nbins_x=50, nbins_y=50):
    """Tworzy heatmapę częstotliwości rzutów na boisku (Histogram2d)."""
    required_cols = ['LOC_X', 'LOC_Y']
    if not all(col in data.columns for col in required_cols):
        st.warning("Brak wymaganych kolumn (LOC_X, LOC_Y) do mapy częstotliwości.")
        return None
    plot_data = data.dropna(subset=required_cols)
    if plot_data.empty:
        st.info("Brak danych lokalizacji (LOC_X, LOC_Y) do stworzenia mapy częstotliwości.")
        return None
    fig = go.Figure()
    fig.add_trace(go.Histogram2d(x=plot_data['LOC_X'], y=plot_data['LOC_Y'], colorscale='YlOrRd', nbinsx=nbins_x, nbinsy=nbins_y, zauto=True,
                                 hovertemplate='<b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Liczba rzutów:</b> %{z}<extra></extra>'))
    fig = add_court_shapes(fig)
    fig.update_layout(title=f'Mapa Częstotliwości Rzutów ({season_name})', xaxis_title="Pozycja X", yaxis_title="Pozycja Y", height=650,
                      xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='rgba(255, 255, 255, 1)')
    fig.update_xaxes(range=[-300, 300])
    fig.update_yaxes(range=[-100, 500])
    return fig

@st.cache_data
def plot_player_quarter_eff(entity_data, entity_name, entity_type="Gracz"):
    """Wykres skuteczności w poszczególnych kwartach/dogrywkach."""
    if 'PERIOD' not in entity_data.columns or 'SHOT_MADE_FLAG' not in entity_data.columns: return None
    quarter_data = entity_data.dropna(subset=['PERIOD', 'SHOT_MADE_FLAG'])
    if quarter_data.empty: return None
    try: quarter_data['PERIOD'] = quarter_data['PERIOD'].astype(int)
    except ValueError: st.warning("Problem z konwersją 'PERIOD' na int."); return None
    quarter_eff = quarter_data.groupby('PERIOD')['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
    quarter_eff['mean'] *= 100
    quarter_eff = quarter_eff[quarter_eff['count'] >= 5]
    if quarter_eff.empty: return None
    def map_period(p):
        if p <= 4: return f"Kwarta {int(p)}"
        elif p == 5: return "OT 1"
        else: return f"OT {int(p-4)}"
    quarter_eff['Okres Gry'] = quarter_eff['PERIOD'].apply(map_period)
    quarter_eff = quarter_eff.sort_values(by='PERIOD')
    fig = px.bar(quarter_eff, x='Okres Gry', y='mean', text='mean', title=f'Skuteczność w kwartach/dogrywkach - {entity_name} ({entity_type})',
                 labels={'Okres Gry': 'Okres Gry', 'mean': 'Skuteczność (%)'}, hover_data=['count'])
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(yaxis_range=[0, 100], uniformtext_minsize=8, uniformtext_mode='hide')
    return fig

@st.cache_data
def plot_player_season_trend(entity_data, entity_name, entity_type="Gracz"):
    """Wykres trendu skuteczności w trakcie sezonu (miesięcznie)."""
    if 'GAME_DATE' not in entity_data.columns or 'SHOT_MADE_FLAG' not in entity_data.columns: return None
    trend_data = entity_data.copy()
    trend_data['GAME_DATE'] = pd.to_datetime(trend_data['GAME_DATE'], errors='coerce')
    trend_data = trend_data.dropna(subset=['GAME_DATE', 'SHOT_MADE_FLAG'])
    if trend_data.empty or len(trend_data) < 10: return None
    trend_data = trend_data.set_index('GAME_DATE')
    monthly_eff = trend_data.resample('ME')['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
    monthly_eff['mean'] *= 100
    monthly_eff = monthly_eff[monthly_eff['count'] >= 10]
    if monthly_eff.empty or len(monthly_eff) < 2: return None
    monthly_eff['Miesiąc'] = monthly_eff['GAME_DATE'].dt.strftime('%Y-%m')
    fig = px.line(monthly_eff, x='Miesiąc', y='mean', markers=True, title=f'Miesięczny trend skuteczności - {entity_name} ({entity_type})',
                  labels={'Miesiąc': 'Miesiąc', 'mean': 'Skuteczność (%)'}, hover_data=['count'])
    max_y = 100
    if not pd.isna(monthly_eff['mean'].max()): max_y = min(100, monthly_eff['mean'].max() * 1.1 + 5)
    fig.update_layout(yaxis_range=[0, max_y])
    return fig

@st.cache_data
def plot_grouped_effectiveness(entity_data, group_col, entity_name, entity_type="Gracz", top_n=10):
    """Tworzy wykres skuteczności pogrupowany wg wybranej kolumny."""
    if group_col not in entity_data.columns or 'SHOT_MADE_FLAG' not in entity_data.columns: return None
    grouped_data = entity_data.dropna(subset=[group_col, 'SHOT_MADE_FLAG'])
    if grouped_data.empty: return None
    grouped_eff = grouped_data.groupby(group_col)['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
    grouped_eff['mean'] *= 100
    grouped_eff = grouped_eff[grouped_eff['count'] >= 5]
    grouped_eff = grouped_eff.sort_values(by='count', ascending=False).head(top_n)
    if grouped_eff.empty: return None
    axis_label = group_col.replace('_',' ').title()
    fig = px.bar(grouped_eff, x=group_col, y='mean', text='mean', title=f'Skuteczność wg {axis_label} - {entity_name} ({entity_type}) (Top {top_n} najczęstszych)',
                 labels={group_col: axis_label, 'mean': 'Skuteczność (%)'}, hover_data=['count'])
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(yaxis_range=[0, 100], uniformtext_minsize=8, uniformtext_mode='hide', xaxis_title=axis_label, xaxis={'categoryorder':'total descending'})
    return fig

@st.cache_data
def plot_comparison_eff_distance(compare_data, selected_players):
    """Porównuje skuteczność graczy względem odległości (linowy)."""
    required_cols = ['SHOT_DISTANCE', 'SHOT_MADE_FLAG', 'PLAYER_NAME']
    if not all(col in compare_data.columns for col in required_cols): return None
    compare_data_eff = compare_data.dropna(subset=['SHOT_MADE_FLAG', 'SHOT_DISTANCE'])
    if compare_data_eff.empty: return None
    if not pd.api.types.is_numeric_dtype(compare_data_eff['SHOT_DISTANCE']): st.warning("SHOT_DISTANCE nie jest numeryczny w danych porównawczych."); return None
    max_dist = compare_data_eff['SHOT_DISTANCE'].max()
    if pd.isna(max_dist) or max_dist <= 0: return None
    bin_width = 3
    distance_bins = np.arange(0, int(max_dist) + bin_width, bin_width)
    compare_data_eff['distance_bin'] = pd.cut(compare_data_eff['SHOT_DISTANCE'], bins=distance_bins, right=False, include_lowest=True)
    effectiveness = compare_data_eff.groupby(['PLAYER_NAME', 'distance_bin'], observed=False)['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
    effectiveness['mean'] *= 100
    min_shots_per_bin = 5
    effectiveness = effectiveness[effectiveness['count'] >= min_shots_per_bin]
    effectiveness['distance_mid'] = effectiveness['distance_bin'].apply(lambda x: x.mid if isinstance(x, pd.Interval) else None)
    effectiveness = effectiveness.dropna(subset=['distance_mid'])
    if not effectiveness.empty:
        max_eff_val = effectiveness['mean'].max()
        yaxis_range = [0, min(100, max_eff_val + 10 if not pd.isna(max_eff_val) else 100)]
        fig = px.line(effectiveness, x='distance_mid', y='mean', color='PLAYER_NAME', title=f'Porównanie skuteczności vs odległości (min. {min_shots_per_bin} prób w przedziale)',
                      labels={'distance_mid': 'Odległość (stopy)', 'mean': 'Skuteczność (%)', 'PLAYER_NAME': 'Gracz'}, markers=True, hover_data=['count'])
        fig.update_layout(yaxis_range=yaxis_range)
        return fig
    return None

@st.cache_data
def plot_comparison_eff_by_zone(compare_data, selected_players, min_shots_per_zone=5):
    """Tworzy grupowany wykres słupkowy porównujący skuteczność graczy wg SHOT_ZONE_BASIC."""
    required_cols = ['PLAYER_NAME', 'SHOT_MADE_FLAG', 'SHOT_ZONE_BASIC']
    if not all(col in compare_data.columns for col in required_cols): st.warning(f"Brak kolumn do porównania wg stref: {required_cols}"); return None
    compare_data['SHOT_MADE_FLAG'] = pd.to_numeric(compare_data['SHOT_MADE_FLAG'], errors='coerce')
    zone_eff_data = compare_data.dropna(subset=required_cols)
    if zone_eff_data.empty: st.info("Brak kompletnych danych do analizy skuteczności wg stref."); return None
    zone_stats = zone_eff_data.groupby(['PLAYER_NAME', 'SHOT_ZONE_BASIC'], observed=False)['SHOT_MADE_FLAG'].agg(Made='sum', Attempts='count').reset_index()
    zone_stats_filtered = zone_stats[zone_stats['Attempts'] >= min_shots_per_zone]
    if zone_stats_filtered.empty: st.info(f"Brak wystarczających danych (min. {min_shots_per_zone} prób na strefę) do porównania skuteczności wg stref."); return None
    zone_stats_filtered['FG%'] = (zone_stats_filtered['Made'] / zone_stats_filtered['Attempts']) * 100
    # !!! DOSTOSUJ TĘ LISTĘ DO NAZW STREF W TWOIM PLIKU !!!
    zone_order = ['Restricted Area', 'In The Paint (Non-RA)', 'Mid-Range', 'Left Corner 3', 'Right Corner 3', 'Above the Break 3', 'Backcourt']
    zone_stats_plot = zone_stats_filtered[zone_stats_filtered['SHOT_ZONE_BASIC'].isin(zone_order)].copy()
    if zone_stats_plot.empty: st.info("Brak danych dla predefiniowanych stref po filtracji."); return None
    fig = px.bar(zone_stats_plot, x='SHOT_ZONE_BASIC', y='FG%', color='PLAYER_NAME', barmode='group', title=f'Porównanie skuteczności (FG%) wg Strefy Rzutowej (min. {min_shots_per_zone} prób)',
                 labels={'SHOT_ZONE_BASIC': 'Strefa Rzutowa', 'FG%': 'Skuteczność (%)', 'PLAYER_NAME': 'Gracz'}, hover_data=['Attempts', 'Made'], category_orders={'SHOT_ZONE_BASIC': zone_order}, text='FG%')
    fig.update_layout(yaxis_range=[0, 100], xaxis={'categoryorder':'array', 'categoryarray':zone_order}, legend_title_text='Gracze')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    return fig

@st.cache_data
def calculate_top_performers(data, group_by_col, min_total_shots, min_2pt_shots, min_3pt_shots, top_n=10):
    """Oblicza rankingi Top N graczy/zespołów wg skuteczności."""
    # ... (kod funkcji bez zmian - skrócono dla czytelności) ...
    if group_by_col not in data.columns or 'SHOT_MADE_FLAG' not in data.columns: return None, None, None
    valid_data = data.dropna(subset=[group_by_col, 'SHOT_MADE_FLAG'])
    if valid_data.empty: return None, None, None
    overall_stats = valid_data.groupby(group_by_col)['SHOT_MADE_FLAG'].agg(Made='sum', Attempts='count').reset_index()
    overall_stats = overall_stats[overall_stats['Attempts'] >= min_total_shots]
    top_overall = pd.DataFrame()
    if not overall_stats.empty:
        overall_stats['FG%'] = (overall_stats['Made'] / overall_stats['Attempts']) * 100
        top_overall = overall_stats.sort_values(by='FG%', ascending=False).head(top_n)
        col_name = group_by_col.replace('_',' ').title()
        top_overall = top_overall.rename(columns={group_by_col: col_name, 'Attempts': 'Próby'})
        top_overall = top_overall[[col_name, 'FG%', 'Próby']]
    shot_type_2pt = '2PT Field Goal'
    top_2pt = pd.DataFrame()
    if 'SHOT_TYPE' in valid_data.columns:
        if shot_type_2pt in valid_data['SHOT_TYPE'].unique():
            data_2pt = valid_data[valid_data['SHOT_TYPE'] == shot_type_2pt]
            stats_2pt = data_2pt.groupby(group_by_col)['SHOT_MADE_FLAG'].agg(Made_2PT='sum', Attempts_2PT='count').reset_index()
            stats_2pt = stats_2pt[stats_2pt['Attempts_2PT'] >= min_2pt_shots]
            if not stats_2pt.empty:
                stats_2pt['2PT FG%'] = (stats_2pt['Made_2PT'] / stats_2pt['Attempts_2PT']) * 100
                top_2pt = stats_2pt.sort_values(by='2PT FG%', ascending=False).head(top_n)
                col_name = group_by_col.replace('_',' ').title()
                top_2pt = top_2pt.rename(columns={group_by_col: col_name, 'Attempts_2PT': 'Próby 2PT'})
                top_2pt = top_2pt[[col_name, '2PT FG%', 'Próby 2PT']]
        else: st.caption(f"Nie znaleziono '{shot_type_2pt}' w SHOT_TYPE dla rankingu 2PT.")
    else: st.caption("Brak 'SHOT_TYPE' dla rankingu 2PT.")
    shot_type_3pt = '3PT Field Goal'
    top_3pt = pd.DataFrame()
    if 'SHOT_TYPE' in valid_data.columns:
        if shot_type_3pt in valid_data['SHOT_TYPE'].unique():
            data_3pt = valid_data[valid_data['SHOT_TYPE'] == shot_type_3pt]
            stats_3pt = data_3pt.groupby(group_by_col)['SHOT_MADE_FLAG'].agg(Made_3PT='sum', Attempts_3PT='count').reset_index()
            stats_3pt = stats_3pt[stats_3pt['Attempts_3PT'] >= min_3pt_shots]
            if not stats_3pt.empty:
                stats_3pt['3PT FG%'] = (stats_3pt['Made_3PT'] / stats_3pt['Attempts_3PT']) * 100
                top_3pt = stats_3pt.sort_values(by='3PT FG%', ascending=False).head(top_n)
                col_name = group_by_col.replace('_',' ').title()
                top_3pt = top_3pt.rename(columns={group_by_col: col_name, 'Attempts_3PT': 'Próby 3PT'})
                top_3pt = top_3pt[[col_name, '3PT FG%', 'Próby 3PT']]
        else: st.caption(f"Nie znaleziono '{shot_type_3pt}' w SHOT_TYPE dla rankingu 3PT.")
    else: st.caption("Brak 'SHOT_TYPE' dla rankingu 3PT.")
    return top_overall, top_2pt, top_3pt

# --- Główna część aplikacji Streamlit ---
st.title("🏀 Interaktywna Analiza Rzutów Graczy NBA (Sezon 2023-24)")

shooting_data = load_shooting_data(CSV_FILE_PATH)

st.sidebar.header("Opcje Filtrowania i Analizy")

if not shooting_data.empty:
    # Sidebar Filters
    available_season_types = ['Wszystko'] + shooting_data['SEASON_TYPE'].dropna().unique().tolist() if 'SEASON_TYPE' in shooting_data.columns else ['Wszystko']
    # Domyślna wartość dla typu sezonu to 'Wszystko', co odpowiada index=0 (bez zmian)
    selected_season_type = st.sidebar.selectbox(
        "Wybierz typ sezonu:",
        options=available_season_types,
        index=0, # 'Wszystko' jest zawsze na pierwszej pozycji
        key='season_select'
        )
    filtered_data = shooting_data[shooting_data['SEASON_TYPE'] == selected_season_type].copy() if selected_season_type != 'Wszystko' else shooting_data.copy()
    st.sidebar.write(f"Wybrano: {selected_season_type} ({len(filtered_data)} rzutów)")

    # --- Ustawienia dla Gracza ---
    available_players = sorted(filtered_data['PLAYER_NAME'].dropna().unique()) if 'PLAYER_NAME' in filtered_data.columns else []
    default_player = "LeBron James"
    default_player_index = None
    if available_players:
        try:
            # Znajdź indeks domyślnego gracza na liście
            default_player_index = available_players.index(default_player)
        except ValueError:
            # Jeśli LeBrona nie ma na liście, wybierz pierwszego gracza
            st.sidebar.warning(f"Domyślny gracz '{default_player}' nie znaleziony w danych. Wybieram pierwszego dostępnego.")
            default_player_index = 0

    selected_player = st.sidebar.selectbox(
        "Gracz do analizy:",
        options=available_players,
        index=default_player_index if available_players else None, # Użyj znalezionego lub zapasowego indeksu
        key='player_select',
        disabled=not available_players
    )

    # --- Ustawienia dla Drużyny ---
    available_teams = sorted(filtered_data['TEAM_NAME'].dropna().unique()) if 'TEAM_NAME' in filtered_data.columns else []
    default_team = "Los Angeles Lakers"
    default_team_index = None
    if available_teams:
        try:
            # Znajdź indeks domyślnej drużyny
            default_team_index = available_teams.index(default_team)
        except ValueError:
            # Jeśli LAL nie ma na liście, wybierz pierwszą drużynę
            st.sidebar.warning(f"Domyślna drużyna '{default_team}' nie znaleziona w danych. Wybieram pierwszą dostępną.")
            default_team_index = 0

    selected_team = st.sidebar.selectbox(
        "Drużyna do analizy:",
        options=available_teams,
        index=default_team_index if available_teams else None, # Użyj znalezionego lub zapasowego indeksu
        key='team_select',
        disabled=not available_teams
    )

    # --- Ustawienia dla Porównania Graczy ---
    default_compare_players_req = ["LeBron James", "Stephen Curry"]
    # Sprawdź, którzy z domyślnych graczy są faktycznie dostępni w danych
    default_compare_players = [p for p in default_compare_players_req if p in available_players]
    if len(default_compare_players) < len(default_compare_players_req):
         missing_defaults = set(default_compare_players_req) - set(default_compare_players)
         st.sidebar.warning(f"Niektórzy domyślni gracze do porównania nie znaleźli się w danych: {', '.join(missing_defaults)}")

    selected_players_compare = st.sidebar.multiselect(
        "Gracze do porównania (2-5):",
        options=available_players,
        default=default_compare_players, # Użyj listy dostępnych domyślnych graczy
        max_selections=5,
        key='player_multi_select',
        disabled=not available_players
    )

    # Tabs Definition
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Rankingi Skuteczności", "⛹️ Analiza Gracza", "🆚 Porównanie Graczy",
        "🏀 Analiza Zespołowa", "🎯 Model Predykcji (KNN)"
    ])

    # --- Tab 1: Rankings ---
    with tab1:
        st.header(f"Rankingi Skuteczności ({selected_season_type})")
        st.markdown("### Ustaw minimalną liczbę prób")
        col_att1, col_att2, col_att3 = st.columns(3)
        with col_att1: min_total = st.number_input("Min. rzutów ogółem:", 10, 1000, 100, 10, key="min_total_r")
        with col_att2: min_2pt = st.number_input("Min. rzutów za 2 pkt:", 5, 500, 50, 5, key="min_2pt_r")
        with col_att3: min_3pt = st.number_input("Min. rzutów za 3 pkt:", 5, 500, 30, 5, key="min_3pt_r")

        tp_ov, tp_2, tp_3 = calculate_top_performers(filtered_data, 'PLAYER_NAME', min_total, min_2pt, min_3pt)
        tt_ov, tt_2, tt_3 = calculate_top_performers(filtered_data, 'TEAM_NAME', min_total*5, min_2pt*5, min_3pt*5)

        st.markdown("---"); st.subheader("Skuteczność Ogółem (FG%)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Top 10 Graczy (min. {min_total} prób)**")
            if tp_ov is not None and not tp_ov.empty: st.dataframe(tp_ov, use_container_width=True, hide_index=True, column_config={"FG%": st.column_config.ProgressColumn("FG%", format="%.1f%%", min_value=0, max_value=100)})
            else: st.caption("Brak graczy.")
        with c2:
            st.markdown(f"**Top 10 Zespołów (min. {min_total*5} prób)**")
            if tt_ov is not None and not tt_ov.empty: st.dataframe(tt_ov, use_container_width=True, hide_index=True, column_config={"FG%": st.column_config.ProgressColumn("FG%", format="%.1f%%", min_value=0, max_value=100)})
            else: st.caption("Brak zespołów.")

        st.markdown("---"); st.subheader("Skuteczność za 2 Punkty (2PT FG%)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Top 10 Graczy (min. {min_2pt} prób)**")
            if tp_2 is not None and not tp_2.empty: st.dataframe(tp_2, use_container_width=True, hide_index=True, column_config={"2PT FG%": st.column_config.ProgressColumn("2PT FG%", format="%.1f%%", min_value=0, max_value=100)})
            else: st.caption("Brak graczy.")
        with c2:
            st.markdown(f"**Top 10 Zespołów (min. {min_2pt*5} prób)**")
            if tt_2 is not None and not tt_2.empty: st.dataframe(tt_2, use_container_width=True, hide_index=True, column_config={"2PT FG%": st.column_config.ProgressColumn("2PT FG%", format="%.1f%%", min_value=0, max_value=100)})
            else: st.caption("Brak zespołów.")

        st.markdown("---"); st.subheader("Skuteczność za 3 Punkty (3PT FG%)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Top 10 Graczy (min. {min_3pt} prób)**")
            if tp_3 is not None and not tp_3.empty: st.dataframe(tp_3, use_container_width=True, hide_index=True, column_config={"3PT FG%": st.column_config.ProgressColumn("3PT FG%", format="%.1f%%", min_value=0, max_value=100)})
            else: st.caption("Brak graczy.")
        with c2:
            st.markdown(f"**Top 10 Zespołów (min. {min_3pt*5} prób)**")
            if tt_3 is not None and not tt_3.empty: st.dataframe(tt_3, use_container_width=True, hide_index=True, column_config={"3PT FG%": st.column_config.ProgressColumn("3PT FG%", format="%.1f%%", min_value=0, max_value=100)})
            else: st.caption("Brak zespołów.")

        # Ogólne Rozkłady
        st.markdown("---"); st.subheader("Ogólne Rozkłady (dane po filtracji sezonu)")
        c1, c2 = st.columns(2)
        with c1:
            if 'SHOT_TYPE' in filtered_data.columns and not filtered_data['SHOT_TYPE'].isnull().all():
                 fig_type = px.pie(filtered_data['SHOT_TYPE'].value_counts().reset_index(), names='SHOT_TYPE', values='count', title='Rozkład Typów Rzutów')
                 st.plotly_chart(fig_type, use_container_width=True)
            else: st.caption("Brak danych dla typów rzutów.")
        with c2:
             if 'ACTION_TYPE' in filtered_data.columns and not filtered_data['ACTION_TYPE'].isnull().all():
                 counts = filtered_data['ACTION_TYPE'].value_counts().head(15).reset_index()
                 fig_action = px.bar(counts, y='ACTION_TYPE', x='count', orientation='h', title='Najczęstsze Typy Akcji (Top 15)', labels={'count':'Liczba Rzutów', 'ACTION_TYPE':''})
                 fig_action.update_layout(yaxis={'categoryorder':'total ascending'}, height=450)
                 st.plotly_chart(fig_action, use_container_width=True)
             else: st.caption("Brak danych dla typów akcji.")

        # MAPA CZĘSTOTLIWOŚCI RZUTÓW (NA DOLE ZAKŁADKI)
        st.markdown("---")
        st.subheader(f"Mapa Częstotliwości Rzutów ({selected_season_type})")
        st.markdown("Mapa pokazuje obszary boiska, z których najczęściej oddawano rzuty. Ciemniejszy/cieplejszy kolor oznacza więcej rzutów.")
        num_bins_freq = st.slider("Dokładność mapy (liczba kwadratów na oś):", 20, 80, 50, 5, key='frequency_map_bins_slider')
        fig_freq_map = plot_shot_frequency_heatmap(filtered_data, selected_season_type, nbins_x=num_bins_freq, nbins_y=int(num_bins_freq * 1.1))
        if fig_freq_map: st.plotly_chart(fig_freq_map, use_container_width=True)
        else: st.caption("Nie udało się wygenerować mapy częstotliwości rzutów.")
        # --- KONIEC MAPY CZĘSTOTLIWOŚCI ---

    # --- Tab 2: Player Analysis (Modified) ---
    with tab2:
        st.header(f"Analiza Gracza: {selected_player}")
        if selected_player:
            player_data = filter_data_by_player(selected_player, filtered_data)
            if not player_data.empty:
                # Stats Section (Top)
                st.subheader("Statystyki Podstawowe")
                total_shots = len(player_data)
                # ... (Calculations for made_shots, shooting_pct, pct_2pt, attempts_2pt_str, pct_3pt, attempts_3pt_str, avg_dist - code shortened for brevity) ...
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
                else: st.warning("Nie można wygenerować mapy rzutów.")
                st.markdown("---")

                st.subheader("Skuteczność vs Odległość")
                cc1, cc2 = st.columns(2)
                with cc1: bin_w = st.slider("Szerokość przedziału (stopy):", 1, 5, 1, key='p_eff_dist_bin')
                with cc2: min_att = st.slider("Min. prób w przedziale:", 1, 50, 10, key='p_eff_dist_min')
                fig_p_eff_dist = plot_player_eff_vs_distance(player_data, selected_player, bin_width=bin_w, min_attempts_per_bin=min_att)
                if fig_p_eff_dist: st.plotly_chart(fig_p_eff_dist, use_container_width=True)
                else: st.caption("Nie udało się wygenerować wykresu skuteczności vs odległość.")
                st.markdown("---")

                st.subheader("Strefy Rzutowe ('Hot Zones')")
                p_hot_zones = calculate_hot_zones(player_data, min_shots_in_zone=5)
                if p_hot_zones is not None and not p_hot_zones.empty:
                    fig_p_hot = plot_hot_zones_heatmap(p_hot_zones, selected_player, "Gracz")
                    if fig_p_hot: st.plotly_chart(fig_p_hot, use_container_width=True)
                    else: st.info("Nie można wygenerować mapy stref.")
                else: st.info("Brak wystarczających danych do analizy stref.")
                st.markdown("---")

                st.subheader("Analiza Czasowa")
                fig_p_q = plot_player_quarter_eff(player_data, selected_player)
                if fig_p_q: st.plotly_chart(fig_p_q, use_container_width=True)
                else: st.info("Brak danych do analizy skuteczności w kwartach.")
                fig_p_t = plot_player_season_trend(player_data, selected_player)
                if fig_p_t: st.plotly_chart(fig_p_t, use_container_width=True)
                else: st.info("Brak danych do analizy trendu sezonowego.")
                st.markdown("---")

                st.subheader("Analiza wg Typu Akcji / Strefy")
                cg1, cg2 = st.columns(2)
                with cg1:
                    fig_p_a = plot_grouped_effectiveness(player_data, 'ACTION_TYPE', selected_player, "Gracz", top_n=10)
                    if fig_p_a: st.plotly_chart(fig_p_a, use_container_width=True)
                    else: st.caption("Brak danych wg typu akcji.")
                with cg2:
                    fig_p_z = plot_grouped_effectiveness(player_data, 'SHOT_ZONE_BASIC', selected_player, "Gracz", top_n=7)
                    if fig_p_z: st.plotly_chart(fig_p_z, use_container_width=True)
                    else: st.caption("Brak danych wg strefy podstawowej.")
            else: st.warning(f"Brak danych dla gracza '{selected_player}'.")
        else: st.info("Wybierz gracza.")

    # --- Tab 3: Player Comparison (Modified) ---
    with tab3:
        st.header("Porównanie Graczy")
        if len(selected_players_compare) >= 2:
            st.write(f"Porównujesz: {', '.join(selected_players_compare)}")
            compare_data_filtered = filtered_data[filtered_data['PLAYER_NAME'].isin(selected_players_compare)].copy()
            if not compare_data_filtered.empty:
                st.subheader("Skuteczność vs Odległość")
                fig_comp_eff_dist = plot_comparison_eff_distance(compare_data_filtered, selected_players_compare)
                if fig_comp_eff_dist: st.plotly_chart(fig_comp_eff_dist, use_container_width=True)
                else: st.caption("Nie można wygenerować wykresu skuteczności vs odległość.")
                st.markdown("---")
                st.subheader("Skuteczność wg Strefy Rzutowej (SHOT_ZONE_BASIC)")
                min_attempts_zone = st.slider("Min. prób w strefie:", 1, 50, 5, key='compare_zone_min_slider')
                fig_comp_zone = plot_comparison_eff_by_zone(compare_data_filtered, selected_players_compare, min_shots_per_zone=min_attempts_zone)
                if fig_comp_zone: st.plotly_chart(fig_comp_zone, use_container_width=True)
                else: st.caption("Nie udało się wygenerować porównania wg stref.")
                st.markdown("---")
                st.subheader("Mapy Rzutów")
                cols = st.columns(len(selected_players_compare))
                for i, player in enumerate(selected_players_compare):
                    with cols[i]:
                        st.markdown(f"**{player}**")
                        player_comp_data = compare_data_filtered[compare_data_filtered['PLAYER_NAME'] == player]
                        if not player_comp_data.empty:
                            fig_comp_chart = plot_shot_chart(player_comp_data, player, "Gracz")
                            if fig_comp_chart: fig_comp_chart.update_layout(height=450, title=""); st.plotly_chart(fig_comp_chart, use_container_width=True, key=f"comp_ch_{player}")
                            else: st.caption("Błąd mapy.")
                        else: st.caption("Brak danych.")
            else: st.warning("Brak danych dla wybranych graczy.")
        else: st.info("Wybierz min. 2 graczy do porównania.")

    # --- Tab 4: Team Analysis ---
    with tab4:
        st.header(f"Analiza Zespołowa: {selected_team}")
        if selected_team:
            team_data = filter_data_by_team(selected_team, filtered_data)
            if not team_data.empty:
                st.subheader("Statystyki Podstawowe Zespołu")
                t_stats = get_basic_stats(team_data, selected_team, "Zespół")
                c1, c2, c3 = st.columns(3)
                c1.metric("Rzuty Zespołu", t_stats['total_shots'])
                c2.metric("Trafione Zespołu", t_stats['made_shots'])
                c3.metric("Skuteczność Zespołu", f"{t_stats['shooting_pct']:.1f}%" if isinstance(t_stats['shooting_pct'], (float, int)) else "N/A")
                st.markdown("---")
                st.subheader("Wizualizacje Rzutów Zespołu")
                fig_t_c = plot_shot_chart(team_data, selected_team, "Zespół")
                if fig_t_c: st.plotly_chart(fig_t_c, use_container_width=True)
                else: st.warning("Nie można wygenerować mapy rzutów zespołu.")
                st.subheader("Skuteczność Zespołu vs Odległość")
                fig_t_ed = plot_player_eff_vs_distance(team_data, selected_team, bin_width=1, min_attempts_per_bin=10)
                if fig_t_ed: st.plotly_chart(fig_t_ed, use_container_width=True)
                else: st.caption("Nie udało się wygenerować wykresu skuteczności vs odległość dla zespołu.")
                st.markdown("---")
                st.subheader("Strefy Rzutowe ('Hot Zones') Zespołu")
                t_hz = calculate_hot_zones(team_data, min_shots_in_zone=10)
                if t_hz is not None and not t_hz.empty:
                    fig_t_h = plot_hot_zones_heatmap(t_hz, selected_team, "Zespół")
                    if fig_t_h: st.plotly_chart(fig_t_h, use_container_width=True)
                    else: st.info("Nie można wygenerować mapy stref zespołu.")
                else: st.info("Brak danych do analizy stref zespołu.")
                st.markdown("---")
                st.subheader("Analiza Czasowa Zespołu")
                fig_t_q = plot_player_quarter_eff(team_data, selected_team, "Zespół")
                if fig_t_q: st.plotly_chart(fig_t_q, use_container_width=True)
                else: st.info("Brak danych do analizy skuteczności zespołu w kwartach.")
                st.markdown("---")
                st.subheader("Analiza Zespołu wg Typu Akcji / Strefy")
                cg1, cg2 = st.columns(2)
                with cg1:
                    fig_t_a = plot_grouped_effectiveness(team_data, 'ACTION_TYPE', selected_team, "Zespół", top_n=10)
                    if fig_t_a: st.plotly_chart(fig_t_a, use_container_width=True)
                    else: st.caption("Brak danych zespołu wg typu akcji.")
                with cg2:
                    fig_t_z = plot_grouped_effectiveness(team_data, 'SHOT_ZONE_BASIC', selected_team, "Zespół", top_n=7)
                    if fig_t_z: st.plotly_chart(fig_t_z, use_container_width=True)
                    else: st.caption("Brak danych zespołu wg strefy podstawowej.")
            else: st.warning(f"Brak danych dla drużyny '{selected_team}'.")
        else: st.info("Wybierz drużynę.")

    # --- Tab 5: KNN Model (Placeholder) ---
    # --- Tab 5: KNN Model (IMPLEMENTED with Interpretation) ---
with tab5:
    st.header(f"Model Predykcji Rzutów (KNN) dla: {selected_player}")

    # TUTAJ ZNAJDUJE SIĘ ZAKTUALIZOWANY TEKST INTERPRETACJI
    st.markdown("""
    ### Interpretacja Wyników Modelu KNN

    Zakładka "Model Predykcji (KNN)" prezentuje prosty model uczenia maszynowego, którego celem jest przewidzenie, czy dany rzut wybranego gracza będzie **celny (1)** czy **niecelny (0)**. Model wykorzystuje algorytm K-Najbliższych Sąsiadów (KNN).

    **Jak działa KNN w tym kontekście?**

    1.  **Dane wejściowe (Cechy):** Model bazuje na trzech podstawowych cechach każdego rzutu:
        * `LOC_X`: Pozycja rzutu na osi X boiska.
        * `LOC_Y`: Pozycja rzutu na osi Y boiska.
        * `SHOT_DISTANCE`: Odległość rzutu od kosza (w stopach).
    2.  **Zmienna docelowa:** Model próbuje przewidzieć wartość kolumny `SHOT_MADE_FLAG` (0 lub 1).
    3.  **Zasada działania:** Kiedy model ma przewidzieć wynik nowego rzutu (np. ze zbioru testowego), znajduje `k` (liczbę sąsiadów wybraną suwakiem) najbardziej podobnych rzutów w danych treningowych, bazując na odległości w przestrzeni cech (uwzględniając LOC_X, LOC_Y i SHOT_DISTANCE po przeskalowaniu). Następnie "głosuje" - jeśli większość z `k` najbliższych sąsiadów to rzuty celne, model przewiduje rzut jako celny; w przeciwnym razie jako niecelny.
    4.  **Skalowanie:** Cechy (pozycja, odległość) mają różne zakresy wartości. Aby zapobiec dominacji cech o większych wartościach (jak LOC_Y) nad cechami o mniejszych wartościach (jak SHOT_DISTANCE) w obliczaniu odległości, dane są **skalowane** za pomocą `StandardScaler`. To zapewnia, że wszystkie cechy mają porównywalny wpływ na model.
    5.  **Podział na zbiór treningowy i testowy:** Dane gracza są dzielone (tutaj 70% na trening, 30% na test). Model "uczy się" wzorców na zbiorze treningowym, a następnie jego wydajność jest oceniana na zbiorze testowym, którego model wcześniej nie widział. Użycie `stratify=y` zapewnia, że proporcje rzutów celnych i niecelnych są podobne w obu zbiorach.

    **Jak interpretować metryki oceny modelu?**

    Po kliknięciu przycisku "Trenuj i Ewaluuj Model KNN", zobaczysz następujące wyniki oceny na **zbiorze testowym**:

    1.  **Dokładność (Accuracy):**
        * **Co oznacza:** Jaki procent wszystkich rzutów w zbiorze testowym model sklasyfikował poprawnie (celne jako celne, niecelne jako niecelne).
        * **Interpretacja:** Wyższa wartość oznacza lepszą ogólną skuteczność modelu. Należy jednak być ostrożnym, jeśli dane są niezbalansowane (np. gracz trafia 80% rzutów) - model przewidujący wszystko jako "celny" miałby wysoką dokładność, ale byłby bezużyteczny.

    2.  **Raport Klasyfikacji:** Dostarcza bardziej szczegółowych metryk dla każdej klasy (0 - Niecelny, 1 - Celny):
        * **Precyzja (Precision):**
            * *Dla klasy "Niecelny (0)":* Spośród wszystkich rzutów, które model *przewidział* jako niecelne, jaki procent faktycznie był niecelny? (Mierzy, jak bardzo można ufać predykcji "niecelny").
            * *Dla klasy "Celny (1)":* Spośród wszystkich rzutów, które model *przewidział* jako celne, jaki procent faktycznie był celny? (Mierzy, jak bardzo można ufać predykcji "celny").
        * **Czułość (Recall / Pełność):**
            * *Dla klasy "Niecelny (0)":* Spośród wszystkich *faktycznie* niecelnych rzutów, jaki procent model poprawnie *wykrył* jako niecelne? (Mierzy zdolność modelu do wyłapywania niecelnych rzutów).
            * *Dla klasy "Celny (1)":* Spośród wszystkich *faktycznie* celnych rzutów, jaki procent model poprawnie *wykrył* jako celne? (Mierzy zdolność modelu do wyłapywania celnych rzutów).
        * **F1-Score:** Średnia harmoniczna precyzji i czułości. Jest to dobra, zbalansowana miara wydajności dla danej klasy, szczególnie przydatna przy niezbalansowanych zbiorach danych.
        * **Support:** Liczba rzeczywistych wystąpień danej klasy w zbiorze testowym.

    3.  **Macierz Pomyłek (Confusion Matrix):** Wizualna reprezentacja wyników klasyfikacji:
        * **Górny lewy (TN - True Negative):** Liczba rzutów *niecelnych*, które model poprawnie sklasyfikował jako *niecelne*.
        * **Dolny prawy (TP - True Positive):** Liczba rzutów *celnych*, które model poprawnie sklasyfikował jako *celne*.
        * **Górny prawy (FP - False Positive):** Liczba rzutów *niecelnych*, które model błędnie sklasyfikował jako *celne* (Błąd typu I).
        * **Dolny lewy (FN - False Negative):** Liczba rzutów *celnych*, które model błędnie sklasyfikował jako *niecelne* (Błąd typu II).
        * **Idealny model** miałby wartości tylko na przekątnej (TN i TP), a zera poza nią (FP i FN).

    **Ograniczenia i Co dalej?**

    * **Prostota modelu:** Używa tylko 3 cech. Bardziej zaawansowane modele mogłyby uwzględniać np. typ akcji, czas gry, obrońcę itp.
    * **Wpływ 'k':** Wybór liczby sąsiadów `k` ma znaczenie. Małe `k` może prowadzić do modelu wrażliwego na szum, duże `k` może nadmiernie wygładzać granice decyzyjne. Optymalne `k` można znaleźć np. przez walidację krzyżową (niezaimplementowane tutaj).
    * **Ilość danych:** Wydajność modelu silnie zależy od ilości i jakości danych dla danego gracza. Dla graczy z małą liczbą rzutów model może być niewiarygodny.
    * **Zbalansowanie klas:** Jeśli gracz ma bardzo dużą lub bardzo małą skuteczność, klasy mogą być niezbalansowane, co wpływa na interpretację metryk (szczególnie dokładności).

    Ten model KNN służy jako **wprowadzenie i demonstracja** możliwości predykcji. Wyniki należy interpretować z uwzględnieniem jego prostoty i potencjalnych ograniczeń.
    """) # KONIEC TEKSTU INTERPRETACJI

    # --- Reszta kodu dla tab5 ---
    if selected_player:
        player_model_data = filter_data_by_player(selected_player, filtered_data)

        if not player_model_data.empty:
            # Cechy, które będą użyte do treningu modelu
            model_features = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE']
            target_variable = 'SHOT_MADE_FLAG'

            # Sprawdzenie, czy wymagane kolumny istnieją
            if all(feat in player_model_data.columns for feat in model_features) and target_variable in player_model_data.columns:

                # Przygotowanie danych - usunięcie wierszy z brakującymi wartościami w kluczowych kolumnach
                pmdc = player_model_data[model_features + [target_variable]].dropna().copy()

                # Upewnienie się, że target jest numeryczny i nie ma NaN po konwersji
                pmdc[target_variable] = pd.to_numeric(pmdc[target_variable], errors='coerce')
                pmdc = pmdc.dropna(subset=[target_variable])

                if not pmdc.empty:
                    # Konwersja targetu na integer (0 lub 1)
                    pmdc[target_variable] = pmdc[target_variable].astype(int)

                    # Sprawdzenie, czy mamy wystarczająco danych i dwie klasy (celny/niecelny)
                    min_samples_for_model = 50 # Minimalna liczba próbek potrzebna do budowy modelu
                    if len(pmdc) >= min_samples_for_model and pmdc[target_variable].nunique() == 2:

                        st.subheader("Konfiguracja Modelu KNN")
                        # Slider do wyboru liczby sąsiadów 'k'
                        k = st.slider("Liczba sąsiadów (k):", min_value=3, max_value=min(25, len(pmdc)//3), value=5, step=2, key='knn_k_slider', # Ograniczono max_value k
                                     help="Parametr 'k' określa, ilu najbliższych sąsiadów będzie branych pod uwagę przy predykcji.")

                        # Przycisk do rozpoczęcia treningu
                        if st.button(f"Trenuj i Ewaluuj Model KNN dla {selected_player}", key='train_knn_button'):
                            with st.spinner(f"Trenowanie modelu KNN (k={k}) dla {selected_player}..."):
                                try:
                                    # Podział danych na cechy (X) i target (y)
                                    X = pmdc[model_features]
                                    y = pmdc[target_variable]

                                    # Podział na zbiór treningowy i testowy (np. 70% trening, 30% test)
                                    # stratify=y zapewnia podobny rozkład klas w obu zbiorach
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.3, random_state=42, stratify=y
                                    )

                                    # Skalowanie cech - bardzo ważne dla KNN!
                                    scaler = StandardScaler()
                                    X_train_scaled = scaler.fit_transform(X_train) # Uczymy skalera na danych treningowych
                                    X_test_scaled = scaler.transform(X_test)     # Transformujemy dane testowe tym samym skalerem

                                    # Inicjalizacja i trenowanie modelu KNN
                                    model = KNeighborsClassifier(n_neighbors=k)
                                    model.fit(X_train_scaled, y_train)

                                    # Predykcja na zbiorze testowym
                                    y_pred = model.predict(X_test_scaled)

                                    # Ewaluacja modelu
                                    accuracy = accuracy_score(y_test, y_pred)
                                    report = classification_report(y_test, y_pred, target_names=['Niecelny (0)', 'Celny (1)'], output_dict=True, zero_division=0) # Dodano target_names
                                    conf_matrix = confusion_matrix(y_test, y_pred)

                                    st.success(f"Model KNN (k={k}) wytrenowany i oceniony pomyślnie!")

                                    # Wyświetlanie wyników
                                    st.metric("Dokładność (Accuracy) na zbiorze testowym:", f"{accuracy:.2%}")

                                    st.subheader("Raport Klasyfikacji:")
                                    st.caption("Pokazuje precyzję, pełność (recall) i F1-score dla każdej klasy.")
                                    # Konwersja raportu (słownika) na DataFrame dla ładniejszego wyświetlenia
                                    report_df = pd.DataFrame(report).transpose()
                                    # Upewnienie się, że kolumna 'support' jest typu int
                                    if 'support' in report_df.columns:
                                        report_df['support'] = report_df['support'].astype(int)
                                    # Formatowanie kolumn procentowych
                                    for col in ['precision', 'recall', 'f1-score']:
                                        if col in report_df.columns:
                                             report_df[col] = report_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

                                    st.dataframe(report_df.drop(columns=['support'], errors='ignore'), use_container_width=True) # Usunięto support dla czytelności, można przywrócić
                                    # st.dataframe(report_df.round(3)) # Alternatywnie, poprzednie wyświetlanie

                                    st.subheader("Macierz Pomyłek:")
                                    st.caption("Pokazuje, ile rzutów zostało poprawnie (TN, TP) i niepoprawnie (FP, FN) sklasyfikowanych.")
                                    # Wizualizacja macierzy pomyłek za pomocą Plotly
                                    fig_cm = px.imshow(conf_matrix, text_auto=True, aspect="auto",
                                                       labels=dict(x="Predykowana Klasa", y="Prawdziwa Klasa", color="Liczba Próbek"),
                                                       x=['Niecelny (0)', 'Celny (1)'],
                                                       y=['Niecelny (0)', 'Celny (1)'],
                                                       color_continuous_scale=px.colors.sequential.Blues, # Można zmienić schemat kolorów
                                                       title=f"Macierz Pomyłek (Zbiór Testowy, k={k})")
                                    fig_cm.update_layout(coloraxis_showscale=False) # Ukrycie paska kolorów dla przejrzystości
                                    st.plotly_chart(fig_cm, use_container_width=True)

                                    st.caption(f"Model wytrenowany na {len(X_train)} próbkach, testowany na {len(X_test)} próbkach.")
                                    st.caption(f"Użyte cechy: {', '.join(model_features)}")

                                except Exception as e:
                                    st.error(f"Wystąpił błąd podczas trenowania lub ewaluacji modelu: {e}")
                                    st.exception(e) # Wyświetla pełny traceback błędu dla debugowania

                    elif pmdc[target_variable].nunique() != 2:
                        st.warning(f"Nie można wytrenować modelu dla '{selected_player}'. Dane zawierają tylko jedną klasę wyniku rzutu (wszystkie celne lub wszystkie niecelne) po przetworzeniu.")
                    else: # len(pmdc) < min_samples_for_model
                        st.warning(f"Niewystarczająca ilość danych ({len(pmdc)} próbek) dla gracza '{selected_player}' do zbudowania wiarygodnego modelu. Wymagane minimum: {min_samples_for_model}.")
                else:
                    st.warning(f"Brak poprawnych danych dla gracza '{selected_player}' po oczyszczeniu (np. usunięciu NaN). Nie można zbudować modelu.")
            else:
                missing_cols = [col for col in model_features + [target_variable] if col not in player_model_data.columns]
                st.warning(f"Brak wymaganych kolumn do zbudowania modelu dla '{selected_player}'. Brakujące kolumny: {', '.join(missing_cols)}")
        else:
            st.warning(f"Brak jakichkolwiek danych dla wybranego gracza: '{selected_player}'.")
    else:
        st.info("Wybierz gracza z panelu bocznego, aby zbudować i ocenić dla niego model predykcyjny KNN.")



# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Rozszerzona Aplikacja Streamlit - Analiza NBA")
try:
    ts = pd.Timestamp.now(tz='Europe/Warsaw').strftime('%Y-%m-%d %H:%M:%S %Z')
except Exception as e:
    st.sidebar.warning(f"Problem ze strefą czasową 'Europe/Warsaw': {e}.")
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S (czas lokalny)')
st.sidebar.markdown(f"Czas: {ts}")