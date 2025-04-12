# nba_app_v2.py
# Wersja kodu uwzględniająca analizę czasową, zespołową, dodatkowe kolumny,
# filtr sezonu, poprawkę błędu describe(), implementację porównania graczy
# ORAZ zmodyfikowaną pierwszą zakładkę (rankingi zamiast describe).

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier # Import dla typu
from sklearn.model_selection import train_test_split # Import dla typu
from sklearn.preprocessing import StandardScaler # Import dla typu
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Import dla typu
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# --- Konfiguracja Początkowa ---
st.set_page_config(layout="wide", page_title="Rozszerzona Analiza Rzutów NBA 2023-24")

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
                 data['SHOT_MADE_FLAG'] = pd.to_numeric(data['SHOT_MADE_FLAG'], errors='coerce') # Próba konwersji, błędy -> NaN


        time_cols = ['PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING']
        missing_time_cols = False
        for col in time_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                if data[col].isnull().any():
                     st.warning(f"W kolumnie '{col}' znaleziono wartości nienumeryczne. Zostały one zastąpione przez NaN.")
            else:
                 missing_time_cols = True


        if not missing_time_cols and all(pd.api.types.is_numeric_dtype(data[col]) for col in time_cols):
             data['GAME_TIME_SEC'] = data['MINUTES_REMAINING'] * 60 + data['SECONDS_REMAINING']
             data['QUARTER_TYPE'] = data['PERIOD'].apply(lambda x: 'Dogrywka' if pd.notna(x) and x > 4 else 'Regularna')
        else:
             st.warning("Nie można utworzyć kolumn pomocniczych 'GAME_TIME_SEC' lub 'QUARTER_TYPE' z powodu braku/błędów w kolumnach czasowych.")


        key_cols = ['PLAYER_NAME', 'TEAM_NAME', 'LOC_X', 'LOC_Y', 'SHOT_DISTANCE', 'SEASON_TYPE',
                    'ACTION_TYPE', 'SHOT_TYPE', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE']
        for col in key_cols:
            if col not in data.columns:
                st.warning(f"Brak oczekiwanej kolumny: '{col}'. Niektóre analizy mogą być niedostępne.")

        # Upewnij się, że SHOT_MADE_FLAG jest numeryczny po wszystkich operacjach
        if 'SHOT_MADE_FLAG' in data.columns:
             data['SHOT_MADE_FLAG'] = pd.to_numeric(data['SHOT_MADE_FLAG'], errors='coerce')

        return data

    except FileNotFoundError:
        st.error(f"Błąd: Nie znaleziono pliku {file_path}.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Błąd podczas wczytywania lub przetwarzania danych: {e}")
        return pd.DataFrame()

# --- Funkcje Wizualizacyjne i Analityczne (bez zmian w definicjach) ---

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
    if 'PLAYER_NAME' not in data.columns or data.empty:
        return pd.DataFrame()
    player_data = data[data['PLAYER_NAME'] == player_name].copy()
    # Usunięto warning stąd, aby nie powtarzał się przy każdej zmianie
    return player_data

@st.cache_data
def filter_data_by_team(team_name, data):
    """Filtruje dane dla wybranej drużyny."""
    if 'TEAM_NAME' not in data.columns or data.empty:
         # st.warning("Brak kolumny 'TEAM_NAME' w danych.") # Mniej inwazyjne
         return pd.DataFrame()
    team_data = data[data['TEAM_NAME'] == team_name].copy()
    return team_data

@st.cache_data
def get_basic_stats(entity_data, entity_name, entity_type="Gracz"):
    """Oblicza podstawowe statystyki dla gracza lub drużyny."""
    stats = {'total_shots': len(entity_data), 'made_shots': "N/A", 'shooting_pct': "N/A"}
    if 'SHOT_MADE_FLAG' in entity_data.columns and stats['total_shots'] > 0:
        numeric_shots = pd.to_numeric(entity_data['SHOT_MADE_FLAG'], errors='coerce').dropna().astype(int)
        if not numeric_shots.empty:
            stats['made_shots'] = int(numeric_shots.sum())
            total_valid_shots = len(numeric_shots)
            if total_valid_shots > 0:
               stats['shooting_pct'] = (stats['made_shots'] / total_valid_shots) * 100
            else:
                stats['made_shots'] = 0
                stats['shooting_pct'] = 0.0
        else:
            stats['made_shots'] = "Błąd danych"
            stats['shooting_pct'] = "N/A"
    elif stats['total_shots'] == 0:
         stats['made_shots'] = 0
         stats['shooting_pct'] = "N/A"
    return stats

@st.cache_data
def plot_distance_histogram(entity_data, entity_name, entity_type="Gracz"):
    """Tworzy interaktywny histogram odległości rzutów."""
    if 'SHOT_DISTANCE' in entity_data.columns and 'SHOT_MADE_FLAG' in entity_data.columns:
        plot_data = entity_data.dropna(subset=['SHOT_MADE_FLAG', 'SHOT_DISTANCE'])
        if plot_data.empty: return None
        plot_data['Wynik Rzutu'] = plot_data['SHOT_MADE_FLAG'].map({0: 'Niecelny', 1: 'Celny'})

        fig = px.histogram(plot_data, x='SHOT_DISTANCE', color='Wynik Rzutu',
                           barmode='stack', nbins=30,
                           title=f'Rozkład odległości rzutów - {entity_name} ({entity_type})',
                           labels={'SHOT_DISTANCE': 'Odległość (stopy)', 'Wynik Rzutu': 'Wynik Rzutu'},
                           category_orders={"Wynik Rzutu": ['Niecelny', 'Celny']},
                           color_discrete_map={'Niecelny': 'red', 'Celny': 'green'})
        return fig
    return None

@st.cache_data
def plot_shot_chart(entity_data, entity_name, entity_type="Gracz"):
    """Tworzy interaktywną mapę rzutów."""
    if 'LOC_X' in entity_data.columns and 'LOC_Y' in entity_data.columns:
        plot_data = entity_data.dropna(subset=['SHOT_MADE_FLAG', 'LOC_X', 'LOC_Y'])
        if plot_data.empty: return None
        plot_data['Wynik Rzutu'] = plot_data['SHOT_MADE_FLAG'].map({0: 'Niecelny', 1: 'Celny'})
        color_col, color_map, cat_orders = 'Wynik Rzutu', {'Niecelny': 'red', 'Celny': 'green'}, {"Wynik Rzutu": ['Niecelny', 'Celny']}

        hover_cols = [col for col in ['SHOT_DISTANCE', 'SHOT_TYPE', 'ACTION_TYPE', 'SHOT_ZONE_BASIC', 'PERIOD'] if col in plot_data.columns]

        fig = px.scatter(plot_data, x='LOC_X', y='LOC_Y',
                         color=color_col,
                         title=f'Mapa rzutów - {entity_name} ({entity_type})',
                         labels={'LOC_X': 'Pozycja X', 'LOC_Y': 'Pozycja Y', 'Wynik Rzutu': 'Wynik'},
                         hover_data=hover_cols if hover_cols else None,
                         category_orders=cat_orders,
                         color_discrete_map=color_map,
                         opacity=0.7)
        fig = add_court_shapes(fig)
        fig.update_layout(height=600)
        return fig
    return None

@st.cache_data
def calculate_hot_zones(entity_data, min_shots_in_zone=5):
    """Oblicza statystyki dla stref rzutowych."""
    required_cols = ['LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']
    if not all(col in entity_data.columns for col in required_cols):
        return pd.DataFrame() # Cicha obsługa - brak ostrzeżenia w cache'owanej funkcji

    zone_data = entity_data.dropna(subset=['LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG'])
    if zone_data.empty: return pd.DataFrame()

    x_bins = np.linspace(-300, 300, 11)
    y_bins = np.linspace(-50, 450, 11)
    zone_data['zone_x'] = pd.cut(zone_data['LOC_X'], bins=x_bins)
    zone_data['zone_y'] = pd.cut(zone_data['LOC_Y'], bins=y_bins)

    zones = zone_data.groupby(['zone_x', 'zone_y'], observed=False).agg(
        total_shots=('SHOT_MADE_FLAG', 'count'),
        made_shots=('SHOT_MADE_FLAG', 'sum'),
        percentage=('SHOT_MADE_FLAG', 'mean')
    ).reset_index()

    zones = zones[zones['total_shots'] >= min_shots_in_zone].copy()
    if zones.empty: return pd.DataFrame()

    zones['percentage'] *= 100
    zones['x_center'] = zones['zone_x'].apply(lambda x: x.mid if isinstance(x, pd.Interval) else None)
    zones['y_center'] = zones['zone_y'].apply(lambda x: x.mid if isinstance(x, pd.Interval) else None)
    zones = zones.dropna(subset=['x_center', 'y_center'])
    return zones

@st.cache_data
def plot_hot_zones_heatmap(hot_zones_df, entity_name, entity_type="Gracz"):
    """Tworzy interaktywną mapę ciepła stref rzutowych."""
    required_cols = ['x_center', 'y_center', 'total_shots', 'percentage', 'made_shots']
    if not hot_zones_df.empty and all(col in hot_zones_df.columns for col in required_cols):
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(hot_zones_df[col]):
                try: hot_zones_df[col] = pd.to_numeric(hot_zones_df[col])
                except ValueError: return None # Błąd konwersji -> brak wykresu

        hot_zones_df = hot_zones_df.dropna(subset=required_cols)
        if hot_zones_df.empty: return None

        min_shots_in_zone = int(hot_zones_df["total_shots"].min())
        min_pct, max_pct = hot_zones_df['percentage'].min(), hot_zones_df['percentage'].max()
        if pd.isna(min_pct): min_pct = 0
        if pd.isna(max_pct): max_pct = 100

        color_range = [max(0, min_pct - 10), min(100, max_pct + 10)]
        if color_range[0] >= color_range[1]: color_range = [0, 100]

        fig = px.scatter(hot_zones_df, x='x_center', y='y_center',
                         size='total_shots', color='percentage',
                         color_continuous_scale=px.colors.diverging.RdYlGn,
                         size_max=40, range_color=color_range,
                         title=f'Skuteczność stref rzutowych ({entity_type}: {entity_name}, min. {min_shots_in_zone} rzutów)',
                         labels={'x_center': 'Pozycja X (środek strefy)', 'y_center': 'Pozycja Y (środek strefy)',
                                 'total_shots': 'Liczba rzutów', 'percentage': 'Skuteczność (%)'},
                         custom_data=['made_shots', 'total_shots']
                        )
        fig.update_traces(
            hovertemplate="<b>Strefa X:</b> %{x:.1f}, <b>Y:</b> %{y:.1f}<br>" +
                          "<b>Liczba rzutów:</b> %{customdata[1]}<br>" +
                          "<b>Trafione:</b> %{customdata[0]}<br>" +
                          "<b>Skuteczność:</b> %{marker.color:.1f}%<extra></extra>"
        )
        fig = add_court_shapes(fig)
        fig.update_layout(height=600)
        return fig
    return None # Jeśli brak danych lub wymaganych kolumn

@st.cache_data
def plot_player_quarter_eff(entity_data, entity_name, entity_type="Gracz"):
    """Wykres skuteczności w poszczególnych kwartach/dogrywkach."""
    if 'PERIOD' not in entity_data.columns or 'SHOT_MADE_FLAG' not in entity_data.columns: return None
    quarter_data = entity_data.dropna(subset=['PERIOD', 'SHOT_MADE_FLAG'])
    if quarter_data.empty: return None
    quarter_data['PERIOD'] = quarter_data['PERIOD'].astype(int)

    quarter_eff = quarter_data.groupby('PERIOD')['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
    quarter_eff['mean'] *= 100
    quarter_eff = quarter_eff[quarter_eff['count'] >= 5]
    if quarter_eff.empty: return None

    def map_period(p):
        if p <= 4: return f"Kwarta {int(p)}"
        elif p == 5: return "OT 1"
        elif p == 6: return "OT 2"
        else: return f"OT {int(p-4)}"
    quarter_eff['Okres Gry'] = quarter_eff['PERIOD'].apply(map_period)
    quarter_eff = quarter_eff.sort_values(by='PERIOD')

    fig = px.bar(quarter_eff, x='Okres Gry', y='mean', text='mean',
                 title=f'Skuteczność w kwartach/dogrywkach - {entity_name} ({entity_type})',
                 labels={'Okres Gry': 'Okres Gry', 'mean': 'Skuteczność (%)'},
                 hover_data=['count'])
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

    fig = px.line(monthly_eff, x='Miesiąc', y='mean', markers=True,
                  title=f'Miesięczny trend skuteczności - {entity_name} ({entity_type})',
                  labels={'Miesiąc': 'Miesiąc', 'mean': 'Skuteczność (%)'},
                  hover_data=['count'])
    max_y = 100
    if not pd.isna(monthly_eff['mean'].max()):
         max_y = min(100, monthly_eff['mean'].max() * 1.1 + 5)
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
    fig = px.bar(grouped_eff, x=group_col, y='mean', text='mean',
                 title=f'Skuteczność wg {axis_label} - {entity_name} ({entity_type}) (Top {top_n})',
                 labels={group_col: axis_label, 'mean': 'Skuteczność (%)'},
                 hover_data=['count'])
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(yaxis_range=[0, 100], uniformtext_minsize=8, uniformtext_mode='hide', xaxis_title=axis_label)
    return fig

@st.cache_data
def plot_comparison_eff_distance(compare_data, selected_players):
    """Porównuje skuteczność graczy względem odległości."""
    required_cols = ['SHOT_DISTANCE', 'SHOT_MADE_FLAG', 'PLAYER_NAME']
    if not all(col in compare_data.columns for col in required_cols): return None
    compare_data_eff = compare_data.dropna(subset=['SHOT_MADE_FLAG', 'SHOT_DISTANCE'])
    if compare_data_eff.empty: return None

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
        fig = px.line(effectiveness, x='distance_mid', y='mean', color='PLAYER_NAME',
                      title=f'Porównanie skuteczności rzutów względem odległości (min. {min_shots_per_bin} rzutów w przedziale)',
                      labels={'distance_mid': 'Środek przedziału odległości (stopy)', 'mean': 'Skuteczność (%)', 'PLAYER_NAME': 'Gracz'},
                      markers=True, hover_data=['count'])
        fig.update_layout(yaxis_range=yaxis_range)
        return fig
    return None

# --- NOWA FUNKCJA ---
@st.cache_data
def calculate_top_performers(data, group_by_col, min_total_shots, min_2pt_shots, min_3pt_shots, top_n=10):
    """Oblicza rankingi Top N graczy/zespołów wg skuteczności."""
    if group_by_col not in data.columns or 'SHOT_MADE_FLAG' not in data.columns:
        st.warning(f"Brak kolumny '{group_by_col}' lub 'SHOT_MADE_FLAG' do rankingu.")
        return None, None, None

    valid_data = data.dropna(subset=[group_by_col, 'SHOT_MADE_FLAG'])
    if valid_data.empty: return None, None, None

    # --- Ranking Ogólny (FG%) ---
    overall_stats = valid_data.groupby(group_by_col)['SHOT_MADE_FLAG'].agg(
        Made='sum', Attempts='count'
    ).reset_index()
    overall_stats = overall_stats[overall_stats['Attempts'] >= min_total_shots]
    if not overall_stats.empty:
        overall_stats['FG%'] = (overall_stats['Made'] / overall_stats['Attempts']) * 100
        top_overall = overall_stats.sort_values(by='FG%', ascending=False).head(top_n)
        # Zmiana nazw kolumn na bardziej przyjazne
        top_overall = top_overall.rename(columns={group_by_col: group_by_col.replace('_',' ').title(), 'Attempts': 'Próby'})
        top_overall = top_overall[[group_by_col.replace('_',' ').title(), 'FG%', 'Próby']] # Kolejność kolumn
    else:
        top_overall = pd.DataFrame()

    # --- Ranking za 2 punkty (2PT FG%) ---
    # !!! DOSTOSUJ WARTOŚĆ '2PT Field Goal' JEŚLI JEST INNA W TWOICH DANYCH !!!
    shot_type_2pt = '2PT Field Goal'
    if 'SHOT_TYPE' in valid_data.columns and shot_type_2pt in valid_data['SHOT_TYPE'].unique():
        data_2pt = valid_data[valid_data['SHOT_TYPE'] == shot_type_2pt]
        stats_2pt = data_2pt.groupby(group_by_col)['SHOT_MADE_FLAG'].agg(
            Made_2PT='sum', Attempts_2PT='count'
        ).reset_index()
        stats_2pt = stats_2pt[stats_2pt['Attempts_2PT'] >= min_2pt_shots]
        if not stats_2pt.empty:
            stats_2pt['2PT FG%'] = (stats_2pt['Made_2PT'] / stats_2pt['Attempts_2PT']) * 100
            top_2pt = stats_2pt.sort_values(by='2PT FG%', ascending=False).head(top_n)
            top_2pt = top_2pt.rename(columns={group_by_col: group_by_col.replace('_',' ').title(), 'Attempts_2PT': 'Próby 2PT'})
            top_2pt = top_2pt[[group_by_col.replace('_',' ').title(), '2PT FG%', 'Próby 2PT']]
        else:
             top_2pt = pd.DataFrame()
    else:
        st.caption(f"Nie znaleziono rzutów typu '{shot_type_2pt}' dla rankingu 2PT.")
        top_2pt = pd.DataFrame()


    # --- Ranking za 3 punkty (3PT FG%) ---
     # !!! DOSTOSUJ WARTOŚĆ '3PT Field Goal' JEŚLI JEST INNA W TWOICH DANYCH !!!
    shot_type_3pt = '3PT Field Goal'
    if 'SHOT_TYPE' in valid_data.columns and shot_type_3pt in valid_data['SHOT_TYPE'].unique():
        data_3pt = valid_data[valid_data['SHOT_TYPE'] == shot_type_3pt]
        stats_3pt = data_3pt.groupby(group_by_col)['SHOT_MADE_FLAG'].agg(
            Made_3PT='sum', Attempts_3PT='count'
        ).reset_index()
        stats_3pt = stats_3pt[stats_3pt['Attempts_3PT'] >= min_3pt_shots]
        if not stats_3pt.empty:
            stats_3pt['3PT FG%'] = (stats_3pt['Made_3PT'] / stats_3pt['Attempts_3PT']) * 100
            top_3pt = stats_3pt.sort_values(by='3PT FG%', ascending=False).head(top_n)
            top_3pt = top_3pt.rename(columns={group_by_col: group_by_col.replace('_',' ').title(), 'Attempts_3PT': 'Próby 3PT'})
            top_3pt = top_3pt[[group_by_col.replace('_',' ').title(), '3PT FG%', 'Próby 3PT']]
        else:
            top_3pt = pd.DataFrame()
    else:
        st.caption(f"Nie znaleziono rzutów typu '{shot_type_3pt}' dla rankingu 3PT.")
        top_3pt = pd.DataFrame()

    return top_overall, top_2pt, top_3pt


# --- Główna część aplikacji Streamlit ---
st.title("🏀 Rozszerzona Interaktywna Analiza Rzutów Graczy NBA (Sezon 2023-24)")

# --- Ładowanie Danych ---
shooting_data = load_shooting_data(CSV_FILE_PATH)

# --- Pasek Boczny (Sidebar) ---
st.sidebar.header("Opcje Filtrowania i Analizy")

if not shooting_data.empty:
    # --- Filtr Typu Sezonu ---
    available_season_types = ['Wszystko']
    if 'SEASON_TYPE' in shooting_data.columns:
         unique_seasons = shooting_data['SEASON_TYPE'].dropna().unique().tolist()
         available_season_types.extend(unique_seasons)
    selected_season_type = st.sidebar.selectbox("Wybierz typ sezonu:", options=available_season_types, index=0, key='season_select')

    if selected_season_type != 'Wszystko':
        filtered_data = shooting_data[shooting_data['SEASON_TYPE'] == selected_season_type].copy()
        st.sidebar.success(f"Filtrujesz wg: {selected_season_type} ({len(filtered_data)} rzutów)")
    else:
        filtered_data = shooting_data.copy()
        st.sidebar.info("Wyświetlanie danych dla wszystkich typów sezonu.")

    # --- Wybór Gracza/Drużyny ---
    # Upewnij się, że PLAYER_NAME i TEAM_NAME istnieją przed próbą pobrania unique()
    available_players = []
    if 'PLAYER_NAME' in filtered_data.columns:
         available_players = sorted(filtered_data['PLAYER_NAME'].dropna().unique())

    selected_player = st.sidebar.selectbox("Wybierz gracza do analizy:", options=available_players, index=0 if available_players else None, key='player_select', disabled=not available_players)

    available_teams = []
    if 'TEAM_NAME' in filtered_data.columns:
         available_teams = sorted(filtered_data['TEAM_NAME'].dropna().unique())

    selected_team = st.sidebar.selectbox("Wybierz drużynę do analizy:", options=available_teams, index=0 if available_teams else None, key='team_select', disabled=not available_teams)

    selected_players_compare = st.sidebar.multiselect("Wybierz graczy do porównania (2-5):", options=available_players, default=available_players[:2] if len(available_players) >= 2 else [], max_selections=5, key='player_multi_select', disabled=not available_players)

    # --- Zakładki Aplikacji ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Rankingi Skuteczności", # Zmieniono nazwę
        "⛹️ Analiza Gracza",
        "🆚 Porównanie Graczy",
        "🏀 Analiza Zespołowa",
        "🎯 Model Predykcji (KNN)"
    ])

    # --- Zakładka 1: Rankingi Skuteczności ---
    with tab1:
        st.header(f"Rankingi Skuteczności ({selected_season_type})")

        # --- Definiowanie Minimalnej Liczby Prób ---
        st.markdown("### Ustaw minimalną liczbę prób do rankingów")
        col_att1, col_att2, col_att3 = st.columns(3)
        # Użyj unikalnych kluczy dla widgetów
        with col_att1:
            min_total_shots = st.number_input("Min. rzutów ogółem:", min_value=10, value=100, step=10, key="min_total")
        with col_att2:
            min_2pt_shots = st.number_input("Min. rzutów za 2 pkt:", min_value=5, value=50, step=5, key="min_2pt")
        with col_att3:
            min_3pt_shots = st.number_input("Min. rzutów za 3 pkt:", min_value=5, value=30, step=5, key="min_3pt")

        # --- Obliczanie Rankingów ---
        # Ustal wyższe progi dla zespołów (np. 5x więcej prób niż dla gracza)
        min_total_shots_team = min_total_shots * 5
        min_2pt_shots_team = min_2pt_shots * 5
        min_3pt_shots_team = min_3pt_shots * 5

        top_players_overall, top_players_2pt, top_players_3pt = calculate_top_performers(
            filtered_data, 'PLAYER_NAME', min_total_shots, min_2pt_shots, min_3pt_shots
        )
        top_teams_overall, top_teams_2pt, top_teams_3pt = calculate_top_performers(
            filtered_data, 'TEAM_NAME', min_total_shots_team, min_2pt_shots_team, min_3pt_shots_team
        )

        # --- Wyświetlanie Rankingów ---
        st.markdown("---")
        st.subheader("Skuteczność Ogółem (FG%)")
        col_fg1, col_fg2 = st.columns(2)
        with col_fg1:
            st.markdown(f"**Top 10 Graczy (min. {min_total_shots} prób)**")
            if top_players_overall is not None and not top_players_overall.empty:
                st.dataframe(top_players_overall, hide_index=True, use_container_width=True,
                             column_config={"FG%": st.column_config.ProgressColumn("FG%", format="%.1f%%", min_value=0, max_value=100),
                                            "Próby": st.column_config.NumberColumn("Próby")})
            else: st.caption("Brak graczy spełniających kryteria.")
        with col_fg2:
             st.markdown(f"**Top 10 Zespołów (min. {min_total_shots_team} prób)**")
             if top_teams_overall is not None and not top_teams_overall.empty:
                 st.dataframe(top_teams_overall, hide_index=True, use_container_width=True,
                              column_config={"FG%": st.column_config.ProgressColumn("FG%", format="%.1f%%", min_value=0, max_value=100),
                                             "Próby": st.column_config.NumberColumn("Próby")})
             else: st.caption("Brak zespołów spełniających kryteria.")

        st.markdown("---")
        st.subheader("Skuteczność za 2 Punkty (2PT FG%)")
        col_2pt1, col_2pt2 = st.columns(2)
        with col_2pt1:
             st.markdown(f"**Top 10 Graczy (min. {min_2pt_shots} prób)**")
             if top_players_2pt is not None and not top_players_2pt.empty:
                 st.dataframe(top_players_2pt, hide_index=True, use_container_width=True,
                              column_config={"2PT FG%": st.column_config.ProgressColumn("2PT FG%", format="%.1f%%", min_value=0, max_value=100),
                                             "Próby 2PT": st.column_config.NumberColumn("Próby 2PT")})
             else: st.caption("Brak graczy spełniających kryteria.")
        with col_2pt2:
            st.markdown(f"**Top 10 Zespołów (min. {min_2pt_shots_team} prób)**")
            if top_teams_2pt is not None and not top_teams_2pt.empty:
                st.dataframe(top_teams_2pt, hide_index=True, use_container_width=True,
                              column_config={"2PT FG%": st.column_config.ProgressColumn("2PT FG%", format="%.1f%%", min_value=0, max_value=100),
                                              "Próby 2PT": st.column_config.NumberColumn("Próby 2PT")})
            else: st.caption("Brak zespołów spełniających kryteria.")

        st.markdown("---")
        st.subheader("Skuteczność za 3 Punkty (3PT FG%)")
        col_3pt1, col_3pt2 = st.columns(2)
        with col_3pt1:
            st.markdown(f"**Top 10 Graczy (min. {min_3pt_shots} prób)**")
            if top_players_3pt is not None and not top_players_3pt.empty:
                st.dataframe(top_players_3pt, hide_index=True, use_container_width=True,
                              column_config={"3PT FG%": st.column_config.ProgressColumn("3PT FG%", format="%.1f%%", min_value=0, max_value=100),
                                              "Próby 3PT": st.column_config.NumberColumn("Próby 3PT")})
            else: st.caption("Brak graczy spełniających kryteria.")
        with col_3pt2:
            st.markdown(f"**Top 10 Zespołów (min. {min_3pt_shots_team} prób)**")
            if top_teams_3pt is not None and not top_teams_3pt.empty:
                st.dataframe(top_teams_3pt, hide_index=True, use_container_width=True,
                              column_config={"3PT FG%": st.column_config.ProgressColumn("3PT FG%", format="%.1f%%", min_value=0, max_value=100),
                                             "Próby 3PT": st.column_config.NumberColumn("Próby 3PT")})
            else: st.caption("Brak zespołów spełniających kryteria.")

        # --- Pozostawiona sekcja ogólnych rozkładów ---
        st.markdown("---")
        st.subheader("Ogólne Statystyki i Rozkłady (dane po filtracji)")
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            overall_stats = get_basic_stats(filtered_data, selected_season_type, "Sezon")
            st.metric("Rzuty Ogółem (Filtrowane)", overall_stats['total_shots'])
            st.metric("Trafione Ogółem (Filtrowane)", overall_stats['made_shots'])
            st.metric("Skuteczność Ogółem (Filtrowana)", f"{overall_stats['shooting_pct']:.1f}%" if isinstance(overall_stats['shooting_pct'], (int, float)) else "N/A")

            fig_hist_all = plot_distance_histogram(filtered_data, selected_season_type, "Sezon")
            if fig_hist_all: st.plotly_chart(fig_hist_all, use_container_width=True)
            else: st.caption("Brak danych do histogramu odległości.")
        with col_dist2:
            if 'SHOT_TYPE' in filtered_data.columns and not filtered_data['SHOT_TYPE'].isnull().all():
                 shot_type_counts = filtered_data['SHOT_TYPE'].value_counts().reset_index()
                 shot_type_counts.columns = ['Typ Rzutu', 'Liczba']
                 fig_type = px.pie(shot_type_counts, names='Typ Rzutu', values='Liczba', title='Ogólny Rozkład Typów Rzutów')
                 st.plotly_chart(fig_type, use_container_width=True)
            else: st.caption("Brak danych dla rozkładu typów rzutów.")

            if 'ACTION_TYPE' in filtered_data.columns and not filtered_data['ACTION_TYPE'].isnull().all():
                 action_type_counts = filtered_data['ACTION_TYPE'].value_counts().head(15).reset_index()
                 action_type_counts.columns = ['Typ Akcji', 'Liczba']
                 fig_action = px.bar(action_type_counts, y='Typ Akcji', x='Liczba', orientation='h',
                                     title='Najczęstsze Typy Akcji (Top 15)', labels={'Liczba':'Liczba Rzutów', 'Typ Akcji':''})
                 fig_action.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                 st.plotly_chart(fig_action, use_container_width=True)
            else: st.caption("Brak danych dla rozkładu typów akcji.")


# --- Zakładka 2: Analiza Gracza ---
    with tab2:
        st.header(f"Analiza Gracza: {selected_player}")
        if selected_player:
            # 1. Filtrowanie danych dla wybranego gracza
            player_data = filter_data_by_player(selected_player, filtered_data)

            # Sprawdzenie, czy dane dla gracza istnieją po filtracji
            if not player_data.empty:

                # --- SEKCJA STATYSTYK PODSTAWOWYCH (NA GÓRZE) ---
                st.subheader("Statystyki Podstawowe")

                # --- Obliczanie Statystyk ---
                # Całkowita liczba rzutów
                total_shots = len(player_data)

                # Rzuty trafione i Skuteczność Ogółem
                made_shots = "N/A"
                shooting_pct = "N/A"
                if 'SHOT_MADE_FLAG' in player_data.columns:
                     made_flag_numeric = pd.to_numeric(player_data['SHOT_MADE_FLAG'], errors='coerce').dropna()
                     if not made_flag_numeric.empty:
                         made_shots = int(made_flag_numeric.sum())
                         if len(made_flag_numeric) > 0: # Użyj len() dla pewności
                              shooting_pct = (made_shots / len(made_flag_numeric)) * 100
                         else:
                              shooting_pct = 0.0
                     else:
                          made_shots = 0
                          shooting_pct = 0.0
                else:
                    st.caption("Brak kolumny 'SHOT_MADE_FLAG' do obliczenia skuteczności.")

                # Skuteczność 2PT i 3PT
                pct_2pt = "N/A"
                attempts_2pt_str = "(brak danych)"
                pct_3pt = "N/A"
                attempts_3pt_str = "(brak danych)"
                if 'SHOT_TYPE' in player_data.columns and not player_data['SHOT_TYPE'].isnull().all():
                    # 2PT%
                    shot_type_2pt = '2PT Field Goal' # !!! Sprawdź tę nazwę w swoich danych !!!
                    data_2pt = player_data[player_data['SHOT_TYPE'] == shot_type_2pt].copy()
                    made_flag_2pt = pd.to_numeric(data_2pt['SHOT_MADE_FLAG'], errors='coerce').dropna()
                    attempts_2pt = len(made_flag_2pt)
                    if attempts_2pt > 0:
                        made_2pt = int(made_flag_2pt.sum())
                        pct_2pt = (made_2pt / attempts_2pt) * 100
                        attempts_2pt_str = f"({made_2pt}/{attempts_2pt})"
                    else:
                        attempts_2pt_str = "(0 prób)"

                    # 3PT%
                    shot_type_3pt = '3PT Field Goal' # !!! Sprawdź tę nazwę w swoich danych !!!
                    data_3pt = player_data[player_data['SHOT_TYPE'] == shot_type_3pt].copy()
                    made_flag_3pt = pd.to_numeric(data_3pt['SHOT_MADE_FLAG'], errors='coerce').dropna()
                    attempts_3pt = len(made_flag_3pt)
                    if attempts_3pt > 0:
                        made_3pt = int(made_flag_3pt.sum())
                        pct_3pt = (made_3pt / attempts_3pt) * 100
                        attempts_3pt_str = f"({made_3pt}/{attempts_3pt})"
                    else:
                         attempts_3pt_str = "(0 prób)"
                else:
                     st.caption("Brak kolumny 'SHOT_TYPE' do obliczenia skuteczności 2PT/3PT.")

                # Średnia odległość
                avg_dist = "N/A"
                if 'SHOT_DISTANCE' in player_data.columns and pd.api.types.is_numeric_dtype(player_data['SHOT_DISTANCE']):
                    valid_distances = player_data['SHOT_DISTANCE'].dropna()
                    if not valid_distances.empty:
                        avg_dist = valid_distances.mean()
                else:
                    st.caption("Brak kolumny 'SHOT_DISTANCE' do obliczenia średniej odległości.")

                # --- Wyświetlanie Statystyk ---
                st.markdown("##### Statystyki Ogólne")
                col1, col2, col3 = st.columns(3)
                col1.metric("Całkowita liczba rzutów", total_shots)
                col2.metric("Rzuty trafione", f"{made_shots}" if isinstance(made_shots, (int)) else "N/A")
                col3.metric("Skuteczność rzutów", f"{shooting_pct:.1f}%" if isinstance(shooting_pct, (float, int)) else "N/A")

                st.markdown("##### Statystyki Szczegółowe")
                col4, col5, col6 = st.columns(3)
                col4.metric(f"Skuteczność rzutów za 2pkt {attempts_2pt_str}", f"{pct_2pt:.1f}%" if isinstance(pct_2pt, (float, int)) else "N/A")
                col5.metric(f"Skuteczność rzutów za 3pkt {attempts_3pt_str}", f"{pct_3pt:.1f}%" if isinstance(pct_3pt, (float, int)) else "N/A")
                col6.metric("Średnia odległość (stopy)", f"{avg_dist:.1f}" if isinstance(avg_dist, (float, int)) else "N/A")

                st.markdown("---") # Linia oddzielająca przed wizualizacjami

                # --- KONIEC SEKCJI STATYSTYK PODSTAWOWYCH ---


                # --- POCZĄTEK SEKCJI WIZUALIZACJI RZUTÓW ---
                st.subheader("Wizualizacje Rzutów")

                # 1. Główna mapa rzutów
                fig_p_chart = plot_shot_chart(player_data, selected_player, "Gracz")
                if fig_p_chart:
                    st.plotly_chart(fig_p_chart, use_container_width=True)
                else:
                    # Ta wiadomość pojawi się, jeśli nadal brakuje danych LOC_X/LOC_Y
                    st.warning("Nie można wygenerować mapy rzutów. Sprawdź, czy dane o lokalizacji (LOC_X, LOC_Y) są dostępne dla tego gracza.")

                # 2. Rozkład odległości i strefy (w rozwijanym panelu)
                with st.expander("Pokaż/Ukryj Rozkład Odległości i Strefy", expanded=False):
                    # Histogram odległości
                    fig_p_dist = plot_distance_histogram(player_data, selected_player, "Gracz")
                    if fig_p_dist:
                        st.plotly_chart(fig_p_dist, use_container_width=True)
                    else:
                        st.caption("Brak wystarczających danych do histogramu odległości (wymagana kolumna SHOT_DISTANCE).")

                    # Strefy rzutowe ('Hot Zones')
                    st.subheader("Strefy Rzutowe ('Hot Zones')")
                    # Ustaw próg minimalnej liczby rzutów w strefie
                    min_shots_for_zone_calc = 5
                    p_hot_zones = calculate_hot_zones(player_data, min_shots_in_zone=min_shots_for_zone_calc)
                    if p_hot_zones is not None and not p_hot_zones.empty:
                        fig_p_hot = plot_hot_zones_heatmap(p_hot_zones, selected_player, "Gracz")
                        if fig_p_hot:
                            st.plotly_chart(fig_p_hot, use_container_width=True)
                        else:
                            st.info("Nie można wygenerować mapy stref (problem z danymi po agregacji).")
                    else:
                        st.info(f"Brak wystarczających danych do analizy stref (wymagane min. {min_shots_for_zone_calc} rzutów w strefie oraz kolumny LOC_X, LOC_Y).")

                # 3. Analiza Czasowa
                st.subheader("Analiza Czasowa")
                # Skuteczność w kwartach
                fig_p_quarter = plot_player_quarter_eff(player_data, selected_player)
                if fig_p_quarter:
                    st.plotly_chart(fig_p_quarter, use_container_width=True)
                else:
                    st.info("Brak wystarczających danych do analizy skuteczności w kwartach (wymagana kolumna PERIOD i min. 5 rzutów na kwartę).")

                # Trend sezonowy (miesięczny)
                fig_p_trend = plot_player_season_trend(player_data, selected_player)
                if fig_p_trend:
                    st.plotly_chart(fig_p_trend, use_container_width=True)
                else:
                    st.info("Brak wystarczających danych do analizy trendu sezonowego (wymagana kolumna GAME_DATE i min. 10 rzutów na miesiąc).")

                # 4. Analiza wg Typu Akcji / Strefy Podstawowej
                st.subheader("Analiza wg Typu Akcji / Strefy")
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    # Skuteczność wg typu akcji
                    fig_p_action = plot_grouped_effectiveness(player_data, 'ACTION_TYPE', selected_player, "Gracz", top_n=10)
                    if fig_p_action:
                        st.plotly_chart(fig_p_action, use_container_width=True)
                    else:
                        st.caption("Brak wystarczających danych dla analizy wg typu akcji (wymagana kolumna ACTION_TYPE).")
                with col_g2:
                    # Skuteczność wg strefy podstawowej
                    fig_p_zone_basic = plot_grouped_effectiveness(player_data, 'SHOT_ZONE_BASIC', selected_player, "Gracz", top_n=7)
                    if fig_p_zone_basic:
                        st.plotly_chart(fig_p_zone_basic, use_container_width=True)
                    else:
                        st.caption("Brak wystarczających danych dla analizy wg strefy podstawowej (wymagana kolumna SHOT_ZONE_BASIC).")
                # --- KONIEC SEKCJI WIZUALIZACJI RZUTÓW ---

            else:
                # Ten kod jest wykonywany, jeśli player_data jest pusty po filtracji
                st.warning(f"Brak danych dla gracza '{selected_player}' przy obecnych filtrach ({selected_season_type}).")
        else:
            # Ten kod jest wykonywany, jeśli żaden gracz nie został wybrany
            st.info("Wybierz gracza z listy po lewej stronie.")


    # --- Zakładka 3: Porównanie Graczy ---
    with tab3:
        st.header("Porównanie Graczy")
        if len(selected_players_compare) >= 2:
            st.write(f"Porównujesz: {', '.join(selected_players_compare)}")
            compare_data_filtered = filtered_data[filtered_data['PLAYER_NAME'].isin(selected_players_compare)].copy()
            if not compare_data_filtered.empty:
                st.subheader("Skuteczność vs Odległość")
                fig_comp_eff_dist = plot_comparison_eff_distance(compare_data_filtered, selected_players_compare)
                if fig_comp_eff_dist:
                    st.plotly_chart(fig_comp_eff_dist, use_container_width=True)
                else:
                    st.caption("Nie można wygenerować wykresu skuteczności vs odległość przy obecnych danych/filtrach.")

                st.subheader("Mapy Rzutów")
                cols = st.columns(len(selected_players_compare))
                for i, player in enumerate(selected_players_compare):
                    with cols[i]:
                        st.markdown(f"**{player}**")
                        player_comp_data = compare_data_filtered[compare_data_filtered['PLAYER_NAME'] == player]
                        if not player_comp_data.empty:
                            fig_comp_chart = plot_shot_chart(player_comp_data, player, "Gracz")
                            if fig_comp_chart:
                                fig_comp_chart.update_layout(height=450, title="")
                                st.plotly_chart(fig_comp_chart, use_container_width=True, key=f"comp_chart_{player}")
                            else: st.caption("Błąd mapy.")
                        else: st.caption("Brak danych.")
            else:
                st.warning("Brak danych dla wybranych graczy do porównania przy obecnych filtrach.")
        else:
            st.info("Wybierz min. 2 graczy do porównania z listy po lewej stronie.")

    # --- Zakładka 4: Analiza Zespołowa ---
    with tab4:
        st.header(f"Analiza Zespołowa: {selected_team}")
        if selected_team:
            team_data = filter_data_by_team(selected_team, filtered_data)
            if not team_data.empty:
                st.subheader("Statystyki Podstawowe Zespołu")
                t_stats = get_basic_stats(team_data, selected_team, "Zespół")
                col1, col2, col3 = st.columns(3)
                col1.metric("Rzuty Zespołu", t_stats['total_shots'])
                col2.metric("Trafione Zespołu", t_stats['made_shots'])
                col3.metric("Skuteczność Zespołu", f"{t_stats['shooting_pct']:.1f}%" if isinstance(t_stats['shooting_pct'], (int, float)) else "N/A")

                st.subheader("Wizualizacje Rzutów Zespołu")
                fig_t_chart = plot_shot_chart(team_data, selected_team, "Zespół")
                if fig_t_chart: st.plotly_chart(fig_t_chart, use_container_width=True)
                else: st.warning("Nie można wygenerować mapy rzutów zespołu.")

                with st.expander("Pokaż/Ukryj Rozkład Odległości i Strefy Zespołu", expanded=False):
                    fig_t_dist = plot_distance_histogram(team_data, selected_team, "Zespół")
                    if fig_t_dist: st.plotly_chart(fig_t_dist, use_container_width=True)
                    else: st.caption("Brak danych do histogramu odległości.")

                    st.subheader("Strefy Rzutowe ('Hot Zones') Zespołu")
                    t_hot_zones = calculate_hot_zones(team_data, min_shots_in_zone=10)
                    if t_hot_zones is not None and not t_hot_zones.empty:
                        fig_t_hot = plot_hot_zones_heatmap(t_hot_zones, selected_team, "Zespół")
                        if fig_t_hot: st.plotly_chart(fig_t_hot, use_container_width=True)
                        else: st.info("Nie można wygenerować mapy stref zespołu.")
                    else:
                        st.info("Brak wystarczających danych do analizy stref dla tego zespołu.")

                st.subheader("Analiza Czasowa Zespołu")
                fig_t_quarter = plot_player_quarter_eff(team_data, selected_team, "Zespół")
                if fig_t_quarter: st.plotly_chart(fig_t_quarter, use_container_width=True)
                else: st.info("Brak wystarczających danych do analizy skuteczności zespołu w kwartach.")

                st.subheader("Analiza Zespołu wg Typu Akcji / Strefy")
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                     fig_t_action = plot_grouped_effectiveness(team_data, 'ACTION_TYPE', selected_team, "Zespół", top_n=10)
                     if fig_t_action: st.plotly_chart(fig_t_action, use_container_width=True)
                     else: st.caption("Brak danych zespołu dla analizy wg typu akcji.")
                with col_g2:
                     fig_t_zone_basic = plot_grouped_effectiveness(team_data, 'SHOT_ZONE_BASIC', selected_team, "Zespół", top_n=7)
                     if fig_t_zone_basic: st.plotly_chart(fig_t_zone_basic, use_container_width=True)
                     else: st.caption("Brak danych zespołu dla analizy wg strefy podstawowej.")

            else:
                st.warning(f"Brak danych dla drużyny '{selected_team}' przy obecnych filtrach ({selected_season_type}).")
        else:
            st.info("Wybierz drużynę z listy po lewej stronie.")

    # --- Zakładka 5: Model Predykcji (KNN) ---
    with tab5:
        st.header(f"Model Predykcji Rzutów (KNN) dla: {selected_player}")
        if selected_player:
            player_model_data = filter_data_by_player(selected_player, filtered_data)
            if not player_model_data.empty:
                st.markdown("""
                    Model K-Najbliższych Sąsiadów (KNN) próbuje przewidzieć **prawdopodobieństwo trafienia** rzutu na podstawie jego lokalizacji i odległości.
                    Obecnie używane cechy: `LOC_X`, `LOC_Y`, `SHOT_DISTANCE`.

                    **Potencjalne przyszłe ulepszenia:** Można by włączyć dodatkowe cechy, takie jak `ACTION_TYPE`, `SHOT_TYPE`, `SHOT_ZONE_BASIC`, `PERIOD`, `GAME_TIME_SEC`,
                    jednak wymagałoby to odpowiedniego przetworzenia (np. kodowania One-Hot dla cech kategorycznych) i mogłoby zwiększyć złożoność modelu oraz czas treningu.
                    *(Implementacja modelu pozostaje jako ćwiczenie - użyj kodu z pierwotnej wersji lub rozbuduj go o nowe cechy).*
                """)

                model_features = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE']
                if all(feat in player_model_data.columns for feat in model_features):
                    player_model_data_cleaned = player_model_data[model_features + ['SHOT_MADE_FLAG']].dropna()
                    # Upewnij się, że SHOT_MADE_FLAG jest numeryczny przed konwersją na int
                    player_model_data_cleaned['SHOT_MADE_FLAG'] = pd.to_numeric(player_model_data_cleaned['SHOT_MADE_FLAG'], errors='coerce')
                    player_model_data_cleaned = player_model_data_cleaned.dropna(subset=['SHOT_MADE_FLAG'])
                    player_model_data_cleaned['SHOT_MADE_FLAG'] = player_model_data_cleaned['SHOT_MADE_FLAG'].astype(int)

                    if len(player_model_data_cleaned) > 50 and player_model_data_cleaned['SHOT_MADE_FLAG'].nunique() == 2:
                        k_neighbors = st.slider("Liczba sąsiadów (k):", 3, 25, 5, 2, key='knn_k_v3')
                        if st.button(f"Trenuj model KNN dla {selected_player}", key='train_model_button_v3'):
                             with st.spinner("Trenowanie modelu... (Uwaga: Funkcjonalność niezaimplementowana w tej wersji)"):
                                st.info("Funkcjonalność trenowania modelu KNN i predykcji nie została w pełni zaimplementowana w tym przykładzie.")
                                st.warning("Aby zobaczyć działanie modelu, należy wkleić lub zaimplementować funkcje: `train_knn_model`, `predict_shot_probability`, `plot_probability_heatmap` oraz kod wyświetlający ich wyniki.")

                    else:
                         st.warning(f"Niewystarczająca ilość danych (wymagane > 50 rzutów z dwoma wynikami: {len(player_model_data_cleaned)} rzutów, {player_model_data_cleaned['SHOT_MADE_FLAG'].nunique()} unikalnych wyników) dla gracza '{selected_player}' do zbudowania sensownego modelu przy obecnych filtrach.")
                else:
                    st.warning(f"Brak wymaganych kolumn {model_features} do trenowania modelu dla gracza '{selected_player}'.")
            else:
                st.warning(f"Brak danych dla gracza '{selected_player}' do trenowania modelu przy obecnych filtrach.")
        else:
            st.info("Wybierz gracza, aby trenować model.")

else:
    st.error("Nie udało się załadować danych. Sprawdź ścieżkę do pliku CSV i jego format.")

# --- Stopka Sidebar ---
st.sidebar.markdown("---")
st.sidebar.info("Rozszerzona Aplikacja Streamlit - Analiza NBA")
try:
    now_local = pd.Timestamp.now(tz='Europe/Warsaw')
    timestamp_str = now_local.strftime('%Y-%m-%d %H:%M:%S %Z')
except Exception as e:
    # Lepszy fallback
    st.sidebar.warning(f"Nie można ustawić strefy czasowej 'Europe/Warsaw': {e}")
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S (czas lokalny serwera)')
st.sidebar.markdown(f"Czas: {timestamp_str}")