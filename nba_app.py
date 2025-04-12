# nba_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt # Potrzebne do rysowania boiska na wykresach Plotly
import warnings
import os

warnings.filterwarnings('ignore')

# --- Konfiguracja Początkowa ---
st.set_page_config(layout="wide", page_title="Analiza Rzutów NBA 2023-24")

# --- Ścieżka do pliku CSV ---
# !!! WAŻNE: Zmień tę ścieżkę, jeśli Twój plik CSV ma inną nazwę lub lokalizację !!!
CSV_FILE_PATH = 'nba_player_shooting_data_2023_24.csv' # Przykład nazwy - dostosuj!

# --- Funkcje Pomocnicze ---

@st.cache_data # Cache'owanie wczytanych danych
def load_shooting_data(file_path):
    """Wczytuje dane o rzutach graczy NBA."""
    try:
        data = pd.read_csv(file_path)
        st.success(f"Wczytano dane z {file_path}. Wymiary: {data.shape}")
        # Podstawowe mapowanie kolumn (jeśli potrzebne, bazując na oryginalnym kodzie)
        required_columns_map = {
            'PLAYER': 'PLAYER_NAME', 'SHOT RESULT': 'SHOT_MADE_FLAG', 'SHOT TYPE': 'SHOT_TYPE',
            'SHOT ZONE BASIC': 'SHOT_ZONE_AREA', 'SHOT ZONE RANGE': 'SHOT_ZONE_RANGE',
            'SHOT DISTANCE': 'SHOT_DISTANCE', 'LOC_X': 'LOC_X', 'LOC_Y': 'LOC_Y'
        }
        rename_map = {old: new for old, new in required_columns_map.items() if old in data.columns and new not in data.columns}
        if rename_map:
             st.info(f"Zmieniam nazwy kolumn: {rename_map}")
             data = data.rename(columns=rename_map)

        # Konwersja SHOT_MADE_FLAG na 0/1
        if 'SHOT_MADE_FLAG' in data.columns and data['SHOT_MADE_FLAG'].dtype == 'object':
             made_values = ['Made', 'made', '1', 1]
             missed_values = ['Missed', 'missed', '0', 0]
             unique_vals = data['SHOT_MADE_FLAG'].unique()
             if any(val in unique_vals for val in made_values) or any(val in unique_vals for val in missed_values):
                data['SHOT_MADE_FLAG'] = data['SHOT_MADE_FLAG'].apply(lambda x: 1 if x in made_values else 0)
             else:
                 try:
                     data['SHOT_MADE_FLAG'] = pd.to_numeric(data['SHOT_MADE_FLAG'])
                 except ValueError:
                     st.error(f"Nie można przekonwertować 'SHOT_MADE_FLAG' na typ numeryczny. Sprawdź dane.")
                     return pd.DataFrame()
        elif 'SHOT_MADE_FLAG' in data.columns:
            try:
                data['SHOT_MADE_FLAG'] = data['SHOT_MADE_FLAG'].astype(int)
            except ValueError:
                 st.error(f"Nie można przekonwertować 'SHOT_MADE_FLAG' na typ całkowity. Sprawdź dane.")
                 return pd.DataFrame()

        return data
    except FileNotFoundError:
        st.error(f"Błąd: Nie znaleziono pliku {file_path}.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Błąd podczas wczytywania danych: {e}")
        return pd.DataFrame()

def add_court_shapes(fig):
    """Dodaje kształty boiska do figury Plotly."""
    fig.add_shape(type="rect", x0=-250, y0=-47.5, x1=250, y1=422.5, line=dict(color="black", width=1))
    fig.add_shape(type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, line=dict(color="orange", width=2), fillcolor="orange", layer='above')
    fig.add_shape(type="line", x0=-30, y0=-7.5, x1=30, y1=-7.5, line=dict(color="black", width=1))
    fig.add_shape(type="path", path=f"M -220 -47.5 L -220 142.5 C -220 280, 0 350, 220 142.5 L 220 -47.5", line=dict(color="black", width=1))
    fig.update_xaxes(range=[-300, 300])
    fig.update_yaxes(range=[-100, 500])
    return fig

@st.cache_data
def analyze_player_data(player_name, data):
    """Analizuje dane rzutowe dla wybranego gracza."""
    if 'PLAYER_NAME' not in data.columns: return None
    player_shots = data[data['PLAYER_NAME'] == player_name].copy()
    if player_shots.empty: return None
    return player_shots

@st.cache_data
def get_player_stats(player_shots):
    """Oblicza podstawowe statystyki gracza."""
    stats = {'total_shots': len(player_shots), 'made_shots': "N/A", 'shooting_pct': "N/A"}
    if 'SHOT_MADE_FLAG' in player_shots.columns and stats['total_shots'] > 0:
        if pd.api.types.is_numeric_dtype(player_shots['SHOT_MADE_FLAG']):
            stats['made_shots'] = int(player_shots['SHOT_MADE_FLAG'].sum())
            stats['shooting_pct'] = (stats['made_shots'] / stats['total_shots']) * 100
        else: stats['made_shots'] = "Błąd typu danych"
    return stats

@st.cache_data
def plot_player_distance_hist(player_shots, player_name):
    """Tworzy interaktywny histogram odległości rzutów gracza."""
    if 'SHOT_DISTANCE' in player_shots.columns and 'SHOT_MADE_FLAG' in player_shots.columns:
        player_shots_hist = player_shots.copy()
        if not pd.api.types.is_numeric_dtype(player_shots_hist['SHOT_MADE_FLAG']):
             try: player_shots_hist['SHOT_MADE_FLAG'] = player_shots_hist['SHOT_MADE_FLAG'].astype(int)
             except ValueError: return None
        fig = px.histogram(player_shots_hist, x='SHOT_DISTANCE', color='SHOT_MADE_FLAG', barmode='stack', nbins=30,
                           title=f'Rozkład odległości rzutów - {player_name}', labels={'SHOT_DISTANCE': 'Odległość (stopy)', 'SHOT_MADE_FLAG': 'Wynik (0=Niecelny, 1=Celny)'},
                           category_orders={"SHOT_MADE_FLAG": [0, 1]})
        return fig
    return None

@st.cache_data
def plot_player_shot_chart(player_shots, player_name):
    """Tworzy interaktywną mapę rzutów gracza."""
    if 'LOC_X' in player_shots.columns and 'LOC_Y' in player_shots.columns:
        player_shots_chart = player_shots.copy()
        color_col, color_discrete_map, category_orders_chart = None, None, None
        if 'SHOT_MADE_FLAG' in player_shots_chart.columns:
             if not pd.api.types.is_numeric_dtype(player_shots_chart['SHOT_MADE_FLAG']):
                 try: player_shots_chart['SHOT_MADE_FLAG'] = player_shots_chart['SHOT_MADE_FLAG'].astype(int)
                 except ValueError: pass # Kontynuuj bez kolorowania
             if pd.api.types.is_numeric_dtype(player_shots_chart['SHOT_MADE_FLAG']): # Sprawdź ponownie po próbie konwersji
                color_col, color_discrete_map, category_orders_chart = 'SHOT_MADE_FLAG', {0: 'red', 1: 'green'}, {"SHOT_MADE_FLAG": [0, 1]}
        hover_data_chart = [col for col in ['SHOT_DISTANCE', 'SHOT_TYPE'] if col in player_shots_chart.columns]
        fig = px.scatter(player_shots_chart, x='LOC_X', y='LOC_Y', color=color_col, title=f'Mapa rzutów - {player_name}',
                         labels={'LOC_X': 'Pozycja X', 'LOC_Y': 'Pozycja Y', 'SHOT_MADE_FLAG': 'Wynik'}, hover_data=hover_data_chart if hover_data_chart else None,
                         category_orders=category_orders_chart, color_discrete_map=color_discrete_map, opacity=0.7)
        fig = add_court_shapes(fig)
        fig.update_layout(height=600)
        return fig
    return None

@st.cache_data
def plot_player_type_eff(player_shots, player_name):
    """Tworzy wykres skuteczności wg typu rzutu."""
    if 'SHOT_TYPE' in player_shots.columns and 'SHOT_MADE_FLAG' in player_shots.columns:
        player_shots_eff = player_shots.copy()
        if not pd.api.types.is_numeric_dtype(player_shots_eff['SHOT_MADE_FLAG']):
            try: player_shots_eff['SHOT_MADE_FLAG'] = player_shots_eff['SHOT_MADE_FLAG'].astype(int)
            except ValueError: return None
        player_shots_eff = player_shots_eff.dropna(subset=['SHOT_MADE_FLAG'])
        if player_shots_eff.empty: return None
        shot_type_pct = player_shots_eff.groupby('SHOT_TYPE')['SHOT_MADE_FLAG'].mean().reset_index()
        shot_type_pct['SHOT_MADE_FLAG'] *= 100
        fig = px.bar(shot_type_pct, x='SHOT_TYPE', y='SHOT_MADE_FLAG', title=f'Skuteczność wg typu rzutu - {player_name}',
                     labels={'SHOT_TYPE': 'Typ rzutu', 'SHOT_MADE_FLAG': 'Skuteczność (%)'})
        fig.update_layout(yaxis_range=[0, 100])
        return fig
    return None

@st.cache_data
def calculate_hot_zones(player_shots, min_shots_in_zone=5):
    """Oblicza statystyki dla stref rzutowych."""
    if 'LOC_X' in player_shots.columns and 'LOC_Y' in player_shots.columns and 'SHOT_MADE_FLAG' in player_shots.columns:
        player_shots_zones = player_shots.copy()
        if not pd.api.types.is_numeric_dtype(player_shots_zones['SHOT_MADE_FLAG']):
            try: player_shots_zones['SHOT_MADE_FLAG'] = player_shots_zones['SHOT_MADE_FLAG'].astype(int)
            except ValueError: return pd.DataFrame()
        player_shots_zones = player_shots_zones.dropna(subset=['SHOT_MADE_FLAG'])
        if player_shots_zones.empty: return pd.DataFrame()
        player_shots_zones['zone_x'] = pd.cut(player_shots_zones['LOC_X'], bins=np.linspace(-300, 300, 11))
        player_shots_zones['zone_y'] = pd.cut(player_shots_zones['LOC_Y'], bins=np.linspace(-50, 450, 11))
        zones = player_shots_zones.groupby(['zone_x', 'zone_y'], observed=False).agg(
            total_shots=('SHOT_MADE_FLAG', 'count'), made_shots=('SHOT_MADE_FLAG', 'sum'), percentage=('SHOT_MADE_FLAG', 'mean')
        ).reset_index()
        zones = zones[zones['total_shots'] >= min_shots_in_zone].copy()
        zones['percentage'] *= 100
        zones['x_center'] = zones['zone_x'].apply(lambda x: x.mid if isinstance(x, pd.Interval) else None)
        zones['y_center'] = zones['zone_y'].apply(lambda x: x.mid if isinstance(x, pd.Interval) else None)
        zones = zones.dropna(subset=['x_center', 'y_center'])
        return zones
    return pd.DataFrame()

# ===========================================================
# === POPRAWIONA FUNKCJA plot_hot_zones_heatmap           ===
# ===========================================================
@st.cache_data
def plot_hot_zones_heatmap(hot_zones, player_name):
    """Tworzy interaktywną mapę ciepła stref rzutowych."""
    required_cols = ['x_center', 'y_center', 'total_shots', 'percentage', 'made_shots']
    if not hot_zones.empty and all(col in hot_zones.columns for col in required_cols):
        for col in required_cols: # Sprawdzenie i próba konwersji typów
            if not pd.api.types.is_numeric_dtype(hot_zones[col]):
                 try: hot_zones[col] = pd.to_numeric(hot_zones[col])
                 except ValueError:
                     st.error(f"Nie można przekonwertować kolumny '{col}' na typ numeryczny dla mapy ciepła.")
                     return None
        hot_zones = hot_zones.dropna(subset=required_cols)
        if hot_zones.empty:
             st.info("Brak danych po przetworzeniu dla mapy ciepła stref.")
             return None

        min_shots_in_zone = int(hot_zones["total_shots"].min())
        min_pct, max_pct = hot_zones['percentage'].min(), hot_zones['percentage'].max()
        color_range = [max(0, min_pct - 5), min(100, max_pct + 5)]

        fig = px.scatter(hot_zones, x='x_center', y='y_center',
                         size='total_shots', color='percentage',
                         color_continuous_scale=px.colors.diverging.RdYlGn,
                         size_max=40, range_color=color_range,
                         title=f'Skuteczność stref rzutowych (min. {min_shots_in_zone} rzutów) - {player_name}',
                         labels={'x_center': 'Pozycja X (środek strefy)', 'y_center': 'Pozycja Y (środek strefy)',
                                 'total_shots': 'Liczba rzutów', 'percentage': 'Skuteczność (%)'},
                         custom_data=['made_shots']
                        )
        # Zastosowanie hovertemplate za pomocą update_traces
        fig.update_traces(
            hovertemplate="<b>Strefa X (środek):</b> %{x:.1f}<br>" +
                          "<b>Strefa Y (środek):</b> %{y:.1f}<br>" +
                          "<b>Liczba rzutów:</b> %{marker.size}<br>" +
                          "<b>Trafione:</b> %{customdata[0]}<br>" +
                          "<b>Skuteczność:</b> %{marker.color:.1f}%<extra></extra>"
        )
        fig = add_court_shapes(fig)
        fig.update_layout(height=600)
        return fig
    else: # Komunikaty informacyjne/ostrzegawcze
         if hot_zones.empty: st.info("Brak danych do utworzenia mapy ciepła stref (po filtrowaniu lub konwersji).")
         else: st.warning(f"Brak wymaganych kolumn do utworzenia mapy ciepła stref: {[col for col in required_cols if col not in hot_zones.columns]}")
         return None
# ===========================================================
# === KONIEC POPRAWIONEJ FUNKCJI                          ===
# ===========================================================

@st.cache_data
def plot_comparison_eff_distance(compare_data, selected_players):
    """Porównuje skuteczność graczy względem odległości."""
    required_cols = ['SHOT_DISTANCE', 'SHOT_MADE_FLAG', 'PLAYER_NAME']
    if all(col in compare_data.columns for col in required_cols):
        compare_data_eff = compare_data.copy()
        if not pd.api.types.is_numeric_dtype(compare_data_eff['SHOT_MADE_FLAG']):
            try: compare_data_eff['SHOT_MADE_FLAG'] = compare_data_eff['SHOT_MADE_FLAG'].astype(int)
            except ValueError: return None
        compare_data_eff = compare_data_eff.dropna(subset=['SHOT_MADE_FLAG', 'SHOT_DISTANCE'])
        if compare_data_eff.empty: return None
        max_dist = compare_data_eff['SHOT_DISTANCE'].max()
        if pd.isna(max_dist) or max_dist <= 0: return None
        distance_bins = np.arange(0, int(max_dist) + 5, 3)
        compare_data_eff['distance_bin'] = pd.cut(compare_data_eff['SHOT_DISTANCE'], bins=distance_bins)
        effectiveness = compare_data_eff.groupby(['PLAYER_NAME', 'distance_bin'], observed=False)['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
        effectiveness['mean'] *= 100
        effectiveness = effectiveness[effectiveness['count'] >= 5]
        effectiveness['distance_mid'] = effectiveness['distance_bin'].apply(lambda x: x.mid if isinstance(x, pd.Interval) else None)
        effectiveness = effectiveness.dropna(subset=['distance_mid'])
        if not effectiveness.empty:
             max_eff_val = effectiveness['mean'].max()
             yaxis_range = [0, min(100, max_eff_val + 10)]
             fig = px.line(effectiveness, x='distance_mid', y='mean', color='PLAYER_NAME', title='Porównanie skuteczności rzutów względem odległości (min. 5 rzutów w przedziale)',
                           labels={'distance_mid': 'Środek przedziału odległości (stopy)', 'mean': 'Skuteczność (%)', 'PLAYER_NAME': 'Gracz'}, markers=True)
             fig.update_layout(yaxis_range=yaxis_range)
             return fig
    return None

# @st.cache_resource # Można odkomentować dla optymalizacji
def train_knn_model(player_shots, features=['LOC_X', 'LOC_Y', 'SHOT_DISTANCE'], k=5):
    """Trenuje model KNN dla danych gracza."""
    required_features_model = features + ['SHOT_MADE_FLAG']
    if not all(col in player_shots.columns for col in required_features_model):
        st.warning(f"Brak wymaganych kolumn do modelowania: {[col for col in required_features_model if col not in player_shots.columns]}")
        return None, None, None, None, None, None
    player_model_data = player_shots[required_features_model].copy()
    for feat in features: # Sprawdzenie typów cech
        if not pd.api.types.is_numeric_dtype(player_model_data[feat]):
            try: player_model_data[feat] = pd.to_numeric(player_model_data[feat])
            except ValueError:
                 st.error(f"Nie można przekonwertować kolumny '{feat}' na typ numeryczny dla modelowania.")
                 return None, None, None, None, None, None
    if not pd.api.types.is_integer_dtype(player_model_data['SHOT_MADE_FLAG']): # Sprawdzenie typu zmiennej celu
        try: player_model_data['SHOT_MADE_FLAG'] = player_model_data['SHOT_MADE_FLAG'].astype(int)
        except ValueError:
            st.error("Nie można przekonwertować 'SHOT_MADE_FLAG' na typ całkowity dla modelowania.")
            return None, None, None, None, None, None
    player_model_data = player_model_data.dropna()
    if len(player_model_data) < 20 or len(player_model_data['SHOT_MADE_FLAG'].unique()) < 2:
         st.warning(f"Niewystarczająca ilość danych ({len(player_model_data)}) lub tylko jedna klasa wyniku rzutu.")
         return None, None, None, None, None, None
    X, y = player_model_data[features], player_model_data['SHOT_MADE_FLAG']
    try: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    except ValueError: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    if len(X_train) == 0 or len(X_test) == 0: return None, None, None, None, None, None
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    try: report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    except ValueError: report = {"Błąd": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}}
    cm = confusion_matrix(y_test, y_pred)
    labels = np.unique(y) # Pobierz unikalne etykiety z oryginalnego y
    if cm.shape != (2, 2) and len(labels) == 2: # Uzupełnienie macierzy, jeśli brakuje klas
         cm_full = np.zeros((2, 2), dtype=int)
         label_map = {label: i for i, label in enumerate(sorted(labels))}
         present_labels = np.unique(y_test)
         present_preds = np.unique(y_pred)
         for true_label_val in present_labels:
             for pred_label_val in present_preds:
                 true_idx = label_map.get(true_label_val, -1)
                 pred_idx = label_map.get(pred_label_val, -1)
                 if true_idx != -1 and pred_idx != -1:
                      # Find the original indices in the possibly reduced cm
                      original_true_idx = np.where(np.unique(y_test) == true_label_val)[0]
                      original_pred_idx = np.where(np.unique(y_pred) == pred_label_val)[0]
                      if len(original_true_idx) > 0 and len(original_pred_idx) > 0:
                           cm_full[true_idx, pred_idx] = cm[original_true_idx[0], original_pred_idx[0]]
         cm = cm_full

    return knn, scaler, features, accuracy, report, cm


@st.cache_data
def predict_shot_probability(_knn_model, _scaler, features, spacing=15.0):
    """Tworzy siatkę boiska i przewiduje prawdopodobieństwo trafienia."""
    x_range = np.arange(-280, 280 + spacing, spacing)
    y_range = np.arange(-60, 400 + spacing, spacing)
    grid_points = [{'LOC_X': x, 'LOC_Y': y, 'SHOT_DISTANCE': np.sqrt(x**2 + y**2)} for x in x_range for y in y_range]
    if not grid_points: return pd.DataFrame()
    grid_df = pd.DataFrame(grid_points)
    try: grid_df_features = grid_df[features]
    except KeyError as e:
        st.error(f"Brak kolumny '{e}' w danych siatki, wymaganej przez model.")
        return pd.DataFrame()
    try: grid_scaled = _scaler.transform(grid_df_features)
    except Exception as e:
        st.error(f"Błąd podczas skalowania danych siatki: {e}")
        return pd.DataFrame()
    try:
        if hasattr(_knn_model, "predict_proba") and (1 in getattr(_knn_model, 'classes_', [])):
             proba_index = np.where(_knn_model.classes_ == 1)[0][0]
             grid_probs = _knn_model.predict_proba(grid_scaled)[:, proba_index]
             grid_df['shot_probability'] = grid_probs
             return grid_df
        else: return grid_df # Zwróć siatkę bez prawdopodobieństw, jeśli nie można ich obliczyć
    except Exception as e:
        st.error(f"Błąd podczas przewidywania prawdopodobieństwa: {e}")
        return pd.DataFrame()

@st.cache_data
def plot_probability_heatmap(grid_df, player_name):
    """Tworzy interaktywną mapę ciepła prawdopodobieństwa trafienia."""
    if not grid_df.empty and 'shot_probability' in grid_df.columns:
        grid_df_plot = grid_df.dropna(subset=['shot_probability'])
        if grid_df_plot.empty: return None
        fig = px.scatter(grid_df_plot, x='LOC_X', y='LOC_Y', color='shot_probability', color_continuous_scale=px.colors.diverging.RdYlGn,
                         range_color=[0, 1], title=f'Mapa Prawdopodobieństwa Trafienia (Model KNN) - {player_name}')
        fig = add_court_shapes(fig)
        fig.update_layout(coloraxis_colorbar=dict(title="Prawdopodob."), height=600)
        fig.update_traces(mode='markers', marker=dict(size=8, opacity=0.7))
        return fig
    elif not grid_df.empty: st.info("Brak kolumny 'shot_probability' w siatce.")
    else: st.info("Pusta siatka danych dla mapy prawdopodobieństwa.")
    return None


# --- Główna część aplikacji Streamlit ---
st.title("🏀 Interaktywna Analiza Rzutów Graczy NBA (Sezon 2023-24)")
shooting_data = load_shooting_data(CSV_FILE_PATH)

if not shooting_data.empty:
    st.sidebar.header("Opcje Analizy")
    available_players = sorted(shooting_data['PLAYER_NAME'].dropna().unique()) if 'PLAYER_NAME' in shooting_data.columns else []
    selected_player = st.sidebar.selectbox("Wybierz gracza do analizy:", options=available_players, index=0 if available_players else None, key='player_select', disabled=not available_players)
    selected_players_compare = st.sidebar.multiselect("Wybierz graczy do porównania (2-5):", options=available_players, default=available_players[:2] if len(available_players) >= 2 else [], max_selections=5, key='player_multi_select', disabled=not available_players)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Przegląd Danych", "👤 Analiza Gracza", "🆚 Porównanie Graczy", "🎯 Model Predykcji Rzutów (KNN)"])

    with tab1:
        st.header("Ogólny Przegląd Danych o Rzutach")
        col1, col2 = st.columns(2)
        with col1:
             st.dataframe(shooting_data.head())
             st.write(f"Wymiary danych: {shooting_data.shape[0]} rzutów, {shooting_data.shape[1]} kolumn.")
             st.write("Brakujące wartości (suma):")
             st.dataframe(shooting_data.isnull().sum().reset_index().rename(columns={0:'Liczba braków', 'index':'Kolumna'}))
        with col2: st.dataframe(shooting_data.describe())
        st.subheader("Ogólna Skuteczność i Rozkład Rzutów")
        col3, col4 = st.columns(2)
        with col3:
            if 'SHOT_MADE_FLAG' in shooting_data.columns and pd.api.types.is_numeric_dtype(shooting_data['SHOT_MADE_FLAG']):
                shot_made_flag_no_nan = shooting_data['SHOT_MADE_FLAG'].dropna()
                if not shot_made_flag_no_nan.empty:
                    shot_success = shot_made_flag_no_nan.value_counts(normalize=True) * 100
                    st.metric("Średnia Skuteczność", f"{shot_success.get(1, 0):.1f}%")
                    st.metric("Rzuty Niecelne", f"{shot_success.get(0, 0):.1f}%")
                    if 'SHOT_DISTANCE' in shooting_data.columns and pd.api.types.is_numeric_dtype(shooting_data['SHOT_DISTANCE']):
                        fig_hist_all = px.histogram(shooting_data.dropna(subset=['SHOT_MADE_FLAG', 'SHOT_DISTANCE']), x='SHOT_DISTANCE', color='SHOT_MADE_FLAG',
                                                    barmode='stack', title='Ogólny Rozkład Odległości', labels={'SHOT_DISTANCE': 'Odległość (stopy)'}, nbins=50, category_orders={"SHOT_MADE_FLAG": [0, 1]})
                        st.plotly_chart(fig_hist_all, use_container_width=True)
        with col4:
            if 'SHOT_TYPE' in shooting_data.columns:
                 shot_type_no_nan = shooting_data['SHOT_TYPE'].dropna()
                 if not shot_type_no_nan.empty:
                     type_counts = shot_type_no_nan.value_counts().reset_index()
                     type_counts.columns = ['Typ Rzutu', 'Liczba']
                     fig_type = px.pie(type_counts, names='Typ Rzutu', values='Liczba', title='Rozkład Typów Rzutów')
                     st.plotly_chart(fig_type, use_container_width=True)

    with tab2:
        st.header(f"Analiza Gracza: {selected_player}")
        if selected_player:
             player_shots_data = analyze_player_data(selected_player, shooting_data)
             if player_shots_data is not None:
                 stats = get_player_stats(player_shots_data)
                 st.subheader("Statystyki")
                 col1, col2, col3 = st.columns(3)
                 col1.metric("Rzuty", stats['total_shots'])
                 col2.metric("Trafione", stats['made_shots'])
                 col3.metric("Skuteczność", f"{stats['shooting_pct']:.1f}%" if isinstance(stats['shooting_pct'], (int, float)) else "N/A")
                 st.subheader("Wizualizacje")
                 fig_dist = plot_player_distance_hist(player_shots_data, selected_player)
                 if fig_dist: st.plotly_chart(fig_dist, use_container_width=True)
                 fig_chart = plot_player_shot_chart(player_shots_data, selected_player)
                 if fig_chart: st.plotly_chart(fig_chart, use_container_width=True)
                 fig_type_eff = plot_player_type_eff(player_shots_data, selected_player)
                 if fig_type_eff: st.plotly_chart(fig_type_eff, use_container_width=True)
                 st.subheader("Strefy Rzutowe ('Hot Zones')")
                 hot_zones_df = calculate_hot_zones(player_shots_data)
                 if hot_zones_df is not None and not hot_zones_df.empty:
                     fig_hot_zones = plot_hot_zones_heatmap(hot_zones_df, selected_player)
                     if fig_hot_zones: st.plotly_chart(fig_hot_zones, use_container_width=True)
                 else: st.info("Brak danych do analizy stref.")
             else: st.warning(f"Brak danych dla gracza {selected_player}.")
        else: st.info("Wybierz gracza.")

    with tab3:
        st.header("Porównanie Graczy")
        if len(selected_players_compare) >= 2:
             st.write(f"Porównujesz: {', '.join(selected_players_compare)}")
             compare_data_filtered = shooting_data[shooting_data['PLAYER_NAME'].isin(selected_players_compare)].copy()
             if not compare_data_filtered.empty:
                 st.subheader("Skuteczność vs Odległość")
                 fig_comp_eff_dist = plot_comparison_eff_distance(compare_data_filtered, selected_players_compare)
                 if fig_comp_eff_dist: st.plotly_chart(fig_comp_eff_dist, use_container_width=True)
                 st.subheader("Mapy Rzutów")
                 cols = st.columns(len(selected_players_compare))
                 for i, player in enumerate(selected_players_compare):
                     with cols[i]:
                         st.markdown(f"**{player}**")
                         player_comp_data = compare_data_filtered[compare_data_filtered['PLAYER_NAME'] == player]
                         if not player_comp_data.empty:
                             fig_comp_chart = plot_player_shot_chart(player_comp_data, player)
                             if fig_comp_chart:
                                 fig_comp_chart.update_layout(height=450, title="")
                                 st.plotly_chart(fig_comp_chart, use_container_width=True, key=f"comp_chart_{player}")
                         else: st.caption("Brak danych.")
             else: st.warning("Brak danych dla wybranych graczy.")
        else: st.info("Wybierz min. 2 graczy.")

    with tab4:
        st.header(f"Model Predykcji Rzutów (KNN) dla: {selected_player}")
        if selected_player:
             player_shots_data_model = analyze_player_data(selected_player, shooting_data)
             if player_shots_data_model is not None:
                 st.markdown("Model K-Najbliższych Sąsiadów (KNN) przewiduje prawdopodobieństwo trafienia rzutu. Cechy: `LOC_X`, `LOC_Y`, `SHOT_DISTANCE`.")
                 model_features = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE']
                 if all(feat in player_shots_data_model.columns for feat in model_features):
                     k_neighbors = st.slider("Liczba sąsiadów (k):", 3, 25, 5, 2, key='knn_k')
                     if st.button(f"Trenuj model KNN dla {selected_player}", key='train_model_button'):
                         with st.spinner("Trenowanie modelu..."):
                             knn_model, scaler_model, features_model, accuracy_model, report_model, cm_model = train_knn_model(
                                 player_shots_data_model, features=model_features, k=k_neighbors)
                             if knn_model and accuracy_model is not None:
                                 st.success(f"Model KNN (k={k_neighbors}) wytrenowany!")
                                 st.subheader("Ocena Modelu")
                                 col1, col2 = st.columns(2)
                                 with col1:
                                     st.metric("Dokładność", f"{accuracy_model:.3f}")
                                     st.text("Raport Klasyfikacji:")
                                     st.dataframe(pd.DataFrame(report_model).transpose().round(2))
                                 with col2:
                                     st.text("Macierz Pomyłek:")
                                     cm_labels = ['Niecelny (0)', 'Celny (1)']
                                     if cm_model is not None and cm_model.shape == (2, 2):
                                         cm_df = pd.DataFrame(cm_model, index=cm_labels, columns=[f"Pred. {l}" for l in cm_labels])
                                         fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", color_continuous_scale='Blues', title="Macierz Pomyłek")
                                         fig_cm.update_layout(xaxis_title="Przewidywana", yaxis_title="Rzeczywista")
                                         st.plotly_chart(fig_cm, use_container_width=True)
                                     else: st.warning("Problem z macierzą pomyłek.")
                                 st.subheader("Mapa Prawdopodobieństwa Trafienia")
                                 prediction_grid = predict_shot_probability(knn_model, scaler_model, features_model, spacing=15)
                                 if prediction_grid is not None:
                                     fig_prob_map = plot_probability_heatmap(prediction_grid, selected_player)
                                     if fig_prob_map: st.plotly_chart(fig_prob_map, use_container_width=True)
                             else: st.error("Nie udało się wytrenować modelu.")
                 else: st.warning(f"Brak wymaganych kolumn do modelu dla {selected_player}: {[f for f in model_features if f not in player_shots_data_model.columns]}.")
             else: st.warning(f"Brak danych dla {selected_player} do modelu.")
        else: st.info("Wybierz gracza.")

else:
    st.error("Nie udało się załadować danych.")

st.sidebar.markdown("---")
st.sidebar.info("Aplikacja Streamlit - Analiza NBA")
st.sidebar.markdown(f"Czas: {pd.Timestamp('now', tz='Europe/Warsaw').strftime('%Y-%m-%d %H:%M:%S %Z')}")