
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
import sys # Added for pytz check
from sklearn.compose import ColumnTransformer
import xgboost as xgb

warnings.filterwarnings('ignore')

# --- Initial Configuration ---
st.set_page_config(
    layout="wide",
    page_title="NBA Shot Analysis 2023-24"
)

# --- Path to CSV file ---
# NOTE: Adjust the path to your file!
# ALSO NOTE: Ensure your CSV file contains a 'LOCATION' column (e.g., 'Home'/'Away') for the new feature to work.
CSV_FILE_PATH = 'nba_player_shooting_data_2023_24.csv' # Actual CSV file


# --- Helper Functions ---
# --- START HELPER FUNCTIONS ---
@st.cache_data
def load_shooting_data(file_path): #
    """Loads and preprocesses NBA player shooting data."""
    load_status = {"success": False} # Default status
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        data = pd.read_csv(file_path, parse_dates=['GAME_DATE'], low_memory=False) # Added low_memory=False
        # Success message will be displayed later

        required_columns_map = {
            'PLAYER': 'PLAYER_NAME', 'TEAM': 'TEAM_NAME',
            'SHOT RESULT': 'SHOT_MADE_FLAG', 'SHOT TYPE': 'SHOT_TYPE',
            'ACTION TYPE': 'ACTION_TYPE',
            'SHOT ZONE BASIC': 'SHOT_ZONE_BASIC', 'SHOT ZONE AREA': 'SHOT_ZONE_AREA',
            'SHOT ZONE RANGE': 'SHOT_ZONE_RANGE', 'SHOT DISTANCE': 'SHOT_DISTANCE',
            'LOC_X': 'LOC_X', 'LOC_Y': 'LOC_Y', 'GAME DATE': 'GAME_DATE',
            'PERIOD': 'PERIOD', 'MINUTES REMAINING': 'MINUTES_REMAINING',
            'SECONDS REMAINING': 'SECONDS_REMAINING',
            'Season Type': 'SEASON_TYPE',
            # Assuming 'LOCATION' column exists for Home/Away info
            'LOCATION': 'LOCATION'
        }
        rename_map = {old: new for old, new in required_columns_map.items() if old in data.columns and new not in data.columns}
        if rename_map:
            data = data.rename(columns=rename_map)

        nan_in_made_flag = False
        if 'SHOT_MADE_FLAG' in data.columns:
            if data['SHOT_MADE_FLAG'].dtype == 'object':
                made_values = ['Made', 'made', '1', 1]
                # Check if there are other values before conversion
                original_unique = data['SHOT_MADE_FLAG'].unique()
                data['SHOT_MADE_FLAG'] = data['SHOT_MADE_FLAG'].apply(lambda x: 1 if str(x).strip().lower() in [str(mv).lower() for mv in made_values] else (0 if pd.notna(x) else np.nan))
                if data['SHOT_MADE_FLAG'].isnull().any():
                    nan_in_made_flag = True
                data['SHOT_MADE_FLAG'] = pd.to_numeric(data['SHOT_MADE_FLAG'], errors='coerce')

            elif pd.api.types.is_numeric_dtype(data['SHOT_MADE_FLAG']):
                if data['SHOT_MADE_FLAG'].isnull().any():
                    nan_in_made_flag = True
                # Convert to int if there are no NaNs, otherwise leave as float and handle NaNs later
                if not data['SHOT_MADE_FLAG'].isnull().any():
                    data['SHOT_MADE_FLAG'] = data['SHOT_MADE_FLAG'].astype(int)
            else: # Other unsupported type
                nan_in_made_flag = True # Assume there might be problems
                data['SHOT_MADE_FLAG'] = pd.to_numeric(data['SHOT_MADE_FLAG'], errors='coerce')

        time_cols = ['PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING']
        missing_time_cols_warning = False
        nan_in_time_cols_warning = False
        for col in time_cols:
            if col in data.columns:
                original_type = data[col].dtype
                # Check if conversion is needed and possible
                if not pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    if data[col].isnull().any():
                        nan_in_time_cols_warning = True
            else:
                missing_time_cols_warning = True

        # Creating time columns
        can_create_time_cols = not missing_time_cols_warning and not nan_in_time_cols_warning
        if can_create_time_cols and all(col in data.columns for col in ['PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING']):
            if all(pd.api.types.is_numeric_dtype(data[col]) for col in time_cols):
                if 'MINUTES_REMAINING' in data.columns and 'SECONDS_REMAINING' in data.columns:
                    data['GAME_TIME_SEC'] = data['MINUTES_REMAINING'] * 60 + data['SECONDS_REMAINING']
                if 'PERIOD' in data.columns:
                    # Ensure PERIOD is int before comparing > 4
                    period_int = pd.to_numeric(data['PERIOD'], errors='coerce').fillna(-1).astype(int)
                    data['QUARTER_TYPE'] = period_int.apply(lambda x: 'Overtime' if x > 4 else 'Regular')
            else:
                # If columns exist but are not numeric after trying conversion
                nan_in_time_cols_warning = True # Set warning flag

        # Checking key columns
        key_cols_to_check = ['PLAYER_NAME', 'TEAM_NAME', 'LOC_X', 'LOC_Y', 'SHOT_DISTANCE', 'SEASON_TYPE',
                             'ACTION_TYPE', 'SHOT_TYPE', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE',
                             'SHOT_MADE_FLAG', 'LOCATION'] # Added LOCATION
        missing_key_cols = [col for col in key_cols_to_check if col not in data.columns]

        if 'SHOT_MADE_FLAG' in data.columns:
            # Ensure NaNs are handled before analysis - often removed later in functions
            pass # Already converted above

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
        error_msg = f"Error: File not found {file_path}."
        st.session_state.load_error_message = error_msg
        load_status["error"] = error_msg
        return pd.DataFrame(), load_status
    except Exception as e:
        error_msg = f"Error during data loading or processing: {e}"
        st.session_state.load_error_message = error_msg
        load_status["error"] = error_msg
        return pd.DataFrame(), load_status

def add_court_shapes(fig): #
    """Adds court shapes to a Plotly figure."""
    # Hoop
    fig.add_shape(type="circle", x0=-7.5, y0=-7.5+52.5, x1=7.5, y1=7.5+52.5, line_color="orange", line_width=2)
    # Backboard
    fig.add_shape(type="rect", x0=-30, y0=-7.5+40, x1=30, y1=-4.5+40, line_color="black", line_width=1, fillcolor="#e8e8e8") # Light gray
    # Three-second lane ("paint")
    fig.add_shape(type="rect", x0=-80, y0=-47.5, x1=80, y1=142.5, line_color="black", line_width=1)
    # Restricted zone (semicircle under the basket)
    fig.add_shape(type="path", path=f"M -40 -7.5 A 40 40 0 0 1 40 -7.5", line_color="black", line_width=1, y0=-47.5) # Corrected y0
    # Free throw circle
    fig.add_shape(type="circle", x0=-60, y0=142.5-60, x1=60, y1=142.5+60, line_color="black", line_width=1)
    # Sidelines for 3-point shots (straight lines)
    fig.add_shape(type="line", x0=-220, y0=-47.5, x1=-220, y1=92.5, line_color="black", line_width=1)
    fig.add_shape(type="line", x0=220, y0=-47.5, x1=220, y1=92.5, line_color="black", line_width=1)
    # 3-point arc
    fig.add_shape(type="path", path=f"M -220 92.5 C -135 300, 135 300, 220 92.5", line_color="black", line_width=1)
    # Center circle
    fig.add_shape(type="circle", x0=-60, y0=470-60, x1=60, y1=470+60, line_color="black", line_width=1)
    # Center line
    fig.add_shape(type="line", x0=-250, y0=422.5, x1=250, y1=422.5, line_color="black", line_width=1)
    # Court boundary lines (rectangle) - NOTE: y1 should be equal to the center line
    fig.add_shape(type="rect", x0=-250, y0=-47.5, x1=250, y1=422.5, line_color="black", line_width=1)

    # Set axis ranges for a standard NBA half-court view
    fig.update_xaxes(range=[-260, 260])
    fig.update_yaxes(range=[-60, 480]) # Adjusted Y range

    return fig

@st.cache_data
def filter_data_by_player(player_name, data): #
    """Filters data for the selected player."""
    if 'PLAYER_NAME' not in data.columns or data.empty: return pd.DataFrame()
    return data[data['PLAYER_NAME'] == player_name].copy()

@st.cache_data
def filter_data_by_team(team_name, data): #
    """Filters data for the selected team."""
    if 'TEAM_NAME' not in data.columns or data.empty: return pd.DataFrame()
    return data[data['TEAM_NAME'] == team_name].copy()

def get_basic_stats(entity_data, entity_name, entity_type="Player"): #
    """Calculates basic stats (overall) for a player or team."""
    stats = {'total_shots': 0, 'made_shots': "N/A", 'shooting_pct': "N/A"}
    if entity_data is None or entity_data.empty:
        return stats

    stats['total_shots'] = len(entity_data)

    if 'SHOT_MADE_FLAG' in entity_data.columns and stats['total_shots'] > 0:
        # Work on a copy to avoid SettingWithCopyWarning
        data_copy = entity_data[['SHOT_MADE_FLAG']].copy()
        # Convert to numeric and remove NaNs *locally* for this calculation
        data_copy['SHOT_MADE_FLAG_NUM'] = pd.to_numeric(data_copy['SHOT_MADE_FLAG'], errors='coerce')
        numeric_shots = data_copy['SHOT_MADE_FLAG_NUM'].dropna()

        if not numeric_shots.empty:
            stats['made_shots'] = int(numeric_shots.sum())
            total_valid_shots = len(numeric_shots) # Use the number of valid (non-NaN) shots
            if total_valid_shots > 0:
                stats['shooting_pct'] = (stats['made_shots'] / total_valid_shots) * 100
            else:
                stats['made_shots'], stats['shooting_pct'] = 0, 0.0
        else:
            # If nothing remains after removing NaNs
            stats['made_shots'], stats['shooting_pct'] = 0, 0.0
    elif stats['total_shots'] == 0:
        stats['made_shots'], stats['shooting_pct'] = 0, 0.0

    return stats


@st.cache_data
def plot_player_eff_vs_distance(player_data, player_name, bin_width=1, min_attempts_per_bin=5): #
    """Creates a line chart of player effectiveness (FG%) vs. shot distance (binned)."""
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
        # Convert bin labels to numeric if they are categorical
        if pd.api.types.is_categorical_dtype(distance_data['distance_bin_mid']):
            distance_data['distance_bin_mid'] = pd.to_numeric(distance_data['distance_bin_mid'].astype(str), errors='coerce')

        effectiveness = distance_data.groupby('distance_bin_mid', observed=False)['SHOT_MADE_FLAG'].agg(Made='sum', Attempts='count').reset_index()
        effectiveness = effectiveness[effectiveness['Attempts'] >= min_attempts_per_bin]
        if effectiveness.empty: return None
        effectiveness['FG%'] = (effectiveness['Made'] / effectiveness['Attempts']) * 100
        effectiveness = effectiveness.sort_values(by='distance_bin_mid')
        if effectiveness.empty: return None
        fig = px.line(effectiveness, x='distance_bin_mid', y='FG%', title=f'Impact of Shot Distance on Effectiveness - {player_name}',
                      labels={'distance_bin_mid': 'Distance Bin Midpoint (feet)', 'FG%': 'Effectiveness (%)'}, markers=True, hover_data=['Attempts', 'Made'])
        fig.update_layout(yaxis_range=[-5, 105], xaxis_title='Shot Distance (feet)', yaxis_title='Effectiveness (%)', hovermode="x unified")
        fig.update_traces(connectgaps=False) # Do not connect data gaps
        return fig
    except ValueError as e: return None
    except Exception as e_general: return None


@st.cache_data
def plot_shot_chart(entity_data, entity_name, entity_type="Player"): #
    """Creates an interactive shot chart."""
    required_cols = ['LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']
    if not all(col in entity_data.columns for col in required_cols): return None
    # Add ACTION_TYPE_SIMPLE to hover data if it exists
    hover_base = ['SHOT_DISTANCE', 'SHOT_TYPE', 'ACTION_TYPE', 'SHOT_ZONE_BASIC', 'PERIOD', 'LOCATION'] # Added LOCATION
    if 'ACTION_TYPE_SIMPLE' in entity_data.columns:
        hover_base.append('ACTION_TYPE_SIMPLE')

    plot_data = entity_data[required_cols + [col for col in hover_base if col in entity_data.columns]].copy()
    plot_data['LOC_X'] = pd.to_numeric(plot_data['LOC_X'], errors='coerce')
    plot_data['LOC_Y'] = pd.to_numeric(plot_data['LOC_Y'], errors='coerce')
    plot_data['SHOT_MADE_FLAG'] = pd.to_numeric(plot_data['SHOT_MADE_FLAG'], errors='coerce')
    plot_data = plot_data.dropna(subset=required_cols)
    if plot_data.empty: return None
    plot_data['Shot Result'] = plot_data['SHOT_MADE_FLAG'].map({0: 'Missed', 1: 'Made'})
    color_col, color_map, cat_orders = 'Shot Result', {'Missed': 'red', 'Made': 'green'}, {"Shot Result": ['Missed', 'Made']}

    hover_cols_present = [col for col in hover_base if col in plot_data.columns] # Use hover_base
    hover_data_config = {col: True for col in hover_cols_present}

    fig = px.scatter(plot_data, x='LOC_X', y='LOC_Y', color=color_col, title=f'Shot Chart - {entity_name} ({entity_type})',
                     labels={'LOC_X': 'X Position', 'LOC_Y': 'Y Position', 'Shot Result': 'Result'},
                     hover_data=hover_data_config if hover_data_config else None,
                     category_orders=cat_orders, color_discrete_map=color_map, opacity=0.7)
    fig = add_court_shapes(fig)
    fig.update_layout(height=600, xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='rgba(255, 255, 255, 1)')
    return fig


@st.cache_data
def calculate_hot_zones(entity_data, min_shots_in_zone=5, n_bins=10): #
    """Calculates statistics for shooting zones."""
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
    n_bins = max(2, n_bins) # Minimum 2 bins
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
def plot_hot_zones_heatmap(hot_zones_df, entity_name, entity_type="Player", min_shots_in_zone=5): #
    """Creates an interactive heatmap of shooting zones (effectiveness)."""
    required_cols = ['x_center', 'y_center', 'total_shots', 'percentage', 'made_shots']
    if hot_zones_df is None or hot_zones_df.empty or not all(col in hot_zones_df.columns for col in required_cols):
        return None
    plot_df = hot_zones_df[required_cols].copy()
    # Ensure columns are numeric before operations
    for col in required_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    plot_df = plot_df.dropna() # Remove rows with NaNs after conversion
    if plot_df.empty: return None

    min_pct, max_pct = plot_df['percentage'].min(), plot_df['percentage'].max()
    # Color range - avoid min > max situation
    color_range = [max(0, min_pct - 5), min(100, max_pct + 5)] if pd.notna(min_pct) and pd.notna(max_pct) else [0, 100]
    if color_range[0] >= color_range[1]: color_range = [0, 100] # Fallback

    # Bubble size scaling
    max_bubble_size = plot_df["total_shots"].max() if not plot_df["total_shots"].empty else 1
    size_ref = max(1, max_bubble_size / 50.0) # Experimentally determined scaling factor

    fig = px.scatter(plot_df, x='x_center', y='y_center', size='total_shots', color='percentage',
                     color_continuous_scale=px.colors.diverging.RdYlGn, # Red-Yellow-Green
                     size_max=60, range_color=color_range,
                     title=f'Shooting Zone Effectiveness ({entity_type}: {entity_name}, min. {min_shots_in_zone} shots)',
                     labels={'x_center': 'X Position', 'y_center': 'Y Position', 'total_shots': 'Number of Shots', 'percentage': 'Effectiveness (%)'},
                     custom_data=['made_shots', 'total_shots']) # Add data to hover

    # Improved hovertemplate
    fig.update_traces(
        hovertemplate="<b>Zone X:</b> %{x:.1f}, <b>Y:</b> %{y:.1f}<br>" +
                      "<b>Number of shots:</b> %{customdata[1]}<br>" +
                      "<b>Made:</b> %{customdata[0]}<br>" +
                      "<b>Effectiveness:</b> %{marker.color:.1f}%<extra></extra>",
        marker=dict(sizeref=size_ref, sizemin=4) # Set scaling and min size
    )

    fig = add_court_shapes(fig)
    fig.update_layout(height=600, xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='rgba(255, 255, 255, 1)')
    return fig


@st.cache_data
def plot_shot_frequency_heatmap(data, season_name, nbins_x=50, nbins_y=50): #
    """Creates a heatmap of shot frequency on the court (Histogram2d)."""
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
        colorscale = 'YlOrRd', # Yellow to Red color scale
        nbinsx = nbins_x,
        nbinsy = nbins_y,
        zauto = True, # Automatic scaling of Z axis (color)
        hovertemplate = '<b>X Range:</b> %{x}<br><b>Y Range:</b> %{y}<br><b>Number of Shots:</b> %{z}<extra></extra>',
        colorbar=dict(title='Number of Shots')
    ))

    fig = add_court_shapes(fig) # Add court lines

    fig.update_layout(
        title=f'Shot Frequency Map ({season_name})',
        xaxis_title="X Position", yaxis_title="Y Position",
        height=650,
        xaxis_showgrid=False, yaxis_showgrid=False,
        plot_bgcolor='rgba(255, 255, 255, 1)' # White background
    )
    return fig


@st.cache_data
def plot_player_quarter_eff(entity_data, entity_name, entity_type="Player", min_attempts=5): #
    """Chart of effectiveness in individual quarters/overtimes."""
    if 'PERIOD' not in entity_data.columns or 'SHOT_MADE_FLAG' not in entity_data.columns: return None
    quarter_data = entity_data[['PERIOD', 'SHOT_MADE_FLAG']].copy()
    quarter_data['PERIOD'] = pd.to_numeric(quarter_data['PERIOD'], errors='coerce')
    quarter_data['SHOT_MADE_FLAG'] = pd.to_numeric(quarter_data['SHOT_MADE_FLAG'], errors='coerce')
    quarter_data = quarter_data.dropna()
    if quarter_data.empty: return None
    quarter_data['PERIOD'] = quarter_data['PERIOD'].astype(int) # Convert to int after removing NaNs

    quarter_eff = quarter_data.groupby('PERIOD')['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
    quarter_eff['mean'] *= 100 # Convert to percentage
    quarter_eff = quarter_eff[quarter_eff['count'] >= min_attempts] # Filter by minimum attempts
    if quarter_eff.empty: return None

    # Map quarter number to a readable label
    def map_period(p):
        if p <= 4: return f"Quarter {int(p)}"
        elif p == 5: return "OT 1"
        else: return f"OT {int(p-4)}"
    quarter_eff['Game Period'] = quarter_eff['PERIOD'].apply(map_period)

    # Sort by quarter/overtime number
    quarter_eff = quarter_eff.sort_values(by='PERIOD')

    fig = px.bar(quarter_eff, x='Game Period', y='mean', text='mean',
                 title=f'Effectiveness in Quarters/Overtimes - {entity_name} ({entity_type}, min. {min_attempts} attempts)',
                 labels={'Game Period': 'Game Period', 'mean': 'Effectiveness (%)'},
                 hover_data=['count']) # Show number of attempts in tooltip

    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside') # Format text on bars
    fig.update_layout(yaxis_range=[0, 105], uniformtext_minsize=8, uniformtext_mode='hide') # Layout adjustments
    return fig


@st.cache_data
def plot_player_season_trend(entity_data, entity_name, entity_type="Player", min_monthly_attempts=10): #
    """Chart of effectiveness trend during the season (monthly)."""
    if 'GAME_DATE' not in entity_data.columns or 'SHOT_MADE_FLAG' not in entity_data.columns: return None
    trend_data = entity_data[['GAME_DATE', 'SHOT_MADE_FLAG']].copy()
    trend_data['GAME_DATE'] = pd.to_datetime(trend_data['GAME_DATE'], errors='coerce')
    trend_data['SHOT_MADE_FLAG'] = pd.to_numeric(trend_data['SHOT_MADE_FLAG'], errors='coerce')
    trend_data = trend_data.dropna()
    if trend_data.empty or len(trend_data) < min_monthly_attempts: return None # Check overall data count

    trend_data = trend_data.set_index('GAME_DATE')
    # Resample to monthly frequency ('M' - Month End is default for 'M')
    monthly_eff = trend_data.resample('M')['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
    monthly_eff['mean'] *= 100 # Effectiveness in percent
    monthly_eff = monthly_eff[monthly_eff['count'] >= min_monthly_attempts] # Filter min monthly attempts
    if monthly_eff.empty or len(monthly_eff) < 2: return None # Need min 2 points for a line

    # Format date to 'YYYY-MM' for X-axis
    monthly_eff['Month'] = monthly_eff['GAME_DATE'].dt.strftime('%Y-%m')

    fig = px.line(monthly_eff, x='Month', y='mean', markers=True,
                  title=f'Monthly Effectiveness Trend - {entity_name} ({entity_type}, min. {min_monthly_attempts} attempts/month)',
                  labels={'Month': 'Month', 'mean': 'Effectiveness (%)'},
                  hover_data=['count']) # Show number of attempts in tooltip

    # Dynamic Y-axis range adjustment for better readability
    max_y = 105
    min_y = -5
    if not pd.isna(monthly_eff['mean'].max()): max_y = min(105, monthly_eff['mean'].max() + 10)
    if not pd.isna(monthly_eff['mean'].min()): min_y = max(-5, monthly_eff['mean'].min() - 10)
    fig.update_layout(yaxis_range=[min_y, max_y])
    return fig


@st.cache_data
def plot_grouped_effectiveness(entity_data, group_col, entity_name, entity_type="Player", top_n=10, min_attempts=5): #
    """Creates a chart of effectiveness grouped by the selected column."""
    if group_col not in entity_data.columns or 'SHOT_MADE_FLAG' not in entity_data.columns: return None
    grouped_data = entity_data[[group_col, 'SHOT_MADE_FLAG']].copy()
    grouped_data[group_col] = grouped_data[group_col].astype(str).str.strip() # Ensure it's a string and remove whitespace
    grouped_data['SHOT_MADE_FLAG'] = pd.to_numeric(grouped_data['SHOT_MADE_FLAG'], errors='coerce')
    grouped_data = grouped_data.dropna(subset=[group_col, 'SHOT_MADE_FLAG']) # Drop NaN also in group_col
    if grouped_data.empty: return None

    grouped_eff = grouped_data.groupby(group_col)['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
    grouped_eff['mean'] *= 100 # Effectiveness in percent
    grouped_eff = grouped_eff[grouped_eff['count'] >= min_attempts] # Filter min attempts

    # Sort by number of attempts descending to select top N most frequent, then sort by category for consistency
    grouped_eff_top = grouped_eff.sort_values(by='count', ascending=False).head(top_n)

    # Special sorting for SHOT_ZONE_BASIC
    if group_col == 'SHOT_ZONE_BASIC':
        zone_order_basic = ['Restricted Area', 'In The Paint (Non-RA)', 'Mid-Range', 'Left Corner 3', 'Right Corner 3', 'Above the Break 3', 'Backcourt']
        # Use pd.Categorical to sort according to the defined order
        present_zones = [z for z in zone_order_basic if z in grouped_eff_top[group_col].unique()]
        grouped_eff_plot = grouped_eff_top.copy() # Work on a copy
        grouped_eff_plot[group_col] = pd.Categorical(
            grouped_eff_plot[group_col],
            categories=present_zones,
            ordered=True
        )
        grouped_eff_plot = grouped_eff_plot.sort_values(by=group_col) # Sort by category
    elif group_col == 'ACTION_TYPE_SIMPLE':
        # Define a logical order for simplified types
        action_order_simple = ['Dunk', 'Layup', 'Tip Shot', 'Hook Shot', 'Floater', 'Driving Shot', 'Jump Shot', 'Bank Shot', 'Alley Oop', 'Other']
        present_actions = [a for a in action_order_simple if a in grouped_eff_top[group_col].unique()]
        # Add remaining unforeseen types sorted alphabetically
        other_actions = sorted([a for a in grouped_eff_top[group_col].unique() if a not in present_actions])
        final_action_order = present_actions + other_actions
        grouped_eff_plot = grouped_eff_top.copy() # Work on a copy
        grouped_eff_plot[group_col] = pd.Categorical(
            grouped_eff_plot[group_col],
            categories=final_action_order,
            ordered=True
        )
        grouped_eff_plot = grouped_eff_plot.sort_values(by=group_col)
    else:
        # Other categories sort alphabetically for consistency (on selected top N)
        grouped_eff_plot = grouped_eff_top.sort_values(by=group_col, ascending=True)

    if grouped_eff_plot.empty: return None

    # Creating title and axis labels
    axis_label = group_col.replace('_',' ').title()
    chart_title = f'Effectiveness by {axis_label} - {entity_name} ({entity_type})'
    # Check if fewer were shown than available after filtering min_attempts
    if len(grouped_eff_plot) < len(grouped_eff):
        chart_title += f' (Top {len(grouped_eff_plot)} most frequent, min. {min_attempts} attempts)'
    else:
        chart_title += f' (min. {min_attempts} attempts)'


    fig = px.bar(grouped_eff_plot, x=group_col, y='mean', text='mean',
                 title=chart_title,
                 labels={group_col: axis_label, 'mean': 'Effectiveness (%)'},
                 hover_data=['count']) # Show number of attempts in tooltip

    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(yaxis_range=[0, 105], uniformtext_minsize=8, uniformtext_mode='hide', xaxis_title=axis_label)

    # Ensure X-axis order matches sorting
    if group_col in ['SHOT_ZONE_BASIC', 'ACTION_TYPE_SIMPLE']:
        category_order = grouped_eff_plot[group_col].cat.categories.tolist() # Get order from Categorical
        fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray': category_order})

    return fig


@st.cache_data
def plot_comparison_eff_distance(compare_data, selected_players, bin_width=3, min_attempts_per_bin=5): #
    """Compares player effectiveness against distance (line chart)."""
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
        effectiveness['mean'] *= 100 # Percent
        effectiveness = effectiveness[effectiveness['count'] >= min_attempts_per_bin] # Filter min attempts
        effectiveness = effectiveness.dropna(subset=['distance_bin_mid']) # Remove NaN in bins
        if effectiveness.empty: return None

        effectiveness = effectiveness.sort_values(by=['PLAYER_NAME', 'distance_bin_mid']) # Sort

        # Dynamic Y-axis range
        max_eff_val = effectiveness['mean'].max()
        yaxis_range = [0, min(105, max_eff_val + 10 if not pd.isna(max_eff_val) else 105)]

        fig = px.line(effectiveness, x='distance_bin_mid', y='mean', color='PLAYER_NAME',
                      title=f'Effectiveness vs Distance Comparison (min. {min_attempts_per_bin} attempts in {bin_width} ft bin)',
                      labels={'distance_bin_mid': 'Distance (feet)', 'mean': 'Effectiveness (%)', 'PLAYER_NAME': 'Player'},
                      markers=True, hover_data=['count'])

        fig.update_layout(yaxis_range=yaxis_range, hovermode="x unified")
        return fig
    except Exception as e: return None


@st.cache_data
def plot_comparison_eff_by_zone(compare_data, selected_players, min_shots_per_zone=5): #
    """Creates a grouped bar chart comparing player effectiveness by SHOT_ZONE_BASIC."""
    required_cols = ['PLAYER_NAME', 'SHOT_MADE_FLAG', 'SHOT_ZONE_BASIC']
    if not all(col in compare_data.columns for col in required_cols): return None
    zone_eff_data = compare_data[required_cols].copy()
    zone_eff_data['SHOT_MADE_FLAG'] = pd.to_numeric(zone_eff_data['SHOT_MADE_FLAG'], errors='coerce')
    zone_eff_data['SHOT_ZONE_BASIC'] = zone_eff_data['SHOT_ZONE_BASIC'].astype(str).str.strip()
    zone_eff_data = zone_eff_data.dropna(subset=required_cols)
    if zone_eff_data.empty: return None

    zone_stats = zone_eff_data.groupby(['PLAYER_NAME', 'SHOT_ZONE_BASIC'], observed=False)['SHOT_MADE_FLAG'].agg(Made='sum', Attempts='count').reset_index()
    zone_stats_filtered = zone_stats[zone_stats['Attempts'] >= min_shots_per_zone] # Filter min attempts
    if zone_stats_filtered.empty: return None

    zone_stats_filtered['FG%'] = (zone_stats_filtered['Made'] / zone_stats_filtered['Attempts']) * 100 # Percent

    # Define zone order as in single player analysis
    zone_order_ideal = ['Restricted Area', 'In The Paint (Non-RA)', 'Mid-Range', 'Left Corner 3', 'Right Corner 3', 'Above the Break 3', 'Backcourt']
    actual_zones_in_data = zone_stats_filtered['SHOT_ZONE_BASIC'].unique()
    # Keep only existing zones in the ideal order, add the rest sorted
    zone_order = [zone for zone in zone_order_ideal if zone in actual_zones_in_data]
    zone_order += sorted([zone for zone in actual_zones_in_data if zone not in zone_order_ideal])
    if not zone_order: return None # If no zones are left

    zone_stats_plot = zone_stats_filtered[zone_stats_filtered['SHOT_ZONE_BASIC'].isin(zone_order)].copy()
    if zone_stats_plot.empty: return None

    fig = px.bar(zone_stats_plot, x='SHOT_ZONE_BASIC', y='FG%', color='PLAYER_NAME', barmode='group',
                 title=f'Effectiveness (FG%) Comparison by Shot Zone (min. {min_shots_per_zone} attempts)',
                 labels={'SHOT_ZONE_BASIC': 'Shot Zone', 'FG%': 'Effectiveness (%)', 'PLAYER_NAME': 'Player'},
                 hover_data=['Attempts', 'Made'],
                 category_orders={'SHOT_ZONE_BASIC': zone_order}, # Use defined order
                 text='FG%')

    fig.update_layout(yaxis_range=[0, 105], xaxis={'categoryorder':'array', 'categoryarray':zone_order}, legend_title_text='Players')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    return fig


@st.cache_data
def calculate_top_performers(data, group_by_col, min_total_shots, min_2pt_shots, min_3pt_shots, top_n=10): #
    """Calculates Top N player/team rankings by effectiveness."""
    if group_by_col not in data.columns or 'SHOT_MADE_FLAG' not in data.columns: return None, None, None
    valid_data = data[[group_by_col, 'SHOT_MADE_FLAG', 'SHOT_TYPE'] if 'SHOT_TYPE' in data.columns else [group_by_col, 'SHOT_MADE_FLAG']].copy()
    valid_data['SHOT_MADE_FLAG'] = pd.to_numeric(valid_data['SHOT_MADE_FLAG'], errors='coerce')
    valid_data[group_by_col] = valid_data[group_by_col].astype(str)
    valid_data = valid_data.dropna(subset=[group_by_col, 'SHOT_MADE_FLAG'])
    if valid_data.empty: return None, None, None

    # Overall Ranking (FG%)
    overall_stats = valid_data.groupby(group_by_col)['SHOT_MADE_FLAG'].agg(Made='sum', Attempts='count').reset_index()
    overall_stats = overall_stats[overall_stats['Attempts'] >= min_total_shots]
    top_overall = pd.DataFrame()
    if not overall_stats.empty:
        overall_stats['FG%'] = (overall_stats['Made'] / overall_stats['Attempts']) * 100
        top_overall = overall_stats.sort_values(by=['FG%', 'Attempts'], ascending=[False, False]).head(top_n)
        col_name = group_by_col.replace('_',' ').title()
        top_overall = top_overall.rename(columns={group_by_col: col_name, 'Attempts': 'Attempts'})
        top_overall = top_overall[[col_name, 'FG%', 'Attempts']] # Keep only necessary columns

    # 2PT% Ranking
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
                    top_2pt = top_2pt.rename(columns={group_by_col: col_name, 'Attempts_2PT': '2PT Attempts'})
                    top_2pt = top_2pt[[col_name, '2PT FG%', '2PT Attempts']]

    # 3PT% Ranking
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
                    top_3pt = top_3pt.rename(columns={group_by_col: col_name, 'Attempts_3PT': '3PT Attempts'})
                    top_3pt = top_3pt[[col_name, '3PT FG%', '3PT Attempts']]

    return top_overall, top_2pt, top_3pt

def simplify_action_type(df): #
    """
    Groups values in the ACTION_TYPE column into simplified categories.
    """
    if 'ACTION_TYPE' not in df.columns:
        print("Warning: Column 'ACTION_TYPE' not found. Cannot simplify.")
        return df

    action_col = df['ACTION_TYPE'].astype(str).str.lower()
    conditions = [
        action_col.str.contains('dunk', na=False),
        action_col.str.contains('layup', na=False),
        action_col.str.contains('jump shot|pullup|step back|fadeaway', na=False, regex=True),
        action_col.str.contains('hook shot', na=False),
        action_col.str.contains('tip|putback', na=False, regex=True),
        action_col.str.contains('driving', na=False),
        action_col.str.contains('floating|floater', na=False, regex=True),
        action_col.str.contains('alley oop', na=False),
        action_col.str.contains('bank shot', na=False)
    ]
    choices = [
        'Dunk', 'Layup', 'Jump Shot', 'Hook Shot', 'Tip Shot',
        'Driving Shot', 'Floater', 'Alley Oop', 'Bank Shot'
    ]
    df['ACTION_TYPE_SIMPLE'] = np.select(conditions, choices, default='Other')
    return df

# --- END HELPER FUNCTIONS ---




# --- Main Streamlit application part ---
st.title("🏀 Interactive Analysis and Prediction of NBA Player Shooting Efficiency in the 2023-24 Season") #

# Calling the data loading function
shooting_data, load_status = load_shooting_data(CSV_FILE_PATH) #

# --- Integration of ACTION_TYPE Simplification ---
if load_status.get("success", False) and not shooting_data.empty:
    shooting_data = simplify_action_type(shooting_data) #

# --- Sidebar ---
st.sidebar.header("Filtering and Analysis Options") #

# Main part of the application rendered only if data is loaded
if load_status.get("success", False) and not shooting_data.empty: #

    # Success message
    success_msg = f"Loaded and processed data from {CSV_FILE_PATH}. Dimensions: {load_status.get('shape')}."
    st.success(success_msg) #

    # Display warnings from the loading process
    if load_status.get("missing_time_cols"): st.warning("Missing time columns. Could not create 'GAME_TIME_SEC' or 'QUARTER_TYPE'.") #
    if load_status.get("nan_in_time_cols"): st.warning("Non-numeric values (NaN) found in time columns after conversion attempt.") #
    if load_status.get("nan_in_made_flag"): st.warning("Ambiguous or non-numeric values (NaN) found in SHOT_MADE_FLAG column after conversion attempt.") #
    if load_status.get("missing_key_cols"): st.warning(f"Missing key columns for full analysis: {', '.join(load_status['missing_key_cols'])}. Note: Models require 'LOCATION' column.") # Updated warning

    # Sidebar Filters
    available_season_types = ['All'] + shooting_data['SEASON_TYPE'].dropna().unique().tolist() if 'SEASON_TYPE' in shooting_data.columns else ['All'] #
    selected_season_type = st.sidebar.selectbox(
        "Select season type:", options=available_season_types, index=0, key='sidebar_season_select'
    ) #
    if selected_season_type != 'All' and 'SEASON_TYPE' in shooting_data.columns:
        filtered_data = shooting_data[shooting_data['SEASON_TYPE'] == selected_season_type].copy()
    else:
        filtered_data = shooting_data.copy()
    st.sidebar.write(f"Selected: {selected_season_type} ({len(filtered_data)} shots)") #

    # Player Selection
    available_players = sorted(filtered_data['PLAYER_NAME'].dropna().unique()) if 'PLAYER_NAME' in filtered_data.columns else [] #
    default_player = "LeBron James" #
    default_player_index = 0
    if available_players:
        try: default_player_index = available_players.index(default_player)
        except ValueError: default_player_index = 0
    selected_player = st.sidebar.selectbox(
        "Player for analysis/modeling:", options=available_players, index=default_player_index,
        key='sidebar_player_select', disabled=not available_players,
        help="Select a player for detailed analysis or model evaluation."
    ) #

    # Team Selection
    available_teams = sorted(filtered_data['TEAM_NAME'].dropna().unique()) if 'TEAM_NAME' in filtered_data.columns else [] #
    default_team = "Los Angeles Lakers" #
    default_team_index = 0
    if available_teams:
        try: default_team_index = available_teams.index(default_team)
        except ValueError: default_team_index = 0
    selected_team = st.sidebar.selectbox(
        "Team for analysis:", options=available_teams, index=default_team_index,
        key='sidebar_team_select', disabled=not available_teams,
        help="Select a team for team analysis."
    ) #

    # Player Comparison Selection
    default_compare_players_req = ["LeBron James", "Stephen Curry"] #
    default_compare_players_available = [p for p in default_compare_players_req if p in available_players]
    selected_players_compare = st.sidebar.multiselect(
        "Players to compare (2-5):", options=available_players, default=default_compare_players_available,
        max_selections=5, key='sidebar_player_multi_select', disabled=not available_players,
        help="Select 2 to 5 players for comparison."
    ) #

    # === Using st.tabs ===
    tab_options = [
        "📊 Overall Analysis",
        "⛹️ Player Analysis",
        "🆚 Player Comparison",
        "🏀 Team Analysis",
        "🎯 Model Evaluation (KNN)",
        "🔍 Model Evaluation (XGBoost)",
        "⚖️ Model Comparison"
    ] #
    tab_rankings, tab_player, tab_compare, tab_team, tab_knn, tab_xgb, tab_model_comp = st.tabs(tab_options) #

    # === Individual Views / Tabs ===

    with tab_rankings: #
        # --- START: Updated code for data sample ---
        # Use 'selected_player' from the sidebar instead of 'default_player'
        st.subheader(f"Data Sample (First 5 Rows for Selected Player: {selected_player})") # Changed from 10 to 5
        st.caption("This shows the structure of the data being analyzed for the currently selected player.")

        # Check if a player is selected and data is available
        if 'shooting_data' in locals() and selected_player and 'PLAYER_NAME' in shooting_data.columns:
            # Filter data for the selected player
            selected_player_data = filter_data_by_player(selected_player, shooting_data) # Use selected_player here
            if not selected_player_data.empty:
                # Display the first 5 rows for the selected player
                st.dataframe(selected_player_data.head(5), use_container_width=True) # Changed from 10 to 5
            else:
                st.caption(f"Could not find data for the selected player '{selected_player}' to display the sample.")
        elif not selected_player:
             st.caption("Select a player from the sidebar ('Player for analysis/modeling') to see a data sample.")
        else:
            st.caption("Could not load data sample (required data missing or no player selected).")
        st.markdown("---") # Add a separator after data sample
        # --- END: Updated code for data sample ---

        # The header now comes AFTER the data sample
        st.header(f"Overall Season Analysis: {selected_season_type}") #
        st.markdown("---") # Separator after the main header

        # --- Calculating and Displaying League Averages ---
        st.subheader("League Average Effectiveness") #
        league_avg_2pt = "N/A"; league_avg_3pt = "N/A" #
        attempts_2pt_league = 0; attempts_3pt_league = 0 #
        made_2pt_league = 0; made_3pt_league = 0 #
        if 'SHOT_TYPE' in filtered_data.columns and 'SHOT_MADE_FLAG' in filtered_data.columns: #
            calc_data = filtered_data[['SHOT_TYPE', 'SHOT_MADE_FLAG']].copy() #
            calc_data['SHOT_MADE_FLAG'] = pd.to_numeric(calc_data['SHOT_MADE_FLAG'], errors='coerce') #
            calc_data.dropna(subset=['SHOT_MADE_FLAG'], inplace=True) #
            if not calc_data.empty: #
                calc_data['SHOT_MADE_FLAG'] = calc_data['SHOT_MADE_FLAG'].astype(int) #
                data_2pt = calc_data[calc_data['SHOT_TYPE'] == '2PT Field Goal'] #
                attempts_2pt_league = len(data_2pt) #
                if attempts_2pt_league > 0: #
                    made_2pt_league = data_2pt['SHOT_MADE_FLAG'].sum() #
                    league_avg_2pt = (made_2pt_league / attempts_2pt_league) * 100 #
                data_3pt = calc_data[calc_data['SHOT_TYPE'] == '3PT Field Goal'] #
                attempts_3pt_league = len(data_3pt) #
                if attempts_3pt_league > 0: #
                    made_3pt_league = data_3pt['SHOT_MADE_FLAG'].sum() #
                    league_avg_3pt = (made_3pt_league / attempts_3pt_league) * 100 #
            else: st.caption("No valid data to calculate league averages.") #
        col_avg1, col_avg2 = st.columns(2) #
        with col_avg1: #
            st.metric(label=f"Average 2PT Effectiveness (League)", value=f"{league_avg_2pt:.1f}%" if isinstance(league_avg_2pt, (float, int)) else "No data", help=f"Calculated based on {attempts_2pt_league:,} shots ({made_2pt_league:,} made) in {selected_season_type}.".replace(',', ' ')) #
        with col_avg2: #
            st.metric(label=f"Average 3PT Effectiveness (League)", value=f"{league_avg_3pt:.1f}%" if isinstance(league_avg_3pt, (float, int)) else "No data", help=f"Calculated based on {attempts_3pt_league:,} shots ({made_3pt_league:,} made) in {selected_season_type}.".replace(',', ' ')) #
        st.markdown("---") #

        st.subheader("General Shot Distributions") #
        st.caption(f"Data for: {selected_season_type}") #
        action_type_col_rank = 'ACTION_TYPE' #
        action_choice = 'Original' #
        if 'ACTION_TYPE_SIMPLE' in filtered_data.columns: #
            action_choice_options = ('Original', 'Simplified') #
            default_action_index = 1 #
            action_choice = st.radio("Show action types:", action_choice_options, key='rank_action_type_choice', horizontal=True, index=default_action_index) #
            if action_choice == 'Simplified': action_type_col_rank = 'ACTION_TYPE_SIMPLE' #
        c1_top, c2_top = st.columns(2) #
        with c1_top: #
            st.markdown("###### Shot Type Distribution") #
            if 'SHOT_TYPE' in filtered_data.columns and not filtered_data['SHOT_TYPE'].isnull().all(): #
                shot_type_counts = filtered_data['SHOT_TYPE'].dropna().value_counts().reset_index() #
                if not shot_type_counts.empty: #
                    if 'index' in shot_type_counts.columns and 'SHOT_TYPE' not in shot_type_counts.columns: #
                        shot_type_counts = shot_type_counts.rename(columns={'index': 'SHOT_TYPE'}) #
                    if 'SHOT_TYPE' in shot_type_counts.columns and 'count' not in shot_type_counts.columns and shot_type_counts.columns[1] != 'SHOT_TYPE': #
                        shot_type_counts = shot_type_counts.rename(columns={shot_type_counts.columns[1]: 'count'}) #
                    fig_type = px.pie(shot_type_counts, names='SHOT_TYPE', values='count', hole=0.3) #
                    fig_type.update_layout(legend_title_text='Shot Type', height=350, margin=dict(t=20, b=0, l=0, r=0)) #
                    st.plotly_chart(fig_type, use_container_width=True) #
                else: st.caption("No data for shot types.") #
            else: st.caption("Missing 'SHOT_TYPE' column.") #
        with c2_top: #
            st.markdown(f"###### Most Frequent Action Types ({action_choice})") #
            if action_type_col_rank in filtered_data.columns and not filtered_data[action_type_col_rank].isnull().all(): #
                action_type_counts = filtered_data[action_type_col_rank].dropna().value_counts().head(15).reset_index() #
                if 'index' in action_type_counts.columns and action_type_col_rank not in action_type_counts.columns: #
                    action_type_counts = action_type_counts.rename(columns={'index': action_type_col_rank}) #
                if action_type_col_rank in action_type_counts.columns and 'count' not in action_type_counts.columns and action_type_counts.columns[1] != action_type_col_rank: #
                    action_type_counts = action_type_counts.rename(columns={action_type_counts.columns[1]: 'count'}) #
                if not action_type_counts.empty: #
                    fig_action_freq = px.bar(action_type_counts, y=action_type_col_rank, x='count', orientation='h', labels={'count':'Number of Shots', action_type_col_rank:''}, text='count') #
                    fig_action_freq.update_layout(yaxis={'categoryorder':'total ascending'}, height=350, margin=dict(t=20, b=0, l=0, r=0)) #
                    fig_action_freq.update_traces(texttemplate='%{text:,}'.replace(',', ' '), textposition='outside') #
                    st.plotly_chart(fig_action_freq, use_container_width=True) #
                else: st.caption(f"No data for action types ('{action_type_col_rank}').") #
            else: st.caption(f"Missing column '{action_type_col_rank}'.") #
        st.markdown("<br>", unsafe_allow_html=True) #
        st.markdown("<br>", unsafe_allow_html=True) #
        st.markdown("<br>", unsafe_allow_html=True) #
        c1_bottom, c2_bottom = st.columns(2) #
        with c1_bottom: #
            st.markdown("###### Shot Distribution by Zone") #
            if 'SHOT_ZONE_BASIC' in filtered_data.columns and not filtered_data['SHOT_ZONE_BASIC'].isnull().all(): #
                zone_basic_counts = filtered_data['SHOT_ZONE_BASIC'].dropna().value_counts().reset_index() #
                if 'index' in zone_basic_counts.columns and 'SHOT_ZONE_BASIC' not in zone_basic_counts.columns: #
                    zone_basic_counts = zone_basic_counts.rename(columns={'index': 'SHOT_ZONE_BASIC'}) #
                if 'SHOT_ZONE_BASIC' in zone_basic_counts.columns and 'count' not in zone_basic_counts.columns and zone_basic_counts.columns[1] != 'SHOT_ZONE_BASIC': #
                    zone_basic_counts = zone_basic_counts.rename(columns={zone_basic_counts.columns[1]: 'count'}) #
                if not zone_basic_counts.empty: #
                    zone_order_basic = ['Restricted Area', 'In The Paint (Non-RA)', 'Mid-Range', 'Left Corner 3', 'Right Corner 3', 'Above the Break 3', 'Backcourt'] #
                    zone_basic_counts['SHOT_ZONE_BASIC'] = pd.Categorical(zone_basic_counts['SHOT_ZONE_BASIC'], categories=[z for z in zone_order_basic if z in zone_basic_counts['SHOT_ZONE_BASIC'].unique()], ordered=True) #
                    zone_basic_counts = zone_basic_counts.sort_values('SHOT_ZONE_BASIC') #
                    fig_zone_basic_dist = px.bar(zone_basic_counts, x='SHOT_ZONE_BASIC', y='count', labels={'SHOT_ZONE_BASIC': 'Shot Zone (Basic)', 'count': 'Number of Shots'}, text='count') #
                    fig_zone_basic_dist.update_traces(texttemplate='%{text:,}'.replace(',', ' '), textposition='auto') #
                    fig_zone_basic_dist.update_layout(xaxis_title=None, yaxis_title="Number of Shots", height=400, margin=dict(t=0, b=0, l=0, r=0)) #
                    max_y_val = zone_basic_counts['count'].max() #
                    if pd.notna(max_y_val): fig_zone_basic_dist.update_layout(yaxis_range=[0, max_y_val * 1.1]) #
                    st.plotly_chart(fig_zone_basic_dist, use_container_width=True) #
                else: st.caption("No data for basic zone distribution.") #
            else: st.caption("Missing 'SHOT_ZONE_BASIC' column.") #
        with c2_bottom: #
            st.markdown(f"###### Most Effective Action Types ({action_choice})") #
            if action_type_col_rank in filtered_data.columns and 'SHOT_MADE_FLAG' in filtered_data.columns: #
                min_attempts_eff_action_b = st.number_input(f"Min. attempts for action effectiveness ranking ({action_choice}):", min_value=5, value=10, step=1, key=f'rank_min_attempts_eff_action_{action_type_col_rank}_bottom', help="Minimum number of shots of a given action type in the ranking.") #
                action_eff_data = filtered_data[[action_type_col_rank, 'SHOT_MADE_FLAG']].copy() #
                action_eff_data[action_type_col_rank] = action_eff_data[action_type_col_rank].astype(str).str.strip() #
                action_eff_data['SHOT_MADE_FLAG'] = pd.to_numeric(action_eff_data['SHOT_MADE_FLAG'], errors='coerce') #
                action_eff_data = action_eff_data.dropna() #
                if not action_eff_data.empty: #
                    action_stats = action_eff_data.groupby(action_type_col_rank)['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index() #
                    action_stats_filtered = action_stats[action_stats['count'] >= min_attempts_eff_action_b].copy() #
                    if not action_stats_filtered.empty: #
                        action_stats_filtered['FG%'] = action_stats_filtered['mean'] * 100 #
                        action_stats_filtered = action_stats_filtered.sort_values(by='FG%', ascending=False).head(15) #
                        fig_action_eff = px.bar(action_stats_filtered, y=action_type_col_rank, x='FG%', orientation='h', labels={'FG%': 'Effectiveness (%)', action_type_col_rank: ''}, text='FG%', hover_data=['count']) #
                        fig_action_eff.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_range=[0, 105], height=400, margin=dict(t=0, b=0, l=0, r=0)) #
                        fig_action_eff.update_traces(texttemplate='%{text:.1f}%', textposition='outside') #
                        st.plotly_chart(fig_action_eff, use_container_width=True) #
                    else: st.caption(f"No action types ('{action_type_col_rank}') with min. {min_attempts_eff_action_b} attempts.") #
                else: st.caption("No data to calculate action effectiveness.") #
            else: st.caption(f"Missing columns '{action_type_col_rank}' or 'SHOT_MADE_FLAG'.") #
        st.markdown("---") #

        st.subheader(f"Shot Frequency Map") #
        st.caption(f"Shot density for: {selected_season_type}") #
        num_bins_freq = st.slider("Map Accuracy (X):", 20, 80, 50, 5, key='rank_frequency_map_bins', help="Number of 'bins' the court's X-axis is divided into for counting shots. The Y-axis will adjust proportionally.") #
        nbins_y_freq = int(num_bins_freq * (470 / 500)) #
        fig_freq_map = plot_shot_frequency_heatmap(filtered_data, selected_season_type, nbins_x=num_bins_freq, nbins_y=nbins_y_freq) #
        if fig_freq_map: st.plotly_chart(fig_freq_map, use_container_width=True) #
        else: st.caption("Could not generate frequency map.") #
        st.markdown("---") #

        st.subheader(f"Player and Team Effectiveness Rankings") #
        st.caption(f"Top 10 for: {selected_season_type}") #
        st.markdown("##### Min. number of attempts") #
        col_att1, col_att2, col_att3 = st.columns(3) #
        with col_att1: min_total = st.number_input("Overall:", 10, 1000, 100, 10, key="rank_min_total") #
        with col_att2: min_2pt = st.number_input("2-pointers:", 5, 500, 50, 5, key="rank_min_2pt") #
        with col_att3: min_3pt = st.number_input("3-pointers:", 5, 500, 30, 5, key="rank_min_3pt") #

        # Calculate rankings
        tp_ov, tp_2, tp_3 = calculate_top_performers(filtered_data, 'PLAYER_NAME', min_total, min_2pt, min_3pt) #
        tt_ov, tt_2, tt_3 = calculate_top_performers(filtered_data, 'TEAM_NAME', min_total*5, min_2pt*5, min_3pt*5) #

        # Display Overall Rankings
        st.markdown("###### Overall Effectiveness (FG%)") #
        c1_rank, c2_rank = st.columns(2) #
        with c1_rank: #
            st.markdown(f"**Top 10 Players (min. {min_total} attempts)**") #
            if tp_ov is not None and not tp_ov.empty: st.dataframe(tp_ov.rename(columns={'Attempts': 'Attempts'}), use_container_width=True, hide_index=True, column_config={"FG%": st.column_config.ProgressColumn("FG%", format="%.1f%%", min_value=0, max_value=100)}) #
            else: st.caption("No players meet the criteria.") #
        with c2_rank: #
            st.markdown(f"**Top 10 Teams (min. {min_total*5} attempts)**") #
            if tt_ov is not None and not tt_ov.empty: st.dataframe(tt_ov.rename(columns={'Attempts': 'Attempts'}), use_container_width=True, hide_index=True, column_config={"FG%": st.column_config.ProgressColumn("FG%", format="%.1f%%", min_value=0, max_value=100)}) #
            else: st.caption("No teams meet the criteria.") #

        # --- START: Implemented Expander for 2PT ---
        with st.expander("Show/Hide 2-Point Effectiveness Rankings (2PT FG%)", expanded=False):
            # Indented 2PT ranking code
            c1_rank_2, c2_rank_2 = st.columns(2) #
            with c1_rank_2: #
                st.markdown(f"**Top 10 Players (min. {min_2pt} attempts)**") #
                if tp_2 is not None and not tp_2.empty: st.dataframe(tp_2.rename(columns={'2PT Attempts': '2PT Attempts'}), use_container_width=True, hide_index=True, column_config={"2PT FG%": st.column_config.ProgressColumn("2PT FG%", format="%.1f%%", min_value=0, max_value=100)}) #
                else: st.caption("No players meet the criteria.") #
            with c2_rank_2: #
                st.markdown(f"**Top 10 Teams (min. {min_2pt*5} attempts)**") #
                if tt_2 is not None and not tt_2.empty: st.dataframe(tt_2.rename(columns={'2PT Attempts': '2PT Attempts'}), use_container_width=True, hide_index=True, column_config={"2PT FG%": st.column_config.ProgressColumn("2PT FG%", format="%.1f%%", min_value=0, max_value=100)}) #
                else: st.caption("No teams meet the criteria.") #
        # --- END: Implemented Expander for 2PT ---

        # --- START: Implemented Expander for 3PT ---
        with st.expander("Show/Hide 3-Point Effectiveness Rankings (3PT FG%)", expanded=False):
             # Indented 3PT ranking code
            c1_rank_3, c2_rank_3 = st.columns(2) #
            with c1_rank_3: #
                st.markdown(f"**Top 10 Players (min. {min_3pt} attempts)**") #
                if tp_3 is not None and not tp_3.empty: st.dataframe(tp_3.rename(columns={'3PT Attempts': '3PT Attempts'}), use_container_width=True, hide_index=True, column_config={"3PT FG%": st.column_config.ProgressColumn("3PT FG%", format="%.1f%%", min_value=0, max_value=100)}) #
                else: st.caption("No players meet the criteria.") #
            with c2_rank_3: #
                st.markdown(f"**Top 10 Teams (min. {min_3pt*5} attempts)**") #
                if tt_3 is not None and not tt_3.empty: st.dataframe(tt_3.rename(columns={'3PT Attempts': '3PT Attempts'}), use_container_width=True, hide_index=True, column_config={"3PT FG%": st.column_config.ProgressColumn("3PT FG%", format="%.1f%%", min_value=0, max_value=100)}) #
                else: st.caption("No teams meet the criteria.") #
        # --- END: Implemented Expander for 3PT ---


    with tab_player:
        # All code for the Player Analysis tab (no changes needed for this request)
        st.header(f"Player Analysis: {selected_player}")
        if selected_player and 'PLAYER_NAME' in filtered_data.columns:
            player_data = filter_data_by_player(selected_player, filtered_data) # Filter data for selected player
            if not player_data.empty:
                st.subheader("Basic Statistics")
                total_shots = len(player_data)
                made_shots, shooting_pct = "N/A", "N/A"
                if 'SHOT_MADE_FLAG' in player_data.columns:
                    made_flag_numeric = pd.to_numeric(player_data['SHOT_MADE_FLAG'], errors='coerce').dropna()
                    if not made_flag_numeric.empty:
                        made_shots = int(made_flag_numeric.sum())
                        if len(made_flag_numeric) > 0: shooting_pct = (made_shots / len(made_flag_numeric)) * 100
                        else: shooting_pct = 0.0
                    else: made_shots, shooting_pct = 0, 0.0
                pct_2pt, attempts_2pt_str = "N/A", "(no data)"
                pct_3pt, attempts_3pt_str = "N/A", "(no data)"
                if 'SHOT_TYPE' in player_data.columns and not player_data['SHOT_TYPE'].isnull().all():
                    shot_type_2pt = '2PT Field Goal'
                    data_2pt = player_data[player_data['SHOT_TYPE'] == shot_type_2pt]
                    made_flag_2pt = pd.to_numeric(data_2pt['SHOT_MADE_FLAG'], errors='coerce').dropna()
                    attempts_2pt = len(made_flag_2pt)
                    if attempts_2pt > 0: made_2pt = int(made_flag_2pt.sum()); pct_2pt = (made_2pt / attempts_2pt) * 100; attempts_2pt_str = f"({made_2pt}/{attempts_2pt})"
                    else: attempts_2pt_str = "(0 attempts)"
                    shot_type_3pt = '3PT Field Goal'
                    data_3pt = player_data[player_data['SHOT_TYPE'] == shot_type_3pt]
                    made_flag_3pt = pd.to_numeric(data_3pt['SHOT_MADE_FLAG'], errors='coerce').dropna()
                    attempts_3pt = len(made_flag_3pt)
                    if attempts_3pt > 0: made_3pt = int(made_flag_3pt.sum()); pct_3pt = (made_3pt / attempts_3pt) * 100; attempts_3pt_str = f"({made_3pt}/{attempts_3pt})"
                    else: attempts_3pt_str = "(0 attempts)"
                avg_dist = "N/A"
                if 'SHOT_DISTANCE' in player_data.columns and pd.api.types.is_numeric_dtype(player_data['SHOT_DISTANCE']):
                    valid_distances = player_data['SHOT_DISTANCE'].dropna()
                    if not valid_distances.empty: avg_dist = valid_distances.mean()
                st.markdown("##### Overall Stats")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Shots", total_shots)
                c2.metric("Shots Made", f"{made_shots}" if isinstance(made_shots, int) else "N/A")
                c3.metric("Effectiveness (FG%)", f"{shooting_pct:.1f}%" if isinstance(shooting_pct, (float, int)) else "N/A")
                st.markdown("##### Detailed Stats")
                c4, c5, c6 = st.columns(3)
                c4.metric(f"2pt Effectiveness {attempts_2pt_str}", f"{pct_2pt:.1f}%" if isinstance(pct_2pt, (float, int)) else "N/A")
                c5.metric(f"3pt Effectiveness {attempts_3pt_str}", f"{pct_3pt:.1f}%" if isinstance(pct_3pt, (float, int)) else "N/A")
                c6.metric("Avg. Distance (feet)", f"{avg_dist:.1f}" if isinstance(avg_dist, (float, int)) else "N/A")
                st.markdown("---")
                st.subheader("Shot Visualizations")
                fig_p_chart = plot_shot_chart(player_data, selected_player, "Player")
                if fig_p_chart: st.plotly_chart(fig_p_chart, use_container_width=True)
                else: st.warning("Cannot generate shot chart for this player (no data?).")
                st.markdown("---")
                st.subheader("Effectiveness vs Distance")
                cc1, cc2 = st.columns([1,2])
                with cc1: bin_w = st.slider("Bin Width (feet):", 1, 5, 1, key='player_eff_dist_bin')
                with cc2: min_att = st.slider("Min. attempts per bin:", 1, 50, 5, key='player_eff_dist_min')
                fig_p_eff_dist = plot_player_eff_vs_distance(player_data, selected_player, bin_width=bin_w, min_attempts_per_bin=min_att)
                if fig_p_eff_dist: st.plotly_chart(fig_p_eff_dist, use_container_width=True)
                else: st.caption(f"Insufficient data for effectiveness vs distance chart (min. {min_att} attempts in {bin_w} ft bin).")
                st.markdown("---")
                st.subheader("Shooting Zones ('Hot Zones')")
                hz_min_shots = st.slider("Min. shots per zone:", 3, 50, 5, key='player_hotzone_min_shots')
                hz_bins = st.slider("Number of zones per axis:", 5, 15, 10, key='player_hotzone_bins')
                p_hot_zones = calculate_hot_zones(player_data, min_shots_in_zone=hz_min_shots, n_bins=hz_bins)
                if p_hot_zones is not None and not p_hot_zones.empty:
                    fig_p_hot = plot_hot_zones_heatmap(p_hot_zones, selected_player, "Player", min_shots_in_zone=hz_min_shots)
                    if fig_p_hot: st.plotly_chart(fig_p_hot, use_container_width=True)
                    else: st.info("Cannot generate hot zone map.")
                else: st.info(f"Insufficient data for hot zone analysis (min. {hz_min_shots} attempts per zone).")
                st.markdown("---")
                st.subheader("Temporal Analysis")
                q_min_shots = st.slider("Min. attempts per Quarter/OT:", 3, 50, 5, key='player_quarter_min_shots')
                fig_p_q = plot_player_quarter_eff(player_data, selected_player, min_attempts=q_min_shots)
                if fig_p_q: st.plotly_chart(fig_p_q, use_container_width=True)
                else: st.info(f"Insufficient data for quarter effectiveness analysis (min. {q_min_shots} attempts).")
                m_min_shots = st.slider("Min. attempts per month:", 5, 100, 10, key='player_month_min_shots')
                fig_p_t = plot_player_season_trend(player_data, selected_player, min_monthly_attempts=m_min_shots)
                if fig_p_t: st.plotly_chart(fig_p_t, use_container_width=True)
                else: st.info(f"Insufficient data for monthly trend analysis (min. {m_min_shots} monthly attempts and/or min. 2 months).")
                st.markdown("---")
                st.subheader("Analysis by Action Type / Zone")
                g_min_shots = st.slider("Min. attempts per group:", 3, 50, 5, key='player_group_min_shots')
                action_type_col_player = 'ACTION_TYPE'
                action_choice_player = 'Original'
                if 'ACTION_TYPE_SIMPLE' in player_data.columns:
                    action_choice_options_player = ('Original', 'Simplified')
                    default_action_index_player = 1
                    action_choice_player = st.radio("Show action types for effectiveness:", action_choice_options_player, key='player_action_type_choice', horizontal=True, index=default_action_index_player)
                    if action_choice_player == 'Simplified': action_type_col_player = 'ACTION_TYPE_SIMPLE'
                cg1, cg2 = st.columns(2)
                with cg1:
                    st.markdown(f"###### Effectiveness by Action Type ({action_choice_player})")
                    fig_p_a = plot_grouped_effectiveness(player_data, action_type_col_player, selected_player, "Player", top_n=10, min_attempts=g_min_shots)
                    if fig_p_a: st.plotly_chart(fig_p_a, use_container_width=True)
                    else: st.caption(f"Insufficient data by action type '{action_choice_player}' (min. {g_min_shots} attempts).")
                with cg2:
                    st.markdown("###### Effectiveness by Basic Zone")
                    fig_p_z = plot_grouped_effectiveness(player_data, 'SHOT_ZONE_BASIC', selected_player, "Player", top_n=7, min_attempts=g_min_shots)
                    if fig_p_z: st.plotly_chart(fig_p_z, use_container_width=True)
                    else: st.caption(f"Insufficient data by basic zone (min. {g_min_shots} attempts).")
            else:
                st.warning(f"No data for player '{selected_player}' in the selected season type '{selected_season_type}'.")
        else:
            st.info("Select a player from the sidebar to see their analysis.")

    with tab_compare:
        # All code for the Player Comparison tab (no changes needed for this request)
        # Widget key prefix: comp_
        st.header("Player Comparison")
        if len(selected_players_compare) >= 2:
            st.write(f"Comparing: {', '.join(selected_players_compare)}")
            compare_data_filtered = filtered_data[filtered_data['PLAYER_NAME'].isin(selected_players_compare)].copy()
            if not compare_data_filtered.empty:
                st.subheader("Effectiveness vs Distance")
                comp_eff_dist_bin = st.slider("Bin Width (feet):", 1, 5, 3, key='comp_eff_dist_bin')
                comp_eff_dist_min = st.slider("Min. attempts per bin:", 3, 50, 5, key='comp_eff_dist_min')
                fig_comp_eff_dist = plot_comparison_eff_distance(compare_data_filtered, selected_players_compare, bin_width=comp_eff_dist_bin, min_attempts_per_bin=comp_eff_dist_min)
                if fig_comp_eff_dist: st.plotly_chart(fig_comp_eff_dist, use_container_width=True)
                else: st.caption(f"Insufficient data for effectiveness vs distance comparison (min. {comp_eff_dist_min} attempts in {comp_eff_dist_bin} ft bin).")
                st.markdown("---")
                st.subheader("Effectiveness by Shot Zone")
                min_attempts_zone = st.slider("Min. attempts per zone:", 3, 50, 5, key='comp_zone_min')
                fig_comp_zone = plot_comparison_eff_by_zone(compare_data_filtered, selected_players_compare, min_shots_per_zone=min_attempts_zone)
                if fig_comp_zone: st.plotly_chart(fig_comp_zone, use_container_width=True)
                else: st.caption(f"Insufficient data for comparison by zone (min. {min_attempts_zone} attempts per zone).")
                st.markdown("---")
                st.subheader("Shot Charts")
                num_players_comp = len(selected_players_compare)
                cols = st.columns(num_players_comp)
                for i, player in enumerate(selected_players_compare):
                    with cols[i]:
                        st.markdown(f"**{player}**")
                        player_comp_data = compare_data_filtered[compare_data_filtered['PLAYER_NAME'] == player]
                        if not player_comp_data.empty:
                            chart_key = f"comp_chart_{player.replace(' ','_').replace('.','')}"
                            fig_comp_chart = plot_shot_chart(player_comp_data, player, "Player")
                            if fig_comp_chart:
                                fig_comp_chart.update_layout(height=450, title="")
                                st.plotly_chart(fig_comp_chart, use_container_width=True, key=chart_key)
                            else: st.caption("Error generating chart.")
                        else: st.caption("No data in selected season.")
            else:
                st.warning(f"No data for selected players to compare in season '{selected_season_type}'.")
        else:
            st.info("Select min. 2 players to compare from the sidebar.")

    with tab_team:
        # All code for the Team Analysis tab (no changes needed for this request)
        # Widget key prefix: team_
        st.header(f"Team Analysis: {selected_team}")
        if selected_team and 'TEAM_NAME' in filtered_data.columns:
            team_data = filter_data_by_team(selected_team, filtered_data) # Filter data for selected team
            if not team_data.empty:
                st.subheader("Basic Team Statistics")
                t_stats = get_basic_stats(team_data, selected_team, "Team")
                c1, c2, c3 = st.columns(3)
                c1.metric("Team Shots", t_stats.get('total_shots', 'N/A'))
                c2.metric("Team Shots Made", t_stats.get('made_shots', 'N/A'))
                pct_val = t_stats.get('shooting_pct')
                c3.metric("Team Effectiveness", f"{pct_val:.1f}%" if isinstance(pct_val, (float, int)) else "N/A")
                st.markdown("---")
                st.subheader("Team Shot Visualizations")
                fig_t_c = plot_shot_chart(team_data, selected_team, "Team")
                if fig_t_c: st.plotly_chart(fig_t_c, use_container_width=True)
                else: st.warning("Cannot generate team shot chart.")
                st.subheader("Team Effectiveness vs Distance")
                t_eff_dist_bin = st.slider("Bin Width (feet):", 1, 5, 2, key='team_eff_dist_bin')
                t_eff_dist_min = st.slider("Min. attempts per bin:", 5, 100, 10, key='team_eff_dist_min')
                fig_t_ed = plot_player_eff_vs_distance(team_data, selected_team, bin_width=t_eff_dist_bin, min_attempts_per_bin=t_eff_dist_min) # Reusing player function
                if fig_t_ed: st.plotly_chart(fig_t_ed, use_container_width=True)
                else: st.caption(f"Insufficient data for team effectiveness vs distance chart (min. {t_eff_dist_min} attempts / {t_eff_dist_bin} ft).")
                st.markdown("---")
                st.subheader("Team Shooting Zones ('Hot Zones')")
                t_hz_min_shots = st.slider("Min. shots per zone:", 5, 100, 10, key='team_hotzone_min_shots')
                t_hz_bins = st.slider("Number of zones per axis:", 5, 15, 10, key='team_hotzone_bins')
                t_hz = calculate_hot_zones(team_data, min_shots_in_zone=t_hz_min_shots, n_bins=t_hz_bins)
                if t_hz is not None and not t_hz.empty:
                    fig_t_h = plot_hot_zones_heatmap(t_hz, selected_team, "Team", min_shots_in_zone=t_hz_min_shots)
                    if fig_t_h: st.plotly_chart(fig_t_h, use_container_width=True)
                    else: st.info("Cannot generate team hot zone map.")
                else: st.info(f"Insufficient data for team zone analysis (min. {t_hz_min_shots} attempts/zone).")
                st.markdown("---")
                st.subheader("Team Temporal Analysis")
                t_q_min_shots = st.slider("Min. attempts per Quarter/OT:", 5, 100, 10, key='team_quarter_min_shots')
                fig_t_q = plot_player_quarter_eff(team_data, selected_team, "Team", min_attempts=t_q_min_shots) # Reusing player function
                if fig_t_q: st.plotly_chart(fig_t_q, use_container_width=True)
                else: st.info(f"Insufficient data for team quarter analysis (min. {t_q_min_shots} attempts).")
                st.markdown("---")
                st.subheader("Team Analysis by Action Type / Zone")
                t_g_min_shots = st.slider("Min. attempts per group:", 5, 100, 10, key='team_group_min_shots')
                action_type_col_team = 'ACTION_TYPE'
                action_choice_team = 'Original'
                if 'ACTION_TYPE_SIMPLE' in team_data.columns:
                    action_choice_options_team = ('Original', 'Simplified')
                    default_action_index_team = 1
                    action_choice_team = st.radio("Show action types for team effectiveness:", action_choice_options_team, key='team_action_type_choice', horizontal=True, index=default_action_index_team)
                    if action_choice_team == 'Simplified': action_type_col_team = 'ACTION_TYPE_SIMPLE'
                cg1, cg2 = st.columns(2)
                with cg1:
                    st.markdown(f"###### Team Effectiveness by Action Type ({action_choice_team})")
                    fig_t_a = plot_grouped_effectiveness(team_data, action_type_col_team, selected_team, "Team", top_n=10, min_attempts=t_g_min_shots) # Reusing player function
                    if fig_t_a: st.plotly_chart(fig_t_a, use_container_width=True)
                    else: st.caption(f"Insufficient team data by action type '{action_choice_team}' (min. {t_g_min_shots} attempts).")
                with cg2:
                    st.markdown("###### Team Effectiveness by Basic Zone")
                    fig_t_z = plot_grouped_effectiveness(team_data, 'SHOT_ZONE_BASIC', selected_team, "Team", top_n=7, min_attempts=t_g_min_shots) # Reusing player function
                    if fig_t_z: st.plotly_chart(fig_t_z, use_container_width=True)
                    else: st.caption(f"Insufficient team data by basic zone (min. {t_g_min_shots} attempts).")
            else:
                st.warning(f"No data for team '{selected_team}' in the selected season type '{selected_season_type}'.")
        else:
            st.info("Select a team from the sidebar to see its analysis.")


    # ================== START KNN TAB MODIFICATIONS ==================
    with tab_knn:
        # Widget key prefix in this view: knn_eval_
        st.header(f"Predictive Model Evaluation (KNN) for: {selected_player}")

        # --- START CHANGE: Define action_type_col_model variable BEFORE description ---
        action_type_col_model = 'ACTION_TYPE' # Default value
        use_simple_action_knn_checked = True # Default checkbox state
        if 'ACTION_TYPE_SIMPLE' in filtered_data.columns: # Check if simplified exists at all
            if selected_player and 'PLAYER_NAME' in filtered_data.columns:
                player_model_data_check = filter_data_by_player(selected_player, filtered_data)
                if 'ACTION_TYPE_SIMPLE' in player_model_data_check.columns:
                    use_simple_action_knn_checked = st.checkbox("Use simplified action types (ACTION_TYPE_SIMPLE) in KNN model?", value=True, key='knn_eval_use_simple_action')
                    if use_simple_action_knn_checked:
                        action_type_col_model = 'ACTION_TYPE_SIMPLE'
                    # St.caption informing about the choice will be shown below, after the description, for better flow
                else:
                    action_type_col_model = 'ACTION_TYPE' # Fallback if column somehow missing for specific player
            else:
                action_type_col_model = 'ACTION_TYPE' # Fallback if no player selected yet
        else:
            action_type_col_model = 'ACTION_TYPE' # Fallback if column doesn't exist in data
        # --- END CHANGE ---


        # --- START CHANGE: Update detailed description for KNN ---
        st.markdown(f"""
        ### K-Nearest Neighbors (KNN) Model Evaluation for Player: {selected_player}

        In this section, we analyze the ability of the **K-Nearest Neighbors (KNN)** model to predict the outcome of a shot (whether it was made or missed - `SHOT_MADE_FLAG`) for the selected player: **{selected_player}**.

        **How does KNN work?**
        KNN is a machine learning algorithm belonging to the group of so-called "lazy learning" (instance-based learning). It doesn't build an explicit model during the training phase. Instead, to predict the outcome of a new shot, the model finds the **k** (user-set number) most similar shots from the training data (the nearest "neighbors" in the feature space) and assigns the new shot the outcome that occurs most frequently among these neighbors. Similarity is measured using a distance metric (most commonly Euclidean).

        **Features Used (Attributes):**
        The model uses the following information to predict the shot outcome:
        * **Position on the court:** `LOC_X`, `LOC_Y` (X and Y coordinates of the shot)
        * **Shot distance:** `SHOT_DISTANCE` (distance from the basket in feet)
        * **Action type:** `{action_type_col_model}` (type of action, e.g., 'Jump Shot', 'Layup'; can be original or simplified version - see checkbox below)
        * **Shot type:** `SHOT_TYPE` (e.g., '2PT Field Goal', '3PT Field Goal')
        * **Shot zone area:** `SHOT_ZONE_AREA` (area on the court where the shot was taken, e.g., 'Left Side(L)', 'Center(C)')
        * **Game period:** `PERIOD` (quarter number or overtime)
        * **Location:** `LOCATION` (game location, e.g., 'Home'/'Away')

        **Data Preparation (Preprocessing):**
        Since KNN relies on distances, appropriate data preparation is crucial:
        1.  **Scaling Numerical Features:** Numerical features (`LOC_X`, `LOC_Y`, `SHOT_DISTANCE`) are **standardized** using `StandardScaler`. This removes the mean and scales to unit variance. This is necessary so that features with larger values (e.g., distance) do not dominate features with smaller values (e.g., coordinates) just because of their scale.
        2.  **Encoding Categorical Features:** Categorical features (`{action_type_col_model}`, `SHOT_TYPE`, `SHOT_ZONE_AREA`, `PERIOD`, `LOCATION`) are transformed into a numerical format using **"One-Hot Encoding"** (`OneHotEncoder`). This creates new, binary columns for each category (dropping one in each feature to avoid multicollinearity - `drop='first'`). The `handle_unknown='ignore'` option ensures robustness against new, unknown categories in the test data.

        The entire data preparation process is integrated with the KNN model into a single pipeline (`Pipeline`), ensuring consistent processing of training and test data.

        **Evaluation Methodology:**
        The performance of the KNN model is assessed in two ways:
        1.  **Cross-Validation (Stratified K-Fold):** The player's data is repeatedly (according to the selected number of folds) split into a training and validation set. The model is trained on the training part and evaluated on the validation part in each fold. This provides a **more stable and reliable assessment** of the model's overall ability to generalize to new data. We use the stratified version (`StratifiedKFold`), which ensures the proportion of classes (made/missed) is maintained in each fold, important for potentially imbalanced data. The average accuracy and its standard deviation across all folds are presented.
        2.  **Evaluation on a Single Train/Test Split:** The data is split once into a training set (used to "teach" the model) and a test set (completely unseen by the model during training). The model trained on the training set is then used to predict on the test set. This allows for **detailed analysis of model errors** on unseen data using:
            * **Accuracy:** Overall percentage of correct predictions.
            * **Classification Report:** Contains metrics like Precision, Recall, and F1-score for each class (Missed/Made), giving insight into how well the model handles each class separately.
            * **Confusion Matrix:** A visual table showing the number of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) predictions.

        **Feature Importance:**
        As mentioned in the dedicated section below, the KNN model does not provide a direct measure of feature importance.
        """)
        # --- END CHANGE ---

        if selected_player:
            # Display information about the selected action column
            st.caption(f"KNN model will use the action type column: `{action_type_col_model}`")

            player_model_data = filter_data_by_player(selected_player, filtered_data)
            if not player_model_data.empty:
                numerical_features = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE']
                # action_type_col_model is already defined above
                # --- START CHANGE: Add LOCATION to categorical features ---
                categorical_features = [action_type_col_model, 'SHOT_TYPE', 'SHOT_ZONE_AREA', 'PERIOD', 'LOCATION'] # <-- ADDED 'LOCATION'
                # --- END CHANGE ---
                target_variable = 'SHOT_MADE_FLAG'
                all_features = numerical_features + categorical_features
                missing_cols = [col for col in all_features + [target_variable] if col not in player_model_data.columns]

                # Removed old checkbox and definition of action_type_col_model from here

                if not missing_cols:
                    pmdc = player_model_data[all_features + [target_variable]].dropna().copy()
                    pmdc[target_variable] = pd.to_numeric(pmdc[target_variable], errors='coerce')
                    pmdc = pmdc.dropna(subset=[target_variable])
                    # Ensure LOCATION is treated as string for OHE
                    for col in categorical_features: pmdc[col] = pmdc[col].astype(str)
                    if pmdc[target_variable].nunique() < 2 and not pmdc.empty:
                        st.warning(f"For player '{selected_player}', only one shot outcome class remains after data processing. Cannot build a classification model.")
                    elif pmdc.empty:
                        st.warning(f"No complete data (after removing NaN) for player '{selected_player}' to build the model.")
                    else:
                        pmdc[target_variable] = pmdc[target_variable].astype(int)
                        min_samples_for_model = 50
                        if len(pmdc) >= min_samples_for_model:
                            st.subheader("KNN Model and Evaluation Configuration")
                            k = st.slider("Number of neighbors (k):", 3, min(25, len(pmdc)//3), 5, 2, key='knn_eval_k')
                            n_splits = st.slider("Number of CV folds:", 3, 10, 5, 1, key='knn_eval_cv_splits')
                            st.markdown("---")
                            st.subheader("Single Test Split Configuration")
                            test_size_percent = st.slider("Test set size (%):", 10, 50, 20, 5, key='knn_eval_test_split', format="%d%%")
                            train_size_percent = 100 - test_size_percent
                            test_size_float = test_size_percent / 100.0
                            if st.button(f"Run KNN Model Evaluation for {selected_player}", key='knn_eval_run_button'):
                                # The preprocessor automatically uses the updated 'categorical_features' list
                                preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_features), ('num', StandardScaler(), numerical_features)], remainder='passthrough')
                                pipeline = Pipeline([('preprocessor', preprocessor), ('knn', KNeighborsClassifier(n_neighbors=k))])
                                X = pmdc[all_features]; y = pmdc[target_variable]
                                st.markdown("---"); st.subheader(f"1. Results of {n_splits}-Fold Cross-Validation (KNN)")
                                with st.spinner(f"Running {n_splits}-fold KNN cross-validation (k={k})..."):
                                    try:
                                        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                                        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                                        st.success("KNN cross-validation completed.")
                                        st.metric("Average accuracy (CV):", f"{scores.mean():.2%}")
                                        st.metric("Standard deviation (CV):", f"{scores.std():.4f}")
                                        st.text("Accuracies in individual folds: " + ", ".join([f"{s:.2%}" for s in scores]))
                                    except Exception as e_cv: st.error(f"An error occurred during KNN cross-validation: {e_cv}")
                                st.markdown("---"); st.subheader(f"2. KNN Evaluation on Train/Test Split ({train_size_percent}%/{test_size_percent}%)")
                                with st.spinner(f"Training and evaluating KNN on single split (k={k})..."):
                                    try:
                                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_float, random_state=42, stratify=y)
                                        if X_test.empty: st.warning("Test set is empty after split. Cannot perform evaluation.")
                                        else:
                                            pipeline.fit(X_train, y_train)
                                            y_pred_single = pipeline.predict(X_test)
                                            accuracy_single = accuracy_score(y_test, y_pred_single)
                                            report_dict = classification_report(y_test, y_pred_single, target_names=['Missed', 'Made'], output_dict=True, zero_division=0)
                                            cm = confusion_matrix(y_test, y_pred_single)
                                            st.success("KNN evaluation on single split completed.")
                                            st.metric("Accuracy (on test split):", f"{accuracy_single:.2%}")
                                            st.subheader("Classification Report (KNN):")
                                            report_df = pd.DataFrame(report_dict).transpose()
                                            st.dataframe(report_df.style.format({'precision': '{:.2%}', 'recall': '{:.2%}', 'f1-score': '{:.2f}', 'support': '{:.0f}'}))
                                            st.subheader("Confusion Matrix (KNN):")
                                            fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Prediction", y="Truth"), x=['Missed (0)', 'Made (1)'], y=['Missed (0)', 'Made (1)'], title="KNN Confusion Matrix (on test split)")
                                            st.plotly_chart(fig_cm, use_container_width=True)

                                            # Explanation about lack of Feature Importance for KNN (already added in v9)
                                            st.markdown("---") # Add separator
                                            st.subheader("Feature Importance (KNN)")
                                            st.info(
                                                """
                                                The K-Nearest Neighbors (KNN) model is a distance-based algorithm and **does not inherently provide a built-in measure of feature importance**, unlike tree-based models (e.g., XGBoost) or linear models.
                                                All features (after scaling) are considered when calculating distances between data points.
                                                """
                                            )

                                    except Exception as e_split: st.error(f"An error occurred during KNN evaluation on single split: {e_split}")
                        else: st.warning(f"Insufficient data ({len(pmdc)}) for player '{selected_player}' to build KNN model. Minimum required: {min_samples_for_model}.")
                else: st.warning(f"Missing required columns to build the model for player '{selected_player}'. Missing columns: {', '.join(missing_cols)}")
            else: st.warning(f"No data for player '{selected_player}' in the selected season type '{selected_season_type}'.")
        else: st.info("Select a player from the sidebar to run KNN model evaluation.")
    # ================== END KNN TAB MODIFICATIONS ==================


    # ================== START XGBOOST TAB MODIFICATIONS (including LOCATION) ==================
    with tab_xgb:
        # Widget key prefix: xgb_eval_
        st.header(f"Predictive Model Evaluation (XGBoost) for: {selected_player}")

        # --- START CHANGE: Define action_type_col_model_xgb variable BEFORE description ---
        action_type_col_model_xgb = 'ACTION_TYPE' # Default value
        use_simple_action_xgb_checked = True # Default checkbox state
        if 'ACTION_TYPE_SIMPLE' in filtered_data.columns: # Check if simplified exists at all
            if selected_player and 'PLAYER_NAME' in filtered_data.columns:
                player_model_data_check_xgb = filter_data_by_player(selected_player, filtered_data)
                if 'ACTION_TYPE_SIMPLE' in player_model_data_check_xgb.columns:
                    use_simple_action_xgb_checked = st.checkbox("Use simplified action types (ACTION_TYPE_SIMPLE) in XGBoost model?", value=True, key='xgb_eval_use_simple_action')
                    if use_simple_action_xgb_checked:
                        action_type_col_model_xgb = 'ACTION_TYPE_SIMPLE'
                    # St.caption informing about the choice will be shown below
                else:
                    action_type_col_model_xgb = 'ACTION_TYPE' # Fallback
            else:
                action_type_col_model_xgb = 'ACTION_TYPE' # Fallback
        else:
            action_type_col_model_xgb = 'ACTION_TYPE' # Fallback
        # --- END CHANGE ---

        # --- START CHANGE: Update detailed description for XGBoost ---
        st.markdown(f"""
        ### XGBoost Model Evaluation for player: {selected_player}

        This tab presents the results of the **XGBoost (Extreme Gradient Boosting)** model trained to predict the shot outcome (made/missed - `SHOT_MADE_FLAG`) for player **{selected_player}**.

        **What is XGBoost?**
        XGBoost is an advanced and highly efficient machine learning algorithm based on the **gradient boosting** technique. It builds an **ensemble of decision trees** sequentially, where each subsequent tree tries to correct the errors made by the previous ones. XGBoost is known for its speed and often achieves **state-of-the-art accuracy** in many classification and regression tasks. It utilizes techniques like regularization (preventing overfitting) and computational optimizations.

        **Features Used (Attributes):**
        The XGBoost model was trained using the **same set of features** as the KNN model to allow for a comparison of their performance. These are:
        * **Position on the court:** `LOC_X`, `LOC_Y`
        * **Shot distance:** `SHOT_DISTANCE`
        * **Action type:** `{action_type_col_model_xgb}` (original or simplified, depending on user choice - see checkbox below)
        * **Shot type:** `SHOT_TYPE`
        * **Shot zone area:** `SHOT_ZONE_AREA` (area on the court where the shot was taken)
        * **Game period:** `PERIOD`
        * **Location:** `LOCATION` (game location, e.g., 'Home'/'Away')

        **Data Preparation (Preprocessing):**
        The **exact same preprocessing pipeline (`Pipeline`)** as used for the KNN model was applied:
        1.  **Scaling Numerical Features:** Standardization (`StandardScaler`) for `LOC_X`, `LOC_Y`, `SHOT_DISTANCE`. Although tree-based models generally do not require feature scaling like distance-based algorithms, applying it here standardizes the process and does not harm the XGBoost model.
        2.  **Encoding Categorical Features:** "One-Hot Encoding" (`OneHotEncoder` with `drop='first'` and `handle_unknown='ignore'`) for `{action_type_col_model_xgb}`, `SHOT_TYPE`, `SHOT_ZONE_AREA`, `PERIOD`, `LOCATION`. This is necessary for the model to process these features.

        Using identical preprocessing is crucial for a **fair comparison** of the performance of both models.

        **Evaluation Methodology:**
        Similar to KNN, XGBoost's performance is assessed in two ways:
        1.  **Cross-Validation (Stratified K-Fold):** Repeated splitting of data and model evaluation in each fold to obtain a **stable estimate** of its generalization ability. Average accuracy and standard deviation are presented.
        2.  **Evaluation on a Single Train/Test Split:** A one-time split of data into training and test sets. This allows for **detailed analysis** of results on unseen data using:
            * Accuracy
            * Classification Report (Precision, Recall, F1-score for both classes)
            * Confusion Matrix

        **Feature Importance (Interpretability):**
        Unlike KNN, XGBoost (as a tree-based model) **provides a built-in measure of feature importance**. In the dedicated section below, a chart shows which of the used features (after transformation, e.g., after OHE) had the **greatest impact** on the decisions made by the ensemble of trees in the XGBoost model. This importance is typically calculated based on how often a feature was used to split the data in the trees or how much "gain" these splits provided in minimizing the model's error function. This helps understand on which information the model based its predictions most heavily.
        """)
        # --- END CHANGE ---

        if selected_player:
            # Display information about the selected action column
            st.caption(f"XGBoost model will use the action type column: `{action_type_col_model_xgb}`")

            player_model_data = filter_data_by_player(selected_player, filtered_data)
            if not player_model_data.empty:
                numerical_features = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE']
                # action_type_col_model_xgb is already defined above
                # --- START CHANGE: Add LOCATION to categorical features ---
                categorical_features = [action_type_col_model_xgb, 'SHOT_TYPE', 'SHOT_ZONE_AREA', 'PERIOD', 'LOCATION'] # <-- ADDED 'LOCATION'
                # --- END CHANGE ---
                target_variable = 'SHOT_MADE_FLAG'
                all_features = numerical_features + categorical_features

                # Removed old checkbox and definition from here

                missing_cols = [col for col in all_features + [target_variable] if col not in player_model_data.columns]
                if not missing_cols:
                    pmdc = player_model_data[all_features + [target_variable]].dropna().copy()
                    pmdc[target_variable] = pd.to_numeric(pmdc[target_variable], errors='coerce')
                    pmdc = pmdc.dropna(subset=[target_variable])
                    # Ensure LOCATION is treated as string for OHE
                    for col in categorical_features: pmdc[col] = pmdc[col].astype(str)

                    if pmdc[target_variable].nunique() < 2 and not pmdc.empty:
                        st.warning(f"For player '{selected_player}', only one shot outcome class remains after data processing. Cannot build a classification model.")
                    elif pmdc.empty:
                        st.warning(f"No complete data (after removing NaN) for player '{selected_player}' to build the model.")
                    else:
                        pmdc[target_variable] = pmdc[target_variable].astype(int)
                        min_samples_for_model = 50
                        if len(pmdc) >= min_samples_for_model:
                            st.subheader("XGBoost Model Evaluation Configuration")
                            n_splits_xgb = st.slider("Number of CV folds:", 3, 10, 5, 1, key='xgb_eval_cv_splits')
                            st.markdown("---")
                            st.subheader("Single Test Split Configuration")
                            test_size_percent_xgb = st.slider("Test set size (%):", 10, 50, 20, 5, key='xgb_eval_test_split', format="%d%%")
                            train_size_percent_xgb = 100 - test_size_percent_xgb
                            test_size_float_xgb = test_size_percent_xgb / 100.0

                            if st.button(f"Run XGBoost Model Evaluation for {selected_player}", key='xgb_eval_run_button'):
                                # The preprocessor automatically uses the updated 'categorical_features' list
                                preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_features),('num', StandardScaler(), numerical_features)],remainder='passthrough')
                                pipeline_xgb = Pipeline([('preprocessor', preprocessor),('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))])
                                X = pmdc[all_features]; y = pmdc[target_variable]

                                # 1. XGBoost Cross-Validation
                                st.markdown("---"); st.subheader(f"1. Results of {n_splits_xgb}-Fold Cross-Validation (XGBoost)")
                                with st.spinner(f"Running {n_splits_xgb}-fold XGBoost cross-validation..."):
                                    try:
                                        cv_xgb = StratifiedKFold(n_splits=n_splits_xgb, shuffle=True, random_state=42)
                                        scores_xgb = cross_val_score(pipeline_xgb, X, y, cv=cv_xgb, scoring='accuracy', n_jobs=-1)
                                        st.success("XGBoost cross-validation completed.")
                                        st.metric("Average accuracy (CV):", f"{scores_xgb.mean():.2%}")
                                        st.metric("Standard deviation (CV):", f"{scores_xgb.std():.4f}")
                                        st.text("Accuracies in individual folds: " + ", ".join([f"{s:.2%}" for s in scores_xgb]))
                                    except Exception as e_cv_xgb: st.error(f"An error occurred during XGBoost cross-validation: {e_cv_xgb}")

                                # 2. XGBoost Evaluation on single split
                                st.markdown("---"); st.subheader(f"2. XGBoost Evaluation on Train/Test Split ({train_size_percent_xgb}%/{test_size_percent_xgb}%)")
                                with st.spinner(f"Training and evaluating XGBoost on single split..."):
                                    try:
                                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_float_xgb, random_state=42, stratify=y)
                                        if X_test.empty: st.warning("Test set is empty after split.")
                                        else:
                                            pipeline_xgb.fit(X_train, y_train)
                                            y_pred_xgb = pipeline_xgb.predict(X_test)
                                            accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
                                            report_dict_xgb = classification_report(y_test, y_pred_xgb, target_names=['Missed', 'Made'], output_dict=True, zero_division=0)
                                            cm_xgb = confusion_matrix(y_test, y_pred_xgb)

                                            st.success("XGBoost evaluation on single split completed.")
                                            st.metric("Accuracy (on test split):", f"{accuracy_xgb:.2%}")

                                            st.subheader("Classification Report (XGBoost):")
                                            report_df_xgb = pd.DataFrame(report_dict_xgb).transpose()
                                            st.dataframe(report_df_xgb.style.format({'precision': '{:.2%}', 'recall': '{:.2%}', 'f1-score': '{:.2f}', 'support': '{:.0f}'}))

                                            st.subheader("Confusion Matrix (XGBoost):")
                                            fig_cm_xgb = px.imshow(cm_xgb, text_auto=True, labels=dict(x="Prediction", y="Truth"), x=['Missed (0)', 'Made (1)'], y=['Missed (0)', 'Made (1)'], title="XGBoost Confusion Matrix (on test split)", color_continuous_scale=px.colors.sequential.Reds)
                                            st.plotly_chart(fig_cm_xgb, use_container_width=True)

                                            # Feature Importance for XGBoost (already added in v9)
                                            st.markdown("---")
                                            st.subheader("3. Feature Importance (XGBoost)")
                                            try:
                                                xgb_model = pipeline_xgb.named_steps['xgb']
                                                preprocessor_xgb = pipeline_xgb.named_steps['preprocessor']
                                                try:
                                                    # This should now include names from SHOT_ZONE_AREA and LOCATION after OHE
                                                    feature_names_out = preprocessor_xgb.get_feature_names_out()
                                                except AttributeError:
                                                    # Fallback for older sklearn versions
                                                    feature_names_out = []
                                                    try:
                                                        cat_encoder = preprocessor_xgb.named_transformers_['cat']
                                                        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features) # Uses updated list
                                                        feature_names_out.extend(cat_feature_names)
                                                    except KeyError: pass
                                                    try:
                                                         if 'num' in preprocessor_xgb.named_transformers_:
                                                             feature_names_out.extend(numerical_features)
                                                    except KeyError: pass
                                                    if not feature_names_out:
                                                         st.warning("Could not automatically retrieve feature names after transformation (older scikit-learn version?). Importance plot might lack labels.")
                                                         feature_names_out = [f"feature_{i}" for i in range(len(xgb_model.feature_importances_))]

                                                importances = xgb_model.feature_importances_

                                                if len(feature_names_out) == len(importances):
                                                    importance_df = pd.DataFrame({'Feature': feature_names_out,'Importance': importances})
                                                    importance_df = importance_df.sort_values(by='Importance', ascending=False)
                                                    top_n = 15
                                                    importance_df_top = importance_df.head(top_n)
                                                    fig_importance = px.bar(
                                                        importance_df_top, x='Importance', y='Feature', orientation='h',
                                                        title=f'Top {top_n} Most Important Features (XGBoost) for {selected_player}',
                                                        labels={'Importance': 'Importance (according to XGBoost)', 'Feature': 'Feature'}, text='Importance'
                                                    )
                                                    fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
                                                    fig_importance.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                                                    st.plotly_chart(fig_importance, use_container_width=True)
                                                    st.caption("Feature importance in XGBoost is usually measured by how often a feature is used to split in the trees and how much improvement (e.g., reduction in error/impurity) the split brings.")
                                                else:
                                                    st.error(f"Mismatch between number of feature names ({len(feature_names_out)}) and number of importances ({len(importances)}). Cannot generate importance plot.")
                                            except Exception as e_importance:
                                                st.error(f"An error occurred during calculation or visualization of XGBoost feature importance: {e_importance}")

                                    except Exception as e_split_xgb: st.error(f"Error evaluating XGBoost on split: {e_split_xgb}")
                        else: st.warning(f"Insufficient data ({len(pmdc)}) for '{selected_player}'. Minimum: {min_samples_for_model}.")
                else: st.warning(f"Missing required columns for '{selected_player}'. Missing: {', '.join(missing_cols)}")
            else: st.warning(f"No data for '{selected_player}' in season '{selected_season_type}'.")
        else: st.info("Select a player to evaluate XGBoost.")
    # ================== END XGBOOST TAB MODIFICATIONS ==================


    # ================== START MODEL COMPARISON TAB MODIFICATIONS ==================
    with tab_model_comp:
        # All code for the Model Comparison tab 
        # Widget key prefix in this view: model_comp_
        st.header(f"Predictive Model Comparison (KNN vs XGBoost) for: {selected_player}")

        # --- START CHANGE: Update description (including LOCATION) ---
        st.markdown(f"""
        This tab allows for a direct comparison of the performance of KNN and XGBoost models for player **{selected_player}**.
        The same input data and the same preprocessing pipeline (scaling numerical features, One-Hot Encoding categorical features including **SHOT_ZONE_AREA** and **LOCATION**) are used for both models to ensure a fair comparison.
        Results from cross-validation and a single test split are compared.
        """)
        # --- END CHANGE ---

        if selected_player:
            player_model_data = filter_data_by_player(selected_player, filtered_data)
            if not player_model_data.empty:
                numerical_features = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE']
                action_type_col_model_comp = 'ACTION_TYPE'
                if 'ACTION_TYPE_SIMPLE' in player_model_data.columns:
                    use_simple_action_comp = st.checkbox("Use simplified action types in both models for comparison?", value=True, key='comp_model_use_simple_action')
                    if use_simple_action_comp: action_type_col_model_comp = 'ACTION_TYPE_SIMPLE'
                    st.caption(f"Models for comparison will use the column: `{action_type_col_model_comp}`")
                # --- START CHANGE: Add LOCATION to categorical features ---
                categorical_features = [action_type_col_model_comp, 'SHOT_TYPE', 'SHOT_ZONE_AREA', 'PERIOD', 'LOCATION']
                # --- END CHANGE ---
                target_variable = 'SHOT_MADE_FLAG'
                all_features = numerical_features + categorical_features
                missing_cols = [col for col in all_features + [target_variable] if col not in player_model_data.columns]
                if not missing_cols:
                    pmdc = player_model_data[all_features + [target_variable]].dropna().copy()
                    pmdc[target_variable] = pd.to_numeric(pmdc[target_variable], errors='coerce')
                    pmdc = pmdc.dropna(subset=[target_variable])
                    # Ensure LOCATION is treated as string for OHE
                    for col in categorical_features: pmdc[col] = pmdc[col].astype(str)
                    if pmdc[target_variable].nunique() < 2 and not pmdc.empty:
                        st.warning(f"For player '{selected_player}', only one shot outcome class remains. Cannot compare models.")
                    elif pmdc.empty:
                        st.warning(f"No complete data for player '{selected_player}' to compare models.")
                    else:
                        pmdc[target_variable] = pmdc[target_variable].astype(int)
                        min_samples_for_model = 50
                        if len(pmdc) >= min_samples_for_model:
                            st.subheader("Model Comparison Configuration")
                            k_comp = st.slider("Number of neighbors (k for KNN):", 3, min(25, len(pmdc)//3), 5, 2, key='model_comp_knn_k')
                            n_splits_comp = st.slider("Number of CV folds:", 3, 10, 5, 1, key='model_comp_cv_splits')
                            test_size_percent_comp = st.slider("Test set size (%):", 10, 50, 20, 5, key='model_comp_test_split', format="%d%%")
                            train_size_percent_comp = 100 - test_size_percent_comp
                            test_size_float_comp = test_size_percent_comp / 100.0
                            if st.button(f"Run KNN vs XGBoost Comparison for {selected_player}", key='model_comp_run_button'):
                                X = pmdc[all_features]; y = pmdc[target_variable]
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_float_comp, random_state=42, stratify=y)
                                if X_test.empty: st.error("Test set is empty after split. Cannot compare models.")
                                else:
                                    # The preprocessor automatically uses the updated 'categorical_features' list
                                    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_features), ('num', StandardScaler(), numerical_features)], remainder='passthrough')
                                    pipeline_knn = Pipeline([('preprocessor', preprocessor), ('knn', KNeighborsClassifier(n_neighbors=k_comp))])
                                    pipeline_xgb = Pipeline([('preprocessor', preprocessor), ('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))])
                                    results = {"knn": {}, "xgb": {}}
                                    st.markdown("---"); st.subheader(f"Cross-Validation Comparison ({n_splits_comp} folds)")
                                    col_cv1, col_cv2 = st.columns(2)
                                    with col_cv1:
                                        st.markdown(f"**KNN (k={k_comp})**")
                                        with st.spinner(f"CV KNN..."):
                                            try:
                                                cv_knn = StratifiedKFold(n_splits=n_splits_comp, shuffle=True, random_state=42)
                                                scores_knn = cross_val_score(pipeline_knn, X, y, cv=cv_knn, scoring='accuracy', n_jobs=-1)
                                                results["knn"]["cv_mean"] = scores_knn.mean(); results["knn"]["cv_std"] = scores_knn.std()
                                                st.metric("Avg. Accuracy:", f"{results['knn']['cv_mean']:.2%}"); st.metric("Std. Dev.:", f"{results['knn']['cv_std']:.4f}")
                                            except Exception as e: results["knn"]["cv_mean"] = "Error"; st.error(f"Error CV KNN: {e}")
                                    with col_cv2:
                                        st.markdown("**XGBoost**")
                                        with st.spinner(f"CV XGBoost..."):
                                            try:
                                                cv_xgb = StratifiedKFold(n_splits=n_splits_comp, shuffle=True, random_state=42)
                                                scores_xgb = cross_val_score(pipeline_xgb, X, y, cv=cv_xgb, scoring='accuracy', n_jobs=-1)
                                                results["xgb"]["cv_mean"] = scores_xgb.mean(); results["xgb"]["cv_std"] = scores_xgb.std()
                                                st.metric("Avg. Accuracy:", f"{results['xgb']['cv_mean']:.2%}"); st.metric("Std. Dev.:", f"{results['xgb']['cv_std']:.4f}")
                                            except Exception as e: results["xgb"]["cv_mean"] = "Error"; st.error(f"Error CV XGBoost: {e}")
                                    st.markdown("---"); st.subheader(f"Comparison on Test Split ({test_size_percent_comp}%)")
                                    st.caption(f"Comparison on {len(y_test)} test samples.")
                                    col_s1, col_s2 = st.columns(2)
                                    with col_s1:
                                        st.markdown(f"**KNN (k={k_comp})**")
                                        with st.spinner("Evaluating KNN..."):
                                            try:
                                                pipeline_knn.fit(X_train, y_train)
                                                y_pred_knn = pipeline_knn.predict(X_test)
                                                results["knn"]["single_acc"] = accuracy_score(y_test, y_pred_knn)
                                                results["knn"]["conf_matrix"] = confusion_matrix(y_test, y_pred_knn)
                                                st.metric("Accuracy:", f"{results['knn']['single_acc']:.2%}")
                                                fig_cm_knn = px.imshow(results["knn"]["conf_matrix"], text_auto=True, x=['M(0)', 'M(1)'], y=['M(0)', 'M(1)'], title=f"KNN Confusion Matrix")
                                                fig_cm_knn.update_layout(height=300, title_font_size=14); st.plotly_chart(fig_cm_knn, use_container_width=True)
                                            except Exception as e: results["knn"]["single_acc"] = "Error"; st.error(f"Error evaluating KNN: {e}")
                                    with col_s2:
                                        st.markdown("**XGBoost**")
                                        with st.spinner("Evaluating XGBoost..."):
                                            try:
                                                pipeline_xgb.fit(X_train, y_train)
                                                y_pred_xgb = pipeline_xgb.predict(X_test)
                                                results["xgb"]["single_acc"] = accuracy_score(y_test, y_pred_xgb)
                                                results["xgb"]["conf_matrix"] = confusion_matrix(y_test, y_pred_xgb)
                                                st.metric("Accuracy:", f"{results['xgb']['single_acc']:.2%}")
                                                fig_cm_xgb = px.imshow(results["xgb"]["conf_matrix"], text_auto=True, x=['M(0)', 'M(1)'], y=['M(0)', 'M(1)'], title="XGBoost Confusion Matrix", color_continuous_scale=px.colors.sequential.Reds)
                                                fig_cm_xgb.update_layout(height=300, title_font_size=14); st.plotly_chart(fig_cm_xgb, use_container_width=True)
                                            except Exception as e: results["xgb"]["single_acc"] = "Error"; st.error(f"Error evaluating XGBoost: {e}")
                                    st.markdown("---"); st.subheader("Comparison Summary")
                                    summary_data = { 'Metric': ['Avg. Accuracy (CV)', 'Accuracy (Test)'], 'KNN': [f"{results['knn'].get('cv_mean', 'N/A'):.2%}" if isinstance(results['knn'].get('cv_mean'), float) else "N/A", f"{results['knn'].get('single_acc', 'N/A'):.2%}" if isinstance(results['knn'].get('single_acc'), float) else "N/A"], 'XGBoost': [f"{results['xgb'].get('cv_mean', 'N/A'):.2%}" if isinstance(results['xgb'].get('cv_mean'), float) else "N/A", f"{results['xgb'].get('single_acc', 'N/A'):.2%}" if isinstance(results['xgb'].get('single_acc'), float) else "N/A"] }
                                    summary_df = pd.DataFrame(summary_data)
                                    st.dataframe(summary_df, hide_index=True, use_container_width=True)
                                    winner_cv = "Tie or error"
                                    if isinstance(results['knn'].get('cv_mean'), float) and isinstance(results['xgb'].get('cv_mean'), float):
                                        if results['xgb']['cv_mean'] > results['knn']['cv_mean']: winner_cv = "XGBoost"
                                        elif results['knn']['cv_mean'] > results['xgb']['cv_mean']: winner_cv = "KNN"
                                        else: winner_cv = "Tie"
                                    winner_test = "Tie or error"
                                    if isinstance(results['knn'].get('single_acc'), float) and isinstance(results['xgb'].get('single_acc'), float):
                                        if results['xgb']['single_acc'] > results['knn']['single_acc']: winner_test = "XGBoost"
                                        elif results['knn']['single_acc'] > results['xgb']['single_acc']: winner_test = "KNN"
                                        else: winner_test = "Tie"
                                    st.markdown(f"**Conclusions:**"); st.markdown(f"- Regarding **average CV accuracy**: **{winner_cv}**"); st.markdown(f"- Regarding **test split accuracy**: **{winner_test}**")
                        else: st.warning(f"Insufficient data ({len(pmdc)}) for player '{selected_player}' to compare models. Minimum required: {min_samples_for_model}.")
                else: st.warning(f"Missing required columns to compare models for player '{selected_player}'. Missing columns: {', '.join(missing_cols)}")
            else: st.warning(f"No data for player '{selected_player}' in the selected season type '{selected_season_type}'.")
        else: st.info("Select a player from the sidebar to run model comparison.")
    # ================== END MODEL COMPARISON TAB MODIFICATIONS ==================

# Handling the case where data was not loaded correctly
else:
    st.error("Failed to load or process data. Check the path to the CSV file and its format.")
    if 'load_error_message' in st.session_state and st.session_state.load_error_message:
        st.error(f"Error details: {st.session_state.load_error_message}")

# --- Sidebar Footer ---
st.sidebar.markdown("---")
try:
    # Attempt to use the timezone, fallback to UTC
    try: tz = pytz.timezone('Europe/Warsaw')
    except pytz.exceptions.UnknownTimeZoneError: tz = pytz.utc; print("Warning: Timezone 'Europe/Warsaw' unavailable, using UTC.")
    # Use the current date (2025-04-21)
    current_dt_context = datetime(2025, 4, 21, 18, 12, 51) # Time from context (assuming it's local time in Wroclaw)
    # If we have pytz, localize the time
    if 'pytz' in sys.modules:
        local_tz = pytz.timezone('Europe/Warsaw')
        localized_dt = local_tz.localize(current_dt_context)
        ts = localized_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
    else: # Fallback if pytz is missing
        ts = current_dt_context.strftime('%Y-%m-%d %H:%M:%S (local time - no pytz)')

except ImportError:
    print("Warning: Library 'pytz' is not installed.")
    # Safer fallback without importing sys in the except block
    ts = datetime(2025, 4, 21, 18, 12, 51).strftime('%Y-%m-%d %H:%M:%S (local time - no pytz)')


except Exception as e_time:
    # Even more general fallback
    ts = datetime(2025, 4, 21, 18, 12, 51).strftime('%Y-%m-%d %H:%M:%S (local time - timezone error)')
    print(f"Error setting timezone: {e_time}")

# st.sidebar.markdown(f"Context date: {ts}") # Changed label for clarity