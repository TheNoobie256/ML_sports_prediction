import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the root directory to the system path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.data_ingestion import fetch_live_odds
from src.utils import find_value_bets
from src.model import SportsPredictor

# -----------------------------------------------------------------------------
# 1. Configuration & Dictionaries
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Value Bet Finder", page_icon="📈", layout="wide")

# Map API keys to readable names for a better UI experience
LEAGUES = {
    "upcoming": "🌍 All Upcoming (Global)",
    "soccer_epl": "⚽ English Premier League",
    "soccer_spain_la_liga": "⚽ Spanish La Liga",  # <-- Update this line right here
    "soccer_uefa_champs": "🏆 UEFA Champions League",
    "basketball_nba": "🏀 NBA Basketball",
    "americanfootball_nfl": "🏈 NFL Football"
}


# -----------------------------------------------------------------------------
# 2. Modular UI Components
# -----------------------------------------------------------------------------
def render_sidebar():
    """Handles all sidebar inputs and returns the selected configuration."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        if "ODDS_API_KEY" in st.secrets:
            api_key = st.secrets["ODDS_API_KEY"]
            st.success("✅ API Key securely loaded.")
        else:
            api_key = st.text_input("The Odds API Key:", type="password")

        # Use our dictionary to show nice names, but return the raw API key (e.g., 'soccer_epl')
        selected_league_key = st.selectbox(
            "Select League / Market",
            options=list(LEAGUES.keys()),
            format_func=lambda x: LEAGUES[x]
        )

        st.markdown("---")
        st.markdown("**Legend:**\n- 🟢 **Green**: +EV (Value Bet)\n- 🔴 **Red**: -EV (Bad Bet)")

        fetch_button = st.button("Fetch Live Market Data", type="primary", use_container_width=True)

        return api_key, selected_league_key, fetch_button


def render_dynamic_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a row of filters so the user can slice the data dynamically."""
    st.markdown("### 🔎 Filter Results")

    col1, col2 = st.columns(2)

    with col1:
        # Filter by specific matches
        all_matches = sorted(df['Matchup'].unique())
        selected_matches = st.multiselect("Filter by Matchup", options=all_matches)

    with col2:
        # Filter by specific bookmakers (e.g., if you only have accounts with DraftKings and BetMGM)
        all_bookies = sorted(df['Bookmaker'].unique())
        selected_bookies = st.multiselect("Filter by Bookmaker", options=all_bookies)

    # Apply the filters if the user selected anything
    filtered_df = df.copy()
    if selected_matches:
        filtered_df = filtered_df[filtered_df['Matchup'].isin(selected_matches)]
    if selected_bookies:
        filtered_df = filtered_df[filtered_df['Bookmaker'].isin(selected_bookies)]

    return filtered_df


def style_ev_dataframe(df: pd.DataFrame):
    """Applies conditional formatting to the EV column."""

    def apply_color(val):
        try:
            numeric_val = float(str(val).strip('%'))
            return 'color: #00FF00; font-weight: bold;' if numeric_val > 0 else 'color: #FF4B4B;'
        except ValueError:
            return ''

    return df.style.map(apply_color, subset=['EV_ROI']) \
        .format({
        'Model_Prob': '{:.1%}',
        'Bookie_Implied_Prob': '{:.1%}',
        'EV_ROI': '{:.1%}'
    })


# -----------------------------------------------------------------------------
# 3. Main Application Logic
# -----------------------------------------------------------------------------
st.title("📈 ML Sports Predictor & EV Calculator")

# 1. Render the sidebar
api_key, league_choice, fetch_triggered = render_sidebar()

# 2. Execute pipeline only when the button is pressed
if fetch_triggered:
    if not api_key:
        st.warning("⚠️ Please enter your API key in the sidebar to continue.")
        st.stop()

    with st.spinner(f"Pulling live odds for {LEAGUES[league_choice]}..."):
        # Fetch data
        odds_df = fetch_live_odds(api_key, league_choice)

        if odds_df.empty:
            st.info("No odds available for this market right now. Try a different league.")
            st.stop()

        # Run ML Inference (Using mock probabilities if model isn't trained yet)
        predictor = SportsPredictor()
        try:
            predictor.load_model()
            # real_predictions = ... (your feature engineering logic here)
            mock_predictions = np.random.uniform(0.1, 0.9, len(odds_df))
        except FileNotFoundError:
            st.warning("🧠 Model not found. Using simulated probabilities for UI testing.")
            mock_predictions = np.random.uniform(0.1, 0.9, len(odds_df))

        # Calculate EV
        value_df = find_value_bets(odds_df, mock_predictions)

        # Save to Streamlit's session state so it doesn't disappear when we interact with filters
        st.session_state['value_df'] = value_df

# 3. Render the interactive UI if data exists in the session state
if 'value_df' in st.session_state:
    st.divider()

    # Pass the data through our new dynamic filter component
    filtered_data = render_dynamic_filters(st.session_state['value_df'])

    # Display the final, styled table
    st.subheader(f"🔥 Live Value Bets: {LEAGUES[league_choice]}")
    st.caption(f"Showing {len(filtered_data)} betting lines.")

    styled_table = style_ev_dataframe(filtered_data)
    st.dataframe(styled_table, use_container_width=True, height=500)