import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Note: We added fetch_fixture_injuries to this import line!
from src.data_ingestion import fetch_live_odds, fetch_league_injuries
from src.utils import find_value_bets
from src.model import SportsPredictor

st.set_page_config(page_title="Value Bet Finder", page_icon="📈", layout="wide")

LEAGUES = {
    "upcoming": "🌍 All Upcoming (Global)",
    "soccer_epl": "⚽ English Premier League",
    "soccer_spain_la_liga": "⚽ Spanish La Liga",
    "soccer_uefa_champs_league": "🏆 UEFA Champions League",
    "basketball_nba": "🏀 NBA Basketball",
    "americanfootball_nfl": "🏈 NFL Football"
}

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")

        if "ODDS_API_KEY" in st.secrets:
            odds_api_key = st.secrets["ODDS_API_KEY"]
            st.success("✅ Odds API Key securely loaded.")
        else:
            odds_api_key = st.text_input("The Odds API Key:", type="password")

        if "API_FOOTBALL_KEY" in st.secrets:
            api_football_key = st.secrets["API_FOOTBALL_KEY"]
            st.success("✅ API-Football Key securely loaded.")
        else:
            api_football_key = st.text_input("API-Football Key:", type="password")

        selected_league_key = st.selectbox(
            "Select League / Market",
            options=list(LEAGUES.keys()),
            format_func=lambda x: LEAGUES[x]
        )

        st.markdown("---")
        st.markdown("**Legend:**\n- 🟢 **Green**: +EV (Value Bet)\n- 🔴 **Red**: -EV (Bad Bet)")

        fetch_button = st.button("Fetch Live Market Data", type="primary", use_container_width=True)

        return odds_api_key, api_football_key, selected_league_key, fetch_button


def render_dynamic_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("### 🔎 Filter Results")
    col1, col2 = st.columns(2)
    with col1:
        all_matches = sorted(df['Matchup'].unique())
        selected_matches = st.multiselect("Filter by Matchup", options=all_matches)
    with col2:
        all_bookies = sorted(df['Bookmaker'].unique())
        selected_bookies = st.multiselect("Filter by Bookmaker", options=all_bookies)

    filtered_df = df.copy()
    if selected_matches:
        filtered_df = filtered_df[filtered_df['Matchup'].isin(selected_matches)]
    if selected_bookies:
        filtered_df = filtered_df[filtered_df['Bookmaker'].isin(selected_bookies)]

    return filtered_df


def style_ev_dataframe(df: pd.DataFrame):
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

st.title("📈 ML Sports Predictor & EV Calculator")

odds_api_key, api_football_key, league_choice, fetch_triggered = render_sidebar()

if fetch_triggered:
    if not odds_api_key or not api_football_key:
        st.warning("⚠️ Please ensure both API keys are entered (or saved in secrets) to continue.")
        st.stop()

    with st.spinner(f"Pulling live odds and fetching injury data for {LEAGUES[league_choice]}..."):
        odds_df = fetch_live_odds(odds_api_key, league_choice)

        if odds_df.empty:
            st.info("No odds available for this market right now. Try a different league.")
            st.stop()

        predictor = SportsPredictor()
        predictor = SportsPredictor()
        try:
            predictor.load_model()
            live_predictions = []
            league_id_map = {
                "soccer_epl": 39,
                "soccer_spain_la_liga": 140,
                "soccer_uefa_champs_league": 2,
            }
            api_league_id = league_id_map.get(league_choice, 39)

            st.info("Fetching league-wide injury reports...")
            league_injuries = fetch_league_injuries(api_football_key, api_league_id, season=2026)

            for index, row in odds_df.iterrows():

                try:
                    home_team, away_team = row['Matchup'].split(" vs ")
                except ValueError:
                    home_team, away_team = "Unknown", "Unknown"

                home_injuries = league_injuries.get(home_team, 0)
                away_injuries = league_injuries.get(away_team, 0)

                BASE_INJURY_WEIGHT = 1.5
                home_impact = home_injuries * BASE_INJURY_WEIGHT
                away_impact = away_injuries * BASE_INJURY_WEIGHT
                differential = away_impact - home_impact

                live_feature_row = pd.DataFrame([{
                    'prob_home_implied': 1 / row['Odds'],
                    'prob_away_implied': 1 / row.get('Away_Odds', row['Odds']),
                    'prob_draw_implied': 1 / row.get('Draw_Odds', row['Odds']),
                    'home_favored': 1 if row.get('Odds', 0) < row.get('Away_Odds', 1) else 0,
                    'home_injury_impact': home_impact,
                    'away_injury_impact': away_impact,
                    'injury_differential': differential
                }])

                prediction = predictor.predict_match(live_feature_row)
                if row['Team'] == home_team:
                    live_predictions.append(prediction['home_win_prob'])
                elif row['Team'] == away_team:
                    live_predictions.append(prediction['away_win_prob'])
                else:
                    live_predictions.append(0.0)

        except FileNotFoundError:
            st.warning(
                "🧠 Model not found! Please run `python src/train.py` in your terminal to train your upgraded 7-feature model.")
            live_predictions = np.random.uniform(0.1, 0.9, len(odds_df))

        # Calculate EV and save to session state
        value_df = find_value_bets(odds_df, live_predictions)
        st.session_state['value_df'] = value_df

if 'value_df' in st.session_state:
    st.divider()
    filtered_data = render_dynamic_filters(st.session_state['value_df'])

    st.subheader(f"🔥 Live Value Bets: {LEAGUES[league_choice]}")
    st.caption(f"Showing {len(filtered_data)} betting lines.")

    styled_table = style_ev_dataframe(filtered_data)
    st.dataframe(styled_table, use_container_width=True, height=500)