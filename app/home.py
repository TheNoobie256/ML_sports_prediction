import sys
from pathlib import Path
import streamlit as st
import pandas as pd

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from src.data_ingestion import fetch_live_odds, fetch_league_injuries
from src.model import SportsPredictor

st.set_page_config(page_title="EV Predictor", layout="wide")
st.title("📈 ML Sports Predictor & EV Calculator")

with st.sidebar:
    st.header("⚙️ Configuration")

    odds_api_key = st.secrets.get("ODDS_API_KEY", "")
    api_football_key = st.secrets.get("API_FOOTBALL_KEY", "")

    if odds_api_key and api_football_key:
        st.success("✅ Successfully read API keys from secrets!")
    else:
        st.warning("⚠️ Secrets not found. Enter keys manually:")
        odds_api_key = st.text_input("Odds API Key", type="password")
        api_football_key = st.text_input("API-Football Key", type="password")

    LEAGUE_DISPLAY_NAMES = {
        "soccer_epl": "English Premier League",
        "soccer_spain_la_liga": "Spanish La Liga",
        "soccer_uefa_champs_league": "UEFA Champions League",
        "basketball_nba": "NBA Basketball"
    }

    league_choice = st.selectbox(
        "Select League",
        options=list(LEAGUE_DISPLAY_NAMES.keys()),
        format_func=lambda x: LEAGUE_DISPLAY_NAMES[x]
    )

    sport_type = "basketball" if "basketball" in league_choice else "soccer"

tab1, tab2 = st.tabs(["🔥 Live Value Bets", "📊 Strategy Backtester"])

with tab1:
    st.subheader("Live Market Dashboard")

    if st.button("Fetch Live Market Data", type="primary"):
        if not odds_api_key or not api_football_key:
            st.warning("Please enter both API keys in the sidebar.")
        else:
            with st.spinner("Fetching live market odds..."):
                odds_df = fetch_live_odds(odds_api_key, league_choice)

                if not odds_df.empty:
                    predictor = SportsPredictor()
                    predictor.load_model(sport_type=sport_type)
                    live_predictions = []

                    if sport_type == "soccer":
                        league_id_map = {
                            "soccer_epl": 39,
                            "soccer_spain_la_liga": 140,
                            "soccer_uefa_champs_league": 2,
                        }
                        api_league_id = league_id_map.get(league_choice, 39)
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
                                'injury_differential': differential,
                                'home_offensive_form': 4.0,
                                'away_offensive_form': 4.0
                            }])

                            prediction = predictor.predict_match(live_feature_row)

                            if row['Team'] == home_team:
                                live_predictions.append(prediction['home_win_prob'])
                            elif row['Team'] == away_team:
                                live_predictions.append(prediction['away_win_prob'])
                            else:
                                live_predictions.append(0.0)

                    elif sport_type == "basketball":
                        for index, row in odds_df.iterrows():
                            try:
                                home_team, away_team = row['Matchup'].split(" vs ")
                            except ValueError:
                                home_team, away_team = "Unknown", "Unknown"

                            live_feature_row = pd.DataFrame([{
                                'prob_home_implied': 1 / row['Odds'],
                                'prob_away_implied': 1 / row.get('Away_Odds', row['Odds']),
                                'home_favored': 1 if row.get('Odds', 0) < row.get('Away_Odds', 1) else 0,
                                'home_back_to_back': 0.0,
                                'away_back_to_back': 0.0
                            }])

                            prediction = predictor.predict_match(live_feature_row)

                            if row['Team'] == home_team:
                                live_predictions.append(prediction['home_win_prob'])
                            elif row['Team'] == away_team:
                                live_predictions.append(prediction['away_win_prob'])
                            else:
                                live_predictions.append(0.0)

                    # Calculate Expected Value (EV)
                    odds_df['Model_Prob'] = live_predictions
                    odds_df['Bookie_Implied_Prob'] = 1 / odds_df['Odds']
                    odds_df['EV_ROI'] = (odds_df['Model_Prob'] * (odds_df['Odds'] - 1)) - (1 - odds_df['Model_Prob'])

                    value_bets = odds_df[odds_df['EV_ROI'] > 0.05].copy()
                    value_bets['Model_Prob'] = (value_bets['Model_Prob'] * 100).map("{:.1f}%".format)
                    value_bets['Bookie_Implied_Prob'] = (value_bets['Bookie_Implied_Prob'] * 100).map("{:.1f}%".format)
                    value_bets['EV_ROI'] = (value_bets['EV_ROI'] * 100).map("{:.1f}%".format)

                    if not value_bets.empty:
                        st.dataframe(value_bets, use_container_width=True)
                    else:
                        st.info("No +EV value bets found in current market lines.")

                else:
                    st.warning(f"No active live market games found for {LEAGUE_DISPLAY_NAMES[league_choice]}. The league may currently be in its off-season.")

with tab2:
    st.subheader("Historical Model Validation")
    st.markdown("Simulate the XGBoost model's performance on past seasons to calculate actual Return on Investment (ROI).")

    col1, col2 = st.columns(2)
    with col1:
        sport_choice_bt = st.selectbox("Select Model to Test", ["Soccer Master", "NBA (Coming Soon)"])
    with col2:
        test_bet_size = st.slider("Flat Bet Size ($)", min_value=1, max_value=100, value=10)

    if st.button("Run Simulation", type="primary", use_container_width=True):
        if sport_choice_bt == "Soccer Master":
            with st.spinner("Running historical simulation on 2024/2025 season..."):
                url = "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
                test_df = pd.read_csv(url).dropna(subset=['FTR', 'B365H', 'B365A', 'B365D'])

                predictor = SportsPredictor()
                predictor.load_model(sport_type="soccer")

                total_wagered = 0.0
                profit = 0.0
                profit_history = [0.0]
                bets_placed = 0

                for index, row in test_df.iterrows():
                    live_features = pd.DataFrame([{
                        'prob_home_implied': 1 / row['B365H'],
                        'prob_away_implied': 1 / row['B365A'],
                        'prob_draw_implied': 1 / row['B365D'],
                        'home_favored': 1 if (1 / row['B365H']) > (1 / row['B365A']) else 0,
                        'home_injury_impact': 0.0,
                        'away_injury_impact': 0.0,
                        'injury_differential': 0.0,
                        'home_offensive_form': 4.0,
                        'away_offensive_form': 4.0
                    }])

                    prediction = predictor.predict_match(live_features)
                    model_home_prob = prediction['home_win_prob']
                    bookie_home_prob = 1 / row['B365H']

                    if (model_home_prob - bookie_home_prob) > 0.05:
                        bets_placed += 1
                        total_wagered += test_bet_size

                        if row['FTR'] == 'H':
                            profit += test_bet_size * (row['B365H'] - 1)
                        else:
                            profit -= test_bet_size

                        profit_history.append(profit)

                roi = (profit / total_wagered) * 100 if total_wagered > 0 else 0

                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Matches Analyzed", len(test_df))
                m2.metric("Value Bets Placed", bets_placed)
                m3.metric("Net Profit", f"${profit:.2f}", delta=f"{roi:.2f}% ROI")
                m4.metric("Total Wagered", f"${total_wagered:.2f}")

                st.markdown("### Cumulative Profit Trajectory")
                st.line_chart(profit_history)