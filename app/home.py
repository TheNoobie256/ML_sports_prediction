import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# 1. Path Setup
# -----------------------------------------------------------------------------
# Add the root directory to the system path so we can import from src/
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.data_ingestion import fetch_live_odds
from src.utils import find_value_bets
from src.model import SportsPredictor

# -----------------------------------------------------------------------------
# 2. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Value Bet Finder",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 ML Sports Predictor & EV Calculator")
st.markdown("""
This dashboard pulls live bookmaker odds and compares them against our XGBoost model's 
win probabilities to identify mathematically profitable (+EV) betting opportunities.
""")

# -----------------------------------------------------------------------------
# 3. Sidebar UI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # In a production app, use st.secrets["API_KEY"] instead of a text input
    api_key = st.text_input("The Odds API Key:", type="password", help="Get a free key from the-odds-api.com")

    sport_choice = st.selectbox(
        "Select Market",
        options=["upcoming", "soccer_epl", "basketball_nba"],
        format_func=lambda x: x.replace("_", " ").title()
    )

    st.markdown("---")
    st.markdown("**How to read the results:**")
    st.markdown("- 🟢 **Green EV**: Model predicts higher probability than the bookie (Value Bet).")
    st.markdown("- 🔴 **Red EV**: Bookie odds are heavily juiced (Bad Bet).")

# -----------------------------------------------------------------------------
# 4. Main Dashboard & Pipeline
# -----------------------------------------------------------------------------
if st.button("Fetch Live Market Data", type="primary"):

    if not api_key:
        st.warning("⚠️ Please enter your API key in the sidebar to continue.")
        st.stop()

    with st.spinner("Pulling live odds and running model inference..."):
        try:
            # Step A: Ingest Live Data
            odds_df = fetch_live_odds(api_key, sport_choice)

            if odds_df.empty:
                st.info("No odds available for this market right now.")
                st.stop()

            st.success(f"Successfully loaded {len(odds_df)} betting lines.")

            # Step B: Initialize Model
            predictor = SportsPredictor()

            # Step C: Model Inference (with fallback for testing)
            try:
                predictor.load_model()

                # If your model is trained, you would engineer features here:
                # live_features_df = engineer_features(odds_df)
                # predictions = [predictor.predict_match(row) for _, row in live_features_df.iterrows()]

                # Placeholder for actual predictions
                model_predictions = np.random.uniform(0.1, 0.9, len(odds_df))

            except FileNotFoundError:
                st.warning(
                    "🧠 Model not found. Using simulated probabilities for UI testing. Train your model in `model.py` to see real predictions.")
                # Generate random probabilities between 10% and 90% for UI testing
                model_predictions = np.random.uniform(0.1, 0.9, len(odds_df))

            # Step D: Calculate Expected Value
            value_df = find_value_bets(odds_df, model_predictions)


            # Step E: UI Styling Magic
            def style_ev(val):
                """Applies green text to positive EV and red to negative EV."""
                try:
                    # Clean the string percentage (e.g., "-5.2%") to a float for logic
                    numeric_val = float(str(val).strip('%'))
                    if numeric_val > 0:
                        return 'color: #00FF00; font-weight: bold;'
                    return 'color: #FF4B4B;'
                except ValueError:
                    return ''


            # Format numbers as percentages
            styled_df = value_df.style.map(style_ev, subset=['EV_ROI']) \
                .format({
                'Model_Prob': '{:.1%}',
                'Bookie_Implied_Prob': '{:.1%}',
                'EV_ROI': '{:.1%}'
            })

            # Render the final interactive table
            st.subheader(f"🔥 Live Value Bets: {sport_choice.replace('_', ' ').title()}")
            st.dataframe(styled_df, use_container_width=True, height=600)

        except Exception as e:
            st.error(f"An error occurred during the pipeline execution: {e}")