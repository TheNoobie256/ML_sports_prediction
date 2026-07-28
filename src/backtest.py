import pandas as pd
from pathlib import Path
from model import SportsPredictor


def run_backtest(df: pd.DataFrame, predictor: SportsPredictor, bet_size: float = 10.0):
    total_bets = 0
    total_wagered = 0.0
    total_profit = 0.0

    print(f"🔄 Starting backtest on {len(df)} historical matches...")

    for index, row in df.iterrows():
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
        edge = model_home_prob - bookie_home_prob
        ev = (model_home_prob * (row['B365H'] - 1)) - (1 - model_home_prob)

        if ev > 0.05:
            total_bets += 1
            total_wagered += bet_size

            if row['FTR'] == 'H':
                profit = bet_size * (row['B365H'] - 1)
                total_profit += profit
            else:
                total_profit -= bet_size

    roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0

    print("\n" + "=" * 30)
    print("📈 BACKTEST RESULTS")
    print("=" * 30)
    print(f"Total Matches Analyzed: {len(df)}")
    print(f"Value Bets Placed: {total_bets}")
    print(f"Total Wagered: ${total_wagered:,.2f}")
    print(f"Total Profit/Loss: ${total_profit:,.2f}")
    print(f"Return on Investment (ROI): {roi:.2f}%")
    print("=" * 30)


def main():
    print("📥 Fetching 2024-2025 holdout data...")
    url = "https://www.football-data.co.uk/mmz4281/2425/E0.csv"

    try:
        test_df = pd.read_csv(url)
        test_df = test_df.dropna(subset=['FTR', 'B365H', 'B365A', 'B365D'])
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    predictor = SportsPredictor()
    try:
        predictor.load_model(sport_type="soccer")
    except FileNotFoundError:
        print("⚠️ Model not found! Please run train.py first.")
        return

    run_backtest(test_df, predictor, bet_size=10.0)


if __name__ == "__main__":
    main()