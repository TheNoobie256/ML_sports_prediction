import pandas as pd


def calculate_ev(model_probability: float, decimal_odds: float) -> float:
    return (model_probability * decimal_odds) - 1.0


def implied_probability(decimal_odds: float) -> float:
    return 1.0 / decimal_odds


def find_value_bets(odds_df: pd.DataFrame, model_predictions: list) -> pd.DataFrame:
    odds_df['Model_Prob'] = model_predictions
    odds_df['Bookie_Implied_Prob'] = odds_df['Odds'].apply(implied_probability)

    odds_df['EV_ROI'] = odds_df.apply(
        lambda row: calculate_ev(row['Model_Prob'], row['Odds']), axis=1
    )

    odds_df = odds_df.sort_values(by='EV_ROI', ascending=False)

    return odds_df