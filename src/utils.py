import pandas as pd


def calculate_ev(model_probability: float, decimal_odds: float) -> float:
    """
    Calculates the Expected Value (ROI) as a percentage.
    Returns a float (e.g., 0.125 for 12.5%).
    """
    return (model_probability * decimal_odds) - 1.0


def implied_probability(decimal_odds: float) -> float:
    """
    Calculates what probability the bookmaker thinks the event has.
    """
    return 1.0 / decimal_odds


def find_value_bets(odds_df: pd.DataFrame, model_predictions: list) -> pd.DataFrame:
    """
    Takes the live odds dataframe and appends our model's predictions and the EV.
    """
    # Assuming 'model_predictions' is a list of probabilities (0.0 to 1.0)
    # matching the exact order of the rows in odds_df
    odds_df['Model_Prob'] = model_predictions
    odds_df['Bookie_Implied_Prob'] = odds_df['Odds'].apply(implied_probability)

    # Calculate the EV
    odds_df['EV_ROI'] = odds_df.apply(
        lambda row: calculate_ev(row['Model_Prob'], row['Odds']), axis=1
    )

    # Sort so the highest +EV bets are at the very top
    odds_df = odds_df.sort_values(by='EV_ROI', ascending=False)

    return odds_df