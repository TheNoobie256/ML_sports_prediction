import joblib
import pandas as pd
from pathlib import Path


class SportsPredictor:
    def __init__(self):
        self.model = None
        self.features = []

    def load_model(self, sport_type: str = "soccer"):
        if sport_type == "basketball":
            model_path = "models/xgb_nba_model.joblib"
            self.features = [
                'prob_home_implied', 'prob_away_implied', 'home_favored',
                'home_back_to_back', 'away_back_to_back'
            ]
        else:
            model_path = "models/xgb_model.joblib"
            self.features = [
                'prob_home_implied', 'prob_away_implied', 'prob_draw_implied', 'home_favored',
                'home_injury_impact', 'away_injury_impact', 'injury_differential',
                'home_offensive_form', 'away_offensive_form'  # <-- New features added here
            ]

        full_path = Path(model_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Model file not found at {full_path}")

        self.model = joblib.load(full_path)

    def predict_match(self, live_features: pd.DataFrame) -> dict:
        X_live = live_features[self.features]
        probabilities = self.model.predict_proba(X_live)[0]

        return {
            'home_win_prob': probabilities[1],
            'away_win_prob': probabilities[0]
        }

    def get_feature_importances(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()

        importances = self.model.feature_importances_

        df = pd.DataFrame({
            "Importance": importances
        }, index=self.features)

        return df.sort_values(by="Importance", ascending=False)