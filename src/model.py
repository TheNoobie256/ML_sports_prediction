import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss


class SportsPredictor:
    def __init__(self, model_path: str = "models/xgb_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.features = [
            'prob_home_implied',
            'prob_away_implied',
            'prob_draw_implied',
            'home_favored',
            'home_injury_impact',  # <-- New Feature
            'away_injury_impact',  # <-- New Feature
            'injury_differential'  # <-- New Feature
        ]

    def train(self, historical_data: pd.DataFrame):

        X = historical_data[self.features]
        y = historical_data['home_win']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            objective='binary:logistic',
            random_state=42
        )

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Model trained. Test Accuracy: {acc:.2f}")

        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def load_model(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"No model found at {self.model_path}. Train it first!")
        self.model = joblib.load(self.model_path)

    def predict_match(self, match_features: pd.DataFrame) -> dict:
        if self.model is None:
            self.load_model()

        X_live = match_features[self.features]

        probabilities = self.model.predict_proba(X_live)[0]

        return {
            "away_win_prob": float(probabilities[0]),
            "home_win_prob": float(probabilities[1])
        }