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
        # We define the specific features the model expects
        self.features = ['prob_home_implied', 'prob_away_implied', 'prob_draw_implied', 'home_favored']

    def train(self, historical_data: pd.DataFrame):
        """
        Trains the XGBoost model on historical data and saves it to disk.
        """
        # 1. Prepare the data
        X = historical_data[self.features]
        # Target: 1 if Home wins, 0 if Away wins (simplified for binary sports like NBA)
        y = historical_data['home_win']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. Initialize XGBoost
        # We use objective='binary:logistic' because we want probabilities, not just a win/loss classification
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            objective='binary:logistic',
            random_state=42
        )

        # 3. Train the model
        self.model.fit(X_train, y_train)

        # 4. Evaluate (Good for your own logging)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Model trained. Test Accuracy: {acc:.2f}")

        # 5. Save the model so Streamlit can load it instantly
        # Make sure the models/ directory exists!
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def load_model(self):
        """Loads the pre-trained model from disk."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"No model found at {self.model_path}. Train it first!")
        self.model = joblib.load(self.model_path)

    def predict_match(self, match_features: pd.DataFrame) -> dict:
        """
        Takes live match features and returns the probability of each outcome.
        """
        if self.model is None:
            self.load_model()

        # Ensure the live data has the exact columns the model was trained on
        X_live = match_features[self.features]

        # predict_proba returns an array: [prob_away_win, prob_home_win]
        probabilities = self.model.predict_proba(X_live)[0]

        return {
            "away_win_prob": float(probabilities[0]),
            "home_win_prob": float(probabilities[1])
        }