import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def engineer_features(df: pd.DataFrame) -> tuple:
    """
    Cleans the CSV and creates predictive features without data leakage.
    """
    # 1. Drop rows missing the target or the Bet365 odds
    df = df.dropna(subset=['FTR', 'B365H', 'B365D', 'B365A']).copy()

    # 2. Target Variable: 1 if Home Team wins, 0 if Away wins or Draw
    df['home_win'] = (df['FTR'] == 'H').astype(int)

    # 3. Features: Implied Probabilities
    # Bookmaker odds are an excellent proxy for team strength (Elo)
    df['prob_home_implied'] = 1 / df['B365H']
    df['prob_away_implied'] = 1 / df['B365A']
    df['prob_draw_implied'] = 1 / df['B365D']

    # 4. Feature: Is the home team the market favorite?
    df['home_favored'] = (df['prob_home_implied'] > df['prob_away_implied']).astype(int)

    features = ['prob_home_implied', 'prob_away_implied', 'prob_draw_implied', 'home_favored']

    return df[features], df['home_win'], features


def main():
    print("📥 Downloading historical EPL data (2023-2024 season)...")
    url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
    df = pd.read_csv(url)

    print("⚙️ Engineering features...")
    X, y, feature_names = engineer_features(df)

    # Train-test split
    # CRITICAL: shuffle=False ensures we train on the past and test on the future
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("🧠 Training XGBoost Model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        objective='binary:logistic',
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate model performance
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Model trained! Test Accuracy: {acc:.2%}")

    # Save the model to disk
    model_path = Path("models/xgb_model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"💾 Model successfully saved to {model_path}")


if __name__ == "__main__":
    main()