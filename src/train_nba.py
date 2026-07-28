import pandas as pd
import xgboost as xgb
import joblib
import os
import kagglehub
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_and_merge_nba_data() -> pd.DataFrame:
    print("📥 Downloading NBA Dataset (This may take a few minutes on the very first run)...")
    path = kagglehub.dataset_download("chevronronson/nba-stats-dataset")
    print(f"✅ Data ready at: {path}")

    print("⚙️ Loading and joining SQL tables...")
    games_df = pd.read_csv(os.path.join(path, "csv", "games_index.csv"))
    odds_df = pd.read_csv(os.path.join(path, "csv", "game_odds.csv"))

    master_df = pd.merge(games_df, odds_df, on="game_id", how="inner")

    return master_df


def engineer_nba_features(df: pd.DataFrame) -> tuple:
    """
    Engineers basketball-specific features for the XGBoost model
    with strict data sanitization to prevent compilation crashes.
    """
    # 1. Standardize columns using the exact names from your Kaggle dataset
    df = df.rename(columns={
        'pts_home': 'HomeScore',
        'pts_away': 'AwayScore',
        'decimal_home': 'HomeOdds',
        'decimal_away': 'AwayOdds'
    })

    # 2. SANITIZATION: Force odds to be numeric. Any weird strings (like "-") become NaN.
    df['HomeOdds'] = pd.to_numeric(df['HomeOdds'], errors='coerce')
    df['AwayOdds'] = pd.to_numeric(df['AwayOdds'], errors='coerce')

    # 3. SANITIZATION: Drop rows with missing odds AND filter out <= 0 odds to prevent Infinity
    df = df.dropna(subset=['HomeScore', 'AwayScore', 'HomeOdds', 'AwayOdds']).copy()
    df = df[(df['HomeOdds'] > 0) & (df['AwayOdds'] > 0)]

    # 4. Create the Target Variable (1 if Home Team won, 0 if Away Team won)
    df['HomeWin'] = (df['HomeScore'] > df['AwayScore']).astype(int)

    # 5. Features: Implied Probabilities
    df['prob_home_implied'] = 1 / df['HomeOdds']
    df['prob_away_implied'] = 1 / df['AwayOdds']
    df['home_favored'] = (df['prob_home_implied'] > df['prob_away_implied']).astype(int)

    # 6. NBA-Specific Features
    df['home_back_to_back'] = df.get('home_b2b', 0.0)
    df['away_back_to_back'] = df.get('away_b2b', 0.0)

    # The exact 5 features the NBA prediction array expects
    features = [
        'prob_home_implied', 'prob_away_implied', 'home_favored',
        'home_back_to_back', 'away_back_to_back'
    ]

    # 7. SANITIZATION: Force the final arrays to strict mathematical types
    X = df[features].astype('float32')
    y = df['HomeWin'].astype('int32')

    return X, y, features


def main():
    # 1. Fetch and merge the data
    df = load_and_merge_nba_data()
    print(f"✅ Total historical NBA records loaded: {len(df)}")

    # 2. Engineer features
    print("⚙️ Engineering NBA-specific features...")
    X, y, feature_names = engineer_nba_features(df)

    # 3. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 4. Train the Model
    print("🧠 Training NBA XGBoost Model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        objective='binary:logistic',
        random_state=42
    )
    model.fit(X_train, y_train)

    # 5. Evaluate Accuracy
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ NBA Model trained! Test Accuracy: {acc:.2%}")

    # 6. Save as a separate file
    model_path = Path("models/xgb_nba_model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"💾 NBA Model successfully saved to {model_path}")


if __name__ == "__main__":
    main()