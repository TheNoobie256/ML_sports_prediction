import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def add_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values(by='Date').reset_index(drop=True)

    df['home_offensive_form'] = df.groupby('HomeTeam')['HST'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    df['away_offensive_form'] = df.groupby('AwayTeam')['AST'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )

    df['home_offensive_form'] = df['home_offensive_form'].fillna(4.0)
    df['away_offensive_form'] = df['away_offensive_form'].fillna(4.0)

    return df


def engineer_soccer_features(df: pd.DataFrame) -> tuple:
    print("   Calculating Offensive Form (xG equivalent)...")
    df = add_rolling_form(df)

    import json
    latest_form = {}

    for team in df['HomeTeam'].dropna().unique():
        team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
        if not team_matches.empty:
            last_match = team_matches.iloc[-1]
            if last_match['HomeTeam'] == team:
                latest_form[team] = float(last_match['home_offensive_form'])
            else:
                latest_form[team] = float(last_match['away_offensive_form'])

    Path("models").mkdir(parents=True, exist_ok=True)
    with open("models/team_xg.json", "w") as f:
        json.dump(latest_form, f)

    df = df.dropna(subset=['FTR', 'B365H', 'B365A', 'B365D']).copy()

    df['HomeWin'] = (df['FTR'] == 'H').astype(int)

    df['prob_home_implied'] = 1 / df['B365H']
    df['prob_away_implied'] = 1 / df['B365A']
    df['prob_draw_implied'] = 1 / df['B365D']
    df['home_favored'] = (df['prob_home_implied'] > df['prob_away_implied']).astype(int)

    df['home_injury_impact'] = 0.0
    df['away_injury_impact'] = 0.0
    df['injury_differential'] = 0.0

    features = [
        'prob_home_implied', 'prob_away_implied', 'prob_draw_implied', 'home_favored',
        'home_injury_impact', 'away_injury_impact', 'injury_differential',
        'home_offensive_form', 'away_offensive_form'
    ]

    X = df[features].astype('float32')
    y = df['HomeWin'].astype('int32')

    return X, y, features


def main():
    print("📥 Downloading multi-league historical data...")

    SOCCER_LEAGUES = {
        "soccer_epl": "E0",
        "soccer_spain_la_liga": "SP1"
    }

    seasons = ['2324', '2223', '2122', '2021', '1920']
    all_data = []

    for league_name, league_code in SOCCER_LEAGUES.items():
        for season in seasons:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
            print(f"   Fetching {league_name} - Season {season}...")
            try:
                season_df = pd.read_csv(url, encoding='unicode_escape')
                all_data.append(season_df)
            except Exception as e:
                print(f"   ⚠️ Could not download {league_name} {season}: {e}")

    master_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Total historical soccer matches loaded: {len(master_df)}")

    print("⚙️ Engineering features...")
    X, y, feature_names = engineer_soccer_features(master_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("🧠 Training Upgraded Master Soccer Model...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        objective='binary:logistic',
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Master Model trained! Test Accuracy: {acc:.2%}")

    model_path = Path("models/xgb_model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"💾 Master model successfully saved to {model_path}")


if __name__ == "__main__":
    main()