import pandas as pd
import requests


def fetch_live_odds(api_key: str, sport: str = "upcoming") -> pd.DataFrame:
    """
    Fetches live odds from The Odds API and returns a flattened Pandas DataFrame.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us,uk",
        "markets": "h2h",  # Head-to-head (Moneyline)
        "oddsFormat": "decimal"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.text}")

    data = response.json()

    # Flatten the JSON into a structured table for our model
    flattened_games = []
    for game in data:
        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    flattened_games.append({
                        "Matchup": f"{game['home_team']} vs {game['away_team']}",
                        "Commence_Time": game["commence_time"],
                        "Bookmaker": bookmaker["title"],
                        "Team": outcome["name"],
                        "Odds": outcome["price"]
                    })

    return pd.DataFrame(flattened_games)