import pandas as pd
import requests


def fetch_live_odds(api_key: str, sport: str = "upcoming") -> pd.DataFrame:
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us,uk",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.text}")

    data = response.json()

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


def fetch_league_injuries(api_football_key: str, league_id: int, season: int = 2026) -> dict:
    url = "https://v3.football.api-sports.io/injuries"
    headers = {
        "x-apisports-key": api_football_key
    }

    params = {
        "league": league_id,
        "season": season
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        return {}

    data = response.json().get("response", [])

    team_injury_counts = {}

    for injury in data:
        team_name = injury.get("team", {}).get("name")
        if team_name:
            team_injury_counts[team_name] = team_injury_counts.get(team_name, 0) + 1

    return team_injury_counts