import json
import os
import time as _time
import requests
from datetime import datetime, time, timedelta
from urllib.parse import urljoin

BASE_SITE = "https://www.statshub.com"

# ---- Calculate start of TODAY and end at 02:00 next day ----
today = datetime.now().date()
start_dt = datetime.combine(today, time.min)          # 00:00:00
end_dt = datetime.combine(today + timedelta(days=1), time(2, 0))  # 02:00:00 next day

startOfDay = int(start_dt.timestamp())
endOfDay = end_dt.timestamp()                          # keep decimals

TOP_5_LEAGUES = ["Premier League", "Serie A", "Ligue 1", "LaLiga", "Bundesliga", 
                 "Eredivisie", "Liga Portugal Betclic", "Super Lig"]


def get_games_data(startOfDay, endOfDay):
    api_url = f'{BASE_SITE}/api/event/by-date'

    params = {
        "startOfDay": startOfDay,
        "endOfDay": endOfDay,
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": BASE_SITE + "/",
    }

    # ---- Request ----
    response = requests.get(api_url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    games_list = data['data']

    game_data_list = []
    for game in games_list:
        match_id = game['events']['id']
        slug = game['events']['slug']
        game_url = f'https://www.statshub.com/fixture/{slug}/{str(match_id)}'

        home_team_id = game['homeTeam']['id']
        home_team_name = game['homeTeam']['name']
        home_team_shortname = game['homeTeam'].get('shortname')
        away_team_id = game['awayTeam']['id']
        away_team_name = game['awayTeam']['name']
        away_team_shortname = game['awayTeam'].get('shortname')

        tournament_id = game['tournaments']['uniqueTournamentId']
        toruname_name = game['tournaments']['name']

        game_data = {'match_id': match_id, 'slug': slug, 'game_url': game_url, 'home_team_id': home_team_id, 'home_team_name': home_team_name,
                     'home_team_shortname': home_team_shortname, 'away_team_id': away_team_id, 'away_team_name': away_team_name,
                     'away_team_shortname': away_team_shortname, 'tournament_id': tournament_id, 'toruname_name': toruname_name}
        
        if game['events']['status'] == "notstarted" and toruname_name in TOP_5_LEAGUES:
            game_data_list.append(game_data)

    return game_data_list


def get_games_history(game_data, team_id, retries=3, backoff_seconds=2):
    game_limit = 10
    api_url = f"https://www.statshub.com/api/team/{team_id}/performance?"

    params = {
        "tournamentId": game_data['tournament_id'],
        "limit": game_limit,
        'location': 'all',
        'eventHalf': 'ALL'
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": game_data['game_url'],
    }

    # ---- Request ----
    last_exc = None
    for attempt in range(retries):
        try:
            response = requests.get(api_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                _time.sleep(backoff_seconds * (attempt + 1))
    return None


def save_games_history(game_data, team_id, output_path="games_history.json"):
    history = get_games_history(game_data, team_id)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return history


def load_games_history(path="games_history.json"):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_stat_averages(history_data, stat_key):
    matches = history_data.get("data", [])
    if not matches:
        return None

    team_total = 0
    opp_total = 0
    count = 0

    for match in matches:
        stats = match.get("statistics") or {}
        opp_stats = match.get("opponentStatistics") or {}
        if stat_key not in stats or stat_key not in opp_stats:
            continue
        team_total += stats[stat_key]
        opp_total += opp_stats[stat_key]
        count += 1

    if count == 0:
        return None

    total = team_total + opp_total
    return {
        "matches": count,
        "avg_per_match": total / count,
        "avg_for_team": team_total / count,
        "avg_against_team": opp_total / count,
    }


def build_stat_averages(history_data):
    stat_map = {
        "cornerKicks": "corners",
        "cards": "cards",
        "shotsOnGoal": "shots_on_target",
        "totalShotsOnGoal": "total_shots",
        "fouls": "fouls",
        "goalkeeperSaves": "goalkeeper_saves",
        "throwIns": "throw_ins",
        "offsides": "offsides",
    }

    results = []
    for stat_key, label in stat_map.items():
        row = calculate_stat_averages(history_data, stat_key)
        if row:
            row["stat"] = label
            results.append(row)
    return results


def build_game_row(game_data, home_history, away_history):
    stat_map = {
        "cornerKicks": "corners",
        "cards": "cards",
        "shotsOnGoal": "shots_on_target",
        "totalShotsOnGoal": "total_shots",
        "fouls": "fouls",
        "goalkeeperSaves": "goalkeeper_saves",
        "throwIns": "throw_ins",
        "offsides": "offsides",
    }

    row = {
        "matchup": f"{game_data['home_team_name']} - {game_data['away_team_name']}",
        "home_team_name": game_data["home_team_name"],
        "away_team_name": game_data["away_team_name"],
        "home_team_shortname": game_data.get("home_team_shortname"),
        "away_team_shortname": game_data.get("away_team_shortname"),
        "match_id": game_data["match_id"],
        "tournament": game_data["toruname_name"],
    }

    for stat_key, label in stat_map.items():
        home_avg = calculate_stat_averages(home_history, stat_key)
        away_avg = calculate_stat_averages(away_history, stat_key)
        if not home_avg or not away_avg:
            continue
        avg_home = home_avg["avg_per_match"]
        avg_away = away_avg["avg_per_match"]
        row[f"average_{label}"] = (avg_home + avg_away) / 2
        row[f"average_{label}_1"] = avg_home
        row[f"average_{label}_2"] = avg_away

    return row


def save_stats_to_excel(rows, output_path="stats_averages.xlsx"):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to write Excel files. Install with: pip install pandas openpyxl"
        ) from exc

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # Second sheet: matchup, tournament, shortnames, and avg_per_match for each stat.
    stat_labels = [
        "corners",
        "cards",
        "shots_on_target",
        "total_shots",
        "fouls",
        "goalkeeper_saves",
        "throw_ins",
        "offsides",
    ]
    summary_cols = [
        "tournament",
        "matchup",
        "home_team_shortname",
        "away_team_shortname",
    ]
    summary_map = {
        "tournament": "tournament",
        "matchup": "matchup",
        "home_team_shortname": "home_team_shortname",
        "away_team_shortname": "away_team_shortname",
    }
    for label in stat_labels:
        for suffix in ("", "_1", "_2"):
            src = f"average_{label}{suffix}"
            summary_cols.append(src)
            summary_map[src] = src

    summary_df = df.reindex(columns=summary_cols).rename(columns=summary_map)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="full", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
    return output_path


if __name__ == "__main__":
    games = get_games_data(startOfDay, endOfDay)
    rows = []

    for game in games:
        home_history = get_games_history(game, game["home_team_id"])
        away_history = get_games_history(game, game["away_team_id"])
        if not home_history or not away_history:
            continue
        row = build_game_row(game, home_history, away_history)
        rows.append(row)

    if rows:
        output_path = save_stats_to_excel(rows)
        if output_path:
            pass
