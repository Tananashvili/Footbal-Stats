import json
import re
from pathlib import Path
from datetime import datetime, time

import pandas as pd
import requests
from difflib import SequenceMatcher


# Set True to print chosen markets for debugging.
DEBUG = True

# ---- Statshub ----
BASE_SITE = "https://www.statshub.com"


def fetch_games_today():
    today = datetime.now().date()
    start_dt = datetime.combine(today, time.min)
    end_dt = datetime.combine(today, time.max)
    start_of_day = int(start_dt.timestamp())
    end_of_day = end_dt.timestamp()

    api_url = f"{BASE_SITE}/api/event/by-date"
    params = {"startOfDay": start_of_day, "endOfDay": end_of_day}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": BASE_SITE + "/",
    }
    resp = requests.get(api_url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])


def get_team_history(team_id, game_url, tournament_id, retries=3, backoff_seconds=2):
    api_url = f"https://www.statshub.com/api/team/{team_id}/performance?"
    params = {
        "tournamentId": tournament_id,
        "limit": 10,
        "location": "all",
        "eventHalf": "ALL",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": game_url,
    }
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(api_url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                import time as _time

                _time.sleep(backoff_seconds * (attempt + 1))
    return None


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
    return {"avg_per_match": total / count}


STAT_MAP = {
    "cornerKicks": "corners",
    "cards": "cards",
    "shotsOnGoal": "shots_on_target",
    "totalShotsOnGoal": "total_shots",
    "fouls": "fouls",
    "goalkeeperSaves": "goalkeeper_saves",
    "throwIns": "throw_ins",
    "offsides": "offsides",
}


def build_row(game):
    match_id = game["events"]["id"]
    slug = game["events"]["slug"]
    game_url = f"{BASE_SITE}/fixture/{slug}/{match_id}"
    tournament_id = game["tournaments"]["uniqueTournamentId"]

    home = game["homeTeam"]
    away = game["awayTeam"]

    home_hist = get_team_history(home["id"], game_url, tournament_id)
    away_hist = get_team_history(away["id"], game_url, tournament_id)
    if not home_hist or not away_hist:
        return None

    row = {
        "tournament": game["tournaments"]["name"],
        "matchup": f"{home['name']} - {away['name']}",
        "home_team_shortname": home.get("shortname"),
        "away_team_shortname": away.get("shortname"),
    }

    for stat_key, label in STAT_MAP.items():
        h = calculate_stat_averages(home_hist, stat_key)
        a = calculate_stat_averages(away_hist, stat_key)
        if not h or not a:
            continue
        avg_home = h["avg_per_match"]
        avg_away = a["avg_per_match"]
        row[f"average_{label}"] = (avg_home + avg_away) / 2
        row[f"average_{label}_1"] = avg_home
        row[f"average_{label}_2"] = avg_away

    return row


# ---- Crocobet ----
LEAGUE_IDS = [
    122128,
    122069,
    122144,
    122074,
    122140,
    122088,
    122280,
    122247,
    122230,
    122129,
    122143,
    133614,
]

EVENTS_URL_TEMPLATE = "https://api.crocobet.com/rest/market/categories/multi/{league_id}/events"
EVENT_API_TEMPLATE = "https://api.crocobet.com/rest/market/events/{event_id}"

CROCO_TARGETS = {
    "corners": ["Corners"],
    "cards": ["Under/Over", "cards"],
    "shots_on_target": ["Shots on target"],
    "total_shots": ["Shots"],
    "fouls": ["Fouls"],
    "goalkeeper_saves": ["Saves"],
    "throw_ins": ["Throw-ins"],
    "offsides": ["Offsides"],
}

EXCLUDE_TERMS = ["Team", "Half"]


def is_valid_market(game_name: str, label: str, targets: list) -> bool:
    name_l = game_name.lower()
    for t in targets:
        if t.lower() not in name_l:
            return False
    if any(term.lower() in name_l for term in EXCLUDE_TERMS):
        return False
    if label == "total_shots" and "shots on target" in name_l:
        return False
    return True


def fetch_event(event_id: int) -> dict:
    url = EVENT_API_TEMPLATE.format(event_id=event_id)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Request-Language": "en",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def find_best_argument(data: dict) -> dict:
    event_games = data.get("data", {}).get("eventGames", []) if isinstance(data, dict) else []
    results = {}
    for label, targets in CROCO_TARGETS.items():
        best = None
        best_name = None
        for game in event_games:
            game_name = game.get("gameName", "")
            if not is_valid_market(game_name, label, targets):
                continue
            outcomes = {o.get("outcomeName"): o for o in game.get("outcomes", [])}
            under = outcomes.get("Under")
            over = outcomes.get("Over")
            if not under or not over:
                continue
            under_odds = under.get("outcomeOdds")
            over_odds = over.get("outcomeOdds")
            if under_odds is None or over_odds is None:
                continue
            diff = abs(under_odds - over_odds)
            row = {"argument": game.get("argument"), "diff": diff}
            if best is None or diff < best["diff"]:
                best = row
                best_name = game_name
        if DEBUG:
            results[f"_debug_{label}"] = best_name
        results[label] = best["argument"] if best else None
    return results


def fetch_events_all():
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Request-Language": "en",
    }
    events = []
    for league_id in LEAGUE_IDS:
        url = EVENTS_URL_TEMPLATE.format(league_id=league_id)
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        events.append(resp.json())
    return events


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def best_event_id(matchup: str, events):
    if " - " not in matchup:
        return None
    home, away = matchup.split(" - ", 1)
    home_n = normalize(home)
    away_n = normalize(away)
    best_id = None
    best_score = 0.0
    for payload in events:
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            continue
        for ev in items:
            name = ev.get("eventName") or ev.get("name") or ev.get("matchName") or ""
            if " - " not in name:
                continue
            h, a = name.split(" - ", 1)
            score = (SequenceMatcher(None, home_n, normalize(h)).ratio() +
                     SequenceMatcher(None, away_n, normalize(a)).ratio()) / 2
            if score > best_score:
                best_score = score
                best_id = ev.get("eventId") or ev.get("id") or ev.get("gameId")
    return best_id


def main():
    raw = input("Enter home/away team names (comma-separated matchups or team names): ").strip()
    if not raw:
        return
    needles = [n.strip().lower() for n in raw.split(",") if n.strip()]

    games = fetch_games_today()
    rows = []
    for game in games:
        home = game["homeTeam"]["name"]
        away = game["awayTeam"]["name"]
        home_short = (game["homeTeam"].get("shortname") or "").lower()
        away_short = (game["awayTeam"].get("shortname") or "").lower()
        matchup = f"{home} - {away}".lower()

        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", s.lower())

        match_found = False
        for n in needles:
            if n in matchup or n in home.lower() or n in away.lower() or n in home_short or n in away_short:
                match_found = True
                break
            # Fuzzy match for full matchup strings
            if " - " in n:
                score = SequenceMatcher(None, norm(n), norm(matchup)).ratio()
                if score >= 0.75:
                    match_found = True
                    break
            # Fuzzy match for single team names against home/away/shortnames
            score_home = SequenceMatcher(None, norm(n), norm(home)).ratio()
            score_away = SequenceMatcher(None, norm(n), norm(away)).ratio()
            score_hs = SequenceMatcher(None, norm(n), norm(home_short)).ratio() if home_short else 0
            score_as = SequenceMatcher(None, norm(n), norm(away_short)).ratio() if away_short else 0
            if max(score_home, score_away, score_hs, score_as) >= 0.75:
                match_found = True
                break

        if not match_found:
            continue
        row = build_row(game)
        if row:
            rows.append(row)

    if not rows:
        return

    events = fetch_events_all()
    for row in rows:
        event_id = best_event_id(row["matchup"], events)
        if event_id is None:
            continue
        data = fetch_event(event_id)
        arguments = find_best_argument(data)
        if DEBUG:
            dbg = {k: v for k, v in arguments.items() if k.startswith("_debug_")}
            print(f"Debug markets for {row['matchup']}: {dbg}")
        for label, value in arguments.items():
            if label.startswith("_debug_"):
                continue
            row[f"cb_{label}"] = value

    new_df = pd.DataFrame(rows)
    order = ["tournament", "matchup", "home_team_shortname", "away_team_shortname"]
    for label in STAT_MAP.values():
        avg = f"average_{label}"
        cb = f"cb_{label}"
        if avg in new_df.columns:
            order.append(avg)
        if cb in new_df.columns:
            order.append(cb)
    remaining = [c for c in new_df.columns if c not in order]
    new_df = new_df.reindex(columns=order + remaining)

    out_path = Path("matchup_report.xlsx")
    if out_path.exists():
        existing = pd.read_excel(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        if "matchup" in combined.columns:
            combined = combined.drop_duplicates(subset=["matchup"], keep="last")
        combined.to_excel(out_path, index=False)
    else:
        new_df.to_excel(out_path, index=False)


if __name__ == "__main__":
    main()
