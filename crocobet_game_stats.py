import json
import re
from pathlib import Path
from datetime import datetime, time, timedelta, timezone

import pandas as pd
import requests
import config


EXCEL_PATH = config.EXCEL_PATH
MATCHES_PATH = config.MATCHES_PATH
EVENT_API_TEMPLATE = config.CROCOBET_EVENT_URL_TEMPLATE
DEBUG_MARKETS_PATH = config.DEBUG_MARKETS_PATH
BASE_SITE = config.BASE_SITE
TEAM_SUMMARY_SHEET = config.SHEET_TEAM_SUMMARY

STAT_TARGETS = config.CROCOBET_STAT_TARGETS
STAT_ORDER = config.STAT_ORDER
EXCLUDE_TERMS = config.CROCOBET_MARKET_EXCLUDE_TERMS
STATSHUB_STAT_KEYS = config.STATSHUB_STAT_KEYS
TEAM_MARKET_TEMPLATES = config.CROCOBET_TEAM_MARKET_TEMPLATES
TEAM_SIDE_ALIASES = {
    "1": ("1st team", "first team"),
    "2": ("2nd team", "second team"),
}
TEAM_MARKET_EXCLUDE_TERMS = ("half", "(ot)", "player", "goalkeeper to get red card")


def normalize_market_text(value: str) -> str:
    return " ".join((value or "").lower().split())


def fetch_event(event_id: int) -> dict:
    url = EVENT_API_TEMPLATE.format(event_id=event_id)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Request-Language": "en",
    }
    resp = requests.get(url, headers=headers, timeout=config.HTTP_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def is_valid_market(game_name: str, label: str) -> bool:
    name = normalize_market_text(game_name)
    if any(term in name for term in EXCLUDE_TERMS):
        return False

    spec = STAT_TARGETS[label]
    phrases = spec.get("phrases", [])
    if phrases and not any(p in name for p in phrases):
        return False

    for phrase in spec.get("exclude_phrases", []):
        if phrase in name:
            return False

    return True


def is_valid_team_market(game_name: str, label: str, side: str) -> bool:
    name = normalize_market_text(game_name)
    side_aliases = TEAM_SIDE_ALIASES.get(side)
    if not side_aliases:
        return False
    if not any(alias in name for alias in side_aliases):
        return False
    if any(term in name for term in TEAM_MARKET_EXCLUDE_TERMS):
        return False

    spec = STAT_TARGETS[label]
    phrases = spec.get("phrases", [])
    if phrases and not any(phrase in name for phrase in phrases):
        return False

    for phrase in spec.get("exclude_phrases", []):
        if phrase in name:
            return False

    return True


def get_event_games(data) -> list:
    if isinstance(data, dict):
        inner = data.get("data")
        if isinstance(inner, dict):
            for key in ("eventGames", "games", "list"):
                value = inner.get(key)
                if isinstance(value, list):
                    return value
            return []
        return []
    return []


def list_market_names(data: dict) -> list:
    return [g.get("gameName", "") for g in get_event_games(data)]


def find_under_over(outcomes: list) -> tuple:
    under = None
    over = None
    for outcome in outcomes:
        name = (outcome.get("outcomeName") or "").strip().lower()
        if name.startswith("under"):
            under = outcome
        elif name.startswith("over"):
            over = outcome
    return under, over


def parse_odds(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_ratio_hits(avg: float, cb: float, values: list[float]) -> int:
    if avg > cb:
        return sum(1 for value in values if value > cb)
    if avg < cb:
        return sum(1 for value in values if value < cb)

    over_hits = sum(1 for value in values if value > cb)
    under_hits = sum(1 for value in values if value < cb)
    push_hits = sum(1 for value in values if value == cb)
    return max(over_hits, under_hits) + push_hits


def fetch_games_for_window() -> list:
    today = datetime.now().date()
    end_date = today + timedelta(days=config.WINDOW_END_DAYS)
    start_ts = int(datetime.combine(today, time.min).timestamp())
    end_ts = int(
        datetime.combine(
            end_date,
            time(config.WINDOW_END_HOUR, 0, 0),
        ).timestamp()
    )

    api_url = f"{BASE_SITE}/api/event/by-date"
    params = {"startOfDay": start_ts, "endOfDay": end_ts}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": BASE_SITE + "/",
    }
    resp = requests.get(
        api_url, params=params, headers=headers, timeout=config.HTTP_TIMEOUT_SECONDS
    )
    resp.raise_for_status()
    data = resp.json()
    rows = data.get("data")
    return rows if isinstance(rows, list) else []


def fetch_team_history(team_id: int, limit: int = config.GAME_STATS_HISTORY_LIMIT) -> list:
    api_url = f"{BASE_SITE}/api/team/{team_id}/performance"
    params = {
        "limit": limit,
        "location": "all",
        "eventHalf": "ALL",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": BASE_SITE + "/",
    }
    resp = requests.get(
        api_url, params=params, headers=headers, timeout=config.HTTP_TIMEOUT_SECONDS
    )
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("data")
    return rows if isinstance(rows, list) else []


def build_team_id_map(games: list) -> dict:
    def norm(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    out = {}
    for game in games:
        if not isinstance(game, dict):
            continue
        home = game.get("homeTeam") or {}
        away = game.get("awayTeam") or {}
        home_name = home.get("name")
        away_name = away.get("name")
        home_id = home.get("id")
        away_id = away.get("id")
        if not isinstance(home_name, str) or not isinstance(away_name, str):
            continue
        if home_id is None or away_id is None:
            continue
        matchup = f"{home_name} - {away_name}"
        out[norm(matchup)] = (int(home_id), int(away_id))
    return out


def build_h2h_matches(
    history_rows: list,
    home_team_id: int,
    away_team_id: int,
    min_year: int = config.H2H_MIN_YEAR,
) -> list[dict]:
    now_ts = int(datetime.now().timestamp())
    seen_event_ids = set()
    out = []

    for row in history_rows:
        if not isinstance(row, dict):
            continue
        event = row.get("event") or {}
        home = row.get("homeTeam") or {}
        away = row.get("awayTeam") or {}
        stats = row.get("statistics") or {}
        opp_stats = row.get("opponentStatistics") or {}

        event_id = event.get("id")
        time_start = event.get("timeStartTimestamp")
        if event_id is None or time_start is None:
            continue
        try:
            event_id = int(event_id)
            ts = int(time_start)
        except (TypeError, ValueError):
            continue

        if event_id in seen_event_ids:
            continue
        if ts >= now_ts:
            continue

        try:
            year = datetime.fromtimestamp(ts, tz=timezone.utc).year
        except (ValueError, OSError):
            continue
        if year < min_year:
            continue

        home_id = home.get("id")
        away_id = away.get("id")
        try:
            ids = {int(home_id), int(away_id)}
        except (TypeError, ValueError):
            continue
        if ids != {home_team_id, away_team_id}:
            continue

        totals = {}
        team_values = {}
        opponent_values = {}
        for label, stat_key in STATSHUB_STAT_KEYS.items():
            a = parse_float(stats.get(stat_key))
            b = parse_float(opp_stats.get(stat_key))
            totals[label] = a + b if a is not None and b is not None else None
            team_values[label] = a
            opponent_values[label] = b

        seen_event_ids.add(event_id)
        out.append(
            {
                "event_id": event_id,
                "timestamp": ts,
                "totals": totals,
                "team_values": team_values,
                "opponent_values": opponent_values,
            }
        )

    out.sort(key=lambda x: x["timestamp"], reverse=True)
    return out


def compute_h2h_ratio(avg_value, cb_value, h2h_matches: list, label: str) -> str | None:
    avg = parse_float(avg_value)
    cb = parse_float(cb_value)
    if avg is None or cb is None:
        return None

    values = [m["totals"].get(label) for m in h2h_matches]
    values = [v for v in values if v is not None]
    if not values:
        return None

    hits = compute_ratio_hits(avg, cb, values)
    return f"{hits}/{len(values)}"


def compute_team_h2h_ratio(
    avg_value,
    cb_value,
    h2h_matches: list,
    label: str,
    side: str,
) -> str | None:
    avg = parse_float(avg_value)
    cb = parse_float(cb_value)
    if avg is None or cb is None:
        return None

    key = "team_values" if side == "1" else "opponent_values"
    values = [m.get(key, {}).get(label) for m in h2h_matches]
    values = [v for v in values if v is not None]
    if not values:
        return None

    hits = compute_ratio_hits(avg, cb, values)
    return f"{hits}/{len(values)}"


def find_best_argument(data: dict) -> dict:
    event_games = get_event_games(data)
    results = {}

    for label in STAT_TARGETS:
        best = None
        for game in event_games:
            game_name = game.get("gameName", "")
            if not is_valid_market(game_name, label):
                continue

            outcomes = game.get("outcomes", [])
            under, over = find_under_over(outcomes)
            if not under or not over:
                continue

            under_odds = parse_odds(under.get("outcomeOdds"))
            over_odds = parse_odds(over.get("outcomeOdds"))
            if under_odds is None or over_odds is None:
                continue

            diff = abs(under_odds - over_odds)
            row = {
                "argument": game.get("argument"),
                "diff": diff,
            }
            if best is None or diff < best["diff"]:
                best = row

        results[label] = best["argument"] if best else None

    return results


def find_best_team_argument(data: dict, side: str) -> dict:
    event_games = get_event_games(data)
    results = {}

    for label in STAT_TARGETS:
        best = None
        for game in event_games:
            game_name = game.get("gameName", "")
            if not is_valid_team_market(game_name, label, side):
                continue

            outcomes = game.get("outcomes", [])
            under, over = find_under_over(outcomes)
            if not under or not over:
                continue

            under_odds = parse_odds(under.get("outcomeOdds"))
            over_odds = parse_odds(over.get("outcomeOdds"))
            if under_odds is None or over_odds is None:
                continue

            diff = abs(under_odds - over_odds)
            row = {
                "argument": game.get("argument"),
                "diff": diff,
            }
            if best is None or diff < best["diff"]:
                best = row

        results[label] = best["argument"] if best else None

    return results


def main():
    if not MATCHES_PATH.exists():
        raise SystemExit("crocobet_event_matches.json not found. Run crocobet_events_fetch.py first.")

    matches = json.loads(MATCHES_PATH.read_text(encoding="utf-8"))
    if isinstance(matches, list):
        def best_record(a, b):
            score_a = float(a.get("score") or 0)
            score_b = float(b.get("score") or 0)
            exact_a = 1 if a.get("matched_name") == a.get("matchup") else 0
            exact_b = 1 if b.get("matched_name") == b.get("matchup") else 0
            key_a = (exact_a, score_a)
            key_b = (exact_b, score_b)
            return a if key_a >= key_b else b

        best_by_matchup = {}
        for rec in matches:
            key = rec.get("matchup") if isinstance(rec, dict) else None
            if not key:
                continue
            if key in best_by_matchup:
                best_by_matchup[key] = best_record(best_by_matchup[key], rec)
            else:
                best_by_matchup[key] = rec
        matches = list(best_by_matchup.values())
    summary_df = pd.read_excel(EXCEL_PATH, sheet_name=config.SHEET_SUMMARY)
    full_df = pd.read_excel(EXCEL_PATH, sheet_name="full")
    team_summary_base_cols = [
        "tournament",
        "matchup",
        "home_team_shortname",
        "away_team_shortname",
    ]
    team_summary_cols = list(team_summary_base_cols)
    for label in STAT_ORDER:
        avg_1 = f"average_{label}_1"
        avg_2 = f"average_{label}_2"
        if avg_1 in full_df.columns:
            team_summary_cols.append(avg_1)
        if avg_2 in full_df.columns:
            team_summary_cols.append(avg_2)
    team_summary_df = (
        full_df.reindex(columns=team_summary_cols)
        .drop_duplicates(subset=["matchup"], keep="last")
        .reset_index(drop=True)
    )

    def norm(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    matchup_to_idx = {}
    for idx, row in summary_df.iterrows():
        matchup = row.get("matchup")
        if isinstance(matchup, str):
            matchup_to_idx[norm(matchup)] = idx
    team_matchup_to_idx = {}
    for idx, row in team_summary_df.iterrows():
        matchup = row.get("matchup")
        if isinstance(matchup, str):
            team_matchup_to_idx[norm(matchup)] = idx

    debug_rows = []

    for m in matches:
        matchup = m.get("matchup")
        event_id = m.get("event_id")
        if not matchup or event_id is None:
            continue
        key = norm(matchup)
        if not key:
            continue
        idx = matchup_to_idx.get(key)
        if idx is None or (isinstance(idx, float) and pd.isna(idx)):
            continue
        team_idx = team_matchup_to_idx.get(key)
        try:
            data = fetch_event(event_id)
        except requests.RequestException:
            continue

        arguments = find_best_argument(data)
        for label, value in arguments.items():
            col = f"cb_{label}"
            if col not in summary_df.columns:
                summary_df[col] = None
            summary_df.at[idx, col] = value

        team_arguments_1 = find_best_team_argument(data, "1")
        team_arguments_2 = find_best_team_argument(data, "2")
        if team_idx is not None:
            for label, value in team_arguments_1.items():
                col = f"cb_{label}_1"
                if col not in team_summary_df.columns:
                    team_summary_df[col] = None
                team_summary_df.at[team_idx, col] = value
            for label, value in team_arguments_2.items():
                col = f"cb_{label}_2"
                if col not in team_summary_df.columns:
                    team_summary_df[col] = None
                team_summary_df.at[team_idx, col] = value

        missing = [label for label, value in arguments.items() if value is None]
        team_missing_1 = [f"{label}_1" for label, value in team_arguments_1.items() if value is None]
        team_missing_2 = [f"{label}_2" for label, value in team_arguments_2.items() if value is None]
        if missing or team_missing_1 or team_missing_2:
            debug_rows.append(
                {
                    "event_id": event_id,
                    "matchup": matchup,
                    "missing": missing + team_missing_1 + team_missing_2,
                    "market_names": list_market_names(data),
                }
            )

    # H2H review (all matches from year >= 2023)
    team_games = fetch_games_for_window()
    matchup_to_team_ids = build_team_id_map(team_games)
    team_history_cache: dict[int, list] = {}

    for idx, row in summary_df.iterrows():
        matchup = row.get("matchup")
        if not isinstance(matchup, str):
            continue
        key = norm(matchup)
        pair = matchup_to_team_ids.get(key)
        if not pair:
            continue

        home_team_id, away_team_id = pair
        if home_team_id not in team_history_cache:
            try:
                team_history_cache[home_team_id] = fetch_team_history(home_team_id)
            except requests.RequestException:
                team_history_cache[home_team_id] = []

        h2h_matches = build_h2h_matches(
            history_rows=team_history_cache[home_team_id],
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            min_year=config.H2H_MIN_YEAR,
        )

        for label in STAT_ORDER:
            h2h_col = f"h2h_{label}"
            avg_col = f"average_{label}"
            cb_col = f"cb_{label}"
            if h2h_col not in summary_df.columns:
                summary_df[h2h_col] = None
            summary_df.at[idx, h2h_col] = compute_h2h_ratio(
                row.get(avg_col),
                row.get(cb_col),
                h2h_matches,
                label,
            )

            for side in ("1", "2"):
                team_avg_col = f"average_{label}_{side}"
                team_cb_col = f"cb_{label}_{side}"
                team_h2h_col = f"h2h_{label}_{side}"
                team_idx = team_matchup_to_idx.get(key)
                if team_idx is None:
                    continue
                if team_h2h_col not in team_summary_df.columns:
                    team_summary_df[team_h2h_col] = None
                team_summary_df.at[team_idx, team_h2h_col] = compute_team_h2h_ratio(
                    team_summary_df.at[team_idx, team_avg_col] if team_avg_col in team_summary_df.columns else None,
                    team_summary_df.at[team_idx, team_cb_col] if team_cb_col in team_summary_df.columns else None,
                    h2h_matches,
                    label,
                    side,
                )

    preferred = ["tournament", "matchup", "home_team_shortname", "away_team_shortname"]
    for label in STAT_ORDER:
        avg_col = f"average_{label}"
        if avg_col in summary_df.columns:
            preferred.append(avg_col)
        cb_col = f"cb_{label}"
        if cb_col in summary_df.columns:
            preferred.append(cb_col)
        h2h_col = f"h2h_{label}"
        if h2h_col in summary_df.columns:
            preferred.append(h2h_col)
    remaining = [c for c in summary_df.columns if c not in preferred]
    summary_df = summary_df.reindex(columns=preferred + remaining)
    round_cols = [
        col for col in summary_df.columns
        if col.startswith("average_") or col.startswith("cb_")
    ]
    if round_cols:
        summary_df.loc[:, round_cols] = summary_df.loc[:, round_cols].round(1)
    team_preferred = ["tournament", "matchup", "home_team_shortname", "away_team_shortname"]
    for label in STAT_ORDER:
        for suffix in ("_1", "_2"):
            avg_col = f"average_{label}{suffix}"
            cb_col = f"cb_{label}{suffix}"
            h2h_col = f"h2h_{label}{suffix}"
            if avg_col in team_summary_df.columns:
                team_preferred.append(avg_col)
            if cb_col in team_summary_df.columns:
                team_preferred.append(cb_col)
            if h2h_col in team_summary_df.columns:
                team_preferred.append(h2h_col)
    team_remaining = [c for c in team_summary_df.columns if c not in team_preferred]
    team_summary_df = team_summary_df.reindex(columns=team_preferred + team_remaining)
    team_round_cols = [
        col for col in team_summary_df.columns
        if col.startswith("average_") or col.startswith("cb_")
    ]
    if team_round_cols:
        team_summary_df.loc[:, team_round_cols] = team_summary_df.loc[:, team_round_cols].round(1)

    writer_kwargs = {"engine": "openpyxl"}
    if EXCEL_PATH.exists():
        writer_kwargs["mode"] = "a"
        writer_kwargs["if_sheet_exists"] = "replace"
    with pd.ExcelWriter(EXCEL_PATH, **writer_kwargs) as writer:
        full_df.to_excel(writer, sheet_name="full", index=False)
        summary_df.to_excel(writer, sheet_name=config.SHEET_SUMMARY, index=False)
        team_summary_df.to_excel(writer, sheet_name=TEAM_SUMMARY_SHEET, index=False)

    if debug_rows:
        DEBUG_MARKETS_PATH.parent.mkdir(exist_ok=True)
        DEBUG_MARKETS_PATH.write_text(
            json.dumps(debug_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
