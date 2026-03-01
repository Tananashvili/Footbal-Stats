from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

import config


GREEN_FILL = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
RED_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
YELLOW_FILL = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_side(avg: float, cb: float) -> str:
    if avg < cb:
        return "under"
    if avg > cb:
        return "over"
    return "push"


def normalize_matchup(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def fetch_events_lookup(timeout: int) -> dict[str, dict]:
    now_utc = datetime.now(timezone.utc)
    start = datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc) - timedelta(days=3)
    end = datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc) + timedelta(days=3)
    params = {"startOfDay": int(start.timestamp()), "endOfDay": int(end.timestamp())}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": config.BASE_SITE + "/",
    }
    try:
        resp = requests.get(
            f"{config.BASE_SITE}/api/event/by-date",
            params=params,
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return {}

    payload = resp.json()
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return {}

    lookup = {}
    for game in rows:
        if not isinstance(game, dict):
            continue
        home = game.get("homeTeam") or {}
        away = game.get("awayTeam") or {}
        events = game.get("events") or {}
        matchup = f"{home.get('name', '')} - {away.get('name', '')}"
        key = normalize_matchup(matchup)
        if not key:
            continue
        event_id = events.get("id")
        home_team_id = home.get("id")
        if event_id is None or home_team_id is None:
            continue
        lookup[key] = {
            "event_id": int(event_id),
            "home_team_id": int(home_team_id),
            "status": str(events.get("status") or "").strip().lower(),
        }
    return lookup


def fetch_event(match_id: int, timeout: int) -> dict | None:
    url = f"{config.BASE_SITE}/api/event/{match_id}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": config.BASE_SITE + "/",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    payload = resp.json()
    return payload if isinstance(payload, dict) else None


def fetch_team_history(team_id: int, timeout: int) -> list[dict]:
    url = f"{config.BASE_SITE}/api/team/{team_id}/performance"
    params = {
        "limit": config.GAME_STATS_HISTORY_LIMIT,
        "location": "all",
        "eventHalf": "ALL",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": config.BASE_SITE + "/",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException:
        return []
    payload = resp.json()
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    return rows if isinstance(rows, list) else []


def find_match_row(history_rows: list[dict], event_id: int) -> dict | None:
    for row in history_rows:
        if not isinstance(row, dict):
            continue
        event = row.get("event") or {}
        if event.get("id") == event_id:
            return row
    return None


def compute_actual_total(match_row: dict | None, stat_label: str) -> float | None:
    if not match_row:
        return None
    stat_key = config.STATSHUB_STAT_KEYS.get(stat_label)
    if not stat_key:
        return None
    stats = match_row.get("statistics") or {}
    opp_stats = match_row.get("opponentStatistics") or {}
    a = parse_float(stats.get(stat_key))
    b = parse_float(opp_stats.get(stat_key))
    if a is None or b is None:
        return None
    return a + b


def evaluate(side: str, cb_value: float, actual_total: float | None) -> tuple[str, PatternFill]:
    if actual_total is None:
        return "N/A", YELLOW_FILL
    if side == "under":
        return ("GREEN", GREEN_FILL) if actual_total < cb_value else ("RED", RED_FILL)
    if side == "over":
        return ("GREEN", GREEN_FILL) if actual_total > cb_value else ("RED", RED_FILL)
    return ("GREEN", GREEN_FILL) if actual_total == cb_value else ("RED", RED_FILL)


def main():
    input_path: Path = config.EXCEL_PATH
    output_path = input_path.with_name(f"{input_path.stem}_results.xlsx")

    if not input_path.exists():
        raise SystemExit(f"Input workbook not found: {input_path}")

    wb = load_workbook(input_path)
    if config.SHEET_SUMMARY not in wb.sheetnames:
        raise SystemExit(f"'{config.SHEET_SUMMARY}' sheet not found in {input_path}")

    # matchup -> match_id from full sheet
    full_df = pd.read_excel(input_path, sheet_name="full")
    matchup_to_match_id = {}
    if "matchup" in full_df.columns and "match_id" in full_df.columns:
        for _, item in full_df.iterrows():
            key = normalize_matchup(item.get("matchup"))
            if not key or key in matchup_to_match_id:
                continue
            match_id = item.get("match_id")
            try:
                matchup_to_match_id[key] = int(match_id)
            except (TypeError, ValueError):
                continue

    ws = wb[config.SHEET_SUMMARY]
    by_date_lookup = fetch_events_lookup(config.HTTP_TIMEOUT_SECONDS)

    header_row = 1
    headers = {}
    for col in range(1, ws.max_column + 1):
        name = ws.cell(row=header_row, column=col).value
        if isinstance(name, str) and name.strip():
            headers[name.strip()] = col

    next_col = ws.max_column + 1
    result_cols = {}
    for stat in config.STAT_ORDER:
        avg_col_name = f"average_{stat}"
        cb_col_name = f"cb_{stat}"
        if avg_col_name not in headers or cb_col_name not in headers:
            continue

        result_col_name = f"result_{stat}"
        if result_col_name in headers:
            result_cols[stat] = headers[result_col_name]
        else:
            ws.cell(row=header_row, column=next_col, value=result_col_name)
            result_cols[stat] = next_col
            next_col += 1

    event_cache: dict[int, dict | None] = {}
    history_cache: dict[int, list[dict]] = {}

    for row in range(2, ws.max_row + 1):
        matchup_col = headers.get("matchup")
        matchup = ws.cell(row=row, column=matchup_col).value if matchup_col else None
        matchup_key = normalize_matchup(matchup)
        match_id = matchup_to_match_id.get(matchup_key)

        event_data = None
        home_team_id = None
        event_status = None
        if match_id is not None:
            if match_id not in event_cache:
                event_cache[match_id] = fetch_event(match_id, config.HTTP_TIMEOUT_SECONDS)
            event_data = event_cache[match_id]
            if event_data:
                events = event_data.get("events") or {}
                event_status = str(events.get("status") or "").strip().lower()
                home_team = event_data.get("homeTeam") or {}
                home_team_id = home_team.get("id")
                if event_data.get("events", {}).get("id"):
                    match_id = int(event_data["events"]["id"])

        if match_id is None and matchup_key in by_date_lookup:
            fallback = by_date_lookup[matchup_key]
            match_id = fallback["event_id"]
            home_team_id = fallback["home_team_id"]
            event_status = fallback["status"]

        match_row = None
        if match_id is not None and home_team_id is not None:
            home_team_id = int(home_team_id)
            if home_team_id not in history_cache:
                history_cache[home_team_id] = fetch_team_history(home_team_id, config.HTTP_TIMEOUT_SECONDS)
            match_row = find_match_row(history_cache[home_team_id], match_id)

        for stat in config.STAT_ORDER:
            avg_col_name = f"average_{stat}"
            cb_col_name = f"cb_{stat}"
            if avg_col_name not in headers or cb_col_name not in headers:
                continue

            avg_col = headers[avg_col_name]
            cb_col = headers[cb_col_name]
            result_col = result_cols.get(stat)
            if result_col is None:
                continue

            avg_value = parse_float(ws.cell(row=row, column=avg_col).value)
            cb_value = parse_float(ws.cell(row=row, column=cb_col).value)

            if avg_value is None or cb_value is None:
                ws.cell(row=row, column=result_col, value="N/A").fill = YELLOW_FILL
                ws.cell(row=row, column=avg_col).fill = YELLOW_FILL
                ws.cell(row=row, column=cb_col).fill = YELLOW_FILL
                continue

            side = infer_side(avg_value, cb_value)
            actual_total = compute_actual_total(match_row, stat)

            if event_status and event_status != "finished":
                result_text = f"{side.upper()} {cb_value:g} | not finished"
                fill = YELLOW_FILL
            else:
                outcome, fill = evaluate(side, cb_value, actual_total)
                actual_text = f"{actual_total:g}" if actual_total is not None else "N/A"
                result_text = f"{side.upper()} {cb_value:g} | actual {actual_text} -> {outcome}"

            ws.cell(row=row, column=result_col, value=result_text).fill = fill
            ws.cell(row=row, column=avg_col).fill = fill
            ws.cell(row=row, column=cb_col).fill = fill

    wb.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
