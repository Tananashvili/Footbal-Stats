from __future__ import annotations

import time as _time

import requests

import config


BASE_SITE = config.BASE_SITE
STAT_KEYS = tuple(config.STATSHUB_STAT_KEYS.values())

_SEASON_YEAR_CACHE: dict[tuple[int, int | None], str | None] = {}
_TEAM_HISTORY_CACHE: dict[tuple[int, int, str], dict] = {}


def _parse_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _request_json(
    url: str,
    *,
    params: dict,
    headers: dict,
    retries: int = 3,
    backoff_seconds: int = 2,
) -> dict | None:
    last_exc = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=config.HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else None
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                _time.sleep(backoff_seconds * (attempt + 1))
    return None


def get_season_year(
    unique_tournament_id: int,
    season_id: int | None,
    *,
    referer: str | None = None,
) -> str | None:
    cache_key = (int(unique_tournament_id), int(season_id) if season_id is not None else None)
    if cache_key in _SEASON_YEAR_CACHE:
        return _SEASON_YEAR_CACHE[cache_key]

    url = f"{BASE_SITE}/api/unique-tournament/{int(unique_tournament_id)}/seasons"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": referer or BASE_SITE + "/",
    }
    payload = _request_json(url, params={}, headers=headers)
    seasons = payload.get("data", []) if payload else []

    season_year = None
    if isinstance(seasons, list):
        if season_id is not None:
            for season in seasons:
                if not isinstance(season, dict):
                    continue
                if season.get("id") == season_id:
                    season_year = season.get("year")
                    break
        if season_year is None and seasons:
            first = seasons[0]
            if isinstance(first, dict):
                season_year = first.get("year")

    if season_year is not None:
        season_year = str(season_year)
    _SEASON_YEAR_CACHE[cache_key] = season_year
    return season_year


def fetch_team_season_history(
    *,
    team_id: int,
    unique_tournament_id: int,
    season_id: int | None,
    referer: str | None = None,
) -> dict | None:
    season_year = get_season_year(
        unique_tournament_id,
        season_id,
        referer=referer,
    )
    if not season_year:
        return None

    cache_key = (int(team_id), int(unique_tournament_id), season_year)
    if cache_key in _TEAM_HISTORY_CACHE:
        return _TEAM_HISTORY_CACHE[cache_key]

    api_url = f"{BASE_SITE}/api/team/{int(team_id)}/event-statistics"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": referer or BASE_SITE + "/",
    }
    common_params = {
        "eventType": "all",
        "eventHalf": "ALL",
        "tournamentIds": str(int(unique_tournament_id)),
        "season": season_year,
    }

    matches_by_event: dict[int, dict] = {}
    for stat_key in STAT_KEYS:
        payload = _request_json(
            api_url,
            params={**common_params, "statisticKey": stat_key},
            headers=headers,
        )
        rows = payload.get("data", []) if payload else None
        if not isinstance(rows, list):
            return None

        for row in rows:
            if not isinstance(row, dict):
                continue
            event_id = row.get("event_id")
            if event_id is None:
                continue
            try:
                event_id = int(event_id)
                home_team_id = int(row.get("home_team_id"))
                away_team_id = int(row.get("away_team_id"))
            except (TypeError, ValueError):
                continue

            team_is_home = home_team_id == int(team_id)
            if not team_is_home and away_team_id != int(team_id):
                continue

            team_value = _parse_float(row.get("home_value" if team_is_home else "away_value"))
            opp_value = _parse_float(row.get("away_value" if team_is_home else "home_value"))
            if team_value is None or opp_value is None:
                continue

            match_row = matches_by_event.setdefault(
                event_id,
                {"statistics": {}, "opponentStatistics": {}},
            )
            match_row["statistics"][stat_key] = team_value
            match_row["opponentStatistics"][stat_key] = opp_value

    history = {"data": list(matches_by_event.values())}
    _TEAM_HISTORY_CACHE[cache_key] = history
    return history
