from __future__ import annotations

import os
import re
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import requests
import config

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

EXCEL_PATH = config.EXCEL_PATH
SHEET_NAME = config.SHEET_SUMMARY
DEFAULT_THRESHOLD_PCT = config.DEFAULT_THRESHOLD_PCT
HIT_OVERRIDE_MIN = config.HIT_OVERRIDE_MIN
HIT_OVERRIDE_TOTAL = config.HIT_OVERRIDE_TOTAL
MATCHES_PATH = config.MATCHES_PATH
EVENT_API_TEMPLATE = config.CROCOBET_EVENT_URL_TEMPLATE
ODDS_DB_PATH = config.ODDS_DB_PATH
ODDS_DB_SHEET = config.ODDS_DB_SHEET
ODDS_TRANSLATIONS_PATH = config.ODDS_TRANSLATIONS_PATH

STAT_ORDER = config.STAT_ORDER
STAT_LABELS = config.STAT_LABELS

BASE_SITE = config.BASE_SITE
STATSHUB_KEY_MAP = config.STATSHUB_STAT_KEYS
CB_MARKET_PHRASES = {
    label: spec.get("phrases", []) for label, spec in config.CROCOBET_STAT_TARGETS.items()
}
CB_EXCLUDE_TERMS = config.CROCOBET_MARKET_EXCLUDE_TERMS
CB_EXCLUDE_PHRASES = {
    label: spec.get("exclude_phrases", [])
    for label, spec in config.CROCOBET_STAT_TARGETS.items()
    if spec.get("exclude_phrases")
}

TEAM_NOISE_TOKENS = {
    "fc", "afc", "cf", "sc", "ac", "as", "ud", "cd", "fk",
    "club", "de", "the", "and",
}


def format_number(value: float) -> str:
    num = float(value)
    text = f"{num:.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def diff_percent(sh_value: float, cb_value: float) -> float | None:
    if sh_value is None or cb_value is None:
        return None
    try:
        sh = float(sh_value)
        cb = float(cb_value)
    except (TypeError, ValueError):
        return None
    denom = (abs(sh) + abs(cb)) / 2
    if denom == 0:
        return None
    return abs(cb - sh) / denom * 100


def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    resp = requests.post(url, json=payload, timeout=config.HTTP_TIMEOUT_SECONDS)
    resp.raise_for_status()


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def normalize_team_for_fuzzy(name: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(name).lower())
    if not tokens:
        return ""
    filtered = [t for t in tokens if t not in TEAM_NOISE_TOKENS]
    if not filtered:
        filtered = tokens
    return "".join(filtered)


def normalize_matchup_for_fuzzy(matchup: str) -> str:
    parts = split_matchup(matchup)
    if not parts:
        return normalize(matchup)
    home, away = parts
    return f"{normalize_team_for_fuzzy(home)}-{normalize_team_for_fuzzy(away)}"


def parse_odds(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_json_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        payload = json.loads(raw)
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def normalize_stat_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def split_matchup(matchup: str) -> tuple[str, str] | None:
    if not isinstance(matchup, str) or " - " not in matchup:
        return None
    home, away = matchup.split(" - ", 1)
    home = home.strip()
    away = away.strip()
    if not home or not away:
        return None
    return home, away


def ensure_translation_file(path: Path, summary_df: pd.DataFrame, odds_df: pd.DataFrame) -> None:
    if path.exists():
        return

    en_teams: set[str] = set()
    if "matchup" in summary_df.columns:
        for matchup in summary_df["matchup"].dropna().astype(str).tolist():
            parts = split_matchup(matchup)
            if not parts:
                continue
            en_teams.add(parts[0])
            en_teams.add(parts[1])

    geo_teams: set[str] = set()
    if "event_name" in odds_df.columns:
        for matchup in odds_df["event_name"].dropna().astype(str).tolist():
            parts = split_matchup(matchup)
            if not parts:
                continue
            geo_teams.add(parts[0])
            geo_teams.add(parts[1])

    default_pos_map = {
        "corners": "corners",
        "cards": "cards",
        "shots_on_target": "",
        "total_shots": "",
        "fouls": "fouls",
        "goalkeeper_saves": "",
        "throw_ins": "throw_ins",
        "offsides": "",
    }

    payload = {
        "notes": "Fill Georgian->English team names. Leave unsupported position mappings empty.",
        "team_geo_to_en": {name: "" for name in sorted(geo_teams)},
        "team_en_to_geo": {name: "" for name in sorted(en_teams)},
        "positions_notify_to_odds": default_pos_map,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_translations(path: Path) -> dict:
    defaults = {
        "team_geo_to_en": {},
        "team_en_to_geo": {},
        "positions_notify_to_odds": {
            "corners": "corners",
            "cards": "cards",
            "shots_on_target": "",
            "total_shots": "",
            "fouls": "fouls",
            "goalkeeper_saves": "",
            "throw_ins": "throw_ins",
            "offsides": "",
        },
    }
    if not path.exists():
        return defaults
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults
    if not isinstance(data, dict):
        return defaults

    for key in ("team_geo_to_en", "team_en_to_geo", "positions_notify_to_odds"):
        val = data.get(key)
        if not isinstance(val, dict):
            continue
        defaults[key].update({str(k): str(v) for k, v in val.items()})
    return defaults


def build_provider_odds_index(odds_df: pd.DataFrame, translations: dict) -> dict[tuple[str, str, float], dict]:
    out: dict[tuple[str, str, float], dict] = {}
    geo_to_en = translations.get("team_geo_to_en", {}) or {}

    required = {
        "event_name",
        "stat_type",
        "line",
        "crocobet_over_odds",
        "crocobet_under_odds",
        "europebet_over_odds",
        "europebet_under_odds",
        "betlive_over_odds",
        "betlive_under_odds",
    }
    if not required.issubset(set(odds_df.columns)):
        return out

    for _, row in odds_df.iterrows():
        matchup_geo = row.get("event_name")
        parts = split_matchup(str(matchup_geo)) if pd.notna(matchup_geo) else None
        if not parts:
            continue
        home_en = str(geo_to_en.get(parts[0], "")).strip()
        away_en = str(geo_to_en.get(parts[1], "")).strip()
        if not home_en or not away_en:
            continue
        matchup_en = f"{home_en} - {away_en}"

        stat_type = normalize_stat_key(str(row.get("stat_type", "")))
        if not stat_type:
            continue
        line = parse_float(row.get("line"))
        if line is None:
            continue
        key = (normalize(matchup_en), stat_type, round(line, 2))
        out[key] = {
            "crocobet_over": parse_odds(row.get("crocobet_over_odds")),
            "crocobet_under": parse_odds(row.get("crocobet_under_odds")),
            "europebet_over": parse_odds(row.get("europebet_over_odds")),
            "europebet_under": parse_odds(row.get("europebet_under_odds")),
            "betlive_over": parse_odds(row.get("betlive_over_odds")),
            "betlive_under": parse_odds(row.get("betlive_under_odds")),
        }
    return out


def get_provider_odds_text(
    provider_index: dict[tuple[str, str, float], dict],
    translations: dict,
    matchup: str,
    label: str,
    cb_line: float,
    side: str,
    cb_fallback: float | None,
) -> str:
    pos_map = translations.get("positions_notify_to_odds", {}) or {}
    mapped_stat = normalize_stat_key(str(pos_map.get(label, label)))
    if not mapped_stat:
        cb_text = format_number(cb_fallback) if cb_fallback is not None else "n/a"
        return f"CB - {cb_text}, EU - n/a, BL - n/a"

    lookup_keys = [
        (normalize(matchup), mapped_stat, round(cb_line, 2)),
        (normalize(matchup), mapped_stat, round(cb_line, 1)),
    ]
    row = None
    for key in lookup_keys:
        row = provider_index.get(key)
        if row is not None:
            break

    if row is None:
        target_mk = normalize(matchup)
        target_fuzzy = normalize_matchup_for_fuzzy(matchup)
        best_score = 0.0
        best_row = None
        for (mk, stat_key, line_key), rec in provider_index.items():
            if stat_key != mapped_stat:
                continue
            if abs(float(line_key) - float(cb_line)) > config.PROVIDER_LINE_TOLERANCE:
                continue
            score_plain = SequenceMatcher(None, target_mk, mk).ratio()
            score_fuzzy = SequenceMatcher(
                None,
                target_fuzzy,
                normalize_matchup_for_fuzzy(mk),
            ).ratio()
            score = max(score_plain, score_fuzzy)
            if score > best_score:
                best_score = score
                best_row = rec
        if best_row is not None and best_score >= config.PROVIDER_MATCHUP_FUZZY_MIN:
            row = best_row

    if row is None:
        cb_text = format_number(cb_fallback) if cb_fallback is not None else "n/a"
        return f"CB - {cb_text}, EU - n/a, BL - n/a"

    cb_val = row.get(f"crocobet_{side}")
    eu_val = row.get(f"europebet_{side}")
    bl_val = row.get(f"betlive_{side}")
    if cb_val is None:
        cb_val = cb_fallback

    cb_text = format_number(cb_val) if cb_val is not None else "n/a"
    eu_text = format_number(eu_val) if eu_val is not None else "n/a"
    bl_text = format_number(bl_val) if bl_val is not None else "n/a"
    return f"CB - {cb_text}, EU - {eu_text}, BL - {bl_text}"


def build_event_id_map(records: list[dict]) -> dict[str, int]:
    out = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        matchup = rec.get("matchup")
        event_id = rec.get("event_id")
        if not isinstance(matchup, str) or event_id is None:
            continue
        try:
            out[normalize(matchup)] = int(event_id)
        except (TypeError, ValueError):
            continue
    return out


def fetch_crocobet_event(event_id: int) -> dict | None:
    url = EVENT_API_TEMPLATE.format(event_id=event_id)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Request-Language": "en",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=config.HTTP_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    data = resp.json()
    return data if isinstance(data, dict) else None


def list_event_games(event_data: dict | None) -> list[dict]:
    if not isinstance(event_data, dict):
        return []
    data = event_data.get("data")
    if not isinstance(data, dict):
        return []
    event_games = data.get("eventGames")
    return event_games if isinstance(event_games, list) else []


def is_valid_cb_market(game_name: str, label: str) -> bool:
    name = (game_name or "").lower()
    if any(t in name for t in CB_EXCLUDE_TERMS):
        return False
    phrases = CB_MARKET_PHRASES.get(label, [])
    if phrases and not any(p in name for p in phrases):
        return False
    for phrase in CB_EXCLUDE_PHRASES.get(label, []):
        if phrase in name:
            return False
    return True


def find_under_over(outcomes: list[dict]) -> tuple[dict | None, dict | None]:
    under = None
    over = None
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        name = str(outcome.get("outcomeName") or "").strip().lower()
        if name.startswith("under"):
            under = outcome
        elif name.startswith("over"):
            over = outcome
    return under, over


def get_cb_side_odd_for_line(
    event_data: dict | None,
    label: str,
    cb_line: float,
    side: str,
) -> float | None:
    best_odd = None
    best_delta = None
    for game in list_event_games(event_data):
        if not isinstance(game, dict):
            continue
        game_name = str(game.get("gameName") or "")
        if not is_valid_cb_market(game_name, label):
            continue
        argument = parse_float(game.get("argument"))
        if argument is None:
            continue
        under, over = find_under_over(game.get("outcomes", []))
        if not under or not over:
            continue
        odd = parse_odds((over if side == "over" else under).get("outcomeOdds"))
        if odd is None:
            continue
        delta = abs(argument - cb_line)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_odd = odd
    if best_delta is None:
        return None
    return best_odd


def fetch_games_today() -> list[dict]:
    today = datetime.now().date()
    start_dt = datetime.combine(today, time.min)
    end_dt = datetime.combine(
        today + timedelta(days=config.WINDOW_END_DAYS),
        time(config.WINDOW_END_HOUR, 0, 0),
    )
    params = {
        "startOfDay": int(start_dt.timestamp()),
        "endOfDay": int(end_dt.timestamp()),
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": BASE_SITE + "/",
    }
    url = f"{BASE_SITE}/api/event/by-date"
    resp = requests.get(
        url, params=params, headers=headers, timeout=config.HTTP_TIMEOUT_SECONDS
    )
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", []) if isinstance(payload, dict) else []
    return data if isinstance(data, list) else []


def build_matchup_context(games: list[dict]) -> dict[str, dict]:
    ctx = {}
    for game in games:
        try:
            home = game["homeTeam"]
            away = game["awayTeam"]
            tournaments = game["tournaments"]
        except (TypeError, KeyError):
            continue
        matchup = f"{home.get('name', '')} - {away.get('name', '')}".strip()
        key = normalize(matchup)
        if not key:
            continue
        ctx[key] = {
            "home_team_id": home.get("id"),
            "away_team_id": away.get("id"),
            "tournament_id": tournaments.get("uniqueTournamentId"),
            "home_name": home.get("name"),
            "away_name": away.get("name"),
        }
    return ctx


def fetch_team_history(team_id: int, tournament_id: int) -> dict | None:
    url = f"{BASE_SITE}/api/team/{team_id}/performance?"
    params = {
        "tournamentId": tournament_id,
        "limit": config.NOTIFY_HISTORY_LIMIT,
        "location": "all",
        "eventHalf": "ALL",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": BASE_SITE + "/",
    }
    try:
        resp = requests.get(
            url, params=params, headers=headers, timeout=config.HTTP_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
    except requests.RequestException:
        return None
    data = resp.json()
    return data if isinstance(data, dict) else None


def parse_float(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_bet_side(sh_value: float, cb_line: float) -> str | None:
    if sh_value > cb_line:
        return "over"
    if sh_value < cb_line:
        return "under"
    return None


def count_hits(history_data: dict | None, stat_key: str, side: str, line_value: float) -> tuple[int, int]:
    if not history_data:
        return 0, 0
    matches = history_data.get("data", [])
    if not isinstance(matches, list):
        return 0, 0

    hits = 0
    total = 0
    for match in matches:
        stats = match.get("statistics") if isinstance(match, dict) else None
        opp_stats = match.get("opponentStatistics") if isinstance(match, dict) else None
        if not isinstance(stats, dict) or not isinstance(opp_stats, dict):
            continue
        if stat_key not in stats or stat_key not in opp_stats:
            continue
        total_value = parse_float(stats.get(stat_key))
        opp_value = parse_float(opp_stats.get(stat_key))
        if total_value is None or opp_value is None:
            continue

        total += 1
        match_total = total_value + opp_value
        if side == "over" and match_total > line_value:
            hits += 1
        elif side == "under" and match_total < line_value:
            hits += 1
    return hits, total


def build_hit_rate_text(
    matchup_ctx: dict | None,
    label: str,
    sh_value: float,
    cb_line: float,
    history_cache: dict[tuple[int, int], dict | None],
) -> tuple[str, int, int] | None:
    stat_key = STATSHUB_KEY_MAP.get(label)
    if not matchup_ctx or not stat_key:
        return None

    home_team_id = matchup_ctx.get("home_team_id")
    away_team_id = matchup_ctx.get("away_team_id")
    tournament_id = matchup_ctx.get("tournament_id")
    if home_team_id is None or away_team_id is None or tournament_id is None:
        return None

    side = infer_bet_side(sh_value, cb_line)
    if side is None:
        return None

    home_cache_key = (int(home_team_id), int(tournament_id))
    away_cache_key = (int(away_team_id), int(tournament_id))
    if home_cache_key not in history_cache:
        history_cache[home_cache_key] = fetch_team_history(home_cache_key[0], home_cache_key[1])
    if away_cache_key not in history_cache:
        history_cache[away_cache_key] = fetch_team_history(away_cache_key[0], away_cache_key[1])

    home_hits, home_total = count_hits(history_cache.get(home_cache_key), stat_key, side, cb_line)
    away_hits, away_total = count_hits(history_cache.get(away_cache_key), stat_key, side, cb_line)
    total_hits = home_hits + away_hits
    total_games = home_total + away_total

    side_text = "Over" if side == "over" else "Under"
    return side_text, total_hits, total_games


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    if not EXCEL_PATH.exists():
        raise SystemExit("stats_averages.xlsx not found. Run statshub.py and crocobet_game_stats.py first.")

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables.")

    try:
        threshold = float(os.environ.get("STAT_DIFF_THRESHOLD_PCT", DEFAULT_THRESHOLD_PCT))
    except ValueError:
        threshold = DEFAULT_THRESHOLD_PCT

    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    if df.empty:
        return
    odds_df = pd.DataFrame()
    if ODDS_DB_PATH.exists():
        try:
            odds_df = pd.read_excel(ODDS_DB_PATH, sheet_name=ODDS_DB_SHEET)
        except Exception:
            odds_df = pd.DataFrame()

    ensure_translation_file(ODDS_TRANSLATIONS_PATH, df, odds_df)
    translations = load_translations(ODDS_TRANSLATIONS_PATH)
    provider_index = build_provider_odds_index(odds_df, translations)

    matchup_context = {}
    try:
        matchup_context = build_matchup_context(fetch_games_today())
    except requests.RequestException:
        matchup_context = {}
    history_cache: dict[tuple[int, int], dict | None] = {}
    event_id_map = build_event_id_map(parse_json_file(MATCHES_PATH))
    event_cache: dict[int, dict | None] = {}

    for _, row in df.iterrows():
        matchup = row.get("matchup")
        if not isinstance(matchup, str) or not matchup.strip():
            continue
        matchup_key = normalize(matchup)
        ctx = matchup_context.get(matchup_key)
        event_id = event_id_map.get(matchup_key)
        event_data = None
        if event_id is not None:
            if event_id not in event_cache:
                event_cache[event_id] = fetch_crocobet_event(event_id)
            event_data = event_cache.get(event_id)

        parts = []
        for label in STAT_ORDER:
            sh_col = f"average_{label}"
            cb_col = f"cb_{label}"
            h2h_col = f"h2h_{label}"
            if sh_col not in df.columns or cb_col not in df.columns:
                continue
            sh_val = row.get(sh_col)
            cb_val = row.get(cb_col)
            if pd.isna(sh_val) or pd.isna(cb_val):
                continue
            h2h_val = row.get(h2h_col) if h2h_col in df.columns else None
            h2h_text_from_sheet = None
            if h2h_val is not None and not pd.isna(h2h_val):
                h2h_text_from_sheet = str(h2h_val).strip() or None
            pct = diff_percent(sh_val, cb_val)
            stat_label = STAT_LABELS.get(label, label)
            if pct is not None and pct > 100:
                continue
            hit_data = build_hit_rate_text(
                matchup_ctx=ctx,
                label=label,
                sh_value=float(sh_val),
                cb_line=float(cb_val),
                history_cache=history_cache,
            )
            total_hits = 0
            total_games = 0
            if hit_data:
                _, total_hits, total_games = hit_data

            hit_override = total_games == HIT_OVERRIDE_TOTAL and total_hits >= HIT_OVERRIDE_MIN
            pct_ok = pct is not None and threshold <= pct <= 100
            if not pct_ok and not hit_override:
                continue

            side = "over" if float(sh_val) > float(cb_val) else "under"
            cb_odd = get_cb_side_odd_for_line(
                event_data=event_data,
                label=label,
                cb_line=float(cb_val),
                side=side,
            )
            providers_text = get_provider_odds_text(
                provider_index=provider_index,
                translations=translations,
                matchup=matchup,
                label=label,
                cb_line=float(cb_val),
                side=side,
                cb_fallback=cb_odd,
            )
            odd_text = format_number(cb_odd) if cb_odd is not None else "n/a"
            odds_part = f"{side} {format_number(cb_val)} {stat_label.lower()}"
            value_part = f"{pct:.0f}%" if pct is not None else "n/a"
            hit_part = "n/a"
            if total_games > 0:
                hit_part = f"{total_hits}/{total_games}"
            h2h_part = h2h_text_from_sheet or "n/a"
            part = f"{odds_part} | {value_part} | {hit_part} | H2H {h2h_part}\n{providers_text}"
            parts.append(part)

        if not parts:
            continue

        message = f"{matchup}\n" + "\n".join(parts)
        send_telegram_message(token, chat_id, message)


if __name__ == "__main__":
    main()
