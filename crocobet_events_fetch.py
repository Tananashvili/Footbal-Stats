import json
from pathlib import Path
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd
import requests
import config


LEAGUE_IDS = config.CROCOBET_LEAGUE_IDS

URL_TEMPLATE = config.CROCOBET_EVENTS_URL_TEMPLATE

ABBREVIATIONS = config.TEAM_ABBREVIATIONS

STOPWORDS = config.TEAM_STOPWORDS

TEAM_SUFFIXES = config.TEAM_SUFFIXES


def main():
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Request-Language": "en"
    }
    json_dir = Path("json")
    json_dir.mkdir(exist_ok=True)

    alias_path = json_dir / "team_aliases.json"
    if not alias_path.exists():
        with open(alias_path, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)

    aliases = load_aliases(alias_path)

    all_events = []
    for league_id in LEAGUE_IDS:
        url = URL_TEMPLATE.format(league_id=league_id)
        resp = requests.get(url, headers=headers, timeout=config.HTTP_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        all_events.append(
            {
                "league_id": league_id,
                "url": url,
                "data": data,
            }
        )

    with open(json_dir / "crocobet_events_all.json", "w", encoding="utf-8") as f:
        json.dump(all_events, f, ensure_ascii=False, indent=2)

    # Compare to Excel matchups (fuzzy)
    summary_df = pd.read_excel("stats_averages.xlsx", sheet_name="summary")
    matchups = [m.strip() for m in summary_df.get("matchup", []).dropna().tolist()]
    tournaments = [
        str(t).strip()
        for t in summary_df.get("tournament", []).dropna().tolist()
    ]
    home_shorts = [
        str(s).strip()
        for s in summary_df.get("home_team_shortname", []).dropna().tolist()
    ]
    away_shorts = [
        str(s).strip()
        for s in summary_df.get("away_team_shortname", []).dropna().tolist()
    ]

    matchup_records = []
    for idx, matchup in enumerate(matchups):
        if " - " not in matchup:
            continue
        m_home, m_away = matchup.split(" - ", 1)
        m_home = apply_alias(m_home, aliases)
        m_away = apply_alias(m_away, aliases)
        record = {
            "matchup": matchup,
            "home": m_home,
            "away": m_away,
            "home_n": normalize_team(m_home),
            "away_n": normalize_team(m_away),
            "tournament": tournaments[idx] if idx < len(tournaments) else "",
        }
        record["tournament_n"] = normalize_tournament(record["tournament"])
        matchup_records.append(record)

    shortname_records = []
    for hs, as_ in zip(home_shorts, away_shorts):
        if not hs or not as_:
            continue
        hs = apply_alias(hs, aliases)
        as_ = apply_alias(as_, aliases)
        shortname_records.append(
            {
                "matchup": f"{hs} - {as_}",
                "home": hs,
                "away": as_,
                "home_n": normalize_team(hs),
                "away_n": normalize_team(as_),
            }
        )

    matched = []
    unmatched = []
    for block in all_events:
        events = extract_events(block["data"])
        for ev in events:
            name = ev.get("eventName") or ev.get("name") or ev.get("matchName") or ""
            if not name:
                continue
            if " - " not in name:
                continue

            ev_home, ev_away = name.split(" - ", 1)
            ev_home = apply_alias(ev_home, aliases)
            ev_away = apply_alias(ev_away, aliases)
            ev_home_n = normalize_team(ev_home)
            ev_away_n = normalize_team(ev_away)

            ev_league = normalize_tournament(
                ev.get("categoryName")
                or ev.get("leagueName")
                or ev.get("tournamentName")
                or ev.get("competitionName")
                or ""
            )

            candidates = matchup_records
            if ev_league:
                league_filtered = [
                    r
                    for r in matchup_records
                    if r["tournament_n"]
                    and similarity(ev_league, r["tournament_n"]) >= 0.6
                ]
                if league_filtered:
                    candidates = league_filtered

            best = None
            best_score = 0.0
            best_swapped = False
            top_candidates = []

            for record in candidates:
                score = score_match(ev_home_n, ev_away_n, record["home_n"], record["away_n"])
                if score > best_score:
                    best_score = score
                    best = record["matchup"]
                    best_swapped = False
                top_candidates.append((record["matchup"], score))

            # Shortname boost (if provided)
            if best_score < 0.75 and shortname_records:
                for record in shortname_records:
                    score = score_match(ev_home_n, ev_away_n, record["home_n"], record["away_n"])
                    if score > best_score:
                        best_score = score
                        best = record["matchup"]
                        best_swapped = False
                    top_candidates.append((record["matchup"], score))

            top_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [
                {"matchup": m, "score": round(s, 3)} for m, s in top_candidates[:3]
            ]

            threshold = dynamic_threshold(ev_home_n, ev_away_n)
            event_id = ev.get("eventId") or ev.get("id") or ev.get("gameId")

            if best and best_score >= threshold:
                matched.append(
                    {
                        "matchup": best,
                        "matched_name": name,
                        "score": round(best_score, 3),
                        "league_id": block["league_id"],
                        "event_id": event_id,
                        "swapped": best_swapped,
                    }
                )
            elif best_score >= 0.6:
                unmatched.append(
                    {
                        "event_name": name,
                        "event_home": ev_home,
                        "event_away": ev_away,
                        "event_home_norm": ev_home_n,
                        "event_away_norm": ev_away_n,
                        "best_matchup": best,
                        "best_score": round(best_score, 3),
                        "league_id": block["league_id"],
                        "event_id": event_id,
                        "candidates": top_candidates,
                    }
                )

    with open(json_dir / "crocobet_event_matches.json", "w", encoding="utf-8") as f:
        def best_record(a, b):
            score_a = float(a.get("score") or 0)
            score_b = float(b.get("score") or 0)
            exact_a = 1 if a.get("matched_name") == a.get("matchup") else 0
            exact_b = 1 if b.get("matched_name") == b.get("matchup") else 0
            key_a = (exact_a, score_a)
            key_b = (exact_b, score_b)
            return a if key_a >= key_b else b

        best_by_matchup = {}
        for rec in matched:
            key = rec.get("matchup")
            if not key:
                continue
            if key in best_by_matchup:
                best_by_matchup[key] = best_record(best_by_matchup[key], rec)
            else:
                best_by_matchup[key] = rec

        deduped = list(best_by_matchup.values())
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    with open(json_dir / "crocobet_event_unmatched.json", "w", encoding="utf-8") as f:
        json.dump(unmatched, f, ensure_ascii=False, indent=2)


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def canonical_tokens(name: str) -> list:
    name = strip_accents(name.lower())
    name = name.replace("&", " and ")
    tokens = re.findall(r"[a-z0-9]+", name)
    cleaned = []
    for t in tokens:
        t = ABBREVIATIONS.get(t, t)
        if re.fullmatch(r"u\d{2}", t):
            continue
        if t in TEAM_SUFFIXES:
            continue
        cleaned.append(t)
    return cleaned


def normalize_team(name: str) -> str:
    tokens = canonical_tokens(name)
    filtered = [t for t in tokens if t not in STOPWORDS]
    if not filtered:
        filtered = tokens
    return " ".join(filtered)


def normalize_tournament(name: str) -> str:
    tokens = canonical_tokens(name)
    tokens = [t for t in tokens if t not in {"league", "cup", "division", "group"}]
    return " ".join(tokens)


def canonical_key(name: str) -> str:
    return " ".join(canonical_tokens(name)).strip()


def load_aliases(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for k, v in data.items():
        if not k or not v:
            continue
        out[canonical_key(k)] = str(v).strip()
    return out


def apply_alias(name: str, aliases: dict) -> str:
    if not name:
        return name
    key = canonical_key(name)
    return aliases.get(key, name)


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def score_match(ev_home_n: str, ev_away_n: str, m_home_n: str, m_away_n: str) -> float:
    score_home = (similarity(ev_home_n, m_home_n) + similarity(ev_away_n, m_away_n)) / 2
    score_swap = (similarity(ev_home_n, m_away_n) + similarity(ev_away_n, m_home_n)) / 2
    score_swap = max(0.0, score_swap - 0.08)
    return max(score_home, score_swap)


def dynamic_threshold(ev_home_n: str, ev_away_n: str) -> float:
    base = 0.75
    min_len = min(len(ev_home_n.replace(" ", "")), len(ev_away_n.replace(" ", "")))
    if min_len < 3:
        base = 0.86
    elif min_len < 4:
        base = 0.82
    return base


def extract_events(payload):
    for key in ("data", "events", "list", "games"):
        if isinstance(payload, dict) and key in payload and isinstance(
            payload[key], list
        ):
            return payload[key]
    if isinstance(payload, list):
        return payload
    return []


if __name__ == "__main__":
    main()
