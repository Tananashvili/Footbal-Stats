from pathlib import Path

# Shared paths
EXCEL_PATH = Path("stats_averages.xlsx")
SHEET_SUMMARY = "summary"
MATCHES_PATH = Path("json") / "crocobet_event_matches.json"
DEBUG_MARKETS_PATH = Path("json") / "crocobet_event_markets_debug.json"

# Shared API settings
BASE_SITE = "https://www.statshub.com"
HTTP_TIMEOUT_SECONDS = 30
CROCOBET_EVENTS_URL_TEMPLATE = "https://api.crocobet.com/rest/market/categories/multi/{league_id}/events"
CROCOBET_EVENT_URL_TEMPLATE = "https://api.crocobet.com/rest/market/events/{event_id}"

# Time window (shared by statshub.py, notify_diffs.py, crocobet_game_stats.py)
# start is always today 00:00; end is today + WINDOW_END_DAYS at WINDOW_END_HOUR:00
WINDOW_END_DAYS = 1
WINDOW_END_HOUR = 2

# Leagues/tournaments
STATSHUB_TARGET_TOURNAMENTS = [
    "Premier League",
    "Serie A",
    "Ligue 1",
    "LaLiga",
    "Bundesliga",
    "Eredivisie",
    "Liga Portugal Betclic",
    "Super Lig",
]

CROCOBET_LEAGUE_IDS = [
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

# Shared stat/market configuration
STAT_ORDER = [
    "corners",
    "cards",
    "shots_on_target",
    "total_shots",
    "fouls",
    "goalkeeper_saves",
    "throw_ins",
    "offsides",
]

STAT_LABELS = {
    "corners": "Corners",
    "cards": "Cards",
    "shots_on_target": "Shots on target",
    "total_shots": "Total shots",
    "fouls": "Fouls",
    "goalkeeper_saves": "Goalkeeper saves",
    "throw_ins": "Throw-ins",
    "offsides": "Offsides",
}

STATSHUB_STAT_KEYS = {
    "corners": "cornerKicks",
    "cards": "cards",
    "shots_on_target": "shotsOnGoal",
    "total_shots": "totalShotsOnGoal",
    "fouls": "fouls",
    "goalkeeper_saves": "goalkeeperSaves",
    "throw_ins": "throwIns",
    "offsides": "offsides",
}

CROCOBET_STAT_TARGETS = {
    "corners": {"phrases": ["corner"]},
    "cards": {"phrases": ["card"]},
    "shots_on_target": {"phrases": ["shots on target"]},
    "total_shots": {"phrases": ["shots"], "exclude_phrases": ["shots on target"]},
    "fouls": {"phrases": ["foul"]},
    "goalkeeper_saves": {"phrases": ["save"]},
    "throw_ins": {"phrases": ["throw-in", "throw in", "throwins"]},
    "offsides": {"phrases": ["offside"]},
}

CROCOBET_MARKET_EXCLUDE_TERMS = ["team", "half"]

# notify_diffs settings
DEFAULT_THRESHOLD_PCT = 20.0
HIT_OVERRIDE_MIN = 16
HIT_OVERRIDE_TOTAL = 20

# History windows
STATSHUB_HISTORY_LIMIT = 10
NOTIFY_HISTORY_LIMIT = 10
GAME_STATS_HISTORY_LIMIT = 200
H2H_MIN_YEAR = 2023
H2H_MAX_MATCHES = 5

# Team-name normalization for crocobet_events_fetch
TEAM_ABBREVIATIONS = {
    "utd": "united",
    "st": "saint",
    "ath": "athletic",
    "atl": "atletico",
    "dept": "deportivo",
    "dep": "deportivo",
    "int": "inter",
    "intl": "international",
    "inzer": "inter",
}

TEAM_STOPWORDS = {
    "fc",
    "sc",
    "ac",
    "as",
    "cf",
    "cd",
    "ud",
    "bk",
    "fk",
    "sk",
    "ssc",
    "sv",
    "tsg",
    "vf",
    "afc",
    "real",
    "stade",
    "olympique",
    "club",
    "de",
    "la",
    "el",
    "the",
    "calcio",
    "sporting",
    "united",
    "city",
    "town",
    "racing",
    "and",
}

TEAM_SUFFIXES = {
    "res",
    "reserve",
    "reserves",
    "ii",
    "iii",
    "b",
    "c",
    "u16",
    "u17",
    "u18",
    "u19",
    "u20",
    "u21",
    "u22",
    "u23",
    "u24",
    "u25",
    "u26",
    "youth",
    "academy",
    "junior",
    "women",
    "w",
    "ladies",
    "femenino",
    "femenina",
    "fem",
}
