# Footbal-Stats

Python pipeline that compares football match stat averages from **Statshub** with market lines from **Crocobet**, writes results to Excel, and optionally sends Telegram alerts when differences are large.

## What this project does

1. Pulls upcoming matches from Statshub (configured leagues/time window).
2. Builds per-match stat averages from current-season team history in the same tournament.
3. Fetches Crocobet events and matches them to Statshub fixtures (fuzzy team-name matching).
4. Adds Crocobet lines + H2H ratio columns into the Excel summary.
5. Sends Telegram notifications for value candidates (threshold-based + hit-rate override).

## Requirements

- Python 3.10+
- Network access to:
  - `https://www.statshub.com`
  - `https://api.crocobet.com`
  - `https://api.telegram.org` (only for notifications)

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt`:
- `requests`
- `pandas`
- `openpyxl`
- `python-dotenv`

## Environment variables

Create `.env` in the project root:

```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
STAT_DIFF_THRESHOLD_PCT=20
```

Notes:
- `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are required by `notify_diffs.py`.
- `STAT_DIFF_THRESHOLD_PCT` is optional (defaults to `20.0` from `config.py`).

## Quick start (full pipeline)

Run everything in order:

```bash
python main.py
```

`main.py` runs:
1. `statshub.py`
2. `crocobet_events_fetch.py`
3. `crocobet_game_stats.py`
4. `notify_diffs.py`

## Run scripts manually

If you want to run step-by-step:

```bash
python statshub.py
python crocobet_events_fetch.py
python crocobet_game_stats.py
python notify_diffs.py
```

Script roles:
- `statshub.py`: builds `json/stats_averages.xlsx` with `full` and `summary` sheets.
- `crocobet_events_fetch.py`: fetches Crocobet events and creates matchup mapping JSON.
- `crocobet_game_stats.py`: writes `cb_*` and `h2h_*` columns into the Excel `summary`.
- `notify_diffs.py`: sends Telegram messages for selected opportunities.
- `matchup_report.py`: older standalone script for combined reporting/debug flow.

## Output files

Generated under `json/`:
- `stats_averages.xlsx`
- `crocobet_events_all.json`
- `crocobet_event_matches.json`
- `crocobet_event_unmatched.json`
- `crocobet_event_markets_debug.json` (only when missing markets are found)
- `team_aliases.json` (created automatically if missing)

`json/` is gitignored by default.

## Main configuration

Centralized in `config.py`:
- time window (`WINDOW_END_DAYS`, `WINDOW_END_HOUR`)
- target Statshub tournaments (`STATSHUB_TARGET_TOURNAMENTS`)
- Crocobet league IDs (`CROCOBET_LEAGUE_IDS`)
- tracked stats (`STAT_ORDER`, labels and mappings)
- notify thresholds (`DEFAULT_THRESHOLD_PCT`, hit override rate)
- H2H rules
- team name normalization dictionaries

Adjust these values in `config.py` to fit your workflow.

## Common issues

- `stats_averages.xlsx not found`:
  - Run `python statshub.py` first.
- `crocobet_event_matches.json not found`:
  - Run `python crocobet_events_fetch.py` first.
- `Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID`:
  - Add both variables to `.env`.
- No alerts sent:
  - Check `STAT_DIFF_THRESHOLD_PCT`, generated `cb_*` columns, and whether fixtures fall in the configured time window.
