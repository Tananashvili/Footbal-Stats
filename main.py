import subprocess
import sys


STEPS = [
    ("statshub.py", "Generate stats_averages.xlsx"),
    ("crocobet_events_fetch.py", "Fetch Crocobet fixtures + matches"),
    ("crocobet_game_stats.py", "Fetch Crocobet event stats and update Excel"),
    ("notify_diffs.py", "Notify Telegram when Statshub vs Crocobet differs"),
]


def run_step(script, label):
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {script} (exit {result.returncode})")


def main():
    for script, label in STEPS:
        run_step(script, label)


if __name__ == "__main__":
    main()
