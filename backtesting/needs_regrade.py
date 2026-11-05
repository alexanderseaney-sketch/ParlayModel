"""Print "yes" when the real-line track record is behind the schedule, "no" otherwise.

The refresh workflow used to regrade only on the run that started in the 12:00 UTC hour
on Tuesday. Scheduled GitHub runs drift by an hour or more (the 2026-11-03 run started at
13:47), so a whole week went ungraded. Now any run regrades once the newest week whose
games all have final scores isn't in line_history_graded.csv yet.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def latest_final_week(schedules: pd.DataFrame) -> tuple[int, int] | None:
    s = schedules[schedules["game_type"].fillna("REG") == "REG"] if "game_type" in schedules else schedules
    season = int(s["season"].max())
    s = s[s["season"] == season]
    done = s.groupby("week")["home_score"].apply(lambda x: x.notna().all())
    done = done[done]
    return (season, int(done.index.max())) if len(done) else None


def latest_graded_week(graded: pd.DataFrame) -> tuple[int, int] | None:
    if graded.empty:
        return None
    season = int(graded["season"].max())
    return season, int(graded.loc[graded["season"] == season, "week"].max())


def main() -> None:
    try:
        sched = pd.read_csv(os.path.join(ROOT, "data", "raw", "schedules.csv"), low_memory=False)
        final = latest_final_week(sched)
    except (OSError, KeyError, ValueError):
        final = None
    try:
        graded = pd.read_csv(os.path.join(ROOT, "backtesting", "line_history_graded.csv"),
                             usecols=["season", "week"])
        done = latest_graded_week(graded)
    except (OSError, ValueError):
        done = None
    behind = final is not None and (done is None or done < final)
    print(f"latest final week {final}, latest graded week {done}", file=sys.stderr)
    print("yes" if behind else "no")


if __name__ == "__main__":
    main()
