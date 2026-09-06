"""
Cross-platform player id bridge -> data/raw/player_ids.csv.

Just the columns needed to line a Yahoo (or other platform) roster up with our
own data: yahoo_id, sleeper_id, espn_id <-> gsis_id (nflverse's key), plus a
normalized name / position / team for a fallback match when an id is missing.

Source: nfl_data_py.import_ids() (the dynastyprocess db_playerids table). Small
and slow-changing -- fine in the 6h refresh, and a manual run any time:

    python data/pull_player_ids.py
"""
import os
from datetime import datetime, timezone

import pandas as pd
import nfl_data_py as nfl

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUT_PATH = os.path.join(RAW_DIR, "player_ids.csv")

_KEEP = ["gsis_id", "yahoo_id", "sleeper_id", "espn_id", "name", "merge_name",
         "position", "team"]


def main() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    ids = nfl.import_ids()
    cols = [c for c in _KEEP if c in ids.columns]
    out = ids[cols].copy()

    # ids come through as floats ("34489.0"); make them clean strings.
    for c in ("yahoo_id", "sleeper_id", "espn_id"):
        if c in out.columns:
            out[c] = (out[c].astype("string")
                      .str.replace(r"\.0$", "", regex=True)
                      .replace({"<NA>": pd.NA, "nan": pd.NA, "": pd.NA}))

    out = out.dropna(subset=["gsis_id"]).drop_duplicates(subset=["gsis_id"])
    out["pulled_at"] = datetime.now(timezone.utc).isoformat()
    out.to_csv(OUT_PATH, index=False)

    have_yahoo = out["yahoo_id"].notna().sum() if "yahoo_id" in out.columns else 0
    print(f"{len(out)} players -> {OUT_PATH}  ({have_yahoo} with a yahoo_id)")


if __name__ == "__main__":
    main()
