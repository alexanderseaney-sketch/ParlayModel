"""
Per-team scheme + personnel identity for the Depth Charts page. One row per team
-> data/raw/team_scheme_tendencies.csv.

Two halves, both kept current automatically:
  * WHO runs it -- head coach / offensive coordinator / defensive coordinator,
    lifted from data/raw/coaching_staff.csv (the Wikipedia pull).
  * WHAT they run -- tendencies derived from play-by-play (personnel mix,
    formation, pass rate / PROE, blitz, box, man vs zone), plus a short
    plain-English descriptor built from those same numbers so nothing here is a
    hand-maintained guess that goes stale on a coordinator change.

Uses the two most recent COMPLETE regular seasons in pbp.csv (>=16 weeks -- the
2024 file is only 15 weeks and was never backfilled, so a blind last-two would
weight a partial season). nflverse participation fields (personnel, box,
pass-rushers) are ~90% charted 2023+, coverage type ~45%.

Usage:
    python models/build_team_scheme_tendencies.py   (needs a fresh pbp.csv + coaching_staff.csv)
"""
import os
import re

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PBP_PATH = os.path.join(RAW_DIR, "pbp.csv")
STAFF_PATH = os.path.join(RAW_DIR, "coaching_staff.csv")
OUT_PATH = os.path.join(RAW_DIR, "team_scheme_tendencies.csv")

MAN_COVERAGES = {"COVER_0", "COVER_1", "2_MAN"}
ZONE_COVERAGES = {"COVER_2", "COVER_3", "COVER_4", "COVER_6", "PREVENT"}
_DB_TOKENS = ("CB", "FS", "SS", "S", "DB", "NB")


def _count(token, s):
    m = re.search(rf"(\d+)\s*{token}\b", s)
    return int(m.group(1)) if m else 0


def _personnel_key(s):
    """(RB+FB) count then TE count -- '1 RB, 1 TE, 3 WR' -> '11'; SF's
    '1 C, 1 FB, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 2 WR' -> '21' (the FB is a back and
    nflverse charts it separately -- miss it and every 21-personnel team reads 11).
    Needs a TE and a WR so defensive-personnel junk strings drop out."""
    if not isinstance(s, str) or "TE" not in s or "WR" not in s:
        return None
    backs = _count("RB", s) + _count("FB", s)
    if backs == 0:
        return None
    return f"{backs}{_count('TE', s)}"


def _db_count(s):
    """nflverse spells out the secondary ('3 CB, 1 FS, 1 SS') instead of a 'DB'
    total; sum the DB-ish tokens. 4 -> base, 5 -> nickel, 6+ -> dime."""
    if not isinstance(s, str):
        return np.nan
    total = sum(_count(tok, s) for tok in _DB_TOKENS)
    return total or np.nan


def _pct(mask, denom):
    return round(100 * mask.sum() / denom, 1) if denom else np.nan


def _load_pbp():
    use = ["season", "season_type", "week", "posteam", "defteam", "play_type", "down", "wp",
           "shotgun", "no_huddle", "qb_dropback", "pass_oe", "offense_formation",
           "offense_personnel", "defense_personnel", "defenders_in_box",
           "number_of_pass_rushers", "defense_coverage_type"]
    df = pd.concat(pd.read_csv(PBP_PATH, usecols=lambda c: c in use, chunksize=100_000,
                               low_memory=False), ignore_index=True)
    reg = df[df.season_type == "REG"]
    weeks = reg.groupby("season").week.nunique()
    complete = sorted(weeks[weeks >= 16].index)[-2:]
    return reg[reg.season.isin(complete)].copy(), complete


def _coordinators():
    if not os.path.exists(STAFF_PATH):
        return pd.DataFrame(columns=["team_abbr", "hc_name", "oc_name", "dc_name"])
    cs = pd.read_csv(STAFF_PATH)

    def pick(g, grp, *pats):
        for pat in pats:
            m = g[(g.group == grp) & g.title.str.contains(pat, case=False, regex=True, na=False)]
            if len(m):
                return m.iloc[0]["name"]
        return ""

    rows = []
    for team, g in cs.groupby("team_abbr"):
        rows.append({
            "team_abbr": team,
            "hc_name": pick(g, "head_coach", r"^head coach$", r"head coach"),
            "oc_name": pick(g, "offense", r"offensive coordinator", r"coordinator"),
            "dc_name": pick(g, "defense", r"defensive coordinator", r"coordinator"),
        })
    return pd.DataFrame(rows)


def _offense_row(g):
    plays = g[g.play_type.isin(["pass", "run"])]
    n = len(plays)
    neutral = plays[(plays.wp.between(0.2, 0.8)) & (plays.down.isin([1, 2, 3]))]
    pers = plays.offense_personnel.map(_personnel_key)
    pn = pers.notna().sum()
    fm = plays.offense_formation.fillna("")
    fn = (fm != "").sum()
    return pd.Series({
        "off_plays": n,
        "off_pass_rate_neutral": _pct(neutral.play_type == "pass", len(neutral)),
        "off_proe": round(plays.pass_oe.mean(), 1) if plays.pass_oe.notna().any() else np.nan,
        "off_shotgun_pct": _pct(plays.shotgun == 1, n),
        "off_under_center_pct": _pct(fm.isin(["UNDER CENTER", "SINGLEBACK", "I_FORM"]), fn),
        "off_pistol_pct": _pct(fm == "PISTOL", fn),
        "off_no_huddle_pct": _pct(plays.no_huddle == 1, n),
        "off_11_pct": _pct(pers == "11", pn),
        "off_12_pct": _pct(pers == "12", pn),
        "off_21_pct": _pct(pers == "21", pn),
        "off_heavy_pct": _pct(pers.isin(["13", "22", "23"]), pn),
    })


def _defense_row(g):
    db = g[g.play_type.isin(["pass", "run"])]
    dbox = db.defenders_in_box
    db_ct = db.defense_personnel.map(_db_count)
    dn = db_ct.notna().sum()
    rushers = db.loc[db.qb_dropback == 1, "number_of_pass_rushers"]
    cov = db.defense_coverage_type.dropna()
    cov = cov[cov.isin(MAN_COVERAGES | ZONE_COVERAGES)]
    top = cov.value_counts().idxmax().replace("_", " ").title() if len(cov) else ""
    return pd.Series({
        "def_box_avg": round(dbox.mean(), 2) if dbox.notna().any() else np.nan,
        "def_base_pct": _pct(db_ct <= 4, dn),
        "def_nickel_pct": _pct(db_ct == 5, dn),
        "def_dime_pct": _pct(db_ct >= 6, dn),
        "def_blitz_pct": _pct(rushers >= 5, rushers.notna().sum()),
        "def_man_pct": _pct(cov.isin(MAN_COVERAGES), len(cov)),
        "def_top_coverage": top,
    })


def _off_identity(r):
    bits = []
    pr, proe = r.off_pass_rate_neutral, r.off_proe
    if pd.notna(pr):
        if pr >= 60 or (pd.notna(proe) and proe >= 2):
            bits.append("pass-first")
        elif pr <= 52 or (pd.notna(proe) and proe <= -3):
            bits.append("run-leaning")
        else:
            bits.append("balanced")
    if pd.notna(r.off_21_pct) and r.off_21_pct >= 18:
        bits.append("heavy two-back / FB usage")
    elif pd.notna(r.off_12_pct) and r.off_12_pct >= 28:
        bits.append("12-personnel lean")
    elif pd.notna(r.off_11_pct) and r.off_11_pct >= 68:
        bits.append("spread, 11-personnel base")
    if pd.notna(r.off_shotgun_pct) and r.off_shotgun_pct >= 78:
        bits.append("shotgun-heavy")
    elif pd.notna(r.off_under_center_pct) and r.off_under_center_pct >= 35:
        bits.append("real under-center rate")
    return ", ".join(bits)


def _def_identity(r):
    bits = []
    if pd.notna(r.def_dime_pct) and r.def_dime_pct >= 15:
        bits.append("dime-heavy")
    elif pd.notna(r.def_base_pct) and r.def_base_pct >= 35:
        bits.append("base-personnel lean")
    elif pd.notna(r.def_nickel_pct) and r.def_nickel_pct >= 60:
        bits.append("nickel base")
    if pd.notna(r.def_blitz_pct):
        bits.append("blitz-heavy" if r.def_blitz_pct >= 30
                    else "4-man rush" if r.def_blitz_pct <= 20 else "average pressure")
    if pd.notna(r.def_man_pct):
        bits.append("man-leaning" if r.def_man_pct >= 42
                    else "zone-heavy" if r.def_man_pct <= 33 else "man/zone mix")
    if r.def_top_coverage:
        bits.append(f"most-used {r.def_top_coverage}")
    return ", ".join(bits)


def main():
    df, seasons = _load_pbp()
    print(f"pbp: {len(df):,} REG plays, seasons {seasons}")

    off = df.groupby("posteam").apply(_offense_row)
    dfn = df.groupby("defteam").apply(_defense_row)
    out = off.join(dfn, how="outer")
    out.index.name = "team_abbr"
    out = out.reset_index()
    out = out[out.team_abbr.str.len() <= 3]  # drop stray non-team codes

    out["off_identity"] = out.apply(_off_identity, axis=1)
    out["def_identity"] = out.apply(_def_identity, axis=1)
    out = out.merge(_coordinators(), on="team_abbr", how="left")
    out["derived_from_seasons"] = ", ".join(str(int(s)) for s in seasons)

    out = out.sort_values("team_abbr")
    out.to_csv(OUT_PATH, index=False)
    print(f"{len(out)} teams -> {OUT_PATH}")
    print(out[["team_abbr", "hc_name", "oc_name", "off_identity"]].to_string(index=False))


if __name__ == "__main__":
    main()
