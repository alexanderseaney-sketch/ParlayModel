"""
Three fresh theories, deliberately chosen in a DIFFERENT category than almost
everything tested before now: opponent/context-side features have failed 6/6 times
across every prop (see README). These are about the player's own statistical shape
and real-world continuity instead, none of it tested anywhere in this project yet
(checked via grep before building, not assumed):

1. CONSISTENCY / VOLATILITY: every feature tried so far describes the LEVEL of a
   player's production (their average, their trend). None describe how PREDICTABLE
   that level actually is. A player who puts up 60-70-65-68 yards is a fundamentally
   different bet than one who puts up 20-140-15-95 with the same average -- the model
   currently can't see that difference at all. Tested as rolling std and rolling
   coefficient of variation (std/mean, more comparable across usage levels) of each
   prop's own primary stat.

2. QB CONTINUITY (receiving/receptions only): a real football mechanism never tested
   -- route-running chemistry and catch-point timing between a WR/TE and their QB
   takes real games to build. A pass-catcher facing a new starter (injury, benching,
   trade) is a different situation than one with their regular QB, and nothing in the
   current features distinguishes this. Built from weekly_stats directly: each team's
   starting QB per week (the QB with the most attempts that week), then whether this
   week's starter differs from last week's for that team.

3. EXPLICIT INTERACTION TERMS (passing only): the passing model uses LogisticRegression
   specifically because XGBoost lost on this smaller dataset (see README) -- but
   LogReg, unlike XGBoost's tree splits, can't learn feature interactions on its own.
   Testing a few football-motivated ones explicitly (team implied total x volume,
   aggressiveness x opponent defense) targets exactly the model type structurally
   least able to find this kind of signal by itself.

Same rigorous pooled 5-season CV as every other round in this project, current
production features + tuned hyperparameters as the base (so "beats base" means
"beats what's actually deployed").
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from current_predictions import PROP_CONFIGS, CONTEXT_MERGERS, _build_base_dataset

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]
XGB_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)

PRIMARY_STAT = {
    "receiving_yards": "receiving_yards",
    "rushing_yards": "rushing_yards",
    "passing_yards": "passing_yards",
    "receptions": "receptions",
}


def production_features(prop_name: str) -> list[str]:
    path = os.path.join(os.path.dirname(__file__), f"player_prop_{prop_name}_model.pkl")
    with open(path, "rb") as f:
        saved = pickle.load(f)
    return saved["features"]


def cv_pooled(df, features, model_fn, target_col="over_proxy_line"):
    all_probs, all_correct = [], []
    for holdout in HOLDOUT_SEASONS:
        train = df[df["season"] != holdout]
        test = df[df["season"] == holdout]
        X_train, y_train = train[features].fillna(0), train[target_col]
        X_test, y_test = test[features].fillna(0), test[target_col].values
        model = model_fn()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs)
        all_correct.extend(preds == y_test)
    return np.array(all_probs), np.array(all_correct)


def summarize(name, probs, correct):
    conf = np.abs(probs - 0.5) * 2
    mask = conf >= 0.4
    conf_acc = correct[mask].mean() if mask.sum() > 0 else float("nan")
    print(f"{name:<32} overall: {correct.mean()*100:>5.1f}%   @0.4 conf: {conf_acc*100:>5.1f}% ({mask.mean()*100:>4.1f}% of games)")


def xgb():
    return XGBClassifier(**XGB_PARAMS)


def logreg():
    return LogisticRegression(max_iter=3000)


def _build_full_context(df, config):
    for merge_name in config["context"]:
        df = CONTEXT_MERGERS[merge_name](df)
    return df


def _add_rolling_volatility(df, stat_col):
    """Rolling std and coefficient of variation of a player's own stat, same
    shift(1).expanding() no-leakage discipline as every other rolling feature."""
    df = df.sort_values(["player_id", "season", "week"]).copy()
    df[f"{stat_col}_rolling_std"] = (
        df.groupby(["player_id", "season"])[stat_col]
        .apply(lambda s: s.shift(1).expanding().std())
        .reset_index(level=[0, 1], drop=True)
    )
    level_col = f"{stat_col}_rolling"
    df[f"{stat_col}_cv"] = df[f"{stat_col}_rolling_std"] / df[level_col].replace(0, np.nan)
    return df


def _add_qb_continuity(df):
    """Each team's starting QB per week (most attempts that week), then whether this
    week's team starter differs from last week's."""
    weekly = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)
    qbs = weekly[weekly["position"] == "QB"].copy()
    starters = (
        qbs.sort_values("attempts", ascending=False)
        .drop_duplicates(subset=["recent_team", "season", "week"])
        [["recent_team", "season", "week", "player_id"]]
        .rename(columns={"player_id": "starting_qb_id"})
        .sort_values(["recent_team", "season", "week"])
    )
    starters["prev_starting_qb_id"] = starters.groupby(["recent_team", "season"])["starting_qb_id"].shift(1)
    starters["qb_changed"] = (
        starters["prev_starting_qb_id"].notna()
        & (starters["starting_qb_id"] != starters["prev_starting_qb_id"])
    ).astype(int)

    return df.merge(starters[["recent_team", "season", "week", "qb_changed"]],
                     on=["recent_team", "season", "week"], how="left")


def test_volatility():
    print("=" * 78)
    print("THEORY 1: player consistency/volatility (rolling std + CV of own stat)")
    print("=" * 78)
    for prop_type, config in PROP_CONFIGS.items():
        print(f"\n--- {prop_type} ---")
        df = _build_base_dataset(prop_type, min_week=4)
        df = df[df["position"].isin(config["positions"])].copy()
        df = df.dropna(subset=[config["proxy_col"]])
        df = _build_full_context(df, config)

        stat_col = PRIMARY_STAT[prop_type]
        df = _add_rolling_volatility(df, stat_col)

        base = production_features(prop_type)
        model_fn = logreg if prop_type == "passing_yards" else xgb

        base_probs, base_correct = cv_pooled(df, base, model_fn)
        summarize("base (current production)", base_probs, base_correct)

        for feat in [f"{stat_col}_rolling_std", f"{stat_col}_cv"]:
            probs, correct = cv_pooled(df, base + [feat], model_fn)
            summarize(f"+ {feat}", probs, correct)

        probs, correct = cv_pooled(df, base + [f"{stat_col}_rolling_std", f"{stat_col}_cv"], model_fn)
        summarize("+ both", probs, correct)


def test_qb_continuity():
    print("\n" + "=" * 78)
    print("THEORY 2: QB continuity (receiving/receptions only)")
    print("=" * 78)
    for prop_type in ["receiving_yards", "receptions"]:
        config = PROP_CONFIGS[prop_type]
        print(f"\n--- {prop_type} ---")
        df = _build_base_dataset(prop_type, min_week=4)
        df = df[df["position"].isin(config["positions"])].copy()
        df = df.dropna(subset=[config["proxy_col"]])
        df = _build_full_context(df, config)
        df = _add_qb_continuity(df)

        base = production_features(prop_type)
        base_probs, base_correct = cv_pooled(df, base, xgb)
        summarize("base (current production)", base_probs, base_correct)

        probs, correct = cv_pooled(df, base + ["qb_changed"], xgb)
        summarize("+ qb_changed", probs, correct)

        n_changed = df["qb_changed"].sum()
        print(f"  ({int(n_changed)} of {len(df)} rows had a QB change from the prior week)")


def test_passing_interactions():
    print("\n" + "=" * 78)
    print("THEORY 3: explicit interaction terms (passing only, LogReg-specific)")
    print("=" * 78)
    config = PROP_CONFIGS["passing_yards"]
    df = _build_base_dataset("passing_yards", min_week=4)
    df = df[df["position"].isin(config["positions"])].copy()
    df = df.dropna(subset=[config["proxy_col"]])
    df = _build_full_context(df, config)

    df["implied_total_x_attempts"] = df["team_implied_total"] * df["attempts_rolling"]
    df["aggressiveness_x_def_epa"] = df["aggressiveness_rolling"] * df["def_epa_allowed_rolling"]
    df["wind_x_intended_air_yards"] = df["wind"] * df["avg_intended_air_yards_rolling"]

    base = production_features("passing_yards")
    base_probs, base_correct = cv_pooled(df, base, logreg)
    summarize("base (current production)", base_probs, base_correct)

    interactions = ["implied_total_x_attempts", "aggressiveness_x_def_epa", "wind_x_intended_air_yards"]
    winners = []
    for feat in interactions:
        probs, correct = cv_pooled(df, base + [feat], logreg)
        summarize(f"+ {feat}", probs, correct)
        if correct.mean() - base_correct.mean() > 0.003:
            winners.append(feat)

    print(f"\nBeat base by >0.3pt: {winners if winners else 'none'}")
    if winners:
        probs, correct = cv_pooled(df, base + winners, logreg)
        summarize("+ combined winners", probs, correct)


if __name__ == "__main__":
    test_volatility()
    test_qb_continuity()
    test_passing_interactions()
