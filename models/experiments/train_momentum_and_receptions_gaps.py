"""
Two genuinely new theories, chosen by looking at what's actually been tried so far
(see README) rather than guessing:

1. USAGE TREND (role momentum): `individual_context_features.add_usage_trend` was
   built during the individual-context round but never actually included in that
   round's tested feature groups (checked -- it's imported and defined, never used).
   Real gap. Matches the established pattern that's actually worked across this
   project: player-side opportunity signals (target share, efficiency) help; the
   *level* of usage is already in every model, but the *trend* (is a player's role
   growing or shrinking right now) isn't, anywhere.

2. GAME-CONTEXT + MATCHUP FEATURES ON RECEPTIONS: the receptions model didn't exist
   yet when the game-context mix-and-match round and the matchup-defense-splits round
   both ran (confirmed via git log), so it never got either treatment despite being
   the strongest of the four models. Tempered expectation on the matchup-features half
   specifically: defense-side granularity has now failed to help in FIVE separate
   rounds (QB stats, PBP efficiency, scheme, game-context, matchup splits) across the
   other props, including receiving -- receptions' closest cousin (same population,
   same opponent mechanic). Testing anyway for completeness on the flagship model, not
   because there's a strong prior it'll help.

Base features for every test = each prop's CURRENT production feature list (read
directly from its .pkl) with CURRENT production hyperparameters, not the older
narrower feature sets used in earlier exploratory rounds -- so a "win" here means
"beats what's actually deployed today," not an outdated baseline.
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from individual_context_features import build_player_injury_status, build_game_flags, add_usage_trend
from game_context_features import build_game_context, add_snap_share
from current_predictions import PROP_CONFIGS

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]

XGB_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)


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


# ============================================================ 1. USAGE TREND

def test_usage_trend():
    print("=" * 78)
    print("THEORY 1: usage trend (role momentum) -- built, never tested, anywhere")
    print("=" * 78)

    injuries = pd.read_csv(os.path.join(RAW_DIR, "injuries.csv"), low_memory=False)
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    inj_status = build_player_injury_status(injuries)
    flags = build_game_flags(schedules).rename(columns={"team": "recent_team"})
    context = build_game_context(schedules).rename(columns={"team": "recent_team"})

    specs = [
        ("receiving_yards", "player_prop_features", "build_receiving_yards_dataset",
         ["WR", "TE"], xgb, ["injury", "game_flags"], "target_share_rolling", "target_share_last3"),
        ("rushing_yards", "player_prop_rushing_features", "build_rushing_yards_dataset",
         ["RB"], xgb, ["game_flags", "game_context", "snap_share"], "carries_rolling", "carries_last3"),
        ("passing_yards", "player_prop_passing_features", "build_passing_yards_dataset",
         ["QB"], logreg, ["game_context"], "attempts_rolling", "attempts_last3"),
        ("receptions", "player_prop_receptions_features", "build_receptions_dataset",
         ["WR", "TE"], xgb, ["game_flags"], "targets_rolling", "targets_last3"),
    ]

    for prop_name, module_name, fn_name, positions, model_fn, context_needed, level_col, last3_col in specs:
        print(f"\n--- {prop_name} ---")
        mod = __import__(module_name)
        df = getattr(mod, fn_name)(min_week=4)
        df = df[df["position"].isin(positions)].copy()
        base = production_features(prop_name)
        # Match production's own dropna column exactly, not the trend's level_col --
        # keeps the "base" number a true apples-to-apples match with what's deployed.
        df = df.dropna(subset=[PROP_CONFIGS[prop_name]["proxy_col"]])

        if "injury" in context_needed:
            df = df.merge(inj_status, on=["player_id", "season", "week"], how="left")
        if "game_flags" in context_needed:
            df = df.merge(flags, on=["recent_team", "season", "week"], how="left")
        if "game_context" in context_needed:
            df = df.merge(context, on=["recent_team", "season", "week"], how="left")
        if "snap_share" in context_needed:
            df = add_snap_share(df)

        df = add_usage_trend(df, level_col, last3_col)

        base_probs, base_correct = cv_pooled(df, base, model_fn)
        summarize("base (current production)", base_probs, base_correct)

        trend_probs, trend_correct = cv_pooled(df, base + ["usage_trend"], model_fn)
        summarize("+ usage_trend", trend_probs, trend_correct)

        # Also try it on the volume last3 column already in every model, since
        # usage_trend is literally last3 - rolling -- some model types can find the
        # same signal via the raw last3 column already present. This checks whether
        # the explicit trend feature adds anything the model can't already infer.
        delta = trend_correct.mean() - base_correct.mean()
        verdict = "WIN" if delta > 0.003 else ("NULL" if abs(delta) <= 0.003 else "WORSE")
        print(f"  -> {verdict} ({delta*100:+.2f}pt)")


# ============================================================ 2. RECEPTIONS GAPS

def test_receptions_game_context():
    print("\n" + "=" * 78)
    print("THEORY 2a: game-context features on receptions (never tested -- model")
    print("didn't exist yet during the game-context round)")
    print("=" * 78)

    from player_prop_receptions_features import build_receptions_dataset

    df = build_receptions_dataset(min_week=4)
    df = df[df["position"].isin(["WR", "TE"])].copy()
    df = df.dropna(subset=["receptions_rolling"])

    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    flags = build_game_flags(schedules).rename(columns={"team": "recent_team"})
    context = build_game_context(schedules).rename(columns={"team": "recent_team"})
    df = df.merge(flags, on=["recent_team", "season", "week"], how="left")
    df = df.merge(context, on=["recent_team", "season", "week"], how="left")
    df = add_snap_share(df)

    base = production_features("receptions")
    base_probs, base_correct = cv_pooled(df, base, xgb)
    summarize("base (current production)", base_probs, base_correct)

    new_groups = {
        "implied_total": ["team_implied_total"],
        "home_away": ["is_home"],
        "weather": ["temp", "wind", "is_dome"],
        "rest": ["rest_days"],
        "snap_share": ["offense_pct_rolling"],
    }
    winners = []
    for name, feats in new_groups.items():
        probs, correct = cv_pooled(df, base + feats, xgb)
        summarize(f"+ {name}", probs, correct)
        if correct.mean() - base_correct.mean() > 0.003:
            winners.append(name)

    print(f"\nBeat base by >0.3pt: {winners if winners else 'none'}")
    if winners:
        combo = base + [f for n in winners for f in new_groups[n]]
        probs, correct = cv_pooled(df, combo, xgb)
        summarize("+ combined winners", probs, correct)


def test_receptions_matchup():
    print("\n" + "=" * 78)
    print("THEORY 2b: matchup-specific defense splits on receptions (confirmatory --")
    print("defense-side granularity has failed 5/5 times on the other props so far)")
    print("=" * 78)

    from player_prop_receptions_features import build_receptions_dataset
    from matchup_features import build_all_matchup_features
    from pbp_features import load_pbp

    df = build_receptions_dataset(min_week=4)
    df = df[df["position"].isin(["WR", "TE"])].copy()
    df = df.dropna(subset=["receptions_rolling"])

    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    flags = build_game_flags(schedules).rename(columns={"team": "recent_team"})
    df = df.merge(flags, on=["recent_team", "season", "week"], how="left")

    pbp = load_pbp()
    weekly_stats = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)
    matchup = build_all_matchup_features(pbp, weekly_stats)
    rolling_cols = [c for c in matchup.columns if c.endswith("_rolling")]
    df = df.merge(matchup[["team", "season", "week"] + rolling_cols].rename(columns={"team": "opponent"}),
                   on=["opponent", "season", "week"], how="left")

    base = production_features("receptions")
    base_probs, base_correct = cv_pooled(df, base, xgb)
    summarize("base (current production)", base_probs, base_correct)

    relevant = ["def_pass_epa_allowed_rolling", "def_pass_success_rate_allowed_rolling",
                "def_epa_allowed_vs_WR_rolling", "def_epa_allowed_vs_TE_rolling"]
    for feat in relevant:
        probs, correct = cv_pooled(df, base + [feat], xgb)
        summarize(f"+ {feat}", probs, correct)

    probs, correct = cv_pooled(df, base + relevant, xgb)
    summarize("+ ALL matchup features", probs, correct)


# ============================================================ 3. SCHEME ON UNTESTED PROPS

def test_scheme_on_untested_props():
    print("\n" + "=" * 78)
    print("THEORY 3: team scheme features (pass_oe, tempo, box counts) on rushing/")
    print("passing/receptions -- only ever tested on receiving (rejected there).")
    print("Rushing has a real mechanistic case: a team's pass-rate-over-expected is")
    print("almost definitionally tied to how much it runs.")
    print("=" * 78)

    from scheme_features import build_team_week_scheme
    from pbp_features import load_pbp

    pbp = load_pbp()
    scheme = build_team_week_scheme(pbp)
    rolling_cols = [c for c in scheme.columns if c.endswith("_rolling")]
    scheme_renamed = scheme[["team", "season", "week"] + rolling_cols].rename(columns={"team": "recent_team"})

    new_groups = {
        "pass_oe": ["pass_oe_rolling"],
        "tempo": ["shotgun_rate_rolling", "no_huddle_rate_rolling"],
        "early_down_pass_rate": ["early_down_pass_rate_rolling"],
        "defenders_in_box": ["avg_defenders_in_box_rolling"],
    }

    injuries = pd.read_csv(os.path.join(RAW_DIR, "injuries.csv"), low_memory=False)
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    inj_status = build_player_injury_status(injuries)
    flags = build_game_flags(schedules).rename(columns={"team": "recent_team"})
    context = build_game_context(schedules).rename(columns={"team": "recent_team"})

    specs = [
        ("rushing_yards", "player_prop_rushing_features", "build_rushing_yards_dataset",
         ["RB"], xgb, ["game_flags", "game_context", "snap_share"]),
        ("passing_yards", "player_prop_passing_features", "build_passing_yards_dataset",
         ["QB"], logreg, ["game_context"]),
        ("receptions", "player_prop_receptions_features", "build_receptions_dataset",
         ["WR", "TE"], xgb, ["game_flags"]),
    ]

    for prop_name, module_name, fn_name, positions, model_fn, context_needed in specs:
        print(f"\n--- {prop_name} ---")
        mod = __import__(module_name)
        df = getattr(mod, fn_name)(min_week=4)
        df = df[df["position"].isin(positions)].copy()
        df = df.dropna(subset=[PROP_CONFIGS[prop_name]["proxy_col"]])

        if "injury" in context_needed:
            df = df.merge(inj_status, on=["player_id", "season", "week"], how="left")
        if "game_flags" in context_needed:
            df = df.merge(flags, on=["recent_team", "season", "week"], how="left")
        if "game_context" in context_needed:
            df = df.merge(context, on=["recent_team", "season", "week"], how="left")
        if "snap_share" in context_needed:
            df = add_snap_share(df)

        df = df.merge(scheme_renamed, on=["recent_team", "season", "week"], how="left")

        base = production_features(prop_name)
        base_probs, base_correct = cv_pooled(df, base, model_fn)
        summarize("base (current production)", base_probs, base_correct)

        winners = []
        for name, feats in new_groups.items():
            probs, correct = cv_pooled(df, base + feats, model_fn)
            summarize(f"+ {name}", probs, correct)
            if correct.mean() - base_correct.mean() > 0.003:
                winners.append(name)

        print(f"  Beat base by >0.3pt: {winners if winners else 'none'}")
        if winners:
            combo = base + [f for n in winners for f in new_groups[n]]
            probs, correct = cv_pooled(df, combo, model_fn)
            summarize("  + combined winners", probs, correct)


if __name__ == "__main__":
    test_usage_trend()
    test_receptions_game_context()
    test_receptions_matchup()
    test_scheme_on_untested_props()
