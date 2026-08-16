"""
Tests a question specific to PARLAYS, not single-prop accuracy: the dashboard's Parlay
Builder combines multiple legs' probabilities by simple multiplication
(dashboard/utils.py's parlay_combined_multiplier: combined_prob = p1 * p2 * ... * pn).
That silently assumes every leg is statistically INDEPENDENT of every other leg. This
has never been tested anywhere in this project -- the single-prop accuracy work all
season has been about getting each p_i right in isolation, never about whether it's
valid to multiply them together.

It's almost certainly wrong for legs on the SAME team in the SAME game. Game script --
a blowout, a shootout, a run-heavy grind-it-out win -- tends to move multiple players'
outcomes together or apart at the same time, and pure independence can't see that.

Measures REAL empirical correlation using 2019-2024 data: for every game, gathers each
rostered skill player's over_proxy_line outcome across all 4 props (receiving yards,
rushing yards, passing yards, receptions), then for same-team pairs across position/
prop combinations, compares the actual joint hit rate to what pure independence would
predict. Reports the phi coefficient (correlation for two binary outcomes, -1 to 1)
per pair-type category, pooled across every historical instance of that pairing.
"""
import os

import numpy as np
import pandas as pd

from current_predictions import PROP_CONFIGS, _build_base_dataset

MIN_PAIR_COUNT = 100  # don't report a "correlation" built on a handful of games


def build_outcomes() -> pd.DataFrame:
    """One row per (player, game, prop_type) with the actual over_proxy_line outcome
    -- no model, no context merges needed, just the real historical result."""
    frames = []
    for prop_type, config in PROP_CONFIGS.items():
        df = _build_base_dataset(prop_type, min_week=4)
        df = df[df["position"].isin(config["positions"])].copy()
        df = df.dropna(subset=[config["proxy_col"], "over_proxy_line"])
        df["prop_type"] = prop_type
        frames.append(df[["player_id", "player_display_name", "position", "recent_team",
                           "season", "week", "prop_type", "over_proxy_line"]])
    return pd.concat(frames, ignore_index=True)


def build_same_team_pairs(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Self-joins on (season, week, recent_team) -- every pair of different players on
    the same team in the same game, across any two prop legs."""
    merged = outcomes.merge(
        outcomes, on=["season", "week", "recent_team"], suffixes=("_a", "_b")
    )
    # Drop self-pairs and de-duplicate (a,b) vs (b,a) -- keep player_id_a < player_id_b,
    # or for the same player across two different prop types, prop_type_a < prop_type_b.
    merged = merged[
        (merged["player_id_a"] < merged["player_id_b"])
        | ((merged["player_id_a"] == merged["player_id_b"]) & (merged["prop_type_a"] < merged["prop_type_b"]))
    ]
    return merged


def phi_coefficient(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Correlation for two binary variables. Returns (phi, p_a, p_b)."""
    p_a, p_b = a.mean(), b.mean()
    p_joint = (a & b).mean()
    denom = np.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
    phi = (p_joint - p_a * p_b) / denom if denom > 0 else 0.0
    return phi, p_a, p_b


def main():
    outcomes = build_outcomes()
    print(f"Built {len(outcomes)} player-game-prop outcome rows across 2019-2024.\n")

    pairs = build_same_team_pairs(outcomes)
    print(f"{len(pairs)} same-team same-game leg pairs found.\n")

    pairs["pair_key"] = pairs.apply(
        lambda r: tuple(sorted([f"{r['position_a']} {r['prop_type_a']}", f"{r['position_b']} {r['prop_type_b']}"])),
        axis=1,
    )

    results = []
    for key, group in pairs.groupby("pair_key"):
        if len(group) < MIN_PAIR_COUNT:
            continue
        a = group["over_proxy_line_a"].values.astype(bool)
        b = group["over_proxy_line_b"].values.astype(bool)
        phi, p_a, p_b = phi_coefficient(a, b)
        results.append({"pair": key, "n": len(group), "phi": phi, "p_a": p_a, "p_b": p_b})

    results.sort(key=lambda r: abs(r["phi"]), reverse=True)

    print(f"{'Pair':<55} {'n':>6}  {'phi':>7}  interpretation")
    print("-" * 100)
    for r in results:
        strength = "REAL" if abs(r["phi"]) >= 0.05 else ("weak" if abs(r["phi"]) >= 0.02 else "~none")
        direction = "positive" if r["phi"] > 0 else "negative"
        pair_str = " + ".join(r["pair"])
        print(f"{pair_str:<55} {r['n']:>6}  {r['phi']:>+7.3f}  {strength} {direction if abs(r['phi']) >= 0.02 else ''}")

    print("\nphi interpretation: 0 = independence confirmed (current parlay math is fine),")
    print("nonzero = independence assumption is measurably wrong for that pairing.")
    print("Rule of thumb: |phi| >= 0.05 is a real, usable effect at these sample sizes;")
    print("below that is likely noise even if technically nonzero.")

    out_path = os.path.join(os.path.dirname(__file__), "parlay_leg_correlations.csv")
    real = [r for r in results if abs(r["phi"]) >= 0.05]
    out_df = pd.DataFrame([{
        "position_prop_a": r["pair"][0], "position_prop_b": r["pair"][1],
        "n": r["n"], "phi": round(r["phi"], 4),
    } for r in real])
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df)} real (|phi|>=0.05) pairings -> {out_path}")


if __name__ == "__main__":
    main()
