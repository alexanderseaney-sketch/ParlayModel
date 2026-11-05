"""Which picks count as "proven" for Picks and "Build my parlay" (2026-11-05).

Until week 9 of 2026 a pick was proven when its STAT TYPE beat the always-take-the-
commoner-side baseline on real Underdog lines (backtesting/line_history_track_record.json).
Re-graded walk-forward (each week judged only with the track record of the weeks before
it, 2026 weeks 4-8, real closing lines), those picks hit 54.6% (n=1326), under the 55.0%
a pick a standard 3-pick entry needs. Alex's week 9 trackers saw the same: catch Unders
went 17/43 in the Recommended set while the "raw model edge" test set made money.

What held up every week was how sure the MODEL is on the side it picks, on any stat type
whose lines are close to a coin flip:

    rule (walk-forward, wks 4-8)                         picks   hit
    old: stat-type verdict "trusted"                      1326   54.6%
    model side prob >= 60%, balanced stat, side not avoid 1148   58.6%
    ... and that side has hit 55%+ when it has 50+ picks   654   61.2%  (worst week 53%)
    model side prob < 53% on any balanced stat            2772   51.3%

The last rule is the one used here. Period (1Q/1H) and longest-reception props are left
out of those numbers because Alex keeps them out of parlays unless near-certain.
"""
CONFIDENT_SIDE_PROB = 0.60   # model's own chance for the side it picks
BALANCED = (0.30, 0.70)      # stat's real-line over rate: skip near-certain TD/sack Unders etc.
SIDE_MIN_N = 50
SIDE_MIN_ACC = 6 ** (-1 / 3)  # 55.0%, a standard 3-pick entry's break-even per pick


def side_of(choice) -> str:
    return "over" if str(choice).lower() in ("over", "higher") else "under"


def is_proven_pick(rec_row: dict | None, side: str, model_side_prob) -> bool:
    """rec_row: the stat's entry in the track record's by_stat (None if never graded)."""
    if not rec_row or model_side_prob is None:
        return False
    try:
        p = float(model_side_prob)
    except (TypeError, ValueError):
        return False
    if not p >= CONFIDENT_SIDE_PROB:
        return False
    over_rate = rec_row.get("over_rate")
    if over_rate is None or not BALANCED[0] < over_rate < BALANCED[1]:
        return False
    if rec_row.get(f"{side}_verdict", rec_row.get("verdict")) == "avoid":
        return False
    n = rec_row.get(f"picked_{side}") or 0
    acc = rec_row.get(f"{side}_pick_accuracy")
    return n < SIDE_MIN_N or (acc is not None and acc >= SIDE_MIN_ACC)
