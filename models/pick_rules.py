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

Added 2026-11-14, two more ways to be proven, both from 55% model confidence instead of
60% (walk-forward on wks 4-9, the 781 picks at 62.5% above become 1215 at 62.9%):

  line moved your way  an Under whose line Underdog RAISED 10%+ since it opened
                       (models/line_moves.moved_your_way): adds 304 picks at 62.2%.
  workload in flux     a catch pick, either side, for a player coming off a snap dip or
                       a missed game (models/role_flux.py): adds 130 picks at 66.9%.

Tested the same day and NOT used, because Underdog's lines already carry them: injury
tags and practice status, teammates out and who replaces them, QB changes, line and
defender injuries, rest, weather, news volume, Overs after a line cut, indoor games, a
player's own record against his lines, depth-chart changes, and negative-binomial
pricing for the trained count models.
"""
CONFIDENT_SIDE_PROB = 0.60   # model's own chance for the side it picks
BALANCED = (0.30, 0.70)      # stat's real-line over rate: skip near-certain TD/sack Unders etc.
SIDE_MIN_N = 50
SIDE_MIN_ACC = 6 ** (-1 / 3)  # 55.0%, a standard 3-pick entry's break-even per pick
MOVED_YOUR_WAY_SIDE_PROB = 0.55  # bar for an Under whose line was raised 10%+ since open
WORKLOAD_FLUX_SIDE_PROB = 0.55   # bar for a catch pick on a player off a snap dip / missed game


def side_of(choice) -> str:
    return "over" if str(choice).lower() in ("over", "higher") else "under"


def is_proven_pick(rec_row: dict | None, side: str, model_side_prob, moved_your_way: bool = False,
                   workload_flux: bool = False) -> bool:
    """rec_row: the stat's entry in the track record's by_stat (None if never graded).
    moved_your_way: line_moves.moved_your_way for this pick (an Under on a raised line).
    workload_flux: role_flux.flag for this pick (a catch prop, player's last game disrupted)."""
    if not rec_row or model_side_prob is None:
        return False
    try:
        p = float(model_side_prob)
    except (TypeError, ValueError):
        return False
    if rec_row.get(f"{side}_verdict", rec_row.get("verdict")) == "avoid":
        return False
    if moved_your_way and side == "under" and p >= MOVED_YOUR_WAY_SIDE_PROB:
        return True
    if workload_flux and p >= WORKLOAD_FLUX_SIDE_PROB:
        return True
    if not p >= CONFIDENT_SIDE_PROB:
        return False
    over_rate = rec_row.get("over_rate")
    if over_rate is None or not BALANCED[0] < over_rate < BALANCED[1]:
        return False
    n = rec_row.get(f"picked_{side}") or 0
    acc = rec_row.get(f"{side}_pick_accuracy")
    return n < SIDE_MIN_N or (acc is not None and acc >= SIDE_MIN_ACC)


# Alex's rule (2026-11-05, and 2026-10-22 for longest reception): quarter and half lines
# and longest-reception lines stay off Picks and out of every auto-built parlay unless the
# model is near-certain. Week 9's trackers leaned on 1Q/1H legs and they hit ~50-54%.
NEAR_CERTAIN = 0.80
NEAR_CERTAIN_ONLY_STATS = ("receiving_long",)


def near_certain_only(stat_name: str) -> bool:
    s = str(stat_name)
    return s.startswith("period_") or s in NEAR_CERTAIN_ONLY_STATS


def annotate(c, by_stat: dict, chance_col: str = "blend_prob"):
    """Adds verdict / record / skip_unless_sure to a candidates frame (ev_finder.
    build_candidates columns: stat_name, choice, model_prob, plus chance_col), so every
    screen and script that picks legs applies the same rules:

      verdict "trusted"  = proven pick (is_proven_pick: model 60%+ on a balanced stat whose
                           side has held up on real lines; or, from 55%, an Under whose
                           line was raised 10%+ since it opened or a catch pick on a
                           player whose last game was disrupted: columns moved_your_way
                           and workload_flux, when the frame has them)
      verdict "avoid"    = the side (or stat) loses on real lines (PR #15: assists Unders,
                           receptions Overs, ...)
      skip_unless_sure   = quarter / half / longest-catch line under NEAR_CERTAIN.
    """
    import numpy as np
    c = c.copy()

    def side_verdict(stat, choice):
        r = by_stat.get(stat, {})
        return r.get(f"{side_of(choice)}_verdict", r.get("verdict", "unproven"))

    def record(stat, choice):
        r = by_stat.get(stat, {})
        s = side_of(choice)
        acc = r.get(f"{s}_pick_accuracy")
        return acc if acc is not None and (r.get(f"picked_{s}") or 0) >= 30 else r.get("accuracy")

    raw = [side_verdict(s, ch) for s, ch in zip(c["stat_name"], c["choice"])]
    moved = c["moved_your_way"] if "moved_your_way" in c.columns else [False] * len(c)
    flux = c["workload_flux"] if "workload_flux" in c.columns else [False] * len(c)
    proven = [is_proven_pick(by_stat.get(s), side_of(ch), p, bool(m), bool(f))
              for s, ch, p, m, f in zip(c["stat_name"], c["choice"], c["model_prob"], moved, flux)]
    c["verdict"] = np.where(proven, "trusted", np.where(np.array(raw) == "avoid", "avoid", "unproven"))
    c["record"] = [record(s, ch) for s, ch in zip(c["stat_name"], c["choice"])]
    c["skip_unless_sure"] = c["stat_name"].map(near_certain_only) & (c[chance_col].astype(float) < NEAR_CERTAIN)
    return c


# "One story per entry" (week 8 review, 2026-10-26): a QB's passing Over next to his own
# pass catchers' Unders (or the reverse) bets on two different games at once. Those
# entries went 1/16 in week 8.
_QB_PASS_STATS = ("passing_yds", "passing_tds", "passing_comps", "passing_att", "passing_and_rushing_yds")
_CATCH_STATS = ("receiving_yds", "receiving_rec", "receiving_tgts", "rush_rec_yds")


def clashing_stories(legs: list[dict]) -> bool:
    for a in legs:
        if a.get("stat_name") not in _QB_PASS_STATS:
            continue
        for b in legs:
            if b is a or b.get("stat_name") not in _CATCH_STATS or not a.get("team") or a.get("team") != b.get("team"):
                continue
            if side_of(a.get("choice")) != side_of(b.get("choice")):
                return True
    return False
