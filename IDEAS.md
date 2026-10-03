# Ideas Backlog

Running list of ideas for ParlayModel / the dashboard / fantasy stuff.
Newest at the top. Move items to **Done** (with date) once built.

## Open

### Multi-book lines: choose which betting app's lines/picks you see (added 2026-10-02)
Let the dashboard show lines and parlay picks from different apps (e.g. PrizePicks,
Sleeper, DraftKings, FanDuel), not just Underdog, with a selector for which one(s).

**Where it fits:**
- Today everything live is **Underdog-only**: `data/pull_underdog.py` → `underdog_props.csv`,
  consumed by `dashboard/app.py`, `dashboard/utils.py`, `models/generate_weekly_bet_slip.py`,
  `models/line_movement_features.py`.
- The Odds API is already wired up (`data/pull_historical_odds.py`, `ODDS_API_KEY`) but only
  for *historical* lines. Its live endpoint covers DK/FD/etc. in one call, but the free tier
  is 500 credits/month and props are expensive, so live multi-book polling eats that budget fast.

**Open questions:**
- Which apps? Pick'em apps (PrizePicks, Sleeper) vs sportsbooks (DK, FD) work differently:
  pick'em has fixed payouts, books have per-leg odds/juice, so "edge" means different things.
- Each app has its own slip rules, e.g. Underdog's same-team restriction (already validated
  in the dashboard). Every new app needs its own rules check.
- Player ID matching: each source names players differently. Underdog already needed an
  ID-mapping fix.
- Bonus feature: **line shopping**. Show the best line for a prop across all apps.

**Progress (2026-10-02):** `data/pull_odds_api.py` pulls live sportsbook props from The
Odds API into a common format (`book, player, stat_name (Underdog naming), line, choice,
american, decimal, implied_prob, fair_prob`) and `consensus()` gives a per-prop median
de-vigged probability + best over/under price (line shopping). The +EV Finder uses the
consensus as a second market opinion. Still open: PrizePicks/Sleeper pullers, a book
selector, per-book slip rules.

**Rough approach:** add a common props format (`book, player, stat, line, over_odds, under_odds,
pulled_at`), one puller per app writing to it, and a book selector in the dashboard. The
slip builder then validates against that book's rules.

## Done

- **+EV Finder, Model Performance page, real-line backtest, leakage audit** (2026-10-02) —
  see the README's 2026-10-02 log entry.
