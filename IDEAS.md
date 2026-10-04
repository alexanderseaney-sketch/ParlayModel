# Ideas Backlog

Running list of ideas for ParlayModel / the dashboard / fantasy stuff.
Newest at the top. Move items to **Done** (with date) once built.

## Open

### Live tracking of placed parlays (added 2026-10-03)
Watch placed entries during games: each leg's current stat vs its line, hit/miss/pending,
and whether the entry is still alive.

**Where it fits:**
- A bet log already exists: `bet_logs/bets.csv` (`date, player, stat, choice, line,
  multiplier_or_odds, stake, result, ...`) via `load_bet_log()` / `append_bet()` in
  `dashboard/utils.py`. It's **empty so far** and stores one row per leg, with no ID
  grouping legs into one parlay. That needs an `entry_id` column first.
- Needs an **in-game** stats feed. nflverse only updates after games, so a live source is
  required (e.g. ESPN's public scoreboard/box-score JSON, which is free and unofficial). Player names
  must map to it the same way Underdog names were mapped.

**Open questions:**
- How do bets get in? Manual entry, one click from the +EV Finder/slip builder, or
  importing from the betting app (no official APIs, so probably manual or screenshot).
- Refresh: auto-poll every ~1 min during game windows vs a refresh button.
- Extras: pace projection ("on pace for 74 yds vs 68.5"), notifications when a leg hits or
  busts, and auto-grading `result` after games, which feeds real ROI tracking on the
  Model Performance page.

### AI chat assistant inside the app (added 2026-10-03)
A chatbot in the dashboard that answers questions and does tasks for you, e.g. "best 3-leg
entry tonight", "why is the model on the under for X", "compare these two WRs",
"log this bet", "run the data pulls".

**Where it fits:**
- Claude is already wired in: `models/interpreter.py` (Opus 5.5, `ANTHROPIC_API_KEY`)
  writes the pick/entry summaries on the +EV Finder. A chat would build on the same setup.
- Give it **tools** that call existing project functions instead of letting it guess:
  predictions, +EV Finder results, consensus lines, player game logs, depth charts,
  the Fantasy page data, `append_bet()`. Same rule as the interpreter: only cite numbers
  the tools return.
- Natural UI: a Streamlit `st.chat_message` page or sidebar panel.

**Progress (2026-10-04):** built as the **AI Assistant** page (`dashboard/assistant.py`):
Claude Opus 5.5 via the SDK tool runner with 10 tools -- best picks, best entries, a
player's props, model predictions + top features, game logs, depth charts, live news,
the real-line track record, the bet log, and `stage_bet_log` (staged; saved only after a
Confirm click). Needs `ANTHROPIC_API_KEY`. Not yet: running data pulls from chat, Fantasy
page data, live parlay tracking.

**Open questions:**
- Which tasks should it be able to *do* vs just answer? Actions like logging bets or running
  pulls should probably need a confirm click.
- Cost: every message is an API call, so pick the model per task (Opus for analysis,
  a cheaper model for simple lookups).
- Ties into the live-tracking idea above ("how's my parlay doing?").

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
