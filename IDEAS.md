# Ideas Backlog

Running list of ideas for ParlayModel / the dashboard / fantasy stuff.
Newest at the top. Move items to **Done** (with date) once built.

## Open

### AI Assistant: remaining extensions (split out 2026-10-07)
The assistant itself shipped (see Done). Not built yet:
- Running data pulls from chat (should need a confirm click, like logging bets).
- Fantasy page data as a tool (projections, start/sit, matchups).
- A live-tracker tool ("how's my parlay doing?") on top of `dashboard/live_tracker.py`.
- Cost: pick the model per task (Opus for analysis, a cheaper model for simple lookups).

### Redesign follow-ups (added 2026-10-07)
- **Practice history chips (W·T·F)** on the Injury Tracker: nflverse `injuries.csv` keeps only
  the latest practice status per player-week, so the page shows one status. Needs a per-day
  practice-report source.
- **Quarter / half models** reset their rolling features each season; a retrain on the
  corrected weekly stats would fix early-season periods.

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

- **2026-10 redesign (Nocturne handoff)** (2026-10-07). New pages: **Home** (bankroll, open
  entries, best edges, injury movers, this week's games), **Bankroll** (balance, drawdown, ROI by
  stat / entry size, saved starting bankroll), **Game Center**, **Player Pages**, **Matchup
  Heatmap**, **Injury Tracker** (status changes + line moves from the new committed
  `data/raw/underdog_line_history.csv`), **Usage Trends**. Navigation is grouped Overview /
  Betting / Research / Admin. Polish passes on Weekly Bet Slip (budget presets), Parlay Builder
  (confidence tiers), Bet Log (entry cards), Data Status, NFL Stats, Depth Charts, Run Data
  Pulls, News, Compare, Fantasy and AI Assistant. No existing feature removed.
- **Live tracking of placed parlays** (2026-10-03). Bet Log live tracker from ESPN box scores
  (`dashboard/live_tracker.py`, 60 s auto-refresh): per-pick value vs line, pace, hit / miss /
  alive, game clock. Picks are grouped into entries (`entry_id`), with one-click "Save final
  results" grading and delete-selected / clear-all.
- **AI chat assistant** (2026-10-03). **AI Assistant** page (`dashboard/assistant.py`): Claude
  Opus 5.5 with tools for best picks / entries, a player's props, predictions + top features,
  game logs, depth charts, news, the real-line track record, the bet log, and
  `stage_bet_log` (saved only after a Confirm click). Remaining ideas are under Open.
- **Hosted bet persistence + live Underdog feed** (2026-10-03). Bets and settings saved to a
  private GitHub repo when `BET_LOG_GITHUB_TOKEN` is set; Underdog lines fetched live with a
  60 s auto-refresh.

- **+EV Finder, Model Performance page, real-line backtest, leakage audit** (2026-10-02) —
  see the README's 2026-10-02 log entry.
