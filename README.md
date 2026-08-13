# ParlayModel

An AI-assisted NFL betting agent: pulls odds + stats, runs prediction models through backtesting,
and (eventually) drives an approval-gated bet-placement flow via browser automation.

## Status

**Currently on: Phase 1 — Data & infra setup**

See the Progress Log below for the detailed handoff notes — the short version:

Not done yet:
- [ ] **Test `data/pull_espn_news.py` for real** — built but completely unverified, see latest log entry
- [ ] Run a real historical pull at home (see next step in latest log entry)
- [ ] Pick and wire up an odds API with **multi-book coverage** (needed for line shopping —
      see Betting Strategy section below; The Odds API vs. SportsDataIO — not yet decided)
- [ ] Then move to Phase 2: baseline power-rating model + backtesting harness

## Progress Log

*This log is also how "work Claude" and "home Claude" hand off to each other — since
neither has live access to the other's session, this file is the shared context. Every
entry is tagged with environment and dated. Newest entry on top. Read the top entry
before starting work to see what the other side left you.*

**Entry format:**
```
**YYYY-MM-DD — [work | home]**
- Did: what got done this session
- Blocked: anything that couldn't be finished here + why (e.g. network restriction, need a decision)
- Next: what the other environment (or next session) should pick up
```

---

**2026-08-13 — [work]**
- Did: Researched Underdog Fantasy's legal status in California. It's operating today via
  its **Champions** format (peer-to-peer picks vs. other users, not against the house) —
  switched from standard Pick'em after CA AG Bonta issued a July 2025 opinion that DFS is
  illegal gambling under state law. That opinion is advisory/non-binding, unenforced by
  courts, and Underdog (a real US company, not offshore) continues operating on the theory
  the P2P structure is defensible. Materially different risk profile than offshore
  sportsbooks — legally contested, not clearly illegal.
- Blocked: nothing.
- Next: **platform decision made — building toward Underdog Champions.** Important
  structural note: since Champions is peer-to-peer (picks vs. other users' picks, not vs.
  a sportsbook line), there's no traditional closing-line to beat — CLV as the primary
  model-quality metric doesn't translate directly here. The EPA/Next Gen Stats model work
  still matters for pick quality, but the backtesting/evaluation approach (Phase 2) needs
  to be rethought for a peer-to-peer format rather than assuming CLV tracking against a
  sportsbook line.

**2026-08-13 — [work]**
- Did: Researched California sports betting legal status, since Alex wants the model to
  eventually cover any bet type. Finding: **California has no legal, regulated sportsbook
  at any age** — voters rejected both legalization measures in 2022, realistic timeline
  for legalization is 2028 at the earliest. Turning 21 (Sept 18) does not unlock sports
  betting in CA the way it would in most states, since the barrier isn't age, it's that
  no legal market exists. Real options: (1) prediction markets like Kalshi/Polymarket,
  federally regulated as event contracts, legal in CA at 18+, though CA lawmakers are
  actively introducing bills targeting sports-specific prediction contracts specifically
  — unsettled ground; (2) betting while physically present in a state where it's actually
  legal, once 21, using real regulated sportsbooks; (3) offshore/unlicensed sportsbooks,
  which is what most CA bettors use in practice but carries real legal risk and no
  consumer protection — not something I'll help automate placing bets on.
- Blocked: waiting on Alex's decision on which platform path to build the execution side
  toward — this changes what "odds API" and the browser automation phase actually target.
- Next: once decided, update Phase 1 (odds API choice) and Phase 5 (browser automation
  target) accordingly.

**2026-08-13 — [work]**
- Did: Researched parlay strategy, bankroll management, and how sharp bettors find edges
  (multiple sources: correlation math, NJ regulatory data, professional betting guides).
  Key findings:
  - Parlay house hold is ~17-26% (4-leg) vs. ~4.5% single bets; same-game parlays run
    15-25%+ due to opaque correlation pricing. NJ 2024 data: parlays are ~22% of handle
    but ~41% of sportsbook net win — parlays are the house's best product, by design.
    Sharp/professional bettors avoid parlays almost entirely for this reason.
  - Bankroll: flat betting (1-3% of bankroll/bet) recommended until a full tracked season
    of results exists. Fractional (quarter/half) Kelly after that, once edge estimates are
    validated — never full Kelly, it assumes exact probability estimates.
  - CLV (closing line value) confirmed again across every source as the strongest
    long-run profitability predictor, independent of variance/win rate.
  - Line shopping matters — same bet prices differently across books; that spread is
    real, capturable edge. Confirms we need multi-book odds coverage, not one source.
  - Timing: lines softest Sun night-Tue, sharpen through the week. Wed-Fri injury
    reports create lagging mispricing in player props specifically (backups' props lag
    behind the target-share shift). Preseason lines are especially soft since outcomes
    depend on which players coaches choose to rest.
- Blocked: nothing — this is a strategy/research finding, not code.
- Next: **project direction change worth discussing with Alex explicitly** — the model
  should prioritize finding +EV straight bets first; parlays (if used at all) should only
  combine legs that are already independently +EV, never used to manufacture edge from
  stacking picks. Also: Phase 1's odds API pick needs to support multi-book comparison,
  not just one book's lines, to support line shopping.

**2026-08-13 — [work]**
- Did: Added a "Pre-approval sanity check" requirement — before showing any bet
  recommendation, do a live web search on that game's teams/players for anything recent
  the structured data feeds might have missed. Documented as a workflow rule (not a
  script) since it relies on live search at recommendation time, not a fixed pull.
- Blocked: nothing — this is a process rule, applies immediately going forward.
- Next: keep this in mind once we're building the actual recommendation/approval flow
  in Phase 4 — make sure the search step is baked into that code's output format, not
  just a mental note.

**2026-08-13 — [work]**
- Did: Built `data/pull_espn_news.py` — pulls league-wide + per-team NFL news (injuries,
  signings, trades, drama, anything storyline-relevant) from ESPN's unofficial public API.
  Tags each article with athletes/teams mentioned where ESPN provides that metadata.
- Blocked: **could not test this at all** — `espn.com` isn't reachable from this sandbox's
  network allowlist (confirmed via direct curl, got `host_not_allowed`). The code is
  reasoned through carefully but completely unverified against a live response.
- Next (home): run `python3 data/pull_espn_news.py` and actually check the output —
  confirm articles come back with real headlines/dates, spot-check that team news
  filtering works, and check `get_team_ids()` returns all 32 teams correctly. ESPN's
  endpoints are undocumented and can change without notice, so don't trust this script
  until it's been seen working on real data at least once.

**2026-08-13 — [work]**
- Did: Added this progress log format so work/home sessions can hand off cleanly.
- Blocked: nothing.
- Next: read this log at the start of every session, here or at home, before doing anything else.

**2026-08-13 — [work]**
- Did: Built and tested `data/pull_nflverse.py` — pulls schedules, play-by-play, weekly
  stats, Next Gen Stats (passing/rushing/receiving), injuries, snap counts. Validation
  checks on every pull (row counts, nulls, duplicates). Tested live for 2023–2024, all
  sources returned clean data.
- Blocked: this sandbox's network can't reach the full historical pull range or confirm
  behavior outside 2023-2024 — needs a real run on an unrestricted network.
- Next (home): run `python3 data/pull_nflverse.py --years 2019 2020 2021 2022 2023 2024`
  (drop `--skip-pbp` if you want play-by-play too, it's slow). Confirm it completes clean,
  then this phase's data side is basically done — next up is picking an odds API.

**2026-08-13 — [work]**
- Did: Repo initialized — folder structure, requirements.txt, .gitignore, .env.example.
- Blocked: nothing.
- Next: build the data feeder script.

## Roadmap

1. **Data & infra setup** — odds API (multi-book, for line shopping) + nflverse stats pipeline *(current phase)*
2. **Baseline model + backtesting harness** — power-rating baseline, ROI/CLV scoring
3. **Model iteration** — logistic regression → gradient-boosted trees on EPA-based features
4. **Bankroll & risk logic** — flat betting (1-3% of bankroll) until a tracked season of
   results exists, then fractional Kelly; approval-time bet summary includes a live
   web-search sanity check (see "Pre-approval sanity check" below)
5. **Browser automation** — Claude in Chrome drives bet placement, human approves each bet
6. **Paper trading** — shadow-mode validation before any real money is wagered

## Betting strategy — key findings (research pass 2026-08-13)

Full writeup in the progress log below; short version, since it changes how the model
should actually be used:

- **Parlays carry a structurally worse house edge than straight bets** — book hold runs
  ~17-26% on multi-leg parlays and 15-25%+ on same-game parlays, vs. ~4.5% on a single
  bet. Sharp/professional bettors avoid parlays almost entirely for this reason.
- **Implication for this project**: the model's real job is finding +EV on individual
  games/props (straight bets). Parlays should only ever combine legs that are *already*
  independently +EV on their own — never used to manufacture value from stacking picks.
- **Line shopping across multiple books is required, not optional** — the odds layer
  needs multi-book coverage so we can compare prices, not just pull from one source.
- **Bet timing**: lines are softest Sun night-Tue (least information priced in), sharpen
  through the week. Injury reports (Wed-Fri) create lagging mispricing in player props.
- **Preseason specifically** (starting this Saturday) is a soft-line environment — outcomes
  hinge on which players coaches rest, which the news feeder should help catch.

## Structure

```
data/
  raw/          # untouched pulls from odds API / nflverse
  processed/    # cleaned, feature-engineered datasets
models/         # model definitions, one file per model type
backtesting/    # backtest engine + results
bet_logs/       # logged bets: odds at bet time, closing odds, outcome, CLV
notebooks/      # exploratory analysis
```

## Data sources (all pulled by `data/pull_nflverse.py`)

- **Schedules & results** — games, scores, closing spread/total lines
- **Play-by-play** — source for EPA, success rate, and other advanced stats *(`--skip-pbp` to skip, it's large)*
- **Weekly player stats** — box score stats by player/week
- **Next Gen Stats** (AWS-powered tracking data) — passing, rushing, receiving: completion probability, separation, time to throw, air yards, etc.
- **Injuries** — weekly injury report status by player
- **Snap counts** — offensive/defensive/special teams snap share by player/week
- **ESPN news** (`data/pull_espn_news.py`) — league-wide + per-team news: injuries, signings,
  trades, suspensions, and general storylines. Untested in this environment (ESPN's domain
  isn't reachable here) — needs a live test run before it's trusted.

Every pull is validated on the way in — row counts, missing key columns, duplicates, and
null rates on important fields are checked and reported, so a broken or changed upstream
source gets caught immediately instead of silently corrupting downstream models.

## Pre-approval sanity check (non-negotiable)

Structured data sources lag — nflverse updates aren't real-time, and even the ESPN news
feeder can miss something from the last few hours before kickoff. **Before any bet
recommendation is shown to Alex for approval, do a live web search on that game's teams
and key players** for anything from the last 24-48 hours the pipeline might have missed:
last-minute inactives, weather, suspensions, coaching decisions, line movement reasons,
anything storyline-relevant. This applies whether the model is being tested here, at home,
or anywhere else — it's a workflow rule, not a script, since it uses live web search
directly rather than a fixed data pull. Skipping this step means presenting a
recommendation that could already be stale by the time it's approved.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your ODDS_API_KEY
```

## Key metric

Model evaluation prioritizes **closing line value (CLV)** — whether a bet beat the closing
line — over raw win rate, since CLV is the stronger long-run signal of model quality.
