# ParlayModel

An AI-assisted NFL betting agent: pulls odds + stats, runs prediction models through backtesting,
and (eventually) drives an approval-gated bet-placement flow via browser automation.

## Status

**Currently on: Phase 1 — Data & infra setup**

Not done yet:
- [ ] Run a real historical pull at home (`--years 2019 2020 2021 2022 2023 2024`) — this
      env's network can't reach the schedules data source, needs testing on unrestricted network
- [ ] Pick and wire up an odds API (The Odds API vs. SportsDataIO — not yet decided)
- [ ] Then move to Phase 2: baseline power-rating model + backtesting harness

## Progress Log

*Newest entry on top. Check this first to see exactly where to pick back up.*

**2026-08-13** — Added README status/progress tracking.

**2026-08-13** — Built and tested `data/pull_nflverse.py`: pulls schedules, play-by-play,
weekly stats, Next Gen Stats (passing/rushing/receiving), injuries, and snap counts.
Added validation checks on every pull (row counts, nulls, duplicates). Tested live for
2023–2024, all sources returned clean data. Not yet run for the full historical range.

**2026-08-13** — Repo initialized: folder structure, requirements.txt, .gitignore, .env.example.

## Roadmap

1. **Data & infra setup** — odds API + nflverse stats pipeline *(current phase)*
2. **Baseline model + backtesting harness** — power-rating baseline, ROI/CLV scoring
3. **Model iteration** — logistic regression → gradient-boosted trees on EPA-based features
4. **Bankroll & risk logic** — staking rules, approval-time bet summary
5. **Browser automation** — Claude in Chrome drives bet placement, human approves each bet
6. **Paper trading** — shadow-mode validation before any real money is wagered

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

Every pull is validated on the way in — row counts, missing key columns, duplicates, and
null rates on important fields are checked and reported, so a broken or changed upstream
source gets caught immediately instead of silently corrupting downstream models.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your ODDS_API_KEY
```

## Key metric

Model evaluation prioritizes **closing line value (CLV)** — whether a bet beat the closing
line — over raw win rate, since CLV is the stronger long-run signal of model quality.
