# ParlayModel

An AI-assisted NFL betting agent: pulls odds + stats, runs prediction models through backtesting,
and (eventually) drives an approval-gated bet-placement flow via browser automation.

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

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your ODDS_API_KEY
```

## Key metric

Model evaluation prioritizes **closing line value (CLV)** — whether a bet beat the closing
line — over raw win rate, since CLV is the stronger long-run signal of model quality.
