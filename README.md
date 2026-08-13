# ParlayModel

An AI-assisted NFL betting agent: pulls odds + stats, runs prediction models through backtesting,
and (eventually) drives an approval-gated bet-placement flow via browser automation.

## Status

**Currently on: Phase 2 — Baseline model + backtesting harness (game model done, player-prop models started)**

Done:
- [x] Real 6-season historical pull (2019-2024) — nflverse works from any environment
- [x] **Game-winner model**: Elo + stats bootstrap ensemble, 64.1% avg accuracy across 5
      seasons (70.7% on high-confidence games) — saved to `models/final_model.pkl`
- [x] **Player-prop model (receiving yards, WR/TE)**: 66.7% avg accuracy across 5 seasons,
      very stable (65.6%-67.2% range) — saved to
      `models/player_prop_receiving_yards_model.pkl`. Strongest signal: target share.
      **Important caveat**: backtested against the player's own rolling average as a
      proxy line, since no historical Underdog line archive exists yet — see log for detail.
- [x] Pulled real play-by-play data and tested proper per-play efficiency features
      (garbage-time filtered) on the game-winner model — **result: flat-to-worse than
      current best (63.5-63.9% vs. 64.1%)**. Real negative result, disproves last
      round's hypothesis that PBP data was the key to closing the market-ceiling gap.

See the Progress Log below for full detail on every round of testing.

Not done yet:
- [ ] **Build the same player-prop pipeline for rushing yards, passing yards, and
      receptions** — only receiving yards is done so far, out of Underdog's likely
      prop menu
- [ ] Start saving real Underdog prop lines over time (once `pull_underdog.py` is
      tested) to replace the proxy-line backtesting approach with real historical lines
- [ ] Wire both `final_model.pkl` and `player_prop_receiving_yards_model.pkl` into the
      dashboard's Parlay Builder, replacing the manual probability slider
- [ ] Consider pulling real play-by-play data (not `--skip-pbp`) for richer EPA features
- [ ] **Test `data/pull_espn_news.py` for real** — built but completely unverified, see log
- [ ] **Test `data/pull_underdog.py` for real** — built but completely unverified, see log

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
- Did: Built `models/feature_engineering.py` — pre-game team-week features from real
  data (EPA offense/defense, injury counts, Next Gen Stats CPOE + avg separation), each
  computed as a strictly-prior rolling average (no leakage, same discipline as the Elo
  backtest). Then ran real iterative model training (`models/train_iterate.py`,
  `models/train_iterate_v2.py`): **trained on 2019-2023, tested on the entire 2024
  season held out** — every number below is real, not estimated.

  | Model | Accuracy (2024 holdout) | Brier |
  |---|---|---|
  | Elo alone (baseline) | 67.1% | — |
  | Logistic regression, EPA only | 66.2% | 0.222 |
  | + injury counts | 66.2% | 0.219 |
  | + Next Gen Stats (CPOE, separation) | 67.1% | 0.213 |
  | XGBoost, same features | 65.0% | 0.227 |
  | **Logistic regression: EPA + injuries + NGS + Elo win prob** | **70.4%** | **0.202** |
  | XGBoost, same combined features | 67.1% | 0.214 |

  **Key finding**: pure box-score/EPA features alone don't beat Elo — they carry
  overlapping information. The real gain came from *combining* Elo's rating with the
  richer features, not replacing it. XGBoost underperformed logistic regression across
  the board here, likely overfitting on only 1,167 training games — simpler model
  generalized better on this amount of data.
- Blocked: nothing — this all ran for real in this environment, using real 2019-2024
  nflverse data.
- Next: **current best model is the "EPA + injuries + NGS + Elo" logistic regression at
  70.4%.** Caveats worth taking seriously before trusting this number fully: it's a
  single 240-game holdout (2024 only) — worth validating with additional holdout seasons
  (e.g. repeat with 2023 held out, 2019-2022+2024 trained) before concluding 70.4% is the
  real generalization accuracy rather than a lucky split. Other features worth testing
  next: turnover margin, weather (temp/wind columns already in schedules.csv), rest days
  (home_rest/away_rest already in schedules.csv, unused so far), and QB-specific NGS
  rather than team-average. This is genuinely a solid working model for now, not a
  finished one — good foundation to keep iterating on.

---

**2026-08-13 — [work]**
- Did: Pulled real play-by-play data (293,478 plays, 2019-2024 — previous runs all used
  `--skip-pbp`). Built proper per-play efficiency features (`models/pbp_features.py`):
  EPA/play and success rate (offense and defense, computed directly from defensive plays
  rather than inferred from opponents), pass/rush EPA split, explosive play rate — all
  with **garbage time filtered out** (16+ pt margin in Q4, or 21+ pt margin in the second
  half — standard practice in public EPA models, since blowout garbage-time plays distort
  true efficiency numbers).
  **Real result, tested the same rigorous 5-season CV way**: these better-constructed
  features did NOT beat the current best model. Current best (EPA totals from weekly_stats
  + Elo + injuries + NGS + turnovers + weather/rest): 64.1% mean. Same setup with PBP
  per-play features swapped in for the EPA totals: 63.5%. Adding PBP features alongside
  the existing ones instead of replacing: 63.9%. Both PBP variants came in flat-to-worse.
- **This disproves the hypothesis from the previous log entry** — I'd flagged real PBP
  data as "the most promising remaining lever" to close the gap toward the 67.6% market-
  ceiling number. It wasn't. Honest read on why: Elo already implicitly captures most of
  the efficiency signal through margin-of-victory updates each week, so a more precise
  version of the same underlying information (per-play rate vs. volume total) doesn't add
  much the model didn't already have access to some other way. This is a real, useful
  negative result, not a wasted afternoon — it rules out a specific hypothesis rather than
  leaving it as an untested "probably would help" assumption.
- Blocked: nothing — real data, real test, real (if unexciting) result.
- Next: the market-ceiling gap (67.6% vs. our 64.1%) likely isn't closed by more precise
  box-score-derived stats at all — it's probably from information categories we don't have
  any version of yet: real injury *severity* (not just a raw count), actual line movement
  over the week (sharp money signals), or coaching/scheme tendencies. Worth being honest
  that closing this gap may need fundamentally different data sources, not more iteration
  on what we already have. The PBP data isn't wasted, though — it's still useful for the
  player-prop models (this round only tested it on the game-winner model).

**2026-08-13 — [work]**
- Did: **First player-prop model** — Alex flagged that everything built so far predicts
  game WINNERS, but Underdog Champions is about PLAYER PROPS. Built a receiving-yards
  model for WR/TE (RBs excluded — Next Gen Stats receiving data has zero coverage for
  RBs in this pull, a real gap not a bug). Features: player's own rolling receiving
  yards/targets/target share/air yards (season-to-date + trailing-3-game), NGS rolling
  (avg separation, cushion, YAC over expectation), and opponent's rolling defensive EPA
  allowed (matchup difficulty).
  - **Important honest limitation**: there's no historical archive of real Underdog prop
    lines to backtest against. Used the player's own trailing rolling average as a
    backtestable proxy line instead — this tests "can the model beat recent-form
    momentum," which is a reasonable stand-in (real lines are usually set close to recent
    form + matchup adjustment) but is NOT the same as testing against real historical
    Underdog lines, which don't exist to test against.
  - **Real cross-validated result (5-season leave-one-out, same discipline as the team
    model)**: 66.7% mean accuracy, remarkably stable — 65.6% to 67.2% across all 5
    seasons (much tighter than the team-win model's 61.3%-67.1% range). Naive "always
    guess over" baseline is only 43-48%, so this beats naive by a wide margin.
  - **Strongest signal by far: target share** (rolling), coefficient dwarfs every other
    feature. Makes sense — target share is the cleanest predictor of receiving
    opportunity. Separation and cushion (open receivers) also help meaningfully.
  - Saved production model (100-model bootstrap ensemble, same approach as the team-win
    model) to `models/player_prop_receiving_yards_model.pkl`.
- Blocked: nothing structurally, but the proxy-line caveat above is real and matters —
  once real Underdog lines start getting pulled (`pull_underdog.py`, still untested),
  backtesting against actual historical lines (saved over time) will be the honest
  version of this test.
- Next: **this same pipeline generalizes directly to other prop types** — rushing yards
  (RB-focused, would need NGS rushing instead of receiving), passing yards (QB-focused,
  NGS passing), and receptions (same data already pulled here, just a different target
  column). Each needs its own model built and validated the same rigorous way — not yet
  done, scoped as the next round of work. Also worth starting to actually save Underdog's
  real prop lines over time (via `pull_underdog.py`, once tested) to build a true
  historical archive — that replaces the proxy-line approximation with the real thing.

**2026-08-13 — [work]**
- Did: Assembled everything validated so far into a final production model
  (`models/train_final.py`), testing the two remaining open questions first:
  1. **QB-specific CPOE vs. team-average CPOE**: tested via the same 5-season CV. Team
     -average won (64.1% mean vs. 63.3% for QB-specific) — another honest negative
     result. Isolating the starting QB's own history sounded like it should sharpen the
     signal, but didn't in practice; kept team-average in the final feature set.
  2. **Market ceiling check (diagnostic only)**: added the raw Vegas closing spread as a
     feature just to see how much information the market has that our stats don't —
     jumped mean accuracy from 64.1% to **67.6%**. This confirms the market knows things
     our current feature set doesn't fully capture (expected and fine — Vegas has far
     more inputs than we do). **This feature is NOT included in the production model** —
     the whole point is finding our own edge on Underdog, not just re-deriving what a
     sportsbook already prices in. Kept as a documented ceiling to measure future
     feature additions against.
  - **Final production model**: 100-model bootstrap ensemble (the same approach
    validated last round — high-agreement predictions hit 70.7% vs. 57.7% low-agreement)
    trained on all 6 seasons (2019-2024) of data, using: Elo win probability, EPA
    offense/defense, injury counts, team-average CPOE, average separation, turnover
    margin, rest-day differential, temperature, wind, and dome/outdoor. Saved to
    `models/final_model.pkl` (100 logistic regression models + feature list, 60KB).
- Blocked: nothing — every number above is real, from actual cross-validated runs.
- Next: this is a genuinely solid, honestly-tested foundation — ~64% average accuracy,
  ~71% on the high-confidence majority of games. To meaningfully improve further, the
  67.6% market-ceiling number suggests real room exists, but closing that gap needs
  fundamentally new information (not just recombining what we already have) — candidates:
  actual play-by-play EPA (not yet pulled, `--skip-pbp` has been used every run so far),
  more granular injury severity (currently just a raw count), or coaching/scheme data we
  don't have access to. `final_model.pkl` is ready for the dashboard's Parlay Builder to
  load and replace the manual "your probability estimate" slider with real predictions —
  that's the natural next integration step.

**2026-08-13 — [work]**
- Did: Tested whether model AGREEMENT is itself a useful signal — trained the same
  logistic regression architecture on 100 different bootstrap resamples of the 2019-2023
  training data, tested each on the full 2024 season, then checked whether games where
  those 100 runs agree with each other are actually the ones it gets right more often.
  **Real result**: when ≥90% of the 100 models agree (188 of 240 games — most games have
  high consensus, median agreement was 100%), accuracy is **70.7%**. When agreement drops
  below 90% (52 harder/closer games), accuracy drops to **57.7%**. Overall: 67.9%.
  Correlation between agreement level and correctness: +0.137 (positive, weak-but-real —
  strongest signal is the high vs. low split, not fine-grained agreement percentages,
  which get noisy with small sample sizes in the middle buckets).
- Blocked: nothing — real bootstrap run, real per-game results saved to
  `models/ensemble_agreement_results.csv`.
- Next: **this is a usable confidence filter for the betting side** — the model
  shouldn't be trusted equally on every game. Once the approval-flow (Phase 4) is built,
  it should surface this agreement/consensus level alongside each pick, and probably
  default to skipping or flagging low-agreement games rather than betting them at the
  same size as high-agreement ones. Worth testing this same bootstrap-agreement approach
  again once QB-specific NGS and market-based features are added, to see if agreement
  gets sharper (more games pushed into the high-consensus bucket) with better features.

**2026-08-13 — [work]**
- Did: Added turnover margin (turnovers forced − committed, via the same opponent-lookup
  pattern as defensive EPA), weather (temp/wind, already in schedules.csv), and rest-day
  differential as features. **Ran proper leave-one-season-out cross-validation across all
  5 of the last 5 seasons (2020-2024)** — train on the other 4-5 seasons, test on the held
  -out one, repeated for each. This is the real test of whether a result generalizes.
  Real results (mean accuracy across all 5 holdout seasons):
  - Elo alone: **63.1%** (range 61.3%-67.1%)
  - Stats-only model (EPA/injuries/NGS/turnovers/weather/rest, no Elo): **62.1%** — at or
    below Elo in every single season, confirms the earlier finding that box-score stats
    alone don't add signal beyond what Elo already captures
  - **Elo + stats combined: 64.1%** — beat or tied Elo alone in *every one* of the 5
    seasons (never worse), by about 1 percentage point on average
- **Correction to the previous log entry**: the earlier reported 70.4% (single 2024
  holdout) was optimistic — a good split, not the honest expected performance. Re-running
  that same 2024 holdout inside this proper CV (now with turnover/weather/rest added too)
  gives 68.3%, and the trustworthy cross-validated average is 64.1%. The real, defensible
  finding is smaller than first reported: **~1 point of consistent improvement over Elo**,
  not ~3 points on one lucky year. Flagging this explicitly rather than letting the
  flashier earlier number stand — worth remembering for how any future "big improvement"
  result should be treated until it's been checked across multiple holdouts.
- Blocked: nothing — real numbers, real cross-validation, in this environment.
- Next: current best, honestly validated model is **logistic regression on Elo + EPA +
  injuries + NGS + turnovers + weather + rest, ~64% average accuracy across 5 seasons of
  holdout testing.** This is a solid, real foundation — modest edge, but a consistent
  and honestly-tested one. Further ideas worth testing the same rigorous way: QB-specific
  NGS instead of team-average, a market-based feature (Vegas spread itself, to see how
  much of the model's signal the market already prices in), and whether XGBoost with
  more training data (once more seasons or weekly granularity is added) starts to
  outperform logistic regression the way it currently doesn't.

**2026-08-13 — [work]**
- Did: **Phase 2 started for real, with actual results (not just written code).** Pulled
  real 6-season historical data (2019-2024, schedules + weekly stats + NGS + injuries +
  snap counts — nflverse is reachable from this sandbox, unlike ESPN/Underdog). Built
  `models/elo_baseline.py` (standard Elo power-rating system with home-field advantage
  and margin-of-victory scaling) and `backtesting/backtest_elo.py` (proper walk-forward
  backtest — predicts each game using only ratings built from prior games, no future
  leakage). Verified the nflverse `spread_line` sign convention against real data
  (positive = home favored) before trusting it in the spread-comparison math.
  **Real results on 1,599 games (2019-2024 regular season):**
  - Straight-up accuracy: **62.3%** (vs. 52.9% home-field-only baseline — real signal)
  - Mean absolute error vs. Vegas closing spread: **2.83 points** (model is "sane" —
    tracks the market's general shape; this is a sanity check, not an edge metric)
  - By season: ranged 59.4%-66.2%, no season looked broken or like an outlier bug
- Blocked: nothing — this all ran and was verified in this environment.
- Next: this Elo baseline is the reference point for every future model (logistic
  regression, XGBoost on EPA/NGS features) to beat. Next step is building out the feature
  engineering pipeline (EPA, NGS, injury/snap-count features) so a real ML model can be
  trained and backtested against this same 1,599-game set, using the same walk-forward
  approach to avoid leakage.

**2026-08-13 — [work]**
- Did: Built `dashboard/` — a Streamlit UI. Originally scoped as a general data browser,
  then Alex redirected it into a **Parlay Builder**: browse Underdog props, add legs to
  a slip, enter a probability estimate per leg, see combined parlay math, get flagged if
  any leg lacks individual edge (enforces the "only stack already-+EV legs" rule from the
  Betting Strategy section). Also has Overview (data status), NFL Stats/ESPN News/Underdog
  Props browsers, a manual Bet Log (form + table), and Run Data Pulls (buttons to trigger
  each pull script). Fully tested here using Streamlit's `AppTest` framework against fake
  data — clicked through all 7 pages and the actual "add a leg" interaction, no exceptions.
  Caught and fixed one real bug during testing (an edit had accidentally deleted the
  `run_pull_script` function signature) — this is why testing before pushing matters, an
  untested "should work" version had a real crash in it.
- Blocked: nothing structurally, but three honest limitations to know about:
  1. No real model yet — "your probability estimate" is manual input until Phase 2/3 exist.
  2. Underdog's actual odds/payout field names are unknown until `pull_underdog.py` runs
     against live data — the UI looks for several likely column names and gracefully
     shows "odds unknown" if none match. Re-check this once real data exists.
  3. Doesn't place bets — that's Phase 5, human-approved, home-only.
- Next: run `streamlit run dashboard/app.py` at home once real data has been pulled, and
  check whether the odds/multiplier column detection actually finds Underdog's real field
  names (see `find_column()` calls in `dashboard/app.py`'s Parlay Builder section) — if
  not, the column candidate lists need updating to match reality.

**2026-08-13 — [work]**
- Did: Built `data/pull_underdog.py`. There's no official Underdog API, but found their
  internal (unofficial) endpoint — `api.underdogfantasy.com/beta/v5/over_under_lines` —
  via an open-source reference scraper (github.com/aidanhall21/underdog-fantasy-pickem-scraper),
  and wrote our own puller against it rather than depending on someone else's script.
  Response schema: players + appearances + over_under_lines (nested options per stat,
  over/under choice). Parses into one row per prop option.
- Blocked: **completely untested** — `api.underdogfantasy.com` isn't reachable from this
  sandbox (confirmed via curl, `host_not_allowed`), same as espn.com earlier. Also worth
  being deliberate about: this is unofficial/reverse-engineered access to a real-money
  platform, which likely isn't covered by their ToS even for read-only data — lower risk
  than automating bet placement, but not zero. Worth a conscious decision, not a default.
- Next (home): run `python3 data/pull_underdog.py` and sanity-check the output — real
  player names, real prop lines, NFL filter working. If the schema has changed since this
  reference scraper was written, this will likely fail loudly (empty result) rather than
  silently — check the validation warnings if so.

**2026-08-13 — [work]**
- Did: Alex confirmed Underdog is available and working from his CA location — platform
  decision is settled, no longer tentative.
- Blocked: nothing.
- Next: figure out what data Underdog's Champions/Pick'em product actually exposes
  (player prop pool, projection lines, how Champions scoring works) — this determines
  what the model needs to predict against, since it's not a traditional sportsbook line.
  Check whether Underdog has any accessible endpoints (official or otherwise) similar to
  how the ESPN news feeder works, or whether this needs manual/browser-based data capture.

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

## Dashboard

Local UI for managing project data and building parlays, built with Streamlit.

```bash
streamlit run dashboard/app.py
```

Pages:
- **Overview** — status of every data file (pulled or not, row count, last updated)
- **Parlay Builder** — browse Underdog props, add legs to a slip, enter a probability
  estimate per leg (manual for now — will pull from the model once Phase 2/3 exist),
  see combined parlay math, and get flagged if any leg lacks individual edge (per the
  "only stack already-+EV legs" rule in the Betting Strategy section above)
- **NFL Stats / ESPN News / Underdog Props** — browse/filter/search each pulled dataset
- **Bet Log** — manually log bets for now (form + table); will connect to the automated
  flow once Phase 4/5 are built
- **Run Data Pulls** — buttons to run each pull script and see its output

**What this doesn't do yet**: place bets. That's Phase 5 (Claude in Chrome, home only,
human-approved each time) — the Parlay Builder gets you to a clean slip to place manually
in the meantime.

## Structure

```
data/
  raw/          # untouched pulls from odds API / nflverse
  processed/    # cleaned, feature-engineered datasets
models/         # model definitions, one file per model type
backtesting/    # backtest engine + results
bet_logs/       # logged bets: odds at bet time, closing odds, outcome, CLV
notebooks/      # exploratory analysis
dashboard/      # Streamlit UI — data browser + Parlay Builder (see Dashboard section below)
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
- **Underdog pick'em props** (`data/pull_underdog.py`) — current NFL player prop over/under
  lines from Underdog's internal (unofficial) API. **Untested here** — see the ToS caveat
  and testing notes in the script itself and the progress log below before relying on this.

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
