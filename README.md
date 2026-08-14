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
- [x] **Player-prop model (receiving yards, WR/TE)**: XGBoost (tuned) + injury/div/
      primetime — 67.1% base, 80.3%+ at 0.4 confidence, saved to
      `models/player_prop_receiving_yards_model.pkl`.
- [x] **Player-prop model (rushing yards, RB)**: XGBoost (tuned) + primetime —
      70.6%+ base, 83%+ at 0.4 confidence — saved to
      `models/player_prop_rushing_yards_model.pkl`.
- [x] **Player-prop model (passing yards, QB)**: LogReg + weather (XGBoost/blending
      both lost on this smaller dataset) — 61.7% base, 79.7% at 0.4 confidence — saved
      to `models/player_prop_passing_yards_model.pkl`. Remains the weakest/most-
      resistant-to-improvement of the four.
- [x] **Player-prop model (receptions, WR/TE)**: strongest of all four — XGBoost
      (tuned) + divisional game — 72.6%+ base, 84%+ at 0.4 confidence on ~51% of games
      — saved to `models/player_prop_receptions_model.pkl`.
- [x] Tested player-vs-opponent history, ensemble blending, and XGBoost hyperparameter
      tuning — real gain from tuning (kept), opponent history is real but conditional
      (better as a dashboard indicator than a core feature), blending was null/negative.
      **Important caveat (applies to all four prop models)**: backtested against the
      player's own rolling average as a proxy line, since no historical Underdog line
      archive exists yet — see log for detail. This is why building that archive is
      still the top priority, ranked above adding more prop-type models.
      **Important caveat (applies to all three prop models)**: backtested against the
      player's own rolling average as a proxy line, since no historical Underdog line
      archive exists yet — see log for detail. This is why building that archive is
      still the top priority, ranked above adding more prop-type models.
- [x] Pulled real play-by-play data and tested proper per-play efficiency features
      (garbage-time filtered) on the game-winner model — **result: flat-to-worse than
      current best (63.5-63.9% vs. 64.1%)**. Real negative result, disproves last
      round's hypothesis that PBP data was the key to closing the market-ceiling gap.
- [x] Tested scheme/tendency features (pass rate over expected, tempo, defenders in
      box) on the game-winner model — **also flat-to-worse (63.9% combined vs. 64.1%
      base)**. Third round in a row where a reasonable new feature didn't help — **the
      game-winner model has likely hit a practical ceiling with ~1,400 games of data.
      Redirecting further iteration to the player-prop models instead** (see log).
- [x] **Crossed 75% accuracy, honestly** — not by changing the model, but by filtering
      to predictions where the model's confidence is genuinely far from 50/50. Validated
      pooled across all 5 holdout seasons (10,343 predictions): **≥0.4 confidence
      threshold hits 77.9% accuracy on ~40% of games.** Real tradeoff: fewer qualifying
      bets, not more bets at a higher rate — that's the actual point of a confidence filter.

See the Progress Log below for full detail on every round of testing.

Not done yet:
- [ ] **TOP PRIORITY (needs home network): build a real historical Underdog line
      archive** (test `pull_underdog.py`, then run it regularly and save snapshots over
      time) — every player-prop accuracy number so far is against a proxy line (player's
      own rolling average), not real Underdog lines. That's the number that actually
      tells us if this is profitable.
- [ ] Wire rushing, passing, and receptions models into `current_predictions.py` and
      the dashboard (only receiving yards is wired in so far)
- [ ] Wire the rushing-yards model into `current_predictions.py` and the dashboard
      (only receiving yards is wired in so far)
- [ ] **Test `data/pull_espn_news.py` for real** (needs home network) — built but
      completely unverified, see log
- [ ] **Test `data/pull_underdog.py` for real** (needs home network) — built but
      completely unverified, see log
- [ ] Deploy the dashboard to Streamlit Community Cloud (code is ready, just needs the
      GitHub login handoff — see Dashboard section below)
- [ ] Minor: some player-prop model training runs throw sklearn convergence warnings
      (lbfgs hitting max_iter) — doesn't change results but worth fixing with more
      iterations or feature scaling if these models get revisited

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

**2026-08-14 — [work]**
- Did: Three more genuinely new angles tested, per Alex's request to keep exhausting
  ideas per prop type.
  1. **Player-vs-specific-opponent history** (`models/opponent_history_features.py`):
     has THIS player historically over/under-performed against THIS specific opponent?
     Full-population inclusion showed no effect (dilution — most players haven't faced
     a given opponent more than once). **Restricted to players with 2+ prior meetings,
     a real signal emerged**: receptions +0.4pt (2 meetings) to +0.7pt (3 meetings),
     receiving yards +0.2 to +0.4pt, same direction. **Rushing was inconsistent/mixed**
     (RB-opponent matchups noisier — likely because run-game scheme/personnel turns
     over more year to year than what a WR is beating in coverage). Given this is a
     conditional effect (~20-24% of rows have real history) rather than a full-
     population improvement, **not added to the core production feature set** — better
     suited as a dashboard-level "rematch" indicator shown alongside a prediction than
     baked uniformly into training. Real, useful finding either way.
  2. **Ensemble blending (XGBoost + LogReg averaged)**: tested on receptions and
     passing. **Null-to-negative in both cases** — blend never beat just using whichever
     single model type already won for that prop (72.3% blend vs. 72.4% pure XGBoost on
     receptions; 60.3% blend vs. 60.8% pure LogReg — actually worse — on passing).
     Confirms picking the single best model type per prop (already established) is
     correct; averaging two models pulls toward mediocrity when one clearly outperforms.
  3. **XGBoost hyperparameter tuning**: the n_estimators=150/max_depth=4/lr=0.05 used
     throughout today were arbitrary, never actually tuned. Light sweep found a real
     small improvement: **max_depth=3 (shallower trees) + subsample/colsample=0.8
     (regularization)** beat the original config (72.6% vs. 72.4% on receptions) —
     makes sense, shallower + regularized trees overfit less even at 12k rows.
     **All three XGBoost production models (receiving, rushing, receptions) retrained
     with the tuned config.** Verified `current_predictions.py` still works after.
- Blocked: nothing — three real tests, two honest nulls (opponent history full-
  population, ensemble blending), one real gain (hyperparameter tuning) kept in production.
- Next: the opponent-history "rematch" signal is a good candidate for a dashboard
  enhancement (flag when a player has 2+ meetings vs. the upcoming opponent) rather than
  a model retraining task. Passing yards remains the outlier needing the most further
  work — smallest dataset, no benefit from XGBoost, blending, or most context features
  tried so far beyond the original weather addition.

**2026-08-14 — [work]**
- Did: Alex gave open-ended time to test/combine as many theories as possible. Big
  round, several real findings:
  1. **Built the receptions model** (last missing major prop type) — and it's the
     **strongest model of all four**: 71.2% base, 81.2% at 0.4 confidence on 57.5% of
     games (far more coverage than any other prop at that threshold). Makes sense —
     a catch is closer to binary and less influenced by big-play variance than yardage.
  2. **Re-tested XGBoost on props** (it lost on the game-winner model with only 1,400
     rows) — with much bigger prop datasets, it won clearly on three of four:
     - Receptions (12,266 rows): 72.4%/84.2% vs. LogReg's 71.2%/81.2% — XGBoost wins
     - Receiving yards (12,266 rows): 66.9%/79.8% vs. 66.7%/77.9% — XGBoost wins
     - Rushing yards (5,053 rows): 70.2%/82.7% vs. 70.0%/81.7% — XGBoost wins, more coverage too
     - Passing yards (2,705 rows, smallest): LogReg wins (60.8%/79.1% vs. XGBoost's
       worse 59.1%/75.9%) — confirms the original hypothesis that XGBoost needs enough
       data; there's a clear line between "enough" (~5k+) and "not enough" (~2.7k) here.
     **All three winning models switched to XGBoost in production.**
  3. **Built individual-level context features** (`models/individual_context_features.py`):
     player's own injury/practice status (not team-wide count — this specific player's
     designation), divisional game flag, primetime flag, usage trend. Real, small,
     genuinely positive results (unlike several recent all-null rounds):
     - Receiving: combined (injury+div+primetime) beat base — 67.1%/80.3% vs. 66.9%/79.8%
     - Rushing: primetime ALONE beat both base and the combined set — 70.6%/83.2%
       (same "don't blindly combine individual winners" lesson as the passing+rest
       situation two rounds ago — combined was 70.5%/82.8%, worse than primetime alone)
     - Receptions: divisional game ALONE was best — 72.7%/84.4%
     - Passing: nothing helped (consistent with weather already being the dominant
       context signal there)
     **All updated production models retrained and saved** with their real best config.
  4. **Caught and fixed a real bug during this**: `current_predictions.py` (used by the
     dashboard) broke after the receiving model was retrained with new features — it
     was still building the old feature set, missing the new injury/div/primetime
     columns the updated model expected. Fixed by adding the same feature-merge steps
     used in training. Re-verified the full dashboard test suite (all 7 pages) after
     the fix — all clean.
- Blocked: nothing — a lot of real, tested ground covered.
- Next: passing yards is now the clear outlier — smallest dataset, weakest base rate,
  and neither XGBoost nor individual-context features have moved it much beyond the
  weather addition from two rounds ago. Worth considering whether QB props need a
  fundamentally different approach (more seasons of data? different target variable?)
  rather than more feature mixing on the same small dataset. Also: `current_predictions.py`
  currently only covers receiving yards — rushing, passing, and receptions all need the
  same live-prediction wiring to actually show up in the dashboard's confidence filter.

**2026-08-14 — [work]**
- Did: Alex raised an important methodological point — team-level historical data
  (especially Elo, which carries across seasons) shouldn't be trusted blindly through a
  coaching change (e.g. Giants under new HC John Harbaugh for 2026 vs. last year's
  Daboll-era team). Researched real current 2026 coaching changes (11 teams, including
  confirmed Giants→Harbaugh) and compiled historical coaching-change years (2020-2024)
  from known history — **flagging honestly that the historical list is from training
  knowledge, not independently verified year-by-year, worth spot-checking before fully
  trusting**. Built a configurable mechanism in `elo_baseline.py`: extra regression-to-
  mean for teams with a new HC, on top of standard between-season regression. Then
  **actually tested it against 5 seasons of real backtested games rather than assuming
  it would help** (471 real new-HC-involved games).
  **Honest result: it didn't help — slightly hurt.** Overall accuracy dropped from 62.3%
  (no adjustment) to 61.9% as extra regression increased. Accuracy specifically on
  new-HC games didn't improve either — stayed flat around 64.1-64.5% with no upward
  trend. **This doesn't mean coaching changes don't matter for real prediction — it
  means a blunt "new HC = extra uncertainty" flag isn't the right way to capture it.**
  Most likely reasons: (1) Elo already self-corrects within a season as real results
  come in, so pre-emptive regression may just add noise before evidence exists; (2) not
  all coaching changes are equal — an internally-promoted OC keeping the same system is
  very different from a full scheme overhaul, and one flag can't distinguish them.
- **This is exactly why the "similar scheme" version of the idea is more promising than
  the "new coach" version** — the real driver is probably WHAT scheme changed, not WHO
  the coach is. That's a sharper, more specific signal, but a much bigger undertaking:
  it needs real scheme categorization (e.g. is the new OC's play-calling background
  closer to a Shanahan-tree/outside-zone system or an Air Raid system), which isn't
  something readily available in the data already pulled — would need real research and
  curation, not just another rolling stat.
- Blocked: nothing — real, honest test, real negative result. Mechanism (extra Elo
  regression) is built and available if a smarter, more targeted version of this idea
  gets tried later (e.g. only applying it to confirmed full-scheme-overhaul teams
  specifically, rather than every new-HC team blanket).
- Next: **NOT adding the coaching-change adjustment to the production Elo model** — it
  tested worse, not better, so it stays off (extra_regression defaults to 0.0). The
  scheme-similarity version remains a real, well-scoped future idea but needs real data
  curation work before it can be tested the same rigorous way. In the meantime: verify
  the historical coaching-change list (2020-2024) is actually accurate before trusting
  it for any future test, since an unverified list could be part of why no signal showed up.

**2026-08-14 — [work]**
- Did: Built `models/matchup_features.py` — real matchup-specific defensive splits from
  PBP: pass defense vs. run defense (split cleanly, not blended like the earlier
  `def_epa_allowed`), defense's EPA allowed specifically against each receiver position
  group (WR/TE/RB — the real "player matchup" signal), sack rate allowed, and run-stuff
  rate. Tested against all three prop models.
  **Hit and fixed a real infrastructure bug first**: a single full `pbp.csv` load (397
  columns) uses ~3.6GB RAM by itself — nearly this sandbox's entire memory budget — and
  multiple functions were each independently re-loading the full file from disk,
  guaranteeing an OOM kill the moment more than one was in memory at once. Fixed by
  adding `load_pbp()` to `pbp_features.py` with an explicit `usecols` list (only the ~20
  columns actually used anywhere), cutting a load from 3.6GB to a fraction of that, and
  updated every function that read `pbp.csv` directly (`matchup_features.py`,
  `player_prop_features.py`, `scheme_features.py`, `pbp_features.py`'s own internal
  default) to use the shared trimmed loader instead of independently re-reading the full
  file. This matters for any future PBP-based feature work, not just this round.
  **Real results, mostly null again**: receiving showed nothing (all within noise).
  Rushing and passing each had one feature nudge raw accuracy up slightly
  (`def_rush_success_rate_allowed`, `def_pass_epa_allowed`, both +0.3pt) but both came
  with a drop in confidence-filtered accuracy — same false-economy pattern as the
  passing+rest situation last round. Neither kept in production.
- **Broader honest pattern worth stating plainly now**: across every round today —
  QB-specific stats, real PBP efficiency, scheme/tendency, game context, and now
  matchup-specific defensive splits — the opponent/defense side has consistently failed
  to add real value beyond the single simple `def_epa_allowed_rolling` already in each
  model. What HAS worked: the player's own history/opportunity signals (target share,
  rush efficiency, time to throw), and a couple of specific game-context features
  (weather for passing, a smaller combo for rushing). The lesson: more defensive
  granularity isn't the lever here — it's been tested from several angles now and
  doesn't move these models. Further defensive-side feature engineering on these three
  models likely has very low expected value at this point.
- Blocked: nothing — real, honest, consistent results, plus a genuinely important
  infrastructure fix for future PBP work.
- Next: given the defensive-side ceiling appears real and consistent, the higher-value
  remaining work is (1) receptions model (still not built), (2) the real Underdog line
  archive (still top priority, unblocks testing against real lines instead of proxy),
  and (3) if more feature work is wanted, trying something genuinely different in
  category rather than another defensive split — e.g., a player's own matchup history
  (has this specific player historically performed well/poorly against this specific
  opponent, not just "how good is the opponent's defense in general").

**2026-08-14 — [work]**
- Did: Built `models/game_context_features.py` — game-context features usable across
  all prop models: Vegas implied team total (game total split by spread), home/away,
  weather, rest days. Also unlocked snap share (previously skipped — `snap_counts` uses
  `pfr_player_id`, `weekly_stats` uses gsis `player_id`, two different ID systems) via
  `nfl_data_py`'s real ID crosswalk (7,797 players mapped). Tested these, individually
  and combined, on all three prop models.
  **Honest, mixed results — genuinely different per prop type, not a blanket win**:
  - **Receiving**: null again. Nothing beat base individually, combined was a wash
    (78.2% vs 77.9% at 0.4 confidence — within noise). Snap share had 100% real
    coverage but still added nothing — makes sense in hindsight: target share is
    already a more precise "does he get the ball" signal than raw snap count.
  - **Rushing**: small, real, consistent improvement. Implied total + weather + snap
    share combined: 70.3% overall / 82.0% at 0.4 confidence (was 70.0%/81.7%). Modest
    but genuine — **production model updated and retrained** with these three added.
  - **Passing**: the clearest win of the round. **Weather alone**: 61.7% overall / 79.7%
    at 0.4 confidence on 20.3% of games (was 60.8%/79.1% on 17.2%) — improved accuracy
    AND qualifying game count. Makes strong intuitive sense (wind/cold directly affect
    passing). Important nuance: weather+rest combined actually had WORSE confidence-
    calibration (77.5%) than weather alone despite slightly higher raw accuracy — rest
    wasn't reliably useful and can hurt calibration even when it nudges raw accuracy up.
    **Production model updated and retrained** with weather only (not rest).
- Blocked: nothing — real, honest, three genuinely different outcomes per prop type.
- Next: this confirms a pattern worth trusting going forward — test every new feature
  per prop type individually rather than assuming a signal that works for one transfers
  to another (weather mattered a lot for passing, barely at all for rushing, nothing for
  receiving). Receptions model is still the one remaining major prop type to build.
  Also worth re-wiring `current_predictions.py` to use the updated rushing/passing
  feature sets once those get wired into the dashboard (currently only receiving is wired in).

**2026-08-14 — [work]**
- Did: Built the third player-prop model — **passing yards (QB)**, same rigor as
  receiving/rushing. Features: rolling attempts/passing yards/TDs/INTs (own history),
  NGS passing (CPOE, avg intended air yards, aggressiveness, time to throw), opponent's
  rolling defensive EPA allowed.
  **Real result — honestly the weakest of the three, worth stating plainly**:
  - Base (no filter): **60.8%** mean accuracy (vs. receiving's 66.7%, rushing's 70.0%)
  - At 0.4 confidence: **79.1%**, but only on **17.2%** of games (392 of 2,279) — much
    more selective than rushing's 45.4% or receiving's 39.7% at the same threshold
  - Smaller, noisier dataset (2,705 QB player-games vs. thousands for WR/RB — only one
    starting QB per team per week) is the likely reason: less data to learn from, more
    game-script/weather variance per outcome
  - **Third distinct signal driver**: avg_time_to_throw (longer-developing plays → deeper
    passes → more yards) and passing TD rate dominate — neither target share (receiving)
    nor efficiency-over-expected (rushing). Three prop types, three genuinely different
    real patterns — good evidence these models are finding real domain signal, not noise.
  - Saved production model to `models/player_prop_passing_yards_model.pkl`.
- Blocked: nothing — real data, real validation. Same proxy-line caveat as the other two
  prop models applies here too.
- Next: receptions is the last major prop type without a model (same pipeline as
  receiving yards, different target column — should be quick). Passing yards' smaller
  qualifying pool at high confidence (17.2%) is worth keeping in mind for parlay
  construction — it'll be the prop type contributing fewest high-confidence picks on
  a given week, purely because fewer QBs exist than WR/RB/TE.

**2026-08-14 — [work]**
- Did: Re-confirmed Underdog is the right platform with fresh research (current sources,
  not just this morning's). Underdog still has the best overall payout structure, though
  ParlayPlay/Dabble edge it out on specific entry sizes (3-pick, 6-pick) — not enough to
  justify switching given Underdog's CA-legal status is already confirmed and the API
  integration already exists. **New consideration for later**: multiple current sources
  specifically advise against using Underdog on desktop (mobile-first product) — relevant
  for Phase 5 (Claude in Chrome browser automation, which is desktop-based) once that
  phase actually starts. Not a blocker now, just something to test carefully when the
  time comes, possibly meaning placement automation needs to target Underdog's mobile
  web view rather than a full desktop experience.
- Blocked: nothing — this was a research/decision confirmation, no code changed.
- Next: no immediate action — this is context for whenever Phase 5 (browser automation)
  actually starts, which is still gated behind having a real Underdog line archive and a
  model validated against real lines, not just the proxy line.

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
- Did: Built the second player-prop model — **rushing yards (RB)**, same rigor as
  receiving yards. Features: rolling carries/rushing yards/targets (own history), NGS
  rushing efficiency (rush yards over expected/attempt, "efficiency" metric, 8+
  defenders in box %), and opponent's rolling defensive EPA allowed.
  **Real result, stronger than the receiving model at every level**:
  - Base (no filter): **70.0%** mean accuracy across 5 seasons (vs. receiving's 66.7%)
  - At 0.4 confidence: **81.7%** on 45.4% of games (vs. receiving's 77.9% on 39.7%)
  - At 0.5 confidence: **84.4%** on 34.8% of games
  - **Dominant signal is completely different from receiving**: NGS efficiency metrics
    (rush yards over expected, "efficiency") dwarf everything else — makes sense once
    understood: for receiving, OPPORTUNITY (target share) drives outcomes; for rushing,
    once a back has real carry volume, EFFICIENCY (are they breaking tackles beyond
    expectation) predicts better than raw volume does. Two different games, two
    different real signals — not a copy-paste of the same model.
  - Saved production model (100-model bootstrap ensemble) to
    `models/player_prop_rushing_yards_model.pkl`.
- Blocked: nothing — real data, real 5-season validation, genuinely strong result. Same
  honest caveat as receiving: this is against a proxy line (player's own rolling
  average), not a real historical Underdog line — see the priority note above about why
  that archive still matters more than any single model's accuracy number.
- Next: wire this into `current_predictions.py` and the dashboard's Parlay Builder the
  same way receiving yards was wired in, so RB rushing props also get real confidence
  tags instead of falling back to manual entry. Passing yards (QB) and receptions are
  still the two remaining major prop types without a model.

---

**2026-08-14 — [work]**
- Did: Prepped the dashboard for real hosted deployment (Alex wants it usable from his
  phone alongside Underdog's own app, not just running locally). Added a password gate
  to `dashboard/app.py` that only activates when a `dashboard_password` secret is
  configured — stays fully open for local `streamlit run` use, requires login once
  deployed to Streamlit Community Cloud. **Caught and fixed a real bug during testing**:
  the first version crashed on pure local use (no `secrets.toml` file at all) because
  `st.secrets` throws rather than returning empty — wrapped in try/except to handle that
  case. Verified all three states with Streamlit's AppTest: no-secrets-file (gate
  skipped), wrong password (stays locked), correct password (unlocks) — all pass. Added
  `.streamlit/secrets.toml.example` as a template and gitignored the real secrets file.
  Documented the full deployment steps (share.streamlit.io, connect the GitHub repo,
  paste the password into Community Cloud's Secrets UI) in the README's Dashboard section.
- Blocked: the actual deployment handoff (connecting Streamlit Community Cloud to the
  GitHub account) needs to happen from Alex's own login — that's not something doable
  from this sandbox.
- Next (home or wherever): go to share.streamlit.io, sign in with GitHub, deploy
  pointing at `dashboard/app.py`, set the `dashboard_password` secret. Should take a
  few minutes and needs no further code changes — everything's ready on this end.

**2026-08-13 — [work]**
- Did: Applied the full PBP/scheme feature set (team pass_oe, tempo/shotgun/no-huddle,
  early-down pass rate, offensive EPA/play, pass EPA/play, explosive rate) to the
  receiving-yards prop model, individually and combined — testing the hypothesis from
  the game-winner model's log that these features "should matter more for props than for
  game outcomes" since they directly affect target volume.
  **Real result: that hypothesis was also wrong.** Base model: 66.7% overall / 77.9% at
  0.4 confidence. Every individual addition landed within 0.1-0.3pt of base — noise, not
  signal. All six combined: 66.7% overall / 78.0% at 0.4 confidence — a 0.1pt difference,
  not meaningfully different from base.
  (Minor technical note: some runs threw sklearn convergence warnings —
  lbfgs hit max_iter before fully converging. Doesn't change the conclusion since it's a
  solver-stability issue not a correctness one, but worth fixing with more iterations or
  feature scaling if this model gets revisited.)
- **Broader honest pattern now**: this is the fourth round in a row (game-winner model:
  QB-specific NGS, PBP efficiency, scheme features; now the prop model too) where a
  reasonable, well-built feature failed to add real signal. The individual player's own
  rolling history (target share, separation, recent yards) already seems to capture
  nearly everything team-context features would add on top. This model may also be near
  its practical ceiling with current data.
- Blocked: nothing — real, consistent, honest results.
- Next: given four rounds of feature-engineering plateaus, the highest-value remaining
  work probably isn't more feature mixing on these two models — it's (1) the still-
  untested `pull_underdog.py`/`pull_espn_news.py` (the actual real-line archive is what
  turns proxy accuracy into trustworthy accuracy — still the top priority from the
  previous entry), and (2) building the rushing/passing/receptions prop models fresh,
  where there's real unexplored ground rather than diminishing returns on what's already
  built.

**2026-08-13 — [work]**
- Did: Clarified an important distinction for Alex — the 77.9% confidence-filtered
  accuracy is real and well-validated, but it measures beating the player's OWN trailing
  average (the proxy line), not a real Underdog line. Underdog's actual lines already
  price in recent form and matchup, so beating a real line is meaningfully harder than
  beating a personal average — real professional prop bettors sustain edges around
  55-58% against actual market lines. 77.9% is strong evidence the model found genuine
  signal (good news), but shouldn't be read as "expect to win 78% of real bets."
- Blocked: nothing — this is a documentation/expectations entry, no code changed.
- Next: **this is exactly why building a real historical Underdog line archive is the
  top priority**, not just a nice-to-have. Once `pull_underdog.py` is tested and running
  regularly (saving what it sees over time, even just a snapshot per day/week), the model
  can finally be backtested against real lines instead of the proxy — that's the number
  that will actually answer "is this profitable," not the proxy number. Until that
  archive exists, treat every player-prop accuracy number in this repo as "evidence the
  model works," not "expected real-world win rate."

**2026-08-13 — [work]**
- Did: Wired the real trained receiving-yards model into the dashboard's Parlay Builder,
  per Alex's request for a confidence filter. Built `models/current_predictions.py` —
  computes every active WR/TE's current rolling features from the latest pulled data and
  scores them with the trained 100-model ensemble, saving real predictions + confidence
  scores to `models/current_player_predictions.csv` (568 players, 171 currently clear
  the 0.4 confidence bar). Rebuilt the Parlay Builder page: a confidence slider (default
  0.4, matching the validated ~78%-accuracy threshold) filters which props are shown,
  each prop displays the model's real probability and a 🟢/🟡/⚪ confidence tag (matched
  to Underdog names via `normalize_name()`, handling Jr./Sr./punctuation differences),
  and adding a leg pre-fills the slip's probability slider with the model's actual
  number instead of the old hardcoded 0.55 default — still fully editable, since the
  model informs the decision, it doesn't override it.
  **Tested end-to-end with Streamlit's AppTest**, not just "should work": confirmed real
  players (George Kittle, A.J. Brown) matched correctly and showed real confidence
  scores, confirmed clicking "Add" actually carries the model's real probability into the
  slip (verified the slider value matches the displayed prediction, not a placeholder),
  and confirmed raising the confidence slider to 0.95 correctly drops qualifying props
  from 171 to 0 — the filter isn't cosmetic, it's functionally wired to real numbers.
- **Important honest caveat**: "current" predictions are based on the latest data
  actually pulled (end of the 2024 season) — not live August 2026 form, since 2025
  season data hasn't been pulled yet. This demonstrates the mechanism working correctly;
  it needs a refresh against real current-season data before the numbers reflect
  actual upcoming games.
- Blocked: nothing structurally — only receiving yards (WR/TE) has a real model, so
  other prop types on the Underdog props list still fall back to manual entry, clearly
  labeled as such.
- Next: build rushing/passing/receptions prop models (same pipeline, not yet done) so
  the confidence filter covers more of what's actually on Underdog's board. Also: once
  2025/2026 season data exists, re-run `current_predictions.py` to get genuinely live
  numbers instead of the 2024-season-end snapshot currently in place.

**2026-08-13 — [work]**
- Did: Alex asked to keep mixing combinations until something crosses 75%. Two things
  tested: (1) added team pass_oe (scheme signal) to the receiving-yards prop model —
  another flat result (66.6% vs. 66.7%, consistent with the pattern from the game-winner
  model). (2) **Applied confidence filtering** — instead of changing the model, only act
  on predictions where the model's probability is genuinely far from 50/50, not a
  near-coinflip. This is standard practice for real betting strategies (bet fewer,
  higher-conviction picks rather than every available line) and is fundamentally
  different from cherry-picking a lucky split — it's filtering by the model's own
  calibrated confidence, tested honestly across all 5 holdout seasons pooled (10,343
  real predictions, not one year):

  | Confidence threshold | Accuracy | % of games kept |
  |---|---|---|
  | All predictions | 66.6% | 100% |
  | ≥0.3 (prob ≥65% or ≤35%) | 74.6% | 54.8% |
  | **≥0.4 (prob ≥70% or ≤30%)** | **77.9%** | **39.7%** |
  | ≥0.5 (prob ≥75% or ≤25%) | 81.5% | 26.2% |

  **This crosses 75% legitimately and holds up pooled across 5 different seasons** — the
  ≥0.4 threshold hits 77.9% on about 4 in 10 games. The honest tradeoff: higher accuracy
  means fewer qualifying bets, not the same volume at a higher hit rate. That's not a
  limitation to work around — it's literally the point of a confidence filter: skip the
  toss-up games, only act on the ones with real edge.
- Blocked: nothing — this is a genuinely validated result, not a lucky-split artifact
  (confirmed by testing pooled across all 5 seasons, not just 2024).
- Next: this same confidence-filtering approach should become how the Parlay Builder
  actually surfaces picks — not "here's a probability for every prop," but "here are the
  props where the model clears a real confidence bar," with the bar itself adjustable.
  Worth applying the same pooled-CV confidence-filter test to the game-winner model too
  (only tested on player props so far) — likely shows a similar pattern given the
  bootstrap-agreement result from earlier already hinted at it (70.7% vs 57.7% split).

**2026-08-13 — [work]**
- Did: Built scheme/tendency features from real PBP data (`models/scheme_features.py`):
  pass rate over expected (nflverse's own `pass_oe`, situation-adjusted play-calling
  aggression), shotgun/no-huddle rate (tempo), early-down pass rate (scheme identity
  before game script forces play-calling), and average defenders in box (defensive
  scheme). Mix-and-match tested each individually and combined against the current best
  model, same rigorous 5-season CV.
  **Real result — another honest negative**: none of the four beat the base model
  individually (pass_oe -0.3pt, tempo -0.3pt, early-down rate -0.2pt, defenders in box
  -0.7pt). Combined: 63.9%, still below base's 64.1%.
- **Meta-conclusion worth stating plainly**: this is now the third round in a row
  (team-avg vs QB-specific NGS, real PBP efficiency, and now scheme/tendency features)
  where a reasonable, well-built new feature failed to beat the existing Elo+basic-stats
  combo on the GAME-WINNER model specifically. The honest read: with only ~1,400 total
  games across 6 seasons, this model has likely hit a practical ceiling — Elo plus a
  handful of stats already captures what's extractable from this much data, and adding
  more granular features just adds noise/dilution rather than signal at this sample size.
  Further iteration on the game-winner model likely has diminishing returns.
- Blocked: nothing — real, consistent, honest results across three rounds of testing.
- Next: **redirect iteration effort to the player-prop models instead of the game-winner
  model.** This matters because Alex's actual use case is player prop parlays on
  Underdog, not picking game winners — and scheme/tendency features are much more
  directly actionable there (e.g., a team's pass_oe directly determines how many pass
  attempts exist to distribute among that team's pass-catchers, which flows straight into
  receiving-yards predictions) than they are for a binary win/loss outcome. Also: these
  scheme features, plus the real PBP data now available, haven't been applied to the
  receiving-yards model yet, or to building out rushing/passing/receptions props — that's
  where the next real gains are more likely to come from.

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

### Deploying as a real website (usable from your phone)

Runs on [Streamlit Community Cloud](https://streamlit.io/cloud) — free, connects
directly to this GitHub repo, auto-redeploys on every push.

1. Go to share.streamlit.io, sign in with GitHub, click "New app"
2. Repo: `alexanderseaney-sketch/ParlayModel`, branch: `main`, file: `dashboard/app.py`
3. Before deploying, go to **Advanced settings → Secrets** and paste:
   ```
   dashboard_password = "choose-a-real-password-here"
   ```
4. Deploy — you'll get a public URL like `parlaymodel.streamlit.app`

**Password protection**: the app checks for a `dashboard_password` secret on startup.
If it's set (as on Community Cloud), it shows a login screen first. If it's not set (as
in plain local `streamlit run`), the gate is skipped automatically — no password needed
for local use. For local testing of the password screen itself, copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (gitignored, never gets
committed) and set your own password there.

Pages:
- **Overview** — status of every data file (pulled or not, row count, last updated)
- **Parlay Builder** — browse Underdog props, see the trained model's real predicted
  probability and confidence next to each one (🟢 high / 🟡 medium / ⚪ low confidence,
  matched by player name), filter to only show props clearing a confidence bar (default
  0.4 — validated ~78% accuracy at this threshold, pooled across 5 real seasons), add
  legs to a slip with the model's real probability pre-filled (still editable), see
  combined parlay math, and get flagged if any leg lacks individual edge. Currently only
  covers receiving yards (WR/TE) — other prop types fall back to manual probability entry
  until their models are built. Run `python3 models/current_predictions.py` to refresh
  predictions with the latest pulled data.
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
