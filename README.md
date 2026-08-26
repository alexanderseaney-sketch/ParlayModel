# ParlayModel

An AI-assisted NFL betting agent: pulls odds + stats, runs prediction models through backtesting,
and (eventually) drives an approval-gated bet-placement flow via browser automation.

## Status

**Currently on: Phase 2/3 — 29 player-prop models + game-winner model, all trained on 2014/2016-2024 (varies by feature requirements), real data pipeline now committed to the repo. See the 2026-08-26 log entry for a full catch-up on everything the Status list below doesn't yet reflect — that entry is more current than the bullets underneath it.**

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
- [x] **Underdog line pull confirmed working from home** (was blocked from the work
      sandbox) — `pull_underdog.py` now saves timestamped snapshots to
      `data/raw/underdog_history/` in addition to the latest-pull file, and is
      scheduled to run daily so the archive actually accumulates. See log for the real
      preseason schema finding (season-long lines dominate right now; no `receptions`
      single-game line live yet).
- [x] **New: SB Nation team-news puller** (`data/pull_sbnation_news.py`) — ESPN's news
      API turned out to be blocked by ESPN's own Akamai bot-protection (403, confirmed
      via curl — a different failure mode than the sandbox network restriction, and not
      something to route around). Built a replacement pulling all 32 teams' SB Nation
      blogs directly via their public Atom feeds, verified live against
      sbnation.com/nfl's own team-site directory rather than guessed. Also scheduled
      daily, accumulating into `data/raw/sbnation_news.csv` deduped by article link.
- [x] **All four prop models now wired into `current_predictions.py` and the
      dashboard** — was receiving-yards-only. Rewrote it generically: rather than
      re-declaring each model's feature list by hand, it trusts the `features` list
      saved inside each model's own `.pkl` (inspected directly to confirm) and merges
      only the shared context features (injury status, div/primetime flags, weather/
      Vegas total, snap share) each specific model actually needs. Also fixed a real
      latent bug this surfaced: the dashboard's Parlay Builder matched predictions to
      Underdog props by **player name only** — harmless with one prop type, but wrong
      with four (would've silently attached e.g. a rushing-yards prediction to that
      same player's receiving-yards prop). Now matches on name **and** stat type.
      Verified directly against live data: 78 of 1024 live Underdog prop-option rows
      matched a model prediction, all correctly stat-scoped (Bijan Robinson only
      matched on rushing_yds, not receiving_yds, etc). Also added an "SB Nation News"
      dashboard page for the puller built earlier this session — it existed but was
      never wired into the UI.
- [x] **Independently re-validated all four prop models with a real walk-forward
      backtest**: retrained each one from scratch using ONLY 2019-2023 data, then
      scored all of 2024 blind (`models/backtest_holdout_2024.py`). Landed within 1-3
      points of every previously reported number (e.g. receiving 66.9% vs. 67.1%
      reported, receptions 72.3% vs. 72.6%) — real evidence the models generalize and
      aren't overfit to the original validation process, not just a re-read of old
      numbers. See log for full comparison table.
- [x] **New theories round: ~20 more feature combinations tested (usage trend, plus
      filling real gaps for receptions/rushing/passing) — clean sweep of honest nulls.**
      Now 6/6 props where opponent/context-side features have failed to beat what's
      deployed. Real signal this sends: feature-engineering-on-the-current-data-source
      looks close to tapped out; the Underdog line archive (already top priority) is
      the more promising remaining lever since it changes the target, not just the
      features. See log for the full breakdown.
- [x] **Added a second news source**: `data/pull_nbcsports_news.py` (NBC Sports/PFT
      rumor mill) — fast, atomic insider roster/injury blurbs, complementing SB
      Nation's longer-form team recaps rather than replacing them. Checked technical
      feasibility for real (ESPN still blocked, PFT's domain moved after a site
      consolidation, one candidate feed turned out stale despite returning HTTP 200)
      before building. Wired into the dashboard and the daily schedule.
- [x] **Fixed two real gaps ahead of the regular season**: predictions used to silently
      go stale with no way to tell (a player with no current-season games fell back to
      last season's finale, treated as if current) and matched game context (injury/
      weather/div-game) to the wrong game (whatever the rolling stats were dated to,
      not the actual upcoming game). Both fixed — `current_predictions.py` now looks up
      each player's real next scheduled game and makes both dates explicit in the
      output. Also fixed `pull_nflverse.py` to merge instead of overwrite (root cause
      of an earlier near-miss this session), which is what makes it safe to now run
      automatically every day alongside predictions regeneration. See log for detail.
- [x] **Added a real depth-chart/injury source**: `data/pull_footballguys_depth.py` —
      all 32 teams, offense/defense/special-teams, structured per-player status tags
      (Q/PUP/IR/SUS/NFI/O). Checked @rclfootball first (Alex's tip) but it's Instagram/
      Threads with no public feed and login-gated content — real dead end, not
      pursued. Evaluated 6 dedicated depth-chart sites instead; most were paywalled or
      had dead links, Footballguys was the one free + comprehensive + fresh option.
      Wired into the dashboard as a browse page AND as a live 🚑 warning tag directly
      in the Parlay Builder next to matching props. Not fed into the trained models
      yet — no historical archive exists for it to validate against.
- [x] **Tested whether models should actually retrain week-by-week as a season
      unfolds** (recency-weighted samples, not just a static one-time fit), per
      Alex's question about seasons having their own trends. Simulated real weekly
      retraining through all of 2024, two decay rates, compared directly against the
      static model on identical test rows. **Null-to-negative both times** — real,
      informative result: the player-level rolling features already reset/adapt every
      season, so the model doesn't need to be retrained to reflect "what this season
      looks like." Not deployed. See log for the full comparison and why this happens.
- [x] **Tested three more genuinely new angles — player consistency/volatility, QB
      continuity, explicit interaction terms — deliberately in a different category
      than the (repeatedly null) context/defense-side features.** Clean sweep of
      nulls again, including a surprising one: QB continuity was completely flat
      despite 12.8% of rows having a real QB change, real football logic notwithstanding.
      **This is now the 7th consecutive round of testing without a real win** across
      dozens of feature/architecture combinations. See log for the honest synthesis of
      what this pattern actually means going forward.
- [x] **Found a real, previously-unexamined gap: the Parlay Builder's combined-
      probability math has always silently assumed every leg is independent** (naive
      `p1 * p2 * ... * pn`). Measured REAL correlation using 2019-2024 data
      (`models/analyze_parlay_correlations.py`) instead of guessing. Found genuine,
      large effects for several same-team pairings — most notably QB passing yards
      correlates POSITIVELY with WR/TE receiving props (phi up to +0.23) and
      NEGATIVELY with RB rushing yards (phi -0.08, pass-heavy vs. run-heavy game
      script trading off) — while most other same-team pairings really are close to
      independent, confirming the old math was fine there. Built a correlation-aware
      combined-probability calculator and wired it into the Parlay Builder: it now
      shows both the naive and the corrected probability whenever a slip contains a
      known-correlated same-team pair, with the specific measured phi shown for
      transparency. This is a different kind of improvement than the 7 null rounds
      above — not another feature trying to nudge single-leg accuracy, but fixing a
      real mathematical error in how legs get COMBINED, which is literally what
      "predicting parlays" (Alex's actual ask) means. See log for the math and how it
      was verified.
- [x] **Found and fixed two real, serious bugs while building a weekly bet-slip
      generator** (Alex wanted suggested bets + sizing; Alex places every bet
      themselves, this project never executes real-money transactions). Bug 1: matched
      predictions to props by player+stat only, without checking which side (over/
      under) the model actually favored -- a confident UNDER call was being shown/used
      as if it endorsed the OVER row. Bug 2, more serious: `predicted_prob_over`
      answers "beats OUR proxy line" (the player's own rolling average), not "beats
      Underdog's actual posted line" -- those are only the same question when the two
      numbers are close, and they often aren't (one real example found: a 4.1 proxy
      compared against a real 65.5 line). Because a big mismatch produces an
      artificially huge-looking "edge," ranking by edge without catching this actively
      SURFACED the worst mismatches as the "best opportunities" -- the exact opposite
      of what should happen. Both bugs existed in the dashboard's Parlay Builder too
      (built earlier this session), not just the new script -- fixed in both places,
      with `MAX_LINE_DIVERGENCE` centralized in `dashboard/utils.py` so the two don't
      drift. Verified against live data: ~74% of matched props are naturally within
      20% of each other, so this excludes genuinely incomparable cases, not most of
      the board. See log for the full before/after numbers.
- [x] **Redesigned the dashboard for actually managing parlays**, per Alex's request
      to make it more polished and user-friendly. Real navigation now (icon-grouped
      sidebar via `st.navigation`/`st.Page` instead of one flat 11-item radio list),
      badges instead of emoji-in-text-strings, and — the actual functional gap this
      closed — Weekly Bet Slip suggestions can now be sent directly into the Parlay
      Builder's slip with one click instead of being two disconnected tools. Bet Log
      also connected: a slip can be sent there as pending bets in one click, and
      results are now editable inline instead of needing a separate form. Caught two
      real bugs live in the browser while testing this, not just in code review: a
      Streamlit "magic" auto-display artifact from a bare ternary expression, and a
      caption regression where consolidating 3 near-identical news pages into one
      shared function leaked a raw internal URL into SB Nation's cards that was never
      shown before. See log for both.

See the Progress Log below for full detail on every round of testing.

Not done yet:
- [ ] **TOP PRIORITY: keep building the real historical Underdog line archive** —
      `pull_underdog.py` confirmed working, scheduled daily, appending timestamped
      snapshots to `data/raw/underdog_history/`. Started 2026-08-15 — needs weeks of
      accumulated snapshots, ideally through the regular season, before it's enough to
      replace the proxy-line backtesting approach. Every player-prop accuracy number
      so far is still against a proxy line (player's own rolling average), not real
      Underdog lines. That's the number that actually tells us if this is profitable.
      Nothing to build here right now — this one just needs time.
- [ ] **Validate the weekly bet-slip generator against real weeks**, not just one
      preseason snapshot. The 3 opportunities it found on 2026-08-15 (Jameson
      Williams, Alec Pierce, Romeo Doubs, all receiving-yards unders) are unproven --
      preseason lines may just be less efficient than regular-season ones, which would
      look identical to "the model has an edge" without actually being one. Watch
      real weeks as they happen.
- [ ] **nflverse behavior against a partial in-progress season is untested** —
      the daily task's nflverse step is written to tolerate this gracefully, but
      that's inferred, not confirmed against real data. Will resolve itself once
      Week 1 (2026-09-09) actually happens.
- [ ] **Receptions single-game Underdog lines aren't posted yet** (preseason only has
      season-long totals) — receptions is the strongest of the four models but
      currently has nothing to bet against. Recheck once the regular season starts.
- [ ] **2025 nflverse `weekly_stats` specifically still isn't published** at the source
      — now doubly confirmed 2026-08-17 (checked both the old per-year URL structure
      AND nflverse's new unified-file structure directly; neither has 2025 data).
      **Also fixed a real, separate problem while checking**: our puller was using the
      old per-year URL structure, which nflverse deprecated before the 2025 season in
      favor of one unified file — fixed to use the new URL, which doesn't unlock 2025
      data (it doesn't exist yet either way) but moves us onto the actively-maintained
      path so future seasons appear automatically once published. Schedules/injuries/
      snap_counts/NGS for 2025 ARE genuinely live (unaffected by this). Since every
      prop model needs `weekly_stats` to build a player-week row, this one file is what's
      actually keeping "current" predictions stuck at 2024. External, out of this
      project's control — recheck periodically (a direct parquet read checking just the
      `season` column is enough, no full pull needed) rather than assuming it'll never update.
- [ ] **Depth-chart data (Footballguys) isn't fed into the trained models yet** — no
      historical archive exists to validate it as a feature against. The daily pull
      is already quietly building that archive; revisit once there's enough of it.
- [ ] **Parlay correlation math only covers same-team pairs** — whether a shootout
      lifts props on BOTH teams (cross-team correlation) is untested. Also: the
      pairwise-multiplication approach only stays valid for 2-leg parlays (the
      one-factor copula fix used in the bankroll simulator would need to move into
      the actual dashboard tool if 3+-leg correlated slips turn out to matter).
- [ ] **Operational, not a build task**: the deployed dashboard's live data (odds,
      news, depth charts) doesn't auto-refresh from the local daily pipeline --
      click through "Run Data Pulls" on the deployed site itself before checking it
      if it's been a while.
- [ ] Minor/cosmetic, low priority: an unused helper function may exist in
      `dashboard/utils.py` per an earlier note — a quick reference-count check today
      didn't turn up a clearly orphaned one (every function has 2+ references), but
      that's not the same as a careful manual read, so leaving this open rather than
      claiming it's resolved; saved model pickles throw harmless XGBoost/scikit-learn
      version-mismatch warnings on load.

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

**2026-08-26 — [work]**
- Did: Alex wants Dev Mode to support drawing, not just text, to communicate intent
  precisely. Honest technical reality first: Streamlit has no way to capture a
  screenshot of itself from Python — there's no browser access. The real flow is
  Alex takes his own screenshot (one keystroke on any OS) and uploads it, then draws
  directly on top of it with a real canvas.
  **Picked the wrong package first, caught it before shipping**: `streamlit-drawable-
  canvas` (the well-known one) imports fine but crashes the moment it actually renders
  — `AttributeError: module 'streamlit.elements.image' has no attribute
  'image_to_url'`. It hasn't been updated in years and calls an internal Streamlit
  function that no longer exists in the current version (1.61.1). Found this by
  actually rendering it through `AppTest`, not by trusting that `pip install`
  succeeding meant it worked. Searched for and found `streamlit-drawable-canvas-fix`,
  a maintained fork built specifically to fix this exact break — same import path
  (drop-in), verified it actually renders with a background image where the original
  crashed.
  **Built**: `_screenshot_annotation_widget()`, a shared helper (upload + canvas,
  scaled to the uploaded image's own aspect ratio capped at 700px wide) used by both
  Dev Mode entry points — the sidebar flow and the player-card-embedded one added last
  round. Kept deliberately OUTSIDE any `st.form` wrapping the note text/submit button,
  since custom components like this need to update on every stroke via their own
  render cycle, which forms suppress until submission. `save_dev_note()` now accepts
  an optional `annotated_image` (numpy array from the canvas) and saves it as a real
  PNG under `dev_notes/images/`, storing the relative path in the JSONL entry rather
  than inline — an annotated image is much bigger than everything else in a note, and
  JSONL is meant to stay one line per record. Dev Notes viewer now shows the saved
  image inline via `st.image()` when present.
  **Tested thoroughly given the real technical risk here**: confirmed the canvas
  renders without crashing via `AppTest` (drawing interaction itself — actual mouse
  strokes — can't be simulated from this sandbox, same limitation as the dialog-
  restore flow two rounds ago), and separately ran the full save round-trip with a
  real RGBA numpy array shaped like actual `canvas_result.image_data` output — saved
  to a real PNG, path recorded correctly, loaded back via PIL with correct dimensions
  and mode. **Also caught a near-miss before it repeated**: `dev_notes/` is meant to
  be tracked in git (same reasoning as the committed `data/raw/` files — Streamlit
  Cloud has no persistent storage, so real notes need to survive a redeploy), so my
  test note needed manual cleanup rather than being gitignored — checked `git status`
  before staging instead of assuming, avoiding the same mistake as last time.
- Blocked: real freehand drawing interaction (actual mouse/touch strokes on the
  canvas) is unverified in a live browser — this sandbox can confirm the component
  renders and the save path works, not that drawing itself feels right to use.
- Next: worth a real check once deployed that `streamlit-drawable-canvas-fix` behaves
  well on mobile/touch (Streamlit Cloud apps get used from phones), since the fork's
  own maintenance history is thinner than the original's ever was before it broke.

---

**2026-08-26 — [work]**
- Did: Alex flagged that the sidebar Dev Mode button can't be clicked while on a player
  card — real root cause, not a bug in the button itself: player cards use `st.dialog`,
  which is a genuine Streamlit modal. Everything outside it, sidebar included, is
  actually inert while it's open, not just visually behind the overlay — confirmed this
  is a platform constraint by checking how `st.dialog` works rather than assuming it
  was fixable from the sidebar side. Since Alex specifically wants to leave notes about
  how player cards should display, and that's exactly the content living inside the
  dialog, **added a self-contained Dev Mode note form directly inside the dialog**
  (`_player_detail_dialog`) instead of trying to route around the modal — an expander
  at the bottom of the card with its own text area and save button, using the same
  `save_dev_note()` already used by the sidebar flow. Notes are filed under
  `"Player Card — {player_name}"` so they're distinguishable per player in the Dev
  Notes list, with page_state capturing player name/team/position/depth_rank/selected
  season — enough context to know exactly what was being looked at even without a
  literal page to restore to.
  **Caught and fixed a real bug while writing this**: `selected_season` is only
  defined conditionally (inside an `if "recent_games" in detail or "snap_trend" in
  detail:` block earlier in the function) — my first draft referenced it via
  `"selected_season" in dir()`, a fragile pattern that doesn't reliably reflect local
  variable state. Fixed by initializing `selected_season = None` before the
  conditional block, so it's always safely defined by the time the note form needs it.
  **Tested the actual save path directly** (dialogs can't be triggered interactively
  via `AppTest`, same limitation noted for the multipage restore flow last round):
  called `save_dev_note()` with the exact structure the dialog uses, confirmed it
  saves and reads back correctly. Also confirmed the "Go there" restore button
  (added last round) correctly stays hidden for these notes, since `"Player Card —
  Josh Allen"` won't match any real nav page in `PAGE_BY_TITLE` — no crash, an honest
  absence where restoration genuinely isn't possible for a dialog rather than a
  broken button.
- Blocked: same category of gap as the last two UI changes — couldn't visually verify
  the expander actually renders correctly inside a real open dialog from this sandbox.
- Next: worth a real check that the expander doesn't feel buried at the bottom of an
  already-long card (injury report, predictions, Underdog lines, games, snap share,
  news, and now this) — if it's easy to miss, moving it near the top might serve the
  actual use case (leaving notes) better than tucking it after everything else.

---

**2026-08-26 — [work]**
- Did: Improved Dev Notes (the viewer for notes left via Dev Mode) three ways. (1)
  **"Go there" button** — the biggest gap: notes showed the frozen page state as
  read-only JSON with no way to actually get back to it. Built a `PAGE_BY_TITLE`
  lookup (page title string → the real `st.Page` object) right after all pages are
  defined, so a note's saved page name can resolve to something `st.switch_page()`
  actually accepts. Clicking "Go there" writes every saved key/value back into
  `st.session_state` before switching, so the destination page's widgets re-initialize
  from the restored values — not just a link to the page, the actual filters/search/
  slip as they were at freeze time. (2) **Page filter** on the notes list — a flat
  reverse-chronological list stops scaling once notes span all 11 pages. (3)
  **Readable state display** — plain `key: value` lines instead of a collapsed JSON
  tree, faster to scan without expanding nested nodes.
  **Tested the actual mechanism, not just the UI**: `AppTest.switch_page()` turned out
  to only support file-based pages, not the function-based `st.Page()` objects this
  app uses — a real tooling limitation, confirmed by checking the method's own
  docstring rather than assuming. Worked around it for testing purposes: verified the
  restore logic (write saved key/value pairs into session_state) in isolation with a
  stub page object standing in for the real `st.switch_page` call, and confirmed all
  three saved values (`search`, `min_confidence`, `stat_filter`) landed correctly in
  session_state after clicking the button. **Did not verify the full click-through in
  the live multipage app** (actually landing on Weekly Bet Slip with its widgets
  showing restored values) — that needs a real browser session this sandbox doesn't
  have, same category of gap as the earlier CSS redesign that couldn't be screenshotted.
  **Near-miss caught before it mattered**: the routine `rm -f data/raw/*.csv` cleanup
  used throughout this project's history is now WRONG — the Aug 23 session
  deliberately allow-listed specific `data/raw/` files in `.gitignore` so Streamlit
  Cloud's deploy has real committed data to serve. Ran the old habit out of muscle
  memory, staged a deletion of 12 real committed data files, caught it before
  committing (`git status` showed deletions, not new files, which was the tell), and
  restored with `git checkout -- data/raw/`. Nothing was actually lost since it never
  got committed, but flagging this clearly so it doesn't happen again.
- Blocked: the full multipage restore flow (click "Go there" → land on the actual
  destination page → confirm its widgets show restored values) is unverified in a real
  browser. Worth a manual check once deployed.
- Next: if the live restore flow doesn't work as expected, the likely culprit is
  widget key mismatches — a saved session_state key needs to exactly match the `key=`
  a widget uses on the destination page, and that wasn't independently re-verified
  against every page's actual widget keys, only the general Streamlit behavior pattern.

---

**2026-08-26 — [work]**
- Did: Alex returned to work after several home Claude Code sessions (Aug 16, 17, 22,
  23) that pushed real substantial progress but never got logged here — the README's
  Status/Log had drifted noticeably behind the actual repo. Read the real commit
  history (git log, not assumptions) to catch up honestly rather than let the doc rot
  further. Real headline changes since the last logged entry:
  - **Prop coverage expanded from 4 models to 29** (Aug 16): season-long props,
    single-game sacks (first defensive-player prop), passing TDs/INTs, RB receiving
    yards, QB rushing yards, period-scoped props (1st quarter / 1st half yards and
    TDs), first-touchdown-scorer, combined rush+rec TDs — closing real position-scope
    coverage gaps (props were going unmatched because no model existed for that
    stat/position combination, not because the underlying signal was weak).
  - **Retrained all 29 models on 5 more years of history** (2014-2018, Aug 16) — but
    only after validating empirically whether more history actually helps (tested
    4 diverse models three ways: current window, full-history equal-weighted, full-
    history recency-weighted). Also found and respected a real data constraint: NGS
    and PBP's defense_players column both start in 2016, capping the usable extension
    at 3-5 years depending on which features a given model needs.
  - **Two real, well-diagnosed bugs fixed** (Aug 23): retired/inactive players (2,476
    of them — DeMarco Murray, Torrey Smith, a guy who left for rugby in 2015) were
    being scored as live props because nothing checked whether a player_id was still
    actually on a roster — fixed with a roster-membership filter, verified live
    (18,279 → 7,395 predictions, stale rate 63.1% → a legitimate 17.0%). Separately:
    the receiving/rushing-yards proxy-line calculation was silently inflated for
    low-target-share players (a qualification filter was applied before computing the
    rolling average instead of after, erasing exactly the low-volume games that should
    have pulled a run-first back's average down — Derrick Henry's receiving proxy was
    15.2 yards against Underdog's real 4.5 line before the fix).
  - **Recalibrated the line-mismatch confidence gate per prop grain** (Aug 16) instead
    of one global 20% threshold — measured real divergence distributions and found
    season-long/period-scoped props are fundamentally noisier than weekly props, so
    the weekly-tuned threshold was wrongly flagging good season props as mismatched.
  - **Real data now committed to the repo** (Aug 23) — schedules/weekly_stats/NGS/
    injuries/snap_counts/players/news/depth-chart/Underdog props are explicitly
    allow-listed in `.gitignore` and committed, specifically so Streamlit Community
    Cloud's deploy (which only ever serves what's in the repo, no persistent storage)
    actually carries fresh data instead of an empty `data/raw/`.
  - **Dev Mode** (Aug 22): freezes the actual underlying page state (not a screenshot)
    at the moment it's switched on, so a note left for Claude describes state that
    can't drift before it's read — addresses the "no visual screenshot tool in the
    sandbox" limitation from a different angle than a literal screenshot would.
  - Also: real weather forecasts (was a fake placeholder the whole time), line
    movement surfaced on Underdog props, a second historical-odds path via The Odds
    API, ESPN news removed (persistently blocked even from Streamlit Cloud, not just
    this sandbox), and various Parlay Builder UX fixes (confidence range filter
    instead of floor-only, week/team/parlay-type filters, clickable player profiles).
  - **The scheduled-automation piece referenced in the `.gitignore` comment** ("ahead
    of scheduled automation") doesn't appear to actually exist as a GitHub Actions
    workflow file in this repo — checked directly, no `.github/workflows/` present.
    The Aug 23 data commit looks like a one-time manual pull+commit, not a recurring
    automated one yet, despite the comment's wording.
- Blocked: nothing — this is a documentation catch-up, no code changed.
- Next: **the real gap is that this log fell behind reality for over a week** — worth
  either committing to updating it every session (as the format asks) or accepting
  that `git log` is the more reliable source of truth going forward and treating the
  README as a periodic summary rather than a complete record. Also worth deciding
  deliberately whether to actually build the GitHub Actions scheduled pull, since
  right now "the archive accumulates automatically" isn't true yet — it accumulates
  whenever someone manually runs and commits it.

---

---

**2026-08-17 — [work]**
- Did: Alex said the dashboard "feels basic and AI-generated" and asked about tools to
  auto-fix it. Real answer: no separate tool needed, Streamlit's own theming (config.toml
  + CSS injection) is fully capable — the issue was that what existed followed generic
  defaults rather than making deliberate choices. Audited the existing theme (added over
  the weekend): dark base with a gold accent was a reasonable start, but Space Grotesk-
  on-headers-only + rounded-8px-everywhere + no distinct data typography is close to the
  generic "clean SaaS dashboard" template — the exact pattern that reads as templated
  regardless of subject matter. **Redesigned with a real signature grounded in the
  actual subject** (NFL broadcast/scouting-terminal, not generic dashboard): condensed
  broadcast-style headers (Oswald, evokes scoreboard/jersey typography), IBM Plex Sans
  for body, **IBM Plex Mono specifically for numeric data** (odds, confidence scores,
  dataframes) so stats read like a real stat sheet rather than decorative UI chrome,
  sharper 4px corners instead of rounded-everywhere, and a signature "yard marker"
  divider (gold hash-tick rule with a small-caps label) replacing all 4 of the app's
  plain `st.divider()` calls, each given a real contextual label (MATCHED PROPS,
  CORRELATION CHECK, STAKE, MANUAL PULLS) rather than being decorative.
  **Tested, not just written**: verified the full app still loads with no exceptions
  after both the CSS rewrite and the divider swaps (regex-replaced with an assertion
  that each target line was actually `st.divider()` before touching it, so no risk of
  silently corrupting unrelated code), and separately verified the new divider's HTML
  renders clean and well-formed in isolation.
- Blocked: couldn't take an actual visual screenshot to self-critique from this sandbox
  (no browser rendering tool available here) — verification was functional (loads
  without error, HTML is well-formed) rather than visual. **Worth a real visual check
  at home or on the deployed site** before considering this done — CSS that's
  functionally correct can still look off in ways only a screenshot catches.
- Next: if the visual result still doesn't land, the signature element (yard-marker
  divider) is the piece most worth iterating on first — it's the one deliberately
  distinctive choice, so it's doing the most work toward "not generic."

---

**2026-08-17 — [work]**
- Did: Quick pass on the minor/cosmetic known-issues list. Bumped `max_iter` from 1000
  to 3000 on every `LogisticRegression` call across the historical exploratory training
  scripts (`train_cv.py`, `train_ensemble_agreement.py`, `train_final.py`,
  `train_iterate.py`, `train_iterate_v2.py`, `train_pbp_test.py`,
  `train_player_props_v2.py`, `train_player_props_v3.py`, `train_scheme_test.py`) to
  reduce sklearn convergence warnings — low-risk, strictly-safer change (higher
  max_iter never hurts, only helps convergence). Verified all edited files still
  compile. Also checked the "orphaned unused helper function" note in `dashboard/
  utils.py` — a quick reference-count check across both dashboard files found every
  function has 2+ references, so nothing obviously orphaned turned up. **Being honest
  that this is a quick automated check, not a careful manual read** — left the item
  open in Status rather than claiming it's resolved on weak evidence.
- Blocked: nothing.
- Next: none of the currently-active production model training (the inline retraining
  done in recent "Big round"-style commits) used the old scripts this touched, so this
  is cleanup of historical/exploratory code, not something that changes any current
  model's numbers.

**2026-08-17 — [work]**
- Did: Alex asked to make data freshness less manual — added a banner that checks on
  every page load instead of silently rendering against empty/stale data. Two states:
  (1) **completely empty** (fresh Streamlit Cloud deploy — `data/raw/` is gitignored on
  purpose, so it never ships with a deployment) → blocking red banner with a one-click
  "Pull all data now" button, `st.stop()`s the rest of the app rather than showing
  broken pages against zero data; (2) **partially missing/stale** (>24h old) → a
  dismissible-by-refreshing yellow banner naming what's stale, with a "Refresh" button,
  app still usable underneath. Built on the existing `EXPECTED_FILES`/`file_status()`
  helpers already in `utils.py` rather than duplicating that logic.
  **Testing this surfaced a real, separate bug**: an empty/corrupt existing CSV (e.g.
  from an interrupted pull, or in this case a leftover 0-byte file from testing) crashed
  the ENTIRE pull for that file type — `save()`'s merge step called `pd.read_csv()` on
  the existing file with no error handling, so "no columns to parse from file" took
  down schedules specifically while every other file pulled fine. **Fixed**: corrupt/
  empty existing files are now treated as "no prior data" (logged clearly) rather than
  crashing that pull — the new data fully replaces it instead. Verified against the
  exact failure (recreated the 0-byte file, confirmed the old code crashed and the
  fixed code doesn't). Tested all three banner states with Streamlit's `AppTest`
  (empty/blocking, partial/warning, fresh/no-banner) before pushing.
- Blocked: nothing — real feature, real bug fix, both genuinely tested.
- Next: the 24-hour staleness threshold is uniform across all file types right now —
  worth reconsidering once the season starts, since Underdog odds arguably need a
  tighter window (hours, not a full day) than slower-moving stats data. Not urgent
  before Week 1.

**2026-08-17 — [work]**
- Did: Alex pushed back that they'd checked online and nflverse looked current —
  investigated further rather than just re-asserting the earlier finding, and found
  something real: **nflverse restructured player stats before the 2025 season**
  (`nflreadr` release notes: `load_player_stats()` switched to `nflfastR::calculate_stats()`
  output, moving from per-year files to ONE unified file with all seasons combined).
  Our puller (via `nfl_data_py`) was still using the old, deprecated per-year URL
  structure — a genuinely real problem, since that path isn't guaranteed to keep
  receiving future seasons at all. **Fixed `pull_weekly_stats()` to hit the new
  unified URL directly** (`.../releases/download/player_stats/player_stats.parquet`),
  verified end-to-end (33,287 rows, correct per-season breakdown matching a direct
  raw-file check). This moves us onto the actively-maintained path going forward.
  **However — checked the new file's actual season coverage directly, and 2025 still
  isn't there.** Confirmed via two independent, genuinely different checks now (old
  per-year URL AND the new unified file) that weekly player stats data for the 2025
  season simply hasn't been published yet, full stop — not a stale-URL problem, not
  our bug. Alex's "nflverse looks current" observation was likely about a different
  part of nflverse (schedules, rosters, general site) — those genuinely are current,
  as confirmed in the previous log entry; weekly player stats specifically is the one
  piece still lagging, and that's now double-confirmed rather than assumed.
- Blocked: still can't unlock 2025 weekly_stats — it doesn't exist at the source under
  either structure. This is now about as rigorously verified as it can be without just
  waiting for nflverse to publish.
- Next: the URL fix is real, permanent value regardless of the 2025 gap — worth keeping.
  Recheck periodically whether the new unified file's season coverage has grown past
  2024 (a quick direct parquet read of just the `season` column is a cheap check, no
  need for a full pull).

**2026-08-17 — [work]**
- Did: Alex reported only seeing 2024 data, asked to verify every pull is current.
  Checked live (not trusting the weekend's note) whether nflverse has published 2025
  season data yet. **Real, mixed finding — progress since the weekend, but one genuine
  remaining gap**: schedules, injuries, snap counts, and all three NGS datasets (passing/
  rushing/receiving) DO now have real 2025 data — pulled and merged in (6,068 injury
  rows, 26,612 snap-count rows, 1,402 NGS-receiving rows, all real 2025 season data,
  verified by season breakdown after the pull). **`weekly_stats` — the core data source
  every prop model's targets/yards/attempts features come from — is still stuck at 2024
  week 22 (Super Bowl)**. Confirmed this is a real source-side gap, not our bug: hit the
  actual file URL directly (`player_stats_2025.parquet`) and got a genuine 404, while
  the 2024 file resolves fine. nflverse simply hasn't published that specific file yet,
  even though four other 2025 datasets are live.
- **Practical impact**: since every prop model needs `weekly_stats` to build a player-
  week row at all, "current" predictions can't advance past 2024 regardless of the other
  four datasets being current — confirmed directly (receiving-yards feature dataset's
  latest available row is still 2024 week 22).
- Blocked: this specific gap isn't fixable in our code — it's waiting on nflverse to
  publish. Worth periodic rechecking (a quick `curl -I` on the file URL is enough,
  doesn't need a full pull) rather than assuming it'll never update.
- Next: **couldn't verify from this sandbox whether the non-nflverse pullers (ESPN,
  SB Nation, NBC Sports, Underdog, Footballguys) are actually current** — those need
  the home network. Worth checking at home whether the "daily scheduled task" mentioned
  in earlier log entries is actually running and keeping those fresh, not just that it
  was built to run daily.

---

**2026-08-15 — [home]**
- Did: First session in a fresh home environment — set up git identity, GitHub CLI auth
  (`gh`), Python 3.12, and the venv/requirements from scratch (none of it existed yet
  here). Cloned the real repo from GitHub rather than trusting the zip export Alex had
  on hand, which turned out to be significantly stale (missing the receptions/rushing/
  passing prop models and several feature modules already on `main`).

  Then picked up the TOP PRIORITY item: **`pull_underdog.py` works from this
  environment** — confirmed unreachable from the work sandbox (`host_not_allowed`), but
  a live pull here returned 1,024 real NFL prop-option rows. Real schema finding: as of
  today (preseason, 2026-08-15) most live lines are season-long totals
  (`season_receiving_yards`, `season_pass_tds`, etc.); per-game lines do exist already
  for preseason games (`receiving_yds`, `rushing_yds`, `passing_yds`, `passing_tds`,
  `passing_ints`, `sacks`) but **no single-game `receptions` line was live yet** — worth
  rechecking once the regular season starts, since the receptions model is currently the
  strongest of the four props. Modified the script to append timestamped snapshots to
  `data/raw/underdog_history/` (previously it only overwrote one file, which can't build
  an archive) and set up a daily 9am scheduled task to run it automatically.

  Also tested `pull_espn_news.py` per the other open "needs home network" item — **it's
  blocked, but not by the sandbox**: ESPN's own Akamai edge returns a real 403 Access
  Denied to this environment too (confirmed via curl, not a code bug). Left it broken
  rather than trying to route around a WAF. Built `data/pull_sbnation_news.py` as a
  replacement per Alex's request — pulls all 32 teams' SB Nation blogs via their public
  Atom feeds (Vox Media Chorus platform, `/rss/index.xml` on every team site, verified
  live against sbnation.com/nfl's own nav rather than guessed). All 32 teams returned
  10 articles cleanly on a full test run. Also added to the daily schedule.
- Blocked: nothing — both new/fixed pullers are real, live-tested pulls, not simulated.
- Next: **the archive needs time to accumulate**, not more code — let the daily schedule
  run for a while (ideally through into the regular season, when per-game receptions
  lines should appear) before trying to replace the proxy-line backtesting with real
  Underdog history. In the meantime, the SB Nation news feed is live and could inform
  the "own_injury_severity" / context features already in the individual-context model
  with more current, team-specific detail than the box-score-derived injury counts
  currently used. Ranked below the line archive but worth a look.

  **Later same day**: wired all four prop models into `current_predictions.py` and the
  dashboard (previously receiving-only). Also pulled real nflverse data into this
  environment for the first time (2024-2025; **2025 `weekly_stats` isn't published at
  nflverse's usual endpoint yet** — pulled 2024 only for that specific file, real
  current limitation worth knowing about, not a bug here). While wiring predictions in,
  inspected each saved model's own `.pkl` directly to get its exact `features` list and
  `position_scope` rather than trusting training-script filenames, since several
  historical training scripts in `models/` are stale/superseded and don't reflect
  what's actually in production — the pickles themselves are the only reliable source
  of truth for what each model expects. Found and fixed a real bug this surfaced: the
  Parlay Builder matched predictions to Underdog props by player name only, which
  would've silently cross-wired predictions once more than one prop type existed (e.g.
  attaching a rushing prediction to a receiving prop for the same player) — now matches
  on name + stat type, verified against live data (78/1024 real prop rows matched,
  spot-checked correct-stat matching for several players). Also wired the SB Nation
  puller into the dashboard as its own page — it was built and scheduled earlier today
  but never actually surfaced in the UI. Tested the whole dashboard live in a browser
  (Overview, Parlay Builder, SB Nation News pages) against real pulled data, not just
  code review.
- Next: Streamlit Cloud deploy is the last open item from the original list, untouched.
  Two small pre-existing rough edges noticed but not touched (out of scope for this
  session): `dashboard/utils.py` has an orphaned, uncallable odds-to-probability helper
  (docstring + body with no `def` line, dead code, nothing currently calls it); and the
  saved model pickles throw harmless XGBoost/scikit-learn version-mismatch warnings on
  load (models were pickled with slightly older library versions than what's installed
  here).

  **Later same day — real out-of-time validation, per Alex's request**: pulled the full
  2019-2023 range too (previously only had 2024-2025 here), specifically to genuinely
  retrain each prop model using ONLY seasons before 2024, then score all of 2024 blind
  — a real walk-forward test, not a re-read of previously reported numbers. New script:
  `models/backtest_holdout_2024.py`. Used the exact same feature lists and
  hyperparameters as production (introspected from each `.pkl`: XGBoost n_estimators=
  200/max_depth=3/lr=0.05/subsample=0.8/colsample_bytree=0.8 for receiving/rushing/
  receptions; LogisticRegression max_iter=3000 for passing), so this is a faithful
  re-creation of the original leave-one-season-out methodology, independently rerun.

  **Result: it holds up.** Fresh 2024-blind accuracy landed within 1-3 points of every
  previously reported number:

  | Prop | Fresh 2024-blind | @0.4 confidence | Previously reported |
  |---|---|---|---|
  | Receiving yards | 66.9% | 79.4% (817/2081 games) | 67.1% / 80.3% |
  | Rushing yards | 69.9% | 81.9% (414/865 games) | 70.6% / 83% |
  | Passing yards | 61.7% | 76.7% (86/465 games) | 61.7% / 79.7% |
  | Receptions | 72.3% | 82.6% (1099/2081 games) | 72.6% / 84% |

  This is real evidence the models aren't overfit to the original CV process — a
  completely independent retrain, on real held-out data, in a different environment,
  reproduces the claimed edge. Small note: passing's 0.4-confidence bucket is only 86
  games, the noisiest of the four given the smaller QB dataset — directionally
  consistent but hold that one number more loosely than the others.
  (Data note: pulling 2019-2023 required overwriting `schedules.csv`/`pbp.csv`
  mid-session since `pull_nflverse.py` overwrites per-file rather than appending —
  recovered by backing up the untouched files first and re-pulling/merging 2024-2025
  back in. All raw data in this environment now spans 2019-2025, with the known 2025
  `weekly_stats` gap still in place.)
- Next: same open items as above (Streamlit Cloud deploy). The backtest script is
  reusable — worth rerunning against 2025 once nflverse publishes `weekly_stats` for
  it, which would be the first genuinely-new-season validation rather than a rerun
  against 2024.

  **Later same day — new theories round, per Alex's request to keep testing and try to
  improve the models**: before picking new ideas, actually checked what had and hadn't
  been tried yet (grepped every `models/train_*.py` script rather than guessing) —
  found real gaps instead of re-testing dead ends: `add_usage_trend` was built during
  the individual-context round but never actually included in that round's tested
  feature groups; the receptions model didn't exist yet when the game-context and
  matchup-defense-splits rounds ran, so it never got either treatment despite being the
  strongest model; team-scheme features (`pass_oe`, tempo, box counts) were only ever
  tested on receiving, never rushing/passing/receptions. New script:
  `models/train_momentum_and_receptions_gaps.py`, same rigorous pooled 5-season CV as
  every other round, current production feature lists + tuned hyperparameters as the
  base (so "beats base" means "beats what's actually deployed").

  **Result: clean sweep of honest nulls, ~20 feature combinations tested:**
  - Usage trend (role momentum): null-to-worse on all 4 props (rushing actually
    dropped, -0.37pt) — the *level* of usage already in every model apparently
    captures what matters; the *trend* doesn't add anything on top.
  - Game-context (implied_total/home_away/weather/rest/snap_share) on receptions:
    null across all 5, consistent with receiving (receptions' closest cousin) already
    being null on the same features.
  - Matchup-specific defense splits on receptions: null across all 4 + combined —
    makes this 6/6 props-tested where defense-side granularity has failed to add
    value beyond the simple `def_epa_allowed_rolling` already in every model.
  - Team-scheme features on rushing/passing/receptions: null across the board. Closest
    thing to a signal all round was `pass_oe` on rushing (70.5%→70.8% raw, a real
    mechanistic hypothesis — a team's pass-rate-over-expected is almost definitionally
    tied to how much it runs) but confidence-filtered accuracy barely moved with
    *fewer* games clearing the 0.4 bar (83.0%→83.2% on 48.5%→47.7% of games) — same
    "false economy" pattern already seen elsewhere in this project. Not kept.

  **Honest bigger-picture read**: this is now roughly six separate rounds (QB-specific
  stats, real PBP efficiency, scheme/tendency, game-context, matchup-specific defense
  splits, and now usage-trend + the receptions/rushing/passing gap-filling above) where
  opponent/context-side features have failed to add value beyond what's already
  deployed. The current feature set — box scores + NGS tracking data + the player's own
  opportunity signals (target share, efficiency, time to throw) — appears to have
  captured most of the readily available signal for the *proxy-line* target
  specifically. This loops back to why the Underdog line archive is still ranked #1:
  it's not just "the number that tells us if this is profitable" (original framing) —
  it's also the only realistic remaining lever for a genuinely different kind of gain,
  since it changes the target itself (real market line) rather than adding another
  feature trying to predict the same proxy target everything so far has already
  squeezed hard.
- Next: this doesn't mean stop iterating — it means the next real gains are more likely
  to come from (a) the Underdog archive accumulating enough history to backtest against
  real lines instead of the proxy, or (b) a genuinely different-in-kind data source like
  the SB Nation news feed (not yet usable for backtesting — no historical archive exists
  for past seasons — but worth exploring for *current* predictions once there's a way to
  turn free text into a structured signal, e.g. detecting real injury/role-change
  language ahead of the official injury report). Feature-engineering-on-the-same-data-
  source appears close to tapped out for now.

  **Later same day — second news source, per Alex asking whether a different one would
  serve better**: SB Nation is longer-form team analysis; the specific thing we said we
  wanted (injury/role signal ahead of the official report) is better served by a fast,
  atomic insider-rumor source. Checked technical feasibility for real before building
  anything (same discipline as every other data source this project uses) rather than
  assuming: ESPN's news API stays blocked (confirmed again); Pro Football Talk's old
  domain (`profootballtalk.nbcsports.com`) now redirects to `nbcsports.com/nfl/
  profootballtalk` after an NBC site consolidation; `rotoworld.com` similarly redirects
  to `nbcsports.com/fantasy`; `nbcsportsedge.com` doesn't resolve at all. No RSS link
  tag on the new PFT page itself, but found two real working feeds by testing URL
  patterns directly: `nbcsports.com/nfl.rss` (genuinely live, same-day items) and
  `nbcsports.com/nfl/profootballtalk.rss` (real but ~1 day behind). A third candidate,
  `.../profootballtalk/rumor-mill.rss`, returned HTTP 200 but with items 3+ weeks stale
  — deliberately excluded rather than pulling dead data just because the endpoint
  responds.

  New script: `data/pull_nbcsports_news.py`, pulling both live feeds, deduped by link
  into `data/raw/nbcsports_news.csv` (same accumulation pattern as the SB Nation
  puller). Wired into the dashboard as its own page ("NBC/PFT Rumor Mill") and added to
  the daily scheduled task alongside Underdog + SB Nation. Real limitation worth
  stating plainly: each feed only holds ~4 items, so this is NOT a deep archive like SB
  Nation's ~10-per-team pull — running it daily will miss items on busy news days. It's
  additive (different in kind: short atomic insider blurbs vs. long-form team recaps),
  not a replacement for SB Nation.
- Next: if the sparse-item-count gap turns out to matter in practice, the fix is
  tightening this specific puller's schedule (e.g. hourly) rather than anything about
  the feeds themselves, which are already the freshest ones available from this
  publisher. Same open items as every other entry above: Streamlit Cloud deploy,
  Underdog archive accumulating.

  **Later same day — getting ready for the real season, per Alex**: it's preseason
  (real games barely mean anything predictively), so rather than chase live preseason
  props, went looking for what would actually break once the regular season starts.
  Found two real gaps by reading `current_predictions.py` critically instead of
  assuming it was season-ready just because it worked today:

  1. **No visibility into stale data.** Rolling stats need 3+ prior games *within a
     season* (by design, avoids noisy small samples) — which means weeks 1-3 of every
     new season have zero qualifying current-season rows for anyone. The code used to
     silently fall back to a player's last game of the *previous* season and treat it
     as current, with no way to tell from the output that it had done this.
  2. **Context wired to the wrong game.** Injury status, div/primetime flag, weather,
     and Vegas implied total were matched to whatever game a player's rolling stats
     happened to be dated to — correct once a season is underway, wrong for the exact
     situation above, and also just conceptually backwards for a "current prediction"
     (a bettor cares about *this week's* matchup context, not last week's).

  Fixed both: `current_predictions.py` now looks up each player's team's actual next
  *unplayed* game from `schedules.csv` (`home_score` still null) and merges injury/
  game-flags/game-context against THAT game specifically, not wherever the rolling
  stats happened to end. Output now makes the two dates explicit instead of collapsing
  them into one ambiguous `season`/`week` — `stats_as_of_season/week` (where the
  rolling stats come from) vs. `next_season/week/opponent` (the actual game being
  predicted). Confirmed the 2026 schedule is already published at the nflverse source
  (272 games, real Vegas lines already set, Week 1 starts 2026-09-09) and pulled it in.
  Real validation, not just code review: reran predictions and found genuine cases the
  fix was built for — e.g. a player whose last qualifying game was **2020 week 16**
  now surfaces that honestly instead of pretending it's current. Wired the same
  distinction into the dashboard's Parlay Builder (⚠️ tag on any prediction older than
  the freshest available data) and verified it renders correctly in a browser.
  Known remaining limitation, stated plainly rather than solved: this assumes a player
  is still on the team from their last recorded game — a trade or signing between then
  and now wouldn't be caught (no roster/depth-chart pull exists in this project).

  **Also fixed the root cause of today's earlier near-miss**: `pull_nflverse.py`'s
  `save()` used to overwrite each file outright rather than merging — fine for one-off
  historical backfills, actively dangerous for the weekly-during-season refresh this
  project now needs, which would otherwise wipe prior weeks every single run. Now
  merges into any existing file by the same key columns `validate()` already checks
  (new rows win on key collisions -- e.g. a corrected stat), tested for real by
  re-pulling 2026 schedules alone and confirming 2019-2025 stayed intact (2232 rows
  before and after). This makes the existing daily scheduled task safe to extend with
  a real nflverse refresh + automatic `current_predictions.py` regeneration, which it
  now does (recent seasons only, `--skip-pbp` for speed) — the dashboard's predictions
  will now actually stay current through the season without a manual re-run, which
  they would NOT have before this fix.
- Next: this was about correctness/plumbing, not new modeling ideas — the actual
  predictive-power work (feature testing, the Underdog archive) is paused until real
  regular-season data exists to test against, which is the right call given how
  unpredictable preseason is. Worth a final live check once Week 1 actually happens:
  confirm the daily task's nflverse step actually succeeds against a partial in-progress
  season (untested — behavior of `nfl_data_py` for "season requested but only some
  weeks have been played yet" is inferred, not confirmed against real data yet).

  **Later same day — found a real depth-chart/roster-move source, per Alex asking
  about @rclfootball on social media**: checked it out rather than assuming — it's
  Instagram/Threads (not Twitter/X), 63K followers. Real technical dead end though:
  neither platform exposes a public feed for third-party accounts, actual post content
  is login-gated (confirmed via curl — raw HTML has only aggregate follower/post
  counts, no actual posts), and getting real content out would mean either logging
  into an account (can't handle credentials) or scraping tools that violate Meta's ToS
  more seriously than anything else in this project so far. Didn't build anything
  against it, said so plainly, and asked whether to check dedicated depth-chart sites
  instead — Alex said yes.

  Evaluated 6 candidates for real before picking one: Ourlads (no clean data found in
  initial pass), Pro Football Network (free, fresh, offense-only, no status tags),
  RotoWire (great per-player status tags but **paywalled** — "reserved for
  subscribers" on all but ~6 teams alphabetically), FantasyPros (Premium/Subscribe/
  Sign-In gated), Lineups.com (**dead link** — /nfl/depth-charts redirects to their
  homepage), and **Footballguys** (free, all 32 teams, offense + defense + special
  teams, explicit same-day "last updated" stamp, and — the actual point — structured
  per-player status tags: Q/PUP/IR/SUS/NFI/O/CEL/EX). Real cross-source disagreement
  found and noted rather than glossed over: PFN and Footballguys disagreed on the
  Falcons' QB1 (Penix vs. Tagovailoa) — exact depth-chart *order* should be trusted
  loosely regardless of source; the status tags are the reliable part.

  New script: `data/pull_footballguys_depth.py`. Confirmed fully server-rendered
  (all 32 teams present in one plain HTTP GET, verified by counting team-name and
  category-class occurrences in the raw response) — no browser automation needed.
  Added `beautifulsoup4` to requirements.txt for real structured HTML parsing (CSS
  classes like `depth-chart-cat-off`, `pos-label`, `player starter`) rather than
  fragile regex, since the nested position/player markup genuinely warranted it — this
  is the first scraper in the project that needed more than `requests` + a basic
  parser. First real pull: 2,626 player-position rows across all 32 teams, 342 with an
  active status tag.

  Wired into the dashboard two ways: a new "Depth Charts" browse page, and — the part
  that actually matters — a live cross-reference in the Parlay Builder itself. Every
  prop now shows a 🚑 warning tag when Footballguys currently has a status flag for
  that player (verified live in a browser: e.g. "Kenyon Sadiq — season_receiving_yards
  over 454.5 🚑 Q (Footballguys, today)"). Deliberately NOT fed into the trained models
  as a feature — no historical archive exists for this source (same limitation as
  SB Nation news), so there's nothing to validate it against yet. It's a live display
  signal for now, same tier as the news feeds, not a backtested input.
- Next: same open items as above. If depth-chart-as-a-feature ever gets tested
  properly, it would need weeks of accumulated snapshots first (mirroring the Underdog
  line archive's own bootstrap problem) — pulling it daily, which is now wired up,
  is what would eventually make that possible.

  **Later same day — verified the daily scheduled task for real, per Alex's request**:
  rather than trust a 6-step chain that had grown across the whole session without
  ever being watched end-to-end, ran every step manually in order and read each
  result. All 6 completed successfully. One real, confirmed finding along the way —
  upgrading what was previously flagged as "inferred, not confirmed" to actually
  verified: **nflverse's `player_stats` (weekly_stats) release for 2025 doesn't exist
  yet**, not a bug here. Checked directly against nflverse's GitHub release API rather
  than assuming: the 2024 release has ~85 file-format variants (csv/parquet/rds/qs ×
  several stat groupings); 2025 has **zero** — genuinely unpublished at the source,
  despite the season having ended back in February. Same root cause affects injuries
  and snap_counts (also 404). NGS (Next Gen Stats) is on a different pipeline and DID
  succeed for 2025. Confirmed the existing per-pull try/except in `pull_nflverse.py`
  already handles this correctly (continues past the failure, reports it, doesn't
  crash the rest of the chain) — this was designed-for but untested until now. Steps
  2-6 (Underdog, SB Nation, NBC/PFT, Footballguys, predictions regen) all ran clean;
  predictions came out byte-identical to before, correctly, since no underlying stats
  actually changed.
- Next: the daily task is confirmed working end-to-end as designed. The nflverse gap
  is out of this project's control — worth re-checking once actual 2026 regular-season
  games start being played, since that's a different, more actively-maintained part of
  nflverse's pipeline than a stalled prior-season backfill might be. If 2025/2026
  weekly_stats keeps 404ing once the season is underway, that would be a real problem
  worth investigating harder; right now it's a known, gracefully-handled external gap.

  **Later same day — does the model need to actually get smarter as a season
  progresses, per Alex?** Real question worth taking seriously: every NFL season has
  its own trends (scheme shifts, rule changes, rookie classes), and right now all four
  models are trained once on historical seasons and never updated as the current
  season unfolds — they're only ever *scored* fresh, not *retrained*. Two ways to
  address that discussed with Alex: (a) add season-level trend features to the
  existing static models, or (b) actually retrain week over week so the model's
  learned parameters themselves evolve. Alex wants (b) — the model getting smarter
  over time, not just fed a trend signal.

  Built `models/backtest_recency_weighting.py` to test this for real before building
  a production retraining pipeline around it: simulates real weekly retraining through
  the entire 2024 season (train on 2019-2023 + all 2024 weeks before W, score week W
  blind, repeat for every week) with samples weighted by recency (`0.75 **
  (2024 - row_season)`, so 2024's own weeks always count most and count more as more
  of them accumulate) — exactly how it would work if wired into the weekly pipeline.
  Compared directly against the static model on the identical test rows.

  **Result: null-to-negative, tested two ways.** Aggressive decay (0.75/season):
  receiving -0.46pt, rushing -0.62pt, passing -0.92pt, receptions +0.05pt — worse or
  flat across the board. Gentler decay (0.9/season), tested as a robustness check
  rather than stopping at one arbitrary parameter choice: receiving -0.41pt, rushing
  +0.12pt, passing +0.23pt, receptions +0.77pt. The receptions number looks like the
  closest thing to a real signal, but at n=1,949 the standard error is ~1.0pt — a
  0.77pt gain doesn't clear normal sampling noise. **Not deployed.**

  Why this is a genuinely useful negative result, not just a null: it explains WHY
  season-to-season trends (which are real, agreed with Alex on this) don't require
  retraining to handle. The player-level rolling features already reset and rebuild
  every season, so each player's OWN current-season form is already fully reflected
  without the model itself needing to change. What the model actually *learns* — how
  those features relate to outcomes — appears stable year over year. Diluting 5+
  years of stable historical relationships with a small, noisy partial-season sample
  mostly adds noise rather than teaching the model something new. This is consistent
  with, not contradictory to, everything else found in the "new theories" round
  earlier today — the current architecture already captures the adaptive part that
  actually matters.
- Next: not pursuing weekly retraining further given two independent null results.
  If this gets revisited, the more promising next step isn't a different decay rate
  (already tested two, both inconclusive) but testing recency weighting on receptions
  specifically in isolation with a larger decay sweep, since it's the only prop that
  showed any positive direction at all — low priority given the effect size found so
  far. Same open items as every entry above otherwise.

  **Later same day — three more angles, per Alex wanting to keep pushing for
  accuracy**: deliberately picked a different CATEGORY than the repeatedly-null
  context/defense-side features, since that pattern is now well-established. New
  script: `models/train_consistency_qb_continuity_interactions.py`.
  1. **Player consistency/volatility** (rolling std + coefficient of variation of each
     prop's own primary stat) — nothing tested before now described how PREDICTABLE
     a player's production is, only its level/trend. Null-to-negative across all 4
     props; rushing was the clearest loser (-0.5 to -0.7pt).
  2. **QB continuity** (receiving/receptions): built a real team-week starting-QB
     lookup from `weekly_stats.csv` and flagged when this week's starter differs from
     last week's — a genuine football mechanism (route/timing chemistry) never
     tested. Completely flat on both props, despite 1,575 of 12,266 rows (12.8%)
     genuinely having a QB change. Surprising given the football logic, but a clean,
     real null, not a bug — worth taking at face value rather than assuming the
     feature must be wrong.
  3. **Explicit interaction terms for passing** (LogReg-specific, since unlike
     XGBoost's tree splits it can't learn interactions on its own): implied-total x
     volume, aggressiveness x opponent defense, wind x air-yards. All three worse than
     base, none close to the bar.

  **Honest synthesis, stated plainly rather than just moving on to the next idea**:
  this is the **7th consecutive round** without a real win, across dozens of
  feature/architecture combinations spanning every category tried — opponent/defense
  context (6/6 null), player-side momentum and trend, model retraining cadence
  (2 variants), and now player-side volatility, real-world continuity, and explicit
  interactions. That's not bad luck on which ideas got picked; it's a genuinely
  strong, consistent signal that this modeling approach — rolling box-score/NGS
  features, binary classification against a proxy line — has reached its practical
  ceiling for this data source. A model that reliably beats a player's OWN rolling
  average doesn't necessarily mean it beats a sportsbook's actual line, which already
  prices in most of what these features capture.
- Next: recommending a real pivot in where effort goes, not just another feature
  round. The two levers that are structurally different (not just another additive
  feature on the same target) are: (a) the Underdog line archive, which changes the
  TARGET itself from a proxy to a real market line — the only thing left that could
  reveal genuine edge rather than re-measuring the same ceiling a different way, and
  (b) an actual architecture change (e.g. predicting the real value via regression
  instead of binary over/under, or a genuinely different target formulation) —
  bigger, riskier, not yet scoped. Continuing to search for incremental features on
  the current architecture has clearly diminishing returns at this point; said so
  directly rather than proposing a plausible-sounding 8th round.

  **Later same day — Alex asked me to do whatever I thought would make the model
  better at predicting PARLAYS specifically**, not single props. Took that literally:
  the 7 rounds above were all about single-leg accuracy, but nothing in this project
  had ever examined how legs get COMBINED. Checked `dashboard/utils.py`'s
  `parlay_combined_multiplier` and the slip math in `app.py` — confirmed both just
  multiply individual probabilities together, a textbook independence assumption,
  never questioned or tested.

  Built `models/analyze_parlay_correlations.py`: for every 2019-2024 game, gathers
  each rostered skill player's real over_proxy_line outcome across all 4 props, then
  for same-team same-game pairs, compares the actual joint hit rate to what pure
  independence predicts (phi coefficient, -1 to +1). Real, substantial findings —
  genuinely different in kind from the null rounds:

  | Pairing (same team) | phi | What it means |
  |---|---|---|
  | TE receiving yds + TE receptions (same player) | +0.345 | Same player's own two stats -- highly redundant, not two independent bets |
  | QB passing yds + WR receiving yds | +0.231 | Good passing games lift the WR corps together |
  | QB passing yds + WR receptions | +0.174 | Same mechanism, receptions |
  | QB passing yds + TE receiving yds | +0.174 | Same mechanism, TE |
  | WR receiving yds + WR receptions (same player) | +0.172 | Same-player redundancy again |
  | QB passing yds + TE receptions | +0.134 | Weaker TE version |
  | **QB passing yds + RB rushing yds** | **-0.081** | **Real negative** -- pass-heavy and run-heavy game scripts trade off |
  | RB rushing yds + RB rushing yds (different RBs) | +0.058 | Mild positive -- "team ran a lot today" outweighs committee competition |

  Everything else tested (RB rushing vs. WR receiving, TE vs. WR, etc. on the same
  team) came back genuinely close to independent (|phi| < 0.02) — confirming the old
  math was actually fine for those pairings specifically, not wrong everywhere.

  Saved the real (|phi| >= 0.05) pairings to `models/parlay_leg_correlations.csv` and
  built `correlation_adjusted_parlay_probability()` in `dashboard/utils.py`: uses the
  phi-coefficient identity P(A∩B) = p_A·p_B + phi·√(p_A(1-p_A)·p_B(1-p_B)), clamped to
  the Frechet-Hoeffding bounds a joint probability can never violate. For 3+ legs this
  multiplies independent pairwise corrections together (an approximation, not an exact
  joint distribution, stated as such in the code) rather than leaving correlated legs
  silently mispriced. Wired into the Parlay Builder: any slip containing a known-
  correlated same-team pair now shows both the naive and corrected combined
  probability, with the specific phi displayed. Removed the now-fully-superseded
  `parlay_combined_multiplier` (confirmed unused anywhere else first).

  Verified the math directly with 3 real sanity checks before trusting it: (1) a real
  live pairing -- Joe Burrow (CIN QB passing_yds, p=47.3%) + Ja'Marr Chase (CIN WR
  receiving_yds, p=32.4%) -- naive combined 15.32% correctly becomes 20.71% adjusted
  (positive correlation raises the true joint probability, correctly lowering the fair
  multiplier from 6.53x to 4.83x); (2) a different-team pair produces zero adjustment,
  confirming the team-matching gate works; (3) a same-team QB+RB pair (known negative
  phi) correctly comes out LOWER after adjustment, not higher. All three behaved
  exactly as the underlying math predicts.

  Honest caveat on verification: couldn't get the actual "Add to slip" button click to
  register through the automated browser this round (a recurring tooling friction all
  session, e.g. the search box earlier) despite the page loading with zero console
  errors and the button code being structurally identical to the already-working
  pre-existing pattern. Trusted the direct mathematical verification above over a
  UI click-path that the tooling wouldn't cooperate with, rather than either skipping
  verification or blocking on a browser automation issue unrelated to the actual code.
- Next: this is the more promising lever going forward, not another single-prop
  feature round. Worth extending: (a) opponent-team pairs, not just same-team (does a
  shootout lift props on BOTH teams?), untested so far; (b) an actual full joint
  distribution for 3+ mutually-correlated legs instead of the pairwise-multiplication
  approximation, if slips with 3+ same-team legs turn out to be common; (c) a real
  click-through UI verification once the automated browser cooperates, or a manual
  check by Alex.

  **Later same day — Alex wants weekly bet suggestions, sized, that Alex places
  themselves**: confirmed directly, this project never executes real-money
  transactions or handles Alex's account, even with per-bet approval offered --
  that's a hard boundary regardless of authorization, not a permission level to
  unlock. What's genuinely useful and in-scope: generate the actual suggestion, sized,
  each week.

  Built `models/generate_weekly_bet_slip.py`: ranks real live Underdog opportunities
  (straight legs + 2-leg correlated parlays, reusing this session's own correlation
  math) by Kelly-criterion edge at REAL prices, splits a fixed weekly budget across
  the top few proportionally. First real run found "7 +EV opportunities" -- which led
  straight into the two-bug discovery below, so those first numbers were wrong and
  were retracted before being presented as usable.

  **Bug 1**: matched props to predictions by player+stat only, hardcoded to the
  "over" side. A player the model confidently favored UNDER was being scored as if
  the model liked the OVER at only that player's low probability -- backwards.
  Fixed: resolve `my_side`/`my_prob` from whichever side `predicted_prob_over`
  actually favors (>=0.5 -> over, else under -> 1-p), then match against THAT side's
  real price specifically. Correlation phi values (measured for "over" outcomes)
  needed a sign flip for legs whose favored side is "under" -- a standard identity,
  corr(A, 1-B) = -corr(A,B) -- applied once per under-side leg in the pair.

  **Bug 2, more serious, caught by manually checking the actual numbers rather than
  trusting a clean-looking output**: `predicted_prob_over` answers "beats OUR proxy
  line" (the player's own rolling average used throughout this whole project), not
  "beats Underdog's actual posted line." Checked a specific "opportunity" the script
  had ranked #1 (Jameson Williams, 91.6% model confidence) and found the proxy line
  was 4.1 receiving yards while Underdog's real live line was 65.5 -- completely
  different questions being silently treated as the same one. Worse: because a big
  line mismatch mechanically produces a huge-looking "probability x price - 1" edge,
  ranking purely by edge was systematically surfacing the WORST mismatches as the
  "best opportunities" -- confirmed by checking the full board: 28 of 39 matched
  props (72%) are naturally within 20% of each other and would have been fine, but
  the ranking logic kept picking from the other 26% specifically.

  Fixed with `MAX_LINE_DIVERGENCE = 0.20`, now centralized in `dashboard/utils.py`
  rather than duplicated, gating out any match where Underdog's line and our proxy
  line diverge by more than 20% -- with the exclusion count printed so this can't
  silently narrow the board again without being visible. Re-ran clean: 3 real
  opportunities survive (Jameson Williams under 65.5 at 9% line divergence, Alec
  Pierce under 56.5 at 11%, Romeo Doubs under 38.5 at 17%), all genuinely priced
  the model likes at REAL Underdog prices, not proxy-line artifacts.

  **Same two bugs existed in the dashboard's Parlay Builder** (built earlier this
  session) -- it does the identical player+stat matching. Fixed there too rather
  than leaving a tool the project points Alex toward silently broken: the per-row
  loop now resolves `model_prob` from the row's actual choice (over/under) and gates
  display/use on the same `MAX_LINE_DIVERGENCE` check, showing a clear "⚠️ model line
  too far from this line to trust" instead of a misleading confidence tag when it
  fails. Verified: script output cross-checked against a direct live-data query
  (confirmed Jameson Williams has two real rows in predictions -- receptions
  proxy=4.1 and receiving_yards proxy=71.8 -- ruling out a name-collision bug before
  concluding the real fix was correct), dashboard confirmed loading clean with no
  console errors and consistent confidence-filter counts (688/1483, matching the
  pre-fix baseline, as expected since the fix changes WHICH matches are trustworthy,
  not the underlying prediction counts).
- Next: the weekly bet-slip generator is real and usable now, but only tested against
  one live snapshot (this week, preseason) -- worth watching over a few real weeks to
  see if the +EV opportunities it finds hold up, especially once the regular season
  starts and Underdog's lines get sharper (preseason lines may simply be less
  efficient, a different explanation for real-looking edge than genuine model skill).
  Not wired into the daily schedule yet -- deliberately left as an on-demand tool
  Alex runs when actually planning a week's bets, rather than another automated
  output to keep track of.

  **Later same day**: "on demand" landed as a dashboard page rather than a schedule
  -- Alex wants it to run "when I want to start making bets," which is a moment, not
  a time of day, so a fixed cron wouldn't actually fit. Added a "Weekly Bet Slip" page
  to `dashboard/app.py`: budget input, a "Generate this week's bets" button, results
  rendered as real cards (stake/description/model prob/real price/edge) instead of
  raw console text. Imports `generate_weekly_bet_slip.py`'s functions directly rather
  than shelling out to the script, so the dashboard and the CLI tool share one
  implementation and can't drift apart. Verified live in a browser: clicked the
  button, got the exact same 3 bets/stakes as the already-verified CLI run, zero
  console errors.
- Next: same as above -- unproven beyond one live snapshot, worth revisiting once
  real weeks of data exist to see if these opportunities were real edge or a
  preseason-line-efficiency artifact.

  **Later same day — deployed to Streamlit Community Cloud**: live at
  parlaymodel.streamlit.app. Verified what would and wouldn't carry over before
  deploying rather than assuming: all 4 trained models, `current_player_predictions.csv`,
  and `parlay_leg_correlations.csv` are tracked in git and deployed correctly;
  `data/raw/*` is gitignored by design, so the live site starts with zero pulled data
  until "Run Data Pulls" is clicked on the deployed instance itself -- confirmed this
  is genuinely true post-deploy (Overview page correctly showed "12 files not pulled").
  GitHub OAuth sign-in and the Secrets/password field were both done by Alex directly
  -- neither is something this project executes on Alex's behalf, a hard boundary
  independent of authorization, and also not something the tooling in this environment
  can technically reach (no access to Alex's authenticated browser session either way).
  First deploy attempt had no password gate active (the `dashboard_password` secret
  hadn't been set yet) -- caught by actually loading the live URL and checking rather
  than assuming the earlier instructions had been followed, not by any error being
  surfaced. Re-verified after Alex set it: real password prompt now blocks the
  dashboard, confirmed via screenshot.

  Clarified for Alex how the pieces fit together, since it's genuinely not obvious:
  the deployed site runs on Streamlit's servers (Alex's PC doesn't need to be on for
  it to be reachable), but the daily local automation still needs this specific PC/
  Claude installation running around 9am for that day's refresh to fire -- and only
  the git-tracked files (models, predictions) auto-sync to the deployed site when that
  runs; live odds/news/depth-chart data needs a manual "Run Data Pulls" click on the
  deployed site itself regardless of the local PC's state. Also confirmed continuing
  development from a work computer is exactly what this project's own existing
  work/home README convention was already built for.
- Next: everything genuinely open at this point needs real-world TIME rather than more
  building -- see the "Not done yet" list at the top, cleaned up today to reflect
  actual current state rather than a stale pre-session snapshot.

  **Later same day — dashboard polish pass, per Alex's request to make it more user-
  friendly for actually managing parlays**: read the whole `dashboard/app.py` fresh
  (709 lines, grown incrementally all session) before touching anything, rather than
  making more piecemeal edits on top of a session's worth of them.

  **Navigation**: confirmed the installed Streamlit (1.61.1) actually supports
  `st.navigation`/`st.Page`/`st.badge` before designing around them, rather than
  assuming. Restructured from one flat 11-item `st.sidebar.radio` into icon-grouped
  sections (Betting / Research / Admin) via `st.navigation({...})` — each existing
  page's body became a function (`def page_xxx():`), same content, no page dropped.
  Weekly Bet Slip is now the default landing page (`default=True`) instead of the old
  data-status Overview page, since it's the actual "start here" action for what this
  tool is for.

  **The real functional gap, not just visual polish**: Weekly Bet Slip and Parlay
  Builder were two disconnected tools -- a suggestion had no way to become a slip leg
  without manually re-finding and re-adding it. Added `leg_details` to
  `generate_weekly_bet_slip.py`'s candidate dicts (player/stat/choice/line/price/team/
  position_prop/my_prob per underlying leg, including both legs of a 2-leg parlay
  candidate separately) so "➕ Add to slip" on a suggestion pushes real leg(s) directly
  into `st.session_state.slip` -- and since that's the exact same slip the Parlay
  Builder already reads, its existing correlation-detection logic picks up
  multi-leg additions automatically, no new logic needed there. Added
  `st.switch_page` so a "go build your parlay" button actually navigates there (this
  needed the `st.Page` objects to be named module-level variables rather than
  inlined in the `st.navigation` dict, so page functions can reference a specific
  target page by name).

  **Bet Log connected too**: a finished slip can now be sent there as pending bets in
  one click (stake split evenly across legs) instead of retyping everything into the
  manual form. Also switched its results table to `st.data_editor` so results can be
  updated inline (double-click a cell) instead of needing a separate edit flow that
  didn't exist before.

  **Visual language**: replaced emoji-concatenated-into-label-strings with real
  `st.badge` calls (confidence tier, staleness, line-mismatch, injury status) inside
  `st.container(border=True)` cards -- same information, actually scannable instead of
  a wall of text per row.

  **Two real bugs caught live in the browser, not in code review**:
  1. A ternary expression used as a bare statement (`st.badge(...) if cond else
     st.badge(...)`) is syntactically valid Python but not valid Streamlit -- its
     result (a DeltaGenerator) got caught by Streamlit's "magic" auto-display and
     dumped a full class docstring onto the page. Only found by actually clicking
     through to that specific card and reading what rendered, not by reading the
     code (which looked fine). Fixed by converting to a real if/else block.
  2. Consolidating the three near-identical news pages (ESPN/SB Nation/NBC-PFT) into
     one shared `_news_page()` helper introduced a real regression: a generic
     `"source"` column reference leaked SB Nation's raw blog URL into its cards' 
     captions, which the original separate page never showed -- because ESPN's
     "source" column is a meaningful league/team label but SB Nation's column of the
     same name is an internal field, and generalizing across both without checking
     collapsed that distinction. Fixed with an explicit `caption_cols` parameter per
     page instead of a column-presence guess, and verified all three news pages
     against their real live data afterward, not just the one that broke.

  Tested every page live in a browser after the rewrite, not just the two that
  changed most: full flow (generate suggestions → add to slip → switch page → adjust
  → send to Bet Log → confirm it landed) plus a spot-check of all 6 otherwise-
  unchanged pages, zero console errors throughout.
- Next: same open items as the rest of today's entries -- this was correctness/UX
  work, not new modeling. Worth a manual click-through by Alex at some point since
  the automated browser's click-registration has been flaky all session (noted
  several times above) even though everything tested clean here.

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
