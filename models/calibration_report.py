"""
Shared calibration diagnostic, called from every train_*.py script right after its
leave-one-season-out holdout loop produces pooled (all_probs, all_y) arrays.

Motivated by a real question: does a model's raw predicted probability mean what it
says, especially at the extreme tail (predictions the dashboard rounds to a "100%"
confidence badge)? Every training script in this project uses RAW, UNCALIBRATED
classifier output (LogisticRegression / XGBoost predict_proba) -- no Platt scaling or
isotonic regression is applied anywhere. Raw probabilities from both model families
are known to often be overconfident at the extremes, so this checks empirically:
within each raw-probability band, what fraction of holdout predictions were ACTUALLY
correct, not just what the model claims. This doesn't refit anything -- it's a
read-only check on predictions each script already computed for its own accuracy
report.
"""
import numpy as np

# Matches the app's own confidence badge bands (dashboard/app.py's _confidence_badge:
# green >=0.4, orange >=0.2, gray below) plus finer bands at the very extreme tail,
# since that's specifically where a raw probability can visually round to a "100%"
# badge in the dashboard (side_prob >= 0.995).
CALIBRATION_BANDS = [
    (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90),
    (0.90, 0.95), (0.95, 0.98), (0.98, 0.995), (0.995, 1.00),
]

MIN_N_TO_FLAG = 5           # bands with fewer holdout samples than this are too noisy to judge
OVERCONFIDENT_GAP_PT = 8.0  # actual hit rate this many points below the band's midpoint gets flagged


def print_calibration_report(all_probs, all_y, label: str) -> list[dict]:
    """all_probs/all_y are the SAME pooled holdout arrays each training script already
    builds (raw predicted P(over), actual outcome over 2020-2024 leave-one-season-out
    holdouts). Returns the per-band rows (for cross-model summarization) as well as
    printing a human-readable table."""
    all_probs = np.asarray(all_probs)
    all_y = np.asarray(all_y)

    # Fold to "the side the model actually favored" -- a raw prob of 0.02 means the
    # model favored "under" with 98% implied confidence, same information content as
    # a raw prob of 0.98 favoring "over" with 98% implied confidence. Calibration is
    # about whichever side was picked, not literally P(over).
    side_prob = np.where(all_probs >= 0.5, all_probs, 1 - all_probs)
    correct = np.where(all_probs >= 0.5, all_y == 1, all_y == 0)

    print(f"\n  --- Calibration check: {label} ({len(all_probs)} pooled holdout predictions) ---")
    print(f"  {'Implied conf.':<18}{'n':>6}  {'Actual hit rate':>16}  {'Gap vs midpoint':>16}")
    rows = []
    for lo, hi in CALIBRATION_BANDS:
        mask = (side_prob >= lo) & (side_prob < hi if hi < 1.0 else side_prob <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        actual = float(correct[mask].mean())
        implied_mid = (lo + hi) / 2
        gap = actual - implied_mid
        flag = ""
        if n >= MIN_N_TO_FLAG and gap * 100 < -OVERCONFIDENT_GAP_PT:
            flag = "  <-- OVERCONFIDENT"
        elif n < MIN_N_TO_FLAG:
            flag = "  (n too small to judge)"
        print(f"  [{lo:.3f}, {hi:.3f})  {n:>6}  {actual*100:>14.1f}%  {gap*100:>+14.1f}pt{flag}")
        rows.append({"label": label, "band_lo": lo, "band_hi": hi, "n": n,
                      "actual_hit_rate": actual, "implied_mid": implied_mid, "gap": gap})
    return rows
