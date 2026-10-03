"""
Probability calibration for every prop model (added 2026-10-02).

calibration_report.py has always measured how far each model's raw predict_proba
drifts from real hit rates on its own leave-one-season-out holdouts -- and those
reports regularly flagged OVERCONFIDENT bands -- but nothing ever corrected for it:
the dashboard, the real-line recompute (utils.recompute_probability_for_real_line)
and every EV number downstream consumed the raw, overconfident probability. That
matters doubly here, because the real-line recompute turns P(over proxy) into a
z-score, so an overconfident 0.97 becomes a ~1.9-sigma shift in the implied mean
and inflates every edge built on top of it.

How it works:
  * Each train_*.py already produces pooled OUT-OF-SAMPLE holdout predictions
    (all_probs, all_y) for its calibration report. print_calibration_report() now
    also calls fit_and_save() on those same arrays.
  * fit_and_save() compares three maps -- none, Platt (logistic on logit(p)) and
    isotonic -- by CROSS-FITTED log loss (fit on 4/5 of the holdout rows, scored on
    the other 1/5; rows are in season order so folds ~= seasons), and keeps the
    winner. "none" wins whenever a map doesn't genuinely help, so a model that was
    already calibrated is left alone rather than being bent by noise.
  * The fitted map is saved to models/calibrators/<prop_type>.pkl together with the
    before/after Brier + log loss, which the Model Performance page displays.
  * current_predictions.score_prop() runs every ensemble's mean probability through
    apply_calibrator(); raw_prob_over is kept alongside for comparison.

Known approximation: the holdout predictions come from one model per held-out
season while the deployed artefact is a 100-model bootstrap average, which is a bit
smoother. The map is fitted on the closest out-of-sample signal available.
"""
import functools
import os
import pickle

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CALIBRATOR_DIR = os.path.join(os.path.dirname(__file__), "calibrators")
_EPS = 0.005
_N_FOLDS = 5
MIN_ROWS_TO_CALIBRATE = 300


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)


def _logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


def log_loss(p, y) -> float:
    p, y = _clip(p), np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


class Calibrator:
    """Picklable probability map. method in {"none", "platt", "isotonic"}."""

    def __init__(self, method: str, model=None):
        self.method = method
        self.model = model

    def transform(self, p):
        p = np.asarray(p, dtype=float)
        if self.method == "platt":
            return self.model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return _clip(self.model.predict(p))
        return p


def _fit(method: str, p, y) -> Calibrator:
    if method == "platt":
        m = LogisticRegression(C=1e6, max_iter=1000)
        m.fit(_logit(p).reshape(-1, 1), y)
        return Calibrator("platt", m)
    if method == "isotonic":
        m = IsotonicRegression(out_of_bounds="clip", y_min=_EPS, y_max=1 - _EPS)
        m.fit(p, y)
        return Calibrator("isotonic", m)
    return Calibrator("none")


def _cross_fitted(method: str, p, y) -> np.ndarray:
    """Out-of-fold calibrated probabilities -- contiguous folds, because training
    scripts append holdout predictions season by season."""
    out = np.empty_like(p, dtype=float)
    folds = np.array_split(np.arange(len(p)), _N_FOLDS)
    for idx in folds:
        mask = np.ones(len(p), dtype=bool)
        mask[idx] = False
        if len(np.unique(y[mask])) < 2:
            out[idx] = p[idx]
            continue
        out[idx] = _fit(method, p[mask], y[mask]).transform(p[idx])
    return out


def choose_and_fit(all_probs, all_y) -> tuple[Calibrator, dict]:
    p = np.asarray(all_probs, dtype=float)
    y = np.asarray(all_y, dtype=int)
    metrics = {"n": int(len(p)), "base_rate": float(y.mean()) if len(y) else None,
               "raw_log_loss": log_loss(p, y), "raw_brier": brier(p, y)}
    if len(p) < MIN_ROWS_TO_CALIBRATE or len(np.unique(y)) < 2:
        metrics.update(method="none", cal_log_loss=metrics["raw_log_loss"],
                       cal_brier=metrics["raw_brier"], reason="too few rows")
        return Calibrator("none"), metrics

    scores = {"none": metrics["raw_log_loss"]}
    briers = {"none": metrics["raw_brier"]}
    for method in ("platt", "isotonic"):
        cf = _cross_fitted(method, p, y)
        scores[method] = log_loss(cf, y)
        briers[method] = brier(cf, y)
    # Require a real improvement (0.1% relative) before bending the model's output.
    best = min(scores, key=scores.get)
    if best != "none" and scores[best] > scores["none"] * 0.999:
        best = "none"
    metrics.update(method=best, cal_log_loss=scores[best], cal_brier=briers[best],
                   cv_log_loss=scores, cv_brier=briers)
    return _fit(best, p, y), metrics


def fit_and_save(all_probs, all_y, prop_type: str, reliability: list[dict] | None = None) -> dict:
    cal, metrics = choose_and_fit(all_probs, all_y)
    os.makedirs(CALIBRATOR_DIR, exist_ok=True)
    # Reliability-curve points (10 equal-width bins, raw vs calibrated) for the
    # dashboard's calibration chart -- computed on the holdout set itself.
    p = np.asarray(all_probs, dtype=float)
    y = np.asarray(all_y, dtype=int)
    bins = np.linspace(0, 1, 11)
    curve = []
    cal_p = cal.transform(p)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() >= 20:
            curve.append({"raw_mean": float(p[m].mean()), "cal_mean": float(cal_p[m].mean()),
                          "actual": float(y[m].mean()), "n": int(m.sum())})
    metrics["reliability"] = curve
    with open(os.path.join(CALIBRATOR_DIR, f"{prop_type}.pkl"), "wb") as f:
        pickle.dump({"calibrator": cal, "metrics": metrics}, f)
    print(f"  calibration [{prop_type}]: {metrics['method']}  log loss "
          f"{metrics['raw_log_loss']:.4f} -> {metrics['cal_log_loss']:.4f} (cross-fitted), "
          f"Brier {metrics['raw_brier']:.4f} -> {metrics['cal_brier']:.4f}")
    _load.cache_clear()
    return metrics


@functools.lru_cache(maxsize=None)
def _load(prop_type: str):
    path = os.path.join(CALIBRATOR_DIR, f"{prop_type}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def apply_calibrator(prop_type: str, probs):
    """Calibrated probabilities for this prop model, or the input unchanged when no
    calibrator has been fitted yet (train script not re-run since this was added)."""
    saved = _load(prop_type)
    if saved is None:
        return np.asarray(probs, dtype=float)
    return saved["calibrator"].transform(probs)


def load_all_metrics() -> dict[str, dict]:
    """prop_type -> saved metrics, for the Model Performance page."""
    out = {}
    if not os.path.isdir(CALIBRATOR_DIR):
        return out
    for fn in sorted(os.listdir(CALIBRATOR_DIR)):
        if fn.endswith(".pkl"):
            saved = _load(fn[:-4])
            if saved:
                out[fn[:-4]] = saved["metrics"]
    return out
