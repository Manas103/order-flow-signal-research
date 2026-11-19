"""Out-of-sample evaluation statistics for the signal study.

Every test here treats a (ticker, date) session as the unit of independence.
Event-level rows are not independent: a horizon-h label overlaps the next h-1
labels, so an event-level standard error would be too small by a large and
unknown factor. Sessions are separately simulated and never share a label (the
features and labels are PARTITION BY ticker, date), so per-session aggregates
are a sound basis for a t-test and for a bootstrap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps


def r2_oos(y: np.ndarray, pred: np.ndarray, mean_ref: float | None = None) -> float:
    """1 - SSE/SST. With mean_ref (the training mean), SST is measured around the
    forecast a no-skill model would actually make, the usual out-of-sample convention."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ref = y.mean() if mean_ref is None else mean_ref
    sst = float(np.sum((y - ref) ** 2))
    return float("nan") if sst == 0 else 1.0 - float(np.sum((y - pred) ** 2)) / sst


@dataclass(frozen=True)
class DMResult:
    mean_loss_diff: float   # mean over sessions of (loss_b - loss_a); > 0 means model a is better
    t_stat: float
    p_value: float          # two-sided
    n_sessions: int


def diebold_mariano_sessions(y, pred_a, pred_b, session) -> DMResult:
    """Diebold-Mariano comparison of squared-error loss, with sessions as clusters.

    d_s = mean over rows in session s of (e_b^2 - e_a^2). The statistic is the
    one-sample t-test of d_s against zero across sessions, which is robust to
    any within-session autocorrelation (overlapping horizons included).
    """
    df = pd.DataFrame({
        "s": np.asarray(session),
        "d": (np.asarray(y, float) - np.asarray(pred_b, float)) ** 2
             - (np.asarray(y, float) - np.asarray(pred_a, float)) ** 2,
    })
    d = df.groupby("s", sort=False)["d"].mean().to_numpy()
    n = len(d)
    if n < 2 or np.allclose(d, d[0]):
        return DMResult(float(d.mean()) if n else float("nan"), 0.0 if n else float("nan"), 1.0, n)
    t = float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))
    p = float(2 * sps.t.sf(abs(t), df=n - 1))
    return DMResult(float(d.mean()), t, p, n)


def session_bootstrap_r2_diff(y, pred_a, pred_b, session, mean_ref: float,
                              n_boot: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    """Point estimate and 95% percentile interval of R2(a) - R2(b), resampling whole sessions.

    Uses per-session sufficient statistics (SSE_a, SSE_b, SST) so each resample is
    O(sessions), not O(rows)."""
    y = np.asarray(y, float)
    ea = (y - np.asarray(pred_a, float)) ** 2
    eb = (y - np.asarray(pred_b, float)) ** 2
    et = (y - mean_ref) ** 2
    g = pd.DataFrame({"s": np.asarray(session), "a": ea, "b": eb, "t": et}).groupby("s", sort=False).sum()
    A, B, T = g["a"].to_numpy(), g["b"].to_numpy(), g["t"].to_numpy()
    point = float((B.sum() - A.sum()) / T.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(T), size=(n_boot, len(T)))
    diffs = (B[idx].sum(1) - A[idx].sum(1)) / T[idx].sum(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


@dataclass(frozen=True)
class ICResult:
    mean_ic: float      # mean over sessions of the Spearman rank correlation
    t_stat: float
    p_value: float
    n_sessions: int


def information_coefficient(y, pred, session) -> ICResult:
    """Per-session Spearman correlation between forecast and realized label, then a
    t-test of the mean across sessions (sessions with a constant forecast are skipped)."""
    df = pd.DataFrame({"s": np.asarray(session), "y": np.asarray(y, float), "p": np.asarray(pred, float)})
    ics = []
    for _, g in df.groupby("s", sort=False):
        if g["p"].nunique() > 1 and g["y"].nunique() > 1:
            ics.append(sps.spearmanr(g["p"], g["y"]).statistic)
    ics = np.asarray(ics, float)
    n = len(ics)
    if n < 2:
        return ICResult(float(ics.mean()) if n else float("nan"), float("nan"), float("nan"), n)
    t = float(ics.mean() / (ics.std(ddof=1) / np.sqrt(n)))
    return ICResult(float(ics.mean()), t, float(2 * sps.t.sf(abs(t), df=n - 1)), n)


def hit_rate(y, pred) -> float:
    """Share of rows with a nonzero realized move whose sign the forecast got right."""
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    m = (y != 0) & (pred != 0)
    return float(np.mean(np.sign(y[m]) == np.sign(pred[m]))) if m.any() else float("nan")


def walk_forward_folds(dates: list, min_train: int, embargo: int = 1):
    """Expanding-window folds over sorted dates: train on dates[:t-embargo], skip the
    embargo day(s), test on dates[t]. Yields (train_dates, embargo_dates, test_date)."""
    dates = sorted(dates)
    for t in range(min_train + embargo, len(dates)):
        yield list(dates[:t - embargo]), list(dates[t - embargo:t]), dates[t]
