"""
Evaluation metrics for climate forecasting.
Standard metrics required for scientific publication.
"""

import numpy as np
import pandas as pd
from typing import Optional


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def acc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    climatology: np.ndarray,
) -> float:
    """
    Anomaly Correlation Coefficient — standard in climate forecasting papers.
    Measures correlation between forecast anomaly and observed anomaly.
    """
    a = y_true - climatology
    f = y_pred - climatology
    num = np.sum(f * a)
    den = np.sqrt(np.sum(f ** 2) * np.sum(a ** 2))
    if den == 0:
        return 0.0
    return float(num / den)


def brier_skill_score(
    y_true_binary: np.ndarray,
    y_pred_prob: np.ndarray,
    climatology_freq: float,
) -> float:
    """
    Brier Skill Score — measures probabilistic skill above climatology.
    BSS > 0: better than climatology. BSS = 1: perfect.
    """
    bs = np.mean((y_pred_prob - y_true_binary) ** 2)
    bs_clim = np.mean((climatology_freq - y_true_binary) ** 2)
    if bs_clim == 0:
        return 0.0
    return float(1 - bs / bs_clim)


def heidke_skill_score(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
) -> float:
    """
    HSS — skill score for categorical/binary events (extremes).
    HSS = 0: no skill. HSS = 1: perfect.
    """
    tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
    tn = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
    fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
    n = tp + tn + fp + fn
    if n == 0:
        return 0.0
    expected = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / n
    denom = n - expected
    if denom == 0:
        return 0.0
    return float((tp + tn - expected) / denom)


def full_scorecard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    climatology: np.ndarray,
    threshold: Optional[float] = None,
) -> dict:
    """Return all metrics in a single dict. Ready for DataFrame export."""
    scores = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "ACC": acc(y_true, y_pred, climatology),
        "MAE_clim": mae(y_true, climatology),
        "RMSE_clim": rmse(y_true, climatology),
    }
    if threshold is not None:
        y_true_bin = (y_true > threshold).astype(int)
        y_pred_bin = (y_pred > threshold).astype(int)
        clim_freq = float(np.mean(y_true_bin))
        y_pred_prob = y_pred_bin.astype(float)
        scores["BSS"] = brier_skill_score(y_true_bin, y_pred_prob, clim_freq)
        scores["HSS"] = heidke_skill_score(y_true_bin, y_pred_bin)
    return scores
