"""
Statistical significance tests for forecast comparison.
Required for scientific publication.
"""

import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano_test(
    errors_model1: np.ndarray,
    errors_model2: np.ndarray,
    h: int = 1,
) -> dict:
    """
    Diebold-Mariano test for equal predictive accuracy.
    H0: model1 and model2 have equal accuracy.
    H1: they differ.

    errors = actual - predicted (or absolute errors, depending on loss function)
    h = forecast horizon (affects autocorrelation correction)
    """
    d = errors_model1 ** 2 - errors_model2 ** 2
    n = len(d)
    d_mean = np.mean(d)

    # Newey-West HAC variance for autocorrelation correction at horizon h
    gamma_0 = np.var(d, ddof=1)
    autocovariances = [gamma_0]
    for lag in range(1, h):
        cov = np.cov(d[lag:], d[:-lag])[0, 1] if lag < n else 0.0
        autocovariances.append(cov)

    var_d = (1 / n) * (autocovariances[0] + 2 * sum(autocovariances[1:]))
    dm_stat = d_mean / np.sqrt(max(var_d, 1e-12))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "DM_stat": round(float(dm_stat), 4),
        "p_value": round(float(p_value), 4),
        "significant_at_05": bool(p_value < 0.05),
        "significant_at_01": bool(p_value < 0.01),
        "interpretation": _interpret_dm(dm_stat, p_value),
    }


def _interpret_dm(stat: float, p: float) -> str:
    if p >= 0.05:
        return "No significant difference between models (p >= 0.05)"
    if stat > 0:
        return f"Model 2 significantly more accurate (p={p:.3f})"
    return f"Model 1 significantly more accurate (p={p:.3f})"
