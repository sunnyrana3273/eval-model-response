"""Calibration metrics."""

from __future__ import annotations

import numpy as np


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Equal-width bins. Returns per-bin mean predicted, observed frequency and counts."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    mean_p, frac_pos, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        counts.append(int(m.sum()))
        mean_p.append(float(p[m].mean()) if m.any() else None)
        frac_pos.append(float(y[m].mean()) if m.any() else None)
    return {"edges": edges.tolist(), "mean_predicted": mean_p, "fraction_positive": frac_pos, "counts": counts}


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    if not len(y):
        return float("nan")
    curve = reliability_curve(y, p, n_bins)
    n = len(y)
    return float(sum(c / n * abs(mp - fp) for mp, fp, c in
                     zip(curve["mean_predicted"], curve["fraction_positive"], curve["counts"]) if c))
