"""Grouped (cluster) bootstrap confidence intervals.

Rows from the same trajectory, and trajectories from the same template, are
correlated. Resampling whole groups (default: template) keeps CIs honest; with few
test templates the intervals are wide, which is the correct conclusion.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from afo.utils import rng_for


def _group_index(groups: np.ndarray) -> list[np.ndarray]:
    _, inv = np.unique(groups, return_inverse=True)
    return [np.flatnonzero(inv == g) for g in range(inv.max() + 1)] if len(groups) else []


def cluster_bootstrap(stat: Callable[[np.ndarray], float], groups: np.ndarray, n_boot: int = 1000,
                      ci: float = 0.95, seed: int = 0) -> dict:
    """CI for ``stat(row_indices)`` by resampling groups with replacement."""
    members = _group_index(np.asarray(groups))
    point = stat(np.arange(len(groups)))
    if len(members) < 2 or n_boot <= 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_groups": len(members), "n_valid": 0}
    g = rng_for("bootstrap", seed)
    vals = []
    for _ in range(n_boot):
        pick = g.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        v = stat(idx)
        if v is not None and not np.isnan(v):
            vals.append(v)
    a = (1 - ci) / 2
    lo, hi = (np.quantile(vals, [a, 1 - a]) if vals else (np.nan, np.nan))
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_groups": len(members), "n_valid": len(vals)}


def paired_difference(stat_a: Callable[[np.ndarray], float], stat_b: Callable[[np.ndarray], float],
                      groups: np.ndarray, n_boot: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """CI for stat_a - stat_b computed on the *same* resampled rows (paired comparison)."""
    def diff(idx):
        a, b = stat_a(idx), stat_b(idx)
        return a - b if not (np.isnan(a) or np.isnan(b)) else np.nan

    out = cluster_bootstrap(diff, groups, n_boot, ci, seed)
    out["excludes_zero"] = bool(not np.isnan(out["lo"]) and (out["lo"] > 0 or out["hi"] < 0))
    return out
