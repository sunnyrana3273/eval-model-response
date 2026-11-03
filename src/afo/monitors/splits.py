"""Leakage-resistant splits: whole task families go to exactly one split.

All variants (strength, placement, difficulty, seed) of a template, and all
templates sharing a family, land in the same split, so test rows always come from
unseen task families and no trajectory contributes steps to two splits.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from afo.utils import rng_for

SPLITS = ("train", "val", "test")


def assign_family_splits(families_by_env: dict[str, list[str]], fractions: dict[str, float], seed: int) -> dict[str, str]:
    """Stratified by environment so every split contains every environment category."""
    out: dict[str, str] = {}
    for env in sorted(families_by_env):
        fams = sorted(set(families_by_env[env]))
        order = rng_for("split", seed, env).permutation(len(fams))
        fams = [fams[i] for i in order]
        n = len(fams)
        n_val = max(1, round(fractions["val"] * n))
        n_test = max(1, round(fractions["test"] * n))
        n_train = n - n_val - n_test
        if n_train < 1:
            raise ValueError(f"environment {env} has too few families ({n}) for a 3-way split")
        for i, f in enumerate(fams):
            out[f] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return out


def splits_from_trajectories(trajectories: list[dict], fractions: dict[str, float], seed: int) -> dict[str, str]:
    by_env: dict[str, list[str]] = defaultdict(list)
    for t in trajectories:
        by_env[t["task"]["environment"]].append(t["task"]["family"])
    return assign_family_splits(by_env, fractions, seed)


def split_indices(meta: list[dict], family_split: dict[str, str]) -> dict[str, np.ndarray]:
    labels = np.asarray([family_split[m["family"]] for m in meta])
    return {s: np.flatnonzero(labels == s) for s in SPLITS}


def check_no_leakage(meta: list[dict], idx: dict[str, np.ndarray]) -> None:
    """Raise if any family, template or trajectory appears in more than one split."""
    for key in ("family", "template_id", "run_id"):
        seen: dict[str, str] = {}
        for split, rows in idx.items():
            for i in rows:
                v = meta[i][key]
                if seen.setdefault(v, split) != split:
                    raise AssertionError(f"leakage: {key}={v} in both {seen[v]} and {split}")
