"""Classification metrics, threshold selection, and early-warning (lead-time) metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

from afo.evaluation.calibration import brier_score, expected_calibration_error


def _safe(fn, y, p) -> float:
    return float(fn(y, p)) if len(np.unique(y)) > 1 else float("nan")


def auroc(y, p) -> float:
    return _safe(roc_auc_score, y, p)


def auprc(y, p) -> float:
    return _safe(average_precision_score, y, p)


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "threshold": float(threshold),
        "auroc": auroc(y, p),
        "auprc": auprc(y, p),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "brier": brier_score(y, p),
        "ece": expected_calibration_error(y, p),
    }


def choose_threshold(y: np.ndarray, p: np.ndarray, method: str = "max_f1", target_fpr: float = 0.1) -> float:
    """Pick a decision threshold on *validation* data.

    max_f1: threshold maximising F1. target_fpr: lowest threshold whose FPR <= target.
    """
    y = np.asarray(y).astype(int)
    cands = np.unique(p)
    if len(cands) == 1 or y.sum() == 0:
        return float(cands.max() + 1e-9)  # never fires: no usable signal on validation
    if method == "max_f1":
        best_t, best_f1 = cands[-1], -1.0
        for t in cands:
            pred = p >= t
            tp = np.sum(pred & (y == 1))
            fp = np.sum(pred & (y == 0))
            fn = np.sum(~pred & (y == 1))
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > best_f1:
                best_t, best_f1 = t, f1
        return float(best_t)
    if method == "target_fpr":
        fpr, _, thr = roc_curve(y, p)
        ok = thr[fpr <= target_fpr]
        return float(ok[-1]) if len(ok) else float(cands.max() + 1e-9)
    raise ValueError(f"unknown threshold method {method!r}")


def lead_time_metrics(meta: list[dict], p: np.ndarray, threshold: float) -> dict:
    """Per-trajectory early warning: first alert at or before failure onset.

    lead = onset - first_alert_step (0 = flagged at the decision point of the failing action,
    which still precedes its execution).
    """
    by_run: dict[str, list[tuple[int, float, int | None]]] = {}
    for m, s in zip(meta, p):
        by_run.setdefault(m["run_id"], []).append((m["step"], float(s), m["failure_onset"]))
    leads, detected, false_alarm, n_fail, n_clean = [], 0, 0, 0, 0
    for rows in by_run.values():
        rows.sort()
        onset = rows[0][2]
        alerts = [t for t, s, _ in rows if s >= threshold]
        if onset is None:
            n_clean += 1
            false_alarm += bool(alerts)
            continue
        n_fail += 1
        if alerts:
            detected += 1
            leads.append(onset - alerts[0])
    leads_arr = np.asarray(leads)
    return {
        "n_failing_trajectories": n_fail,
        "n_clean_trajectories": n_clean,
        "detection_rate": detected / n_fail if n_fail else float("nan"),
        "early_detection_rate": float(np.sum(leads_arr >= 1)) / n_fail if n_fail else float("nan"),
        "median_lead_steps": float(np.median(leads_arr)) if len(leads_arr) else float("nan"),
        "mean_lead_steps": float(np.mean(leads_arr)) if len(leads_arr) else float("nan"),
        "lead_histogram": {int(k): int(v) for k, v in zip(*np.unique(leads_arr, return_counts=True))} if len(leads_arr) else {},
        "clean_trajectory_false_alarm_rate": false_alarm / n_clean if n_clean else float("nan"),
    }


def auroc_by_steps_to_onset(meta: list[dict], p: np.ndarray, max_k: int = 6) -> list[dict]:
    """Separability of pre-failure states k steps before onset vs. states from clean trajectories."""
    p = np.asarray(p)
    neg = np.asarray([not m["trajectory_fails"] for m in meta])
    stoo = np.asarray([m["steps_to_onset"] if m["steps_to_onset"] is not None else -1 for m in meta])
    out = []
    for k in range(max_k + 1):
        pos = stoo == k
        if pos.sum() == 0 or neg.sum() == 0:
            continue
        y = np.concatenate([np.ones(pos.sum()), np.zeros(neg.sum())])
        s = np.concatenate([p[pos], p[neg]])
        out.append({"k": k, "auroc": auroc(y, s), "n_pos": int(pos.sum()), "n_neg": int(neg.sum())})
    return out
