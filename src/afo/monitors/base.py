"""Monitor base classes.

A monitor maps decision-point inputs (behavioural features and/or activations) to a
failure probability. Preprocessing lives inside an sklearn Pipeline that is fit on
training rows only; the regularisation strength is chosen by validation AUROC.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from afo.monitors.features import StepDataset, behavioral_features


class Monitor:
    name = "monitor"
    uses_behavioral = False
    uses_activations = False

    def __init__(self, layer: int | None = None):
        self.layer = layer          # index into the dataset's stored layers
        self.layer_id: int | None = None  # model layer number, for reporting
        self.selection: dict = {}

    # -- inputs ---------------------------------------------------------------------------
    def inputs(self, ds: StepDataset) -> np.ndarray:
        parts = []
        if self.uses_behavioral:
            parts.append(ds.behavioral)
        if self.uses_activations:
            if ds.activations is None:
                raise ValueError(f"{self.name} needs activations but the dataset has none")
            parts.append(ds.activations[:, self.layer, :])
        return np.concatenate(parts, axis=1) if parts else np.zeros((len(ds), 1), np.float32)

    def step_inputs(self, prefix: list[dict], observation: str, activations: np.ndarray | None,
                    max_steps: int) -> np.ndarray:
        parts = []
        if self.uses_behavioral:
            parts.append(behavioral_features(prefix, observation, max_steps))
        if self.uses_activations:
            if activations is None:
                raise ValueError(f"{self.name} needs activations at runtime")
            parts.append(activations[self.layer])
        return np.concatenate(parts)[None, :] if parts else np.zeros((1, 1), np.float32)

    # -- API ------------------------------------------------------------------------------
    def fit(self, train: StepDataset, y_train: np.ndarray, val: StepDataset, y_val: np.ndarray,
            cfg: dict) -> "Monitor":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score(self, ds: StepDataset) -> np.ndarray:
        return self.predict_proba(self.inputs(ds))

    def score_step(self, prefix: list[dict], observation: str, activations: np.ndarray | None,
                   max_steps: int) -> float:
        return float(self.predict_proba(self.step_inputs(prefix, observation, activations, max_steps))[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "Monitor":
        with Path(path).open("rb") as f:
            return pickle.load(f)

    def describe(self) -> dict:
        return {"name": self.name, "layer": self.layer_id, **self.selection}


class MajorityMonitor(Monitor):
    """Predicts the training base rate for every row (AUROC 0.5 by construction)."""

    name = "majority"

    def fit(self, train, y_train, val, y_val, cfg):
        self.p = float(np.mean(y_train)) if len(y_train) else 0.0
        self.majority_label = int(self.p >= 0.5)
        self.selection = {"train_prevalence": self.p}
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.p)


class LogisticMonitor(Monitor):
    """Standardise -> (optional PCA) -> L2 logistic regression, C chosen on validation AUROC."""

    def _pipeline(self, C: float, cfg: dict, n_features: int) -> Pipeline:
        steps = [("scale", StandardScaler())]
        k = cfg.get("probe_pca") if self.uses_activations else None
        if k:
            steps.append(("pca", PCA(n_components=min(k, n_features), random_state=0)))
        steps.append(("clf", LogisticRegression(C=C, class_weight=cfg.get("class_weight"), max_iter=5000)))
        return Pipeline(steps)

    def fit(self, train, y_train, val, y_val, cfg):
        Xtr, Xva = self.inputs(train), self.inputs(val)
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"{self.name}: training labels contain a single class")
        best = None
        scores = {}
        for C in cfg["c_grid"]:
            pipe = self._pipeline(C, cfg, Xtr.shape[1]).fit(Xtr, y_train)
            auc = roc_auc_score(y_val, pipe.predict_proba(Xva)[:, 1]) if len(np.unique(y_val)) > 1 else float("nan")
            scores[C] = auc
            if best is None or (not np.isnan(auc) and auc > best[1]):
                best = (C, auc, pipe)
        self.C, self.val_auroc, self.pipe = best
        self.selection = {"C": self.C, "val_auroc": self.val_auroc,
                          "val_auroc_by_C": {str(k): v for k, v in scores.items()}}
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)[:, 1]
