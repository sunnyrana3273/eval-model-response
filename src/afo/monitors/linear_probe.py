"""Activation-based linear probe, with a validation-based layer sweep."""

from __future__ import annotations

import numpy as np

from afo.monitors.base import LogisticMonitor
from afo.monitors.features import StepDataset


class LinearProbeMonitor(LogisticMonitor):
    name = "probe"
    uses_activations = True


def sweep_layers(train: StepDataset, y_train: np.ndarray, val: StepDataset, y_val: np.ndarray,
                 cfg: dict) -> tuple[LinearProbeMonitor, list[dict]]:
    """Fit one probe per stored layer; return the best by validation AUROC and the full sweep."""
    sweep, best = [], None
    for i, layer_id in enumerate(train.layers):
        probe = LinearProbeMonitor(layer=i)
        probe.layer_id = layer_id
        probe.fit(train, y_train, val, y_val, cfg)
        sweep.append({"layer_index": i, "layer": layer_id, "val_auroc": probe.val_auroc, "C": probe.C, "probe": probe})
        if best is None or probe.val_auroc > best.val_auroc:
            best = probe
    return best, sweep
