"""Distribution-shift evaluation (RQ5): leave-one-environment-out.

For each environment category E: fit monitors on the train split of the other two
categories, choose C and thresholds on their val split, and test on *all*
trajectories of E. This is a stronger shift than the main unseen-template split.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from afo.collection.storage import ExperimentStore
from afo.evaluation.metrics import auroc, binary_metrics, choose_threshold
from afo.evaluation.statistics import cluster_bootstrap
from afo.monitors.splits import split_indices
from afo.monitors.training import fit_monitors, get_splits, load_dataset
from afo.utils import dump_json


def leave_one_environment_out(store: ExperimentStore, log: Callable[[str], None] = print) -> dict:
    cfg = store.config
    ecfg = cfg["evaluation"]
    h = int(ecfg["primary_horizon"])
    ds, trajs = load_dataset(store)
    idx = split_indices(ds.meta, get_splits(store, trajs))
    envs = ds.column("environment")
    split_of = np.empty(len(ds), dtype=object)
    for s, rows in idx.items():
        split_of[rows] = s
    out = {"horizon": h, "method": "leave-one-environment-out", "held_out": {}}
    for env in sorted(set(envs)):
        train = ds.subset(np.flatnonzero((envs != env) & (split_of == "train")))
        val = ds.subset(np.flatnonzero((envs != env) & (split_of == "val")))
        test = ds.subset(np.flatnonzero(envs == env))
        monitors, _ = fit_monitors(train, val, h, cfg)
        y = test.labels[h]
        groups = test.column(ecfg["bootstrap_group"])
        res = {}
        for name, m in monitors.items():
            thr = choose_threshold(val.labels[h], m.score(val), ecfg["threshold_method"], ecfg["target_fpr"])
            p = m.score(test)
            r = binary_metrics(y, p, thr)
            r["auroc_ci"] = cluster_bootstrap(lambda i: auroc(y[i], p[i]), groups, n_boot=ecfg["bootstrap"],
                                              ci=ecfg["ci"], seed=cfg["seed"])
            r["layer"] = getattr(m, "layer_id", None)
            res[name] = r
        out["held_out"][env] = res
        log(f"[generalize] held out {env}: " + ", ".join(f"{n}={r['auroc']:.3f}" for n, r in res.items()))
    dump_json(store.path("eval", "generalization.json"), out)
    return out
