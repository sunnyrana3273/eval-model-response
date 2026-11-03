"""Evaluation stage: score the held-out test split and write metrics, predictions and plots.

Thresholds come from the validation split (fixed at training time). The test split
contains only task families never seen during training or threshold selection.
"""

from __future__ import annotations

from typing import Callable

from afo.collection.storage import ExperimentStore
from afo.evaluation import plots
from afo.evaluation.calibration import reliability_curve
from afo.evaluation.metrics import (
    auprc,
    auroc,
    auroc_by_steps_to_onset,
    binary_metrics,
    brier_score,
    lead_time_metrics,
)
from afo.evaluation.statistics import cluster_bootstrap, paired_difference
from afo.monitors.base import Monitor
from afo.monitors.splits import split_indices
from afo.monitors.training import get_splits, load_dataset
from afo.utils import dump_json, load_json, write_jsonl

COMPARISONS = [("behavioral", "majority"), ("probe", "behavioral"), ("combined", "behavioral"),
               ("combined", "probe")]


def evaluate_monitors(store: ExperimentStore, log: Callable[[str], None] = print) -> dict:
    cfg = store.config
    ecfg = cfg["evaluation"]
    index = load_json(store.root / "monitors" / "index.json")
    ds, trajs = load_dataset(store)
    idx = split_indices(ds.meta, get_splits(store, trajs))
    test = ds.subset(idx["test"])
    groups = test.column(ecfg["bootstrap_group"])
    boot = dict(n_boot=ecfg["bootstrap"], ci=ecfg["ci"], seed=cfg["seed"])

    out: dict = {
        "experiment_id": store.experiment_id,
        "test_split": {"rows": len(test), "trajectories": len(set(test.column("run_id"))),
                       "templates": sorted(set(test.column("template_id"))),
                       "bootstrap_group": ecfg["bootstrap_group"], "n_groups": len(set(groups))},
        "threshold_method": index["threshold_method"],
        "results": {}, "by_category": {}, "by_steps_to_onset": {}, "lead_time": {}, "paired_differences": {},
        "layer_sweep": {}, "calibration": {},
    }
    predictions = []
    plot_dir = store.root / "eval" / "plots"

    for h_str, entries in index["horizons"].items():
        h = int(h_str)
        y = test.labels[h]
        scores, res, by_cat, stoo, leads, calib = {}, {}, {}, {}, {}, {}
        for name, e in entries.items():
            mon = Monitor.load(store.root / e["path"])
            p = mon.score(test)
            scores[name] = p
            thr = e["threshold"]
            m = binary_metrics(y, p, thr)
            m["auroc_ci"] = cluster_bootstrap(lambda i: auroc(y[i], p[i]), groups, **boot)
            m["auprc_ci"] = cluster_bootstrap(lambda i: auprc(y[i], p[i]), groups, **boot)
            m["brier_ci"] = cluster_bootstrap(lambda i: brier_score(y[i], p[i]), groups, **boot)
            m["layer"] = e.get("layer")
            res[name] = m
            envs = test.column("environment")
            by_cat[name] = {env: binary_metrics(y[envs == env], p[envs == env], thr) for env in sorted(set(envs))}
            stoo[name] = auroc_by_steps_to_onset(test.meta, p)
            leads[name] = lead_time_metrics(test.meta, p, thr)
            calib[name] = reliability_curve(y, p)
            # Predictions on every split, tagged, for the trajectory explorer.
            p_all = mon.score(ds)
            for split, rows in idx.items():
                for i in rows:
                    predictions.append({"run_id": ds.meta[i]["run_id"], "step": ds.meta[i]["step"], "split": split,
                                        "horizon": h, "monitor": name, "score": float(p_all[i]),
                                        "threshold": thr, "label": int(ds.labels[h][i])})
        diffs = {}
        for a, b in COMPARISONS:
            if a in scores and b in scores:
                pa, pb = scores[a], scores[b]
                diffs[f"{a}-{b}"] = paired_difference(lambda i: auroc(y[i], pa[i]), lambda i: auroc(y[i], pb[i]),
                                                      groups, **boot)
        sweep = []
        for r in index["layer_sweep"].get(h_str, []):
            probe = Monitor.load(store.root / r["path"])
            sweep.append({**r, "test_auroc": auroc(y, probe.score(test))})
        out["results"][h_str] = res
        out["by_category"][h_str] = by_cat
        out["by_steps_to_onset"][h_str] = stoo
        out["lead_time"][h_str] = leads
        out["paired_differences"][h_str] = diffs
        out["layer_sweep"][h_str] = sweep
        out["calibration"][h_str] = calib

        plots.roc_pr(y, scores, h, plot_dir / f"roc_pr_h{h}.png")
        plots.calibration(calib, h, plot_dir / f"calibration_h{h}.png")
        plots.by_steps_to_onset(stoo, h, plot_dir / f"early_warning_h{h}.png")
        if sweep:
            plots.layer_sweep(sweep, h, plot_dir / f"layer_sweep_h{h}.png")
        log(f"[eval] h={h} test AUROC: " + ", ".join(
            f"{n}={r['auroc']:.3f} [{r['auroc_ci']['lo']:.3f},{r['auroc_ci']['hi']:.3f}]" for n, r in res.items()))

    names = [n for n in ("majority", "behavioral", "probe", "combined") if n in index["horizons"][next(iter(index["horizons"]))]]
    plots.by_horizon(out["results"], names, plot_dir / "auroc_by_horizon.png")
    dump_json(store.path("eval", "metrics.json"), out)
    write_jsonl(store.path("eval", "predictions.jsonl"), predictions)
    return out
