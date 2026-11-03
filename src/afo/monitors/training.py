"""Monitor training stage: dataset -> family splits -> fitted monitors + validation thresholds."""

from __future__ import annotations

from typing import Callable

from afo.collection.storage import ExperimentStore
from afo.evaluation.metrics import choose_threshold
from afo.monitors.base import MajorityMonitor, Monitor
from afo.monitors.behavioral import BehavioralMonitor
from afo.monitors.combined import CombinedMonitor
from afo.monitors.features import StepDataset, build_step_dataset
from afo.monitors.linear_probe import sweep_layers
from afo.monitors.splits import check_no_leakage, split_indices, splits_from_trajectories
from afo.utils import dump_json, load_json


def load_dataset(store: ExperimentStore) -> tuple[StepDataset, list[dict]]:
    cfg = store.config
    trajs = store.load_trajectories()
    use_acts = bool(cfg["collection"]["activations"]["enabled"])
    ds = build_step_dataset(trajs, store, cfg["monitors"]["horizons"], cfg["agent"]["max_steps"], use_acts)
    return ds, trajs


def get_splits(store: ExperimentStore, trajs: list[dict]) -> dict[str, str]:
    path = store.root / "splits.json"
    if path.exists():
        return load_json(path)["family_split"]
    cfg = store.config
    fam = splits_from_trajectories(trajs, cfg["split"], cfg["split"]["seed"])
    dump_json(path, {"family_split": fam, "method": "family-level, stratified by environment",
                     "fractions": cfg["split"]})
    return fam


def fit_monitors(train: StepDataset, val: StepDataset, horizon: int, cfg: dict) -> tuple[dict[str, Monitor], list[dict]]:
    mcfg = cfg["monitors"]
    ytr, yva = train.labels[horizon], val.labels[horizon]
    monitors: dict[str, Monitor] = {
        "majority": MajorityMonitor().fit(train, ytr, val, yva, mcfg),
        "behavioral": BehavioralMonitor().fit(train, ytr, val, yva, mcfg),
    }
    sweep: list[dict] = []
    if train.activations is not None:
        probe, sweep = sweep_layers(train, ytr, val, yva, mcfg)
        monitors["probe"] = probe
        combined = CombinedMonitor(layer=probe.layer)
        combined.layer_id = probe.layer_id
        monitors["combined"] = combined.fit(train, ytr, val, yva, mcfg)
    return monitors, sweep


def train_monitors(store: ExperimentStore, log: Callable[[str], None] = print) -> dict:
    cfg = store.config
    ecfg = cfg["evaluation"]
    ds, trajs = load_dataset(store)
    fam = get_splits(store, trajs)
    idx = split_indices(ds.meta, fam)
    check_no_leakage(ds.meta, idx)
    train, val = ds.subset(idx["train"]), ds.subset(idx["val"])
    ds.save(store.path("features", "dataset.npz"))

    summary = {s: {"rows": int(len(i)), "trajectories": len({ds.meta[j]["run_id"] for j in i}),
                   "templates": sorted({ds.meta[j]["template_id"] for j in i}),
                   "prevalence": {str(h): float(ds.labels[h][i].mean()) if len(i) else None
                                  for h in cfg["monitors"]["horizons"]}}
               for s, i in idx.items()}
    dump_json(store.path("features", "summary.json"), {"splits": summary, "features": ds.feature_names,
                                                        "layers": ds.layers})

    index: dict = {"horizons": {}, "layer_sweep": {}}
    for h in cfg["monitors"]["horizons"]:
        monitors, sweep = fit_monitors(train, val, h, cfg)
        entry = {}
        for name, m in monitors.items():
            p_val = m.score(val)
            thr = choose_threshold(val.labels[h], p_val, ecfg["threshold_method"], ecfg["target_fpr"])
            path = store.path("monitors", f"{name}_h{h}.pkl")
            m.threshold = thr
            m.save(path)
            entry[name] = {"path": str(path.relative_to(store.root)), "threshold": thr, **m.describe()}
        index["horizons"][str(h)] = entry
        rows = []
        for r in sweep:
            p = store.path("monitors", "layers", f"probe_L{r['layer']}_h{h}.pkl")
            r["probe"].save(p)
            rows.append({k: v for k, v in r.items() if k != "probe"} | {"path": str(p.relative_to(store.root))})
        index["layer_sweep"][str(h)] = rows
        log(f"[train] h={h}: " + ", ".join(
            f"{n} val_auroc={e.get('val_auroc', float('nan')):.3f}" for n, e in entry.items() if n != "majority"))
    index["threshold_method"] = {"method": ecfg["threshold_method"], "target_fpr": ecfg["target_fpr"],
                                 "fit_on": "validation split"}
    dump_json(store.path("monitors", "index.json"), index)
    return index


def load_monitor(store: ExperimentStore, name: str, horizon: int) -> tuple[Monitor, float]:
    entry = load_json(store.root / "monitors" / "index.json")["horizons"][str(horizon)][name]
    return Monitor.load(store.root / entry["path"]), float(entry["threshold"])


__all__ = ["fit_monitors", "get_splits", "load_dataset", "load_monitor", "train_monitors"]
