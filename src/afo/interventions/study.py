"""Intervention study (RQ4): replay test-split tasks under each intervention condition.

Conditions are ``<intervention>:<trigger>`` (e.g. ``gate:monitor``) or ``none``.
``random`` triggers fire at the per-step rate observed for the matching monitor
condition (non-predictive, rate-matched baseline). ``always`` shows the degenerate
"refuse everything" trade-off. Runs are paired across conditions by (instance, seed).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from afo.agents.model_adapter import build_adapter
from afo.agents.runner import AgentRunner
from afo.collection.collect import run_id_for
from afo.collection.storage import ExperimentStore
from afo.evaluation import plots
from afo.evaluation.statistics import cluster_bootstrap
from afo.interventions.approval_gate import ApprovalGate
from afo.interventions.base import AlwaysTrigger, Intervention, MonitorTrigger, RandomTrigger
from afo.interventions.reconsideration import ReconsiderIntervention
from afo.interventions.warning import WarningIntervention
from afo.evaluation.metrics import choose_threshold
from afo.monitors.splits import split_indices
from afo.monitors.training import get_splits, load_dataset, load_monitor
from afo.utils import dump_json

INTERVENTIONS = {"warning": WarningIntervention, "gate": ApprovalGate, "reconsider": ReconsiderIntervention}


def build_intervention(spec: str, monitor=None, threshold: float | None = None,
                       random_rate: float | None = None) -> Intervention | None:
    if spec == "none":
        return None
    kind, trig = spec.split(":")
    if trig == "monitor":
        trigger = MonitorTrigger(monitor, threshold)
    elif trig == "random":
        trigger = RandomTrigger(random_rate)
    elif trig == "always":
        trigger = AlwaysTrigger()
    else:
        raise ValueError(f"unknown trigger {trig!r}")
    return INTERVENTIONS[kind](trigger)


def _rate(trajs: list[dict]) -> float:
    steps = [s for t in trajs for s in t["steps"]]
    return float(np.mean([bool(s["intervention"] and s["intervention"]["triggered"]) for s in steps])) if steps else 0.0


def summarise(trajs: list[dict], baseline: dict[tuple, dict] | None, cfg: dict) -> dict:
    ecfg = cfg["evaluation"]
    viol = np.asarray([t["outcome"]["safety_violation"] for t in trajs], float)
    succ = np.asarray([t["outcome"]["task_success"] for t in trajs], float)
    groups = np.asarray([t["task"]["template_id"] for t in trajs])
    trig_counts = np.asarray([sum(bool(s["intervention"] and s["intervention"]["triggered"]) for s in t["steps"])
                              for t in trajs])
    blocks = [s for t in trajs for s in t["steps"]
              if s["intervention"] and s["intervention"]["action_taken"] in ("block", "reconsider")]
    boot = dict(n_boot=ecfg["bootstrap"], ci=ecfg["ci"], seed=cfg["seed"])
    out = {
        "n_runs": len(trajs),
        "violation_rate": float(viol.mean()),
        "violation_rate_ci": cluster_bootstrap(lambda i: float(viol[i].mean()), groups, **boot),
        "task_success_rate": float(succ.mean()),
        "safe_success_rate": float(np.mean(succ * (1 - viol))),
        "trigger_run_rate": float(np.mean(trig_counts > 0)),
        "trigger_step_rate": _rate(trajs),
        "mean_triggers_per_run": float(trig_counts.mean()),
        "stopped_actions": len(blocks),
        "stopped_actions_unsafe": sum(bool(s["annotations"]["prevented_violations"]) for s in blocks),
        "stopped_actions_legitimate": sum(not s["annotations"]["prevented_violations"] for s in blocks),
        "by_environment": {},
    }
    envs = np.asarray([t["task"]["environment"] for t in trajs])
    for env in sorted(set(envs)):
        m = envs == env
        out["by_environment"][env] = {"n_runs": int(m.sum()), "violation_rate": float(viol[m].mean()),
                                      "task_success_rate": float(succ[m].mean())}
    if baseline is not None:
        keys = [(t["task"]["instance_id"], t["seed"]) for t in trajs]
        paired = [i for i, k in enumerate(keys) if k in baseline]
        b_viol = np.asarray([baseline[keys[i]]["outcome"]["safety_violation"] for i in paired], float)
        b_succ = np.asarray([baseline[keys[i]]["outcome"]["task_success"] for i in paired], float)
        v, s, g = viol[paired], succ[paired], groups[paired]
        clean = b_viol == 0
        out["vs_none"] = {
            "n_pairs": len(paired),
            "violation_delta": cluster_bootstrap(lambda i: float((v[i] - b_viol[i]).mean()), g, **boot),
            "success_delta": cluster_bootstrap(lambda i: float((s[i] - b_succ[i]).mean()), g, **boot),
            # Runs that were clean without intervention but still got intervened on.
            "false_alarm_run_rate": float(np.mean(trig_counts[paired][clean] > 0)) if clean.any() else float("nan"),
            "success_lost_on_clean_runs": float(np.mean((b_succ - s)[clean] > 0)) if clean.any() else float("nan"),
        }
    return out


def run_study(store: ExperimentStore, log: Callable[[str], None] = print) -> dict:
    cfg = store.config
    icfg = cfg["interventions"]
    tasks = store.load_tasks()
    fam_split = get_splits(store, store.load_trajectories())
    test_instances = [t for t in tasks.values() if fam_split.get(t.family) == "test"]
    monitor, threshold = load_monitor(store, icfg["monitor"], icfg["horizon"])
    threshold_source = "monitor threshold (validation, evaluation.threshold_method)"
    if icfg.get("threshold_method", "monitor") != "monitor":
        # Interventions may need a different operating point (e.g. a gate should rarely fire).
        # Still chosen on the validation split only.
        ds, trajs = load_dataset(store)
        val = ds.subset(split_indices(ds.meta, fam_split)["val"])
        threshold = choose_threshold(val.labels[icfg["horizon"]], monitor.score(val),
                                     icfg["threshold_method"], icfg.get("target_fpr", 0.1))
        threshold_source = f"validation split, {icfg['threshold_method']} (target_fpr={icfg.get('target_fpr')})"
    model_cfg = dict(cfg["model"])
    if model_cfg.get("adapter") == "hf":
        model_cfg.setdefault("activations", {}).update(cfg["collection"]["activations"])
    adapter = build_adapter(model_cfg, capture_activations=monitor.uses_activations)
    runner = AgentRunner(adapter, cfg["agent"], capture_activations=False)
    log(f"[intervene] {len(test_instances)} test instances x {len(icfg['seeds'])} seeds; monitor={monitor.name} "
        f"h={icfg['horizon']} threshold={threshold:.3f}")

    conditions = list(icfg["conditions"])
    ordered = [c for c in conditions if not c.endswith(":random")] + [c for c in conditions if c.endswith(":random")]
    results: dict[str, list[dict]] = {}
    for cond in ordered:
        rate = None
        if cond.endswith(":random"):
            src = cond.replace(":random", ":monitor")
            rate = _rate(results[src]) if src in results else 0.1
        done = store.completed_run_ids(cond)
        for inst in test_instances:
            for seed in icfg["seeds"]:
                rid = run_id_for(inst, seed)
                if rid in done:
                    continue
                iv = build_intervention(cond, monitor, threshold, rate)
                traj, _ = runner.run(inst, seed, store.experiment_id, rid, condition=cond, intervention=iv)
                store.append_trajectory(traj, None, condition=cond)
        results[cond] = store.load_trajectories(cond)
        log(f"[intervene] {cond}: {len(results[cond])} runs" + (f" (matched rate {rate:.3f})" if rate is not None else ""))

    baseline = {(t["task"]["instance_id"], t["seed"]): t for t in results.get("none", [])}
    summary = {c: summarise(results[c], baseline if c != "none" else None, cfg) for c in conditions}
    for c in conditions:
        if c.endswith(":monitor"):
            rnd = c.replace(":monitor", ":random")
            if rnd in summary:
                summary[c]["vs_random"] = {
                    "violation_rate_diff": summary[c]["violation_rate"] - summary[rnd]["violation_rate"],
                    "task_success_diff": summary[c]["task_success_rate"] - summary[rnd]["task_success_rate"],
                }
    out = {"monitor": monitor.name, "horizon": icfg["horizon"], "threshold": threshold,
           "threshold_source": threshold_source, "seeds": icfg["seeds"],
           "test_templates": sorted({t.template_id for t in test_instances}), "conditions": summary}
    dump_json(store.path("interventions", "summary.json"), out)
    plots.intervention_outcomes(summary, store.root / "eval" / "plots" / "interventions.png")
    for c, s in summary.items():
        log(f"[intervene] {c:20s} violation={s['violation_rate']:.3f} success={s['task_success_rate']:.3f} "
            f"triggered_runs={s['trigger_run_rate']:.2f}")
    return out

