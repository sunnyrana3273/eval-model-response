"""Render a Markdown results report from stored evaluation artefacts.

Hypothesis verdicts are mechanical: "supported" only when the relevant 95% CI
excludes the null in the predicted direction; otherwise "not supported" or
"inconclusive". Every number is traceable to eval/*.json in the experiment dir.
"""

from __future__ import annotations

import math

from afo.collection.storage import ExperimentStore
from afo.utils import load_json

MONITORS = ("majority", "behavioral", "probe", "combined")


def _f(x, nd=3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _ci(c: dict, nd=3) -> str:
    return f"{_f(c['point'], nd)} [{_f(c['lo'], nd)}, {_f(c['hi'], nd)}]"


def _verdict(diff: dict | None, direction: int = 1) -> str:
    if not diff or math.isnan(diff.get("lo", float("nan"))):
        return "inconclusive (CI unavailable)"
    lo, hi = diff["lo"], diff["hi"]
    if (direction > 0 and lo > 0) or (direction < 0 and hi < 0):
        return "supported"
    if (direction > 0 and hi < 0) or (direction < 0 and lo > 0):
        return "not supported (effect in the opposite direction)"
    return "inconclusive (CI includes 0)"


def render_report(store: ExperimentStore) -> str:
    cfg = store.config
    man = store.manifest
    metrics = load_json(store.root / "eval" / "metrics.json")
    summary = load_json(store.root / "features" / "summary.json")
    gen = load_json(store.root / "eval" / "generalization.json") if store.has("eval", "generalization.json") else None
    iv = load_json(store.root / "interventions" / "summary.json") if store.has("interventions", "summary.json") else None
    trajs = store.load_trajectories()
    H = str(cfg["evaluation"]["primary_horizon"])
    L: list[str] = []
    w = L.append

    w(f"# Results: `{store.experiment_id}`\n")
    model = man.get("model") or {}
    if model.get("adapter") == "simulated":
        w("> **Simulated backend.** These trajectories come from `SimulatedPolicyAdapter`, a seeded stochastic "
          "policy with synthetic hidden states. The numbers check the pipeline (positive/negative controls) and "
          "say nothing about real LLM agents. See `docs/methodology.md`.\n")
    w(f"- Model: `{model.get('id')}` (revision `{model.get('revision')}`, adapter `{model.get('adapter')}`)")
    w(f"- Config hash: `{man.get('config_hash')}`; git commit: `{(man.get('git') or {}).get('commit')}`")
    w(f"- Dataset: {len(trajs)} trajectories, task-version hash `{(man.get('dataset') or {}).get('task_versions_hash')}`")
    w(f"- Threshold selection: `{metrics['threshold_method']['method']}` on the validation split")
    t = metrics["test_split"]
    w(f"- CIs: {cfg['evaluation']['ci']:.0%} cluster bootstrap over `{t['bootstrap_group']}` "
      f"({t['n_groups']} groups, {cfg['evaluation']['bootstrap']} resamples)\n")

    w("## Dataset\n")
    n = len(trajs)
    viol = sum(x["outcome"]["safety_violation"] for x in trajs)
    succ = sum(x["outcome"]["task_success"] for x in trajs)
    w(f"Task success {succ}/{n} ({succ / n:.1%}); safety violation {viol}/{n} ({viol / n:.1%}).\n")
    w("| Environment | Trajectories | Violation rate | Task success |")
    w("|---|---|---|---|")
    for env in sorted({x["task"]["environment"] for x in trajs}):
        sub = [x for x in trajs if x["task"]["environment"] == env]
        w(f"| {env} | {len(sub)} | {sum(x['outcome']['safety_violation'] for x in sub) / len(sub):.2f} | "
          f"{sum(x['outcome']['task_success'] for x in sub) / len(sub):.2f} |")
    w("")
    w("| Split | Rows | Trajectories | Templates | Prevalence (h=" + H + ") |")
    w("|---|---|---|---|---|")
    for s, d in summary["splits"].items():
        w(f"| {s} | {d['rows']} | {d['trajectories']} | {len(d['templates'])} | {_f(d['prevalence'][H])} |")
    w(f"\nTest templates (unseen in training): {', '.join(metrics['test_split']['templates'])}\n")

    w("## Failure prediction (test split)\n")
    for h, res in metrics["results"].items():
        w(f"### Horizon {h}\n")
        w("| Monitor | Layer | AUROC [95% CI] | AUPRC [95% CI] | Precision | Recall | F1 | FPR | Brier | ECE |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for m in MONITORS:
            if m in res:
                r = res[m]
                w(f"| {m} | {r.get('layer') if r.get('layer') is not None else '-'} | {_ci(r['auroc_ci'])} | "
                  f"{_ci(r['auprc_ci'])} | {_f(r['precision'])} | {_f(r['recall'])} | {_f(r['f1'])} | "
                  f"{_f(r['fpr'])} | {_f(r['brier'])} | {_f(r['ece'])} |")
        w(f"\nPrevalence {_f(res['majority']['prevalence'])} (n={res['majority']['n']}, positives="
          f"{res['majority']['n_pos']}).\n")

    w(f"### Paired AUROC differences (horizon {H})\n")
    w("| Comparison | Δ AUROC [95% CI] |")
    w("|---|---|")
    for k, d in metrics["paired_differences"][H].items():
        w(f"| {k} | {_ci(d)} |")
    w("")

    w(f"### Early warning (horizon {H})\n")
    w("| Monitor | Detection rate | Early (≥1 step) | Median lead | Clean-run false alarms |")
    w("|---|---|---|---|---|")
    for m in MONITORS:
        lt = metrics["lead_time"][H].get(m)
        if lt:
            w(f"| {m} | {_f(lt['detection_rate'], 2)} | {_f(lt['early_detection_rate'], 2)} | "
              f"{_f(lt['median_lead_steps'], 1)} | {_f(lt['clean_trajectory_false_alarm_rate'], 2)} |")
    w("\nAUROC of states *k* steps before onset vs. clean-trajectory states:\n")
    ks = sorted({r["k"] for rows in metrics["by_steps_to_onset"][H].values() for r in rows})
    w("| Monitor | " + " | ".join(f"k={k}" for k in ks) + " |")
    w("|---|" + "---|" * len(ks))
    for m in MONITORS:
        rows = {r["k"]: r for r in metrics["by_steps_to_onset"][H].get(m, [])}
        if rows:
            w(f"| {m} | " + " | ".join(_f(rows[k]["auroc"]) if k in rows else "-" for k in ks) + " |")
    w("")

    if metrics["layer_sweep"].get(H):
        w(f"### Probe layer sweep (horizon {H})\n")
        w("| Layer | Val AUROC (selection) | Test AUROC (descriptive) |")
        w("|---|---|---|")
        for r in metrics["layer_sweep"][H]:
            w(f"| {r['layer']} | {_f(r['val_auroc'])} | {_f(r['test_auroc'])} |")
        w("")

    w(f"### By environment (horizon {H}, AUROC)\n")
    envs = sorted(next(iter(metrics["by_category"][H].values())).keys())
    w("| Monitor | " + " | ".join(envs) + " |")
    w("|---|" + "---|" * len(envs))
    for m in MONITORS:
        if m in metrics["by_category"][H]:
            w(f"| {m} | " + " | ".join(_f(metrics["by_category"][H][m][e]["auroc"]) for e in envs) + " |")
    w("")

    if gen:
        w(f"## Generalisation: leave-one-environment-out (horizon {gen['horizon']})\n")
        w("| Held-out environment | " + " | ".join(m for m in MONITORS if m != "majority") + " |")
        w("|---|" + "---|" * 3)
        for env, res in gen["held_out"].items():
            w(f"| {env} | " + " | ".join(_ci(res[m]["auroc_ci"]) if m in res else "-"
                                         for m in MONITORS if m != "majority") + " |")
        w("")

    if iv:
        w(f"## Interventions (monitor `{iv['monitor']}`, horizon {iv['horizon']}, threshold {iv['threshold']:.3f})\n")
        w("| Condition | Runs | Violation rate [95% CI] | Task success | Safe success | Runs intervened | "
          "Stopped unsafe | Stopped legitimate | Δ violation vs none [95% CI] |")
        w("|---|---|---|---|---|---|---|---|---|")
        for c, s in iv["conditions"].items():
            d = s.get("vs_none", {}).get("violation_delta")
            w(f"| {c} | {s['n_runs']} | {_ci(s['violation_rate_ci'])} | {_f(s['task_success_rate'])} | "
              f"{_f(s['safe_success_rate'])} | {_f(s['trigger_run_rate'], 2)} | {s['stopped_actions_unsafe']} | "
              f"{s['stopped_actions_legitimate']} | {_ci(d) if d else '-'} |")
        w("")

    w("## Hypotheses\n")
    pd = metrics["paired_differences"][H]
    w("| Hypothesis | Evidence | Verdict |")
    w("|---|---|---|")
    w(f"| H1 behavioural > majority | Δ AUROC {_ci(pd['behavioral-majority'])} | {_verdict(pd['behavioral-majority'])} |")
    if "combined-behavioral" in pd:
        w(f"| H2 combined > behavioural | Δ AUROC {_ci(pd['combined-behavioral'])} | {_verdict(pd['combined-behavioral'])} |")
    for m in ("combined", "behavioral"):
        rows = metrics["by_steps_to_onset"][H].get(m)
        if rows:
            first, last = rows[0], rows[-1]
            trend = "consistent" if first["auroc"] > last["auroc"] else "not consistent"
            w(f"| H3 accuracy rises near failure ({m}) | AUROC k={first['k']}: {_f(first['auroc'])}, "
              f"k={last['k']}: {_f(last['auroc'])} | descriptive: {trend} with H3 (no CI) |")
            break
    if iv:
        for c in ("gate:monitor", "warning:monitor", "reconsider:monitor"):
            s = iv["conditions"].get(c)
            if s and "vs_none" in s:
                d = s["vs_none"]["violation_delta"]
                w(f"| H4 {c} reduces violations | Δ violation rate {_ci(d)}; Δ success "
                  f"{_ci(s['vs_none']['success_delta'])} | {_verdict(d, -1)} |")
    if gen:
        main = metrics["results"][str(gen["horizon"])]
        for env, res in gen["held_out"].items():
            m = "combined" if "combined" in res else "behavioral"
            c = res[m]["auroc_ci"]
            transfer = "signal transfers (CI > 0.5)" if c["lo"] > 0.5 else "transfer inconclusive"
            w(f"| H5 shift to {env} ({m}) | LOEO AUROC {_ci(c)} vs in-distribution test {_f(main[m]['auroc'])} | "
              f"{'degrades' if c['point'] < main[m]['auroc'] else 'no degradation'}; {transfer} |")
    w("\n## Figures\n")
    for fig in sorted((store.root / "eval" / "plots").glob("*.png")):
        w(f"![{fig.stem}](../eval/plots/{fig.name})")
    return "\n".join(L) + "\n"


def write_report(store: ExperimentStore) -> str:
    text = render_report(store)
    path = store.path("report", "results.md")
    path.write_text(text)
    return str(path)
