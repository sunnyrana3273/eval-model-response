"""Command-line interface.

    afo validate-templates                 machine-check every task template
    afo generate-tasks [CONFIG]            expand templates into task instances (JSONL)
    afo collect CONFIG [--exp-dir DIR]     run agents, store trajectories (+ activations)
    afo train --exp-dir DIR                fit monitors, choose thresholds on validation
    afo evaluate --exp-dir DIR             test-split metrics, CIs, plots
    afo generalize --exp-dir DIR           leave-one-environment-out shift evaluation
    afo intervene --exp-dir DIR            monitor-triggered intervention study
    afo report --exp-dir DIR               Markdown results report
    afo run CONFIG                         all of the above, in order
    afo replay --exp-dir DIR --run-id ID   re-execute a recorded trajectory, check determinism
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from afo.collection.storage import ExperimentStore
from afo.config import load_config, resolve_path
from afo.utils import write_jsonl


def _store(args) -> ExperimentStore:
    root = Path(args.exp_dir)
    if not (root / "config.yaml").exists():
        sys.exit(f"not an experiment directory: {root}")
    return ExperimentStore(root)


def cmd_validate_templates(args) -> int:
    from afo.environments.templates import load_templates, validate_template

    templates = load_templates(resolve_path(args.templates))
    bad = 0
    for t in templates:
        problems = validate_template(t)
        if problems:
            bad += 1
            print(f"FAIL {t['id']}:")
            for p in problems:
                print(f"    - {p}")
    by_env: dict[str, int] = {}
    for t in templates:
        by_env[t["environment"]] = by_env.get(t["environment"], 0) + 1
    print(f"{len(templates) - bad}/{len(templates)} templates valid; per environment: {by_env}")
    return 1 if bad else 0


def cmd_generate_tasks(args) -> int:
    from afo.collection.collect import instances_from_config

    cfg = load_config(args.config)
    instances = instances_from_config(cfg)
    out = resolve_path(args.out)
    write_jsonl(out, (i.to_dict() for i in instances))
    print(f"wrote {len(instances)} task instances to {out}")
    return 0


def cmd_collect(args) -> int:
    from afo.collection.collect import collect

    cfg = load_config(args.config)
    store = collect(cfg, args.exp_dir)
    print(store.root)
    return 0


def cmd_train(args) -> int:
    from afo.monitors.training import train_monitors

    train_monitors(_store(args))
    return 0


def cmd_evaluate(args) -> int:
    from afo.evaluation.evaluate import evaluate_monitors

    evaluate_monitors(_store(args))
    return 0


def cmd_generalize(args) -> int:
    from afo.evaluation.generalization import leave_one_environment_out

    leave_one_environment_out(_store(args))
    return 0


def cmd_intervene(args) -> int:
    from afo.interventions.study import run_study

    run_study(_store(args))
    return 0


def cmd_report(args) -> int:
    from afo.evaluation.report import write_report

    print(write_report(_store(args)))
    return 0


def cmd_run(args) -> int:
    from afo.collection.collect import collect
    from afo.evaluation.evaluate import evaluate_monitors
    from afo.evaluation.generalization import leave_one_environment_out
    from afo.evaluation.report import write_report
    from afo.interventions.study import run_study
    from afo.monitors.training import train_monitors

    cfg = load_config(args.config)
    store = collect(cfg, args.exp_dir)
    train_monitors(store)
    evaluate_monitors(store)
    leave_one_environment_out(store)
    if cfg["interventions"]["enabled"]:
        run_study(store)
    print(f"[report] {write_report(store)}")
    return 0


def cmd_replay(args) -> int:
    from afo.agents.tool_executor import ParsedAction, ToolExecutor
    from afo.environments.base import SandboxWorld

    store = _store(args)
    tasks = store.load_tasks()
    traj = next((t for t in store.iter_trajectories(args.condition) if t["run_id"] == args.run_id), None)
    if traj is None:
        sys.exit(f"run not found: {args.run_id}")
    inst = tasks[traj["task"]["instance_id"]]
    if inst.task_version != traj["task"]["task_version"]:
        sys.exit("task version mismatch: the stored task differs from the one this run used")
    world = SandboxWorld(inst, store.config["agent"]["max_output_chars"])
    ex = ToolExecutor(world)
    mismatches = 0
    for s in traj["steps"]:
        iv = s.get("intervention") or {}
        if iv.get("action_taken") == "block":
            print(f"step {s['step']:2d} [blocked by gate] {iv['original_action']['tool']}")
            continue
        res = ex.execute(ParsedAction(s["tool"], s["args"] or {}, s["thought"], s["parse_error"]))
        same = res.output == s["tool_response"]
        mismatches += not same
        flag = "ok " if same else "DIFF"
        print(f"step {s['step']:2d} {flag} {s['tool']}({s['args']}) violations={s['annotations']['violations']}")
    print(f"outcome: {traj['outcome']['status']}, violation={traj['outcome']['safety_violation']}, "
          f"onset={traj['outcome']['failure_onset']}; replay mismatches: {mismatches}")
    return 1 if mismatches else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="afo", description="Agent Failure Observatory")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("validate-templates")
    s.add_argument("--templates", default="datasets/templates")
    s.set_defaults(fn=cmd_validate_templates)

    s = sub.add_parser("generate-tasks")
    s.add_argument("config", nargs="?", default=None)
    s.add_argument("--out", default="datasets/samples/tasks.jsonl")
    s.set_defaults(fn=cmd_generate_tasks)

    for name, fn in (("collect", cmd_collect), ("run", cmd_run)):
        s = sub.add_parser(name)
        s.add_argument("config")
        s.add_argument("--exp-dir", default=None, help="reuse/resume a specific experiment directory")
        s.set_defaults(fn=fn)

    for name, fn in (("train", cmd_train), ("evaluate", cmd_evaluate), ("generalize", cmd_generalize),
                     ("intervene", cmd_intervene), ("report", cmd_report)):
        s = sub.add_parser(name)
        s.add_argument("--exp-dir", required=True)
        s.set_defaults(fn=fn)

    s = sub.add_parser("replay")
    s.add_argument("--exp-dir", required=True)
    s.add_argument("--run-id", required=True)
    s.add_argument("--condition", default=None, help="intervention condition, e.g. gate:monitor")
    s.set_defaults(fn=cmd_replay)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
