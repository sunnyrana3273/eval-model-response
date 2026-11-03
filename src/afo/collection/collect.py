"""Trajectory collection stage."""

from __future__ import annotations

from collections import Counter
from typing import Callable

from afo.agents.model_adapter import ModelAdapter, build_adapter
from afo.agents.runner import AgentRunner
from afo.collection.storage import ExperimentStore
from afo.config import resolve_path
from afo.environments.base import TaskInstance
from afo.environments.templates import expand_instances, load_templates
from afo.utils import stable_hash


def instances_from_config(cfg: dict) -> list[TaskInstance]:
    t = cfg["tasks"]
    templates = load_templates(resolve_path(t["templates_dir"]), t["environments"])
    if t.get("templates"):
        templates = [x for x in templates if x["id"] in set(t["templates"])]
    instances = expand_instances(templates, t["strengths"], t["placements"], t["difficulties"])
    if t.get("limit"):
        instances = instances[: t["limit"]]
    return instances


def run_id_for(instance: TaskInstance, seed: int) -> str:
    return f"{instance.instance_id}.s{seed}"


def collect(cfg: dict, exp_dir=None, adapter: ModelAdapter | None = None,
            log: Callable[[str], None] = print) -> ExperimentStore:
    store = ExperimentStore.create(cfg, exp_dir)
    instances = instances_from_config(cfg)
    store.save_tasks(instances)
    capture = cfg["collection"]["activations"]["enabled"]
    if adapter is None:
        model_cfg = dict(cfg["model"])
        if model_cfg.get("adapter") == "hf":
            model_cfg.setdefault("activations", {}).update(cfg["collection"]["activations"])
        adapter = build_adapter(model_cfg, capture_activations=capture)
    runner = AgentRunner(adapter, cfg["agent"], capture_activations=capture)
    done = store.completed_run_ids()
    todo = [(i, s) for i in instances for s in cfg["tasks"]["seeds"] if run_id_for(i, s) not in done]
    log(f"[collect] {len(instances)} task instances x {len(cfg['tasks']['seeds'])} seeds; "
        f"{len(done)} cached, {len(todo)} to run -> {store.root}")
    statuses: Counter = Counter()
    for k, (inst, seed) in enumerate(todo, 1):
        traj, acts = runner.run(inst, seed, store.experiment_id, run_id_for(inst, seed), condition="collection")
        store.append_trajectory(traj, acts)
        statuses[traj["outcome"]["status"]] += 1
        if k % 50 == 0 or k == len(todo):
            log(f"[collect] {k}/{len(todo)} {dict(statuses)}")
    store.update_manifest(
        dataset={"n_instances": len(instances), "seeds": cfg["tasks"]["seeds"],
                 "task_versions_hash": stable_hash(sorted((i.instance_id, i.task_version) for i in instances))},
        model=adapter.info(),
        activations=adapter.activation_info() if capture else None,
    )
    return store
