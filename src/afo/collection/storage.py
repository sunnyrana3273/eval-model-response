"""Experiment directories: config, manifest, trajectories, activations, artefacts.

Layout::

    experiments/<name>_<timestamp>_<confighash>/
        config.yaml          resolved config (reproduces the setup)
        manifest.json        versions, git commit, dataset hash, activation metadata
        tasks.jsonl          exact task instances used (with task_version)
        trajectories.jsonl   one trajectory per line
        activations/<run_id>.npy   float16 [n_steps, n_layers, hidden_dim]
        monitors/ eval/ interventions/ report/   downstream artefacts
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import yaml

from afo.collection.trajectory import validate_trajectory
from afo.config import config_hash, dump_config, resolve_path
from afo.environments.base import TaskInstance
from afo.utils import append_jsonl, dump_json, env_info, git_info, load_json, read_jsonl, write_jsonl


class ExperimentStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    # -- creation ------------------------------------------------------------------------
    @classmethod
    def create(cls, cfg: dict, exp_dir: Path | str | None = None) -> "ExperimentStore":
        if exp_dir is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            exp_dir = resolve_path(cfg["output_dir"]) / f"{cfg['name']}_{stamp}_{config_hash(cfg)}"
        store = cls(exp_dir)
        store.root.mkdir(parents=True, exist_ok=True)
        if not (store.root / "config.yaml").exists():
            dump_config(cfg, store.root / "config.yaml")
            dump_json(store.root / "manifest.json", {
                "experiment_id": store.experiment_id,
                "config_hash": config_hash(cfg),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "git": git_info(),
                "environment": env_info(),
            })
        return store

    @property
    def experiment_id(self) -> str:
        return self.root.name

    @property
    def config(self) -> dict:
        return yaml.safe_load((self.root / "config.yaml").read_text())

    @property
    def manifest(self) -> dict:
        return load_json(self.root / "manifest.json")

    def update_manifest(self, **fields) -> None:
        m = self.manifest
        m.update(fields)
        dump_json(self.root / "manifest.json", m)

    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # -- tasks -----------------------------------------------------------------------------
    def save_tasks(self, instances: list[TaskInstance]) -> None:
        write_jsonl(self.root / "tasks.jsonl", (i.to_dict() for i in instances))

    def load_tasks(self) -> dict[str, TaskInstance]:
        return {d["instance_id"]: TaskInstance.from_dict(d) for d in read_jsonl(self.root / "tasks.jsonl")}

    # -- trajectories -------------------------------------------------------------------------
    def trajectories_path(self, condition: str | None = None) -> Path:
        if condition in (None, "collection"):
            return self.root / "trajectories.jsonl"
        return self.root / "interventions" / condition.replace(":", "_") / "trajectories.jsonl"

    def append_trajectory(self, traj: dict, activations: np.ndarray | None,
                          condition: str | None = None) -> None:
        validate_trajectory(traj)
        if activations is not None and condition in (None, "collection"):
            np.save(self.path("activations", f"{traj['run_id']}.npy"), activations.astype(np.float16))
        append_jsonl(self.trajectories_path(condition), traj)

    def completed_run_ids(self, condition: str | None = None) -> set[str]:
        p = self.trajectories_path(condition)
        return {t["run_id"] for t in read_jsonl(p)} if p.exists() else set()

    def iter_trajectories(self, condition: str | None = None,
                          where: Callable[[dict], bool] | None = None) -> Iterator[dict]:
        p = self.trajectories_path(condition)
        if not p.exists():
            return
        for t in read_jsonl(p):
            validate_trajectory(t)
            if where is None or where(t):
                yield t

    def load_trajectories(self, condition: str | None = None, **filters) -> list[dict]:
        """Load trajectories, optionally filtering on task fields (environment=..., template_id=...)."""
        def where(t):
            return all(t["task"].get(k) == v or t["outcome"].get(k) == v for k, v in filters.items())
        return list(self.iter_trajectories(condition, where if filters else None))

    def load_activations(self, run_id: str) -> np.ndarray | None:
        p = self.root / "activations" / f"{run_id}.npy"
        return np.load(p).astype(np.float32) if p.exists() else None

    def has(self, *parts: str) -> bool:
        return self.root.joinpath(*parts).exists()
