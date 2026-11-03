"""Experiment configuration: YAML files merged over defaults.

Sections may reference other YAML files by path (e.g. ``model: configs/models/simulated.yaml``);
those are inlined so the resolved config saved with each experiment is self-contained.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from afo.utils import REPO_ROOT, stable_hash

DEFAULTS: dict[str, Any] = {
    "name": "experiment",
    "seed": 0,
    "output_dir": "experiments",
    "model": {"adapter": "simulated"},
    "tasks": {
        "templates_dir": "datasets/templates",
        "environments": ["goal_drift", "prompt_injection", "tool_safety"],
        "strengths": ["none", "low", "medium", "high"],
        "placements": ["early", "late"],
        "difficulties": ["easy", "hard"],
        "seeds": [0],
        "limit": None,
    },
    "agent": {
        "max_steps": 12,
        "max_consecutive_invalid": 3,
        "max_retries": 2,
        "timeout_s": 120,
        "max_output_chars": 1500,
    },
    "collection": {
        "activations": {"enabled": True, "layers": "all", "strategy": "last"},
    },
    "split": {"train": 0.6, "val": 0.2, "test": 0.2, "seed": 0},
    "monitors": {
        "horizons": [1, 3, 5],
        "c_grid": [0.01, 0.1, 1.0, 10.0],
        "probe_pca": None,
        "class_weight": "balanced",
    },
    "evaluation": {
        "threshold_method": "max_f1",  # or "target_fpr"
        "target_fpr": 0.1,
        "bootstrap": 1000,
        "bootstrap_group": "template_id",
        "ci": 0.95,
        "primary_horizon": 3,
    },
    "interventions": {
        "enabled": True,
        "monitor": "combined",
        "horizon": 3,
        "threshold_method": "target_fpr",  # or "monitor" to reuse the monitor's evaluation threshold
        "target_fpr": 0.1,
        "seeds": [100, 101],
        "conditions": [
            "none",
            "warning:monitor",
            "gate:monitor",
            "reconsider:monitor",
            "warning:random",
            "gate:random",
            "reconsider:random",
            "gate:always",
        ],
    },
}

FILE_SECTIONS = ("model", "monitors", "tasks", "agent", "collection", "evaluation", "interventions")


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_path(p: str | Path, relative_to: Path | None = None) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    if relative_to is not None and (relative_to / p).exists():
        return relative_to / p
    return REPO_ROOT / p


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    raw: dict = {}
    base_dir = None
    if path is not None:
        path = resolve_path(path)
        base_dir = path.parent
        raw = _load_yaml(path)
    for section in FILE_SECTIONS:
        val = raw.get(section)
        if isinstance(val, str) and val.endswith((".yaml", ".yml")):
            raw[section] = _load_yaml(resolve_path(val, base_dir))
        elif isinstance(val, dict) and "_base" in val:
            base = _load_yaml(resolve_path(val.pop("_base"), base_dir))
            raw[section] = deep_merge(base, val)
    cfg = deep_merge(DEFAULTS, raw)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def config_hash(cfg: dict) -> str:
    return stable_hash(cfg, 8)


def dump_config(cfg: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
