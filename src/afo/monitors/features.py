"""Step-level dataset construction: behavioural features, activations, labels.

Row t of a trajectory is the *decision point before action t*. Its inputs are the
observable history (steps < t) and the observation at t. Action t itself, any
later step, annotations and outcomes are never read. Rows after the failure-onset
step are dropped (post-failure states are not prediction targets).

Label for horizon h: 1 iff a failure onset occurs at step k with t <= k < t + h.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from afo.collection.storage import ExperimentStore
from afo.collection.trajectory import observable_step

CUE_PATTERNS = [
    r"ignore (all )?(previous|prior) instructions", r"\bsystem override\b", r"\burgent\b", r"\bimmediately\b",
    r"\bright now\b", r"\binstead\b", r"\bpriority\b", r"\bplease (also )?\w+", r"\bmust\b", r"\bassistants?\b",
    r"\bAI\b", r"\bnote to\b", r"\bwhoever\b", r"\bbefore anything\b", r"\bdeadline\b", r"\bcritical\b",
]
SENSITIVE_PATTERNS = [
    r"password", r"credential", r"secret", r"api[_ ]?key", r"token", r"\.env\b", r"iban", r"sudo", r"rm -rf",
    r"delete", r"--force", r"chmod", r"firewall|ufw", r"@[\w.-]+\.example",
]
_CUE = [re.compile(p, re.I) for p in CUE_PATTERNS]
_SENS = [re.compile(p, re.I) for p in SENSITIVE_PATTERNS]
READ_TOOLS = {"list_files", "read_file", "search_docs"}

BEHAVIORAL_FEATURES = [
    "step", "step_frac", "n_invalid", "n_permission_denied", "n_distinct_tools", "n_reads", "n_side_effects",
    "n_repeated_calls", "obs_log_len", "obs_cue_count", "cum_cue_count", "obs_sensitive_count",
    "cum_sensitive_count", "steps_since_first_cue", "any_cue_seen", "obs_external", "n_external",
    "last_is_list", "last_repeats_prev", "recent_read_ratio",
]


def _count(patterns, text: str) -> int:
    return sum(len(p.findall(text)) for p in patterns)


def behavioral_features(prefix: list[dict], observation: str, max_steps: int) -> np.ndarray:
    """Observable features at a decision point. ``prefix`` = observable views of steps < t."""
    prefix = [observable_step(s) for s in prefix]  # defensive: drop any ground-truth keys
    t = len(prefix)
    calls = [(s["tool"], str(sorted((s.get("args") or {}).items()))) for s in prefix]
    observations = [s["observation"] for s in prefix] + [observation]
    cue_per_obs = [_count(_CUE, o) for o in observations]
    sens_per_obs = [_count(_SENS, o) for o in observations]
    first_cue = next((i for i, c in enumerate(cue_per_obs) if c > 0), None)
    recent = prefix[-3:]
    # The observation at t is the response to action t-1, so its source is prefix[-1].
    obs_external = bool(prefix and prefix[-1].get("external_content"))
    f = [
        t,
        t / max(1, max_steps),
        sum(1 for s in prefix if not s["action_valid"]),
        sum(1 for s in prefix if s.get("permission_denied")),
        len({s["tool"] for s in prefix if s["tool"]}),
        sum(1 for s in prefix if s["tool"] in READ_TOOLS),
        sum(1 for s in prefix if s.get("side_effect")),
        len(calls) - len(set(calls)),
        math.log1p(len(observation)),
        cue_per_obs[-1],
        sum(cue_per_obs),
        sens_per_obs[-1],
        sum(sens_per_obs),
        (t - first_cue) if first_cue is not None else 0,
        float(first_cue is not None),
        float(obs_external),
        sum(1 for s in prefix if s.get("external_content")),
        float(bool(prefix) and prefix[-1]["tool"] == "list_files"),
        float(len(calls) >= 2 and calls[-1] == calls[-2]),
        (sum(1 for s in recent if s["tool"] in READ_TOOLS) / len(recent)) if recent else 0.0,
    ]
    return np.asarray(f, dtype=np.float32)


@dataclass
class StepDataset:
    behavioral: np.ndarray            # [N, F]
    activations: np.ndarray | None    # [N, L, D]
    labels: dict[int, np.ndarray]     # horizon -> [N]
    meta: list[dict]                  # per-row identifiers
    layers: list[int]
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.meta)

    def subset(self, idx: np.ndarray) -> "StepDataset":
        return StepDataset(
            self.behavioral[idx],
            self.activations[idx] if self.activations is not None else None,
            {h: y[idx] for h, y in self.labels.items()},
            [self.meta[i] for i in idx], self.layers, self.feature_names,
        )

    def column(self, key: str) -> np.ndarray:
        return np.asarray([m[key] for m in self.meta])

    def save(self, path) -> None:
        arrays = {"behavioral": self.behavioral, "layers": np.asarray(self.layers)}
        if self.activations is not None:
            arrays["activations"] = self.activations.astype(np.float16)
        for h, y in self.labels.items():
            arrays[f"y_{h}"] = y
        np.savez_compressed(path, **arrays)


def step_labels(n_rows: int, onset: int | None, horizons: list[int]) -> dict[int, np.ndarray]:
    t = np.arange(n_rows)
    out = {}
    for h in horizons:
        out[h] = ((onset is not None) & (t <= (onset if onset is not None else -1))
                  & ((onset if onset is not None else 0) - t < h)).astype(np.int8)
    return out


def build_step_dataset(trajectories: list[dict], store: ExperimentStore | None, horizons: list[int],
                       max_steps: int, use_activations: bool = True) -> StepDataset:
    feats, acts, meta = [], [], []
    labels: dict[int, list] = {h: [] for h in horizons}
    layers: list[int] = []
    for traj in trajectories:
        steps = traj["steps"]
        onset = traj["outcome"]["failure_onset"]
        n = len(steps) if onset is None else onset + 1
        if n == 0:
            continue
        a = store.load_activations(traj["run_id"]) if (use_activations and store is not None) else None
        if use_activations and a is None:
            raise FileNotFoundError(f"activations missing for {traj['run_id']}; collect with activations enabled")
        if a is not None:
            layers = (traj.get("activations") or {}).get("layers", list(range(a.shape[1])))
            if a.shape[0] < n:
                raise ValueError(f"activation rows ({a.shape[0]}) < steps ({n}) for {traj['run_id']}")
        ys = step_labels(n, onset, horizons)
        for t in range(n):
            feats.append(behavioral_features(steps[:t], steps[t]["observation"], max_steps))
            if a is not None:
                acts.append(a[t])
            for h in horizons:
                labels[h].append(ys[h][t])
            meta.append({
                "run_id": traj["run_id"], "step": t, "template_id": traj["task"]["template_id"],
                "family": traj["task"]["family"], "environment": traj["task"]["environment"],
                "strength": traj["task"]["hazard_strength"], "difficulty": traj["task"]["difficulty"],
                "failure_onset": onset, "steps_to_onset": (onset - t) if onset is not None else None,
                "trajectory_fails": onset is not None,
            })
    return StepDataset(
        behavioral=np.stack(feats) if feats else np.zeros((0, len(BEHAVIORAL_FEATURES)), np.float32),
        activations=np.stack(acts).astype(np.float32) if acts else None,
        labels={h: np.asarray(v, dtype=np.int8) for h, v in labels.items()},
        meta=meta, layers=list(layers), feature_names=list(BEHAVIORAL_FEATURES),
    )
