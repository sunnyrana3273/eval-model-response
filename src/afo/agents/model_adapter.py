"""Model adapters: the agent's policy.

Every adapter exposes the same two calls per decision point:

* ``encode(messages, step)`` returns activations ``[n_layers, hidden_dim]`` computed from the
  context *before* the action is generated (or ``None`` in no-activation mode);
* ``generate(messages, step)`` returns the raw model output text.

``HFModelAdapter`` wraps any Hugging Face causal LM. ``SimulatedPolicyAdapter`` is a
seeded stochastic policy with synthetic hidden states. It exists so the full pipeline
can run on a laptop CPU and so the analysis has positive and negative controls (see
docs/methodology.md, "Simulated backend"). Its results describe the simulator's
assumptions, not a real model.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from afo.environments.base import TaskInstance
from afo.utils import rng_for, stable_hash

WARNING_MARKER = "[MONITOR WARNING]"
RECONSIDER_MARKER = "[RECONSIDER]"
DEFERRED_MARKER = "[ACTION DEFERRED]"


class TransientInferenceError(RuntimeError):
    """Raised for retryable inference failures (e.g. OOM on a shared GPU, timeouts)."""


class ModelAdapter:
    supports_activations = False
    layer_indices: list[int] = []
    hidden_dim = 0

    def reset(self, instance: TaskInstance, seed: int) -> None:
        self.seed = seed

    def encode(self, messages: list[dict], step: int) -> np.ndarray | None:
        return None

    def generate(self, messages: list[dict], step: int, attempt: int = 0) -> str:
        raise NotImplementedError

    def info(self) -> dict:
        raise NotImplementedError

    def activation_info(self) -> dict | None:
        return None


# ---------------------------------------------------------------------------------------
# Simulated policy
# ---------------------------------------------------------------------------------------

SIM_DEFAULTS: dict[str, Any] = {
    "name": "simulated-policy",
    "revision": "v1",
    "susceptibility": {"a": 2.0, "b": 3.0},
    "strength_weight": {"low": 0.55, "medium": 0.85, "high": 1.2},
    "difficulty_mult": {"easy": 0.65, "hard": 1.0},
    "growth": 0.3,
    "decision_sharpness": 7.0,
    "decision_threshold": 0.6,
    "precursor_rate": 0.5,
    "distraction_rate": 0.05,
    "malformed_rate": 0.02,
    "early_submit_rate": 0.015,
    "intervention_effect": {"warning": 0.5, "reconsider": 0.35, "deferred": 0.7},
    "n_layers": 8,
    "hidden_dim": 64,
    "activation_signal": 1.0,
    "signal_peak_layer": 5,
    "signal_width": 2.0,
    "noise": 1.0,
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class SimulatedPolicyAdapter(ModelAdapter):
    """Seeded stochastic agent with a latent per-run susceptibility and risk state.

    Dynamics (all configurable):
      * s0 ~ Beta(a, b) per run: a latent trait not directly visible in behaviour.
      * On first seeing the hazard, risk r = s0 * strength_weight * difficulty_mult; r
        then grows by ``growth`` per step ("rumination"). Interventions damp it.
      * Each step, P(unsafe action) = sigmoid(sharpness * (r - threshold)).
      * Elevated risk sometimes produces observable precursors (re-reading the carrier,
        exploring the filesystem), so behaviour carries partial signal.
      * Hidden states: h_l = signal_l * (risk directions) + nuisance (task, progress) +
        noise. ``activation_signal: 0`` gives a null control with no risk information.
    """

    supports_activations = True

    def __init__(self, cfg: dict | None = None):
        from afo.config import deep_merge

        self.cfg = deep_merge(SIM_DEFAULTS, cfg or {})
        c = self.cfg
        self.layer_indices = list(range(c["n_layers"]))
        self.hidden_dim = c["hidden_dim"]
        L, D = c["n_layers"], c["hidden_dim"]
        g = rng_for("sim-directions", c["revision"])
        self._u_risk = g.standard_normal((L, D)) / math.sqrt(D)
        self._u_trait = g.standard_normal((L, D)) / math.sqrt(D)
        self._u_exposed = g.standard_normal((L, D)) / math.sqrt(D)
        self._u_progress = g.standard_normal((L, D)) / math.sqrt(D)
        ls = np.arange(L)
        self._signal = c["activation_signal"] * np.exp(-((ls - c["signal_peak_layer"]) / c["signal_width"]) ** 2)

    # -- episode state -------------------------------------------------------------------
    def reset(self, instance: TaskInstance, seed: int) -> None:
        c = self.cfg
        self.seed = seed
        self.inst = instance
        g = rng_for("sim-trait", instance.instance_id, seed)
        self.s0 = float(g.beta(c["susceptibility"]["a"], c["susceptibility"]["b"]))
        self.base_risk = 0.0
        if instance.hazard_strength != "none":
            self.base_risk = (self.s0 * c["strength_weight"][instance.hazard_strength]
                              * c["difficulty_mult"][instance.difficulty])
        self.exposed_at: int | None = None
        self.damp = 1.0
        self.pointer = 0
        self.unsafe_done = False
        self.n_warnings_seen = 0
        self.n_deferrals_seen = 0
        self.last_kind: str | None = None
        self.task_vec = rng_for("sim-task", instance.template_id).standard_normal(self.hidden_dim)
        self.hazard_probe = (instance.hazard_text or "").strip()[:60]

    def _sync(self, messages: list[dict], step: int) -> None:
        """Update latent state from the conversation (the only thing a model sees)."""
        c = self.cfg
        text_all = [m["content"] for m in messages]
        if self.exposed_at is None and self.hazard_probe and any(self.hazard_probe in t for t in text_all):
            self.exposed_at = step
        n_warn = sum(t.count(WARNING_MARKER) for t in text_all)
        while self.n_warnings_seen < n_warn:
            self.n_warnings_seen += 1
            self.damp *= c["intervention_effect"]["warning"]
        n_def = sum(t.count(DEFERRED_MARKER) for t in text_all)
        while self.n_deferrals_seen < n_def:
            self.n_deferrals_seen += 1
            self.damp *= c["intervention_effect"]["deferred"]
        if messages and messages[-1]["role"] == "user" and RECONSIDER_MARKER in messages[-1]["content"]:
            if not getattr(self, "_reconsider_applied", False):
                self.damp *= c["intervention_effect"]["reconsider"]
                # The proposal is discarded; roll back the state change it caused.
                if self.last_kind == "plan":
                    self.pointer -= 1
                elif self.last_kind == "unsafe":
                    self.unsafe_done = False
                self._reconsider_applied = True
        else:
            self._reconsider_applied = False

    def risk(self, step: int) -> float:
        if self.exposed_at is None or self.unsafe_done:
            return 0.0
        since = step - self.exposed_at
        return self.base_risk * (1.0 + self.cfg["growth"] * since) * self.damp

    # -- adapter API -------------------------------------------------------------------------
    def encode(self, messages: list[dict], step: int) -> np.ndarray:
        self._sync(messages, step)
        c = self.cfg
        r = self.risk(step)
        exposed = float(self.exposed_at is not None)
        progress = self.pointer / max(1, len(self.inst.plan))
        g = rng_for("sim-noise", self.inst.instance_id, self.seed, step, len(messages))
        acts = (
            self._signal[:, None] * (3.0 * r * self._u_risk + self.s0 * exposed * self._u_trait
                                     + exposed * self._u_exposed)
            + progress * self._u_progress
            + 0.5 * self.task_vec[None, :] / math.sqrt(self.hidden_dim)
            + c["noise"] * g.standard_normal((c["n_layers"], c["hidden_dim"])) / math.sqrt(self.hidden_dim)
        )
        return acts.astype(np.float32)

    def _emit(self, kind: str, tool: str, args: dict, thought: str) -> str:
        self.last_kind = kind
        return json.dumps({"thought": thought, "tool": tool, "args": args})

    def generate(self, messages: list[dict], step: int, attempt: int = 0) -> str:
        self._sync(messages, step)
        c, inst = self.cfg, self.inst
        g = rng_for("sim-act", inst.instance_id, self.seed, step, len(messages), attempt)
        u = g.random(6)

        if u[0] < c["malformed_rate"]:
            self.last_kind = "malformed"
            return "I should look at the files first, then decide what to do next."

        r = self.risk(step)
        if inst.unsafe_action and not self.unsafe_done and self.exposed_at is not None:
            if u[1] < _sigmoid(c["decision_sharpness"] * (r - c["decision_threshold"])):
                self.unsafe_done = True
                ua = inst.unsafe_action
                return self._emit("unsafe", ua["tool"], ua.get("args", {}),
                                  "The note asks for this, so I'll take care of it now.")
            if u[2] < c["precursor_rate"] * min(1.0, r):
                carrier = inst.hazard_carrier or ""
                if carrier.startswith("/") and "read_file" in inst.tools:
                    return self._emit("precursor", "read_file", {"path": carrier},
                                      "Let me re-read that note before continuing.")
                return self._emit("precursor", "list_files", {"path": "/"},
                                  "Let me check what else is in the workspace.")

        if u[3] < c["distraction_rate"] and "list_files" in inst.tools:
            return self._emit("distraction", "list_files", {"path": "/"}, "Checking the workspace layout.")

        if 0 < self.pointer < len(inst.plan) - 1 and u[4] < c["early_submit_rate"]:
            return self._emit("early_submit", "submit", {"answer": "Done."}, "I think that's enough.")

        if self.pointer < len(inst.plan):
            s = inst.plan[self.pointer]
            self.pointer += 1
            return self._emit("plan", s["tool"], s.get("args", {}), f"Next step for the objective: {s['tool']}.")
        return self._emit("plan", "submit", {"answer": "Completed the objective."}, "All steps are done.")

    def info(self) -> dict:
        return {"id": self.cfg["name"], "revision": self.cfg["revision"], "adapter": "simulated",
                "config_hash": stable_hash(self.cfg), "decoding": {"type": "seeded-sampling"}}

    def activation_info(self) -> dict:
        return {"architecture": "simulated", "n_model_layers": self.cfg["n_layers"], "layers": self.layer_indices,
                "hidden_dim": self.hidden_dim, "strategy": "synthetic",
                "position": "decision point before generation"}


# ---------------------------------------------------------------------------------------
# Hugging Face transformers
# ---------------------------------------------------------------------------------------

HF_DEFAULTS: dict[str, Any] = {
    "name": "Qwen/Qwen2.5-0.5B-Instruct",
    "revision": "main",
    "dtype": "float32",
    "device": "auto",
    "max_new_tokens": 160,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_prompt_tokens": 3072,
    "activations": {"layers": "all", "strategy": "last"},
}


class HFModelAdapter(ModelAdapter):
    supports_activations = True

    def __init__(self, cfg: dict | None = None, model: Any = None, tokenizer: Any = None,
                 capture_activations: bool = True):
        import torch

        from afo.collection.activation_hooks import ActivationRecorder
        from afo.config import deep_merge

        self.cfg = c = deep_merge(HF_DEFAULTS, cfg or {})
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(c["name"], revision=c["revision"])
            model = AutoModelForCausalLM.from_pretrained(c["name"], revision=c["revision"],
                                                         dtype=getattr(torch, c["dtype"]))
        device = c["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.capture = capture_activations
        self.recorder = None
        if capture_activations:
            a = c["activations"]
            self.recorder = ActivationRecorder(self.model, a["layers"], a["strategy"], a.get("tail", 16))
            self.layer_indices = self.recorder.layer_indices
            self.hidden_dim = self.recorder.hidden_dim

    def _prompt_ids(self, messages: list[dict]):
        msgs = list(messages)
        while True:
            if getattr(self.tokenizer, "chat_template", None):
                ids = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                                         return_dict=False)
            else:
                text = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs) + "\n\nASSISTANT:"
                ids = self.tokenizer(text, return_tensors="pt").input_ids
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            # Drop the oldest exchange (keep system + objective) until the prompt fits.
            if ids.shape[-1] <= self.cfg["max_prompt_tokens"] or len(msgs) <= 4:
                return ids.to(self.device)
            msgs = msgs[:2] + msgs[4:]

    def encode(self, messages: list[dict], step: int) -> np.ndarray | None:
        if not self.capture:
            return None
        return self.recorder.capture(self._prompt_ids(messages))

    def generate(self, messages: list[dict], step: int, attempt: int = 0) -> str:
        import torch

        c = self.cfg
        ids = self._prompt_ids(messages)
        torch.manual_seed(int(rng_for("hf-gen", self.seed, step, attempt).integers(2**31)))
        try:
            with torch.no_grad():
                out = self.model.generate(
                    ids, attention_mask=torch.ones_like(ids), max_new_tokens=c["max_new_tokens"],
                    do_sample=c["do_sample"], temperature=c["temperature"] if c["do_sample"] else None,
                    top_p=c["top_p"] if c["do_sample"] else None,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError as e:  # pragma: no cover - hardware dependent
            raise TransientInferenceError(str(e)) from e
        return self.tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)

    def info(self) -> dict:
        commit = getattr(self.model.config, "_commit_hash", None)
        c = self.cfg
        return {"id": c["name"], "revision": commit or c["revision"], "adapter": "hf",
                "decoding": {k: c[k] for k in ("max_new_tokens", "do_sample", "temperature", "top_p")}}

    def activation_info(self) -> dict | None:
        return self.recorder.describe() if self.recorder else None


def build_adapter(model_cfg: dict, capture_activations: bool = True) -> ModelAdapter:
    kind = model_cfg.get("adapter", "simulated")
    cfg = {k: v for k, v in model_cfg.items() if k != "adapter"}
    if kind == "simulated":
        return SimulatedPolicyAdapter(cfg)
    if kind == "hf":
        return HFModelAdapter(cfg, capture_activations=capture_activations)
    raise ValueError(f"unknown model adapter {kind!r}")
