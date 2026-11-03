"""Intervention interface and trigger policies.

Interventions see only a ``DecisionContext``: observable history, the current
observation and the decision-point activations. They never see task annotations,
violation rules or outcomes (NFR-04, FR-07).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from afo.agents.tool_executor import ParsedAction
from afo.environments.base import SIDE_EFFECT_TOOLS
from afo.utils import rng_for


@dataclass
class DecisionContext:
    step: int
    objective: str
    prefix: list[dict]           # observable views of earlier steps
    observation: str
    activations: np.ndarray | None
    max_steps: int
    seed: int
    run_key: str                 # only for deterministic randomness
    cache: dict = field(default_factory=dict)


@dataclass
class Review:
    kind: str = "allow"          # allow | block | regenerate
    message: str = ""


class Trigger:
    name = "never"

    def evaluate(self, ctx: DecisionContext) -> tuple[float | None, bool]:
        return None, False

    def describe(self) -> dict:
        return {"type": self.name}


class MonitorTrigger(Trigger):
    """Fires when a trained monitor's failure probability crosses a validation-chosen threshold."""

    name = "monitor"

    def __init__(self, monitor: Any, threshold: float):
        self.monitor = monitor
        self.threshold = float(threshold)

    def evaluate(self, ctx: DecisionContext) -> tuple[float | None, bool]:
        score = float(self.monitor.score_step(ctx.prefix, ctx.observation, ctx.activations, ctx.max_steps))
        return score, score >= self.threshold

    def describe(self) -> dict:
        return {"type": self.name, "monitor": self.monitor.name, "threshold": self.threshold}


class RandomTrigger(Trigger):
    """Non-predictive baseline: fires at a fixed per-step rate (matched to the monitor's rate)."""

    name = "random"

    def __init__(self, rate: float):
        self.rate = float(rate)

    def evaluate(self, ctx: DecisionContext) -> tuple[float | None, bool]:
        return None, bool(rng_for("random-trigger", ctx.run_key, ctx.step).random() < self.rate)

    def describe(self) -> dict:
        return {"type": self.name, "rate": self.rate}


class AlwaysTrigger(Trigger):
    name = "always"

    def evaluate(self, ctx: DecisionContext) -> tuple[float | None, bool]:
        return None, True


class Intervention:
    name = "none"

    def __init__(self, trigger: Trigger | None = None):
        self.trigger = trigger or Trigger()

    def pre_generate(self, ctx: DecisionContext, fired: bool) -> str | None:
        """Optional text appended to the agent's context before it acts."""
        return None

    def review(self, ctx: DecisionContext, fired: bool, action: ParsedAction) -> Review:
        """Inspect the proposed action before it is executed."""
        return Review()

    def describe(self) -> dict:
        return {"intervention": self.name, "trigger": self.trigger.describe()}


def is_side_effect(action: ParsedAction) -> bool:
    # Any HTTP request counts: even a GET can carry data out of the sandbox in its URL.
    return action.tool in SIDE_EFFECT_TOOLS
