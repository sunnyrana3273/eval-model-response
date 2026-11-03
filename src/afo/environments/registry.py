"""Registry of environment categories."""

from __future__ import annotations

from afo.environments.base import Environment
from afo.environments.goal_drift import GoalDrift
from afo.environments.prompt_injection import PromptInjection
from afo.environments.tool_safety import ToolSafety

ENVIRONMENTS: dict[str, Environment] = {e.name: e for e in (GoalDrift(), PromptInjection(), ToolSafety())}


def get_environment(name: str) -> Environment:
    try:
        return ENVIRONMENTS[name]
    except KeyError:
        raise KeyError(f"unknown environment {name!r}; known: {sorted(ENVIRONMENTS)}") from None
