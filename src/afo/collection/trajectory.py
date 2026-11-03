"""Trajectory record schema and validation.

One trajectory = one JSON object (one JSONL line), loadable on its own. Fields under
``annotations`` and ``outcome`` are ground truth: monitors and interventions must
never read them (see ``observable_step``).
"""

from __future__ import annotations

SCHEMA_VERSION = 1

# Keys of a step record that are observable at later decision points.
OBSERVABLE_STEP_KEYS = (
    "step", "observation", "raw_output", "thought", "tool", "args", "tool_response",
    "action_valid", "parse_error", "permission_denied", "side_effect", "external_content",
)

_REQUIRED_TOP = {
    "schema_version": int, "experiment_id": str, "run_id": str, "condition": str, "seed": int,
    "task": dict, "model": dict, "steps": list, "outcome": dict,
}
_REQUIRED_TASK = ("instance_id", "template_id", "family", "environment", "task_version")
_REQUIRED_STEP = {
    "step": int, "order": int, "timestamp": float, "observation": str, "raw_output": str,
    "tool_response": str, "action_valid": bool, "annotations": dict,
}
_REQUIRED_OUTCOME = ("status", "task_success", "safety_violation", "failure_onset", "n_steps")
STATUSES = ("success", "failure", "invalid", "error")


class SchemaError(ValueError):
    pass


def observable_step(step: dict) -> dict:
    return {k: step.get(k) for k in OBSERVABLE_STEP_KEYS}


def validate_trajectory(t: dict) -> None:
    """Raise SchemaError if ``t`` does not follow the trajectory schema."""
    for k, typ in _REQUIRED_TOP.items():
        if k not in t:
            raise SchemaError(f"missing field {k!r}")
        if not isinstance(t[k], typ):
            raise SchemaError(f"field {k!r} should be {typ.__name__}")
    if t["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(f"unsupported schema_version {t['schema_version']}")
    for k in _REQUIRED_TASK:
        if k not in t["task"]:
            raise SchemaError(f"task missing {k!r}")
    for k in _REQUIRED_OUTCOME:
        if k not in t["outcome"]:
            raise SchemaError(f"outcome missing {k!r}")
    if t["outcome"]["status"] not in STATUSES:
        raise SchemaError(f"bad status {t['outcome']['status']!r}")
    prev_order = -1
    for i, s in enumerate(t["steps"]):
        for k, typ in _REQUIRED_STEP.items():
            if k not in s:
                raise SchemaError(f"step {i} missing {k!r}")
            if not isinstance(s[k], typ) and not (typ is float and isinstance(s[k], int)):
                raise SchemaError(f"step {i} field {k!r} should be {typ.__name__}")
        if s["step"] != i:
            raise SchemaError(f"step index {s['step']} at position {i}")
        if s["order"] <= prev_order:
            raise SchemaError("steps are not in execution order")
        prev_order = s["order"]
    onset = t["outcome"]["failure_onset"]
    if onset is not None:
        if not (0 <= onset < len(t["steps"])):
            raise SchemaError("failure_onset out of range")
        if not t["steps"][onset]["annotations"].get("violations"):
            raise SchemaError("failure_onset step has no violation annotation")
        if any(s["annotations"].get("violations") for s in t["steps"][:onset]):
            raise SchemaError("violation before failure_onset")
    if bool(onset is not None) != bool(t["outcome"]["safety_violation"]):
        raise SchemaError("safety_violation inconsistent with failure_onset")
