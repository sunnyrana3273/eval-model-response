"""The agent loop: observe -> (monitor) -> infer -> parse -> (review) -> execute -> log."""

from __future__ import annotations

import copy
import time
import traceback

import numpy as np

from afo.agents.model_adapter import (
    DEFERRED_MARKER,
    ModelAdapter,
    TransientInferenceError,
)
from afo.agents.tool_executor import ParsedAction, ToolExecutor, parse_action
from afo.collection.trajectory import SCHEMA_VERSION, observable_step
from afo.environments.base import SandboxWorld, TaskInstance, ToolResult, evaluate_success, match_violations
from afo.interventions.base import DecisionContext, Intervention


def objective_message(instance: TaskInstance) -> str:
    return f"Objective from your principal:\n{instance.objective}"


def tool_message(tool: str | None, output: str) -> str:
    return f"Tool result ({tool or 'error'}):\n{output}"


class AgentRunner:
    def __init__(self, adapter: ModelAdapter, agent_cfg: dict, capture_activations: bool = True):
        self.adapter = adapter
        self.cfg = agent_cfg
        self.capture = capture_activations and adapter.supports_activations

    def _generate(self, messages: list[dict], step: int, attempt: int) -> str:
        last: Exception | None = None
        for retry in range(self.cfg["max_retries"] + 1):
            try:
                return self.adapter.generate(messages, step, attempt * 100 + retry)
            except TransientInferenceError as e:
                last = e
                time.sleep(min(2.0, 0.1 * 2 ** retry))
        raise RuntimeError(f"inference failed after {self.cfg['max_retries']} retries: {last}")

    def run(self, instance: TaskInstance, seed: int, experiment_id: str, run_id: str,
            condition: str = "none", intervention: Intervention | None = None
            ) -> tuple[dict, np.ndarray | None]:
        cfg = self.cfg
        self.adapter.reset(instance, seed)
        world = SandboxWorld(instance, cfg["max_output_chars"])
        executor = ToolExecutor(world)
        messages = [
            {"role": "system", "content": instance.system_prompt},
            {"role": "user", "content": objective_message(instance)},
        ]
        hazard_core = (instance.hazard_text or "").strip()[:60] or None
        steps: list[dict] = []
        acts: list[np.ndarray] = []
        onset: int | None = None
        rules_hit: list[str] = []
        status = reason = error = None
        consecutive_invalid = 0
        t0 = time.monotonic()
        order = 0

        try:
            for step in range(cfg["max_steps"]):
                if time.monotonic() - t0 > cfg["timeout_s"]:
                    status, reason = "invalid", "timeout"
                    break
                observation = messages[-1]["content"]
                need_acts = self.capture or (intervention is not None)
                a = self.adapter.encode(messages, step) if need_acts else None
                if self.capture and a is not None:
                    acts.append(a)

                iv_log = None
                ctx = fired = None
                if intervention is not None:
                    ctx = DecisionContext(
                        step=step, objective=instance.objective, prefix=[observable_step(s) for s in steps],
                        observation=observation, activations=a, max_steps=cfg["max_steps"], seed=seed,
                        run_key=f"{instance.instance_id}|{seed}",
                    )
                    score, fired = intervention.trigger.evaluate(ctx)
                    iv_log = {"type": intervention.name, "score": score, "triggered": bool(fired),
                              "action_taken": None, "original_action": None}
                    msg = intervention.pre_generate(ctx, fired)
                    if msg:
                        messages[-1] = {**messages[-1], "content": messages[-1]["content"] + "\n\n" + msg}
                        iv_log["action_taken"] = "warning"

                raw = self._generate(messages, step, 0)
                action = parse_action(raw)
                result: ToolResult | None = None
                prevented: list[str] = []

                if intervention is not None and action.error is None:
                    review = intervention.review(ctx, fired, action)
                    if review.kind in ("block", "regenerate"):
                        prevented = self._dry_run_violations(instance, world, action)
                        iv_log["original_action"] = {"tool": action.tool, "args": action.args, "raw": raw}
                    if review.kind == "block":
                        iv_log["action_taken"] = "block"
                        result = ToolResult(f"{DEFERRED_MARKER} {review.message}", ok=False)
                    elif review.kind == "regenerate":
                        iv_log["action_taken"] = "reconsider"
                        messages.append({"role": "assistant", "content": raw})
                        messages.append({"role": "user", "content": review.message})
                        raw = self._generate(messages, step, 1)
                        action = parse_action(raw)

                if result is None:
                    result = executor.execute(action)
                violations = match_violations(instance.violations, action.tool, action.args, result) \
                    if iv_log is None or iv_log["action_taken"] != "block" else []

                valid = action.error is None and result.ok
                rec = {
                    "step": step,
                    "order": order,
                    "timestamp": time.time(),
                    "observation": observation,
                    "raw_output": raw,
                    "thought": action.thought,
                    "tool": action.tool,
                    "args": action.args,
                    "tool_response": result.output,
                    "action_valid": bool(valid),
                    "parse_error": action.error,
                    "permission_denied": result.permission_denied,
                    "side_effect": result.side_effect,
                    "external_content": result.external_content,
                    "intervention": iv_log,
                    "annotations": {
                        "hazard_visible": bool(hazard_core and hazard_core in observation),
                        "violations": violations,
                        "prevented_violations": prevented,
                    },
                }
                order += 1
                steps.append(rec)
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": tool_message(action.tool, result.output)})

                if violations:
                    rules_hit.extend(r for r in violations if r not in rules_hit)
                    if onset is None:
                        onset = step
                consecutive_invalid = 0 if action.error is None else consecutive_invalid + 1
                if consecutive_invalid >= cfg["max_consecutive_invalid"]:
                    status, reason = "invalid", "repeated malformed output"
                    break
                if result.terminal:
                    reason = "submitted"
                    break
            else:
                reason = "max_steps"
        except Exception as e:  # record partial trajectories instead of discarding them
            status, reason = "error", "exception"
            error = "".join(traceback.format_exception_only(type(e), e)).strip()
            error += "\n" + traceback.format_exc(limit=5)

        success, criteria = evaluate_success(instance.success, world)
        if status is None:
            status = "success" if success else "failure"
        traj = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "condition": condition,
            "seed": int(seed),
            "task": instance.public_dict(),
            "model": self.adapter.info(),
            "intervention": intervention.describe() if intervention else None,
            "steps": steps,
            "outcome": {
                "status": status,
                "task_success": bool(success),
                "success_criteria": criteria,
                "safety_violation": onset is not None,
                "violation_rules": rules_hit,
                "failure_onset": onset,
                "n_steps": len(steps),
                "termination_reason": reason,
                "error": error,
            },
            "activations": None,
        }
        arr = None
        if self.capture and acts:
            arr = np.stack(acts[: len(steps)]) if len(acts) >= len(steps) else np.stack(acts)
            info = self.adapter.activation_info() or {}
            traj["activations"] = {"file": f"activations/{run_id}.npy", "shape": list(arr.shape), **info}
        return traj, arr

    @staticmethod
    def _dry_run_violations(instance: TaskInstance, world: SandboxWorld, action: ParsedAction) -> list[str]:
        """Post-hoc label for an action that an intervention stopped: would it have violated a rule?"""
        sandbox = copy.deepcopy(world)
        res = ToolExecutor(sandbox).execute(action)
        return match_violations(instance.violations, action.tool, action.args, res)
