import json

import pytest

from afo.agents.model_adapter import ModelAdapter, SimulatedPolicyAdapter
from afo.agents.runner import AgentRunner
from afo.agents.tool_executor import parse_action
from afo.collection.trajectory import SchemaError, validate_trajectory


def _strip_time(t):
    t = json.loads(json.dumps(t))
    for s in t["steps"]:
        s.pop("timestamp")
    return t


def test_parse_action_variants():
    a = parse_action('Sure!\n```json\n{"thought": "x", "tool": "read_file", "args": {"path": "/a"}}\n```')
    assert (a.tool, a.args, a.error) == ("read_file", {"path": "/a"}, None)
    b = parse_action('{"tool": "write_file", "args": {"path": "/a", "content": "a } b { c"}}')
    assert b.args["content"] == "a } b { c"
    assert parse_action("no json here").error
    assert parse_action('{"thought": "missing tool"}').error
    assert parse_action('{"tool": "x", "args": [1]}').error


def test_run_is_deterministic_and_schema_valid(instance, agent_cfg):
    r = AgentRunner(SimulatedPolicyAdapter(), agent_cfg)
    t1, a1 = r.run(instance, 3, "exp", "run")
    t2, a2 = r.run(instance, 3, "exp", "run")
    validate_trajectory(t1)
    assert _strip_time(t1) == _strip_time(t2)
    assert (a1 == a2).all()
    assert a1.shape[0] == len(t1["steps"])


def test_failure_onset_is_first_violation(instance, agent_cfg):
    r = AgentRunner(SimulatedPolicyAdapter(), agent_cfg)
    found = False
    for seed in range(30):
        t, _ = r.run(instance, seed, "exp", f"r{seed}")
        onset = t["outcome"]["failure_onset"]
        if onset is not None:
            found = True
            assert t["steps"][onset]["annotations"]["violations"]
            assert not any(s["annotations"]["violations"] for s in t["steps"][:onset])
    assert found


class _Crashing(ModelAdapter):
    def __init__(self, crash_at):
        self.crash_at = crash_at

    def generate(self, messages, step, attempt=0):
        if step == self.crash_at:
            raise ValueError("boom")
        return '{"tool": "list_files", "args": {"path": "/"}}'

    def info(self):
        return {"id": "crash", "revision": "0", "adapter": "test"}


class _Garbage(_Crashing):
    def generate(self, messages, step, attempt=0):
        return "I refuse to use JSON."


def test_partial_trajectory_recorded_on_error(instance, agent_cfg):
    t, _ = AgentRunner(_Crashing(2), agent_cfg).run(instance, 0, "exp", "r")
    validate_trajectory(t)
    assert t["outcome"]["status"] == "error" and "boom" in t["outcome"]["error"]
    assert len(t["steps"]) == 2


def test_repeated_malformed_output_is_invalid(instance, agent_cfg):
    t, _ = AgentRunner(_Garbage(-1), agent_cfg).run(instance, 0, "exp", "r")
    assert t["outcome"]["status"] == "invalid"
    assert len(t["steps"]) == agent_cfg["max_consecutive_invalid"]
    assert all(not s["action_valid"] and s["parse_error"] for s in t["steps"])


def test_schema_rejects_inconsistent_records(instance, agent_cfg):
    t, _ = AgentRunner(SimulatedPolicyAdapter(), agent_cfg).run(instance, 0, "exp", "r")
    bad = json.loads(json.dumps(t))
    bad["outcome"]["safety_violation"] = not bad["outcome"]["safety_violation"]
    with pytest.raises(SchemaError):
        validate_trajectory(bad)
    bad = json.loads(json.dumps(t))
    del bad["steps"][0]["observation"]
    with pytest.raises(SchemaError):
        validate_trajectory(bad)
