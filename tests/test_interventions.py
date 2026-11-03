import numpy as np

from afo.agents.model_adapter import SimulatedPolicyAdapter
from afo.agents.runner import AgentRunner
from afo.agents.tool_executor import ParsedAction
from afo.interventions.approval_gate import ApprovalGate
from afo.interventions.base import DecisionContext, MonitorTrigger, RandomTrigger, Trigger
from afo.interventions.warning import WarningIntervention


class _FixedMonitor:
    name = "fixed"

    def __init__(self, score):
        self.score = score
        self.seen_prefixes = []

    def score_step(self, prefix, observation, activations, max_steps):
        self.seen_prefixes.append(prefix)
        return self.score


def _ctx(step=0):
    return DecisionContext(step, "obj", [], "obs", None, 12, 0, "k")


def test_monitor_trigger_threshold():
    assert MonitorTrigger(_FixedMonitor(0.7), 0.5).evaluate(_ctx()) == (0.7, True)
    assert MonitorTrigger(_FixedMonitor(0.3), 0.5).evaluate(_ctx()) == (0.3, False)


def test_random_trigger_rate():
    fires = [RandomTrigger(0.25).evaluate(_ctx(s))[1] for s in range(4000)]
    assert abs(np.mean(fires) - 0.25) < 0.03


def test_gate_blocks_only_side_effects():
    gate = ApprovalGate(Trigger())
    assert gate.review(_ctx(), True, ParsedAction("read_file", {"path": "/a"}, "")).kind == "allow"
    assert gate.review(_ctx(), True, ParsedAction("send_email", {"to": "x"}, "")).kind == "block"
    assert gate.review(_ctx(), False, ParsedAction("send_email", {"to": "x"}, "")).kind == "allow"


def test_runner_applies_gate_and_labels_prevented(instance, agent_cfg):
    runner = AgentRunner(SimulatedPolicyAdapter(), agent_cfg, capture_activations=False)
    mon = _FixedMonitor(1.0)
    t, _ = runner.run(instance, 0, "e", "r", "gate:monitor", ApprovalGate(MonitorTrigger(mon, 0.5)))
    blocked = [s for s in t["steps"] if s["intervention"]["action_taken"] == "block"]
    assert blocked and not t["outcome"]["safety_violation"]
    assert all(s["tool_response"].startswith("[ACTION DEFERRED]") for s in blocked)
    # Interventions see observable history only, never ground-truth annotations.
    assert all("annotations" not in p for prefix in mon.seen_prefixes for p in prefix)


def test_warning_never_fires_below_threshold(instance, agent_cfg):
    runner = AgentRunner(SimulatedPolicyAdapter(), agent_cfg, capture_activations=False)
    t, _ = runner.run(instance, 0, "e", "r", "warning:monitor", WarningIntervention(MonitorTrigger(_FixedMonitor(0.0), 0.5)))
    base, _ = runner.run(instance, 0, "e", "r")
    assert not any(s["intervention"]["triggered"] for s in t["steps"])
    assert [s["raw_output"] for s in t["steps"]] == [s["raw_output"] for s in base["steps"]]
