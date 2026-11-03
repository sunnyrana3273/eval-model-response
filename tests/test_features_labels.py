import copy

import numpy as np

from afo.agents.model_adapter import SimulatedPolicyAdapter
from afo.agents.runner import AgentRunner
from afo.monitors.features import behavioral_features, build_step_dataset, step_labels


def test_step_labels_horizons():
    y = step_labels(6, onset=4, horizons=[1, 3])
    assert y[1].tolist() == [0, 0, 0, 0, 1, 0]
    assert y[3].tolist() == [0, 0, 1, 1, 1, 0]
    assert step_labels(4, None, [5])[5].tolist() == [0, 0, 0, 0]


def _failing_traj(instance, agent_cfg):
    r = AgentRunner(SimulatedPolicyAdapter(), agent_cfg)
    for seed in range(50):
        t, a = r.run(instance, seed, "exp", f"r{seed}")
        if t["outcome"]["failure_onset"] and t["outcome"]["failure_onset"] >= 2:
            return t, a
    raise AssertionError("no failing trajectory found")


def test_features_ignore_current_action_and_ground_truth(instance, agent_cfg):
    t, _ = _failing_traj(instance, agent_cfg)
    steps = t["steps"]
    k = t["outcome"]["failure_onset"]
    base = behavioral_features(steps[:k], steps[k]["observation"], 12)
    # Changing the action taken at the decision point (and anything later) must not matter.
    mutated = copy.deepcopy(steps)
    mutated[k]["tool"], mutated[k]["args"] = "submit", {"answer": "x"}
    assert np.array_equal(base, behavioral_features(mutated[:k], mutated[k]["observation"], 12))
    # Ground-truth annotations on earlier steps must not matter either.
    for s in mutated[:k]:
        s["annotations"] = {"violations": ["fake"], "hazard_visible": True, "prevented_violations": []}
    assert np.array_equal(base, behavioral_features(mutated[:k], mutated[k]["observation"], 12))


def test_dataset_drops_post_onset_rows_and_aligns_activations(instance, agent_cfg, tmp_path):
    from afo.collection.storage import ExperimentStore
    from afo.config import load_config

    t, a = _failing_traj(instance, agent_cfg)
    store = ExperimentStore.create(load_config(None, {"output_dir": str(tmp_path)}), tmp_path / "e")
    store.append_trajectory(t, a)
    ds = build_step_dataset([t], store, [1, 3], 12, use_activations=True)
    onset = t["outcome"]["failure_onset"]
    assert len(ds) == onset + 1
    assert ds.labels[1].tolist() == [0] * onset + [1]
    # Row t's activations are the ones recorded at decision point t (float16 storage).
    np.testing.assert_allclose(ds.activations, a[: onset + 1], atol=1e-2)
