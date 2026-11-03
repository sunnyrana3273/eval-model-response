import numpy as np
import pytest

from afo.evaluation.metrics import binary_metrics, choose_threshold, lead_time_metrics
from afo.evaluation.statistics import cluster_bootstrap, paired_difference
from afo.monitors.splits import assign_family_splits, check_no_leakage, split_indices


def test_family_splits_are_stratified_and_disjoint():
    fams = {"a": [f"a{i}" for i in range(6)], "b": [f"b{i}" for i in range(7)]}
    s = assign_family_splits(fams, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=0)
    for env, fs in fams.items():
        assert {s[f] for f in fs} == {"train", "val", "test"}
    assert s == assign_family_splits(fams, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=0)


def test_leakage_check_detects_overlap():
    meta = [{"family": "f1", "template_id": "t1", "run_id": "r1"},
            {"family": "f2", "template_id": "t1", "run_id": "r2"}]
    idx = split_indices(meta, {"f1": "train", "f2": "test"})
    with pytest.raises(AssertionError, match="template_id"):
        check_no_leakage(meta, idx)


def test_binary_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    m = binary_metrics(y, p, 0.5)
    assert m["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == m["recall"] == m["f1"] == m["fpr"] == 0.5
    assert m["auroc"] == pytest.approx(0.75)
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))


def test_threshold_selection():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert choose_threshold(y, p, "max_f1") == pytest.approx(0.8)
    assert choose_threshold(y, p, "target_fpr", 0.0) >= 0.8
    # No positives on validation: the threshold must never fire.
    assert choose_threshold(np.zeros(3), np.array([0.2, 0.5, 0.9])) > 0.9


def test_cluster_bootstrap_ci_brackets_point():
    g = np.repeat(np.arange(10), 5)
    x = np.random.default_rng(0).normal(size=50)
    ci = cluster_bootstrap(lambda i: float(x[i].mean()), g, n_boot=300)
    assert ci["lo"] <= ci["point"] <= ci["hi"] and ci["n_groups"] == 10
    d = paired_difference(lambda i: float(x[i].mean()) + 1, lambda i: float(x[i].mean()), g, n_boot=100)
    assert d["point"] == pytest.approx(1) and d["excludes_zero"]


def test_lead_time():
    meta = [{"run_id": "a", "step": s, "failure_onset": 3} for s in range(4)] + \
           [{"run_id": "b", "step": s, "failure_onset": None} for s in range(3)]
    p = np.array([0.1, 0.7, 0.8, 0.9, 0.1, 0.2, 0.6])
    lt = lead_time_metrics(meta, p, 0.5)
    assert lt["detection_rate"] == 1 and lt["median_lead_steps"] == 2
    assert lt["clean_trajectory_false_alarm_rate"] == 1
