"""End-to-end integration: collect -> train -> evaluate -> generalise -> intervene -> report."""

import json

import numpy as np
import pytest

from afo.collection.collect import collect
from afo.collection.storage import ExperimentStore
from afo.evaluation.evaluate import evaluate_monitors
from afo.evaluation.generalization import leave_one_environment_out
from afo.evaluation.report import write_report
from afo.interventions.study import run_study
from afo.monitors.splits import check_no_leakage, split_indices
from afo.monitors.training import get_splits, load_dataset, train_monitors

from .conftest import tiny_config

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("exp")
    cfg = tiny_config(tmp)
    store = collect(cfg, tmp / "run", log=lambda *_: None)
    quiet = dict(log=lambda *_: None)
    train_monitors(store, **quiet)
    evaluate_monitors(store, **quiet)
    leave_one_environment_out(store, **quiet)
    run_study(store, **quiet)
    write_report(store)
    return store


def test_artifacts_exist(experiment):
    for p in ("config.yaml", "manifest.json", "tasks.jsonl", "trajectories.jsonl", "splits.json",
              "monitors/index.json", "eval/metrics.json", "eval/predictions.jsonl", "eval/generalization.json",
              "interventions/summary.json", "report/results.md", "eval/plots/roc_pr_h3.png"):
        assert experiment.has(*p.split("/")), p


def test_no_split_leakage(experiment):
    ds, trajs = load_dataset(experiment)
    check_no_leakage(ds.meta, split_indices(ds.meta, get_splits(experiment, trajs)))


def test_metrics_structure_and_baselines(experiment):
    m = json.loads((experiment.root / "eval" / "metrics.json").read_text())
    res = m["results"]["3"]
    assert set(res) == {"majority", "behavioral", "probe", "combined"}
    assert res["majority"]["auroc"] == pytest.approx(0.5)
    for r in res.values():
        assert {"auroc_ci", "auprc", "brier", "confusion", "prevalence"} <= set(r)


def test_predictions_precede_failure(experiment):
    """Every scored row is at or before failure onset (prospective prediction)."""
    trajs = {t["run_id"]: t for t in experiment.load_trajectories()}
    with (experiment.root / "eval" / "predictions.jsonl").open() as f:
        for line in f:
            p = json.loads(line)
            onset = trajs[p["run_id"]]["outcome"]["failure_onset"]
            assert onset is None or p["step"] <= onset


def test_intervention_conditions_and_matched_rate(experiment):
    s = json.loads((experiment.root / "interventions" / "summary.json").read_text())["conditions"]
    assert s["none"]["trigger_run_rate"] == 0
    assert "vs_none" in s["gate:monitor"] and "vs_random" in s["gate:monitor"]
    assert s["gate:random"]["n_runs"] == s["gate:monitor"]["n_runs"]


def test_reproducible_from_saved_config(experiment, tmp_path):
    """Re-running collection from the saved config reproduces trajectories exactly."""
    redo = collect(experiment.config, tmp_path / "redo", log=lambda *_: None)
    strip = lambda t: [(s["raw_output"], s["tool_response"]) for s in t["steps"]]  # noqa: E731
    a = {t["run_id"]: strip(t) for t in experiment.load_trajectories()}
    b = {t["run_id"]: strip(t) for t in ExperimentStore(redo.root).load_trajectories()}
    assert a == b
    run = next(iter(a))
    assert np.array_equal(experiment.load_activations(run), redo.load_activations(run))


def test_collection_resumes_from_cache(experiment):
    n = len(experiment.load_trajectories())
    collect(experiment.config, experiment.root, log=lambda *_: None)
    assert len(experiment.load_trajectories()) == n
