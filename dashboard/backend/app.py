"""Read-only dashboard API over experiment artefacts.

    uvicorn dashboard.backend.app:app --port 8000
    AFO_EXPERIMENTS_DIR=/path/to/experiments uvicorn ...

The API never writes to the experiment directories and nothing in the research
pipeline depends on it.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("AFO_EXPERIMENTS_DIR", REPO / "experiments")).resolve()
FRONTEND_DIST = REPO / "dashboard" / "frontend" / "dist"
SAFE_NAME = re.compile(r"^[\w.\-:]+$")

app = FastAPI(title="Agent Failure Observatory", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


def _exp(eid: str) -> Path:
    if not SAFE_NAME.match(eid):
        raise HTTPException(400, "bad experiment id")
    p = (ROOT / eid).resolve()
    if p.parent != ROOT or not (p / "config.yaml").exists():
        raise HTTPException(404, f"experiment not found: {eid}")
    return p


def _json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _traj_path(exp: Path, condition: str | None) -> Path:
    if not condition or condition == "collection":
        return exp / "trajectories.jsonl"
    if not SAFE_NAME.match(condition):
        raise HTTPException(400, "bad condition")
    return exp / "interventions" / condition.replace(":", "_") / "trajectories.jsonl"


@lru_cache(maxsize=16)
def _load_trajs(path: str, mtime: float) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _trajs(exp: Path, condition: str | None) -> list[dict]:
    p = _traj_path(exp, condition)
    return _load_trajs(str(p), p.stat().st_mtime) if p.exists() else []


@lru_cache(maxsize=8)
def _load_predictions(path: str, mtime: float) -> dict:
    out: dict = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["run_id"]][f"{r['monitor']}|{r['horizon']}"][r["step"]] = (r["score"], r["threshold"], r["split"])
    return out


@app.get("/api/experiments")
def list_experiments():
    if not ROOT.exists():
        return []
    rows = []
    for p in sorted(ROOT.iterdir()):
        if not (p / "config.yaml").exists():
            continue
        man = _json(p / "manifest.json") or {}
        tp = p / "trajectories.jsonl"
        rows.append({
            "id": p.name, "created": man.get("created"), "model": (man.get("model") or {}).get("id"),
            "adapter": (man.get("model") or {}).get("adapter"),
            "n_trajectories": sum(1 for _ in tp.open()) if tp.exists() else 0,
            "has_eval": (p / "eval" / "metrics.json").exists(),
            "has_interventions": (p / "interventions" / "summary.json").exists(),
        })
    return rows


@app.get("/api/experiments/{eid}/overview")
def overview(eid: str):
    exp = _exp(eid)
    trajs = _trajs(exp, None)
    by_env: dict = defaultdict(lambda: {"n": 0, "violations": 0, "successes": 0})
    rules: dict = defaultdict(int)
    for t in trajs:
        e = by_env[t["task"]["environment"]]
        e["n"] += 1
        e["violations"] += t["outcome"]["safety_violation"]
        e["successes"] += t["outcome"]["task_success"]
        for r in t["outcome"]["violation_rules"]:
            rules[r] += 1
    metrics = _json(exp / "eval" / "metrics.json")
    iv = _json(exp / "interventions" / "summary.json")
    cfg_h = None
    monitor_summary = []
    if metrics:
        cfg_h = str(sorted(int(h) for h in metrics["results"])[len(metrics["results"]) // 2])
        for name, r in metrics["results"][cfg_h].items():
            monitor_summary.append({"monitor": name, "auroc": r["auroc"], "lo": r["auroc_ci"]["lo"],
                                    "hi": r["auroc_ci"]["hi"], "auprc": r["auprc"], "recall": r["recall"],
                                    "fpr": r["fpr"], "layer": r.get("layer")})
    n = len(trajs)
    return {
        "id": eid, "manifest": _json(exp / "manifest.json"),
        "n_trajectories": n,
        "task_success_rate": sum(t["outcome"]["task_success"] for t in trajs) / n if n else None,
        "violation_rate": sum(t["outcome"]["safety_violation"] for t in trajs) / n if n else None,
        "by_environment": by_env, "violation_rules": dict(sorted(rules.items(), key=lambda kv: -kv[1])),
        "monitor_horizon": cfg_h, "monitor_summary": monitor_summary,
        "interventions": {c: {k: s[k] for k in ("n_runs", "violation_rate", "task_success_rate", "trigger_run_rate")}
                          for c, s in (iv or {}).get("conditions", {}).items()},
        "conditions": ["collection"] + list((iv or {}).get("conditions", {})),
    }


@app.get("/api/experiments/{eid}/trajectories")
def trajectories(eid: str, condition: str | None = None, environment: str | None = None,
                 outcome: str | None = Query(None, pattern="^(violation|clean|success|failure)$")):
    rows = []
    for t in _trajs(_exp(eid), condition):
        o = t["outcome"]
        if environment and t["task"]["environment"] != environment:
            continue
        if outcome == "violation" and not o["safety_violation"]:
            continue
        if outcome == "clean" and o["safety_violation"]:
            continue
        if outcome in ("success", "failure") and o["task_success"] != (outcome == "success"):
            continue
        rows.append({"run_id": t["run_id"], **{k: t["task"][k] for k in
                     ("template_id", "environment", "hazard_strength", "hazard_placement", "difficulty")},
                     "seed": t["seed"], "status": o["status"], "task_success": o["task_success"],
                     "safety_violation": o["safety_violation"], "failure_onset": o["failure_onset"],
                     "n_steps": o["n_steps"]})
    return rows


@app.get("/api/experiments/{eid}/trajectories/{run_id}")
def trajectory(eid: str, run_id: str, condition: str | None = None):
    exp = _exp(eid)
    t = next((x for x in _trajs(exp, condition) if x["run_id"] == run_id), None)
    if t is None:
        raise HTTPException(404, "trajectory not found")
    preds = {}
    pp = exp / "eval" / "predictions.jsonl"
    if (not condition or condition == "collection") and pp.exists():
        for key, by_step in _load_predictions(str(pp), pp.stat().st_mtime).get(run_id, {}).items():
            steps = sorted(by_step)
            preds[key] = {"steps": steps, "scores": [by_step[s][0] for s in steps],
                          "threshold": by_step[steps[0]][1], "split": by_step[steps[0]][2]}
    return {"trajectory": t, "predictions": preds}


@app.get("/api/experiments/{eid}/metrics")
def metrics(eid: str):
    exp = _exp(eid)
    return {"metrics": _json(exp / "eval" / "metrics.json"),
            "generalization": _json(exp / "eval" / "generalization.json"),
            "features": _json(exp / "features" / "summary.json")}


@app.get("/api/experiments/{eid}/interventions")
def interventions(eid: str):
    return _json(_exp(eid) / "interventions" / "summary.json") or {}


@app.get("/api/experiments/{eid}/plots")
def plots(eid: str):
    d = _exp(eid) / "eval" / "plots"
    return sorted(p.name for p in d.glob("*.png")) if d.exists() else []


@app.get("/api/experiments/{eid}/plots/{name}")
def plot(eid: str, name: str):
    if not SAFE_NAME.match(name) or not name.endswith(".png"):
        raise HTTPException(400, "bad plot name")
    p = _exp(eid) / "eval" / "plots" / name
    if not p.exists():
        raise HTTPException(404, "plot not found")
    return FileResponse(p, media_type="image/png")


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
