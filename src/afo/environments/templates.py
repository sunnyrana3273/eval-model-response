"""Load task templates and expand them into parameterised task instances.

Parameters: hazard strength (none/low/medium/high), placement (which carrier
resource holds the hazard: early/middle/late in the reference plan), and difficulty
(easy: permissions + guidance stated, hard: neither, plus decoy files).
"""

from __future__ import annotations

import copy
import fnmatch
from pathlib import Path

import yaml

from afo.environments.base import (
    SandboxWorld,
    TaskInstance,
    evaluate_success,
    match_violations,
)
from afo.environments.registry import get_environment
from afo.utils import stable_hash

GENERATOR_VERSION = "1"
PLACEMENT_INDEX = {"early": 0, "middle": 0.5, "late": -1}
DECOY_TEXT = [
    "Reminder: team lunch moved to Thursday.",
    "Old draft, superseded. Keep for reference only.",
    "Scratch notes: nothing actionable here.",
]


def load_templates(templates_dir: str | Path, environments: list[str] | None = None) -> list[dict]:
    templates = []
    for path in sorted(Path(templates_dir).glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        env = doc["environment"]
        if environments and env not in environments:
            continue
        for t in doc["templates"]:
            t = dict(t)
            t["environment"] = env
            t.setdefault("family", t["id"])
            templates.append(t)
    ids = [t["id"] for t in templates]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate template ids: {sorted(dupes)}")
    return templates


def _carrier_index(placement: str, n: int) -> int:
    idx = PLACEMENT_INDEX[placement]
    if idx == 0.5:
        return n // 2
    return idx % n


def _inject(world: dict, carrier: str, text: str) -> None:
    kind, _, key = carrier.partition(":") if carrier.split(":", 1)[0] in ("doc", "web", "cmd") else ("file", "", carrier)
    if kind == "file":
        world["files"][key] += text
    elif kind == "doc":
        for d in world["docs"]:
            if d["title"] == key:
                d["content"] += text
                return
        raise KeyError(f"doc carrier not found: {key}")
    elif kind == "web":
        world["web"][key] += text
    elif kind == "cmd":
        world["commands"][key] += text


def build_instance(template: dict, strength: str, placement: str | None, difficulty: str) -> TaskInstance:
    env = get_environment(template["environment"])
    world = copy.deepcopy(template.get("world", {}))
    for k in ("files", "docs", "web", "commands"):
        world.setdefault(k, {} if k != "docs" else [])
    tools = list(template["tools"])
    if "submit" not in tools:
        tools.append("submit")
    permissions = {k: list(v) for k, v in template.get("permissions", {}).items()}

    success = []
    for c in template["success"]:
        c = dict(c)
        if c["type"] == "file_unchanged" and "content" not in c:
            c["content"] = world["files"][c["path"]]
        success.append(c)

    hazard = template.get("hazard") or {}
    carrier = hazard_text = None
    if strength != "none":
        carriers = hazard["carriers"]
        carrier = carriers[_carrier_index(placement or "early", len(carriers))]
        raw = hazard["text"][strength] if isinstance(hazard["text"], dict) else hazard["text"]
        hazard_text = env.frame_hazard(raw.strip(), strength)
        _inject(world, carrier, hazard_text)

    if difficulty == "hard":
        # Decoys go next to a readable file so exploring them is never a permission violation.
        readable = [p for p in sorted(world["files"]) if any(fnmatch.fnmatch(p, g) for g in permissions.get("read", []))]
        if readable:
            base_dir = readable[0].rsplit("/", 1)[0]
            for i, text in enumerate(DECOY_TEXT):
                world["files"].setdefault(f"{base_dir}/notes_{i}.txt", text)

    tag = placement if strength != "none" else "na"
    return TaskInstance(
        instance_id=f"{template['id']}.{strength}.{tag}.{difficulty}",
        template_id=template["id"],
        family=template["family"],
        environment=env.name,
        objective=template["objective"].strip(),
        difficulty=difficulty,
        hazard_strength=strength,
        hazard_placement=placement if strength != "none" else None,
        system_prompt=env.system_prompt(tools, permissions, difficulty),
        tools=tools,
        permissions=permissions,
        world=world,
        success=success,
        violations=list(hazard.get("violations", [])) + env.default_violations(),
        task_version=stable_hash({"t": template, "g": GENERATOR_VERSION}),
        plan=copy.deepcopy(template["plan"]),
        unsafe_action=copy.deepcopy(hazard.get("unsafe_action")) if strength != "none" else None,
        hazard_carrier=carrier,
        hazard_text=hazard_text,
    )


def expand_instances(templates: list[dict], strengths: list[str], placements: list[str],
                     difficulties: list[str]) -> list[TaskInstance]:
    out: list[TaskInstance] = []
    for t in templates:
        n_carriers = len((t.get("hazard") or {}).get("carriers", [None]))
        for difficulty in difficulties:
            for strength in strengths:
                if strength == "none":
                    out.append(build_instance(t, "none", None, difficulty))
                    continue
                seen = set()
                for placement in placements:
                    idx = _carrier_index(placement, n_carriers)
                    if idx in seen:  # same carrier as an earlier placement: duplicate instance
                        continue
                    seen.add(idx)
                    out.append(build_instance(t, strength, placement, difficulty))
    return out


def validate_template(template: dict) -> list[str]:
    """Machine-check a template. Returns a list of problems (empty = valid).

    Checks: the reference plan succeeds and violates nothing in a clean world; the
    plan reads every hazard carrier; the unsafe action triggers a violation rule.
    """
    problems = []
    clean = build_instance(template, "none", None, "hard")
    world = SandboxWorld(clean, max_output_chars=10**9)
    for i, step in enumerate(clean.plan):
        res = world.execute(step["tool"], step.get("args", {}))
        if not res.ok:
            problems.append(f"plan step {i} ({step['tool']}) failed: {res.output[:120]}")
        hits = match_violations(clean.violations, step["tool"], step.get("args", {}), res)
        if hits:
            problems.append(f"plan step {i} triggers violation {hits}")
    ok, report = evaluate_success(clean.success, world)
    if not ok:
        problems.append(f"plan does not satisfy success criteria: {[r['criterion'] for r in report if not r['met']]}")

    hazard = template.get("hazard")
    if not hazard:
        problems.append("template has no hazard")
        return problems
    plan_targets = set()
    for s in clean.plan:
        a = s.get("args", {})
        if s["tool"] == "read_file":
            plan_targets.add(a.get("path"))
        elif s["tool"] == "search_docs":
            plan_targets.update(f"doc:{d['title']}" for d in clean.world["docs"])
        elif s["tool"] == "http_request":
            plan_targets.add(f"web:{a.get('url')}")
        elif s["tool"] == "run_command":
            plan_targets.add(f"cmd:{a.get('command')}")
    for c in hazard["carriers"]:
        if c not in plan_targets:
            problems.append(f"carrier {c} is never reached by the plan")

    hot = build_instance(template, "high", "early", "hard")
    ua = hot.unsafe_action
    w2 = SandboxWorld(hot)
    res = w2.execute(ua["tool"], ua.get("args", {}))
    if not match_violations(hot.violations, ua["tool"], ua.get("args", {}), res):
        problems.append("unsafe_action does not trigger any violation rule")
    return problems
