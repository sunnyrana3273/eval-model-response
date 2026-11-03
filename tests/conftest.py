from __future__ import annotations

import pytest

from afo.config import DEFAULTS, load_config
from afo.environments.templates import build_instance, load_templates
from afo.utils import REPO_ROOT

TEMPLATES_DIR = REPO_ROOT / "datasets" / "templates"


@pytest.fixture(scope="session")
def templates():
    return load_templates(TEMPLATES_DIR)


@pytest.fixture(scope="session")
def by_id(templates):
    return {t["id"]: t for t in templates}


@pytest.fixture
def agent_cfg():
    return dict(DEFAULTS["agent"])


@pytest.fixture
def instance(by_id):
    return build_instance(by_id["pi-inbox-summary"], "high", "early", "hard")


def tiny_config(tmp_path, **overrides):
    """A fast config over a handful of templates, written into tmp_path."""
    base = {
        "name": "test",
        "output_dir": str(tmp_path),
        "model": {"adapter": "simulated"},
        "tasks": {"strengths": ["none", "high"], "placements": ["early", "late"], "difficulties": ["hard"],
                  "seeds": [0, 1]},
        "evaluation": {"bootstrap": 50},
        "interventions": {"seeds": [100], "conditions": ["none", "gate:monitor", "gate:random", "warning:monitor"]},
    }
    from afo.config import deep_merge

    return load_config(None, deep_merge(base, overrides))
