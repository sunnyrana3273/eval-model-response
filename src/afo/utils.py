"""Small shared helpers: hashing, seeded RNG, JSON I/O, provenance."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=json_default)


def stable_hash(obj: Any, n: int = 8) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()[:n]


def rng_for(*keys: Any) -> np.random.Generator:
    """Deterministic generator derived from arbitrary keys (seed, step, ...)."""
    digest = hashlib.sha256(canonical_json(list(keys)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def dump_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=json_default) + "\n")


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def append_jsonl(path: Path | str, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=json_default) + "\n")
        f.flush()


def write_jsonl(path: Path | str, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, default=json_default) + "\n")


def read_jsonl(path: Path | str) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def git_info() -> dict:
    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True
            ).stdout.strip()
        except Exception:
            return None

    return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))}


def env_info() -> dict:
    pkgs = {}
    for name in ("numpy", "scikit-learn", "torch", "transformers", "pyyaml", "matplotlib"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": pkgs}
