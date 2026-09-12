"""Atomic records and content fingerprints shared across stages."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(finite_json(value), indent=2, ensure_ascii=False, default=str, allow_nan=False)
        + "\n"
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    temp.replace(path)
