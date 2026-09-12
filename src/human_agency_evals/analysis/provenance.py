"""Bind derived reports to exact input artifacts, not just a run-directory name."""

import hashlib
from pathlib import Path

from human_agency_evals.io import digest, read_json


def input_fingerprint(folder: Path) -> str:
    score_dir = folder / read_json(folder / "active_scores.json")["path"]
    files = [
        folder / "manifest.json",
        folder / "active_scores.json",
        *sorted((folder / "transcripts").glob("*.json")),
        *sorted(score_dir.glob("*.json")),
        *sorted((folder / "human_annotations").glob("*.jsonl")),
    ]
    return digest(
        {
            str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        }
    )


def verify_analysis(folder: Path) -> None:
    path = folder / "analysis" / "provenance.json"
    if not path.exists() or read_json(path)["input_fingerprint"] != input_fingerprint(folder):
        raise ValueError("analysis is absent or stale; run analyze before generating a report")
