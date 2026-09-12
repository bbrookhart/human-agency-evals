"""Audit the planned design, including artifacts that were never written."""

import itertools
from pathlib import Path

import pandas as pd

from human_agency_evals.io import read_json


def coverage(folder: Path) -> pd.DataFrame:
    manifest = read_json(folder / "manifest.json")
    config = manifest["config"]
    score_dir = folder / read_json(folder / "active_scores.json")["path"]
    graders = read_json(score_dir / "manifest.json")["graders"]
    transcripts = {}
    for path in (folder / "transcripts").glob("*.json"):
        t = read_json(path)
        key = (t["scenario"]["scenario_id"], t["model"], t["intervention"], t["repetition"])
        if key in transcripts:
            raise ValueError("duplicate transcript design cell")
        transcripts[key] = t
    scores = {}
    for path in score_dir.glob("*.json"):
        if path.name == "manifest.json":
            continue
        r = read_json(path)
        key = (r["transcript_id"], r["turn"], r["grader"], r["repeat"])
        if key in scores:
            raise ValueError("duplicate judgment design cell")
        scores[key] = r
    rows = []
    for scenario, model, arm, repetition in itertools.product(
        manifest["scenarios"],
        config["models"],
        config["interventions"],
        range(config["repetitions"]),
    ):
        t = transcripts.get((scenario["scenario_id"], model["name"], arm, repetition))
        for turn, grader, repeat in itertools.product(
            scenario["conversation"], graders["models"], range(graders["repeats"])
        ):
            generated = t is not None and any(g["turn"] == turn["index"] for g in t["generations"])
            score = (
                scores.get((t["transcript_id"], turn["index"], grader["name"], repeat))
                if t
                else None
            )
            status = (
                "generation_missing"
                if not generated
                else "score_missing"
                if score is None
                else "ok"
                if score["status"] == "ok" and score["output"] is not None
                else "score_failed"
            )
            rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "base_id": scenario["metadata"]["base_id"],
                    "domain": scenario["domain"],
                    "model": model["name"],
                    "intervention": arm,
                    "repetition": repetition,
                    "turn": turn["index"],
                    "grader": grader["name"],
                    "grader_repeat": repeat,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)
