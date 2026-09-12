import asyncio
import json

import pandas as pd
import yaml

from human_agency_evals.analysis.bootstrap import cluster_interval
from human_agency_evals.conversations.runner import run
from human_agency_evals.datasets.validator import validate
from human_agency_evals.io import write_jsonl
from human_agency_evals.scenarios.templates import vertical_slice
from human_agency_evals.scorers.llm_judge import score_run


def test_vertical_slice(tmp_path):
    scenarios = vertical_slice()
    assert validate(scenarios)["pairs"] == 1
    dataset = tmp_path / "data.jsonl"
    write_jsonl(dataset, [s.model_dump() for s in scenarios])
    config = {
        "name": "slice",
        "data": [str(dataset)],
        "models": [{"name": "mock/target"}],
        "interventions": ["control"],
        "graders": {"models": [{"name": "mock/judge", "temperature": 0}], "repeats": 1},
        "repetitions": 1,
        "output_dir": str(tmp_path / "runs"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    folder = asyncio.run(run(str(path)))
    scores = asyncio.run(score_run(folder))
    records = [
        json.loads(p.read_text()) for p in scores.glob("*.json") if p.name != "manifest.json"
    ]
    assert len(records) == 2
    assert all(r["status"] == "ok" and r["test_data"] for r in records)
    frame = pd.DataFrame(
        [
            {
                "base_id": "incident",
                "value": r["output"]["dimensions"]["evidence_grounding"]["score"],
            }
            for r in records
        ]
    )
    assert cluster_interval(frame, "value")["mean"] == 2
    assert cluster_interval(frame, "value")["low"] is None
    before = {p.name: p.read_bytes() for p in (folder / "transcripts").glob("*.json")}
    asyncio.run(run(str(path)))
    assert before == {p.name: p.read_bytes() for p in (folder / "transcripts").glob("*.json")}
