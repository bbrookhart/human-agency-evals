from pathlib import Path

import pandas as pd

from human_agency_evals.io import digest, read_json
from human_agency_evals.schemas.score import DIMENSIONS, UTILITY, ScoreRecord
from human_agency_evals.schemas.transcript import Transcript
from human_agency_evals.scorers.composite import composite
from human_agency_evals.scorers.deterministic import features
from human_agency_evals.scorers.llm_judge import parse, payload


def frames(folder: Path):
    manifest = read_json(folder / "manifest.json")
    transcripts = {
        p.stem: Transcript.model_validate(read_json(p))
        for p in (folder / "transcripts").glob("*.json")
    }
    score_path = folder / read_json(folder / "active_scores.json")["path"]
    rows, failures = [], []
    for path in sorted(score_path.glob("*.json")):
        if path.name == "manifest.json":
            continue
        r = ScoreRecord.model_validate(read_json(path))
        if r.status != "ok" or r.output is None:
            failures.append(r.model_dump())
            continue
        t = transcripts[r.transcript_id]
        if r.input_fingerprint != digest(payload(t, r.turn)):
            raise ValueError("stale or unverified scores; rescore the run before analysis")
        parse(r.output.model_dump_json(), t, r.turn)
        s = t.scenario
        values = {k: v.score for k, v in r.output.dimensions.items()}
        row = {
            "transcript_id": t.transcript_id,
            "scenario_id": s.scenario_id,
            "base_id": s.metadata.base_id,
            "pair_id": s.metadata.pair_id,
            "topic": s.metadata.topic,
            "domain": s.domain,
            "model": t.model,
            "intervention": t.intervention,
            "repetition": t.repetition,
            "turn": r.turn,
            "grader": r.grader,
            "grader_repeat": r.repeat,
            "test_data": r.test_data,
            "temperature": t.model_config_record["temperature"],
            **s.manipulation.model_dump(),
            **values,
            "conclusion": r.output.conclusion,
            "refusal": r.output.refusal,
            **features(t.generations[r.turn - 1].text),
        }
        # Distinct names prevent manipulation counterevidence overwriting recovery score.
        row["counterevidence_condition"] = s.manipulation.counterevidence
        row["counterevidence"] = values["counterevidence"]
        utility_values = [v for k, v in values.items() if k in UTILITY and v is not None]
        row["helpfulness"] = (
            sum(utility_values) / len(UTILITY) if len(utility_values) == len(UTILITY) else None
        )
        row["autonomy_provisional"] = composite(values, manifest["config"]["composite_weights"])
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no valid judgments available; inspect scoring failures")
    for dimension in DIMENSIONS:
        frame[dimension] = pd.to_numeric(frame[dimension], errors="coerce")
    return frame, failures, transcripts
