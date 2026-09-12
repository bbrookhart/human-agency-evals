"""Blind exports omit system prompts, labels and target identities. Text may still reveal style."""

import itertools
import json
import random
import uuid
from collections import defaultdict
from pathlib import Path

import pandas as pd

from human_agency_evals.io import atomic_json, digest, read_json, write_jsonl
from human_agency_evals.schemas.score import DIMENSIONS, Judgment
from human_agency_evals.schemas.transcript import Transcript
from human_agency_evals.scorers.llm_judge import applicable
from human_agency_evals.scorers.rubric import RUBRIC, VERSION


def export_annotations(folder: Path, destination: Path, limit: int = 100, seed: int = 1729):
    if destination.exists():
        raise ValueError("annotation output already exists; choose a new path")
    key_path = folder / "annotation_keys" / f"{destination.stem}.json"
    if key_path.exists():
        raise ValueError("annotation batch name already used; choose a distinct output stem")
    candidates = []
    for path in sorted((folder / "transcripts").glob("*.json")):
        t = Transcript.model_validate(read_json(path))
        for g in t.generations:
            candidates.append((t, g.turn))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    strata = defaultdict(list)
    for t, turn in candidates:
        strata[(t.scenario.domain, t.model, t.intervention, turn)].append((t, turn))
    candidates = [
        item
        for row in itertools.zip_longest(*[strata[k] for k in sorted(strata)])
        for item in row
        if item is not None
    ]
    candidates = candidates[:limit]
    rng.shuffle(candidates)
    rows, mapping = [], {}
    for t, turn in candidates[:limit]:
        anonymous = uuid.uuid4().hex
        mapping[anonymous] = {
            "transcript_id": t.transcript_id,
            "turn": turn,
            "applicable": sorted(applicable(t, turn)),
            "content_fingerprint": digest([m.model_dump() for m in t.messages[1 : 2 * turn + 1]]),
        }
        rows.append(
            {
                "example_id": anonymous,
                "transcript": json.dumps([m.model_dump() for m in t.messages[1 : 2 * turn + 1]]),
                "response_to_score": turn,
                "rubric_version": VERSION,
                "rubric": json.dumps(RUBRIC),
                "applicable_dimensions": json.dumps(sorted(applicable(t, turn))),
                **{d: "" for d in DIMENSIONS},
                "notes": "",
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    atomic_json(key_path, mapping)
    return destination


def import_annotations(folder: Path, source: Path, key: Path, annotator: str):
    if not annotator.strip():
        raise ValueError("annotator identity is required")
    mapping = read_json(key)
    frame = pd.read_csv(source, keep_default_na=False)
    if frame.example_id.duplicated().any():
        raise ValueError("duplicate annotation IDs")
    rows = []
    for row in frame.to_dict("records"):
        if row["example_id"] not in mapping:
            raise ValueError("unknown annotation ID")
        entry = mapping[row["example_id"]]
        transcript = Transcript.model_validate(
            read_json(folder / "transcripts" / f"{entry['transcript_id']}.json")
        )
        original = [m.model_dump() for m in transcript.messages[1 : 2 * entry["turn"] + 1]]
        if entry.get("content_fingerprint") != digest(original) or digest(
            json.loads(row["transcript"])
        ) != digest(original):
            raise ValueError("annotation content or underlying transcript changed")
        if int(row["response_to_score"]) != entry["turn"] or row["rubric_version"] != VERSION:
            raise ValueError("annotation response index or rubric version changed")
        if (
            json.loads(row["rubric"]) != RUBRIC
            or sorted(json.loads(row["applicable_dimensions"])) != entry["applicable"]
        ):
            raise ValueError("annotation rubric or applicability changed")
        scores: dict[str, int | None] = {}
        for d in DIMENSIONS:
            raw = row[d]
            if raw == "":
                scores[d] = None
            else:
                number = float(raw)
                if number != int(number):
                    raise ValueError("ratings must be integers")
                scores[d] = Judgment(score=int(number), rationale="human annotation").score
            if d not in entry["applicable"] and scores[d] is not None:
                raise ValueError(f"inapplicable rating: {d}")
        rows.append(
            {
                "transcript_id": entry["transcript_id"],
                "turn": entry["turn"],
                "annotator": annotator,
                "scores": scores,
                "notes": row.get("notes", ""),
            }
        )
    destination = folder / "human_annotations" / f"{digest(annotator)[:16]}.jsonl"
    if destination.exists():
        previous = [json.loads(line) for line in destination.read_text().splitlines()]
        previous_keys = {(r["transcript_id"], r["turn"]) for r in previous}
        if any((r["transcript_id"], r["turn"]) in previous_keys for r in rows):
            raise ValueError("annotator already rated one of these response prefixes")
        rows = previous + rows
    write_jsonl(destination, rows)
    return destination
