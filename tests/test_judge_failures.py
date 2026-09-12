import asyncio
import json

from test_invariants import make_run

from human_agency_evals.conversations.models import Response
from human_agency_evals.conversations.runner import run
from human_agency_evals.io import read_json
from human_agency_evals.schemas.experiment import GraderConfig
from human_agency_evals.schemas.transcript import Transcript
from human_agency_evals.scorers.llm_judge import mock_judgment, score_run


def test_invalid_judge_retries_logged_and_cache(tmp_path, monkeypatch):
    folder = asyncio.run(run(str(make_run(tmp_path))))
    templates = [
        Transcript.model_validate(read_json(p)) for p in (folder / "transcripts").glob("*.json")
    ]
    valid = mock_judgment(templates[0], 1).model_dump_json()
    calls = []

    class FakeJudge:
        async def generate(self, messages, seed):
            calls.append(messages)
            return Response("invalid JSON" if len(calls) % 2 else valid, 100, 100)

    monkeypatch.setattr("human_agency_evals.scorers.llm_judge.make_model", lambda spec: FakeJudge())
    graders = GraderConfig.model_validate(
        {"models": [{"name": "test/judge", "temperature": 0}], "repeats": 1, "retries": 2}
    )
    output = asyncio.run(score_run(folder, graders))
    assert len(calls) == 4
    attempts = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
    assert len(attempts) == 2 and attempts[0]["raw"] == "invalid JSON"
    asyncio.run(score_run(folder, graders))
    assert len(calls) == 4
    usage = read_json(output / "usage/cost.json")
    assert usage["successful"] == 2
    assert usage["provider_calls"] == 4
    assert usage["input_tokens"] == 400  # Includes both invalid JSON replies.
    assert usage["output_tokens"] == 400
