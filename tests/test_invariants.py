import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from human_agency_evals.analysis.agreement import agreement, ordinal_alpha
from human_agency_evals.analysis.bootstrap import cluster_interval
from human_agency_evals.analysis.metrics import fdr, pareto
from human_agency_evals.conversations.runner import run
from human_agency_evals.conversations.state import render_turn
from human_agency_evals.datasets.loader import load
from human_agency_evals.datasets.pairing import pairs
from human_agency_evals.datasets.validator import validate
from human_agency_evals.io import read_json, write_jsonl
from human_agency_evals.scenarios.templates import vertical_slice
from human_agency_evals.schemas.scenario import Scenario
from human_agency_evals.schemas.score import Judgment
from human_agency_evals.scorers.composite import composite
from human_agency_evals.scorers.deterministic import conclusion_shift, features
from human_agency_evals.scorers.llm_judge import mock_judgment, parse, payload, score_run


def test_pair_evidence_and_context_invariance():
    data = vertical_slice()
    data[1].evidence[0].text = "different fact"
    with pytest.raises(ValueError, match="changes evidence"):
        pairs(data)


def test_schema_and_dataset_errors():
    data = vertical_slice()
    with pytest.raises(ValueError, match="duplicate"):
        validate(data + data)
    with pytest.raises(ValueError, match="broken"):
        validate(data[:1])
    data[1].metadata.split = "heldout"
    with pytest.raises(ValueError, match="leakage"):
        validate(data)
    value = vertical_slice()[0].model_dump()
    value["conversation"][0]["index"] = 2
    with pytest.raises(ValidationError):
        Scenario.model_validate(value)
    for bad in (True, 1.5, -1, 4, "2"):
        with pytest.raises(ValidationError):
            Judgment(score=bad, rationale="test")


def test_full_dataset_and_evidence_timing():
    data = load([str(p) for p in Path("data").glob("*/scenarios.jsonl")])
    assert validate(data) == {"conditions": 720, "base_scenarios": 30, "pairs": 360}
    scenario = next(s for s in data if s.manipulation.counterevidence == "strong")
    update = scenario.evidence[-1].text
    assert update not in render_turn(scenario, 1)
    assert update in render_turn(scenario, 4)
    assert scenario.expected_behavior.desirable not in render_turn(scenario, 1)


def test_composite_and_surface_diagnostics():
    assert (
        composite(
            {"action_pressure": 3, "evidence_grounding": 3},
            {"action_pressure": 1, "evidence_grounding": 1},
        )
        == 1.5
    )
    assert composite({"value_elicitation": None}, {"value_elicitation": 1}) is None
    with pytest.raises(ValueError):
        composite({}, {"evidence_grounding": -1})
    assert features("Certainly?\n- option\n- another")["enumerated_items"] == 2
    assert conclusion_shift("A", "B") == 1
    assert conclusion_shift("unclear", "B") is None


def test_cluster_dependence_and_reproducibility():
    frame = pd.DataFrame({"base_id": ["a", "b", "c"], "value": [0.0, 1.0, 3.0]})
    original = cluster_interval(frame, "value", samples=500)
    repeated = cluster_interval(pd.concat([frame] * 10), "value", samples=500)
    assert original == repeated
    assert original["low"] <= original["mean"] <= original["high"]
    assert cluster_interval(frame, "value", seed=4) == cluster_interval(frame, "value", seed=4)


def test_agreement_and_fdr():
    assert agreement([0, 1, 2, 3], [0, 1, 2, 3])["kappa"] == 1
    assert agreement([2, 2], [2, 2])["kappa"] is None
    assert agreement([0, None], [0, 2])["n"] == 1
    assert agreement([0, 1], [3, 2])["percent_agreement"] == 0
    assert ordinal_alpha(pd.DataFrame([[0, 0], [3, 3]])) == 1
    assert np.allclose(fdr([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])
    assert pareto(
        pd.DataFrame({"helpfulness": [1, 2, 3], "autonomy_provisional": [1, 3, 2]})
    ).tolist() == [False, True, True]


def make_run(tmp_path, monkeypatch=None):
    data = tmp_path / "data.jsonl"
    write_jsonl(data, [s.model_dump() for s in vertical_slice()])
    config = {
        "name": "test",
        "data": [str(data)],
        "models": [{"name": "mock/target"}],
        "interventions": ["control"],
        "graders": {"models": [{"name": "mock/judge"}], "repeats": 1},
        "repetitions": 1,
        "retries": 1,
        "output_dir": str(tmp_path / "runs"),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_failure_recovery_and_rescore(tmp_path, monkeypatch):
    from human_agency_evals.conversations.models import MockModel

    config = make_run(tmp_path)
    original = MockModel.generate

    async def fail(self, messages, seed):
        raise RuntimeError("TEST DATA simulated failure")

    monkeypatch.setattr(MockModel, "generate", fail)
    folder = asyncio.run(run(str(config), "resume"))
    assert read_json(folder / "cost.json")["failed"] == 2
    monkeypatch.setattr(MockModel, "generate", original)
    asyncio.run(run(str(config), "resume"))
    assert read_json(folder / "cost.json")["completed"] == 2
    monkeypatch.setattr(MockModel, "generate", fail)
    asyncio.run(score_run(folder))  # No target call permitted.
    transcript = next((folder / "transcripts").glob("*.json"))
    from human_agency_evals.schemas.transcript import Transcript

    t = Transcript.model_validate(read_json(transcript))
    assert t.errors and t.status == "complete"
    blind = json.dumps(payload(t, 1))
    assert (
        t.model not in blind and "system_prompt" not in blind and "expected_behavior" not in blind
    )
    output = mock_judgment(t, 1)
    assert parse(output.model_dump_json(), t, 1) == output
    output.dimensions["evidence_grounding"].evidence_quote = "fabricated quotation"
    with pytest.raises(ValueError, match="quote"):
        parse(output.model_dump_json(), t, 1)
    values = yaml.safe_load(config.read_text())
    values["seed"] = 1
    config.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="changed"):
        asyncio.run(run(str(config), "resume"))


def test_mock_report_and_annotations(tmp_path):
    from human_agency_evals.analysis.annotations import export_annotations, import_annotations
    from human_agency_evals.analysis.pipeline import analyze
    from human_agency_evals.reporting.report import report

    folder = asyncio.run(run(str(make_run(tmp_path))))
    asyncio.run(score_run(folder))
    annotation = export_annotations(folder, tmp_path / "blind.csv")
    frame = pd.read_csv(annotation)
    assert not {"model", "intervention", "condition_id"} & set(frame.columns)
    assert "mock/target" not in annotation.read_text()
    for name in json.loads(frame.iloc[0].applicable_dimensions):
        frame[name] = 2
    frame.to_csv(annotation, index=False)
    imported = import_annotations(
        folder, annotation, folder / "annotation_keys/blind.json", "human-test"
    )
    assert imported.exists()
    analyze(folder)
    assert "TEST DATA" in report(folder).read_text()
    assert len(list((folder / "analysis/figures").glob("*.png"))) == 6


def test_inspect_task_import():
    from human_agency_evals.inspect.tasks import autonomy_task

    task = autonomy_task()
    assert len(task.dataset) > 0


def test_inspect_native_offline(tmp_path, monkeypatch):
    from inspect_ai import eval

    from human_agency_evals.inspect.tasks import autonomy_task

    monkeypatch.setattr(
        "inspect_ai._util.appdirs.user_data_path", lambda package: tmp_path / "inspect-data"
    )
    monkeypatch.setattr(
        "inspect_ai._util.appdirs.user_cache_path", lambda package: tmp_path / "inspect-cache"
    )
    from inspect_ai.model import ModelOutput, ModelUsage

    outputs = [
        ModelOutput.from_content(model="mockllm", content="TEST DATA: native Inspect fixture")
        for _ in range(5)
    ]
    for output in outputs:
        output.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    logs = eval(
        autonomy_task(),
        model="mockllm/model",
        model_args={"custom_outputs": outputs},
        limit=1,
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert logs[0].status == "success", logs[0].error
    assert len(logs[0].samples[0].messages) == 11


def test_mid_conversation_resume_preserves_turns(tmp_path, monkeypatch):
    from human_agency_evals.conversations.models import MockModel
    from human_agency_evals.schemas.scenario import TurnTemplate

    config = make_run(tmp_path)
    scenarios = vertical_slice()
    for scenario in scenarios:
        scenario.conversation.extend(
            [
                TurnTemplate(index=2, text="What information is missing?"),
                TurnTemplate(index=3, text="What would you suggest next?"),
            ]
        )
    write_jsonl(tmp_path / "data.jsonl", [s.model_dump() for s in scenarios])
    original = MockModel.generate

    async def fail_second(self, messages, seed):
        if len(messages) == 4:
            raise RuntimeError("TEST DATA: interrupted second turn")
        return await original(self, messages, seed)

    monkeypatch.setattr(MockModel, "generate", fail_second)
    folder = asyncio.run(run(str(config), "partial"))
    saved = {
        p.name: read_json(p)["generations"][0] for p in (folder / "transcripts").glob("*.json")
    }
    monkeypatch.setattr(MockModel, "generate", original)
    asyncio.run(run(str(config), "partial"))
    for path in (folder / "transcripts").glob("*.json"):
        result = read_json(path)
        assert result["generations"][0] == saved[path.name]
        assert len(result["messages"]) == 7 and result["status"] == "complete"


def test_provider_adapter_without_network(monkeypatch):
    from inspect_ai.model import ModelOutput, ModelUsage

    from human_agency_evals.inspect.adapter import InspectModel
    from human_agency_evals.schemas.experiment import ModelSpec
    from human_agency_evals.schemas.transcript import Message

    captured = {}

    class FakeProvider:
        async def generate(self, messages, config):
            captured["config"] = config
            output = ModelOutput.from_content(
                model="test/model", content="TEST DATA: provider response"
            )
            output.usage = ModelUsage(input_tokens=4, output_tokens=5, total_tokens=9)
            return output

    def get_model(name, **options):
        captured["name"], captured["options"] = name, options
        return FakeProvider()

    monkeypatch.setattr("human_agency_evals.inspect.adapter.get_model", get_model)
    monkeypatch.setenv("TEST_PROVIDER_KEY", "TEST DATA dummy credential")
    model = InspectModel(
        ModelSpec(name="test/model", api_key_env="TEST_PROVIDER_KEY", seed_supported=True)
    )
    result = asyncio.run(model.generate([Message(role="user", content="TEST DATA")], 17))
    assert result.input_tokens == 4 and result.output_tokens == 5
    assert captured["config"].seed == 17
