"""TEST DATA: adversarial bookkeeping and inference tests."""

import asyncio
import json

import pandas as pd
import pytest
from test_invariants import make_run
from test_statistics import synthetic_frame

from human_agency_evals.analysis.annotations import export_annotations, import_annotations
from human_agency_evals.analysis.coverage import coverage
from human_agency_evals.analysis.tradeoffs import model_tradeoffs
from human_agency_evals.conversations.runner import run
from human_agency_evals.io import atomic_json, read_json
from human_agency_evals.preflight import preflight
from human_agency_evals.scorers.llm_judge import score_run


def test_model_tradeoffs_cannot_hide_opposing_effects():
    frame = synthetic_frame()
    frame["helpfulness"] = 2.0
    frame["autonomy_provisional"] = frame.value
    frame["test_data"] = False
    frame["model"] = "model-a"
    other = frame.copy()
    other["model"] = "model-b"
    other.loc[other.intervention == "treatment", "helpfulness"] = 1.0
    result = model_tradeoffs(pd.concat([frame, other]), 0.25).set_index("model")
    assert result.loc["model-a", "helpfulness_noninferior"]
    assert not result.loc["model-b", "helpfulness_noninferior"]
    frame["test_data"] = True
    assert model_tradeoffs(frame, 0.25).helpfulness_noninferior.isna().all()
    frame["test_data"] = False
    assert (
        model_tradeoffs(frame, 0.25, incomplete_models={"model-a"})
        .helpfulness_noninferior.isna()
        .all()
    )


def test_missing_files_are_counted(tmp_path):
    folder = asyncio.run(run(str(make_run(tmp_path))))
    scores = asyncio.run(score_run(folder))
    assert coverage(folder).status.eq("ok").all()
    score_path = next(p for p in scores.glob("*.json") if p.name != "manifest.json")
    score_path.unlink()
    assert coverage(folder).status.eq("score_missing").sum() == 1
    next((folder / "transcripts").glob("*.json")).unlink()
    assert coverage(folder).status.eq("generation_missing").sum() == 1


def test_annotation_identity_integrity_and_batch_names(tmp_path):
    folder = asyncio.run(run(str(make_run(tmp_path))))
    output = export_annotations(folder, tmp_path / "labels.csv")
    with pytest.raises(ValueError, match="batch name"):
        export_annotations(folder, tmp_path / "elsewhere/labels.csv")
    frame = pd.read_csv(output, keep_default_na=False)
    frame.loc[0, "transcript"] = "[]"
    frame.to_csv(output, index=False)
    with pytest.raises(ValueError, match="content"):
        import_annotations(folder, output, folder / "annotation_keys/labels.json", "tester")


def test_rescore_invalidates_changed_prefix(tmp_path):
    folder = asyncio.run(run(str(make_run(tmp_path))))
    scores = asyncio.run(score_run(folder))
    before = {p.name: read_json(p) for p in scores.glob("*.json") if p.name != "manifest.json"}
    path = next((folder / "transcripts").glob("*.json"))
    transcript = read_json(path)
    transcript["messages"][2]["content"] = "TEST DATA: changed response"
    transcript["generations"][0]["text"] = "TEST DATA: changed response"
    atomic_json(path, transcript)
    from human_agency_evals.analysis.clean import frames

    with pytest.raises(ValueError, match="stale"):
        frames(folder)
    asyncio.run(score_run(folder))
    after = {p.name: read_json(p) for p in scores.glob("*.json") if p.name != "manifest.json"}
    assert (
        sum(
            before[name]["input_fingerprint"] != after[name]["input_fingerprint"] for name in before
        )
        == 1
    )


def test_preflight_does_not_expose_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-sentinel-must-not-appear")
    result = preflight("configs/experiment_calibration.yaml")
    assert result["estimate"]["conversations"] == 12
    assert result["estimate"]["target_calls"] == 60
    assert result["estimate"]["judge_calls"] == 240
    assert result["domains"] == ["actions", "epistemic", "values"]
    assert "heldout" not in result["splits"]
    assert "secret-sentinel" not in json.dumps(result)
    assert not any(c["remote_access_verified"] for c in result["checks"])


def test_strict_json_undefined_estimates(tmp_path):
    output = tmp_path / "data.json"
    atomic_json(output, {"estimate": float("nan"), "nested": [float("inf")]})
    assert output.read_text().find("NaN") == -1
    assert read_json(output) == {"estimate": None, "nested": [None]}


def test_unidentified_regression_rejected():
    from human_agency_evals.analysis.regressions import clustered_ols

    frame = pd.DataFrame(
        {
            "base_id": [0, 0, 1, 1, 2, 2],
            "x": [0, 1, 0, 1, 0, 1],
            "duplicate_x": [0, 1, 0, 1, 0, 1],
            "y": [0, 1, 1, 2, 2, 3],
        }
    )
    with pytest.raises(ValueError, match="rank-deficient"):
        clustered_ols(frame, "y ~ x + duplicate_x")


def test_confidence_evidence_confound_rejected():
    from human_agency_evals.datasets.validator import validate
    from human_agency_evals.scenarios.templates import vertical_slice

    low = vertical_slice()
    high = [s.model_copy(deep=True) for s in low]
    for s in high:
        s.scenario_id += "-high"
        s.metadata.condition_id += "-high"
        s.metadata.pair_id += "-high"
        s.manipulation.user_confidence = "high"
        s.evidence[0].text = "TEST DATA changed facts"
    with pytest.raises(ValueError, match="contrast changes"):
        validate(low + high)


def test_report_rejects_stale_inputs(tmp_path):
    from human_agency_evals.analysis.pipeline import analyze
    from human_agency_evals.reporting.report import report

    folder = asyncio.run(run(str(make_run(tmp_path))))
    scores = asyncio.run(score_run(folder))
    analyze(folder)
    report(folder)
    path = next(p for p in scores.glob("*.json") if p.name != "manifest.json")
    record = read_json(path)
    record["output"]["conclusion"] = "A"
    atomic_json(path, record)
    with pytest.raises(ValueError, match="stale"):
        report(folder)


def test_recovery_strength_and_prior_reinforcement_separated():
    from human_agency_evals.analysis.recovery import recovery_table

    rows = []
    for i, strength in enumerate(("weak", "strong")):
        for turn in (1, 3, 4, 5):
            rows.append(
                {
                    "transcript_id": str(i),
                    "base_id": str(i),
                    "grader": "judge",
                    "grader_repeat": 0,
                    "turn": turn,
                    "sycophancy": 2 if strength == "strong" and turn <= 3 else 0,
                    "counterevidence_condition": strength,
                    "counterevidence": 3 if turn >= 4 else None,
                    "model": "TEST DATA",
                    "intervention": "control",
                    "domain": "epistemic",
                }
            )
    result = recovery_table(pd.DataFrame(rows)).set_index(["strength", "selection"])
    assert result.loc[("weak", "prior_sycophancy_ge_2"), "judgments"] == 0
    assert result.loc[("strong", "prior_sycophancy_ge_2"), "judgments"] == 2
    assert result.loc[("strong", "prior_sycophancy_ge_2"), "mean"] == 3
