import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from human_agency_evals.analysis.agreement import agreement
from human_agency_evals.analysis.bootstrap import cluster_interval
from human_agency_evals.analysis.clean import frames
from human_agency_evals.analysis.coverage import coverage
from human_agency_evals.analysis.metrics import bccs, fdr, paired_effect, pareto
from human_agency_evals.analysis.plots import make_plots
from human_agency_evals.analysis.provenance import input_fingerprint
from human_agency_evals.analysis.readiness import pilot_readiness
from human_agency_evals.analysis.recovery import recovery_table
from human_agency_evals.analysis.regressions import clustered_ols
from human_agency_evals.analysis.robustness import (
    intervention_sensitivity,
    leave_one_scenario_out,
    stratified,
    weight_sensitivity,
)
from human_agency_evals.analysis.tradeoffs import model_tradeoffs
from human_agency_evals.io import atomic_json, read_json
from human_agency_evals.schemas.score import DIMENSIONS


def summarize(frame, metrics, groups, samples, seed):
    rows = []
    for key, group in frame.groupby(groups, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for metric in metrics:
            rows.append(
                {
                    **dict(zip(groups, key, strict=True)),
                    "metric": metric,
                    **cluster_interval(group, metric, samples=samples, seed=seed),
                }
            )
    return pd.DataFrame(rows, columns=groups + ["metric", "mean", "low", "high", "n_clusters"])


def agreement_table(frame, folder):
    ratings = {}
    for (grader, repeat), group in frame.groupby(["grader", "grader_repeat"]):
        ratings[f"{grader}/repeat-{repeat}"] = group.set_index(["transcript_id", "turn"])[
            list(DIMENSIONS)
        ]
    for path in (folder / "human_annotations").glob("*.jsonl"):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if records:
            ratings["human/" + records[0]["annotator"]] = pd.DataFrame(
                [
                    {**r["scores"], "transcript_id": r["transcript_id"], "turn": r["turn"]}
                    for r in records
                ]
            ).set_index(["transcript_id", "turn"])
    rows = []
    for a, b in itertools.combinations(sorted(ratings), 2):
        joined = ratings[a].join(ratings[b], how="inner", lsuffix="_a", rsuffix="_b")
        for d in DIMENSIONS:
            result = agreement(joined[d + "_a"], joined[d + "_b"])
            rows.append(
                {
                    "rater_a": a,
                    "rater_b": b,
                    "metric": d,
                    **{k: v for k, v in result.items() if k != "confusion"},
                    "confusion": json.dumps(result["confusion"]),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "rater_a",
            "rater_b",
            "metric",
            "n",
            "percent_agreement",
            "kappa",
            "weighted_kappa",
            "confusion",
        ],
    )


def analyze(folder: Path) -> Path:
    manifest = read_json(folder / "manifest.json")
    config = manifest["config"]
    samples, seed = config["bootstrap_samples"], config["seed"]
    output = folder / "analysis"
    output.mkdir(exist_ok=True)
    (output / "provenance.json").unlink(missing_ok=True)
    coverage_frame = coverage(folder)
    coverage_frame.to_csv(output / "coverage.csv", index=False)
    coverage_frame.groupby(["model", "intervention", "domain", "status"]).size().rename(
        "count"
    ).reset_index().to_csv(output / "missingness.csv", index=False)
    frame, failures, transcripts = frames(folder)
    frame.to_csv(output / "scores_long.csv", index=False)
    atomic_json(output / "score_failures.json", failures)
    metrics = list(DIMENSIONS) + ["helpfulness", "autonomy_provisional", "refusal"]
    summary = summarize(frame, metrics, ["model", "intervention"], samples, seed)
    summary.to_csv(output / "metrics.csv", index=False)
    summarize(frame, metrics, ["model", "intervention", "domain"], samples, seed).to_csv(
        output / "per_domain.csv", index=False
    )
    drift = summarize(frame, metrics, ["model", "intervention", "domain", "turn"], samples, seed)
    drift.to_csv(output / "drift.csv", index=False)
    recovery_table(frame, samples, seed).to_csv(output / "recovery.csv", index=False)
    paired = bccs(frame)
    paired.to_csv(output / "paired_bccs.csv", index=False)
    paired_summary = summarize(paired, ["bccs"], ["model", "intervention"], samples, seed)
    paired_summary.to_csv(output / "bccs_summary.csv", index=False)
    effects = []
    # H1/H2 use pre-counterevidence responses to avoid pooling recovery into initial susceptibility.
    pre = frame[frame.turn <= 3]
    for factor, control, treatment in [
        ("user_confidence", "low", "high"),
        ("validation_seeking", "absent", "present"),
    ]:
        effects.append(paired_effect(pre, factor, control, treatment, "sycophancy", samples, seed))
    for arm in sorted(set(frame.intervention) - {"control"}):
        for metric in (
            "action_pressure",
            "evidence_grounding",
            "helpfulness",
            "autonomy_provisional",
        ):
            effects.append(
                paired_effect(frame, "intervention", "control", arm, metric, samples, seed)
            )
    effects_frame = pd.DataFrame(effects)
    primary = effects_frame.metric.isin(["sycophancy", "action_pressure", "evidence_grounding"])
    effects_frame["family"] = np.where(primary, "primary", "exploratory")
    effects_frame["q_bh"] = np.nan
    effects_frame.loc[primary, "q_bh"] = fdr(effects_frame.loc[primary, "p_value"])
    effects_frame.to_csv(output / "hypothesis_effects.csv", index=False)
    frontier = (
        frame.groupby(["model", "intervention", "base_id"])[["helpfulness", "autonomy_provisional"]]
        .mean()
        .groupby(level=[0, 1])
        .mean()
        .reset_index()
    )
    frontier["pareto"] = pareto(frontier)
    frontier.to_csv(output / "frontier.csv", index=False)
    incomplete_models = set(coverage_frame.loc[coverage_frame.status != "ok", "model"])
    model_tradeoffs(frame, config["helpfulness_margin"], samples, seed, incomplete_models).to_csv(
        output / "tradeoffs.csv", index=False
    )
    agreement_frame = agreement_table(frame, folder)
    agreement_frame.to_csv(output / "agreement.csv", index=False)
    pilot_readiness(frame, coverage_frame, agreement_frame).to_csv(
        output / "pilot_readiness.csv", index=False
    )
    leave_one_scenario_out(frame, "autonomy_provisional").to_csv(
        output / "leave_one_out.csv", index=False
    )
    intervention_sensitivity(frame, samples, seed).to_csv(
        output / "intervention_sensitivity.csv", index=False
    )
    schemes = {
        "configured": config["composite_weights"],
        "epistemic": {"evidence_grounding": 2, "uncertainty": 1, "alternatives": 1},
        "pressure_emphasis": {
            "evidence_grounding": 1,
            "action_pressure": 3,
            "value_elicitation": 1,
        },
    }
    weight_sensitivity(frame, schemes).to_csv(output / "weight_sensitivity.csv", index=False)
    for factor in ("grader", "wording_variant", "temperature", "domain", "model"):
        # model is already a grouping key; avoid duplicate grouper.
        if factor == "model":
            continue
        stratified(frame, factor, "autonomy_provisional").to_csv(
            output / f"robustness_{factor}.csv", index=False
        )
    diagnostics = {
        "test_data": bool(frame.test_data.any()),
        "base_scenarios": frame.base_id.nunique(),
        "transcripts": len(transcripts),
        "failed_transcripts": sum(t.status != "complete" for t in transcripts.values()),
        "failed_scores": len(failures),
        "planned_judgments": len(coverage_frame),
        "unavailable_judgments": int((coverage_frame.status != "ok").sum()),
        "score_variance": {
            d: None if frame[d].dropna().empty else float(frame[d].var()) for d in DIMENSIONS
        },
        "floor_fraction": {
            d: float((frame[d].dropna() == 0).mean()) if frame[d].notna().any() else None
            for d in DIMENSIONS
        },
        "ceiling_fraction": {
            d: float((frame[d].dropna() == 3).mean()) if frame[d].notna().any() else None
            for d in DIMENSIONS
        },
        "human_annotations_present": any((folder / "human_annotations").glob("*.jsonl")),
        "prompt_robustness_available": frame.wording_variant.nunique() > 1,
        "temperature_robustness_available": frame.temperature.nunique() > 1,
        "note": "Constant mock scores deliberately fail variance gates. Agreement kappa is undefined for constant ratings. Real pilot success is not established by software tests.",
    }
    atomic_json(output / "diagnostics.json", diagnostics)
    regression_status = {}
    # Average judge repeats before exploratory clustered regressions.
    columns = [
        "base_id",
        "scenario_id",
        "model",
        "intervention",
        "turn",
        "repetition",
        "domain",
        "user_confidence",
        "validation_seeking",
        "counterevidence_condition",
    ]
    observations = frame.groupby(columns)[list(DIMENSIONS)].mean().reset_index()
    formula = "sycophancy ~ turn * C(intervention) + C(model) + C(domain) + C(user_confidence) + C(validation_seeking) + C(counterevidence_condition)"
    try:
        (output / "regression.csv").unlink(missing_ok=True)
        if observations.sycophancy.nunique() < 2:
            raise ValueError("constant outcome; regression not estimable")
        fit = clustered_ols(observations, formula)
        pd.DataFrame(
            {
                "coefficient": fit.params,
                "low": fit.conf_int()[0],
                "high": fit.conf_int()[1],
                "p_value_exploratory": fit.pvalues,
            }
        ).to_csv(output / "regression.csv")
        regression_status = {
            "status": "exploratory",
            "formula": formula,
            "note": "Gaussian ordinal approximation; sparse clusters and design rank require review",
        }
    except (ValueError, np.linalg.LinAlgError) as exc:
        regression_status = {"status": "not_estimable", "reason": str(exc), "formula": formula}
    atomic_json(output / "regression_status.json", regression_status)
    make_plots(
        summary,
        frontier,
        drift,
        paired_summary,
        agreement_frame,
        output / "figures",
        bool(frame.test_data.any()),
    )
    atomic_json(
        output / "provenance.json",
        {
            "input_fingerprint": input_fingerprint(folder),
            "analysis_version": "0.2",
            "score_path": read_json(folder / "active_scores.json")["path"],
        },
    )
    return output
