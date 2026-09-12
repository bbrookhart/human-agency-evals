"""TEST DATA: numerical examples test estimands, not model behavior."""

import numpy as np
import pandas as pd
import pytest

from human_agency_evals.analysis.metrics import bccs, paired_effect
from human_agency_evals.analysis.mixed_effects import mixed_model
from human_agency_evals.analysis.regressions import clustered_ols, logistic_odds_ratios


def synthetic_frame():
    rows = []
    for base in range(6):
        for arm in ("control", "treatment"):
            for position in ("A", "B"):
                rows.append(
                    {
                        "base_id": str(base),
                        "pair_id": str(base),
                        "domain": "epistemic",
                        "model": "TEST DATA",
                        "repetition": 0,
                        "turn": 1,
                        "grader": "TEST DATA",
                        "grader_repeat": 0,
                        "position": position,
                        "user_confidence": "low",
                        "validation_seeking": "absent",
                        "counterevidence_condition": "absent",
                        "intervention": arm,
                        "value": base / 5 + (1 if arm == "treatment" else 0),
                        "conclusion": position,
                    }
                )
    return pd.DataFrame(rows)


def test_matched_effect_and_missingness():
    frame = synthetic_frame()
    effect = paired_effect(frame, "intervention", "control", "treatment", "value")
    assert effect["mean"] == pytest.approx(1)
    assert effect["n_clusters"] == 6
    assert effect["matched_rows"] == 12
    effect = paired_effect(frame.iloc[1:], "intervention", "control", "treatment", "value")
    assert effect["unmatched_or_missing_rows"] == 1
    assert effect["mean"] == pytest.approx(1)
    paired = bccs(frame)
    assert paired.bccs.eq(1).all()
    assert paired.follows_both_positions.all()
    with pytest.raises(pd.errors.MergeError):
        paired_effect(pd.concat([frame, frame]), "intervention", "control", "treatment", "value")


def test_regression_helpers():
    rng = np.random.default_rng(123)
    group = np.repeat(np.arange(30), 12)
    x = np.tile(np.arange(12) % 2, 30)
    intercepts = rng.normal(0, 1.5, 30)
    y = 1 + 0.8 * x + intercepts[group] + rng.normal(0, 0.3, len(x))
    frame = pd.DataFrame({"base_id": group, "x": x, "y": y})
    assert clustered_ols(frame, "y ~ x").params["x"] == pytest.approx(0.8, abs=0.15)
    assert mixed_model(frame, "y ~ x").params["x"] == pytest.approx(0.8, abs=0.15)
    frame["binary"] = rng.binomial(1, 1 / (1 + np.exp(-(0.7 * x - 0.5))))
    ratios = logistic_odds_ratios(frame, "binary ~ x")
    assert ratios["low"]["x"] < ratios["odds_ratio"]["x"] < ratios["high"]["x"]
