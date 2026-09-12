"""Exploratory fits reject unidentified designs rather than reporting spurious precision."""

import numpy as np
import statsmodels.formula.api as smf


def validate_design(model, frame, cluster):
    if frame[cluster].isna().any() or frame[cluster].nunique() < 3:
        raise ValueError(
            "at least three nonmissing clusters required; small-cluster inference remains fragile"
        )
    matrix = model.exog
    if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        raise ValueError("rank-deficient design; revise the prespecified estimable model")
    if matrix.shape[0] <= matrix.shape[1]:
        raise ValueError("no residual degrees of freedom")


def clustered_ols(frame, formula, cluster="base_id"):
    model = smf.ols(formula, data=frame, missing="raise")
    validate_design(model, frame, cluster)
    return model.fit(cov_type="cluster", cov_kwds={"groups": frame[cluster]})


def logistic_odds_ratios(frame, formula, cluster="base_id"):
    model = smf.logit(formula, data=frame, missing="raise")
    validate_design(model, frame, cluster)
    if not set(np.unique(model.endog)) == {0, 1}:
        raise ValueError("binary regression requires both zero and one outcomes")
    fit = model.fit(disp=False, cov_type="cluster", cov_kwds={"groups": frame[cluster]})
    if not fit.mle_retvals["converged"] or not np.isfinite(fit.params).all():
        raise ValueError("logistic fit did not converge to finite estimates")
    intervals = np.exp(fit.conf_int())
    return {
        "odds_ratio": np.exp(fit.params).to_dict(),
        "low": intervals[0].to_dict(),
        "high": intervals[1].to_dict(),
    }
