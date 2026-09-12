import statsmodels.formula.api as smf


def mixed_model(frame, formula, group="base_id", variance_components=None):
    """Exploratory Gaussian approximation for ordinal outcomes; inspect convergence."""
    fit = smf.mixedlm(
        formula, frame, groups=frame[group], vc_formula=variance_components, missing="raise"
    ).fit(reml=False, method="lbfgs")
    if not fit.converged:
        raise ValueError("mixed model failed to converge")
    return fit
