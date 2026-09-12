from pathlib import Path

import pandas as pd

from human_agency_evals.analysis.provenance import verify_analysis
from human_agency_evals.io import read_json


def table(frame: pd.DataFrame, limit=20):
    if frame.empty:
        return "No observations available."
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.head(limit).itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(
                "NA"
                if pd.isna(v)
                else str(round(v, 3))
                if isinstance(v, float)
                else str(v).replace("|", "\\|").replace("\n", " ")
                for v in row
            )
            + " |"
        )
    if len(frame) > limit:
        lines += [f"\nShowing {limit} of {len(frame)} rows; see the complete CSV."]
    return "\n".join(lines)


def report(folder: Path) -> Path:
    verify_analysis(folder)
    analysis = folder / "analysis"
    diagnostics = read_json(analysis / "diagnostics.json")
    manifest = read_json(folder / "manifest.json")
    test = diagnostics["test_data"]
    sections = [
        "# Human Agency Evals",
        "**TEST DATA — pipeline verification only. No model findings.**"
        if test
        else "Behavioral evaluation under the recorded conditions; no demonstrated human harm.",
        "## Executive summary",
        "Results pending execution of a validated real-model study."
        if test
        else "This report summarizes recorded behavioral proxies. Interpretation requires human validation and review of missingness and experimental controls.",
        "## Experimental setup",
        f"Run fingerprint: `{manifest['fingerprint']}`. Seed: {manifest['config']['seed']}. Equal-weight base-scenario estimates; 95% scenario-cluster bootstrap intervals. Dimensions are ordinal proxies; composite validity is unestablished.",
        "## Dataset summary",
        f"{diagnostics['base_scenarios']} base scenarios; {diagnostics['transcripts']} transcripts; {diagnostics['failed_transcripts']} incomplete/failed transcripts; {diagnostics['failed_scores']} failed judgments. See manifest for every condition and evidence schedule.",
        "## Pilot readiness and missingness",
        f"{diagnostics.get('unavailable_judgments', 0)} of {diagnostics.get('planned_judgments', 0)} planned judgments are unavailable or failed. Missing artifacts are included in this denominator.",
        table(pd.read_csv(analysis / "pilot_readiness.csv")),
        "See [coverage.csv](analysis/coverage.csv) and [missingness.csv](analysis/missingness.csv) for every planned cell and grouped failure counts.",
        "## Model configuration",
        table(pd.DataFrame(manifest["config"]["models"])),
        "## Primary metrics",
        table(pd.read_csv(analysis / "metrics.csv")),
        "## Primary hypothesis results, effect sizes and confidence intervals",
        "Hypotheses are preregistered-style expectations, not established findings. H1/H2 compare pre-update responses; H5 contrasts control and intervention. Matched failures are reported in the CSV. H3/H4 remain exploratory.",
        table(pd.read_csv(analysis / "hypothesis_effects.csv")),
        "## Safety–utility analysis",
        "Helpfulness is independent of autonomy scores. Provisional noninferiority uses the recorded margin and is assessed separately for each model. Mock and incomplete designs remain unassessed. Pareto labels are descriptive and do not establish superiority.",
        table(pd.read_csv(analysis / "tradeoffs.csv")),
        "![Safety–utility frontier](analysis/figures/02_frontier.png)",
        "## Per-domain results",
        table(pd.read_csv(analysis / "per_domain.csv")),
        "## Counterevidence recovery",
        "Weak and strong updates are separated. The prior-sycophancy subset conditions on a previous model response; its comparisons are descriptive and subject to post-treatment selection.",
        table(pd.read_csv(analysis / "recovery.csv")),
        "## Multi-turn analysis",
        "Turn indexes assistant responses. Accumulated pressure, evidence exposure and turn are confounded; slopes do not isolate time effects.",
        "![Interaction drift](analysis/figures/04_drift.png)",
        "## Human/LLM agreement",
        "Human labels present: "
        + str(diagnostics["human_annotations_present"])
        + ". Agreement is descriptive on matched ratings. Repeated graders and responses are dependent. Constant-label kappa is undefined, not perfect reliability.",
        "See [agreement.csv](analysis/agreement.csv) for rater identities, confusion matrices and exact/weighted agreement.",
        "## Robustness analyses",
        "See intervention_sensitivity.csv, leave_one_out.csv, weight_sensitivity.csv, robustness_grader.csv, robustness_domain.csv, robustness_temperature.csv and robustness_wording_variant.csv. Single-level strata cannot establish robustness. Controlled verbosity/style diagnostic experiments require separate execution.",
        "## Failure examples",
        "Generation attempts remain in events.jsonl and transcript errors; failed judgments remain in score_failures.json. No failure examples are invented. Inspect scores_long.csv with blinded transcript exports for qualitative review.",
        "## Limitations",
        "Fictional convenience sample; small topic clusters; public heldout; unvalidated ordinal rubrics and composite; possible judge-family/style bias; static user follow-ups; imperfect blinding through response style; no human-outcome evidence. Token estimates are approximate and failed-call charges may be unavailable.",
        "Composite validity is an empirical question and should not be assumed merely because dimensions can be numerically combined.",
    ]
    path = folder / "report.md"
    path.write_text("\n\n".join(sections) + "\n")
    return path
