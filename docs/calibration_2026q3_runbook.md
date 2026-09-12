# Calibration 2026 Q3 execution runbook

**Status:** preparation only. Do not interpret this document as authorization to spend or as evidence that model access succeeds.

This runbook implements issue #3 using only development scenarios. It supplements `docs/calibration_protocol.md`; the original protocol remains the methodological source of truth.

## Why a new configuration exists

The repository's original example configuration references `anthropic/claude-sonnet-4-20250514`. Anthropic's official model-deprecation documentation lists that model as retired on 2026-06-15 and identifies `claude-sonnet-4-6` as its replacement. The original `openai/gpt-4o` example is also no longer representative of the current OpenAI frontier family.

`configs/experiment_calibration_2026q3.yaml` therefore uses a deliberately small cross-family development matrix:

- target: `openai/gpt-5.6-terra`
- target: `anthropic/claude-sonnet-4-6`
- grader: `openai/gpt-5.6-sol`
- grader: `anthropic/claude-sonnet-4-6`

The OpenAI model IDs and short-context prices were checked against OpenAI's official API documentation on 2026-09-06. Anthropic price fields are left null rather than inferred from memory. Re-check all model IDs and prices immediately before execution because provider availability and pricing can change.

Official references used for this preparation:

- OpenAI API changelog: https://developers.openai.com/api/docs/changelog
- OpenAI API pricing: https://developers.openai.com/api/docs/pricing
- Anthropic model deprecations: https://docs.anthropic.com/en/docs/about-claude/model-deprecations

## Scientific boundary

This is a **measurement-development calibration**, not a confirmatory model comparison. With `base_limit: 3`, the unit count remains only three independent authored base scenarios. Adding a second target family helps identify obvious model-family dependence; it does not make three scenarios representative.

No heldout scenario may be inspected or modified because of calibration output.

## 1. Independent scenario review

Before model calls, make two independent copies of:

`data/human_annotations/scenario_review.csv`

Each scenario reviewer should assess factual coherence, realism, acceptable option space, update reliability, expected-behavior leakage, dimension applicability, and whether the evidence manipulation actually has the intended strength.

Preserve both initial reviews before adjudication.

Do not use the response-rater blinding workflow for scenario reviewers; these are different roles.

## 2. Provider and cost preflight

Install provider extras and export credentials locally:

```bash
uv sync --extra providers
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

Never commit credentials or `.env` files.

Run:

```bash
uv run autonomy-evals preflight --config configs/experiment_calibration_2026q3.yaml
```

Archive the output under a local run-preparation directory together with:

- Git commit SHA;
- resolved configuration;
- Python and package versions;
- whether each credential environment variable is present;
- token estimate;
- estimated cost where prices are known;
- an explicit operator-set maximum spend.

Credential presence is not proof that a model ID is accessible. If the first deliberate provider call returns a model-not-found/retired/access error, stop and record the failure rather than silently substituting another model.

## 3. Freeze the development batch

Before inference:

- resolve any critical scenario-review defects;
- commit the exact config used;
- record its SHA-256;
- record current provider documentation retrieval date;
- record the Git SHA;
- document the planned run ID;
- do not modify prompts/rubrics after inspecting model outputs without versioning the calibration as a new development run.

Recommended first run ID:

`calibration-2026q3-001`

## 4. Execute target inference and grading

```bash
uv run autonomy-evals run \
  --config configs/experiment_calibration_2026q3.yaml \
  --run-id calibration-2026q3-001
```

Because the configuration includes two targets, verify the generated manifest before continuing. Confirm that both exact target names and both exact grader names match the frozen configuration.

Preserve all provider failures, retries and raw judge outputs. Do not delete failed attempts merely because a later retry succeeds.

## 5. Export one blinded batch

```bash
uv run autonomy-evals annotate-export \
  --run calibration-2026q3-001 \
  --output annotations/calibration-2026q3-001.csv \
  --limit 120
```

The final limit should be chosen to cover the intended response prefixes without breaking the balancing assumptions. Do not assume 120 is correct if the generated run structure differs; inspect the export summary first.

Give two independent human response raters separate copies of the same exported batch. Prefer three raters if available.

Do **not** share:

- target identity;
- grader outputs;
- intervention arm;
- system prompt;
- run manifest;
- linkage key;
- scenario-review expected-behavior fields.

## 6. Import human ratings

Example:

```bash
uv run autonomy-evals annotate-import \
  --run calibration-2026q3-001 \
  --input annotations/rater-a.csv \
  --key runs/calibration-2026q3-001/annotation_keys/calibration-2026q3-001.json \
  --annotator rater-a

uv run autonomy-evals annotate-import \
  --run calibration-2026q3-001 \
  --input annotations/rater-b.csv \
  --key runs/calibration-2026q3-001/annotation_keys/calibration-2026q3-001.json \
  --annotator rater-b
```

Retain stable rater IDs. Do not overwrite or fabricate ratings to satisfy readiness checks.

## 7. Regenerate analysis after final annotation import

```bash
uv run autonomy-evals analyze --run calibration-2026q3-001
uv run autonomy-evals report --run calibration-2026q3-001
```

Review at minimum:

- `coverage.csv`
- `missingness.csv`
- `pilot_readiness.csv`
- `agreement.csv`
- raw score dimensions
- refusal / excessive-hedging diagnostics
- supported-vs-unsupported agreement diagnostics
- per-domain results
- intervention sensitivity
- grader/repeat stability
- target-family differences
- cost/error ledgers

## 8. Decision gate

The calibration can justify a next pilot only if the review team can defend the measurement process, not merely because an aggregate score moved in the preferred direction.

Before freezing v0.2, answer explicitly:

1. Which rubric dimensions show adequate human interpretability?
2. Where do graders disagree with humans or each other?
3. Does the intervention reduce unsupported agreement without suppressing warranted agreement?
4. Does it create generalized skepticism, refusal, or excessive hedging?
5. Are effects consistent enough across the two target families to motivate broader testing?
6. Which scenarios/rubric items must change on development data?
7. Is the current helpfulness/noninferiority margin defensible?
8. What claims remain `not_assessed`?

## 9. Public evidence package

After the final calibration analysis, create `docs/calibration-2026q3-findings.md` containing:

- exact model/provider identifiers and dates;
- claim boundary;
- scenario count and true independent-unit count;
- human-rater count;
- agreement results;
- failure/missingness summary;
- supported-vs-unsupported agreement diagnostic;
- overcorrection/refusal findings;
- changes made before the next frozen pilot;
- limitations.

Do not publish private annotation linkage keys or provider credentials.

## Stop conditions

Stop rather than improvise if:

- a configured model is retired/unavailable;
- cost estimate exceeds the operator-approved ceiling;
- scenario review identifies a critical construct-validity defect;
- target or grader identity differs from the frozen manifest;
- human annotation blinding is compromised materially;
- generated evidence fails integrity checks.

A stopped calibration with a documented reason is preferable to a completed run whose evidence cannot support interpretation.
