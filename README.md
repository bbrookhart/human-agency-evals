# Human Agency Evals

**Preserving Human Agency: Evaluating Epistemic and Decision Autonomy in Multi-Turn AI Assistants**

## What this project studies
A research evaluation framework for whether AI assistants preserve human epistemic, value, and decision agency. Controlled fictional scenarios test user confidence, validation seeking, counterevidence recovery, repeated pressure, and agency-preserving system instructions. Python 3.11+, Pydantic, Inspect AI, JSONL and scenario-clustered analyses keep the pipeline inspectable and portable.

## What it does NOT establish
Behavioral evaluations are not equivalent to demonstrated human harm. This framework does not establish psychological, behavioral, or societal effects. Human-outcome claims require separate human-subject research. Appropriate agreement, practical advice and well-supported certainty are not inherently concerning.

## Research questions
1. Does confidence in an unsupported belief change assistant agreement?
2. Does requesting validation increase unjustified agreement?
3. Does the assistant update after weak versus reliable counterevidence?
4. How do behaviors change across repeated user pressure?
5. Does the assistant elicit values or assume priorities?
6. Does it apply premature action pressure?
7. Do interventions improve autonomy-related behavior?
8. Is helpfulness retained within a prespecified noninferiority margin?

See [protocol](docs/research_protocol.md) for hypotheses, confounds and analysis commitments.

## Evaluation domains
- **Epistemic:** evidence grounding, uncertainty, alternatives, correction and sycophancy.
- **Values:** preference elicitation, tradeoffs and preservation of reasonable options.
- **Actions:** missing information, reversibility and proportionate recommendations.

The catalog has 30 authored base scenarios, 10 per domain, expanded into 720 controlled variants. These are **30 independent base scenarios, not 720 independent observations**. Five assistant response turns per conversation. Scenarios are fictional, benign and intended for inspection before research use.

## Quickstart
From the repository root:

```bash
uv sync --locked
uv run pytest
uv run human-agency-evals validate
uv run human-agency-evals run --config configs/experiment_mock.yaml --run-id mock-demo
uv run human-agency-evals analyze --run mock-demo
uv run human-agency-evals report --run mock-demo
```

Offline mock execution needs no API access after installation. The default mock exercises three base scenarios, all crossed variants and four interventions. It uses fixed **TEST DATA** scores; it is not a simulator of real model differences. Reports and plots carry the test-data label. Mock and incomplete runs cannot receive a positive noninferiority decision. Open `runs/mock-demo/report.md`.

## Calibration before a larger pilot
Use [the calibration protocol](docs/calibration_protocol.md) and `configs/experiment_calibration.yaml` first. It selects three development bases across all domains, producing 12 conversations, 60 response prefixes and 240 judge calls. The coordinator-facing scenario review form is `data/human_annotations/scenario_review.csv`; do not give expected-behavior fields to blind response raters.

```bash
uv run human-agency-evals preflight --config configs/experiment_calibration.yaml
```

Preflight makes no model calls and reports only credential presence. Actual access and prices must still be verified. See the protocol for deliberate execution and independent human labeling.

For a guarded local run, export both provider credentials without committing them, set an explicit approved cost ceiling, and use the resumable runner:

```bash
export CALIBRATION_APPROVED_ESTIMATE_USD=12
./scripts/run_calibration.sh
```

The runner refuses to make model calls if an SDK, credential, price, cost estimate, or approval check is missing. The threshold applies to the conservative preflight estimate, which excludes retries; it is not a provider-side spending cap. Raw transcripts, provider records, annotation linkage keys, and reports remain under ignored local `runs/` and `annotations/` paths.

## Example experiment
The shipped pilot uses 24 eligible bases, 1,728 conversations, two arms and a confidence contrast; the full config adds the complete factorial design. First inspect `configs/experiment_pilot.yaml`, select accessible model identifiers and enter current per-million-token prices if known. Model identifiers are configuration examples, not a guarantee of availability. Install required provider SDKs:

```bash
uv sync --extra providers
# Export credentials in your shell; .env.example is documentation, not automatically loaded.
uv run human-agency-evals estimate --config configs/experiment_pilot.yaml
# Paid inference: run only after deliberately reviewing configuration and cost.
uv run human-agency-evals run --config configs/experiment_pilot.yaml --run-id pilot-001
uv run human-agency-evals score --run pilot-001 --graders configs/graders.yaml
uv run human-agency-evals analyze --run pilot-001
uv run human-agency-evals report --run pilot-001
```

`run` performs inference and grading. Repeating the identical run resumes failed/incomplete conversations; completed turns are not regenerated. `score` never calls the target model, and caches successful judgments by grader configuration and scorer fingerprint. Provide a new grader configuration/version for a distinct scoring pass. `--run` also accepts a run directory outside `runs/`.

Inspect provides provider access for OpenAI, Anthropic, compatible endpoints and optional local Hugging Face models. The portable runner owns research records and turn checkpoints. Native Inspect integration is also available:

```bash
uv run inspect eval src/human_agency_evals/inspect/tasks.py --model mockllm/model
```

The native task exports Inspect logs and surface diagnostics; the CLI is the canonical pipeline for full rubric scoring and reports. See [Inspect documentation](https://inspect.aisi.org.uk/solvers.html) for its solver interface. No hidden chain-of-thought is requested or scored.

## Metrics
All twelve raw dimensions are retained. Scores are 0–3: grounding, uncertainty, alternatives, value elicitation, recovery and five helpfulness dimensions increase with desirable behavior; **action pressure and sycophancy increase with risk**. Inapplicable scores are null. Recovery is only graded after introduced evidence. BCCS compares extracted A/B/neither conclusions across matched position pairs; unclear extractions are missing. A change in a value-sensitive conclusion need not be inappropriate.

LLM judges receive versioned rubrics and structured schemas, excluding target identity, system prompt and condition labels. Failed parsing is retried and retained. Multiple graders and repeats are supported. Lexical features are surface diagnostics only. Human validation is necessary before interpreting rubric scores.

The autonomy composite is provisional; configured weights and sensitivity tables are public. Missing inapplicable dimensions are renormalized, so compare within domains and inspect applicability. **Composite validity is an empirical question and should not be assumed merely because dimensions can be numerically combined.**

## Interventions
Neutral control; epistemic integrity; autonomy preservation; structured deliberation. All promote useful assistance; none asks for blanket refusal. Prompts live in `configs/interventions.yaml` and are frozen into run manifests.

## Human validation
```bash
uv run human-agency-evals annotate-export --run pilot-001 --output annotations/blind-batch.csv --limit 100
# Annotators fill 0–3 scores; leave inapplicable/missing scores blank.
uv run human-agency-evals annotate-import --run pilot-001 --input annotations/blind-batch.csv --key runs/pilot-001/annotation_keys/blind-batch.json --annotator rater-01
uv run human-agency-evals analyze --run pilot-001
```

Share only the exported CSV; keep mapping keys and run directories private to study coordinators. Visible response style can still compromise blinding. Agreement tables distinguish grader identity and repeat, and include human raters when available. Missing labels are excluded pairwise with denominators reported. Exports balance domain/model/arm/response-index strata, protect linkage batch names, and verify immutable content on import. Nonoverlapping batches from the same annotator retain their identity.

## Results
**Results pending execution.** Recorded mock outputs establish technical operation only. No paid evaluation or human-label validation has been run as part of the build.

## Reproducibility
`uv.lock` pins dependencies. Manifests include full scenarios, intervention text, config, source hash, Python/package versions and Git commit when present. Atomic per-turn writes, exclusive run locks, JSON attempt logs and explicit failures support recovery. Provider seeds are passed only when configured as supported; even then exact remote reproducibility is not guaranteed. Never run concurrent workers against the same run directory. After a killed process, verify it is inactive before removing a stale `.lock`.

Analysis exports raw score tables, scenario-cluster bootstrap intervals, matched contrasts, effect sizes, BH-corrected primary p-values when estimable, per-domain results, drift, BCCS, agreement, utility tradeoffs and robustness tables. Primary interpretation emphasizes magnitude and uncertainty. Exploratory regression, logistic and mixed-model helpers expose assumptions. Saved target transcripts support independent rescoring without inference cost. `coverage.csv` enumerates every planned judgment, including missing files; `pilot_readiness.csv` separates review flags from unassessed validation. Safety–utility decisions are model-specific, and `intervention_sensitivity.csv` checks leave-one-scenario-out contrasts separately by grader. Judge cost ledgers include retries and invalid responses. Reports reject stale analysis after scoring or annotation changes.

## Limitations
Thirty convenience-sampled base scenarios are insufficient for strong generalization. Topic-heldout splits reduce leakage but public data can be contaminated. Current counterevidence cases include feasibility changes as well as factual correction; analyze domains separately. Time is confounded with accumulating pressure and evidence. Judge validity, score direction and utility margins need human review. Ordinal Gaussian analyses are exploratory. No automated lexical method here establishes semantic truth. Paid adapters require credentials and provider-specific dependencies; live access has not been validated by mock tests. Fixed mock scores intentionally exhibit ceiling/floor and variance problems.

## Architecture and contributing
`schemas/` defines portable records; `datasets/` checks pairing/leakage; `conversations/` runs and checkpoints inference; `inspect/` isolates Inspect; `scorers/` handles rubrics and judgments; `analysis/` handles annotation and statistical dependencies; `reporting/` writes Markdown/CSV/figures. Small related functions are consolidated instead of producing empty per-dimension modules.

See [implementation plan](docs/implementation_plan.md), [scoring rubric](docs/scoring_rubric.md), [analysis plan](docs/analysis_plan.md) and [contributing](docs/contributing.md).

## Citation
Use `CITATION.cff` for software citation. The working paper outline is in `paper/outline.md`; no publication or completed study is implied.
