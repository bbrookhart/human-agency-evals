# Engineering and research handoff

## Built
Portable Pydantic schemas; 30 authored fictional base scenarios / 720 crossed condition records / 360 fixed-evidence pairs; topic splits and dataset validation; four interventions; stateful inference with atomic turn checkpoints, retry/backoff, explicit failures, manifests and cost reports; Inspect provider adapter and native task; deterministic surface diagnostics, structured multi-judge rubric scoring and cached rescoring; blinded annotation export/import; agreement and statistical helpers; analysis tables, six PNG/SVG figures and Markdown reports; documentation, notebooks, paper outline, lockfile and CI workflow.

## Architecture
`schemas`, `datasets` and `scenarios` define the portable research records. `conversations` owns inference and resume; `inspect` contains provider-specific integration. `scorers` operates on saved transcript prefixes independently of target inference. `analysis` handles clustering, matched contrasts, regression, mixed effects, agreement, robustness and plotting. `reporting` composes the recorded outputs. Small rubric dimensions share a registry rather than empty wrapper modules.

## Methodology
Epistemic, value and action autonomy are latent constructs; ordinal behavioral proxies do not establish human harm. Position pairs preserve evidence and factual context. Confidence, validation and counterevidence are explicit crossed variables; five response turns and four system arms support intervention analysis. Helpfulness is independently scored. Composite and noninferiority margin are provisional. Topic splits contain 10 development, 14 pilot and 6 heldout base cases. The pilot uses 24 eligible bases, two arms, a confidence contrast and strong counterevidence, totaling 1,728 conversations. Full configuration includes the factorial design on heldout; more independently authored heldout scenarios are needed before credible confirmatory inference.

## Run
```bash
uv sync --locked
uv run pytest
uv run human-agency-evals validate
uv run human-agency-evals run --config configs/experiment_mock.yaml --run-id mock-demo
uv run human-agency-evals score --run mock-demo
uv run human-agency-evals analyze --run mock-demo
uv run human-agency-evals report --run mock-demo
```
Review provider IDs, credentials and current prices before deliberate paid execution. `uv sync --extra providers` installs optional remote SDKs. `estimate --config configs/experiment_pilot.yaml` makes no model calls. A null cost means pricing is unknown, not free.

## Validation executed
Python 3.13 environment installed from `uv.lock`; locked offline sync passed after dependency installation. All 16 pytest tests passed, including offline native Inspect, mid-conversation failure recovery, immutable saved turns, judge retries, annotation round-trip, statistics and report generation. Ruff lint and formatting passed; mypy passed for 47 source files. Dataset validation passed for all 720 records. `mock-final` completed 288 conversations / 1,440 assistant responses / 5,760 test judgments with zero failures, covering all three domains and four arms. Cached resume and rescoring completed. Analysis exported CSV tables and six figures in both PNG and SVG; frontier layout was visually inspected and corrected. See `runs/mock-final/report.md`.

## Limitations
No paid models or human-label study ran. Live provider access is unverified; provider adapter plumbing is tested with a fake transport. All mock scores are fixed TEST DATA and do not establish rubric reliability, variance or research findings. The sample is small, public and convenience-authored, with unequal domain split sizes. Drift confounds time, accumulated pressure and evidence exposure. Counterevidence in value/action cases often changes feasibility. Ordinal Gaussian analyses and composites require validation. Costs are heuristic estimates and may omit unavailable failed-call billing. Native Inspect logs and canonical CLI reports have distinct formats.

## Recommended next experiment
Conduct blinded calibration before a larger pilot: independently review the 24 development/pilot scenarios; then deliberately run a small, domain-balanced batch, have two independent humans and two grader families label the same 60 response prefixes, and include semantically matched concise/verbose variants. Assess disagreement, ceiling/floor effects, refusals, helpfulness and cost reconciliation. Refine only development/pilot data and freeze the rubric, margins and protocol before new heldout inference.

## Review first
1. `README.md`
2. `docs/research_protocol.md`
3. `docs/construct_definition.md`
4. `configs/experiment_pilot.yaml`
5. `src/human_agency_evals/scorers/llm_judge.py`

## Subsequent hardening
The current status supersedes the original validation count above: 26 tests pass and 53 source files pass mypy. See `docs/completion_status.md` for the audit and `runs/mock-hardened/report.md` for the updated mock report. A smaller 12-conversation calibration configuration and `docs/calibration_protocol.md` now precede the large pilot. SDKs are installed; credentials, live access, pricing and independent labels remain pending.
