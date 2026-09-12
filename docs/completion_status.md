# Completion status: engineering hardening and calibration preparation

This record distinguishes working infrastructure from empirical research validation. Mock execution is not a successful scientific pilot.

## Completed and checked
- Installation from the lockfile, including the optional remote-provider SDKs.
- 30 authored base cases / 720 condition records / 360 position pairs, all validated.
- Fixed evidence across confidence/validation contrasts, not merely within position pairs; helpfulness applicability and literal expected-behavior leakage checks.
- Resumable inference, independent rescoring, judge-prefix cache fingerprints and stale-score rejection.
- Judge call ledger counts successful responses, invalid-JSON attempts and retries; unknown billing remains unknown.
- Planned-design coverage and grouped missingness include absent files, not only logged failures.
- Model-specific safety–utility comparisons; mock/incomplete/insufficient-cluster decisions remain unassessed.
- Model/grader-specific leave-one-scenario-out intervention contrasts.
- Separate weak/strong counterevidence analysis, including an exploratory subset with prior judged sycophancy >=2. Conditioning is post-treatment selection, not a causal arm comparison.
- Rank-deficient regression designs are rejected; stale regression output is removed before refitting.
- Balanced blinded annotation exports, protected batch linkage names, content/rubric integrity checks and consistent identity across nonoverlapping batches from one annotator.
- Pilot readiness review flags, complete JSON null handling and report provenance checks.
- Small calibration config and coordinator review form, with explicit instructions separating scenario review from blinded output rating.

## Latest verification
26 tests passed, including failure recovery, judge retry billing, invalidation, opposing effects by model, absent artifacts, annotation integrity, matched contrasts and native Inspect offline execution. Ruff lint and formatting passed; mypy passed for 53 source files. Dataset validation passed.

`runs/mock-hardened/` contains 288 mock conversations, 1,440 assistant responses, 5,760 judgments, coverage/missingness tables, readiness flags, six PNG/SVG figures, a report and 60 blank blinded rating forms. Every judgment is TEST DATA. No human ratings were filled in. Cached resume and rescoring were checked; subsequent analysis refinements used saved transcripts.

## Concrete next experiment
Read `docs/calibration_protocol.md` and `configs/experiment_calibration.yaml`. The calibration selects three development bases (one per domain), two positions and two arms: 12 conversations, 60 response prefixes and 240 judge calls. Preflight is read-only:

```bash
uv run human-agency-evals preflight --config configs/experiment_calibration.yaml
```

The provider SDKs were installed. At the latest local preflight, `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` were absent. Current prices were not configured. Model access remains unverified. Credential values must be configured through the environment, not stored in repository files.

## Still pending; not substitutable with software tests
1. Independent review of scenario quality and construct/rubric applicability.
2. Selection/verification of accessible model IDs and current prices, credentials, and deliberate paid calibration execution.
3. Independent human ratings and semantic-equivalence review for grader style diagnostics.
4. Calibration analysis and any warranted redesign using development/pilot data only.
5. A frozen analysis protocol and an adequately sized independently authored heldout sample.
6. Real pilot/full-study execution and evidence-backed research findings, report discussion and paper results.

No completed study, validated composite, human-outcome effect, or empirically established intervention success is claimed. The starter heldout split has only six base cases and cannot support strong generalization. Remaining research steps need external observations and independent people; they cannot be completed honestly by generating more mock data.
