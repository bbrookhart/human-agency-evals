# Dataset card
Version 1; MIT license, matching repository licensing. Intended use: research on observable conversational behaviors potentially related to autonomy; not clinical assessment or a human-harm benchmark.

Creation: 30 individually authored fictional cases (10 per domain) in `src/human_agency_evals/scenarios/catalog.py`. A deterministic builder crosses position, confidence, validation seeking and counterevidence strength. Five scripted user turns. Strong evidence excludes both initial options; weak evidence wraps that update as unverified second-hand information. This design keeps update text constant across belief positions. It simplifies reality and mixes correction with feasibility changes in values/actions; analyze these separately.

Topic hashes fix development/pilot/heldout assignment before results. All variants of a topic stay together. The starter pilot necessarily uses fewer than 30 base scenarios to preserve heldout. Split counts and domain coverage are machine-auditable. No tuning on heldout. Public availability means heldout is not secret or contamination-proof.

Scope: low-risk workplace, hobby, education, scheduling, everyday consumer and project choices. Excludes self-harm, medicine, law, high-risk finance, political persuasion, abuse and other high-risk contexts. No personal records or real participants.

Limitations: English-language convenience sample, culturally specific deliberation norms, templated follow-ups, only two initial position anchors, intentionally clear strong corrections, no counterbalanced update timing, no realistic user adaptation. Authoring has not received independent human review. Automated leakage checks detect structural violations and explicit markers, not all semantic hints. Releasing data risks benchmark overfitting and model contamination.

Review each case for realism, expected behavior, rubric applicability and factual invariance. Record changes as a new version, rebuild without inspecting model results, and update fingerprints. Do not generate thousands of low-quality substitutes.
