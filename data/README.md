# Dataset
30 authored fictional base cases → 720 controlled condition records → 360 fixed-evidence position pairs. Run `uv run human-agency-evals validate` for audit counts. Inspect the compact authored source in `src/human_agency_evals/scenarios/catalog.py`; rebuild with `uv run python scripts/build_dataset.py`.

Development and pilot may inform design; heldout must not. Split assignment uses topics, never individual conditions. Human annotation exports and private linkage keys are run artifacts, not benchmark data. See `docs/dataset_card.md` for licensing, limitations and authoring details.
