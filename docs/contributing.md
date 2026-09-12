# Contributing
Use Python 3.11+, `uv sync --locked`, `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, and `uv run mypy src/human_agency_evals`.

Keep data conditions machine-readable and changes scientifically motivated. Never tune on heldout. Add regression tests for pair invariance, timing, missingness and resume when changing those behaviors. Preserve raw dimensions and score direction. Version rubrics when semantic interpretation changes. Human-review non-substantive wording variants before calling them equivalent.

Avoid fabricated findings/citations and claims of human harm from model outputs. Do not include secrets, identifying participant data or paid run artifacts in commits. Mock fixtures must say TEST DATA. Report validation evidence and remaining limitations in changes.
