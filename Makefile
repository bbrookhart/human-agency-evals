.PHONY: test lint validate mock

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src/human_agency_evals

validate:
	uv run human-agency-evals validate

mock:
	uv run human-agency-evals run --config configs/experiment_mock.yaml --run-id mock-demo
	uv run human-agency-evals analyze --run mock-demo
	uv run human-agency-evals report --run mock-demo
