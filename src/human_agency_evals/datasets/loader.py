from pathlib import Path

from human_agency_evals.schemas.scenario import Scenario


def load(paths: list[str]) -> list[Scenario]:
    result = []
    for name in paths:
        path = Path(name)
        for index, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                try:
                    result.append(Scenario.model_validate_json(line))
                except ValueError as exc:
                    raise ValueError(f"{path}:{index}: {exc}") from exc
    if not result:
        raise ValueError("empty dataset")
    return result
