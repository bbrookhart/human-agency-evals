from human_agency_evals.io import read_yaml


def load_interventions(path: str) -> dict[str, str]:
    prompts = read_yaml(path)
    if not isinstance(prompts, dict) or not all(
        isinstance(v, str) and v.strip() for v in prompts.values()
    ):
        raise ValueError("interventions must map names to nonempty prompts")
    return prompts
