from human_agency_evals.schemas.scenario import Scenario


def wording_variant(scenario: Scenario, suffix: str, replacements: dict[str, str]) -> Scenario:
    """Create an audited wording variant; human semantic review remains required."""
    copy = scenario.model_copy(deep=True)
    for turn in copy.conversation:
        for old, new in replacements.items():
            turn.text = turn.text.replace(old, new)
    copy.manipulation.wording_variant = suffix
    copy.scenario_id += "-" + suffix
    copy.metadata.pair_id += "-" + suffix
    copy.metadata.condition_id += "-" + suffix
    return Scenario.model_validate(copy.model_dump())
