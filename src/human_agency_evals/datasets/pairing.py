from collections import defaultdict

from human_agency_evals.schemas.scenario import Scenario


def pairs(scenarios: list[Scenario]) -> dict[str, tuple[Scenario, Scenario]]:
    grouped = defaultdict(list)
    for scenario in scenarios:
        grouped[scenario.metadata.pair_id].append(scenario)
    result = {}
    for key, members in grouped.items():
        if len(members) != 2 or {s.manipulation.position for s in members} != {"A", "B"}:
            raise ValueError(f"broken position pair: {key}")
        a, b = sorted(members, key=lambda s: s.manipulation.position)
        for field in ("context", "evidence", "conversation", "domain", "expected_behavior"):
            if getattr(a, field) != getattr(b, field):
                raise ValueError(f"pair {key} changes {field}")
        if a.manipulation.model_dump(exclude={"position"}) != b.manipulation.model_dump(
            exclude={"position"}
        ):
            raise ValueError(f"confounded pair: {key}")
        for field in ("base_id", "split", "topic"):
            if getattr(a.metadata, field) != getattr(b.metadata, field):
                raise ValueError(f"pair {key} changes {field}")
        if a.metadata.condition_id == b.metadata.condition_id or a.user_position == b.user_position:
            raise ValueError(f"pair {key} lacks distinct conditions")
        result[key] = (a, b)
    return result
