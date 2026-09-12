from collections import defaultdict
from string import Formatter

from human_agency_evals.datasets.pairing import pairs
from human_agency_evals.io import digest
from human_agency_evals.schemas.scenario import Scenario
from human_agency_evals.schemas.score import DIMENSIONS, UTILITY


def validate(scenarios: list[Scenario]) -> dict:
    ids = [s.scenario_id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scenario IDs")
    split_groups = defaultdict(set)
    controls = defaultdict(set)
    conditions = set()
    for s in scenarios:
        condition = (s.metadata.base_id, s.metadata.condition_id)
        if condition in conditions:
            raise ValueError("duplicate condition within base scenario")
        conditions.add(condition)
        control_key = (
            s.metadata.base_id,
            s.manipulation.counterevidence,
            s.manipulation.wording_variant,
            s.manipulation.conversation_length,
        )
        controls[control_key].add(
            digest(
                [
                    s.context,
                    [e.model_dump() for e in s.evidence],
                    [t.model_dump() for t in s.conversation],
                ]
            )
        )
        for key in (
            ("base", s.metadata.base_id),
            ("topic", s.metadata.topic),
            ("context", s.context),
        ):
            split_groups[key].add(s.metadata.split)
        if not set(s.expected_behavior.applicable_dimensions) <= set(DIMENSIONS):
            raise ValueError(f"rubric mismatch: {s.scenario_id}")
        if not set(UTILITY) <= set(s.expected_behavior.applicable_dimensions):
            raise ValueError("helpfulness dimensions must remain applicable")
        visible = s.context + "\n" + "\n".join(t.text for t in s.conversation)
        if s.expected_behavior.desirable in visible:
            raise ValueError("expected behavior leaked verbatim into model input")
        for turn in s.conversation:
            fields = {f for _, f, _, _ in Formatter().parse(turn.text) if f}
            if not fields <= {"position", "confidence", "validation", "emotion"}:
                raise ValueError(f"unknown or answer-leaking template field: {fields}")
            if any(
                x in turn.text.lower()
                for x in ("expected_behavior", "ground_truth", "desired score")
            ):
                raise ValueError("answer leakage marker")
        future = [e for e in s.evidence if e.available_at > 1]
        if s.manipulation.counterevidence == "absent" and future:
            raise ValueError("absent counterevidence has later evidence")
        if s.manipulation.counterevidence != "absent" and not future:
            raise ValueError("counterevidence missing")
    if any(len(v) > 1 for v in controls.values()):
        raise ValueError("confidence/validation contrast changes evidence, context or templates")
    if any(len(v) > 1 for v in split_groups.values()):
        raise ValueError("base/topic/context split leakage")
    return {
        "conditions": len(scenarios),
        "base_scenarios": len({s.metadata.base_id for s in scenarios}),
        "pairs": len(pairs(scenarios)),
    }
