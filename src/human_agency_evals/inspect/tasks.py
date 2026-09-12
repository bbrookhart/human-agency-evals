from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import system_message

from human_agency_evals.conversations.state import render_turn
from human_agency_evals.datasets.loader import load
from human_agency_evals.datasets.validator import validate
from human_agency_evals.inspect.scorers import response_diagnostics
from human_agency_evals.inspect.solvers import autonomy_conversation
from human_agency_evals.interventions.loader import load_interventions


@task
def autonomy_task(
    dataset: str = "data/development/scenarios.jsonl", intervention: str = "control"
) -> Task:
    scenarios = load([dataset])
    validate(scenarios)
    prompt = load_interventions("configs/interventions.yaml")[intervention]
    return Task(
        dataset=[
            Sample(id=s.scenario_id, input=render_turn(s, 1), metadata={"scenario": s.model_dump()})
            for s in scenarios
        ],
        solver=[system_message(prompt), autonomy_conversation()],
        scorer=response_diagnostics(),
    )
