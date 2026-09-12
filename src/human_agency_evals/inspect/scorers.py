from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from human_agency_evals.scorers.deterministic import features


@scorer(metrics=[mean()])
def response_diagnostics() -> Scorer:
    """Surface diagnostics only. Use portable CLI rubric scoring for substantive judgments."""

    async def score(state: TaskState, target: Target) -> Score:
        return Score(
            value=features(state.output.completion),
            explanation="Surface diagnostics, not autonomy scores.",
        )

    return score
