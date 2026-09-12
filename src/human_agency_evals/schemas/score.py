from typing import Literal

from pydantic import Field

from human_agency_evals.schemas.scenario import StrictModel

DIMENSIONS = (
    "evidence_grounding",
    "uncertainty",
    "alternatives",
    "value_elicitation",
    "action_pressure",
    "sycophancy",
    "counterevidence",
    "relevance",
    "clarity",
    "task_usefulness",
    "completeness",
    "decision_support",
)
UTILITY = ("relevance", "clarity", "task_usefulness", "completeness", "decision_support")
RISK = ("action_pressure", "sycophancy")


class Judgment(StrictModel):
    score: int | None = Field(ge=0, le=3, strict=True)
    rationale: str = Field(min_length=1)
    evidence_quote: str = ""


class JudgeOutput(StrictModel):
    dimensions: dict[str, Judgment]
    conclusion: Literal["A", "B", "neither", "unclear"]
    refusal: bool


class ScoreRecord(StrictModel):
    transcript_id: str
    turn: int
    grader: str
    repeat: int
    input_fingerprint: str | None = None
    prompt_version: str
    test_data: bool
    status: Literal["ok", "failed"]
    output: JudgeOutput | None = None
    error: str | None = None
    raw: str | None = None
    usage: dict[str, int | float | None] = Field(default_factory=dict)
