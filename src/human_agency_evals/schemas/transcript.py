from typing import Literal

from pydantic import Field, model_validator

from human_agency_evals.schemas.scenario import Scenario, StrictModel


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Generation(StrictModel):
    turn: int
    text: str
    latency: float
    timestamp: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    raw: dict = Field(default_factory=dict)


class Transcript(StrictModel):
    transcript_id: str
    scenario: Scenario
    model: str
    model_config_record: dict
    intervention: str
    repetition: int
    seed: int
    system_prompt: str
    test_data: bool
    messages: list[Message]
    generations: list[Generation] = Field(default_factory=list)
    status: Literal["pending", "complete", "failed"] = "pending"
    errors: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent_history(self):
        if [g.turn for g in self.generations] != list(range(1, len(self.generations) + 1)):
            raise ValueError("generation turns must be contiguous")
        if len(self.messages) != 1 + 2 * len(self.generations):
            raise ValueError("transcript message/generation count mismatch")
        if self.messages[0].role != "system" or self.messages[0].content != self.system_prompt:
            raise ValueError("system prompt mismatch")
        for index, generation in enumerate(self.generations):
            user, assistant = self.messages[1 + 2 * index : 3 + 2 * index]
            if (
                user.role != "user"
                or assistant.role != "assistant"
                or assistant.content != generation.text
            ):
                raise ValueError("malformed transcript role sequence or completion mismatch")
        if self.status == "complete" and len(self.generations) != len(self.scenario.conversation):
            raise ValueError("complete transcript is missing turns")
        return self
