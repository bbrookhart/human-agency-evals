"""Mock is a test fixture, never a proxy for real model behavior."""

from dataclasses import dataclass, field
from typing import Protocol

from human_agency_evals.schemas.experiment import ModelSpec
from human_agency_evals.schemas.transcript import Message


@dataclass
class Response:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw: dict = field(default_factory=dict)


class Model(Protocol):
    async def generate(self, messages: list[Message], seed: int) -> Response: ...


class MockModel:
    async def generate(self, messages: list[Message], seed: int) -> Response:
        text = "TEST DATA: The available evidence is insufficient to choose confidently. Consider both options and check the missing information. Which tradeoffs matter most to you? A reversible next step is to gather more information."
        return Response(
            text,
            sum(len(m.content.split()) for m in messages),
            len(text.split()),
            {"fixture": True, "seed": seed},
        )


def make_model(spec: ModelSpec) -> Model:
    if spec.name.startswith("mock/"):
        return MockModel()
    from human_agency_evals.inspect.adapter import InspectModel

    return InspectModel(spec)
