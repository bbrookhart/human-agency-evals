from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceItem(StrictModel):
    evidence_id: str
    text: str = Field(min_length=1)
    available_at: int = Field(default=1, ge=1)
    reliability: Literal["ambiguous", "weak", "strong"] = "ambiguous"


class ManipulationSpec(StrictModel):
    position: Literal["A", "B"]
    user_confidence: Literal["low", "medium", "high"] = "low"
    validation_seeking: Literal["absent", "present"] = "absent"
    counterevidence: Literal["absent", "weak", "strong"] = "absent"
    emotional_pressure: Literal["neutral", "elevated"] = "neutral"
    conversation_length: Literal["short", "medium", "long"] = "medium"
    wording_variant: str = "original"


class TurnTemplate(StrictModel):
    index: int = Field(ge=1)
    text: str = Field(min_length=1)


class ExpectedBehavior(StrictModel):
    desirable: str
    failure_mode: str
    ground_truth: str | None = None
    applicable_dimensions: list[str]


class ScenarioMetadata(StrictModel):
    base_id: str
    pair_id: str
    condition_id: str
    split: Literal["development", "pilot", "heldout"]
    topic: str
    target_construct: str = Field(validation_alias=AliasChoices("target_construct", "construct"))
    fictional: Literal[True] = True
    authoring_version: str = "1"


class Scenario(StrictModel):
    scenario_id: str
    domain: Literal["epistemic", "values", "actions"]
    title: str
    context: str
    evidence: list[EvidenceItem] = Field(min_length=1)
    user_position: str
    manipulation: ManipulationSpec
    conversation: list[TurnTemplate] = Field(min_length=1)
    expected_behavior: ExpectedBehavior
    metadata: ScenarioMetadata

    @model_validator(mode="after")
    def valid_sequence(self):
        if [t.index for t in self.conversation] != list(range(1, len(self.conversation) + 1)):
            raise ValueError("turn indices must be contiguous from 1")
        if len({e.evidence_id for e in self.evidence}) != len(self.evidence):
            raise ValueError("duplicate evidence IDs")
        if any(e.available_at > len(self.conversation) for e in self.evidence):
            raise ValueError("evidence is scheduled after conversation ends")
        return self
