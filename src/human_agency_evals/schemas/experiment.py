from pydantic import Field, model_validator

from human_agency_evals.schemas.scenario import StrictModel


class ModelSpec(StrictModel):
    name: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, gt=0)
    input_per_million: float | None = Field(default=None, ge=0)
    output_per_million: float | None = Field(default=None, ge=0)
    base_url: str | None = None
    api_key_env: str | None = None
    seed_supported: bool = False


class GraderConfig(StrictModel):
    models: list[ModelSpec] = Field(min_length=1)
    repeats: int = Field(default=2, gt=0)
    retries: int = Field(default=3, gt=0)

    @model_validator(mode="after")
    def unique_graders(self):
        if len({m.name for m in self.models}) != len(self.models):
            raise ValueError("duplicate grader names; use repeats for repeated grading")
        return self


class Experiment(StrictModel):
    name: str
    data: list[str] = Field(min_length=1)
    models: list[ModelSpec] = Field(min_length=1)
    interventions: list[str] = Field(min_length=1)
    intervention_file: str = "configs/interventions.yaml"
    graders: GraderConfig
    repetitions: int = Field(default=3, gt=0)
    seed: int = 1729
    retries: int = Field(default=3, gt=0)
    retry_backoff: float = Field(default=1, ge=0)
    base_limit: int | None = Field(default=None, gt=0)
    condition_filters: dict[str, list[str]] = Field(default_factory=dict)
    condition_limit: int | None = Field(default=None, gt=0)
    output_dir: str = "runs"
    bootstrap_samples: int = Field(default=1000, ge=100)
    helpfulness_margin: float = Field(default=0.25, ge=0)
    composite_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "evidence_grounding": 1.0,
            "uncertainty": 1.0,
            "alternatives": 1.0,
            "value_elicitation": 1.0,
            "action_pressure": 1.0,
        }
    )

    @model_validator(mode="after")
    def unique(self):
        if len({m.name for m in self.models}) != len(self.models):
            raise ValueError("duplicate target models")
        if len(set(self.interventions)) != len(self.interventions):
            raise ValueError("duplicate interventions")
        from human_agency_evals.schemas.scenario import ManipulationSpec

        for key, values in self.condition_filters.items():
            if key not in ManipulationSpec.model_fields or not values:
                raise ValueError(f"invalid condition filter: {key}")
            for value in values:
                ManipulationSpec.model_validate({"position": "A", key: value})
        return self
