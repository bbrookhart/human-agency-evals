"""Provider portability through Inspect; no provider SDK in the research core."""

import os
from typing import Any

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)

from human_agency_evals.conversations.models import Response
from human_agency_evals.schemas.experiment import ModelSpec
from human_agency_evals.schemas.transcript import Message


class InspectModel:
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    async def generate(self, messages: list[Message], seed: int) -> Response:
        options: dict[str, Any] = {}
        if self.spec.base_url:
            options["base_url"] = self.spec.base_url
        if self.spec.api_key_env:
            key = os.environ.get(self.spec.api_key_env)
            if not key:
                raise ValueError(
                    f"missing credential environment variable: {self.spec.api_key_env}"
                )
            options["api_key"] = key
        model = get_model(self.spec.name, **options)
        classes = {
            "system": ChatMessageSystem,
            "user": ChatMessageUser,
            "assistant": ChatMessageAssistant,
        }
        config = GenerateConfig(
            temperature=self.spec.temperature,
            max_tokens=self.spec.max_tokens,
            seed=seed if self.spec.seed_supported else None,
        )
        output = await model.generate(
            [classes[m.role](content=m.content) for m in messages], config=config
        )
        usage = output.usage
        return Response(
            output.completion,
            usage.input_tokens if usage else None,
            usage.output_tokens if usage else None,
            output.model_dump(mode="json"),
        )
