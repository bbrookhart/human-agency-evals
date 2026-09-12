"""Read-only execution audit; reports credential presence, never credential contents."""

import importlib.util
import os
from pathlib import Path

from human_agency_evals.conversations.runner import estimate, prepare
from human_agency_evals.io import digest

PROVIDERS = {
    "openai": ("openai", "OPENAI_API_KEY"),
    "anthropic": ("anthropic", "ANTHROPIC_API_KEY"),
    "google": ("google.genai", "GOOGLE_API_KEY"),
    "hf": ("transformers", None),
}


def preflight(config_path: str) -> dict:
    config, scenarios, prompts = prepare(config_path)
    checks = []
    for role, specs in (("target", config.models), ("grader", config.graders.models)):
        for spec in specs:
            provider = spec.name.split("/", 1)[0]
            module, default_key = PROVIDERS.get(provider, (None, None))
            try:
                installed = importlib.util.find_spec(module) is not None if module else None
            except ModuleNotFoundError:
                installed = False
            key_name = spec.api_key_env or default_key
            mock = provider == "mock"
            checks.append(
                {
                    "role": role,
                    "model": spec.name,
                    "sdk": module,
                    "sdk_available": True if mock else installed,
                    "credential_variable": key_name,
                    "credential_present": bool(os.environ.get(key_name)) if key_name else None,
                    "pricing_known": mock
                    or (spec.input_per_million is not None and spec.output_per_million is not None),
                    "remote_access_verified": False,
                }
            )
    return {
        "config": str(Path(config_path)),
        "config_fingerprint": digest(config.model_dump()),
        "dataset_fingerprint": digest([s.model_dump() for s in scenarios]),
        "base_scenarios": len({s.metadata.base_id for s in scenarios}),
        "domains": sorted({s.domain for s in scenarios}),
        "splits": sorted({s.metadata.split for s in scenarios}),
        "estimate": estimate(config, scenarios, prompts),
        "checks": checks,
        "note": "No model calls made. Presence of a credential or SDK does not verify model access. Unknown pricing is not zero cost.",
    }
