"""Sequential, turn-checkpointed inference with immutable run specification."""

import asyncio
import importlib.metadata
import itertools
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from human_agency_evals.conversations.models import make_model
from human_agency_evals.conversations.state import render_turn
from human_agency_evals.datasets.loader import load
from human_agency_evals.datasets.pairing import pairs
from human_agency_evals.datasets.validator import validate
from human_agency_evals.interventions.loader import load_interventions
from human_agency_evals.io import atomic_json, digest, read_json, read_yaml
from human_agency_evals.schemas.experiment import Experiment
from human_agency_evals.schemas.transcript import Generation, Message, Transcript


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare(config_path: str):
    config = Experiment.model_validate(read_yaml(config_path))
    scenarios = load(config.data)
    validate(scenarios)
    if config.base_limit:
        by_domain = [
            sorted({s.metadata.base_id for s in scenarios if s.domain == domain})
            for domain in ("epistemic", "values", "actions")
        ]
        base_ids = [
            base for row in itertools.zip_longest(*by_domain) for base in row if base is not None
        ][: config.base_limit]
        scenarios = [s for s in scenarios if s.metadata.base_id in base_ids]
    if config.condition_limit:
        pair_ids = sorted(pairs(scenarios))[: config.condition_limit]
        scenarios = [s for s in scenarios if s.metadata.pair_id in pair_ids]
    if config.condition_filters:
        scenarios = [
            s
            for s in scenarios
            if all(
                getattr(s.manipulation, key) in values
                for key, values in config.condition_filters.items()
            )
        ]
        if not scenarios:
            raise ValueError("condition filters selected no scenarios")
        validate(scenarios)
    prompts = load_interventions(config.intervention_file)
    for name in config.interventions:
        if name not in prompts:
            raise ValueError(f"unknown intervention: {name}")
    return config, scenarios, {k: prompts[k] for k in config.interventions}


def estimate(config, scenarios, prompts):
    conversations = len(scenarios) * len(config.models) * len(prompts) * config.repetitions
    calls = (
        sum(len(s.conversation) for s in scenarios)
        * len(config.models)
        * len(prompts)
        * config.repetitions
    )
    target_input = target_output = 0
    cost = 0.0
    known = True
    for s, model, prompt in itertools.product(scenarios, config.models, prompts.values()):
        history = len(prompt) / 4
        for turn in s.conversation:
            history += len(render_turn(s, turn.index)) / 4 + 8
            target_input += int(history) * config.repetitions
            target_output += model.max_tokens * config.repetitions
            if model.name.startswith("mock/"):
                pass
            elif model.input_per_million is None or model.output_per_million is None:
                known = False
            else:
                cost += (
                    config.repetitions
                    * (
                        history * model.input_per_million
                        + model.max_tokens * model.output_per_million
                    )
                    / 1e6
                )
            history += model.max_tokens
    judge_calls = calls * len(config.graders.models) * config.graders.repeats
    # Full available transcript prefix plus rubric/schema overhead, rough character/token heuristic.
    judge_input = int(
        (target_input + target_output + calls * 2500)
        * len(config.graders.models)
        * config.graders.repeats
    )
    judge_output = calls * config.graders.repeats * sum(g.max_tokens for g in config.graders.models)
    for g in config.graders.models:
        if g.name.startswith("mock/"):
            continue
        if g.input_per_million is None or g.output_per_million is None:
            known = False
        else:
            cost += (
                judge_input / len(config.graders.models) * g.input_per_million
                + calls * config.graders.repeats * g.max_tokens * g.output_per_million
            ) / 1e6
    return {
        "conversations": conversations,
        "target_calls": calls,
        "judge_calls": judge_calls,
        "expected_prompt_tokens": target_input,
        "completion_token_budget": target_output,
        "grader_prompt_tokens": judge_input,
        "grader_completion_token_budget": judge_output,
        "estimated_total_usd": cost if known else None,
        "assumption": "rough 4 characters/token; completion budgets; excludes retries; unknown price is null",
    }


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    return digest({str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*.py"))})


async def run(config_path: str, run_id: str | None = None) -> Path:
    config, scenarios, prompts = prepare(config_path)
    spec = {
        "config": config.model_dump(),
        "scenarios": [s.model_dump() for s in scenarios],
        "prompts": prompts,
        "source_hash": source_fingerprint(),
    }
    fingerprint = digest(spec)
    run_id = run_id or f"{config.name}-{fingerprint[:12]}"
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run ID must be a single directory name")
    folder = Path(config.output_dir) / run_id
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / ".lock"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError as exc:
        raise ValueError("run locked; verify no worker is active before removing .lock") from exc
    try:
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest["fingerprint"] != fingerprint:
                raise ValueError("resume specification changed; use a new run ID")
        else:
            git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
            manifest = {
                **spec,
                "fingerprint": fingerprint,
                "created_at": now(),
                "git_commit": git.stdout.strip() or None,
                "python": platform.python_version(),
                "packages": {
                    p: importlib.metadata.version(p)
                    for p in (
                        "autonomy-evals",
                        "inspect-ai",
                        "pydantic",
                        "numpy",
                        "pandas",
                        "scipy",
                        "statsmodels",
                    )
                },
                "test_data": all(m.name.startswith("mock/") for m in config.models),
                "estimate": estimate(config, scenarios, prompts),
            }
            atomic_json(manifest_path, manifest)
        for s, model_spec, arm, repetition in itertools.product(
            scenarios, config.models, prompts, range(config.repetitions)
        ):
            tid = digest([s.scenario_id, model_spec.model_dump(), arm, repetition, config.seed])[
                :24
            ]
            path = folder / "transcripts" / f"{tid}.json"
            if path.exists():
                transcript = Transcript.model_validate(read_json(path))
                if transcript.status == "complete":
                    continue
            else:
                transcript = Transcript(
                    transcript_id=tid,
                    scenario=s,
                    model=model_spec.name,
                    model_config_record=model_spec.model_dump(),
                    intervention=arm,
                    repetition=repetition,
                    seed=config.seed + repetition,
                    system_prompt=prompts[arm],
                    test_data=model_spec.name.startswith("mock/"),
                    messages=[Message(role="system", content=prompts[arm])],
                )
            model = make_model(model_spec)
            transcript.status = "pending"
            for index in range(len(transcript.generations) + 1, len(s.conversation) + 1):
                messages = transcript.messages + [
                    Message(role="user", content=render_turn(s, index))
                ]
                for attempt in range(config.retries):
                    started = time.perf_counter()
                    try:
                        response = await model.generate(messages, transcript.seed + index)
                        if not response.text.strip():
                            raise ValueError("empty model completion")
                        cost = 0.0 if transcript.test_data else None
                        if (
                            model_spec.input_per_million is not None
                            and model_spec.output_per_million is not None
                            and response.input_tokens is not None
                            and response.output_tokens is not None
                        ):
                            cost = (
                                response.input_tokens * model_spec.input_per_million
                                + response.output_tokens * model_spec.output_per_million
                            ) / 1e6
                        transcript.generations.append(
                            Generation(
                                turn=index,
                                text=response.text,
                                latency=time.perf_counter() - started,
                                timestamp=now(),
                                input_tokens=response.input_tokens,
                                output_tokens=response.output_tokens,
                                cost_usd=cost,
                                raw=response.raw,
                            )
                        )
                        transcript.messages = messages + [
                            Message(role="assistant", content=response.text)
                        ]
                        atomic_json(path, transcript.model_dump())
                        break
                    except Exception as exc:
                        # Avoid logging provider exceptions that can contain credential-bearing URLs.
                        error = {
                            "turn": index,
                            "attempt": attempt + 1,
                            "type": type(exc).__name__,
                            "at": now(),
                        }
                        transcript.errors.append(error)
                        with (folder / "events.jsonl").open("a") as log:
                            import json

                            log.write(json.dumps({"transcript_id": tid, **error}) + "\n")
                        atomic_json(path, transcript.model_dump())
                        if attempt + 1 < config.retries:
                            await asyncio.sleep(config.retry_backoff * 2**attempt)
                else:
                    transcript.status = "failed"
                    atomic_json(path, transcript.model_dump())
                    break
            if len(transcript.generations) == len(s.conversation):
                transcript.status = "complete"
            atomic_json(path, transcript.model_dump())
        transcripts = [
            Transcript.model_validate(read_json(p))
            for p in sorted((folder / "transcripts").glob("*.json"))
        ]
        generations = [g for t in transcripts for g in t.generations]
        atomic_json(
            folder / "cost.json",
            {
                "estimate": manifest["estimate"],
                "completed": sum(t.status == "complete" for t in transcripts),
                "failed": sum(t.status == "failed" for t in transcripts),
                "input_tokens": sum(g.input_tokens or 0 for g in generations),
                "output_tokens": sum(g.output_tokens or 0 for g in generations),
                "unknown_usage_generations": sum(
                    g.input_tokens is None or g.output_tokens is None for g in generations
                ),
                "known_target_cost_usd": sum(g.cost_usd or 0 for g in generations),
                "unknown_cost_generations": sum(g.cost_usd is None for g in generations),
                "note": "Failed request billing may be unavailable; judge usage recorded separately.",
            },
        )
        with (folder / "events.jsonl").open("a") as log:
            import json

            log.write(
                json.dumps(
                    {
                        "event": "inference_finished",
                        "at": now(),
                        "completed": sum(t.status == "complete" for t in transcripts),
                    }
                )
                + "\n"
            )
        return folder
    finally:
        lock.unlink(missing_ok=True)
