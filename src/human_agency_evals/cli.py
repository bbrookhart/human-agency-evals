import argparse
import asyncio
import json
from pathlib import Path

from human_agency_evals.io import read_yaml


def resolve_run(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        path = Path("runs") / value
    if not (path / "manifest.json").exists():
        raise ValueError(f"run not found: {value}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Human-agency evaluation infrastructure")
    sub = parser.add_subparsers(dest="command", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("paths", nargs="*")
    for command in ("run", "estimate", "preflight"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        if command == "run":
            p.add_argument("--run-id")
    for command in (
        "score",
        "analyze",
        "report",
        "export-results",
        "annotate-export",
        "annotate-import",
    ):
        p = sub.add_parser(command)
        p.add_argument("--run", required=True)
        if command == "score":
            p.add_argument("--graders")
        elif command == "annotate-export":
            p.add_argument("--output", required=True)
            p.add_argument("--limit", type=int, default=100)
        elif command == "annotate-import":
            p.add_argument("--input", required=True)
            p.add_argument("--key", required=True)
            p.add_argument("--annotator", required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            from human_agency_evals.datasets.loader import load
            from human_agency_evals.datasets.validator import validate

            paths = args.paths or [str(p) for p in sorted(Path("data").glob("*/scenarios.jsonl"))]
            print(json.dumps(validate(load(paths)), indent=2))
        elif args.command in ("run", "estimate", "preflight"):
            from human_agency_evals.conversations.runner import estimate, prepare, run

            config, scenarios, prompts = prepare(args.config)
            if args.command == "preflight":
                from human_agency_evals.preflight import preflight

                print(json.dumps(preflight(args.config), indent=2))
            else:
                print(json.dumps(estimate(config, scenarios, prompts), indent=2))
            if args.command == "run":
                from human_agency_evals.scorers.llm_judge import score_run

                folder = asyncio.run(run(args.config, args.run_id))
                score_folder = asyncio.run(score_run(folder))
                from human_agency_evals.io import read_json

                if read_json(score_folder / "usage/cost.json")["failed"]:
                    raise ValueError(
                        "run contains failed judgments; inspect scorer attempts and rescore"
                    )
                print(folder)
                from human_agency_evals.io import read_json

                if read_json(folder / "cost.json")["failed"]:
                    raise ValueError("run contains failed conversations; inspect events and resume")
        else:
            folder = resolve_run(args.run)
            if args.command == "score":
                from human_agency_evals.schemas.experiment import GraderConfig
                from human_agency_evals.scorers.llm_judge import score_run

                graders = (
                    GraderConfig.model_validate(read_yaml(args.graders)) if args.graders else None
                )
                score_folder = asyncio.run(score_run(folder, graders))
                print(score_folder)
                from human_agency_evals.io import read_json

                if read_json(score_folder / "usage/cost.json")["failed"]:
                    raise ValueError("failed judgments retained; inspect scorer attempts and retry")
            elif args.command in ("analyze", "export-results"):
                from human_agency_evals.analysis.pipeline import analyze

                print(analyze(folder))
            elif args.command == "report":
                from human_agency_evals.reporting.report import report

                print(report(folder))
            elif args.command == "annotate-export":
                from human_agency_evals.analysis.annotations import export_annotations

                if args.limit < 1:
                    raise ValueError("limit must be positive")
                print(export_annotations(folder, Path(args.output), args.limit))
            elif args.command == "annotate-import":
                from human_agency_evals.analysis.annotations import import_annotations

                print(import_annotations(folder, Path(args.input), Path(args.key), args.annotator))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
