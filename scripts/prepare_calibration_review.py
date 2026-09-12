"""Export coordinator-facing scenario review forms, excluding heldout cases."""

import csv
from pathlib import Path

from human_agency_evals.conversations.runner import prepare


def main():
    _, scenarios, _ = prepare("configs/experiment_calibration.yaml")
    output = Path("data/human_annotations/scenario_review.csv")
    if output.exists():
        raise ValueError("review form already exists; preserve reviewer input before regenerating")
    cases = {s.metadata.base_id: s for s in scenarios}
    fields = [
        "base_id",
        "domain",
        "context",
        "evidence",
        "desired_behavior",
        "failure_mode",
        "reviewer",
        "factual_coherence",
        "realism",
        "pair_invariance",
        "rubric_applicability",
        "answer_leakage",
        "update_validity",
        "decision",
        "notes",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for base_id, s in sorted(cases.items()):
            writer.writerow(
                {
                    "base_id": base_id,
                    "domain": s.domain,
                    "context": s.context,
                    "evidence": "\n".join(
                        f"Response {e.available_at} ({e.reliability}): {e.text}" for e in s.evidence
                    ),
                    "desired_behavior": s.expected_behavior.desirable,
                    "failure_mode": s.expected_behavior.failure_mode,
                }
            )
    print(output)


if __name__ == "__main__":
    main()
