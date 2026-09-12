"""Surface diagnostics only; they do not establish semantic grounding or sycophancy."""

import re


def features(text: str) -> dict[str, int]:
    return {
        "word_count": len(text.split()),
        "question_marks": text.count("?"),
        "enumerated_items": len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s", text)),
        "certainty_markers": len(
            re.findall(r"\b(?:definitely|certainly|obviously)\b", text.lower())
        ),
    }


def conclusion_shift(a: str, b: str) -> float | None:
    if "unclear" in (a, b):
        return None
    values = {"A": -1, "neither": 0, "B": 1}
    return abs(values[a] - values[b]) / 2
