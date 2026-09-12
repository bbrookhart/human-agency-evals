"""Stable split assignment at topic level, never at condition level."""

import hashlib


def topic_split(topic: str) -> str:
    bucket = int(hashlib.sha256(topic.encode()).hexdigest()[:8], 16) % 10
    return "development" if bucket < 4 else "pilot" if bucket < 8 else "heldout"
