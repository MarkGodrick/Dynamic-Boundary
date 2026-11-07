"""Generic IO helpers for dataset and response-matrix JSONL files."""

import json


def load_dataset(path: str) -> list[dict]:
    """Load edge case dataset JSONL."""
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries
