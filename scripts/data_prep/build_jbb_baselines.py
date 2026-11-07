"""Build JailbreakBench easy and hard baseline datasets.

Easy: JBB-Behaviors direct harmful requests (no attack).
Hard: JBB artifacts PAIR attack prompts (successful jailbreaks only).

Output format matches edge_case_dataset.jsonl for direct comparison.

Usage:
    python scripts/build_jbb_baselines.py \
        --output-dir benchmarks/harmful_refusal

Requires: pip install datasets  (in boundary env)
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path


JBB_BEHAVIORS_URL = (
    "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors"
    "/resolve/main/data/harmful-behaviors.csv"
)

# PAIR black-box attack on GPT-3.5 — well-studied, high ASR
PAIR_ARTIFACT_URL = (
    "https://raw.githubusercontent.com/JailbreakBench/artifacts"
    "/main/attack-artifacts/PAIR/black_box/gpt-3.5-turbo-1106.json"
)


def download_text(url: str) -> str:
    """Download a URL and return text content."""
    print(f"  Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Python"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def build_easy(output_dir: Path) -> None:
    """Build JBB-Easy: direct harmful requests without attack."""
    csv_text = download_text(JBB_BEHAVIORS_URL)

    # Parse CSV manually (avoid pandas dependency)
    lines = csv_text.strip().split("\n")
    header = lines[0].split(",")
    goal_idx = None
    behavior_idx = None
    category_idx = None
    for i, col in enumerate(header):
        col_clean = col.strip().strip('"')
        if col_clean == "Goal":
            goal_idx = i
        elif col_clean == "Behavior":
            behavior_idx = i
        elif col_clean == "Category":
            category_idx = i

    if goal_idx is None:
        raise ValueError(f"Could not find 'Goal' column in header: {header}")

    # CSV may have commas inside quoted fields — use csv module
    import csv
    import io
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    header_row = rows[0]

    # Re-find indices from parsed header
    goal_idx = header_row.index("Goal")
    behavior_idx = header_row.index("Behavior") if "Behavior" in header_row else None
    category_idx = header_row.index("Category") if "Category" in header_row else None

    dataset = []
    for i, row in enumerate(rows[1:], 1):
        entry = {
            "prompt_id": f"jbb_easy_{i:03d}",
            "prompt": row[goal_idx],
            "bare_request": row[goal_idx],  # same as prompt (no attack)
            "skills": [],
            "source_model": None,
            "p_hat": None,
            "round_found": None,
            "category": "harmful_refusal",
            "benchmark": "jbb_easy",
        }
        if behavior_idx is not None:
            entry["behavior"] = row[behavior_idx]
        if category_idx is not None:
            entry["harm_category"] = row[category_idx]
        dataset.append(entry)

    out_path = output_dir / "jbb_easy.jsonl"
    with open(out_path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  JBB-Easy: {len(dataset)} prompts → {out_path}")


def build_hard(output_dir: Path) -> None:
    """Build JBB-Hard: PAIR attack prompts (successful jailbreaks only)."""
    raw = download_text(PAIR_ARTIFACT_URL)
    data = json.loads(raw)

    jailbreaks = data.get("jailbreaks", [])

    dataset = []
    skipped = 0
    for jb in jailbreaks:
        # Skip failed attacks and null prompts
        if not jb.get("jailbroken", False):
            skipped += 1
            continue
        if not jb.get("prompt"):
            skipped += 1
            continue

        idx = jb.get("index", len(dataset))
        entry = {
            "prompt_id": f"jbb_hard_{idx + 1:03d}",
            "prompt": jb["prompt"],
            "bare_request": jb.get("goal", ""),
            "skills": ["PAIR"],
            "source_model": None,
            "p_hat": None,
            "round_found": None,
            "category": "harmful_refusal",
            "benchmark": "jbb_hard",
            "behavior": jb.get("behavior", ""),
            "harm_category": jb.get("category", ""),
        }
        dataset.append(entry)

    out_path = output_dir / "jbb_hard.jsonl"
    with open(out_path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  JBB-Hard: {len(dataset)} prompts (skipped {skipped} failed) → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build JailbreakBench easy/hard baseline datasets",
    )
    parser.add_argument(
        "--output-dir", default="benchmarks/harmful_refusal",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building JBB-Easy (direct requests, no attack)...")
    build_easy(output_dir)

    print("Building JBB-Hard (PAIR attack prompts)...")
    build_hard(output_dir)

    print("\nDone. Benchmarks saved to:", output_dir)


if __name__ == "__main__":
    main()
