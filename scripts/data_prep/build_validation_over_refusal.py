"""Build validation benchmark for over_refusal (Category A').

Single tier:
  FalseReject test split (Zhang et al. COLM 2025) — 1187 human-annotated
  benign prompts spanning 44 safety-related categories.

V3 satisfied trivially: FalseReject is a completely independent source
from XSTest and OR-Bench (different team, different generation pipeline).

Output:
  data/over_refusal/validation/all.jsonl

Usage:
  conda run --no-capture-output -n boundary python -u \
      scripts/data_prep/build_validation_over_refusal.py
"""

import argparse
import json
from datetime import date
from pathlib import Path

from datasets import load_dataset


DOWNLOAD_DATE = date.today().isoformat()


def build_falsereject_test() -> list[dict]:
    """Load FalseReject test split (human-annotated over-refusal prompts)."""
    ds = load_dataset("AmazonScience/FalseReject", split="test")
    items = []
    # Inspect available columns
    columns = ds.column_names
    print(f"  FalseReject columns: {columns}")
    print(f"  FalseReject test size: {len(ds)}")
    if len(ds) > 0:
        print(f"  Sample row keys: {list(ds[0].keys())}")

    for i, row in enumerate(ds):
        # FalseReject schema: prompt + category (+ possibly others)
        prompt = row.get("prompt", "")
        category_id = row.get("category", -1)
        category_text = row.get("category_text", "")
        items.append({
            "id": f"falsereject-{i:04d}",
            "tier": "single",
            "prompt": prompt,
            "metadata": {
                "source": "falsereject_test",
                "source_version": "AmazonScience/FalseReject (Zhang et al. COLM 2025)",
                "download_date": DOWNLOAD_DATE,
                "category_id": category_id,
                "category": category_text,
                "source_id": i,
            },
        })
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str,
                        default="data/over_refusal/validation")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FalseReject test split...")
    items = build_falsereject_test()
    print(f"  Total validation items: {len(items)}")

    # Category distribution
    from collections import Counter
    cats = Counter(item["metadata"]["category"] for item in items)
    print(f"\n  Category distribution ({len(cats)} categories):")
    for cat, cnt in cats.most_common():
        print(f"    {cat}: {cnt}")

    # Save
    all_path = out_dir / "all.jsonl"
    with open(all_path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(items)} validation items to {all_path}")


if __name__ == "__main__":
    main()
