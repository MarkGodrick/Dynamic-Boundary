"""Diff AVG vs ALO validation subsamples → emit alo_only benchmark + content mapping.

For each category:
  - Read benchmarks/{cat}/validation_sample.jsonl       (AVG, default prompt_id naming)
  - Read benchmarks/{cat}/validation_sample_alo.jsonl   (ALO, regenerated 2026-05-17)
  - Match by prompt TEXT (not prompt_id; the rng-driven stratified sampler under
    different per-tier quotas produces different items at the same prompt_id index)
  - For each ALO item:
      - if its prompt text appears in AVG → "overlap"; record (alo_prompt_id → avg_prompt_id)
      - otherwise → "alo_only"; needs new inference

Outputs:
  benchmarks/{cat}/validation_sample_alo_only.jsonl   (alo_only items, ready for inference)
  benchmarks/{cat}/validation_sample_alo_mapping.json (full per-item provenance:
      overlap → avg_prompt_id, alo_only → null)
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

CATS = ["harmful_refusal", "over_refusal", "constrained_completion", "sycophancy_resistance"]


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def diff_one(cat: str) -> dict:
    avg_items = load_jsonl(ROOT / "benchmarks" / cat / "validation_sample.jsonl")
    alo_items = load_jsonl(ROOT / "benchmarks" / cat / "validation_sample_alo.jsonl")

    avg_by_prompt: dict[str, str] = {it["prompt"]: it["prompt_id"] for it in avg_items}

    overlap: list[dict] = []
    alo_only: list[dict] = []
    mapping: dict[str, str | None] = {}
    for it in alo_items:
        avg_pid = avg_by_prompt.get(it["prompt"])
        if avg_pid is not None:
            overlap.append(it)
            mapping[it["prompt_id"]] = avg_pid
        else:
            alo_only.append(it)
            mapping[it["prompt_id"]] = None

    out_dir = ROOT / "benchmarks" / cat
    out_dir.mkdir(parents=True, exist_ok=True)
    only_path = out_dir / "validation_sample_alo_only.jsonl"
    map_path = out_dir / "validation_sample_alo_mapping.json"

    with open(only_path, "w") as f:
        for it in alo_only:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    payload = {
        "category": cat,
        "avg_source": "benchmarks/{cat}/validation_sample.jsonl".format(cat=cat),
        "alo_source": "benchmarks/{cat}/validation_sample_alo.jsonl".format(cat=cat),
        "n_alo": len(alo_items),
        "n_overlap": len(overlap),
        "n_alo_only": len(alo_only),
        "mapping": mapping,  # alo_prompt_id → avg_prompt_id (null if alo_only)
    }
    with open(map_path, "w") as f:
        json.dump(payload, f, indent=2)

    return {
        "cat": cat,
        "n_alo": len(alo_items),
        "n_overlap": len(overlap),
        "n_alo_only": len(alo_only),
        "out_jsonl": str(only_path.relative_to(ROOT)),
        "out_map": str(map_path.relative_to(ROOT)),
    }


def main() -> None:
    print(f"{'category':<26s} {'|ALO|':>6s} {'overlap':>8s} {'alo_only':>9s}  output")
    print("-" * 100)
    for cat in CATS:
        r = diff_one(cat)
        print(f"{r['cat']:<26s} {r['n_alo']:>6d} {r['n_overlap']:>8d} {r['n_alo_only']:>9d}  "
              f"{r['out_jsonl']}")


if __name__ == "__main__":
    main()
