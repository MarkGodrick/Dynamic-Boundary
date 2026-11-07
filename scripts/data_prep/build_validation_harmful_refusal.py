"""Build the three-tier validation benchmark for harmful_refusal (Pillar 3).

Tiers:
  easy   (994):  JBB-Behaviors filtered to 55 Original items + DoNotAnswer 939
  medium (5694): AIR-Bench 2024 (Stanford CRFM)
  hard   (2000): WildJailbreak eval/adversarial_harmful (Allen AI)

JBB filter rationale: JBB-Behaviors comprises 55 Original + 27 TDC/HarmBench + 18
AdvBench. Including the 45 sourced-from-bare items would create item-level leakage
into our bare pool (HarmBench + AdvBench + StrongREJECT). We keep only the 55
Originals — the JBB Source field carries the provenance, so the filter is exact.

Each entry uses the same schema as bare_requests.jsonl:
    id, prompt, source, source_id, source_version, download_date, source_method,
    category, source_meta
"""

import argparse
import json
from datetime import date
from pathlib import Path

from datasets import load_dataset


DOWNLOAD_DATE = date.today().isoformat()


def load_jbb_originals() -> list[dict]:
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    out = []
    for row in ds:
        if row["Source"] != "Original":
            continue
        out.append({
            "prompt": row["Goal"],
            "source": "jailbreakbench",
            "source_id": int(row["Index"]),
            "source_version": "JailbreakBench/JBB-Behaviors@behaviors/harmful",
            "download_date": DOWNLOAD_DATE,
            "source_method": "human-written",
            "category": row["Category"],
            "source_meta": {
                "behavior": row["Behavior"],
                "original_source_field": row["Source"],
            },
        })
    return out


def load_donotanswer() -> list[dict]:
    ds = load_dataset("LibrAI/do-not-answer", split="train")
    out = []
    for row in ds:
        out.append({
            "prompt": row["question"],
            "source": "donotanswer",
            "source_id": int(row["id"]),
            "source_version": "LibrAI/do-not-answer",
            "download_date": DOWNLOAD_DATE,
            "source_method": "machine-generated",
            "category": row["risk_area"],
            "source_meta": {
                "types_of_harm": row["types_of_harm"],
                "specific_harms": row["specific_harms"],
            },
        })
    return out


def load_airbench() -> list[dict]:
    ds = load_dataset("stanford-crfm/air-bench-2024", split="test")
    out = []
    for i, row in enumerate(ds):
        out.append({
            "prompt": row["prompt"],
            "source": "airbench-2024",
            "source_id": i,
            "source_version": "stanford-crfm/air-bench-2024@test",
            "download_date": DOWNLOAD_DATE,
            "source_method": "mixed",
            "category": row["l2-name"],
            "source_meta": {
                "cate_idx": row["cate-idx"],
                "l2_name": row["l2-name"],
                "l3_name": row["l3-name"],
                "l4_name": row["l4-name"],
            },
        })
    return out


def load_wildjailbreak_hard() -> list[dict]:
    ds = load_dataset("allenai/wildjailbreak", "eval", split="train")
    out = []
    for i, row in enumerate(ds):
        if row["data_type"] != "adversarial_harmful":
            continue
        out.append({
            "prompt": row["adversarial"],
            "source": "wildjailbreak",
            "source_id": i,
            "source_version": "allenai/wildjailbreak@eval",
            "download_date": DOWNLOAD_DATE,
            "source_method": "machine-generated",
            "category": "adversarial_harmful",
            "source_meta": {
                "label": row["label"],
                "data_type": row["data_type"],
            },
        })
    return out


def write_tier(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, it in enumerate(items):
            it["id"] = i
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"  → {path} ({len(items)} items)")


def main():
    parser = argparse.ArgumentParser(description="Build harmful_refusal validation benchmark")
    parser.add_argument("--out-dir", type=str, default="data/harmful_refusal/validation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    print("Loading JBB originals (filter Source==Original)...")
    jbb = load_jbb_originals()
    print(f"  JBB originals: {len(jbb)}")

    print("Loading DoNotAnswer...")
    dna = load_donotanswer()
    print(f"  DoNotAnswer: {len(dna)}")

    easy = jbb + dna
    print(f"\nEasy tier total: {len(easy)} ({len(jbb)} JBB + {len(dna)} DoNotAnswer)")
    write_tier(easy, out_dir / "easy.jsonl")

    print("\nLoading AIR-Bench 2024...")
    air = load_airbench()
    print(f"  AIR-Bench: {len(air)}")
    write_tier(air, out_dir / "medium.jsonl")

    print("\nLoading WildJailbreak adversarial_harmful (eval split)...")
    wj = load_wildjailbreak_hard()
    print(f"  WildJailbreak hard: {len(wj)}")
    write_tier(wj, out_dir / "hard.jsonl")

    print("\n=== Validation benchmark summary ===")
    print(f"  easy.jsonl   : {len(easy)}")
    print(f"  medium.jsonl : {len(air)}")
    print(f"  hard.jsonl   : {len(wj)}")
    print(f"  total        : {len(easy) + len(air) + len(wj)}")


if __name__ == "__main__":
    main()
