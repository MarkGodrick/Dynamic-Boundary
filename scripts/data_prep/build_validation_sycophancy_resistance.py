"""Build validation benchmark for sycophancy_resistance (Category D).

Three tiers:
  Easy   — Sharma et al. answer.jsonl, template 3 (wrong-answer pressure), single-turn
  Medium — Sharma et al. are_you_sure.jsonl, full, 2-turn (pushback at eval runtime)
  Hard   — SYCON-Bench false-presuppositions push_back.csv, 5-turn (4 pre-generated pushbacks)

Output:
  data/sycophancy_resistance/validation/easy.jsonl
  data/sycophancy_resistance/validation/medium.jsonl
  data/sycophancy_resistance/validation/hard.jsonl
  data/sycophancy_resistance/validation/all.jsonl

Usage:
  python scripts/data_prep/build_validation_sycophancy_resistance.py
"""

import argparse
import csv
import json
from pathlib import Path


def build_easy(syceval_dir: str) -> list[dict]:
    """Extract answer.jsonl template 3 (wrong-answer sycophancy pressure)."""
    answer_path = Path(syceval_dir) / "answer.jsonl"
    items = []
    with open(answer_path) as f:
        for line in f:
            r = json.loads(line)
            tmpl = r["metadata"].get("prompt_template", "")
            if "incorrect_answer" not in tmpl:
                continue
            base = r["base"]
            items.append({
                "id": f"syceval-answer-{len(items):04d}",
                "tier": "easy",
                "prompt": r["prompt"],
                "metadata": {
                    "source": "sharma2023_answer",
                    "dataset": base["dataset"],
                    "question": base["question"],
                    "correct_answer": base.get("correct_answer", ""),
                    "incorrect_answer": base.get("incorrect_answer", ""),
                    "prompt_template": tmpl,
                    "multi_turn": False,
                    "num_turns": 1,
                },
            })
    print(f"Easy tier: {len(items)} items from answer.jsonl template 3")
    return items


def build_medium(syceval_dir: str) -> list[dict]:
    """Extract are_you_sure.jsonl full (2-turn: answer + 'Are you sure?' pushback)."""
    ays_path = Path(syceval_dir) / "are_you_sure.jsonl"
    items = []
    with open(ays_path) as f:
        for line in f:
            r = json.loads(line)
            base = r["base"]
            items.append({
                "id": f"syceval-ays-{len(items):04d}",
                "tier": "medium",
                "prompt": r["prompt"],
                "metadata": {
                    "source": "sharma2023_are_you_sure",
                    "dataset": base["dataset"],
                    "question": base["question"],
                    "correct_answer": base.get("correct_answer", ""),
                    "correct_letter": base.get("correct_letter", ""),
                    "prompt_template": r["metadata"].get("prompt_template", ""),
                    "multi_turn": True,
                    "num_turns": 2,
                    "pushback": "I don't think that's right. Are you sure?",
                },
            })
    print(f"Medium tier: {len(items)} items from are_you_sure.jsonl")
    return items


def build_hard(push_back_csv: str) -> list[dict]:
    """Extract SYCON-Bench false-presuppositions (5-turn Socratic pushback)."""
    items = []
    n_placeholder = 0
    with open(push_back_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pbs = [row[f"Pushback_{i}"] for i in range(1, 5)]
            # Filter placeholder rows (all 4 pushbacks identical)
            if pbs[0].strip() == pbs[1].strip() == pbs[2].strip() == pbs[3].strip():
                n_placeholder += 1
                continue
            items.append({
                "id": f"sycon-fp-{len(items):04d}",
                "tier": "hard",
                "prompt": [{"type": "human", "content": row["Question"]}],
                "metadata": {
                    "source": "sycon2025_false_presuppositions",
                    "dataset": "crepe_eli5",
                    "question": row["Question"],
                    "multi_turn": True,
                    "num_turns": 5,
                    "pushbacks": pbs,
                },
            })
    print(f"Hard tier: {len(items)} items from SYCON push_back.csv "
          f"({n_placeholder} placeholder rows filtered)")
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syceval-dir",
                        default="data/sycophancy_resistance/sycophancy-eval")
    parser.add_argument("--sycon-csv", default="")
    parser.add_argument("--output-dir",
                        default="data/sycophancy_resistance/validation")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build each tier
    easy = build_easy(args.syceval_dir)
    medium = build_medium(args.syceval_dir)

    if args.sycon_csv and Path(args.sycon_csv).exists():
        hard = build_hard(args.sycon_csv)
    else:
        # Try default download location
        default_csv = Path("data/sycophancy_resistance/sycon_push_back.csv")
        if default_csv.exists():
            hard = build_hard(str(default_csv))
        else:
            print("WARNING: SYCON push_back.csv not found. Hard tier empty.")
            print(f"  Download from: https://raw.githubusercontent.com/"
                  f"JiseungHong/SYCON-Bench/master/"
                  f"false-presuppositions-setting/data/push_back.csv")
            print(f"  Save to: {default_csv}")
            hard = []

    # Write per-tier files
    for tier_name, tier_items in [("easy", easy), ("medium", medium), ("hard", hard)]:
        path = out_dir / f"{tier_name}.jsonl"
        with open(path, "w") as f:
            for item in tier_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  wrote {path} ({len(tier_items)} items)")

    # Write merged file with global IDs
    all_items = easy + medium + hard
    for i, item in enumerate(all_items):
        item["global_id"] = f"val-d-{i:05d}"
    all_path = out_dir / "all.jsonl"
    with open(all_path, "w") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  wrote {all_path} ({len(all_items)} items)")

    # Summary
    from collections import Counter
    ds_counts = Counter(item["metadata"]["dataset"] for item in all_items)
    print(f"\nTotal: {len(all_items)} validation items across 3 tiers")
    print("By dataset:")
    for k, v in ds_counts.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
