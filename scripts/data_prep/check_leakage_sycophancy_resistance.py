"""V3 leakage check for sycophancy_resistance validation benchmark.

Compares bare pool questions against each validation tier using:
  1. Exact normalized string match
  2. Sentence-embedding cosine similarity (all-MiniLM-L6-v2, threshold 0.85)

Outputs:
  - Per-tier hit counts (exact + semantic)
  - Filtered validation files with overlapping items removed
  - leakage_report.json with full details

Usage:
  python scripts/data_prep/check_leakage_sycophancy_resistance.py
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np


def normalize(text: str) -> str:
    """Normalize text for exact match comparison."""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_questions(path: str, question_key: str = "question") -> list[tuple[int, str]]:
    """Load (index, question_string) pairs from JSONL."""
    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            q = r.get("metadata", {}).get(question_key, "")
            if not q:
                q = r.get("prompt", [{}])[0].get("content", "") if isinstance(r.get("prompt"), list) else ""
            items.append((i, q))
    return items


def run_check(bare_path: str, val_dir: str, output_dir: str,
              embed_threshold: float = 0.85):
    from sentence_transformers import SentenceTransformer

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load bare questions
    bare_qs = load_questions(bare_path)
    bare_texts = [q for _, q in bare_qs]
    bare_norms = {normalize(q) for q in bare_texts}
    print(f"Bare pool: {len(bare_qs)} items, {len(bare_norms)} unique normalized")

    # Encode bare
    print("Encoding bare pool...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    bare_embs = model.encode(bare_texts, normalize_embeddings=True,
                             show_progress_bar=False)

    report = {"bare_size": len(bare_qs), "tiers": {}}

    val_path = Path(val_dir)
    all_kept = []

    for tier_name in ["easy", "medium", "hard"]:
        tier_file = val_path / f"{tier_name}.jsonl"
        if not tier_file.exists():
            print(f"  {tier_name}: skipped (file not found)")
            continue

        # Load tier items
        tier_items = [json.loads(line) for line in open(tier_file)]
        tier_qs = [item["metadata"].get("question", "") for item in tier_items]
        print(f"\n=== {tier_name} tier: {len(tier_items)} items ===")

        # Exact match
        exact_hits = set()
        for i, q in enumerate(tier_qs):
            if normalize(q) in bare_norms:
                exact_hits.add(i)
        print(f"  Exact match hits: {len(exact_hits)}")

        # Embedding similarity
        print(f"  Encoding {tier_name} tier...")
        tier_embs = model.encode(tier_qs, normalize_embeddings=True,
                                 show_progress_bar=False)
        # Max cosine similarity of each val item against all bare items
        # Process in chunks to avoid memory issues
        embed_hits = set()
        chunk_size = 500
        for start in range(0, len(tier_embs), chunk_size):
            end = min(start + chunk_size, len(tier_embs))
            sims = tier_embs[start:end] @ bare_embs.T
            max_sims = sims.max(axis=1)
            for j, sim in enumerate(max_sims):
                if sim >= embed_threshold:
                    embed_hits.add(start + j)

        semantic_only = embed_hits - exact_hits
        all_hits = exact_hits | embed_hits
        print(f"  Embed (>={embed_threshold}) hits: {len(embed_hits)} "
              f"({len(semantic_only)} semantic-only)")
        print(f"  Total hits (union): {len(all_hits)}")

        # Filter
        kept = [item for i, item in enumerate(tier_items) if i not in all_hits]
        dropped = [item for i, item in enumerate(tier_items) if i in all_hits]
        print(f"  Kept: {len(kept)}, Dropped: {len(dropped)}")

        # Write filtered tier
        filtered_path = val_path / f"{tier_name}.jsonl"
        with open(filtered_path, "w") as f:
            for item in kept:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        all_kept.extend(kept)

        # Report
        report["tiers"][tier_name] = {
            "raw_size": len(tier_items),
            "exact_hits": len(exact_hits),
            "embed_hits": len(embed_hits),
            "semantic_only_hits": len(semantic_only),
            "total_hits": len(all_hits),
            "kept": len(kept),
            "dropped": len(dropped),
            "dropped_questions": [tier_qs[i][:200] for i in sorted(all_hits)[:20]],
        }

    # Write filtered all.jsonl
    for i, item in enumerate(all_kept):
        item["global_id"] = f"val-d-{i:05d}"
    all_path = val_path / "all.jsonl"
    with open(all_path, "w") as f:
        for item in all_kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    report["total_kept"] = len(all_kept)
    report["total_dropped"] = sum(t["dropped"] for t in report["tiers"].values())

    # Write report
    report_path = out_dir / "leakage_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n=== Summary ===")
    print(f"Total validation: {report['total_kept']} kept, "
          f"{report['total_dropped']} dropped")
    print(f"Report: {report_path}")
    print(f"Filtered all.jsonl: {all_path} ({len(all_kept)} items)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bare",
                        default="data/sycophancy_resistance/bare_requests.jsonl")
    parser.add_argument("--val-dir",
                        default="data/sycophancy_resistance/validation")
    parser.add_argument("--output-dir",
                        default="results/data_prep/sycophancy_resistance")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    run_check(args.bare, args.val_dir, args.output_dir, args.threshold)


if __name__ == "__main__":
    main()
