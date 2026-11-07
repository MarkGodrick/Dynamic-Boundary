"""V3 leakage check for over_refusal validation benchmark.

Compares bare pool prompts against validation (FalseReject test) using:
  1. Exact normalized string match
  2. Sentence-embedding cosine similarity (all-MiniLM-L6-v2, threshold 0.85)

Since bare and validation come from completely independent sources (XSTest +
OR-Bench vs FalseReject), we expect zero or near-zero leakage.  This script
produces the hard evidence for the paper appendix.

Outputs:
  - results/data_prep/over_refusal/leakage_report.json

Usage:
  conda run --no-capture-output -n boundary python -u \
      scripts/data_prep/check_leakage_over_refusal.py
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bare",
                        default="data/over_refusal/bare_requests.jsonl")
    parser.add_argument("--val",
                        default="data/over_refusal/validation/all.jsonl")
    parser.add_argument("--output-dir",
                        default="results/data_prep/over_refusal")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load bare prompts
    bare_items = [json.loads(line) for line in open(args.bare)]
    bare_prompts = [item["prompt"] for item in bare_items]
    bare_norms = {normalize(p) for p in bare_prompts}
    print(f"Bare pool: {len(bare_items)} items, {len(bare_norms)} unique normalized")

    # Load validation prompts
    val_items = [json.loads(line) for line in open(args.val)]
    val_prompts = [item["prompt"] for item in val_items]
    print(f"Validation: {len(val_items)} items")

    # 1. Exact match
    exact_hits = set()
    for i, p in enumerate(val_prompts):
        if normalize(p) in bare_norms:
            exact_hits.add(i)
    print(f"\nExact match hits: {len(exact_hits)}")

    # 2. Embedding similarity
    from sentence_transformers import SentenceTransformer
    print("Encoding bare pool...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    bare_embs = np.array(model.encode(bare_prompts, normalize_embeddings=True,
                                       show_progress_bar=True))
    print("Encoding validation...")
    val_embs = np.array(model.encode(val_prompts, normalize_embeddings=True,
                                      show_progress_bar=True))

    # Chunk to avoid memory blowup
    embed_hits = set()
    embed_details = []
    chunk_size = 200
    for start in range(0, len(val_embs), chunk_size):
        end = min(start + chunk_size, len(val_embs))
        sims = val_embs[start:end] @ bare_embs.T
        for j in range(end - start):
            max_idx = int(sims[j].argmax())
            max_sim = float(sims[j, max_idx])
            if max_sim >= args.threshold:
                embed_hits.add(start + j)
                embed_details.append({
                    "val_idx": start + j,
                    "val_prompt": val_prompts[start + j][:200],
                    "bare_idx": max_idx,
                    "bare_prompt": bare_prompts[max_idx][:200],
                    "sim": round(max_sim, 4),
                })

    semantic_only = embed_hits - exact_hits
    all_hits = exact_hits | embed_hits
    print(f"Embed (>={args.threshold}) hits: {len(embed_hits)} "
          f"({len(semantic_only)} semantic-only)")
    print(f"Total hits (union): {len(all_hits)}")

    # Filter validation
    kept = [item for i, item in enumerate(val_items) if i not in all_hits]
    dropped = [item for i, item in enumerate(val_items) if i in all_hits]

    # Overwrite filtered validation
    val_path = Path(args.val)
    with open(val_path, "w") as f:
        for i, item in enumerate(kept):
            item["id"] = f"falsereject-{i:04d}"
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nKept: {len(kept)}, Dropped: {len(dropped)}")
    print(f"Filtered validation saved to {val_path}")

    # Report
    report = {
        "bare_size": len(bare_items),
        "val_raw_size": len(val_items),
        "val_kept": len(kept),
        "val_dropped": len(dropped),
        "exact_hits": len(exact_hits),
        "embed_hits": len(embed_hits),
        "semantic_only_hits": len(semantic_only),
        "total_hits": len(all_hits),
        "embed_threshold": args.threshold,
        "embed_model": "all-MiniLM-L6-v2",
        "hit_details": embed_details[:50],
    }
    report_path = out_dir / "leakage_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
