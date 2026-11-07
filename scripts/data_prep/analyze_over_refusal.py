"""Compute pairwise embedding similarity (P6/S5) and distribution stats
for over_refusal bare pool.

Outputs:
  results/data_prep/over_refusal/similarity/bare/stats.json
  results/data_prep/over_refusal/similarity/bare/histogram.png

Usage:
  conda run --no-capture-output -n boundary python -u \
      scripts/data_prep/analyze_over_refusal.py
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bare",
                        default="data/over_refusal/bare_requests.jsonl")
    parser.add_argument("--output-dir",
                        default="results/data_prep/over_refusal/similarity/bare")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = [json.loads(line) for line in open(args.bare)]
    prompts = [item["prompt"] for item in items]
    print(f"Loaded {len(prompts)} bare prompts")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    embs = np.array(model.encode(prompts, normalize_embeddings=True,
                                  show_progress_bar=True, batch_size=64))

    # Pairwise cosine similarity (upper triangle only)
    print("Computing pairwise similarities...")
    n = len(embs)
    total_pairs = n * (n - 1) // 2
    sims = []
    chunk = 200
    for i in range(0, n, chunk):
        end_i = min(i + chunk, n)
        block = embs[i:end_i] @ embs.T  # (chunk, n)
        for ri, row_idx in enumerate(range(i, end_i)):
            for col_idx in range(row_idx + 1, n):
                sims.append(float(block[ri, col_idx]))

    sims = np.array(sims)
    print(f"Total pairs: {total_pairs} (computed {len(sims)})")

    above_80 = int((sims >= 0.80).sum())
    above_85 = int((sims >= 0.85).sum())
    above_90 = int((sims >= 0.90).sum())

    stats = {
        "n_items": n,
        "n_pairs": total_pairs,
        "mean": round(float(sims.mean()), 4),
        "median": round(float(np.median(sims)), 4),
        "std": round(float(sims.std()), 4),
        "p5": round(float(np.percentile(sims, 5)), 4),
        "p95": round(float(np.percentile(sims, 95)), 4),
        "min": round(float(sims.min()), 4),
        "max": round(float(sims.max()), 4),
        "above_0.80": above_80,
        "above_0.80_pct": round(100.0 * above_80 / total_pairs, 4),
        "above_0.85": above_85,
        "above_0.85_pct": round(100.0 * above_85 / total_pairs, 4),
        "above_0.90": above_90,
        "above_0.90_pct": round(100.0 * above_90 / total_pairs, 4),
    }
    print(f"\nStats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Histogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sims, bins=100, edgecolor="none", alpha=0.8, color="#4C72B0")
    ax.axvline(0.80, color="red", linestyle="--", alpha=0.7, label="0.80 threshold")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Pair Count")
    ax.set_title(f"Over-Refusal Bare Pool: Pairwise Similarity (n={n}, pairs={total_pairs:,})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "histogram.png", dpi=150)
    plt.close()
    print(f"\nSaved stats and histogram to {out_dir}")


if __name__ == "__main__":
    main()
