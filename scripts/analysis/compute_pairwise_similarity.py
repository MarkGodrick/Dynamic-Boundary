"""Generic intra-set pairwise embedding similarity (P6 / S5).

Reads a jsonl or json file of items, extracts a text field, computes pairwise
cosine similarity using sentence-transformers, and outputs:
  - histogram PNG
  - high-similarity pair list (sim > threshold) as JSON
  - summary stats (mean / median / fraction above threshold)

Used for both bare request pools (P6) and skill dictionaries (S5).

Usage:
    # Bare pool (P6) — text field is `prompt`
    python scripts/analysis/compute_pairwise_similarity.py \
        --input data/harmful_refusal/bare_requests.jsonl \
        --text-field prompt \
        --out-dir results/data_prep/harmful_refusal/similarity/bare \
        --label "Harmful Refusal — Bare Request Pool"

    # Skill dict (S5) — text field is `description`
    python scripts/analysis/compute_pairwise_similarity.py \
        --input skills/harmful_refusal/skill_dictionary.json \
        --text-field description \
        --out-dir results/data_prep/harmful_refusal/similarity/skill \
        --label "Harmful Refusal — Skill Dictionary"
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_items(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        items = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, list):
        return d
    raise ValueError(f"Unsupported JSON structure in {path}")


def compute_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return np.array(model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--text-field", type=str, required=True,
                        help="Top-level key holding the text to embed")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--label", type=str, default="dataset",
                        help="Display label for plot title")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--max-pairs-recorded", type=int, default=200,
                        help="Max number of high-sim pairs to dump verbatim")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(in_path)
    texts = [it[args.text_field] for it in items]
    n = len(texts)
    print(f"Loaded {n} items from {in_path}")
    print(f"Computing embeddings ({args.model})...")
    embs = compute_embeddings(texts, args.model)

    print("Computing pairwise cosine similarity matrix...")
    sim_matrix = embs @ embs.T
    iu = np.triu_indices(n, k=1)
    sims = sim_matrix[iu]
    n_pairs = len(sims)
    print(f"Total pairs: {n_pairs}")

    mean_sim = float(sims.mean())
    median_sim = float(np.median(sims))
    p95 = float(np.percentile(sims, 95))
    p99 = float(np.percentile(sims, 99))
    above_threshold = int((sims >= args.threshold).sum())
    pct_above = above_threshold / n_pairs * 100

    print(f"  mean    : {mean_sim:.4f}")
    print(f"  median  : {median_sim:.4f}")
    print(f"  p95     : {p95:.4f}")
    print(f"  p99     : {p99:.4f}")
    print(f"  pairs ≥ {args.threshold}: {above_threshold} ({pct_above:.3f}%)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sims, bins=80, edgecolor="black", linewidth=0.4)
    ax.axvline(args.threshold, color="red", linestyle="--",
               label=f"threshold = {args.threshold}")
    ax.set_xlabel("Pairwise cosine similarity")
    ax.set_ylabel("Pair count")
    ax.set_title(f"{args.label}\nintra-set similarity (n={n}, pairs={n_pairs})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "similarity_hist.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir / 'similarity_hist.png'}")

    indices_above = np.where(sims >= args.threshold)[0]
    sorted_idx = indices_above[np.argsort(-sims[indices_above])]
    high_pairs = []
    for k in sorted_idx[:args.max_pairs_recorded]:
        i, j = int(iu[0][k]), int(iu[1][k])
        high_pairs.append({
            "i": i, "j": j,
            "sim": round(float(sims[k]), 4),
            "text_i": texts[i][:200],
            "text_j": texts[j][:200],
        })

    summary = {
        "input": str(in_path),
        "n_items": n,
        "n_pairs": n_pairs,
        "embedding_model": args.model,
        "threshold": args.threshold,
        "mean": round(mean_sim, 4),
        "median": round(median_sim, 4),
        "p95": round(p95, 4),
        "p99": round(p99, 4),
        "pairs_above_threshold": above_threshold,
        "pct_above_threshold": round(pct_above, 4),
        "top_high_sim_pairs": high_pairs,
    }
    with open(out_dir / "similarity_stats.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  → {out_dir / 'similarity_stats.json'}")


if __name__ == "__main__":
    main()
