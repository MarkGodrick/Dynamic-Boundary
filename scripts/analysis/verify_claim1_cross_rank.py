"""Cross-subset rank consistency check for Claim 1.

Split the 600 ECs into 6 subsets by source_model. On each subset, compute the
mean p_hat (and equivalently per-subset θ) for each of the 6 panel models, and
check whether the model ranking is stable across subsets.

If Rasch's assumption holds (one-dimensional difficulty), all 6 subsets should
yield approximately the same model ranking.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difficulty.rasch import load_response_matrix


DATA_DIR = "results/constrained_completion/validation/edge_case"
DATASET = "benchmarks/constrained_completion/edge_case_dataset.jsonl"


def main():
    # Source map
    source_map = {}
    with open(DATASET) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                source_map[e["prompt_id"]] = e["source_model"]

    rm = load_response_matrix(DATA_DIR)
    models = rm["models"]               # sorted list, 6 models
    items = rm["items"]                 # 600 prompt_ids
    p_hat = rm["p_hat"]                 # (6, 600)

    # Group items by source_model
    src_models = sorted(set(source_map.values()))
    item_groups = {sm: [j for j, p in enumerate(items) if source_map[p] == sm]
                   for sm in src_models}

    # Compute mean p_hat per (eval_model, source_group)
    M = len(models)
    G = len(src_models)
    mean_p = np.zeros((M, G))
    for j, sm in enumerate(src_models):
        idx = item_groups[sm]
        mean_p[:, j] = p_hat[:, idx].mean(axis=1)

    # Print the (model × group) mean p_hat matrix
    print("=" * 100)
    print("  Mean p_hat: rows = evaluating model, columns = EC source group")
    print("=" * 100)
    short = lambda s: s.split("/")[-1][:24]
    header = "  " + " " * 26 + " | " + " | ".join(f"{short(s):>12}" for s in src_models)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, m in enumerate(models):
        row = f"  {short(m):>26} | " + " | ".join(f"{mean_p[i, j]:>12.3f}" for j in range(G))
        print(row)

    # Per-group ranking (1 = strongest, M = weakest)
    print("\n" + "=" * 100)
    print("  Per-group model ranking (1 = highest mean p_hat = strongest)")
    print("=" * 100)
    ranks = np.zeros((M, G), dtype=int)
    for j in range(G):
        order = np.argsort(-mean_p[:, j])     # descending
        for r, i in enumerate(order):
            ranks[i, j] = r + 1
    header = "  " + " " * 26 + " | " + " | ".join(f"{short(s):>12}" for s in src_models)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, m in enumerate(models):
        row = f"  {short(m):>26} | " + " | ".join(f"{ranks[i, j]:>12d}" for j in range(G))
        print(row)

    # Pairwise Spearman ρ between subset rankings
    print("\n" + "=" * 100)
    print("  Pairwise Spearman ρ between subset rankings (G × G)")
    print("=" * 100)
    rho_mat = np.zeros((G, G))
    for a in range(G):
        for b in range(G):
            rho_mat[a, b] = spearmanr(ranks[:, a], ranks[:, b])[0]
    header = "  " + " " * 26 + " | " + " | ".join(f"{short(s):>12}" for s in src_models)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for a in range(G):
        row = f"  {short(src_models[a]):>26} | " + " | ".join(f"{rho_mat[a, b]:>12.3f}" for b in range(G))
        print(row)

    # Average off-diagonal ρ
    off_diag = rho_mat[~np.eye(G, dtype=bool)]
    print(f"\n  Mean off-diagonal Spearman ρ = {off_diag.mean():.4f}")
    print(f"  Min  off-diagonal Spearman ρ = {off_diag.min():.4f}")

    # Kendall's W (concordance) — single number for "do all G subsets agree?"
    # W = 12·S / (G²(M³ − M)), where S = Σ(R_i − mean R)²
    R_sum = ranks.sum(axis=1)
    S = ((R_sum - R_sum.mean()) ** 2).sum()
    W = 12 * S / (G ** 2 * (M ** 3 - M))
    print(f"\n  Kendall's W (concordance across {G} groups) = {W:.4f}")
    print(f"    W = 1.0 → perfect agreement; W = 0 → no agreement")


if __name__ == "__main__":
    main()
