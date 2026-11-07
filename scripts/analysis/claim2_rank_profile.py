"""Claim 2 appendix: panel-rank profile of EC(m) vs PC items.

For each item x and panel model m, rank m on p_hat(., x) among the M=9
panel models (rank 1 = highest pass rate, rank M = lowest). If SGBS is
genuinely m-anchored, m's rank on EC(m) should track m's panel position:
the strongest panel model gets rank 1 on its own EC items, the weakest
gets rank M.

We report two complementary signals:
  (1) Per-source-model median rank: rank_m_med = median over x in EC(m)
      of rank_m(x). We expect rank_m_med to be MONOTONE in m's panel
      ordering, captured by Spearman rho(panel_theta_order, rank_m_med).
      A strong negative rho (~ -1) is the target-anchoring signal:
      stronger models (low panel_theta_order index when sorted ascending
      by capability => high theta) get LOWER rank_m_med (rank 1).
  (2) Pooled rank distribution on PC items (panel-mean p in [0.45, 0.55]):
      rank of EACH panel model on each PC item, pooled across (m, x).
      Under H0 (PC items have no target preference) this is approximately
      uniform on {1, ..., M}; KS distance ~ 0 confirms PC is a good null.

The two together: PC items have approximately uniform per-model ranks
across the panel (near-zero KS); EC items have ranks anchored to each
m's specific panel position (strong rho).

Statistic (1) is non-parametric (independent of the 1PL link function,
the masked-MLE fit, and the sum-to-zero gauge); statistic (2) is also
non-parametric in this sense.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from claim2_boundary_anchored import (
    PANEL,
    CATEGORY_PATHS,
    FILTER_LOW,
    FILTER_HIGH,
    category_paths,
    load_anchor_matrix,
    load_ec_blocks,
)


PC_LO = 0.45
PC_HI = 0.55


def rank_with_random_tiebreak(p_row: np.ndarray, target_idx: int, rng: np.random.Generator) -> int:
    """Rank target among p_row entries: 1 = highest pass rate, M = lowest.
    Ties broken at random (so the target's expected rank under tied panel pass
    rates is the midpoint, not biased to top or bottom)."""
    M = len(p_row)
    # Add small random perturbation to break ties; target's perturbation is also random
    perturbed = p_row + rng.uniform(-1e-9, 1e-9, size=M)
    # Rank by descending pass rate -> rank 1 has highest p
    order = np.argsort(-perturbed, kind="stable")
    rank = int(np.where(order == target_idx)[0][0]) + 1
    return rank


def ranks_for_set(p_hat_block: np.ndarray, target_indices: list[int], rng: np.random.Generator) -> np.ndarray:
    """p_hat_block: (M, J) panel pass rates. target_indices: length J, panel index of the target for that item.
    Returns array (J,) of ranks of target_idx[j] in column j."""
    J = p_hat_block.shape[1]
    out = np.empty(J, dtype=int)
    for j in range(J):
        out[j] = rank_with_random_tiebreak(p_hat_block[:, j], target_indices[j], rng)
    return out


def ks_to_uniform(ranks: np.ndarray, M: int) -> float:
    """KS distance between empirical rank ECDF and discrete uniform on {1,...,M}."""
    ranks = np.asarray(ranks)
    n = len(ranks)
    if n == 0:
        return float("nan")
    # Empirical CDF at each integer 1..M
    cdf_emp = np.array([(ranks <= k).sum() / n for k in range(1, M + 1)])
    cdf_unif = np.array([k / M for k in range(1, M + 1)])
    return float(np.max(np.abs(cdf_emp - cdf_unif)))


def summary(ranks: np.ndarray, M: int) -> dict:
    if len(ranks) == 0:
        return {"n": 0, "median": float("nan"), "q1": float("nan"), "q3": float("nan"),
                "mid3": float("nan"), "ks_uniform": float("nan")}
    q1, med, q3 = np.percentile(ranks, [25, 50, 75])
    mid_lo, mid_hi = (M // 2), (M // 2 + 2)  # M=9 -> {4,5,6}
    mid3 = float(((ranks >= mid_lo) & (ranks <= mid_hi)).mean())
    return {
        "n": int(len(ranks)),
        "median": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "mid3": mid3,
        "ks_uniform": ks_to_uniform(ranks, M),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=sorted(CATEGORY_PATHS.keys()), required=True)
    parser.add_argument("--seed", type=int, default=20260506)
    args = parser.parse_args()

    paths = category_paths(args.category)
    out_dir = paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = list(PANEL)
    M = len(panel)
    rng = np.random.default_rng(args.seed)

    print(f"Category: {args.category}")
    # Anchor (used to find PC items by panel-mean band)
    anchor_ids, anchor_K, anchor_N = load_anchor_matrix(paths["anchor_dir"], panel)
    p_hat_anchor = np.where(anchor_N > 0, anchor_K / anchor_N, np.nan)

    # EC block
    ec_ids, ec_K, ec_N, source_map = load_ec_blocks(paths["ec_dataset"], paths["ec_val_dir"], panel)
    p_hat_ec = np.where(ec_N > 0, ec_K / ec_N, np.nan)

    # EC ranks: per item, rank the source model among panel
    target_indices = [panel.index(source_map[pid]) for pid in ec_ids]
    ec_ranks = ranks_for_set(p_hat_ec, target_indices, rng)
    ec_sum = summary(ec_ranks, M)
    print(f"  EC pooled (n={ec_sum['n']}): median={ec_sum['median']:.1f}, IQR=[{ec_sum['q1']:.1f},{ec_sum['q3']:.1f}], "
          f"mid-3={ec_sum['mid3']:.3f}, KS={ec_sum['ks_uniform']:.3f}")

    # EC per-source-model breakdown
    per_model_ec = {}
    median_per_m = []
    for m_idx, model in enumerate(panel):
        mask = np.array([t == m_idx for t in target_indices])
        sub_ranks = ec_ranks[mask]
        per_model_ec[model] = summary(sub_ranks, M)
        median_per_m.append(per_model_ec[model]["median"])
    median_per_m = np.array(median_per_m)

    # Per-source rank tracking: load calibrated theta from the masked Claim 2 fit.
    metrics_path = out_dir / "claim2_metrics.json"
    if metrics_path.exists():
        m_d = json.load(open(metrics_path))
        theta = np.array(m_d["theta"])
        if m_d["panel"] != panel:
            raise RuntimeError("panel order mismatch with claim2_metrics.json")
    else:
        theta = None

    if theta is not None:
        rho, p_val = spearmanr(theta, median_per_m)
        # Stronger model (higher theta) => SHOULD have LOWER rank median (rank 1 = top pass rate).
        # So expected rho is NEGATIVE; we report magnitude.
        print(f"  Spearman rho(theta_m, rank_med_m on EC(m)) = {rho:+.3f} (p={p_val:.3g})  [expect strongly negative]")
    else:
        rho, p_val = float("nan"), float("nan")
        print("  (claim2_metrics.json missing; skipping Spearman rho theta vs rank-median)")

    # PC ranks: PC pool is panel-mean in [PC_LO, PC_HI] from the random pool.
    # Since PC items have no associated target, we measure the rank of EACH
    # panel model on each PC item, pooling across (m, x). Under H0 (PC items
    # are panel-centered with no target preference) this should be uniform.
    panel_mean = np.nanmean(p_hat_anchor, axis=0)
    pc_idx = np.where((panel_mean >= PC_LO) & (panel_mean <= PC_HI))[0]
    print(f"  PC pool size: {len(pc_idx)}")
    if len(pc_idx) == 0:
        raise RuntimeError(f"PC pool empty for category {args.category}")
    p_hat_pc = p_hat_anchor[:, pc_idx]
    pc_ranks_all = []
    for j in range(p_hat_pc.shape[1]):
        for m_idx in range(M):
            pc_ranks_all.append(rank_with_random_tiebreak(p_hat_pc[:, j], m_idx, rng))
    pc_ranks_all = np.array(pc_ranks_all)
    pc_sum = summary(pc_ranks_all, M)
    print(f"  PC (n={pc_sum['n']}): median={pc_sum['median']:.1f}, IQR=[{pc_sum['q1']:.1f},{pc_sum['q3']:.1f}], "
          f"mid-3={pc_sum['mid3']:.3f}, KS={pc_sum['ks_uniform']:.3f}")

    out = {
        "category": args.category,
        "M": M,
        "ec_pooled": ec_sum,
        "ec_per_source_model": per_model_ec,
        "ec_rank_median_per_source_model": median_per_m.tolist(),
        "spearman_theta_vs_rank_median": {"rho": float(rho), "p_value": float(p_val)},
        "pc": pc_sum,
        "pc_band": [PC_LO, PC_HI],
        "n_pc_pool": int(len(pc_idx)),
        "panel_order": panel,
    }
    out_path = out_dir / "claim2_rank_profile.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
