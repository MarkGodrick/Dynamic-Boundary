"""Claim 2 appendix: panel-centered (PC) counterfactual baseline.

For each category we sample K=50 items from the random-composed pool whose
panel-mean pass rate satisfies p_bar in [0.45, 0.55] (the strict panel-boundary
band). PC items are not associated with any target model; they capture "items
hard for the panel as a whole." We add the PC block to the Claim 2 response
matrix as additional non-masked items and refit the same masked-MLE 1PL Rasch
model used in scripts/analysis/claim2_boundary_anchored.py, then compare
beta(PC) against the per-target distributions beta(EC(m)).

Outputs (per category):
  - mean(beta(PC))     : grand-mean PC difficulty under sum-to-zero theta gauge
                         (expected ~ 0 if PC items are panel-centered)
  - mean(beta(EC(.)))  : range across 9 panel models (target-anchored items)
  - std_cross-m EC     : std of mean(beta(EC(m))) across the 9 panel models
                         (tracks panel theta spread if SGBS items track theta_m)
  - std_boot PC        : std across B=100 PC bootstraps of mean(beta(PC))
  - ratio              : std_cross-m EC / std_boot PC
                         (>>1 indicates target-specific signal; ~1 indicates
                          PC-equivalent dispersion)
"""

import argparse
import json
from pathlib import Path

import numpy as np

from claim2_boundary_anchored import (
    PANEL,
    CATEGORY_PATHS,
    PROJECT_ROOT,
    FILTER_LOW,
    FILTER_HIGH,
    category_paths,
    fit_masked_mle,
    load_anchor_matrix,
    load_ec_blocks,
    model_to_dirname,
)


PC_LO = 0.45
PC_HI = 0.55
K_PC = 50
N_BOOT = 100


def select_pc_pool(anchor_K: np.ndarray, anchor_N: np.ndarray, anchor_ids: list[str]) -> tuple[list[int], list[str]]:
    """Indices and ids of pool items with panel-mean p_hat in [PC_LO, PC_HI]."""
    p_hat = np.where(anchor_N > 0, anchor_K / anchor_N, np.nan)
    panel_mean = np.nanmean(p_hat, axis=0)
    keep = (panel_mean >= PC_LO) & (panel_mean <= PC_HI)
    idx = [int(i) for i, k in enumerate(keep) if k]
    return idx, [anchor_ids[i] for i in idx]


def fit_with_pc_block(
    pass_counts: np.ndarray,
    total_trials: np.ndarray,
    fit_mask: np.ndarray,
    pc_K: np.ndarray,
    pc_N: np.ndarray,
    frozen_beta_anchor: np.ndarray | None = None,
    n_anchor: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit Rasch on (anchor || EC || PC) with the same diagonal mask, plus all-1 mask on PC.

    If ``frozen_beta_anchor`` is supplied (length n_anchor), those anchor difficulties
    are held fixed (anchor-locked calibration). Returns (theta, beta_full, beta_pc)
    where beta_full is over [anchor || EC || PC] in that column order.
    """
    M = pass_counts.shape[0]
    K_pc = pc_K.shape[1]
    K_combined = np.concatenate([pass_counts, pc_K], axis=1)
    N_combined = np.concatenate([total_trials, pc_N], axis=1)
    mask_combined = np.concatenate([fit_mask, np.ones((M, K_pc), dtype=int)], axis=1)
    J_total = K_combined.shape[1]
    if frozen_beta_anchor is not None:
        frozen = np.full(J_total, np.nan)
        frozen[:n_anchor] = frozen_beta_anchor
    else:
        frozen = None
    theta, beta, _ = fit_masked_mle(K_combined, N_combined, mask_combined, frozen_beta=frozen)
    J0 = pass_counts.shape[1]
    return theta, beta[:J0], beta[J0:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        choices=sorted(CATEGORY_PATHS.keys()),
        required=True,
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--no-mask-diagonal", action="store_true")
    parser.add_argument(
        "--anchor-locked",
        action="store_true",
        help="anchor-locked calibration: freeze beta_anchor from anchor-only fit, "
             "then fit theta + beta_EC + beta_PC with anchor frozen.",
    )
    args = parser.parse_args()

    paths = category_paths(args.category)
    out_dir = paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = list(PANEL)

    print(f"Category: {args.category}")
    # Load anchor + EC matrices the same way as the main Claim 2 script
    anchor_ids, anchor_K, anchor_N = load_anchor_matrix(paths["anchor_dir"], panel)
    p_hat_anchor = np.where(anchor_N > 0, anchor_K / anchor_N, 0.0)
    keep_filter = (p_hat_anchor.mean(axis=0) >= FILTER_LOW) & (p_hat_anchor.mean(axis=0) <= FILTER_HIGH)
    anchor_K_kept = anchor_K[:, keep_filter]
    anchor_N_kept = anchor_N[:, keep_filter]
    anchor_ids_kept = [pid for pid, k in zip(anchor_ids, keep_filter) if k]
    print(f"  Anchor (filtered): {len(anchor_ids_kept)} items")

    ec_ids, ec_K, ec_N, source_map = load_ec_blocks(paths["ec_dataset"], paths["ec_val_dir"], panel)
    print(f"  EC: {len(ec_ids)} items")

    # PC pool (drawn from the FULL random pool, BEFORE the AVG[0.05, 0.95] filter,
    # so that the PC band [0.45, 0.55] is independent of the anchor-set definition).
    pc_pool_idx, pc_pool_ids = select_pc_pool(anchor_K, anchor_N, anchor_ids)
    print(f"  PC pool size (panel-mean in [{PC_LO}, {PC_HI}]): {len(pc_pool_idx)}")
    if len(pc_pool_idx) == 0:
        raise RuntimeError(
            f"PC pool empty for category {args.category} at band [{PC_LO}, {PC_HI}]; "
            f"widen the band or extend the random pool."
        )

    # Combined matrix (anchor_kept || EC); same diagonal-mask logic
    M = len(panel)
    pass_counts = np.concatenate([anchor_K_kept, ec_K], axis=1)
    total_trials = np.concatenate([anchor_N_kept, ec_N], axis=1)
    J_anchor_kept = anchor_K_kept.shape[1]

    diagonal_mask = np.ones(pass_counts.shape, dtype=int)
    for j_off, pid in enumerate(ec_ids):
        diagonal_mask[panel.index(source_map[pid]), J_anchor_kept + j_off] = 0
    fit_mask = np.ones_like(diagonal_mask) if args.no_mask_diagonal else diagonal_mask

    # Step 1 of anchor-locked: pre-calibrate beta_anchor on anchor-only matrix.
    if args.anchor_locked:
        anchor_only_mask = np.ones((M, J_anchor_kept), dtype=int)
        _, beta_anchor_step1, _ = fit_masked_mle(anchor_K_kept, anchor_N_kept, anchor_only_mask)
        print(f"  Anchor-locked: pre-calibrated beta_anchor on {J_anchor_kept} items")
    else:
        beta_anchor_step1 = None

    rng = np.random.default_rng(args.seed)

    # Bootstrap: B fresh PC samples; for each, fit Rasch on combined+PC and record mean(beta(PC))
    pool_K = anchor_K[:, pc_pool_idx]
    pool_N = anchor_N[:, pc_pool_idx]
    n_pool = len(pc_pool_idx)
    take_K = K_PC
    # Sample with replacement: standard bootstrap. Decouples the spread estimate
    # from the pool size, so categories with narrow PC bands still admit a
    # meaningful baseline (each replicate draws K items i.i.d. from the pool).
    replace = True

    pc_means = []
    pc_thetas = []
    for b in range(args.n_bootstrap):
        sel = rng.choice(n_pool, size=take_K, replace=replace)
        pc_K_b = pool_K[:, sel]
        pc_N_b = pool_N[:, sel]
        theta_b, beta_full, beta_pc = fit_with_pc_block(
            pass_counts, total_trials, fit_mask, pc_K_b, pc_N_b,
            frozen_beta_anchor=beta_anchor_step1, n_anchor=J_anchor_kept,
        )
        pc_means.append(float(beta_pc.mean()))
        pc_thetas.append(theta_b.tolist())

    pc_mean_arr = np.array(pc_means)
    print(f"  PC bootstrap (B={args.n_bootstrap}, K_PC={take_K}):")
    print(f"    grand-mean(beta(PC))  = {pc_mean_arr.mean():+.4f}")
    print(f"    std_boot(mean beta PC) = {pc_mean_arr.std(ddof=1):.4f}")

    # EC anchored: fit once (no PC), compute mean(beta(EC(m))) per panel model
    theta_ec, beta_ec_only, _ = fit_with_pc_block(
        pass_counts, total_trials, fit_mask,
        np.zeros((M, 0)), np.zeros((M, 0)),  # empty PC block
        frozen_beta_anchor=beta_anchor_step1, n_anchor=J_anchor_kept,
    )
    ec_beta = beta_ec_only[J_anchor_kept:]  # length len(ec_ids)
    ec_mean_per_m = []
    for m_idx, model in enumerate(panel):
        idx = [j for j, pid in enumerate(ec_ids) if source_map[pid] == model]
        ec_mean_per_m.append(float(ec_beta[idx].mean()))
    ec_mean_per_m = np.array(ec_mean_per_m)
    ec_range_lo = float(ec_mean_per_m.min())
    ec_range_hi = float(ec_mean_per_m.max())
    ec_std_cross = float(ec_mean_per_m.std(ddof=1))
    print(f"  EC: mean(beta(EC(m))) cross-m: range [{ec_range_lo:+.3f}, {ec_range_hi:+.3f}]; std={ec_std_cross:.4f}")
    if pc_mean_arr.std(ddof=1) > 0:
        ratio = ec_std_cross / pc_mean_arr.std(ddof=1)
    else:
        ratio = float("inf")
    print(f"  Spread ratio std_cross-m(EC) / std_boot(PC) = {ratio:.2f}")

    out = {
        "category": args.category,
        "filter": f"AVG[{FILTER_LOW}, {FILTER_HIGH}]",
        "pc_band": [PC_LO, PC_HI],
        "K_pc": int(take_K),
        "n_pc_pool": int(n_pool),
        "n_bootstrap": int(args.n_bootstrap),
        "n_anchor_kept": int(J_anchor_kept),
        "n_ec": int(len(ec_ids)),
        "panel": panel,
        "ec_mean_beta_per_m": ec_mean_per_m.tolist(),
        "ec_range": [ec_range_lo, ec_range_hi],
        "ec_std_cross_m": ec_std_cross,
        "pc_mean_grand": float(pc_mean_arr.mean()),
        "pc_std_bootstrap": float(pc_mean_arr.std(ddof=1)),
        "spread_ratio": float(ratio),
        "pc_means_each_replicate": pc_means,
        "fit_includes_diagonal": bool(args.no_mask_diagonal),
        "fit_mode": "anchor_locked" if args.anchor_locked else "joint",
    }
    parts = []
    if args.no_mask_diagonal:
        parts.append("with_diagonal")
    if args.anchor_locked:
        parts.append("anchor_locked")
    suffix = "_" + "_".join(parts) if parts else ""
    fname = f"claim2_counterfactual{suffix}.json"
    out_path = out_dir / fname
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
