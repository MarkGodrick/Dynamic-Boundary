"""Claim 1 diagnostic: is the |Delta beta| LOO tail rank-order-benign?

Examines the location, local spacing, and rank effects of items with large
leave-one-out difficulty displacement. Reports three diagnostics per category
for the pooled M=9 LOO replicates and the single lowest-rho replicate:

  Q1  Are large-|Delta beta| items located in the TAILS of the beta_full
      difficulty distribution? -> Spearman rho(|Delta beta|, |beta_full -
      median|), plus the mean |beta_full| z-score of the top-5% |Delta beta|
      items vs the overall mean.

  Q2  Do large-|Delta beta| items sit in SPARSE regions (large nearest-
      neighbor gap in beta_full)? -> ratio of the mean nearest-neighbor gap
      of the top-5% |Delta beta| items to the global median gap.

  Q3  Does the rank churn (Spearman rho < 1) COME FROM the tail? -> recompute
      rho after removing the top-k highest-|Delta beta| items, for k growing;
      report how fast rho recovers toward 1.0. If churn is tail-driven, rho
      jumps to ~1 after dropping a small fraction.

Uses the ALO[0.20, 0.80] anchor filter (the paper's anchor filter). Reuses
difficulty.rasch (core lib) for loading + fitting; LOO alignment mirrors
scripts/analysis/claim1_calibration_stability.py.

Usage:
    conda run -n boundary --no-capture-output python \\
        scripts/analysis/claim1_loo_tail_diagnostic.py \\
        --output results/analysis/claim1_loo_tail_diagnostic.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difficulty.rasch import load_response_matrix, fit_ols, fit_mle


CATEGORIES = {
    "harmful_refusal":        "filter_ablation_raw_v2_judgefix",
    "over_refusal":           "filter_ablation_raw_v2_judgefix",
    "constrained_completion": "filter_ablation_raw_v2",
    "sycophancy_resistance":  "filter_ablation_raw_v2",
}

ALO_LO, ALO_HI = 0.20, 0.80
TOP_FRAC = 0.05          # "tail" := top 5% by |Delta beta|


def alo_filter_mask(p_hat: np.ndarray,
                    lo: float = ALO_LO, hi: float = ALO_HI) -> np.ndarray:
    """AT-LEAST-ONE panel model has per-cell pass rate inside [lo, hi]."""
    return ((p_hat >= lo) & (p_hat <= hi)).any(axis=0)


def nearest_neighbor_gaps(beta: np.ndarray) -> np.ndarray:
    """Per-item distance to its nearest neighbor along the sorted beta axis."""
    order = np.argsort(beta)
    b_sorted = beta[order]
    left = np.full_like(b_sorted, np.inf)
    right = np.full_like(b_sorted, np.inf)
    left[1:] = b_sorted[1:] - b_sorted[:-1]
    right[:-1] = b_sorted[1:] - b_sorted[:-1]
    gap_sorted = np.minimum(left, right)
    gap = np.empty_like(gap_sorted)
    gap[order] = gap_sorted
    return gap


def rho_recovery_curve(beta_full: np.ndarray,
                       beta_loo: np.ndarray,
                       abs_dbeta: np.ndarray,
                       ks: list[int]) -> list[dict]:
    """Recompute Spearman rho after dropping the top-k |Delta beta| items."""
    order = np.argsort(-abs_dbeta)      # descending |Delta beta|
    J = len(abs_dbeta)
    out = []
    for k in ks:
        if k >= J - 2:
            break
        keep = np.sort(order[k:])       # drop the k largest-displacement items
        rho, _ = spearmanr(beta_full[keep], beta_loo[keep])
        out.append({"dropped": k,
                    "dropped_frac": round(k / J, 4),
                    "rho": float(rho)})
    return out


def diagnose_one_loo(beta_full: np.ndarray, beta_loo_aligned: np.ndarray) -> dict:
    abs_dbeta = np.abs(beta_loo_aligned - beta_full)
    J = len(abs_dbeta)

    # Q1: are big-displacement items in the beta_full tails?
    beta_center = beta_full - np.median(beta_full)
    beta_z = beta_center / (beta_full.std() + 1e-12)
    rho_disp_extremity, _ = spearmanr(abs_dbeta, np.abs(beta_center))
    n_top = max(1, int(round(TOP_FRAC * J)))
    top_idx = np.argsort(-abs_dbeta)[:n_top]
    q1 = {
        "spearman_disp_vs_extremity": float(rho_disp_extremity),
        "top_frac": TOP_FRAC,
        "n_top": int(n_top),
        "top_mean_abs_beta_z": float(np.abs(beta_z[top_idx]).mean()),
        "overall_mean_abs_beta_z": float(np.abs(beta_z).mean()),
    }

    # Q2: are big-displacement items in sparse (large-gap) regions?
    gaps = nearest_neighbor_gaps(beta_full)
    global_med_gap = float(np.median(gaps))
    q2 = {
        "global_median_nn_gap": global_med_gap,
        "top_mean_nn_gap": float(gaps[top_idx].mean()),
        "top_to_global_gap_ratio": float(gaps[top_idx].mean() / (global_med_gap + 1e-12)),
    }

    # Q3: does rho recover once the tail is dropped?
    ks = [0, 1, 2, 3, 5, 8, 12, 20, max(1, int(0.05 * J))]
    ks = sorted(set(k for k in ks if k < J - 2))
    q3 = rho_recovery_curve(beta_full, beta_loo_aligned, abs_dbeta, ks)

    return {"Q1_tail_extremity": q1, "Q2_tail_sparsity": q2,
            "Q3_rho_recovery": q3}


def panel_loo_diag(category: str, response_dir: str) -> dict:
    rm = load_response_matrix(response_dir)
    pc_all, Nt_all, p_all, models = (
        rm["pass_counts"], rm["total_trials"], rm["p_hat"], rm["models"])
    M = pc_all.shape[0]

    mask = alo_filter_mask(p_all)
    pc, Nt, p = pc_all[:, mask], Nt_all[:, mask], p_all[:, mask]
    J = int(mask.sum())

    th_init, b_init = fit_ols(p)
    theta_full, beta_full, _ = fit_mle(pc, Nt, th_init, b_init)

    per_loo = []
    for m_drop in range(M):
        keep = [i for i in range(M) if i != m_drop]
        th_i, b_i = fit_ols(p[keep])
        theta_loo, beta_loo, _ = fit_mle(pc[keep], Nt[keep], th_i, b_i)
        shift = float(theta_full[keep].mean() - theta_loo.mean())
        beta_loo_aligned = beta_loo + shift
        abs_dbeta = np.abs(beta_loo_aligned - beta_full)
        rho, _ = spearmanr(beta_full, beta_loo_aligned)
        d = diagnose_one_loo(beta_full, beta_loo_aligned)
        d.update({"held_out": models[m_drop], "rho": float(rho),
                  "median_abs_dbeta": float(np.median(abs_dbeta)),
                  "max_abs_dbeta": float(abs_dbeta.max())})
        per_loo.append(d)

    # Aggregate + spotlight the worst-rho LOO (the one that stresses the claim)
    worst = min(per_loo, key=lambda r: r["rho"])
    agg = {
        "spearman_disp_vs_extremity_mean":
            float(np.mean([r["Q1_tail_extremity"]["spearman_disp_vs_extremity"]
                           for r in per_loo])),
        "top_to_global_gap_ratio_mean":
            float(np.mean([r["Q2_tail_sparsity"]["top_to_global_gap_ratio"]
                           for r in per_loo])),
        "top_mean_abs_beta_z_mean":
            float(np.mean([r["Q1_tail_extremity"]["top_mean_abs_beta_z"]
                           for r in per_loo])),
        "overall_mean_abs_beta_z":
            float(per_loo[0]["Q1_tail_extremity"]["overall_mean_abs_beta_z"]),
    }
    return {"category": category, "n_panel": M, "n_anchor_alo": J,
            "rho_min": float(worst["rho"]), "worst_held_out": worst["held_out"],
            "aggregate": agg, "worst_loo": worst, "per_loo": per_loo}


def print_summary(s: dict) -> None:
    a = s["aggregate"]
    print(f"\n=== {s['category']}  (J_anchor_ALO={s['n_anchor_alo']}, "
          f"rho_min={s['rho_min']:.4f} @ drop {s['worst_held_out']}) ===")
    print(f"  Q1 disp-vs-extremity rho (mean over LOO): "
          f"{a['spearman_disp_vs_extremity_mean']:+.3f}   "
          f"(>0 => big displacement concentrates in beta tails)")
    print(f"     top-5% |dbeta| mean |beta z| = {a['top_mean_abs_beta_z_mean']:.3f}  "
          f"vs overall {a['overall_mean_abs_beta_z']:.3f}")
    print(f"  Q2 tail nn-gap / global-median-gap (mean): "
          f"{a['top_to_global_gap_ratio_mean']:.2f}x   "
          f"(>1 => tail items are in sparse regions)")
    print(f"  Q3 rho recovery on WORST LOO (drop {s['worst_held_out']}):")
    for row in s["worst_loo"]["Q3_rho_recovery"]:
        print(f"       drop top {row['dropped']:>3d} "
              f"({row['dropped_frac']*100:4.1f}%)  ->  rho = {row['rho']:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Claim 1 LOO tail rank diagnostic.")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--output",
                    default="results/analysis/claim1_loo_tail_diagnostic.json")
    ap.add_argument("--categories", nargs="+", default=None)
    args = ap.parse_args()

    cats = args.categories or list(CATEGORIES.keys())
    root = Path(args.results_root)
    summaries = {}
    for cat in cats:
        rdir = root / cat / CATEGORIES[cat]
        if not rdir.exists():
            print(f"  [skip] {cat}: missing {rdir}")
            continue
        s = panel_loo_diag(cat, str(rdir))
        summaries[cat] = s
        print_summary(s)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"filter": f"ALO[{ALO_LO}, {ALO_HI}]", "top_frac": TOP_FRAC,
                   "categories": summaries}, f, indent=2, ensure_ascii=False)
    print(f"\nJSON -> {args.output}")


if __name__ == "__main__":
    main()
