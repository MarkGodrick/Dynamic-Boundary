"""Empirical check: which filter-necessity argument survives scrutiny, and
which filter candidate is optimal by each argument?

See docs/irt_concepts_primer.md §3.6 for the 5 reasons to filter. This script
runs three concrete experiments on a response matrix, iterating over filter
candidates:

  Test A — β_x estimability (argument #2)
    Fit Rasch on each filter's kept items. Report:
      * SE(β_x) distribution
      * Fraction with SE > τ (unusable β's)
      * |β| distribution (saturation detection)

  Test B — CAT convergence simulation (argument #3)
    For each filter, simulate CAT on the kept items at θ_new ∈ {-1, 0, +1}.
    Report steps to reach SE(θ_new) < 0.2.

  Test C — Leave-one-model-out β stability (Claim 4, argument #5 proxy)
    Fit Rasch on kept items with full panel vs leaving each model out.
    Report median and max |Δβ| across leave-one-out refits.

  Test D — Resolution metrics (primer §3.2 / §3.5, the 5 formal defs)
    For each filter, using full-pool panel θ + filter-kept β̂:
      (a) mean_d12    — pairwise pass-rate diff (intensive)
      (b) SE(θ̂_m)    — mean/max across panel (extensive) + budget_SE
      (c) mean_res    — pairwise resolution |Δθ|/σ_Δ (extensive)
      (d) frac_disj95 — CI disjoint fraction at 95%
      (e) tif_cv      — TIF flatness across panel θ (low = flatter = better)
    Reuses functions from compare_filters.py for consistency.

Arguments #1 (inference cost) and #4 (QC) are not measurable with this data
alone — they require cross-validation across future experiments.

Usage:
    python scripts/analysis/filter_necessity_check.py \\
        --response-dir results/constrained_completion/validation/random_baseline \\
        --output results/constrained_completion/filter_necessity_check.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from difficulty.rasch import load_response_matrix, fit_ols, fit_mle
from compare_filters import (  # noqa: E402
    compute_pairwise_resolution,
    compute_pairwise_pass_rate_diff,
    compute_tif_at_panel,
)


def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


# ---------------------------------------------------------------------------
# Filter candidates
# ---------------------------------------------------------------------------

def filter_by_se_nonparam(p_hat, N, tau):
    """Non-parametric SE filter: keep items with per-item β estimation SE ≤ τ.

    SE(β̂_x) = 1 / sqrt(Σ_m N · p̂_m(1-p̂_m))

    Uses observed p̂ (no Rasch fit). Saturation items (p̂=0 or 1) get
    info=0 → SE→∞ → automatically filtered. This is the "definition B"
    metric: total Fisher info over panel, representing the item's
    informativeness across the θ-range spanned by the panel.
    """
    info = (N * p_hat * (1 - p_hat)).sum(axis=0)  # (J,)
    with np.errstate(divide="ignore"):
        se = np.where(info > 0, 1.0 / np.sqrt(info), np.inf)
    return se <= tau, se


def build_filter_candidates(p_hat, panel_theta, N_avg, tau_equiv_n=40):
    """Return list of (name, mask) tuples for filter candidates.

    Includes AVG (panel-mean proxy) + SE (direct principled) filters.

    τ values are calibrated at N=tau_equiv_n (default 40). When current data
    N_avg differs, τ is rescaled by √(tau_equiv_n / N_avg) to preserve the
    underlying info-budget threshold (Σ_m p(1-p) ≥ 1/(N·τ²)). Example: at
    N=10 with τ_equiv_n=40, τ=0.22 becomes τ=0.44 (same info floor).
    """
    panel_mean = p_hat.mean(axis=0)
    candidates = [
        ("NO FILTER",         np.ones(len(panel_mean), dtype=bool)),
        # AVG panel-mean filters (indirect proxy; N-independent by construction)
        ("AVG [0.05, 0.95]",  (panel_mean >= 0.05) & (panel_mean <= 0.95)),
        ("AVG [0.10, 0.90]",  (panel_mean >= 0.10) & (panel_mean <= 0.90)),
        ("AVG [0.20, 0.80]",  (panel_mean >= 0.20) & (panel_mean <= 0.80)),
        ("AVG [0.30, 0.70]",  (panel_mean >= 0.30) & (panel_mean <= 0.70)),
        ("AVG [0.40, 0.60]",  (panel_mean >= 0.40) & (panel_mean <= 0.60)),
    ]
    # SE filters (direct, non-parametric) — τ rescaled to current N
    tau_scale = float(np.sqrt(tau_equiv_n / N_avg)) if N_avg > 0 else 1.0
    base_taus = [0.13, 0.17, 0.20, 0.22, 0.30, 0.50]
    for tau_base in base_taus:
        tau = tau_base * tau_scale
        mask, _ = filter_by_se_nonparam(p_hat, N_avg, tau)
        label = (f"SE τ={tau_base:.2f}" if tau_scale == 1.0
                 else f"SE τ_eq={tau_base:.2f} (→{tau:.2f}@N={int(N_avg)})")
        candidates.append((label, mask))
    return candidates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_beta_se(beta, theta, N_per_cell):
    """SE(β_x) = 1 / sqrt(Σ_m N · p(1-p))."""
    p = sigmoid(theta[:, None] - beta[None, :])
    info = (N_per_cell * p * (1 - p)).sum(axis=0)
    return np.where(info > 0, 1.0 / np.sqrt(info), np.inf)


def fit_rasch_subset(pass_counts, N_trials, p_hat, item_mask, model_idx=None):
    """Fit Rasch MLE on a subset of items (optionally a subset of models)."""
    if model_idx is None:
        model_idx = np.arange(pass_counts.shape[0])
    pc = pass_counts[np.ix_(model_idx, item_mask)]
    nt = N_trials[np.ix_(model_idx, item_mask)]
    ph = p_hat[np.ix_(model_idx, item_mask)]
    if pc.shape[1] == 0:
        return np.zeros(len(model_idx)), np.zeros(0)
    ti, bi = fit_ols(ph)
    theta, beta, _ = fit_mle(pc, nt, ti, bi)
    return theta, beta


# ---------------------------------------------------------------------------
# Test A: β estimability
# ---------------------------------------------------------------------------

def test_a_one_filter(mask, pass_counts, N_trials, p_hat, tau=0.5):
    """Fit Rasch on kept items and report β estimability stats."""
    n_kept = int(mask.sum())
    if n_kept < 2:
        return {
            "n_kept": n_kept, "fit_ok": False,
            "se_median": np.inf, "se_max": np.inf,
            "pct_se_gt_tau": 100.0, "pct_abs_beta_gt_3": 100.0,
            "beta_range": [0.0, 0.0], "max_abs_beta": 0.0,
        }

    theta, beta = fit_rasch_subset(pass_counts, N_trials, p_hat, mask)
    N_avg = float(np.median(N_trials[:, mask]))
    se = compute_beta_se(beta, theta, N_avg)
    se_finite = se[np.isfinite(se)]

    return {
        "n_kept": n_kept,
        "fit_ok": True,
        "se_median": float(np.median(se)),
        "se_max": float(se_finite.max()) if len(se_finite) > 0 else np.inf,
        "se_p95": float(np.percentile(se_finite, 95)) if len(se_finite) > 0 else np.inf,
        "n_se_gt_tau": int((se > tau).sum()),
        "pct_se_gt_tau": float((se > tau).mean() * 100),
        "n_abs_beta_gt_3": int((np.abs(beta) > 3).sum()),
        "pct_abs_beta_gt_3": float((np.abs(beta) > 3).mean() * 100),
        "beta_range": [float(beta.min()), float(beta.max())],
        "max_abs_beta": float(np.abs(beta).max()),
        "beta_std": float(beta.std()),
    }


# ---------------------------------------------------------------------------
# Test B: CAT convergence simulation
# ---------------------------------------------------------------------------

def cat_simulate_one(beta_bank, N_per_cell, theta_target, se_target=0.2,
                     max_steps=30, rng_seed=42):
    """Simulate CAT on the item bank for a given target θ.

    We generate observations under the Rasch model at θ_target using the
    bank's β values as ground truth (oracle mode). This is an upper bound
    on CAT efficiency — real data adds noise from compose/judge.
    """
    if len(beta_bank) == 0:
        return {"steps": max_steps, "final_se": np.inf, "converged": False}

    rng = np.random.default_rng(rng_seed)
    used = []           # list of item indices picked
    obs = []            # list of k (pass counts)
    theta_est = 0.0     # prior

    for step in range(max_steps):
        # Pick unused item with max Fisher info at current θ_est
        remaining = [i for i in range(len(beta_bank)) if i not in used]
        if not remaining:
            break
        rem = np.array(remaining)
        p_rem = sigmoid(theta_est - beta_bank[rem])
        info_rem = N_per_cell * p_rem * (1 - p_rem)
        pick = int(rem[int(np.argmax(info_rem))])

        # Oracle-generate observation at θ_target
        p_true = float(sigmoid(theta_target - beta_bank[pick]))
        k_obs = int(rng.binomial(int(round(N_per_cell)), p_true))

        used.append(pick)
        obs.append(k_obs)

        # MLE update of θ on all observations so far
        used_beta = beta_bank[np.array(used)]
        obs_arr = np.array(obs)
        N_int = int(round(N_per_cell))

        def nll(th):
            p_ = sigmoid(th - used_beta)
            p_ = np.clip(p_, 1e-9, 1 - 1e-9)
            return -float(np.sum(obs_arr * np.log(p_)
                                 + (N_int - obs_arr) * np.log(1 - p_)))

        res = minimize_scalar(nll, bounds=(-5, 5), method="bounded",
                              options={"xatol": 1e-4})
        theta_est = float(res.x)

        # Fisher info & SE at current θ_est
        p_all = sigmoid(theta_est - used_beta)
        info_total = float((N_per_cell * p_all * (1 - p_all)).sum())
        se_est = float(1.0 / np.sqrt(info_total)) if info_total > 0 else np.inf

        if se_est < se_target:
            return {"steps": step + 1, "final_se": se_est, "converged": True}

    return {"steps": max_steps, "final_se": se_est, "converged": False}


def test_b_one_filter(mask, pass_counts, N_trials, p_hat):
    """CAT convergence at θ_new ∈ {-1, 0, +1} using filter's bank."""
    n_kept = int(mask.sum())
    if n_kept < 3:
        return {"n_kept": n_kept, "steps_by_theta": {}, "mean_steps": np.inf}

    # Fit β on kept items only (this is the bank we'd actually ship)
    _theta, beta = fit_rasch_subset(pass_counts, N_trials, p_hat, mask)
    N_avg = float(np.median(N_trials[:, mask]))

    steps_by_theta = {}
    for tt in [-1.0, 0.0, +1.0]:
        r = cat_simulate_one(beta, N_avg, tt, se_target=0.2, max_steps=30, rng_seed=42)
        steps_by_theta[str(tt)] = r

    return {
        "n_kept": n_kept,
        "steps_by_theta": steps_by_theta,
        "mean_steps": float(np.mean([r["steps"] for r in steps_by_theta.values()])),
    }


# ---------------------------------------------------------------------------
# Test C: leave-one-out β stability
# ---------------------------------------------------------------------------

def test_c_one_filter(mask, pass_counts, N_trials, p_hat):
    """Leave-one-model-out β stability for items in filter's kept set."""
    M = pass_counts.shape[0]
    n_kept = int(mask.sum())
    if n_kept < 2 or M < 3:
        return {"n_kept": n_kept, "median_delta": np.inf,
                "p95_delta": np.inf, "max_delta": np.inf}

    # Full fit on all models + kept items
    _theta_full, beta_full = fit_rasch_subset(pass_counts, N_trials, p_hat, mask)

    # Leave-one-out refits
    deltas = []
    for m_out in range(M):
        remaining = np.array([i for i in range(M) if i != m_out])
        _t, b = fit_rasch_subset(pass_counts, N_trials, p_hat, mask,
                                 model_idx=remaining)
        deltas.append(np.abs(b - beta_full))
    deltas = np.stack(deltas)  # (M, n_kept)

    return {
        "n_kept": n_kept,
        "median_delta": float(np.median(deltas)),
        "max_delta": float(deltas.max()),
        "p95_delta": float(np.percentile(deltas, 95)),
    }


# ---------------------------------------------------------------------------
# Test D: resolution metrics (primer §3.2 / §3.5, 5 formal definitions)
# ---------------------------------------------------------------------------

def test_d_one_filter(mask, beta_full, N_avg, panel_theta, n_ref=200):
    """Resolution metrics for filter-kept items.

    Uses full-pool β̂ and full-pool panel θ̂ (fixed reference). For each
    filter, evaluates the 5 formal resolution definitions from primer §3.5:

      (a) mean_d12   = mean over C(M,2) of (1/|F|) Σ_x |σ(θ_i−β_x) − σ(θ_j−β_x)|
      (b) SE(θ̂_m)  = 1/√(Σ_{x∈F} N·p(1−p)) for each m; report mean / max
      (c) mean_res  = mean over C(M,2) of |Δθ| / √(SE_i² + SE_j²)
      (d) frac_disjoint_95 = frac of pairs with |Δθ| > 1.96·(SE_i + SE_j)
      (e) tif_cv   = std(TIF(θ_m)) / mean(TIF(θ_m)) (lower = flatter = better)

    Budget normalization for extensive metrics (b)(c):
      budget_SE = max(SE) · √(n_kept / n_ref)   (primer §3.6.3)
      budget_res = mean_res / √(n_kept / n_ref)
    Makes filters with different n_kept directly comparable.
    """
    n_kept = int(mask.sum())
    if n_kept < 2:
        return {
            "n_kept": n_kept, "fit_ok": False,
            "mean_d12": 0.0, "mean_res": 0.0,
            "se_theta_mean": np.inf, "se_theta_max": np.inf,
            "budget_se": np.inf, "budget_res": 0.0,
            "frac_disjoint_95": 0.0, "tif_cv": np.nan,
        }

    beta_sub = beta_full[mask]

    # (a) pairwise pass-rate diff (intensive)
    pd = compute_pairwise_pass_rate_diff(beta_sub, panel_theta)

    # (b) + (c) + (d) pairwise resolution metrics (uses filter-kept info at panel θ)
    pr = compute_pairwise_resolution(beta_sub, N_avg, panel_theta)
    se_panel = np.array(pr["se_per_panel"], dtype=float)
    se_panel_finite = se_panel[np.isfinite(se_panel)]

    # (e) TIF at panel
    tif_panel = compute_tif_at_panel(beta_sub, panel_theta) * N_avg
    tif_mean = float(tif_panel.mean()) if len(tif_panel) > 0 else 0.0
    tif_cv = float(tif_panel.std() / tif_mean) if tif_mean > 0 else float("nan")

    # Budget-normalized versions for fair cross-n_kept comparison
    scale = float(np.sqrt(n_kept / n_ref))
    se_max = float(se_panel_finite.max()) if len(se_panel_finite) > 0 else np.inf
    budget_se = se_max * scale
    budget_res = pr["mean_res"] / scale if scale > 0 else 0.0

    return {
        "n_kept": n_kept,
        "fit_ok": True,
        # (a) intensive
        "mean_d12": pd["mean_d"],
        # (b) extensive + budget-normalized
        "se_theta_mean": float(se_panel_finite.mean()) if len(se_panel_finite) else np.inf,
        "se_theta_max": se_max,
        "budget_se": budget_se,
        # (c) extensive + budget-normalized
        "mean_res": pr["mean_res"],
        "budget_res": budget_res,
        # (d) intensive-ish (via z-transform, but CI width scales with SE → weakly extensive)
        "frac_disjoint_95": pr["frac_ci_disjoint_95"],
        # (e) TIF flatness (intensive: ratio of std to mean)
        "tif_mean": tif_mean,
        "tif_cv": tif_cv,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Empirical check across filter candidates."
    )
    parser.add_argument("--response-dir", required=True)
    parser.add_argument("--tau", type=float, default=0.5,
                        help="SE(β) threshold for 'unusable' (default 0.5 logit)")
    parser.add_argument("--n-ref", type=int, default=200,
                        help="Reference n_kept for budget-normalized SE (Test D)")
    parser.add_argument("--tau-equiv-n", type=int, default=40,
                        help="N at which SE τ values are calibrated; rescales "
                             "τ to current N_avg to keep info-budget constant")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rm = load_response_matrix(args.response_dir)
    p_hat = rm["p_hat"]
    pass_counts = rm["pass_counts"]
    N_trials = rm["total_trials"]
    M, J = p_hat.shape
    N_avg = float(np.median(N_trials))

    print("=" * 86)
    print("  Filter Necessity Check — Compare All Filter Candidates")
    print("=" * 86)
    print(f"  Response dir: {args.response_dir}")
    print(f"  Dimensions:   {M} models × {J} items × N={int(N_avg)} trials")
    print(f"  τ (SE unusable threshold): {args.tau}")

    # Panel θ + full-pool β̂ (fixed reference for Test D)
    theta_panel, beta_full = fit_rasch_subset(pass_counts, N_trials, p_hat,
                                              np.ones(J, dtype=bool))

    print(f"\n  Panel θ range: [{theta_panel.min():+.3f}, {theta_panel.max():+.3f}]")
    print(f"  n_ref for budget-SE (Test D): {args.n_ref}")
    if N_avg != args.tau_equiv_n:
        scale = float(np.sqrt(args.tau_equiv_n / N_avg))
        print(f"  τ rescale: base τ (calibrated @ N={args.tau_equiv_n}) × "
              f"√({args.tau_equiv_n}/{int(N_avg)}) = {scale:.3f}")

    candidates = build_filter_candidates(p_hat, theta_panel, N_avg,
                                         tau_equiv_n=args.tau_equiv_n)

    # ---- Evaluate each filter across all 4 tests ----
    print(f"\n  Evaluating {len(candidates)} filter candidates × 4 tests...\n")

    results = []
    for name, mask in candidates:
        print(f"  • {name} (n_kept={int(mask.sum())})")
        r_a = test_a_one_filter(mask, pass_counts, N_trials, p_hat, tau=args.tau)
        r_b = test_b_one_filter(mask, pass_counts, N_trials, p_hat)
        r_c = test_c_one_filter(mask, pass_counts, N_trials, p_hat)
        r_d = test_d_one_filter(mask, beta_full, N_avg, theta_panel,
                                 n_ref=args.n_ref)
        results.append({
            "name": name,
            "n_kept": r_a["n_kept"],
            "test_a": r_a,
            "test_b": r_b,
            "test_c": r_c,
            "test_d": r_d,
        })

    # ---- Print master comparison table ----
    print("\n" + "=" * 86)
    print("  MASTER TABLE")
    print("=" * 86)

    print(f"\n  Test A — β estimability (SE(β) and |β| distribution)\n")
    hdr_a = (f"  {'Filter':<22} {'n_kept':>7} "
             f"{'|β|>3 %':>8} {'|β|max':>7} "
             f"{'SE med':>7} {'SE max':>8} {'SE>τ %':>8}")
    print(hdr_a)
    print("  " + "─" * (len(hdr_a) - 2))
    for r in results:
        ra = r["test_a"]
        print(f"  {r['name']:<22} {r['n_kept']:>7} "
              f"{ra['pct_abs_beta_gt_3']:>7.1f}% {ra['max_abs_beta']:>7.2f} "
              f"{ra['se_median']:>7.3f} {ra['se_max']:>8.3f} "
              f"{ra['pct_se_gt_tau']:>7.1f}%")

    print(f"\n  Test B — CAT convergence (steps to SE(θ)<0.2, oracle mode)\n")
    hdr_b = (f"  {'Filter':<22} {'n_kept':>7} "
             f"{'θ=-1':>6} {'θ=0':>6} {'θ=+1':>6} {'mean':>6}")
    print(hdr_b)
    print("  " + "─" * (len(hdr_b) - 2))
    for r in results:
        rb = r["test_b"]
        if not rb["steps_by_theta"]:
            print(f"  {r['name']:<22} {r['n_kept']:>7}  (too few items)")
            continue
        s_m1 = rb["steps_by_theta"]["-1.0"]["steps"]
        s_0 = rb["steps_by_theta"]["0.0"]["steps"]
        s_p1 = rb["steps_by_theta"]["1.0"]["steps"]
        print(f"  {r['name']:<22} {r['n_kept']:>7} "
              f"{s_m1:>6} {s_0:>6} {s_p1:>6} {rb['mean_steps']:>6.1f}")

    print(f"\n  Test C — Leave-one-out β stability on kept items\n")
    hdr_c = (f"  {'Filter':<22} {'n_kept':>7} "
             f"{'median':>8} {'p95':>8} {'max':>8}")
    print(hdr_c)
    print("  " + "─" * (len(hdr_c) - 2))
    for r in results:
        rc = r["test_c"]
        print(f"  {r['name']:<22} {r['n_kept']:>7} "
              f"{rc['median_delta']:>8.3f} {rc['p95_delta']:>8.3f} {rc['max_delta']:>8.3f}")

    print(f"\n  Test D — Resolution metrics (5 formal defs, primer §3.5)\n")
    print(f"  Legend: (a) mean_d12    pairwise pass-rate diff  [intensive, higher=better]")
    print(f"          (b) SE(θ̂)_max  worst-case SE of panel θ [extensive, lower=better]")
    print(f"          (b') budget_SE  = SE_max · √(n/n_ref)    [intensive, lower=better]")
    print(f"          (c) mean_res    pairwise resolution       [extensive, higher=better]")
    print(f"          (d) disj95%     CI-disjoint fraction      [higher=better]")
    print(f"          (e) tif_cv      TIF flatness across panel [lower=better]\n")
    hdr_d = (f"  {'Filter':<22} {'n_kept':>7} "
             f"{'mean_d12':>9} {'SE_max':>8} {'budgSE':>8} "
             f"{'m_res':>7} {'disj95':>7} {'tif_cv':>7}")
    print(hdr_d)
    print("  " + "─" * (len(hdr_d) - 2))
    for r in results:
        rd = r["test_d"]
        if not rd.get("fit_ok", False):
            print(f"  {r['name']:<22} {r['n_kept']:>7}  (too few items)")
            continue
        print(f"  {r['name']:<22} {r['n_kept']:>7} "
              f"{rd['mean_d12']:>9.3f} {rd['se_theta_max']:>8.3f} "
              f"{rd['budget_se']:>8.3f} "
              f"{rd['mean_res']:>7.2f} {rd['frac_disjoint_95']*100:>6.1f}% "
              f"{rd['tif_cv']:>7.3f}")

    # ---- AVG vs SE filter overlap analysis ----
    print("\n" + "=" * 86)
    print("  AVG vs SE FILTER — OVERLAP ANALYSIS (interpretation B)")
    print("=" * 86)
    print("  If panel were uniform, AVG and SE filters would select identical items.")
    print("  Deviations reveal where SE filter captures panel-density structure.\n")

    # Find AVG and SE candidates, map to their masks
    mask_by_name = {name: m for name, m in candidates}
    avg_names = [n for n in mask_by_name if n.startswith("AVG")]
    se_names = [n for n in mask_by_name if n.startswith("SE")]

    # Jaccard similarity between AVG [0.20, 0.80] and each SE filter
    ref_name = "AVG [0.20, 0.80]"
    ref_mask = mask_by_name[ref_name]
    print(f"  Jaccard similarity and overlap with {ref_name} (n={int(ref_mask.sum())}):\n")
    print(f"  {'Filter':<22} {'n_kept':>7} {'n_intersect':>12} {'n_only_this':>12} "
          f"{'n_only_ref':>11} {'Jaccard':>8}")
    print("  " + "─" * 76)
    def jaccard_row(name, m):
        inter = int((m & ref_mask).sum())
        only_this = int((m & ~ref_mask).sum())
        only_ref = int((~m & ref_mask).sum())
        union = int((m | ref_mask).sum())
        jac = inter / union if union > 0 else 0
        print(f"  {name:<22} {int(m.sum()):>7} {inter:>12} {only_this:>12} "
              f"{only_ref:>11} {jac:>8.3f}")
        return {"n_kept": int(m.sum()), "n_intersect": inter,
                "n_only_this": only_this, "n_only_ref": only_ref, "jaccard": jac}

    overlap_stats = {}
    for name in avg_names + se_names:
        if name == ref_name:
            continue
        overlap_stats[name] = jaccard_row(name, mask_by_name[name])

    # β distribution in panel gap region — SE filter should capture more
    # items in sparse panel-θ zones than AVG filter at same n_kept.
    # Our panel θ: [-1.40, -0.55, +0.08, +0.18, +0.65, +1.05]
    # Largest gap: Mistral-24B (-0.55) to R1-70B (+0.08), ~0.63 logit,
    #              and Gemma3-4B (+0.18) to Gemma3-12B (+0.65), ~0.47.
    # Find β for all items using a full-pool fit
    print("\n  β density in panel θ gap regions (higher = more informative items "
          "in sparse zones):")
    _theta_fp, beta_fp = fit_rasch_subset(pass_counts, N_trials, p_hat,
                                          np.ones(J, dtype=bool))
    # Define gap regions: sorted panel θ, take midpoints as region centers
    theta_sorted = np.sort(theta_panel)
    gaps = []
    for i in range(len(theta_sorted) - 1):
        lo, hi = theta_sorted[i], theta_sorted[i + 1]
        if hi - lo > 0.3:  # non-trivial gap
            gaps.append((lo, hi))

    if gaps:
        print(f"\n  Gap regions (panel θ): " +
              ", ".join(f"[{lo:+.2f}, {hi:+.2f}]" for lo, hi in gaps))
        print(f"\n  {'Filter':<22} {'n_kept':>7} " +
              " ".join(f"{f'β∈[{lo:+.1f},{hi:+.1f}]':>14}" for lo, hi in gaps) +
              f" {'gap %':>8}")
        print("  " + "─" * (30 + 15 * len(gaps) + 10))
        for name, mask in candidates:
            if mask.sum() == 0:
                continue
            kept_beta = beta_fp[mask]
            counts = []
            total_gap = 0
            for lo, hi in gaps:
                c = int(((kept_beta >= lo) & (kept_beta <= hi)).sum())
                counts.append(c)
                total_gap += c
            gap_pct = 100 * total_gap / int(mask.sum())
            print(f"  {name:<22} {int(mask.sum()):>7} " +
                  " ".join(f"{c:>14}" for c in counts) +
                  f" {gap_pct:>7.1f}%")

    # ---- Optimal filter by each criterion ----
    print("\n" + "=" * 86)
    print("  OPTIMAL FILTER BY EACH CRITERION")
    print("=" * 86)

    # A: strictest filter with 0% unusable (filters above this have more items)
    a_acceptable = [r for r in results if r["test_a"]["pct_se_gt_tau"] == 0]
    if a_acceptable:
        a_best = max(a_acceptable, key=lambda r: r["n_kept"])
        print(f"  A (β estimability): loosest filter with 0% unusable β → "
              f"{a_best['name']} (n_kept={a_best['n_kept']})")
    else:
        a_best = min(results, key=lambda r: r["test_a"]["pct_se_gt_tau"])
        print(f"  A (β estimability): lowest unusable fraction → "
              f"{a_best['name']} ({a_best['test_a']['pct_se_gt_tau']:.1f}%)")

    # B: lowest mean steps (but ties expected)
    b_best = min(results, key=lambda r: r["test_b"]["mean_steps"])
    print(f"  B (CAT convergence): lowest mean steps → "
          f"{b_best['name']} ({b_best['test_b']['mean_steps']:.1f} steps)")

    # C: lowest median Δβ
    c_best = min(results, key=lambda r: r["test_c"]["median_delta"])
    print(f"  C (β stability): lowest median |Δβ| → "
          f"{c_best['name']} ({c_best['test_c']['median_delta']:.3f})")

    # D: lowest budget_SE (intensive primary, primer §3.6)
    d_ok = [r for r in results if r["test_d"].get("fit_ok") and
            np.isfinite(r["test_d"]["budget_se"])]
    if d_ok:
        d_best = min(d_ok, key=lambda r: r["test_d"]["budget_se"])
        print(f"  D (resolution, budget_SE): lowest → "
              f"{d_best['name']} ({d_best['test_d']['budget_se']:.3f})")
        # Cross-check intensive (a): mean_d12 winner
        d_d12_best = max(d_ok, key=lambda r: r["test_d"]["mean_d12"])
        if d_d12_best["name"] == d_best["name"]:
            print(f"      (cross-check) mean_d12 winner agrees: ✓")
        else:
            print(f"      (cross-check) mean_d12 winner = {d_d12_best['name']} "
                  f"— intensive metrics disagree")
    else:
        d_best = None

    # ---- Verdict ----
    print("\n" + "=" * 86)
    print("  VERDICT")
    print("=" * 86)
    print("  Is filter necessary?")
    no_filter = next(r for r in results if r["name"] == "NO FILTER")
    if no_filter["test_a"]["pct_se_gt_tau"] > 5:
        print(f"    ✓ YES — NO FILTER has {no_filter['test_a']['pct_se_gt_tau']:.1f}% unusable β's")
    else:
        print(f"    ? NO FILTER only has {no_filter['test_a']['pct_se_gt_tau']:.1f}% unusable β's "
              f"— filter may be optional")

    print("  Which test drives the decision?")
    # Compare spread across filters for each test
    a_range = max(r["test_a"]["pct_se_gt_tau"] for r in results) - \
              min(r["test_a"]["pct_se_gt_tau"] for r in results)
    b_range = max(r["test_b"]["mean_steps"] for r in results) - \
              min(r["test_b"]["mean_steps"] for r in results)
    c_range = max(r["test_c"]["median_delta"] for r in results) - \
              min(r["test_c"]["median_delta"] for r in results)
    print(f"    Test A spread:  {a_range:>6.1f}%  ← fractional unusable β")
    print(f"    Test B spread:  {b_range:>6.1f}   ← CAT steps")
    print(f"    Test C spread:  {c_range:>6.3f}  ← |Δβ| median")
    print(f"    The test with the largest spread is the DISCRIMINATING driver.")

    # Save
    out = {
        "metadata": {
            "response_dir": args.response_dir,
            "n_models": M,
            "n_items": J,
            "n_trials_median": N_avg,
            "tau_unusable_threshold": args.tau,
            "panel_theta_range": [float(theta_panel.min()), float(theta_panel.max())],
        },
        "results": results,
        "best_by_test": {
            "test_a_beta_estimability": a_best["name"],
            "test_b_cat_convergence": b_best["name"],
            "test_c_beta_stability": c_best["name"],
            "test_d_resolution_budget_se": d_best["name"] if d_best else None,
        },
        "test_spread": {
            "a_pct_unusable_range": float(a_range),
            "b_mean_steps_range": float(b_range),
            "c_median_delta_range": float(c_range),
        },
    }
    out_path = args.output or str(
        Path(args.response_dir) / "filter_necessity_check.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON → {out_path}")


if __name__ == "__main__":
    main()
