"""Verify the N-scaling relationship between Binomial and Bernoulli z².

Claim: for a cell (m, x) with observed k-out-of-N trials and model-predicted p,
    z²_bin   = (k - Np)² / [N p (1-p)]              (what we currently report)
    z²_bern  = [k(1-p)/p + (N-k)p/(1-p)] / N        (avg over N Bernoulli trials)

Relationship: z²_bin ≈ 1 + N · (z²_bern - 1)  [in expectation, under model]

Equivalently: z²_bern ≈ 1 + (z²_bin - 1) / N

If this holds empirically, our mean_z² = 15 on Binomial data corresponds to
mean_z² ≈ 1.35 under classical single-trial Bernoulli framing — well within
the conventional [0.5, 1.5] "acceptable fit" range.

Usage:
    python scripts/analysis/verify_n_scaling.py \\
        --response-dir results/constrained_completion/filter_ablation_raw
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difficulty.rasch import load_response_matrix, fit_ols, fit_mle


def sigmoid(x):
    x = np.clip(x, -40.0, 40.0)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def compute_z2_binomial(pass_counts, N, theta, beta):
    """Current-style Binomial standardized residual squared.

    z²_bin[m, x] = (k_mx - N*p_pred)² / (N * p_pred * (1-p_pred))
    """
    diff = theta[:, None] - beta[None, :]
    p = np.clip(sigmoid(diff), 1e-6, 1 - 1e-6)
    expected = N * p
    variance = N * p * (1 - p)
    z = (pass_counts - expected) / np.sqrt(variance)
    return z * z, p


def compute_z2_bernoulli_avg(pass_counts, N, p_pred):
    """Average of Bernoulli per-trial z² across N trials within each cell.

    For cell (m, x) with k observed ones and N-k observed zeros under p_pred:
        z²_per_1 = (1 - p_pred)² / [p_pred (1-p_pred)] = (1 - p_pred) / p_pred
        z²_per_0 = (0 - p_pred)² / [p_pred (1-p_pred)] = p_pred / (1 - p_pred)
        avg z² across N trials = [k * (1-p)/p + (N-k) * p/(1-p)] / N

    Returns (M, J) matrix of cell-level average Bernoulli z².
    """
    k = pass_counts
    p = np.clip(p_pred, 1e-6, 1 - 1e-6)
    z2_per_1 = (1 - p) / p     # contribution of each "1" trial
    z2_per_0 = p / (1 - p)     # contribution of each "0" trial
    total = k * z2_per_1 + (N - k) * z2_per_0
    return total / N


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-dir", required=True)
    parser.add_argument("--filter-lo", type=float, default=None,
                        help="If set, filter items with panel-mean p̂ ≥ lo")
    parser.add_argument("--filter-hi", type=float, default=None,
                        help="If set, filter items with panel-mean p̂ ≤ hi")
    args = parser.parse_args()

    rm = load_response_matrix(args.response_dir)
    pass_counts = rm["pass_counts"]
    N_trials = rm["total_trials"]
    p_hat = rm["p_hat"]
    M, J = pass_counts.shape
    N_med = int(np.median(N_trials))

    print(f"Loaded: M={M}, J={J}, N_median={N_med}")
    print(f"  from {args.response_dir}")

    # Optional filter to edge-case subset
    if args.filter_lo is not None and args.filter_hi is not None:
        panel_mean = p_hat.mean(axis=0)
        mask = (panel_mean >= args.filter_lo) & (panel_mean <= args.filter_hi)
        pass_counts = pass_counts[:, mask]
        N_trials = N_trials[:, mask]
        p_hat = p_hat[:, mask]
        J = pass_counts.shape[1]
        print(f"  filtered to J={J} items with panel_mean ∈ [{args.filter_lo}, {args.filter_hi}]")

    # Fit 1PL Rasch
    theta_init, beta_init = fit_ols(p_hat)
    theta, beta, nll = fit_mle(pass_counts, N_trials, theta_init, beta_init)
    print(f"\n1PL fit: NLL={nll:.1f}")
    print(f"  θ range: [{theta.min():+.3f}, {theta.max():+.3f}]")
    print(f"  β range: [{beta.min():+.3f}, {beta.max():+.3f}]")

    # Compute both flavors of z²
    z2_bin, p_pred = compute_z2_binomial(pass_counts, N_trials, theta, beta)
    z2_bern = compute_z2_bernoulli_avg(pass_counts, N_trials, p_pred)

    mean_bin = z2_bin.mean()
    median_bin = np.median(z2_bin)
    mean_bern = z2_bern.mean()
    median_bern = np.median(z2_bern)

    print(f"\nBinomial z² (what we currently report):")
    print(f"  mean   = {mean_bin:.4f}")
    print(f"  median = {median_bin:.4f}")

    print(f"\nBernoulli-equivalent z² (avg over N trials per cell):")
    print(f"  mean   = {mean_bern:.4f}")
    print(f"  median = {median_bern:.4f}")

    # Predicted relationship: mean_bin ≈ 1 + N * (mean_bern - 1)
    N_eff = float(np.mean(N_trials))
    predicted_bin = 1 + N_eff * (mean_bern - 1)
    predicted_bern = 1 + (mean_bin - 1) / N_eff

    print(f"\nN-scaling relationship (expected to hold in expectation):")
    print(f"  z²_bin ≈ 1 + N · (z²_bern - 1)")
    print(f"    predicted z²_bin from z²_bern: {predicted_bin:.4f}")
    print(f"    actual z²_bin:                 {mean_bin:.4f}")
    print(f"    relative error:                {100*abs(predicted_bin - mean_bin)/mean_bin:.2f}%")
    print()
    print(f"  z²_bern ≈ 1 + (z²_bin - 1) / N")
    print(f"    predicted z²_bern from z²_bin: {predicted_bern:.4f}")
    print(f"    actual z²_bern:                {mean_bern:.4f}")
    print(f"    relative error:                {100*abs(predicted_bern - mean_bern)/mean_bern:.2f}%")

    # Classical IRT verdict under Bernoulli framing
    if 0.5 <= mean_bern <= 1.5:
        verdict = "WITHIN classical acceptable range [0.5, 1.5] ✓"
    elif 0.3 <= mean_bern <= 2.0:
        verdict = "marginal (outside [0.5, 1.5] but within [0.3, 2.0])"
    else:
        verdict = "FAIL classical IRT fit criterion"
    print(f"\nClassical IRT verdict under Bernoulli framing: {verdict}")

    # Per-cell correlation check
    # Under model correctness + independent trials: z²_bin ≈ 1 + N·(z²_bern - 1)
    # This should hold cell-wise in expectation; check correlation.
    corr = np.corrcoef(z2_bin.ravel(), z2_bern.ravel())[0, 1]
    print(f"\nPer-cell correlation corr(z²_bin, z²_bern) = {corr:.4f}")
    print(f"  (should be high if N-scaling is consistent per-cell too)")


if __name__ == "__main__":
    main()
