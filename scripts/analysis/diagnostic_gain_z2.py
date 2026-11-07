"""Recompute mean z²_bern on the CIF ALO[0.20, 0.80] subset (J=183, M=9).

Used for paper §app:diagnostic-gain refresh after the filter switch from
AVG[0.05, 0.95] (J=171) to ALO[0.20, 0.80] (J=183) on 2026-05-15.

Methodology mirrors compare_filters.py / irt_diagnostics.py:
  1. Load full response matrix from filter_ablation_raw_v2.
  2. Apply ALO[0.20, 0.80] mask: keep items where at least one panel
     model has p_hat in [0.20, 0.80].
  3. Fit 1PL Rasch (MLE, OLS warm-start) on the masked subset.
  4. Compute per-cell z²_bern = (k - N p̂)² / (N p̂(1-p̂)) and report
     the mean over all M×J cells, plus NaN cell count.

Run:
    conda run -n boundary --no-capture-output python -u \
        scripts/analysis/diagnostic_gain_z2.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difficulty.rasch import load_response_matrix, fit_ols, fit_mle
from difficulty.irt_diagnostics import compute_z2_bernoulli_avg


RESPONSE_DIR = "results/constrained_completion/filter_ablation_raw_v2"
ALO_LO, ALO_HI = 0.20, 0.80


def main() -> None:
    rm = load_response_matrix(RESPONSE_DIR)
    pass_counts = rm["pass_counts"]  # (M, J_full)
    N_trials = rm["total_trials"]
    p_hat = rm["p_hat"]
    M, J_full = pass_counts.shape

    # ALO mask: at least one panel model with p_hat in [lo, hi]
    mask = ((p_hat >= ALO_LO) & (p_hat <= ALO_HI)).any(axis=0)
    J = int(mask.sum())

    pc_sub = pass_counts[:, mask]
    N_sub = N_trials[:, mask]
    p_sub = p_hat[:, mask]

    # 1PL fit on the kept subset (OLS warm-start → MLE), same as
    # compare_filters._fit_irt_for_diagnostics for irt_model='1pl'.
    theta_init, beta_init = fit_ols(p_sub)
    theta, beta, _nll = fit_mle(pc_sub, N_sub, theta_init, beta_init)

    # p̂_pred = σ(θ_m - β_j)
    z = theta[:, None] - beta[None, :]
    p_pred = 1.0 / (1.0 + np.exp(-z))

    z2_bern = compute_z2_bernoulli_avg(pc_sub, N_sub, p_pred)
    finite = np.isfinite(z2_bern)
    n_nan = int((~finite).sum())
    mean_z2 = float(z2_bern[finite].mean())

    print(f"category       : constrained_completion")
    print(f"filter         : ALO[{ALO_LO}, {ALO_HI}]")
    print(f"M, J_full      : {M}, {J_full}")
    print(f"J (kept)       : {J}")
    print(f"total cells    : {M * J}")
    print(f"NaN cells      : {n_nan}  (p̂ saturated at 0 or 1)")
    print(f"mean z²_bern   : {mean_z2:.6f}")
    print(f"median z²_bern : {np.median(z2_bern[finite]):.6f}")


if __name__ == "__main__":
    main()
