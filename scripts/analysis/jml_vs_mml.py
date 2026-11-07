"""JML vs MML robustness check for 1PL Rasch β estimates.

Loads the same K=500 random-composed pool response matrix used for the
calibrated anchor set, applies the chosen anchor filter (AVG or ALO),
then fits β under both estimators:

  - JML (joint MLE): existing fit_mle from difficulty/rasch.py
  - MML (marginal MLE): EM with Gauss-Hermite quadrature

Reports |β_JML - β_MML| distribution + Spearman rank correlation per
category to assess difficulty-estimate sensitivity to the estimator.

Usage:
    python -m scripts.analysis.jml_vs_mml [--filter-rule {avg,alo}]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difficulty.rasch import (
    _sigmoid,
    fit_mle,
    fit_ols,
    load_response_matrix,
)


CATEGORIES = {
    "harmful_refusal":         "results/harmful_refusal/filter_ablation_raw_v2_judgefix",
    "over_refusal":            "results/over_refusal/filter_ablation_raw_v2_judgefix",
    "constrained_completion":  "results/constrained_completion/filter_ablation_raw_v2",
    "sycophancy_resistance":   "results/sycophancy_resistance/filter_ablation_raw_v2",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Marginal MLE via EM with Gauss-Hermite quadrature
# ---------------------------------------------------------------------------

def fit_mml_em(
    K: np.ndarray,
    N: np.ndarray,
    n_quad: int = 41,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
) -> dict:
    """Fit 1PL Rasch via Marginal Maximum Likelihood with EM.

    θ_m ~ N(μ, σ²) is the random-effect prior; we integrate it out per
    panel model with Gauss-Hermite quadrature (n_quad nodes), then run
    EM:
      E-step: posterior q(θ_m | data, β^(t)) on the quadrature grid.
      M-step: update β by maximizing expected log-likelihood; update
              (μ, σ²) by posterior moment matching.

    Args:
        K: (M, J) pass counts.
        N: (M, J) trial counts.
        n_quad: Gauss-Hermite nodes for the θ integral.
        max_iter: EM iteration cap.
        tol: marginal log-likelihood improvement tolerance.

    Returns:
        Dict with keys: theta_eap, beta, mu, sigma, n_iter, marg_ll.
    """
    M, J = K.shape

    # Init from OLS
    p_hat = np.where(N > 0, K / N, 0.5)
    theta_init, beta_init = fit_ols(p_hat)
    beta = beta_init.astype(np.float64).copy()
    mu = float(theta_init.mean())
    sigma = float(max(theta_init.std(ddof=0), 0.5))

    # Gauss-Hermite nodes for ∫ f(θ) N(θ; μ, σ²) dθ
    # = sum_k w_k / sqrt(π) * f(sqrt(2)·σ·z_k + μ)
    z_nodes, raw_weights = np.polynomial.hermite.hermgauss(n_quad)
    base_weights = raw_weights / np.sqrt(np.pi)  # ∫ N(z;0,1) integrating weights up to scale

    prev_marg_ll = -np.inf
    n_iter = 0
    marg_ll = -np.inf

    for it in range(max_iter):
        n_iter = it + 1
        # Quadrature θ-points under current (μ, σ)
        theta_q = np.sqrt(2.0) * sigma * z_nodes + mu  # (Q,)
        # Importance weight from change-of-variable cancels: keep base_weights
        w_q = base_weights  # (Q,) sums to 1 (modulo numerical)

        # Per-(q, x) log-prob
        diff = theta_q[:, None] - beta[None, :]  # (Q, J)
        p = np.clip(_sigmoid(diff), 1e-12, 1 - 1e-12)
        log_p = np.log(p)
        log_1mp = np.log1p(-p)

        # ll[m, q] = sum_x [K[m,x] log_p[q,x] + (N-K)[m,x] log(1-p)[q,x]]
        ll_mq = K @ log_p.T + (N - K) @ log_1mp.T  # (M, Q)

        # E-step: posterior q(θ | data_m, β) ∝ w_q * exp(ll_mq)
        log_w = np.log(w_q)[None, :]
        log_post_unn = ll_mq + log_w
        log_post_max = log_post_unn.max(axis=1, keepdims=True)
        post = np.exp(log_post_unn - log_post_max)
        norm = post.sum(axis=1, keepdims=True)
        post /= norm  # (M, Q)

        # Marginal log-likelihood
        marg_ll = float((np.log(norm[:, 0]) + log_post_max[:, 0]).sum())

        # Posterior moments per m
        E_theta = (post * theta_q[None, :]).sum(axis=1)         # (M,)
        E_theta2 = (post * theta_q[None, :] ** 2).sum(axis=1)    # (M,)

        # M-step: update (μ, σ) by moment matching
        mu_new = float(E_theta.mean())
        var_new = float((E_theta2 - 2.0 * mu_new * E_theta + mu_new ** 2).mean())
        sigma_new = float(max(np.sqrt(max(var_new, 1e-6)), 1e-3))

        # M-step: update β by maximizing Q-function w.r.t. β
        #   Q(β) = sum_m sum_q post[m,q] * ll_mq[q, x](β)
        # Gradient w.r.t. β_x:
        #   d/dβ_x = sum_m sum_q post[m,q] * (N[m,x] σ(θ_q - β_x) - K[m,x])
        #         = (post.T @ N)[q,x] * σ(θ_q - β_x) summed - K.sum(0)
        W = post.T @ N  # (Q, J)
        K_col = K.sum(axis=0)  # (J,)

        def neg_q(b):
            d = theta_q[:, None] - b[None, :]
            pp = np.clip(_sigmoid(d), 1e-12, 1 - 1e-12)
            lp = np.log(pp)
            l1p = np.log1p(-pp)
            ll = K @ lp.T + (N - K) @ l1p.T  # (M, Q)
            return -float((post * ll).sum())

        def neg_q_grad(b):
            d = theta_q[:, None] - b[None, :]
            pp = np.clip(_sigmoid(d), 1e-12, 1 - 1e-12)
            term1 = (W * pp).sum(axis=0)  # (J,)
            return term1 - K_col

        result = minimize(
            neg_q, beta, jac=neg_q_grad,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-8},
        )
        beta_new = result.x

        delta_ll = marg_ll - prev_marg_ll
        if verbose:
            print(
                f"  iter {it+1:3d}  marg_ll={marg_ll:+.4f}  Δ={delta_ll:+.2e}  "
                f"μ={mu_new:+.3f} σ={sigma_new:.3f}",
                flush=True,
            )

        beta = beta_new
        mu = mu_new
        sigma = sigma_new

        if abs(delta_ll) < tol and it > 5:
            break
        prev_marg_ll = marg_ll

    # EAP θ estimates (final E-step output)
    theta_eap = E_theta

    # Apply identifiability shift: sum-to-zero on θ (matches JML convention
    # in difficulty/rasch.py:185-187).
    shift = float(theta_eap.mean())
    theta_eap = theta_eap - shift
    beta = beta - shift
    mu = mu - shift

    return {
        "theta_eap": theta_eap,
        "beta": beta,
        "mu": mu,
        "sigma": sigma,
        "n_iter": n_iter,
        "marg_ll": marg_ll,
    }


# ---------------------------------------------------------------------------
# Anchor set extraction + comparison
# ---------------------------------------------------------------------------

_DEFAULT_BANDS = {
    "avg": (0.05, 0.95),
    "alo": (0.20, 0.80),
}


def apply_anchor_filter(
    pass_counts: np.ndarray,
    total_trials: np.ndarray,
    rule: str = "avg",
    p_lo: float | None = None,
    p_hi: float | None = None,
) -> np.ndarray:
    """Anchor filter: 'avg' = panel-mean p̂ in [lo, hi]; 'alo' = at-least-one
    panel member with p̂ in [lo, hi]."""
    if rule not in _DEFAULT_BANDS:
        raise ValueError(f"Unknown filter rule: {rule!r}")
    default_lo, default_hi = _DEFAULT_BANDS[rule]
    lo = default_lo if p_lo is None else p_lo
    hi = default_hi if p_hi is None else p_hi
    p_hat = np.where(total_trials > 0, pass_counts / total_trials, 0.5)
    if rule == "avg":
        bar_p = p_hat.mean(axis=0)
        return (bar_p >= lo) & (bar_p <= hi)
    # alo
    in_band = (p_hat >= lo) & (p_hat <= hi)
    return in_band.any(axis=0)


def compare_one_category(
    category: str,
    data_dir: Path,
    filter_rule: str = "avg",
    verbose: bool = False,
) -> dict:
    print(f"\n=== {category} ===", flush=True)
    rm = load_response_matrix(str(data_dir))
    M = len(rm["models"])
    J = len(rm["items"])
    print(f"  Loaded {M} models × {J} items", flush=True)

    lo, hi = _DEFAULT_BANDS[filter_rule]
    keep = apply_anchor_filter(rm["pass_counts"], rm["total_trials"], rule=filter_rule)
    K = rm["pass_counts"][:, keep]
    N = rm["total_trials"][:, keep]
    J_kept = int(keep.sum())
    print(f"  {filter_rule.upper()}[{lo},{hi}] retains {J_kept}/{J} items", flush=True)

    # JML
    p_hat = np.where(N > 0, K / N, 0.5)
    theta_ols, beta_ols = fit_ols(p_hat)
    theta_jml, beta_jml, jml_nll = fit_mle(K, N, theta_ols, beta_ols)
    print(f"  JML: nll={jml_nll:.2f}  θ range=[{theta_jml.min():+.2f}, {theta_jml.max():+.2f}]",
          flush=True)

    # MML
    print("  MML EM ...", flush=True)
    res = fit_mml_em(K, N, verbose=verbose)
    theta_mml = res["theta_eap"]
    beta_mml = res["beta"]
    print(f"  MML: marg_ll={res['marg_ll']:.2f}  iters={res['n_iter']}  "
          f"μ̂={res['mu']:+.3f} σ̂={res['sigma']:.3f}", flush=True)

    # Comparison
    delta = beta_jml - beta_mml
    abs_delta = np.abs(delta)
    rho = float(spearmanr(beta_jml, beta_mml)[0])
    pearson = float(np.corrcoef(beta_jml, beta_mml)[0, 1])

    # Andersen scale check: if JML is over-shrunk by M/(M-1), then
    # β_JML ≈ ((M-1)/M) · β_MML. Fit OLS slope β_JML = a·β_MML + b.
    a, b = np.polyfit(beta_mml, beta_jml, 1)
    expected_andersen = (M - 1) / M
    # Residual after removing scale-shift
    beta_jml_rescaled = (beta_jml - b) / a
    delta_resc = beta_jml_rescaled - beta_mml
    rms_resc = float(np.sqrt(np.mean(delta_resc ** 2)))

    summary = {
        "category": category,
        "filter_rule": filter_rule,
        "filter_band": [lo, hi],
        "n_models": M,
        "n_items_pool": J,
        "n_items_anchor": J_kept,
        "spearman_rho_beta": round(rho, 5),
        "pearson_r_beta": round(pearson, 5),
        "max_abs_delta_beta": round(float(abs_delta.max()), 4),
        "p95_abs_delta_beta": round(float(np.quantile(abs_delta, 0.95)), 4),
        "p50_abs_delta_beta": round(float(np.median(abs_delta)), 4),
        "mean_abs_delta_beta": round(float(abs_delta.mean()), 4),
        "rms_delta_beta": round(float(np.sqrt(np.mean(delta ** 2))), 4),
        # Andersen (1973) scale-shrinkage diagnostic
        "ols_slope_jml_vs_mml": round(float(a), 4),
        "ols_intercept_jml_vs_mml": round(float(b), 4),
        "andersen_predicted_slope": round(expected_andersen, 4),
        "rms_residual_after_rescale": round(rms_resc, 4),
        "rescale_explains_pct": round(100 * (1 - rms_resc / float(np.sqrt(np.mean(delta ** 2)))), 1),
        "mml_mu": round(res["mu"], 4),
        "mml_sigma": round(res["sigma"], 4),
        "theta_jml_range": [round(float(theta_jml.min()), 3), round(float(theta_jml.max()), 3)],
        "theta_mml_range": [round(float(theta_mml.min()), 3), round(float(theta_mml.max()), 3)],
        "theta_range_ratio_mml_over_jml": round(
            (float(theta_mml.max()) - float(theta_mml.min())) /
            (float(theta_jml.max()) - float(theta_jml.min())), 3,
        ),
        "rho_theta": round(float(spearmanr(theta_jml, theta_mml)[0]), 5),
    }

    print(f"  Δβ: median={summary['p50_abs_delta_beta']:.4f}  "
          f"p95={summary['p95_abs_delta_beta']:.4f}  "
          f"max={summary['max_abs_delta_beta']:.4f}  "
          f"RMS={summary['rms_delta_beta']:.4f}", flush=True)
    print(f"  Spearman ρ(β_JML, β_MML) = {rho:.5f}  Pearson r = {pearson:.5f}", flush=True)
    print(f"  OLS slope (β_JML = a·β_MML + b): a={a:.4f}  b={b:.4f}  "
          f"(Andersen predicts {expected_andersen:.4f} for M={M})", flush=True)
    print(f"  After rescale: RMS residual = {rms_resc:.4f}  "
          f"({summary['rescale_explains_pct']}% of original RMS explained by scale)",
          flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter-rule", choices=["avg", "alo"], default="avg",
                        help="Anchor filter rule (default 'avg' for back-compat).")
    parser.add_argument("--output", default=None,
                        help="Output JSON path. Default: "
                             "results/analysis/jml_vs_mml[_<rule>].json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out = {}
    for cat, rel in CATEGORIES.items():
        data_dir = PROJECT_ROOT / rel
        if not data_dir.exists():
            print(f"SKIP {cat}: {data_dir} missing", flush=True)
            continue
        out[cat] = compare_one_category(
            cat, data_dir, filter_rule=args.filter_rule, verbose=args.verbose,
        )

    if args.output is None:
        suffix = "" if args.filter_rule == "avg" else f"_{args.filter_rule}"
        out_path = PROJECT_ROOT / "results" / "analysis" / f"jml_vs_mml{suffix}.json"
    else:
        out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
