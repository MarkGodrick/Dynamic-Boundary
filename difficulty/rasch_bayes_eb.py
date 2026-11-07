"""Empirical Bayes shrinkage for 1PL Rasch β estimates.

Approach: Laplace approximation + method-of-moments prior variance.
  - Fit 1PL Rasch MLE → β̂_MLE, θ̂_MLE
  - Observed Fisher info at MLE → SE(β̂_x) for each item (via pseudo-inverse
    of full Hessian to handle Σθ=0 identifiability)
  - Assume β_x ~ N(μ, τ²) prior with empirical Bayes estimation:
        Var(β̂_MLE) ≈ τ² + mean(SE²)
        τ̂² = max(0, Var(β̂_MLE) − mean(SE²))   [MoM]
        μ̂ = mean(β̂_MLE)
  - Posterior mean under Laplace observation model β̂_x | β_x ~ N(β_x, SE²):
        β̂_EB_x = (τ̂² / (τ̂² + SE_x²)) · β̂_MLE_x  +  (SE_x² / (τ̂² + SE_x²)) · μ̂
  - Posterior SD:
        SD(β̂_EB_x) = √(τ̂² · SE_x² / (τ̂² + SE_x²))

Shrinkage factor B_x = SE_x² / (τ̂² + SE_x²) ∈ [0, 1]:
  - Well-determined items (small SE): B → 0, β̂_EB ≈ β̂_MLE
  - Poorly-determined items (large SE, typically extreme β̂): B → 1, pulled to μ̂

This is a Laplace-approximation stand-in for full hierarchical Bayesian MCMC
(e.g., PyMC). The approximation is accurate when the likelihood surface is
locally Gaussian at the MLE (which holds for 1PL Binomial at non-extreme p).
For the extreme cells where p̂ ∈ {0, 1}, fit_mle clips σ(·) ∈ [1e−10, 1−1e−10]
but β̂ is still far from its true posterior mode — full MCMC handles these
better. EB here serves as a fast diagnostic; upgrade to PyMC if signal
warrants.

References:
  - Efron, B. (2010). Large-Scale Inference: Empirical Bayes Methods.
    Cambridge University Press. Ch. 1-2.
  - Efron, B., & Morris, C. (1973). Stein's estimation rule and its
    competitors — an empirical Bayes approach. JASA, 68(341), 117–130.
  - Fox, J.-P. (2010). Bayesian Item Response Modeling. Springer. Ch. 4.
"""

from __future__ import annotations

import numpy as np

from difficulty.rasch import fit_mle, fit_ols, _sigmoid


def observed_fisher_se(theta: np.ndarray,
                       beta: np.ndarray,
                       N_trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Fisher-info-based SE for 1PL Rasch (θ, β) at MLE.

    Observed Fisher info (full Hessian of −ℓ at MLE):
        I_θθ[m,m] = Σ_x N_mx · p_mx (1−p_mx)       (diagonal, M × M)
        I_ββ[x,x] = Σ_m N_mx · p_mx (1−p_mx)       (diagonal, J × J)
        I_θβ[m,x] = −N_mx · p_mx (1−p_mx)           (dense, M × J)

    Joint Hessian has rank M+J−1 under Σθ=0 identifiability (null
    direction (1,1,...,1) across all params). We invert via Moore–Penrose
    pseudo-inverse to project out this direction; SE in the complement
    is the correct asymptotic CRLB under the identifying constraint.

    Returns (se_theta, se_beta).
    """
    M, J = len(theta), len(beta)
    p = _sigmoid(theta[:, None] - beta[None, :])
    q = p * (1.0 - p)
    W = N_trials * q

    H = np.zeros((M + J, M + J))
    np.fill_diagonal(H[:M, :M], W.sum(axis=1))
    np.fill_diagonal(H[M:, M:], W.sum(axis=0))
    H[:M, M:] = -W
    H[M:, :M] = -W.T

    Cov = np.linalg.pinv(H, rcond=1e-10)
    se_theta = np.sqrt(np.maximum(np.diag(Cov[:M, :M]), 0.0))
    se_beta = np.sqrt(np.maximum(np.diag(Cov[M:, M:]), 0.0))
    return se_theta, se_beta


def empirical_bayes_shrinkage(beta_mle: np.ndarray,
                              se_beta: np.ndarray,
                              mu: float | None = None,
                              ) -> tuple[np.ndarray, np.ndarray,
                                         float, float, np.ndarray]:
    """Apply method-of-moments EB shrinkage to β̂_MLE.

    Args:
        beta_mle: MLE β estimates (J,).
        se_beta: Asymptotic SE of β̂_MLE from observed Fisher info (J,).
        mu: Prior mean. If None, uses grand mean of β̂_MLE.

    Returns:
        beta_eb: Shrunken β estimates (J,).
        se_eb: Posterior SD (J,).
        mu_hat: Estimated prior mean (scalar).
        tau_sq: Estimated prior variance τ² (scalar, ≥ 0).
        shrinkage: Per-item shrinkage factor B_x ∈ [0, 1] (J,).
    """
    mu_hat = float(beta_mle.mean()) if mu is None else float(mu)
    var_beta = float(np.var(beta_mle - mu_hat, ddof=1))
    mean_var_err = float(np.mean(se_beta ** 2))
    tau_sq = max(0.0, var_beta - mean_var_err)

    denom = tau_sq + se_beta ** 2
    # Guard against denom=0 (only possible if tau_sq=0 and some SE=0, which
    # requires a perfectly determined item — degenerate, fall back to MLE)
    denom = np.maximum(denom, 1e-12)
    shrinkage = se_beta ** 2 / denom
    beta_eb = (1.0 - shrinkage) * beta_mle + shrinkage * mu_hat
    se_eb = np.sqrt(tau_sq * se_beta ** 2 / denom)

    return beta_eb, se_eb, mu_hat, tau_sq, shrinkage


def fit_eb(pass_counts: np.ndarray,
           N_trials: np.ndarray,
           mu: float | None = None) -> dict:
    """Full EB pipeline: OLS → MLE → Fisher SE → empirical Bayes shrinkage.

    Args:
        pass_counts: (M, J) Binomial pass counts.
        N_trials: (M, J) trial counts.
        mu: Fix prior mean (None → estimate as mean β̂_MLE).

    Returns dict with:
        theta: (M,)       MLE θ (unchanged; EB only shrinks β)
        beta_mle: (J,)    original MLE β̂
        beta_eb: (J,)     shrunken β̂
        se_beta_fisher: (J,) asymptotic SE of β̂_MLE
        se_beta_eb: (J,)  posterior SD under EB
        se_theta: (M,)    asymptotic SE of θ̂
        mu_hat: float     estimated prior mean
        tau_sq: float     estimated prior variance
        tau: float        √tau_sq (interpretable logit scale)
        shrinkage: (J,)   per-item B_x factor
        neg_loglik: float MLE NLL
    """
    p_hat = pass_counts / np.maximum(N_trials, 1.0)
    theta_init, beta_init = fit_ols(p_hat)
    theta_mle, beta_mle, nll = fit_mle(pass_counts, N_trials,
                                       theta_init, beta_init)

    se_theta, se_beta = observed_fisher_se(theta_mle, beta_mle, N_trials)
    beta_eb, se_eb, mu_hat, tau_sq, shrinkage = empirical_bayes_shrinkage(
        beta_mle, se_beta, mu=mu)

    return {
        "theta": theta_mle,
        "beta_mle": beta_mle,
        "beta_eb": beta_eb,
        "se_theta": se_theta,
        "se_beta_fisher": se_beta,
        "se_beta_eb": se_eb,
        "mu_hat": mu_hat,
        "tau_sq": tau_sq,
        "tau": float(np.sqrt(tau_sq)),
        "shrinkage": shrinkage,
        "neg_loglik": float(nll),
    }
