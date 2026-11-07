"""2PL IRT (two-parameter logistic) fit for Binomial response matrices.

Model:
    p(m, x) = σ(α_x · (θ_m - β_x))

Parameters:
    θ (M,):  panel ability
    β (J,):  item difficulty
    α (J,):  item discrimination (> 0)

Compared to 1PL Rasch (α_x ≡ 1 for all x), 2PL adds per-item discrimination
and can represent items whose steepness differs (e.g. mixed-response items
whose panel-wide pass-rate variance differs from what β alone predicts).

Used as the "better-model" comparison in Phase 9.3 IRT ablation and in
filter ablation sanity-check to distinguish "filter bad" from "1PL bad".

Identifiability: 2PL is invariant under joint rescaling (α → α/c, θ → c·θ,
β → c·β). We anchor mean(θ) = 0 to pin the scale.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from difficulty.rasch import fit_ols as _fit_ols_1pl


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable: clip at ±40 (σ(40) ≈ 1 - 4e-18; further saturation
    # contributes nothing and causes RuntimeWarnings in np.where).
    x_clipped = np.clip(x, -40.0, 40.0)
    return np.where(x_clipped >= 0,
                    1 / (1 + np.exp(-x_clipped)),
                    np.exp(x_clipped) / (1 + np.exp(x_clipped)))


def _pack(theta: np.ndarray, beta: np.ndarray, log_alpha: np.ndarray) -> np.ndarray:
    return np.concatenate([theta, beta, log_alpha])


def _unpack(params: np.ndarray, M: int, J: int) -> tuple:
    theta = params[:M]
    beta = params[M:M + J]
    log_alpha = params[M + J:]
    return theta, beta, log_alpha


def _neg_log_likelihood(params, pass_counts, total_trials, M, J):
    theta, beta, log_alpha = _unpack(params, M, J)
    alpha = np.exp(log_alpha)          # (J,) enforce α > 0
    diff = theta[:, None] - beta[None, :]          # (M, J)
    eta = alpha[None, :] * diff                    # (M, J)
    p = np.clip(_sigmoid(eta), 1e-10, 1 - 1e-10)
    ll = (pass_counts * np.log(p)
          + (total_trials - pass_counts) * np.log(1 - p)).sum()
    return -ll


def _neg_log_likelihood_grad(params, pass_counts, total_trials, M, J):
    theta, beta, log_alpha = _unpack(params, M, J)
    alpha = np.exp(log_alpha)
    diff = theta[:, None] - beta[None, :]
    eta = alpha[None, :] * diff
    p = np.clip(_sigmoid(eta), 1e-10, 1 - 1e-10)

    residual = pass_counts - total_trials * p  # (M, J); ∂LL/∂η

    # ∂LL/∂θ_m  = Σ_x residual[m, x] · α_x
    grad_theta = -(residual * alpha[None, :]).sum(axis=1)
    # ∂LL/∂β_x  = -Σ_m residual[m, x] · α_x
    grad_beta = (residual * alpha[None, :]).sum(axis=0)
    # ∂LL/∂log_α_x = α_x · Σ_m residual[m, x] · (θ_m - β_x)
    grad_log_alpha = -alpha * (residual * diff).sum(axis=0)

    return np.concatenate([grad_theta, grad_beta, grad_log_alpha])


def fit_ols(p_hat: np.ndarray) -> tuple:
    """Warm-start init for 2PL.

    Reuses 1PL OLS for (θ, β). Initializes α_x = 1 for all x (i.e. start
    at the 1PL solution; MLE will then deviate α where supported).
    """
    theta0, beta0 = _fit_ols_1pl(p_hat)
    J = p_hat.shape[1]
    log_alpha0 = np.zeros(J)  # log(1) = 0
    return theta0, beta0, log_alpha0


def fit_mle(
    pass_counts: np.ndarray,
    total_trials: np.ndarray,
    theta_init: np.ndarray | None = None,
    beta_init: np.ndarray | None = None,
    log_alpha_init: np.ndarray | None = None,
    max_iter: int = 2000,
    log_alpha_bounds: tuple = (-2.0, 2.0),
) -> tuple:
    """2PL MLE via L-BFGS-B on log-α (bounded).

    Args:
        pass_counts:  (M, J) Binomial pass counts.
        total_trials: (M, J) Binomial trial counts.
        theta_init, beta_init, log_alpha_init: Optional warm starts. If None,
            initialize from fit_ols.
        max_iter: L-BFGS-B iteration cap.
        log_alpha_bounds: Bounds on log α (default (-2, 2) = α ∈ [0.135, 7.389]).
            Prevents the classic 2PL pathology where α → ∞ on uniform-response
            items (fit degenerates into a step function overfitting the M=6
            panel points). θ and β are unbounded.

    Returns:
        (theta, beta, alpha, neg_loglik)

    Notes:
        - Scale anchor (mean(θ) = 0) is applied AFTER fit completes.
    """
    M, J = pass_counts.shape

    if theta_init is None or beta_init is None or log_alpha_init is None:
        p_hat = pass_counts / np.maximum(total_trials, 1)
        t0, b0, la0 = fit_ols(p_hat)
        theta_init = t0 if theta_init is None else theta_init
        beta_init = b0 if beta_init is None else beta_init
        log_alpha_init = la0 if log_alpha_init is None else log_alpha_init

    x0 = _pack(theta_init, beta_init, log_alpha_init)

    # Bounds: θ and β unbounded, log α in [lo, hi]
    lo, hi = log_alpha_bounds
    bounds = [(None, None)] * (M + J) + [(lo, hi)] * J

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(pass_counts, total_trials, M, J),
        jac=_neg_log_likelihood_grad,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-10},
    )

    theta, beta, log_alpha = _unpack(result.x, M, J)
    alpha = np.exp(log_alpha)

    # Scale anchor: mean(θ) = 0.
    # 2PL is invariant under (α → α/c, θ → c·θ, β → c·β). We pick c such
    # that the shifted θ has zero mean.
    shift = theta.mean()
    theta = theta - shift
    beta = beta - shift
    # α unchanged (shift, not scale)

    return theta, beta, alpha, float(result.fun)


def predict(theta: np.ndarray, beta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Model-implied p̂(m, x) for all cells."""
    diff = theta[:, None] - beta[None, :]
    eta = alpha[None, :] * diff
    return _sigmoid(eta)
