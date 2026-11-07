"""Multidimensional IRT (MIRT, compensatory) fit for Binomial response data.

Model (compensatory D-dimensional MIRT):
    p(m, x) = σ( a_x · θ_m - β_x )
    where a_x ∈ ℝ^D, θ_m ∈ ℝ^D, β_x ∈ ℝ

For D = 1, this reduces to 2PL (scalar α_x, θ_m). D = 2+ adds latent
ability dimensions — each item has a D-vector of discriminations that
describes how strongly it loads on each latent trait.

Used as a sanity check against the 1PL/2PL choice: if MIRT pseudo-R²
on held-out cells is substantially higher than 1PL, the data has
multi-dimensional structure that unidimensional IRT cannot capture.

Identifiability:
    Invariant under joint rotation (R · θ, R · a). We fix:
        - mean(θ_d) = 0 for each d (translation)
        - log ||a_x|| bounded (scale)
    Rotational indeterminacy is NOT resolved here (we only care about
    likelihood / predictive quality, not parameter interpretation).
    Use post-hoc varimax if parameter interpretability is needed.

Regularization: log ||a_x|| bounded to [-2, 2], analogous to 2PL.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from difficulty.rasch import fit_ols as _fit_ols_1pl


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -40.0, 40.0)
    return np.where(x_clipped >= 0,
                    1 / (1 + np.exp(-x_clipped)),
                    np.exp(x_clipped) / (1 + np.exp(x_clipped)))


def _pack(theta: np.ndarray, beta: np.ndarray,
          log_a_norm: np.ndarray, a_dir: np.ndarray) -> np.ndarray:
    """Flatten parameters for scipy.minimize.

    θ (M, D) → first M*D entries
    β (J,)   → next J entries
    log_a_norm (J,) → next J entries (norm of a_x in log scale)
    a_dir (J, D) → next J*D entries (unit-vector direction of a_x)
    """
    return np.concatenate([theta.ravel(), beta, log_a_norm, a_dir.ravel()])


def _unpack(params: np.ndarray, M: int, J: int, D: int) -> tuple:
    idx = 0
    theta = params[idx:idx + M * D].reshape(M, D); idx += M * D
    beta = params[idx:idx + J]; idx += J
    log_a_norm = params[idx:idx + J]; idx += J
    a_dir = params[idx:idx + J * D].reshape(J, D); idx += J * D
    return theta, beta, log_a_norm, a_dir


def _reconstruct_a(log_a_norm: np.ndarray, a_dir: np.ndarray) -> np.ndarray:
    """Build a (J, D) loading matrix from log-norm + unnormalized direction.

    We parameterize a_x = exp(log_a_norm_x) * a_dir_x to make the norm
    bound easy to enforce. During optimization a_dir does NOT have to be
    unit-norm — we divide by its norm at prediction time.
    """
    a_norm = np.exp(log_a_norm)                                  # (J,)
    dir_norm = np.linalg.norm(a_dir, axis=1, keepdims=True)      # (J, 1)
    dir_norm = np.maximum(dir_norm, 1e-10)
    a_unit = a_dir / dir_norm                                    # unit vectors
    return a_norm[:, None] * a_unit                              # (J, D)


def predict(theta: np.ndarray, beta: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Model-implied p̂(m, x) given (θ, β, a).

    theta: (M, D), beta: (J,), a: (J, D).
    Returns p̂ of shape (M, J).
    """
    if a.ndim == 1:
        a = a[:, None]
    if theta.ndim == 1:
        theta = theta[:, None]
    logits = theta @ a.T - beta[None, :]
    return _sigmoid(logits)


def _neg_log_likelihood(params, pass_counts, total_trials, M, J, D):
    theta, beta, log_a_norm, a_dir = _unpack(params, M, J, D)
    a = _reconstruct_a(log_a_norm, a_dir)
    logits = theta @ a.T - beta[None, :]
    p = np.clip(_sigmoid(logits), 1e-10, 1 - 1e-10)
    ll = (pass_counts * np.log(p)
          + (total_trials - pass_counts) * np.log(1 - p)).sum()
    return -ll


def _neg_log_likelihood_grad(params, pass_counts, total_trials, M, J, D):
    theta, beta, log_a_norm, a_dir = _unpack(params, M, J, D)
    a_norm = np.exp(log_a_norm)
    dir_norm = np.linalg.norm(a_dir, axis=1, keepdims=True)
    dir_norm = np.maximum(dir_norm, 1e-10)
    a_unit = a_dir / dir_norm
    a = a_norm[:, None] * a_unit                                   # (J, D)

    logits = theta @ a.T - beta[None, :]                           # (M, J)
    p = np.clip(_sigmoid(logits), 1e-10, 1 - 1e-10)
    residual = pass_counts - total_trials * p                      # (M, J)

    # ∂LL/∂θ_{m,d} = Σ_x residual[m, x] · a[x, d]
    grad_theta = -(residual @ a)                                   # (M, D)

    # ∂LL/∂β_x = -Σ_m residual[m, x]
    grad_beta = residual.sum(axis=0)                               # (J,)

    # For a_x: ∂LL/∂a_{x,d} = Σ_m residual[m, x] · θ_{m,d}
    grad_a = -(residual.T @ theta)                                 # (J, D)

    # log_a_norm_x : ∂a / ∂log_a_norm = a_norm · a_unit = a (chain: d(exp)/dx = exp)
    # ∂LL/∂log_a_norm_x = Σ_d grad_a[x, d] · a[x, d] = <grad_a, a>
    grad_log_a_norm = (grad_a * a).sum(axis=1)                     # (J,)

    # a_dir_x : a = a_norm * a_dir / ||a_dir||
    # ∂a_d / ∂a_dir_e = a_norm · [δ_{de} ||a_dir|| - a_dir_e · a_dir_d / ||a_dir||] / ||a_dir||²
    #                 = (a_norm / ||a_dir||) · (I - a_unit a_unit^T)_{de}
    # ∂LL/∂a_dir_{x,e} = (a_norm_x / ||a_dir_x||) · Σ_d grad_a[x, d] · (δ_{de} - a_unit_{x,d} a_unit_{x,e})
    #                  = (a_norm_x / ||a_dir_x||) · (grad_a[x, :] - (grad_a[x, :] · a_unit_x) · a_unit_x)_e
    # Equivalently: project grad_a onto orthogonal complement of a_unit, then scale.
    scalar = (grad_a * a_unit).sum(axis=1, keepdims=True)          # (J, 1)
    grad_a_dir = (grad_a - scalar * a_unit) * (a_norm[:, None] / dir_norm)

    return np.concatenate([
        grad_theta.ravel(),
        grad_beta,
        grad_log_a_norm,
        grad_a_dir.ravel()
    ])


def fit_mle(
    pass_counts: np.ndarray,
    total_trials: np.ndarray,
    D: int = 2,
    theta_init: np.ndarray | None = None,
    beta_init: np.ndarray | None = None,
    a_init: np.ndarray | None = None,
    max_iter: int = 2000,
    log_a_norm_bounds: tuple = (-2.0, 2.0),
    init_noise: float = 0.1,
    seed: int = 0,
) -> tuple:
    """Compensatory MIRT MLE via L-BFGS-B.

    Args:
        pass_counts:  (M, J) Binomial pass counts.
        total_trials: (M, J) Binomial trial counts.
        D: Latent dimensionality. D=1 recovers 2PL.
        theta_init: (M, D) or None. If None, first dim from 1PL OLS, rest random.
        beta_init: (J,) or None. If None, from 1PL OLS.
        a_init: (J, D) or None. If None, first dim = 1, rest small random.
        max_iter, log_a_norm_bounds: L-BFGS-B options.
        init_noise: Std of noise for random init of extra dims.
        seed: RNG seed for init noise.

    Returns:
        (theta, beta, a, neg_loglik)
        theta: (M, D); beta: (J,); a: (J, D).

    Post-fit housekeeping:
        - mean(θ_d) = 0 for each d (center)
        - Report final log_a_norm diagnostics in caller if needed

    Caveats:
        - MIRT with small M (e.g., M=6) on D≥2 has identification & variance
          issues. log_a_norm bounds ([-2, 2] → ||a|| ∈ [0.135, 7.389]) prevent
          the same "α → ∞" pathology seen in 2PL, but cannot rescue the
          bias-variance tradeoff of per-item parameter estimation under
          small panels.
        - Rotational indeterminacy unresolved — likelihood and predictions
          are invariant, but (θ, a) individually are not interpretable
          without a rotation convention (e.g. varimax).
    """
    M, J = pass_counts.shape
    rng = np.random.RandomState(seed)

    # ---- Initial values ----
    p_hat = pass_counts / np.maximum(total_trials, 1)
    t0_1d, b0 = _fit_ols_1pl(p_hat)                  # (M,), (J,)

    if theta_init is None:
        theta_init = np.zeros((M, D))
        theta_init[:, 0] = t0_1d
        if D > 1:
            theta_init[:, 1:] = init_noise * rng.randn(M, D - 1)
    if beta_init is None:
        beta_init = b0
    if a_init is None:
        a_init = np.zeros((J, D))
        a_init[:, 0] = 1.0
        if D > 1:
            a_init[:, 1:] = init_noise * rng.randn(J, D - 1)

    # Convert a_init into (log_a_norm, a_dir) parameterization
    a_norm_init = np.linalg.norm(a_init, axis=1)
    a_norm_init = np.maximum(a_norm_init, 1e-3)
    log_a_norm_init = np.log(a_norm_init)
    a_dir_init = a_init / a_norm_init[:, None]

    x0 = _pack(theta_init, beta_init, log_a_norm_init, a_dir_init)

    # Bounds: θ unbounded; β unbounded; log_a_norm bounded; a_dir unbounded.
    lo, hi = log_a_norm_bounds
    bounds = (
        [(None, None)] * (M * D)       # θ
        + [(None, None)] * J           # β
        + [(lo, hi)] * J               # log_a_norm
        + [(None, None)] * (J * D)     # a_dir
    )

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(pass_counts, total_trials, M, J, D),
        jac=_neg_log_likelihood_grad,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-10},
    )

    theta, beta, log_a_norm, a_dir = _unpack(result.x, M, J, D)
    a = _reconstruct_a(log_a_norm, a_dir)

    # Center: mean(θ_d) = 0 per dim. Adjust β to preserve predictions:
    # shift θ_m → θ_m - μ  ⇒  logit = (θ_m - μ) · a - β + μ · a
    # Equivalently: β_x → β_x - μ · a_x
    mu = theta.mean(axis=0)                         # (D,)
    theta = theta - mu[None, :]
    beta = beta - a @ mu                            # (J,) = (J, D) · (D,)

    return theta, beta, a, float(result.fun)
