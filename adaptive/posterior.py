"""Posterior estimators for θ given a single subject's Binomial trials on items
with frozen difficulties β.

Two implementations sharing a single estimate(k, N, beta) -> (theta_hat, se):

  EAPEstimator           — Gauss–Hermite quadrature on a Normal(μ, σ²) prior.
                            Returns posterior mean and posterior std.
  MAPLaplaceEstimator    — Newton on the log-posterior; SE from observed Fisher.

Both follow the 1PL Rasch likelihood
    P(pass | θ, β_x) = σ(θ - β_x)
with independent Binomial(N_x, σ(θ - β_x)) cells.

The two estimators are interchangeable behind PosteriorEstimator (Protocol).
EAP is used by default; MAP+Laplace is provided for ablation.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _log_likelihood(theta: np.ndarray | float,
                    k: np.ndarray, N: np.ndarray, beta: np.ndarray) -> np.ndarray | float:
    """log P(k | θ, β, N) up to a θ-independent Binomial coefficient.

    Vectorized over θ: theta is broadcast against (k, N, beta) on a new axis.
    """
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    diff = theta_arr[:, None] - beta[None, :]   # (Q, J)
    p = np.clip(_sigmoid(diff), 1e-12, 1.0 - 1e-12)
    ll = (k[None, :] * np.log(p) + (N[None, :] - k[None, :]) * np.log(1.0 - p)).sum(axis=1)
    if np.isscalar(theta):
        return float(ll[0])
    return ll if theta_arr.ndim == 1 else ll.reshape(np.shape(theta))


class PosteriorEstimator(Protocol):
    """Common interface: given trial cells, return (theta_hat, se(theta_hat))."""

    def estimate(self, k: np.ndarray, N: np.ndarray,
                 beta: np.ndarray) -> tuple[float, float]: ...


class EAPEstimator:
    """Posterior mean / std via adaptive Gauss–Hermite quadrature.

    Why adaptive: with J anchor items × N trials, the likelihood is much
    narrower than the prior (Fisher info scales with J·N·p(1-p)). Fixed-grid
    GH centered on the prior places almost all nodes outside the posterior's
    effective support, collapsing the integral to a single node. Adaptive GH
    (Pinheiro & Bates 1995) first locates the posterior mode θ̂ via Newton,
    sets a local scale s from the observed Fisher info there, and places GH
    nodes at θ_q = θ̂ + s · x_q. The Gaussian weight in the GH rule is
    undone via the factor exp(x_q² / 2) so the integrand is just the
    posterior density (× θ for the mean).
    """

    # Clamp θ to ±10 logits during Newton iterations. Sigma(±10) ≈ {4.5e-5,
    # 1−4.5e-5} is wider than any plausible θ̂; clamp only kicks in when the
    # likelihood saturates (all cells at p=0 or p=1) and the mode runs away
    # toward ±∞.
    THETA_CLAMP: float = 10.0

    def __init__(self, prior_mu: float = 0.0, prior_sigma: float = 1.5,
                 n_quad: int = 21, newton_max_iter: int = 50,
                 newton_tol: float = 1e-8):
        if prior_sigma <= 0:
            raise ValueError("prior_sigma must be > 0")
        if n_quad < 5:
            raise ValueError("n_quad must be >= 5")
        self.prior_mu = float(prior_mu)
        self.prior_sigma = float(prior_sigma)
        self.n_quad = int(n_quad)
        self.newton_max_iter = int(newton_max_iter)
        self.newton_tol = float(newton_tol)
        # GH nodes/weights for ∫ f(x) exp(-x²/2) dx (probabilists' convention)
        self._gh_nodes, self._gh_weights = np.polynomial.hermite_e.hermegauss(self.n_quad)

    def _find_mode(self, k: np.ndarray, N: np.ndarray,
                   beta: np.ndarray) -> tuple[float, float]:
        """Newton on the log-posterior; returns (mode, local_scale = SE_Laplace)."""
        sigma2 = self.prior_sigma ** 2
        theta = self.prior_mu
        for _ in range(self.newton_max_iter):
            p = _sigmoid(theta - beta)
            score = float((k - N * p).sum() - (theta - self.prior_mu) / sigma2)
            hess = float(-(N * p * (1.0 - p)).sum() - 1.0 / sigma2)
            step = score / hess
            theta_new = theta - step
            theta_new = float(np.clip(theta_new, -self.THETA_CLAMP, self.THETA_CLAMP))
            if abs(theta_new - theta) < self.newton_tol:
                theta = theta_new
                break
            theta = theta_new
        p = _sigmoid(theta - beta)
        hess = float(-(N * p * (1.0 - p)).sum() - 1.0 / sigma2)
        scale = float(np.sqrt(-1.0 / hess))
        return float(theta), scale

    def estimate(self, k: np.ndarray, N: np.ndarray,
                 beta: np.ndarray) -> tuple[float, float]:
        k = np.asarray(k, dtype=np.float64)
        N = np.asarray(N, dtype=np.float64)
        beta = np.asarray(beta, dtype=np.float64)
        if k.shape != N.shape or k.shape != beta.shape:
            raise ValueError(f"shape mismatch: k={k.shape}, N={N.shape}, beta={beta.shape}")

        # Step 1: locate the mode and its local scale
        mode, scale = self._find_mode(k, N, beta)

        # Step 2: shifted+scaled GH nodes, integrand = posterior density × jacobian
        theta_nodes = mode + scale * self._gh_nodes
        ll = _log_likelihood(theta_nodes, k, N, beta)              # (Q,)
        # log of unnormalized posterior at each node
        log_prior = -0.5 * ((theta_nodes - self.prior_mu) / self.prior_sigma) ** 2 \
                    - np.log(self.prior_sigma * np.sqrt(2.0 * np.pi))
        # GH rule: ∫ g(θ) dθ ≈ s · Σ w_q exp(x_q²/2) g(mode + s x_q)
        # Inside the integral g(θ) = posterior_unnormalized(θ).
        log_integrand = log_prior + ll + 0.5 * self._gh_nodes ** 2
        log_w = np.log(self._gh_weights + 1e-300) + log_integrand
        log_w_max = log_w.max()
        w = np.exp(log_w - log_w_max) * scale
        Z = w.sum()
        if not np.isfinite(Z) or Z <= 0:
            raise RuntimeError("EAP normalization underflow")
        post = w / Z
        mean = float((post * theta_nodes).sum())
        var = float((post * (theta_nodes - mean) ** 2).sum())
        return mean, float(np.sqrt(max(var, 0.0)))


class MAPLaplaceEstimator:
    """Posterior mode via Newton; SE from observed Fisher at the mode.

    Log-posterior (up to const):
        ℓ(θ) = Σ_j [k_j log σ(θ-β_j) + (N_j-k_j) log(1-σ(θ-β_j))] − (θ-μ)²/(2σ²)

    Score:
        s(θ)  = Σ_j (k_j − N_j σ(θ-β_j)) − (θ-μ)/σ²
    Hessian:
        H(θ)  = − Σ_j N_j σ(θ-β_j)(1-σ(θ-β_j)) − 1/σ²

    Newton: θ ← θ − s/H. Converges in ~5 iters for typical inputs.
    SE = sqrt(−1/H_at_mode); H<0 by construction so the inner expression > 0.
    """

    THETA_CLAMP: float = 10.0  # see EAPEstimator.THETA_CLAMP

    def __init__(self, prior_mu: float = 0.0, prior_sigma: float = 1.5,
                 max_iter: int = 50, tol: float = 1e-8):
        if prior_sigma <= 0:
            raise ValueError("prior_sigma must be > 0")
        self.prior_mu = float(prior_mu)
        self.prior_sigma = float(prior_sigma)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def estimate(self, k: np.ndarray, N: np.ndarray,
                 beta: np.ndarray) -> tuple[float, float]:
        k = np.asarray(k, dtype=np.float64)
        N = np.asarray(N, dtype=np.float64)
        beta = np.asarray(beta, dtype=np.float64)
        if k.shape != N.shape or k.shape != beta.shape:
            raise ValueError(f"shape mismatch: k={k.shape}, N={N.shape}, beta={beta.shape}")

        sigma2 = self.prior_sigma ** 2
        theta = float(self.prior_mu)
        for _ in range(self.max_iter):
            p = _sigmoid(theta - beta)
            score = float((k - N * p).sum() - (theta - self.prior_mu) / sigma2)
            hess = float(-(N * p * (1.0 - p)).sum() - 1.0 / sigma2)
            step = score / hess  # H < 0, so this moves toward the mode
            theta_new = theta - step
            theta_new = float(np.clip(theta_new, -self.THETA_CLAMP, self.THETA_CLAMP))
            if abs(theta_new - theta) < self.tol:
                theta = theta_new
                break
            theta = theta_new

        # Re-evaluate Hessian at the mode for SE
        p = _sigmoid(theta - beta)
        hess = float(-(N * p * (1.0 - p)).sum() - 1.0 / sigma2)
        se = float(np.sqrt(-1.0 / hess))
        return theta, se


def build_estimator(name: str, cfg: dict) -> PosteriorEstimator:
    """Factory: `name` ∈ {'eap', 'laplace'}; `cfg` reads prior/quad fields."""
    name = name.lower()
    prior_mu = cfg.get("prior_mu", 0.0)
    prior_sigma = cfg.get("prior_sigma", 1.5)
    if name == "eap":
        return EAPEstimator(
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            n_quad=cfg.get("n_quad", 21),
        )
    if name == "laplace":
        return MAPLaplaceEstimator(
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            max_iter=cfg.get("max_iter", 50),
            tol=cfg.get("tol", 1e-8),
        )
    raise ValueError(f"unknown estimator: {name!r}; choose 'eap' or 'laplace'")
