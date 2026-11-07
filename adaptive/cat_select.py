"""Fisher-info computerized adaptive testing (CAT) item selection.

For 1PL Rasch with N trials per item, the Fisher information at θ for an
item with difficulty β is
    I(θ, β) = N · σ(θ - β) · (1 - σ(θ - β))
maximized at θ = β with I_max = N/4. The CAT rule picks, among unused items,
the one whose β is closest to the current θ̂ — this minimizes posterior
variance of the next θ̂ update.
"""

from __future__ import annotations

import numpy as np


def fisher_info_select(theta_hat: float, beta: np.ndarray,
                       used_mask: np.ndarray, N: int = 40) -> int:
    """Return the unused-item index with maximum Fisher info at θ̂.

    Args:
        theta_hat: current ability estimate.
        beta: shape (J,) anchor item difficulties.
        used_mask: shape (J,) bool; True = already evaluated.
        N: trials per item (scales information uniformly; we keep it for
           explicitness even though argmax is invariant in N).

    Returns:
        Index in [0, J) of the selected unused item.
    """
    beta = np.asarray(beta, dtype=np.float64)
    used = np.asarray(used_mask, dtype=bool)
    if used.shape != beta.shape:
        raise ValueError(f"shape mismatch: beta={beta.shape}, used_mask={used.shape}")
    if used.all():
        raise ValueError("no unused items left")

    diff = theta_hat - beta
    p = np.where(diff >= 0, 1.0 / (1.0 + np.exp(-diff)),
                 np.exp(diff) / (1.0 + np.exp(diff)))
    info = N * p * (1.0 - p)                 # max at θ̂ = β
    info = np.where(used, -np.inf, info)
    return int(np.argmax(info))


def pilot_indices(beta: np.ndarray, quantiles: list[float]) -> list[int]:
    """Pick anchor items closest to the empirical β quantiles for the pilot stage.

    Each quantile maps to the index of the unused anchor item whose β is
    closest to that quantile of `beta`. Duplicates within the same call are
    avoided greedily (ties broken by item index).
    """
    beta = np.asarray(beta, dtype=np.float64)
    if len(beta) < len(quantiles):
        raise ValueError(f"len(beta)={len(beta)} < len(quantiles)={len(quantiles)}")
    targets = np.quantile(beta, quantiles)
    chosen: list[int] = []
    used = np.zeros(len(beta), dtype=bool)
    for q in targets:
        d = np.abs(beta - q)
        d = np.where(used, np.inf, d)
        idx = int(np.argmin(d))
        chosen.append(idx)
        used[idx] = True
    return chosen
