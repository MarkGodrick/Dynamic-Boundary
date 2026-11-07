"""Rasch (1PL IRT) model fitting for difficulty calibration.

Three estimation methods:
  1. OLS on logit-transformed pass rates (analytic)
  2. MLE on binomial data (L-BFGS)
  3. Simple average baseline

References:
  - Rasch, G. (1960). Probabilistic Models for Some Intelligence and
    Attainment Tests.
  - Paper plan §Step 2: Global Difficulty Calibration
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_response_matrix(data_dir: str) -> dict:
    """Load all response_matrix.jsonl files from model subdirectories.

    Args:
        data_dir: Directory containing model subdirectories, each with
                  response_matrix.jsonl and metadata.json.

    Returns:
        Dict with keys:
            models: list[str] — sorted model names
            items: list[str] — sorted prompt IDs
            pass_counts: np.array (M × J)
            total_trials: np.array (M × J)
            p_hat: np.array (M × J)
    """
    data_path = Path(data_dir)

    raw = {}
    for model_dir in sorted(data_path.iterdir()):
        rm_path = model_dir / "response_matrix.jsonl"
        if not rm_path.exists():
            continue
        meta_path = model_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                model_name = json.load(f).get("model", model_dir.name)
        else:
            model_name = model_dir.name

        with open(rm_path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                pid = entry["prompt_id"]
                raw[(model_name, pid)] = (entry["pass_count"], entry["total_trials"])

    if not raw:
        raise ValueError(f"No response_matrix.jsonl found in {data_dir}")

    models = sorted({k[0] for k in raw})
    items = sorted({k[1] for k in raw})
    model_idx = {m: i for i, m in enumerate(models)}
    item_idx = {p: j for j, p in enumerate(items)}

    M, J = len(models), len(items)
    pass_counts = np.zeros((M, J), dtype=np.float64)
    total_trials = np.zeros((M, J), dtype=np.float64)

    for (m, p), (k, n) in raw.items():
        i, j = model_idx[m], item_idx[p]
        pass_counts[i, j] = k
        total_trials[i, j] = n

    p_hat = np.where(total_trials > 0, pass_counts / total_trials, 0.5)

    return {
        "models": models,
        "items": items,
        "pass_counts": pass_counts,
        "total_trials": total_trials,
        "p_hat": p_hat,
    }


# ---------------------------------------------------------------------------
# Method 1: OLS on logit-transformed data
# ---------------------------------------------------------------------------

def fit_ols(p_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS Rasch model on logit-transformed pass rates.

    θ̂_m = ȳ_{m·} − ȳ_{··}
    β̂_x = −(ȳ_{·x} − ȳ_{··})

    Uses clipping to [0.025, 0.975] to avoid logit(0) and logit(1).

    Returns:
        (theta, beta) with constraint Σ theta = 0.
    """
    p_smooth = np.clip(p_hat, 0.025, 0.975)
    y = np.log(p_smooth / (1 - p_smooth))

    row_mean = y.mean(axis=1)
    col_mean = y.mean(axis=0)
    grand_mean = y.mean()

    theta = row_mean - grand_mean
    beta = -(col_mean - grand_mean)

    theta -= theta.mean()
    return theta, beta


# ---------------------------------------------------------------------------
# Method 2: MLE on binomial data
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)),
    )


def _neg_log_likelihood(params, K, N, M, J):
    theta = params[:M]
    beta = params[M:]
    diff = theta[:, None] - beta[None, :]
    p = np.clip(_sigmoid(diff), 1e-10, 1 - 1e-10)
    ll = (K * np.log(p) + (N - K) * np.log(1 - p)).sum()
    return -ll


def _neg_log_likelihood_grad(params, K, N, M, J):
    theta = params[:M]
    beta = params[M:]
    diff = theta[:, None] - beta[None, :]
    p = np.clip(_sigmoid(diff), 1e-10, 1 - 1e-10)
    residual = K - N * p
    grad_theta = -residual.sum(axis=1)
    grad_beta = residual.sum(axis=0)
    return np.concatenate([grad_theta, grad_beta])


def fit_mle(
    pass_counts: np.ndarray,
    total_trials: np.ndarray,
    theta_init: np.ndarray,
    beta_init: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """MLE Rasch model via L-BFGS-B.

    Args:
        pass_counts: (M, J) pass count matrix.
        total_trials: (M, J) trial count matrix.
        theta_init: Initial theta (from OLS).
        beta_init: Initial beta (from OLS).

    Returns:
        (theta, beta, neg_loglik)
    """
    M, J = pass_counts.shape
    x0 = np.concatenate([theta_init, beta_init])

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(pass_counts, total_trials, M, J),
        jac=_neg_log_likelihood_grad,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    theta_mle = result.x[:M]
    beta_mle = result.x[M:]

    shift = theta_mle.mean()
    theta_mle -= shift
    beta_mle -= shift

    return theta_mle, beta_mle, result.fun


# ---------------------------------------------------------------------------
# Method 3: Simple average baseline
# ---------------------------------------------------------------------------

def fit_baseline(p_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simple average baseline.

    ability(m) = mean_x(p_hat_mx)
    difficulty(x) = 1 - mean_m(p_hat_mx)

    Returns:
        (ability, difficulty)
    """
    ability = p_hat.mean(axis=1)
    difficulty = 1 - p_hat.mean(axis=0)
    return ability, difficulty


# ---------------------------------------------------------------------------
# Fit statistics
# ---------------------------------------------------------------------------

def compute_fit_stats(
    p_hat: np.ndarray,
    pass_counts: np.ndarray,
    total_trials: np.ndarray,
    theta_ols: np.ndarray,
    beta_ols: np.ndarray,
    theta_mle: np.ndarray,
    beta_mle: np.ndarray,
    d_avg: np.ndarray,
    mle_nll: float,
) -> dict:
    """Compute fit statistics and method agreement metrics."""
    M, J = p_hat.shape

    # OLS: RMSE in logit space
    p_smooth = np.clip(p_hat, 0.025, 0.975)
    y_actual = np.log(p_smooth / (1 - p_smooth))
    y_pred_ols = theta_ols[:, None] - beta_ols[None, :]
    ols_rmse = float(np.sqrt(((y_actual - y_pred_ols) ** 2).mean()))

    # MLE: RMSE in probability space
    p_pred_mle = _sigmoid(theta_mle[:, None] - beta_mle[None, :])
    mle_p_rmse = float(np.sqrt(((p_hat - p_pred_mle) ** 2).mean()))

    # Rank correlations
    rho_ols_mle = float(spearmanr(beta_ols, beta_mle)[0])
    rho_mle_avg = float(spearmanr(beta_mle, d_avg)[0])
    rho_ols_avg = float(spearmanr(beta_ols, d_avg)[0])

    return {
        "n_models": M,
        "n_items": J,
        "ols_logit_rmse": round(ols_rmse, 4),
        "mle_p_rmse": round(mle_p_rmse, 4),
        "mle_neg_loglik": round(float(mle_nll), 2),
        "rank_corr_ols_mle": round(rho_ols_mle, 4),
        "rank_corr_mle_avg": round(rho_mle_avg, 4),
        "rank_corr_ols_avg": round(rho_ols_avg, 4),
    }
