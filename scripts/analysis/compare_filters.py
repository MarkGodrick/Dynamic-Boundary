"""Compare filter candidates on a fixed response-matrix pool.

Given a directory of per-model response matrices (produced by
collect_response_matrix.py or filter_random_compose.py), this script:

  1. Fits a joint Rasch model on the FULL pool to get θ_panel and β_item
     (Option A: shared β across all filter candidates).
  2. Applies a suite of filter candidates (AVG p̄, β range, AT-LEAST-ONE).
  3. For each candidate, computes the primary metric:
        max SE(θ) over target θ grid [panel θ_min − buffer, panel θ_max + buffer]
     plus secondary metrics (TIF stats, coverage, per-item efficiency).
  4. Prints a comparison table and saves a JSON report.

No GPU work. Inputs are pre-existing response matrices.

Usage:
    python scripts/analysis/compare_filters.py \\
        --response-dir results/constrained_completion/validation/edge_case \\
        --target-buffer 0.5 \\
        --n-grid 50 \\
        --output results/constrained_completion/filter_ablation_edge_case.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difficulty.rasch import load_response_matrix, fit_ols, fit_mle
from difficulty.irt_diagnostics import (
    _compute_logits,
    compute_standardized_residuals,
    compute_item_fit_diagnostics,
    compute_q3,
)


def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


# ---------------------------------------------------------------------------
# Filter candidates — each returns a boolean mask over items
# Signature: (p_hat, beta, panel_theta) → bool mask; unused args prefixed _
# ---------------------------------------------------------------------------

def filter_none(p_hat, _beta, _panel_theta):
    """Keep everything."""
    return np.ones(p_hat.shape[1], dtype=bool)


def filter_avg(p_hat, _beta, _panel_theta, lo, hi):
    """Panel-mean p̂ in [lo, hi]."""
    panel_mean = p_hat.mean(axis=0)
    return (panel_mean >= lo) & (panel_mean <= hi)


def filter_beta(_p_hat, beta, panel_theta, lo_offset, hi_offset):
    """β in [θ_min - lo_offset, θ_max + hi_offset]."""
    lo = panel_theta.min() - lo_offset
    hi = panel_theta.max() + hi_offset
    return (beta >= lo) & (beta <= hi)


def filter_at_least_one(p_hat, _beta, _panel_theta, lo, hi):
    """At least one model has p̂ in [lo, hi]."""
    return ((p_hat >= lo) & (p_hat <= hi)).any(axis=0)


def filter_low_variance(p_hat, _beta, _panel_theta,
                        std_threshold=0.05, mean_lo=0.05, mean_hi=0.95):
    """ATLAS §3.2 low-variance + extreme-mean removal.

    ATLAS original thresholds are single-trial: std < 1% or mean > 95%.
    In our N=40 Binomial setting, pure Binomial floor of cross-model pass-rate
    std at p=0.5 is sqrt(p(1-p)/N) ≈ 0.079, so ATLAS's 1% threshold is
    un-reachable by design. We adapt:

        - std_threshold = 0.05 (below ATLAS's semantic 'near-constant' bar,
          above Binomial floor)
        - mean bounds [0.05, 0.95] to exclude panel-uniform items

    An item is kept iff all three conditions hold.
    """
    panel_std = p_hat.std(axis=0)
    panel_mean = p_hat.mean(axis=0)
    return (panel_std >= std_threshold) & (panel_mean >= mean_lo) & (panel_mean <= mean_hi)


def filter_point_biserial(p_hat, _beta, _panel_theta, r_threshold=0.1):
    """ATLAS §3.2 point-biserial discrimination filter.

    r_pb(x) = Pearson corr(p_hat[:, x], T_m) where T_m = Σ_j p_hat[m, j].
    Low |r_pb| means the item's pass-rate pattern is uncorrelated with the
    panel's overall ability ordering — likely off the main ability axis
    (multi-dimensional contamination) or pure noise.

    Original ATLAS version uses binary responses → point-biserial (binary-
    continuous). Our pass rate is continuous, so this degenerates to regular
    Pearson correlation. Threshold 0.1 follows ATLAS convention.

    Returns: items with |r_pb| ≥ r_threshold.
    """
    panel_total = p_hat.sum(axis=1)  # (M,)
    p_centered = p_hat - p_hat.mean(axis=0, keepdims=True)
    t_centered = panel_total - panel_total.mean()
    num = (p_centered * t_centered[:, None]).sum(axis=0)
    denom = np.sqrt((p_centered ** 2).sum(axis=0) * (t_centered ** 2).sum())
    r_pb = np.where(denom > 0, num / denom, 0.0)
    return np.abs(r_pb) >= r_threshold


def filter_atlas_combined(p_hat, beta, panel_theta,
                          std_threshold=0.05, mean_lo=0.05, mean_hi=0.95,
                          r_threshold=0.1):
    """ATLAS §3.2 full pipeline: low-variance AND point-biserial, ANDed."""
    m1 = filter_low_variance(p_hat, beta, panel_theta,
                             std_threshold, mean_lo, mean_hi)
    m2 = filter_point_biserial(p_hat, beta, panel_theta, r_threshold)
    return m1 & m2


def build_filter_list(panel_theta):
    """Return list of (name, config_dict, fn) tuples."""
    filters = []
    filters.append((
        "NO FILTER",
        {"type": "none"},
        lambda p, b, t: filter_none(p, b, t),
    ))
    for lo, hi in [(0.05, 0.95), (0.10, 0.90), (0.20, 0.80),
                   (0.30, 0.70), (0.40, 0.60)]:
        filters.append((
            f"AVG [{lo:.2f}, {hi:.2f}]",
            {"type": "avg", "lo": lo, "hi": hi},
            lambda p, b, t, lo=lo, hi=hi: filter_avg(p, b, t, lo, hi),
        ))
    for off in [0.25, 0.50, 1.00, 1.50]:
        t_lo = panel_theta.min() - off
        t_hi = panel_theta.max() + off
        filters.append((
            f"β [θmin-{off:.2f}, θmax+{off:.2f}]"
            f"  ([{t_lo:+.2f}, {t_hi:+.2f}])",
            {"type": "beta", "offset": off, "lo": t_lo, "hi": t_hi},
            lambda p, b, t, off=off: filter_beta(p, b, t, off, off),
        ))
    for lo, hi in [(0.20, 0.80), (0.30, 0.70)]:
        filters.append((
            f"AT-LEAST-ONE [{lo:.2f}, {hi:.2f}]",
            {"type": "at_least_one", "lo": lo, "hi": hi},
            lambda p, b, t, lo=lo, hi=hi: filter_at_least_one(p, b, t, lo, hi),
        ))
    # ATLAS §3.2 filters (std + mean + point-biserial)
    filters.append((
        "ATLAS low-var [std≥0.05, mean∈[0.05,0.95]]",
        {"type": "atlas_low_variance", "std_threshold": 0.05,
         "mean_lo": 0.05, "mean_hi": 0.95},
        lambda p, b, t: filter_low_variance(p, b, t, 0.05, 0.05, 0.95),
    ))
    filters.append((
        "ATLAS point-biserial [|r_pb|≥0.10]",
        {"type": "atlas_point_biserial", "r_threshold": 0.1},
        lambda p, b, t: filter_point_biserial(p, b, t, 0.1),
    ))
    filters.append((
        "ATLAS combined [low-var AND point-biserial]",
        {"type": "atlas_combined", "std_threshold": 0.05,
         "mean_lo": 0.05, "mean_hi": 0.95, "r_threshold": 0.1},
        lambda p, b, t: filter_atlas_combined(p, b, t, 0.05, 0.05, 0.95, 0.1),
    ))
    return filters


# ---------------------------------------------------------------------------
# Pareto frontier on (budget_SE ↓, FIR ↑)
# ---------------------------------------------------------------------------

def compute_pareto_frontier(results: list) -> list:
    """Return names of filters non-dominated on (budget_SE min, FIR max).

    A filter f is DOMINATED iff there exists g with:
        budget_SE[g] ≤ budget_SE[f]  AND  FIR[g] ≥ FIR[f]
        with at least one strict inequality.
    Non-dominated filters sit on the Pareto frontier; neither axis can
    improve without hurting the other.

    Rationale: budget_SE and FIR measure complementary properties —
    budget_SE rewards per-item info density (Axis I), FIR rewards not
    discarding signal (Axis II). A filter dominating on both axes is
    strictly better; others make a trade-off that needs a value judgment.
    """
    n = len(results)
    pareto_names = []
    for i in range(n):
        bs_i = results[i].get("budget_se", float("inf"))
        fir_i = results[i].get("fir", 0.0)
        if not np.isfinite(bs_i):
            continue
        dominated = False
        for j in range(n):
            if i == j:
                continue
            bs_j = results[j].get("budget_se", float("inf"))
            fir_j = results[j].get("fir", 0.0)
            if not np.isfinite(bs_j):
                continue
            if (bs_j <= bs_i and fir_j >= fir_i
                    and (bs_j < bs_i or fir_j > fir_i)):
                dominated = True
                break
        if not dominated:
            pareto_names.append(results[i]["name"])
    return pareto_names


def check_fit_thresholds(result: dict, thresholds: dict,
                         irt_model: str = "1pl") -> dict:
    """Return per-axis hard-threshold compliance flags.

    Thresholds are INFORMATIONAL (not filter pre-screens). Edge-case
    filter ablation data is intrinsically IRT-mis-specified (see
    smoke-test findings 2026-04-20), so absolute fit thresholds kill
    most candidates. We report per-filter compliance and let the reader
    judge; primary selection still relies on budget_SE × FIR Pareto.

    Returns dict with boolean flags per check.
    """
    diag = result.get("diagnostics_by_irt_model", {}).get(irt_model, {})
    # mean_z² uses Bernoulli-equivalent (N-invariant) for threshold check;
    # Binomial mean_z² is ×N-amplified and not a fit score in isolation.
    # Same reasoning for Infit: Bernoulli-equivalent pass rate is the
    # defensible version. See reading guide + §9.4.1.
    return {
        "pseudo_r2_ok": (diag.get("pseudo_r2", float("nan"))
                        >= thresholds.get("pseudo_r2", -np.inf)),
        "mean_z_sq_ok": (diag.get("mean_z_sq_bernoulli", float("nan"))
                         <= thresholds.get("mean_z_sq", np.inf)),
        "infit_pass_ok": (diag.get("infit_bernoulli_pass_rate", float("nan"))
                          >= thresholds.get("infit_pass_rate", 0.0)),
        "q3_viol_ok": (diag.get("q3_violation_rate", float("nan"))
                       <= thresholds.get("q3_violation_rate", 1.0)),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_se_grid(beta_subset, N, target_grid):
    """SE(θ) over target grid, using only items in subset."""
    if len(beta_subset) == 0:
        return np.full(len(target_grid), np.inf)
    se = np.zeros(len(target_grid))
    for i, t in enumerate(target_grid):
        p = sigmoid(t - beta_subset)
        info = (N * p * (1 - p)).sum()
        se[i] = (1.0 / np.sqrt(info)) if info > 0 else np.inf
    return se


def compute_tif_at_panel(beta_subset, panel_theta):
    """TIF(θ_m) = Σ_x p(1-p) for each panel model m."""
    if len(beta_subset) == 0:
        return np.zeros(len(panel_theta))
    tif = np.zeros(len(panel_theta))
    for i, t in enumerate(panel_theta):
        p = sigmoid(t - beta_subset)
        tif[i] = (p * (1 - p)).sum()
    return tif


def beta_coverage_gap(beta_subset):
    """Largest adjacent gap in sorted β values."""
    if len(beta_subset) < 2:
        return float("inf")
    sorted_b = np.sort(beta_subset)
    return float(np.max(np.diff(sorted_b)))


def compute_q_bar_all(p_hat: np.ndarray) -> np.ndarray:
    """Panel-average per-item Fisher information q̄_x = mean_m p_mx (1 - p_mx).

    Pre-computed once before the filter loop; used for FIR (Axis II).
    Shape: (J,) where J = total items in pool.
    """
    return (p_hat * (1 - p_hat)).mean(axis=0)


def compute_fir(q_bar_all: np.ndarray, mask: np.ndarray) -> float:
    """Fisher Info Retention ratio (Axis II primary).

    FIR(F) = Σ_{x ∈ F} q̄_x / Σ_{x} q̄_x ∈ [0, 1]

    See docs/filter_metric_construction.md §9.1 for derivation.
    Intuition: how much of the total (panel-averaged) Fisher info does the
    filter retain? Complementary to budget_SE (which only sees kept items).
    FIR = 1.0 for NO FILTER; stricter filters have lower FIR.
    """
    total = float(q_bar_all.sum())
    if total <= 0:
        return float("nan")
    return float(q_bar_all[mask].sum() / total)


# ---------------------------------------------------------------------------
# Axis III: per-item Rasch fit diagnostics (retained-set independent fit)
# ---------------------------------------------------------------------------

def fit_irt_on_subset(pass_counts_sub: np.ndarray,
                      N_sub: np.ndarray,
                      irt_model: str = "1pl") -> tuple:
    """Option B: independent IRT fit on the retained subset only.

    Each filter candidate gets its own parameter estimates, fit only on the
    items it retains. This is the methodologically correct basis for
    per-item and global fit diagnostics (Axis III and IV).

    Args:
        pass_counts_sub: (M, n_kept) Binomial pass counts.
        N_sub: (M, n_kept) Binomial trial counts.
        irt_model: "1pl" (Rasch) / "2pl" / "mirt-2d" / "mirt-3d".

    Returns:
        (theta, beta, alpha, fit_ok)
        - 1PL: theta (M,), beta (J,), alpha = ones(J,)
        - 2PL: theta (M,), beta (J,), alpha (J,)
        - MIRT-Dd: theta (M, D), beta (J,), alpha (J, D)
        - fit_ok=False on failure
    """
    try:
        from difficulty.rasch import fit_ols as _fit_ols, fit_mle as _fit_mle_1pl
        p_hat_sub = pass_counts_sub / np.maximum(N_sub, 1)

        if irt_model == "ctt":
            # Classical Test Theory baseline: use OLS (logit-of-pass-rate)
            # only, skip MLE refinement. Equivalent to the 1PL warm-start
            # estimator. β̂_x ∝ -logit(mean_m p̂), θ̂_m = logit(mean_x p̂) -
            # grand_mean. No probabilistic fit, no convergence issues.
            theta, beta = _fit_ols(p_hat_sub)
            alpha = np.ones(beta.shape[0])
        elif irt_model == "1pl":
            theta_init, beta_init = _fit_ols(p_hat_sub)
            theta, beta, _nll = _fit_mle_1pl(pass_counts_sub, N_sub,
                                             theta_init, beta_init)
            alpha = np.ones(beta.shape[0])
        elif irt_model == "2pl":
            from difficulty.irt_2pl import fit_mle as _fit_mle_2pl
            theta, beta, alpha, _nll = _fit_mle_2pl(pass_counts_sub, N_sub)
        elif irt_model.startswith("mirt-") and irt_model.endswith("d"):
            from difficulty.mirt import fit_mle as _fit_mle_mirt
            D = int(irt_model[5:-1])  # "mirt-2d" → 2
            theta, beta, alpha, _nll = _fit_mle_mirt(pass_counts_sub, N_sub, D=D)
        else:
            raise ValueError(f"Unknown irt_model: {irt_model!r}")

        if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(beta))
                and np.all(np.isfinite(alpha))):
            return None, None, None, False
        return theta, beta, alpha, True
    except Exception:
        return None, None, None, False


# ---------------------------------------------------------------------------
# Axis IV primary: held-out Binomial pseudo-R²
# ---------------------------------------------------------------------------

def _binomial_log_likelihood(k: np.ndarray, N: np.ndarray, p: np.ndarray) -> float:
    """Σ Binomial log-likelihood (ignores log C(N, k) normalizer since it
    cancels across model comparisons on the same (k, N) holdout).

    Sum over all cells; p is clipped to avoid log(0).
    """
    p_clip = np.clip(p, 1e-10, 1 - 1e-10)
    return float((k * np.log(p_clip) + (N - k) * np.log(1 - p_clip)).sum())


def compute_holdout_pseudo_r2(pass_counts: np.ndarray,
                              N_trials: np.ndarray,
                              seed: int = 42,
                              holdout_frac: float = 0.2,
                              irt_model: str = "1pl") -> dict:
    """Axis IV primary: pseudo-R² on a random holdout of cells.

    Protocol:
        1. Fix seed, randomly assign 20% of (m, x) cells to holdout.
        2. Fit IRT on training cells only (masked MLE for 1PL; zero-weight
           approximation for 2PL, see note below).
        3. Score holdout cells under fitted model vs. two baselines:
             LL_model = Σ BinomLL(k, N, p̂_model)
             LL_null  = Σ BinomLL(k, N, 0.5)       (no-information baseline)
             LL_sat   = Σ BinomLL(k, N, k / N)     (saturated upper bound)
        4. pseudo_r2 = (LL_model - LL_null) / (LL_sat - LL_null)
           ∈ [0, 1] in theory; can be negative if model fits worse than p=0.5.

    Does NOT rely on asymptotic χ² theory — works at M=6. Suitable as
    Axis IV primary metric.

    For 1PL: uses _fit_mle_masked (exact cell mask honored in likelihood).
    For 2PL: zeroes out holdout cells in (k, N) before fitting. This is an
    approximation (not equivalent to proper masked 2PL MLE) but gives a
    valid held-out evaluation for filter comparison.
    """
    from difficulty.rasch import fit_ols as _fit_ols

    M, J = pass_counts.shape
    if M < 2 or J < 2:
        return {"pseudo_r2": float("nan"), "ll_model": float("nan"),
                "ll_null": float("nan"), "ll_sat": float("nan"),
                "n_holdout": 0, "n_train": 0, "seed": seed,
                "fit_ok": False}

    rng = np.random.RandomState(seed)
    n_cells = M * J
    n_holdout = max(1, int(round(holdout_frac * n_cells)))
    flat_idx = rng.permutation(n_cells)
    holdout_flat = flat_idx[:n_holdout]
    train_flat = flat_idx[n_holdout:]
    holdout_mask = np.zeros((M, J), dtype=bool)
    holdout_mask.flat[holdout_flat] = True
    train_mask = ~holdout_mask

    # Safety check: every model and every item must have ≥ 1 training cell,
    # else fit is degenerate (infinite θ or β).
    if (train_mask.any(axis=1).sum() < M) or (train_mask.any(axis=0).sum() < J):
        return {"pseudo_r2": float("nan"), "ll_model": float("nan"),
                "ll_null": float("nan"), "ll_sat": float("nan"),
                "n_holdout": n_holdout, "n_train": n_cells - n_holdout,
                "seed": seed, "fit_ok": False,
                "reason": "holdout split left some row or column empty"}

    # Initialize from OLS on training p̂ (use training mean for missing cells)
    p_hat_train = np.where(train_mask, pass_counts / np.maximum(N_trials, 1), np.nan)
    # Impute missing cells with row/col mean for OLS init (just a warm start)
    col_means = np.nanmean(p_hat_train, axis=0, keepdims=True)
    p_hat_imputed = np.where(np.isnan(p_hat_train),
                             np.broadcast_to(col_means, p_hat_train.shape),
                             p_hat_train)
    # If any column was entirely NaN (shouldn't happen after safety check),
    # fall back to 0.5
    p_hat_imputed = np.where(np.isnan(p_hat_imputed), 0.5, p_hat_imputed)

    try:
        theta_init, beta_init = _fit_ols(p_hat_imputed)
        if irt_model == "ctt":
            # CTT: OLS on train cells only (via imputed p_hat matrix, which
            # imputes holdout cells with col/grand means). No MLE refinement.
            theta, beta = theta_init, beta_init
            alpha = np.ones(beta.shape[0])
        elif irt_model == "1pl":
            theta, beta, _nll = _fit_mle_masked(pass_counts, N_trials, train_mask,
                                                theta_init, beta_init)
            alpha = np.ones(beta.shape[0])
        elif irt_model == "2pl":
            # Zero-weight approximation for 2PL: mask holdout cells by zeroing
            # both k and N (contribute zero to likelihood).
            from difficulty.irt_2pl import fit_mle as _fit_mle_2pl
            train_mask_f = train_mask.astype(float)
            pc_train = pass_counts * train_mask_f
            N_train = N_trials * train_mask_f
            theta, beta, alpha, _nll = _fit_mle_2pl(pc_train, N_train)
        elif irt_model.startswith("mirt-") and irt_model.endswith("d"):
            # Same zero-weight mask approximation for MIRT
            from difficulty.mirt import fit_mle as _fit_mle_mirt
            D = int(irt_model[5:-1])
            train_mask_f = train_mask.astype(float)
            pc_train = pass_counts * train_mask_f
            N_train = N_trials * train_mask_f
            theta, beta, alpha, _nll = _fit_mle_mirt(pc_train, N_train, D=D)
        else:
            raise ValueError(f"Unknown irt_model: {irt_model!r}")

        if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(beta))
                and np.all(np.isfinite(alpha))):
            return {"pseudo_r2": float("nan"), "ll_model": float("nan"),
                    "ll_null": float("nan"), "ll_sat": float("nan"),
                    "n_holdout": n_holdout, "n_train": n_cells - n_holdout,
                    "seed": seed, "fit_ok": False, "reason": "non-finite fit"}
    except Exception as e:
        return {"pseudo_r2": float("nan"), "ll_model": float("nan"),
                "ll_null": float("nan"), "ll_sat": float("nan"),
                "n_holdout": n_holdout, "n_train": n_cells - n_holdout,
                "seed": seed, "fit_ok": False, "reason": f"fit exception: {e}"}

    # Score holdout cells — use unified logit helper
    k_h = pass_counts[holdout_mask]
    n_h = N_trials[holdout_mask]
    logits_full = _compute_logits(theta, beta, alpha)
    p_pred_h = 1.0 / (1.0 + np.exp(-logits_full[holdout_mask]))

    ll_model = _binomial_log_likelihood(k_h, n_h, p_pred_h)
    ll_null = _binomial_log_likelihood(k_h, n_h, np.full_like(k_h, 0.5, dtype=float))
    # Saturated: p = k/N per cell; handle N=0 gracefully
    p_sat = np.where(n_h > 0, k_h / np.maximum(n_h, 1), 0.5)
    ll_sat = _binomial_log_likelihood(k_h, n_h, p_sat)

    denom = ll_sat - ll_null
    if denom <= 0:
        # Degenerate holdout (e.g. all cells happen to have p=0.5 observed)
        pseudo_r2 = float("nan")
    else:
        pseudo_r2 = float((ll_model - ll_null) / denom)

    return {
        "pseudo_r2": pseudo_r2,
        "ll_model": ll_model,
        "ll_null": ll_null,
        "ll_sat": ll_sat,
        "n_holdout": int(n_holdout),
        "n_train": int(n_cells - n_holdout),
        "seed": seed,
        "fit_ok": True,
    }


def _fit_mle_masked(pass_counts: np.ndarray,
                    total_trials: np.ndarray,
                    train_mask: np.ndarray,
                    theta_init: np.ndarray,
                    beta_init: np.ndarray) -> tuple:
    """Masked-cell MLE for 1PL Rasch. Cells with train_mask=False are
    excluded from the likelihood (used for holdout evaluation).

    Mirrors difficulty/rasch._neg_log_likelihood but applies a mask.
    """
    from scipy.optimize import minimize

    M, J = pass_counts.shape
    mask_f = train_mask.astype(float)

    def neg_ll(params):
        theta = params[:M]
        beta = params[M:]
        diff = theta[:, None] - beta[None, :]
        # Numerically stable sigmoid
        p = np.where(diff >= 0,
                     1 / (1 + np.exp(-diff)),
                     np.exp(diff) / (1 + np.exp(diff)))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        ll = mask_f * (pass_counts * np.log(p)
                       + (total_trials - pass_counts) * np.log(1 - p))
        return -ll.sum()

    def neg_ll_grad(params):
        theta = params[:M]
        beta = params[M:]
        diff = theta[:, None] - beta[None, :]
        p = np.where(diff >= 0,
                     1 / (1 + np.exp(-diff)),
                     np.exp(diff) / (1 + np.exp(diff)))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        residual = mask_f * (pass_counts - total_trials * p)
        grad_theta = -residual.sum(axis=1)
        grad_beta = residual.sum(axis=0)
        return np.concatenate([grad_theta, grad_beta])

    x0 = np.concatenate([theta_init, beta_init])
    result = minimize(neg_ll, x0, jac=neg_ll_grad, method="L-BFGS-B",
                      options={"maxiter": 1000, "ftol": 1e-10})
    theta_mle = result.x[:M]
    beta_mle = result.x[M:]
    return theta_mle, beta_mle, float(result.fun)


def compute_m2_rmsea(pass_counts: np.ndarray,
                     N_trials: np.ndarray,
                     theta: np.ndarray,
                     beta: np.ndarray,
                     alpha: np.ndarray = None) -> dict:
    """Limited-information fit statistic (paper tertiary, ATLAS comparison).

    Simplified Maydeu-Olivares (2013)-style M_2 + RMSEA. Full M_2 requires
    inversion of a large dense Σ matrix (J² + J marginals). We use a
    diagonal-Σ approximation:

        χ²_uni = Σ_x (π̂_x - E[π_x])² / Var[π̂_x]
        χ²_bi  = Σ_{x<y} (π̂_xy - E[π_xy])² / Var[π̂_xy]
        M₂ ≈ χ²_uni + χ²_bi
        df = (J + J(J-1)/2) - n_params
        RMSEA = sqrt(max(0, (M₂/df - 1) / (M - 1)))

    Protocol (Option A binarization):
        binary[m, x] = 1 if k_mx > N_mx/2 else 0
    This aligns with method_pipeline.md M2 §decision B.

    NOT exact M_2 — the diagonal Σ ignores cross-marginal covariance, so
    the test is conservative (absolute M₂ value may be biased but relative
    comparisons across filters remain valid).

    Caveat: M=6 makes asymptotic χ² weak. RMSEA here is directional, not
    a strict Type-I-error test. Use for paper reporting with this caveat
    explicit; pseudo_R² on held-out cells is the preferred Axis IV metric.
    """
    M, J = pass_counts.shape
    if M < 2 or J < 2:
        return {"m2": float("nan"), "m2_df": 0, "rmsea": float("nan"),
                "method": "diagonal-Σ approximation", "M": M, "J": J}

    # Binarize (Option A)
    binary = (pass_counts > N_trials / 2.0).astype(float)

    # Observed marginals
    obs_uni = binary.mean(axis=0)                              # (J,)
    obs_bi_mat = (binary.T @ binary) / M                       # (J, J) — E[Y_x Y_y]

    # Model-implied cell probabilities (unified 1PL/2PL/MIRT)
    logits = _compute_logits(theta, beta, alpha)
    p = 1.0 / (1.0 + np.exp(-logits))                          # (M, J)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    imp_uni = p.mean(axis=0)                                   # (J,)
    imp_bi_mat = (p.T @ p) / M                                 # (J, J) under local indep.

    # Univariate χ² (diagonal Σ = p(1-p)/M)
    var_uni = np.maximum(imp_uni * (1 - imp_uni) / M, 1e-10)
    chi_uni = ((obs_uni - imp_uni) ** 2 / var_uni).sum()

    # Bivariate χ² — upper triangle only
    iu = np.triu_indices(J, k=1)
    obs_bi = obs_bi_mat[iu]
    imp_bi = imp_bi_mat[iu]
    var_bi = np.maximum(imp_bi * (1 - imp_bi) / M, 1e-10)
    chi_bi = ((obs_bi - imp_bi) ** 2 / var_bi).sum()

    m2 = float(chi_uni + chi_bi)
    # Degrees of freedom: # marginals − # params
    n_marginals = J + J * (J - 1) // 2
    n_params = M + J + (J if alpha is not None else 0)
    df = max(1, n_marginals - n_params)

    # RMSEA (M-1 degrees, per ATLAS convention for the per-person scale)
    rmsea = float(np.sqrt(max(0.0, (m2 / df - 1) / max(M - 1, 1))))

    return {
        "m2": m2,
        "m2_df": int(df),
        "rmsea": rmsea,
        "chi_uni": float(chi_uni),
        "chi_bi": float(chi_bi),
        "method": "diagonal-Σ approximation (Maydeu-Olivares 2013-style)",
        "M": int(M),
        "J": int(J),
        "caveat": "diagonal Σ; M=6 weakens asymptotic χ² — treat RMSEA as directional",
    }


def compute_pairwise_resolution(beta_sub, N, panel_theta):
    """Pairwise resolution metrics using filter's kept items.

    Derivation: see docs/irt_concepts_primer.md §3.5.4 and §3.5.5.

    For each pair of panel models (i, j):
        SE_m = 1 / sqrt(Σ_{x in kept} N · p_m(1-p_m)),
               p_m = σ(θ_m - β_x)
        res_ij = |θ_i - θ_j| / sqrt(SE_i² + SE_j²)
                 (non-centrality parameter = benchmark resolution for this pair)
        CI disjoint at α (conservative disjoint-CI test):
            |θ_i - θ_j| > z_{α/2} · (SE_i + SE_j)

    Aggregated over C(M,2) pairs:
        - mean_res, min_res, max_res
        - frac_ci_disjoint at 95% and 99% levels

    Intensive vs extensive: mean_res is extensive (∝ sqrt(|kept|)), so
    NO FILTER tends to win on it. The "filter contribution" is captured
    by mean_res / sqrt(|kept|) ≈ per-item resolution density.
    """
    M = len(panel_theta)
    if len(beta_sub) == 0:
        return {
            "se_per_panel": [],
            "res_per_pair": [],
            "mean_res": 0.0,
            "min_res": 0.0,
            "max_res": 0.0,
            "frac_ci_disjoint_95": 0.0,
            "frac_ci_disjoint_99": 0.0,
            "n_pairs": 0,
        }

    # SE at each panel θ (Fisher info at θ_m from filter-kept items)
    p_panel = sigmoid(panel_theta[:, None] - beta_sub[None, :])  # (M, J_kept)
    info_panel = N * (p_panel * (1 - p_panel)).sum(axis=1)       # (M,)
    se_panel = np.where(info_panel > 0, 1.0 / np.sqrt(info_panel), np.inf)

    res_list = []
    pair_records = []
    disjoint_95 = 0
    disjoint_99 = 0
    n_pairs = 0
    for i in range(M):
        for j in range(i + 1, M):
            dtheta = abs(panel_theta[i] - panel_theta[j])
            sigma_delta = float(np.sqrt(se_panel[i] ** 2 + se_panel[j] ** 2))
            if sigma_delta == 0 or not np.isfinite(sigma_delta):
                continue
            res = dtheta / sigma_delta
            res_list.append(res)
            # Disjoint CI at 95%: |Δθ| > 1.96 (SE_i + SE_j)
            # Disjoint CI at 99%: |Δθ| > 2.576 (SE_i + SE_j)
            ci_95 = dtheta > 1.96 * (se_panel[i] + se_panel[j])
            ci_99 = dtheta > 2.576 * (se_panel[i] + se_panel[j])
            disjoint_95 += int(ci_95)
            disjoint_99 += int(ci_99)
            pair_records.append({
                "i": int(i), "j": int(j),
                "delta_theta": float(dtheta),
                "se_i": float(se_panel[i]),
                "se_j": float(se_panel[j]),
                "res": float(res),
                "ci_disjoint_95": bool(ci_95),
                "ci_disjoint_99": bool(ci_99),
            })
            n_pairs += 1

    if n_pairs == 0 or not res_list:
        return {
            "se_per_panel": [float(x) for x in se_panel],
            "res_per_pair": [],
            "mean_res": 0.0, "min_res": 0.0, "max_res": 0.0,
            "frac_ci_disjoint_95": 0.0, "frac_ci_disjoint_99": 0.0,
            "n_pairs": 0,
        }

    return {
        "se_per_panel": [float(x) for x in se_panel],
        "res_per_pair": pair_records,
        "mean_res": float(np.mean(res_list)),
        "min_res": float(np.min(res_list)),
        "max_res": float(np.max(res_list)),
        "frac_ci_disjoint_95": disjoint_95 / n_pairs,
        "frac_ci_disjoint_99": disjoint_99 / n_pairs,
        "n_pairs": n_pairs,
    }


def compute_pairwise_pass_rate_diff(beta_sub, panel_theta):
    """Mean pairwise pass-rate difference D_ij (intensive metric).

    D_ij = (1 / |kept|) Σ_x |p_i^x - p_j^x|
    where p_m^x = σ(θ_m - β_x).

    Derivation (docs/irt_concepts_primer.md §3.5.6): Taylor expansion at
    the midpoint θ_c = (θ_i + θ_j)/2 gives
        D_ij ≈ |Δθ_ij| · q̄_c
    so D_ij is the pure per-item info density metric (intensive:
    independent of |kept|). NO FILTER has no natural advantage here.

    Returns mean / min / max across C(M,2) pairs.
    """
    M = len(panel_theta)
    if len(beta_sub) == 0:
        return {"mean_d": 0.0, "min_d": 0.0, "max_d": 0.0}

    p = sigmoid(panel_theta[:, None] - beta_sub[None, :])  # (M, J_kept)
    d_list = []
    for i in range(M):
        for j in range(i + 1, M):
            d_list.append(float(np.abs(p[i] - p[j]).mean()))

    if not d_list:
        return {"mean_d": 0.0, "min_d": 0.0, "max_d": 0.0}
    return {
        "mean_d": float(np.mean(d_list)),
        "min_d": float(np.min(d_list)),
        "max_d": float(np.max(d_list)),
    }


def evaluate_filter(name, cfg, mask, beta, _p_hat, pass_counts, N_trials,
                    panel_theta, target_grid, n_ref, q_bar_all,
                    irt_model: str = "1pl"):
    """Compute all metrics for one filter candidate.

    Primary metric (Axis I): budget-normalized max SE (Framing C).
        budget_se = max_se * sqrt(n_kept / n_ref)

    Under uniform per-item Fisher info, SE ∝ 1/sqrt(n), so SE·sqrt(n) is
    invariant under random subsetting. budget_se is therefore independent
    of n_kept and only rewards filters that keep genuinely higher-quality
    items (higher per-item info at target θ). Coverage gaps at extreme
    target θ are still caught because max over grid catches them.

    Axis II metric: FIR (Fisher Info Retention) = fraction of total
    panel-averaged Fisher info retained by the filter.

    Axis III metrics: Infit / Outfit pass rates (retained-set IRT fit).
    Axis IV: pseudo_r² (primary) + mean_z² (secondary).

    Args:
        irt_model: "1pl" (Rasch) or "2pl" for Axis III/IV fit.
    """
    n_kept = int(mask.sum())
    K = len(beta)
    if n_kept == 0:
        return {
            "name": name, "config": cfg, "n_kept": 0, "pass_rate": 0.0,
            "max_se": float("inf"), "mean_se": float("inf"),
            "budget_se": float("inf"),
            "tif_mean": 0.0, "tif_min": 0.0, "tif_cv": float("nan"),
            "per_item_tif": float("nan"), "beta_coverage_gap": float("inf"),
            "se_grid": [],
            # Axis II
            "fir": 0.0,
            # Axis III
            "fit_ok": False,
            "theta_retained": [], "beta_retained": [],
            "infit": [], "outfit": [],
            "infit_bernoulli": [], "outfit_bernoulli": [],
            "infit_pass_rate": float("nan"),
            "outfit_pass_rate": float("nan"),
            "infit_bernoulli_pass_rate": float("nan"),
            "outfit_bernoulli_pass_rate": float("nan"),
            "q3_violation_rate": float("nan"),
            "q3_max": float("nan"),
            # Axis IV primary
            "pseudo_r2": float("nan"),
            "ll_model": float("nan"),
            "ll_null": float("nan"),
            "ll_sat": float("nan"),
            "holdout_n_cells": 0,
            "holdout_seed": 42,
            "holdout_fit_ok": False,
            # Axis IV tertiary
            "m2": float("nan"), "m2_df": 0, "rmsea": float("nan"),
            # Axis IV secondary
            "mean_z_sq": float("nan"),
            "mean_z_sq_bernoulli": float("nan"),
            # Resolution metrics (all zero for empty set)
            "mean_res": 0.0, "min_res": 0.0, "max_res": 0.0,
            "frac_ci_disjoint_95": 0.0, "frac_ci_disjoint_99": 0.0,
            "mean_d12": 0.0, "min_d12": 0.0, "max_d12": 0.0,
            "se_per_panel": [], "res_per_pair": [],
        }

    beta_sub = beta[mask]
    # Effective N per cell (median across subset, across models)
    N_sub = float(np.median(N_trials[:, mask]))

    se_grid = compute_se_grid(beta_sub, N_sub, target_grid)
    tif_panel = compute_tif_at_panel(beta_sub, panel_theta)

    max_se = float(se_grid.max())
    mean_se = float(se_grid.mean())
    budget_se = max_se * float(np.sqrt(n_kept / n_ref))

    # NEW: Pairwise resolution metrics (definition (c) + (d) in primer §3.5)
    pr = compute_pairwise_resolution(beta_sub, N_sub, panel_theta)
    # NEW: Pairwise pass-rate differentiation (definition (a) in primer §3.5.6)
    pd = compute_pairwise_pass_rate_diff(beta_sub, panel_theta)

    # Axis II: Fisher Info Retention
    fir = compute_fir(q_bar_all, mask)

    # Axis III + Axis IV secondary: retained-set IRT fit + residual diagnostics
    pass_sub = pass_counts[:, mask]
    N_trials_sub = N_trials[:, mask]
    theta_F, beta_F, alpha_F, fit_ok = fit_irt_on_subset(pass_sub, N_trials_sub, irt_model)
    if fit_ok and theta_F is not None:
        z = compute_standardized_residuals(pass_sub, N_trials_sub, theta_F, beta_F, alpha_F)
        fit_diag = compute_item_fit_diagnostics(z, pass_sub, N_trials_sub,
                                                theta_F, beta_F, alpha_F)
        q3_diag = compute_q3(z)
        infit_arr = fit_diag["infit"]
        outfit_arr = fit_diag["outfit"]
        infit_bern_arr = fit_diag["infit_bernoulli"]
        outfit_bern_arr = fit_diag["outfit_bernoulli"]
        infit_pass = fit_diag["infit_pass_rate"]
        outfit_pass = fit_diag["outfit_pass_rate"]
        infit_bern_pass = fit_diag["infit_bernoulli_pass_rate"]
        outfit_bern_pass = fit_diag["outfit_bernoulli_pass_rate"]
        mean_z_sq = fit_diag["mean_z_sq"]
        mean_z_sq_bernoulli = fit_diag["mean_z_sq_bernoulli"]
        q3_violation = q3_diag["q3_violation_rate"]
        q3_max = q3_diag["q3_max"]
        # theta may be (M,) for 1PL/2PL or (M, D) for MIRT; same for alpha
        theta_out = theta_F.tolist() if hasattr(theta_F, "tolist") else list(theta_F)
        beta_out = [float(x) for x in beta_F]
        alpha_out = alpha_F.tolist() if hasattr(alpha_F, "tolist") else list(alpha_F)
    else:
        infit_arr = np.full(n_kept, np.nan)
        outfit_arr = np.full(n_kept, np.nan)
        infit_bern_arr = np.full(n_kept, np.nan)
        outfit_bern_arr = np.full(n_kept, np.nan)
        infit_pass = float("nan")
        outfit_pass = float("nan")
        infit_bern_pass = float("nan")
        outfit_bern_pass = float("nan")
        mean_z_sq = float("nan")
        mean_z_sq_bernoulli = float("nan")
        q3_violation = float("nan")
        q3_max = float("nan")
        theta_out = []
        beta_out = []
        alpha_out = []

    # Axis IV primary: held-out pseudo-R² (under the same IRT model)
    r2_diag = compute_holdout_pseudo_r2(pass_sub, N_trials_sub, seed=42,
                                        irt_model=irt_model)

    # Axis IV tertiary (paper-only, ATLAS comparison): M_2 / RMSEA
    if fit_ok and theta_F is not None:
        m2_diag = compute_m2_rmsea(pass_sub, N_trials_sub, theta_F, beta_F, alpha_F)
    else:
        m2_diag = {"m2": float("nan"), "m2_df": 0, "rmsea": float("nan")}

    return {
        "name": name,
        "config": cfg,
        "n_kept": n_kept,
        "pass_rate": n_kept / K,
        "max_se": max_se,
        "mean_se": mean_se,
        "budget_se": budget_se,
        "se_grid": [float(x) for x in se_grid],
        "tif_mean": float(tif_panel.mean()),
        "tif_min": float(tif_panel.min()),
        "tif_cv": float(tif_panel.std() / tif_panel.mean()) if tif_panel.mean() > 0 else float("nan"),
        "per_item_tif": float(tif_panel.mean() / n_kept),
        "beta_coverage_gap": beta_coverage_gap(beta_sub),
        "n_effective": N_sub,
        # Axis II
        "fir": fir,
        # Axis III: retained-set IRT fit diagnostics
        "irt_model": irt_model,
        "fit_ok": bool(fit_ok),
        "theta_retained": theta_out,
        "beta_retained": beta_out,
        "alpha_retained": alpha_out,
        "infit": [float(x) for x in infit_arr],
        "outfit": [float(x) for x in outfit_arr],
        "infit_bernoulli": [float(x) for x in infit_bern_arr],
        "outfit_bernoulli": [float(x) for x in outfit_bern_arr],
        "infit_pass_rate": infit_pass,
        "outfit_pass_rate": outfit_pass,
        "infit_bernoulli_pass_rate": infit_bern_pass,
        "outfit_bernoulli_pass_rate": outfit_bern_pass,
        "q3_violation_rate": q3_violation,
        "q3_max": q3_max,
        # Axis IV primary
        "pseudo_r2": r2_diag["pseudo_r2"],
        "ll_model": r2_diag["ll_model"],
        "ll_null": r2_diag["ll_null"],
        "ll_sat": r2_diag["ll_sat"],
        "holdout_n_cells": r2_diag["n_holdout"],
        "holdout_seed": r2_diag["seed"],
        "holdout_fit_ok": r2_diag.get("fit_ok", False),
        # Axis IV tertiary (paper-only, ATLAS comparison)
        "m2": m2_diag["m2"],
        "m2_df": m2_diag["m2_df"],
        "rmsea": m2_diag["rmsea"],
        # Axis IV secondary
        "mean_z_sq": mean_z_sq,
        "mean_z_sq_bernoulli": mean_z_sq_bernoulli,
        # Resolution metrics — (c) pairwise z / res
        "mean_res": pr["mean_res"],
        "min_res": pr["min_res"],
        "max_res": pr["max_res"],
        # Resolution metrics — (d) CI disjoint fraction
        "frac_ci_disjoint_95": pr["frac_ci_disjoint_95"],
        "frac_ci_disjoint_99": pr["frac_ci_disjoint_99"],
        # Resolution metrics — (a) pairwise pass-rate diff (intensive)
        "mean_d12": pd["mean_d"],
        "min_d12": pd["min_d"],
        "max_d12": pd["max_d"],
        # Diagnostic detail (for JSON only; not printed)
        "se_per_panel": pr["se_per_panel"],
        "res_per_pair": pr["res_per_pair"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare filter candidates on a response matrix pool")
    parser.add_argument("--response-dir", required=True,
                        help="Directory containing {model_short}/response_matrix.jsonl")
    parser.add_argument("--target-buffer", type=float, default=0.5,
                        help="Extend target θ range by this much beyond panel [θ_min, θ_max]")
    parser.add_argument("--n-grid", type=int, default=50,
                        help="Number of points in target θ grid")
    parser.add_argument("--n-ref", type=int, default=200,
                        help="Reference n_kept for budget-normalized max_SE "
                             "(budget_se = max_se * sqrt(n_kept / n_ref))")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: {response_dir}/filter_ablation.json)")
    parser.add_argument("--irt-model",
                        choices=["ctt", "1pl", "2pl", "mirt-2d", "mirt-3d",
                                 "both", "all", "ctt-vs-1pl"],
                        default="1pl",
                        help="Calibration model for Axis III/IV. Options: "
                             "'ctt' (classical test theory, OLS only), "
                             "'1pl' / '2pl' / 'mirt-2d' / 'mirt-3d', "
                             "'both' (1PL+2PL), 'all' (1PL+2PL+MIRT-2D+MIRT-3D), "
                             "'ctt-vs-1pl' (CTT + 1PL for head-to-head).")
    # Hard thresholds for Axis III/IV compliance (INFORMATIONAL only — not
    # used to pre-screen candidates; reported per filter for transparency).
    parser.add_argument("--threshold-pseudo-r2", type=float, default=0.0,
                        help="Axis IV primary soft threshold: pseudo_R² ≥ this")
    parser.add_argument("--threshold-mean-z-sq", type=float, default=1.5,
                        help=("Axis IV secondary soft threshold: "
                              "mean_z²_bernoulli ≤ this (classical IRT range "
                              "[0.5, 1.5]; N-invariant). Threshold is "
                              "compared against the Bernoulli-equivalent, "
                              "NOT the Binomial mean_z² (which is ×N bias-"
                              "amplified)."))
    parser.add_argument("--threshold-infit-pass", type=float, default=0.95,
                        help=("Axis III soft threshold: "
                              "infit_bernoulli_pass_rate ≥ this. Uses the "
                              "N-normalized Bernoulli-equivalent Infit; the "
                              "Binomial Infit pass rate is near-zero for N=40 "
                              "due to ×N amplification (not a fit score "
                              "in isolation)."))
    parser.add_argument("--threshold-q3-viol", type=float, default=0.80,
                        help="Axis III soft threshold: Q3 violation rate ≤ this")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--exclude-models", nargs="+", default=[],
                        help="Full model names (e.g. 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B') "
                             "to exclude from panel. For panel ablation / LOO sensitivity.")
    args = parser.parse_args()

    # ---- Load ----
    rm = load_response_matrix(args.response_dir)
    if args.exclude_models:
        missing = [m for m in args.exclude_models if m not in rm["models"]]
        if missing:
            raise ValueError(
                f"--exclude-models not in response dir: {missing}\n"
                f"Available: {rm['models']}")
        keep = np.array([m not in args.exclude_models for m in rm["models"]])
        excluded = [m for m in rm["models"] if m in args.exclude_models]
        rm["models"] = [m for m, k in zip(rm["models"], keep) if k]
        rm["p_hat"] = rm["p_hat"][keep]
        rm["pass_counts"] = rm["pass_counts"][keep]
        rm["total_trials"] = rm["total_trials"][keep]
        print(f"  [panel filter] excluded {len(excluded)}: {excluded}")
    models = rm["models"]
    items = rm["items"]
    K = len(items)
    M = len(models)
    p_hat = rm["p_hat"]                     # (M, J)
    pass_counts = rm["pass_counts"]
    N_trials = rm["total_trials"]

    print("=" * 72)
    print("  Filter Ablation")
    print("=" * 72)
    print(f"  Response dir:  {args.response_dir}")
    print(f"  Dimensions:    {M} models × {K} items")
    print(f"  N_trials:      min={int(N_trials.min())}, median={int(np.median(N_trials))}, max={int(N_trials.max())}")

    # ---- Joint Rasch fit ----
    print("\n[1/3] Fitting joint Rasch model on full pool (OLS init → MLE)...")
    theta_init, beta_init = fit_ols(p_hat)
    theta, beta, _nll = fit_mle(pass_counts, N_trials, theta_init, beta_init)
    print(f"  θ_panel range: [{theta.min():+.3f}, {theta.max():+.3f}]")
    print(f"  β range:       [{beta.min():+.3f}, {beta.max():+.3f}] "
          f"(p50={np.percentile(beta, 50):+.3f})")

    target_lo = theta.min() - args.target_buffer
    target_hi = theta.max() + args.target_buffer
    target_grid = np.linspace(target_lo, target_hi, args.n_grid)
    print(f"  Target θ grid: [{target_lo:+.3f}, {target_hi:+.3f}] ({args.n_grid} points)")

    # Short model names for display
    def short(m):
        s = m.split("/")[-1]
        for old, new in [("DeepSeek-R1-Distill-Llama", "R1"),
                         ("Qwen3-30B-A3B-Instruct-2507", "Qwen-30B"),
                         ("Mistral-Small-3.1-24B-Instruct-2503", "Mistral-24B"),
                         ("gemma-3-", "Gemma3-"), ("-it", "")]:
            s = s.replace(old, new)
        return s[:14]

    print(f"\n  Panel θ (sorted):")
    for i in np.argsort(theta):
        print(f"    {short(models[i]):<14}  θ = {theta[i]:+.3f}")

    # ---- Pre-compute Axis II atom ----
    # q̄_x = mean over panel of p_mx (1 - p_mx); per-item Fisher info density.
    # Used for FIR (Fisher Info Retention), computed once on full pool.
    q_bar_all = compute_q_bar_all(p_hat)

    # ---- Build and evaluate filters ----
    # Which IRT model(s) to run for Axis III/IV diagnostics
    if args.irt_model == "both":
        irt_models_to_run = ["1pl", "2pl"]
    elif args.irt_model == "all":
        irt_models_to_run = ["1pl", "2pl", "mirt-2d", "mirt-3d"]
    elif args.irt_model == "ctt-vs-1pl":
        irt_models_to_run = ["ctt", "1pl"]
    else:
        irt_models_to_run = [args.irt_model]

    print(f"\n[2/3] Evaluating filter candidates (β source: Option A — shared; "
          f"n_ref={args.n_ref}; IRT model(s): {irt_models_to_run})...")
    filters = build_filter_list(theta)
    # results is the list for the PRIMARY irt_model (first in list); extra
    # models stored in results[i]["diagnostics_{irt_model}"].
    results = []
    for name, cfg, fn in filters:
        mask = fn(p_hat, beta, theta)
        # Primary eval (first IRT model in list)
        r = evaluate_filter(name, cfg, mask, beta, p_hat, pass_counts, N_trials,
                            theta, target_grid, args.n_ref, q_bar_all,
                            irt_model=irt_models_to_run[0])
        # Additional IRT model diagnostics (if requested)
        r["diagnostics_by_irt_model"] = {
            irt_models_to_run[0]: {
                k: r[k] for k in ["fit_ok",
                                  "infit_pass_rate", "outfit_pass_rate",
                                  "infit_bernoulli_pass_rate",
                                  "outfit_bernoulli_pass_rate",
                                  "q3_violation_rate",
                                  "mean_z_sq", "mean_z_sq_bernoulli",
                                  "pseudo_r2"]
            }
        }
        for extra_model in irt_models_to_run[1:]:
            r_extra = evaluate_filter(name, cfg, mask, beta, p_hat, pass_counts,
                                      N_trials, theta, target_grid, args.n_ref,
                                      q_bar_all, irt_model=extra_model)
            r["diagnostics_by_irt_model"][extra_model] = {
                k: r_extra[k] for k in ["fit_ok",
                                        "infit_pass_rate", "outfit_pass_rate",
                                        "infit_bernoulli_pass_rate",
                                        "outfit_bernoulli_pass_rate",
                                        "q3_violation_rate",
                                        "mean_z_sq", "mean_z_sq_bernoulli",
                                        "pseudo_r2"]
            }
        results.append(r)

    # Sort results by primary metric (budget_se), ascending
    results_sorted = sorted(
        results,
        key=lambda r: r["budget_se"] if np.isfinite(r["budget_se"]) else float("inf"),
    )

    # ---- Print table ----
    print(f"\n[3/3] Results (PRIMARY: budget_SE — LOWER IS BETTER — Framing C, n_ref={args.n_ref})\n")
    header = (f"  {'Filter':<42} {'n_kept':>7} {'pass%':>6} "
              f"{'budget_SE':>10} {'FIR':>6} {'max_SE':>8} {'mean_SE':>8} "
              f"{'per_item_tif':>12} {'β_gap':>7}")
    print(header)
    print("  " + "─" * (len(header) - 2))
    best_budget_se = min(r["budget_se"] for r in results if np.isfinite(r["budget_se"]))
    for r in results_sorted:
        marker = " ⭐" if r["budget_se"] == best_budget_se else "  "
        line = (f"  {r['name']:<42} {r['n_kept']:>7} {r['pass_rate']*100:>5.1f}% "
                f"{r['budget_se']:>10.4f} {r['fir']:>6.3f} "
                f"{r['max_se']:>8.4f} {r['mean_se']:>8.4f} "
                f"{r['per_item_tif']:>12.4f} {r['beta_coverage_gap']:>7.3f}{marker}")
        print(line)

    # ---- Highlight winner ----
    winner = results_sorted[0]
    runner_ups = results_sorted[1:3]
    print(f"\n  ⭐ Winner (min budget_SE): {winner['name']}")
    print(f"     budget_SE={winner['budget_se']:.4f}  "
          f"(max_SE={winner['max_se']:.4f}  n_kept={winner['n_kept']})")
    for ru in runner_ups:
        gap = ru["budget_se"] - winner["budget_se"]
        print(f"     runner-up: {ru['name']:<42} "
              f"budget_SE={ru['budget_se']:.4f} (+{gap:.4f})")

    # Also report who wins on secondary metrics
    min_max_se = min(results, key=lambda r: r["max_se"] if np.isfinite(r["max_se"]) else float("inf"))
    max_per_item = max(results, key=lambda r: r["per_item_tif"] if np.isfinite(r["per_item_tif"]) else -1)
    print(f"\n  Secondary winners:")
    print(f"     min max_SE (unlimited budget):  {min_max_se['name']}  "
          f"({min_max_se['max_se']:.4f})")
    print(f"     max per-item TIF (efficiency):   {max_per_item['name']}  "
          f"({max_per_item['per_item_tif']:.4f})")

    # ---- Axis III + Axis IV (secondary) fit diagnostics table ----
    # Retained-set Rasch fit; tells us whether the filter's surviving items
    # are compatible with 1PL Rasch. mean_z² ≈ 1 under correct model.
    # Axis III/IV table; if multiple IRT models were run, show one row per
    # (filter, model) to enable direct 1PL↔2PL comparison.
    for irt_m in irt_models_to_run:
        label = irt_m.upper()
        print(f"\n  ─── Fit diagnostics (Axis III + IV) under {label} ───\n")
        header3 = (f"  {'Filter':<42} {'n_kept':>7} "
                   f"{'Inf%bin':>8} {'Inf%bern':>9} "
                   f"{'Out%bin':>8} {'Out%bern':>9} "
                   f"{'Q3viol%':>8} "
                   f"{'mean_z²':>8} {'mz²_bern':>9} {'pseudo_R²':>10}")
        print(header3)
        print("  " + "─" * (len(header3) - 2))
        for r in results_sorted:
            marker = " ⭐" if r["budget_se"] == best_budget_se else "  "
            diag = r.get("diagnostics_by_irt_model", {}).get(irt_m, {})
            infit_pct = diag.get("infit_pass_rate", float("nan"))
            outfit_pct = diag.get("outfit_pass_rate", float("nan"))
            infit_bern_pct = diag.get("infit_bernoulli_pass_rate", float("nan"))
            outfit_bern_pct = diag.get("outfit_bernoulli_pass_rate", float("nan"))
            q3_viol = diag.get("q3_violation_rate", float("nan"))
            mz2 = diag.get("mean_z_sq", float("nan"))
            mz2_bern = diag.get("mean_z_sq_bernoulli", float("nan"))
            pr2 = diag.get("pseudo_r2", float("nan"))
            line3 = (f"  {r['name']:<42} {r['n_kept']:>7} "
                     f"{infit_pct*100:>7.1f}% {infit_bern_pct*100:>8.1f}% "
                     f"{outfit_pct*100:>7.1f}% {outfit_bern_pct*100:>8.1f}% "
                     f"{q3_viol*100:>7.1f}% "
                     f"{mz2:>8.3f} {mz2_bern:>9.3f} "
                     f"{pr2:>10.3f}{marker}")
            print(line3)
    print(f"\n  Axis III/IV reading guide:")
    print(f"    - Inf%bin / Out%bin: fraction of items with Binomial Infit/Outfit ∈ [0.5, 1.5]")
    print(f"      E[Infit_bin] ≈ 1 + N·(Σ q·δ²)/(Σ q); same ×N amplification as mean_z²")
    print(f"      → near-zero pass rates are EXPECTED for N=40 even with tiny real misfit")
    print(f"    - Inf%bern / Out%bern: N-normalized Bernoulli-equivalent pass rates")
    print(f"      Same z²_bern machinery as mz²_bern; this is what single-trial Fluid/ATLAS/ReEval")
    print(f"      implicitly report. **Use these for actual fit judgment**, not Binomial columns.")
    print(f"    - Q3viol%: |Q3| > 0.2; M=6 null baseline ≈ 70%, so differences < 3% are noise")
    print(f"    - mean_z²: Binomial residual sq mean; E[·] ≈ 1 + N·δ²/p(1−p) under δ=p_true−p_pred")
    print(f"      → bias amplified ×N vs single-trial Bernoulli; not a 'fit score' in isolation")
    print(f"    - mz²_bern: N-normalized Bernoulli-equivalent z² (z²_bern = 1 + (z²_bin−1)/N)")
    print(f"      classical single-trial IRT range [0.5, 1.5]; gap from mean_z² = N-fold")
    print(f"      diagnostic power, not misfit magnitude")
    print(f"    - pseudo_R²: holdout Binomial LL relative to p=0.5 baseline; N-invariant.")
    print(f"      NO FILTER row is the uncompressed reference. Filtered subsets naturally")
    print(f"      trade pseudo_R² for Fisher-info density: boundary items have p≈0.5, so the")
    print(f"      p=0.5 null becomes competitive — low/negative filtered pseudo_R² is by design,")
    print(f"      not Rasch failure. Compare row-to-row against NO FILTER as anchor.")
    if len(irt_models_to_run) > 1:
        print(f"    - Rankings robust across 1PL/2PL → filter choice is IRT-model-independent")
        print(f"      Rankings differ → use the IRT model paper commits to (currently 1PL)")

    # pseudo_R² framing summary: NO FILTER = uncompressed baseline
    no_filter_row = next((r for r in results_sorted if r["name"] == "NO FILTER"), None)
    if no_filter_row is not None:
        primary_diag = no_filter_row.get("diagnostics_by_irt_model", {}).get(
            irt_models_to_run[0], {})
        no_filter_pr2 = primary_diag.get("pseudo_r2", float("nan"))
        winner_diag = results_sorted[0].get("diagnostics_by_irt_model", {}).get(
            irt_models_to_run[0], {})
        winner_pr2 = winner_diag.get("pseudo_r2", float("nan"))
        print(f"\n  pseudo_R² framing (§filter_metric_construction.md §9.3.0):")
        print(f"    NO FILTER pseudo_R² = {no_filter_pr2:.3f}   (uncompressed baseline — full K-item pool)")
        print(f"    Winner    pseudo_R² = {winner_pr2:.3f}   (filtered subset)")
        drop = no_filter_pr2 - winner_pr2
        print(f"    Δ = {drop:+.3f} — filter concentrates boundary items (p ≈ 0.5) where")
        print(f"    the p=0.5 null baseline becomes competitive with Rasch's cell-specific p̂.")
        print(f"    Low/negative filtered pseudo_R² is BY DESIGN (information-density trade-off),")
        print(f"    not Rasch failure. Use NO FILTER row as anchor when reading filtered rows.")

    # ---- Secondary table: resolution metrics (§3.5 definitions (a), (c), (d)) ----
    # These are INDEPENDENT cross-checks. If winner agrees across
    # budget_SE, mean_d12 (intensive), frac_ci_disjoint → robust.
    print(f"\n  ─── Resolution metrics (primer §3.5: (a) D_ij, (c) res, (d) CI disjoint) ───\n")
    header2 = (f"  {'Filter':<42} {'n_kept':>7} "
               f"{'mean_d12':>9} {'min_res':>9} {'mean_res':>9} "
               f"{'disjoint95':>11} {'disjoint99':>11}")
    print(header2)
    print("  " + "─" * (len(header2) - 2))
    # Print in same order as primary table (by budget_se)
    for r in results_sorted:
        marker = " ⭐" if r["budget_se"] == best_budget_se else "  "
        line2 = (f"  {r['name']:<42} {r['n_kept']:>7} "
                 f"{r['mean_d12']:>9.4f} {r['min_res']:>9.2f} {r['mean_res']:>9.2f} "
                 f"{r['frac_ci_disjoint_95']*100:>10.1f}% "
                 f"{r['frac_ci_disjoint_99']*100:>10.1f}%{marker}")
        print(line2)

    # Cross-check: intensive metrics should agree with budget_SE (both
    # are n_kept-invariant per-item-quality measures). Extensive metrics
    # (min_res / frac_ci_disjoint) are EXPECTED to favor NO FILTER by
    # construction — they are reported for inspection, not for ranking.
    print(f"\n  Cross-check (primer §3.6): intensive metrics — should agree with budget_SE")
    best_d12 = max(results, key=lambda r: r["mean_d12"])
    print(f"     max mean_d12 (intensive, pass-rate diff):     {best_d12['name']} "
          f"(d12={best_d12['mean_d12']:.4f})")
    # winner is from budget_SE
    intensive_agree = best_d12["name"] == winner["name"]
    if intensive_agree:
        print(f"     ✓ mean_d12 winner = budget_SE winner → intensive cross-check passes")
    else:
        # Find rank of budget_SE winner in d12 order
        d12_ranked = sorted(results, key=lambda r: -r["mean_d12"])
        winner_d12_rank = next(i for i, r in enumerate(d12_ranked) if r["name"] == winner["name"]) + 1
        print(f"     ⚠ mean_d12 winner differs from budget_SE winner")
        print(f"       budget_SE winner ranks #{winner_d12_rank} on mean_d12 "
              f"(d12={winner['mean_d12']:.4f}, gap from d12 winner={best_d12['mean_d12']-winner['mean_d12']:+.4f})")
        print(f"       → usually strict filters (small n_kept) win d12 at cost of coverage;")
        print(f"         budget_SE balances d12 × coverage → prefer budget_SE unless d12 gap is large")

    # Report extensive metrics (informational, not for ranking)
    print(f"\n  Extensive metrics (informational — bias toward large n_kept):")
    best_min_res = max(results, key=lambda r: r["min_res"])
    best_disjoint = max(results, key=lambda r: r["frac_ci_disjoint_95"])
    print(f"     max min_res  (worst-pair res, extensive):       {best_min_res['name']} "
          f"(min_res={best_min_res['min_res']:.2f})")
    print(f"     max frac_ci_disjoint_95 (extensive):             {best_disjoint['name']} "
          f"({best_disjoint['frac_ci_disjoint_95']*100:.1f}%)")
    print(f"     note: these extensive metrics naturally favor NO FILTER;")
    print(f"           read as 'is the benchmark saturated at this resolution level?' not 'which filter wins?'")

    # ---- Pareto frontier on (budget_SE ↓, FIR ↑) ----
    print(f"\n  ─── Pareto frontier on (budget_SE ↓ × FIR ↑) ───\n")
    pareto_names = compute_pareto_frontier(results)
    pareto_names_set = set(pareto_names)
    print(f"  {'Filter':<42} {'n_kept':>7} {'budget_SE':>10} {'FIR':>7} {'Pareto?':>8}")
    print("  " + "─" * 78)
    # Sort results by budget_SE for display
    for r in results_sorted:
        pareto_mark = "✓" if r["name"] in pareto_names_set else " "
        line = (f"  {r['name']:<42} {r['n_kept']:>7} "
                f"{r['budget_se']:>10.4f} {r['fir']:>7.3f}   {pareto_mark:>4}")
        print(line)
    print(f"\n  Pareto frontier ({len(pareto_names)} filters): "
          f"any non-member is strictly dominated on both budget_SE and FIR.")
    print(f"  Winner (min budget_SE in frontier): {results_sorted[0]['name']}")

    # ---- Threshold compliance (informational, under primary IRT model) ----
    thresholds = {
        "pseudo_r2": args.threshold_pseudo_r2,
        "mean_z_sq": args.threshold_mean_z_sq,
        "infit_pass_rate": args.threshold_infit_pass,
        "q3_violation_rate": args.threshold_q3_viol,
    }
    primary_irt = irt_models_to_run[0]
    print(f"\n  ─── Axis III/IV threshold compliance (informational, under {primary_irt.upper()}) ───")
    print(f"  Thresholds:  pseudo_R² ≥ {thresholds['pseudo_r2']:.2f}  "
          f"mean_z²_bern ≤ {thresholds['mean_z_sq']:.1f}  "
          f"Infit%_bern ≥ {thresholds['infit_pass_rate']*100:.0f}%  "
          f"Q3viol% ≤ {thresholds['q3_violation_rate']*100:.0f}%")
    print(f"  (Axis III/IV thresholds are compared against Bernoulli-equivalent metrics,")
    print(f"   not the ×N bias-amplified Binomial counterparts.)")
    print(f"\n  {'Filter':<42} {'pR²✓':>6} {'mz²✓':>6} {'Inf✓':>6} {'Q3✓':>6} {'all✓':>6}")
    print("  " + "─" * 78)
    compliance_summary = {}
    for r in results_sorted:
        flags = check_fit_thresholds(r, thresholds, primary_irt)
        compliance_summary[r["name"]] = flags
        all_ok = all(flags.values())
        c = lambda ok: "✓" if ok else "✗"
        print(f"  {r['name']:<42} "
              f"{c(flags['pseudo_r2_ok']):>6} {c(flags['mean_z_sq_ok']):>6} "
              f"{c(flags['infit_pass_ok']):>6} {c(flags['q3_viol_ok']):>6} "
              f"{c(all_ok):>6}")
    print(f"\n  Filter selection uses the Pareto frontier above.")
    print(f"  Fit thresholds provide supplementary model-fit diagnostics.")

    # Binarized-marginal diagnostics discard within-cell trial variance.
    print(f"\n  ─── Axis IV: M₂ / RMSEA on binarized marginals ───\n")
    print(f"  {'Filter':<42} {'n_kept':>7} {'M₂':>10} {'df':>7} {'M₂/df':>7} {'RMSEA':>8}")
    print("  " + "─" * 84)
    for r in results_sorted:
        marker = " ⭐" if r["budget_se"] == best_budget_se else "  "
        m2 = r.get("m2", float("nan"))
        m2_df = r.get("m2_df", 0)
        rmsea = r.get("rmsea", float("nan"))
        m2_over_df = m2 / max(m2_df, 1)
        line = (f"  {r['name']:<42} {r['n_kept']:>7} "
                f"{m2:>10.1f} {m2_df:>7} {m2_over_df:>7.3f} {rmsea:>8.3f}{marker}")
        print(line)
    print(f"\n  Interpretation:")
    print(f"    M₂ uses a diagonal-covariance approximation on binarized cells.")
    print(f"    With M={M} panel models, asymptotic χ² calibration is approximate.")
    print(f"    RMSEA is zero whenever M₂/df < 1; this alone does not establish fit.")
    print(f"    Binarization discards within-cell N-trial variance. For count-based")
    print(f"    fit and predictive diagnostics, use mean_z²_bern, Infit_bern, and pseudo_R².")

    # ---- Save JSON ----
    out_path = args.output or str(Path(args.response_dir) / "filter_ablation.json")
    out = {
        "metadata": {
            "response_dir": args.response_dir,
            "n_models": M,
            "n_items": K,
            "n_trials_median": int(np.median(N_trials)),
            "panel_theta": {models[i]: float(theta[i]) for i in range(M)},
            "beta_range": [float(beta.min()), float(beta.max())],
            "target_buffer": args.target_buffer,
            "target_range": [float(target_lo), float(target_hi)],
            "n_grid": args.n_grid,
            "beta_source": "Option A (shared from joint fit on full pool)",
            "primary_metric": "budget_se = max_SE * sqrt(n_kept / n_ref)",
            "n_ref": args.n_ref,
            "four_axis_framework": {
                "axis_I_retained_quality": {
                    "primary": "budget_se",
                    "secondary": ["mean_d12", "frac_ci_disjoint_95"],
                    "doc": "docs/filter_metric_construction.md §2",
                },
                "axis_II_info_retention": {
                    "primary": "fir",
                    "secondary": ["pass_rate", "beta_coverage_gap"],
                    "doc": "docs/filter_metric_construction.md §9.1",
                    "formula": "fir = sum(q_bar[kept]) / sum(q_bar[all]); q_bar_x = mean_m p_mx(1-p_mx)",
                },
                "axis_III_per_item_fit": {
                    "primary": ["infit_bernoulli_pass_rate",
                                "q3_violation_rate"],
                    "primary_note": (
                        "Bernoulli-equivalent Infit is the defensible "
                        "fit-pass metric (N-invariant, classical IRT range "
                        "[0.5, 1.5]). Binomial Infit/Outfit pass rates are "
                        "also reported but mechanically near-zero for N=40 "
                        "due to ×N bias amplification — same story as mean_z². "
                        "See docs/filter_metric_construction.md §9.4.1."
                    ),
                    "implemented": ["infit_pass_rate", "outfit_pass_rate",
                                    "infit_bernoulli_pass_rate",
                                    "outfit_bernoulli_pass_rate",
                                    "q3_violation_rate", "q3_max"],
                    "fit_method": "Option B retained-set fit (OLS → MLE)",
                    "q3_threshold": 0.2,
                    "irt_models_run": irt_models_to_run,
                    "irt_2pl_log_alpha_bounds": [-2.0, 2.0],
                },
                "axis_IV_global_fit": {
                    "primary": "pseudo_r2 (holdout Binomial LL)",
                    "secondary": ["mean_z_sq", "mean_z_sq_bernoulli"],
                    "secondary_note": (
                        "mean_z_sq is Binomial N-trial residual (amplifies "
                        "bias by ×N); mean_z_sq_bernoulli is the N-normalized "
                        "single-trial equivalent (classical IRT range "
                        "[0.5, 1.5]). Their ratio quantifies N-fold "
                        "diagnostic power vs Fluid/ATLAS/ReEval."
                    ),
                    "tertiary_informational_only": ["m2", "rmsea"],
                    "tertiary_downgrade_note": (
                        "M_2 / RMSEA are Bernoulli-designed and require "
                        "Option A binarization for our Binomial observations. "
                        "Binarization discards the N-trial cell variance info "
                        "that pillar #5 (N-trial Binomial) relies on to expose "
                        "misfit, so 'perfect fit' readings here are structurally "
                        "uninformative. Downgraded from 'paper-tertiary' to "
                        "'informational only'; not used for filter selection, "
                        "not cited as fit evidence in paper. Binomial-native "
                        "M_2 derivation is future work. See "
                        "docs/filter_metric_construction.md §9.3.1."
                    ),
                    "implemented": ["mean_z_sq", "mean_z_sq_bernoulli",
                                    "pseudo_r2", "m2", "rmsea"],
                    "holdout_frac": 0.2,
                    "holdout_seed": 42,
                    "m2_method": "diagonal-Σ approximation (Maydeu-Olivares 2013)",
                    "m2_binarization": "Option A: binary = (k > N/2)",
                    "m2_caveat": "M=6 weakens asymptotic χ²; RMSEA directional",
                },
            },
            "resolution_metrics_doc": "docs/irt_concepts_primer.md §3.5, §3.6",
            "pareto_frontier": pareto_names,
            "pareto_winner": results_sorted[0]["name"] if results_sorted else None,
            "thresholds": thresholds,
            "threshold_compliance": compliance_summary,
            "resolution_metrics": {
                "mean_d12": "intensive pairwise pass-rate diff (§3.5.6); |kept|-invariant",
                "min_res / mean_res / max_res": "pairwise resolution = |Δθ| / sqrt(SE_i²+SE_j²) (§3.5.4); extensive",
                "frac_ci_disjoint_95": "fraction of pairs with |Δθ| > 1.96 (SE_i+SE_j) (§3.5.5, conservative disjoint-CI test)",
                "frac_ci_disjoint_99": "same at 2.576 threshold",
                "se_per_panel": "SE of θ̂ at each panel θ under filter's kept items",
                "res_per_pair": "per-pair detail: delta_theta, se_i, se_j, res, CI disjoint flags",
            },
        },
        "target_grid": [float(x) for x in target_grid],
        "filters": results,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON → {out_path}")


if __name__ == "__main__":
    main()
