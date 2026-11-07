"""IRT misfit diagnostics — shared helpers for 1PL / 2PL / MIRT.

Used by both `scripts/analysis/compare_filters.py` (filter ablation Axis III/IV)
and `scripts/analysis/irt_misfit_diagnostics.py` (standalone misfit inspection).

Provides:
  - _compute_logits: unified logit matrix for 1PL / 2PL / MIRT
  - compute_standardized_residuals: z_{mx} = (k − Np) / sqrt(Np(1−p))
  - compute_z2_bernoulli_avg: per-cell Bernoulli-equivalent z² (N-normalized)
  - compute_item_fit_diagnostics: per-item Infit / Outfit + pass rates
  - compute_q3: pairwise residual correlation (local independence check)

References:
  - docs/irt_concepts_primer.md §2.6 (formulas)
  - docs/filter_metric_construction.md §9 (4-axis metric framework)
  - docs/method_pipeline.md Cross-cutting A (N-scaling power asymmetry)
"""

from __future__ import annotations

import numpy as np


def _compute_logits(theta: np.ndarray, beta: np.ndarray,
                    alpha: np.ndarray | None = None) -> np.ndarray:
    """Unified logit matrix computation supporting 1PL / 2PL / MIRT.

    Shape dispatch:
        1PL: theta (M,), alpha None
            → logit_{mx} = θ_m - β_x
        2PL: theta (M,), alpha (J,)
            → logit_{mx} = α_x · (θ_m - β_x)
        MIRT D-dim: theta (M, D), alpha (J, D)
            → logit_{mx} = Σ_d θ_{m,d} · a_{x,d} - β_x

    Clipped to [-40, 40] for numerical stability.
    Returns (M, J) matrix.
    """
    if alpha is None:
        return np.clip(theta[:, None] - beta[None, :], -40.0, 40.0)
    if alpha.ndim == 1 and theta.ndim == 1:
        return np.clip(alpha[None, :] * (theta[:, None] - beta[None, :]),
                       -40.0, 40.0)
    if theta.ndim == 1:
        theta = theta[:, None]
    if alpha.ndim == 1:
        alpha = alpha[:, None]
    return np.clip(theta @ alpha.T - beta[None, :], -40.0, 40.0)


def compute_standardized_residuals(pass_counts: np.ndarray,
                                   N_trials: np.ndarray,
                                   theta: np.ndarray,
                                   beta: np.ndarray,
                                   alpha: np.ndarray | None = None) -> np.ndarray:
    """Standardized residuals z[m, x] = (k - N*p̂) / sqrt(N*p̂*(1-p̂)).

    Binomial-native (primer §2.6.1); no dichotomization needed.

    Supports 1PL (alpha=None), 2PL (alpha shape (J,)), and MIRT (alpha shape
    (J, D) + theta shape (M, D)).

    Output shape: (M, n_kept). Used by Infit/Outfit/Q3 and mean_z².
    """
    logits = _compute_logits(theta, beta, alpha)
    p = 1.0 / (1.0 + np.exp(-logits))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    expected = N_trials * p
    variance = N_trials * p * (1 - p)
    z = (pass_counts - expected) / np.sqrt(variance)
    return z


def compute_z2_bernoulli_avg(pass_counts: np.ndarray,
                             N_trials: np.ndarray,
                             p: np.ndarray) -> np.ndarray:
    """Per-cell average of Bernoulli-per-trial z² across N trials.

    For cell (m, x) with k observed ones and N−k observed zeros under model-
    predicted p:
        z²_per_1 = (1 − p)² / [p(1−p)] = (1 − p) / p
        z²_per_0 = (0 − p)² / [p(1−p)] = p / (1 − p)
        z²_bern_avg = [k·(1−p)/p + (N−k)·p/(1−p)] / N

    Under correct model, E[z²_bern_avg] ≈ 1 (classical IRT single-trial
    reference). Relationship to Binomial z²:
        z²_bin ≈ 1 + N · (z²_bern_avg − 1)

    This exposes the statistical-power asymmetry between N-trial Binomial
    (bias amplified ×N) and single-trial Bernoulli frameworks used by
    Fluid / ATLAS / ReEval. See docs/method_pipeline.md Cross-cutting A.

    Args:
        pass_counts: (M, J) observed pass counts.
        N_trials: scalar or (M, J) trial counts per cell.
        p: (M, J) model-predicted pass probabilities.

    Returns:
        (M, J) matrix of per-cell Bernoulli-equivalent z² averages.
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z2_per_1 = (1 - p) / p
    z2_per_0 = p / (1 - p)
    total = pass_counts * z2_per_1 + (N_trials - pass_counts) * z2_per_0
    return total / N_trials


def compute_item_fit_diagnostics(z: np.ndarray,
                                 pass_counts: np.ndarray,
                                 N_trials: np.ndarray,
                                 theta: np.ndarray,
                                 beta: np.ndarray,
                                 alpha: np.ndarray | None = None) -> dict:
    """Per-item Infit / Outfit + global mean_z².

    Formulas (primer §2.6.2):
        Outfit_x = mean_m(z_mx²)                              [unweighted]
        Infit_x  = Σ_m (k_mx - N*p_mx)² / Σ_m N*p_mx*(1-p_mx) [Fisher-info weighted]

    Expected under Rasch + MLE convergence:
        Infit_x, Outfit_x ≈ 1.0
    Acceptable range: [0.5, 1.5]; < 0.5 = overfitting, > 1.5 = misfit.

    Also returns Bernoulli-equivalent (N-normalized) twins for every
    residual-derived statistic (see compute_z2_bernoulli_avg):
        mean_z_sq_bernoulli, infit_bernoulli, outfit_bernoulli
        (+ pass rates for Infit/Outfit Bernoulli)

    Under correct model, all Bernoulli-equivalent variants ≈ 1 regardless
    of N; Binomial variants amplify bias by factor N. Both families are
    reported so the reader can distinguish N-amplified bias detection
    from true mis-specification magnitude (see docs/method_pipeline.md
    Cross-cutting A and docs/filter_metric_construction.md §9.4.1).

    Returns dict with keys:
        infit (n_kept,), outfit (n_kept,),
        infit_bernoulli (n_kept,), outfit_bernoulli (n_kept,),
        infit_pass_rate, outfit_pass_rate,
        infit_bernoulli_pass_rate, outfit_bernoulli_pass_rate,
        mean_z_sq, mean_z_sq_bernoulli.
    """
    outfit = (z * z).mean(axis=0)

    logits = _compute_logits(theta, beta, alpha)
    p = 1.0 / (1.0 + np.exp(-logits))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    residual_sq = (pass_counts - N_trials * p) ** 2
    weight = N_trials * p * (1 - p)
    num = residual_sq.sum(axis=0)
    den = weight.sum(axis=0)
    infit = np.where(den > 0, num / den, np.nan)

    z2_bern = compute_z2_bernoulli_avg(pass_counts, N_trials, p)
    outfit_bernoulli = z2_bern.mean(axis=0)
    # Bernoulli Infit uses the same Fisher-info weights as Binomial Infit
    # (w_m = N·p(1-p)); only the per-cell z² is replaced by its N-normalized
    # Bernoulli-equivalent. This preserves the "information-weighted" semantics
    # while removing the ×N bias amplification.
    infit_bern_num = (z2_bern * weight).sum(axis=0)
    infit_bernoulli = np.where(den > 0, infit_bern_num / den, np.nan)

    infit_pass = float(np.mean((infit >= 0.5) & (infit <= 1.5)))
    outfit_pass = float(np.mean((outfit >= 0.5) & (outfit <= 1.5)))
    infit_bern_pass = float(np.mean(
        (infit_bernoulli >= 0.5) & (infit_bernoulli <= 1.5)))
    outfit_bern_pass = float(np.mean(
        (outfit_bernoulli >= 0.5) & (outfit_bernoulli <= 1.5)))
    mean_z_sq = float((z * z).mean())
    mean_z_sq_bernoulli = float(z2_bern.mean())

    return {
        "infit": infit,
        "outfit": outfit,
        "infit_bernoulli": infit_bernoulli,
        "outfit_bernoulli": outfit_bernoulli,
        "infit_pass_rate": infit_pass,
        "outfit_pass_rate": outfit_pass,
        "infit_bernoulli_pass_rate": infit_bern_pass,
        "outfit_bernoulli_pass_rate": outfit_bern_pass,
        "mean_z_sq": mean_z_sq,
        "mean_z_sq_bernoulli": mean_z_sq_bernoulli,
    }


def compute_q3(z: np.ndarray, threshold: float = 0.2) -> dict:
    """Q3 statistic: local independence check via residual correlations.

    Q3(x1, x2) = Pearson corr(z[:, x1], z[:, x2]) across panel models.
    Under Rasch + local independence: Q3 ≈ 0 for all pairs.
    |Q3| > threshold (0.2 conventional) indicates pairwise dependence,
    which violates 1PL unidimensional + local independence assumption.

    See docs/irt_concepts_primer.md §2.6.4.
    """
    n_kept = z.shape[1]
    if n_kept < 2:
        return {
            "q3_upper": np.array([]),
            "q3_violation_rate": float("nan"),
            "q3_max": float("nan"),
        }
    corr = np.corrcoef(z.T)
    iu = np.triu_indices(n_kept, k=1)
    q3_upper = corr[iu]
    valid = np.isfinite(q3_upper)
    if not valid.any():
        return {
            "q3_upper": q3_upper,
            "q3_violation_rate": float("nan"),
            "q3_max": float("nan"),
        }
    q3_valid = q3_upper[valid]
    violation_rate = float(np.mean(np.abs(q3_valid) > threshold))
    q3_max = float(np.max(np.abs(q3_valid)))
    return {
        "q3_upper": q3_upper,
        "q3_violation_rate": violation_rate,
        "q3_max": q3_max,
    }
