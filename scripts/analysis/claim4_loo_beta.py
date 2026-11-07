"""Claim 4: Leave-one-model-out β stability (specific objectivity validation).

Rasch's specific objectivity theorem guarantees that item difficulty β
estimates are invariant under panel perturbation — i.e. removing or adding
panel models should not systematically shift β̂. This script empirically
tests that property on a fitted Rasch model by holding each panel model
out in turn, re-fitting, and measuring Δβ against the full-panel fit.

If the overall median |Δβ| < 0.1 logit and max |Δβ| < 0.3 logit, Rasch
is empirically stable under panel perturbation → specific objectivity
holds → Rasch is defensible as the IRT model choice regardless of
residual diagnostic magnitudes (Infit, Outfit, mean_z²).

Reference:
  - docs/irt_concepts_primer.md §2.6.5 (LOO β stability, thresholds)
  - docs/paper_plan.md Claim 4
  - docs/method_pipeline.md M5 (Rasch calibration on reference panel)

Coordinate-system alignment: Rasch is identifiable up to a translation
(θ + c, β + c gives the same likelihood). fit_mle normalizes so Σ θ = 0,
but the LOO fit uses only M-1 panel models, so its Σ = 0 constraint lives
in a different coordinate system. We re-anchor the LOO fit by matching
the mean of surviving θ̂s between full and LOO fits:
    shift = mean(θ_full[kept]) − mean(θ_loo)      (≈ −θ_full[dropped]/(M−1))
    β_loo_aligned = β_loo + shift
Δβ is then the residual β drift after removing translation invariance.

Usage:
    conda run -n boundary --no-capture-output python \
        scripts/analysis/claim4_loo_beta.py \
        --response-dir results/constrained_completion/filter_ablation_raw \
        --filters winner no_filter \
        --output results/constrained_completion/claim4_loo_beta.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difficulty.rasch import load_response_matrix, fit_ols, fit_mle
from difficulty.rasch_bayes_eb import fit_eb
from difficulty import irt_2pl
from difficulty import mirt


def fit_model(pass_counts: np.ndarray,
              N_trials: np.ndarray,
              irt_model: str = "1pl") -> tuple:
    """Fit 1PL Rasch / 1PL-EB / 2PL IRT.

    Returns:
        (theta (M,), beta (J,), alpha (J,) | None)
        For 1PL and 1PL-EB, alpha is None (implicitly α ≡ 1).
        1PL-EB returns the empirical-Bayes shrunken β in place of β_MLE.
    """
    p_hat = pass_counts / np.clip(N_trials, 1, None)
    if irt_model == "1pl":
        theta_init, beta_init = fit_ols(p_hat)
        theta, beta, _ = fit_mle(pass_counts, N_trials, theta_init, beta_init)
        return theta, beta, None
    if irt_model == "1pl_eb":
        res = fit_eb(pass_counts, N_trials)
        return res["theta"], res["beta_eb"], None
    if irt_model == "2pl":
        theta_init, beta_init, log_alpha_init = irt_2pl.fit_ols(p_hat)
        theta, beta, alpha, _ = irt_2pl.fit_mle(
            pass_counts, N_trials,
            theta_init=theta_init,
            beta_init=beta_init,
            log_alpha_init=log_alpha_init,
        )
        return theta, beta, alpha
    raise ValueError(f"Unknown irt_model: {irt_model!r}")


def align_coordinates(theta_full_keep: np.ndarray,
                      theta_loo: np.ndarray,
                      irt_model: str) -> tuple[float, float]:
    """Solve for affine (a, b) such that a·θ_loo + b ≈ θ_full[keep].

    - 1PL: invariant only under translation → a ≡ 1, b = mean(θ_full[keep]) − mean(θ_loo)
    - 2PL: invariant under (α→α/c, θ→c·θ+b, β→c·β+b) → free a, b by OLS

    β transforms identically under the same (a, b); α transforms as α/a.
    """
    if irt_model in ("1pl", "1pl_eb"):
        return 1.0, float(theta_full_keep.mean() - theta_loo.mean())
    # 2PL OLS fit for a·θ_loo + b ≈ θ_full[keep]
    x = theta_loo
    y = theta_full_keep
    x_bar = x.mean()
    y_bar = y.mean()
    var_x = ((x - x_bar) ** 2).sum()
    if var_x < 1e-12:
        return 1.0, float(y_bar - x_bar)
    cov_xy = ((x - x_bar) * (y - y_bar)).sum()
    a = float(cov_xy / var_x)
    b = float(y_bar - a * x_bar)
    return a, b


def compute_point_biserial(p_hat: np.ndarray,
                           panel_theta: np.ndarray) -> np.ndarray:
    """Per-item Pearson correlation r_pb(x) = corr(p̂_{·,x}, panel_theta).

    For unidimensional 1PL, r_pb should be ≥ 0 for all items (harder models
    pass hard items with lower p, easier models with higher p, so p̂ and θ
    are positively correlated). r_pb < 0 flags items whose response pattern
    is anti-correlated with panel ability — a signature of multi-dimensional
    item behavior.

    Handles zero-variance columns (all p̂ equal) by returning 0 for those.
    """
    J = p_hat.shape[1]
    r_pb = np.zeros(J)
    theta_c = panel_theta - panel_theta.mean()
    theta_norm = np.sqrt((theta_c ** 2).sum())
    if theta_norm < 1e-12:
        return r_pb
    for j in range(J):
        p_col = p_hat[:, j]
        p_c = p_col - p_col.mean()
        p_norm = np.sqrt((p_c ** 2).sum())
        if p_norm < 1e-12:
            r_pb[j] = 0.0
        else:
            r_pb[j] = float((p_c @ theta_c) / (p_norm * theta_norm))
    return r_pb


def resolve_filter(filter_spec: str,
                   p_hat: np.ndarray,
                   beta_full: np.ndarray,
                   panel_theta: np.ndarray,
                   rpb_threshold: float = 0.0) -> np.ndarray:
    """Map a filter spec string to a boolean item mask.

    Supported specs:
        winner            β in [θ_min − 0.5, θ_max + 0.5]
        avg_0.2_0.8       panel-mean p̂ ∈ [0.2, 0.8]
        no_filter         all items
        winner_rpb_pos    winner AND r_pb ≥ rpb_threshold (default 0)
        rpb_pos           r_pb ≥ rpb_threshold (no β range filter)
    """
    J = len(beta_full)
    if filter_spec == "winner":
        lo = panel_theta.min() - 0.5
        hi = panel_theta.max() + 0.5
        return (beta_full >= lo) & (beta_full <= hi)
    if filter_spec == "avg_0.2_0.8":
        panel_mean = p_hat.mean(axis=0)
        return (panel_mean >= 0.2) & (panel_mean <= 0.8)
    if filter_spec.startswith("alo_"):
        parts = filter_spec.split("_")
        lo, hi = float(parts[1]), float(parts[2])
        in_band = (p_hat >= lo) & (p_hat <= hi)
        return in_band.any(axis=0)
    if filter_spec == "no_filter":
        return np.ones(J, dtype=bool)
    if filter_spec == "winner_rpb_pos":
        lo = panel_theta.min() - 0.5
        hi = panel_theta.max() + 0.5
        winner = (beta_full >= lo) & (beta_full <= hi)
        r_pb = compute_point_biserial(p_hat, panel_theta)
        return winner & (r_pb >= rpb_threshold)
    if filter_spec == "rpb_pos":
        r_pb = compute_point_biserial(p_hat, panel_theta)
        return r_pb >= rpb_threshold
    raise ValueError(f"Unknown filter spec: {filter_spec!r}")


def _procrustes_align(X: np.ndarray,
                      Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal Procrustes with translation: find R (orthogonal) and t
    (vector) such that Y @ R + t ≈ X, where X and Y are (N, D) matrices.

    Returns (R, t). Used for MIRT coordinate alignment.
    """
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X_c = X - X_mean
    Y_c = Y - Y_mean
    U, _s, Vt = np.linalg.svd(Y_c.T @ X_c, full_matrices=False)
    R = U @ Vt
    t = X_mean - Y_mean @ R
    return R, t


def mirt_loo(pass_counts: np.ndarray,
             N_trials: np.ndarray,
             models: list[str],
             D: int = 2,
             seed: int = 0) -> dict:
    """MIRT (D-dim) leave-one-model-out β stability via Procrustes alignment.

    MIRT is invariant under joint rotation + translation:
        θ → θ·R + t       (R orthogonal D×D, t ∈ R^D)
        a → a·R
        β → β + a·R·t^T   (β transforms to preserve logit)

    For each held-out model m, we fit MIRT on M-1 panel models and align
    back to the full-fit coordinate system via orthogonal Procrustes on
    the surviving θ's. β is then transformed via β' = β_loo + a'·t^T.

    Args:
        pass_counts: (M, J) Binomial pass counts.
        N_trials: (M, J) trial counts.
        models: panel model names.
        D: latent dimensionality (D=1 → 2PL; D=2,3 are typical).
        seed: MIRT init noise seed.

    Returns dict with per-held-out-model stats + overall pooled stats
    (same schema as claim4_loo for easy comparison).
    """
    M, J = pass_counts.shape
    theta_full, beta_full, a_full, _ = mirt.fit_mle(
        pass_counts, N_trials, D=D, seed=seed)

    per_model = []
    pooled_abs_diffs = []
    for m_drop in range(M):
        keep = [i for i in range(M) if i != m_drop]
        t_l, b_l, a_l, _ = mirt.fit_mle(
            pass_counts[keep], N_trials[keep], D=D, seed=seed)

        R, t = _procrustes_align(theta_full[keep], t_l)
        # theta_aligned = t_l @ R + t   (we only need this if reporting θ drift)
        a_aligned = a_l @ R
        beta_aligned = b_l + a_aligned @ t

        delta = beta_aligned - beta_full
        abs_d = np.abs(delta)
        pooled_abs_diffs.append(abs_d)
        per_model.append({
            "held_out": models[m_drop],
            "held_out_idx": int(m_drop),
            "procrustes_R_det": float(np.linalg.det(R)),
            "procrustes_t_norm": float(np.linalg.norm(t)),
            "median_abs_delta_beta": float(np.median(abs_d)),
            "max_abs_delta_beta": float(abs_d.max()),
            "mean_abs_delta_beta": float(abs_d.mean()),
            "frac_exceed_0p1": float((abs_d > 0.1).mean()),
            "frac_exceed_0p3": float((abs_d > 0.3).mean()),
            "frac_exceed_0p5": float((abs_d > 0.5).mean()),
        })

    pooled = np.concatenate(pooled_abs_diffs)
    overall = {
        "n_items": int(J),
        "n_panel": int(M),
        "n_loo_comparisons": int(pooled.size),
        "median_abs_delta_beta": float(np.median(pooled)),
        "mean_abs_delta_beta": float(pooled.mean()),
        "p90_abs_delta_beta": float(np.quantile(pooled, 0.90)),
        "p95_abs_delta_beta": float(np.quantile(pooled, 0.95)),
        "max_abs_delta_beta": float(pooled.max()),
        "frac_exceed_0p1": float((pooled > 0.1).mean()),
        "frac_exceed_0p3": float((pooled > 0.3).mean()),
        "frac_exceed_0p5": float((pooled > 0.5).mean()),
    }
    return {
        "irt_model": f"mirt_D{D}",
        "mirt_dim": D,
        "per_model": per_model,
        "overall": overall,
        "theta_full": theta_full.tolist(),
        "beta_full": beta_full.tolist(),
        "a_full": a_full.tolist(),
    }


def mirt_null_simulation(theta: np.ndarray,
                         beta: np.ndarray,
                         a: np.ndarray,
                         N_trials: np.ndarray,
                         D: int,
                         n_bootstrap: int = 50,
                         seed: int = 42) -> dict:
    """Parametric null for MIRT LOO β stability.

    Under correct MIRT(D) at fitted (θ, β, a), simulate Binomial responses
    and re-run mirt_loo to get null |Δβ| distribution.
    """
    rng = np.random.default_rng(seed)
    M, _D = theta.shape
    logits = theta @ a.T - beta[None, :]
    p_true = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    N_int = N_trials.astype(int)
    dummy_models = [f"sim_m{i}" for i in range(M)]

    null_medians, null_maxes, null_p95s = [], [], []
    for b in range(n_bootstrap):
        pc_sim = rng.binomial(N_int, p_true)
        res = mirt_loo(pc_sim, N_trials, dummy_models, D=D, seed=b)
        null_medians.append(res["overall"]["median_abs_delta_beta"])
        null_maxes.append(res["overall"]["max_abs_delta_beta"])
        null_p95s.append(res["overall"]["p95_abs_delta_beta"])
    nm, nx, np95 = map(np.asarray, (null_medians, null_maxes, null_p95s))
    return {
        "n_bootstrap": n_bootstrap,
        "mirt_dim": D,
        "null_median_abs_delta_beta": {
            "mean": float(nm.mean()),
            "p50": float(np.quantile(nm, 0.50)),
            "p95": float(np.quantile(nm, 0.95)),
            "p99": float(np.quantile(nm, 0.99)),
        },
        "null_max_abs_delta_beta": {
            "mean": float(nx.mean()),
            "p50": float(np.quantile(nx, 0.50)),
            "p95": float(np.quantile(nx, 0.95)),
            "p99": float(np.quantile(nx, 0.99)),
        },
        "null_p95_abs_delta_beta": {
            "mean": float(np95.mean()),
            "p95": float(np.quantile(np95, 0.95)),
        },
    }


def claim4_loo(pass_counts: np.ndarray,
               N_trials: np.ndarray,
               models: list[str],
               irt_model: str = "1pl") -> dict:
    """Run leave-one-model-out β stability analysis on a (filtered) subset.

    For each held-out model m, refits IRT (1PL or 2PL) on the remaining
    M-1 models and compares β estimates against the full-panel fit after
    coordinate-system alignment:
        1PL: pure translation (a=1, b=shift)
        2PL: affine (a, b via OLS), since 2PL has scale invariance
             (α→α/c, θ→c·θ, β→c·β). β transforms with (a,b); α as α/a.

    Returns dict with per-held-out-model stats + overall pooled stats.
    """
    M, J = pass_counts.shape
    theta_full, beta_full, alpha_full = fit_model(pass_counts, N_trials, irt_model)

    per_model = []
    pooled_abs_diffs = []
    for m_drop in range(M):
        keep = [i for i in range(M) if i != m_drop]
        pc_loo = pass_counts[keep]
        N_loo = N_trials[keep]
        theta_loo, beta_loo, alpha_loo = fit_model(pc_loo, N_loo, irt_model)

        a, b = align_coordinates(theta_full[keep], theta_loo, irt_model)
        beta_loo_aligned = a * beta_loo + b

        delta = beta_loo_aligned - beta_full
        abs_d = np.abs(delta)
        pooled_abs_diffs.append(abs_d)

        row = {
            "held_out": models[m_drop],
            "held_out_idx": int(m_drop),
            "anchor_a": a,
            "anchor_b": b,
            "median_abs_delta_beta": float(np.median(abs_d)),
            "max_abs_delta_beta": float(abs_d.max()),
            "mean_abs_delta_beta": float(abs_d.mean()),
            "frac_exceed_0p1": float((abs_d > 0.1).mean()),
            "frac_exceed_0p3": float((abs_d > 0.3).mean()),
            "frac_exceed_0p5": float((abs_d > 0.5).mean()),
        }
        if alpha_full is not None and alpha_loo is not None:
            # α transforms as α/a under the alignment transformation
            alpha_loo_aligned = alpha_loo / max(a, 1e-6)
            d_log_alpha = np.log(alpha_loo_aligned) - np.log(alpha_full)
            row["median_abs_delta_log_alpha"] = float(np.median(np.abs(d_log_alpha)))
            row["max_abs_delta_log_alpha"] = float(np.abs(d_log_alpha).max())
        per_model.append(row)

    pooled = np.concatenate(pooled_abs_diffs)
    overall = {
        "n_items": int(J),
        "n_panel": int(M),
        "n_loo_comparisons": int(pooled.size),
        "median_abs_delta_beta": float(np.median(pooled)),
        "mean_abs_delta_beta": float(pooled.mean()),
        "p90_abs_delta_beta": float(np.quantile(pooled, 0.90)),
        "p95_abs_delta_beta": float(np.quantile(pooled, 0.95)),
        "max_abs_delta_beta": float(pooled.max()),
        "frac_exceed_0p1": float((pooled > 0.1).mean()),
        "frac_exceed_0p3": float((pooled > 0.3).mean()),
        "frac_exceed_0p5": float((pooled > 0.5).mean()),
    }
    out = {
        "irt_model": irt_model,
        "per_model": per_model,
        "overall": overall,
        "theta_full": theta_full.tolist(),
        "beta_full": beta_full.tolist(),
    }
    if alpha_full is not None:
        out["alpha_full"] = alpha_full.tolist()
    return out


def verdict(overall: dict) -> str:
    """Classify stability per irt_concepts_primer.md §2.6.5 thresholds.

    NOTE: These absolute thresholds (0.1, 0.3 logit) were calibrated on
    large-M (10-20) panels. For M=6, sampling noise alone yields
    |Δβ| ≈ 0.1-0.2 logit under a correctly-specified Rasch model, so
    this absolute-threshold verdict is over-strict. Use the parametric
    null-bootstrap check (see --with-null-sim) for an M=6-calibrated
    verdict that asks: "is observed |Δβ| consistent with correct-model
    sampling noise?"
    """
    med = overall["median_abs_delta_beta"]
    mx = overall["max_abs_delta_beta"]
    if med < 0.1 and mx < 0.3:
        return "✓ STABLE (large-M threshold) — specific objectivity holds"
    if med < 0.2 and mx < 0.5:
        return "⚠ MARGINAL (large-M threshold) — check parametric null"
    return "✗ UNSTABLE (large-M threshold) — check parametric null for M=6 verdict"


def null_simulation(theta: np.ndarray,
                    beta: np.ndarray,
                    N_trials: np.ndarray,
                    alpha: np.ndarray | None = None,
                    irt_model: str = "1pl",
                    n_bootstrap: int = 100,
                    seed: int = 42) -> dict:
    """Parametric null: under fitted (θ, β), simulate Binomial responses
    and re-run LOO pipeline to build null distribution of |Δβ| statistics.

    Question asked: "if Rasch were perfectly correct at the fitted
    (θ̂, β̂), how large would LOO |Δβ| be just from M=6 × N=40 sampling
    noise?" Observed |Δβ| within 95% null CI → consistent with correct
    model (no evidence for mis-specification at this panel size).

    Args:
        theta, beta: fitted parameters to use as ground truth.
        N_trials: (M, J) trial counts (unchanged across bootstrap).
        n_bootstrap: number of synthetic datasets.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with null distribution summaries: median, p95, p99 of both
        median|Δβ| and max|Δβ| across bootstrap samples.
    """
    rng = np.random.default_rng(seed)
    M, J = len(theta), len(beta)
    if irt_model == "2pl" and alpha is not None:
        diff = alpha[None, :] * (theta[:, None] - beta[None, :])
    else:
        diff = theta[:, None] - beta[None, :]
    p_true = 1.0 / (1.0 + np.exp(-np.clip(diff, -40, 40)))
    N_int = N_trials.astype(int)
    dummy_models = [f"sim_m{i}" for i in range(M)]

    null_medians = []
    null_maxes = []
    null_p95s = []
    for b in range(n_bootstrap):
        pc_sim = rng.binomial(N_int, p_true)
        res = claim4_loo(pc_sim, N_trials, dummy_models, irt_model=irt_model)
        null_medians.append(res["overall"]["median_abs_delta_beta"])
        null_maxes.append(res["overall"]["max_abs_delta_beta"])
        null_p95s.append(res["overall"]["p95_abs_delta_beta"])

    nm = np.asarray(null_medians)
    nx = np.asarray(null_maxes)
    np95 = np.asarray(null_p95s)
    return {
        "n_bootstrap": n_bootstrap,
        "null_median_abs_delta_beta": {
            "mean": float(nm.mean()),
            "p50": float(np.quantile(nm, 0.50)),
            "p95": float(np.quantile(nm, 0.95)),
            "p99": float(np.quantile(nm, 0.99)),
        },
        "null_max_abs_delta_beta": {
            "mean": float(nx.mean()),
            "p50": float(np.quantile(nx, 0.50)),
            "p95": float(np.quantile(nx, 0.95)),
            "p99": float(np.quantile(nx, 0.99)),
        },
        "null_p95_abs_delta_beta": {
            "mean": float(np95.mean()),
            "p95": float(np.quantile(np95, 0.95)),
        },
    }


def null_verdict(overall: dict, null: dict) -> str:
    """Two-tailed bootstrap check: is observed |Δβ| within null 95% CI?"""
    obs_med = overall["median_abs_delta_beta"]
    obs_max = overall["max_abs_delta_beta"]
    null_med_95 = null["null_median_abs_delta_beta"]["p95"]
    null_max_95 = null["null_max_abs_delta_beta"]["p95"]
    med_ok = obs_med <= null_med_95
    max_ok = obs_max <= null_max_95
    if med_ok and max_ok:
        return ("✓ RASCH-CONSISTENT — observed |Δβ| within 95% null CI "
                "(expected drift under correct-model + M=6 sampling noise)")
    if med_ok or max_ok:
        return ("⚠ BORDERLINE — one of {median, max} |Δβ| exceeds 95% "
                "null CI; inspect heavy tail")
    return ("✗ MIS-SPECIFIED — both median and max |Δβ| exceed 95% null; "
            "drift is larger than M=6 sampling noise can explain")


def main():
    parser = argparse.ArgumentParser(
        description="Claim 4: leave-one-model-out β stability")
    parser.add_argument("--response-dir", required=True,
                        help="Directory with per-model response_matrix.jsonl")
    parser.add_argument("--filters", nargs="+",
                        default=["winner", "avg_0.2_0.8", "alo_0.20_0.80", "no_filter"],
                        help=("Filter specs: winner, avg_0.2_0.8, "
                              "alo_<lo>_<hi> (any-LOO p̂ in band), no_filter, "
                              "winner_rpb_pos, rpb_pos"))
    parser.add_argument("--rpb-threshold", type=float, default=0.0,
                        help=("Threshold for r_pb-based filters "
                              "(winner_rpb_pos, rpb_pos). Items with "
                              "r_pb ≥ threshold kept. Default 0.0."))
    parser.add_argument("--irt-model",
                        choices=["1pl", "1pl_eb", "2pl", "mirt2", "mirt3"],
                        default="1pl",
                        help=("IRT model: 1pl (Rasch MLE), "
                              "1pl_eb (Rasch + empirical-Bayes shrinkage), "
                              "2pl (2PL IRT), "
                              "mirt2/mirt3 (compensatory MIRT D=2/3)"))
    parser.add_argument("--output", default=None,
                        help="Path to save full JSON report (optional)")
    parser.add_argument("--with-null-sim", action="store_true",
                        help=("Run parametric null bootstrap under fitted "
                              "(θ̂, β̂). M=6-calibrated verdict asks whether "
                              "observed |Δβ| is consistent with correct-"
                              "model sampling noise."))
    parser.add_argument("--n-bootstrap", type=int, default=100,
                        help="Bootstrap samples for null simulation "
                             "(default: 100)")
    parser.add_argument("--null-sim-filter", default="winner",
                        help=("Which filter to run null sim on "
                              "(default: winner — only needed for paper "
                              "Claim 4; other filters are informational)"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-models", nargs="+", default=[],
                        help="Full model names to exclude from panel "
                             "(for panel-size ablation / M=N vs M=N-k).")
    args = parser.parse_args()

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
    pass_counts = rm["pass_counts"]
    N_trials = rm["total_trials"]
    p_hat = rm["p_hat"]
    models = rm["models"]
    M, J_pool = pass_counts.shape

    # Full-pool Rasch fit for filter definition (same convention as compare_filters.py)
    def irt_label(key):
        return {
            "1pl": "1PL Rasch",
            "1pl_eb": "1PL Rasch + Empirical Bayes shrinkage",
            "2pl": "2PL IRT",
            "mirt2": "MIRT D=2 (compensatory)",
            "mirt3": "MIRT D=3 (compensatory)",
        }[key]

    print(f"\nClaim 4: Leave-One-Model-Out β Stability")
    print(f"  IRT model: {args.irt_model.upper()}  ({irt_label(args.irt_model)})")
    print(f"  Response dir: {args.response_dir}")
    print(f"  Panel (M={M}):")
    for i, m in enumerate(models):
        print(f"    [{i}] {m}")
    print(f"  Full pool: J={J_pool}, N_median={int(np.median(N_trials))}\n")

    # Full pool fit for filter definition always uses 1PL (shared β pool
    # across filter candidates — convention matches compare_filters.py).
    print("  Fitting full pool 1PL Rasch for filter definition...")
    theta_pool, beta_pool, _ = fit_model(pass_counts, N_trials, "1pl")
    print(f"    θ_pool range: [{theta_pool.min():+.3f}, {theta_pool.max():+.3f}]")

    # r_pb diagnostic (if any rpb-based filter requested or user asked)
    uses_rpb = any("rpb" in f for f in args.filters)
    if uses_rpb:
        r_pb_all = compute_point_biserial(p_hat, theta_pool)
        print(f"\n  r_pb diagnostic (full pool, J={J_pool}):")
        print(f"    r_pb range:  [{r_pb_all.min():+.3f}, {r_pb_all.max():+.3f}]")
        print(f"    r_pb mean:   {r_pb_all.mean():+.3f}")
        print(f"    r_pb median: {np.median(r_pb_all):+.3f}")
        n_neg = int((r_pb_all < 0).sum())
        n_small_pos = int(((r_pb_all >= 0) & (r_pb_all < 0.2)).sum())
        print(f"    items with r_pb < 0:     {n_neg:4d} "
              f"({n_neg/J_pool*100:.1f}%)  [anti-correlated multi-dim]")
        print(f"    items with 0 ≤ r_pb < 0.2: {n_small_pos:4d} "
              f"({n_small_pos/J_pool*100:.1f}%)  [weakly scalable]")
        print(f"    threshold for rpb-based filter: r_pb ≥ {args.rpb_threshold}")

    all_results = {}
    for fspec in args.filters:
        mask = resolve_filter(fspec, p_hat, beta_pool, theta_pool,
                              rpb_threshold=args.rpb_threshold)
        n_kept = int(mask.sum())
        if n_kept < 2:
            print(f"\n━━━ Filter: {fspec}  (J={n_kept}) — SKIP, too few items ━━━")
            continue

        header = f"━━━ Filter: {fspec}   (J={n_kept}) ━━━"
        print(f"\n{header}")

        if args.irt_model == "1pl_eb":
            eb_diag = fit_eb(pass_counts[:, mask], N_trials[:, mask])
            shr = eb_diag["shrinkage"]
            print(f"\n  EB diagnostics (full panel fit):")
            print(f"    μ̂ (prior mean β)         = {eb_diag['mu_hat']:+.4f}")
            print(f"    τ̂ (prior SD β)           = {eb_diag['tau']:.4f} logit "
                  f"(τ²̂ = {eb_diag['tau_sq']:.4f})")
            print(f"    SE(β̂_MLE): mean={eb_diag['se_beta_fisher'].mean():.4f}, "
                  f"median={np.median(eb_diag['se_beta_fisher']):.4f}, "
                  f"max={eb_diag['se_beta_fisher'].max():.4f}")
            print(f"    shrinkage B_x: mean={shr.mean():.3f}, "
                  f"median={np.median(shr):.3f}, "
                  f"p90={np.quantile(shr, 0.9):.3f}, "
                  f"max={shr.max():.3f}")
            # Most-shrunken items
            top_shr_idx = np.argsort(shr)[-5:][::-1]
            print(f"    Top-5 shrunken items (β̂_MLE → β̂_EB):")
            for j in top_shr_idx:
                print(f"      B_x={shr[j]:.3f}  "
                      f"β_MLE={eb_diag['beta_mle'][j]:+7.3f}  "
                      f"β_EB={eb_diag['beta_eb'][j]:+7.3f}  "
                      f"SE_MLE={eb_diag['se_beta_fisher'][j]:.3f}")

        if args.irt_model.startswith("mirt"):
            D = 2 if args.irt_model == "mirt2" else 3
            res = mirt_loo(pass_counts[:, mask], N_trials[:, mask], models,
                           D=D, seed=args.seed)
        else:
            res = claim4_loo(pass_counts[:, mask], N_trials[:, mask], models,
                             irt_model=args.irt_model)
        res["n_kept"] = n_kept
        res["filter_spec"] = fspec
        all_results[fspec] = res

        # Per-held-out table
        if args.irt_model in ("1pl", "1pl_eb"):
            anchor_label = "shift"
            anchor_fmt = lambda r: f"{r['anchor_b']:>+8.3f}"
        elif args.irt_model == "2pl":
            anchor_label = "(a, b)"
            anchor_fmt = lambda r: f"({r['anchor_a']:.2f},{r['anchor_b']:+.2f})"
        else:
            anchor_label = "(det R, ‖t‖)"
            anchor_fmt = lambda r: (
                f"({r['procrustes_R_det']:+.2f},{r['procrustes_t_norm']:.3f})")
        align_desc = {
            "1pl": "1PL translation",
            "1pl_eb": "1PL translation (EB-shrunken β)",
            "2pl": "2PL full affine",
            "mirt2": "MIRT D=2 Procrustes",
            "mirt3": "MIRT D=3 Procrustes",
        }[args.irt_model]
        print(f"\n  Per held-out model (aligned — {align_desc}):")
        print(f"  {'held_out':<48} {anchor_label:>16} {'med|Δβ|':>9} "
              f"{'max|Δβ|':>9} {'%|Δβ|>0.1':>11} {'%|Δβ|>0.3':>11}")
        print("  " + "─" * 108)
        for row in res["per_model"]:
            held = row["held_out"][:48]
            print(f"  {held:<48} {anchor_fmt(row):>16} "
                  f"{row['median_abs_delta_beta']:>9.4f} "
                  f"{row['max_abs_delta_beta']:>9.4f} "
                  f"{row['frac_exceed_0p1']*100:>10.1f}% "
                  f"{row['frac_exceed_0p3']*100:>10.1f}%")

        o = res["overall"]
        print(f"\n  Overall (pooled across {M} LOO runs × {n_kept} items = "
              f"{o['n_loo_comparisons']} |Δβ| observations):")
        print(f"    median|Δβ|     = {o['median_abs_delta_beta']:.4f} logit"
              f"  (target: < 0.10)")
        print(f"    mean|Δβ|       = {o['mean_abs_delta_beta']:.4f} logit")
        print(f"    p90|Δβ|        = {o['p90_abs_delta_beta']:.4f} logit")
        print(f"    p95|Δβ|        = {o['p95_abs_delta_beta']:.4f} logit")
        print(f"    max|Δβ|        = {o['max_abs_delta_beta']:.4f} logit"
              f"  (target: < 0.30)")
        print(f"    %|Δβ|>0.1 logit = {o['frac_exceed_0p1']*100:.1f}%")
        print(f"    %|Δβ|>0.3 logit = {o['frac_exceed_0p3']*100:.1f}%")
        print(f"    %|Δβ|>0.5 logit = {o['frac_exceed_0p5']*100:.1f}%")
        print(f"\n  Verdict (large-M threshold): {verdict(o)}")

        if args.with_null_sim and fspec == args.null_sim_filter:
            print(f"\n  Running parametric null bootstrap "
                  f"(n_bootstrap={args.n_bootstrap})...")
            if args.irt_model.startswith("mirt"):
                D = 2 if args.irt_model == "mirt2" else 3
                null = mirt_null_simulation(
                    np.asarray(res["theta_full"]),
                    np.asarray(res["beta_full"]),
                    np.asarray(res["a_full"]),
                    N_trials[:, mask],
                    D=D,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed,
                )
            else:
                alpha_full_arr = (np.asarray(res["alpha_full"])
                                  if "alpha_full" in res else None)
                null = null_simulation(
                    np.asarray(res["theta_full"]),
                    np.asarray(res["beta_full"]),
                    N_trials[:, mask],
                    alpha=alpha_full_arr,
                    irt_model=args.irt_model,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed,
                )
            res["null_simulation"] = null
            print(f"\n  Null distribution (under correct Rasch + M=6 sampling):")
            print(f"    median|Δβ|: mean={null['null_median_abs_delta_beta']['mean']:.4f}, "
                  f"p95={null['null_median_abs_delta_beta']['p95']:.4f}, "
                  f"p99={null['null_median_abs_delta_beta']['p99']:.4f}")
            print(f"    max|Δβ|:    mean={null['null_max_abs_delta_beta']['mean']:.4f}, "
                  f"p95={null['null_max_abs_delta_beta']['p95']:.4f}, "
                  f"p99={null['null_max_abs_delta_beta']['p99']:.4f}")
            print(f"\n  Observed vs null 95% CI:")
            print(f"    median|Δβ|: observed={o['median_abs_delta_beta']:.4f}  "
                  f"null-95%={null['null_median_abs_delta_beta']['p95']:.4f}")
            print(f"    max|Δβ|:    observed={o['max_abs_delta_beta']:.4f}  "
                  f"null-95%={null['null_max_abs_delta_beta']['p95']:.4f}")
            print(f"\n  Verdict (M=6-calibrated): {null_verdict(o, null)}")

    if args.output:
        payload = {
            "response_dir": args.response_dir,
            "n_models": M,
            "n_items_pool": J_pool,
            "panel": models,
            "irt_model": args.irt_model,
            "coordinate_alignment": (
                "translation-only (a=1)" if args.irt_model == "1pl"
                else "affine (a, b via OLS on θ)"
            ),
            "thresholds_doc": "docs/irt_concepts_primer.md §2.6.5",
            "results": all_results,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nJSON → {args.output}")


if __name__ == "__main__":
    main()
