"""IRT validation on random_filter/final_n40 response matrix.

Runs three validation sections:
  Section 1 — Baseline fit: 1PL vs 2PL vs 3PL (AIC / BIC / LRT)
  Section 5 — Predictive: trial-split holdout + cell-mask holdout + calibration
  Section 7 — Boundary consistency: |θ_m − b_i| ≈ 0 ⟺ p̂ ≈ 0.5

Usage:
    python scripts/analysis/validate_irt.py \\
        --data results/harmful_refusal/random_filter/final_n40 \\
        --output results/harmful_refusal/random_filter/final_n40/irt_validation
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difficulty.rasch import _sigmoid, fit_mle, fit_ols, load_response_matrix


# ---------------------------------------------------------------------------
# Trial-level loader (for Section 5 trial-split holdout)
# ---------------------------------------------------------------------------

def load_trial_tensor(data_dir: str) -> tuple[list, list, np.ndarray]:
    """Load (M, J, T) binary trial tensor plus models/items lists.

    Missing cells / short trials filled with NaN.
    """
    data_path = Path(data_dir)
    raw: dict = {}
    for model_dir in sorted(data_path.iterdir()):
        if not model_dir.is_dir():
            continue
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
                trials = [bool(t["passed"]) for t in entry["trials"]]
                raw[(model_name, pid)] = trials
    models = sorted({k[0] for k in raw})
    items = sorted({k[1] for k in raw})
    M, J = len(models), len(items)
    T = max(len(v) for v in raw.values())
    model_idx = {m: i for i, m in enumerate(models)}
    item_idx = {p: j for j, p in enumerate(items)}
    tensor = np.full((M, J, T), np.nan)
    for (m, p), trials in raw.items():
        i, j = model_idx[m], item_idx[p]
        tensor[i, j, : len(trials)] = trials
    return models, items, tensor


# ---------------------------------------------------------------------------
# 2PL: P = σ(a_j · (θ_m − b_j))
# ---------------------------------------------------------------------------

def _nll_2pl(params, K, N, M, J, log_a_sigma):
    """2PL NLL with log-normal(0, log_a_sigma) regularization on log(a)."""
    theta = params[:M]
    log_a = params[M : M + J]
    b = params[M + J :]
    a = np.exp(log_a)
    logit = a[None, :] * (theta[:, None] - b[None, :])
    p = np.clip(_sigmoid(logit), 1e-10, 1 - 1e-10)
    ll = (K * np.log(p) + (N - K) * np.log(1 - p)).sum()
    # Regularizer: sum_j 0.5 * (log a_j / sigma)^2
    reg = 0.5 * (log_a ** 2).sum() / (log_a_sigma ** 2)
    return -ll + reg


def _nll_2pl_grad(params, K, N, M, J, log_a_sigma):
    theta = params[:M]
    log_a = params[M : M + J]
    b = params[M + J :]
    a = np.exp(log_a)
    diff = theta[:, None] - b[None, :]
    p = np.clip(_sigmoid(a[None, :] * diff), 1e-10, 1 - 1e-10)
    residual = K - N * p  # (M, J)
    grad_theta = -(a[None, :] * residual).sum(axis=1)
    grad_b = (a[None, :] * residual).sum(axis=0)
    grad_log_a = -((diff * residual).sum(axis=0)) * a + log_a / (log_a_sigma ** 2)
    return np.concatenate([grad_theta, grad_log_a, grad_b])


def fit_2pl(K, N, theta_init, b_init, log_a_sigma=1.0):
    """Fit 2PL with log-normal regularization on discrimination.

    Returns unregularized NLL for information-criterion comparisons.
    """
    M, J = K.shape
    x0 = np.concatenate([theta_init, np.zeros(J), b_init])
    result = minimize(
        _nll_2pl,
        x0,
        args=(K, N, M, J, log_a_sigma),
        jac=_nll_2pl_grad,
        method="L-BFGS-B",
        options={"maxiter": 3000, "ftol": 1e-10},
    )
    theta = result.x[:M]
    a = np.exp(result.x[M : M + J])
    b = result.x[M + J :]
    shift = theta.mean()
    theta -= shift
    b -= shift
    # Recompute raw NLL without regularizer for IC
    p = np.clip(_sigmoid(a[None, :] * (theta[:, None] - b[None, :])), 1e-10, 1 - 1e-10)
    raw_nll = -(K * np.log(p) + (N - K) * np.log(1 - p)).sum()
    return theta, a, b, float(raw_nll)


# ---------------------------------------------------------------------------
# 3PL: P = c_j + (1 − c_j) · σ(a_j · (θ_m − b_j))
#   with Beta(5, 17) penalty on c (mean ≈ 0.22) to stabilize
# ---------------------------------------------------------------------------

def _nll_3pl(params, K, N, M, J, log_a_sigma, c_prior_a, c_prior_b):
    theta = params[:M]
    log_a = params[M : M + J]
    b = params[M + J : M + 2 * J]
    logit_c = params[M + 2 * J :]
    a = np.exp(log_a)
    c = _sigmoid(logit_c)
    p_irt = _sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
    p = np.clip(c[None, :] + (1 - c[None, :]) * p_irt, 1e-10, 1 - 1e-10)
    ll = (K * np.log(p) + (N - K) * np.log(1 - p)).sum()
    reg_a = 0.5 * (log_a ** 2).sum() / (log_a_sigma ** 2)
    log_prior_c = (
        (c_prior_a - 1) * np.log(np.clip(c, 1e-10, 1))
        + (c_prior_b - 1) * np.log(np.clip(1 - c, 1e-10, 1))
    ).sum()
    return -(ll + log_prior_c) + reg_a


def fit_3pl(K, N, theta_2pl, a_2pl, b_2pl, log_a_sigma=1.0, c_prior=(1.0, 10.0)):
    """Warm-start 3PL from 2PL fit. c prior defaults to Beta(1, 10), mean ≈ 0.09."""
    M, J = K.shape
    log_a_init = np.log(np.maximum(a_2pl, 1e-3))
    x0 = np.concatenate(
        [theta_2pl, log_a_init, b_2pl, np.full(J, -3.0)]  # c_init ≈ 0.047
    )
    result = minimize(
        _nll_3pl,
        x0,
        args=(K, N, M, J, log_a_sigma, c_prior[0], c_prior[1]),
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-8},
    )
    theta = result.x[:M]
    a = np.exp(result.x[M : M + J])
    b = result.x[M + J : M + 2 * J]
    c = _sigmoid(result.x[M + 2 * J :])
    shift = theta.mean()
    theta -= shift
    b -= shift
    p_irt = _sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
    p = np.clip(c[None, :] + (1 - c[None, :]) * p_irt, 1e-10, 1 - 1e-10)
    raw_nll = -(K * np.log(p) + (N - K) * np.log(1 - p)).sum()
    return theta, a, b, c, float(raw_nll)


# ---------------------------------------------------------------------------
# Section 1 — baseline fit comparison
# ---------------------------------------------------------------------------

def section1_baseline(K, N, theta_ols, b_ols):
    M, J = K.shape
    n_cells = int((N > 0).sum())

    # 1PL (reuse existing)
    theta1, b1, nll1 = fit_mle(K, N, theta_ols, b_ols)
    # 2PL — regularize log(a) with log-normal(0, 1)
    theta2, a2, b2, nll2 = fit_2pl(K, N, theta1, b1, log_a_sigma=1.0)
    # 3PL — warm-start from 2PL, weak Beta(1, 10) on c
    theta3, a3, b3, c3, nll3 = fit_3pl(K, N, theta2, a2, b2)

    def _ic(nll, p):
        return {
            "nll": float(nll),
            "n_params": int(p),
            "aic": float(2 * nll + 2 * p),
            "bic": float(2 * nll + p * np.log(n_cells)),
        }

    # Effective free params after identifiability constraints
    # 1PL: M + J − 1 (mean(θ) fixed)
    # 2PL: M + 2J − 2 (mean(θ), scale of a)
    # 3PL: M + 3J − 2
    p1, p2, p3 = M + J - 1, M + 2 * J - 2, M + 3 * J - 2
    ic1, ic2, ic3 = _ic(nll1, p1), _ic(nll2, p2), _ic(nll3, p3)

    def _lrt(nll_null, nll_alt, df):
        stat = 2.0 * (nll_null - nll_alt)
        return {
            "lrt_stat": float(stat),
            "df": int(df),
            "p_value": float(1.0 - chi2.cdf(max(stat, 0.0), df)),
        }

    return {
        "n_cells": n_cells,
        "1PL": ic1,
        "2PL": ic2,
        "3PL": ic3,
        "lrt_1pl_vs_2pl": _lrt(nll1, nll2, J - 1),
        "lrt_2pl_vs_3pl": _lrt(nll2, nll3, J),
        "discrimination_2pl": {
            "mean": float(a2.mean()),
            "std": float(a2.std()),
            "min": float(a2.min()),
            "max": float(a2.max()),
        },
        "guessing_3pl": {
            "mean": float(c3.mean()),
            "median": float(np.median(c3)),
            "max": float(c3.max()),
        },
        "params": {
            "theta_1pl": theta1.tolist(),
            "b_1pl": b1.tolist(),
            "theta_2pl": theta2.tolist(),
            "a_2pl": a2.tolist(),
            "b_2pl": b2.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Section 5 — predictive validation
# ---------------------------------------------------------------------------

def _eval_pred(p_pred, K_test, N_test, trial_tensor, train_mask):
    """Compute Brier, log-loss, AUC given predicted (M, J) probabilities."""
    from sklearn.metrics import roc_auc_score

    mask = N_test > 0
    p_test_obs = np.where(mask, K_test / np.maximum(N_test, 1), np.nan)
    brier = float(((p_pred[mask] - p_test_obs[mask]) ** 2).mean())
    p_clip = np.clip(p_pred[mask], 1e-10, 1 - 1e-10)
    ll = (
        K_test[mask] * np.log(p_clip)
        + (N_test[mask] - K_test[mask]) * np.log(1 - p_clip)
    ).sum() / N_test[mask].sum()
    M, J, _T = trial_tensor.shape
    y_true, y_score = [], []
    for i in range(M):
        for j in range(J):
            if not mask[i, j]:
                continue
            t_idx = np.where(~train_mask[i, j] & ~np.isnan(trial_tensor[i, j]))[0]
            for t in t_idx:
                y_true.append(int(trial_tensor[i, j, t]))
                y_score.append(float(p_pred[i, j]))
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")
    return {"brier": brier, "log_loss": float(-ll), "auc": auc}


def trial_split_holdout(trial_tensor, n_train, n_repeats, seed=42):
    """For each cell: random n_train/n_test split, fit 1PL and 2PL on train, predict test."""
    rng = np.random.default_rng(seed)
    M, J, T = trial_tensor.shape
    n_test = T - n_train
    reps_1pl, reps_2pl = [], []
    for rep in range(n_repeats):
        perm = np.argsort(rng.random((M, J, T)), axis=2)
        train_mask = perm < n_train
        obs_valid = ~np.isnan(trial_tensor)
        tv = np.nan_to_num(trial_tensor, nan=0.0)
        K_train = np.where(train_mask & obs_valid, tv, 0.0).sum(axis=2)
        N_train = (train_mask & obs_valid).sum(axis=2).astype(float)
        K_test = np.where(~train_mask & obs_valid, tv, 0.0).sum(axis=2)
        N_test = (~train_mask & obs_valid).sum(axis=2).astype(float)

        p_fill = np.where(N_train > 0, K_train / np.maximum(N_train, 1), 0.5)
        theta_ols, b_ols = fit_ols(p_fill)
        theta1, b1, _ = fit_mle(K_train, N_train, theta_ols, b_ols)
        theta2, a2, b2, _ = fit_2pl(K_train, N_train, theta1, b1, log_a_sigma=1.0)

        p_pred_1pl = _sigmoid(theta1[:, None] - b1[None, :])
        p_pred_2pl = _sigmoid(a2[None, :] * (theta2[:, None] - b2[None, :]))

        reps_1pl.append(_eval_pred(p_pred_1pl, K_test, N_test, trial_tensor, train_mask))
        reps_2pl.append(_eval_pred(p_pred_2pl, K_test, N_test, trial_tensor, train_mask))

    def _agg(reps, k):
        vals = [r[k] for r in reps]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    def _summary(reps):
        return {k: _agg(reps, k) for k in ("brier", "log_loss", "auc")}

    return {
        "n_train": n_train,
        "n_test": n_test,
        "n_repeats": n_repeats,
        "1PL": _summary(reps_1pl),
        "2PL": _summary(reps_2pl),
    }


def cell_mask_holdout(K, N, n_folds, frac, seed=42):
    """Mask random frac of cells, fit 1PL on remaining, predict masked cells' p_hat."""
    rng = np.random.default_rng(seed)
    M, J = K.shape
    all_idx = np.array([(i, j) for i in range(M) for j in range(J)])
    p_hat_obs = K / np.maximum(N, 1)
    results = []
    for fold in range(n_folds):
        perm = rng.permutation(len(all_idx))
        n_mask = int(len(all_idx) * frac)
        mask_idx = all_idx[perm[:n_mask]]
        mask_mat = np.zeros_like(K, dtype=bool)
        mask_mat[mask_idx[:, 0], mask_idx[:, 1]] = True

        K_train = np.where(mask_mat, 0.0, K)
        N_train = np.where(mask_mat, 0.0, N)
        # avoid zero-trial rows/cols issue: use OLS with p_hat=0.5 fill
        p_fill = np.where(N_train > 0, K_train / np.maximum(N_train, 1), 0.5)
        theta_ols, b_ols = fit_ols(p_fill)
        theta, b, _ = fit_mle(K_train, N_train, theta_ols, b_ols)
        p_pred = _sigmoid(theta[:, None] - b[None, :])
        diff = p_pred[mask_mat] - p_hat_obs[mask_mat]
        results.append(
            {
                "mae": float(np.abs(diff).mean()),
                "rmse": float(np.sqrt((diff ** 2).mean())),
                "corr_pearson": float(np.corrcoef(p_pred[mask_mat], p_hat_obs[mask_mat])[0, 1]),
                "corr_spearman": float(spearmanr(p_pred[mask_mat], p_hat_obs[mask_mat])[0]),
            }
        )

    def _agg(k):
        vals = [r[k] for r in results]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return {
        "n_folds": n_folds,
        "frac_masked": frac,
        "mae": _agg("mae"),
        "rmse": _agg("rmse"),
        "corr_pearson": _agg("corr_pearson"),
        "corr_spearman": _agg("corr_spearman"),
    }


def calibration_data(p_pred, p_hat, n_bins=10):
    """Return binned (mean_pred, mean_obs, count) for calibration plot."""
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(p_pred, edges) - 1, 0, n_bins - 1)
    mean_pred, mean_obs, counts = [], [], []
    for b in range(n_bins):
        sel = bin_idx == b
        if sel.sum() == 0:
            continue
        mean_pred.append(float(p_pred[sel].mean()))
        mean_obs.append(float(p_hat[sel].mean()))
        counts.append(int(sel.sum()))
    return mean_pred, mean_obs, counts


# ---------------------------------------------------------------------------
# Section 7 — boundary consistency
# ---------------------------------------------------------------------------

def section7_boundary(theta, b, p_hat, eps=0.3):
    M, J = p_hat.shape
    diff = theta[:, None] - b[None, :]  # (M, J)
    boundary_mask = np.abs(diff) < eps
    n_boundary = int(boundary_mask.sum())
    if n_boundary > 0:
        p_boundary = p_hat[boundary_mask]
        boundary_stats = {
            "n_cells": n_boundary,
            "p_hat_mean": float(p_boundary.mean()),
            "p_hat_median": float(np.median(p_boundary)),
            "p_hat_std": float(p_boundary.std()),
            "frac_in_0.3_to_0.7": float(
                ((p_boundary > 0.3) & (p_boundary < 0.7)).mean()
            ),
        }
    else:
        boundary_stats = {"n_cells": 0}

    # Global: σ(θ - b) vs p_hat
    p_pred = _sigmoid(diff).flatten()
    p_obs = p_hat.flatten()
    slope, intercept = np.polyfit(p_pred, p_obs, 1)
    r2 = 1 - ((p_obs - (slope * p_pred + intercept)) ** 2).sum() / (
        (p_obs - p_obs.mean()) ** 2
    ).sum()
    return {
        "eps": eps,
        "boundary_zone": boundary_stats,
        "global_scatter": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2),
            "pearson_r": float(np.corrcoef(p_pred, p_obs)[0, 1]),
            "spearman_r": float(spearmanr(p_pred, p_obs)[0]),
        },
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_calibration(p_pred_flat, p_obs_flat, out_path):
    mean_pred, mean_obs, counts = calibration_data(p_pred_flat, p_obs_flat, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    sizes = np.array(counts) / max(counts) * 200
    ax.scatter(mean_pred, mean_obs, s=sizes, alpha=0.7, edgecolor="black")
    ax.set_xlabel("predicted P(pass)")
    ax.set_ylabel("observed p̂")
    ax.set_title("1PL calibration (cell-level)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_boundary_scatter(theta, b, p_hat, out_path):
    diff = theta[:, None] - b[None, :]
    p_pred = _sigmoid(diff).flatten()
    p_obs = p_hat.flatten()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="y=x")
    ax.scatter(p_pred, p_obs, s=4, alpha=0.25)
    ax.set_xlabel("predicted P(pass) = σ(θ − b)")
    ax.set_ylabel("observed p̂")
    ax.set_title("Boundary consistency (all cells)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_info_criteria(bic_dict, out_path):
    labels = ["1PL", "2PL", "3PL"]
    bic = [bic_dict[k]["bic"] for k in labels]
    aic = [bic_dict[k]["aic"] for k in labels]
    nll = [bic_dict[k]["nll"] for k in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    w = 0.25
    ax.bar(x - w, nll, w, label="NLL")
    ax.bar(x, aic, w, label="AIC")
    ax.bar(x + w, bic, w, label="BIC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("value (lower = better)")
    ax.set_title("Baseline model comparison")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--n-repeats-trial-split", type=int, default=5)
    ap.add_argument("--n-folds-cell-mask", type=int, default=5)
    ap.add_argument("--cell-mask-frac", type=float, default=0.1)
    ap.add_argument("--boundary-eps", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(exist_ok=True)

    # ----- Load -----
    print(f"Loading aggregated matrix from {args.data}")
    rm = load_response_matrix(args.data)
    models = rm["models"]
    items = rm["items"]
    K, N, p_hat = rm["pass_counts"], rm["total_trials"], rm["p_hat"]
    print(f"  matrix: {len(models)} models × {len(items)} items, "
          f"mean N/cell = {N.mean():.1f}")

    print(f"Loading trial tensor for trial-split holdout…")
    _m, _i, trial_tensor = load_trial_tensor(args.data)
    print(f"  tensor shape: {trial_tensor.shape}, "
          f"NaN frac = {np.isnan(trial_tensor).mean():.4f}")

    # ----- Section 1: baseline comparison -----
    print("\n=== Section 1: 1PL / 2PL / 3PL baseline fit ===")
    theta_ols, b_ols = fit_ols(p_hat)
    sec1 = section1_baseline(K, N, theta_ols, b_ols)
    for mdl in ("1PL", "2PL", "3PL"):
        r = sec1[mdl]
        print(f"  {mdl}: NLL={r['nll']:.1f}  AIC={r['aic']:.1f}  "
              f"BIC={r['bic']:.1f}  params={r['n_params']}")
    print(f"  LRT 1PL→2PL: stat={sec1['lrt_1pl_vs_2pl']['lrt_stat']:.1f} "
          f"df={sec1['lrt_1pl_vs_2pl']['df']} p={sec1['lrt_1pl_vs_2pl']['p_value']:.4g}")
    print(f"  LRT 2PL→3PL: stat={sec1['lrt_2pl_vs_3pl']['lrt_stat']:.1f} "
          f"df={sec1['lrt_2pl_vs_3pl']['df']} p={sec1['lrt_2pl_vs_3pl']['p_value']:.4g}")
    print(f"  2PL discrim a: mean={sec1['discrimination_2pl']['mean']:.3f} "
          f"std={sec1['discrimination_2pl']['std']:.3f}")
    print(f"  3PL guessing c: mean={sec1['guessing_3pl']['mean']:.4f} "
          f"max={sec1['guessing_3pl']['max']:.4f}")

    # ----- Section 5: predictive validation -----
    print("\n=== Section 5: Predictive validation (1PL) ===")
    print("  → trial-split holdout")
    trial_split = trial_split_holdout(
        trial_tensor, args.n_train, args.n_repeats_trial_split, seed=args.seed
    )
    for tag in ("1PL", "2PL"):
        r = trial_split[tag]
        print(f"    {tag}: Brier={r['brier']['mean']:.4f}±{r['brier']['std']:.4f}  "
              f"log-loss={r['log_loss']['mean']:.4f}  "
              f"AUC={r['auc']['mean']:.4f}")
    print("  → cell-mask holdout")
    cell_mask = cell_mask_holdout(
        K, N, args.n_folds_cell_mask, args.cell_mask_frac, seed=args.seed
    )
    print(f"    MAE={cell_mask['mae']['mean']:.4f}  "
          f"RMSE={cell_mask['rmse']['mean']:.4f}  "
          f"ρ_spearman={cell_mask['corr_spearman']['mean']:.4f}")

    # ----- Section 7: boundary consistency -----
    print("\n=== Section 7: Boundary consistency ===")
    theta1 = np.array(sec1["params"]["theta_1pl"])
    b1 = np.array(sec1["params"]["b_1pl"])
    sec7 = section7_boundary(theta1, b1, p_hat, eps=args.boundary_eps)
    bz = sec7["boundary_zone"]
    gs = sec7["global_scatter"]
    print(f"  boundary cells |θ−b|<{args.boundary_eps}: n={bz['n_cells']}")
    if bz["n_cells"] > 0:
        print(f"    p̂ mean={bz['p_hat_mean']:.3f}  median={bz['p_hat_median']:.3f}  "
              f"std={bz['p_hat_std']:.3f}")
        print(f"    frac p̂∈(0.3, 0.7): {bz['frac_in_0.3_to_0.7']:.3f}")
    print(f"  global scatter: slope={gs['slope']:.3f}  intercept={gs['intercept']:.3f}  "
          f"R²={gs['r2']:.3f}  r={gs['pearson_r']:.3f}  ρ={gs['spearman_r']:.3f}")

    # ----- Plots -----
    print("\nSaving plots…")
    p_pred_1pl = _sigmoid(theta1[:, None] - b1[None, :])
    plot_calibration(p_pred_1pl.flatten(), p_hat.flatten(), fig_dir / "calibration.png")
    plot_boundary_scatter(theta1, b1, p_hat, fig_dir / "boundary_scatter.png")
    plot_info_criteria(
        {k: sec1[k] for k in ("1PL", "2PL", "3PL")},
        fig_dir / "info_criteria.png",
    )

    # ----- Save JSON -----
    report = {
        "data_dir": str(args.data),
        "models": models,
        "n_items": len(items),
        "section1_baseline": sec1,
        "section5_trial_split": trial_split,
        "section5_cell_mask": cell_mask,
        "section7_boundary": sec7,
        "args": vars(args),
    }
    report_path = out_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport → {report_path}")
    print(f"Plots  → {fig_dir}/")


if __name__ == "__main__":
    main()
