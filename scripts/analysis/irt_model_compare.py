"""IRT model comparison (1PL / 2PL / 3PL / MIRT-2D / MIRT-3D), per category.

Two data sources are supported:
  --data-source sgbs   (default): Claim 2 response matrix (anchor + 9*K EC),
      held-out = diagonal block {(m, x) : x in EC(m)}.
  --data-source anchor: filtered anchor block only, held-out = random cell
      mask with fixed seed (default seed=0, hold-out rate=0.1).

All models share the same fit/score pipeline. The decisive metric is
held-out NLL: if a richer model does not reduce held-out NLL despite
extra parameters, the added flexibility does not generalize and 1PL is
the right choice given specific objectivity for cross-panel beta
transfer.

Reported per category × model:
  - Visible-cell in-sample NLL and BIC
  - Held-out NLL, mean p_tilde, p_tilde range, cell-RMSE
  - 2PL/3PL: alpha distribution; 3PL: c distribution
  - MIRT: log-||a|| distribution

Usage:
    conda run -n boundary python scripts/analysis/irt_model_compare.py
    conda run -n boundary python scripts/analysis/irt_model_compare.py --data-source anchor
    conda run -n boundary python scripts/analysis/irt_model_compare.py --category harmful_refusal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from difficulty import mirt as _mirt  # noqa: E402


PANEL = [
    "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "google/gemma-3-27b-it",
    "Qwen/Qwen3-8B",
    "google/gemma-3-12b-it",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "google/gemma-3-4b-it",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
]


CATEGORY_CONFIG = {
    "harmful_refusal": {
        "anchor_dir": Path("results/harmful_refusal/filter_ablation_raw_v2_judgefix"),
        "ec_dataset": Path("benchmarks/harmful_refusal/edge_case_dataset_slot50.jsonl"),
        "ec_val_dir": Path("results/harmful_refusal/validation/edge_case_slot50"),
    },
    "over_refusal": {
        "anchor_dir": Path("results/over_refusal/filter_ablation_raw_v2_judgefix"),
        "ec_dataset": Path("benchmarks/over_refusal/edge_case_dataset_slot50.jsonl"),
        "ec_val_dir": Path("results/over_refusal/validation/edge_case_slot50"),
    },
    "constrained_completion": {
        "anchor_dir": Path("results/constrained_completion/filter_ablation_raw_v2"),
        "ec_dataset": Path("benchmarks/constrained_completion/edge_case_dataset_slot50.jsonl"),
        "ec_val_dir": Path("results/constrained_completion/validation/edge_case_slot50"),
    },
    "sycophancy_resistance": {
        "anchor_dir": Path("results/sycophancy_resistance/filter_ablation_raw_v2"),
        "ec_dataset": Path("benchmarks/sycophancy_resistance/edge_case_dataset_slot50.jsonl"),
        "ec_val_dir": Path("results/sycophancy_resistance/validation/edge_case_slot50"),
    },
}

FILTER_LOW = 0.05
FILTER_HIGH = 0.95


def model_to_dirname(model: str) -> str:
    return model.replace("/", "_").replace("-", "_")


def load_anchor_matrix(anchor_dir: Path, panel: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    item_ids = None
    K_rows, N_rows = [], []
    for model in panel:
        sub = anchor_dir / model_to_dirname(model)
        rm = sub / "response_matrix.jsonl"
        if not rm.exists():
            raise FileNotFoundError(f"missing anchor response matrix: {rm}")
        rows = [json.loads(line) for line in rm.read_text().split("\n") if line.strip()]
        rows.sort(key=lambda r: r["prompt_id"])
        ids = [r["prompt_id"] for r in rows]
        if item_ids is None:
            item_ids = ids
        elif item_ids != ids:
            raise ValueError(f"anchor item_ids mismatch for {model}")
        K_rows.append(np.array([r["pass_count"] for r in rows], dtype=float))
        N_rows.append(np.array([r["total_trials"] for r in rows], dtype=float))
    return item_ids, np.stack(K_rows), np.stack(N_rows)


def load_ec_blocks(
    ec_dataset: Path, ec_val_dir: Path, panel: list[str]
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, str]]:
    panel_set = set(panel)
    rows = [json.loads(line) for line in ec_dataset.read_text().split("\n") if line.strip()]
    rows = [r for r in rows if r["source_model"] in panel_set]
    rows.sort(key=lambda r: r["prompt_id"])
    item_ids = [r["prompt_id"] for r in rows]
    source_map = {r["prompt_id"]: r["source_model"] for r in rows}
    diag_pass = {r["prompt_id"]: r["pass_count"] for r in rows}
    diag_total = {r["prompt_id"]: r["total_trials"] for r in rows}

    M, J = len(panel), len(item_ids)
    K = np.zeros((M, J), dtype=float)
    N = np.zeros((M, J), dtype=float)
    for m_idx, model in enumerate(panel):
        sub = ec_val_dir / model_to_dirname(model)
        rm = sub / "response_matrix.jsonl"
        if not rm.exists():
            raise FileNotFoundError(f"missing EC validation matrix: {rm}")
        per_pid = {}
        for line in rm.read_text().split("\n"):
            if not line.strip():
                continue
            r = json.loads(line)
            per_pid[r["prompt_id"]] = (r["pass_count"], r["total_trials"], len(r.get("trials", [])))
        n_missing = 0
        for j, pid in enumerate(item_ids):
            pc, tt, n_trials = per_pid[pid]
            if n_trials == 0:
                if source_map[pid] == model:
                    # diagonal cell — load m's responses on EC(m) from SGBS dataset
                    K[m_idx, j] = diag_pass[pid]
                    N[m_idx, j] = diag_total[pid]
                else:
                    # missing off-diagonal: leave (K, N) = (0, 0); mask layer drops it
                    n_missing += 1
            else:
                K[m_idx, j] = pc
                N[m_idx, j] = tt
        if n_missing > 0:
            print(f"    [warn] {model}: {n_missing} non-diagonal EC cells missing trials")
    return item_ids, K, N, source_map


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _ols_init(K: np.ndarray, N: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    M, J = K.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(N > 0, K / N, 0.5)
    p = np.clip(p, 0.025, 0.975)
    y = np.log(p / (1 - p))
    masked_y = np.where(mask > 0, y, np.nan)
    grand = np.nanmean(masked_y)
    row = np.nanmean(masked_y, axis=1)
    col = np.nanmean(masked_y, axis=0)
    theta0 = np.nan_to_num(row - grand, nan=0.0)
    beta0 = np.nan_to_num(-(col - grand), nan=0.0)
    return theta0, beta0


def fit_masked_1pl(K: np.ndarray, N: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    M, J = K.shape
    W = mask.astype(float)
    theta0, beta0 = _ols_init(K, N, mask)

    def nll(p):
        th = p[:M]
        be = p[M:]
        diff = th[:, None] - be[None, :]
        pp = np.clip(_sigmoid(diff), 1e-10, 1 - 1e-10)
        ll = K * np.log(pp) + (N - K) * np.log(1 - pp)
        return -float((W * ll).sum())

    def grad(p):
        th = p[:M]
        be = p[M:]
        diff = th[:, None] - be[None, :]
        pp = np.clip(_sigmoid(diff), 1e-10, 1 - 1e-10)
        r = (K - N * pp) * W
        return np.concatenate([-r.sum(axis=1), r.sum(axis=0)])

    x0 = np.concatenate([theta0, beta0])
    out = minimize(nll, x0, jac=grad, method="L-BFGS-B",
                   options={"maxiter": 5000, "ftol": 1e-11, "gtol": 1e-8})
    th, be = out.x[:M], out.x[M:]
    sh = th.mean()
    return th - sh, be - sh, float(out.fun)


def fit_masked_2pl(
    K: np.ndarray,
    N: np.ndarray,
    mask: np.ndarray,
    log_alpha_bounds: tuple[float, float] = (-2.0, 2.0),
    theta_init: np.ndarray | None = None,
    beta_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """2PL: p(m, x) = sigma(alpha_x * (theta_m - beta_x))."""
    M, J = K.shape
    W = mask.astype(float)

    if theta_init is None or beta_init is None:
        # Warm-start from masked 1PL — gives 2PL a head start at the 1PL solution.
        th0, be0, _ = fit_masked_1pl(K, N, mask)
        if theta_init is None:
            theta_init = th0
        if beta_init is None:
            beta_init = be0
    log_alpha_init = np.zeros(J)

    def nll(p):
        th = p[:M]
        be = p[M:M + J]
        la = p[M + J:]
        a = np.exp(la)
        eta = a[None, :] * (th[:, None] - be[None, :])
        pp = np.clip(_sigmoid(eta), 1e-10, 1 - 1e-10)
        ll = K * np.log(pp) + (N - K) * np.log(1 - pp)
        return -float((W * ll).sum())

    def grad(p):
        th = p[:M]
        be = p[M:M + J]
        la = p[M + J:]
        a = np.exp(la)
        diff = th[:, None] - be[None, :]
        eta = a[None, :] * diff
        pp = np.clip(_sigmoid(eta), 1e-10, 1 - 1e-10)
        r = (K - N * pp) * W           # ∂LL/∂η; weighted
        g_th = -(r * a[None, :]).sum(axis=1)
        g_be = (r * a[None, :]).sum(axis=0)
        g_la = -(a * (r * diff).sum(axis=0))
        return np.concatenate([g_th, g_be, g_la])

    x0 = np.concatenate([theta_init, beta_init, log_alpha_init])
    bounds = [(None, None)] * (M + J) + [log_alpha_bounds] * J
    out = minimize(nll, x0, jac=grad, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 5000, "ftol": 1e-11, "gtol": 1e-8})
    th = out.x[:M]
    be = out.x[M:M + J]
    a = np.exp(out.x[M + J:])
    sh = th.mean()
    return th - sh, be - sh, a, float(out.fun)


def fit_masked_3pl(
    K: np.ndarray,
    N: np.ndarray,
    mask: np.ndarray,
    log_alpha_bounds: tuple[float, float] = (-2.0, 2.0),
    logit_c_bounds: tuple[float, float] = (-5.0, -1.0),
    theta_init: np.ndarray | None = None,
    beta_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """3PL: p(m, x) = c_x + (1 - c_x) * sigma(alpha_x * (theta_m - beta_x)).

    Parameters: theta (M), beta (J), log_alpha (J), logit_c (J).
    c_x = sigmoid(logit_c_x) bounded into [sigma(-5), sigma(-1)] = [0.007, 0.269]
    so c stays in the "guessing" range.
    """
    M, J = K.shape
    W = mask.astype(float)
    if theta_init is None or beta_init is None:
        th0, be0, _ = fit_masked_1pl(K, N, mask)
        if theta_init is None:
            theta_init = th0
        if beta_init is None:
            beta_init = be0
    log_alpha_init = np.zeros(J)
    logit_c_init = np.full(J, -3.0)  # c ≈ 0.047

    def nll(p):
        th = p[:M]
        be = p[M:M + J]
        la = p[M + J:M + 2 * J]
        lc = p[M + 2 * J:]
        a = np.exp(la)
        c = _sigmoid(lc)
        eta = a[None, :] * (th[:, None] - be[None, :])
        q = _sigmoid(eta)                                            # 2PL part
        pp = c[None, :] + (1 - c[None, :]) * q
        pp = np.clip(pp, 1e-10, 1 - 1e-10)
        ll = K * np.log(pp) + (N - K) * np.log(1 - pp)
        return -float((W * ll).sum())

    def grad(p):
        th = p[:M]
        be = p[M:M + J]
        la = p[M + J:M + 2 * J]
        lc = p[M + 2 * J:]
        a = np.exp(la)
        c = _sigmoid(lc)
        diff = th[:, None] - be[None, :]
        eta = a[None, :] * diff
        q = _sigmoid(eta)
        one_mc = (1 - c)[None, :]
        pp = c[None, :] + one_mc * q
        pp = np.clip(pp, 1e-10, 1 - 1e-10)
        # ∂L/∂P scaled by W
        dL_dP = (K / pp - (N - K) / (1 - pp)) * W
        # ∂P/∂η = (1-c) * q * (1-q)
        dP_deta = one_mc * q * (1 - q)
        # ∂P/∂c = 1 - q
        dP_dc = 1 - q
        r_eta = dL_dP * dP_deta                  # (M, J)
        r_c = dL_dP * dP_dc                      # (M, J)
        g_th = -(r_eta * a[None, :]).sum(axis=1)
        g_be = (r_eta * a[None, :]).sum(axis=0)
        g_la = -(a * (r_eta * diff).sum(axis=0))
        # ∂c/∂logit_c = c(1-c)
        g_lc = -((c * (1 - c)) * r_c.sum(axis=0))
        return np.concatenate([g_th, g_be, g_la, g_lc])

    x0 = np.concatenate([theta_init, beta_init, log_alpha_init, logit_c_init])
    bounds = (
        [(None, None)] * M
        + [(None, None)] * J
        + [log_alpha_bounds] * J
        + [logit_c_bounds] * J
    )
    out = minimize(nll, x0, jac=grad, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 5000, "ftol": 1e-11, "gtol": 1e-8})
    th = out.x[:M]
    be = out.x[M:M + J]
    a = np.exp(out.x[M + J:M + 2 * J])
    c = _sigmoid(out.x[M + 2 * J:])
    sh = th.mean()
    return th - sh, be - sh, a, c, float(out.fun)


def fit_masked_mirt(
    K: np.ndarray, N: np.ndarray, mask: np.ndarray, D: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compensatory MIRT-D fit with diagonal mask.

    Implements masking by zeroing K and N at masked cells before delegating
    to difficulty.mirt.fit_mle. Cells with K=N=0 contribute exactly zero to
    both NLL and gradient, so this is equivalent to a true mask.

    Returns (theta (M, D), beta (J,), a (J, D), nll on visible cells).
    """
    K_masked = np.where(mask > 0, K, 0.0)
    N_masked = np.where(mask > 0, N, 0.0)
    theta, beta, a, nll_v = _mirt.fit_mle(K_masked, N_masked, D=D, seed=0)
    return theta, beta, a, nll_v


def held_out_metrics(
    K: np.ndarray, N: np.ndarray, held_out_mask: np.ndarray, p_pred: np.ndarray
) -> dict:
    """held_out_mask: 1 on cells to score (held-out and observed), 0 elsewhere.

    Computes mean p_tilde, RMSE_cell, NLL across scored cells.
    """
    held = held_out_mask.astype(bool)
    p_tilde = p_pred[held]
    K_h = K[held]
    N_h = N[held]
    p_hat = K_h / np.maximum(N_h, 1)
    rmse_cell = float(np.sqrt(np.mean((p_tilde - p_hat) ** 2)))
    p_clip = np.clip(p_tilde, 1e-10, 1 - 1e-10)
    nll = -float((K_h * np.log(p_clip) + (N_h - K_h) * np.log(1 - p_clip)).sum())
    bias = float(p_tilde.mean() - p_hat.mean())
    return {
        "n_cells": int(held.sum()),
        "mean_p_tilde": float(p_tilde.mean()),
        "mean_p_hat_observed": float(p_hat.mean()),
        "mean_bias_pred_minus_obs": bias,
        "min_p_tilde": float(p_tilde.min()),
        "max_p_tilde": float(p_tilde.max()),
        "rmse_cell": rmse_cell,
        "nll": nll,
    }


def filter_anchor(K: np.ndarray, N: np.ndarray, rule: str = "avg") -> np.ndarray:
    """Anchor filter mask.

    rule = 'avg' : panel-mean p̄_x ∈ [FILTER_LOW, FILTER_HIGH]  (legacy)
    rule = 'alo' : ALO[0.20, 0.80] — ∃ m s.t. p̂_{m,x} ∈ [0.20, 0.80]
    """
    p_hat = np.where(N > 0, K / N, 0.5)
    if rule == "avg":
        panel_mean = p_hat.mean(axis=0)
        return (panel_mean >= FILTER_LOW) & (panel_mean <= FILTER_HIGH)
    if rule == "alo":
        return ((p_hat >= 0.20) & (p_hat <= 0.80)).any(axis=0)
    raise ValueError(f"unknown filter rule: {rule!r}")


def run_category(
    category: str,
    *,
    data_source: str = "sgbs",
    mask_seed: int = 0,
    mask_rate: float = 0.1,
    filter_rule: str = "avg",
    verbose: bool = True,
) -> dict:
    cfg = CATEGORY_CONFIG[category]
    panel = list(PANEL)
    M = len(panel)

    # Anchor block (filtered)
    anchor_ids, anchor_K, anchor_N = load_anchor_matrix(cfg["anchor_dir"], panel)
    keep = filter_anchor(anchor_K, anchor_N, rule=filter_rule)
    J_anchor = int(keep.sum())
    anchor_K, anchor_N = anchor_K[:, keep], anchor_N[:, keep]
    anchor_ids = [pid for pid, k in zip(anchor_ids, keep) if k]

    if data_source == "anchor":
        # Anchor-only mode: random uniform cell mask, fit on visible (1-mask_rate),
        # score held-out on the masked cells.
        K = anchor_K
        N = anchor_N
        J_total = J_anchor
        ec_ids: list[str] = []
        rng = np.random.default_rng(mask_seed)
        rand_held = rng.random((M, J_total)) < mask_rate
        diag_mask = (~rand_held).astype(int)  # 1 = visible, 0 = held-out
    elif data_source == "sgbs":
        # EC block
        ec_ids, ec_K, ec_N, source_map = load_ec_blocks(
            cfg["ec_dataset"], cfg["ec_val_dir"], panel,
        )
        # Combined matrix; diagonal mask: 0 on (m, x) where source(x) == m; 1 elsewhere
        K = np.concatenate([anchor_K, ec_K], axis=1)
        N = np.concatenate([anchor_N, ec_N], axis=1)
        J_total = K.shape[1]
        diag_mask = np.ones((M, J_total), dtype=int)
        for j_off, pid in enumerate(ec_ids):
            m_idx = panel.index(source_map[pid])
            diag_mask[m_idx, J_anchor + j_off] = 0
    else:
        raise ValueError(f"unknown data_source: {data_source}")
    # Cells with no observation (N=0) are also held out from the fit; the
    # held-out evaluation only scores cells where diag_mask=0 AND N>0 (i.e.
    # the diagonal block, restricted to cells we actually observed).
    obs_mask = (N > 0).astype(int)
    fit_mask = diag_mask * obs_mask          # 1 = visible & observed
    held_out_mask = (1 - diag_mask) * obs_mask  # 1 = held-out & observed
    n_held = int(held_out_mask.sum())
    n_visible = int(fit_mask.sum())
    n_missing = int(M * J_total - obs_mask.sum())
    if n_missing > 0:
        print(f"  [info] {n_missing} cells with N=0 (excluded from fit and eval)")

    n_eff = float((N * fit_mask).sum())  # total Binomial trials on visible cells

    def _bic(nll: float, dof: int) -> float:
        return 2 * nll + dof * float(np.log(max(n_eff, 1.0)))

    # ---- 1PL ----
    th_1, be_1, nll_1 = fit_masked_1pl(K, N, fit_mask)
    p_1 = _sigmoid(th_1[:, None] - be_1[None, :])
    dof_1 = (M - 1) + J_total
    h_1 = held_out_metrics(K, N, held_out_mask, p_1)

    # ---- 2PL ----
    th_2, be_2, a_2, nll_2 = fit_masked_2pl(K, N, fit_mask, theta_init=th_1, beta_init=be_1)
    p_2 = _sigmoid(a_2[None, :] * (th_2[:, None] - be_2[None, :]))
    dof_2 = (M - 1) + J_total + J_total
    h_2 = held_out_metrics(K, N, held_out_mask, p_2)

    # ---- 3PL ----
    th_3, be_3, a_3, c_3, nll_3 = fit_masked_3pl(K, N, fit_mask, theta_init=th_1, beta_init=be_1)
    eta_3 = a_3[None, :] * (th_3[:, None] - be_3[None, :])
    p_3 = c_3[None, :] + (1 - c_3[None, :]) * _sigmoid(eta_3)
    dof_3 = (M - 1) + J_total + J_total + J_total
    h_3 = held_out_metrics(K, N, held_out_mask, p_3)

    # ---- MIRT-2D ----
    th_m2, be_m2, a_m2, nll_m2 = fit_masked_mirt(K, N, fit_mask, D=2)
    p_m2 = _sigmoid(th_m2 @ a_m2.T - be_m2[None, :])
    dof_m2 = M * 2 - 2 + J_total + J_total * 2  # theta (M·D - D) + beta + a
    h_m2 = held_out_metrics(K, N, held_out_mask, p_m2)

    # ---- MIRT-3D ----
    th_m3, be_m3, a_m3, nll_m3 = fit_masked_mirt(K, N, fit_mask, D=3)
    p_m3 = _sigmoid(th_m3 @ a_m3.T - be_m3[None, :])
    dof_m3 = M * 3 - 3 + J_total + J_total * 3
    h_m3 = held_out_metrics(K, N, held_out_mask, p_m3)

    def _alpha_summary(arr: np.ndarray) -> dict:
        return {
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }

    def _maybe_summary(arr: np.ndarray) -> dict | None:
        return _alpha_summary(arr) if arr.size > 0 else None

    summary = {
        "category": category,
        "data_source": data_source,
        "M": M,
        "J_anchor": J_anchor,
        "J_ec": len(ec_ids),
        "J_total": J_total,
        "n_visible_cells": n_visible,
        "n_held_out_cells": n_held,
        "n_visible_trials": n_eff,
        "models": {
            "1pl": {
                "in_sample_nll": nll_1, "dof": dof_1, "bic": _bic(nll_1, dof_1),
                "held_out": h_1,
            },
            "2pl": {
                "in_sample_nll": nll_2, "dof": dof_2, "bic": _bic(nll_2, dof_2),
                "held_out": h_2,
                "alpha_anchor": _maybe_summary(a_2[:J_anchor]),
                "alpha_ec": _maybe_summary(a_2[J_anchor:]),
            },
            "3pl": {
                "in_sample_nll": nll_3, "dof": dof_3, "bic": _bic(nll_3, dof_3),
                "held_out": h_3,
                "alpha_anchor": _maybe_summary(a_3[:J_anchor]),
                "alpha_ec": _maybe_summary(a_3[J_anchor:]),
                "c_anchor": _maybe_summary(c_3[:J_anchor]),
                "c_ec": _maybe_summary(c_3[J_anchor:]),
            },
            "mirt_2d": {
                "in_sample_nll": nll_m2, "dof": dof_m2, "bic": _bic(nll_m2, dof_m2),
                "held_out": h_m2,
                "log_a_norm_anchor": _maybe_summary(
                    np.log(np.linalg.norm(a_m2[:J_anchor], axis=1))
                ) if J_anchor > 0 else None,
                "log_a_norm_ec": _maybe_summary(
                    np.log(np.linalg.norm(a_m2[J_anchor:], axis=1))
                ) if a_m2.shape[0] > J_anchor else None,
            },
            "mirt_3d": {
                "in_sample_nll": nll_m3, "dof": dof_m3, "bic": _bic(nll_m3, dof_m3),
                "held_out": h_m3,
                "log_a_norm_anchor": _maybe_summary(
                    np.log(np.linalg.norm(a_m3[:J_anchor], axis=1))
                ) if J_anchor > 0 else None,
                "log_a_norm_ec": _maybe_summary(
                    np.log(np.linalg.norm(a_m3[J_anchor:], axis=1))
                ) if a_m3.shape[0] > J_anchor else None,
            },
        },
    }

    if verbose:
        print(f"\n=== Category: {category} ===")
        print(f"  J_anchor={J_anchor}  J_ec={len(ec_ids)}  visible={n_visible}  held_out={n_held}")
        models = summary["models"]
        print(f"  {'model':<10} {'dof':>6} {'in-NLL':>10} {'BIC':>11} "
              f"{'held NLL':>10} {'mean p̃':>9} {'p̃ range':>17} {'RMSE':>7}")
        for name in ["1pl", "2pl", "3pl", "mirt_2d", "mirt_3d"]:
            r = models[name]
            ho = r["held_out"]
            rng = f"[{ho['min_p_tilde']:.2f}, {ho['max_p_tilde']:.2f}]"
            print(f"  {name:<10} {r['dof']:>6d} {r['in_sample_nll']:>10.0f} {r['bic']:>11.0f} "
                  f"{ho['nll']:>10.0f} {ho['mean_p_tilde']:>9.3f} {rng:>17} {ho['rmse_cell']:>7.3f}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None,
                        choices=list(CATEGORY_CONFIG.keys()),
                        help="run a single category; default = all 4")
    parser.add_argument("--data-source", default="sgbs", choices=["sgbs", "anchor"],
                        help="sgbs = anchor + EC w/ diagonal mask; "
                             "anchor = anchor-only w/ random cell mask")
    parser.add_argument("--mask-seed", type=int, default=0,
                        help="RNG seed for random cell mask (anchor mode)")
    parser.add_argument("--mask-rate", type=float, default=0.1,
                        help="fraction of cells held out (anchor mode)")
    parser.add_argument("--filter-rule", default="avg", choices=["avg", "alo"],
                        help="anchor filter for --data-source anchor: "
                             "avg = panel-mean ∈ [0.05, 0.95]; "
                             "alo = ∃m s.t. p̂_{m,x} ∈ [0.20, 0.80]")
    parser.add_argument("--out", default=None,
                        help="output JSON path (relative to project root); "
                             "defaults to irt_model_compare[_anchor[_alo]].json")
    args = parser.parse_args()

    if args.out is None:
        if args.data_source == "sgbs":
            args.out = "results/analysis/irt_model_compare.json"
        else:
            suffix = "_alo" if args.filter_rule == "alo" else ""
            args.out = f"results/analysis/irt_model_compare_anchor{suffix}.json"

    cats = [args.category] if args.category else list(CATEGORY_CONFIG.keys())

    results = {}
    for cat in cats:
        try:
            results[cat] = run_category(
                cat,
                data_source=args.data_source,
                mask_seed=args.mask_seed,
                mask_rate=args.mask_rate,
                filter_rule=args.filter_rule,
            )
        except FileNotFoundError as e:
            print(f"  [skip {cat}] {e}")
            results[cat] = {"error": str(e)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote: {out_path}")

    # Final compact: held-out NLL gap to 1PL, per model, per category.
    model_order = ["1pl", "2pl", "3pl", "mirt_2d", "mirt_3d"]
    print("\n=== Held-out NLL: model − 1PL (positive = 1PL better) ===")
    print(f"  {'category':<26}" + " ".join(f"{m:>12}" for m in model_order))
    for cat, r in results.items():
        if "error" in r:
            print(f"  {cat:<26} (skipped)")
            continue
        nll_1pl = r["models"]["1pl"]["held_out"]["nll"]
        cells = []
        for m in model_order:
            d = r["models"][m]["held_out"]["nll"] - nll_1pl
            cells.append(f"{d:>+12.1f}")
        print(f"  {cat:<26}" + " ".join(cells))

    print("\n=== Held-out RMSE_cell: model − 1PL (positive = 1PL better) ===")
    print(f"  {'category':<26}" + " ".join(f"{m:>12}" for m in model_order))
    for cat, r in results.items():
        if "error" in r:
            print(f"  {cat:<26} (skipped)")
            continue
        rmse_1pl = r["models"]["1pl"]["held_out"]["rmse_cell"]
        cells = []
        for m in model_order:
            d = r["models"][m]["held_out"]["rmse_cell"] - rmse_1pl
            cells.append(f"{d:>+12.4f}")
        print(f"  {cat:<26}" + " ".join(cells))

    print("\n=== BIC on visible: model − 1PL (negative = richer model wins by BIC) ===")
    print(f"  {'category':<26}" + " ".join(f"{m:>12}" for m in model_order))
    for cat, r in results.items():
        if "error" in r:
            print(f"  {cat:<26} (skipped)")
            continue
        bic_1pl = r["models"]["1pl"]["bic"]
        cells = []
        for m in model_order:
            d = r["models"][m]["bic"] - bic_1pl
            cells.append(f"{d:>+12.0f}")
        print(f"  {cat:<26}" + " ".join(cells))


if __name__ == "__main__":
    main()
