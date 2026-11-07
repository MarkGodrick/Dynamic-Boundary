"""Claim 2: SGBS items are boundary-anchored.

Builds the Claim 2 response matrix (anchor + 9*K SGBS EC) for the 9-model panel,
fits a masked-MLE 1PL Rasch model (diagonal block {(m, x) : x in EC(m)} hidden),
and reports held-out statistics.

Per-panel-model statistics:
  - mean held-out prediction p_tilde = sigma(theta_m - beta_x), x in EC(m)
    Target ~ 0.5 (boundary-anchored on calibrated logit scale).
  - cell-aggregate RMSE on held-out (m, x) cells: sqrt(mean over x of (p_tilde - p_hat)^2),
    where p_hat = k_{m,x}/N is the empirical pass rate.
    Binomial floor sqrt(0.25 / N) ~ 0.079 at N = 40, p = 0.5.
    (Note: paper §sec:experiments-claim2 writes the RMSE with a per-trial sum
     `Σ_{x, t}(p_tilde - y_{m,x,t})^2` but cites the cell-aggregate Binomial SE
     sqrt(0.25/N) as the floor — the per-trial Brier floor at p=0.5 is 0.5,
     so we follow the cited floor and report the cell-aggregate form.)

Panel-level supplement:
  - Spearman rho(theta_m, beta_bar(EC(m))) across the 9 panel models.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr


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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Per-category source paths. Anchor dir mirrors the convention of
# scripts/analysis/{mixed_vs_anchor_metrics, skill_diversity, jml_vs_mml}.py.
CATEGORY_PATHS: dict[str, dict[str, str]] = {
    "harmful_refusal":        {"anchor_subdir": "filter_ablation_raw_v2_judgefix"},
    "over_refusal":           {"anchor_subdir": "filter_ablation_raw_v2_judgefix"},
    "constrained_completion": {"anchor_subdir": "filter_ablation_raw_v2"},
    "sycophancy_resistance":  {"anchor_subdir": "filter_ablation_raw_v2"},
}

FILTER_LOW = 0.05
FILTER_HIGH = 0.95


def category_paths(category: str) -> dict[str, Path]:
    if category not in CATEGORY_PATHS:
        raise ValueError(f"unknown category {category}; expected one of {list(CATEGORY_PATHS)}")
    sub = CATEGORY_PATHS[category]["anchor_subdir"]
    return {
        "anchor_dir": PROJECT_ROOT / "results" / category / sub,
        "ec_dataset": PROJECT_ROOT / "benchmarks" / category / "edge_case_dataset_slot50.jsonl",
        "ec_val_dir": PROJECT_ROOT / "results" / category / "validation" / "edge_case_slot50",
        "out_dir":    PROJECT_ROOT / "results" / category / "claim2_boundary_anchored",
    }


def model_to_dirname(model: str) -> str:
    return model.replace("/", "_").replace("-", "_").replace(".", ".")


def load_anchor_matrix(
    anchor_dir: Path,
    panel: list[str] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (item_ids, pass_counts, total_trials) for the K=500 random pool.

    Tolerates occasional corrupt JSONL lines (truncated rows from interrupted
    writes): skips them per-model and takes the intersection of prompt_ids
    across the panel. A warning is printed when any line is dropped.
    """
    if panel is None:
        panel = PANEL
    per_model: dict[str, dict[str, dict]] = {}

    for model in panel:
        sub = anchor_dir / model_to_dirname(model)
        rm = sub / "response_matrix.jsonl"
        if not rm.exists():
            raise FileNotFoundError(f"missing anchor response matrix: {rm}")

        bad = 0
        rows: dict[str, dict] = {}
        for ln in rm.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                bad += 1
                continue
            rows[r["prompt_id"]] = r
        if bad:
            print(f"  WARN: {model}: dropped {bad} corrupt anchor rows")
        per_model[model] = rows

    common = sorted(set.intersection(*(set(r) for r in per_model.values())))
    n_dropped = max(len(r) for r in per_model.values()) - len(common)
    if n_dropped:
        print(f"  Anchor: keeping {len(common)} prompt_ids common across panel "
              f"({n_dropped} dropped from at least one model)")

    item_ids = list(common)
    pass_counts_per_model = []
    total_trials_per_model = []
    for model in panel:
        rows = per_model[model]
        pass_counts_per_model.append(np.array([rows[pid]["pass_count"] for pid in item_ids], dtype=float))
        total_trials_per_model.append(np.array([rows[pid]["total_trials"] for pid in item_ids], dtype=float))

    return item_ids, np.stack(pass_counts_per_model), np.stack(total_trials_per_model)


def filter_anchor(p_hat: np.ndarray) -> np.ndarray:
    """AVG[0.05, 0.95] panel-mean filter; returns boolean keep mask of length J."""
    panel_mean = p_hat.mean(axis=0)
    return (panel_mean >= FILTER_LOW) & (panel_mean <= FILTER_HIGH)


def load_ec_blocks(
    ec_dataset: Path,
    ec_val_dir: Path,
    panel: list[str] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, str]]:
    """Build the EC block: J*50 = J*50 items, full panel responses.

    Off-diagonal cells from validation/edge_case_slot50/<model>/response_matrix.jsonl;
    diagonal cells (m on EC(m)) from the SGBS dataset's pass_count/total_trials.
    Returns (item_ids, pass_counts (M, J_ec), total_trials (M, J_ec), source_map: pid -> source_model).
    Items whose source_model is not in `panel` are dropped.
    """
    if panel is None:
        panel = PANEL
    panel_set = set(panel)
    # Source map from dataset; drop items sourced from excluded models
    dataset_rows = [json.loads(line) for line in ec_dataset.read_text().splitlines() if line.strip()]
    dataset_rows = [r for r in dataset_rows if r["source_model"] in panel_set]
    dataset_rows.sort(key=lambda r: r["prompt_id"])
    candidate_ids = [r["prompt_id"] for r in dataset_rows]
    source_map = {r["prompt_id"]: r["source_model"] for r in dataset_rows}
    diag_pass = {r["prompt_id"]: r["pass_count"] for r in dataset_rows}
    diag_total = {r["prompt_id"]: r["total_trials"] for r in dataset_rows}

    # Two-pass: first read each panel-model's per-pid rows (tolerating corrupt
    # JSONL lines), then drop any pid missing on a non-source model.
    per_model: dict[str, dict[str, tuple[int, int, int]]] = {}
    for model in panel:
        sub = ec_val_dir / model_to_dirname(model)
        rm = sub / "response_matrix.jsonl"
        if not rm.exists():
            raise FileNotFoundError(f"missing EC validation matrix: {rm}")
        per_pid: dict[str, tuple[int, int, int]] = {}
        bad = 0
        for line in rm.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            per_pid[r["prompt_id"]] = (r["pass_count"], r["total_trials"], len(r.get("trials", [])))
        if bad:
            print(f"  WARN: EC val for {model}: dropped {bad} corrupt rows")
        per_model[model] = per_pid

    item_ids: list[str] = []
    dropped: list[str] = []
    for pid in candidate_ids:
        ok = True
        for model in panel:
            if model == source_map[pid]:
                continue  # diagonal cell sourced from dataset, off-diagonal not required
            if pid not in per_model[model]:
                ok = False
                break
        (item_ids if ok else dropped).append(pid)
    if dropped:
        print(f"  EC: dropped {len(dropped)} items missing on at least one off-diagonal panel model")

    M = len(panel)
    J = len(item_ids)
    pass_counts = np.zeros((M, J), dtype=float)
    total_trials = np.zeros((M, J), dtype=float)

    for m_idx, model in enumerate(panel):
        per_pid = per_model[model]
        for j, pid in enumerate(item_ids):
            entry = per_pid.get(pid)
            n_trials = entry[2] if entry is not None else 0
            if n_trials == 0:
                # off-diagonal slot for this model is empty -> diagonal cell from dataset
                if source_map[pid] != model:
                    raise ValueError(
                        f"empty trials at non-diagonal cell ({model}, {pid}); source={source_map[pid]}"
                    )
                pass_counts[m_idx, j] = diag_pass[pid]
                total_trials[m_idx, j] = diag_total[pid]
            else:
                pc, tt, _ = entry
                pass_counts[m_idx, j] = pc
                total_trials[m_idx, j] = tt

    return item_ids, pass_counts, total_trials, source_map


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def fit_masked_mle(
    pass_counts: np.ndarray,
    total_trials: np.ndarray,
    mask: np.ndarray,
    theta_init: np.ndarray | None = None,
    beta_init: np.ndarray | None = None,
    frozen_beta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """1PL Rasch MLE on cells where mask[m, j] > 0.

    If ``frozen_beta`` is provided (length J, NaN for free entries, otherwise the
    fixed value), those beta entries are held fixed and the scale is pinned by
    them (no post-fit mean-zero shift). When all entries are free, theta is
    centered to mean 0. Returns (theta, beta, neg_loglik).
    """
    M, J = pass_counts.shape
    if frozen_beta is None:
        frozen_beta = np.full(J, np.nan)
    free_mask = np.isnan(frozen_beta)
    n_free = int(free_mask.sum())
    any_frozen = bool((~free_mask).any())

    if theta_init is None or beta_init is None:
        # OLS init on visible cells: row/col means of logit(p_hat) with smoothing
        with np.errstate(divide="ignore", invalid="ignore"):
            p_hat = np.where(total_trials > 0, pass_counts / total_trials, 0.5)
        p_smooth = np.clip(p_hat, 0.025, 0.975)
        y = np.log(p_smooth / (1 - p_smooth))
        masked_y = np.where(mask > 0, y, np.nan)
        with np.errstate(invalid="ignore"):
            grand = np.nanmean(masked_y)
            row_means = np.nanmean(masked_y, axis=1)
            col_means = np.nanmean(masked_y, axis=0)
        if theta_init is None:
            theta_init = np.nan_to_num(row_means - grand, nan=0.0)
        if beta_init is None:
            beta_init = np.nan_to_num(-(col_means - grand), nan=0.0)
    # Make sure frozen entries of beta_init match the frozen values (cosmetic;
    # they are not optimized but used in unpack only for the frozen slots).
    beta_init = beta_init.copy()
    beta_init[~free_mask] = frozen_beta[~free_mask]

    K = pass_counts
    N = total_trials
    W = mask.astype(float)
    frozen_vals = np.where(free_mask, 0.0, frozen_beta)

    def unpack(params):
        theta = params[:M]
        beta = frozen_vals.copy()
        beta[free_mask] = params[M:]
        return theta, beta

    def nll(params):
        theta, beta = unpack(params)
        diff = theta[:, None] - beta[None, :]
        p = np.clip(_sigmoid(diff), 1e-10, 1 - 1e-10)
        ll_cells = K * np.log(p) + (N - K) * np.log(1 - p)
        return -float((W * ll_cells).sum())

    def grad(params):
        theta, beta = unpack(params)
        diff = theta[:, None] - beta[None, :]
        p = np.clip(_sigmoid(diff), 1e-10, 1 - 1e-10)
        residual = (K - N * p) * W
        g_theta = -residual.sum(axis=1)
        g_beta_full = residual.sum(axis=0)
        return np.concatenate([g_theta, g_beta_full[free_mask]])

    x0 = np.concatenate([theta_init, beta_init[free_mask]])
    result = minimize(
        nll, x0, jac=grad, method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-11, "gtol": 1e-8},
    )
    theta, beta = unpack(result.x)
    if not any_frozen:
        shift = theta.mean()
        theta = theta - shift
        beta = beta - shift
    return theta, beta, float(result.fun)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        choices=sorted(CATEGORY_PATHS.keys()),
        default="constrained_completion",
        help="evaluation category whose Claim 2 response matrix to fit.",
    )
    parser.add_argument(
        "--no-anchor",
        action="store_true",
        help="ablation: drop the anchor block from the fit (EC-only).",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="exclude these panel models (substring match); their EC items are also dropped.",
    )
    parser.add_argument(
        "--no-mask-diagonal",
        action="store_true",
        help="ablation: include the diagonal (m, EC(m)) cells in the fit instead of holding them out.",
    )
    parser.add_argument(
        "--anchor-locked",
        action="store_true",
        help="Two-step fit: first calibrate beta_anchor on anchor-only, then freeze "
             "beta_anchor and jointly fit theta + beta_EC on combined matrix. "
             "Scale is pinned by anchor (no post-fit recentering).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="output filename suffix; default derived from flags.",
    )
    args = parser.parse_args()
    use_anchor = not args.no_anchor
    if args.anchor_locked and not use_anchor:
        parser.error("--anchor-locked requires anchor block; remove --no-anchor")

    paths = category_paths(args.category)
    anchor_dir = paths["anchor_dir"]
    ec_dataset = paths["ec_dataset"]
    ec_val_dir = paths["ec_val_dir"]
    out_dir = paths["out_dir"]

    if args.exclude:
        excluded = [m for m in PANEL if any(ex in m for ex in args.exclude)]
        panel = [m for m in PANEL if m not in excluded]
        print(f"Excluding panel models: {excluded}")
    else:
        excluded = []
        panel = list(PANEL)
    print(f"Panel size: {len(panel)} models")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Category: {args.category}")
    print(f"Anchor dir: {anchor_dir}")
    print(f"EC dataset: {ec_dataset}")
    print(f"EC val dir: {ec_val_dir}")

    # Anchor block
    if use_anchor:
        anchor_ids, anchor_K, anchor_N = load_anchor_matrix(anchor_dir, panel)
        anchor_p_hat = np.where(anchor_N > 0, anchor_K / anchor_N, 0.5)
        keep = filter_anchor(anchor_p_hat)
        J_anchor = int(keep.sum())
        anchor_K = anchor_K[:, keep]
        anchor_N = anchor_N[:, keep]
        anchor_ids = [pid for pid, k in zip(anchor_ids, keep) if k]
        print(f"Anchor: kept {J_anchor}/{len(keep)} items after AVG[{FILTER_LOW}, {FILTER_HIGH}] filter")
    else:
        anchor_ids = []
        anchor_K = np.zeros((len(panel), 0), dtype=float)
        anchor_N = np.zeros((len(panel), 0), dtype=float)
        J_anchor = 0
        print("Ablation --no-anchor: skipping anchor block; fitting EC only")

    # EC block (only items sourced from non-excluded panel models)
    ec_ids, ec_K, ec_N, source_map = load_ec_blocks(ec_dataset, ec_val_dir, panel)
    K_per_target = sum(1 for v in source_map.values() if v == panel[0])
    print(f"EC: {len(ec_ids)} items, K={K_per_target} per target model x {len(panel)} panel models")

    # Combined matrix
    pass_counts = np.concatenate([anchor_K, ec_K], axis=1)
    total_trials = np.concatenate([anchor_N, ec_N], axis=1)
    item_ids = anchor_ids + ec_ids
    M, J = pass_counts.shape

    # Mask: 1 = visible, 0 = held out (diagonal block on EC items only)
    # Always track which cells are "diagonal" for evaluation; only honor the mask
    # in the fit when --no-mask-diagonal is False.
    diagonal_mask = np.ones((M, J), dtype=int)
    for j_offset, pid in enumerate(ec_ids):
        j = J_anchor + j_offset
        target = source_map[pid]
        m_idx = panel.index(target)
        diagonal_mask[m_idx, j] = 0  # 0 means "this is a diagonal cell"

    if args.no_mask_diagonal:
        mask = np.ones((M, J), dtype=int)  # fit uses ALL cells
        print(f"Ablation --no-mask-diagonal: diagonal cells INCLUDED in fit")
    else:
        mask = diagonal_mask.copy()
    n_held_out_in_fit = int((mask == 0).sum())
    n_diagonal = int((diagonal_mask == 0).sum())
    print(f"Combined matrix: {M} models x {J} items; {n_diagonal} diagonal cells; {n_held_out_in_fit} held out in fit")

    # Sanity: cells used for evaluation must have observed responses
    if (total_trials[diagonal_mask == 0] == 0).any():
        n_zero = int((total_trials[diagonal_mask == 0] == 0).sum())
        raise RuntimeError(f"{n_zero} diagonal cells have no observed trials")

    # Fit masked MLE
    if args.anchor_locked:
        # Step 1: anchor-only fit -> beta_anchor (and theta_init)
        anchor_mask = np.ones((M, J_anchor), dtype=int)
        theta_step1, beta_anchor_step1, nll_step1 = fit_masked_mle(
            anchor_K, anchor_N, anchor_mask,
        )
        print(f"Step 1 (anchor-only) MLE done. NLL = {nll_step1:.2f}")
        # Step 2: freeze beta_anchor, fit theta + beta_EC on combined matrix
        frozen_beta = np.full(J, np.nan)
        frozen_beta[:J_anchor] = beta_anchor_step1
        theta, beta, nll = fit_masked_mle(
            pass_counts, total_trials, mask,
            theta_init=theta_step1,
            frozen_beta=frozen_beta,
        )
        print(f"Step 2 (anchor-locked combined) MLE done. NLL = {nll:.2f}")
    else:
        theta, beta, nll = fit_masked_mle(pass_counts, total_trials, mask)
        print(f"Masked MLE done. NLL = {nll:.2f}")
    print("theta (sorted by panel order):")
    for m_name, t in zip(panel, theta):
        print(f"  {m_name:<55} theta = {t:+.3f}")

    # Held-out predictions
    diff = theta[:, None] - beta[None, :]
    p_pred = _sigmoid(diff)

    # Per panel model: mean p_tilde over EC(m), cell-level RMSE
    rows = []
    beta_bar_per_model = []
    for m_idx, model in enumerate(panel):
        ec_idx = [J_anchor + j for j, pid in enumerate(ec_ids) if source_map[pid] == model]
        p_tilde = p_pred[m_idx, ec_idx]  # held-out cells for model m
        mean_p_tilde = float(p_tilde.mean())

        K_held = pass_counts[m_idx, ec_idx]
        N_held = total_trials[m_idx, ec_idx]

        # Cell-aggregate RMSE: sqrt(mean over x of (p_tilde - p_hat)^2). Comparable
        # to the Binomial floor sqrt(0.25/N) ~ 0.079 at N = 40, p = 0.5.
        p_obs = K_held / N_held
        rmse_cell = float(np.sqrt(((p_obs - p_tilde) ** 2).mean()))

        # Per-trial Brier (auxiliary). Floor at p=0.5 is 0.5, not 0.079.
        sse = K_held * (1 - p_tilde) ** 2 + (N_held - K_held) * p_tilde ** 2
        rmse_per_trial = float(np.sqrt(sse.sum() / N_held.sum()))

        beta_bar = float(beta[ec_idx].mean())
        beta_bar_per_model.append(beta_bar)

        # SGBS-run observed mean p_hat on EC(m) by m itself (target ~ 0.5 by SGBS reward)
        mean_p_hat_diag = float(p_obs.mean())

        # Panel-mean observed pass rate on EC(m), excluding m itself
        ec_global_idx = ec_idx
        panel_K = pass_counts[:, ec_global_idx]
        panel_N = total_trials[:, ec_global_idx]
        # Exclude diagonal (only m's row is the diagonal for these items)
        others_mask = np.ones(M, dtype=bool)
        others_mask[m_idx] = False
        mean_p_hat_panel = float((panel_K[others_mask] / panel_N[others_mask]).mean())

        rows.append({
            "panel_model": model,
            "n_ec": len(ec_idx),
            "theta_hat": float(theta[m_idx]),
            "beta_bar_EC": beta_bar,
            "mean_p_tilde": mean_p_tilde,
            "mean_p_hat_self_EC": mean_p_hat_diag,
            "mean_p_hat_panel_others_EC": mean_p_hat_panel,
            "rmse_cell": rmse_cell,
            "rmse_per_trial": rmse_per_trial,
        })

    # Spearman rho(theta_m, beta_bar(EC(m)))
    rho, p_val = spearmanr(theta, np.array(beta_bar_per_model))
    binomial_floor = float(np.sqrt(0.25 / 40))
    print()
    print(f"Binomial floor (N=40, p=0.5): sqrt(0.25/40) = {binomial_floor:.4f}")
    print(f"Spearman rho(theta_m, beta_bar(EC(m))) = {rho:+.3f} (p = {p_val:.3g})")
    print()
    print(f"{'Panel model':<55} {'theta':>7} {'beta_bar':>9} {'p_tilde':>8} {'p_hat_self':>11} {'p_hat_pnl':>10} {'RMSE_cell':>10}")
    for r in rows:
        print(
            f"{r['panel_model']:<55} {r['theta_hat']:+7.3f} {r['beta_bar_EC']:+9.3f}"
            f" {r['mean_p_tilde']:8.3f} {r['mean_p_hat_self_EC']:11.3f}"
            f" {r['mean_p_hat_panel_others_EC']:10.3f} {r['rmse_cell']:10.4f}"
        )

    out = {
        "category": args.category,
        "filter": f"AVG[{FILTER_LOW}, {FILTER_HIGH}]",
        "n_anchor": J_anchor,
        "n_ec": len(ec_ids),
        "K_per_target": K_per_target,
        "n_panel": len(panel),
        "n_total_cells": int(M * J),
        "n_held_out_cells": n_held_out_in_fit,
        "n_diagonal_cells": n_diagonal,
        "fit_includes_diagonal": bool(args.no_mask_diagonal),
        "fit_mode": "anchor_locked" if args.anchor_locked else "joint",
        "trials_per_cell": int(total_trials.max()),
        "binomial_floor_N40": binomial_floor,
        "panel": panel,
        "excluded_models": excluded,
        "use_anchor": use_anchor,
        "theta": theta.tolist(),
        "beta_anchor": beta[:J_anchor].tolist(),
        "beta_ec": beta[J_anchor:].tolist(),
        "ec_ids": ec_ids,
        "anchor_ids": anchor_ids,
        "ec_source_model": [source_map[pid] for pid in ec_ids],
        "per_panel_model": rows,
        "spearman_theta_vs_beta_bar_EC": {"rho": float(rho), "p_value": float(p_val)},
        "neg_log_likelihood": nll,
    }
    if args.tag:
        suffix = args.tag
    else:
        parts = []
        if not use_anchor:
            parts.append("no_anchor")
        if args.anchor_locked:
            parts.append("anchor_locked")
        if args.no_mask_diagonal:
            parts.append("with_diagonal")
        if excluded:
            parts.append(f"excl_{len(excluded)}")
        suffix = "_".join(parts) if parts else None
    fname = "claim2_metrics.json" if suffix is None else f"claim2_metrics_{suffix}.json"
    out_path = out_dir / fname
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
