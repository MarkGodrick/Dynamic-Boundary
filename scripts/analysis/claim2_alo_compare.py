"""Claim 2 boundary-localization under ALO[0.20, 0.80] vs AVG[0.05, 0.95].

Re-runs the same joint 1PL Rasch fit as claim2_boundary_anchored.py for all
four categories, swapping the anchor-set filter between AVG and ALO. EC items
and their panel responses are identical across the two filters (filter only
affects the anchor block). Reports paper Table 2 numbers side by side:
mean p_tilde over EC(m), per-cell RMSE, and panel-wide Spearman ρ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.claim2_boundary_anchored import (  # noqa: E402
    CATEGORY_PATHS, PANEL, category_paths,
    load_anchor_matrix, load_ec_blocks, fit_masked_mle, _sigmoid,
)


def avg_mask(p_hat: np.ndarray, lo=0.05, hi=0.95) -> np.ndarray:
    pm = p_hat.mean(axis=0)
    return (pm >= lo) & (pm <= hi)


def alo_mask(p_hat: np.ndarray, lo=0.20, hi=0.80) -> np.ndarray:
    return ((p_hat >= lo) & (p_hat <= hi)).any(axis=0)


def run_one(category: str, mask_fn, hold_diagonal: bool,
            anchor_subdir_override: str | None = None) -> dict:
    """Joint 1PL fit on [anchor | EC], anchor block filtered by `mask_fn`.

    `hold_diagonal=False` reproduces paper Table 2 main result;
    `hold_diagonal=True` reproduces the robustness check tab:claim2-masked.
    """
    paths = category_paths(category)
    panel = list(PANEL)

    anchor_dir = paths["anchor_dir"]
    if anchor_subdir_override is not None:
        anchor_dir = anchor_dir.parent / anchor_subdir_override
    anchor_ids, anchor_K, anchor_N = load_anchor_matrix(anchor_dir, panel)
    anchor_p_hat = np.where(anchor_N > 0, anchor_K / anchor_N, 0.5)
    keep = mask_fn(anchor_p_hat)
    J_anchor = int(keep.sum())
    anchor_K = anchor_K[:, keep]
    anchor_N = anchor_N[:, keep]
    anchor_ids = [pid for pid, k in zip(anchor_ids, keep) if k]

    ec_ids, ec_K, ec_N, source_map = load_ec_blocks(
        paths["ec_dataset"], paths["ec_val_dir"], panel
    )

    pass_counts = np.concatenate([anchor_K, ec_K], axis=1)
    total_trials = np.concatenate([anchor_N, ec_N], axis=1)
    M, J = pass_counts.shape

    mask = np.ones((M, J), dtype=int)
    if hold_diagonal:
        for j_off, pid in enumerate(ec_ids):
            m_idx = panel.index(source_map[pid])
            mask[m_idx, J_anchor + j_off] = 0
    theta, beta, _ = fit_masked_mle(pass_counts, total_trials, mask)
    p_pred = _sigmoid(theta[:, None] - beta[None, :])

    rows = []
    beta_bar = []
    for m_idx, model in enumerate(panel):
        ec_idx = [J_anchor + j for j, pid in enumerate(ec_ids)
                   if source_map[pid] == model]
        p_tilde = p_pred[m_idx, ec_idx]
        K_held = pass_counts[m_idx, ec_idx]
        N_held = total_trials[m_idx, ec_idx]
        p_obs = K_held / N_held
        rmse_cell = float(np.sqrt(((p_obs - p_tilde) ** 2).mean()))
        rows.append({
            "panel_model":  model,
            "mean_p_tilde": float(p_tilde.mean()),
            "rmse_cell":    rmse_cell,
        })
        beta_bar.append(float(beta[ec_idx].mean()))

    theta_arr = np.asarray(theta)
    beta_bar_arr = np.array(beta_bar)
    rho, _ = spearmanr(theta_arr, beta_bar_arr)

    # Inner-7 trim: drop per-category top- and bottom-theta panel models.
    order = np.argsort(theta_arr)
    inner_idx = order[1:-1]
    rho_inner, _ = spearmanr(theta_arr[inner_idx], beta_bar_arr[inner_idx])
    return {"category": category, "J_anchor": J_anchor,
            "rho": float(rho), "rho_inner7": float(rho_inner),
            "rows": rows}


def short(model: str) -> str:
    return (model.split("/")[-1]
            .replace("Qwen2.5-72B-Instruct-AWQ", "Qwen2.5-72B-AWQ")
            .replace("Qwen3-30B-A3B-Instruct-2507", "Qwen3-30B-A3B")
            .replace("Mistral-Small-3.1-24B-Instruct-2503", "Mistral-Small-3.1-24B")
            .replace("Mistral-Nemo-Instruct-2407", "Mistral-Nemo-12B")
            .replace("DeepSeek-R1-Distill-Llama-8B", "DS-R1-Distill-Llama-8B"))


def _print_block(label: str, store_avg: dict, store_alo: dict,
                  cats: list, panel_labels: list) -> None:
    print("=" * 110)
    print(label)
    print(f"  J_anchor (AVG / ALO):")
    for c, lab in zip(cats, panel_labels):
        print(f"    {lab:<10}  AVG={store_avg[c]['J_anchor']:>3}  "
              f"ALO={store_alo[c]['J_anchor']:>3}")
    print()
    hdr = (f"  {'panel model':<24}  " +
           "  ".join(f"{lab:<14}" for lab in panel_labels))
    print(hdr)
    sub = ("  " + " " * 24 + "  " +
           "  ".join(f"{'p̃    RMSE':<14}" for _ in panel_labels))
    print(sub)
    print("  " + "-" * (len(hdr) - 2))

    for filt_name, store in [("AVG", store_avg), ("ALO", store_alo)]:
        print(f"  --- {filt_name} ---")
        for m_idx, model in enumerate(PANEL):
            line = f"  {short(model):<24}  "
            cells = []
            for c in cats:
                r = store[c]["rows"][m_idx]
                cells.append(f"{r['mean_p_tilde']:.3f}  {r['rmse_cell']:.3f}")
            line += "  ".join(f"{c:<14}" for c in cells)
            print(line)
        rho_line = f"  {'Spearman ρ (full 9)':<24}  "
        cells = []
        for c in cats:
            cells.append(f"{store[c]['rho']:+.3f}")
        rho_line += "  ".join(f"{c:<14}" for c in cells)
        print(rho_line)
        rho_inner_line = f"  {'Spearman ρ (inner 7)':<24}  "
        cells = []
        for c in cats:
            cells.append(f"{store[c]['rho_inner7']:+.3f}")
        rho_inner_line += "  ".join(f"{c:<14}" for c in cells)
        print(rho_inner_line)
        print()

    print(f"  Summary aggregates ({label.split('(')[-1].rstrip(')')})")
    print(f"  {'metric':<28}  " +
          "  ".join(f"{lab:<14}" for lab in panel_labels))
    for filt_name, store in [("AVG", store_avg), ("ALO", store_alo)]:
        mean_abs_dev = []
        mean_rmse = []
        mean_ptilde_raw = []
        for c in cats:
            ptilde = np.array([r["mean_p_tilde"] for r in store[c]["rows"]])
            rmse = np.array([r["rmse_cell"] for r in store[c]["rows"]])
            mean_abs_dev.append(float(np.mean(np.abs(ptilde - 0.5))))
            mean_rmse.append(float(np.mean(rmse)))
            mean_ptilde_raw.append(float(np.mean(ptilde)))
        print(f"  {filt_name + ' mean p̃':<28}  " +
              "  ".join(f"{v:<14.3f}" for v in mean_ptilde_raw))
        print(f"  {filt_name + ' mean |p̃−0.5|':<28}  " +
              "  ".join(f"{v:<14.3f}" for v in mean_abs_dev))
        print(f"  {filt_name + ' mean RMSE':<28}  " +
              "  ".join(f"{v:<14.3f}" for v in mean_rmse))
    print()


def main():
    cats = ["harmful_refusal", "over_refusal",
            "constrained_completion", "sycophancy_resistance"]
    panel_labels = ["A (harm)", "A' (over)", "B (cons)", "C (syco)"]

    # With-diagonal fit (paper Table 2 main)
    avg_wd = {c: run_one(c, avg_mask, hold_diagonal=False) for c in cats}
    alo_wd = {c: run_one(c, alo_mask, hold_diagonal=False) for c in cats}
    # Masked-diagonal fit (paper tab:claim2-masked robustness check)
    avg_md = {c: run_one(c, avg_mask, hold_diagonal=True) for c in cats}
    alo_md = {c: run_one(c, alo_mask, hold_diagonal=True) for c in cats}

    _print_block(
        "Claim 2 with-diagonal fit (paper Table 2): AVG[0.05,0.95] vs ALO[0.20,0.80]",
        avg_wd, alo_wd, cats, panel_labels,
    )
    _print_block(
        "Claim 2 masked-diagonal fit (paper tab:claim2-masked): AVG[0.05,0.95] vs ALO[0.20,0.80]",
        avg_md, alo_md, cats, panel_labels,
    )


if __name__ == "__main__":
    main()
