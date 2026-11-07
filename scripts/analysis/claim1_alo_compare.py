"""Claim 1 panel-LOO stability under ALO[0.20, 0.80] vs AVG[0.05, 0.95].

Re-runs the same panel-LOO pipeline as claim1_calibration_stability.py but
swaps the AVG mask for the AT-LEAST-ONE mask (∃ panel m with p̂ ∈ [0.2, 0.8]).
Reports both filters' results side by side for direct comparison with paper
Table tab:claim1.

Reports empirical sensitivity without external acceptance thresholds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from difficulty.rasch import load_response_matrix, fit_ols, fit_mle  # noqa: E402

from scripts.analysis.claim1_calibration_stability import (  # noqa: E402
    fit_theta_fixed_beta,
)

CATEGORIES = {
    "A (harm)":  ("harmful_refusal",        "filter_ablation_raw_v2_judgefix"),
    "A' (over)": ("over_refusal",           "filter_ablation_raw_v2_judgefix"),
    "B (cons)":  ("constrained_completion", "filter_ablation_raw_v2"),
    "C (syco)":  ("sycophancy_resistance",  "filter_ablation_raw_v2"),
}


def avg_mask(p_hat: np.ndarray, lo=0.05, hi=0.95) -> np.ndarray:
    pm = p_hat.mean(axis=0)
    return (pm >= lo) & (pm <= hi)


def alo_mask(p_hat: np.ndarray, lo=0.20, hi=0.80) -> np.ndarray:
    return ((p_hat >= lo) & (p_hat <= hi)).any(axis=0)


def panel_loo(pc: np.ndarray, Nt: np.ndarray, p: np.ndarray) -> dict:
    """Returns Claim 1 rank, displacement, and held-out ability metrics.

    Mirrors claim1_calibration_stability.py:_panel_loo_inner but inlined
    here so we can call it on either filter's mask.
    """
    M, J = pc.shape
    th_init, b_init = fit_ols(p)
    theta_full, beta_full, _ = fit_mle(pc, Nt, th_init, b_init)

    rho_loo = []
    pooled_abs_dbeta = []
    abs_dtheta_pooled = []

    for m_drop in range(M):
        keep = [i for i in range(M) if i != m_drop]
        pc_l, Nt_l, p_l = pc[keep], Nt[keep], p[keep]
        th_init_l, b_init_l = fit_ols(p_l)
        theta_loo, beta_loo, _ = fit_mle(pc_l, Nt_l, th_init_l, b_init_l)
        # Align: 1PL identifiable up to translation, match mean θ of survivors
        # 1PL invariant under (θ, β) → (θ+c, β+c); shift both by the same amount.
        shift = float(theta_full[keep].mean() - theta_loo.mean())
        beta_loo_a = beta_loo + shift
        theta_loo_a = theta_loo + shift

        rho, _ = spearmanr(beta_full, beta_loo_a)
        rho_loo.append(rho)
        pooled_abs_dbeta.extend(np.abs(beta_full - beta_loo_a).tolist())

        # Out-of-panel θ recovery for the dropped model: fit its θ on fixed β_loo,
        # initializing at the full-fit θ (same as claim1_calibration_stability.py).
        theta_m_refit = fit_theta_fixed_beta(pc[m_drop], Nt[m_drop], beta_loo_a,
                                              theta_init=float(theta_full[m_drop]))
        abs_dtheta_pooled.append(abs(theta_m_refit - theta_full[m_drop]))

    arr_dbeta = np.array(pooled_abs_dbeta)
    return {
        "J": int(J),
        "rho_min": float(min(rho_loo)),
        "median_dbeta": float(np.median(arr_dbeta)),
        "p95_dbeta":    float(np.percentile(arr_dbeta, 95)),
        "max_dtheta":   float(max(abs_dtheta_pooled)),
    }


def run(label: str, dir_name: tuple) -> dict:
    cat, sub = dir_name
    rdir = REPO / "results" / cat / sub
    rm = load_response_matrix(str(rdir))
    p_hat = rm["p_hat"]
    pc = rm["pass_counts"]
    Nt = rm["total_trials"]

    avg_m = avg_mask(p_hat)
    alo_m = alo_mask(p_hat)
    res_avg = panel_loo(pc[:, avg_m], Nt[:, avg_m], p_hat[:, avg_m])
    res_alo = panel_loo(pc[:, alo_m], Nt[:, alo_m], p_hat[:, alo_m])

    overlap = int((avg_m & alo_m).sum())
    avg_only = int((avg_m & ~alo_m).sum())
    alo_only = int((~avg_m & alo_m).sum())
    return {
        "label": label,
        "AVG": res_avg, "ALO": res_alo,
        "overlap": overlap, "avg_only": avg_only, "alo_only": alo_only,
    }


def fmt(reports: list[dict]) -> str:
    lines = []
    lines.append("\nPanel-LOO stability: AVG[0.05,0.95] vs ALO[0.20,0.80]")
    lines.append("=" * 96)
    hdr = (f"  {'Category':<10}  {'filter':<6}  {'J':>4}  "
           f"{'ρ_min':>6}  {'med|Δβ̂|':>9}  {'p95|Δβ̂|':>9}  "
           f"{'max|Δθ̂|':>9}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    refs = {
        "AVG": "AVG", "ALO": "ALO",
    }
    for rep in reports:
        for k in ["AVG", "ALO"]:
            r = rep[k]
            lines.append(
                f"  {rep['label']:<10}  {refs[k]:<6}  {r['J']:>4}  "
                f"{r['rho_min']:>6.3f}  {r['median_dbeta']:>9.3f}  "
                f"{r['p95_dbeta']:>9.3f}  "
                f"{r['max_dtheta']:>9.3f}"
            )
        lines.append(f"    overlap={rep['overlap']}, AVG-only={rep['avg_only']}, ALO-only={rep['alo_only']}")
    lines.append("")
    lines.append("  Ability shifts compare fixed-LOO calibration with the full-panel fit; neither is ground truth.")
    return "\n".join(lines)


def main():
    reports = []
    for label, dirinfo in CATEGORIES.items():
        rdir = REPO / "results" / dirinfo[0] / dirinfo[1]
        if not rdir.exists():
            print(f"[SKIP] {label}: {rdir} not found")
            continue
        reports.append(run(label, dirinfo))
    print(fmt(reports))


if __name__ == "__main__":
    main()
