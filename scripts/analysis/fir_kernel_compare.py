"""Axis I & II metric comparison on K=500 random pool, 4 categories × 15 filters.

Tests three Axis I variants and two Axis II kernels:

  Axis I (per-item SE max):
    bSE_A     = max_x [1/√(Σ_m N · p̂(1−p̂))] · √(n_ref / n_kept)   — fit-free, penalty √(n_ref/n_kept)
                  (max over items of per-item asymptotic SE of β̂_x;
                   max independent of n_kept, so penalty must grow as n_kept shrinks)

  Axis I (per-model SE max):
    bSE_B_free = max_m [1/√(Σ_{x∈kept} N · p̂(1−p̂))] · √(n_kept / n_ref)   — fit-free
                  (max over panel members of plug-in SE of θ̂_m on kept set;
                   raw max ~ 1/√n_kept, penalty cancels it)
    bSE_B_fit  = max_θ [1/√(Σ_{x∈kept} N · σ(θ−β̂_x)(1−σ(θ−β̂_x)))] · √(n_kept / n_ref)
                  — current compare_filters.py implementation, requires 1PL fit;
                  reported for reference.

  Axis II (FIR):
    FIR_real  = Σ_{x∈kept} mean_m p̂(1−p̂)  /  Σ_x mean_m p̂(1−p̂)   — code current
    FIR_proxy = Σ_{x∈kept} p̄(1−p̄)         /  Σ_x p̄(1−p̄)          — paper-text claim

Outputs per-filter bSE/FIR table + (bSE, FIR) Pareto under each (bSE, FIR) pairing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from difficulty.rasch import load_response_matrix, fit_ols, fit_mle  # noqa: E402
from scripts.analysis.compare_filters import (  # noqa: E402
    filter_avg, filter_beta, filter_at_least_one,
    filter_low_variance, filter_point_biserial, filter_atlas_combined,
    compute_se_grid,
)


CATEGORIES = {
    "HRR": "results/harmful_refusal/filter_ablation_raw_v2_judgefix",
    "OR":  "results/over_refusal/filter_ablation_raw_v2_judgefix",
    "CIF": "results/constrained_completion/filter_ablation_raw_v2",
    "SR":  "results/sycophancy_resistance/filter_ablation_raw_v2",
}

N_REF = 200
TARGET_BUFFER = 0.5
N_GRID = 50

# bSE_A explosion gate: kept items must collectively avoid Σ_m p̂(1−p̂) = 0
# cells. Normal-range bSE_A is 0.2–0.9 logits; explosion gives 10⁵+. Threshold
# of 5.0 logits is generous (≈ 6× normal max) yet decisively eliminates
# explosion. Override via --bse-a-gate.
BSE_A_GATE_DEFAULT = 5.0


# ---------------------------------------------------------------------------
# Axis II kernels
# ---------------------------------------------------------------------------

def q_bar_real(p_hat: np.ndarray) -> np.ndarray:
    """mean_m p̂_{m,x}(1−p̂_{m,x})  — Fisher info kernel (code current)."""
    return (p_hat * (1.0 - p_hat)).mean(axis=0)


def q_bar_proxy(p_hat: np.ndarray) -> np.ndarray:
    """p̄_x(1−p̄_x)  — proxy kernel (paper text)."""
    p_bar = p_hat.mean(axis=0)
    return p_bar * (1.0 - p_bar)


def fir(q: np.ndarray, mask: np.ndarray) -> float:
    total = float(q.sum())
    if total <= 0:
        return float("nan")
    return float(q[mask].sum() / total)


# ---------------------------------------------------------------------------
# Axis I metrics
# ---------------------------------------------------------------------------

def bse_A_per_item(p_hat: np.ndarray, N_trials: np.ndarray,
                    mask: np.ndarray, n_ref: int = N_REF) -> float:
    """Case A: max over items of per-item SE of β̂_x, penalty √(n_ref/n_kept).

    SE_x = 1/√(Σ_m N · p̂(1−p̂))   (panel-summed plug-in Fisher info per item)
    bSE_A = max_x SE_x · √(n_ref/n_kept)
    """
    n_kept = int(mask.sum())
    if n_kept == 0:
        return float("nan")
    kept_p = p_hat[:, mask]                              # (M, n_kept)
    kept_N = N_trials[:, mask]                           # (M, n_kept)
    info_per_item = (kept_N * kept_p * (1.0 - kept_p)).sum(axis=0)  # (n_kept,)
    info_per_item = np.clip(info_per_item, 1e-12, None)
    se = 1.0 / np.sqrt(info_per_item)
    return float(se.max() * np.sqrt(n_ref / n_kept))


def bse_B_free(p_hat: np.ndarray, N_trials: np.ndarray,
                mask: np.ndarray, n_ref: int = N_REF) -> float:
    """Case B-free: max over panel models of per-model SE of θ̂_m on kept set,
    penalty √(n_kept/n_ref).

    SE_m = 1/√(Σ_{x∈kept} N · p̂(1−p̂))   (item-summed plug-in TIF per model)
    bSE_B_free = max_m SE_m · √(n_kept/n_ref)
    """
    n_kept = int(mask.sum())
    if n_kept == 0:
        return float("nan")
    kept_p = p_hat[:, mask]                              # (M, n_kept)
    kept_N = N_trials[:, mask]                           # (M, n_kept)
    tif_per_model = (kept_N * kept_p * (1.0 - kept_p)).sum(axis=1)  # (M,)
    tif_per_model = np.clip(tif_per_model, 1e-12, None)
    se = 1.0 / np.sqrt(tif_per_model)
    return float(se.max() * np.sqrt(n_kept / n_ref))


def bse_B_fit(beta_fit: np.ndarray, panel_theta_fit: np.ndarray,
               N_trials: np.ndarray, mask: np.ndarray,
               n_ref: int = N_REF,
               target_buffer: float = TARGET_BUFFER,
               n_grid: int = N_GRID) -> float:
    """Case B-fit: code current. max over θ-grid of TIF SE, fit-based.

    Mirrors compare_filters.py:845-850 exactly:
        target_grid = linspace(θ_min−buf, θ_max+buf, n_grid)
        SE(t) = 1/√(Σ_{x∈kept} N · σ(t−β̂_x)(1−σ(t−β̂_x)))
        bSE_B_fit = max_t SE(t) · √(n_kept/n_ref)
    """
    n_kept = int(mask.sum())
    if n_kept == 0:
        return float("nan")
    target_lo = panel_theta_fit.min() - target_buffer
    target_hi = panel_theta_fit.max() + target_buffer
    target_grid = np.linspace(target_lo, target_hi, n_grid)
    beta_sub = beta_fit[mask]
    N_sub = float(np.median(N_trials[:, mask]))
    se_grid = compute_se_grid(beta_sub, N_sub, target_grid)
    return float(se_grid.max() * np.sqrt(n_kept / n_ref))


# ---------------------------------------------------------------------------
# Filters (cheap β / θ proxies for filter selection only — bSE_B_fit uses
# the proper 1PL fit values via beta_fit / panel_theta_fit)
# ---------------------------------------------------------------------------

def cheap_beta(p_hat: np.ndarray) -> np.ndarray:
    """β proxy = −logit(p̄_x); used only by the β-window filters' selection."""
    p_bar = p_hat.mean(axis=0).clip(1e-6, 1 - 1e-6)
    return -np.log(p_bar / (1 - p_bar))


def cheap_panel_theta(p_hat: np.ndarray) -> np.ndarray:
    """θ proxy = logit(row-mean pass rate); used only for β-window endpoints."""
    rm = p_hat.mean(axis=1).clip(1e-6, 1 - 1e-6)
    return np.log(rm / (1 - rm))


def filter_specs(p_hat: np.ndarray, beta_fit: np.ndarray,
                  panel_theta_fit: np.ndarray):
    """Build the 15 filter candidates. β-window filters use the proper 1PL
    fit β̂_x / θ̂_m so that filter membership matches compare_filters.py."""
    specs = [
        ("NO_FILTER",       np.ones(p_hat.shape[1], dtype=bool)),
        ("AVG[.05,.95]",    filter_avg(p_hat, beta_fit, panel_theta_fit, 0.05, 0.95)),
        ("AVG[.10,.90]",    filter_avg(p_hat, beta_fit, panel_theta_fit, 0.10, 0.90)),
        ("AVG[.20,.80]",    filter_avg(p_hat, beta_fit, panel_theta_fit, 0.20, 0.80)),
        ("AVG[.30,.70]",    filter_avg(p_hat, beta_fit, panel_theta_fit, 0.30, 0.70)),
        ("AVG[.40,.60]",    filter_avg(p_hat, beta_fit, panel_theta_fit, 0.40, 0.60)),
        ("β[±0.25]",        filter_beta(p_hat, beta_fit, panel_theta_fit, -0.25, 0.25)),
        ("β[±0.50]",        filter_beta(p_hat, beta_fit, panel_theta_fit, -0.50, 0.50)),
        ("β[±1.00]",        filter_beta(p_hat, beta_fit, panel_theta_fit, -1.00, 1.00)),
        ("β[±1.50]",        filter_beta(p_hat, beta_fit, panel_theta_fit, -1.50, 1.50)),
        ("ALO[.20,.80]",    filter_at_least_one(p_hat, beta_fit, panel_theta_fit, 0.20, 0.80)),
        ("ALO[.30,.70]",    filter_at_least_one(p_hat, beta_fit, panel_theta_fit, 0.30, 0.70)),
        ("ATLAS-LV",        filter_low_variance(p_hat, beta_fit, panel_theta_fit,
                                                 std_threshold=0.05,
                                                 mean_lo=0.05, mean_hi=0.95)),
        ("ATLAS-PB",        filter_point_biserial(p_hat, beta_fit, panel_theta_fit,
                                                   r_threshold=0.10)),
        ("ATLAS-CB",        filter_atlas_combined(p_hat, beta_fit, panel_theta_fit,
                                                    std_threshold=0.05,
                                                    mean_lo=0.05, mean_hi=0.95,
                                                    r_threshold=0.10)),
    ]
    return specs


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------

def pareto_mask(bse, fir_vals):
    """(lower bSE, higher FIR) Pareto."""
    n = len(bse)
    keep = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j or not keep[j]:
                continue
            bi, fi = bse[i], fir_vals[i]
            bj, fj = bse[j], fir_vals[j]
            if not (np.isfinite(bi) and np.isfinite(fi)
                    and np.isfinite(bj) and np.isfinite(fj)):
                continue
            if (bj <= bi and fj >= fi and (bj < bi or fj > fi)):
                keep[i] = False
                break
    return keep


def pareto_3d(bse_a, bse_b, fir_vals):
    """3D Pareto: (bse_a ↓, bse_b ↓, fir ↑).

    F dominated by F' iff bse_a' ≤ bse_a AND bse_b' ≤ bse_b AND fir' ≥ fir
    with at least one strict inequality.
    """
    n = len(bse_a)
    keep = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j or not keep[j]:
                continue
            ai, bi, fi = bse_a[i], bse_b[i], fir_vals[i]
            aj, bj, fj = bse_a[j], bse_b[j], fir_vals[j]
            vals = [ai, bi, fi, aj, bj, fj]
            if not all(np.isfinite(v) for v in vals):
                continue
            if (aj <= ai and bj <= bi and fj >= fi
                    and (aj < ai or bj < bi or fj > fi)):
                keep[i] = False
                break
    return keep


# Paper's Stage A failures (from §app:filter-ablation text). True = ADMITTED.
# HRR/OR/CIF use Infit pass = 1.0 (non-binding); pseudo_R² gate fails some;
# SR uses uniform quasi-B bootstrap, admitting 3 filters whose 95%CI brackets
# the 0.95 gate.
STAGE_A_ADMITTED = {
    "HRR": {
        # AVG[0.30,0.70] and AVG[0.40,0.60] fail pseudo-R² gate (negative)
        "NO_FILTER": True, "AVG[.05,.95]": True, "AVG[.10,.90]": True,
        "AVG[.20,.80]": True, "AVG[.30,.70]": False, "AVG[.40,.60]": False,
        "β[±0.25]": True, "β[±0.50]": True, "β[±1.00]": True, "β[±1.50]": True,
        "ALO[.20,.80]": True, "ALO[.30,.70]": True,
        "ATLAS-LV": True, "ATLAS-PB": True, "ATLAS-CB": True,
    },
    "OR": {
        # AVG[0.40,0.60] fails pseudo-R² gate
        "NO_FILTER": True, "AVG[.05,.95]": True, "AVG[.10,.90]": True,
        "AVG[.20,.80]": True, "AVG[.30,.70]": True, "AVG[.40,.60]": False,
        "β[±0.25]": True, "β[±0.50]": True, "β[±1.00]": True, "β[±1.50]": True,
        "ALO[.20,.80]": True, "ALO[.30,.70]": True,
        "ATLAS-LV": True, "ATLAS-PB": True, "ATLAS-CB": True,
    },
    "CIF": {
        # AVG[0.40,0.60] fails pseudo-R² gate
        "NO_FILTER": True, "AVG[.05,.95]": True, "AVG[.10,.90]": True,
        "AVG[.20,.80]": True, "AVG[.30,.70]": True, "AVG[.40,.60]": False,
        "β[±0.25]": True, "β[±0.50]": True, "β[±1.00]": True, "β[±1.50]": True,
        "ALO[.20,.80]": True, "ALO[.30,.70]": True,
        "ATLAS-LV": True, "ATLAS-PB": True, "ATLAS-CB": True,
    },
    "SR": {
        # NO_FILTER fails Infit (saturated); 11 filters fail Infit, quasi-B
        # rescues AVG[0.05,0.95], β[±1.50], ATLAS-LV.
        "NO_FILTER": False,
        "AVG[.05,.95]": True,   # quasi-B admit
        "AVG[.10,.90]": False, "AVG[.20,.80]": False,
        "AVG[.30,.70]": False, "AVG[.40,.60]": False,
        "β[±0.25]": False, "β[±0.50]": False, "β[±1.00]": False,
        "β[±1.50]": True,       # quasi-B admit
        "ALO[.20,.80]": True, "ALO[.30,.70]": True,
        "ATLAS-LV": True,       # quasi-B admit
        "ATLAS-PB": True, "ATLAS-CB": True,
    },
}


# ---------------------------------------------------------------------------
# Per-category driver
# ---------------------------------------------------------------------------

def run_category(label: str, response_dir: Path,
                  bse_a_gate: float = BSE_A_GATE_DEFAULT) -> dict:
    rm = load_response_matrix(str(response_dir))
    p_hat = rm["p_hat"]                       # (M, J)
    N_trials = rm["total_trials"]             # (M, J)
    M, J = p_hat.shape

    # Fit 1PL once (only used by bSE_B_fit and to make β-window filters
    # match compare_filters.py)
    theta_init, beta_init = fit_ols(p_hat)
    theta_fit, beta_fit, _ = fit_mle(rm["pass_counts"], N_trials,
                                       theta_init, beta_init)

    # Axis II kernels (pool-level)
    q_r_all = q_bar_real(p_hat)
    q_p_all = q_bar_proxy(p_hat)

    rows = []
    for name, mask in filter_specs(p_hat, beta_fit, theta_fit):
        rows.append({
            "filter":   name,
            "n_kept":   int(mask.sum()),
            "bSE_A":      bse_A_per_item(p_hat, N_trials, mask),
            "bSE_B_free": bse_B_free(p_hat, N_trials, mask),
            "bSE_B_fit":  bse_B_fit(beta_fit, theta_fit, N_trials, mask),
            "FIR_real":   fir(q_r_all, mask),
            "FIR_proxy":  fir(q_p_all, mask),
        })

    # Pareto under each (bSE-variant, FIR-variant) pairing
    bsea = [r["bSE_A"] for r in rows]
    bseb = [r["bSE_B_free"] for r in rows]
    bsebf = [r["bSE_B_fit"] for r in rows]
    fr   = [r["FIR_real"]   for r in rows]
    fp   = [r["FIR_proxy"]  for r in rows]

    # Stage A admission: paper's pseudo-R² / Infit / quasi-B gates
    paper_gate = [STAGE_A_ADMITTED[label].get(r["filter"], True) for r in rows]
    # bSE_A explosion gate: kept set must yield finite-range bSE_A.
    bse_a_gate_pass = [np.isfinite(r["bSE_A"]) and r["bSE_A"] < bse_a_gate
                        for r in rows]
    admitted = [p and g for p, g in zip(paper_gate, bse_a_gate_pass)]
    for r, p, g, a in zip(rows, paper_gate, bse_a_gate_pass, admitted):
        r["paper_gate_pass"]  = p
        r["bse_a_gate_pass"]  = g
        r["bse_a_gate_thr"]   = bse_a_gate
    # Stage B: Pareto, but only consider Stage-A admitted as candidates;
    # non-admitted filters get P=False regardless.
    bsea_adm = [v if a else np.inf for v, a in zip(bsea,  admitted)]
    bseb_adm = [v if a else np.inf for v, a in zip(bseb,  admitted)]
    bsebf_adm = [v if a else np.inf for v, a in zip(bsebf, admitted)]
    fr_adm   = [v if a else -np.inf for v, a in zip(fr,   admitted)]
    fp_adm   = [v if a else -np.inf for v, a in zip(fp,   admitted)]

    pareto_map = {
        "A_real":     pareto_mask(bsea_adm,  fr_adm),
        "A_proxy":    pareto_mask(bsea_adm,  fp_adm),
        "Bfree_real":  pareto_mask(bseb_adm,  fr_adm),
        "Bfree_proxy": pareto_mask(bseb_adm,  fp_adm),
        "Bfit_real":   pareto_mask(bsebf_adm, fr_adm),
        "Bfit_proxy":  pareto_mask(bsebf_adm, fp_adm),
        # 3D Pareto: bSE_A + bSE_Bfree + FIR
        "AB_real":   pareto_3d(bsea_adm, bseb_adm, fr_adm),
        "AB_proxy":  pareto_3d(bsea_adm, bseb_adm, fp_adm),
    }
    # Force Stage A non-admitted to P=False (in case Pareto picked them up
    # via degenerate ±∞ injection)
    for key in pareto_map:
        pareto_map[key] = [p and a for p, a in zip(pareto_map[key], admitted)]
    for r, a in zip(rows, admitted):
        r["stage_A"] = a
    for r, *_p in zip(rows, *pareto_map.values()):
        pass
    for key, mask in pareto_map.items():
        for r, m in zip(rows, mask):
            r[f"P_{key}"] = m

    return {
        "category": label,
        "M": M, "J": J,
        "rows": rows,
        "theta_range": [float(theta_fit.min()), float(theta_fit.max())],
        "kernel_corr": float(np.corrcoef(q_r_all, q_p_all)[0, 1]),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_table(report: dict) -> str:
    lines = []
    lines.append(f"\n=== {report['category']}  (M={report['M']}, J={report['J']}, "
                 f"θ_panel ∈ [{report['theta_range'][0]:+.2f}, "
                 f"{report['theta_range'][1]:+.2f}]) ===")
    lines.append(f"  Axis II kernel corr (real vs proxy item-wise): "
                 f"{report['kernel_corr']:.4f}")
    hdr = (f"  {'filter':<16}  {'n':>4}  "
           f"{'bSE_A':>8}  {'bSE_Bfr':>8}  {'bSE_Bft':>8}  "
           f"{'FIR_r':>6}  {'FIR_p':>6}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for r in report["rows"]:
        lines.append(
            f"  {r['filter']:<16}  {r['n_kept']:>4}  "
            f"{r['bSE_A']:>8.4f}  {r['bSE_B_free']:>8.4f}  {r['bSE_B_fit']:>8.4f}  "
            f"{r['FIR_real']:>6.3f}  {r['FIR_proxy']:>6.3f}"
        )
    # Pareto summaries for the 6 (bSE × FIR) combinations
    lines.append("")
    lines.append("  Pareto frontier members:")
    combos = [
        ("A × real",    "P_A_real"),
        ("A × proxy",   "P_A_proxy"),
        ("Bfree × real",  "P_Bfree_real"),
        ("Bfree × proxy", "P_Bfree_proxy"),
        ("Bfit × real",   "P_Bfit_real"),
        ("Bfit × proxy",  "P_Bfit_proxy"),
    ]
    for label, key in combos:
        members = [r["filter"] for r in report["rows"] if r[key]]
        has_avg = "AVG[.05,.95]" in members
        lines.append(f"    {label:<14}  AVG[.05,.95]: {'✓' if has_avg else '✗'}  "
                     f"({len(members)} members: {members})")
    return "\n".join(lines)


def cross_category_avg_table(reports: list[dict]) -> str:
    """For each (bSE-variant × FIR-variant), report which categories have
    AVG[.05,.95] on the Pareto frontier, plus 4-of-4 winner check."""
    lines = ["", "=" * 78,
             "CROSS-CATEGORY: AVG[.05,.95] Pareto membership across 4 categories",
             "=" * 78]
    combos = [
        ("A × real",      "P_A_real"),
        ("A × proxy",     "P_A_proxy"),
        ("Bfree × real",   "P_Bfree_real"),
        ("Bfree × proxy",  "P_Bfree_proxy"),
        ("Bfit × real",    "P_Bfit_real"),
        ("Bfit × proxy",   "P_Bfit_proxy"),
        ("A+B 3D × real",  "P_AB_real"),
        ("A+B 3D × proxy", "P_AB_proxy"),
    ]
    cat_labels = [r["category"] for r in reports]
    hdr = f"  {'metric pair':<16}  " + "  ".join(f"{c:>5}" for c in cat_labels) + "   count"
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for label, key in combos:
        marks = []
        n_yes = 0
        for rep in reports:
            row = next(r for r in rep["rows"] if r["filter"] == "AVG[.05,.95]")
            yes = bool(row[key])
            marks.append("  ✓  " if yes else "     ")
            if yes:
                n_yes += 1
        lines.append(f"  {label:<16}  " + " ".join(marks) + f"   {n_yes}/4")

    # Stage C: per filter, count cats on Pareto (under A+B 3D × real)
    lines.append("")
    lines.append("=" * 78)
    lines.append("STAGE C (A+B 3D × real_FIR): per-filter Pareto count across 4 cats")
    lines.append("=" * 78)
    all_filters = [r["filter"] for r in reports[0]["rows"]]
    rows_stage_c = []
    for f in all_filters:
        count = 0
        marks_per_cat = []
        for rep in reports:
            row = next(r for r in rep["rows"] if r["filter"] == f)
            yes = bool(row["P_AB_real"])
            adm = bool(row["stage_A"])
            if not adm:
                marks_per_cat.append("  -  ")
            else:
                marks_per_cat.append("  ✓  " if yes else "     ")
                if yes:
                    count += 1
        # Mean FIR across categories admitted by Stage A (tie-break key)
        firs = []
        for rep in reports:
            row = next(r for r in rep["rows"] if r["filter"] == f)
            if row["stage_A"] and np.isfinite(row["FIR_real"]):
                firs.append(row["FIR_real"])
        mean_fir = float(np.mean(firs)) if firs else float("nan")
        rows_stage_c.append((f, count, mean_fir, marks_per_cat))
    rows_stage_c.sort(key=lambda x: (-x[1], -x[2] if not np.isnan(x[2]) else 0))
    hdr2 = (f"  {'filter':<16}  " + " ".join(f"{c:>5}" for c in cat_labels) +
            f"   {'count':>5}  {'mean FIR':>9}")
    lines.append(hdr2)
    lines.append("  " + "-" * (len(hdr2) - 2))
    for f, count, mean_fir, marks_per_cat in rows_stage_c:
        lines.append(f"  {f:<16}  " + " ".join(marks_per_cat) +
                     f"   {count:>5}  {mean_fir:>9.4f}")
    lines.append("")
    lines.append("  Legend:  - = Stage A rejected;  blank = Stage A pass, Pareto fail;")
    lines.append("           ✓ = Stage A pass AND on A+B 3D Pareto.")
    if rows_stage_c:
        winner = rows_stage_c[0]
        lines.append("")
        lines.append(f"  >>> Stage C winner (A+B 3D × real_FIR): "
                     f"{winner[0]}  (count={winner[1]}/4, mean FIR={winner[2]:.4f})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--bse-a-gate", type=float, default=BSE_A_GATE_DEFAULT,
                     help=(f"Stage A bSE_A explosion gate (logits); reject "
                           f"filters with bSE_A ≥ this. Default {BSE_A_GATE_DEFAULT}."))
    args = ap.parse_args()

    print(f"\nStage A gates: paper (pseudo-R²/Infit) + bSE_A < {args.bse_a_gate}")
    reports = []
    for label, rel in CATEGORIES.items():
        path = REPO / rel
        if not path.exists():
            print(f"[SKIP] {label}: {path} not found")
            continue
        rep = run_category(label, path, bse_a_gate=args.bse_a_gate)
        reports.append(rep)
        print(fmt_table(rep))

    print(cross_category_avg_table(reports))

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(reports, indent=2))
        print(f"\nSaved JSON to {args.out_json}")


if __name__ == "__main__":
    main()
