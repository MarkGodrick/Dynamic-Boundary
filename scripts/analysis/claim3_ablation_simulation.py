#!/usr/bin/env python3
"""
Ablation simulation for adaptive evaluation (Claim 3 diagnosis).

Replays the adaptive algorithm offline using the full anchor response matrix
under different config variants to isolate which algorithmic component(s)
cause the observed θ̂ bias.

Configs:
  C0_baseline:        current behavior (τ_se=0.30 early exit allowed)
  C1_gate_se_by_tmin: gate τ_se exit by t >= t_min (fix B1)
  C2_tighter_se:      C1 + tighten τ_se to 0.10
  C3_loose_prior:     C2 + loosen EAP prior σ to 3.0
  C_inf_weak_prior:   C2 + EAP σ=10 (~MLE)
  CT_run_full_T:      no early exit at all, force T_max=50

Coverage trigger v1 (anchor β based) vs v2 (panel θ̂ range based) recorded
at every iter to diagnose B2 (coverage check semantics).

Outputs (under results/adaptive_ablation/):
  claim3_ablation_trajectories.csv  long-form per-iter trajectory
  claim3_ablation_summary.csv       per-(model, cat, config) summary
  claim3_ablation_trigger.csv       B2 trigger timing analysis
  claim3_ablation_convergence.pdf   |θ̂_t - θ̂_full| vs t curves
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive.anchor_bundle import load_anchor_bundle
from adaptive.replay import (
    CATEGORIES,
    bundle_path_for,
    find_full_anchor_pairs,
    load_response_matrix,
    response_matrix_path_for,
    simulate,
)

CONFIGS: Dict[str, dict] = {
    # C0: replicates ACTUAL current behavior — no t_min gate on SE exit (bug B1).
    "C0_baseline":        dict(t_min=5, tau_se=0.30, prior_sigma=1.5,  apply_tmin_gate=False, force_no_early=False, termination="se_post"),
    # C1: B1 fix — gate SE exit by t >= t_min so trigger guards have a chance.
    "C1_gate_se_by_tmin": dict(t_min=5, tau_se=0.30, prior_sigma=1.5,  apply_tmin_gate=True,  force_no_early=False, termination="se_post"),
    # C2: C1 + tighten SE threshold.
    "C2_tighter_se":      dict(t_min=5, tau_se=0.10, prior_sigma=1.5,  apply_tmin_gate=True,  force_no_early=False, termination="se_post"),
    # C3: C2 + looser prior (t_min=5).
    "C3_loose_prior":     dict(t_min=5, tau_se=0.10, prior_sigma=3.0,  apply_tmin_gate=True,  force_no_early=False, termination="se_post"),
    # C3_tmin2: same as C3 but t_min=2 (test trigger-window/SE-gate trade-off).
    "C3_tmin2":           dict(t_min=2, tau_se=0.10, prior_sigma=3.0,  apply_tmin_gate=True,  force_no_early=False, termination="se_post"),
    # C_inf: σ=10 ≈ likelihood-only (with damped Newton clamp).
    "C_inf_weak_prior":   dict(t_min=5, tau_se=0.10, prior_sigma=10.0, apply_tmin_gate=True,  force_no_early=False, termination="se_post"),
    # CT: force-run T_max iters regardless of SE/trigger.
    "CT_run_full_T":      dict(t_min=5, tau_se=0.00, prior_sigma=1.5,  apply_tmin_gate=False, force_no_early=True,  termination="se_post"),
    # --- New: alternative termination criteria (all on σ=3.0, t_min=2 base) ---
    # T1: tighter posterior SE threshold (0.10 → 0.05).
    "T1_strict_se":       dict(t_min=2, tau_se=0.05, prior_sigma=3.0,  apply_tmin_gate=True,  force_no_early=False, termination="se_post"),
    # T2: stability-based termination (|Δθ̂_t| < ε for K consecutive iters).
    "T2_stability":       dict(t_min=2, prior_sigma=3.0, apply_tmin_gate=True, force_no_early=False, termination="stability",
                               stab_eps=0.05, stab_K=3),
    # T3: likelihood-only Fisher SE threshold (no prior contribution).
    "T3_lik_se":          dict(t_min=2, tau_se=0.10, prior_sigma=3.0,  apply_tmin_gate=True,  force_no_early=False, termination="se_lik"),
    # --- Path 1: tightened T1 / T2 thresholds ---
    "T1b_se0.03":         dict(t_min=2, tau_se=0.03, prior_sigma=3.0,  apply_tmin_gate=True,  force_no_early=False, termination="se_post"),
    "T1c_se0.02":         dict(t_min=2, tau_se=0.02, prior_sigma=3.0,  apply_tmin_gate=True,  force_no_early=False, termination="se_post"),
    "T2b_eps0.03":        dict(t_min=2, prior_sigma=3.0, apply_tmin_gate=True, force_no_early=False, termination="stability",
                               stab_eps=0.03, stab_K=3),
    "T2c_eps0.02":        dict(t_min=2, prior_sigma=3.0, apply_tmin_gate=True, force_no_early=False, termination="stability",
                               stab_eps=0.02, stab_K=3),
    # --- Path 2: combined stability AND likelihood-SE (both must satisfy) ---
    "T23a_stab+lik":      dict(t_min=2, tau_se=0.05, prior_sigma=3.0,  apply_tmin_gate=True,  force_no_early=False, termination="stability_and_lik",
                               stab_eps=0.05, stab_K=3),
    "T23b_strict_combo":  dict(t_min=2, tau_se=0.03, prior_sigma=3.0,  apply_tmin_gate=True,  force_no_early=False, termination="stability_and_lik",
                               stab_eps=0.03, stab_K=3),
    "T23c_very_strict":   dict(t_min=2, tau_se=0.02, prior_sigma=3.0,  apply_tmin_gate=True,  force_no_early=False, termination="stability_and_lik",
                               stab_eps=0.02, stab_K=3),
    # --- K-robustness for T2c (ε=0.02 fixed, vary K) ---
    "T2c_K2":             dict(t_min=2, prior_sigma=3.0, apply_tmin_gate=True, force_no_early=False, termination="stability",
                               stab_eps=0.02, stab_K=2),
    "T2c_K4":             dict(t_min=2, prior_sigma=3.0, apply_tmin_gate=True, force_no_early=False, termination="stability",
                               stab_eps=0.02, stab_K=4),
    "T2c_K5":             dict(t_min=2, prior_sigma=3.0, apply_tmin_gate=True, force_no_early=False, termination="stability",
                               stab_eps=0.02, stab_K=5),
    # --- K-robustness for T2b (ε=0.03 fixed, vary K) ---
    "T2b_K2":             dict(t_min=2, prior_sigma=3.0, apply_tmin_gate=True, force_no_early=False, termination="stability",
                               stab_eps=0.03, stab_K=2),
    "T2b_K4":             dict(t_min=2, prior_sigma=3.0, apply_tmin_gate=True, force_no_early=False, termination="stability",
                               stab_eps=0.03, stab_K=4),
    "T2b_K5":             dict(t_min=2, prior_sigma=3.0, apply_tmin_gate=True, force_no_early=False, termination="stability",
                               stab_eps=0.03, stab_K=5),
}

def build_summary_row(cat: str, model: str, cfg_name: str, sim: dict) -> dict:
    traj = sim['trajectory']
    theta_full = sim['theta_full']
    panel_min, panel_max = sim['panel_min'], sim['panel_max']

    checkpoints = {1: None, 5: None, 10: None, 20: None, 50: None}
    t_converge = -1
    for step in traj:
        t = step['t']
        if t in checkpoints and checkpoints[t] is None:
            checkpoints[t] = step['theta_hat']
        if t > 0 and t_converge < 0 and abs(step['theta_hat'] - theta_full) < 0.1:
            t_converge = t

    def absdelta(theta):
        return abs(theta - theta_full) if theta is not None else None

    return {
        "category": cat,
        "model": model,
        "config": cfg_name,
        "theta_full": theta_full,
        "se_full": sim['se_full'],
        "n_iters": len(traj) - 1,
        "stopped_reason": sim['stopped_reason'],
        "theta_at_t1": checkpoints[1],
        "theta_at_t5": checkpoints[5],
        "theta_at_t10": checkpoints[10],
        "theta_at_t20": checkpoints[20],
        "theta_at_t50": checkpoints[50],
        "theta_final": traj[-1]['theta_hat'],
        "abs_delta_t1": absdelta(checkpoints[1]),
        "abs_delta_t5": absdelta(checkpoints[5]),
        "abs_delta_t10": absdelta(checkpoints[10]),
        "abs_delta_t20": absdelta(checkpoints[20]),
        "abs_delta_t50": absdelta(checkpoints[50]),
        "abs_delta_final": abs(traj[-1]['theta_hat'] - theta_full),
        "t_converge": t_converge,
        "panel_min": panel_min,
        "panel_max": panel_max,
        "is_outside_panel_range": (theta_full < panel_min or theta_full > panel_max),
    }


def first_trigger_combined(traj: List[dict], key: str, t_min: int, L: int) -> int:
    """First t at which (cov[key] True for L consecutive iters) AND t >= t_min."""
    c = 0
    for step in traj:
        t = step['t']
        if t == 0:
            continue
        if step[key]:
            c += 1
            if c >= L and t >= t_min:
                return int(t)
        else:
            c = 0
    return -1


def build_trigger_row(cat: str, model: str, cfg_name: str, sim: dict, summary_row: dict) -> dict:
    traj = sim['trajectory']
    row = {
        "category": cat,
        "model": model,
        "config": cfg_name,
        "should_trigger": summary_row['is_outside_panel_range'],
        "panel_min": sim['panel_min'],
        "panel_max": sim['panel_max'],
        "theta_full": summary_row['theta_full'],
    }
    # Cross-product of (cov key, t_min, L)
    for key in ('cov_v1', 'cov_v2'):
        for t_min in (2, 5):
            for L in (1, 2, 3):
                col = f"t_{key}_tmin{t_min}_L{L}"
                row[col] = first_trigger_combined(traj, key, t_min, L)
    return row


def plot_convergence(df_traj: pd.DataFrame, output_path: Path) -> None:
    pairs = (df_traj.drop_duplicates(subset=['category', 'model'])
                    [['category', 'model']].values.tolist())
    n_pairs = len(pairs)
    n_cols = 4
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.7 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    colors = {
        "C0_baseline":        "tab:red",
        "C1_gate_se_by_tmin": "tab:orange",
        "C2_tighter_se":      "tab:olive",
        "C3_loose_prior":     "tab:green",
        "C3_tmin2":           "tab:cyan",
        "C_inf_weak_prior":   "tab:blue",
        "CT_run_full_T":      "tab:purple",
        "T1_strict_se":       "tab:brown",
        "T2_stability":       "tab:pink",
        "T3_lik_se":          "tab:gray",
        "T1b_se0.03":         "lightcoral",
        "T1c_se0.02":         "darkred",
        "T2b_eps0.03":        "violet",
        "T2c_eps0.02":        "darkviolet",
        "T23a_stab+lik":      "gold",
        "T23b_strict_combo":  "orange",
        "T23c_very_strict":   "darkorange",
        "T2c_K2":             "lightblue",
        "T2c_K4":             "steelblue",
        "T2c_K5":             "midnightblue",
        "T2b_K2":             "thistle",
        "T2b_K4":             "mediumorchid",
        "T2b_K5":             "purple",
    }

    for i, (cat, model) in enumerate(pairs):
        ax = axes[i]
        sub = df_traj[(df_traj['category'] == cat) & (df_traj['model'] == model)]
        for cfg_name in CONFIGS.keys():
            cfg_sub = sub[sub['config'] == cfg_name].sort_values('t')
            if len(cfg_sub) == 0:
                continue
            # Only plot main iters (t > 0); pilot is t=0 reference
            cfg_main = cfg_sub[cfg_sub['t'] > 0]
            ax.plot(cfg_main['t'], cfg_main['abs_delta_theta'],
                    label=cfg_name, color=colors[cfg_name],
                    linewidth=1.4, alpha=0.85)
        ax.axhline(0.1, color='black', linestyle='--', linewidth=0.6, alpha=0.6)
        ax.set_yscale('log')
        ax.set_ylim(0.001, 5)
        cat_short = {"harmful_refusal": "HR", "over_refusal": "OR",
                     "constrained_completion": "CC",
                     "sycophancy_resistance": "SR"}.get(cat, cat[:3])
        model_short = (model.replace('_', ' ')
                            .replace('allenai ', '').replace('deepseek ai ', '')
                            .replace('microsoft ', '').replace('tiiuae ', '')
                            .replace('ibm granite ', ''))[:32]
        ax.set_title(f"{cat_short} / {model_short}", fontsize=8)
        ax.set_xlabel("iter t")
        ax.set_ylabel(r"$|\hat\theta_t - \hat\theta_{\rm full}|$")
        ax.grid(alpha=0.3, which='both')

    for j in range(n_pairs, len(axes)):
        axes[j].axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, bbox_inches='tight', dpi=120)
    plt.close()


def print_summary_table(df_summary: pd.DataFrame) -> None:
    print()
    print("=" * 110)
    print("Mean |Δθ̂| at iter checkpoints (averaged over all (model, cat) pairs with full anchor)")
    print("=" * 110)
    header = f"  {'config':25s}  " + "  ".join(f"t={t:<3d}" for t in (1, 5, 10, 20, 50)) + "   final  conv_rate  med_t_conv"
    print(header)
    print("-" * 110)

    for cfg in CONFIGS.keys():
        sub = df_summary[df_summary['config'] == cfg]
        cells = []
        for col in ('abs_delta_t1', 'abs_delta_t5', 'abs_delta_t10',
                    'abs_delta_t20', 'abs_delta_t50'):
            m = sub[col].dropna().mean()
            cells.append(f"{m:5.3f}" if pd.notna(m) else "  -  ")
        m_final = sub['abs_delta_final'].dropna().mean()
        n_total = len(sub)
        n_conv = (sub['t_converge'] > 0).sum()
        med_t_conv = sub.loc[sub['t_converge'] > 0, 't_converge'].median()
        med_t_conv_str = f"{med_t_conv:.0f}" if pd.notna(med_t_conv) else " -"
        print(f"  {cfg:25s}  " + "  ".join(cells) +
              f"   {m_final:5.3f}    {n_conv:2d}/{n_total:2d}      {med_t_conv_str}")


def print_trigger_table(df_trigger: pd.DataFrame) -> None:
    print()
    print("=" * 140)
    print("Coverage trigger timing — under CT_run_full_T (T_max=50, no early exit)")
    print("=" * 140)
    print("Per (model, cat) — first t s.t. (cov[v_x] True for L consec iters) AND (t >= t_min)")
    print()
    print(f"  {'cat':3s}  {'model':32s}  {'should':6s}  "
          f"{'v1 t2 L1':9s}  {'v1 t2 L2':9s}  {'v1 t5 L2':9s}  {'v1 t5 L3':9s}  "
          f"{'v2 t2 L1':9s}  {'v2 t2 L2':9s}  {'v2 t5 L2':9s}  {'v2 t5 L3':9s}")
    print("-" * 140)

    sub = df_trigger[df_trigger['config'] == 'CT_run_full_T'].copy()
    sub = sub.sort_values(['should_trigger', 'category', 'model'], ascending=[False, True, True])
    cat_short_map = {"harmful_refusal": "HR", "over_refusal": "OR",
                     "constrained_completion": "CC", "sycophancy_resistance": "SR"}
    cols_to_show = [
        't_cov_v1_tmin2_L1', 't_cov_v1_tmin2_L2', 't_cov_v1_tmin5_L2', 't_cov_v1_tmin5_L3',
        't_cov_v2_tmin2_L1', 't_cov_v2_tmin2_L2', 't_cov_v2_tmin5_L2', 't_cov_v2_tmin5_L3',
    ]
    for _, row in sub.iterrows():
        cells = []
        for col in cols_to_show:
            v = int(row[col])
            cells.append(f"t={v}" if v > 0 else "—")
        cat_s = cat_short_map.get(row['category'], row['category'][:3])
        should = "YES" if row['should_trigger'] else "no"
        print(f"  {cat_s:3s}  {row['model'][:32]:32s}  {should:6s}  " +
              "  ".join(f"{c:9s}" for c in cells))

    print()
    print("Precision / Recall / F1 (vs. should_trigger ground truth):")
    print(f"  {'judge':30s}  {'TP':>3s}  {'FP':>3s}  {'FN':>3s}  {'TN':>3s}  "
          f"{'prec':>5s}  {'recall':>6s}  {'F1':>5s}")
    print("-" * 80)

    judges = []
    for key, key_label in (('cov_v1', 'cov_v1 (anchor β)'), ('cov_v2', 'cov_v2 (panel θ̂)')):
        for t_min in (2, 5):
            for L in (1, 2, 3):
                col = f"t_{key}_tmin{t_min}_L{L}"
                judges.append((f"{key_label} t_min={t_min} L={L}", col))

    for label, col in judges:
        fired = sub[col] > 0
        should = sub['should_trigger'].astype(bool)
        TP = int(((fired) & (should)).sum())
        FP = int(((fired) & (~should)).sum())
        FN = int(((~fired) & (should)).sum())
        TN = int(((~fired) & (~should)).sum())
        prec = TP / (TP + FP) if (TP + FP) > 0 else float('nan')
        recall = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else float('nan')
        print(f"  {label:30s}  {TP:>3d}  {FP:>3d}  {FN:>3d}  {TN:>3d}  "
              f"{prec:>5.2f}  {recall:>6.2f}  {f1:>5.2f}")


def print_tmin_inrange_table(df_summary: pd.DataFrame) -> None:
    """Compare in-range accuracy under C3 (t_min=5) vs C3_tmin2."""
    print()
    print("=" * 100)
    print("t_min effect on IN-RANGE accuracy (C3 with σ=3.0, τ_se=0.10)")
    print("=" * 100)
    in_range = df_summary[df_summary['is_outside_panel_range'] == False]

    for cfg in ('C3_loose_prior', 'C3_tmin2'):
        sub = in_range[in_range['config'] == cfg]
        if len(sub) == 0:
            continue
        n_total = len(sub)
        mean_d = sub['abs_delta_final'].mean()
        med_d = sub['abs_delta_final'].median()
        n_conv = (sub['abs_delta_final'] < 0.1).sum()
        # Mean iters consumed (excluding pilot)
        mean_iters = sub['n_iters'].mean()
        tag = "t_min=5" if cfg == 'C3_loose_prior' else "t_min=2"
        print(f"  {tag:10s}: n={n_total}, mean |Δθ̂|_final = {mean_d:.3f}, median = {med_d:.3f}, "
              f"final<0.1: {n_conv}/{n_total}, mean iters: {mean_iters:.1f}")

    # Per-model comparison
    print()
    print("Per-model in-range comparison:")
    print(f"  {'category':22s}  {'model':35s}  {'t_min=5 |Δ|':12s}  {'t_min=2 |Δ|':12s}  {'Δ(t2-t5)':10s}")
    print("-" * 100)
    pivot = in_range.pivot_table(index=['category', 'model'],
                                  columns='config',
                                  values='abs_delta_final')
    for (cat, model), row in pivot.iterrows():
        t5 = row.get('C3_loose_prior', float('nan'))
        t2 = row.get('C3_tmin2', float('nan'))
        delta = t2 - t5 if pd.notna(t2) and pd.notna(t5) else float('nan')
        print(f"  {cat:22s}  {model[:35]:35s}  {t5:>12.3f}  {t2:>12.3f}  {delta:>+10.3f}")


def sanity_check_t1_simulation(pairs: List[Tuple[str, str]], filter_rule: str) -> None:
    """Compare simulated t=1 θ̂ vs original adaptive run's t=1 θ̂_post."""
    print("=" * 110)
    print("Sanity check: simulated θ̂@t=1 vs original adaptive run's θ̂_post@t=1")
    print("(differences come from response_matrix vs original-run sampling noise; should be < 0.10)")
    print("-" * 110)

    for cat, model in pairs:
        bundle = load_anchor_bundle(bundle_path_for(cat, filter_rule))
        rm_path = response_matrix_path_for(cat, model, filter_rule)
        response_lookup = load_response_matrix(rm_path)

        sim = simulate(bundle, response_lookup, CONFIGS['C0_baseline'], T_max=1)
        if "error" in sim:
            continue
        sim_t1_theta = next((s['theta_hat'] for s in sim['trajectory'] if s['t'] == 1), None)

        # Find original adaptive result
        adaptive_dir = PROJECT_ROOT / "results" / "adaptive" / cat
        run_dirs = list(adaptive_dir.glob(f"{model}_{cat}_eap_*"))
        if not run_dirs:
            continue
        result_path = run_dirs[0] / "result.json"
        if not result_path.exists():
            continue
        with open(result_path) as f:
            res = json.load(f)
        orig_t1 = next((h['theta_hat_post'] for h in res['history'] if h['t'] == 1), None)

        if sim_t1_theta is not None and orig_t1 is not None:
            diff = abs(sim_t1_theta - orig_t1)
            flag = "OK" if diff < 0.15 else "HIGH"
            print(f"  {cat:22s}  {model[:35]:35s}  sim θ̂={sim_t1_theta:+6.3f}  orig θ̂={orig_t1:+6.3f}  Δ={diff:.3f}  [{flag}]")


def main():
    parser = argparse.ArgumentParser(
        description="Offline replay of Claim 3 adaptive evaluation from full response matrices.",
    )
    parser.add_argument(
        "--filter-rule",
        choices=["avg", "alo"],
        default="alo",
        help="avg uses anchor_bundle_M9 + anchor_set; alo uses anchor_alo_bundle_M9 + anchor_set_alo.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to results/adaptive_ablation_alo for ALO and results/adaptive_ablation for AVG.",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip the AVG-era t=1 check against original online adaptive runs.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        default_name = "adaptive_ablation_alo" if args.filter_rule == "alo" else "adaptive_ablation"
        output_dir = PROJECT_ROOT / "results" / default_name
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_full_anchor_pairs(CATEGORIES, args.filter_rule)
    print(
        f"Found {len(pairs)} (model, category) pairs with full "
        f"{args.filter_rule.upper()} anchor data\n"
    )

    # The sanity check compares against old online adaptive result.json files,
    # which were produced on the AVG anchor path. Skip it for ALO replay.
    if args.filter_rule == "avg" and not args.skip_sanity:
        sanity_check_t1_simulation(pairs, args.filter_rule)
        print()
    else:
        print("Skipping t=1 online-run sanity check for this replay.\n")

    # Run all configs
    all_traj_rows: List[dict] = []
    all_summary_rows: List[dict] = []
    all_trigger_rows: List[dict] = []

    for cat, model in pairs:
        bundle = load_anchor_bundle(bundle_path_for(cat, args.filter_rule))
        rm_path = response_matrix_path_for(cat, model, args.filter_rule)
        response_lookup = load_response_matrix(rm_path)

        for cfg_name, cfg in CONFIGS.items():
            sim = simulate(bundle, response_lookup, cfg)
            if "error" in sim:
                print(f"  SKIP {cat} / {model} / {cfg_name}: {sim['error']}")
                continue

            theta_full = sim['theta_full']
            for step in sim['trajectory']:
                all_traj_rows.append({
                    "category": cat,
                    "model": model,
                    "config": cfg_name,
                    "t": step['t'],
                    "phase": step['phase'],
                    "item_id": step['item_id'],
                    "beta_x": step['beta_x'],
                    "theta_hat": step['theta_hat'],
                    "theta_se": step['theta_se'],
                    "abs_delta_theta": abs(step['theta_hat'] - theta_full),
                    "cov_v1": step['cov_v1'],
                    "cov_v2": step['cov_v2'],
                })

            summary = build_summary_row(cat, model, cfg_name, sim)
            all_summary_rows.append(summary)
            all_trigger_rows.append(build_trigger_row(cat, model, cfg_name, sim, summary))

        print(f"  done: {cat} / {model}")

    df_traj = pd.DataFrame(all_traj_rows)
    df_summary = pd.DataFrame(all_summary_rows)
    df_trigger = pd.DataFrame(all_trigger_rows)

    df_traj.to_csv(output_dir / "claim3_ablation_trajectories.csv", index=False)
    df_summary.to_csv(output_dir / "claim3_ablation_summary.csv", index=False)
    df_trigger.to_csv(output_dir / "claim3_ablation_trigger.csv", index=False)
    print(f"\nCSVs saved to {output_dir}")

    plot_convergence(df_traj, output_dir / "claim3_ablation_convergence.pdf")
    print(f"Figure saved to {output_dir}/claim3_ablation_convergence.pdf")

    print_summary_table(df_summary)
    print_trigger_table(df_trigger)
    print_tmin_inrange_table(df_summary)
    print_termination_comparison(df_summary)
    print_K_robustness(df_summary)


def print_K_robustness(df_summary: pd.DataFrame) -> None:
    """K ∈ {2,3,4,5} robustness for stability-based termination."""
    from scipy import stats

    print()
    print("=" * 110)
    print("K-robustness check for stability-based termination (in-range, n=13)")
    print("=" * 110)
    print(f"  {'config':25s}  {'eps':>5s}  {'K':>2s}  "
          f"{'mean|Δ|':>8s}  {'med|Δ|':>7s}  {'ρ_S':>5s}  {'τ_K':>5s}  "
          f"{'<0.3':>5s}  {'iters':>6s}  {'verdict':10s}")
    print("-" * 110)

    in_range = df_summary[df_summary['is_outside_panel_range'] == False]

    K_configs_T2c = [
        ('T2c_K2',     0.02, 2),
        ('T2c_eps0.02', 0.02, 3),  # this is the original T2c (K=3)
        ('T2c_K4',     0.02, 4),
        ('T2c_K5',     0.02, 5),
    ]
    K_configs_T2b = [
        ('T2b_K2',     0.03, 2),
        ('T2b_eps0.03', 0.03, 3),  # original T2b (K=3)
        ('T2b_K4',     0.03, 4),
        ('T2b_K5',     0.03, 5),
    ]

    print("# T2c series (ε = 0.02)")
    for cfg, eps, K in K_configs_T2c:
        sub = in_range[in_range['config'] == cfg]
        if len(sub) == 0:
            continue
        theta_pred = sub['theta_final'].values
        theta_true = sub['theta_full'].values
        abs_delta = np.abs(theta_pred - theta_true)
        spearman = stats.spearmanr(theta_pred, theta_true).statistic
        kendall = stats.kendalltau(theta_pred, theta_true).statistic
        n_03 = int((abs_delta < 0.3).sum())
        mean_iters = sub['n_iters'].mean()

        if spearman >= 0.95 and kendall >= 0.85:
            verdict = "✓ ReEval"
        elif spearman >= 0.85 and kendall >= 0.75:
            verdict = "~Fluid"
        elif kendall >= 0.7:
            verdict = "≥Madaan"
        else:
            verdict = "✗ poor"

        print(f"  {cfg:25s}  {eps:>5.2f}  {K:>2d}  "
              f"{abs_delta.mean():>8.3f}  {np.median(abs_delta):>7.3f}  "
              f"{spearman:>5.3f}  {kendall:>5.3f}  "
              f"{n_03:>2d}/13  {mean_iters:>6.1f}  {verdict:10s}")

    print()
    print("# T2b series (ε = 0.03)")
    for cfg, eps, K in K_configs_T2b:
        sub = in_range[in_range['config'] == cfg]
        if len(sub) == 0:
            continue
        theta_pred = sub['theta_final'].values
        theta_true = sub['theta_full'].values
        abs_delta = np.abs(theta_pred - theta_true)
        spearman = stats.spearmanr(theta_pred, theta_true).statistic
        kendall = stats.kendalltau(theta_pred, theta_true).statistic
        n_03 = int((abs_delta < 0.3).sum())
        mean_iters = sub['n_iters'].mean()

        if spearman >= 0.95 and kendall >= 0.85:
            verdict = "✓ ReEval"
        elif spearman >= 0.85 and kendall >= 0.75:
            verdict = "~Fluid"
        elif kendall >= 0.7:
            verdict = "≥Madaan"
        else:
            verdict = "✗ poor"

        print(f"  {cfg:25s}  {eps:>5.2f}  {K:>2d}  "
              f"{abs_delta.mean():>8.3f}  {np.median(abs_delta):>7.3f}  "
              f"{spearman:>5.3f}  {kendall:>5.3f}  "
              f"{n_03:>2d}/13  {mean_iters:>6.1f}  {verdict:10s}")


def print_termination_comparison(df_summary: pd.DataFrame) -> None:
    """Compare alternative termination criteria (T1, T2, T3) vs C3 and CT."""
    from scipy import stats

    print()
    print("=" * 110)
    print("Termination-criterion comparison (in-range, n=13) — vs literature standards")
    print("=" * 110)
    print(f"  {'config':25s}  {'mean|Δ|':>8s}  {'med|Δ|':>7s}  {'RMSE':>5s}  "
          f"{'ρ_P':>5s}  {'ρ_S':>5s}  {'τ_K':>5s}  {'<0.3':>5s}  {'iters':>5s}  "
          f"{'verdict':10s}")
    print("-" * 110)

    in_range = df_summary[df_summary['is_outside_panel_range'] == False]

    configs_to_show = ['C0_baseline', 'C3_loose_prior',
                       'T1_strict_se', 'T1b_se0.03', 'T1c_se0.02',
                       'T2_stability', 'T2b_eps0.03', 'T2c_eps0.02',
                       'T3_lik_se',
                       'T23a_stab+lik', 'T23b_strict_combo', 'T23c_very_strict',
                       'CT_run_full_T']
    for cfg in configs_to_show:
        sub = in_range[in_range['config'] == cfg].copy()
        if len(sub) == 0:
            continue
        theta_pred = sub['theta_final'].values
        theta_true = sub['theta_full'].values
        abs_delta = np.abs(theta_pred - theta_true)
        rmse = float(np.sqrt((abs_delta ** 2).mean()))
        pearson = stats.pearsonr(theta_pred, theta_true).statistic
        spearman = stats.spearmanr(theta_pred, theta_true).statistic
        kendall = stats.kendalltau(theta_pred, theta_true).statistic
        n_03 = int((abs_delta < 0.3).sum())
        n_total = len(sub)
        mean_iters = sub['n_iters'].mean()

        # Verdict against literature
        if spearman >= 0.95 and kendall >= 0.85:
            verdict = "✓ ReEval"
        elif spearman >= 0.85 and kendall >= 0.75:
            verdict = "~Fluid"
        elif kendall >= 0.7:
            verdict = "≥Madaan"
        else:
            verdict = "✗ poor"

        print(f"  {cfg:25s}  {abs_delta.mean():>8.3f}  {np.median(abs_delta):>7.3f}  "
              f"{rmse:>5.3f}  {pearson:>5.3f}  {spearman:>5.3f}  {kendall:>5.3f}  "
              f"{n_03:>2d}/{n_total:<2d}  {mean_iters:>5.1f}  {verdict:10s}")

    print()
    print("Literature reference:")
    print("  ReEval (within-generator):   ρ_S = 0.96")
    print("  Fluid validity (102 LMs):    τ_K = 0.70-0.85")
    print("  Madaan critique baseline:    τ_K = 0.76-0.91 (he calls this BAD)")
    print("  Educational CAT standard:    mean |Δ| ≤ 0.30 logits")


if __name__ == "__main__":
    main()
