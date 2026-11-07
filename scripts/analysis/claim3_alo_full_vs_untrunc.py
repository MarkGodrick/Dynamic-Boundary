"""Phase 3 main-table simulator + full-vs-untrunc Rasch fit comparison.

For each (V model, category) on the merged ALO anchor:

  Option 1 (Full)      — (k, n) = (pass_count, total_trials)            [paper convention]
  Option 2 (Untrunc)   — (k, n) = (pass_count_among_nontrunc, total_trials - n_truncated)

Computes two estimates per option:
  theta_full  : EAP on full ALO anchor with frozen panel β̂ (no simulation)
  theta_adp   : Algorithm-1 simulator C0_baseline replay using response_lookup

Frozen β̂ + panel θ̂ range come from benchmarks/{cat}/anchor_alo_bundle_M9.json.
Merged response matrices come from results/{cat}/anchor_set_alo/<model>/.

Output:
  results/adaptive_alo/claim3_full_vs_untrunc.csv  (long form)
  Console table per category.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from adaptive.anchor_bundle import load_anchor_bundle
from scripts.analysis.claim3_ablation_simulation import (
    RobustEAPEstimator,
    simulate,
    CONFIGS,
)


CATS = [
    ("HRR", "harmful_refusal"),
    ("OR",  "over_refusal"),
    ("CIF", "constrained_completion"),
    ("SR",  "sycophancy_resistance"),
]
V_MODELS = [
    ("Olmo-3-7B",      "allenai_Olmo_3_7B_Instruct"),
    ("granite-4.1-8b", "ibm_granite_granite_4.1_8b"),
    ("phi-4",          "microsoft_phi_4"),
    ("Phi-4-r-plus",   "microsoft_Phi_4_reasoning_plus"),
    ("Falcon-H1R-7B",  "tiiuae_Falcon_H1R_7B"),
]

SIM_CFG_NAME = "T2c_eps0.02"  # paper main-table regime (plan §3.2)
SIM_CFG = CONFIGS[SIM_CFG_NAME]
PRIOR_SIGMA = SIM_CFG["prior_sigma"]


def build_response_lookups(path: Path) -> Tuple[Dict[str, Tuple[int, int]],
                                                Dict[str, Tuple[int, int]]]:
    """Return (full, untrunc) response lookups per prompt_id.

    Untrunc lookup omits items where every trial was truncated (n_untrunc = 0).
    """
    full: Dict[str, Tuple[int, int]] = {}
    untrunc: Dict[str, Tuple[int, int]] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            pid = row["prompt_id"]
            full[pid] = (int(row["pass_count"]), int(row["total_trials"]))
            n_untrunc = 0
            k_untrunc = 0
            for t in row.get("trials") or []:
                if not t.get("truncated"):
                    n_untrunc += 1
                    if t.get("passed"):
                        k_untrunc += 1
            if n_untrunc > 0:
                untrunc[pid] = (k_untrunc, n_untrunc)
    return full, untrunc


def eap_theta_full(bundle, lookup: Dict[str, Tuple[int, int]]):
    """EAP on all bundle items present in lookup; returns (theta, se, n_items_used)."""
    ks, ns, betas = [], [], []
    for item in bundle.items:
        if item.item_id not in lookup:
            continue
        k, n = lookup[item.item_id]
        if n <= 0:
            continue
        ks.append(k)
        ns.append(n)
        betas.append(item.beta)
    if not ks:
        return None, None, 0
    estimator = RobustEAPEstimator(prior_mu=0.0, prior_sigma=PRIOR_SIGMA, n_quad=21)
    theta, se = estimator.estimate(
        np.array(ks, dtype=np.float64),
        np.array(ns, dtype=np.float64),
        np.array(betas, dtype=np.float64),
    )
    return float(theta), float(se), len(ks)


def adp_theta(bundle, lookup: Dict[str, Tuple[int, int]]):
    """Run C0_baseline simulator; return (theta_hat_final, se_final, T_stop)."""
    # Restrict bundle items to those present in lookup (simulator returns error otherwise).
    if any(it.item_id not in lookup for it in bundle.items):
        kept = [it for it in bundle.items if it.item_id in lookup]
        b = type(bundle)(
            category=bundle.category,
            panel_models=bundle.panel_models,
            panel_theta=bundle.panel_theta,
            items=kept,
            filter_spec=bundle.filter_spec,
            M=bundle.M,
            N=bundle.N,
        )
    else:
        b = bundle
    sim = simulate(b, lookup, SIM_CFG, T_max=50)
    if "error" in sim:
        return None, None, None
    traj = sim["trajectory"]
    last = traj[-1]
    return float(last["theta_hat"]), float(last["theta_se"]), int(last["t"])


def main() -> None:
    out_rows: List[dict] = []
    panel_range_lines: List[str] = []
    for cat_lbl, cat in CATS:
        bundle_path = ROOT / "benchmarks" / cat / "anchor_alo_bundle_M9.json"
        bundle = load_anchor_bundle(bundle_path)
        pvals = list(bundle.panel_theta.values())
        pmin, pmax = float(min(pvals)), float(max(pvals))
        panel_range_lines.append(f"  {cat_lbl}: [{pmin:+.2f}, {pmax:+.2f}]   (J = {len(bundle.items)})")

        print(f"\n{'='*132}")
        print(f"  Category: {cat_lbl}    panel θ̂ range = [{pmin:+.2f}, {pmax:+.2f}]    J = {len(bundle.items)}")
        print(f"{'='*132}")
        hdr = (f"{'model':<16s} | {'opt':<8s} | "
               f"{'J_used':>6s} {'θ_full':>8s} {'SE_full':>8s} {'In/Out':>7s} | "
               f"{'θ_adp':>8s} {'SE_adp':>8s} {'T_stop':>6s} | "
               f"{'|Δ θ_full|':>11s} {'|Δ θ_adp|':>10s}")
        print(hdr)
        print("-" * len(hdr))

        for m_lbl, mkey in V_MODELS:
            rm_path = ROOT / "results" / cat / "anchor_set_alo" / mkey / "response_matrix.jsonl"
            if not rm_path.exists():
                print(f"{m_lbl:<16s} | MISSING")
                continue
            full_lookup, untrunc_lookup = build_response_lookups(rm_path)

            results = {}
            for opt_label, lookup in [("Full", full_lookup), ("Untrunc", untrunc_lookup)]:
                tf, sef, nf = eap_theta_full(bundle, lookup)
                ta, sea, ts = adp_theta(bundle, lookup)
                in_panel = (tf is not None) and (pmin <= tf <= pmax)
                results[opt_label] = {
                    "J_used": nf, "theta_full": tf, "se_full": sef,
                    "theta_adp": ta, "se_adp": sea, "t_stop": ts,
                    "in_panel": in_panel,
                }
                out_rows.append({
                    "category": cat_lbl, "model": m_lbl, "option": opt_label,
                    "J_total": len(bundle.items), "J_used": nf,
                    "theta_full": tf, "se_full": sef,
                    "theta_adp": ta, "se_adp": sea, "t_stop": ts,
                    "in_panel": in_panel,
                    "panel_min": pmin, "panel_max": pmax,
                })

            tf_full = results["Full"]["theta_full"]
            tf_un   = results["Untrunc"]["theta_full"]
            ta_full = results["Full"]["theta_adp"]
            ta_un   = results["Untrunc"]["theta_adp"]
            d_full = abs(tf_full - tf_un) if (tf_full is not None and tf_un is not None) else None
            d_adp  = abs(ta_full - ta_un) if (ta_full is not None and ta_un is not None) else None

            for opt_label in ["Full", "Untrunc"]:
                r = results[opt_label]
                io = "IN " if r["in_panel"] else "OUT"
                d_f_str = f"{d_full:+.3f}" if (opt_label == "Untrunc" and d_full is not None) else ("" if opt_label == "Full" else "  —  ")
                d_a_str = f"{d_adp:+.3f}"  if (opt_label == "Untrunc" and d_adp  is not None) else ("" if opt_label == "Full" else "  —  ")
                print(f"{m_lbl:<16s} | {opt_label:<8s} | "
                      f"{r['J_used']:>6d} {r['theta_full']:>+8.3f} {r['se_full']:>8.3f} {io:>7s} | "
                      f"{r['theta_adp']:>+8.3f} {r['se_adp']:>8.3f} {r['t_stop']:>6d} | "
                      f"{d_f_str:>11s} {d_a_str:>10s}")
            print()

    print(f"\nPanel θ̂ ranges (from anchor_alo_bundle_M9.json):")
    for line in panel_range_lines:
        print(line)

    out_dir = ROOT / "results" / "adaptive_alo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "claim3_full_vs_untrunc.csv"
    with open(out_csv, "w", newline="") as f:
        if out_rows:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            for r in out_rows:
                writer.writerow(r)
    print(f"\nSaved: {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
