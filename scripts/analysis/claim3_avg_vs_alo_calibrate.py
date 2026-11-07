"""Full validation-model calibration table: AVG vs ALO filter, all 5 V × 4 cats.

For each (V model, category) we compute θ_full under both filters:

  AVG (paper's current convention):
    β̂   ← benchmarks/{cat}/anchor_bundle_M9.json           (J=306/308/175/414)
    resp ← results/{cat}/anchor_set/<model>/response_matrix.jsonl

  ALO (proposed filter switch):
    β̂   ← benchmarks/{cat}/anchor_alo_bundle_M9.json       (J=293/299/183/416)
    resp ← results/{cat}/anchor_set_alo/<model>/response_matrix.jsonl
                                (merged overlap + alo_only, judge_revision = cot_strip)

Also reports Untrunc θ_full (trunc trials excluded) under both filters, so paper
can decide on the convention.

Output:
  results/adaptive_alo/claim3_avg_vs_alo_calibrate.csv
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
from scripts.analysis.claim3_ablation_simulation import RobustEAPEstimator, CONFIGS

PRIOR_SIGMA = CONFIGS["T2c_eps0.02"]["prior_sigma"]

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


def build_response_lookups(path: Path) -> Tuple[Dict[str, Tuple[int, int]],
                                                Dict[str, Tuple[int, int]]]:
    full: Dict[str, Tuple[int, int]] = {}
    untrunc: Dict[str, Tuple[int, int]] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            pid = row["prompt_id"]
            full[pid] = (int(row["pass_count"]), int(row["total_trials"]))
            n_u = k_u = 0
            for t in row.get("trials") or []:
                if not t.get("truncated"):
                    n_u += 1
                    if t.get("passed"):
                        k_u += 1
            if n_u > 0:
                untrunc[pid] = (k_u, n_u)
    return full, untrunc


def eap_on_bundle(bundle, lookup: Dict[str, Tuple[int, int]]):
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
    est = RobustEAPEstimator(prior_mu=0.0, prior_sigma=PRIOR_SIGMA, n_quad=21)
    theta, se = est.estimate(
        np.array(ks, dtype=np.float64),
        np.array(ns, dtype=np.float64),
        np.array(betas, dtype=np.float64),
    )
    return float(theta), float(se), len(ks)


def panel_range(bundle):
    v = list(bundle.panel_theta.values())
    return float(min(v)), float(max(v))


def fmt_pm(theta, se):
    return "  —  " if theta is None else f"{theta:+6.2f}±{se:.2f}"


def fmt_io(theta, lo, hi):
    if theta is None:
        return "?"
    return "IN " if (lo <= theta <= hi) else "OUT"


def main() -> None:
    rows: List[dict] = []
    panel_ranges = {}

    # Pre-load bundles
    bundles = {}
    for cat_lbl, cat in CATS:
        bundles[(cat_lbl, "AVG")] = load_anchor_bundle(ROOT / "benchmarks" / cat / "anchor_bundle_M9.json")
        bundles[(cat_lbl, "ALO")] = load_anchor_bundle(ROOT / "benchmarks" / cat / "anchor_alo_bundle_M9.json")
        panel_ranges[(cat_lbl, "AVG")] = panel_range(bundles[(cat_lbl, "AVG")])
        panel_ranges[(cat_lbl, "ALO")] = panel_range(bundles[(cat_lbl, "ALO")])

    # Print panel ranges header
    print("Panel θ̂ ranges:")
    for cat_lbl, _ in CATS:
        a_lo, a_hi = panel_ranges[(cat_lbl, "AVG")]
        l_lo, l_hi = panel_ranges[(cat_lbl, "ALO")]
        print(f"  {cat_lbl}:  AVG = [{a_lo:+.2f}, {a_hi:+.2f}]   ALO = [{l_lo:+.2f}, {l_hi:+.2f}]")

    for cat_lbl, cat in CATS:
        bundle_avg = bundles[(cat_lbl, "AVG")]
        bundle_alo = bundles[(cat_lbl, "ALO")]
        a_lo, a_hi = panel_ranges[(cat_lbl, "AVG")]
        l_lo, l_hi = panel_ranges[(cat_lbl, "ALO")]

        print(f"\n{'='*146}")
        print(f"  Category: {cat_lbl}    "
              f"AVG: J={len(bundle_avg.items)} panel=[{a_lo:+.2f}, {a_hi:+.2f}]    "
              f"ALO: J={len(bundle_alo.items)} panel=[{l_lo:+.2f}, {l_hi:+.2f}]")
        print(f"{'='*146}")
        hdr = (f"{'model':<16s} | "
               f"{'AVG/Full':>13s} {'IO':>3s} "
               f"{'AVG/Unt':>13s} {'IO':>3s} | "
               f"{'ALO/Full':>13s} {'IO':>3s} "
               f"{'ALO/Unt':>13s} {'IO':>3s} | "
               f"{'|Δ Full|':>9s} {'|Δ Unt|':>9s}")
        print(hdr)
        print("-" * len(hdr))

        for m_lbl, mkey in V_MODELS:
            rm_avg = ROOT / "results" / cat / "anchor_set" / mkey / "response_matrix.jsonl"
            rm_alo = ROOT / "results" / cat / "anchor_set_alo" / mkey / "response_matrix.jsonl"
            if not rm_avg.exists() or not rm_alo.exists():
                print(f"{m_lbl:<16s} | MISSING ({rm_avg.exists()=} {rm_alo.exists()=})")
                continue
            avg_full, avg_unt = build_response_lookups(rm_avg)
            alo_full, alo_unt = build_response_lookups(rm_alo)

            t_av_f, s_av_f, j_av_f = eap_on_bundle(bundle_avg, avg_full)
            t_av_u, s_av_u, j_av_u = eap_on_bundle(bundle_avg, avg_unt)
            t_al_f, s_al_f, j_al_f = eap_on_bundle(bundle_alo, alo_full)
            t_al_u, s_al_u, j_al_u = eap_on_bundle(bundle_alo, alo_unt)

            io_av_f = fmt_io(t_av_f, a_lo, a_hi)
            io_av_u = fmt_io(t_av_u, a_lo, a_hi)
            io_al_f = fmt_io(t_al_f, l_lo, l_hi)
            io_al_u = fmt_io(t_al_u, l_lo, l_hi)

            d_full = abs(t_av_f - t_al_f) if (t_av_f is not None and t_al_f is not None) else None
            d_unt  = abs(t_av_u - t_al_u) if (t_av_u is not None and t_al_u is not None) else None
            d_full_s = f"{d_full:+.3f}" if d_full is not None else "  —  "
            d_unt_s  = f"{d_unt:+.3f}"  if d_unt  is not None else "  —  "

            print(f"{m_lbl:<16s} | "
                  f"{fmt_pm(t_av_f, s_av_f):>13s} {io_av_f:>3s} "
                  f"{fmt_pm(t_av_u, s_av_u):>13s} {io_av_u:>3s} | "
                  f"{fmt_pm(t_al_f, s_al_f):>13s} {io_al_f:>3s} "
                  f"{fmt_pm(t_al_u, s_al_u):>13s} {io_al_u:>3s} | "
                  f"{d_full_s:>9s} {d_unt_s:>9s}")

            rows.append({
                "category": cat_lbl, "model": m_lbl,
                "avg_J_full": j_av_f, "avg_theta_full": t_av_f, "avg_se_full": s_av_f,
                "avg_J_unt": j_av_u, "avg_theta_unt": t_av_u, "avg_se_unt": s_av_u,
                "alo_J_full": j_al_f, "alo_theta_full": t_al_f, "alo_se_full": s_al_f,
                "alo_J_unt": j_al_u, "alo_theta_unt": t_al_u, "alo_se_unt": s_al_u,
                "avg_panel_min": a_lo, "avg_panel_max": a_hi,
                "alo_panel_min": l_lo, "alo_panel_max": l_hi,
                "io_avg_full": io_av_f, "io_avg_unt": io_av_u,
                "io_alo_full": io_al_f, "io_alo_unt": io_al_u,
                "delta_full_avg_alo": d_full, "delta_unt_avg_alo": d_unt,
            })

    out_dir = ROOT / "results" / "adaptive_alo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "claim3_avg_vs_alo_calibrate.csv"
    with open(out_csv, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    print(f"\nSaved: {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
