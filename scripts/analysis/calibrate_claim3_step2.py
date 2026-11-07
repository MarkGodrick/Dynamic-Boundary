#!/usr/bin/env python3
"""Claim 3 Step-2 calibration: augmented θ̂_new for out-of-panel validation cells.

Offline analysis (no GPU). Per (category, model) cell that fell outside the
reference-panel θ range in Step 1, this fits the SGBS boundary items through the
panel and re-estimates the validation model's ability on the augmented set.

Pipeline producing the inputs (run beforehand, per cell):
  1. scripts/run_sgbs.py            (mode: async, K=50) -> edge_cases.jsonl
  2. scripts/data_prep/build_edge_case_dataset.py       -> edge_case_dataset.jsonl
  3. scripts/collect_response_matrix.py --models <panel + V model>
        -> <matrix_root>/<model_short>/response_matrix.jsonl  (panel inferred;
           the V model's search-phase rows reused verbatim, source=search_phase)

This script then, with panel θ̂ and original anchor β̂ frozen, fits only:
  - β̂ of each surviving boundary item   (1-D MLE, panel cells; adaptive.calibrate._fit_beta_1d)
  - θ̂_new of the validation model        (EAP over anchor ∪ boundary items)

Two anchor scopes are reported:
  - full   : θ̂ over (full anchor ∪ E_new)         — paper full-anchor target framing
  - cat    : θ̂ over (CAT-used anchor subset ∪ E_new) — runtime-DBE narrative

Output: results/adaptive_alo/claim3_step2_calibrated.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive.anchor_bundle import load_anchor_bundle
from adaptive.calibrate import _fit_beta_1d
from adaptive.posterior import EAPEstimator
from adaptive.replay import (
    T2C_CONFIG,
    bundle_path_for,
    load_response_matrix,
    response_matrix_path_for,
    simulate,
)

# The 7 out-of-panel cells (ALO filter), locked from
# results/adaptive_ablation_alo/claim3_ablation_summary.csv (is_outside_panel_range).
# model strings match the results/<cat>/anchor_set_alo/<model>/ directory names.
DEFAULT_CELLS = [
    ("harmful_refusal", "microsoft_Phi_4_reasoning_plus"),
    ("harmful_refusal", "microsoft_phi_4"),
    ("harmful_refusal", "tiiuae_Falcon_H1R_7B"),
    ("over_refusal", "tiiuae_Falcon_H1R_7B"),
    ("constrained_completion", "tiiuae_Falcon_H1R_7B"),
    ("sycophancy_resistance", "microsoft_Phi_4_reasoning_plus"),
    ("sycophancy_resistance", "tiiuae_Falcon_H1R_7B"),
]

# Same prior σ as the T2c simulator that produced tab:claim3 (paper primary).
PRIOR_SIGMA = float(T2C_CONFIG["prior_sigma"])


def _short(model_hf_name: str) -> str:
    """Match collect_response_matrix's output dir naming."""
    return model_hf_name.replace("/", "_").replace("-", "_")


def calibrate_cell(category: str, model: str, matrix_root: Path,
                   filter_rule: str = "alo") -> dict:
    bundle = load_anchor_bundle(bundle_path_for(category, filter_rule))
    panel_names = list(bundle.panel_models)
    panel_theta = np.array([bundle.panel_theta[n] for n in panel_names], dtype=np.float64)

    # --- Boundary items E_new: V-model (k,n) from the search-phase dataset ---
    cell_dir = matrix_root / category / "claim3_step2" / model
    matrix_dir = cell_dir / "matrix"
    ds_candidates = sorted(cell_dir.glob("edge_case_dataset*.jsonl"))
    if not ds_candidates:
        raise FileNotFoundError(f"no edge_case_dataset*.jsonl in {cell_dir}")
    vmodel_new = {}
    with open(ds_candidates[0]) as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            vmodel_new[e["prompt_id"]] = (int(e["pass_count"]), int(e["total_trials"]))
    new_ids = sorted(vmodel_new.keys())

    # --- Panel responses on E_new ---
    panel_resp = {
        name: load_response_matrix(matrix_dir / _short(name) / "response_matrix.jsonl")
        for name in panel_names
    }

    # --- Fit β̂ per new item (panel θ frozen); drop saturated / missing-cell ---
    new_beta, new_k, new_n = [], [], []
    n_dropped = 0
    for pid in new_ids:
        cells_k, cells_n, miss = [], [], False
        for name in panel_names:
            row = panel_resp[name].get(pid)
            if row is None:
                miss = True
                break
            cells_k.append(row[0])
            cells_n.append(row[1])
        if miss:
            n_dropped += 1
            continue
        beta_x = _fit_beta_1d(
            np.asarray(cells_k, dtype=np.float64),
            np.asarray(cells_n, dtype=np.float64),
            panel_theta,
        )
        if beta_x is None:  # saturated → unidentifiable
            n_dropped += 1
            continue
        k_v, n_v = vmodel_new[pid]
        new_beta.append(beta_x)
        new_k.append(k_v)
        new_n.append(n_v)

    if not new_beta:
        raise RuntimeError(f"{category}/{model}: all E_new items dropped (saturated/missing)")

    new_beta = np.asarray(new_beta, dtype=np.float64)
    new_k = np.asarray(new_k, dtype=np.float64)
    new_n = np.asarray(new_n, dtype=np.float64)

    # --- V-model on full anchor (existing) + Step-1 replay for CAT-used subset ---
    anchor_lookup = load_response_matrix(response_matrix_path_for(category, model, filter_rule))
    anchor_beta = np.array([it.beta for it in bundle.items], dtype=np.float64)
    anchor_k = np.array([anchor_lookup[it.item_id][0] for it in bundle.items], dtype=np.float64)
    anchor_n = np.array([anchor_lookup[it.item_id][1] for it in bundle.items], dtype=np.float64)

    sim = simulate(bundle, anchor_lookup, T2C_CONFIG)
    used = sim["used"]
    pre_step2_adaptive = sim["trajectory"][-1]["theta_hat"]
    theta_full_target = sim["theta_full"]

    est = EAPEstimator(prior_mu=0.0, prior_sigma=PRIOR_SIGMA, n_quad=21)

    # 3a: full anchor ∪ E_new
    theta_aug_full, se_aug_full = est.estimate(
        np.concatenate([anchor_k, new_k]),
        np.concatenate([anchor_n, new_n]),
        np.concatenate([anchor_beta, new_beta]),
    )
    # 3b: CAT-used anchor subset ∪ E_new
    theta_aug_cat, se_aug_cat = est.estimate(
        np.concatenate([anchor_k[used], new_k]),
        np.concatenate([anchor_n[used], new_n]),
        np.concatenate([anchor_beta[used], new_beta]),
    )

    return {
        "category": category,
        "model": model,
        "panel_min": float(sim["panel_min"]),
        "panel_max": float(sim["panel_max"]),
        "theta_full_target": float(theta_full_target),
        "pre_step2_adaptive": float(pre_step2_adaptive),
        "n_new_total": len(new_ids),
        "n_new_survived": int(new_beta.size),
        "n_dropped": n_dropped,
        "beta_new_median": float(np.median(new_beta)),
        "beta_new_min": float(new_beta.min()),
        "beta_new_max": float(new_beta.max()),
        "theta_aug_full": float(theta_aug_full),
        "se_aug_full": float(se_aug_full),
        "theta_aug_cat": float(theta_aug_cat),
        "se_aug_cat": float(se_aug_cat),
        "n_cat_used": int(used.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filter-rule", choices=["avg", "alo"], default="alo")
    parser.add_argument("--matrix-root", default=str(PROJECT_ROOT / "results"),
                        help="Root holding <cat>/claim3_step2/<model>/matrix/ dirs.")
    parser.add_argument("--cells", nargs="*", default=None,
                        help="Override cells as <category>:<model> tokens; "
                             "default = the 7 locked OOP cells.")
    parser.add_argument("--output", default=str(
        PROJECT_ROOT / "results" / "adaptive_alo" / "claim3_step2_calibrated.csv"))
    args = parser.parse_args()

    if args.cells:
        cells = [tuple(c.split(":", 1)) for c in args.cells]
    else:
        cells = DEFAULT_CELLS

    matrix_root = Path(args.matrix_root)
    rows = []
    for cat, model in cells:
        try:
            row = calibrate_cell(cat, model, matrix_root, args.filter_rule)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  [skip] {cat}/{model}: {e}")
            continue
        rows.append(row)
        print(
            f"  {cat:24s} {model:34s} "
            f"target={row['theta_full_target']:+.3f} "
            f"pre={row['pre_step2_adaptive']:+.3f} "
            f"aug_full={row['theta_aug_full']:+.3f}(SE {row['se_aug_full']:.3f}) "
            f"aug_cat={row['theta_aug_cat']:+.3f} "
            f"J_new={row['n_new_survived']}/{row['n_new_total']} "
            f"β̃={row['beta_new_median']:+.2f}"
        )

    if not rows:
        print("No cells calibrated (missing inputs?).")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
