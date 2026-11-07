#!/usr/bin/env python3
"""Per-cell Claim 3 table: rows = 5 validation models, cols = HR/OR/CIF/SR.

For each (model, category) cell we report:
  theta_full   : EAP on full anchor set with frozen panel β̂
  abs_delta    : |theta_adaptive - theta_full|
  se_full      : posterior SE of theta_full
  in_or_out    : IN / OUT of panel θ̂ range

Falls back to .bak_20260510_<model>/response_matrix.jsonl when the canonical
path is absent (the bak files are the response matrices used to produce the
current paper table).
"""

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from adaptive.posterior import EAPEstimator
from adaptive.anchor_bundle import load_anchor_bundle


CATEGORIES = [
    ("harmful_refusal", "HR"),
    ("over_refusal", "OR"),
    ("constrained_completion", "CIF"),
    ("sycophancy_resistance", "SR"),
]

VALIDATION_MODELS = [
    ("allenai_Olmo_3_7B_Instruct",       "Olmo-3-7B-Instruct"),
    ("ibm_granite_granite_4.1_8b",       "granite-4.1-8b"),
    ("microsoft_phi_4",                  "phi-4"),
    ("microsoft_Phi_4_reasoning_plus",   "Phi-4-reasoning-plus"),
    ("tiiuae_Falcon_H1R_7B",             "Falcon-H1R-7B"),
]


def find_response_matrix(category: str, model_key: str) -> Path | None:
    base = ROOT / "results" / category / "anchor_set"
    cur = base / model_key / "response_matrix.jsonl"
    if cur.exists():
        return cur
    bak = base / f".bak_20260510_{model_key}" / "response_matrix.jsonl"
    if bak.exists():
        return bak
    return None


def rerun_status(category: str, model_key: str) -> str:
    """Reports rerun (longer max_model_len / max_token) status per cell.

    - DONE:    CUR response_matrix exists (latest run with extended context).
    - PENDING: only BAK exists (stale run; rerun not yet completed).
    - N/A:     model never needed rerun (no BAK present).
    """
    base = ROOT / "results" / category / "anchor_set"
    cur_exists = (base / model_key / "response_matrix.jsonl").exists()
    bak_exists = (base / f".bak_20260510_{model_key}" / "response_matrix.jsonl").exists()
    if cur_exists and bak_exists:
        return "DONE"
    if cur_exists and not bak_exists:
        return "N/A"
    if not cur_exists and bak_exists:
        return "PENDING"
    return "MISSING"


def find_adaptive_result(category: str, model_key: str) -> Path | None:
    base = ROOT / "results" / "adaptive" / category
    matches = sorted(base.glob(f"{model_key}_{category}_eap_*/result.json"))
    return matches[-1] if matches else None


def load_response_matrix(path: Path):
    ids, k, n = [], [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["prompt_id"])
            k.append(row["pass_count"])
            n.append(row["total_trials"])
    return ids, np.array(k), np.array(n)


def compute_theta_full(response_path: Path, bundle_path: Path):
    bundle = load_anchor_bundle(bundle_path)
    beta_dict = {item.item_id: item.beta for item in bundle.items}
    ids, k, n = load_response_matrix(response_path)
    beta = np.array([beta_dict[i] for i in ids])
    est = EAPEstimator(prior_mu=0.0, prior_sigma=1.5, n_quad=21)
    theta, se = est.estimate(k, n, beta)
    return float(theta), float(se)


def main():
    cells = {}
    panel_ranges = {}
    for cat_key, _ in CATEGORIES:
        bundle_path = ROOT / "benchmarks" / cat_key / "anchor_bundle_M9.json"
        for model_key, _ in VALIDATION_MODELS:
            rm_path = find_response_matrix(cat_key, model_key)
            adp_path = find_adaptive_result(cat_key, model_key)
            cell = {"resp": rm_path, "adp": adp_path}
            if rm_path is None:
                cells[(cat_key, model_key)] = {"status": "no_response_matrix"}
                continue
            theta_full, se_full = compute_theta_full(rm_path, bundle_path)

            theta_adp = None
            panel_theta = None
            if adp_path is not None:
                with open(adp_path) as f:
                    adp = json.load(f)
                theta_adp = adp["theta_hat_final"]
                panel_theta = adp["panel_theta"]
                panel_ranges[cat_key] = (
                    min(panel_theta.values()),
                    max(panel_theta.values()),
                )

            in_panel = None
            if panel_theta is not None:
                pmin, pmax = panel_ranges[cat_key]
                in_panel = (theta_full >= pmin) and (theta_full <= pmax)

            cells[(cat_key, model_key)] = {
                "status": "ok",
                "theta_full": theta_full,
                "se_full": se_full,
                "theta_adp": theta_adp,
                "abs_delta": abs(theta_adp - theta_full) if theta_adp is not None else None,
                "in_panel": in_panel,
                "resp_src": "BAK" if ".bak_" in str(rm_path) else "CUR",
                "rerun": rerun_status(cat_key, model_key),
            }

    # Print cell table: rows = models, cols = HR/OR/CIF/SR
    print("\n=== Cell table: θ̂_full | |Δθ̂| | SE | IN/OUT ===\n")
    print(f"Panel θ̂ ranges:")
    for cat_key, cat_label in CATEGORIES:
        if cat_key in panel_ranges:
            lo, hi = panel_ranges[cat_key]
            print(f"  {cat_label}: [{lo:+.2f}, {hi:+.2f}]")

    header = f"\n{'Model':<22}"
    for _, cat_label in CATEGORIES:
        header += f" | {cat_label:<26}"
    print(header)
    print("-" * len(header))

    for model_key, model_label in VALIDATION_MODELS:
        line = f"{model_label:<22}"
        for cat_key, _ in CATEGORIES:
            c = cells[(cat_key, model_key)]
            if c["status"] != "ok":
                line += f" | {'--':<26}"
            else:
                io = "OUT" if c["in_panel"] is False else ("IN " if c["in_panel"] else "?  ")
                d = c["abs_delta"]
                d_str = f"{d:.2f}" if d is not None else " - "
                line += f" | θ={c['theta_full']:+5.2f} Δ={d_str} SE={c['se_full']:.2f} {io}"
        print(line)

    print("\n=== Compact cell table: |Δθ̂| (SE_full) ===\n")
    header = f"\n{'Model':<22}"
    for _, cat_label in CATEGORIES:
        header += f" | {cat_label:<14}"
    print(header)
    print("-" * len(header))
    for model_key, model_label in VALIDATION_MODELS:
        line = f"{model_label:<22}"
        for cat_key, _ in CATEGORIES:
            c = cells[(cat_key, model_key)]
            if c["status"] != "ok":
                line += f" | {'--':<14}"
            else:
                d = c["abs_delta"]
                io = "*" if c["in_panel"] is False else ""
                d_str = f"{d:.2f}" if d is not None else "  -"
                line += f" | {d_str}({c['se_full']:.2f}){io:<3}".ljust(17)
        print(line)
    print("\n* = OUT of panel θ̂ range")

    # Rerun status grid: which cells have been re-evaluated with longer
    # max_model_len / max_token, which are still pending.
    print("\n=== Rerun status (longer max_model_len / max_token) ===\n")
    header = f"{'Model':<22}"
    for _, cat_label in CATEGORIES:
        header += f" | {cat_label:<8}"
    print(header)
    print("-" * len(header))
    pending = []
    for model_key, model_label in VALIDATION_MODELS:
        line = f"{model_label:<22}"
        for cat_key, cat_label in CATEGORIES:
            c = cells[(cat_key, model_key)]
            tag = c.get("rerun", "?") if c["status"] == "ok" else "MISSING"
            line += f" | {tag:<8}"
            if tag == "PENDING":
                pending.append(f"{cat_label} x {model_label}")
        print(line)
    print("\nLegend: DONE = rerun completed (CUR exists); "
          "PENDING = only BAK present, rerun not yet done; "
          "N/A = model never needed rerun.")
    if pending:
        print("\nPending cells:")
        for p in pending:
            print(f"  - {p}")
    else:
        print("\nAll validation cells have been re-run.")

    # Save CSV
    out = ROOT / "results" / "adaptive" / "claim3_cell_table.csv"
    with open(out, "w") as f:
        f.write("model,category,theta_full,se_full,theta_adp,abs_delta,in_panel,resp_src,rerun\n")
        for model_key, model_label in VALIDATION_MODELS:
            for cat_key, cat_label in CATEGORIES:
                c = cells[(cat_key, model_key)]
                if c["status"] != "ok":
                    f.write(f"{model_label},{cat_label},,,,,,,MISSING\n")
                else:
                    f.write(f"{model_label},{cat_label},{c['theta_full']:.4f},{c['se_full']:.4f},"
                            f"{c['theta_adp']:.4f},{c['abs_delta']:.4f},"
                            f"{'IN' if c['in_panel'] else 'OUT'},{c['resp_src']},{c['rerun']}\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
