"""Phase 0 §1.2 sanity check: AVG vs ALO β̂ structural divergence on overlap items.

For each (category, V-model) pair where the V-model has AVG-era anchor responses
(`results/{cat}/anchor_set/<v_model>/response_matrix.jsonl`):

  1. Identify overlap items = items present in BOTH AVG and ALO anchor bundles.
  2. Restrict V-model's response matrix to overlap items.
  3. Fit θ̂_v on the overlap subset twice:
       - β̂ from AVG bundle  (anchor_bundle_M9.json)
       - β̂ from ALO bundle  (anchor_alo_bundle_M9.json)
  4. Report Δθ̂ = θ̂(ALO-β̂) − θ̂(AVG-β̂) plus per-source Brier / RMSE / log-loss
     for held-out generalization (predict p_obs on overlap with each β̂).

Decision gate: if max |Δθ̂| < 0.1 logit across all (cat, V-model) rows, ALO and
AVG β̂ structurally agree on the overlap and Phase 1 can proceed.

Usage:
    conda run -n boundary --no-capture-output python -u \\
        scripts/analysis/claim3_overlap_sanity.py \\
        --output results/sanity/claim3_overlap_alo_vs_avg.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from adaptive.anchor_bundle import load_anchor_bundle
from adaptive.posterior import EAPEstimator


CATEGORIES = [
    "harmful_refusal",
    "over_refusal",
    "constrained_completion",
    "sycophancy_resistance",
]

V_MODELS = [
    # Primary 5
    "allenai_Olmo_3_7B_Instruct",
    "ibm_granite_granite_4.1_8b",
    "microsoft_phi_4",
    "microsoft_Phi_4_reasoning_plus",
    "tiiuae_Falcon_H1R_7B",
    # Secondary 3
    "allenai_Olmo_3.1_32B_Instruct",
    "deepseek_ai_DeepSeek_R1_Distill_Llama_70B",
    "deepseek_ai_DeepSeek_R1_Distill_Qwen_32B",
]

PRIMARY_SET = set(V_MODELS[:5])
DRIFT_THRESHOLD = 0.1


def load_response_matrix(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids, k, n = [], [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["prompt_id"])
            k.append(row["pass_count"])
            n.append(row["total_trials"])
    return ids, np.array(k), np.array(n)


def metrics(theta: float, k: np.ndarray, n: np.ndarray,
            beta: np.ndarray) -> dict:
    p_pred = 1.0 / (1.0 + np.exp(-(theta - beta)))
    p_obs = k / n
    brier = float(np.mean((p_pred - p_obs) ** 2))
    p_clip = np.clip(p_pred, 1e-12, 1 - 1e-12)
    log_loss = float(np.mean(
        -(k * np.log(p_clip) + (n - k) * np.log(1 - p_clip)) / n))
    return {"brier": brier, "rmse": float(np.sqrt(brier)),
            "log_loss": log_loss}


def analyse(cat: str, v_model: str, estimator: EAPEstimator) -> dict:
    resp_path = Path(f"results/{cat}/anchor_set/{v_model}/response_matrix.jsonl")
    avg_path = Path(f"benchmarks/{cat}/anchor_bundle_M9.json")
    alo_path = Path(f"benchmarks/{cat}/anchor_alo_bundle_M9.json")

    for p in (resp_path, avg_path, alo_path):
        if not p.exists():
            return {"status": "missing", "missing": str(p)}

    avg_bundle = load_anchor_bundle(avg_path)
    alo_bundle = load_anchor_bundle(alo_path)
    beta_avg = {it.item_id: it.beta for it in avg_bundle.items}
    beta_alo = {it.item_id: it.beta for it in alo_bundle.items}

    overlap_ids = set(beta_avg) & set(beta_alo)
    item_ids, k_all, n_all = load_response_matrix(resp_path)
    mask = np.array([iid in overlap_ids for iid in item_ids])
    if mask.sum() == 0:
        return {"status": "empty_overlap"}

    kept_ids = [iid for iid, m in zip(item_ids, mask) if m]
    k = k_all[mask]
    n = n_all[mask]
    b_avg = np.array([beta_avg[i] for i in kept_ids])
    b_alo = np.array([beta_alo[i] for i in kept_ids])

    theta_avg, se_avg = estimator.estimate(k, n, b_avg)
    theta_alo, se_alo = estimator.estimate(k, n, b_alo)

    return {
        "status": "ok",
        "n_overlap": int(mask.sum()),
        "n_responses": int(len(item_ids)),
        "n_in_avg_bundle": len(beta_avg),
        "n_in_alo_bundle": len(beta_alo),
        "theta_avg": float(theta_avg),
        "se_avg": float(se_avg),
        "theta_alo": float(theta_alo),
        "se_alo": float(se_alo),
        "delta_theta": float(theta_alo - theta_avg),
        "metrics_avg": metrics(theta_avg, k, n, b_avg),
        "metrics_alo": metrics(theta_alo, k, n, b_alo),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/sanity/claim3_overlap_alo_vs_avg.json")
    args = ap.parse_args()

    estimator = EAPEstimator(prior_mu=0.0, prior_sigma=1.5, n_quad=21)

    rows: list[dict] = []
    print(f"{'cat':<24} {'V-model':<48} {'tier':<4} {'n_ov':>5} "
          f"{'θ̂_AVG':>8} {'θ̂_ALO':>8} {'Δθ̂':>8} {'flag':>5}")
    print("-" * 120)

    for cat in CATEGORIES:
        for v in V_MODELS:
            tier = "P" if v in PRIMARY_SET else "S"
            res = analyse(cat, v, estimator)
            row = {"category": cat, "v_model": v, "tier": tier, **res}
            rows.append(row)
            if res["status"] != "ok":
                print(f"{cat:<24} {v:<48} {tier:<4} ! status={res['status']}"
                      + (f" ({res.get('missing','')})" if res.get('missing') else ""))
                continue
            dtheta = res["delta_theta"]
            flag = "FAIL" if abs(dtheta) >= DRIFT_THRESHOLD else "ok"
            print(f"{cat:<24} {v:<48} {tier:<4} {res['n_overlap']:>5} "
                  f"{res['theta_avg']:+8.3f} {res['theta_alo']:+8.3f} "
                  f"{dtheta:+8.3f} {flag:>5}")

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    dthetas = np.array([abs(r["delta_theta"]) for r in ok_rows])
    n_fail = int((dthetas >= DRIFT_THRESHOLD).sum())

    print()
    print(f"Summary: {len(ok_rows)} rows analysed; "
          f"max |Δθ̂| = {dthetas.max():.4f} logit; "
          f"median |Δθ̂| = {np.median(dthetas):.4f} logit; "
          f"{n_fail} rows ≥ {DRIFT_THRESHOLD} logit")
    print("DECISION:",
          "PROCEED with Phase 1" if n_fail == 0 else
          "PAUSE — structural divergence detected; investigate before Phase 1")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "drift_threshold": DRIFT_THRESHOLD,
        "n_rows": len(ok_rows),
        "max_abs_delta_theta": float(dthetas.max()),
        "median_abs_delta_theta": float(np.median(dthetas)),
        "n_above_threshold": n_fail,
        "decision": "proceed" if n_fail == 0 else "pause",
        "rows": rows,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
