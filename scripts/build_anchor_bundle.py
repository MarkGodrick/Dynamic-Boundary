"""Build the per-category anchor bundle consumed by adaptive evaluation.

Joins three sources:
  1. {response_dir}/<model>/response_matrix.jsonl   — k, N cells (panel × items)
  2. {pool_path}                                    — prompt_id → (prompt, metadata, ...)
  3. Filter rule:
       avg  — panel-mean p̂ ∈ [lo, hi]                          (default lo=0.05, hi=0.95)
       alo  — at-least-one panel member with p̂ ∈ [lo, hi]      (default lo=0.20, hi=0.80)

Refits a 1PL Rasch model on the kept subset (OLS init + L-BFGS-B MLE) to obtain
β̂ on the same logit scale as the panel θ̂; writes a single JSON bundle.

Usage:
    python scripts/build_anchor_bundle.py \\
        --category harmful_refusal \\
        --response-dir results/harmful_refusal/filter_ablation_raw_v2_judgefix \\
        --pool benchmarks/harmful_refusal/filter_ablation_pool_k500.jsonl \\
        --filter-rule alo \\
        --output benchmarks/harmful_refusal/anchor_alo_bundle_M9.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from difficulty.rasch import load_response_matrix, fit_ols, fit_mle
from adaptive.anchor_bundle import AnchorBundle, AnchorItem, save_anchor_bundle


_DEFAULT_BANDS = {
    "avg": (0.05, 0.95),
    "alo": (0.20, 0.80),
}


def apply_filter(rule: str, p_hat: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if rule == "avg":
        panel_mean = p_hat.mean(axis=0)
        return (panel_mean >= lo) & (panel_mean <= hi)
    if rule == "alo":
        in_band = (p_hat >= lo) & (p_hat <= hi)
        return in_band.any(axis=0)
    raise ValueError(f"Unknown filter rule: {rule!r}")


def load_pool(pool_path: str) -> dict[str, dict]:
    """item_id -> {prompt, metadata, ...} index."""
    pool = {}
    with open(pool_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            pool[entry["prompt_id"]] = entry
    return pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--response-dir", required=True,
                        help="Directory with per-model response_matrix.jsonl")
    parser.add_argument("--pool", required=True,
                        help="Path to the K-pool jsonl with prompt_id → prompt+metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--filter-rule", choices=["avg", "alo"], default="avg",
                        help=("Filter rule: 'avg' = panel-mean p̂ in [lo, hi]; "
                              "'alo' = at-least-one panel member with p̂ in [lo, hi]. "
                              "Default 'avg' preserves backward-compatible behavior."))
    parser.add_argument("--filter-lo", type=float, default=None,
                        help="Low band edge; defaults: avg→0.05, alo→0.20")
    parser.add_argument("--filter-hi", type=float, default=None,
                        help="High band edge; defaults: avg→0.95, alo→0.80")
    args = parser.parse_args()

    default_lo, default_hi = _DEFAULT_BANDS[args.filter_rule]
    lo = default_lo if args.filter_lo is None else args.filter_lo
    hi = default_hi if args.filter_hi is None else args.filter_hi

    print(f"[1/4] Loading response matrix from {args.response_dir}...", flush=True)
    rm = load_response_matrix(args.response_dir)
    M, J = rm["pass_counts"].shape
    print(f"      M={M} models, J={J} items, N_median={int(np.median(rm['total_trials']))}")

    rule_label = args.filter_rule.upper()
    print(f"[2/4] Applying {rule_label}[{lo}, {hi}] filter...", flush=True)
    keep_mask = apply_filter(args.filter_rule, rm["p_hat"], lo, hi)
    n_kept = int(keep_mask.sum())
    print(f"      kept {n_kept}/{J} items")

    K_kept = rm["pass_counts"][:, keep_mask]
    N_kept = rm["total_trials"][:, keep_mask]
    p_kept = rm["p_hat"][:, keep_mask]

    print(f"[3/4] Fitting 1PL Rasch on kept subset (OLS init + L-BFGS-B MLE)...", flush=True)
    theta_ols, beta_ols = fit_ols(p_kept)
    theta_mle, beta_mle, nll = fit_mle(K_kept, N_kept, theta_ols, beta_ols)
    print(f"      θ̂ range: [{theta_mle.min():+.3f}, {theta_mle.max():+.3f}]")
    print(f"      β̂ range: [{beta_mle.min():+.3f}, {beta_mle.max():+.3f}]")
    print(f"      NLL = {nll:.2f}")

    print(f"[4/4] Joining with pool from {args.pool}...", flush=True)
    pool = load_pool(args.pool)
    kept_item_ids = [iid for j, iid in enumerate(rm["items"]) if keep_mask[j]]
    missing = [iid for iid in kept_item_ids if iid not in pool]
    if missing:
        raise ValueError(f"{len(missing)} kept items missing from pool: {missing[:5]}...")

    items = []
    for j, iid in enumerate(kept_item_ids):
        entry = pool[iid]
        items.append(AnchorItem(
            item_id=iid,
            prompt=entry["prompt"],
            metadata=entry.get("metadata", {}),
            beta=float(beta_mle[j]),
        ))

    panel_theta = {m: float(t) for m, t in zip(rm["models"], theta_mle)}
    bundle = AnchorBundle(
        category=args.category,
        panel_models=list(rm["models"]),
        panel_theta=panel_theta,
        items=items,
        filter_spec=f"{rule_label}[{lo:.2f}, {hi:.2f}]",
        M=M,
        N=int(np.median(rm["total_trials"])),
    )
    save_anchor_bundle(bundle, args.output)
    print(f"      wrote {args.output}  (J={bundle.J}, M={bundle.M}, N={bundle.N})")
    print()
    print("Panel θ̂ (sorted):")
    for m, t in sorted(panel_theta.items(), key=lambda kv: -kv[1]):
        print(f"  {t:+.3f}  {m}")


if __name__ == "__main__":
    main()
