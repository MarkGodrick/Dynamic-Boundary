"""Score-based resolution metrics on tier vs anchor-subsample.

Metrics (all bounded, score-based, no IRT artifact):
  - Res_Var:    mean_x var_m(p_smooth_mx)              in [0, 0.25]
  - Res_JSD:    mean_x [H(p_bar_x) - mean_m H(p_smooth_mx)] / log M   in [0, 1]
  - Res_PR:     #{pairs (i,j) : 95% CI of s_i, s_j non-overlapping} / C(M,2)
  - saturation: mean of indicator{p_hat in {0,1}} over (m,x) cells

Where:
  p_hat   = K / N               raw pass rate
  p_smooth = (K+1) / (N+2)      Laplace add-1 smoothing (avoids log(0))
  s_i     = mean_x p_hat_ix     model's overall pass rate on the benchmark
  Var(s_i) = (1/J^2) sum_x p_hat_ix (1-p_hat_ix) / N

Same data flow as scripts/analysis/tier_vs_anchor_subsample.py:
  tier   = validation_benchmark_response (split by prompt_id prefix)
  anchor = filter_ablation pool (apply AVG[0.05, 0.95] filter, then subsample)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


CATEGORY_CONFIG = {
    "harmful_refusal": {
        "tier_dir": "results/harmful_refusal/validation/validation_benchmark_response",
        "anchor_dir": "results/harmful_refusal/filter_ablation_raw_v2_judgefix",
        "tier_prefixes": ["val_A_easy", "val_A_medium", "val_A_hard"],
    },
    "constrained_completion": {
        "tier_dir": "results/constrained_completion/validation/validation_benchmark_response",
        "anchor_dir": "results/constrained_completion/filter_ablation_raw_v2",
        "tier_prefixes": ["val_B_ifbench", "val_B_manyifeval"],
    },
}

FILTER_LO, FILTER_HI = 0.05, 0.95
Z = 1.96


# -------------------------------------------------------------------
# Loading (duplicated from tier_vs_anchor_subsample.py per project rule:
#  no cross-script imports; if reused again, hoist to a core module.)
# -------------------------------------------------------------------

def load_response_matrix_dict(data_dir: Path) -> dict:
    raw = {}
    models = []
    for model_subdir in sorted(data_dir.iterdir()):
        if not model_subdir.is_dir():
            continue
        rm_path = model_subdir / "response_matrix.jsonl"
        if not rm_path.exists():
            continue
        meta_path = model_subdir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                model_name = json.load(f).get("model", model_subdir.name)
        else:
            model_name = model_subdir.name
        models.append(model_name)
        with open(rm_path) as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                raw[(model_name, e["prompt_id"])] = (e["pass_count"], e["total_trials"])
    return {"raw": raw, "models": sorted(models)}


def build_kn_matrix(raw, models, items):
    M, J = len(models), len(items)
    K = np.zeros((M, J), dtype=np.float64)
    N = np.zeros((M, J), dtype=np.float64)
    for i, m in enumerate(models):
        for j, p in enumerate(items):
            k, n = raw.get((m, p), (0, 0))
            K[i, j] = k
            N[i, j] = n
    return K, N


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------

def H_bernoulli(p: np.ndarray) -> np.ndarray:
    """Binary entropy in nats. Clipped for numerical safety."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def compute_metrics(K: np.ndarray, N: np.ndarray) -> dict:
    """K, N: (M, J) arrays. Return all 4 metrics."""
    M, J = K.shape
    N_safe = np.maximum(N, 1)
    p_hat = K / N_safe
    p_smooth = (K + 1) / (N + 2)

    # Res_Var
    res_var = float(p_smooth.var(axis=0).mean())

    # Res_JSD = mean_x JSD_panel(p_x) / log M
    p_bar = p_smooth.mean(axis=0)
    H_pbar = H_bernoulli(p_bar)
    H_each = H_bernoulli(p_smooth).mean(axis=0)
    jsd_per_item = H_pbar - H_each
    res_jsd = float(jsd_per_item.mean() / np.log(M))

    # Res_PR: per-pair Wald CI (Gaussian) on s_i
    s_i = p_hat.mean(axis=1)
    var_s = (p_hat * (1 - p_hat) / N_safe).sum(axis=1) / (J ** 2)
    se_s = np.sqrt(var_s)
    lo = s_i - Z * se_s
    hi = s_i + Z * se_s
    n_pairs = M * (M - 1) // 2
    n_distinct = 0
    for i in range(M):
        for j in range(i + 1, M):
            if hi[i] < lo[j] or hi[j] < lo[i]:
                n_distinct += 1
    res_pr = n_distinct / n_pairs

    # Saturation
    saturation = float(((p_hat == 0) | (p_hat == 1)).mean())

    return {
        "res_var": res_var,
        "res_jsd": res_jsd,
        "res_pr": res_pr,
        "saturation": saturation,
    }


def percentile_rank(value: float, dist: np.ndarray) -> float:
    return float((dist <= value).mean() * 100)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, choices=list(CATEGORY_CONFIG.keys()))
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = CATEGORY_CONFIG[args.category]
    tier_dir = Path(cfg["tier_dir"])
    anchor_dir = Path(cfg["anchor_dir"])
    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*84}")
    print(f"  CATEGORY: {args.category}")
    print(f"{'='*84}")

    # ----- Tier data -----
    tier_data = load_response_matrix_dict(tier_dir)
    tier_raw = tier_data["raw"]
    tier_models = tier_data["models"]
    print(f"Tier dir: {tier_dir}")
    print(f"  panel size: {len(tier_models)}")

    all_tier_items = sorted({pid for (_, pid) in tier_raw.keys()})
    items_by_tier = {}
    for prefix in cfg["tier_prefixes"]:
        items_by_tier[prefix] = sorted(p for p in all_tier_items if p.startswith(prefix + "_"))
        print(f"  {prefix}: {len(items_by_tier[prefix])}")

    # ----- Per-tier metrics -----
    tier_results = {}
    print(f"\n{'-'*84}")
    print(f"  Per-tier metrics")
    print(f"{'-'*84}")
    print(f"  {'tier':<22} {'n_items':>8}  {'Res_Var':>9} {'Res_JSD':>9} "
          f"{'Res_PR':>9} {'saturation':>11}")
    for prefix, items in items_by_tier.items():
        K, N = build_kn_matrix(tier_raw, tier_models, items)
        m = compute_metrics(K, N)
        tier_results[prefix] = {**m, "n_items": len(items)}
        print(f"  {prefix:<22} {len(items):>8d}  {m['res_var']:>9.4f} {m['res_jsd']:>9.4f} "
              f"{m['res_pr']:>9.4f} {m['saturation']:>11.4f}")

    # ----- Anchor pool, AVG filter -----
    print(f"\nAnchor dir: {anchor_dir}")
    anchor_data = load_response_matrix_dict(anchor_dir)
    anchor_raw = anchor_data["raw"]
    anchor_models = anchor_data["models"]
    if anchor_models != tier_models:
        print(f"  WARN: panel mismatch.\n   tier:   {tier_models}\n   anchor: {anchor_models}")

    anchor_pool_items = sorted({pid for (_, pid) in anchor_raw.keys()})
    K_pool, N_pool = build_kn_matrix(anchor_raw, anchor_models, anchor_pool_items)
    p_pool = K_pool / np.maximum(N_pool, 1)
    panel_mean = p_pool.mean(axis=0)
    keep_mask = (panel_mean >= FILTER_LO) & (panel_mean <= FILTER_HI)
    anchor_items = [anchor_pool_items[j] for j in range(len(anchor_pool_items)) if keep_mask[j]]
    print(f"  raw pool: {len(anchor_pool_items)} -> AVG[{FILTER_LO},{FILTER_HI}] anchor: {len(anchor_items)}")

    K_anchor, N_anchor = build_kn_matrix(anchor_raw, anchor_models, anchor_items)
    n_anchor = len(anchor_items)

    # ----- Anchor subsamples -----
    print(f"\n{'-'*84}")
    print(f"  Anchor subsamples (B={args.B}, seed={args.seed})")
    print(f"{'-'*84}")
    sizes = sorted({len(items) for items in items_by_tier.values()})
    subsample_dist = {}
    for n in sizes:
        if n > n_anchor:
            print(f"  skip n={n} (anchor only has {n_anchor})")
            continue
        arrs = {k: np.empty(args.B) for k in ["res_var", "res_jsd", "res_pr", "saturation"]}
        for b in range(args.B):
            idx = rng.choice(n_anchor, size=n, replace=False)
            mvals = compute_metrics(K_anchor[:, idx], N_anchor[:, idx])
            for k in arrs:
                arrs[k][b] = mvals[k]
        subsample_dist[n] = arrs
        print(f"  n={n}:")
        for k, v in arrs.items():
            print(f"    {k:<12} mean={v.mean():.4f}  std={v.std():.4f}  "
                  f"q05={np.percentile(v,5):.4f}  q50={np.percentile(v,50):.4f}  "
                  f"q95={np.percentile(v,95):.4f}")

    # ----- Tier vs subsample comparison -----
    print(f"\n{'='*84}")
    print(f"  TIER vs ANCHOR SUBSAMPLE (matched n_items)")
    print(f"{'='*84}")
    print(f"  {'tier':<22} {'metric':<14} {'tier_val':>10} {'anchor_q50':>11} "
          f"{'anchor_q05':>11} {'anchor_q95':>11} {'pct_rank':>9}")
    for prefix, tres in tier_results.items():
        n = tres["n_items"]
        if n not in subsample_dist:
            continue
        for key in ["res_var", "res_jsd", "res_pr", "saturation"]:
            d = subsample_dist[n][key]
            q05, q50, q95 = np.percentile(d, [5, 50, 95])
            pr = percentile_rank(tres[key], d)
            print(f"  {prefix:<22} {key:<14} {tres[key]:>10.4f} {q50:>11.4f} "
                  f"{q05:>11.4f} {q95:>11.4f} {pr:>8.1f}%")
        print()

    # ----- Save -----
    out = {
        "category": args.category,
        "filter": {"lo": FILTER_LO, "hi": FILTER_HI},
        "B": args.B,
        "seed": args.seed,
        "n_anchor_pool": len(anchor_pool_items),
        "n_anchor_filtered": len(anchor_items),
        "panel_models": anchor_models,
        "tier_results": tier_results,
        "anchor_subsample_dist": {
            str(n): {
                key: {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "q05": float(np.percentile(arr, 5)),
                    "q50": float(np.percentile(arr, 50)),
                    "q95": float(np.percentile(arr, 95)),
                    "values": arr.tolist(),
                }
                for key, arr in d.items()
            }
            for n, d in subsample_dist.items()
        },
    }
    out_path = args.output or f"/tmp/score_metrics_{args.category}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
