"""Per-tier Rasch fit vs matched-size anchor subsamples.

Methodology (Claim 1 resolution check):
  1. Split each category's validation benchmark by prompt source (tier).
  2. For each tier: fit Rasch on the (model x item) (K, N) matrix; report
     theta_range = max(theta_hat) - min(theta_hat) and beta_range similarly.
  3. From the same category's anchor pool (random pool, K=500), apply the
     locked AVG[0.05, 0.95] filter to obtain the anchor set, then draw B
     subsamples of size n_match (= tier size). Fit Rasch per subsample,
     record (theta_range, beta_range) distribution.
  4. Report each tier's metrics with their percentile rank within the anchor
     subsample distribution.

Locked config (project memory):
  - filter: AVG p_hat in [0.05, 0.95]
  - 9-model panel
  - K=500 random pool

Usage:
  python scripts/analysis/tier_vs_anchor_subsample.py \
      --category harmful_refusal --B 500 --seed 0
  python scripts/analysis/tier_vs_anchor_subsample.py \
      --category constrained_completion --B 500 --seed 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difficulty.rasch import fit_ols, fit_mle


# -------------------------------------------------------------------
# Per-category config
# -------------------------------------------------------------------

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


# -------------------------------------------------------------------
# Loading
# -------------------------------------------------------------------

def load_response_matrix_dict(data_dir: Path) -> dict:
    """Load (model, item) -> (K, N) for all model subdirs."""
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


def build_kn_matrix(raw: dict, models: list, items: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (K, N, p_hat) with shape (M, J) for the given model + item lists."""
    M, J = len(models), len(items)
    K = np.zeros((M, J), dtype=np.float64)
    N = np.zeros((M, J), dtype=np.float64)
    for i, m in enumerate(models):
        for j, p in enumerate(items):
            k, n = raw.get((m, p), (0, 0))
            K[i, j] = k
            N[i, j] = n
    p_hat = np.where(N > 0, K / N, 0.5)
    return K, N, p_hat


# -------------------------------------------------------------------
# Rasch metrics
# -------------------------------------------------------------------

def rasch_metrics(K: np.ndarray, N: np.ndarray) -> dict:
    """Fit Rasch (OLS init + MLE) and return theta/beta ranges."""
    p_hat = np.where(N > 0, K / N, 0.5)
    theta0, beta0 = fit_ols(p_hat)
    theta_mle, beta_mle, nll = fit_mle(K, N, theta0, beta0)
    return {
        "theta": theta_mle,
        "beta": beta_mle,
        "theta_range": float(theta_mle.max() - theta_mle.min()),
        "beta_range": float(beta_mle.max() - beta_mle.min()),
        "theta_p5_separation": float(np.percentile(
            [abs(theta_mle[i] - theta_mle[j])
             for i in range(len(theta_mle)) for j in range(i+1, len(theta_mle))], 5)),
        "theta_med_separation": float(np.median(
            [abs(theta_mle[i] - theta_mle[j])
             for i in range(len(theta_mle)) for j in range(i+1, len(theta_mle))])),
        "saturation": float(((p_hat == 0) | (p_hat == 1)).mean()),
        "nll": float(nll),
    }


def percentile_rank(value: float, distribution: np.ndarray) -> float:
    """Fraction of distribution <= value, in percent."""
    return float((distribution <= value).mean() * 100)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, choices=list(CATEGORY_CONFIG.keys()))
    ap.add_argument("--B", type=int, default=500, help="Number of anchor subsamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = CATEGORY_CONFIG[args.category]
    tier_dir = Path(cfg["tier_dir"])
    anchor_dir = Path(cfg["anchor_dir"])
    rng = np.random.default_rng(args.seed)

    # ----- Load tier data -----
    print(f"\n{'='*72}")
    print(f"  CATEGORY: {args.category}")
    print(f"{'='*72}")
    tier_data = load_response_matrix_dict(tier_dir)
    tier_raw = tier_data["raw"]
    tier_models = tier_data["models"]
    print(f"Tier directory: {tier_dir}")
    print(f"  models: {len(tier_models)}")

    all_tier_items = sorted({pid for (_, pid) in tier_raw.keys()})
    items_by_tier = {}
    for prefix in cfg["tier_prefixes"]:
        items_by_tier[prefix] = sorted(p for p in all_tier_items if p.startswith(prefix + "_"))
        print(f"  {prefix}: {len(items_by_tier[prefix])} items")

    # ----- Per-tier Rasch fit -----
    tier_results = {}
    print(f"\n{'-'*72}")
    print(f"  Per-tier Rasch fit (M=9, MLE)")
    print(f"{'-'*72}")
    print(f"  {'tier':<22} {'n_items':>8} {'theta_range':>12} {'beta_range':>11} "
          f"{'theta_med_sep':>14} {'saturation':>11}")
    for prefix, items in items_by_tier.items():
        K, N, _ = build_kn_matrix(tier_raw, tier_models, items)
        m = rasch_metrics(K, N)
        tier_results[prefix] = {**m, "n_items": len(items)}
        print(f"  {prefix:<22} {len(items):>8d} {m['theta_range']:>12.3f} "
              f"{m['beta_range']:>11.3f} {m['theta_med_separation']:>14.3f} "
              f"{m['saturation']:>11.3f}")

    # ----- Load anchor pool, apply AVG filter -----
    print(f"\nAnchor directory: {anchor_dir}")
    anchor_data = load_response_matrix_dict(anchor_dir)
    anchor_raw = anchor_data["raw"]
    anchor_models = anchor_data["models"]

    # Sanity: same panel
    if anchor_models != tier_models:
        print(f"  WARN: anchor models != tier models")
        print(f"    tier:   {tier_models}")
        print(f"    anchor: {anchor_models}")
    anchor_pool_items = sorted({pid for (_, pid) in anchor_raw.keys()})
    print(f"  raw pool size: {len(anchor_pool_items)}")

    K_pool, N_pool, p_pool = build_kn_matrix(anchor_raw, anchor_models, anchor_pool_items)
    panel_mean = p_pool.mean(axis=0)
    keep_mask = (panel_mean >= FILTER_LO) & (panel_mean <= FILTER_HI)
    anchor_items = [anchor_pool_items[j] for j in range(len(anchor_pool_items)) if keep_mask[j]]
    print(f"  AVG filter [{FILTER_LO}, {FILTER_HI}] -> anchor size: {len(anchor_items)}")

    # ----- Anchor subsamples -----
    print(f"\n{'-'*72}")
    print(f"  Anchor subsamples: B={args.B}, seed={args.seed}")
    print(f"{'-'*72}")

    K_anchor, N_anchor, _ = build_kn_matrix(anchor_raw, anchor_models, anchor_items)
    n_anchor = len(anchor_items)

    # Per tier size, draw B subsamples
    sizes = sorted({len(items) for items in items_by_tier.values()})
    subsample_dist = {}  # n -> {theta_range: [...], beta_range: [...], ...}
    for n in sizes:
        if n > n_anchor:
            print(f"  skip n={n} (anchor only has {n_anchor} items)")
            continue
        theta_ranges = np.empty(args.B)
        beta_ranges = np.empty(args.B)
        theta_med_seps = np.empty(args.B)
        saturations = np.empty(args.B)
        for b in range(args.B):
            idx = rng.choice(n_anchor, size=n, replace=False)
            K_sub, N_sub = K_anchor[:, idx], N_anchor[:, idx]
            try:
                m = rasch_metrics(K_sub, N_sub)
                theta_ranges[b] = m["theta_range"]
                beta_ranges[b] = m["beta_range"]
                theta_med_seps[b] = m["theta_med_separation"]
                saturations[b] = m["saturation"]
            except Exception as ex:
                theta_ranges[b] = np.nan
                beta_ranges[b] = np.nan
                theta_med_seps[b] = np.nan
                saturations[b] = np.nan
        subsample_dist[n] = {
            "theta_range": theta_ranges,
            "beta_range": beta_ranges,
            "theta_med_separation": theta_med_seps,
            "saturation": saturations,
        }
        valid = ~np.isnan(theta_ranges)
        print(f"  n={n}: {valid.sum()}/{args.B} fits OK")
        for key in ["theta_range", "beta_range", "theta_med_separation", "saturation"]:
            d = subsample_dist[n][key][valid]
            print(f"    {key:<24} mean={d.mean():.3f}  std={d.std():.3f}  "
                  f"q05={np.percentile(d, 5):.3f}  q50={np.percentile(d, 50):.3f}  "
                  f"q95={np.percentile(d, 95):.3f}")

    # ----- Tier vs anchor subsample comparison -----
    print(f"\n{'='*72}")
    print(f"  TIER vs ANCHOR SUBSAMPLE (matched n_items)")
    print(f"{'='*72}")
    print(f"  {'tier':<22} {'metric':<22} {'tier_val':>10} {'anchor_q50':>11} "
          f"{'anchor_q05':>11} {'anchor_q95':>11} {'pct_rank':>9}")
    for prefix, tres in tier_results.items():
        n = tres["n_items"]
        if n not in subsample_dist:
            continue
        for key in ["theta_range", "beta_range", "theta_med_separation", "saturation"]:
            dist = subsample_dist[n][key]
            valid = ~np.isnan(dist)
            d = dist[valid]
            pr = percentile_rank(tres[key], d)
            q05, q50, q95 = np.percentile(d, [5, 50, 95])
            print(f"  {prefix:<22} {key:<22} {tres[key]:>10.3f} {q50:>11.3f} "
                  f"{q05:>11.3f} {q95:>11.3f} {pr:>8.1f}%")
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
        "tier_results": {
            prefix: {
                "n_items": tres["n_items"],
                "theta_range": tres["theta_range"],
                "beta_range": tres["beta_range"],
                "theta_med_separation": tres["theta_med_separation"],
                "theta_p5_separation": tres["theta_p5_separation"],
                "saturation": tres["saturation"],
                "theta": tres["theta"].tolist(),
                "beta_summary": {
                    "min": float(tres["beta"].min()),
                    "max": float(tres["beta"].max()),
                    "median": float(np.median(tres["beta"])),
                },
            }
            for prefix, tres in tier_results.items()
        },
        "anchor_subsample_dist": {
            str(n): {
                key: {
                    "mean": float(np.nanmean(arr)),
                    "std": float(np.nanstd(arr)),
                    "q05": float(np.nanpercentile(arr, 5)),
                    "q50": float(np.nanpercentile(arr, 50)),
                    "q95": float(np.nanpercentile(arr, 95)),
                    "values": arr.tolist(),
                }
                for key, arr in d.items()
            }
            for n, d in subsample_dist.items()
        },
    }
    out_path = args.output or f"/tmp/tier_vs_anchor_{args.category}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
