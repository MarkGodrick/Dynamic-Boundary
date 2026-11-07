"""Per-tier AVG filter test: anchor vs each filtered tier.

For each tier of the fixed benchmark, apply the same AVG[0.05, 0.95] filter
that was used to construct the anchor set, then compare anchor vs each
filtered tier head-to-head with bootstrap CIs.

Goal: quantify how much of anchor's metric advantage comes from the filter
(vs from the random-compose pool itself). If anchor still wins against
each filtered tier, the pool has intrinsic value; if filtered tiers match
or beat anchor, the filter is doing all the work.

Categories with tier structure:
  - harmful_refusal: val_A_easy / val_A_medium / val_A_hard (each 100)
  - constrained_completion: val_B_ifbench (25) / val_B_manyifeval (150)
  - over_refusal: single tier (skipped — same as mixed_vs_anchor)

Usage:
  python scripts/analysis/per_tier_filter_test.py --category harmful_refusal
  python scripts/analysis/per_tier_filter_test.py --category constrained_completion
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


def apply_avg_filter(raw, models, items):
    """Apply AVG[FILTER_LO, FILTER_HI] on panel-mean p_hat. Return surviving items."""
    K, N = build_kn_matrix(raw, models, items)
    p = K / np.maximum(N, 1)
    pm = p.mean(axis=0)
    keep = (pm >= FILTER_LO) & (pm <= FILTER_HI)
    return [items[j] for j in range(len(items)) if keep[j]]


def H_bernoulli(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def compute_metrics(K, N):
    M, J = K.shape
    if J == 0:
        return {k: float("nan") for k in ["res_var", "res_jsd", "res_pr", "saturation"]}
    N_safe = np.maximum(N, 1)
    p_hat = K / N_safe
    p_smooth = (K + 1) / (N + 2)
    res_var = float(p_smooth.var(axis=0).mean())
    p_bar = p_smooth.mean(axis=0)
    res_jsd = float((H_bernoulli(p_bar) - H_bernoulli(p_smooth).mean(axis=0)).mean() / np.log(M))
    s_i = p_hat.mean(axis=1)
    var_s = (p_hat * (1 - p_hat) / N_safe).sum(axis=1) / (J ** 2)
    se_s = np.sqrt(var_s)
    lo, hi = s_i - Z * se_s, s_i + Z * se_s
    n_pairs = M * (M - 1) // 2
    n_distinct = sum(
        1 for i in range(M) for j in range(i + 1, M)
        if hi[i] < lo[j] or hi[j] < lo[i]
    )
    res_pr = n_distinct / n_pairs
    saturation = float(((p_hat == 0) | (p_hat == 1)).mean())
    return {"res_var": res_var, "res_jsd": res_jsd, "res_pr": res_pr, "saturation": saturation}


def bootstrap_metrics(K, N, B, rng):
    J = K.shape[1]
    if J == 0:
        return {k: np.full(B, np.nan) for k in ["res_var", "res_jsd", "res_pr", "saturation"]}
    keys = ["res_var", "res_jsd", "res_pr", "saturation"]
    out = {k: np.empty(B) for k in keys}
    for b in range(B):
        idx = rng.integers(0, J, size=J)
        m = compute_metrics(K[:, idx], N[:, idx])
        for k in keys:
            out[k][b] = m[k]
    return out


def fmt_ci(x, lo, hi):
    return f"{x:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, choices=list(CATEGORY_CONFIG.keys()))
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = CATEGORY_CONFIG[args.category]
    print(f"\n{'='*92}")
    print(f"  CATEGORY: {args.category}  |  PER-TIER FILTER TEST  |  B={args.B}")
    print(f"{'='*92}")

    # Load tier data
    tier_data = load_response_matrix_dict(Path(cfg["tier_dir"]))
    tier_raw = tier_data["raw"]
    tier_models = tier_data["models"]
    all_tier_items = sorted({pid for (_, pid) in tier_raw.keys()})
    items_by_tier = {p: sorted(pid for pid in all_tier_items if pid.startswith(p + "_"))
                     for p in cfg["tier_prefixes"]}

    # Apply filter per tier
    items_by_tier_filt = {p: apply_avg_filter(tier_raw, tier_models, items)
                          for p, items in items_by_tier.items()}

    # Load anchor
    anchor_data = load_response_matrix_dict(Path(cfg["anchor_dir"]))
    anchor_raw = anchor_data["raw"]
    anchor_models = anchor_data["models"]
    anchor_pool_items = sorted({pid for (_, pid) in anchor_raw.keys()})
    anchor_items = apply_avg_filter(anchor_raw, anchor_models, anchor_pool_items)

    if anchor_models != tier_models:
        print(f"WARN: panel mismatch")

    # ----- Filter shrinkage report -----
    print(f"\n{'-'*92}")
    print(f"  Filter shrinkage  (AVG[{FILTER_LO},{FILTER_HI}] on panel mean)")
    print(f"{'-'*92}")
    print(f"  {'tier':<22} {'n_full':>8} -> {'n_filt':>8}  ({'kept_%':>7})")
    for p in cfg["tier_prefixes"]:
        n_full = len(items_by_tier[p])
        n_filt = len(items_by_tier_filt[p])
        pct = 100.0 * n_filt / n_full if n_full > 0 else 0
        print(f"  {p:<22} {n_full:>8d} -> {n_filt:>8d}  ({pct:>6.1f}%)")
    print(f"  {'anchor (filtered)':<22} {len(anchor_pool_items):>8d} -> {len(anchor_items):>8d}  "
          f"({100.0 * len(anchor_items) / len(anchor_pool_items):>6.1f}%)")

    # ----- Compute metrics for each set -----
    sets = {}
    for p in cfg["tier_prefixes"]:
        K, N = build_kn_matrix(tier_raw, tier_models, items_by_tier[p])
        sets[f"{p}_full"] = (K, N, len(items_by_tier[p]))
        K, N = build_kn_matrix(tier_raw, tier_models, items_by_tier_filt[p])
        sets[f"{p}_filt"] = (K, N, len(items_by_tier_filt[p]))
    K, N = build_kn_matrix(anchor_raw, anchor_models, anchor_items)
    sets["anchor"] = (K, N, len(anchor_items))

    metrics = {name: compute_metrics(K, N) for name, (K, N, _) in sets.items()}

    # ----- Print metric table -----
    print(f"\n{'-'*92}")
    print(f"  Metrics (point estimate)")
    print(f"{'-'*92}")
    print(f"  {'set':<24} {'n':>5}  {'Res_Var':>9} {'Res_JSD':>9} {'Res_PR':>9} "
          f"{'saturation':>11}")
    ordered_names = []
    for p in cfg["tier_prefixes"]:
        ordered_names.append(f"{p}_full")
        ordered_names.append(f"{p}_filt")
    ordered_names.append("anchor")
    for name in ordered_names:
        K, N, n = sets[name]
        m = metrics[name]
        print(f"  {name:<24} {n:>5d}  {m['res_var']:>9.4f} {m['res_jsd']:>9.4f} "
              f"{m['res_pr']:>9.4f} {m['saturation']:>11.4f}")

    # ----- Bootstrap CIs and per-tier delta vs anchor -----
    rng = np.random.default_rng(args.seed)
    boots = {}
    for i, name in enumerate(ordered_names):
        K, N, _ = sets[name]
        sub_rng = np.random.default_rng(args.seed + i)
        boots[name] = bootstrap_metrics(K, N, args.B, sub_rng)

    print(f"\n{'-'*92}")
    print(f"  Filtered-tier vs anchor head-to-head  (delta = anchor - filtered_tier)")
    print(f"  '*' = 95% CI excludes 0  |  direction: '+' on Var/JSD/PR, '-' on saturation = anchor better")
    print(f"{'-'*92}")
    print(f"  {'filtered_tier':<22} {'metric':<12} {'tier_filt [95% CI]':<24} "
          f"{'anchor [95% CI]':<24} {'delta CI':<22} {'verdict':<14}")
    direction = {"res_var": "higher", "res_jsd": "higher", "res_pr": "higher",
                 "saturation": "lower"}
    deltas = {}
    for p in cfg["tier_prefixes"]:
        name_filt = f"{p}_filt"
        for k in ["res_var", "res_jsd", "res_pr", "saturation"]:
            t = boots[name_filt][k]
            a = boots["anchor"][k]
            t_lo, t_hi = np.percentile(t, [2.5, 97.5])
            a_lo, a_hi = np.percentile(a, [2.5, 97.5])
            delta = a - t
            deltas[(p, k)] = delta
            d_lo, d_hi = np.percentile(delta, [2.5, 97.5])
            sig = "*" if (d_lo > 0 or d_hi < 0) else " "
            point_a = metrics["anchor"][k]
            point_t = metrics[name_filt][k]
            anchor_wins = ((point_a > point_t) if direction[k] == "higher"
                           else (point_a < point_t))
            verdict = ("ANCHOR" if anchor_wins else "tier") + (" *" if sig == "*" else "")
            print(f"  {p:<22} {k:<12} {fmt_ci(point_t, t_lo, t_hi):<24} "
                  f"{fmt_ci(point_a, a_lo, a_hi):<24} "
                  f"[{d_lo:+.3f}, {d_hi:+.3f}]{sig:<2} {verdict:<14}")
        print()

    # ----- Save -----
    out = {
        "category": args.category,
        "filter": {"lo": FILTER_LO, "hi": FILTER_HI},
        "B": args.B,
        "seed": args.seed,
        "panel_models": anchor_models,
        "n_anchor": len(anchor_items),
        "tiers": {
            p: {
                "n_full": len(items_by_tier[p]),
                "n_filt": len(items_by_tier_filt[p]),
                "metrics_full": metrics[f"{p}_full"],
                "metrics_filt": metrics[f"{p}_filt"],
            }
            for p in cfg["tier_prefixes"]
        },
        "metrics_anchor": metrics["anchor"],
    }
    out_path = args.output or f"/tmp/per_tier_filter_{args.category}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
