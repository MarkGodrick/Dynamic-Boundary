"""Head-to-head: anchor set vs mixed fixed benchmark (Claim 1 main test).

The 'mixed fixed benchmark' = the full validation_benchmark_response produced
by sample_validation_benchmarks.py. For category A this combines 3 tiers
(easy + medium + hard, 300 items); for category B this combines IFBench +
ManyIFEval (175 items).

The anchor set = filter_ablation pool AVG[0.05, 0.95]-filtered (306 / 175).

For each: compute the 4 score-based metrics at native size; bootstrap items
with replacement for 95% CIs; report delta (anchor - mixed) with CI.

Usage:
  python scripts/analysis/mixed_vs_anchor_metrics.py --category harmful_refusal
  python scripts/analysis/mixed_vs_anchor_metrics.py --category constrained_completion
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


CATEGORY_CONFIG = {
    "harmful_refusal": {
        "mixed_dir": "results/harmful_refusal/validation/validation_benchmark_response",
        "anchor_dir": "results/harmful_refusal/filter_ablation_raw_v2_judgefix",
    },
    "constrained_completion": {
        "mixed_dir": "results/constrained_completion/validation/validation_benchmark_response",
        "anchor_dir": "results/constrained_completion/filter_ablation_raw_v2",
    },
    "over_refusal": {
        "mixed_dir": "results/over_refusal/validation/validation_benchmark_response",
        "anchor_dir": "results/over_refusal/filter_ablation_raw_v2_judgefix",
    },
    "sycophancy_resistance": {
        "mixed_dir": "results/sycophancy_resistance/validation/validation_benchmark_response",
        "anchor_dir": "results/sycophancy_resistance/filter_ablation_raw_v2",
    },
}

FILTER_LO, FILTER_HI = 0.05, 0.95
Z = 1.96


# -------------------------------------------------------------------
# Loaders
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

def H_bernoulli(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def compute_metrics(K, N):
    M, J = K.shape
    N_safe = np.maximum(N, 1)
    p_hat = K / N_safe
    p_smooth = (K + 1) / (N + 2)

    res_var = float(p_smooth.var(axis=0).mean())

    p_bar = p_smooth.mean(axis=0)
    H_pbar = H_bernoulli(p_bar)
    H_each = H_bernoulli(p_smooth).mean(axis=0)
    res_jsd = float((H_pbar - H_each).mean() / np.log(M))

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

    saturation = float(((p_hat == 0) | (p_hat == 1)).mean())
    return {"res_var": res_var, "res_jsd": res_jsd, "res_pr": res_pr, "saturation": saturation}


def bootstrap_metrics(K, N, B, rng):
    """Resample items with replacement at native size, B times."""
    J = K.shape[1]
    keys = ["res_var", "res_jsd", "res_pr", "saturation"]
    out = {k: np.empty(B) for k in keys}
    for b in range(B):
        idx = rng.integers(0, J, size=J)
        m = compute_metrics(K[:, idx], N[:, idx])
        for k in keys:
            out[k][b] = m[k]
    return out


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
    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*84}")
    print(f"  CATEGORY: {args.category}  |  ANCHOR vs MIXED FIXED  |  B={args.B}")
    print(f"{'='*84}")

    # ----- mixed fixed (full validation set) -----
    mixed_data = load_response_matrix_dict(Path(cfg["mixed_dir"]))
    mixed_raw = mixed_data["raw"]
    mixed_models = mixed_data["models"]
    mixed_items = sorted({pid for (_, pid) in mixed_raw.keys()})
    K_m, N_m = build_kn_matrix(mixed_raw, mixed_models, mixed_items)
    print(f"Mixed fixed: {len(mixed_items)} items, {len(mixed_models)} models")

    # ----- mixed-fixed-FILTERED (apply same AVG filter as anchor) -----
    p_m = K_m / np.maximum(N_m, 1)
    panel_mean_m = p_m.mean(axis=0)
    keep_m = (panel_mean_m >= FILTER_LO) & (panel_mean_m <= FILTER_HI)
    mixed_filtered_items = [mixed_items[j] for j in range(len(mixed_items)) if keep_m[j]]
    K_mf, N_mf = build_kn_matrix(mixed_raw, mixed_models, mixed_filtered_items)
    print(f"Mixed (filtered AVG[{FILTER_LO},{FILTER_HI}]): {len(mixed_filtered_items)} items")

    # ----- anchor (filtered) -----
    anchor_data = load_response_matrix_dict(Path(cfg["anchor_dir"]))
    anchor_raw = anchor_data["raw"]
    anchor_models = anchor_data["models"]
    anchor_pool_items = sorted({pid for (_, pid) in anchor_raw.keys()})
    K_pool, N_pool = build_kn_matrix(anchor_raw, anchor_models, anchor_pool_items)
    p_pool = K_pool / np.maximum(N_pool, 1)
    panel_mean = p_pool.mean(axis=0)
    keep_mask = (panel_mean >= FILTER_LO) & (panel_mean <= FILTER_HI)
    anchor_items = [anchor_pool_items[j] for j in range(len(anchor_pool_items)) if keep_mask[j]]
    K_a, N_a = build_kn_matrix(anchor_raw, anchor_models, anchor_items)
    print(f"Anchor: pool {len(anchor_pool_items)} -> AVG[{FILTER_LO},{FILTER_HI}] = {len(anchor_items)} items")

    if anchor_models != mixed_models:
        print(f"WARN panel mismatch.")
        print(f"  mixed:  {mixed_models}")
        print(f"  anchor: {anchor_models}")

    # ----- Point estimates -----
    metrics_mixed = compute_metrics(K_m, N_m)
    metrics_mixed_filt = compute_metrics(K_mf, N_mf)
    metrics_anchor = compute_metrics(K_a, N_a)

    print(f"\n{'-'*84}")
    print(f"  Point estimates (full set, no resampling)")
    print(f"{'-'*84}")
    print(f"  {'metric':<14} {'mixed_full':>11} {'mixed_filt':>11} {'anchor':>11}  "
          f"{'a-mfull':>9} {'a-mfilt':>9}")
    for k in ["res_var", "res_jsd", "res_pr", "saturation"]:
        d_full = metrics_anchor[k] - metrics_mixed[k]
        d_filt = metrics_anchor[k] - metrics_mixed_filt[k]
        print(f"  {k:<14} {metrics_mixed[k]:>11.4f} {metrics_mixed_filt[k]:>11.4f} "
              f"{metrics_anchor[k]:>11.4f}  {d_full:>+9.4f} {d_filt:>+9.4f}")

    # ----- Bootstrap CIs -----
    rng_m = np.random.default_rng(args.seed)
    rng_mf = np.random.default_rng(args.seed + 2)
    rng_a = np.random.default_rng(args.seed + 1)
    boot_mixed = bootstrap_metrics(K_m, N_m, args.B, rng_m)
    boot_mixed_filt = bootstrap_metrics(K_mf, N_mf, args.B, rng_mf)
    boot_anchor = bootstrap_metrics(K_a, N_a, args.B, rng_a)

    print(f"\n{'-'*84}")
    print(f"  Bootstrap CI (B={args.B})  |  delta_full = anchor - mixed_full,  "
          f"delta_filt = anchor - mixed_filt")
    print(f"{'-'*84}")
    print(f"  {'metric':<12} {'mixed_full [95% CI]':<22} {'mixed_filt [95% CI]':<22} "
          f"{'anchor [95% CI]':<22} {'delta_full CI':<18} {'delta_filt CI':<18}")

    delta_full_dist = {}
    delta_filt_dist = {}
    for k in ["res_var", "res_jsd", "res_pr", "saturation"]:
        m_lo, m_hi = np.percentile(boot_mixed[k], [2.5, 97.5])
        mf_lo, mf_hi = np.percentile(boot_mixed_filt[k], [2.5, 97.5])
        a_lo, a_hi = np.percentile(boot_anchor[k], [2.5, 97.5])
        delta_full_dist[k] = boot_anchor[k] - boot_mixed[k]
        delta_filt_dist[k] = boot_anchor[k] - boot_mixed_filt[k]
        df_lo, df_hi = np.percentile(delta_full_dist[k], [2.5, 97.5])
        dfilt_lo, dfilt_hi = np.percentile(delta_filt_dist[k], [2.5, 97.5])
        sig_full = "*" if (df_lo > 0 or df_hi < 0) else " "
        sig_filt = "*" if (dfilt_lo > 0 or dfilt_hi < 0) else " "
        print(f"  {k:<12} {metrics_mixed[k]:.3f} [{m_lo:.3f}, {m_hi:.3f}]   "
              f"{metrics_mixed_filt[k]:.3f} [{mf_lo:.3f}, {mf_hi:.3f}]   "
              f"{metrics_anchor[k]:.3f} [{a_lo:.3f}, {a_hi:.3f}]   "
              f"[{df_lo:+.3f}, {df_hi:+.3f}]{sig_full}  "
              f"[{dfilt_lo:+.3f}, {dfilt_hi:+.3f}]{sig_filt}")

    print(f"\n  '*' = 95% CI excludes 0 (significant)")

    # ----- Verdict -----
    print(f"\n{'-'*84}")
    print(f"  Verdict — anchor vs mixed_filtered (apples-to-apples filter test)")
    print(f"{'-'*84}")
    direction = {"res_var": "higher", "res_jsd": "higher", "res_pr": "higher",
                 "saturation": "lower"}
    for k in ["res_var", "res_jsd", "res_pr", "saturation"]:
        a, mf = metrics_anchor[k], metrics_mixed_filt[k]
        dfilt_lo, dfilt_hi = np.percentile(delta_filt_dist[k], [2.5, 97.5])
        ci_excludes_zero = (dfilt_lo > 0 or dfilt_hi < 0)
        anchor_wins = (a > mf) if direction[k] == "higher" else (a < mf)
        verdict = "ANCHOR wins" if anchor_wins else "mixed_filt wins"
        sig = "(significant)" if ci_excludes_zero else "(n.s.)"
        print(f"  {k:<14} {verdict:<18} {sig}")

    # ----- Save -----
    out = {
        "category": args.category,
        "B": args.B,
        "seed": args.seed,
        "n_mixed": len(mixed_items),
        "n_mixed_filtered": len(mixed_filtered_items),
        "n_anchor": len(anchor_items),
        "panel_models": mixed_models,
        "metrics_mixed": metrics_mixed,
        "metrics_mixed_filtered": metrics_mixed_filt,
        "metrics_anchor": metrics_anchor,
        "bootstrap_mixed": {k: v.tolist() for k, v in boot_mixed.items()},
        "bootstrap_mixed_filtered": {k: v.tolist() for k, v in boot_mixed_filt.items()},
        "bootstrap_anchor": {k: v.tolist() for k, v in boot_anchor.items()},
        "bootstrap_delta_full": {k: v.tolist() for k, v in delta_full_dist.items()},
        "bootstrap_delta_filt": {k: v.tolist() for k, v in delta_filt_dist.items()},
    }
    out_path = args.output or f"/tmp/mixed_vs_anchor_{args.category}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
