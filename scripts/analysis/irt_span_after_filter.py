"""IRT theta_span / beta_span after filter — Claim 1 robustness check.

The earlier IRT analysis (tier_vs_anchor_subsample.py) showed inflated
theta/beta ranges due to saturated cells pushing MLE estimates to +-inf.
After applying AVG[0.05, 0.95] filter, saturation is reduced; this script
re-runs the IRT comparison on the filtered sets to see if the original
"Claim 1 via theta/beta range" framing survives.

Two sub-experiments:

  A) mixed_filtered (full validation set, AVG-filtered) vs anchor (AVG-filtered)
     - Single point estimate + B bootstrap items for CI on each.

  B) Per-tier filtered fixed benchmark vs anchor sub-sampled at matched n
     - For each tier: fit IRT on filtered tier items.
     - For anchor: B random sub-samples of size = filtered tier size, fit IRT each.
     - Report tier theta/beta range against anchor sub-sample distribution.

Metrics:
  theta_range = max(theta_mle) - min(theta_mle)   (panel ability spread, logits)
  beta_range  = max(beta_mle)  - min(beta_mle)    (item difficulty spread, logits)

Higher = wider, but be aware: cells with p_hat=0 or 1 still push to ~ +-23
(due to 1e-10 clip in sigmoid). Post-filter, this should be reduced but not
eliminated entirely.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difficulty.rasch import fit_ols, fit_mle


CATEGORY_CONFIG = {
    "harmful_refusal": {
        "mixed_dir": "results/harmful_refusal/validation/validation_benchmark_response",
        "anchor_dir": "results/harmful_refusal/filter_ablation_raw_v2_judgefix",
        "tier_prefixes": ["val_A_easy", "val_A_medium", "val_A_hard"],
    },
    "constrained_completion": {
        "mixed_dir": "results/constrained_completion/validation/validation_benchmark_response",
        "anchor_dir": "results/constrained_completion/filter_ablation_raw_v2",
        "tier_prefixes": ["val_B_ifbench", "val_B_manyifeval"],
    },
    "over_refusal": {
        "mixed_dir": "results/over_refusal/validation/validation_benchmark_response",
        "anchor_dir": "results/over_refusal/filter_ablation_raw_v2_judgefix",
        "tier_prefixes": ["val_Aprime"],
    },
    "sycophancy_resistance": {
        "mixed_dir": "results/sycophancy_resistance/validation/validation_benchmark_response",
        "anchor_dir": "results/sycophancy_resistance/filter_ablation_raw_v2",
        "tier_prefixes": ["val_C_easy", "val_C_medium", "val_C_hard"],
    },
}

FILTER_LO, FILTER_HI = 0.05, 0.95


def load_response_matrix_dict(data_dir: Path):
    raw = {}
    models = []
    for sd in sorted(data_dir.iterdir()):
        if not sd.is_dir():
            continue
        rm = sd / "response_matrix.jsonl"
        if not rm.exists():
            continue
        meta = sd / "metadata.json"
        name = sd.name
        if meta.exists():
            with open(meta) as f:
                name = json.load(f).get("model", name)
        models.append(name)
        with open(rm) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    raw[(name, e["prompt_id"])] = (e["pass_count"], e["total_trials"])
    return {"raw": raw, "models": sorted(models)}


def build_kn(raw, models, items):
    M, J = len(models), len(items)
    K = np.zeros((M, J))
    N = np.zeros((M, J))
    for i, m in enumerate(models):
        for j, p in enumerate(items):
            k, n = raw.get((m, p), (0, 0))
            K[i, j] = k
            N[i, j] = n
    return K, N


def apply_avg_filter(raw, models, items):
    K, N = build_kn(raw, models, items)
    p = K / np.maximum(N, 1)
    pm = p.mean(axis=0)
    keep = (pm >= FILTER_LO) & (pm <= FILTER_HI)
    return [items[j] for j in range(len(items)) if keep[j]]


def irt_spans(K, N) -> dict:
    """Fit Rasch (OLS init + MLE), return theta_range, beta_range, plus OLS for robustness."""
    if K.shape[1] == 0:
        return {k: float("nan") for k in
                ["theta_range", "beta_range", "theta_range_ols", "beta_range_ols"]}
    p_hat = np.where(N > 0, K / np.maximum(N, 1), 0.5)
    theta_ols, beta_ols = fit_ols(p_hat)
    try:
        theta_mle, beta_mle, _ = fit_mle(K, N, theta_ols, beta_ols)
    except Exception:
        theta_mle, beta_mle = theta_ols, beta_ols
    return {
        "theta_range": float(theta_mle.max() - theta_mle.min()),
        "beta_range": float(beta_mle.max() - beta_mle.min()),
        "theta_range_ols": float(theta_ols.max() - theta_ols.min()),
        "beta_range_ols": float(beta_ols.max() - beta_ols.min()),
    }


def bootstrap_irt(K, N, B, rng):
    J = K.shape[1]
    keys = ["theta_range", "beta_range", "theta_range_ols", "beta_range_ols"]
    out = {k: np.empty(B) for k in keys}
    if J == 0:
        for k in keys:
            out[k][:] = np.nan
        return out
    for b in range(B):
        idx = rng.integers(0, J, size=J)
        m = irt_spans(K[:, idx], N[:, idx])
        for k in keys:
            out[k][b] = m[k]
    return out


def subsample_irt(K, N, n_sub, B, rng):
    J = K.shape[1]
    keys = ["theta_range", "beta_range", "theta_range_ols", "beta_range_ols"]
    out = {k: np.empty(B) for k in keys}
    for b in range(B):
        idx = rng.choice(J, size=min(n_sub, J), replace=False)
        m = irt_spans(K[:, idx], N[:, idx])
        for k in keys:
            out[k][b] = m[k]
    return out


def fmt_ci(x, lo, hi):
    return f"{x:.2f} [{lo:.2f}, {hi:.2f}]"


def percentile_rank(value, dist):
    return float((dist <= value).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, choices=list(CATEGORY_CONFIG.keys()))
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = CATEGORY_CONFIG[args.category]
    print(f"\n{'='*100}")
    print(f"  CATEGORY: {args.category}  |  IRT theta/beta range after filter  |  B={args.B}")
    print(f"{'='*100}")

    # ----- Load -----
    mixed_data = load_response_matrix_dict(Path(cfg["mixed_dir"]))
    anchor_data = load_response_matrix_dict(Path(cfg["anchor_dir"]))
    panel = mixed_data["models"]
    if panel != anchor_data["models"]:
        print(f"WARN: panel mismatch")

    mixed_items = sorted({pid for (_, pid) in mixed_data["raw"].keys()})
    anchor_pool_items = sorted({pid for (_, pid) in anchor_data["raw"].keys()})

    # AVG filter on each
    mixed_filt_items = apply_avg_filter(mixed_data["raw"], panel, mixed_items)
    anchor_filt_items = apply_avg_filter(anchor_data["raw"], panel, anchor_pool_items)

    K_mix, N_mix = build_kn(mixed_data["raw"], panel, mixed_filt_items)
    K_anc, N_anc = build_kn(anchor_data["raw"], panel, anchor_filt_items)

    print(f"\n  Mixed filtered: {len(mixed_filt_items)} items (from {len(mixed_items)})")
    print(f"  Anchor filtered: {len(anchor_filt_items)} items (from {len(anchor_pool_items)})")

    # ============================================================
    # Sub-experiment A: mixed_filtered vs anchor
    # ============================================================
    print(f"\n{'-'*100}")
    print(f"  Sub-experiment A: mixed_filtered (full) vs anchor (full)")
    print(f"{'-'*100}")

    sp_mix = irt_spans(K_mix, N_mix)
    sp_anc = irt_spans(K_anc, N_anc)
    boot_mix = bootstrap_irt(K_mix, N_mix, args.B, np.random.default_rng(args.seed))
    boot_anc = bootstrap_irt(K_anc, N_anc, args.B, np.random.default_rng(args.seed + 1))

    print(f"  {'metric':<20} {'mixed_filt [95% CI]':<26} {'anchor [95% CI]':<26} "
          f"{'delta CI (a-m)':<22} {'verdict':<14}")
    sub_a = {}
    for key in ["theta_range", "beta_range", "theta_range_ols", "beta_range_ols"]:
        m_lo, m_hi = np.percentile(boot_mix[key], [2.5, 97.5])
        a_lo, a_hi = np.percentile(boot_anc[key], [2.5, 97.5])
        delta = boot_anc[key] - boot_mix[key]
        d_lo, d_hi = np.percentile(delta, [2.5, 97.5])
        # Higher = wider; we'll just report sign and significance
        if d_lo > 0:
            verdict = "ANCHOR *"
        elif d_hi < 0:
            verdict = "mixed *"
        else:
            verdict = "tie"
        print(f"  {key:<20} {fmt_ci(sp_mix[key], m_lo, m_hi):<26} "
              f"{fmt_ci(sp_anc[key], a_lo, a_hi):<26} "
              f"[{d_lo:+.2f}, {d_hi:+.2f}]      {verdict:<14}")
        sub_a[key] = {
            "mixed": sp_mix[key], "anchor": sp_anc[key],
            "mixed_q025": float(m_lo), "mixed_q975": float(m_hi),
            "anchor_q025": float(a_lo), "anchor_q975": float(a_hi),
            "delta_q025": float(d_lo), "delta_q975": float(d_hi),
            "verdict": verdict,
        }

    # ============================================================
    # Sub-experiment B: per-tier filtered vs anchor sub-sampled at matched n
    # ============================================================
    print(f"\n{'-'*100}")
    print(f"  Sub-experiment B: per-tier filtered vs anchor sub-sample (matched n)")
    print(f"{'-'*100}")

    all_tier_items = sorted({pid for (_, pid) in mixed_data["raw"].keys()})

    sub_b = {}
    for prefix in cfg["tier_prefixes"]:
        tier_items_full = sorted(p for p in all_tier_items if p.startswith(prefix + "_"))
        tier_items_filt = apply_avg_filter(mixed_data["raw"], panel, tier_items_full)
        n_filt = len(tier_items_filt)
        if n_filt < 9:  # IRT needs at least M items for stability
            print(f"\n  {prefix} (filt {n_filt} items): TOO SMALL, skipping")
            continue

        K_t, N_t = build_kn(mixed_data["raw"], panel, tier_items_filt)
        sp_tier = irt_spans(K_t, N_t)
        boot_tier = bootstrap_irt(K_t, N_t, args.B, np.random.default_rng(args.seed + 100))

        # Anchor sub-sampled to match tier filt size
        if n_filt > K_anc.shape[1]:
            n_sub = K_anc.shape[1]
        else:
            n_sub = n_filt
        sub_anc = subsample_irt(K_anc, N_anc, n_sub, args.B,
                                np.random.default_rng(args.seed + 200))

        print(f"\n  --- {prefix}_filt (n={n_filt})  vs  anchor_subsample (n={n_sub}, B={args.B}) ---")
        print(f"  {'metric':<20} {'tier (point + boot CI)':<32} "
              f"{'anchor sub-sample (q05, q50, q95)':<36} {'tier pct rank':<14}")
        sub_b[prefix] = {"n_filt": n_filt, "n_sub": n_sub}
        for key in ["theta_range", "beta_range", "theta_range_ols", "beta_range_ols"]:
            t_lo, t_hi = np.percentile(boot_tier[key], [2.5, 97.5])
            a_q05, a_q50, a_q95 = np.percentile(sub_anc[key], [5, 50, 95])
            pr = percentile_rank(sp_tier[key], sub_anc[key])
            tier_str = f"{sp_tier[key]:.2f} [{t_lo:.2f}, {t_hi:.2f}]"
            anchor_str = f"({a_q05:.2f}, {a_q50:.2f}, {a_q95:.2f})"
            print(f"  {key:<20} {tier_str:<32} {anchor_str:<36} {pr:>6.1f}%")
            sub_b[prefix][key] = {
                "tier": sp_tier[key],
                "tier_q025": float(t_lo), "tier_q975": float(t_hi),
                "anchor_q05": float(a_q05), "anchor_q50": float(a_q50),
                "anchor_q95": float(a_q95),
                "tier_pct_rank": pr,
            }

    # ----- Save -----
    out = {
        "category": args.category,
        "B": args.B, "seed": args.seed,
        "panel": panel,
        "n_mixed_filt": len(mixed_filt_items),
        "n_anchor_filt": len(anchor_filt_items),
        "sub_experiment_a": sub_a,
        "sub_experiment_b": sub_b,
    }
    out_path = args.output or f"/tmp/irt_span_after_filter_{args.category}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
