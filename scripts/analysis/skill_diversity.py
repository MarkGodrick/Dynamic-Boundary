"""Skill diversity (panel-deviation spectrum) of anchor vs filtered fixed benchmarks.

For each benchmark with item set {x_1, ..., x_J} and 9-model panel:
  q_x = p_hat_x - mean_panel(p_hat_x)         # panel-deviation, 9-D, sum=0
  Q = stack(q_x)                              # (J, 9)
  Cov = Q.T @ Q / J                           # (9, 9), rank <= 8
  lambda_1 >= lambda_2 >= ... >= lambda_8 > 0  (the 9th eigval ~= 0)

Three summary metrics on the spectrum (all scale-invariant):
  D_eff       = (sum lambda)^2 / sum lambda^2          in [1, 8]
  top1_frac   = lambda_1 / sum lambda                  in [1/8, 1]   (lower = more diverse)
  spec_entropy = -sum p_i log p_i / log(8), p = lambda/sum lambda    in [0, 1]
                                                     (higher = more diverse)

Compare anchor vs each filtered tier and vs mixed_filtered.

Usage:
  python scripts/analysis/skill_diversity.py --category harmful_refusal
  python scripts/analysis/skill_diversity.py --category constrained_completion
  python scripts/analysis/skill_diversity.py --category over_refusal
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
    "over_refusal": {
        "tier_dir": "results/over_refusal/validation/validation_benchmark_response",
        "anchor_dir": "results/over_refusal/filter_ablation_raw_v2_judgefix",
        "tier_prefixes": ["val_Aprime"],
    },
    "sycophancy_resistance": {
        "tier_dir": "results/sycophancy_resistance/validation/validation_benchmark_response",
        "anchor_dir": "results/sycophancy_resistance/filter_ablation_raw_v2",
        "tier_prefixes": ["val_C_easy", "val_C_medium", "val_C_hard"],
    },
}

FILTER_LO, FILTER_HI = 0.05, 0.95


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


def build_p_hat_matrix(raw, models, items):
    """Return p_hat (M, J)."""
    M, J = len(models), len(items)
    K = np.zeros((M, J))
    N = np.zeros((M, J))
    for i, m in enumerate(models):
        for j, p in enumerate(items):
            k, n = raw.get((m, p), (0, 0))
            K[i, j] = k
            N[i, j] = n
    return K / np.maximum(N, 1)


def apply_avg_filter(raw, models, items):
    p = build_p_hat_matrix(raw, models, items)
    pm = p.mean(axis=0)
    keep = (pm >= FILTER_LO) & (pm <= FILTER_HI)
    return [items[j] for j in range(len(items)) if keep[j]]


def panel_deviation_spectrum(p_hat: np.ndarray) -> np.ndarray:
    """p_hat: (M, J). Return descending eigenvalues of Cov = Q^T Q / J, length M."""
    M, J = p_hat.shape
    if J == 0:
        return np.zeros(M)
    Q = (p_hat - p_hat.mean(axis=0, keepdims=True)).T  # (J, M)
    cov = Q.T @ Q / J  # (M, M), rank <= M-1
    lam = np.linalg.eigvalsh(cov)
    lam = np.sort(lam)[::-1]  # descending
    lam = np.clip(lam, 0, None)  # numerical safety
    return lam


def diversity_metrics(lam: np.ndarray) -> dict:
    """Compute D_eff, top1_frac, spectrum entropy. lam: descending eigvals."""
    M = len(lam)
    Mm1 = M - 1  # rank bound after centering
    if lam.sum() < 1e-12:
        return {"d_eff": float("nan"), "top1_frac": float("nan"),
                "spec_entropy": float("nan"), "total_var": 0.0}

    s = lam.sum()
    s2 = (lam ** 2).sum()
    d_eff = float(s ** 2 / s2)
    top1_frac = float(lam[0] / s)

    # Spectrum entropy on the M-1 nonzero eigvals (drop the trivially zero one)
    p = lam[:Mm1] / lam[:Mm1].sum() if lam[:Mm1].sum() > 0 else np.zeros(Mm1)
    p = np.clip(p, 1e-12, 1.0)
    spec_entropy = float(-(p * np.log(p)).sum() / np.log(Mm1))

    return {
        "d_eff": d_eff,
        "top1_frac": top1_frac,
        "spec_entropy": spec_entropy,
        "total_var": float(s),
    }


def bootstrap_diversity(p_hat, B, rng):
    """Item-level bootstrap. p_hat: (M, J)."""
    M, J = p_hat.shape
    keys = ["d_eff", "top1_frac", "spec_entropy", "total_var"]
    out = {k: np.empty(B) for k in keys}
    for b in range(B):
        idx = rng.integers(0, J, size=J)
        lam = panel_deviation_spectrum(p_hat[:, idx])
        m = diversity_metrics(lam)
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
    print(f"\n{'='*100}")
    print(f"  CATEGORY: {args.category}  |  SKILL DIVERSITY  |  B={args.B}")
    print(f"{'='*100}")

    # Load tier data
    tier_data = load_response_matrix_dict(Path(cfg["tier_dir"]))
    tier_raw = tier_data["raw"]
    tier_models = tier_data["models"]
    all_tier_items = sorted({pid for (_, pid) in tier_raw.keys()})

    # Per-tier filtered (and overall mixed_filtered)
    items_by_tier = {p: sorted(pid for pid in all_tier_items if pid.startswith(p + "_"))
                     for p in cfg["tier_prefixes"]}
    items_by_tier_filt = {p: apply_avg_filter(tier_raw, tier_models, items)
                          for p, items in items_by_tier.items()}
    mixed_filtered_items = apply_avg_filter(tier_raw, tier_models, all_tier_items)

    # Anchor
    anchor_data = load_response_matrix_dict(Path(cfg["anchor_dir"]))
    anchor_raw = anchor_data["raw"]
    anchor_models = anchor_data["models"]
    anchor_pool_items = sorted({pid for (_, pid) in anchor_raw.keys()})
    anchor_items = apply_avg_filter(anchor_raw, anchor_models, anchor_pool_items)

    # Build p_hat matrices
    sets = {}
    for p in cfg["tier_prefixes"]:
        sets[f"{p}_filt"] = build_p_hat_matrix(tier_raw, tier_models, items_by_tier_filt[p])
    sets["mixed_filt"] = build_p_hat_matrix(tier_raw, tier_models, mixed_filtered_items)
    sets["anchor"] = build_p_hat_matrix(anchor_raw, anchor_models, anchor_items)

    n_items = {name: arr.shape[1] for name, arr in sets.items()}

    # ----- Point estimates -----
    spectra = {name: panel_deviation_spectrum(arr) for name, arr in sets.items()}
    metrics = {name: diversity_metrics(lam) for name, lam in spectra.items()}

    print(f"\n{'-'*100}")
    print(f"  Point estimates  (D_eff in [1, {len(anchor_models)-1}],  "
          f"top1_frac in [1/{len(anchor_models)-1}, 1] -- lower is more diverse,  "
          f"spec_entropy in [0, 1] -- higher is more diverse)")
    print(f"{'-'*100}")
    print(f"  {'set':<24} {'n':>5}  {'D_eff':>9} {'top1_frac':>11} {'spec_entropy':>13} "
          f"{'total_var':>11}")
    ordered = [f"{p}_filt" for p in cfg["tier_prefixes"]] + ["mixed_filt", "anchor"]
    for name in ordered:
        m = metrics[name]
        print(f"  {name:<24} {n_items[name]:>5d}  {m['d_eff']:>9.3f} "
              f"{m['top1_frac']:>11.3f} {m['spec_entropy']:>13.3f} {m['total_var']:>11.4f}")

    # ----- Bootstrap CIs and head-to-head delta -----
    rng = np.random.default_rng(args.seed)
    boots = {}
    for i, name in enumerate(ordered):
        sub_rng = np.random.default_rng(args.seed + i * 17)
        boots[name] = bootstrap_diversity(sets[name], args.B, sub_rng)

    print(f"\n{'-'*100}")
    print(f"  Anchor vs each filtered fixed  (delta = anchor - tier; '*' = 95% CI excludes 0)")
    print(f"  Anchor 'better' = higher D_eff, lower top1_frac, higher spec_entropy")
    print(f"{'-'*100}")
    print(f"  {'tier':<24} {'metric':<14} {'tier [95% CI]':<24} "
          f"{'anchor [95% CI]':<24} {'delta CI':<22} {'verdict':<14}")

    direction = {"d_eff": "higher", "top1_frac": "lower", "spec_entropy": "higher"}
    deltas_save = {}

    for name in ordered:
        if name == "anchor":
            continue
        deltas_save[name] = {}
        for k in ["d_eff", "top1_frac", "spec_entropy"]:
            t = boots[name][k]
            a = boots["anchor"][k]
            t_lo, t_hi = np.percentile(t, [2.5, 97.5])
            a_lo, a_hi = np.percentile(a, [2.5, 97.5])
            d = a - t
            d_lo, d_hi = np.percentile(d, [2.5, 97.5])
            sig = "*" if (d_lo > 0 or d_hi < 0) else " "
            point_a = metrics["anchor"][k]
            point_t = metrics[name][k]
            anchor_wins = (point_a > point_t) if direction[k] == "higher" else (point_a < point_t)
            verdict = ("ANCHOR" if anchor_wins else "tier") + (" *" if sig == "*" else "")
            print(f"  {name:<24} {k:<14} {fmt_ci(point_t, t_lo, t_hi):<24} "
                  f"{fmt_ci(point_a, a_lo, a_hi):<24} "
                  f"[{d_lo:+.3f}, {d_hi:+.3f}]{sig:<2} {verdict:<14}")
            deltas_save[name][k] = {
                "delta_q025": float(d_lo), "delta_q975": float(d_hi),
                "tier_value": point_t, "anchor_value": point_a,
            }
        print()

    # ----- Eigenvalue table for inspection -----
    print(f"{'-'*100}")
    print(f"  Eigenvalue spectrum (sorted, top 8)")
    print(f"{'-'*100}")
    print(f"  {'set':<24} " + " ".join(f"{f'lam_{i+1}':>8}" for i in range(8)))
    for name in ordered:
        lam = spectra[name]
        row = " ".join(f"{lam[i]:>8.4f}" for i in range(min(8, len(lam))))
        print(f"  {name:<24} {row}")

    # ----- Save -----
    out = {
        "category": args.category,
        "B": args.B,
        "seed": args.seed,
        "panel_models": anchor_models,
        "n_items": n_items,
        "metrics": metrics,
        "spectra": {name: lam.tolist() for name, lam in spectra.items()},
        "deltas_vs_anchor": deltas_save,
    }
    out_path = args.output or f"/tmp/skill_diversity_{args.category}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
