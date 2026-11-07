"""W/M/S validation: 3 spanning models construct filter, 6 held-out evaluate.

For each category, pick the weakest, median, strongest panel models (by IRT
theta on the AVG-filtered anchor pool). Use these 3 to construct the filter
(panel mean p_hat in [0.05, 0.95]); evaluate metrics on the remaining 6.

This addresses two issues with the previous LOO (n_train=6, n_test=3):
  1. Test panel of 3 was too small for stable diversity metrics.
  2. n_train=6 used most models for filter construction, leaving little
     held-out signal.

Here we deliberately span the ability range with the 3 filter models
(weakest + median + strongest), giving the filter a representative panel-
mean signal even from just 3 models. Metrics on 6 held-out models retain
strong statistical power.

Usage:
  python scripts/analysis/wms_validation_test.py --category harmful_refusal
  python scripts/analysis/wms_validation_test.py --category constrained_completion
  python scripts/analysis/wms_validation_test.py --category over_refusal
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# Filter-construction triples (weakest / median / strongest) per category,
# determined from IRT theta on AVG-filtered anchor pool of each category.
WMS = {
    "harmful_refusal": {
        "weakest": "mistralai/Mistral-Nemo-Instruct-2407",
        "median": "google/gemma-3-12b-it",
        "strongest": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    },
    "constrained_completion": {
        "weakest": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "median": "google/gemma-3-12b-it",
        "strongest": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    },
    "over_refusal": {
        "weakest": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "median": "Qwen/Qwen3-8B",
        "strongest": "google/gemma-3-12b-it",
    },
    "sycophancy_resistance": {
        "weakest": "google/gemma-3-4b-it",
        "median": "mistralai/Mistral-Nemo-Instruct-2407",
        "strongest": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    },
}

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


def load_response_matrix_dict(data_dir: Path) -> dict:
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


def build_kn_matrix(raw, models, items):
    M, J = len(models), len(items)
    K = np.zeros((M, J))
    N = np.zeros((M, J))
    for i, m in enumerate(models):
        for j, p in enumerate(items):
            k, n = raw.get((m, p), (0, 0))
            K[i, j] = k
            N[i, j] = n
    return K, N


def H_bernoulli(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def compute_metrics(K, N):
    M, J = K.shape
    if J == 0 or M < 2:
        return {k: float("nan") for k in
                ["res_var", "res_jsd", "res_pr", "saturation",
                 "d_eff", "top1_frac", "spec_entropy"]}
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
    res_pr = n_distinct / max(n_pairs, 1)
    saturation = float(((p_hat == 0) | (p_hat == 1)).mean())

    Q = (p_hat - p_hat.mean(axis=0, keepdims=True)).T
    cov = Q.T @ Q / J
    lam = np.sort(np.linalg.eigvalsh(cov))[::-1]
    lam = np.clip(lam, 0, None)
    if lam.sum() < 1e-12:
        d_eff = top1_frac = spec_entropy = float("nan")
    else:
        s_l = lam.sum()
        d_eff = float(s_l ** 2 / (lam ** 2).sum())
        top1_frac = float(lam[0] / s_l)
        Mm1 = M - 1
        if Mm1 > 1 and lam[:Mm1].sum() > 0:
            p_dist = np.clip(lam[:Mm1] / lam[:Mm1].sum(), 1e-12, 1.0)
            spec_entropy = float(-(p_dist * np.log(p_dist)).sum() / np.log(Mm1))
        else:
            spec_entropy = float("nan")

    return {
        "res_var": res_var, "res_jsd": res_jsd, "res_pr": res_pr,
        "saturation": saturation, "d_eff": d_eff, "top1_frac": top1_frac,
        "spec_entropy": spec_entropy,
    }


def bootstrap_metrics(K, N, B, rng):
    J = K.shape[1]
    keys = ["res_var", "res_jsd", "res_pr", "saturation",
            "d_eff", "top1_frac", "spec_entropy"]
    out = {k: np.empty(B) for k in keys}
    if J == 0:
        for k in keys:
            out[k][:] = np.nan
        return out
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
    wms = WMS[args.category]

    mixed_data = load_response_matrix_dict(Path(cfg["mixed_dir"]))
    anchor_data = load_response_matrix_dict(Path(cfg["anchor_dir"]))
    panel = mixed_data["models"]
    if panel != anchor_data["models"]:
        print(f"WARN panel mismatch")

    # Indices of train (w/m/s) and test (rest)
    train_models = [wms["weakest"], wms["median"], wms["strongest"]]
    train_idx = [panel.index(m) for m in train_models]
    test_idx = [i for i in range(len(panel)) if i not in train_idx]

    print(f"\n{'='*100}")
    print(f"  CATEGORY: {args.category}  |  W/M/S filter (n_train=3) -> 6 held-out eval")
    print(f"{'='*100}")
    print(f"  Filter constructors (weakest, median, strongest):")
    for label in ["weakest", "median", "strongest"]:
        print(f"    {label:<10} {wms[label]}")
    print(f"  Held-out (6):")
    for i in test_idx:
        print(f"    {panel[i]}")

    # Item sets
    mixed_items = sorted({pid for (_, pid) in mixed_data["raw"].keys()})
    anchor_items_pool = sorted({pid for (_, pid) in anchor_data["raw"].keys()})
    K_mix, N_mix = build_kn_matrix(mixed_data["raw"], panel, mixed_items)
    K_anc, N_anc = build_kn_matrix(anchor_data["raw"], panel, anchor_items_pool)

    # Apply filter using train_idx panel mean
    p_anc_train_mean = (K_anc[train_idx] / np.maximum(N_anc[train_idx], 1)).mean(axis=0)
    keep_anc = (p_anc_train_mean >= FILTER_LO) & (p_anc_train_mean <= FILTER_HI)
    K_anc_filt = K_anc[:, keep_anc]
    N_anc_filt = N_anc[:, keep_anc]

    p_mix_train_mean = (K_mix[train_idx] / np.maximum(N_mix[train_idx], 1)).mean(axis=0)
    keep_mix = (p_mix_train_mean >= FILTER_LO) & (p_mix_train_mean <= FILTER_HI)
    K_mix_filt = K_mix[:, keep_mix]
    N_mix_filt = N_mix[:, keep_mix]

    print(f"\n{'-'*100}")
    print(f"  Filter shrinkage (3 train models)")
    print(f"{'-'*100}")
    print(f"  anchor pool ({len(anchor_items_pool)}) -> filt {keep_anc.sum()}  "
          f"({100*keep_anc.sum()/len(anchor_items_pool):.1f}%)")
    print(f"  mixed pool ({len(mixed_items)}) -> filt {keep_mix.sum()}  "
          f"({100*keep_mix.sum()/len(mixed_items):.1f}%)")

    # Compute metrics on 6 held-out models
    K_anc_test = K_anc_filt[test_idx]
    N_anc_test = N_anc_filt[test_idx]
    K_mix_test = K_mix_filt[test_idx]
    N_mix_test = N_mix_filt[test_idx]

    m_anc = compute_metrics(K_anc_test, N_anc_test)
    m_mix = compute_metrics(K_mix_test, N_mix_test)

    rng = np.random.default_rng(args.seed)
    rng2 = np.random.default_rng(args.seed + 1)
    boot_anc = bootstrap_metrics(K_anc_test, N_anc_test, args.B, rng)
    boot_mix = bootstrap_metrics(K_mix_test, N_mix_test, args.B, rng2)

    # ----- Report -----
    direction = {"res_var": "higher", "res_jsd": "higher", "res_pr": "higher",
                 "saturation": "lower", "d_eff": "higher",
                 "top1_frac": "lower", "spec_entropy": "higher"}

    print(f"\n{'-'*100}")
    print(f"  Metrics on 6 held-out models  (B={args.B} bootstrap)")
    print(f"{'-'*100}")
    print(f"  {'metric':<14} {'anchor [95% CI]':<24} {'mixed [95% CI]':<24} "
          f"{'delta CI':<22} {'verdict':<14}")
    summary = {}
    for k in ["res_var", "res_jsd", "res_pr", "saturation",
              "d_eff", "top1_frac", "spec_entropy"]:
        a = boot_anc[k]
        m = boot_mix[k]
        a_lo, a_hi = np.percentile(a, [2.5, 97.5])
        m_lo, m_hi = np.percentile(m, [2.5, 97.5])
        d = a - m
        d_lo, d_hi = np.percentile(d, [2.5, 97.5])
        if direction[k] == "higher":
            verdict = "ANCHOR *" if d_lo > 0 else ("mixed *" if d_hi < 0 else "tie")
        else:
            verdict = "ANCHOR *" if d_hi < 0 else ("mixed *" if d_lo > 0 else "tie")
        print(f"  {k:<14} {fmt_ci(m_anc[k], a_lo, a_hi):<24} "
              f"{fmt_ci(m_mix[k], m_lo, m_hi):<24} "
              f"[{d_lo:+.3f}, {d_hi:+.3f}]    {verdict:<14}")
        summary[k] = {
            "anchor": m_anc[k], "anchor_q025": float(a_lo), "anchor_q975": float(a_hi),
            "mixed": m_mix[k], "mixed_q025": float(m_lo), "mixed_q975": float(m_hi),
            "delta_q025": float(d_lo), "delta_q975": float(d_hi),
            "verdict": verdict,
        }

    # ----- Save -----
    out = {
        "category": args.category,
        "B": args.B, "seed": args.seed,
        "panel": panel,
        "train_models": train_models, "test_models": [panel[i] for i in test_idx],
        "n_anchor_pool": len(anchor_items_pool), "n_mixed_pool": len(mixed_items),
        "n_anchor_filt": int(keep_anc.sum()), "n_mixed_filt": int(keep_mix.sum()),
        "metrics_anchor": m_anc, "metrics_mixed": m_mix,
        "summary": summary,
    }
    out_path = args.output or f"/tmp/wms_validation_{args.category}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
