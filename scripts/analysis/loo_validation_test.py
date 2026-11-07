"""Leave-3-out validation: filter on 6 panel models, evaluate metrics on 3 held-out.

For each of C(9, 3) = 84 splits:
  1. Pick 3 models as held-out; 6 as filter-construction.
  2. Apply AVG[0.05, 0.95] filter using the 6-model panel mean p_hat per item
     (separately on anchor pool and on mixed validation pool).
  3. On the surviving items, compute resolution + diversity metrics using
     ONLY the 3 held-out models.
  4. Compare anchor vs mixed_filtered on each metric (paired by split).

This addresses the confound that the original filter was constructed on the
full 9 panel: if anchor still beats mixed on metrics computed on completely
unseen 3 models, the filter's "graded item" property generalizes to new
models (not just an artifact of overfitting to the panel).

Metrics on 3-panel:
  - score-based: Res_Var, Res_JSD, Res_PR, saturation
  - skill diversity: D_eff, top1_frac, spec_entropy (bounded by 2 with M=3)

Aggregation across 84 splits:
  - mean / std of each metric for anchor and mixed
  - paired delta (anchor - mixed)
  - % splits where anchor wins
  - 95% percentile CI of delta
"""

import argparse
import itertools
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


def compute_score_metrics(K, N):
    M, J = K.shape
    if J == 0 or M < 2:
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
    res_pr = n_distinct / max(n_pairs, 1)
    saturation = float(((p_hat == 0) | (p_hat == 1)).mean())
    return {"res_var": res_var, "res_jsd": res_jsd, "res_pr": res_pr, "saturation": saturation}


def compute_diversity_metrics(K, N):
    M, J = K.shape
    if J == 0 or M < 2:
        return {k: float("nan") for k in ["d_eff", "top1_frac", "spec_entropy"]}
    p_hat = K / np.maximum(N, 1)
    Q = (p_hat - p_hat.mean(axis=0, keepdims=True)).T  # (J, M)
    cov = Q.T @ Q / J
    lam = np.sort(np.linalg.eigvalsh(cov))[::-1]
    lam = np.clip(lam, 0, None)
    if lam.sum() < 1e-12:
        return {"d_eff": float("nan"), "top1_frac": float("nan"), "spec_entropy": float("nan")}
    s = lam.sum()
    d_eff = float(s ** 2 / (lam ** 2).sum())
    top1_frac = float(lam[0] / s)
    Mm1 = M - 1
    p = lam[:Mm1] / lam[:Mm1].sum() if lam[:Mm1].sum() > 0 else np.zeros(Mm1)
    p = np.clip(p, 1e-12, 1.0)
    spec_entropy = float(-(p * np.log(p)).sum() / np.log(Mm1)) if Mm1 > 1 else float("nan")
    return {"d_eff": d_eff, "top1_frac": top1_frac, "spec_entropy": spec_entropy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, choices=list(CATEGORY_CONFIG.keys()))
    ap.add_argument("--n_test", type=int, default=3)
    ap.add_argument("--max_splits", type=int, default=84)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = CATEGORY_CONFIG[args.category]
    mixed_data = load_response_matrix_dict(Path(cfg["mixed_dir"]))
    anchor_data = load_response_matrix_dict(Path(cfg["anchor_dir"]))
    panel = mixed_data["models"]
    if panel != anchor_data["models"]:
        print(f"WARN panel mismatch")
    M = len(panel)
    n_test = args.n_test
    n_train = M - n_test

    print(f"\n{'='*100}")
    print(f"  CATEGORY: {args.category}  |  LOO (n_train={n_train}, n_test={n_test})")
    print(f"  Filter: AVG[{FILTER_LO},{FILTER_HI}] on {n_train}-panel mean")
    print(f"  Metrics evaluated on {n_test} held-out models")
    print(f"{'='*100}")

    mixed_items = sorted({pid for (_, pid) in mixed_data["raw"].keys()})
    anchor_items_pool = sorted({pid for (_, pid) in anchor_data["raw"].keys()})

    K_mix, N_mix = build_kn_matrix(mixed_data["raw"], panel, mixed_items)
    K_anc, N_anc = build_kn_matrix(anchor_data["raw"], panel, anchor_items_pool)

    all_splits = list(itertools.combinations(range(M), n_test))
    if len(all_splits) > args.max_splits:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(all_splits), size=args.max_splits, replace=False)
        all_splits = [all_splits[i] for i in idx]
    n_splits = len(all_splits)
    print(f"  splits: {n_splits}")

    metrics_list = ["res_var", "res_jsd", "res_pr", "saturation",
                    "d_eff", "top1_frac", "spec_entropy"]
    direction = {"res_var": "higher", "res_jsd": "higher", "res_pr": "higher",
                 "saturation": "lower", "d_eff": "higher",
                 "top1_frac": "lower", "spec_entropy": "higher"}

    results = {kind: {k: [] for k in metrics_list} for kind in ["anchor", "mixed", "delta"]}
    n_filt = {"anchor": [], "mixed": []}

    for test_idx in all_splits:
        train_idx = [i for i in range(M) if i not in set(test_idx)]
        # Anchor filter from train panel
        p_anc_train_mean = (K_anc[train_idx] / np.maximum(N_anc[train_idx], 1)).mean(axis=0)
        keep_anc = (p_anc_train_mean >= FILTER_LO) & (p_anc_train_mean <= FILTER_HI)
        K_anc_test = K_anc[list(test_idx)][:, keep_anc]
        N_anc_test = N_anc[list(test_idx)][:, keep_anc]

        p_mix_train_mean = (K_mix[train_idx] / np.maximum(N_mix[train_idx], 1)).mean(axis=0)
        keep_mix = (p_mix_train_mean >= FILTER_LO) & (p_mix_train_mean <= FILTER_HI)
        K_mix_test = K_mix[list(test_idx)][:, keep_mix]
        N_mix_test = N_mix[list(test_idx)][:, keep_mix]

        n_filt["anchor"].append(int(keep_anc.sum()))
        n_filt["mixed"].append(int(keep_mix.sum()))

        m_anc = {**compute_score_metrics(K_anc_test, N_anc_test),
                 **compute_diversity_metrics(K_anc_test, N_anc_test)}
        m_mix = {**compute_score_metrics(K_mix_test, N_mix_test),
                 **compute_diversity_metrics(K_mix_test, N_mix_test)}

        for k in metrics_list:
            results["anchor"][k].append(m_anc[k])
            results["mixed"][k].append(m_mix[k])
            results["delta"][k].append(m_anc[k] - m_mix[k])

    a_n = np.array(n_filt["anchor"])
    m_n = np.array(n_filt["mixed"])
    print(f"\n{'-'*100}")
    print(f"  Filter shrinkage")
    print(f"{'-'*100}")
    print(f"  anchor pool ({len(anchor_items_pool)}) -> filt: {a_n.mean():.0f} ± {a_n.std():.1f}  "
          f"(min {a_n.min()}, max {a_n.max()})")
    print(f"  mixed pool  ({len(mixed_items)}) -> filt: {m_n.mean():.0f} ± {m_n.std():.1f}  "
          f"(min {m_n.min()}, max {m_n.max()})")

    print(f"\n{'-'*100}")
    print(f"  Metrics on {n_test} held-out models  (anchor vs mixed_filt; paired by split)")
    print(f"{'-'*100}")
    print(f"  {'metric':<14} {'anchor (mean ± std)':<22} {'mixed (mean ± std)':<22} "
          f"{'delta (mean ± std)':<22} {'wins':<14} {'verdict':<14}")

    summary = {}
    for k in metrics_list:
        a = np.array(results["anchor"][k])
        m = np.array(results["mixed"][k])
        d = np.array(results["delta"][k])
        mask = ~(np.isnan(a) | np.isnan(m))
        a, m, d = a[mask], m[mask], d[mask]
        if len(a) == 0:
            continue
        if direction[k] == "higher":
            wins = int((d > 0).sum())
        else:
            wins = int((d < 0).sum())
        n = len(a)
        d_lo, d_hi = np.percentile(d, [2.5, 97.5])
        if direction[k] == "higher":
            verdict = "ANCHOR *" if d_lo > 0 else ("mixed *" if d_hi < 0 else "tie")
        else:
            verdict = "ANCHOR *" if d_hi < 0 else ("mixed *" if d_lo > 0 else "tie")
        print(f"  {k:<14} {a.mean():.4f} ± {a.std():.4f}      "
              f"{m.mean():.4f} ± {m.std():.4f}      "
              f"{d.mean():+.4f} ± {d.std():.4f}      "
              f"{wins:>3}/{n:<3} ({100*wins/n:.0f}%)   {verdict}")
        summary[k] = {
            "anchor_mean": float(a.mean()), "anchor_std": float(a.std()),
            "mixed_mean": float(m.mean()), "mixed_std": float(m.std()),
            "delta_mean": float(d.mean()), "delta_std": float(d.std()),
            "delta_q025": float(d_lo), "delta_q975": float(d_hi),
            "wins": wins, "n": n, "verdict": verdict,
        }

    out = {
        "category": args.category,
        "n_train": n_train, "n_test": n_test, "n_splits": n_splits,
        "panel": panel,
        "n_anchor_pool": len(anchor_items_pool),
        "n_mixed_pool": len(mixed_items),
        "n_filt_anchor": n_filt["anchor"],
        "n_filt_mixed": n_filt["mixed"],
        "summary": summary,
        "per_split": {
            kind: {k: results[kind][k] for k in metrics_list}
            for kind in ["anchor", "mixed", "delta"]
        },
    }
    out_path = args.output or f"/tmp/loo_validation_{args.category}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
