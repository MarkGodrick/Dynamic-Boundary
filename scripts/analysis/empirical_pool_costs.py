#!/usr/bin/env python3
"""Empirical pool shape + two-stage split cost projection across 4 categories.

Reads each category's canonical filter_ablation dir (resolved via
`core.result_paths.filter_ablation_dir`), computes panel-mean p̄ per item
restricted to the locked M=9 panel members that are present, and projects
two-stage (N_c=10, k_min=1) cost savings under the empirical p̄ distribution.

Complementary to derive_coarse_split.py:
  derive_coarse_split.py     — parametric Beta(a, b) pool models (educational)
  empirical_pool_costs.py    — real response_matrix.jsonl data (this file)

Canonical spec consumer: docs/filter_method_spec.md §F.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom, beta as beta_dist, kstest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from core.result_paths import filter_ablation_dir

TARGET_M = 9
N_C = 7       # locked 2026-04-24: smallest feasible N_c at α=0.05, M=9, k_min=1
K_MIN = 1
N_TOTAL = 40
P_BOUNDARY = 0.05

M9_PANEL = {
    "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-8B",
    "google/gemma-3-27b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-4b-it",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}

CATEGORIES = {
    "constrained_completion": filter_ablation_dir("constrained_completion"),
    "harmful_refusal":        filter_ablation_dir("harmful_refusal"),
    "over_refusal":           filter_ablation_dir("over_refusal"),
    "sycophancy_resistance":  filter_ablation_dir("sycophancy_resistance"),
}


def load_p_bar(response_dir: Path, keep_only_m9: bool = True):
    """Read per-item panel-mean p̄ from a filter_ablation directory.

    Returns (p_bars: np.ndarray, models_used: list[str]).
    Items with at least one zero-trial cell across kept models are skipped.
    """
    model_dict = {}
    for mdir in sorted(response_dir.iterdir()):
        if not mdir.is_dir():
            continue
        meta_path = mdir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            name = json.load(f).get("model", mdir.name)
        if keep_only_m9 and name not in M9_PANEL:
            continue
        rm_path = mdir / "response_matrix.jsonl"
        if not rm_path.exists():
            continue
        items = {}
        with open(rm_path) as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                n = e.get("total_trials", 0)
                if n and n > 0:
                    items[e["prompt_id"]] = e["pass_count"] / n
        model_dict[name] = items

    if not model_dict:
        return np.array([]), []

    models = sorted(model_dict.keys())
    common = set.intersection(*[set(d.keys()) for d in model_dict.values()])
    p_bars = np.array([
        np.mean([model_dict[m][pid] for m in models]) for pid in sorted(common)
    ])
    return p_bars, models


def coarse_pass_prob(M, N_c, k_min, p):
    """P(k_min <= S <= M*N_c - k_min | S ~ Bin(M*N_c, p))."""
    MN = M * N_c
    lower = binom.cdf(k_min - 1, MN, p) if k_min > 0 else 0.0
    upper = binom.cdf(MN - k_min, MN, p)
    return float(upper - lower)


def fnr_exact(M, N_c, k_min, p=P_BOUNDARY):
    if k_min <= 0:
        return 0.0
    return float(binom.cdf(k_min - 1, M * N_c, p))


def empirical_r_c(p_bars, M, N_c, k_min):
    return float(np.mean([coarse_pass_prob(M, N_c, k_min, p) for p in p_bars]))


def beta_ks_match(p_bars):
    candidates = [
        ("uniform Beta(1,1)", 1.0, 1.0),
        ("mild-U Beta(0.7,0.7)", 0.7, 0.7),
        ("heavy-U Beta(0.3,0.3)", 0.3, 0.3),
        ("extreme Beta(0.15,0.15)", 0.15, 0.15),
    ]
    results = []
    for name, a, b in candidates:
        D, _ = kstest(p_bars, lambda x, a=a, b=b: beta_dist.cdf(x, a, b))
        results.append((name, float(D)))
    results.sort(key=lambda x: x[1])
    return results


def report_headline():
    print("=" * 92)
    print(f"Empirical pool cost projection  (M={TARGET_M}, N_c={N_C}, k_min={K_MIN}, fine=[0.05, 0.95])")
    print(f"FNR(p=0.05) under exact Binomial = {fnr_exact(TARGET_M, N_C, K_MIN):.4f}")
    print("=" * 92)
    print(f"{'category':<28} {'M_raw':>5} {'K':>5} {'r_c':>7} {'stage1':>7} "
          f"{'stage2':>7} {'total':>7} {'base':>6} {'savings':>8}")
    print("-" * 92)
    rows = []
    for cat, path in CATEGORIES.items():
        p_bars, models = load_p_bar(path, keep_only_m9=True)
        if len(p_bars) == 0:
            print(f"{cat:<28} (no M=9 overlap — raw dir missing or empty)")
            continue
        r_c = empirical_r_c(p_bars, TARGET_M, N_C, K_MIN)
        stage1 = TARGET_M * N_C
        stage2 = r_c * TARGET_M * (N_TOTAL - N_C)
        total = stage1 + stage2
        baseline = TARGET_M * N_TOTAL
        sav = 1 - total / baseline
        rows.append((cat, len(models), len(p_bars), r_c, stage1, stage2, total, baseline, sav))
        print(f"{cat:<28} {len(models):>5} {len(p_bars):>5} {r_c:>7.4f} "
              f"{stage1:>7.1f} {stage2:>7.1f} {total:>7.1f} {baseline:>6.0f} {sav:>7.1%}")
    return rows


def report_distribution(n_bins: int = 20):
    print("\n" + "=" * 92)
    print(f"Per-cat p̄ histogram ({n_bins} bins) + nearest parametric Beta (KS distance)")
    print("=" * 92)
    width = {10: 1, 20: 2, 40: 3, 50: 3}.get(n_bins, 2)
    for cat, path in CATEGORIES.items():
        p_bars, models = load_p_bar(path, keep_only_m9=True)
        if len(p_bars) == 0:
            continue
        print(f"\n--- {cat}  (M_raw={len(models)}, K={len(p_bars)}) ---")
        pcts = np.percentile(p_bars, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        labels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"  p̄ percentiles: "
              + "  ".join(f"p{L}={v:.3f}" for L, v in zip(labels, pcts)))
        # Tail focus: concentration at the extreme 2% bins of the fine filter
        frac_le_005 = float((p_bars <= 0.05).mean())
        frac_ge_095 = float((p_bars >= 0.95).mean())
        frac_in_fine = float(((p_bars > 0.05) & (p_bars < 0.95)).mean())
        print(f"  Fine-filter zones:  p̄ ≤ 0.05: {frac_le_005:.1%}   "
              f"p̄ ∈ (0.05, 0.95): {frac_in_fine:.1%}   "
              f"p̄ ≥ 0.95: {frac_ge_095:.1%}")
        hist, edges = np.histogram(p_bars, bins=n_bins, range=(0, 1))
        tot = hist.sum()
        print(f"  Histogram:")
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            frac = hist[i] / tot
            bar = "█" * int(frac * 60)
            print(f"    [{lo:.{width}f}, {hi:.{width}f})  {hist[i]:>4}  {frac:>5.1%}  {bar}")
        ks = beta_ks_match(p_bars)
        print(f"  Nearest Beta (KS): {ks[0][0]}  (D={ks[0][1]:.3f})")


def sweep_empirical(alpha: float = 0.05):
    """Sweep (N_c, k_min) over empirical per-cat r_c and report savings."""
    # Pre-load all p_bars once.
    p_bars_per_cat = {}
    for cat, path in CATEGORIES.items():
        p_bars, models = load_p_bar(path, keep_only_m9=True)
        if len(p_bars) > 0:
            p_bars_per_cat[cat] = (p_bars, len(models))

    N_c_range = range(6, 17)
    k_min_range = range(0, 4)

    print("\n" + "=" * 100)
    print(f"Empirical (N_c, k_min) sweep at M={TARGET_M}, alpha={alpha:.0%}, "
          f"fine=[0.05,0.95], N_total={N_TOTAL}")
    print("Savings = 1 - (M·N_c + r_c·M·(N_total−N_c)) / (M·N_total)")
    print("=" * 100)

    # Table 1: full sweep per cat
    cats_short = {"constrained_completion": "B",
                  "harmful_refusal": "A",
                  "over_refusal": "A'",
                  "sycophancy_resistance": "D"}
    cat_order = ["constrained_completion", "harmful_refusal",
                 "over_refusal", "sycophancy_resistance"]

    print(f"\n  {'N_c':>4} {'k_min':>5} {'FNR@.05':>8} {'ell_c':>7}   "
          + "   ".join(f"{cats_short[c]:>7}" for c in cat_order)
          + "   mean")
    print("  " + "-" * 98)

    all_rows = []  # for universal best later
    for N_c in N_c_range:
        for k_min in k_min_range:
            fnr = fnr_exact(TARGET_M, N_c, k_min)
            if fnr > alpha:
                continue
            ell_c = k_min / (TARGET_M * N_c) if k_min > 0 else 0.0
            savings = []
            for cat in cat_order:
                if cat not in p_bars_per_cat:
                    savings.append(None)
                    continue
                p_bars, _ = p_bars_per_cat[cat]
                r_c = empirical_r_c(p_bars, TARGET_M, N_c, k_min)
                total = TARGET_M * N_c + r_c * TARGET_M * (N_TOTAL - N_c)
                baseline = TARGET_M * N_TOTAL
                savings.append(1 - total / baseline)
            mean_sav = np.mean([s for s in savings if s is not None])
            all_rows.append((N_c, k_min, fnr, ell_c, savings, mean_sav))
            sav_str = "   ".join(
                f"{s*100:>6.1f}%" if s is not None else "   —"
                for s in savings)
            print(f"  {N_c:>4} {k_min:>5} {fnr:>7.2%} {ell_c:>7.4f}   "
                  f"{sav_str}   {mean_sav*100:>5.1f}%")

    # Per-cat best
    print("\n  Per-cat best (N_c, k_min):")
    for cat in cat_order:
        if cat not in p_bars_per_cat: continue
        best = max(all_rows, key=lambda r: r[4][cat_order.index(cat)])
        s = best[4][cat_order.index(cat)]
        print(f"    {cats_short[cat]:>3} ({cat:<25}):  "
              f"N_c={best[0]}, k_min={best[1]}, ell_c={best[3]:.4f}, "
              f"savings={s*100:.1f}%")

    # Universal best (argmax mean across cats)
    universal = max(all_rows, key=lambda r: r[5])
    print(f"\n  Universal best (argmax mean savings): "
          f"N_c={universal[0]}, k_min={universal[1]}, "
          f"ell_c={universal[3]:.4f}, FNR@.05={universal[2]:.2%}, "
          f"mean savings={universal[5]*100:.1f}%")
    print(f"    per-cat: "
          + ", ".join(f"{cats_short[c]}={universal[4][i]*100:.1f}%"
                      for i, c in enumerate(cat_order)
                      if universal[4][i] is not None))

    # Locked baseline for comparison
    locked = next(r for r in all_rows if r[0] == 10 and r[1] == 1)
    print(f"\n  Current locked (N_c=10, k_min=1): "
          f"mean savings={locked[5]*100:.1f}%")
    print(f"    per-cat: "
          + ", ".join(f"{cats_short[c]}={locked[4][i]*100:.1f}%"
                      for i, c in enumerate(cat_order)
                      if locked[4][i] is not None))
    print(f"\n  Δ universal − locked: "
          f"{(universal[5] - locked[5]) * 100:+.1f} pp mean savings")


def report_M11_vs_M9_for_B():
    print("\n" + "=" * 92)
    print("B-only: side-by-side M=11 (raw) vs M=9 (panel locked)")
    print("=" * 92)
    path_B = CATEGORIES["constrained_completion"]
    for label, keep_m9 in [("M=11 (all raw)", False), ("M=9 (panel locked)", True)]:
        p_bars, models = load_p_bar(path_B, keep_only_m9=keep_m9)
        M_sim = len(models)
        r_c = empirical_r_c(p_bars, M_sim, N_C, K_MIN)
        total = M_sim * N_C + r_c * M_sim * (N_TOTAL - N_C)
        baseline = M_sim * N_TOTAL
        print(f"  {label:<22}  M_sim={M_sim}  K={len(p_bars)}  r_c={r_c:.4f}  "
              f"cost={total:.1f} vs {baseline}  ({1 - total / baseline:.1%} savings)")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bins", type=int, default=20,
                    help="Histogram bin count (default 20; try 40 for finer tails)")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.05],
                    help="FNR tolerance at p=0.05 for sweep. "
                         "Can pass multiple: --alpha 0.05 0.01")
    ap.add_argument("--skip-headline", action="store_true")
    ap.add_argument("--skip-distribution", action="store_true")
    ap.add_argument("--skip-m11-vs-m9", action="store_true")
    args = ap.parse_args()
    if not args.skip_headline:
        report_headline()
    if not args.skip_distribution:
        report_distribution(n_bins=args.bins)
    if not args.skip_m11_vs_m9:
        report_M11_vs_M9_for_B()
    for alpha in args.alpha:
        sweep_empirical(alpha=alpha)


if __name__ == "__main__":
    main()
