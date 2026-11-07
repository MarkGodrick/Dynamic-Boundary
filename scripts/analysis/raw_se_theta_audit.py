"""SE_θ audit: raw vs penalized + Stage A 0.1-logit gate compliance.

Produces the §8.3 / §8.4 raw SE_θ tables (2026-05-17 first revision —
SE_θ penalty removal + Stage A panel-coverage gate addition).

Inputs:
  results/filter_comparison_20260428/{harmful_refusal,over_refusal,
    constrained_completion,sycophancy_resistance}.json
    Each filter row carries:
      max_se     = raw SE_θ                     (Q2 candidate, no penalty)
      budget_se  = max_se · √(n_kept/200)       (legacy penalized convention)

Outputs:
  stdout — per-cat 15-filter table (n_kept, raw SE_θ, penalized SE_θ, FIR)
           + rank comparison under raw vs penalized + 0.1-gate pass/fail.

Used by:
  - paper/neurips_2026.tex §app:filter-ablation L1234 (SE_θ < 0.1 gate
    non-binding statement, 0.083 worst-cell value)
  - 毕业论文_实验数据汇总.txt §8.3 / §8.4 (raw SE_θ 4×15 table, spread
    comparison vs penalized)
"""
import json
import math
from pathlib import Path

CATS = [
    ("HRR", "results/filter_comparison_20260428/harmful_refusal.json"),
    ("OR",  "results/filter_comparison_20260428/over_refusal.json"),
    ("CIF", "results/filter_comparison_20260428/constrained_completion.json"),
    ("SR",  "results/filter_comparison_20260428/sycophancy_resistance.json"),
]
THRESHOLD = 0.10   # Q1: 试探 bSE_θ < 0.1 上限


def fmt(x, w=8, p=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return f"{'inf':>{w}s}"
    return f"{x:{w}.{p}f}"


def main():
    for cat_lbl, path in CATS:
        with open(path) as f:
            d = json.load(f)
        rows = d["filters"]
        print(f"\n{'='*125}")
        print(f"  Category: {cat_lbl}    n filters: {len(rows)}")
        print(f"{'='*125}")
        hdr = (f"{'filter':<30s} {'n_kept':>7s}  "
               f"{'bSE_θ_raw':>11s} {'bSE_θ_pen':>11s} {'FIR':>7s}  "
               f"{'raw<0.1?':>10s} {'pen<0.1?':>10s}")
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            name = r["name"]
            n = r["n_kept"]
            raw = r["max_se"]            # = bSE_θ_raw (no penalty)
            pen = r["budget_se"]         # = max_se · √(n_kept/200)
            fir_v = r["fir"]
            raw_pass = "PASS" if (raw is not None and raw < THRESHOLD) else "FAIL"
            pen_pass = "PASS" if (pen is not None and pen < THRESHOLD) else "FAIL"
            print(f"{name:<30s} {n:>7d}  {fmt(raw):>11s} {fmt(pen):>11s} {fmt(fir_v, 7, 4):>7s}  "
                  f"{raw_pass:>10s} {pen_pass:>10s}")

    # Per-cat ranking under both conventions
    print(f"\n\n{'='*100}")
    print(f"  ranking by bSE_θ ascending (小者优)，paper winner ALO[.20,.80]")
    print(f"{'='*100}")
    for cat_lbl, path in CATS:
        with open(path) as f:
            d = json.load(f)
        rows = d["filters"]
        # Sort by raw and penalized
        by_raw = sorted([(r["name"], r["max_se"], r["budget_se"], r["n_kept"]) for r in rows],
                        key=lambda x: x[1] if x[1] is not None else float('inf'))
        by_pen = sorted([(r["name"], r["max_se"], r["budget_se"], r["n_kept"]) for r in rows],
                        key=lambda x: x[2] if x[2] is not None else float('inf'))
        print(f"\n--- {cat_lbl} ---")
        print(f"{'rank':>5s}  {'by RAW (no penalty)':<50s}      {'by PEN (current)':<50s}")
        for i in range(len(rows)):
            n_raw, r_raw, p_raw, k_raw = by_raw[i]
            n_pen, r_pen, p_pen, k_pen = by_pen[i]
            star_raw = " ★" if n_raw == "ALO [0.20, 0.80]" else "  "
            star_pen = " ★" if n_pen == "ALO [0.20, 0.80]" else "  "
            print(f"{i+1:>5d}  {n_raw:<30s}{star_raw} (raw={r_raw:.4f}, n={k_raw})      "
                  f"{n_pen:<30s}{star_pen} (pen={p_pen:.4f}, n={k_pen})")


if __name__ == "__main__":
    main()
