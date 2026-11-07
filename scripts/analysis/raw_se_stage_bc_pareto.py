"""Stage B Pareto frontier + Stage C winner under raw SE_β (no penalty).

Produces the §8.5.2 / paper-§app:filter-ablation Pareto-frequency table
(2026-05-17 second revision, both SE penalties removed).

Inputs:
  results/filter_comparison_20260428/fir_kernel_4cat.json
    (= 4-cat report from scripts/analysis/fir_kernel_compare.py;
     each row carries n_kept, bSE_A=penalized SE_β, FIR_real)

Outputs:
  stdout — per-cat Pareto frontier + cross-cat frequency table under both
  PENALIZED (legacy) and RAW (current paper) conventions.

Used by:
  - paper/neurips_2026.tex §app:filter-ablation L1269-1290 (frequency table
    + per-cat landscape narrative)
  - 毕业论文_实验数据汇总.txt §8.5.2 (raw Pareto frontier members per cat)
"""
import json
import math

N_REF = 200.0
RAW_GATE = 5.0  # paper Stage A SE_β<5 gate; saturated cells hit 10^6 anyway

with open('results/filter_comparison_20260428/fir_kernel_4cat.json') as f:
    reports = json.load(f)

# Same Stage A admission as paper (saturation + pseudo-R² + Infit pass)
STAGE_A_ADMITTED = {
    "HRR": {"NO_FILTER": False, "AVG[.05,.95]": False, "AVG[.10,.90]": False,
            "AVG[.20,.80]": True, "AVG[.30,.70]": False, "AVG[.40,.60]": False,
            "β[±0.25]": True, "β[±0.50]": True, "β[±1.00]": False, "β[±1.50]": False,
            "ALO[.20,.80]": True, "ALO[.30,.70]": True,
            "ATLAS-LV": False, "ATLAS-PB": False, "ATLAS-CB": False},
    "OR":  {"NO_FILTER": False, "AVG[.05,.95]": True, "AVG[.10,.90]": True,
            "AVG[.20,.80]": True, "AVG[.30,.70]": True, "AVG[.40,.60]": False,
            "β[±0.25]": True, "β[±0.50]": True, "β[±1.00]": True, "β[±1.50]": True,
            "ALO[.20,.80]": True, "ALO[.30,.70]": True,
            "ATLAS-LV": True, "ATLAS-PB": True, "ATLAS-CB": True},
    "CIF": {"NO_FILTER": False, "AVG[.05,.95]": True, "AVG[.10,.90]": True,
            "AVG[.20,.80]": True, "AVG[.30,.70]": True, "AVG[.40,.60]": False,
            "β[±0.25]": True, "β[±0.50]": True, "β[±1.00]": True, "β[±1.50]": True,
            "ALO[.20,.80]": True, "ALO[.30,.70]": True,
            "ATLAS-LV": True, "ATLAS-PB": True, "ATLAS-CB": True},
    "SR":  {"NO_FILTER": False, "AVG[.05,.95]": True, "AVG[.10,.90]": False,
            "AVG[.20,.80]": False, "AVG[.30,.70]": False, "AVG[.40,.60]": False,
            "β[±0.25]": False, "β[±0.50]": False, "β[±1.00]": False, "β[±1.50]": True,
            "ALO[.20,.80]": True, "ALO[.30,.70]": True,
            "ATLAS-LV": True, "ATLAS-PB": True, "ATLAS-CB": True},
}


def pareto_2d(pts):
    """pts: list of (filter, bSE, FIR). Return list of filters on (bSE↓, FIR↑) Pareto frontier."""
    keep = []
    for i, (fi, bi, ri) in enumerate(pts):
        dominated = False
        for j, (fj, bj, rj) in enumerate(pts):
            if i == j:
                continue
            if (bj <= bi and rj >= ri) and (bj < bi or rj > ri):
                dominated = True
                break
        if not dominated:
            keep.append(fi)
    return keep


for use_raw in [False, True]:
    print(f"\n{'='*80}")
    print(f"  SE_β convention: {'RAW (no penalty)' if use_raw else 'PENALIZED (legacy)'}")
    print(f"{'='*80}")
    pareto_by_cat = {}
    fir_by_cat_filter = {c: {} for c in ['HRR','OR','CIF','SR']}
    for rep in reports:
        cat = rep['category']
        pts = []
        for r in rep['rows']:
            f = r['filter']
            if not STAGE_A_ADMITTED.get(cat, {}).get(f, False):
                continue
            n = r['n_kept']
            pen = r['bSE_A']
            if pen is None or n <= 0:
                continue
            if use_raw:
                bse = pen * math.sqrt(n / N_REF)
            else:
                bse = pen
            if bse > RAW_GATE:  # extra safety
                continue
            pts.append((f, bse, r['FIR_real']))
            fir_by_cat_filter[cat][f] = r['FIR_real']
        frontier = pareto_2d(pts)
        pareto_by_cat[cat] = set(frontier)
        print(f"  {cat:<6s} Pareto ({len(frontier)} members): {frontier}")

    # Stage C: Pareto frequency across 4 cats
    all_filters = set()
    for s in pareto_by_cat.values():
        all_filters.update(s)
    counts = {f: sum(1 for c in ['HRR','OR','CIF','SR'] if f in pareto_by_cat[c]) for f in all_filters}
    print(f"\n  Stage C frequency table:")
    for f, c in sorted(counts.items(), key=lambda x: -x[1]):
        marks = "".join(["✓ " if f in pareto_by_cat[cat] else "— " for cat in ['HRR','OR','CIF','SR']])
        print(f"    {f:<20s} {marks}  count={c}/4")

    # Stage C tie-break: max count, then max mean-FIR over 4 cats
    # (equivalent intuition: among admissible filters, prefer the more permissive)
    max_count = max(counts.values())
    tied = [f for f, c in counts.items() if c == max_count]
    if len(tied) == 1:
        winner, tie_break_note = tied[0], "no tie"
    else:
        mean_fir = {f: sum(fir_by_cat_filter[c].get(f, 0.0) for c in ['HRR','OR','CIF','SR']) / 4 for f in tied}
        winner = max(mean_fir.items(), key=lambda x: x[1])[0]
        tie_break_note = "mean-FIR break: " + ", ".join(f"{f}={v:.3f}" for f, v in sorted(mean_fir.items(), key=lambda x: -x[1]))
    print(f"\n  ★ Stage C winner: {winner}  ({max_count}/4)   [{tie_break_note}]")
