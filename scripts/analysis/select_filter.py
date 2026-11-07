#!/usr/bin/env python3
"""
Filter selection pipeline over per-category filter_comparison_*.json outputs.

Pipeline (per category, Stage A → B → C):
  Stage A  Hard gates:  Infit_bern_pass ≥ --gate-infit (default 0.95)
                        pseudo_R²       ≥ --gate-pr2   (default 0.0)
                        mean_z²_bern    ≤ --gate-mz2   (default 1.5)
  Stage B  Pareto frontier on (bSE ascending, FIR descending)
  Stage C  Report top-N rows by bSE ascending, with parallel columns:
              #bSE  = rank by budget_SE (lower = better)
              #FIR  = rank by FIR       (higher = better)
              P     = on Pareto frontier? (✓)

Two rank sets:
  per-cat    (default)  each category independent; cross-cat summary follows
  aggregate  (--agg)    --agg-cats worst|mean over a subset of cats,
                        then one Pareto + rank on the aggregated (bSE, FIR)

D-category fallback:
  When Stage A empties out, --gate-fallback=show (default) ranks all 15 rows
  on bSE/FIR without gates and prints a banner; --gate-fallback=skip blanks
  the category.

Usage
-----
    python scripts/analysis/select_filter.py \
        --dir results/filter_comparison_20260423 \
        [--top-n 5] \
        [--gate-infit 0.95 --gate-pr2 0.0 --gate-mz2 1.5] \
        [--gate-fallback show|skip] \
        [--agg] [--agg-cats A,A',B] [--agg-method worst|mean]
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Filename stem → canonical category label
CAT_MAP = {
    "harmful_refusal":               "A",
    "over_refusal":                  "A'",
    "constrained_completion_M11":    "B",
    "constrained_completion":        "B",
    "sycophancy_resistance":         "D",
}

CAT_ORDER = ["A", "A'", "B", "D"]


@dataclass
class FilterRow:
    name: str
    n_kept: int
    pass_rate: float
    bse: float
    fir: float
    infit_bern: float   # pass rate, already ∈ [0, 1]
    outfit_bern: float
    pr2: float
    mz2_bern: float
    mz2_bin: float

    def gates_pass(self, g_infit: float, g_pr2: float, g_mz2: float) -> bool:
        return (
            self.infit_bern >= g_infit
            and self.pr2 >= g_pr2
            and self.mz2_bern <= g_mz2
        )


def canonical_name(raw: str) -> str:
    """Strip cat-specific panel-θ suffix so β filters collapse across cats.
    'β [θmin-0.25, θmax+0.25]  ([-0.82, +0.61])' → 'β [θmin-0.25, θmax+0.25]'
    """
    idx = raw.find("  (")
    return raw[:idx].strip() if idx != -1 else raw.strip()


def load_cat_json(path: str | os.PathLike) -> list[FilterRow]:
    with open(path) as f:
        data = json.load(f)
    rows: list[FilterRow] = []
    for r in data["filters"]:
        rows.append(FilterRow(
            name=canonical_name(r["name"]),
            n_kept=int(r["n_kept"]),
            pass_rate=float(r.get("pass_rate", float("nan"))),
            bse=float(r.get("budget_se", float("nan"))),
            fir=float(r.get("fir", float("nan"))),
            infit_bern=float(r.get("infit_bernoulli_pass_rate") or float("nan")),
            outfit_bern=float(r.get("outfit_bernoulli_pass_rate") or float("nan")),
            pr2=float(r.get("pseudo_r2") if r.get("pseudo_r2") is not None else float("nan")),
            mz2_bern=float(r.get("mean_z_sq_bernoulli") or float("nan")),
            mz2_bin=float(r.get("mean_z_sq") or float("nan")),
        ))
    return rows


def discover_categories(dir_path: str) -> dict[str, list[FilterRow]]:
    """Scan dir for known *.json files and load them into {cat_label: rows}."""
    found: dict[str, list[FilterRow]] = {}
    for p in sorted(Path(dir_path).glob("*.json")):
        stem = p.stem
        label = CAT_MAP.get(stem)
        if label is None:
            print(f"  [skip unknown file] {p.name}")
            continue
        # If B has both M=6 and M=11 present, prefer M=11
        if label == "B" and label in found and "_M11" not in stem:
            continue
        if label == "B" and label in found and "_M11" in stem:
            found[label] = load_cat_json(p)
            continue
        found[label] = load_cat_json(p)
    return found


def pareto_mask(bse: Sequence[float], fir: Sequence[float]) -> list[bool]:
    """(lower bse, higher fir) Pareto. Returns bool list."""
    n = len(bse)
    on = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_eq = (bse[j] <= bse[i]) and (fir[j] >= fir[i])
            strictly  = (bse[j] <  bse[i]) or  (fir[j] >  fir[i])
            if better_eq and strictly:
                on[i] = False
                break
    return on


def rank_with_ties(values: Sequence[float], ascending: bool) -> list[int]:
    """1-based dense rank; ties share the min rank within their group."""
    idx = sorted(range(len(values)), key=lambda i: values[i] if ascending else -values[i])
    ranks = [0] * len(values)
    for pos, i in enumerate(idx):
        if pos > 0 and values[i] == values[idx[pos - 1]]:
            ranks[i] = ranks[idx[pos - 1]]
        else:
            ranks[i] = pos + 1
    return ranks


def print_cat_block(cat_label: str, rows: list[FilterRow], args) -> dict:
    """Run Stage A→B→C for one category, print the result, return summary dict."""
    n_all = len(rows)
    gates = [r.gates_pass(args.gate_infit, args.gate_pr2, args.gate_mz2) for r in rows]
    n_pass = sum(gates)

    banner = f"[{cat_label}]  stage A survivors: {n_pass}/{n_all}"
    fallback = False
    if n_pass > 0:
        kept = [(r, i) for i, (r, g) in enumerate(zip(rows, gates)) if g]
    elif args.gate_fallback == "show":
        kept = [(r, i) for i, r in enumerate(rows)]
        fallback = True
        banner += "  [GATE-FALLBACK: showing all rows for info, ranks ignore gates]"
    else:
        print(f"\n  {banner}\n  (skipped — --gate-fallback=skip)")
        return {"cat": cat_label, "rows": [], "fallback": False, "skipped": True}

    kept_rows = [k[0] for k in kept]
    orig_idx  = [k[1] for k in kept]
    bse_vals = [r.bse for r in kept_rows]
    fir_vals = [r.fir for r in kept_rows]

    on_p   = pareto_mask(bse_vals, fir_vals)
    rb_bse = rank_with_ties(bse_vals, ascending=True)
    rb_fir = rank_with_ties(fir_vals, ascending=False)
    n_pareto = sum(on_p)

    # Sort display by bse ascending (tie: fir descending)
    order = sorted(range(len(kept_rows)),
                   key=lambda i: (bse_vals[i], -fir_vals[i]))
    order_top = order[: args.top_n]

    print(f"\n  {banner}")
    print(f"  Pareto frontier: {n_pareto}/{len(kept_rows)}")
    hdr = (f"    {'#bSE':>4}  {'#FIR':>4}  {'P':>1}  "
           f"{'filter':<30}  {'n':>4}  {'bSE':>7}  {'FIR':>5}  "
           f"{'Inf%':>5}  {'Out%':>5}  {'pR²':>7}  {'mz²bn':>6}  {'gates':>5}")
    print(hdr)
    print("    " + "─" * (len(hdr) - 4))
    for i in order_top:
        r = kept_rows[i]
        oi = orig_idx[i]
        p_mark = "✓" if on_p[i] else " "
        gate_mark = "PASS" if gates[oi] else ("FAIL" if not fallback else "fb")
        name_short = (r.name[:28] + "..") if len(r.name) > 30 else r.name
        print(f"    {rb_bse[i]:>4}  {rb_fir[i]:>4}  {p_mark:>1}  "
              f"{name_short:<30}  {r.n_kept:>4}  {r.bse:>7.4f}  {r.fir:>5.3f}  "
              f"{r.infit_bern*100:>5.1f}  {r.outfit_bern*100:>5.1f}  "
              f"{r.pr2:>7.3f}  {r.mz2_bern:>6.3f}  {gate_mark:>5}")

    return {
        "cat": cat_label,
        "rows": kept_rows,
        "orig_idx": orig_idx,
        "on_p": on_p,
        "rb_bse": rb_bse,
        "rb_fir": rb_fir,
        "gates_pass_all_rows": gates,
        "fallback": fallback,
        "skipped": False,
    }


def print_cross_cat_summary(per_cat: dict[str, dict],
                            all_rows_by_cat: dict[str, list[FilterRow]],
                            args) -> None:
    """For each filter name, show gate-pass count and per-cat (rb_bse, rb_fir, on_P)."""
    all_names: list[str] = []
    for cat in CAT_ORDER:
        if cat not in all_rows_by_cat:
            continue
        for r in all_rows_by_cat[cat]:
            if r.name not in all_names:
                all_names.append(r.name)

    present_cats = [c for c in CAT_ORDER if c in per_cat and not per_cat[c]["skipped"]]

    print("\n" + "=" * 100)
    print("  CROSS-CATEGORY SUMMARY  (δ scheme: parallel rank_bSE / rank_FIR / on-Pareto)")
    print("=" * 100)
    hdr_cats = "  ".join(f"{c:<10}" for c in present_cats)
    print(f"  {'filter':<30}  {'gates':>5}  {hdr_cats}")
    print(f"  {'':<30}  {'':>5}  " + "  ".join(f"{'#bSE/#FIR/P':<10}" for _ in present_cats))
    print("  " + "─" * 100)

    for name in all_names:
        # gates pass count across categories that actually have this filter
        gpass = 0
        gtotal = 0
        cells = []
        for cat in present_cats:
            info = per_cat[cat]
            # find index in kept_rows by name
            idx = next((k for k, r in enumerate(info["rows"]) if r.name == name), None)
            # also look up in the ALL rows to get gate-pass truth even if it failed
            full_rows = all_rows_by_cat.get(cat, [])
            full_idx = next((k for k, r in enumerate(full_rows) if r.name == name), None)
            if full_idx is not None:
                gtotal += 1
                gp = full_rows[full_idx].gates_pass(args.gate_infit, args.gate_pr2, args.gate_mz2)
                if gp:
                    gpass += 1
            if idx is None:
                cells.append("   —      ")
                continue
            rbse = info["rb_bse"][idx]
            rfir = info["rb_fir"][idx]
            p_mark = "✓" if info["on_p"][idx] else " "
            cells.append(f"{rbse:>2}/{rfir:>2}/{p_mark:>1}    ")
        name_short = (name[:28] + "..") if len(name) > 30 else name
        print(f"  {name_short:<30}  {gpass}/{gtotal:<3}  " + "  ".join(cells))

    print("\n  Legend:  #bSE = rank by budget_SE ascending;  #FIR = rank by FIR descending;")
    print("           P ✓ = on (bSE↓, FIR↑) Pareto frontier within Stage-A survivors.")
    print("           gates x/y = passes 3 hard gates in x out of y categories where the filter appears.")


# ---------------------------------------------------------------------------
# Aggregate mode
# ---------------------------------------------------------------------------

def aggregate_rows(all_rows_by_cat: dict[str, list[FilterRow]],
                   agg_cat_labels: list[str],
                   method: str) -> list[FilterRow]:
    """Build synthetic FilterRow per filter name aggregated across agg_cat_labels."""
    common: set[str] | None = None
    for cat in agg_cat_labels:
        if cat not in all_rows_by_cat:
            continue
        names = {r.name for r in all_rows_by_cat[cat]}
        common = names if common is None else common & names
    if not common:
        return []

    out = []
    for name in common:
        bse_list, fir_list, inf_list, out_list, pr2_list, mz2_list, mz2b_list, n_list = [], [], [], [], [], [], [], []
        for cat in agg_cat_labels:
            rows = all_rows_by_cat.get(cat, [])
            r = next((x for x in rows if x.name == name), None)
            if r is None:
                continue
            bse_list.append(r.bse); fir_list.append(r.fir)
            inf_list.append(r.infit_bern); out_list.append(r.outfit_bern)
            pr2_list.append(r.pr2); mz2_list.append(r.mz2_bern); mz2b_list.append(r.mz2_bin)
            n_list.append(r.n_kept)
        if not bse_list:
            continue
        if method == "worst":
            agg_bse = max(bse_list); agg_fir = min(fir_list)
            agg_inf = min(inf_list); agg_out = min(out_list)
            agg_pr2 = min(pr2_list); agg_mz2 = max(mz2_list); agg_mz2b = max(mz2b_list)
        elif method == "mean":
            agg_bse = sum(bse_list)/len(bse_list); agg_fir = sum(fir_list)/len(fir_list)
            agg_inf = sum(inf_list)/len(inf_list); agg_out = sum(out_list)/len(out_list)
            agg_pr2 = sum(pr2_list)/len(pr2_list); agg_mz2 = sum(mz2_list)/len(mz2_list)
            agg_mz2b = sum(mz2b_list)/len(mz2b_list)
        else:
            raise ValueError(f"unknown agg method: {method}")
        out.append(FilterRow(
            name=name,
            n_kept=int(sum(n_list)/len(n_list)),
            pass_rate=float("nan"),
            bse=agg_bse, fir=agg_fir,
            infit_bern=agg_inf, outfit_bern=agg_out,
            pr2=agg_pr2, mz2_bern=agg_mz2, mz2_bin=agg_mz2b,
        ))
    # preserve original filter order (use the first agg_cat as reference)
    first_cat = next((c for c in agg_cat_labels if c in all_rows_by_cat), None)
    if first_cat is not None:
        ref_order = [r.name for r in all_rows_by_cat[first_cat]]
        out.sort(key=lambda r: ref_order.index(r.name) if r.name in ref_order else 999)
    return out


def main():
    ap = argparse.ArgumentParser(description="Filter selection pipeline (Stage A→B→C) on per-category filter_comparison JSONs.")
    ap.add_argument("--dir", required=True, help="Directory containing {category}.json files")
    ap.add_argument("--top-n", type=int, default=5, help="Top-N per category to display (default 5)")
    ap.add_argument("--gate-infit", type=float, default=0.95, help="Axis III: Infit_bern pass rate ≥ this (default 0.95)")
    ap.add_argument("--gate-pr2",   type=float, default=0.0,  help="Axis IV primary: pseudo_R² ≥ this (default 0.0)")
    ap.add_argument("--gate-mz2",   type=float, default=1.5,  help="Axis IV secondary: mean_z²_bern ≤ this (default 1.5)")
    ap.add_argument("--gate-fallback", choices=["show", "skip"], default="show",
                    help="When Stage A is empty: 'show' = rank all rows without gates (default); 'skip' = blank the category")
    ap.add_argument("--agg", action="store_true",
                    help="Also run aggregate mode: combine cats into a single (bSE, FIR) Pareto")
    ap.add_argument("--agg-cats", default="A,A',B",
                    help="Comma-separated cats for aggregate mode (default 'A,A\\',B' — skips D)")
    ap.add_argument("--agg-method", choices=["worst", "mean"], default="worst",
                    help="Aggregation across cats: 'worst' = max bSE / min FIR (default); 'mean' = arithmetic mean")
    args = ap.parse_args()

    cats = discover_categories(args.dir)
    if not cats:
        print(f"No recognized filter_comparison JSON files under {args.dir}.")
        return

    print("=" * 100)
    print(f"  Filter Selection  —  dir={args.dir}")
    print(f"  Hard gates:  Infit_bern ≥ {args.gate_infit}   pseudo_R² ≥ {args.gate_pr2}   mean_z²_bern ≤ {args.gate_mz2}")
    print(f"  Top-N:       {args.top_n}   gate-fallback: {args.gate_fallback}")
    print("=" * 100)

    per_cat: dict[str, dict] = {}
    for cat in CAT_ORDER:
        if cat in cats:
            per_cat[cat] = print_cat_block(cat, cats[cat], args)

    print_cross_cat_summary(per_cat, cats, args)

    # ---- aggregate mode ----
    if args.agg:
        agg_cats = [c.strip() for c in args.agg_cats.split(",") if c.strip()]
        print("\n" + "=" * 100)
        print(f"  AGGREGATE MODE  cats={agg_cats}  method={args.agg_method}")
        print("=" * 100)
        agg_rows = aggregate_rows(cats, agg_cats, args.agg_method)
        if not agg_rows:
            print("  No common filters across the requested cats.")
        else:
            print_cat_block(f"AGG-{args.agg_method}", agg_rows, args)


if __name__ == "__main__":
    main()
