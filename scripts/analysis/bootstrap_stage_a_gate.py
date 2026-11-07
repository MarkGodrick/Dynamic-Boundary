"""Trial-level non-parametric bootstrap CI for the Stage A Infit_bern gate.

For a (cat, filter) cell:
  1. Apply filter mask to the response matrix → kept items.
  2. For b = 1..B:
       resample 40 trials within each (m, j) cell with replacement
         (≡ K_b[m, j] ~ Binomial(N=40, p=K_orig[m, j]/40))
       refit 1PL on (M=9, J_kept) → θ_b, β_b
       compute Infit_bern[j] for j in kept items
       record pass_rate_b = #(Infit_bern ∈ [0.5, 1.5]) / J_kept
  3. Report 95% percentile CI on pass rate.

Used to characterize sampling noise around the Stage A gate (≥ 0.95) for
borderline cells (notably SR / AVG[0.05, 0.95] = 0.944, vs gate 0.95).

If 0.95 ∈ CI → the formal pass/fail decision is sampling-noise-bound;
filter_method_spec.md §I.3 can cite the CI to justify treating SR's gate
fail as "effective pass" while keeping AVG[0.05, 0.95] as the unified
4-cat anchor (see also B1: §D Stage C narrative tie-break threshold).

Run:
    conda run --no-capture-output -n boundary python \\
        scripts/analysis/bootstrap_stage_a_gate.py \\
        --comparison-dir results/filter_comparison_20260428 \\
        --filter "AVG [0.05, 0.95]" \\
        --B 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.result_paths import filter_ablation_dir
from difficulty.rasch import load_response_matrix, fit_ols, fit_mle
from difficulty.irt_diagnostics import (
    compute_standardized_residuals,
    compute_item_fit_diagnostics,
)


CAT_ORDER = [
    ("harmful_refusal",         "HR"),
    ("over_refusal",            "OR"),
    ("constrained_completion",  "CC"),
    ("sycophancy_resistance",   "SR"),
]

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


# ---------------------------------------------------------------------------
# Filter mask (mirrors compare_filters.py & build_filter_comparison_summary.py)
# ---------------------------------------------------------------------------

def _point_biserial_panel_total(p_hat: np.ndarray) -> np.ndarray:
    panel_total = p_hat.sum(axis=1)
    p_centered = p_hat - p_hat.mean(axis=0, keepdims=True)
    t_centered = panel_total - panel_total.mean()
    num = (p_centered * t_centered[:, None]).sum(axis=0)
    denom = np.sqrt((p_centered ** 2).sum(axis=0) * (t_centered ** 2).sum())
    return np.where(denom > 0, num / denom, 0.0)


def filter_mask(cfg: dict, p_hat: np.ndarray, beta_full: np.ndarray) -> np.ndarray:
    t = cfg["type"]
    J = p_hat.shape[1]
    if t == "none":
        return np.ones(J, dtype=bool)
    if t == "avg":
        pm = p_hat.mean(axis=0)
        return (pm >= cfg["lo"]) & (pm <= cfg["hi"])
    if t == "beta":
        return (beta_full >= cfg["lo"]) & (beta_full <= cfg["hi"])
    if t == "at_least_one":
        return ((p_hat >= cfg["lo"]) & (p_hat <= cfg["hi"])).any(axis=0)
    if t == "atlas_low_variance":
        std = p_hat.std(axis=0, ddof=0)
        pm = p_hat.mean(axis=0)
        return (std >= cfg["std_threshold"]) & (pm >= cfg["mean_lo"]) & (pm <= cfg["mean_hi"])
    if t == "atlas_point_biserial":
        r = _point_biserial_panel_total(p_hat)
        return np.abs(r) >= cfg["r_threshold"]
    if t == "atlas_combined":
        std = p_hat.std(axis=0, ddof=0)
        pm = p_hat.mean(axis=0)
        r = _point_biserial_panel_total(p_hat)
        lv = (std >= cfg["std_threshold"]) & (pm >= cfg["mean_lo"]) & (pm <= cfg["mean_hi"])
        return lv & (np.abs(r) >= cfg["r_threshold"])
    raise ValueError(f"Unknown filter type: {t!r}")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def fit_1pl(K: np.ndarray, N: np.ndarray):
    p_hat = K / np.clip(N, 1, None)
    th0, bt0 = fit_ols(p_hat)
    th, bt, _ = fit_mle(K, N, th0, bt0)
    return th, bt


def infit_bern_pass_rate(K: np.ndarray, N: np.ndarray) -> tuple[float, bool]:
    """Returns (pass_rate, ok). ok=False if fit failed."""
    try:
        th, bt = fit_1pl(K, N)
    except Exception:
        return float("nan"), False
    z = compute_standardized_residuals(K, N, th, bt)
    diag = compute_item_fit_diagnostics(z, K, N, th, bt)
    return float(diag["infit_bernoulli_pass_rate"]), True


def load_response_for_cat(cat: str):
    src = filter_ablation_dir(cat)
    data = load_response_matrix(str(src))
    keep_idx = [i for i, m in enumerate(data["models"]) if m in M9_PANEL]
    if len(keep_idx) != 9:
        raise RuntimeError(f"{cat}: expected 9 M9_PANEL members, found {len(keep_idx)}")
    K = data["pass_counts"][keep_idx]
    N = data["total_trials"][keep_idx]
    p_hat = K / np.clip(N, 1, None)
    return K, N, p_hat


def bootstrap_one(K: np.ndarray, N: np.ndarray, mask: np.ndarray,
                  B: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    K_kept = K[:, mask]
    N_kept = N[:, mask]
    if K_kept.shape[1] == 0:
        return {"observed": float("nan"), "rates": [], "n_ok": 0}

    obs, _ = infit_bern_pass_rate(K_kept, N_kept)

    p_cell = K_kept / np.clip(N_kept, 1, None)
    N_int = N_kept.astype(np.int64)
    rates = []
    n_ok = 0
    t0 = time.time()
    for b in range(B):
        K_b = rng.binomial(N_int, p_cell).astype(np.float64)
        rate, ok = infit_bern_pass_rate(K_b, N_kept)
        if ok:
            rates.append(rate)
            n_ok += 1
        if (b + 1) % 100 == 0:
            el = time.time() - t0
            print(f"    {b+1}/{B}  rate={n_ok/(b+1):.0%}  el={el:.0f}s", flush=True)
    return {"observed": obs, "rates": rates, "n_ok": n_ok}


def summarize(rates: list[float]) -> dict:
    if not rates:
        return {"mean": float("nan"), "median": float("nan"),
                "ci_lo_95": float("nan"), "ci_hi_95": float("nan"),
                "p005": float("nan"), "p995": float("nan")}
    a = np.asarray(rates)
    return {
        "mean":     float(a.mean()),
        "median":   float(np.median(a)),
        "ci_lo_95": float(np.percentile(a, 2.5)),
        "ci_hi_95": float(np.percentile(a, 97.5)),
        "p005":     float(np.percentile(a, 0.5)),
        "p995":     float(np.percentile(a, 99.5)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison-dir",
                    default="results/filter_comparison_20260428")
    ap.add_argument("--filter", default=None,
                    help="Filter name to bootstrap (matches compare_filters JSON 'name'). "
                         "If omitted with --all, runs on every filter that failed Infit_bern gate.")
    ap.add_argument("--all", action="store_true",
                    help="Iterate over every filter in every cat that has "
                         "observed Infit_bern < gate (i.e., uniform quasi-B)")
    ap.add_argument("--cats", nargs="+", default=None,
                    help="Cat tags to run; default = all 4")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gate", type=float, default=0.95)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cmp_dir = Path(args.comparison_dir).resolve()
    if not cmp_dir.is_absolute():
        cmp_dir = (ROOT / cmp_dir).resolve()
    out_path = Path(args.out) if args.out else cmp_dir / f"bootstrap_stage_a_{args.B}.json"

    target_cats = args.cats or [tag for _, tag in CAT_ORDER]
    cats_to_run = [(c, t) for c, t in CAT_ORDER if t in target_cats]

    if not args.all and args.filter is None:
        ap.error("must pass --filter NAME or --all")

    summary = {
        "B": args.B,
        "seed": args.seed,
        "gate": args.gate,
        "mode": "all" if args.all else "single",
        "filter_single": args.filter,
        "results": [],  # list of per-cell entries
    }

    # Pre-load each cat's data once (expensive)
    cat_data = {}
    cat_json = {}
    for cat, tag in cats_to_run:
        json_path = cmp_dir / f"{cat}_M9.json"
        if not json_path.exists():
            print(f"  [skip] {tag}: missing {json_path}")
            continue
        with json_path.open() as f:
            cat_json[tag] = json.load(f)
        cat_data[tag] = (cat, *load_response_for_cat(cat))

    for tag, (cat, K, N, p_hat) in cat_data.items():
        cmp_data = cat_json[tag]
        th_full, bt_full = fit_1pl(K, N)

        # Decide which filters to run
        if args.all:
            cells = [r for r in cmp_data["filters"]
                     if r["name"] != "NO FILTER"
                     and (r["infit_bernoulli_pass_rate"] or 0) < args.gate]
        else:
            row = next((r for r in cmp_data["filters"] if r["name"] == args.filter), None)
            cells = [row] if row is not None else []

        if not cells:
            print(f"\n--- {tag} ({cat}) — no cells to bootstrap (all Inf_bern ≥ gate) ---")
            continue

        for row in cells:
            cfg = row["config"]
            json_inf = row["infit_bernoulli_pass_rate"]
            print(f"\n--- {tag} ({cat}) — {row['name']}  json_inf_bern={json_inf:.4f} ---")
            mask = filter_mask(cfg, p_hat, bt_full)
            n_kept = int(mask.sum())
            print(f"  n_kept={n_kept}  (json={row['n_kept']})")

            seed = args.seed + (hash(tag) ^ hash(row["name"])) % 10**6
            res = bootstrap_one(K, N, mask, B=args.B, seed=seed)
            stats = summarize(res["rates"])
            in_ci = stats["ci_lo_95"] <= args.gate <= stats["ci_hi_95"]
            verdict = "✓ gate ∈ 95% CI" if in_ci else "✗ gate OUTSIDE 95% CI"

            print(f"  observed={res['observed']:.4f}   bootstrap mean={stats['mean']:.4f}")
            print(f"  95% CI: [{stats['ci_lo_95']:.4f}, {stats['ci_hi_95']:.4f}]   "
                  f"99% CI: [{stats['p005']:.4f}, {stats['p995']:.4f}]")
            print(f"  Stage A gate ({args.gate}):  {verdict}")

            summary["results"].append({
                "cat_tag": tag,
                "category": cat,
                "filter_name": row["name"],
                "filter_config": cfg,
                "n_kept_json": row["n_kept"],
                "n_kept_mask": n_kept,
                "infit_bern_json": json_inf,
                "infit_bern_observed": res["observed"],
                "B_ok": res["n_ok"],
                "stats": stats,
                "in_95_ci": in_ci,
                "in_99_ci": stats["p005"] <= args.gate <= stats["p995"],
            })

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"\n=== uniform quasi-B summary (gate = {args.gate}) ===")
    print(f"  {'cat':>3}  {'filter':<48}  {'obs':>6}  {'95% CI':>20}  {'verdict':<8}")
    for r in summary["results"]:
        print(f"  {r['cat_tag']:>3}  {r['filter_name'][:48]:<48}  "
              f"{r['infit_bern_observed']:>6.3f}  "
              f"[{r['stats']['ci_lo_95']:.3f}, {r['stats']['ci_hi_95']:.3f}]   "
              f"{'PASS quasi-B' if r['in_95_ci'] else 'FAIL'}")


if __name__ == "__main__":
    main()
