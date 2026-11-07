"""Standalone 1PL Rasch misfit diagnostics.

Given a response matrix (panel M × items J × Binomial N-trials), fits 1PL
Rasch and reports:
  - Cell-level |z| > 2 / > 3 rates (vs null 5% / 0.3%)
  - Per-item Infit / Outfit distributions + pass-rate gates
  - Q3 pair violation rate (local independence)
  - Top-k worst-fit items with full 6-model response patterns
        (observed p̂ vs Rasch-predicted σ(θ̂ − β̂))

Runs on both (a) raw item pool and (b) retained subset after the
winner filter β [θ ± 0.50].

Output: terminal pretty-print + JSON + minimal standalone HTML table
        to `results/<category>/irt_misfit_diagnostics/`.

See docs/irt_concepts_primer.md §2.6 and docs/filter_metric_construction.md §9.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make repo root importable (mirrors compare_filters.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from difficulty.rasch import load_response_matrix, fit_ols, fit_mle
from difficulty.irt_diagnostics import (
    compute_standardized_residuals,
    compute_item_fit_diagnostics,
    compute_q3,
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Fit + diagnostics
# ---------------------------------------------------------------------------

def fit_1pl(pass_counts: np.ndarray, total_trials: np.ndarray) -> tuple:
    """Fit 1PL Rasch. Returns (theta, beta, nll)."""
    p_hat = pass_counts / np.maximum(total_trials, 1)
    theta_init, beta_init = fit_ols(p_hat)
    theta, beta, nll = fit_mle(pass_counts, total_trials,
                               theta_init, beta_init)
    return theta, beta, nll


def run_diagnostics(
    models: list[str],
    items: list[str],
    pass_counts: np.ndarray,
    total_trials: np.ndarray,
    prompt_lookup: dict,
    top_k: int = 5,
) -> dict:
    """Fit 1PL, compute all misfit metrics, return a serializable report."""
    M, J = pass_counts.shape
    if M < 2 or J < 2:
        raise ValueError(f"Degenerate matrix: M={M}, J={J}")

    theta, beta, nll = fit_1pl(pass_counts, total_trials)
    p_hat = pass_counts / np.maximum(total_trials, 1)

    # Predicted p under Rasch
    p_pred = sigmoid(theta[:, None] - beta[None, :])

    # Cell-level standardized residuals
    z = compute_standardized_residuals(pass_counts, total_trials, theta, beta)
    cell_abs_z = np.abs(z)
    cell_rate_2 = float((cell_abs_z > 2.0).mean())
    cell_rate_3 = float((cell_abs_z > 3.0).mean())

    # Per-item Infit / Outfit
    fit = compute_item_fit_diagnostics(z, pass_counts, total_trials, theta, beta)
    infit = fit["infit"]
    outfit = fit["outfit"]

    # Bucketed Infit stats
    def frac_in(arr, lo, hi):
        return float(np.mean((arr >= lo) & (arr <= hi)))

    infit_bins = {
        "[0.5, 1.5]": frac_in(infit, 0.5, 1.5),
        "[0.7, 1.3]": frac_in(infit, 0.7, 1.3),
        ">2.0": float(np.mean(infit > 2.0)),
    }
    outfit_bins = {
        "[0.5, 1.5]": frac_in(outfit, 0.5, 1.5),
        "[0.7, 1.3]": frac_in(outfit, 0.7, 1.3),
        ">2.0": float(np.mean(outfit > 2.0)),
    }

    # Q3 pairs
    q3 = compute_q3(z, threshold=0.2)

    # Top-k worst-fit items by Infit
    order = np.argsort(-infit)  # descending
    top_indices = order[:top_k].tolist()
    worst_items = []
    for j in top_indices:
        pid = items[j]
        prompt_meta = prompt_lookup.get(pid, {})
        worst_items.append({
            "prompt_id": pid,
            "prompt_preview": (prompt_meta.get("prompt") or "")[:220],
            "skill_count": prompt_meta.get("skill_count"),
            "skills": prompt_meta.get("skills") or [],
            "beta_hat": float(beta[j]),
            "infit": float(infit[j]),
            "outfit": float(outfit[j]),
            "per_model": [
                {
                    "model": models[m],
                    "theta_hat": float(theta[m]),
                    "p_observed": float(p_hat[m, j]),
                    "p_predicted": float(p_pred[m, j]),
                    "pass_count": int(pass_counts[m, j]),
                    "total_trials": int(total_trials[m, j]),
                    "z": float(z[m, j]),
                }
                for m in range(M)
            ],
        })

    # Panel-level summary
    panel_summary = []
    for m in range(M):
        z_m = z[m]
        panel_summary.append({
            "model": models[m],
            "theta_hat": float(theta[m]),
            "outfit_model": float((z_m * z_m).mean()),
            "frac_abs_z_gt2": float((np.abs(z_m) > 2.0).mean()),
        })

    report = {
        "n_models": M,
        "n_items": J,
        "nll_1pl": float(nll),
        "fit": {
            "cell_abs_z_gt2_rate": cell_rate_2,
            "cell_abs_z_gt3_rate": cell_rate_3,
            "cell_null_rate_2": 0.0455,  # 2 * (1 − Φ(2))
            "cell_null_rate_3": 0.0027,  # 2 * (1 − Φ(3))
            "infit_bins": infit_bins,
            "outfit_bins": outfit_bins,
            "mean_z_sq": fit["mean_z_sq"],
            "q3_violation_rate_0.2": q3["q3_violation_rate"],
            "q3_max": q3["q3_max"],
        },
        "panel": panel_summary,
        "beta_range": [float(beta.min()), float(beta.max())],
        "worst_items": worst_items,
        # Raw arrays for downstream histograms (saved separately)
        "_arrays": {
            "infit": infit.tolist(),
            "outfit": outfit.tolist(),
            "beta": beta.tolist(),
            "theta": theta.tolist(),
        },
    }
    return report


# ---------------------------------------------------------------------------
# Prompt lookup
# ---------------------------------------------------------------------------

def load_prompt_lookup(benchmark_path: Path) -> dict:
    """Read benchmarks/<cat>/random_baseline_k500.jsonl (or similar) → dict
    keyed by prompt_id with 'prompt', 'skills', 'skill_count' fields."""
    if not benchmark_path.exists():
        return {}
    lookup = {}
    with open(benchmark_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pid = d.get("prompt_id") or d.get("id")
            if pid is None:
                continue
            lookup[pid] = {
                "prompt": d.get("prompt"),
                "skills": d.get("skills"),
                "skill_count": d.get("skill_count"),
            }
    return lookup


# ---------------------------------------------------------------------------
# Filter: β [θ ± 0.50]
# ---------------------------------------------------------------------------

def apply_beta_filter(theta_panel: np.ndarray, beta: np.ndarray,
                      half_width: float = 0.5) -> np.ndarray:
    """Winner filter β [θ±0.50]: keep items whose β̂ is within ±half_width
    logit of the panel θ̂ range [min, max]. Matches the 'β filter' in
    compare_filters.py:build_filter_list."""
    theta_lo = theta_panel.min() - half_width
    theta_hi = theta_panel.max() + half_width
    return (beta >= theta_lo) & (beta <= theta_hi)


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_report(label: str, report: dict) -> None:
    fit = report["fit"]
    print(f"\n{'=' * 72}")
    print(f" {label}  (M={report['n_models']}, J={report['n_items']})")
    print(f"{'=' * 72}")

    print("\n[Cell-level |z| rates]")
    print(f"  |z| > 2:   {fit['cell_abs_z_gt2_rate']:.4f}  "
          f"(null ≈ {fit['cell_null_rate_2']:.4f}, "
          f"excess ×{fit['cell_abs_z_gt2_rate']/fit['cell_null_rate_2']:.1f})")
    print(f"  |z| > 3:   {fit['cell_abs_z_gt3_rate']:.4f}  "
          f"(null ≈ {fit['cell_null_rate_3']:.4f}, "
          f"excess ×{fit['cell_abs_z_gt3_rate']/fit['cell_null_rate_3']:.1f})")
    print(f"  mean z²:   {fit['mean_z_sq']:.2f}  (null = 1.0)")

    print("\n[Per-item Infit distribution]")
    for label_b, frac in fit["infit_bins"].items():
        print(f"  {label_b:<12}  {frac:.3f}")
    print("\n[Per-item Outfit distribution]")
    for label_b, frac in fit["outfit_bins"].items():
        print(f"  {label_b:<12}  {frac:.3f}")

    print("\n[Q3 local independence]")
    print(f"  |Q3| > 0.2 pairs:  {fit['q3_violation_rate_0.2']:.3f}")
    print(f"  max |Q3|:          {fit['q3_max']:.3f}")

    print("\n[Panel per-model residual summary]")
    print(f"  {'model':<52} {'θ̂':>7} {'outfit':>8} {'%|z|>2':>8}")
    for row in report["panel"]:
        print(f"  {row['model']:<52} {row['theta_hat']:+.3f}  "
              f"{row['outfit_model']:>7.2f}  {row['frac_abs_z_gt2']:>7.3f}")

    print(f"\n[Top-{len(report['worst_items'])} worst-fit items by Infit]")
    for i, item in enumerate(report["worst_items"], 1):
        print(f"\n  #{i}  prompt_id={item['prompt_id']}  "
              f"β̂={item['beta_hat']:+.2f}  "
              f"Infit={item['infit']:.2f}  Outfit={item['outfit']:.2f}  "
              f"skill_count={item['skill_count']}")
        preview = item.get("prompt_preview", "") or ""
        if preview:
            snippet = preview.replace("\n", " ")[:140]
            print(f"     prompt: {snippet}...")
        print(f"     {'model':<52} {'θ̂':>7} {'p_obs':>8} {'p_pred':>8} {'z':>6}")
        for row in item["per_model"]:
            print(f"     {row['model']:<52} {row['theta_hat']:+.3f}  "
                  f"{row['p_observed']:>6.3f}  {row['p_predicted']:>6.3f}  "
                  f"{row['z']:+.2f}")


# ---------------------------------------------------------------------------
# Minimal HTML writer (standalone, no deps)
# ---------------------------------------------------------------------------

def write_html_report(path: Path, label: str, report: dict) -> None:
    fit = report["fit"]
    M = report["n_models"]

    def fmt_pct(x):
        return f"{100*x:.1f}%"

    def fmt_num(x, p=3):
        return f"{x:+.{p}f}" if x is not None else "—"

    html = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
            f"<title>{label} — 1PL misfit</title>",
            "<style>",
            "body{font-family:-apple-system,sans-serif;max-width:1200px;"
            "margin:24px auto;padding:0 20px;color:#222}",
            "h2{border-bottom:2px solid #ccc;padding-bottom:4px;margin-top:32px}",
            "table{border-collapse:collapse;margin:12px 0;font-size:13px}",
            "td,th{border:1px solid #ddd;padding:5px 10px;text-align:right}",
            "th{background:#f4f4f4;text-align:center}",
            "td.left,th.left{text-align:left}",
            ".misfit{background:#fee}", ".bad{background:#fcc}",
            ".ok{background:#efe}",
            "code{background:#f4f4f4;padding:1px 4px}",
            ".prompt{color:#555;font-size:12px;max-width:900px}",
            "</style></head><body>"]

    html.append(f"<h1>{label}</h1>")
    html.append(f"<p>M = {M}, J = {report['n_items']}, "
                f"β range = [{report['beta_range'][0]:+.2f}, "
                f"{report['beta_range'][1]:+.2f}], "
                f"NLL = {report['nll_1pl']:.1f}</p>")

    # Aggregate table
    html.append("<h2>Aggregate fit</h2><table>")
    html.append("<tr><th class='left'>Metric</th><th>Value</th>"
                "<th>Null expectation</th></tr>")
    html.append(f"<tr><td class='left'>Cell |z| &gt; 2 rate</td>"
                f"<td>{fmt_pct(fit['cell_abs_z_gt2_rate'])}</td>"
                f"<td>{fmt_pct(fit['cell_null_rate_2'])}</td></tr>")
    html.append(f"<tr><td class='left'>Cell |z| &gt; 3 rate</td>"
                f"<td>{fmt_pct(fit['cell_abs_z_gt3_rate'])}</td>"
                f"<td>{fmt_pct(fit['cell_null_rate_3'])}</td></tr>")
    html.append(f"<tr><td class='left'>Mean z²</td>"
                f"<td>{fit['mean_z_sq']:.2f}</td><td>1.0</td></tr>")
    for bn, v in fit["infit_bins"].items():
        html.append(f"<tr><td class='left'>Infit ∈ {bn}</td>"
                    f"<td>{fmt_pct(v)}</td><td>—</td></tr>")
    for bn, v in fit["outfit_bins"].items():
        html.append(f"<tr><td class='left'>Outfit ∈ {bn}</td>"
                    f"<td>{fmt_pct(v)}</td><td>—</td></tr>")
    html.append(f"<tr><td class='left'>|Q3| &gt; 0.2 pair rate</td>"
                f"<td>{fmt_pct(fit['q3_violation_rate_0.2'])}</td>"
                f"<td>≤ 5%</td></tr>")
    html.append(f"<tr><td class='left'>max |Q3|</td>"
                f"<td>{fit['q3_max']:.3f}</td><td>—</td></tr>")
    html.append("</table>")

    # Panel summary
    html.append("<h2>Per-model residual summary</h2><table>")
    html.append("<tr><th class='left'>Model</th><th>θ̂</th>"
                "<th>Outfit_model</th><th>% |z|&gt;2</th></tr>")
    for row in report["panel"]:
        html.append(f"<tr><td class='left'><code>{row['model']}</code></td>"
                    f"<td>{row['theta_hat']:+.3f}</td>"
                    f"<td>{row['outfit_model']:.2f}</td>"
                    f"<td>{fmt_pct(row['frac_abs_z_gt2'])}</td></tr>")
    html.append("</table>")

    # Worst items
    html.append(f"<h2>Top {len(report['worst_items'])} worst-fit items</h2>")
    for i, item in enumerate(report["worst_items"], 1):
        html.append(f"<h3>#{i} &nbsp; <code>{item['prompt_id']}</code> "
                    f"&nbsp; β̂={item['beta_hat']:+.2f} &nbsp; "
                    f"Infit={item['infit']:.2f} &nbsp; "
                    f"Outfit={item['outfit']:.2f} &nbsp; "
                    f"skill_count={item['skill_count']}</h3>")
        if item.get("prompt_preview"):
            preview = item["prompt_preview"].replace("<", "&lt;").replace(
                ">", "&gt;").replace("\n", " ")
            html.append(f"<div class='prompt'>prompt: {preview}…</div>")
        html.append("<table>")
        html.append("<tr><th class='left'>Model</th><th>θ̂</th>"
                    "<th>p̂ obs</th><th>p pred</th><th>k/N</th>"
                    "<th>z</th></tr>")
        for row in item["per_model"]:
            css = "bad" if abs(row["z"]) > 3 else (
                "misfit" if abs(row["z"]) > 2 else "")
            html.append(f"<tr class='{css}'>"
                        f"<td class='left'><code>{row['model']}</code></td>"
                        f"<td>{row['theta_hat']:+.3f}</td>"
                        f"<td>{row['p_observed']:.3f}</td>"
                        f"<td>{row['p_predicted']:.3f}</td>"
                        f"<td>{row['pass_count']}/{row['total_trials']}</td>"
                        f"<td>{row['z']:+.2f}</td></tr>")
        html.append("</table>")

    html.append("</body></html>")
    path.write_text("\n".join(html))


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-dir", required=True,
                        help="e.g. results/constrained_completion/filter_ablation_raw")
    parser.add_argument("--benchmark",
                        help="optional path to jsonl with prompt text "
                             "(benchmarks/<cat>/random_baseline_k500.jsonl)")
    parser.add_argument("--output-dir", required=True,
                        help="results/<category>/irt_misfit_diagnostics")
    parser.add_argument("--label", default="category",
                        help="display label (e.g. constrained_completion)")
    parser.add_argument("--beta-half-width", type=float, default=0.5,
                        help="retained filter half-width (default 0.5)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="how many worst-fit items to show")
    args = parser.parse_args()

    response_dir = Path(args.response_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading response matrix from {response_dir} …")
    data = load_response_matrix(str(response_dir))
    models = data["models"]
    items = data["items"]
    pass_counts = data["pass_counts"]
    total_trials = data["total_trials"]
    print(f"  M = {len(models)}, J = {len(items)}")

    prompt_lookup = (load_prompt_lookup(Path(args.benchmark))
                     if args.benchmark else {})
    if prompt_lookup:
        print(f"  loaded {len(prompt_lookup)} prompt texts")

    # -- Raw --
    raw_report = run_diagnostics(
        models, items, pass_counts, total_trials,
        prompt_lookup, top_k=args.top_k,
    )
    print_report(f"{args.label} — RAW (all items)", raw_report)

    raw_json = output_dir / "raw.json"
    with open(raw_json, "w") as f:
        json.dump(raw_report, f, indent=2)
    write_html_report(output_dir / "raw.html",
                      f"{args.label} — raw (M={len(models)}, J={len(items)})",
                      raw_report)
    print(f"\n→ raw report saved to {raw_json}")

    # -- Retained (β [θ ± 0.50]) --
    #   Use panel θ from the raw fit (theta is in raw_report["_arrays"])
    theta_raw = np.array(raw_report["_arrays"]["theta"])
    beta_raw = np.array(raw_report["_arrays"]["beta"])
    mask = apply_beta_filter(theta_raw, beta_raw,
                             half_width=args.beta_half_width)
    n_kept = int(mask.sum())
    print(f"\nApplying β[θ±{args.beta_half_width}] filter: "
          f"{n_kept}/{len(items)} items kept")

    if n_kept >= 2:
        items_ret = [items[j] for j in range(len(items)) if mask[j]]
        retained_report = run_diagnostics(
            models, items_ret,
            pass_counts[:, mask], total_trials[:, mask],
            prompt_lookup, top_k=args.top_k,
        )
        print_report(f"{args.label} — RETAINED (β[θ±{args.beta_half_width}])",
                     retained_report)
        retained_json = output_dir / "retained.json"
        with open(retained_json, "w") as f:
            json.dump(retained_report, f, indent=2)
        write_html_report(
            output_dir / "retained.html",
            f"{args.label} — retained β[θ±{args.beta_half_width}] "
            f"(M={len(models)}, J={n_kept})",
            retained_report,
        )
        print(f"\n→ retained report saved to {retained_json}")
    else:
        print("  skipped retained — not enough items after filter")


if __name__ == "__main__":
    main()
