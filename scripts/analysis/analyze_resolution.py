"""Analyze and compare resolution across benchmarks.

Compares edge_case, jbb_easy, jbb_hard benchmarks on:
  1. θ separation (model distinguishability)
  2. β distribution (difficulty coverage)
  3. Edge case group pass rates by model (boundary specificity)
  4. Heatmaps of response matrices

Usage:
    python scripts/analyze_resolution.py \
        --benchmarks results/validation/edge_case results/validation/jbb_easy results/validation/jbb_hard \
        --labels edge_case jbb_easy jbb_hard \
        --edge-case-dataset benchmarks/harmful_refusal/edge_case_dataset.jsonl \
        --output-dir results/validation/analysis
"""

import argparse
import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difficulty.rasch import load_response_matrix, fit_ols, fit_mle, fit_baseline

# Try importing matplotlib; skip plots if unavailable
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available, skipping plots")


# ---------------------------------------------------------------------------
# Model display order (by approximate size)
# ---------------------------------------------------------------------------

MODEL_SIZE_ORDER = [
    "google/gemma-3-4b-it",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "google/gemma-3-12b-it",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
]

MODEL_SHORT_NAMES = {
    "google/gemma-3-4b-it": "Gemma3-4B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "R1-8B",
    "google/gemma-3-12b-it": "Gemma3-12B",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "Mistral-24B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "Qwen3-30B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "R1-70B",
}


def short_name(model: str) -> str:
    return MODEL_SHORT_NAMES.get(model, model.split("/")[-1])


def sort_models(models: list[str]) -> list[str]:
    """Sort models by size order."""
    order = {m: i for i, m in enumerate(MODEL_SIZE_ORDER)}
    return sorted(models, key=lambda m: order.get(m, 999))


# ---------------------------------------------------------------------------
# Resolution metrics
# ---------------------------------------------------------------------------

def compute_resolution_metrics(
    theta: np.ndarray, models: list[str], label: str,
) -> dict:
    """Compute metrics measuring how well a benchmark distinguishes models."""
    M = len(theta)

    # θ range
    theta_range = float(theta.max() - theta.min())

    # Pairwise separation
    pairs = list(combinations(range(M), 2))
    separations = [abs(theta[i] - theta[j]) for i, j in pairs]
    avg_separation = float(np.mean(separations))
    min_separation = float(np.min(separations))

    # Effective discrimination: pairs with |Δθ| > 0.5
    n_distinguished = sum(1 for s in separations if s > 0.5)

    return {
        "benchmark": label,
        "n_models": M,
        "theta_range": round(theta_range, 4),
        "avg_pairwise_separation": round(avg_separation, 4),
        "min_pairwise_separation": round(min_separation, 4),
        "n_distinguished_pairs": n_distinguished,
        "total_pairs": len(pairs),
    }


# ---------------------------------------------------------------------------
# Claim 1: Edge case β ≈ source model θ
# ---------------------------------------------------------------------------

def compute_claim1_source_beta(
    models: list[str],
    items: list[str],
    theta: np.ndarray,
    beta: np.ndarray,
    dataset_path: str,
) -> dict | None:
    """Test Claim 1: mean β of ECs from source model k correlates with θ_k.

    For each source model that produced edge cases, compute the mean β across
    its ECs (in the edge_case fit). Then correlate (θ_panel, mean_β) over the
    set of source models.

    Returns:
        Dict with per-source rows, Spearman ρ, Pearson r. None if dataset
        unavailable or no overlap.
    """
    if not Path(dataset_path).exists():
        return None

    with open(dataset_path) as f:
        dataset = [json.loads(l) for l in f if l.strip()]

    # Group prompt_ids by source_model
    source_groups: dict[str, list[str]] = {}
    for entry in dataset:
        src = entry.get("source_model")
        if not src:
            continue
        source_groups.setdefault(src, []).append(entry["prompt_id"])

    item_idx = {p: j for j, p in enumerate(items)}
    model_idx = {m: i for i, m in enumerate(models)}

    rows = []
    for src_model, pids in sorted(source_groups.items()):
        if src_model not in model_idx:
            continue
        col_indices = [item_idx[p] for p in pids if p in item_idx]
        if not col_indices:
            continue
        mean_beta = float(np.mean(beta[col_indices]))
        theta_src = float(theta[model_idx[src_model]])
        rows.append({
            "source_model": src_model,
            "n_edge_cases": len(col_indices),
            "theta_panel": round(theta_src, 4),
            "mean_beta_of_ECs": round(mean_beta, 4),
        })

    if len(rows) < 2:
        return None

    thetas = np.array([r["theta_panel"] for r in rows])
    betas = np.array([r["mean_beta_of_ECs"] for r in rows])

    rho, rho_p = spearmanr(thetas, betas)
    r, r_p = pearsonr(thetas, betas)

    return {
        "n_source_models": len(rows),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(rho_p), 4),
        "pearson_r": round(float(r), 4),
        "pearson_p": round(float(r_p), 4),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Plot 1: β distribution comparison
# ---------------------------------------------------------------------------

def plot_beta_distributions(
    all_betas: dict[str, np.ndarray], output_dir: Path,
) -> None:
    """Histogram of β values across benchmarks."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, len(all_betas), figsize=(5 * len(all_betas), 4), sharey=True)
    if len(all_betas) == 1:
        axes = [axes]

    colors = {"edge_case": "#2196F3", "jbb_easy": "#4CAF50", "jbb_hard": "#F44336"}

    for ax, (label, beta) in zip(axes, all_betas.items()):
        color = colors.get(label, "#9E9E9E")
        ax.hist(beta, bins=20, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(np.median(beta), color="black", linestyle="--", linewidth=1)
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_xlabel("β (item difficulty)")
        if ax == axes[0]:
            ax.set_ylabel("Count")

    fig.suptitle("Item Difficulty (β) Distribution by Benchmark", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "beta_distributions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved beta_distributions.png")


# ---------------------------------------------------------------------------
# Plot 2: θ comparison across benchmarks
# ---------------------------------------------------------------------------

def plot_theta_comparison(
    all_thetas: dict[str, tuple[list[str], np.ndarray]], output_dir: Path,
) -> None:
    """Bar chart of θ values across benchmarks, models sorted by size."""
    if not HAS_MPL:
        return

    # Get common models
    all_model_sets = [set(models) for models, _ in all_thetas.values()]
    common_models = sorted(
        set.intersection(*all_model_sets),
        key=lambda m: MODEL_SIZE_ORDER.index(m) if m in MODEL_SIZE_ORDER else 999,
    )

    labels = list(all_thetas.keys())
    x = np.arange(len(common_models))
    width = 0.8 / len(labels)
    colors = {"edge_case": "#2196F3", "jbb_easy": "#4CAF50", "jbb_hard": "#F44336"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, label in enumerate(labels):
        models, theta = all_thetas[label]
        model_to_theta = dict(zip(models, theta))
        vals = [model_to_theta.get(m, 0) for m in common_models]
        ax.bar(x + i * width, vals, width, label=label,
               color=colors.get(label, "#9E9E9E"), alpha=0.8)

    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels([short_name(m) for m in common_models], rotation=30, ha="right")
    ax.set_ylabel("θ (model ability)")
    ax.set_title("Model Ability (θ) by Benchmark")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_dir / "theta_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved theta_comparison.png")


# ---------------------------------------------------------------------------
# Plot 3: Edge case group pass rates
# ---------------------------------------------------------------------------

def plot_edge_case_group_passrates(
    data_dir: str, dataset_path: str, output_dir: Path,
) -> None:
    """Line plot: pass rate per edge-case source group, X=models by size."""
    if not HAS_MPL:
        return

    # Load dataset to get source_model grouping
    with open(dataset_path) as f:
        dataset = [json.loads(l) for l in f if l.strip()]

    # Group prompt_ids by source_model
    source_groups = {}
    for entry in dataset:
        src = entry.get("source_model", "unknown")
        source_groups.setdefault(src, []).append(entry["prompt_id"])

    # Load response matrix
    rm = load_response_matrix(data_dir)
    models = rm["models"]
    items = rm["items"]
    p_hat = rm["p_hat"]
    item_idx = {p: j for j, p in enumerate(items)}

    # Sort models by size
    sorted_models = sort_models(models)
    model_order = [models.index(m) for m in sorted_models]

    # For each source group, compute average p_hat per model
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(source_groups)))

    for color, (src_model, pids) in zip(colors, sorted(source_groups.items())):
        col_indices = [item_idx[pid] for pid in pids if pid in item_idx]
        if not col_indices:
            continue
        avg_p_per_model = p_hat[:, col_indices].mean(axis=1)
        ordered_p = [avg_p_per_model[i] for i in model_order]

        ax.plot(
            range(len(sorted_models)), ordered_p,
            marker="o", linewidth=2, markersize=6,
            label=f"EC from {short_name(src_model)}",
            color=color,
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(range(len(sorted_models)))
    ax.set_xticklabels([short_name(m) for m in sorted_models], rotation=30, ha="right")
    ax.set_ylabel("Average Pass Rate")
    ax.set_xlabel("Model (by size →)")
    ax.set_title("Pass Rate by Edge Case Source Group")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_dir / "edge_case_group_passrates.png", dpi=150)
    plt.close(fig)
    print(f"  Saved edge_case_group_passrates.png")


# ---------------------------------------------------------------------------
# Plot 4: Response matrix heatmap
# ---------------------------------------------------------------------------

def plot_response_heatmap(
    data_dir: str, label: str, output_dir: Path,
) -> None:
    """Heatmap of p_hat values, models sorted by θ, items sorted by β."""
    if not HAS_MPL:
        return

    rm = load_response_matrix(data_dir)
    models = rm["models"]
    items = rm["items"]
    p_hat = rm["p_hat"]

    theta_ols, beta_ols = fit_ols(p_hat)

    # Sort: models by θ descending, items by β ascending
    m_order = np.argsort(theta_ols)[::-1]
    j_order = np.argsort(beta_ols)

    sorted_p = p_hat[m_order][:, j_order]
    sorted_models = [short_name(models[i]) for i in m_order]

    fig, ax = plt.subplots(figsize=(max(8, len(items) * 0.15), max(4, len(models) * 0.6)))
    im = ax.imshow(sorted_p, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(sorted_models)))
    ax.set_yticklabels(sorted_models, fontsize=9)
    ax.set_xlabel("Items (sorted by β →)")
    ax.set_title(f"Response Matrix: {label}")
    fig.colorbar(im, ax=ax, label="Pass Rate", shrink=0.8)

    fig.tight_layout()
    fig.savefig(output_dir / f"heatmap_{label}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved heatmap_{label}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze resolution across benchmarks")
    parser.add_argument(
        "--benchmarks", nargs="+", required=True,
        help="Directories with response matrix data (e.g., results/validation/edge_case)",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Labels for each benchmark (default: directory names)",
    )
    parser.add_argument(
        "--edge-case-dataset", default="benchmarks/harmful_refusal/edge_case_dataset.jsonl",
        help="Edge case dataset for group analysis",
    )
    parser.add_argument(
        "--output-dir", default="results/validation/analysis",
        help="Output directory for results and plots",
    )
    args = parser.parse_args()

    labels = args.labels or [Path(b).name for b in args.benchmarks]
    assert len(labels) == len(args.benchmarks), "labels must match benchmarks count"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Fit IRT for each benchmark ---
    all_thetas = {}
    all_betas = {}
    all_items = {}
    all_metrics = []

    for bench_dir, label in zip(args.benchmarks, labels):
        print(f"\n{'='*60}")
        print(f"  {label}: {bench_dir}")
        print(f"{'='*60}")

        try:
            rm = load_response_matrix(bench_dir)
        except ValueError as e:
            print(f"  Skipping: {e}")
            continue

        theta_ols, beta_ols = fit_ols(rm["p_hat"])
        theta_mle, beta_mle, _ = fit_mle(
            rm["pass_counts"], rm["total_trials"], theta_ols, beta_ols,
        )

        all_thetas[label] = (rm["models"], theta_mle)
        all_betas[label] = beta_mle
        all_items[label] = rm["items"]

        metrics = compute_resolution_metrics(theta_mle, rm["models"], label)
        all_metrics.append(metrics)

        print(f"  θ range: {metrics['theta_range']:.3f}")
        print(f"  Avg pairwise separation: {metrics['avg_pairwise_separation']:.3f}")
        print(f"  Distinguished pairs: {metrics['n_distinguished_pairs']}/{metrics['total_pairs']}")

    # --- Print comparison table ---
    print(f"\n{'='*60}")
    print("  Resolution Comparison")
    print(f"{'='*60}")
    print(f"  {'Benchmark':<15} {'θ range':>10} {'Avg sep':>10} {'Dist pairs':>12}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*12}")
    for m in all_metrics:
        print(f"  {m['benchmark']:<15} {m['theta_range']:>10.3f} "
              f"{m['avg_pairwise_separation']:>10.3f} "
              f"{m['n_distinguished_pairs']:>5}/{m['total_pairs']:<5}")

    # --- Save metrics ---
    with open(output_dir / "resolution_comparison.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Saved resolution_comparison.json")

    # --- Claim 1: Edge case β ≈ source model θ ---
    print(f"\n{'='*60}")
    print("  Claim 1: Edge case β vs source model θ")
    print(f"{'='*60}")
    for label in labels:
        if "edge_case" not in label or label not in all_thetas:
            continue
        models_ec, theta_ec = all_thetas[label]
        beta_ec = all_betas[label]
        items_ec = all_items[label]
        claim1 = compute_claim1_source_beta(
            models_ec, items_ec, theta_ec, beta_ec, args.edge_case_dataset,
        )
        if claim1 is None:
            print("  Skipped (dataset missing or insufficient overlap)")
            break

        print(f"  {'Source model':<50} {'n_EC':>5} {'θ_panel':>10} {'mean β':>10}")
        print(f"  {'-'*50} {'-'*5} {'-'*10} {'-'*10}")
        for r in claim1["rows"]:
            print(f"  {short_name(r['source_model']):<50} "
                  f"{r['n_edge_cases']:>5} "
                  f"{r['theta_panel']:>10.3f} "
                  f"{r['mean_beta_of_ECs']:>10.3f}")
        print(f"\n  Spearman ρ = {claim1['spearman_rho']:.3f}  (p={claim1['spearman_p']:.3f})")
        print(f"  Pearson  r = {claim1['pearson_r']:.3f}  (p={claim1['pearson_p']:.3f})")

        with open(output_dir / "claim1_source_beta.json", "w") as f:
            json.dump(claim1, f, indent=2)
        print(f"\n  Saved claim1_source_beta.json")
        break

    # --- Generate plots ---
    print("\nGenerating plots...")

    if all_betas:
        plot_beta_distributions(all_betas, output_dir)

    if all_thetas:
        plot_theta_comparison(all_thetas, output_dir)

    # Edge case group pass rates (only if edge_case benchmark present)
    for bench_dir, label in zip(args.benchmarks, labels):
        if "edge_case" in label and Path(args.edge_case_dataset).exists():
            plot_edge_case_group_passrates(bench_dir, args.edge_case_dataset, output_dir)
            break

    # Heatmaps for each benchmark
    for bench_dir, label in zip(args.benchmarks, labels):
        try:
            plot_response_heatmap(bench_dir, label, output_dir)
        except ValueError:
            pass

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
