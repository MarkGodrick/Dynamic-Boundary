"""Fit IRT (Rasch) model on response matrix data.

Thin CLI wrapper around difficulty.rasch.

Usage:
    python scripts/fit_irt.py --data results/validation/edge_case
    python scripts/fit_irt.py --data results/validation/jbb_easy
    python scripts/fit_irt.py --data results/validation/jbb_hard
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difficulty.rasch import (
    load_response_matrix, fit_ols, fit_mle, fit_baseline, compute_fit_stats,
)


def main():
    parser = argparse.ArgumentParser(description="Fit IRT (Rasch) model on response matrix")
    parser.add_argument("--data", required=True, help="Directory with model subdirs containing response_matrix.jsonl")
    parser.add_argument("--output", default=None, help="Output JSON path (default: {data}/irt_results.json)")
    args = parser.parse_args()

    output_path = args.output or str(Path(args.data) / "irt_results.json")

    # Load
    rm = load_response_matrix(args.data)
    models = rm["models"]
    items = rm["items"]
    K, N, p_hat = rm["pass_counts"], rm["total_trials"], rm["p_hat"]
    print(f"Response matrix: {len(models)} models × {len(items)} items")

    # Fit
    print("\nFitting OLS...")
    theta_ols, beta_ols = fit_ols(p_hat)

    print("Fitting MLE...")
    theta_mle, beta_mle, nll = fit_mle(K, N, theta_ols, beta_ols)

    print("Fitting baseline...")
    a_avg, d_avg = fit_baseline(p_hat)

    stats = compute_fit_stats(p_hat, K, N, theta_ols, beta_ols, theta_mle, beta_mle, d_avg, nll)

    # Print summary
    print(f"\n{'='*60}")
    print("  Model abilities (θ) — higher = stronger")
    print(f"{'='*60}")
    print(f"  {'Model':<50} {'θ_OLS':>8} {'θ_MLE':>8} {'A_avg':>8}")
    print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*8}")
    for i in np.argsort(theta_mle)[::-1]:
        print(f"  {models[i]:<50} {theta_ols[i]:>8.3f} {theta_mle[i]:>8.3f} {a_avg[i]:>8.3f}")

    print(f"\n  Item difficulty (β) — higher = harder")
    print(f"  OLS:  min={beta_ols.min():.3f}  med={np.median(beta_ols):.3f}  max={beta_ols.max():.3f}")
    print(f"  MLE:  min={beta_mle.min():.3f}  med={np.median(beta_mle):.3f}  max={beta_mle.max():.3f}")
    print(f"  Davg: min={d_avg.min():.3f}  med={np.median(d_avg):.3f}  max={d_avg.max():.3f}")

    print(f"\n  Fit statistics:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    # Save
    result = {
        "models": {
            m: {
                "theta_ols": round(float(theta_ols[i]), 4),
                "theta_mle": round(float(theta_mle[i]), 4),
                "ability_avg": round(float(a_avg[i]), 4),
            }
            for i, m in enumerate(models)
        },
        "items": {
            p: {
                "beta_ols": round(float(beta_ols[j]), 4),
                "beta_mle": round(float(beta_mle[j]), 4),
                "d_avg": round(float(d_avg[j]), 4),
            }
            for j, p in enumerate(items)
        },
        "fit_stats": stats,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
