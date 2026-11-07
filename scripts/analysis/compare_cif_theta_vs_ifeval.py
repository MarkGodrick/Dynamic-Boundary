#!/usr/bin/env python3
"""Compare DBE CIF theta_hat against public IFEval leaderboard scores.

Two model groups share one calibrated logit scale:
  - reference panel (M=9): theta_hat read from the anchor bundle
  - closed-weight rebuttal targets: theta_hat read from each run's result.json

For every model that also appears on the public IFEval leaderboard we report
both numbers side by side, then compute rank agreement. The point of the
comparison is resolution, not ranking: IFEval is a fixed-difficulty pass rate
and saturates near its ceiling, whereas theta_hat is estimated on a calibrated
difficulty scale and keeps separating models above that ceiling.

IFEval scores are transcribed from https://llm-stats.com/benchmarks/ifeval
(all entries there are self-reported, none independently verified — see
LEADERBOARD_NOTE).

Usage:
    python scripts/analysis/compare_cif_theta_vs_ifeval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LEADERBOARD_URL = "https://llm-stats.com/benchmarks/ifeval"
LEADERBOARD_NOTE = (
    "All 65 leaderboard entries are self-reported (0 independently verified); "
    "treat the scores as a public reference point, not ground truth."
)
LEADERBOARD_ACCESSED = "2026-07-28"

# Transcribed IFEval scores, keyed by the model id used in this repo.
# rank is the leaderboard rank at LEADERBOARD_ACCESSED.
IFEVAL_SCORES: dict[str, tuple[float, int, str]] = {
    "Qwen/Qwen2.5-72B-Instruct-AWQ": (0.841, 41, "Qwen2.5 72B Instruct"),
    "google/gemma-3-27b-it": (0.904, 13, "Gemma 3 27B"),
    "google/gemma-3-12b-it": (0.889, 22, "Gemma 3 12B"),
    "google/gemma-3-4b-it": (0.902, 15, "Gemma 3 4B"),
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": (
        0.829, 47, "Mistral Small 3 24B Instruct",
    ),
    "openai/gpt-4.1-mini": (0.841, 41, "GPT-4.1 mini"),
}

GEMMA_FAMILY = [
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
]

def load_panel_theta(bundle_path: Path) -> dict[str, float]:
    """Read the panel theta_hat map from an anchor bundle."""
    with open(bundle_path) as f:
        bundle = json.load(f)
    return {k: float(v) for k, v in bundle["panel_theta"].items()}


def load_closed_targets(results_dir: Path) -> list[dict]:
    """Read every finished closed-weight run under results_dir."""
    out: list[dict] = []
    for path in sorted(results_dir.glob("*/result.json")):
        with open(path) as f:
            d = json.load(f)
        history = d.get("history") or []
        out.append({
            "model": d["target_model"],
            "theta": float(d["theta_hat_final"]),
            "se": float(d["theta_se_final"]),
            "converged": bool(d["converged"]),
            "stopped_reason": d.get("stopped_reason"),
            "iters": len(history),
            "coverage_failures": sum(
                1 for s in history if s.get("coverage_failed")
            ),
            "triggered": any(s.get("triggered") for s in history),
            "run_dir": str(path.parent.relative_to(PROJECT_ROOT)),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bundle",
        default="benchmarks/constrained_completion/anchor_alo_bundle_M9.json",
    )
    ap.add_argument(
        "--results-dir",
        default="results/rebuttal/closed_models/constrained_completion",
    )
    ap.add_argument(
        "--output",
        default="results/rebuttal/closed_models/cif_theta_vs_ifeval.csv",
    )
    args = ap.parse_args()

    panel_theta = load_panel_theta(PROJECT_ROOT / args.bundle)
    closed = load_closed_targets(PROJECT_ROOT / args.results_dir)
    panel_min, panel_max = min(panel_theta.values()), max(panel_theta.values())

    rows: list[dict] = []
    for name, theta in panel_theta.items():
        ifeval = IFEVAL_SCORES.get(name)
        rows.append({
            "model": name,
            "group": "panel",
            "theta": theta,
            "se": "",
            "ifeval": ifeval[0] if ifeval else "",
            "ifeval_rank": ifeval[1] if ifeval else "",
            "path": "panel",
            "stopped_reason": "",
        })
    for entry in closed:
        ifeval = IFEVAL_SCORES.get(entry["model"])
        rows.append({
            "model": entry["model"],
            "group": "closed_target",
            "theta": entry["theta"],
            "se": entry["se"],
            "ifeval": ifeval[0] if ifeval else "",
            "ifeval_rank": ifeval[1] if ifeval else "",
            "path": "out_of_panel" if entry["triggered"] else "in_panel",
            "stopped_reason": entry["stopped_reason"],
        })
    rows.sort(key=lambda r: -r["theta"])

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"IFEval leaderboard: {LEADERBOARD_URL} (accessed {LEADERBOARD_ACCESSED})")
    print(f"NOTE: {LEADERBOARD_NOTE}\n")
    print(f"Panel theta range: [{panel_min:+.3f}, {panel_max:+.3f}]\n")

    print(f"{'model':<46} {'group':<14} {'theta':>8} {'SE':>7} "
          f"{'IFEval':>7} {'rank':>5} {'path':<13}")
    for r in rows:
        se = f"{r['se']:.4f}" if r["se"] != "" else "--"
        ie = f"{r['ifeval']:.3f}" if r["ifeval"] != "" else "--"
        rk = str(r["ifeval_rank"]) if r["ifeval_rank"] != "" else "--"
        print(f"{r['model']:<46} {r['group']:<14} {r['theta']:>+8.3f} {se:>7} "
              f"{ie:>7} {rk:>5} {r['path']:<13}")

    paired = [r for r in rows if r["ifeval"] != ""]
    if len(paired) >= 3:
        theta = np.array([r["theta"] for r in paired])
        ifeval = np.array([r["ifeval"] for r in paired])
        rho = spearmanr(theta, ifeval)
        print(f"\nPaired models: {len(paired)}")
        print(f"Spearman rho(theta, IFEval) = {rho.statistic:+.4f} "
              f"(p = {rho.pvalue:.4f})")
        print(f"IFEval spread among paired models: "
              f"{ifeval.min():.3f}-{ifeval.max():.3f} "
              f"(range {ifeval.max() - ifeval.min():.3f})")
        print(f"theta spread among paired models:  "
              f"{theta.min():+.3f}-{theta.max():+.3f} "
              f"(range {theta.max() - theta.min():.3f} logits)")

    print("\nGemma-3 family (same architecture, 3 sizes):")
    for name in GEMMA_FAMILY:
        score, rank, _ = IFEVAL_SCORES[name]
        print(f"  {name:<26} theta={panel_theta[name]:+.3f}  "
              f"IFEval={score:.3f} (rank {rank})")
    gem_theta = [panel_theta[n] for n in GEMMA_FAMILY]
    gem_ifeval = [IFEVAL_SCORES[n][0] for n in GEMMA_FAMILY]
    print(f"  theta monotone with size: "
          f"{all(x < y for x, y in zip(gem_theta, gem_theta[1:]))} "
          f"(spread {max(gem_theta) - min(gem_theta):.3f} logits)")
    print(f"  IFEval monotone with size: "
          f"{all(x < y for x, y in zip(gem_ifeval, gem_ifeval[1:]))} "
          f"(spread {max(gem_ifeval) - min(gem_ifeval):.3f})")

    print(f"\nWrote {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
