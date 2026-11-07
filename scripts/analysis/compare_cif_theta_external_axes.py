#!/usr/bin/env python3
"""Rank-agreement between CIF theta_hat and external references (HiES W1).

The closed-weight targets have no published parameter counts and their release
dates tie, so neither can order them. We therefore compare the calibrated CIF
theta_hat against three external references:

  IFEval   public IFEval leaderboard score (llm-stats.com). Reported for
           transparency: it DISAGREES with theta_hat. The leaderboard band is
           narrow (0.83-0.90) and non-monotone within the Gemma-3 family, i.e.
           it is saturated on these models and cannot resolve them.
  B price  provider output-token price (USD / 1M tokens, OpenRouter). The only
           axis covering all five closed targets.
  C arena  Arena (formerly LMArena/Chatbot Arena) text Elo. Covers three of
           the five; Nova Lite / Nova Micro have no Arena entry.
  A tier   within-vendor capability tier as positioned by the provider.
           Compared only inside a vendor (GPT vs GPT, Nova vs Nova), never
           across vendors, since tiers carry no shared scale.

Rank agreement is Spearman rho / Kendall tau against theta_hat. For axis A,
which only yields ordered pairs inside a vendor, we report pairwise accuracy
and its tau-equivalent (2*acc - 1) so it can enter the mean.

Sources are pinned by URL + access date in SOURCES below; every score is
transcribed, not scraped at runtime, so the numbers stay reproducible.

Usage:
    python scripts/analysis/compare_cif_theta_external_axes.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ACCESSED = "2026-07-28"

SOURCES = {
    "ifeval": ("https://llm-stats.com/benchmarks/ifeval",
               "self-reported, 0 of 65 entries independently verified"),
    "price": ("https://openrouter.ai/api/v1/models",
              "provider list price, output tokens, USD per 1M"),
    "arena": ("https://arena.ai/leaderboard/text",
              "text Elo; Nova Lite / Nova Micro absent from the board"),
    "tier": ("provider product documentation",
             "within-vendor tier only; no cross-vendor scale"),
}

# Closed-weight rebuttal targets. None has a published parameter count.
CLOSED_TARGETS = [
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4.1-mini",
    "openai/gpt-4o-mini",
    "amazon/nova-lite-v1",
    "amazon/nova-micro-v1",
]

# Axis B: OpenRouter output-token list price, USD per 1M tokens.
PRICE_OUT = {
    "anthropic/claude-haiku-4.5": 5.000,
    "openai/gpt-4.1-mini": 1.600,
    "openai/gpt-4o-mini": 0.600,
    "amazon/nova-lite-v1": 0.240,
    "amazon/nova-micro-v1": 0.140,
}

# Axis C: Arena text Elo. Nova Lite / Nova Micro have no entry.
ARENA_ELO = {
    "anthropic/claude-haiku-4.5": 1392,
    "openai/gpt-4.1-mini": 1340,
    "openai/gpt-4o-mini": 1287,
}

# Axis A: within-vendor tier as positioned by the provider (higher = stronger).
VENDOR_TIERS = {
    "openai": {"openai/gpt-4.1-mini": 2, "openai/gpt-4o-mini": 1},
    "amazon": {"amazon/nova-lite-v1": 2, "amazon/nova-micro-v1": 1},
}

# IFEval leaderboard scores, for the disagreement note.
IFEVAL = {
    "openai/gpt-4.1-mini": 0.841,
    "amazon/nova-lite-v1": 0.897,
    "amazon/nova-micro-v1": 0.872,
}


def load_theta(results_dir: Path) -> dict[str, tuple[float, float]]:
    """Read {model: (theta_hat, se)} from every finished run."""
    out: dict[str, tuple[float, float]] = {}
    for path in sorted(results_dir.glob("*/result.json")):
        with open(path) as f:
            d = json.load(f)
        out[d["target_model"]] = (
            float(d["theta_hat_final"]), float(d["theta_se_final"]),
        )
    return out


def agreement(theta: dict[str, tuple[float, float]],
              axis: dict[str, float]) -> dict:
    """Spearman / Kendall agreement between theta_hat and one external axis."""
    models = [m for m in theta if m in axis]
    th = np.array([theta[m][0] for m in models])
    ax = np.array([axis[m] for m in models])
    rho = spearmanr(th, ax)
    tau = kendalltau(th, ax)
    return {
        "n": len(models),
        "models": models,
        "rho": float(rho.statistic),
        "rho_p": float(rho.pvalue),
        "tau": float(tau.statistic),
    }


def vendor_tier_accuracy(theta: dict[str, tuple[float, float]]) -> dict:
    """Pairwise ordering accuracy of within-vendor tiers."""
    pairs: list[dict] = []
    for vendor, tiers in VENDOR_TIERS.items():
        members = [m for m in tiers if m in theta]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                tier_says = tiers[a] > tiers[b]
                theta_says = theta[a][0] > theta[b][0]
                pairs.append({
                    "vendor": vendor, "a": a, "b": b,
                    "theta_a": theta[a][0], "theta_b": theta[b][0],
                    "consistent": tier_says == theta_says,
                })
    n_ok = sum(1 for p in pairs if p["consistent"])
    acc = n_ok / len(pairs) if pairs else float("nan")
    return {"pairs": pairs, "n_ok": n_ok, "n": len(pairs),
            "accuracy": acc, "tau_equiv": 2 * acc - 1}
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir",
        default="results/rebuttal/closed_models/constrained_completion",
    )
    ap.add_argument(
        "--output",
        default="results/rebuttal/closed_models/cif_theta_external_axes.csv",
    )
    args = ap.parse_args()

    theta = load_theta(PROJECT_ROOT / args.results_dir)
    closed = {m: v for m, v in theta.items() if m in CLOSED_TARGETS}
    if not closed:
        print(f"no closed-target runs found under {args.results_dir}")
        return 1

    print(f"Sources (accessed {ACCESSED}):")
    for key, (url, note) in SOURCES.items():
        print(f"  {key:<7} {url}\n          {note}")
    print()

    print(f"{'model':<28} {'theta':>8} {'SE':>7} {'price':>7} "
          f"{'arena':>6} {'IFEval':>7}")
    for m in sorted(closed, key=lambda x: -closed[x][0]):
        th, se = closed[m]
        pr = PRICE_OUT.get(m)
        el = ARENA_ELO.get(m)
        ie = IFEVAL.get(m)
        print(f"{m:<28} {th:>+8.3f} {se:>7.4f} "
              f"{(f'{pr:.3f}' if pr else '--'):>7} "
              f"{(str(el) if el else '--'):>6} "
              f"{(f'{ie:.3f}' if ie else '--'):>7}")

    axes = {
        "B price": agreement(closed, PRICE_OUT),
        "C arena": agreement(closed, ARENA_ELO),
    }
    ifeval_res = agreement(closed, IFEVAL)
    tier = vendor_tier_accuracy(closed)

    print("\n=== rank agreement with CIF theta_hat ===")
    for name, r in axes.items():
        print(f"  {name:<10} n={r['n']}  rho={r['rho']:+.4f} "
              f"(p={r['rho_p']:.4f})  tau={r['tau']:+.4f}")
    print(f"  {'A tier':<10} n={tier['n']} pairs  "
          f"accuracy={tier['accuracy']:.3f} "
          f"({tier['n_ok']}/{tier['n']})  tau_equiv={tier['tau_equiv']:+.4f}")
    print(f"  {'IFEval':<10} n={ifeval_res['n']}  "
          f"rho={ifeval_res['rho']:+.4f} "
          f"(p={ifeval_res['rho_p']:.4f})  tau={ifeval_res['tau']:+.4f}"
          "   <-- DISAGREES")

    print("\n  within-vendor tier detail:")
    for p in tier["pairs"]:
        mark = "ok " if p["consistent"] else "NO "
        print(f"    {mark} {p['vendor']:<7} "
              f"{p['a'].split('/')[-1]} ({p['theta_a']:+.3f}) vs "
              f"{p['b'].split('/')[-1]} ({p['theta_b']:+.3f})")

    taus = [axes["B price"]["tau"], axes["C arena"]["tau"],
            tier["tau_equiv"]]
    print(f"\n  mean tau over B, C, A = {np.mean(taus):+.4f}")
    print(f"  mean rho over B, C     = "
          f"{np.mean([axes['B price']['rho'], axes['C arena']['rho']]):+.4f}")

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "theta_hat", "se", "price_out_usd_per_1m",
                    "arena_elo", "ifeval_score"])
        for m in sorted(closed, key=lambda x: -closed[x][0]):
            th, se = closed[m]
            w.writerow([m, f"{th:.6f}", f"{se:.6f}",
                        PRICE_OUT.get(m, ""), ARENA_ELO.get(m, ""),
                        IFEVAL.get(m, "")])
        w.writerow([])
        w.writerow(["axis", "n", "spearman_rho", "p_value", "kendall_tau"])
        for name, r in axes.items():
            w.writerow([name, r["n"], f"{r['rho']:.6f}",
                        f"{r['rho_p']:.6f}", f"{r['tau']:.6f}"])
        w.writerow(["A tier (pairwise acc)", tier["n"],
                    f"{tier['accuracy']:.6f}", "", f"{tier['tau_equiv']:.6f}"])
        w.writerow(["IFEval (disagrees)", ifeval_res["n"],
                    f"{ifeval_res['rho']:.6f}", f"{ifeval_res['rho_p']:.6f}",
                    f"{ifeval_res['tau']:.6f}"])
        w.writerow(["mean tau (B,C,A)", "", "", "", f"{np.mean(taus):.6f}"])

    print(f"\nWrote {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
