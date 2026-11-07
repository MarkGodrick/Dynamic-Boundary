"""Reference panel (M=9) per-category θ̂ under ALO[0.20, 0.80] filter.

Replaces the AVG[0.05, 0.95] panel-θ̂ table in paper §panel-selection with the
ALO-filter analogue for thesis Table 4.1. Sum-to-zero θ gauge.

Run:
  conda run -n boundary --no-capture-output python -u \\
    scripts/analysis/panel_theta_alo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from difficulty.rasch import fit_mle, fit_ols  # noqa: E402


PANEL = [
    "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "google/gemma-3-27b-it",
    "Qwen/Qwen3-8B",
    "google/gemma-3-12b-it",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "google/gemma-3-4b-it",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
]
SHORT = [
    "Qwen2.5-72B-AWQ", "Qwen3-30B-A3B", "Gemma-3-27B", "Qwen3-8B", "Gemma-3-12B",
    "Mistral-Small-3.1-24B", "Gemma-3-4B", "Mistral-Nemo-12B", "DS-R1-Distill-Llama-8B",
]
FAMILY = ["Qwen-2.5", "Qwen-3", "Gemma-3", "Qwen-3", "Gemma-3",
          "Mistral-S", "Gemma-3", "Mistral-N", "DS-Distill"]
PARAMS = ["72B (int4)", "30B MoE", "27B", "8B", "12B",
          "24B", "4B", "12B", "8B"]

CATS = [
    ("A",  "harmful_refusal",        "filter_ablation_raw_v2_judgefix"),
    ("A'", "over_refusal",           "filter_ablation_raw_v2_judgefix"),
    ("B",  "constrained_completion", "filter_ablation_raw_v2"),
    ("C",  "sycophancy_resistance",  "filter_ablation_raw_v2"),
]
ALO_LO, ALO_HI = 0.20, 0.80


def _model_dir(m: str) -> str:
    return m.replace("/", "_").replace("-", "_")


def load_panel_matrix(base: Path, panel: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    per_model: dict[str, dict[str, dict]] = {}
    for m in panel:
        rm = base / _model_dir(m) / "response_matrix.jsonl"
        rows: dict[str, dict] = {}
        for ln in rm.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            rows[r["prompt_id"]] = r
        per_model[m] = rows
    common = sorted(set.intersection(*(set(r) for r in per_model.values())))
    M, J = len(panel), len(common)
    K = np.zeros((M, J))
    N = np.zeros((M, J))
    for i, m in enumerate(panel):
        for j, pid in enumerate(common):
            r = per_model[m][pid]
            K[i, j] = r["pass_count"]
            N[i, j] = r["total_trials"]
    return K, N, J


def alo_mask(p_hat: np.ndarray, lo: float = ALO_LO, hi: float = ALO_HI) -> np.ndarray:
    return ((p_hat >= lo) & (p_hat <= hi)).any(axis=0)


def main():
    results = {}
    for lbl, cat, sub in CATS:
        base = REPO / "results" / cat / sub
        K, N, J = load_panel_matrix(base, PANEL)
        p_hat = np.where(N > 0, K / N, 0.5)
        keep = alo_mask(p_hat)
        Jk = int(keep.sum())
        K2, N2, p2 = K[:, keep], N[:, keep], p_hat[:, keep]
        th0, b0 = fit_ols(p2)
        theta, beta, _ = fit_mle(K2, N2, th0, b0)
        theta = theta - theta.mean()
        results[lbl] = {
            "theta": theta.tolist(),
            "kept": Jk,
            "pool": J,
            "range": float(theta.max() - theta.min()),
        }

    # Print as cell table
    header = f"{'#':>2}  {'Model':<24} {'Family':<11} {'Params':<11} " + " ".join(
        f"theta_{lbl:<3}" for lbl, _, _ in CATS
    )
    print(header)
    print("-" * len(header))
    for i, (nm, fam, p) in enumerate(zip(SHORT, FAMILY, PARAMS)):
        row = f"{i+1:>2}  {nm:<24} {fam:<11} {p:<11} "
        row += " ".join(f"{results[lbl]['theta'][i]:+8.2f}" for lbl, _, _ in CATS)
        print(row)
    print("-" * len(header))
    print(
        f"{'':>2}  {'theta range':<24} {'':<11} {'':<11} "
        + " ".join(f"{results[lbl]['range']:+8.2f}" for lbl, _, _ in CATS)
    )
    print(
        f"{'':>2}  {'kept J (ALO)':<24} {'':<11} {'':<11} "
        + " ".join(f"{results[lbl]['kept']:>8d}" for lbl, _, _ in CATS)
    )
    print(
        f"{'':>2}  {'pool size':<24} {'':<11} {'':<11} "
        + " ".join(f"{results[lbl]['pool']:>8d}" for lbl, _, _ in CATS)
    )

    out_dir = REPO / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "filter": f"ALO[{ALO_LO}, {ALO_HI}]",
        "panel": PANEL,
        "panel_short": SHORT,
        "family": FAMILY,
        "params": PARAMS,
        "categories": [lbl for lbl, _, _ in CATS],
        "category_paths": {lbl: sub for lbl, _, sub in CATS},
        "results": results,
    }
    (out_dir / "panel_theta_alo.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {out_dir / 'panel_theta_alo.json'}")


if __name__ == "__main__":
    main()
