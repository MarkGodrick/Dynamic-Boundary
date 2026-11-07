"""Verify Claim 1: edge case β ≈ source model θ.

Uses the IRT fit from `scripts/fit_irt.py` plus the original edge case dataset
(to recover prompt_id → source_model) and the response matrices (to compute
mean validation p_hat per item for post-hoc filtering).

Reports:
  - Spearman ρ(θ_m, mean β of M's own ECs), with and without filtering
  - Per-model mean β table
  - Monotonicity check: mean validation p_hat across β bins
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difficulty.rasch import load_response_matrix


def load_source_map(dataset_path: str) -> dict[str, str]:
    """prompt_id -> source_model."""
    out = {}
    with open(dataset_path) as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            out[e["prompt_id"]] = e["source_model"]
    return out


def claim1_correlation(
    theta: dict[str, float],
    beta: dict[str, float],
    source_map: dict[str, str],
    keep_items: set[str] | None = None,
) -> tuple[float, float, dict]:
    """Compute Spearman ρ(θ_m, mean β of M's own ECs).

    Returns (rho, pearson_r, per_model_dict).
    """
    per_model = {}
    for m in theta:
        own = [
            beta[pid] for pid, src in source_map.items()
            if src == m and pid in beta and (keep_items is None or pid in keep_items)
        ]
        if not own:
            per_model[m] = {"theta": theta[m], "n": 0, "mean_beta": float("nan")}
            continue
        per_model[m] = {
            "theta": theta[m],
            "n": len(own),
            "mean_beta": float(np.mean(own)),
            "median_beta": float(np.median(own)),
            "std_beta": float(np.std(own)),
        }

    valid = [(d["theta"], d["mean_beta"]) for d in per_model.values() if d["n"] > 0]
    if len(valid) < 2:
        return float("nan"), float("nan"), per_model

    thetas, mean_betas = zip(*valid)
    rho = float(spearmanr(thetas, mean_betas)[0])
    pr = float(pearsonr(thetas, mean_betas)[0])
    return rho, pr, per_model


def monotonicity_check(
    p_hat: np.ndarray,
    items: list[str],
    beta: dict[str, float],
    n_bins: int = 5,
) -> list[dict]:
    """Bin items by β, compute mean validation p_hat per bin."""
    item_beta = np.array([beta[p] for p in items])
    order = np.argsort(item_beta)  # ascending β = easy → hard

    bins = np.array_split(order, n_bins)
    out = []
    for i, idx in enumerate(bins):
        b_low, b_high = item_beta[idx].min(), item_beta[idx].max()
        # Mean validation p_hat across all (model, item) cells in the bin
        mean_p = float(p_hat[:, idx].mean())
        out.append({
            "bin": i,
            "n_items": len(idx),
            "beta_range": (round(float(b_low), 3), round(float(b_high), 3)),
            "mean_p_hat": round(mean_p, 4),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/constrained_completion/validation/edge_case")
    ap.add_argument("--dataset", default="benchmarks/constrained_completion/edge_case_dataset.jsonl")
    ap.add_argument("--irt", default=None,
                    help="Default: {data}/irt_results.json")
    args = ap.parse_args()

    data_dir = Path(args.data)
    irt_path = Path(args.irt) if args.irt else data_dir / "irt_results.json"

    # Load
    irt = json.load(open(irt_path))
    theta_mle = {m: d["theta_mle"] for m, d in irt["models"].items()}
    beta_mle = {p: d["beta_mle"] for p, d in irt["items"].items()}
    source_map = load_source_map(args.dataset)

    rm = load_response_matrix(str(data_dir))
    items = rm["items"]
    p_hat = rm["p_hat"]
    item_idx = {p: j for j, p in enumerate(items)}

    # Mean validation p_hat per item (across the 6 panel models)
    item_mean_p = p_hat.mean(axis=0)

    # ------------------------------------------------------------------
    # 1. Coverage
    # ------------------------------------------------------------------
    print(f"{'='*68}")
    print(f"  Coverage")
    print(f"{'='*68}")
    print(f"  models in IRT:        {len(theta_mle)}")
    print(f"  items in IRT:         {len(beta_mle)}")
    print(f"  items in dataset:     {len(source_map)}")
    print(f"  items in p_hat matrix: {len(items)}")
    src_counts = {}
    for pid, src in source_map.items():
        src_counts[src] = src_counts.get(src, 0) + 1
    print(f"  ECs per source model:")
    for m, c in sorted(src_counts.items()):
        print(f"    {c:4d}  {m}")

    # ------------------------------------------------------------------
    # 2. Main: ρ(θ, mean β of own ECs) — no filter
    # ------------------------------------------------------------------
    def report(label, keep):
        rho, pr, per = claim1_correlation(theta_mle, beta_mle, source_map, keep)
        n_kept = len(keep) if keep is not None else len(beta_mle)
        print(f"\n{'='*68}")
        print(f"  {label}  (n_items_kept = {n_kept})")
        print(f"{'='*68}")
        print(f"  {'Model':<50} {'θ_MLE':>8} {'n':>4} {'mean β':>8} {'std β':>7}")
        print(f"  {'-'*50} {'-'*8} {'-'*4} {'-'*8} {'-'*7}")
        for m in sorted(per, key=lambda x: per[x]["theta"], reverse=True):
            d = per[m]
            mb = f"{d['mean_beta']:.3f}" if d["n"] > 0 else "  nan"
            sb = f"{d.get('std_beta', float('nan')):.3f}" if d["n"] > 0 else "  nan"
            print(f"  {m:<50} {d['theta']:>8.3f} {d['n']:>4d} {mb:>8} {sb:>7}")
        print(f"\n  Spearman ρ(θ, mean β) = {rho:.4f}")
        print(f"  Pearson  r(θ, mean β) = {pr:.4f}")
        return rho, pr

    rho_all, _ = report("ALL EDGE CASES (no filter)", None)

    # ------------------------------------------------------------------
    # 3. Post-hoc filter by validation mean p_hat
    # ------------------------------------------------------------------
    for lo, hi in [(0.30, 0.70), (0.35, 0.65), (0.40, 0.60), (0.20, 0.80)]:
        keep = {items[j] for j in range(len(items))
                if lo <= item_mean_p[j] <= hi}
        report(f"Filter mean p_hat ∈ [{lo:.2f}, {hi:.2f}]", keep)

    # ------------------------------------------------------------------
    # 4. Monotonicity: bin items by β, mean p_hat per bin
    # ------------------------------------------------------------------
    print(f"\n{'='*68}")
    print(f"  Monotonicity check (5 β-bins, easy → hard)")
    print(f"{'='*68}")
    bins = monotonicity_check(p_hat, items, beta_mle, n_bins=5)
    print(f"  {'bin':>3} {'n':>5} {'β range':>20} {'mean p_hat':>12}")
    for b in bins:
        rng = f"[{b['beta_range'][0]:>6.3f}, {b['beta_range'][1]:>6.3f}]"
        print(f"  {b['bin']:>3} {b['n_items']:>5} {rng:>20} {b['mean_p_hat']:>12.4f}")

    # ------------------------------------------------------------------
    # 5. Bonus: ρ between θ and median β (more robust to outlier ECs)
    # ------------------------------------------------------------------
    print(f"\n{'='*68}")
    print(f"  Bonus: Spearman ρ using median β instead of mean β")
    print(f"{'='*68}")
    for label, keep in [
        ("no filter", None),
        ("[0.30, 0.70]", {items[j] for j in range(len(items)) if 0.30 <= item_mean_p[j] <= 0.70}),
        ("[0.40, 0.60]", {items[j] for j in range(len(items)) if 0.40 <= item_mean_p[j] <= 0.60}),
    ]:
        per = {}
        for m in theta_mle:
            own = [beta_mle[pid] for pid, src in source_map.items()
                   if src == m and pid in beta_mle and (keep is None or pid in keep)]
            if own:
                per[m] = (theta_mle[m], float(np.median(own)), len(own))
        if len(per) >= 2:
            ts, mbs, _ = zip(*per.values())
            rho = float(spearmanr(ts, mbs)[0])
            print(f"  {label:<20} ρ={rho:+.4f}  (n_models={len(per)})")


if __name__ == "__main__":
    main()
