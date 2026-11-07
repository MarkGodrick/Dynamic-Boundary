"""Budget-fair comparison of SGBS vs random + filter.

Addresses:
  Q1: Fix histogram — show extreme tail count separately, no clipping artifacts
  Q2: Per-θ TIF for filtered random (see if filter helps panel-edge models)
  Q3: Use ACTUAL SGBS search budget from stats.json (not estimate)
      Use SGBS's per-round trial count (15 coarse + up to 25 fine)
      not the 20 used in response matrix collection.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difficulty.rasch import load_response_matrix


EC_DIR = "results/constrained_completion/validation/edge_case"
RB_DIR = "results/constrained_completion/validation/random_baseline"
SGBS_DIR = "results/constrained_completion/edge_case"


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def get_sgbs_budget():
    """Read all 6 SGBS run stats.json files."""
    rows = []
    total_evaluated = 0
    total_run = 0
    total_skipped = 0
    for d in sorted(Path(SGBS_DIR).iterdir()):
        sp = d / "stats.json"
        if not sp.exists():
            continue
        s = json.load(open(sp))
        run = s["rounds_run"]
        skipped = s.get("skipped_rounds", 0)
        evaluated = run - skipped
        rows.append({
            "model": d.name.split("_constrained")[0],
            "rounds_run": run,
            "skipped": skipped,
            "evaluated": evaluated,
            "ECs": s["edge_cases_found"],
            "time_min": s.get("total_time_seconds", 0) / 60,
        })
        total_evaluated += evaluated
        total_run += run
        total_skipped += skipped
    return rows, total_run, total_skipped, total_evaluated


def main():
    print("=" * 78)
    print("  Q3: Precise SGBS search budget (from stats.json)")
    print("=" * 78)
    rows, T_run, T_skip, T_eval = get_sgbs_budget()
    print(f"\n  {'Model':<32} {'rounds_run':>12} {'skipped':>10} {'evaluated':>10} {'time(min)':>10}")
    print(f"  {'-'*32} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for r in rows:
        print(f"  {r['model'][:32]:<32} {r['rounds_run']:>12} {r['skipped']:>10} "
              f"{r['evaluated']:>10} {r['time_min']:>10.0f}")
    print(f"  {'TOTAL':<32} {T_run:>12} {T_skip:>10} {T_eval:>10}")

    COARSE_N = 15
    FINE_N = 25
    print(f"\n  SGBS progressive eval: coarse N={COARSE_N}, fine N={FINE_N} (cumulative {COARSE_N+FINE_N})")

    sgbs_search_lower = T_eval * COARSE_N
    sgbs_search_upper = T_eval * (COARSE_N + FINE_N)
    sgbs_search_realistic = T_eval * (COARSE_N + 0.5 * FINE_N)

    print(f"\n  SGBS search trials:")
    print(f"    LOWER bound (coarse only):    {sgbs_search_lower:>10,}")
    print(f"    REALISTIC (50% reach fine):   {int(sgbs_search_realistic):>10,}")
    print(f"    UPPER bound (all reach fine): {sgbs_search_upper:>10,}")

    RM_TRIALS = 600 * 6 * 20
    print(f"\n  Response matrix collection (BOTH methods): "
          f"600 × 6 × 20 = {RM_TRIALS:,} trials")

    sgbs_total_lower = sgbs_search_lower + RM_TRIALS
    sgbs_total_upper = sgbs_search_upper + RM_TRIALS
    sgbs_total_realistic = sgbs_search_realistic + RM_TRIALS

    print(f"\n  SGBS TOTAL inference budget:")
    print(f"    LOWER:     {sgbs_total_lower:>10,}")
    print(f"    REALISTIC: {int(sgbs_total_realistic):>10,}")
    print(f"    UPPER:     {sgbs_total_upper:>10,}")
    print(f"\n  Random TOTAL: {RM_TRIALS:,} (search/generation is template-only, free)")
    print(f"\n  Ratio (SGBS / Random):")
    print(f"    LOWER:     {sgbs_total_lower / RM_TRIALS:>5.2f}×")
    print(f"    REALISTIC: {sgbs_total_realistic / RM_TRIALS:>5.2f}×")
    print(f"    UPPER:     {sgbs_total_upper / RM_TRIALS:>5.2f}×")

    # Q1 & Q2
    print("\n" + "=" * 78)
    print("  Q1: True β distribution (no clipping)")
    print("=" * 78)
    data = json.load(open("results/constrained_completion/joint_fit_data.json"))
    beta_ec = np.array(data["beta_ec"])
    beta_rb = np.array(data["beta_rb"])

    inside_ec = (beta_ec >= -3) & (beta_ec <= 3)
    inside_rb = (beta_rb >= -3) & (beta_rb <= 3)
    n_ec_below = (beta_ec < -3).sum()
    n_ec_above = (beta_ec > 3).sum()
    n_rb_below = (beta_rb < -3).sum()
    n_rb_above = (beta_rb > 3).sum()

    print(f"  SGBS:   in [−3, +3] = {inside_ec.sum()}, "
          f"|β|>3 = {n_ec_below + n_ec_above}")
    print(f"  Random: in [−3, +3] = {inside_rb.sum()}, "
          f"|β|>3 = {n_rb_below + n_rb_above} "
          f"({(n_rb_below+n_rb_above)/600*100:.0f}%)")

    print("\n" + "=" * 78)
    print("  Q2: Per-θ TIF — does filter help panel-edge models?")
    print("=" * 78)

    theta = np.array(data["theta_joint"])
    rb_rm = load_response_matrix(RB_DIR)
    K_rb = rb_rm["pass_counts"]
    N_rb = rb_rm["total_trials"]
    p_rb = K_rb / N_rb
    rb_models = rb_rm["models"]

    short = lambda m: m.split("/")[-1].replace("DeepSeek-R1-Distill-Llama", "R1") \
                                       .replace("Qwen3-30B-A3B-Instruct-2507", "Qwen-30B") \
                                       .replace("Mistral-Small-3.1-24B-Instruct-2503", "Mistral-24B") \
                                       .replace("gemma-3-", "Gemma3-").replace("-it", "")[:14]

    item_mean = p_rb.mean(axis=0)

    methods = [
        ("SGBS (n=600)", beta_ec),
        ("Random unfilt", beta_rb),
    ]
    for label, lo, hi in [
        ("OLD [0.20, 0.80]", 0.20, 0.80),
        ("OLD [0.30, 0.70]", 0.30, 0.70),
        ("OLD [0.40, 0.60]", 0.40, 0.60),
    ]:
        mask = (item_mean >= lo) & (item_mean <= hi)
        methods.append((f"{label} (n={mask.sum()})", beta_rb[mask]))

    print(f"\n  per-θ TIF total (per-item efficiency)\n")
    print(f"  {'Model':<22}", end="")
    for label, _ in methods:
        print(f"{label:>22}", end="")
    print()
    print("  " + "-" * (22 + 22 * len(methods)))

    for i, (m, t) in enumerate(zip(rb_models, theta)):
        m_short = short(m)
        print(f"  {m_short:<10} (θ={t:+.2f})  ", end="")
        for label, beta_set in methods:
            if len(beta_set) == 0:
                print(f"{'—':>22}", end="")
                continue
            p = sigmoid(t - np.asarray(beta_set))
            tif = float((p * (1 - p)).sum())
            per_item = tif / len(beta_set)
            print(f"{tif:>9.1f}({per_item:>5.3f})", end=" " * 4)
        print()

    print("\n" + "=" * 78)
    print("  Budget-fair extrapolation: how much budget does random need?")
    print("=" * 78)
    sgbs_tif_total = sum(
        float((sigmoid(t - beta_ec) * (1 - sigmoid(t - beta_ec))).sum())
        for t in theta
    )
    print(f"\n  SGBS total TIF (sum over panel θ): {sgbs_tif_total:.1f}")

    print(f"\n  {'Filter':<22} {'pass%':>7} {'TIF/item':>10} {'kept needed':>12} "
          f"{'gen needed':>12} {'inf cost':>14} {'×SGBS_real':>12}")
    print(f"  {'-'*22} {'-'*7} {'-'*10} {'-'*12} {'-'*12} {'-'*14} {'-'*12}")

    for label, lo, hi in [
        ("OLD [0.20, 0.80]", 0.20, 0.80),
        ("OLD [0.30, 0.70]", 0.30, 0.70),
        ("OLD [0.40, 0.60]", 0.40, 0.60),
    ]:
        mask = (item_mean >= lo) & (item_mean <= hi)
        n_kept = mask.sum()
        if n_kept == 0:
            continue
        pass_rate = n_kept / 600
        beta_filt = beta_rb[mask]
        per_item_total = sum(
            float((sigmoid(t - beta_filt) * (1 - sigmoid(t - beta_filt))).sum()) / n_kept
            for t in theta
        )
        kept_needed = sgbs_tif_total / per_item_total
        gen_needed = kept_needed / pass_rate
        inf_cost = gen_needed * 6 * 20
        ratio = inf_cost / sgbs_total_realistic
        print(f"  {label:<22} {pass_rate*100:>6.1f}% {per_item_total:>10.4f} "
              f"{kept_needed:>12.0f} {gen_needed:>12.0f} {inf_cost:>14,.0f} "
              f"{ratio:>11.2f}×")


if __name__ == "__main__":
    main()
