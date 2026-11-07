#!/usr/bin/env python3
"""Derive optimal (N_c, coarse range) for filter split under fine [0.05, 0.95].

Probabilistic framework
-----------------------
Setup:
  - Fine filter (locked): keep item iff panel-mean p_bar in [ell_f, h_f] = [0.05, 0.95].
  - N_total = 40 trials per panel model per item (budget).
  - Coarse stage uses N_c trials; fine stage adds N_f = 40 - N_c.
  - Panel size M = 9 (locked 2026-04-24; see docs/filter_method_spec.md §G).

Panel-sum statistic:
  S = sum_m k_m  ~  Binomial(M*N_c, p)
  Coarse passes iff  k_min <= S <= M*N_c - k_min
  Effective lower threshold ell_c = k_min / (M*N_c)
  Effective upper threshold h_c = 1 - ell_c   (symmetric by fine filter symmetry)

Critical point: p_0 = 0.05 (fine's lower boundary).
  Normal approx fails here: SE(p_bar) = sqrt(0.05*0.95/(M*N_c)) ~ 0.03, but
  distance to boundary is p_0 itself = 0.05. So lower tail of Normal crosses 0.
  Use exact Binomial CDF throughout.

Constraint:
  FNR(p=0.05) = P(S < k_min | S ~ Bin(M*N_c, 0.05))  <=  alpha

Cost objective:
  Cost(N_c, k_min) = M*N_c + r_c(N_c, k_min) * M * (N_total - N_c)
  where r_c = fraction of pool passing coarse, depends on pool's p-distribution.

Pool models (Beta(a,b) discretization):
  - uniform       Beta(1,1)      — diffuse, few extreme items
  - mild U        Beta(0.7,0.7)  — some mass near boundaries
  - heavy U       Beta(0.3,0.3)  — most mass near 0 and 1 (our real case)
  - extreme       Beta(0.15,0.15) — nearly all mass at boundaries
"""
import math
from scipy.stats import binom, beta as beta_dist

FINE_LO, FINE_HI = 0.05, 0.95
N_TOTAL = 40
P_BOUNDARY = FINE_LO


def fnr_exact(M, N_c, k_min, p=P_BOUNDARY):
    """FNR at true p: P(S < k_min | S ~ Bin(M*N_c, p)). Exact Binomial."""
    if k_min <= 0:
        return 0.0
    return float(binom.cdf(k_min - 1, M * N_c, p))


def coarse_pass_prob(M, N_c, k_min, p):
    """P(k_min <= S <= M*N_c - k_min | S ~ Bin(M*N_c, p))."""
    MN = M * N_c
    lower = binom.cdf(k_min - 1, MN, p) if k_min > 0 else 0.0
    upper = binom.cdf(MN - k_min, MN, p)
    return float(upper - lower)


def max_k_min_for_alpha(M, N_c, alpha, p=P_BOUNDARY):
    """Largest integer k_min with FNR(p) <= alpha."""
    MN = M * N_c
    for k in range(MN // 2 + 1, -1, -1):
        if fnr_exact(M, N_c, k, p) <= alpha:
            return k
    return 0


def beta_pool(a, b, n_grid=199):
    """Discretize Beta(a, b) on (0,1) via midpoints."""
    grid = [(i + 0.5) / n_grid for i in range(n_grid)]
    weights = [float(beta_dist.pdf(p, a, b)) for p in grid]
    total = sum(weights)
    return grid, [w / total for w in weights]


def pool_r_c(M, N_c, k_min, grid, weights):
    return sum(w * coarse_pass_prob(M, N_c, k_min, p) for p, w in zip(grid, weights))


def normal_ci_halfwidth(M, N, p, z=1.96):
    """95% CI half-width for panel-mean p_hat under Normal approx (for comparison)."""
    se = math.sqrt(p * (1 - p) / (M * N))
    return z * se


def print_frontier_table(M, alpha):
    print(f"\n[Frontier] M={M}, FNR tolerance at p=0.05 is alpha={alpha:.0%}")
    print(f"  k_min* = largest int with FNR(0.05) <= alpha;  ell_c* = k_min* / (M*N_c)")
    print(f"  {'N_c':>4} {'N_f':>4} {'MN_c':>5} {'k_min*':>7} {'ell_c*':>8} {'h_c*':>8}"
          f" {'FNR@.05':>9} {'FNR@.02':>9} {'FNR@.10':>9} {'NormalSE':>9}")
    print("  " + "-" * 85)
    for N_c in [3, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30]:
        k = max_k_min_for_alpha(M, N_c, alpha)
        ell = k / (M * N_c)
        fnr = fnr_exact(M, N_c, k, 0.05)
        fnr02 = fnr_exact(M, N_c, k, 0.02)
        fnr10 = fnr_exact(M, N_c, k, 0.10)
        nse = math.sqrt(0.05 * 0.95 / (M * N_c))
        print(f"  {N_c:>4} {N_TOTAL-N_c:>4} {M*N_c:>5} {k:>7} {ell:>8.4f} {1-ell:>8.4f}"
              f" {fnr:>9.2%} {fnr02:>9.2%} {fnr10:>9.2%} {nse:>9.4f}")


def print_cost_table(M, alpha, pool_label, a, b):
    grid, weights = beta_pool(a, b)
    print(f"\n[Cost] M={M}, alpha={alpha:.0%}, pool={pool_label} Beta({a},{b})")
    print(f"  Cost = M*N_c + r_c * M * N_f  (per item, measured in single-model-trial units)")
    print(f"  Baseline single-stage N=40 cost = {M * N_TOTAL} trials per item")
    print(f"  {'N_c':>4} {'k_min':>6} {'ell_c':>8} {'r_c':>6} {'stage1':>7} {'stage2':>7}"
          f" {'TOTAL':>7} {'vs N=40':>9}")
    print("  " + "-" * 64)
    best_cost, best_Nc = math.inf, None
    for N_c in range(3, 21):
        k = max_k_min_for_alpha(M, N_c, alpha)
        if k == 0:
            ell, r = 0.0, 1.0
        else:
            ell = k / (M * N_c)
            r = pool_r_c(M, N_c, k, grid, weights)
        s1 = M * N_c
        s2 = r * M * (N_TOTAL - N_c)
        total = s1 + s2
        ratio = total / (M * N_TOTAL)
        mark = ""
        if total < best_cost:
            best_cost, best_Nc = total, N_c
            mark = " <-- best"
        print(f"  {N_c:>4} {k:>6} {ell:>8.4f} {r:>6.3f} {s1:>7.1f} {s2:>7.1f}"
              f" {total:>7.1f} {ratio:>9.1%}{mark}")
    print(f"  => Best N_c = {best_Nc}, total = {best_cost:.1f} trials, savings = {1 - best_cost/(M*N_TOTAL):.1%}")


def print_summary():
    print("\n" + "=" * 82)
    print("SUMMARY: optimal (N_c, coarse range) across pool shapes")
    print("=" * 82)
    print(f"  {'M':>3} {'pool':>25} {'alpha':>6} {'best N_c':>9} {'k_min':>6} {'ell_c':>8}"
          f" {'h_c':>8} {'savings':>9}")
    print("  " + "-" * 78)
    for M in [9]:
        for alpha in [0.03, 0.05, 0.10]:
            for label, a, b in [("uniform (1,1)", 1, 1),
                                ("mild-U (0.7,0.7)", 0.7, 0.7),
                                ("heavy-U (0.3,0.3)", 0.3, 0.3),
                                ("extreme (0.15,0.15)", 0.15, 0.15)]:
                grid, weights = beta_pool(a, b)
                best = (math.inf, None, None, None)
                for N_c in range(3, 21):
                    k = max_k_min_for_alpha(M, N_c, alpha)
                    if k == 0:
                        r = 1.0
                    else:
                        r = pool_r_c(M, N_c, k, grid, weights)
                    total = M * N_c + r * M * (N_TOTAL - N_c)
                    if total < best[0]:
                        best = (total, N_c, k, r)
                total, N_c, k, r = best
                ell = k / (M * N_c) if k > 0 else 0.0
                sav = 1 - total / (M * N_TOTAL)
                print(f"  {M:>3} {label:>25} {alpha:>6.0%} {N_c:>9} {k:>6} {ell:>8.4f}"
                      f" {1-ell:>8.4f} {sav:>9.1%}")


def main():
    print("=" * 82)
    print(f"Derivation: optimal (N_c, coarse range) under fine [{FINE_LO}, {FINE_HI}]")
    print(f"N_total = {N_TOTAL}, using exact Binomial throughout (Normal fails at p=0.05)")
    print("=" * 82)

    # Part 1: probabilistic frontier (FNR constraint only, no cost)
    print("\n" + "#" * 82)
    print("# Part 1: Probabilistic frontier — largest feasible k_min per N_c")
    print("#" * 82)
    for M in [9]:
        for alpha in [0.05, 0.03]:
            print_frontier_table(M, alpha)

    # Part 2: cost analysis per pool shape
    print("\n" + "#" * 82)
    print("# Part 2: Cost analysis (includes r_c and stage-2 savings)")
    print("#" * 82)
    for M in [9]:
        for label, a, b in [("uniform", 1.0, 1.0),
                             ("mild U-shape", 0.7, 0.7),
                             ("heavy U-shape", 0.3, 0.3)]:
            print_cost_table(M, 0.05, label, a, b)

    # Part 3: summary across scenarios
    print_summary()

    # Part 4: sanity — current proposal [0.005, 0.995]
    print("\n" + "=" * 82)
    print("SANITY CHECK: session proposal coarse = [0.005, 0.995]")
    print("=" * 82)
    for M in [9]:
        for N_c in [10]:
            MN = M * N_c
            k = math.ceil(MN * 0.005)
            if k == 0:
                k = 1  # degenerate: ell_c=0 means no threshold
            ell_eff = k / MN
            fnr = fnr_exact(M, N_c, k, 0.05)
            grid, weights = beta_pool(0.3, 0.3)
            r = pool_r_c(M, N_c, k, grid, weights)
            total = M * N_c + r * M * (N_TOTAL - N_c)
            print(f"  M={M}, N_c={N_c}: ceil({MN}*0.005)={k}, effective ell_c={ell_eff:.4f}, "
                  f"FNR(p=0.05)={fnr:.2%}, r_c(heavyU)={r:.3f}, cost={total:.1f} "
                  f"(vs {M*N_TOTAL} single-stage, saves {1-total/(M*N_TOTAL):.1%})")


if __name__ == "__main__":
    main()
