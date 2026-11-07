#!/usr/bin/env python3
"""Verify closed-form FNR derivation for ALO[0.20, 0.80] coarse-fine protocol.

Claims to verify (thesis §coarse-fine / paper §app:two-stage):
  C1. FNR(p_0=0.20; N_c, k_lo) = P(Bin(N_c, 0.20) ≤ k_lo)  (analytical)
      Specifically:
        FNR(0.20; 14, 0) ≈ 0.0440
        FNR(0.20; 22, 1) ≈ 0.0480
        FNR(0.20;  7, 0) ≈ 0.2097  (AVG-era N_c, infeasible under ALO)

  C2. The "single-boundary-model" anchor configuration (one model at p_0=0.20,
      others at p=0) is the worst case for coarse-rejection of ALO-survivable
      items. We scan all configurations of the form
          (one m* at p_0, M-1 others at p_other ∈ {0, 0.05, 0.10, 0.15})
      and confirm the item-level coarse-FNR is maximized at p_other = 0.

  C3. Symmetry: FNR at upper boundary p_0=0.80 with k_hi = N_c - k_lo equals
      FNR at lower boundary p_0=0.20 with k_lo (exact Binomial symmetry).

Each claim is verified analytically (scipy.binom) AND via 200k-trial Monte
Carlo to confirm the formula evaluates correctly.

Inputs: none (purely analytic + simulated)
Outputs: stdout — pass/fail per claim + numerical values.

Used by:
  paper/neurips_2026.tex §app:two-stage (closed-form derivation L1353+)
  毕业论文_附录_coarse_fine_ALO.txt §6 (verification log)
"""
import numpy as np
from scipy.stats import binom

P0 = 0.20         # ALO inner boundary (lower)
P0_UP = 0.80      # ALO inner boundary (upper)
M = 9             # locked panel size
N_SIM = 200_000   # Monte Carlo trials
RNG = np.random.default_rng(20260517)


def fnr_analytic(N_c, k_lo, p):
    return float(binom.cdf(k_lo, N_c, p))


def fnr_monte_carlo(N_c, k_lo, p, n_sim=N_SIM):
    """Single boundary model: P(K^(c) ≤ k_lo)."""
    k = RNG.binomial(N_c, p, size=n_sim)
    return float((k <= k_lo).mean())


def item_fnr_analytic(N_c, k_lo, p_boundary, p_others, M=M):
    """P(item coarse-rejected | one m* at p_boundary, M-1 others at p_others).

    Item rejected iff ALL M cells classified L or H. For a cell with rate p:
        P(L | p) = P(Bin(N_c, p) ≤ k_lo)
        P(H | p) = P(Bin(N_c, p) ≥ N_c - k_lo) = 1 - P(Bin ≤ N_c - k_lo - 1)
    """
    def p_extreme(p):
        return binom.cdf(k_lo, N_c, p) + (1 - binom.cdf(N_c - k_lo - 1, N_c, p))
    return p_extreme(p_boundary) * p_extreme(p_others) ** (M - 1)


def item_fnr_monte_carlo(N_c, k_lo, p_boundary, p_others, M=M, n_sim=N_SIM):
    """Same as item_fnr_analytic but Monte Carlo."""
    p_vec = np.array([p_boundary] + [p_others] * (M - 1))
    K = RNG.binomial(N_c, p_vec[:, None] * np.ones((1, n_sim)), size=(M, n_sim))
    is_L = K <= k_lo
    is_H = K >= N_c - k_lo
    is_extreme = is_L | is_H
    all_extreme = is_extreme.all(axis=0)
    return float(all_extreme.mean())


def banner(s):
    print()
    print("=" * 80)
    print(f"  {s}")
    print("=" * 80)


def check(label, lhs, rhs, tol=1e-3):
    ok = abs(lhs - rhs) <= tol
    flag = "✓" if ok else "✗"
    print(f"  {flag}  {label}: analytic={lhs:.6f}  vs  MC={rhs:.6f}   "
          f"|Δ|={abs(lhs-rhs):.6f}  (tol={tol})")
    return ok


def main():
    banner("C1. Closed-form FNR(p_0=0.20; N_c, k_lo) for headline configurations")
    cases = [(14, 0, 0.0440), (22, 1, 0.0480), (7, 0, 0.2097),
             (21, 1, 0.0576), (13, 0, 0.0550)]
    all_ok = True
    for N_c, k_lo, expected in cases:
        ana = fnr_analytic(N_c, k_lo, P0)
        mc = fnr_monte_carlo(N_c, k_lo, P0)
        flag = "✓" if abs(ana - expected) < 5e-4 else "✗"
        flag_mc = "✓" if abs(ana - mc) < 5e-3 else "✗"
        print(f"  {flag}  FNR(0.20; N_c={N_c:>2}, k_lo={k_lo}) = {ana:.4f}   "
              f"(thesis claim {expected:.4f}, |Δ|={abs(ana-expected):.4f})   "
              f"{flag_mc} MC={mc:.4f}")
        all_ok &= (abs(ana - expected) < 5e-4) & (abs(ana - mc) < 5e-3)

    banner("C2. Worst-case anchor: is (one m* at p_0=0.20, others at p=0) the maximum FNR?")
    print(f"  Scanning p_others over [0, 0.20] for N_c=14, k_lo=0, M={M}")
    print(f"  {'p_others':>10}  {'item-FNR (ana)':>16}  {'item-FNR (MC)':>16}")
    p_other_grid = [0.00, 0.02, 0.05, 0.10, 0.15, 0.19, 0.20]
    item_fnrs = []
    for p_o in p_other_grid:
        ana = item_fnr_analytic(14, 0, P0, p_o)
        mc = item_fnr_monte_carlo(14, 0, P0, p_o)
        item_fnrs.append((p_o, ana, mc))
        print(f"  {p_o:>10.2f}  {ana:>16.6f}  {mc:>16.6f}")
    max_idx = int(np.argmax([t[1] for t in item_fnrs]))
    p_max = item_fnrs[max_idx][0]
    fnr_max = item_fnrs[max_idx][1]
    fnr_at_zero = item_fnrs[0][1]
    print(f"\n  → argmax over p_others: p={p_max:.2f}, item-FNR={fnr_max:.6f}")
    print(f"    item-FNR at p_others=0: {fnr_at_zero:.6f}")
    if abs(p_max) < 1e-6:
        print(f"  ✓  Worst-case at p_others=0 confirmed (one-model anchor, "
              f"others saturated low)")
        c2_ok = True
    else:
        print(f"  ✗  Argmax NOT at p_others=0 — closed-form worst-case claim WRONG")
        c2_ok = False

    print(f"\n  At p_others=0, item-FNR equals single-cell FNR ({fnr_at_zero:.6f}) "
          f"because others' P(L)=1 (saturated zero).")
    print(f"  This collapses item-level FNR to single-boundary-model FNR — "
          f"the closed-form (eq:fnr) is exact for this config.")

    banner("C3. Symmetry: FNR at upper boundary p=0.80 equals lower at p=0.20")
    cases_sym = [(14, 0), (22, 1), (7, 0)]
    c3_ok = True
    for N_c, k_lo in cases_sym:
        fnr_lo = fnr_analytic(N_c, k_lo, P0)
        # Upper boundary: P(K ≥ N_c - k_lo | p=0.80) = 1 - P(K ≤ N_c-k_lo-1)
        fnr_up = 1 - float(binom.cdf(N_c - k_lo - 1, N_c, P0_UP))
        flag = "✓" if abs(fnr_lo - fnr_up) < 1e-10 else "✗"
        print(f"  {flag}  N_c={N_c:>2}, k_lo={k_lo}: "
              f"FNR_lo(p=0.20)={fnr_lo:.6f}, FNR_up(p=0.80)={fnr_up:.6f}")
        c3_ok &= (abs(fnr_lo - fnr_up) < 1e-10)

    banner("Summary")
    print(f"  C1 (closed-form numerical accuracy):           {'PASS' if all_ok else 'FAIL'}")
    print(f"  C2 (worst-case anchor configuration argmax):    {'PASS' if c2_ok else 'FAIL'}")
    print(f"  C3 (upper/lower boundary FNR symmetry):         {'PASS' if c3_ok else 'FAIL'}")
    overall = all_ok and c2_ok and c3_ok
    print(f"\n  Overall: {'✓ ALL VERIFIED' if overall else '✗ FAILURES — re-check derivation'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
