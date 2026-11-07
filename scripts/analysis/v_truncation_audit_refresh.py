"""Recompute the two CIF cells of tab:claim3-truncation-V after the
ConstraintJudge CoT-strip fix (judge_revision 2026-05-16_cot_strip).

Original audit (paper L2399-L2400, neurips_2026.tex):
  Phi-4-reasoning-plus  CIF: 40.8% / +1.00 (flagged)
  Falcon-H1R-7B         CIF: 20.8% / +0.07

Re-judging flipped many CIF `passed` labels for both reasoning models
(rule-based IFEval/IFBench was previously fed full <think>...</think>
content). r_t is unchanged (it's a property of finish_reason, not pass
label). Δ_p is the quantity that moves; the original Phi-4-r-plus
+1.00 was a CoT-pollution artifact, not a small-sample artifact.

Metrics:
  r_t = (# trials with truncated=True) / (# total trials), reported as %.
  Δ_p = (pass rate among truncated) − (pass rate among non-truncated)
        Undefined when there are zero truncated trials.

Usage:
    conda run -n boundary --no-capture-output python -u \\
        scripts/analysis/v_truncation_audit_refresh.py
"""

import json
from pathlib import Path


TARGETS = [
    ("Phi-4-reasoning-plus", "results/constrained_completion/anchor_set/microsoft_Phi_4_reasoning_plus/response_matrix.jsonl"),
    ("Falcon-H1R-7B",       "results/constrained_completion/anchor_set/tiiuae_Falcon_H1R_7B/response_matrix.jsonl"),
]


def audit(path: Path) -> dict:
    n_total = 0
    n_trunc = 0
    pass_trunc = 0
    pass_non = 0
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            for t in row.get("trials") or []:
                n_total += 1
                truncated = bool(t.get("truncated"))
                passed = bool(t.get("passed"))
                if truncated:
                    n_trunc += 1
                    if passed:
                        pass_trunc += 1
                else:
                    if passed:
                        pass_non += 1
    r_t = n_trunc / n_total if n_total else float("nan")
    n_non = n_total - n_trunc
    p_trunc = pass_trunc / n_trunc if n_trunc else None
    p_non = pass_non / n_non if n_non else None
    delta_p = (p_trunc - p_non) if (p_trunc is not None and p_non is not None) else None
    return {
        "n_total": n_total,
        "n_trunc": n_trunc,
        "n_non": n_non,
        "pass_trunc": pass_trunc,
        "pass_non": pass_non,
        "r_t": r_t,
        "p_trunc": p_trunc,
        "p_non": p_non,
        "delta_p": delta_p,
        "flag_metric": (r_t * delta_p) if delta_p is not None else None,
    }


def main() -> None:
    print(f"{'model':<24s} {'n':>6s} {'n_t':>5s} {'r_t %':>7s} "
          f"{'p_t':>6s} {'p_non':>6s} {'Δ_p':>7s} {'r_t·Δ_p':>9s} {'flag':>5s}")
    print("-" * 90)
    for label, path in TARGETS:
        r = audit(Path(path))
        flag = "YES" if (r["flag_metric"] is not None and abs(r["flag_metric"]) >= 0.04) else "no"
        rt_pct = r["r_t"] * 100
        dp = r["delta_p"]
        fm = r["flag_metric"]
        print(f"{label:<24s} {r['n_total']:>6d} {r['n_trunc']:>5d} "
              f"{rt_pct:>7.2f} {r['p_trunc']:>6.3f} {r['p_non']:>6.3f} "
              f"{dp:>+7.3f} {fm:>+9.3f}  {flag:>4s}")


if __name__ == "__main__":
    main()
