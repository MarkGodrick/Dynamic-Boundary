"""Refresh tab:claim3-truncation-V on the full ALO anchor (anchor_set_alo/).

For each (V model, category) on the merged ALO matrix:
  r_t = (# truncated trials) / (# total trials)
  Δ_p = pass_rate(truncated) - pass_rate(non-truncated), undefined if r_t == 0
  flag = |r_t * Δ_p| >= 0.04

Run:
  conda run -n boundary --no-capture-output python -u \\
      scripts/analysis/v_truncation_audit_alo_full.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

CATS = [("HRR", "harmful_refusal"),
        ("OR",  "over_refusal"),
        ("CIF", "constrained_completion"),
        ("SR",  "sycophancy_resistance")]
V_MODELS = [
    ("Olmo-3-7B-Instruct",   "allenai_Olmo_3_7B_Instruct"),
    ("granite-4.1-8b",       "ibm_granite_granite_4.1_8b"),
    ("phi-4",                "microsoft_phi_4"),
    ("Phi-4-reasoning-plus", "microsoft_Phi_4_reasoning_plus"),
    ("Falcon-H1R-7B",        "tiiuae_Falcon_H1R_7B"),
]


def audit(path: Path) -> dict:
    n_total = n_trunc = pass_trunc = pass_non = 0
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            for t in row.get("trials") or []:
                n_total += 1
                tr = bool(t.get("truncated"))
                pa = bool(t.get("passed"))
                if tr:
                    n_trunc += 1
                    if pa:
                        pass_trunc += 1
                else:
                    if pa:
                        pass_non += 1
    n_non = n_total - n_trunc
    r_t = n_trunc / n_total if n_total else 0.0
    p_t = pass_trunc / n_trunc if n_trunc else None
    p_non = pass_non / n_non if n_non else None
    delta_p = (p_t - p_non) if (p_t is not None and p_non is not None) else None
    flag = (delta_p is not None) and (abs(r_t * delta_p) >= 0.04)
    return {"r_t": r_t, "delta_p": delta_p, "flag": flag,
            "n_total": n_total, "n_trunc": n_trunc}


def fmt_cell(r: dict) -> str:
    rt_pct = r["r_t"] * 100
    if r["delta_p"] is None:
        body = f"{rt_pct:.1f}% / ---"
    else:
        body = f"{rt_pct:.1f}% / {r['delta_p']:+.2f}"
    return f"**{body}**" if r["flag"] else body


def main():
    print(f"\n{'Model':<24s} | {'HRR':<22s} | {'OR':<22s} | {'CIF':<22s} | {'SR':<22s}")
    print("-" * 124)
    for m_label, mkey in V_MODELS:
        cells = {}
        for cat_lbl, cat in CATS:
            path = ROOT / "results" / cat / "anchor_set_alo" / mkey / "response_matrix.jsonl"
            if not path.exists():
                cells[cat_lbl] = "MISSING"
            else:
                cells[cat_lbl] = fmt_cell(audit(path))
        print(f"{m_label:<24s} | {cells['HRR']:<22s} | {cells['OR']:<22s} | {cells['CIF']:<22s} | {cells['SR']:<22s}")


if __name__ == "__main__":
    main()
