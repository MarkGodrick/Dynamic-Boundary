"""Compute per-(category, model) truncation rate and pass-rate delta from
anchor_set response matrices. Covers all 9 panel + 5 validation models on the
4 categories. Helps decide which cells need a max_tokens bump + re-run.

Outputs:
  - stdout: markdown table sorted by trunc_rate
  - results/analysis/truncation_audit/truncation_summary.json

Definition:
  truncation_rate     = (#trials with trial-level truncated=True) / (#trials with truncated field)
  pass_trunc          = mean(passed) over truncated trials
  pass_nontrunc       = mean(passed) over non-truncated trials
  delta_pass          = pass_trunc - pass_nontrunc
                        (negative => truncation hurts pass; positive => truncation looks like pass)

Trials that lack a `truncated` field (legacy Gen 0/1 data) are reported under
`n_unknown`; if `n_unknown == n_trials` the cell is reported as 'no schema'.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT_DIR = RESULTS / "analysis" / "truncation_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATS = [
    "harmful_refusal",
    "over_refusal",
    "constrained_completion",
    "sycophancy_resistance",
]

PANEL = {
    "Qwen_Qwen2.5_72B_Instruct_AWQ",
    "Qwen_Qwen3_30B_A3B_Instruct_2507",
    "google_gemma_3_27b_it",
    "Qwen_Qwen3_8B",
    "google_gemma_3_12b_it",
    "mistralai_Mistral_Small_3.1_24B_Instruct_2503",
    "google_gemma_3_4b_it",
    "mistralai_Mistral_Nemo_Instruct_2407",
    "deepseek_ai_DeepSeek_R1_Distill_Llama_8B",
}

VALIDATION = {
    "allenai_Olmo_3_7B_Instruct",
    "ibm_granite_granite_4.1_8b",
    "microsoft_phi_4",
    "microsoft_Phi_4_reasoning_plus",
    "tiiuae_Falcon_H1R_7B",
}

# Extra OOD models that show up in the panel dirs but are not in M=9 panel /
# V=5 validation set (e.g. Olmo-3.1-32B, R1-Distill-70B, R1-Distill-Qwen-32B,
# gpt-oss-20b). Tag as 'other' so they show up but are clearly separated.
def role_of(model_dir: str) -> str:
    if model_dir in PANEL:
        return "panel"
    if model_dir in VALIDATION:
        return "V"
    return "other"


def cell_stats(rm_path: Path) -> dict:
    n_trials = 0
    n_unknown = 0  # missing `truncated` field
    n_trunc = 0
    pass_trunc_sum = 0
    pass_nontrunc_sum = 0
    n_trunc_with_pass = 0
    n_nontrunc_with_pass = 0
    completion_tokens_trunc = []
    completion_tokens_all = []

    with open(rm_path) as f:
        for line in f:
            row = json.loads(line)
            for t in row.get("trials", []):
                n_trials += 1
                tr = t.get("truncated")
                ct = t.get("completion_tokens")
                if isinstance(ct, (int, float)):
                    completion_tokens_all.append(int(ct))
                if tr is None:
                    n_unknown += 1
                    continue
                tr = bool(tr)
                if tr:
                    n_trunc += 1
                    if isinstance(ct, (int, float)):
                        completion_tokens_trunc.append(int(ct))
                p = t.get("passed")
                if p is None:
                    continue
                p = bool(p)
                if tr:
                    pass_trunc_sum += int(p)
                    n_trunc_with_pass += 1
                else:
                    pass_nontrunc_sum += int(p)
                    n_nontrunc_with_pass += 1

    n_known = n_trials - n_unknown
    out = {
        "n_trials": n_trials,
        "n_unknown_trunc": n_unknown,
        "n_trunc": n_trunc,
        "trunc_rate": (n_trunc / n_known) if n_known else None,
        "pass_trunc": (pass_trunc_sum / n_trunc_with_pass) if n_trunc_with_pass else None,
        "pass_nontrunc": (pass_nontrunc_sum / n_nontrunc_with_pass) if n_nontrunc_with_pass else None,
        "delta_pass": None,
        "p95_completion_tokens": None,
        "max_completion_tokens": None,
    }
    if out["pass_trunc"] is not None and out["pass_nontrunc"] is not None:
        out["delta_pass"] = out["pass_trunc"] - out["pass_nontrunc"]
    if completion_tokens_all:
        ct = sorted(completion_tokens_all)
        out["p95_completion_tokens"] = ct[int(0.95 * (len(ct) - 1))]
        out["max_completion_tokens"] = ct[-1]
    return out


def fmt(x: Optional[float], digits: int = 3, percent: bool = False, signed: bool = False) -> str:
    if x is None:
        return "—"
    if percent:
        return f"{x*100:.{digits}f}%"
    if signed:
        return f"{x:+.{digits}f}"
    return f"{x:.{digits}f}"


# Per-category data source: V + others live in anchor_set; panel lives in
# filter_ablation_raw_v2_judgefix (HR/OR) or filter_ablation_raw_v2 (CC/SR).
PANEL_DIR = {
    "harmful_refusal": "filter_ablation_raw_v2_judgefix",
    "over_refusal": "filter_ablation_raw_v2_judgefix",
    "constrained_completion": "filter_ablation_raw_v2",
    "sycophancy_resistance": "filter_ablation_raw_v2",
}


def main():
    rows = []
    for cat in CATS:
        seen = set()
        # Anchor_set: V + held-out extras
        anchor = RESULTS / cat / "anchor_set"
        if anchor.exists():
            for d in sorted(anchor.iterdir()):
                rm = d / "response_matrix.jsonl"
                if not rm.exists():
                    continue
                stats = cell_stats(rm)
                stats["category"] = cat
                stats["model"] = d.name
                stats["role"] = role_of(d.name)
                stats["source"] = "anchor_set"
                rows.append(stats)
                seen.add(d.name)
        # Panel
        panel_root = RESULTS / cat / PANEL_DIR[cat]
        if panel_root.exists():
            for d in sorted(panel_root.iterdir()):
                if d.name in seen:
                    continue
                rm = d / "response_matrix.jsonl"
                if not rm.exists():
                    continue
                stats = cell_stats(rm)
                stats["category"] = cat
                stats["model"] = d.name
                stats["role"] = role_of(d.name)
                stats["source"] = PANEL_DIR[cat]
                rows.append(stats)

    # Save JSON
    out_json = OUT_DIR / "truncation_summary.json"
    out_json.write_text(json.dumps(rows, indent=2))

    # Print markdown grouped by category, sorted by trunc_rate desc
    print("# Truncation summary — anchor_set")
    print()
    print(f"Models: panel={len(PANEL)}, V={len(VALIDATION)}, plus 'other' (held-out / extra).\n")
    for cat in CATS:
        print(f"\n## {cat}\n")
        cat_rows = [r for r in rows if r["category"] == cat]
        cat_rows.sort(key=lambda r: (-(r["trunc_rate"] or -1), r["model"]))
        print("| role  | model | n_trials | trunc | trunc_rate | p95_tok | max_tok | pass_trunc | pass_nontrunc | Δpass |")
        print("|-------|-------|---------:|------:|-----------:|--------:|--------:|-----------:|--------------:|------:|")
        for r in cat_rows:
            print(
                f"| {r['role']:5s} | {r['model']:50s} "
                f"| {r['n_trials']:7d} "
                f"| {r['n_trunc']:5d} "
                f"| {fmt(r['trunc_rate'], 4, percent=True)} "
                f"| {r['p95_completion_tokens'] if r['p95_completion_tokens'] is not None else '—'} "
                f"| {r['max_completion_tokens'] if r['max_completion_tokens'] is not None else '—'} "
                f"| {fmt(r['pass_trunc'], 3)} "
                f"| {fmt(r['pass_nontrunc'], 3)} "
                f"| {fmt(r['delta_pass'], 3, signed=True)} |"
            )

    # Aggregate by role × cat: weighted trunc_rate
    print("\n\n## Aggregate by role × category\n")
    print("| category | role | cells | n_trials | n_trunc | trunc_rate | mean Δpass (cell-weighted) |")
    print("|----------|------|------:|---------:|--------:|-----------:|---------------------------:|")
    for cat in CATS:
        for role in ["panel", "V", "other"]:
            sel = [r for r in rows if r["category"] == cat and r["role"] == role]
            if not sel:
                continue
            n = sum(r["n_trials"] for r in sel)
            tr = sum(r["n_trunc"] for r in sel)
            n_known = sum(r["n_trials"] - r["n_unknown_trunc"] for r in sel)
            rate = tr / n_known if n_known else None
            dps = [r["delta_pass"] for r in sel if r["delta_pass"] is not None]
            mdp = sum(dps) / len(dps) if dps else None
            print(
                f"| {cat:25s} | {role:5s} | {len(sel):3d} | {n:7d} | {tr:6d} "
                f"| {fmt(rate, 4, percent=True)} | {fmt(mdp, 3, signed=True)} |"
            )

    print(f"\nJSON saved to {out_json}")


if __name__ == "__main__":
    main()
