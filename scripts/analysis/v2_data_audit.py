"""Audit canonical filter_ablation cells for schema integrity, errors, and judge consistency.

Per category dir is resolved via core.result_paths.filter_ablation_dir() —
HR/OR read v2_judgefix (post 2026-04-26 rejudge), CC/SR read v2.

Three checks per (category, model) cell:
  1. Schema integrity — every trial has the expected Gen 2 keys.
  2. Errors — judge_outcome != "ok", judge_error counts, api_failed, trial_failed,
     truncated trials, judge truncation, judge_label parse failures.
  3. Response ↔ judge consistency:
        HR: judge_label in {safe, unsafe}; passed == (judge_label == "safe").
        OR: judge_label in {appropriate, over_refusal}; passed == (judge_label == "appropriate").
        CC: rule-based, judge_label is always None by design — check passed in {True, False}.

Run:
    python scripts/analysis/v2_data_audit.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.result_paths import filter_ablation_dir

CATEGORIES = {
    "harmful_refusal": "HR",
    "over_refusal": "OR",
    "constrained_completion": "CC",
}

REQUIRED_TOP_KEYS = {
    "prompt_id", "model", "trials", "pass_count", "total_trials",
    "n_attempted", "n_safety_classified", "n_api_failed", "n_trial_failed",
    "n_truncated", "n_judge_truncated", "p_hat",
}
REQUIRED_TRIAL_KEYS = {
    "seed", "response", "passed", "failed_at_turn",
    "judge_outcome", "judge_error", "judge_label",
    "truncated", "completion_tokens", "per_turn_meta", "per_turn_judge_meta",
}

EXPECTED_LABELS = {
    "harmful_refusal":   {"safe", "unsafe"},
    # OR judge is 3-class: 'appropriate' (pass), 'over_refusal' (fail by refusing benign),
    # 'unsafe' (fail by producing harmful content despite benign prompt).
    "over_refusal":      {"appropriate", "over_refusal", "unsafe"},
    "constrained_completion": set(),  # rule-based; label is always None
}
PASS_LABEL = {
    "harmful_refusal":   "safe",
    "over_refusal":      "appropriate",
}


def audit_cell(cat: str, cell_dir: Path) -> dict:
    rm_path = cell_dir / "response_matrix.jsonl"
    n_prompts = 0
    n_trials = 0

    missing_top_keys = Counter()
    missing_trial_keys = Counter()

    judge_outcomes = Counter()
    n_judge_error = 0
    n_api_failed_sum = 0
    n_trial_failed_sum = 0
    n_truncated_sum = 0
    n_judge_truncated_sum = 0

    label_dist = Counter()
    n_label_unexpected = 0
    n_judge_inconsistent = 0  # passed mismatches judge_label
    n_pass_label_none = 0     # passed=True but label=None (HR/OR only)

    pass_count_total = 0
    completion_tokens = []

    with rm_path.open() as f:
        for line in f:
            r = json.loads(line)
            n_prompts += 1
            for k in REQUIRED_TOP_KEYS:
                if k not in r:
                    missing_top_keys[k] += 1

            pass_count_total += r.get("pass_count", 0)
            for t in r.get("trials", []):
                n_trials += 1
                for k in REQUIRED_TRIAL_KEYS:
                    if k not in t:
                        missing_trial_keys[k] += 1

                outcome = t.get("judge_outcome")
                judge_outcomes[outcome] += 1
                if t.get("judge_error"):
                    n_judge_error += 1
                if t.get("truncated"):
                    n_truncated_sum += 1
                ct = t.get("completion_tokens")
                if isinstance(ct, int):
                    completion_tokens.append(ct)

                # judge truncation check via per_turn_judge_meta
                ptjm = t.get("per_turn_judge_meta") or []
                for m in ptjm:
                    if isinstance(m, dict) and m.get("truncated"):
                        n_judge_truncated_sum += 1
                        break

                label = t.get("judge_label")
                passed = t.get("passed")
                label_dist[label] += 1

                if cat in PASS_LABEL:  # HR/OR
                    if label is not None and label not in EXPECTED_LABELS[cat]:
                        n_label_unexpected += 1
                    if label is None and passed is True:
                        n_pass_label_none += 1
                    if label is not None and (passed == (label == PASS_LABEL[cat])) is False:
                        n_judge_inconsistent += 1
                else:  # CC
                    if label is not None:
                        n_label_unexpected += 1

            n_api_failed_sum += r.get("n_api_failed", 0)
            n_trial_failed_sum += r.get("n_trial_failed", 0)

    return {
        "n_prompts": n_prompts,
        "n_trials": n_trials,
        "missing_top_keys": dict(missing_top_keys),
        "missing_trial_keys": dict(missing_trial_keys),
        "judge_outcomes": dict(judge_outcomes),
        "n_judge_error": n_judge_error,
        "n_api_failed": n_api_failed_sum,
        "n_trial_failed": n_trial_failed_sum,
        "n_truncated": n_truncated_sum,
        "n_judge_truncated": n_judge_truncated_sum,
        "label_dist": dict(label_dist),
        "n_label_unexpected": n_label_unexpected,
        "n_judge_inconsistent": n_judge_inconsistent,
        "n_pass_label_none": n_pass_label_none,
        "p_hat_overall": pass_count_total / n_trials if n_trials else 0.0,
        "ct_p50": _pct(completion_tokens, 50),
        "ct_p95": _pct(completion_tokens, 95),
        "ct_max": max(completion_tokens) if completion_tokens else 0,
    }


def _pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[int(len(xs) * p / 100)]


def main():
    print(f"Audit root: {ROOT}\n")
    for cat, tag in CATEGORIES.items():
        cat_root = filter_ablation_dir(cat)
        if not cat_root.exists():
            print(f"[{tag}] no dir at {cat_root}, skip")
            continue
        cells = sorted(p for p in cat_root.iterdir() if p.is_dir())
        print(f"=== {tag} ({cat}) — {len(cells)} cell(s) at {cat_root.name} ===")
        for cell in cells:
            r = audit_cell(cat, cell)
            n_t = r["n_trials"]
            print(f"\n  {cell.name}")
            print(f"    prompts={r['n_prompts']:<4} trials={n_t:<6} p_hat_overall={r['p_hat_overall']:.3f}")
            # schema
            if r["missing_top_keys"] or r["missing_trial_keys"]:
                print(f"    SCHEMA MISSING top={r['missing_top_keys']} trial={r['missing_trial_keys']}")
            else:
                print(f"    schema    OK (all keys present)")
            # errors
            print(f"    judge_outcome: {r['judge_outcomes']}")
            print(f"    judge_error={r['n_judge_error']} ({_pct_str(r['n_judge_error'], n_t)})  "
                  f"api_failed={r['n_api_failed']}  trial_failed={r['n_trial_failed']}")
            print(f"    truncated={r['n_truncated']} ({_pct_str(r['n_truncated'], n_t)})  "
                  f"judge_truncated={r['n_judge_truncated']} ({_pct_str(r['n_judge_truncated'], n_t)})")
            print(f"    completion_tokens p50={r['ct_p50']} p95={r['ct_p95']} max={r['ct_max']}")
            # consistency
            print(f"    label_dist: {r['label_dist']}")
            if r["n_label_unexpected"]:
                print(f"    !! UNEXPECTED LABELS: {r['n_label_unexpected']}")
            if cat in PASS_LABEL:
                print(f"    judge_inconsistent (passed != expected): {r['n_judge_inconsistent']} ({_pct_str(r['n_judge_inconsistent'], n_t)})")
                print(f"    label=None & passed=True: {r['n_pass_label_none']}")


def _pct_str(num, denom):
    if denom == 0:
        return "—"
    return f"{100 * num / denom:.2f}%"


if __name__ == "__main__":
    main()
