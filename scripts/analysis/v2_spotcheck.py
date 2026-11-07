"""Spot-check v2 trials: pull prompt + response + judge label, hand-check whether they agree.

Picks targeted samples from each category to expose disagreement risks:
  HR: judge_label=None & passed=True (parser fall-through bias)
  HR: truncated trial (does the partial response justify the label?)
  OR: judge_label='unsafe' (model produced harmful content on benign-looking prompt — sanity check)
  OR: judge_label='over_refusal' (model refused a benign request)
  CC: passed=True (rule-based pass — does response actually satisfy the constraint type?)
  CC: passed=False (rule-based fail — does response actually violate?)

Run:
    python scripts/analysis/v2_spotcheck.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.result_paths import filter_ablation_dir

POOL_PATH = {
    "harmful_refusal":   ROOT / "benchmarks/harmful_refusal/filter_ablation_pool_k500.jsonl",
    "over_refusal":      ROOT / "benchmarks/over_refusal/filter_ablation_pool_k500.jsonl",
    "constrained_completion": ROOT / "benchmarks/constrained_completion/filter_ablation_pool_k500.jsonl",
}


def load_prompts(cat: str) -> dict:
    """Return {prompt_id: full_record} from the filter_ablation pool."""
    out = {}
    with POOL_PATH[cat].open() as f:
        for line in f:
            r = json.loads(line)
            out[r["prompt_id"]] = r
    return out


def find_trials(rm_path: Path, predicate, n=2):
    out = []
    with rm_path.open() as f:
        for line in f:
            r = json.loads(line)
            for t in r["trials"]:
                if predicate(t, r):
                    out.append((r["prompt_id"], t))
                    if len(out) >= n:
                        return out
    return out


def show(label: str, prompts: dict, trials):
    print(f"\n{'='*100}\n{label}\n{'='*100}")
    for pid, t in trials:
        p = prompts.get(pid, {})
        print(f"\n--- prompt_id={pid} ---")
        print(f"BARE_REQUEST: {p.get('bare_request', '<missing>')[:200]}")
        print(f"SKILLS: {p.get('skills', [])}")
        if "constraints" in p:
            print(f"CONSTRAINTS: {p.get('constraints')}")
        print(f"PROMPT (first 300 chars):\n{p.get('prompt', '')[:300]}")
        print(f"\nRESPONSE (first 500 chars):\n{t['response'][:500]}")
        print(f"\nRESPONSE (last 200 chars):\n...{t['response'][-200:]}")
        print(f"\n>> truncated={t['truncated']}, judge_label={t['judge_label']!r}, passed={t['passed']}, completion_tokens={t['completion_tokens']}")


def main():
    # Harmful refusal
    hr_path = filter_ablation_dir("harmful_refusal") / "Qwen_Qwen2.5_72B_Instruct_AWQ" / "response_matrix.jsonl"
    hr_prompts = load_prompts("harmful_refusal")
    show(
        "HR / Qwen2.5-72B  judge_label=None & passed=True (parser fall-through bias)",
        hr_prompts,
        find_trials(hr_path, lambda t, r: t["judge_label"] is None and t["passed"] is True, n=2),
    )
    show(
        "HR / Qwen2.5-72B  truncated=True (was the cut-off response correctly judged?)",
        hr_prompts,
        find_trials(hr_path, lambda t, r: t["truncated"] is True, n=2),
    )
    show(
        "HR / gemma-3-27b  truncated=True (gemma3 has 12% truncation — sample one)",
        load_prompts("harmful_refusal"),
        find_trials(
            filter_ablation_dir("harmful_refusal") / "google_gemma_3_27b_it" / "response_matrix.jsonl",
            lambda t, r: t["truncated"] is True, n=1,
        ),
    )

    # Over refusal
    or_path = filter_ablation_dir("over_refusal") / "Qwen_Qwen2.5_72B_Instruct_AWQ" / "response_matrix.jsonl"
    or_prompts = load_prompts("over_refusal")
    show(
        "OR / Qwen2.5-72B  judge_label='unsafe' (model produced harmful content)",
        or_prompts,
        find_trials(or_path, lambda t, r: t["judge_label"] == "unsafe", n=2),
    )
    show(
        "OR / Qwen2.5-72B  judge_label='over_refusal' (model refused benign request)",
        or_prompts,
        find_trials(or_path, lambda t, r: t["judge_label"] == "over_refusal", n=2),
    )

    # Constrained completion
    cc_path = filter_ablation_dir("constrained_completion") / "google_gemma_3_12b_it" / "response_matrix.jsonl"
    cc_prompts = load_prompts("constrained_completion")
    show(
        "CC / gemma-3-12b  passed=True (does response satisfy IFEval constraints?)",
        cc_prompts,
        find_trials(cc_path, lambda t, r: t["passed"] is True and not t["truncated"], n=2),
    )
    show(
        "CC / gemma-3-12b  passed=False (does response actually violate?)",
        cc_prompts,
        find_trials(cc_path, lambda t, r: t["passed"] is False and not t["truncated"], n=2),
    )


if __name__ == "__main__":
    main()
