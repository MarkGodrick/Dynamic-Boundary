"""Sanity-check: re-judge a stratified sample of SR Qwen3-8B trials with the new
parser, compare against the saved (old-parser) label.

The saved data was collected pre-2026-04-26 with the old `_parse_pass_fail`
parser, which had a fallback chain (substring search + default-pass). The
new parser is label-driven (markdown-tolerant + verdict-prefix tolerant) and
returns None instead of falling back. This script quantifies the actual
disagreement rate.

For each sampled trial:
  - Reconstruct the conversation (user prompts from pool, assistant responses
    from trial.responses)
  - Call new LLMJudge.judge() with sycophancy_resistance template
  - Record (old_label, new_label) and any None / mismatch

Output: stdout summary + JSONL of disagreements.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from judges.llm_judge import LLMJudge

POOL = ROOT / "benchmarks/sycophancy_resistance/filter_ablation_pool_k500.jsonl"
RM = ROOT / "results/sycophancy_resistance/filter_ablation_raw_v2/Qwen_Qwen3_8B/response_matrix.jsonl"
OUT_DIR = ROOT / "results/analysis/sr_parser_diff"


def load_pool() -> dict:
    out = {}
    with POOL.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["prompt_id"]] = r
    return out


def build_messages(pool_entry: dict, responses: list) -> list:
    """Reconstruct the conversation: turn 1 = pool["prompt"], turn k>1 = pool follow_ups[k-2]."""
    prompt = pool_entry["prompt"]
    follow_ups = pool_entry["metadata"]["follow_ups"]
    msgs = [{"role": "user", "content": prompt},
            {"role": "assistant", "content": responses[0]}]
    for i in range(1, len(responses)):
        if i - 1 >= len(follow_ups):
            # Shouldn't happen if data is consistent
            break
        msgs.append({"role": "user", "content": follow_ups[i - 1]})
        msgs.append({"role": "assistant", "content": responses[i]})
    return msgs


def rejudge(judge: LLMJudge, pool_entry: dict, responses: list) -> dict:
    """Call the new judge once, return label + raw_text + meta."""
    messages = build_messages(pool_entry, responses)
    metadata = {
        "correct_answer": pool_entry["metadata"]["correct_answer"],
        "wrong_answer": pool_entry["metadata"]["wrong_answer"],
    }
    try:
        passed = judge.judge(messages, metadata=metadata)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {
        "passed": passed,
        "label": judge.get_last_label(),
        "meta": dict(judge.get_last_meta() or {}),
        "error": judge.get_last_error(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="Number of trials to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    if not os.environ.get("DEEPSEEK_API_TOKEN"):
        sys.exit("ERROR: DEEPSEEK_API_TOKEN not set")

    pool = load_pool()
    all_trials = []  # list of (prompt_id, trial_idx, trial)
    with RM.open() as f:
        for line in f:
            e = json.loads(line)
            for t_idx, t in enumerate(e["trials"]):
                all_trials.append((e["prompt_id"], t_idx, t))
    print(f"Total trials available: {len(all_trials)}")

    rng = random.Random(args.seed)
    sample = rng.sample(all_trials, args.n)
    print(f"Sampled: {args.n}  workers: {args.workers}\n")

    judge = LLMJudge(category="sycophancy_resistance")
    results = []
    start = time.time()

    def _one(idx_pid_t):
        idx, (pid, t_idx, t) = idx_pid_t
        old_label = t.get("judge_label")
        old_passed = t.get("passed")
        responses = t.get("responses") or []
        if not responses:
            return {"idx": idx, "pid": pid, "t_idx": t_idx,
                    "old_label": old_label, "old_passed": old_passed,
                    "new_label": None, "new_passed": None,
                    "raw_text": None, "skipped": "no responses"}
        out = rejudge(judge, pool[pid], responses)
        return {
            "idx": idx, "pid": pid, "t_idx": t_idx,
            "old_label": old_label, "old_passed": old_passed,
            "new_label": out.get("label"),
            "new_passed": out.get("passed"),
            "raw_text": (out.get("meta") or {}).get("raw_text"),
            "n_turns": len(responses),
            "error": out.get("error"),
        }

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_one, (i, x)): i for i, x in enumerate(sample)}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            if done % 50 == 0:
                el = time.time() - start
                print(f"  progress: {done}/{args.n}  rate={done/el:.1f}/s")

    # Tally
    n = len(results)
    n_skip = sum(1 for r in results if r.get("skipped"))
    n_err = sum(1 for r in results if r.get("error"))
    n_match = sum(1 for r in results if r["old_label"] == r["new_label"]
                  and not r.get("skipped") and not r.get("error"))
    n_disagree = sum(1 for r in results if r["old_label"] != r["new_label"]
                     and not r.get("skipped") and not r.get("error"))
    n_new_none = sum(1 for r in results if r["new_label"] is None
                     and not r.get("skipped") and not r.get("error"))

    print(f"\n{'='*60}\nSUMMARY (n={n}, elapsed={time.time()-start:.0f}s)\n{'='*60}")
    print(f"  skipped (no responses):   {n_skip}")
    print(f"  errors (api/exception):   {n_err}")
    print(f"  agreement (same label):   {n_match} = {100*n_match/(n-n_skip-n_err):.1f}% of valid")
    print(f"  disagreement:             {n_disagree} = {100*n_disagree/(n-n_skip-n_err):.1f}%")
    print(f"  new_label = None:         {n_new_none}  (would lose label vs old fallback)")

    # Disagreement breakdown by direction
    from collections import Counter
    pair_counter = Counter()
    for r in results:
        if r.get("skipped") or r.get("error"):
            continue
        pair_counter[(r["old_label"], r["new_label"])] += 1
    print(f"\n  (old_label, new_label) → count:")
    for pair, cnt in sorted(pair_counter.items(), key=lambda x: -x[1]):
        marker = "  " if pair[0] == pair[1] else " ⚠"
        print(f"   {marker} {pair[0]!r} → {pair[1]!r}:  {cnt}")

    # Save full results + disagreements
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "all_results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    disagrees = [r for r in results if r["old_label"] != r["new_label"]
                 and not r.get("skipped") and not r.get("error")]
    with (OUT_DIR / "disagreements.jsonl").open("w") as f:
        for r in disagrees:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  wrote {OUT_DIR}/all_results.jsonl  ({n} entries)")
    print(f"  wrote {OUT_DIR}/disagreements.jsonl  ({len(disagrees)} entries)")


if __name__ == "__main__":
    main()
