"""Step 1 — Diagnostic study: why do v2 LLM-judge cells produce judge_label=None?

Stratified random sample → fresh re-judge → capture raw text + meta. No
pre-baked bucket classification — raw_text is preserved per-trial so we
can inspect patterns post-hoc.

Cells (parameterized for SR future):
  HR: Qwen2.5-72B / Qwen3-30B-A3B / gemma-3-27b
  OR: Qwen2.5-72B / Qwen3-30B-A3B
  SR: skipped (no v2 data; CELLS["sycophancy_resistance"] is empty)

Strata = unique values of saved `judge_label` per category.

Output:
  results/analysis/judge_audit/per_trial.jsonl
  results/analysis/judge_audit/summary.json

Run:
  python scripts/analysis/judge_label_audit.py
  python scripts/analysis/judge_label_audit.py --per-stratum 100 --workers 24
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from judges.llm_judge import LLMJudge

POOL = {
    "harmful_refusal":       ROOT / "benchmarks/harmful_refusal/filter_ablation_pool_k500.jsonl",
    "over_refusal":          ROOT / "benchmarks/over_refusal/filter_ablation_pool_k500.jsonl",
    "sycophancy_resistance": ROOT / "benchmarks/sycophancy_resistance/filter_ablation_pool_k500.jsonl",
}

# (cat, model_dir_name) per cell, cells share the v2 data root.
CELLS = {
    "harmful_refusal": [
        "Qwen_Qwen2.5_72B_Instruct_AWQ",
        "Qwen_Qwen3_30B_A3B_Instruct_2507",
        "google_gemma_3_27b_it",
    ],
    "over_refusal": [
        "Qwen_Qwen2.5_72B_Instruct_AWQ",
        "Qwen_Qwen3_30B_A3B_Instruct_2507",
    ],
    "sycophancy_resistance": [],  # no v2 data yet
}

# Strata = saved-label vocabulary per category. None is encoded as the
# Python None object inside the trial dict; we keep it as None here so
# `stratified_sample` keys match `trial["judge_label"]` directly.
STRATA = {
    "harmful_refusal":       [None, "safe", "unsafe"],
    "over_refusal":          [None, "appropriate", "over_refusal", "unsafe"],
    "sycophancy_resistance": [None, "pass", "fail"],
}


def load_pool(cat: str) -> dict:
    out = {}
    with POOL[cat].open() as f:
        for line in f:
            r = json.loads(line)
            out[r["prompt_id"]] = r
    return out


def gather_trials_by_label(rm_path: Path) -> dict:
    """Walk a response_matrix.jsonl, return {saved_label: [(pid, idx, trial), ...]}."""
    by_label = defaultdict(list)
    with rm_path.open() as f:
        for line in f:
            r = json.loads(line)
            for i, t in enumerate(r["trials"]):
                by_label[t["judge_label"]].append((r["prompt_id"], i, t))
    return by_label


def stratified_sample(by_label: dict, strata: list, per_stratum: int, seed: int) -> list:
    rng = random.Random(seed)
    out = []
    for s in strata:
        candidates = by_label.get(s, [])
        n = min(len(candidates), per_stratum)
        chosen = rng.sample(candidates, n) if n > 0 else []
        for pid, idx, t in chosen:
            out.append({
                "prompt_id": pid,
                "trial_idx": idx,
                "trial": t,
                "saved_label": s,
            })
    return out


def fresh_judge(cat: str, user_prompt: str, ai_response: str) -> dict:
    """Re-run LLMJudge on (prompt, response). Capture raw text via
    monkey-patch of `_call_api` since the production code doesn't stash
    it on `_local.meta` yet (that's Step 0)."""
    judge = LLMJudge(category=cat)
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": ai_response},
    ]
    captured = {}
    orig = judge._call_api

    def wrap(msgs):
        text = orig(msgs)
        captured["text"] = text
        return text

    judge._call_api = wrap
    try:
        verdict = judge.judge(messages)
    except Exception as e:
        return {
            "fresh_passed": None,
            "fresh_label": None,
            "fresh_raw_text": captured.get("text", ""),
            "fresh_finish_reason": None,
            "fresh_completion_tokens": None,
            "fresh_truncated": None,
            "fresh_error": f"{type(e).__name__}: {e}",
        }

    meta = judge._local.meta or {}
    return {
        "fresh_passed": verdict,
        "fresh_label": judge._local.last_label,
        "fresh_raw_text": captured.get("text", ""),
        "fresh_finish_reason": meta.get("finish_reason"),
        "fresh_completion_tokens": meta.get("completion_tokens"),
        "fresh_truncated": meta.get("truncated"),
        "fresh_error": None,
    }


def process_sample(cat: str, cell: str, pool: dict, sample: dict) -> dict:
    pid = sample["prompt_id"]
    trial = sample["trial"]
    fresh = fresh_judge(cat, pool[pid]["prompt"], trial["response"])
    return {
        "cat": cat,
        "cell": cell,
        "prompt_id": pid,
        "trial_idx": sample["trial_idx"],
        "saved_label": sample["saved_label"],
        "saved_passed": trial["passed"],
        "saved_truncated_target": trial.get("truncated"),
        "saved_per_turn_judge_meta": trial.get("per_turn_judge_meta"),
        **fresh,
    }


def build_summary(results: list) -> dict:
    """Per (cell, saved_stratum): agreement rate, fresh label dist,
    fresh truncation rate, error rate. No bucket classification — that's
    done post-hoc on the per_trial.jsonl."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[f"{r['cat']}/{r['cell']}"][str(r["saved_label"])].append(r)

    out = {}
    for cell_key, strata in grouped.items():
        out[cell_key] = {}
        for stratum, items in strata.items():
            n = len(items)
            agree = sum(
                1 for r in items
                if (r["fresh_label"] == r["saved_label"])
                or (r["fresh_label"] is None and r["saved_label"] in (None, "None"))
            )
            fresh_dist = Counter(str(r["fresh_label"]) for r in items)
            trunc = sum(1 for r in items if r["fresh_truncated"])
            errs = sum(1 for r in items if r.get("fresh_error"))
            out[cell_key][stratum] = {
                "n": n,
                "agreement_rate": agree / n if n else 0,
                "fresh_label_dist": dict(fresh_dist),
                "fresh_truncated_count": trunc,
                "fresh_error_count": errs,
            }
    return out


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 100)
    print("SUMMARY  (per cell × saved-label stratum)")
    print("=" * 100)
    for cell_key, strata in summary.items():
        print(f"\n  {cell_key}")
        for stratum, s in strata.items():
            tag = f"saved={stratum:>14}"
            print(f"    {tag} (n={s['n']:>3})  "
                  f"agree={s['agreement_rate']:.2f}  "
                  f"fresh_truncated={s['fresh_truncated_count']:>2}  "
                  f"errs={s['fresh_error_count']}")
            print(f"        fresh_label_dist: {s['fresh_label_dist']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=50)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/analysis/judge_audit")
    ap.add_argument("--cats", nargs="+",
                    default=["harmful_refusal", "over_refusal"])
    ap.add_argument("--cells", nargs="*", default=None,
                    help="Optional model_dir filter (substring match)")
    args = ap.parse_args()

    if not os.environ.get("DEEPSEEK_API_TOKEN"):
        sys.exit("ERROR: DEEPSEEK_API_TOKEN not set in env")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    per_trial_path = out_dir / "per_trial.jsonl"
    summary_path = out_dir / "summary.json"

    pools = {cat: load_pool(cat) for cat in args.cats}

    # 1) Build the sample list across cells × strata
    all_samples = []
    for cat in args.cats:
        for cell in CELLS.get(cat, []):
            if args.cells and not any(f in cell for f in args.cells):
                continue
            rm_path = ROOT / "results" / cat / "filter_ablation_raw_v2" / cell / "response_matrix.jsonl"
            if not rm_path.exists():
                print(f"  skip {cat}/{cell}: not found")
                continue
            by_label = gather_trials_by_label(rm_path)
            available = {str(k): len(v) for k, v in by_label.items()}
            samples = stratified_sample(by_label, STRATA[cat], args.per_stratum, args.seed)
            print(f"  {cat}/{cell}: available {available}  → sampled {len(samples)}")
            for s in samples:
                s["cat"] = cat
                s["cell"] = cell
                all_samples.append(s)

    print(f"\nTotal samples: {len(all_samples)}\n"
          f"Workers: {args.workers}\n"
          f"Per-trial out: {per_trial_path}\n"
          f"Summary out:   {summary_path}\n")

    # 2) Run fresh judge in parallel
    results = []
    with per_trial_path.open("w") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(process_sample, s["cat"], s["cell"], pools[s["cat"]], s): s
                for s in all_samples
            }
            done = 0
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {
                        "cat": s["cat"], "cell": s["cell"],
                        "prompt_id": s["prompt_id"], "trial_idx": s["trial_idx"],
                        "saved_label": s["saved_label"],
                        "fresh_error": f"OUTER: {type(e).__name__}: {e}",
                    }
                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                f_out.flush()
                results.append(r)
                done += 1
                if done % 50 == 0:
                    print(f"  progress: {done}/{len(all_samples)}")

    # 3) Summarize
    summary = build_summary(results)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print_summary(summary)


if __name__ == "__main__":
    main()
