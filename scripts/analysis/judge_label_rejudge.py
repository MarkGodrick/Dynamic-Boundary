"""Step 3 — Re-judge all v2 LLMJudge trials with judge_label=None.

For each cell under `results/{cat}/filter_ablation_raw_v2/`, find trials where
`judge_label is None` AND `judge_outcome == "ok"` (i.e., the judge API call
succeeded but the prior strict regex couldn't extract a verdict). Re-call the
judge with the fixed parser (Step 0: markdown-tolerant + verdict-prefix-tolerant
+ raw_text stash). Replace `passed`, `judge_label`, `judge_outcome`,
`judge_error`, and the corresponding `per_turn_judge_meta` entry. Re-aggregate
per-prompt stats. Write to `filter_ablation_raw_v2_judgefix/` in parallel
(does not modify v2).

Skipped:
  - `judge_outcome != "ok"` (api_exhausted / safety_classified / etc.) —
    re-running won't recover these; they're tracked separately.
  - Cells without LLMJudge (constrained_completion: rule-based; SR: no v2 data).
  - Trials already with non-None `judge_label` — kept as-is (no re-judge).

Cells are auto-discovered from the v2 directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.trial_utils import aggregate_trials, judge_and_capture
from judges.llm_judge import LLMJudge

POOL = {
    "harmful_refusal":   ROOT / "benchmarks/harmful_refusal/filter_ablation_pool_k500.jsonl",
    "over_refusal":      ROOT / "benchmarks/over_refusal/filter_ablation_pool_k500.jsonl",
}

CATS = ["harmful_refusal", "over_refusal"]


def load_pool(cat: str) -> dict:
    out = {}
    with POOL[cat].open() as f:
        for line in f:
            r = json.loads(line)
            out[r["prompt_id"]] = r
    return out


def discover_cells(cat: str) -> list:
    base = ROOT / "results" / cat / "filter_ablation_raw_v2"
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def rejudge_one(judge, prompt_text: str, response_text: str) -> tuple:
    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": response_text},
    ]
    try:
        return judge_and_capture(judge, messages, metadata=None)
    except Exception as e:
        return (None, "trial_exception",
                {"message": f"{type(e).__name__}: {e}"}, None, None)


def process_cell(cat: str, cell: str, workers: int, dry_run: bool) -> dict:
    src_dir = ROOT / "results" / cat / "filter_ablation_raw_v2" / cell
    src_rm = src_dir / "response_matrix.jsonl"
    src_meta = src_dir / "metadata.json"
    if not src_rm.exists():
        print(f"  {cat}/{cell}: no response_matrix.jsonl, skip")
        return {}

    dst_dir = ROOT / "results" / cat / "filter_ablation_raw_v2_judgefix" / cell
    dst_rm = dst_dir / "response_matrix.jsonl"
    dst_meta = dst_dir / "metadata.json"

    pool = load_pool(cat)
    entries = []
    with src_rm.open() as f:
        for line in f:
            entries.append(json.loads(line))

    # Worklist: only re-judge trials where the API succeeded but the parser
    # returned None. Other failure modes (api_exhausted etc.) need different
    # handling (out of scope for parser fix).
    worklist = []
    for e_idx, e in enumerate(entries):
        for t_idx, t in enumerate(e["trials"]):
            if t.get("judge_label") is None and t.get("judge_outcome") == "ok":
                worklist.append((e_idx, t_idx))

    n_total_trials = sum(len(e["trials"]) for e in entries)
    print(f"  {cat}/{cell}: {len(entries)} prompts, {n_total_trials} trials, "
          f"{len(worklist)} to re-judge ({100*len(worklist)/n_total_trials:.1f}%)")

    if dry_run:
        return {"cell": cell, "n_to_rejudge": len(worklist), "n_total": n_total_trials}

    if not worklist:
        # Nothing to do — but still copy through so downstream scripts can
        # use the judgefix dir uniformly.
        dst_dir.mkdir(parents=True, exist_ok=True)
        with dst_rm.open("w") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        if src_meta.exists():
            md = json.loads(src_meta.read_text())
            md["judge_rejudge"] = {
                "rejudged_at": datetime.now().isoformat(),
                "n_rejudged": 0,
                "n_total_trials": n_total_trials,
                "judge_code_version": "Step 0 (label-driven, markdown-tolerant)",
            }
            dst_meta.write_text(json.dumps(md, indent=2))
        return {"cell": cell, "n_rejudged": 0, "n_total": n_total_trials}

    judge = LLMJudge(category=cat)
    start = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(rejudge_one, judge, pool[entries[e_idx]["prompt_id"]]["prompt"],
                      entries[e_idx]["trials"][t_idx]["response"]): (e_idx, t_idx)
            for (e_idx, t_idx) in worklist
        }
        for fut in as_completed(futures):
            e_idx, t_idx = futures[fut]
            try:
                passed, outcome, err, label, meta = fut.result()
            except Exception as e:
                passed, outcome, err, label, meta = (
                    None, "trial_exception",
                    {"message": f"OUTER: {type(e).__name__}: {e}"},
                    None, None,
                )
            t = entries[e_idx]["trials"][t_idx]
            t["passed"] = passed
            t["judge_label"] = label
            t["judge_outcome"] = outcome
            t["judge_error"] = err
            # Single-turn for HR/OR: replace per_turn_judge_meta[0] with
            # the fresh meta dict (now includes raw_text from Step 0).
            ptjm = t.get("per_turn_judge_meta") or []
            if ptjm:
                ptjm[0] = meta or {}
            else:
                ptjm = [meta or {}]
            t["per_turn_judge_meta"] = ptjm
            done += 1
            if done % 200 == 0:
                elapsed = time.time() - start
                rate = done / elapsed
                eta = (len(worklist) - done) / rate
                print(f"    progress: {done}/{len(worklist)}  "
                      f"rate={rate:.1f}/s  eta={eta:.0f}s")

    # Re-aggregate per-prompt stats (pass_count, total_trials, p_hat,
    # n_judge_label_unparsed, etc.)
    for e in entries:
        agg = aggregate_trials(e["trials"])
        for k, v in agg.items():
            e[k] = v

    # Write to v2_judgefix/
    dst_dir.mkdir(parents=True, exist_ok=True)
    with dst_rm.open("w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    if src_meta.exists():
        md = json.loads(src_meta.read_text())
        md["judge_rejudge"] = {
            "rejudged_at": datetime.now().isoformat(),
            "n_rejudged": len(worklist),
            "n_total_trials": n_total_trials,
            "judge_code_version": "Step 0 (label-driven, markdown-tolerant)",
            "elapsed_seconds": time.time() - start,
        }
        dst_meta.write_text(json.dumps(md, indent=2))

    # Quick before/after summary for this cell
    new_unparsed = sum(
        1 for e in entries for t in e["trials"]
        if t.get("judge_outcome") == "judge_label_unparsed"
    )
    new_p_hat_avg = sum(e["p_hat"] or 0 for e in entries) / len(entries)
    return {
        "cell": cell,
        "n_rejudged": len(worklist),
        "n_total": n_total_trials,
        "n_still_unparsed_after": new_unparsed,
        "recovery_rate": 1 - new_unparsed / len(worklist) if worklist else 1.0,
        "new_p_hat_avg": new_p_hat_avg,
        "elapsed_s": time.time() - start,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--cats", nargs="+", default=CATS)
    ap.add_argument("--cells", nargs="*", default=None,
                    help="Optional substring filter")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("DEEPSEEK_API_TOKEN"):
        sys.exit("ERROR: DEEPSEEK_API_TOKEN not set in env")

    print(f"Workers: {args.workers}  dry_run: {args.dry_run}\n")

    summary = []
    for cat in args.cats:
        cells = discover_cells(cat)
        if args.cells:
            cells = [c for c in cells if any(f in c for f in args.cells)]
        print(f"[{cat}] cells discovered: {cells}")
        for cell in cells:
            r = process_cell(cat, cell, args.workers, args.dry_run)
            r["cat"] = cat
            summary.append(r)

    # Print aggregate summary
    print("\n" + "=" * 100)
    print("REJUDGE SUMMARY")
    print("=" * 100)
    for s in summary:
        if "n_rejudged" not in s:
            print(f"  {s['cat']}/{s['cell']}: dry-run, {s.get('n_to_rejudge', '?')} would be rejudged")
            continue
        print(f"\n  {s['cat']}/{s['cell']}")
        print(f"    rejudged: {s['n_rejudged']:>5}/{s['n_total']:>5}  "
              f"({100*s['n_rejudged']/s['n_total']:.1f}% of total)")
        if s.get("n_rejudged"):
            print(f"    still unparsed after: {s['n_still_unparsed_after']:>4}  "
                  f"(recovery rate {100*s['recovery_rate']:.1f}%)")
            print(f"    new mean p_hat: {s['new_p_hat_avg']:.4f}")
            print(f"    elapsed: {s['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
