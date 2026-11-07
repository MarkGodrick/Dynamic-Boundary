"""Rerun failed trials in a response_matrix.jsonl cell.

Designed for the case where some trials failed due to:
  - `judge_outcome == "api_exhausted"`: judge API call exhausted retries
    (typical cause: DeepSeek 402 Insufficient Balance). Model responses
    were generated and saved; we only need to re-call the judge.
  - `judge_outcome == "trial_exception"` with `failed_at_turn == -1`:
    target model failed before any response was produced. We must
    re-run the full trial (model + judge) from scratch.
  - `judge_outcome == "judge_label_unparsed"` (optional, --include-judge-unparsed):
    judge replied but the verdict line couldn't be parsed. Re-call the
    judge with whatever fix is now in place.

For each trial we re-execute the same FSM as `scripts/online_response_matrix.py`:
model turn N → judge turn N → if passed and more follow_ups → model turn N+1, etc.
When saved responses exist (api_exhausted path), we reuse them for the first
len(responses) turns and only call the target model if the re-judged trial
needs to continue beyond what was saved.

Writes the updated matrix to <cell-dir>/response_matrix.jsonl, backing up the
original to <cell-dir>/.bak_pre_rerun_<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.trial_utils import (  # noqa: E402
    _accumulate_judge_meta,
    _accumulate_meta_from_dict,
    aggregate_trials,
    build_trial,
    do_model_step,
    judge_and_capture,
    trial_exception_record,
)
from judges.loader import create_judge  # noqa: E402
from models.loader import create_target_model  # noqa: E402

RERUN_OUTCOMES = {"api_exhausted", "trial_exception"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell-dir", required=True, type=Path,
                   help="Directory containing response_matrix.jsonl + metadata.json")
    p.add_argument("--dataset", type=Path, default=None,
                   help="Anchor bundle path (default: read from metadata.json)")
    p.add_argument("--falcon-base-url", default="http://localhost:8300/v1",
                   help="Target-model OpenAI-compatible base URL")
    p.add_argument("--falcon-api-key-env", default="VLLM_API_KEY")
    p.add_argument("--falcon-max-new-tokens", type=int, default=2048,
                   help="Per-turn max_tokens for target model. Match the value used "
                        "in the original run (SR adaptive.yaml: 2048).")
    p.add_argument("--falcon-temperature", type=float, default=1.0)
    p.add_argument("--judge-model", default="deepseek-chat")
    p.add_argument("--judge-api-key-env", default="DEEPSEEK_API_TOKEN")
    p.add_argument("--judge-base-url", default="https://api.deepseek.com")
    p.add_argument("--judge-max-retries", type=int, default=5)
    p.add_argument("--workers", type=int, default=32,
                   help="ThreadPoolExecutor size (parallel trials in flight)")
    p.add_argument("--include-judge-unparsed", action="store_true",
                   help="Also rerun trials with judge_outcome == 'judge_label_unparsed'")
    p.add_argument("--max-trials", type=int, default=None,
                   help="Process at most N trials (for smoke testing; default: all). "
                        "Mixes outcomes so both rejudge and full-rerun paths get exercised.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Write updated matrix here (default: in-place in --cell-dir, "
                        "with backup at .bak_pre_rerun_<ts>/)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print worklist and exit without re-running anything")
    return p.parse_args()


def load_cell(cell_dir: Path) -> tuple[list[dict], dict]:
    rm_path = cell_dir / "response_matrix.jsonl"
    meta_path = cell_dir / "metadata.json"
    if not rm_path.exists():
        sys.exit(f"ERROR: {rm_path} not found")
    if not meta_path.exists():
        sys.exit(f"ERROR: {meta_path} not found")
    rows = [json.loads(l) for l in open(rm_path)]
    meta = json.loads(meta_path.read_text())
    return rows, meta


def load_dataset(dataset_path: Path) -> dict[str, dict]:
    out = {}
    for line in open(dataset_path):
        item = json.loads(line)
        out[item["prompt_id"]] = item
    return out


def need_rerun(trial: dict, include_judge_unparsed: bool) -> bool:
    oc = trial.get("judge_outcome")
    if oc in RERUN_OUTCOMES:
        return True
    if include_judge_unparsed and oc == "judge_label_unparsed":
        return True
    return False


def replay_trial(item: dict, original_trial: dict, falcon_model, judge) -> dict:
    """Replay a trial through the model+judge FSM, reusing saved responses where possible.

    For api_exhausted / judge_label_unparsed: reuse saved `responses` for turns
    1..len(responses); call falcon_model only for any additional turns.
    For trial_exception (failed_at_turn == -1): re-run from scratch (no saved
    responses to reuse).
    """
    pid = item["prompt_id"]
    prompt = item["prompt"]
    metadata = item.get("metadata") or {}
    follow_ups = metadata.get("follow_ups") or []
    max_turns = 1 + len(follow_ups)
    seed = int(original_trial["seed"])
    is_multi_turn = max_turns > 1

    orig_outcome = original_trial.get("judge_outcome")
    if orig_outcome == "trial_exception":
        saved_responses: list[str] = []
        saved_meta: list[dict] = []
    else:
        saved_responses = list(original_trial.get("responses") or [])
        saved_meta = list(original_trial.get("per_turn_meta") or [])

    messages = [{"role": "user", "content": prompt}]
    responses: list[str] = []
    meta_accum = {"truncated": None, "completion_tokens": None, "per_turn_meta": None}
    judge_meta_list: list[dict] | None = None
    last_response = ""

    for turn in range(1, max_turns + 1):
        # ---- model step ----
        if turn <= len(saved_responses):
            response = saved_responses[turn - 1]
            meta = saved_meta[turn - 1] if turn - 1 < len(saved_meta) else {}
        else:
            try:
                arg = messages if turn > 1 else prompt
                response, meta = do_model_step(
                    falcon_model, arg, seed, is_conversation=(turn > 1),
                )
            except Exception as e:
                return trial_exception_record(seed, e)

        responses.append(response)
        last_response = response
        messages.append({"role": "assistant", "content": response})
        meta_accum = _accumulate_meta_from_dict(meta_accum, meta)

        # ---- judge step ----
        try:
            passed, outcome, err, label, judge_meta = judge_and_capture(
                judge, list(messages), metadata,
            )
        except Exception as e:
            return trial_exception_record(seed, e)
        judge_meta_list = _accumulate_judge_meta(judge_meta_list, judge_meta)

        # If the judge itself errored out (api_exhausted again, safety classified,
        # etc.) record using build_trial so all the diagnostic fields stay aligned.
        if outcome != "ok":
            return build_trial(
                seed=seed,
                response=last_response,
                responses=responses,
                is_multi_turn=is_multi_turn,
                passed=passed,
                failed_at_turn=turn,
                judge_outcome=outcome,
                judge_error=err,
                judge_label=label,
                truncated=meta_accum.get("truncated"),
                completion_tokens=meta_accum.get("completion_tokens"),
                per_turn_meta=meta_accum.get("per_turn_meta"),
                per_turn_judge_meta=judge_meta_list,
            )

        # Verdict parsed (or unparsed → passed=None): same shape, build_trial
        # records judge_outcome='ok' but passed=None means out-of-denominator.
        if passed is None:
            return build_trial(
                seed=seed, response=last_response, responses=responses,
                is_multi_turn=is_multi_turn, passed=None, failed_at_turn=turn,
                judge_outcome="judge_label_unparsed", judge_error=None,
                judge_label=label,
                truncated=meta_accum.get("truncated"),
                completion_tokens=meta_accum.get("completion_tokens"),
                per_turn_meta=meta_accum.get("per_turn_meta"),
                per_turn_judge_meta=judge_meta_list,
            )
        if not passed:
            return build_trial(
                seed=seed, response=last_response, responses=responses,
                is_multi_turn=is_multi_turn, passed=False, failed_at_turn=turn,
                judge_outcome="ok", judge_error=None, judge_label=label,
                truncated=meta_accum.get("truncated"),
                completion_tokens=meta_accum.get("completion_tokens"),
                per_turn_meta=meta_accum.get("per_turn_meta"),
                per_turn_judge_meta=judge_meta_list,
            )
        # Passed this turn — advance if more follow_ups, else finalize as pass-all.
        if turn < max_turns:
            messages.append({"role": "user", "content": follow_ups[turn - 1]})

    return build_trial(
        seed=seed, response=last_response, responses=responses,
        is_multi_turn=is_multi_turn, passed=True, failed_at_turn=0,
        judge_outcome="ok", judge_error=None, judge_label=None,  # last label is in judge_meta_list
        truncated=meta_accum.get("truncated"),
        completion_tokens=meta_accum.get("completion_tokens"),
        per_turn_meta=meta_accum.get("per_turn_meta"),
        per_turn_judge_meta=judge_meta_list,
    )


def main() -> None:
    args = parse_args()

    rows, meta = load_cell(args.cell_dir)
    category = meta["category"]
    model_name = meta["model"]

    dataset_path = args.dataset or (ROOT / meta["dataset_path"])
    ds = load_dataset(dataset_path)

    # Build worklist
    worklist: list[tuple[int, int]] = []  # (row_idx, trial_idx)
    counts = {"api_exhausted": 0, "trial_exception": 0, "judge_label_unparsed": 0}
    needs_model_call = 0  # trials that may require Falcon (full rerun or rejudge tail)
    for ri, r in enumerate(rows):
        item = ds.get(r["prompt_id"])
        if item is None:
            continue
        n_follow = len(((item.get("metadata") or {}).get("follow_ups")) or [])
        max_turns = 1 + n_follow
        for ti, t in enumerate(r.get("trials") or []):
            if not need_rerun(t, args.include_judge_unparsed):
                continue
            oc = t.get("judge_outcome")
            counts[oc] = counts.get(oc, 0) + 1
            worklist.append((ri, ti))
            if oc == "trial_exception":
                needs_model_call += 1
            else:
                n_resp = len(t.get("responses") or [])
                if n_resp < max_turns:
                    needs_model_call += 1

    if args.max_trials is not None and len(worklist) > args.max_trials:
        # Stratified sample so both outcome paths get covered.
        by_oc: dict[str, list[tuple[int, int]]] = {}
        for k in worklist:
            oc = rows[k[0]]["trials"][k[1]]["judge_outcome"]
            by_oc.setdefault(oc, []).append(k)
        sampled: list[tuple[int, int]] = []
        # take at least 1 from each outcome, then fill proportionally
        remaining = args.max_trials
        for oc, lst in by_oc.items():
            sampled.append(lst[0]); remaining -= 1
        for oc, lst in by_oc.items():
            for k in lst[1:]:
                if remaining <= 0: break
                sampled.append(k); remaining -= 1
        worklist = sampled
        # Recount and reset needs_model_call after subsampling
        counts = {oc: 0 for oc in counts}
        needs_model_call = 0
        for ri, ti in worklist:
            t = rows[ri]["trials"][ti]
            counts[t["judge_outcome"]] = counts.get(t["judge_outcome"], 0) + 1
            item = ds[rows[ri]["prompt_id"]]
            max_turns = 1 + len(((item.get("metadata") or {}).get("follow_ups")) or [])
            n_resp = len(t.get("responses") or [])
            if t["judge_outcome"] == "trial_exception" or n_resp < max_turns:
                needs_model_call += 1

    print(f"Cell: {args.cell_dir}")
    print(f"  category={category}  model={model_name}  rows={len(rows)}")
    print(f"  worklist: {len(worklist)} trials")
    for k, v in counts.items():
        print(f"    {k}: {v}")
    print(f"  trials that may require target-model calls: {needs_model_call}")

    if args.dry_run:
        print("\n(dry-run, exiting before any rerun)")
        return
    if not worklist:
        print("\nNothing to do.")
        return

    # Build judge + (optionally) falcon model
    judge_cfg = {
        "type": "llm_api",
        "model": args.judge_model,
        "api_key_env": args.judge_api_key_env,
        "base_url": args.judge_base_url,
        "max_retries": args.judge_max_retries,
        "temperature": 0.0,
    }
    judge = create_judge(judge_cfg, category=category)

    falcon_model = None
    if needs_model_call:
        falcon_cfg = {
            "type": "api",
            "name": model_name,
            "api_key_env": args.falcon_api_key_env,
            "base_url": args.falcon_base_url,
            "max_retries": 5,
            "temperature": args.falcon_temperature,
            "max_new_tokens": args.falcon_max_new_tokens,
        }
        falcon_model = create_target_model(falcon_cfg)

    # Backup
    out_dir = args.output_dir or args.cell_dir
    if out_dir == args.cell_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak_dir = args.cell_dir.parent / f".bak_pre_rerun_{ts}_{args.cell_dir.name}"
        bak_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(args.cell_dir / "response_matrix.jsonl", bak_dir / "response_matrix.jsonl")
        shutil.copy2(args.cell_dir / "metadata.json", bak_dir / "metadata.json")
        print(f"\nBackup: {bak_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Execute worklist in parallel
    print(f"\nReplaying {len(worklist)} trials with workers={args.workers}...")
    t0 = time.time()
    new_trials: dict[tuple[int, int], dict] = {}
    n_done = 0
    n_progress_every = max(1, len(worklist) // 20)

    def task(key: tuple[int, int]) -> tuple[tuple[int, int], dict | Exception]:
        ri, ti = key
        item = ds[rows[ri]["prompt_id"]]
        orig = rows[ri]["trials"][ti]
        try:
            new = replay_trial(item, orig, falcon_model, judge)
            return key, new
        except Exception as e:
            return key, e

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(task, k) for k in worklist]
        for fut in as_completed(futures):
            key, result = fut.result()
            if isinstance(result, Exception):
                # Should not happen — replay_trial catches model/judge errors and
                # converts them into trial_exception records. Anything reaching
                # here is a programming bug; surface it loudly.
                print(f"  [bug] replay_trial raised for {key}: {result!r}")
                new_trials[key] = trial_exception_record(
                    rows[key[0]]["trials"][key[1]]["seed"], result,
                )
            else:
                new_trials[key] = result
            n_done += 1
            if n_done % n_progress_every == 0 or n_done == len(worklist):
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(worklist) - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{len(worklist)}] elapsed={elapsed:.0f}s  rate={rate:.2f}/s  eta={eta:.0f}s")

    # Splice new trials back into rows, re-aggregate touched rows.
    touched_rows = sorted({ri for ri, _ in new_trials})
    for (ri, ti), new in new_trials.items():
        rows[ri]["trials"][ti] = new
    for ri in touched_rows:
        agg = aggregate_trials(rows[ri]["trials"])
        rows[ri].update(agg)

    # Summary by new outcome
    out_summary = {}
    for new in new_trials.values():
        out_summary[new.get("judge_outcome")] = out_summary.get(new.get("judge_outcome"), 0) + 1
    print("\nReplay outcome counts:")
    for k, v in sorted(out_summary.items()):
        print(f"  {k}: {v}")

    # Write
    rm_out = out_dir / "response_matrix.jsonl"
    tmp = rm_out.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(rm_out)
    print(f"\nWrote: {rm_out}")

    # Update metadata.json (last_updated only; preserve everything else)
    meta_out = out_dir / "metadata.json"
    meta["last_updated"] = datetime.now().isoformat()
    meta["last_rerun"] = {
        "timestamp": meta["last_updated"],
        "n_trials_rerun": len(worklist),
        "outcomes_in": counts,
        "outcomes_out": out_summary,
    }
    meta_out.write_text(json.dumps(meta, indent=2))
    print(f"Wrote: {meta_out}")


if __name__ == "__main__":
    main()
