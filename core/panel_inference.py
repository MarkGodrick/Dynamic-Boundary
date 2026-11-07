"""Shared panel inference helper: run one model on a list of items.

Two layers:
  - `run_jobs_on_model`: FSM-driven runner. Takes a built model + judge,
    pipelines all (item × trial) jobs through two independent thread pools
    (model and judge) so vLLM and the judge API saturate independently.
    Returns aggregated results in memory.
  - `run_model_on_items`: thin wrapper for callers that just want the
    happy-path "load VLLMModel → run → save jsonl → unload" flow. Used by
    `scripts/filter_random_compose.py` and `scripts/patches/*`.

Direct callers of `run_jobs_on_model` (e.g. `scripts/collect_response_matrix.py`)
manage the model lifecycle themselves — useful when the script needs to
interleave inference results with reused data from a separate source (e.g.
search-phase trials for the model's own edge cases).

The model→judge FSM replaces the legacy "one trial = one thread that runs
generate then judge sequentially" pattern. Under that pattern, half of the
pool's threads were always blocked on the judge API while vLLM sat
underutilized; with two pools and event-driven dispatch, both saturate
independently. Multi-turn (sycophancy follow-ups) reuses the same FSM by
re-entering NEEDS_MODEL after each judged turn.
"""

import json
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from core.trial_utils import (
    _accumulate_judge_meta,
    _accumulate_meta_from_dict,
    aggregate_trials,
    build_trial,
    do_model_step,
    judge_and_capture,
    trial_exception_record,
)


# ---------------------------------------------------------------------------
# Trial state machine
# ---------------------------------------------------------------------------

_STAGE_MODEL = "model"
_STAGE_JUDGE = "judge"
_STAGE_DONE = "done"


@dataclass
class _TrialJob:
    """Per-trial state machine. One instance per (item, trial) pair.

    Lifecycle:
      stage=model → submit do_model_step → on success, stage=judge
      stage=judge → submit judge_and_capture → on result:
        - judge None / passed=False → stage=done
        - passed=True with follow-ups remaining → turn += 1, stage=model
        - passed=True with no more follow-ups → stage=done
    """
    # Identity
    i: int                                   # item index (in the run's items list)
    t: int                                   # trial index within item
    seed: int
    item: dict                               # full item dict (for prompt + metadata)

    # FSM state
    stage: str = _STAGE_MODEL
    turn: int = 1
    messages: list[dict] = field(default_factory=list)  # OpenAI-format conversation

    # Accumulators (populated across turns)
    responses: list[str] = field(default_factory=list)
    meta_accum: dict = field(default_factory=lambda: {
        "truncated": None,
        "completion_tokens": None,
        "per_turn_meta": None,
    })
    judge_meta_list: list[dict] | None = None

    # Last-step results (carried into next stage / final trial dict)
    last_response: str = ""
    last_passed: bool | None = None
    last_outcome: str = "ok"
    last_err: dict | None = None
    last_label: str | None = None

    # Final
    final_trial: dict | None = None          # populated when stage=done

    @property
    def follow_ups(self) -> list[str]:
        return (self.item.get("metadata") or {}).get("follow_ups") or []

    @property
    def is_multi_turn(self) -> bool:
        return bool(self.follow_ups)


def _advance_after_model(job: _TrialJob, response: str, meta: dict | None) -> None:
    """Apply a model step's result. Transitions to NEEDS_JUDGE."""
    job.last_response = response
    job.responses.append(response)
    job.messages.append({"role": "assistant", "content": response})
    job.meta_accum = _accumulate_meta_from_dict(job.meta_accum, meta)
    job.stage = _STAGE_JUDGE


def _advance_after_judge(
    job: _TrialJob,
    judge_result: tuple,
) -> None:
    """Apply a judge step's result. Transitions to NEEDS_MODEL (next turn)
    or DONE.
    """
    passed, outcome, err, label, judge_meta = judge_result
    job.last_passed = passed
    job.last_outcome = outcome
    job.last_err = err
    job.last_label = label
    job.judge_meta_list = _accumulate_judge_meta(job.judge_meta_list, judge_meta)

    # passed=None → judge couldn't render verdict → out of denominator.
    # passed=False → trial fails at this turn.
    # passed=True + more follow-ups → next turn.
    # passed=True + no follow-ups left → trial passed all turns.
    if passed is None or not passed:
        _finalize_trial(job, failed_at_turn=job.turn)
        return
    next_idx = job.turn - 1  # follow_ups[0] feeds turn 2
    if next_idx >= len(job.follow_ups):
        _finalize_trial(job, failed_at_turn=0)
        return
    job.messages.append({"role": "user", "content": job.follow_ups[next_idx]})
    job.turn += 1
    job.stage = _STAGE_MODEL


def _finalize_trial(job: _TrialJob, *, failed_at_turn: int) -> None:
    """Build the canonical trial record dict and stash on the job."""
    job.stage = _STAGE_DONE
    job.final_trial = build_trial(
        seed=job.seed,
        response=job.last_response,
        responses=job.responses,
        is_multi_turn=job.is_multi_turn,
        passed=job.last_passed,
        failed_at_turn=failed_at_turn,
        judge_outcome=job.last_outcome,
        judge_error=job.last_err,
        judge_label=job.last_label,
        truncated=job.meta_accum["truncated"],
        completion_tokens=job.meta_accum["completion_tokens"],
        per_turn_meta=job.meta_accum["per_turn_meta"],
        per_turn_judge_meta=job.judge_meta_list,
    )


def _finalize_exception(job: _TrialJob, exc: Exception) -> None:
    """Trial-level exception path (model.generate or judge raised after
    retries). Mirrors the legacy `trial_exception_record` schema:
    passed=None, failed_at_turn=-1, judge_outcome="trial_exception".
    """
    job.stage = _STAGE_DONE
    job.final_trial = trial_exception_record(job.seed, exc)


# ---------------------------------------------------------------------------
# Pool dispatch
# ---------------------------------------------------------------------------

def _submit_model(
    job: _TrialJob, model, model_pool: ThreadPoolExecutor, events_q: queue.Queue,
) -> None:
    """Dispatch a model step. Turn 1 calls generate(prompt); turn 2+ calls
    generate_conversation(messages). messages must already be primed
    (turn 1: just user prompt; turn 2+: includes the new follow-up user msg).
    """
    is_conv = job.turn > 1
    arg = job.messages if is_conv else job.item["prompt"]
    fut = model_pool.submit(
        do_model_step, model, arg, job.seed, is_conversation=is_conv,
    )
    fut.add_done_callback(lambda f, _job=job: events_q.put((_job, f)))


def _submit_judge(
    job: _TrialJob, judge, judge_pool: ThreadPoolExecutor, events_q: queue.Queue,
) -> None:
    """Dispatch a judge step over the current job.messages."""
    metadata = job.item.get("metadata")
    fut = judge_pool.submit(
        judge_and_capture, judge, list(job.messages), metadata,
    )
    fut.add_done_callback(lambda f, _job=job: events_q.put((_job, f)))


# ---------------------------------------------------------------------------
# FSM-driven runner (low layer)
# ---------------------------------------------------------------------------

def run_jobs_on_model(
    items: list[dict],
    model,
    judge,
    n_trials: int,
    seed: int,
    max_workers: int,
    *,
    model_name: str,
    model_pool_size: int | None = None,
    judge_pool_size: int | None = None,
    prior_completed: dict[str, list[dict]] | None = None,
    save_callback: Callable[[list[dict]], None] | None = None,
    save_interval: float = 30.0,
    stop_event: threading.Event | None = None,
) -> list[dict]:
    """Run all (item × trial) jobs against a single live model + judge.

    Pipelines model and judge through two independent thread pools; each
    trial is a state machine that re-enters the model pool for each
    multi-turn follow-up (when item.metadata.follow_ups is set).

    Args:
        items: list of {"prompt", "prompt_id", optional "metadata"}.
        model: TargetModel (e.g. VLLMModel) — caller manages start/shutdown.
        judge: Judge implementing judge(messages, metadata).
        n_trials: trials per item.
        seed: master seed; per-trial seeds are pre-generated in item-major
            order so completion order can never shift seed assignment
            (matches legacy collect_responses semantics).
        max_workers: legacy single-knob pool size. Used as fallback for both
            pools when model_pool_size / judge_pool_size are None.
        model_name: stamped on each result row's `"model"` field. Pass the
            *evaluating* model name (not source_model — set that via the
            item dict if needed downstream).
        model_pool_size: model thread pool size. Defaults to max_workers.
        judge_pool_size: judge thread pool size. Defaults to max_workers.
        prior_completed: optional {prompt_id: [trial,...]} preloaded from a
            previous partial run. Trials whose `seed` matches the regenerated
            seed for that (item, trial_idx) are reused as-is; the remaining
            (item, trial_idx) pairs are run fresh. Resume requires the dataset
            content + order, master seed, and n_trials to be unchanged.
        save_callback: if given, called periodically with a partial-results
            list (same schema as the function's final return). Caller is
            responsible for atomic write semantics.
        save_interval: seconds between save_callback invocations. Ignored if
            save_callback is None.
        stop_event: optional Event. When set, the FSM enters drain mode:
            stops dispatching multi-turn next-turn model steps (records those
            jobs as trial_exception so resume retries them), still runs judge
            for in-flight model results, and exits the loop once all
            already-dispatched work has completed.

    Returns:
        results in original item order, one dict per item with aggregated
        stats (matches the legacy collect_responses output schema).
    """
    if model_pool_size is None:
        model_pool_size = max_workers
    if judge_pool_size is None:
        judge_pool_size = max(1, max_workers)

    # Pre-generate seeds in item-major order — completion order in the FSM
    # cannot shift seed assignment.
    rng = random.Random(seed)
    trial_seeds: list[list[int]] = []
    for _ in items:
        trial_seeds.append([rng.randint(0, 2**31 - 1) for _ in range(n_trials)])

    trials_by_item: list[list[dict | None]] = [
        [None] * n_trials for _ in items
    ]

    # Resume prefill: trials whose (prompt_id, seed) appear in prior_completed
    # are loaded into trials_by_item directly and skipped during job creation.
    n_skipped_resume = 0
    if prior_completed:
        prior_by_pid_seed: dict[str, dict[int, dict]] = {
            pid: {t["seed"]: t for t in trials}
            for pid, trials in prior_completed.items()
        }
        for i, item in enumerate(items):
            seed_map = prior_by_pid_seed.get(item["prompt_id"])
            if not seed_map:
                continue
            for t in range(n_trials):
                trial_obj = seed_map.get(trial_seeds[i][t])
                if trial_obj is not None:
                    trials_by_item[i][t] = trial_obj
                    n_skipped_resume += 1

    jobs: list[_TrialJob] = []
    for i, item in enumerate(items):
        for t in range(n_trials):
            if trials_by_item[i][t] is not None:
                continue  # already loaded from prior_completed
            jobs.append(_TrialJob(
                i=i, t=t, seed=trial_seeds[i][t], item=item,
                messages=[{"role": "user", "content": item["prompt"]}],
            ))
    n_jobs = len(jobs)

    def _build_partial_results() -> list[dict]:
        """Snapshot trials_by_item → result rows. Filters None (in-flight)."""
        out: list[dict] = []
        for i, item in enumerate(items):
            trials = [t for t in trials_by_item[i] if t is not None]
            stats = aggregate_trials(trials)
            out.append({
                "prompt_id": item["prompt_id"],
                "model": model_name,
                "trials": trials,
                **stats,
            })
        return out

    if n_skipped_resume:
        print(
            f"  [resume] {n_skipped_resume}/{len(items) * n_trials} trials "
            f"loaded from prior_completed; {n_jobs} new trials to run",
            flush=True,
        )

    if n_jobs == 0:
        # All trials already done from resume — nothing to dispatch.
        return _build_partial_results()

    events_q: queue.Queue = queue.Queue()
    n_done = 0
    progress_step = max(500, n_jobs // 20)

    print(
        f"  pool sizes: model={model_pool_size}, judge={judge_pool_size}",
        flush=True,
    )

    with ThreadPoolExecutor(
        max_workers=model_pool_size, thread_name_prefix="model",
    ) as model_pool, ThreadPoolExecutor(
        max_workers=judge_pool_size, thread_name_prefix="judge",
    ) as judge_pool:

        # Initial dispatch: every job starts in NEEDS_MODEL.
        for jb in jobs:
            _submit_model(jb, model, model_pool, events_q)

        last_save_t = time.time()
        stopping = False

        # FSM loop — single-threaded orchestration, no locks needed.
        # Worker callbacks only put (job, fut) onto events_q.
        while n_done < n_jobs:
            # Promote stop_event → drain mode (idempotent).
            if not stopping and stop_event is not None and stop_event.is_set():
                stopping = True
                print(
                    "  [stop] graceful stop requested, draining in-flight…",
                    flush=True,
                )

            try:
                job, fut = events_q.get(timeout=0.5)
            except queue.Empty:
                # Periodic save tick.
                if save_callback and time.time() - last_save_t >= save_interval:
                    save_callback(_build_partial_results())
                    last_save_t = time.time()
                # If stopping and no jobs are still in flight, we're done.
                if stopping and all(j.stage == _STAGE_DONE for j in jobs):
                    break
                continue

            try:
                result = fut.result()
            except Exception as e:
                _finalize_exception(job, e)
            else:
                if job.stage == _STAGE_MODEL:
                    response, meta = result
                    _advance_after_model(job, response, meta)
                else:  # _STAGE_JUDGE
                    _advance_after_judge(job, result)

            if job.stage == _STAGE_MODEL:
                if stopping:
                    # Multi-turn next-turn dispatch — abandon as trial_exception
                    # so resume retries from turn 1 cleanly. The first turn's
                    # model output is sunk cost; we accept that.
                    _finalize_exception(
                        job, RuntimeError("interrupted between turns"),
                    )
                    trials_by_item[job.i][job.t] = job.final_trial
                    n_done += 1
                else:
                    _submit_model(job, model, model_pool, events_q)
            elif job.stage == _STAGE_JUDGE:
                # Always dispatch judge — model step is already paid for.
                _submit_judge(job, judge, judge_pool, events_q)
            else:
                trials_by_item[job.i][job.t] = job.final_trial
                n_done += 1
                if n_done % progress_step == 0 or n_done == n_jobs:
                    print(f"    [{n_done}/{n_jobs}] trials completed", flush=True)

            # Periodic save tick after an event too (in case events keep
            # arriving and queue.Empty path never fires).
            if save_callback and time.time() - last_save_t >= save_interval:
                save_callback(_build_partial_results())
                last_save_t = time.time()

    results = []
    for i, item in enumerate(items):
        trials = trials_by_item[i]
        stats = aggregate_trials(trials)
        results.append({
            "prompt_id": item["prompt_id"],
            "model": model_name,
            "trials": trials,
            **stats,
        })
    return results


# ---------------------------------------------------------------------------
# Convenience wrapper: build vLLM, run, save, shutdown (high layer)
# ---------------------------------------------------------------------------

def run_model_on_items(
    items: list[dict],
    model_cfg: dict,
    judge,
    n_trials: int,
    seed: int,
    port: int,
    max_workers: int,
    server_timeout: int,
    out_path: Path,
    max_tokens: int = 1024,
    max_model_len: int = 8192,
    *,
    model_pool_size: int | None = None,
    judge_pool_size: int | None = None,
) -> tuple[bool, str]:
    """Build a VLLMModel, run all (item × trial) jobs through the FSM,
    save jsonl, shut down.

    Args:
        max_tokens: per-call generation cap. Callers running multiple panel
            models on a single category should pass the category-appropriate
            cap (see core/inference_budgets.py:get_max_tokens).
        max_model_len: vLLM KV-cache window. Must be ≥ prompt + max_tokens
            (or prompt + 3 × max_tokens for multi-turn SR). See
            core/inference_budgets.py:get_max_model_len.
        model_pool_size / judge_pool_size: passed through to
            `run_jobs_on_model`. Default to max_workers and max_workers.

    Returns (ok, status_message). Server-level construction failure returns
    (False, "FAILED (server): ..."); per-trial errors are caught inside the
    FSM and recorded as trial_exception trials (function still returns OK).
    """
    from models.vllm_model import VLLMModel

    name = model_cfg["name"]
    print(f"\n{'='*60}")
    print(f"  {datetime.now().strftime('%H:%M:%S')} | Starting: {name}")
    print(f"{'='*60}")
    n_jobs = len(items) * n_trials
    print(f"  n_items = {len(items)}, n_trials = {n_trials}, jobs = {n_jobs}, "
          f"max_workers = {max_workers}, seed = {seed}, max_tokens = {max_tokens}")

    try:
        vllm = VLLMModel(
            model_name=name,
            conda_env=model_cfg["conda_env"],
            cuda_devices=model_cfg["cuda_devices"],
            port=port,
            dtype=model_cfg.get("dtype", "bfloat16"),
            max_model_len=max_model_len,
            gpu_memory_utilization=model_cfg["gpu_mem_util"],
            extra_args=model_cfg.get("extra_args", []),
            server_timeout=server_timeout,
            max_tokens=max_tokens,
        )
    except RuntimeError as e:
        print(f"  FAILED: {e}")
        return False, f"FAILED (server): {e}"

    try:
        results = run_jobs_on_model(
            items=items,
            model=vllm,
            judge=judge,
            n_trials=n_trials,
            seed=seed,
            max_workers=max_workers,
            model_name=name,
            model_pool_size=model_pool_size,
            judge_pool_size=judge_pool_size,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        p_hats = [r["p_hat"] for r in results if r["p_hat"] is not None]
        n_bad = sum(1 for r in results if r["p_hat"] is None)
        avg = sum(p_hats) / len(p_hats) if p_hats else float("nan")
        print(f"  Done: avg p_hat = {avg:.3f} over {len(p_hats)} valid items"
              f"{f' ({n_bad} items had every trial fail — check for env/judge issues)' if n_bad else ''}"
              f"  →  {out_path}")
        return True, "OK"
    finally:
        vllm.shutdown()
        time.sleep(3)
