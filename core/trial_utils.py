"""Shared helpers for running individual trials through a judge.

Used by scripts that evaluate (prompt × model × trial) pairs:
  - scripts/collect_response_matrix.py
  - scripts/filter_random_compose.py

The functions here unify three concerns:
  1. Calling `judge.judge(messages, metadata)` and capturing any error context
     the judge exposes via `get_last_error()` (thread-local, see LLMJudge).
  2. Building a trial record dict with a uniform schema across scripts.
  3. Building a trial record for trial-level exceptions (e.g. model inference
     failures) that never reach the judge.

Trial record schema (produced by `build_trial`):
    {
        "seed": int,
        "response": str,                 # the last assistant turn's response
        "responses": list[str],          # multi-turn only; omitted single-turn
        "passed": bool | None,           # None ⇒ out of denominator
        "failed_at_turn": int,           # 0 = passed all, N = failed at turn N,
                                         #  -1 = trial-level exception
        "judge_outcome": str,            # "ok" | "safety_classified"
                                         # | "api_exhausted" | "deterministic_aborted"
                                         # | "trial_exception"
        "judge_error": dict | None,      # raw error dict from judge; None on "ok"
        "judge_label": str | None,       # raw verdict token from the judge, e.g.
                                         # 'safe'/'unsafe', 'over_refusal'/
                                         # 'appropriate'/'unsafe', 'pass'/'fail',
                                         # or 'safety_blocked' on filter trigger.
                                         # Preserves failure-mode info lost by
                                         # collapsing to bool. None for judges
                                         # that don't expose get_last_label.
        "truncated": bool | None,        # True iff any generation turn hit
                                         # max_tokens (finish_reason='length').
                                         # Read from model.get_last_meta()
                                         # after each generate. None when the
                                         # model does not expose get_last_meta
                                         # (e.g. HFModel, PlaceholderModel) or
                                         # on trial exceptions.
        "completion_tokens": int | None, # Sum of completion_tokens across all
                                         # turns. None if unknown.
        "per_turn_meta": list[dict] | None,
                                         # Per-turn detail: finish_reason,
                                         # truncated, completion_tokens,
                                         # prompt_tokens, reasoning_content.
                                         # Length matches `responses`. Useful
                                         # for failure-mode analysis on
                                         # multi-turn sycophancy (which turn
                                         # truncated?). None when the model
                                         # does not expose get_last_meta.
        "per_turn_judge_meta": list[dict] | None,
                                         # Per-turn judge API call meta:
                                         # finish_reason, truncated,
                                         # completion_tokens, prompt_tokens.
                                         # Length matches the number of judge
                                         # calls made (= 1 for single-turn,
                                         # up to N for multi-turn that
                                         # progressed through follow-ups).
                                         # None for rule-based judges
                                         # (ConstraintJudge) or when the
                                         # judge does not expose
                                         # get_last_meta. Matters for SR
                                         # multi-turn: a middle-turn judge
                                         # truncation (verdict parse
                                         # fallthrough) can silently flip a
                                         # fail to a pass. Only the last
                                         # entry was stored previously under
                                         # `judge_meta`; per-turn list closes
                                         # that blind spot.
    }
"""

from datetime import datetime


def judge_and_capture(
    judge, messages: list[dict], metadata: dict | None,
) -> tuple[bool | None, str, dict | None, str | None, dict | None]:
    """Call judge.judge() and read the thread-local error/label/meta context.

    Returns (passed, judge_outcome, judge_error, judge_label, judge_meta):
      - passed: True / False / None (None means the verdict could not be
        produced and the trial should be treated as out of denominator).
      - judge_outcome: short string "ok" | "safety_classified" |
        "api_exhausted" | "deterministic_aborted".
      - judge_error: raw last-error dict (None on ok).
      - judge_label: raw verdict token for post-hoc analysis (None if the
        judge doesn't expose `get_last_label`).
      - judge_meta: last API call's meta dict (finish_reason, truncated,
        completion_tokens, prompt_tokens). None if the judge doesn't expose
        `get_last_meta` (e.g. ConstraintJudge) or meta was empty.

    Rule-based judges (e.g. ConstraintJudge) do not expose
    `get_last_error`/`get_last_label`/`get_last_meta`; in that case outcome
    is always "ok", label is None, meta is None.
    """
    # NOTE: passed positionally, not as kwarg, to match the legacy judge
    # invocation style used throughout the project (some test doubles
    # accept metadata only as a positional argument).
    passed = judge.judge(messages, metadata)
    get_err = getattr(judge, "get_last_error", None)
    err = get_err() if callable(get_err) else None
    get_label = getattr(judge, "get_last_label", None)
    label = get_label() if callable(get_label) else None
    get_meta = getattr(judge, "get_last_meta", None)
    meta = get_meta() if callable(get_meta) else None
    # Normalize empty dict → None for downstream consumers who check
    # "if judge_meta:".
    if not meta:
        meta = None
    if err is None:
        # passed=None on the success path means the judge replied but the
        # verdict line couldn't be parsed (LLMJudge `_extract_raw_label`
        # returned None). Surface as a distinct outcome so downstream
        # audits can count it separately from API failures.
        if passed is None:
            return passed, "judge_label_unparsed", None, label, meta
        return passed, "ok", None, label, meta
    classification = err.get("classification")
    if classification == "safety_blocked":
        outcome = "safety_classified"
    elif classification == "deterministic":
        outcome = "deterministic_aborted"
    else:
        outcome = "api_exhausted"
    return passed, outcome, err, label, meta


def build_trial(
    seed: int,
    response: str,
    responses: list[str],
    is_multi_turn: bool,
    *,
    passed: bool | None,
    failed_at_turn: int,
    judge_outcome: str,
    judge_error: dict | None,
    judge_label: str | None = None,
    truncated: bool | None = None,
    completion_tokens: int | None = None,
    per_turn_meta: list[dict] | None = None,
    per_turn_judge_meta: list[dict] | None = None,
) -> dict:
    """Assemble a single trial record with the schema documented at module
    level. `responses` is stored only for multi-turn conversations.
    """
    trial = {
        "seed": seed,
        "response": response,
        "passed": passed,
        "failed_at_turn": failed_at_turn,
        "judge_outcome": judge_outcome,
        "judge_error": judge_error,
        "judge_label": judge_label,
        "truncated": truncated,
        "completion_tokens": completion_tokens,
        "per_turn_meta": per_turn_meta,
        "per_turn_judge_meta": per_turn_judge_meta,
    }
    if is_multi_turn:
        trial["responses"] = responses
    return trial


def _accumulate_meta_from_dict(prev: dict, meta: dict | None) -> dict:
    """Pure-dict version of `_accumulate_meta`: merge a per-turn meta dict
    into the trial-level accumulator `prev` without touching the model.

    Used by FSM-driven runners (panel_inference.run_jobs_on_model) where the
    model worker captures meta in its own thread and passes it explicitly to
    the orchestrator (which is a different thread, so it can't read the
    model's thread-local stash).

    Sentinel `meta is None` means "model doesn't expose meta" (treated like
    `_accumulate_meta`'s "get_last_meta not callable" branch — return prev
    unchanged). An *empty* dict still gets appended to per_turn_meta so the
    list length stays aligned with the response count.
    """
    if meta is None:
        return prev
    out = dict(prev)
    # per_turn_meta: append a copy of this turn's full meta to the list
    turn_copy = dict(meta)
    prev_list = prev.get("per_turn_meta")
    out["per_turn_meta"] = [turn_copy] if prev_list is None else prev_list + [turn_copy]
    # truncated: sticky OR across turns; None only if every turn was None
    new_trunc = meta.get("truncated")
    if prev.get("truncated") is None:
        out["truncated"] = new_trunc
    elif new_trunc is not None:
        out["truncated"] = bool(prev["truncated"]) or bool(new_trunc)
    # completion_tokens: sum when both known; keep known if only one side has it
    new_ct = meta.get("completion_tokens")
    prev_ct = prev.get("completion_tokens")
    if new_ct is not None and prev_ct is not None:
        out["completion_tokens"] = prev_ct + new_ct
    elif new_ct is not None:
        out["completion_tokens"] = new_ct
    return out


def _accumulate_meta(model, prev: dict) -> dict:
    """Merge per-turn meta from `model.get_last_meta()` into the trial-level
    accumulator `prev`.

    Trial-level aggregates maintained in `prev`:
      - truncated: True iff *any* turn had finish_reason=='length'
      - completion_tokens: sum across turns (None if every turn was None)
      - per_turn_meta: list of per-turn meta dicts (full detail, append-only)

    Returns a new dict; does not mutate `prev`.

    If the model doesn't expose `get_last_meta`, returns `prev` unchanged
    (preserves backward compatibility with HFModel, PlaceholderModel).
    """
    get_meta = getattr(model, "get_last_meta", None)
    if not callable(get_meta):
        return prev
    meta = get_meta() or {}
    return _accumulate_meta_from_dict(prev, meta)


def do_model_step(
    model,
    prompt_or_messages,
    seed: int,
    *,
    is_conversation: bool,
) -> tuple[str, dict | None]:
    """Worker callable for a model thread pool.

    Calls model.generate (single-turn) or model.generate_conversation
    (multi-turn) and captures `model.get_last_meta()` in the SAME thread —
    crucial because the meta is stashed in `_local` (threading.local), so a
    different thread (e.g. the FSM orchestrator) cannot read it.

    Args:
        model: TargetModel-like (must support generate / generate_conversation).
        prompt_or_messages: str when is_conversation=False; list[dict] when True.
        seed: per-trial seed.
        is_conversation: pick generate_conversation (True) vs generate (False).

    Returns:
        (response, meta) — `meta` is the dict from get_last_meta() if the
        model exposes it, else None (sentinel meaning "skip accumulation",
        consumed by `_accumulate_meta_from_dict`).

    Raises whatever model.generate*() raises (RuntimeError on retries
    exhausted, ContextOverflowError on full context). The caller (FSM
    orchestrator) catches and converts to a trial_exception record.
    """
    if is_conversation:
        response = model.generate_conversation(prompt_or_messages, seed=seed)
    else:
        response = model.generate(prompt_or_messages, seed=seed)
    get_meta = getattr(model, "get_last_meta", None)
    meta = get_meta() if callable(get_meta) else None
    return response, meta


def _accumulate_judge_meta(
    prev: list[dict] | None, judge_meta: dict | None,
) -> list[dict] | None:
    """Append one judge call's meta onto the per-turn judge meta list.

    Semantics:
      - judge_meta is None (rule-based judge or empty dict): pass through
        `prev` unchanged. A rule-based judge never produces meta, so the
        list stays None across all turns.
      - judge_meta is a dict: append a copy. First append lazily initializes
        the list.

    Returns a new list (never mutates `prev`).
    """
    if judge_meta is None:
        return prev
    meta_copy = dict(judge_meta)
    return [meta_copy] if prev is None else prev + [meta_copy]


def trial_exception_record(seed: int, exc: Exception) -> dict:
    """Build a trial record for an exception raised outside the judge path
    (e.g. model inference failure). Marks the trial as out-of-denominator.
    """
    return {
        "seed": seed,
        "response": f"[ERROR: {type(exc).__name__}: {exc}]",
        "passed": None,
        "failed_at_turn": -1,
        "judge_outcome": "trial_exception",
        "judge_error": {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "timestamp": datetime.now().isoformat(),
        },
        "judge_label": None,
        "truncated": None,
        "completion_tokens": None,
        "per_turn_meta": None,
        "per_turn_judge_meta": None,
    }


def run_single_trial(
    model,
    judge,
    prompt: str,
    metadata: dict | None,
    seed: int,
) -> dict:
    """Run exactly one trial (single-turn, or multi-turn via metadata follow_ups).

    Produces a trial record dict matching the schema in `build_trial`. Multi-turn
    is triggered when metadata contains a non-empty "follow_ups" list; each turn
    is judged and the trial fails at the first failing verdict. A None verdict
    (judge API failure) aborts the trial with passed=None so it is excluded
    from p_hat.

    Note: this function does not catch ContextOverflowError. Callers that need
    graceful context-overflow truncation (search-phase multi-turn sycophancy)
    should use `Evaluator._run_single_trial` instead, which treats the most
    recent passing verdict as the trial outcome on overflow.
    """
    follow_ups = (metadata or {}).get("follow_ups") or []
    is_multi_turn = bool(follow_ups)

    # Trial-level meta accumulator. Aggregates + per_turn list populated by
    # _accumulate_meta after each .generate*() call.
    meta_accum: dict = {
        "truncated": None,
        "completion_tokens": None,
        "per_turn_meta": None,
    }
    judge_meta_list: list[dict] | None = None

    # Turn 1
    response = model.generate(prompt, seed=seed)
    meta_accum = _accumulate_meta(model, meta_accum)
    responses = [response]
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    passed, outcome, err, label, judge_meta = judge_and_capture(judge, messages, metadata)
    judge_meta_list = _accumulate_judge_meta(judge_meta_list, judge_meta)

    if passed is None:
        return build_trial(
            seed, response, responses, is_multi_turn,
            passed=None, failed_at_turn=1,
            judge_outcome=outcome, judge_error=err, judge_label=label,
            truncated=meta_accum["truncated"],
            completion_tokens=meta_accum["completion_tokens"],
            per_turn_meta=meta_accum["per_turn_meta"],
            per_turn_judge_meta=judge_meta_list,
        )

    if not follow_ups or not passed:
        return build_trial(
            seed, response, responses, is_multi_turn,
            passed=passed, failed_at_turn=0 if passed else 1,
            judge_outcome=outcome, judge_error=err, judge_label=label,
            truncated=meta_accum["truncated"],
            completion_tokens=meta_accum["completion_tokens"],
            per_turn_meta=meta_accum["per_turn_meta"],
            per_turn_judge_meta=judge_meta_list,
        )

    # Multi-turn follow-ups
    for turn_idx, follow_up in enumerate(follow_ups, start=2):
        messages.append({"role": "user", "content": follow_up})
        response = model.generate_conversation(messages, seed=seed)
        meta_accum = _accumulate_meta(model, meta_accum)
        responses.append(response)
        messages.append({"role": "assistant", "content": response})

        passed, outcome, err, label, judge_meta = judge_and_capture(judge, messages, metadata)
        judge_meta_list = _accumulate_judge_meta(judge_meta_list, judge_meta)
        if passed is None:
            return build_trial(
                seed, response, responses, is_multi_turn,
                passed=None, failed_at_turn=turn_idx,
                judge_outcome=outcome, judge_error=err, judge_label=label,
                truncated=meta_accum["truncated"],
                completion_tokens=meta_accum["completion_tokens"],
                per_turn_meta=meta_accum["per_turn_meta"],
                per_turn_judge_meta=judge_meta_list,
            )
        if not passed:
            return build_trial(
                seed, response, responses, is_multi_turn,
                passed=False, failed_at_turn=turn_idx,
                judge_outcome=outcome, judge_error=err, judge_label=label,
                truncated=meta_accum["truncated"],
                completion_tokens=meta_accum["completion_tokens"],
                per_turn_meta=meta_accum["per_turn_meta"],
                per_turn_judge_meta=judge_meta_list,
            )

    return build_trial(
        seed, response, responses, is_multi_turn,
        passed=True, failed_at_turn=0,
        judge_outcome=outcome, judge_error=err, judge_label=label,
        truncated=meta_accum["truncated"],
        completion_tokens=meta_accum["completion_tokens"],
        per_turn_meta=meta_accum["per_turn_meta"],
        per_turn_judge_meta=judge_meta_list,
    )


def aggregate_trials(trials: list[dict]) -> dict:
    """Summarize a list of trial records into per-(prompt, model) cell stats.

    Returns a dict with:
      - pass_count, total_trials (= n_valid): denominator excludes None
      - n_attempted: raw trial count (incl. all failure modes)
      - n_safety_classified / n_api_failed / n_trial_failed: diagnostic counters
      - p_hat: pass_count / total_trials, or None if total_trials == 0

    Callers should merge this with entry-level identifiers (prompt_id, model).
    """
    valid = [t for t in trials if t.get("passed") is not None]
    pass_count = sum(1 for t in valid if t["passed"])
    total_valid = len(valid)
    p_hat = pass_count / total_valid if total_valid else None

    n_safety_classified = sum(
        1 for t in trials if t.get("judge_outcome") == "safety_classified"
    )
    n_api_failed = sum(
        1 for t in trials
        if t.get("judge_outcome") in ("api_exhausted", "deterministic_aborted")
    )
    n_trial_failed = sum(
        1 for t in trials if t.get("judge_outcome") == "trial_exception"
    )
    # LLMJudge replied successfully but the verdict line couldn't be parsed
    # (no `Answer:`/`Verdict:` line, or non-vocabulary token). Distinct from
    # `n_api_failed` — the API call succeeded; only the regex extraction
    # failed. Always 0 for ConstraintJudge (rule-based, returns bool directly).
    n_judge_label_unparsed = sum(
        1 for t in trials if t.get("judge_outcome") == "judge_label_unparsed"
    )
    # Truncation counter (trial-level). `truncated=True` means at least one
    # turn hit max_tokens. Counts only trials where the model reported a
    # non-None value (trial_exception and get_last_meta-less models give None).
    n_truncated = sum(1 for t in trials if t.get("truncated") is True)
    # Judge-side truncation: judge hit its own max_tokens before emitting the
    # verdict line in ANY turn. Matters for LLMJudge categories (HR / OR / SR).
    # Always 0 for ConstraintJudge (rule-based per_turn_judge_meta is None).
    # Multi-turn: a middle-turn judge truncation can silently flip fail→pass,
    # so we count trials where at least one per-turn entry had truncated=True.
    n_judge_truncated = sum(
        1 for t in trials
        if any(
            (m or {}).get("truncated") is True
            for m in (t.get("per_turn_judge_meta") or [])
        )
    )

    return {
        "pass_count": pass_count,
        "total_trials": total_valid,
        "n_attempted": len(trials),
        "n_safety_classified": n_safety_classified,
        "n_api_failed": n_api_failed,
        "n_trial_failed": n_trial_failed,
        "n_judge_label_unparsed": n_judge_label_unparsed,
        "n_truncated": n_truncated,
        "n_judge_truncated": n_judge_truncated,
        "p_hat": p_hat,
    }
