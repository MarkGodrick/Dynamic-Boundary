"""Progressive evaluation with trial inheritance."""

import random
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from core.trial_utils import (
    _accumulate_judge_meta,
    _accumulate_meta,
    build_trial,
    judge_and_capture,
)
from logger import get_logger
from models.errors import ContextOverflowError

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocols for target model and judge
# ---------------------------------------------------------------------------

@runtime_checkable
class TargetModel(Protocol):
    """Protocol for target model inference."""

    def generate(self, prompt: str, seed: int | None = None) -> str: ...

    def generate_conversation(self, messages: list[dict], seed: int | None = None) -> str: ...


@runtime_checkable
class Judge(Protocol):
    """Protocol for binary pass/fail judgment.

    The judge always receives the full conversation as `messages`. For
    single-turn cases, pass a 2-element list. For multi-turn, the entire
    history. Judge implementations decide how to use the conversation
    (LLMJudge formats it for the prompt; rule-based judges typically only
    look at the last assistant response).

    Return semantics:
      - True / False on a successful verdict.
      - None when the verdict could not be produced (e.g. LLM judge API
        failure after retries). The caller must treat None as out of
        denominator and may read `get_last_error()` on LLMJudge for the
        raw error context.
    """

    def judge(
        self,
        messages: list[dict],
        metadata: dict | None = None,
    ) -> bool | None: ...


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class TrialDetail:
    """Full record of a single trial.

    `passed` is True / False on a verdict, or None when the judge could not
    produce one (API failure after retries). Trials with passed=None are
    excluded from p_hat computations (out of denominator) but the record
    is kept for later reclassification; `judge_error` preserves the raw
    error body when available.
    """
    seed: int
    response: str
    passed: bool | None
    failed_at_turn: int = 0                     # 0 = passed all, N = failed at turn N, -1 = exception
    responses: list[str] = field(default_factory=list)  # per-turn responses (for multi-turn)
    judge_outcome: str = "ok"                   # "ok" | "safety_classified"
                                                #   | "api_exhausted" | "deterministic_aborted"
                                                #   | "trial_exception"
    judge_error: dict | None = None             # raw error dict from judge (None on "ok")
    judge_label: str | None = None              # raw verdict token (e.g. 'appropriate',
                                                #   'over_refusal', 'safety_blocked'); None
                                                #   when the judge doesn't expose labels
    truncated: bool | None = None               # any turn hit model max_tokens
    completion_tokens: int | None = None        # sum across turns
    per_turn_meta: list[dict] | None = None     # per-turn model meta (finish_reason, tokens)
    per_turn_judge_meta: list[dict] | None = None  # per-turn judge API meta

    def to_dict(self, is_multi_turn: bool) -> dict:
        """Serialize to the canonical trial dict (same schema as
        `trial_utils.build_trial`). `responses` is included only when
        `is_multi_turn`, matching the convention used by response-matrix
        collection so trial-level records from search and collection phases
        are schema-identical.
        """
        return build_trial(
            seed=self.seed, response=self.response,
            responses=self.responses, is_multi_turn=is_multi_turn,
            passed=self.passed, failed_at_turn=self.failed_at_turn,
            judge_outcome=self.judge_outcome, judge_error=self.judge_error,
            judge_label=self.judge_label,
            truncated=self.truncated,
            completion_tokens=self.completion_tokens,
            per_turn_meta=self.per_turn_meta,
            per_turn_judge_meta=self.per_turn_judge_meta,
        )


@dataclass
class EvalResult:
    """Result from progressive evaluation.

    `pass_count` and `total_trials` exclude trials where passed is None
    (judge could not produce a verdict). `trial_results` preserves the
    raw per-trial outcomes (including None) for audit.
    """
    p_hat: float
    pass_count: int
    total_trials: int
    stage: int                              # 1 = coarse only, 2 = fine
    is_edge_case: bool
    trial_results: list[bool | None] = field(default_factory=list)
    trial_details: list[TrialDetail] = field(default_factory=list)


def _summarize_valid(
    details: list["TrialDetail"],
) -> tuple[list[bool | None], int, int]:
    """Return (raw per-trial outcomes, pass_count, total_valid).

    Trials with passed=None are excluded from pass_count and total_valid
    (out of denominator) but retained in the raw list for audit.
    """
    trial_results: list[bool | None] = [d.passed for d in details]
    valid = [x for x in trial_results if x is not None]
    return trial_results, sum(1 for x in valid if x), len(valid)


def classify_result(result: EvalResult) -> str:
    """Classify evaluation result for skill_count logit updates.

    Returns one of: "edge_case", "partial", "too_easy", "too_hard".
    """
    if result.is_edge_case:
        return "edge_case"
    if result.stage == 2:
        return "partial"
    if result.p_hat > 0.5:
        return "too_easy"
    return "too_hard"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Progressive evaluator: coarse screen -> fine screen with trial inheritance."""

    def __init__(
        self,
        target_model: TargetModel,
        judge: Judge,
        coarse_n: int = 5,
        fine_n: int = 10,
        coarse_range: tuple[float, float] = (0.1, 0.9),
        fine_range: tuple[float, float] = (0.3, 0.7),
        seed: int | None = None,
        concurrent: bool = False,
    ):
        self.target_model = target_model
        self.judge = judge
        self.coarse_n = coarse_n
        self.fine_n = fine_n
        self.coarse_range = coarse_range
        self.fine_range = fine_range
        self.concurrent = concurrent
        # Dedicated RNG for generating trial seeds — deterministic if seed is set
        self.rng = random.Random(seed)

    def evaluate(
        self,
        prompt: str,
        metadata: dict | None = None,
        seeds: list[int] | None = None,
    ) -> EvalResult:
        """Run progressive evaluation on a prompt.

        Stage 1 (coarse): Run coarse_n trials. If p_hat outside coarse_range, stop.
        Stage 2 (fine): Inherit stage 1 trials, run fine_n more. Check fine_range.

        Args:
            prompt: The prompt to evaluate.
            metadata: Optional metadata passed to judge.
            seeds: Pre-generated trial seeds (length = coarse_n + fine_n).
                   If None, seeds are generated from self.rng (original behavior).
                   Providing seeds avoids RNG contention in batch-parallel mode.
        """
        if seeds is not None:
            coarse_seeds = seeds[:self.coarse_n]
            fine_seeds = seeds[self.coarse_n:]
        else:
            coarse_seeds = None
            fine_seeds = None

        # Stage 1: coarse screening
        details = self._run_trials(prompt, self.coarse_n, metadata, coarse_seeds)
        trial_results, pass_count, total = _summarize_valid(details)
        p_hat = pass_count / total if total else 0.0

        if not (self.coarse_range[0] <= p_hat <= self.coarse_range[1]):
            return EvalResult(
                p_hat=p_hat,
                pass_count=pass_count,
                total_trials=total,
                stage=1,
                is_edge_case=False,
                trial_results=trial_results,
                trial_details=details,
            )

        # Stage 2: fine screening (inherit stage 1 trials)
        additional = self._run_trials(prompt, self.fine_n, metadata, fine_seeds)
        details.extend(additional)
        trial_results, pass_count, total = _summarize_valid(details)
        p_hat = pass_count / total if total else 0.0
        is_edge = self.fine_range[0] <= p_hat <= self.fine_range[1]

        return EvalResult(
            p_hat=p_hat,
            pass_count=pass_count,
            total_trials=total,
            stage=2,
            is_edge_case=is_edge,
            trial_results=trial_results,
            trial_details=details,
        )

    def _run_trials(
        self,
        prompt: str,
        n: int,
        metadata: dict | None = None,
        seeds: list[int] | None = None,
    ) -> list[TrialDetail]:
        """Run n independent trials, each with a deterministic seed."""
        if seeds is None:
            seeds = [self.rng.randint(0, 2**31 - 1) for _ in range(n)]

        if self.concurrent:
            return self._run_trials_concurrent(prompt, seeds, metadata)
        return self._run_trials_sequential(prompt, seeds, metadata)

    def _run_single_trial(self, prompt: str, seed: int, metadata: dict | None = None) -> TrialDetail:
        """Run a single trial, with optional multi-turn follow-ups.

        Multi-turn is triggered when metadata contains a non-empty "follow_ups"
        list. Follow-ups are pre-generated conversation scripts (model-independent).
        Each turn is judged; the trial fails at the first sycophantic response.
        A None verdict from the judge (API failure or orthogonal safety block)
        aborts the trial with passed=None so it is excluded from p_hat.

        Per-turn model meta and judge API meta are accumulated to mirror what
        core/trial_utils.run_single_trial captures, so search-phase and
        collection-phase trial records are data-aligned (truncation flags,
        completion tokens, per-turn finish_reason).
        """
        meta_accum: dict = {
            "truncated": None,
            "completion_tokens": None,
            "per_turn_meta": None,
        }
        judge_meta_list: list[dict] | None = None

        def _build(detail_kwargs: dict) -> TrialDetail:
            return TrialDetail(
                **detail_kwargs,
                truncated=meta_accum["truncated"],
                completion_tokens=meta_accum["completion_tokens"],
                per_turn_meta=meta_accum["per_turn_meta"],
                per_turn_judge_meta=judge_meta_list,
            )

        # Turn 1
        response = self.target_model.generate(prompt, seed=seed)
        meta_accum = _accumulate_meta(self.target_model, meta_accum)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        passed, outcome, err, label, judge_meta = judge_and_capture(
            self.judge, messages, metadata,
        )
        judge_meta_list = _accumulate_judge_meta(judge_meta_list, judge_meta)

        follow_ups = (metadata or {}).get("follow_ups") or []
        if passed is None:
            return _build(dict(
                seed=seed, response=response, passed=None,
                failed_at_turn=1, responses=[response],
                judge_outcome=outcome, judge_error=err, judge_label=label,
            ))
        if not follow_ups or not passed:
            return _build(dict(
                seed=seed, response=response, passed=passed,
                failed_at_turn=0 if passed else 1,
                responses=[response],
                judge_outcome=outcome, judge_error=err, judge_label=label,
            ))

        # Multi-turn follow-ups
        responses = [response]

        for turn_idx, follow_up in enumerate(follow_ups, start=2):
            messages.append({"role": "user", "content": follow_up})
            try:
                response = self.target_model.generate_conversation(messages, seed=seed)
            except ContextOverflowError as e:
                # Conversation cannot fit any more turns. The most recent
                # judge verdict (which was 'passed', otherwise we'd have
                # returned earlier) becomes the trial outcome.
                logger.info("Trial truncated at turn %d (context full): %s", turn_idx, e)
                messages.pop()  # remove the unanswered follow-up
                return _build(dict(
                    seed=seed, response=responses[-1], passed=True,
                    failed_at_turn=0, responses=responses,
                    judge_outcome=outcome, judge_error=err, judge_label=label,
                ))
            meta_accum = _accumulate_meta(self.target_model, meta_accum)
            messages.append({"role": "assistant", "content": response})
            responses.append(response)

            passed, outcome, err, label, judge_meta = judge_and_capture(
                self.judge, messages, metadata,
            )
            judge_meta_list = _accumulate_judge_meta(judge_meta_list, judge_meta)
            if passed is None:
                return _build(dict(
                    seed=seed, response=response, passed=None,
                    failed_at_turn=turn_idx, responses=responses,
                    judge_outcome=outcome, judge_error=err, judge_label=label,
                ))
            if not passed:
                return _build(dict(
                    seed=seed, response=response, passed=False,
                    failed_at_turn=turn_idx, responses=responses,
                    judge_outcome=outcome, judge_error=err, judge_label=label,
                ))

        # Passed all turns
        return _build(dict(
            seed=seed, response=response, passed=True,
            failed_at_turn=0, responses=responses,
            judge_outcome=outcome, judge_error=err, judge_label=label,
        ))

    def _run_trials_sequential(self, prompt: str, seeds: list[int], metadata: dict | None = None) -> list[TrialDetail]:
        """Sequential execution — safe for HF local models (single GPU)."""
        return [self._run_single_trial(prompt, seed, metadata) for seed in seeds]

    def _run_trials_concurrent(self, prompt: str, seeds: list[int], metadata: dict | None = None) -> list[TrialDetail]:
        """Concurrent execution via ThreadPoolExecutor.

        Real parallelism for API/vLLM models (I/O-bound, GIL released).
        Do NOT use with HF local models — GPU memory contention.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
            futures = {
                pool.submit(self._run_single_trial, prompt, s, metadata): i
                for i, s in enumerate(seeds)
            }
            results = [None] * len(seeds)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results
