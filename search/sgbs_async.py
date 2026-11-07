"""Async SGBS: decouple prompt search from prompt evaluation.

This module keeps the SGBS policy/update logic in ``search.sgbs`` and only
replaces the outer scheduler.  The main difference from ``SGBSv2`` is that
composition and evaluation run concurrently across rounds.  Completed eval
results are buffered and processed in round_num order via a sliding window,
so that state updates (especially ``rounds_since_last_edge``) remain correct
even when evals finish out of order.
"""

import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

from logger import get_logger

from . import skill_count, ucb
from .evaluator import EvalResult, Evaluator, Judge, TargetModel
from .sgbs import (
    BareRequest,
    BatchTask,
    RoundRecord,
    SGBSv2,
    SampledTask,
    SearchResult,
    SearchState,
    _compose_task,
    _process_result,
    _save_state_snapshot,
)
from .skill_selector import Skill, load_skills as load_follow_up_skills

logger = get_logger(__name__)


class SGBSAsync(SGBSv2):
    """Skill-Guided Boundary Search with asynchronous compose/eval scheduling."""

    def run(
        self,
        target_model: TargetModel,
        judge: Judge,
        bare_requests: list[BareRequest],
        skills: dict[str, Skill],
        budget: int | None = None,
        initial_state: SearchState | None = None,
        initial_rng_state: tuple | None = None,
        initial_np_rng_state: dict | None = None,
        initial_skipped: int = 0,
        initial_round_records: list[RoundRecord] | None = None,
        prev_elapsed: float = 0.0,
        output_dir: str | None = None,
        stop_event=None,
    ) -> SearchResult:
        """Run SGBS with search and evaluation decoupled.

        The main thread owns all mutable search state.  Worker threads only run
        composition and evaluation calls, then return immutable task/result
        payloads for the main thread to process.
        """
        cfg_search = self.config.get("search", {})
        cfg_ucb = self.config.get("ucb", {})
        cfg_sc = self.config.get("skill_count", {})
        cfg_eval = self.config.get("evaluator", {})

        budget = budget or cfg_search.get("budget", 100)
        target_k = cfg_search.get("target_edge_cases", 10)
        dim_returns_r = cfg_search.get("diminishing_returns", 200)
        max_skip_retries = cfg_search.get("max_skip_retries", 5)
        skip_penalty = cfg_search.get("skip_penalty", 0.1)
        max_compositions = cfg_search.get("max_compositions", 3)
        prior_alpha = cfg_ucb.get("prior_alpha", 0.3)
        prior_beta = cfg_ucb.get("prior_beta", 0.7)
        c = cfg_ucb.get("c", 1.0)
        skill_method = cfg_ucb.get("skill_method", "thompson")
        sc_alpha = cfg_sc.get("alpha", 0.05)
        sc_beta = cfg_sc.get("beta", 0.1)
        mis_size = self.config.get("skills", {}).get("mis_size", 8)
        top_k = cfg_sc.get("top_k", 3)
        top_p = cfg_sc.get("top_p", 0.9)
        max_count = cfg_sc.get("max_count", 4)
        lower_bound = cfg_sc.get("lower_bound", 0)

        br_lookup = {br.prompt: br for br in bare_requests}

        if initial_state is not None:
            state = initial_state
            start_round = state.round_num + 1
            logger.info("Resuming async SGBS from round %d", start_round)
        else:
            state = SearchState(
                bare_request_profiles=ucb.create_profiles(
                    [br.prompt for br in bare_requests], prior_alpha, prior_beta,
                ),
                skill_profiles=ucb.create_profiles(
                    list(skills.keys()), prior_alpha, prior_beta,
                ),
                skill_logits=skill_count.init_skill_logits(
                    max_count,
                    cfg_sc.get("bare_request_bias", 3.0),
                ),
                tried_combinations=defaultdict(Counter),
                edge_cases=[],
            )
            start_round = 1

        fu_dict_path = self.config.get("skills", {}).get("follow_up_dictionary_path")
        max_follow_up_turns = cfg_eval.get("max_follow_up_turns", 0)
        if fu_dict_path and max_follow_up_turns > 0:
            follow_up_skill_pool = list(load_follow_up_skills(fu_dict_path).values())
            logger.info(
                "Loaded %d follow-up skills (turns uniform in [1, %d])",
                len(follow_up_skill_pool), max_follow_up_turns,
            )
        else:
            follow_up_skill_pool = []

        exp_seed = self.config.get("experiment", {}).get("seed", 42)
        composer = self._build_composer()

        from models.api_model import APIModel
        from models.vllm_model import VLLMModel

        prompt_concurrent = (
            isinstance(target_model, (APIModel, VLLMModel))
            or bool(getattr(target_model, "supports_concurrent", False))
        )
        default_in_flight = cfg_search.get(
            "batch_size", 4 if prompt_concurrent else 1,
        )
        max_in_flight = max(1, int(cfg_search.get("max_in_flight", default_in_flight)))
        compose_workers = max(1, int(cfg_search.get("compose_workers", 1)))
        eval_workers = max(
            1,
            int(cfg_search.get(
                "eval_workers", max_in_flight if prompt_concurrent else 1,
            )),
        )
        trial_concurrent = bool(cfg_search.get("async_trial_concurrent", False))

        evaluator = Evaluator(
            target_model=target_model,
            judge=judge,
            coarse_n=cfg_eval.get("coarse_n", 5),
            fine_n=cfg_eval.get("fine_n", 10),
            coarse_range=tuple(cfg_eval.get("coarse_range", [0.1, 0.9])),
            fine_range=tuple(cfg_eval.get("fine_range", [0.3, 0.7])),
            seed=exp_seed,
            concurrent=trial_concurrent,
        )
        if initial_rng_state is not None:
            evaluator.rng.setstate(initial_rng_state)

        np_rng = np.random.default_rng(exp_seed)
        if initial_np_rng_state is not None:
            np_rng.bit_generator.state = initial_np_rng_state

        skipped = initial_skipped
        round_records: list[RoundRecord] = list(initial_round_records or [])
        t_start = time.time()
        _out_dir = Path(output_dir) if output_dir else None
        seeds_per_task = evaluator.coarse_n + evaluator.fine_n

        logger.info(
            "Async SGBS starting: selector=%s, budget=%d, start_round=%d, "
            "bare_requests=%d, skills=%d, seed=%s, max_in_flight=%d, "
            "compose_workers=%d, eval_workers=%d, trial_concurrent=%s",
            self.selector, budget, start_round, len(bare_requests), len(skills),
            exp_seed, max_in_flight, compose_workers, eval_workers,
            trial_concurrent,
        )

        next_round = start_round
        stop_submitting = False
        in_flight: dict[Future, tuple[str, SampledTask | BatchTask]] = {}
        # Ordered-processing state: buffer eval results and consume in
        # round_num order so that _process_result (and especially
        # rounds_since_last_edge) sees a monotonically increasing sequence.
        pending_eval_results: dict[int, tuple[BatchTask, EvalResult]] = {}
        skipped_round_nums: set[int] = set()
        next_process_round = start_round

        def _submit_until_full(compose_pool: ThreadPoolExecutor) -> None:
            nonlocal next_round, skipped, stop_submitting
            if stop_event is not None and stop_event.is_set():
                stop_submitting = True
            while (
                not stop_submitting
                and next_round <= budget
                and len(in_flight) < max_in_flight
            ):
                round_num = next_round
                next_round += 1

                sampled = self._sample_fn(
                    state=state,
                    round_num=round_num,
                    br_lookup=br_lookup,
                    skills=skills,
                    np_rng=np_rng,
                    c=c,
                    skill_method=skill_method,
                    max_skip_retries=max_skip_retries,
                    skip_penalty=skip_penalty,
                    mis_size=mis_size,
                    top_k=top_k,
                    top_p=top_p,
                    lower_bound=lower_bound,
                    sc_alpha=sc_alpha,
                    sc_beta=sc_beta,
                    follow_up_skill_pool=follow_up_skill_pool,
                    max_follow_up_turns=max_follow_up_turns,
                    max_compositions=max_compositions,
                )
                if sampled is None:
                    skipped += 1
                    skipped_round_nums.add(round_num)
                    continue

                future = compose_pool.submit(_compose_task, sampled, composer)
                in_flight[future] = ("compose", sampled)

        with ThreadPoolExecutor(max_workers=compose_workers) as compose_pool:
            with ThreadPoolExecutor(max_workers=eval_workers) as eval_pool:
                while True:
                    _submit_until_full(compose_pool)

                    if not in_flight and not pending_eval_results:
                        if stop_submitting or next_round > budget:
                            break
                        continue

                    done, _pending = wait(
                        list(in_flight.keys()),
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        kind, payload = in_flight.pop(future)

                        if kind == "compose":
                            task = future.result()
                            if task is None:
                                skipped += 1
                                skipped_round_nums.add(payload.round_num)
                                continue
                            task.seeds = [
                                evaluator.rng.randint(0, 2**31 - 1)
                                for _ in range(seeds_per_task)
                            ]
                            eval_future = eval_pool.submit(
                                evaluator.evaluate,
                                task.prompt,
                                task.metadata,
                                task.seeds,
                            )
                            in_flight[eval_future] = ("eval", task)
                            continue

                        # kind == "eval": buffer result for ordered processing
                        task = payload
                        eval_result = future.result()
                        pending_eval_results[task.round_num] = (task, eval_result)

                    # Sliding window: process buffered results in round_num
                    # order so that _process_result (and especially
                    # rounds_since_last_edge) sees a monotonically increasing
                    # sequence, matching the synchronous SGBSv2 contract.
                    while (
                        next_process_round in pending_eval_results
                        or next_process_round in skipped_round_nums
                    ):
                        if next_process_round in skipped_round_nums:
                            skipped_round_nums.discard(next_process_round)
                            next_process_round += 1
                            continue

                        task, eval_result = pending_eval_results.pop(
                            next_process_round,
                        )
                        state.round_num = next_process_round
                        next_process_round += 1
                        _process_result(
                            state=state,
                            task=task,
                            eval_result=eval_result,
                            round_records=round_records,
                            sc_alpha=sc_alpha,
                            sc_beta=sc_beta,
                            lower_bound=lower_bound,
                            out_dir=_out_dir,
                        )

                        if _out_dir:
                            _save_state_snapshot(
                                _out_dir, state, skipped,
                                evaluator.rng.getstate(),
                                np_rng.bit_generator.state,
                                state.round_num,
                                t_start,
                                prev_elapsed,
                                exp_seed,
                            )

                        if state.round_num % 10 == 0 or state.round_num >= budget:
                            logger.info(
                                "Async SGBS round %d/%d: edge_cases=%d, "
                                "skipped=%d, in_flight=%d, logits=%s",
                                state.round_num, budget,
                                len(state.edge_cases),
                                skipped, len(in_flight),
                                [f"{l:.2f}" for l in state.skill_logits],
                            )

                        if stop_event is not None and stop_event.is_set():
                            if not stop_submitting:
                                logger.info(
                                    "Stop event set; draining %d in-flight"
                                    " tasks",
                                    len(in_flight),
                                )
                            stop_submitting = True
                        if len(state.edge_cases) >= target_k:
                            if not stop_submitting:
                                logger.info(
                                    "Target edge cases reached (%d); "
                                    "draining %d in-flight tasks",
                                    target_k, len(in_flight),
                                )
                            stop_submitting = True
                        elif state.rounds_since_last_edge >= dim_returns_r:
                            if not stop_submitting:
                                logger.info(
                                    "Diminishing returns (%d rounds "
                                    "without edge case); draining %d "
                                    "in-flight tasks",
                                    dim_returns_r, len(in_flight),
                                )
                            stop_submitting = True

        elapsed = time.time() - t_start + prev_elapsed
        stats = {
            "rounds_run": state.round_num,
            "edge_cases_found": len(state.edge_cases),
            "skipped_rounds": skipped,
            "final_logits": list(state.skill_logits),
            "total_time_seconds": round(elapsed, 1),
            "total_time_human": f"{elapsed / 60:.1f} minutes",
            "seed": exp_seed,
            "mode": "async",
            "max_in_flight": max_in_flight,
            "compose_workers": compose_workers,
            "eval_workers": eval_workers,
        }
        logger.info("Async SGBS finished: %s", stats)

        return SearchResult(
            edge_cases=state.edge_cases,
            round_records=sorted(round_records, key=lambda r: r.round_num),
            state=state,
            stats=stats,
            rng_state=evaluator.rng.getstate(),
            np_rng_state=np_rng.bit_generator.state,
            skipped=skipped,
        )
