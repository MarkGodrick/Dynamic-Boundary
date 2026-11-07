"""SGBS v2: Skill-Guided Boundary Search — main flat loop."""

import json
import platform
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from logger import get_logger

from . import ucb, skill_count
from .composer import Composer, TemplateComposer
from .evaluator import Evaluator, TargetModel, Judge, TrialDetail, classify_result
from .skill_selector import Skill, load_skills as load_follow_up_skills, select_skills

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BareRequest:
    """A base request with optional metadata for the judge."""
    prompt: str
    metadata: dict = field(default_factory=dict)


@dataclass
class EdgeCase:
    """A discovered boundary prompt."""
    prompt: str
    bare_request: str
    skills: list[str]
    p_hat: float
    pass_count: int
    total_trials: int
    round_found: int
    follow_ups: list[str] = field(default_factory=list)
    follow_up_skills: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class RoundRecord:
    """Detailed record of a single SGBS round (for trial_details.jsonl)."""
    round_num: int
    bare_request: str
    skill_count: int
    skills: list[str]
    prompt: str
    p_hat: float
    stage: int
    result_type: str       # edge_case / partial / too_easy / too_hard
    is_edge_case: bool
    trials: list[dict]     # [{seed, response, passed}, ...]
    follow_ups: list[str] = field(default_factory=list)
    # Composer API token usage for this round (None for TemplateComposer /
    # runs predating token accounting). See core.usage.accumulate_usage.
    compose_usage: dict | None = None


@dataclass
class SampledTask:
    """Result of sampling, ready for compose.  No LLM call yet."""
    round_num: int
    q_key: str
    q: BareRequest
    count: int
    selected_skills: list[Skill]
    fu_skills: list[Skill] | None


@dataclass
class BatchTask:
    """Pre-computed data for one round, ready for parallel evaluation."""
    round_num: int
    q_key: str
    q: BareRequest
    count: int
    selected_skills: list[Skill]
    prompt: str
    metadata: dict
    seeds: list[int]
    fu_skills: list[Skill] | None = None
    compose_usage: dict | None = None


@dataclass
class SearchState:
    """Full mutable state of the SGBS search loop."""
    bare_request_profiles: dict[str, ucb.ArmProfile]
    skill_profiles: dict[str, ucb.ArmProfile]
    skill_logits: list[float]
    # per-q: Counter keyed by (frozenset(main_skills), tuple(fu_skills)) -> n_tried
    tried_combinations: dict[str, Counter]
    edge_cases: list[EdgeCase]
    round_num: int = 0
    rounds_since_last_edge: int = 0


@dataclass
class SearchResult:
    """Final output of a SGBS run."""
    edge_cases: list[EdgeCase]
    round_records: list[RoundRecord]
    state: SearchState
    stats: dict = field(default_factory=dict)
    rng_state: tuple | None = None
    np_rng_state: dict | None = None
    skipped: int = 0


# ---------------------------------------------------------------------------
# Selector dispatch
# ---------------------------------------------------------------------------

def _resolve_selector(config: dict):
    """Resolve `search.selector` to (name, sampling function).

    "thompson" (default) — the SGBS policy: Beta posteriors for bare request
        and skills, logit distribution for the skill count.
    "uniform" — the naive baseline: same composition/evaluation machinery,
        uniform draws for (bare request, skill count, skill subset).

    Both functions share `_sample`'s keyword-only signature and are called
    interchangeably by the sync and async loops.
    """
    name = (config.get("search") or {}).get("selector", "thompson")
    if name == "thompson":
        return name, _sample
    if name == "uniform":
        # Local import: search.uniform imports from this module.
        from .uniform import sample_uniform
        return name, sample_uniform
    raise ValueError(
        f"Unsupported search.selector: {name!r} (expected 'thompson' or 'uniform')"
    )


# ---------------------------------------------------------------------------
# SGBS v2
# ---------------------------------------------------------------------------

class SGBSv2:
    """Skill-Guided Boundary Search v2 — single flat loop."""

    def __init__(self, config: dict):
        self.config = config
        self.selector, self._sample_fn = _resolve_selector(config)

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
        """Run the SGBS v2 search loop.

        Args:
            target_model: Model to evaluate.
            judge: Binary pass/fail judge.
            bare_requests: Pool of base requests.
            skills: Skill dictionary (name -> Skill).
            budget: Max rounds (overrides config if provided).
            initial_state: Restored state for resumption (None = fresh start).
            initial_rng_state: Restored evaluator RNG state.
            initial_np_rng_state: Restored numpy RNG state (for skill_count sampling).
            initial_skipped: Restored skipped counter.
            initial_round_records: Restored round records from previous run.
            prev_elapsed: Elapsed time from previous run (for cumulative timing).
            output_dir: If provided, incrementally save results during the loop.
            stop_event: Optional threading.Event. When set, the loop exits
                gracefully at the next batch boundary (after snapshot is
                flushed). Wired to two-stage Ctrl-C handlers; see
                core/signal_utils.py.
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
        skill_method = cfg_ucb.get("skill_method", "thompson")  # "thompson" or "ucb"
        sc_alpha = cfg_sc.get("alpha", 0.05)
        sc_beta = cfg_sc.get("beta", 0.1)
        mis_size = self.config.get("skills", {}).get("mis_size", 8)
        top_k = cfg_sc.get("top_k", 3)
        top_p = cfg_sc.get("top_p", 0.9)
        max_count = cfg_sc.get("max_count", 4)
        lower_bound = cfg_sc.get("lower_bound", 0)

        # --- Initialize state ---
        # Build prompt -> BareRequest lookup for UCB key resolution
        br_lookup = {br.prompt: br for br in bare_requests}

        if initial_state is not None:
            state = initial_state
            start_round = state.round_num + 1
            logger.info("Resuming from round %d", start_round)
        else:
            state = SearchState(
                bare_request_profiles=ucb.create_profiles(
                    [br.prompt for br in bare_requests], prior_alpha, prior_beta,
                ),
                skill_profiles=ucb.create_profiles(
                    list(skills.keys()), prior_alpha, prior_beta,
                ),
                skill_logits=skill_count.init_skill_logits(max_count,
                                                            cfg_sc.get("bare_request_bias", 3.0)),
                tried_combinations=defaultdict(Counter),
                edge_cases=[],
            )
            start_round = 1

        # Follow-up skills for multi-turn evaluation (sycophancy etc.)
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

        # --- Build composer and evaluator ---
        composer = self._build_composer()

        # Auto-detect concurrent mode: API/vLLM models and external adapters
        # with an explicitly thread-safe client support true parallelism.
        from models.api_model import APIModel
        from models.vllm_model import VLLMModel
        concurrent = (
            isinstance(target_model, (APIModel, VLLMModel))
            or bool(getattr(target_model, "supports_concurrent", False))
        )

        evaluator = Evaluator(
            target_model=target_model,
            judge=judge,
            coarse_n=cfg_eval.get("coarse_n", 5),
            fine_n=cfg_eval.get("fine_n", 10),
            coarse_range=tuple(cfg_eval.get("coarse_range", [0.1, 0.9])),
            fine_range=tuple(cfg_eval.get("fine_range", [0.3, 0.7])),
            seed=exp_seed,
            concurrent=concurrent,
        )

        # Restore evaluator RNG state if resuming
        if initial_rng_state is not None:
            evaluator.rng.setstate(initial_rng_state)

        # Numpy RNG for skill_count sampling
        np_rng = np.random.default_rng(exp_seed)
        if initial_np_rng_state is not None:
            np_rng.bit_generator.state = initial_np_rng_state

        skipped = initial_skipped
        round_records: list[RoundRecord] = list(initial_round_records or [])
        t_start = time.time()

        # Incremental save setup
        _out_dir = Path(output_dir) if output_dir else None

        batch_size = cfg_search.get("batch_size", 1)
        seeds_per_task = evaluator.coarse_n + evaluator.fine_n

        logger.info(
            "SGBS v2 starting: selector=%s, budget=%d, start_round=%d, "
            "bare_requests=%d, skills=%d, seed=%s, batch_size=%d",
            self.selector, budget, start_round, len(bare_requests), len(skills),
            exp_seed, batch_size,
        )

        # --- Batch loop ---
        terminated = False
        round_cursor = start_round

        while round_cursor <= budget and not terminated:
            actual_size = min(batch_size, budget + 1 - round_cursor)

            # Phase 1a: Sequential sampling (fast, no LLM calls)
            ready_tasks: list[BatchTask] = []   # count=0, no compose needed
            to_compose: list[SampledTask] = []  # count>0, need LLM compose
            for i in range(actual_size):
                round_num = round_cursor + i
                state.round_num = round_num

                result = self._sample_fn(
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
                if result is None:
                    skipped += 1
                elif isinstance(result, BatchTask):
                    ready_tasks.append(result)
                else:
                    to_compose.append(result)

            # Phase 1b: Parallel compose (LLM API calls)
            if to_compose:
                if len(to_compose) == 1 or not concurrent:
                    for s in to_compose:
                        task = _compose_task(s, composer)
                        if task is not None:
                            ready_tasks.append(task)
                        else:
                            skipped += 1
                else:
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    with ThreadPoolExecutor(max_workers=len(to_compose)) as pool:
                        futures = {
                            pool.submit(_compose_task, s, composer): s
                            for s in to_compose
                        }
                        for f in as_completed(futures):
                            task = f.result()
                            if task is not None:
                                ready_tasks.append(task)
                            else:
                                skipped += 1

            # Assign seeds and sort by round_num for deterministic order
            tasks: list[BatchTask] = sorted(
                ready_tasks, key=lambda t: t.round_num,
            )
            for task in tasks:
                task.seeds = [evaluator.rng.randint(0, 2**31 - 1)
                              for _ in range(seeds_per_task)]

            round_cursor += actual_size

            if not tasks:
                continue

            # Phase 2: Evaluate (parallel if batch_size > 1 and concurrent)
            if len(tasks) == 1 or not concurrent:
                eval_results = [
                    (t, evaluator.evaluate(t.prompt, t.metadata, seeds=t.seeds))
                    for t in tasks
                ]
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
                    futures = {
                        pool.submit(
                            evaluator.evaluate, t.prompt, t.metadata, t.seeds,
                        ): t
                        for t in tasks
                    }
                    eval_results = []
                    for f in as_completed(futures):
                        eval_results.append((futures[f], f.result()))
                # Sort by round_num for deterministic update order
                eval_results.sort(key=lambda x: x[0].round_num)

            # Phase 3: Sequential update (fast)
            for task, eval_result in eval_results:
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

            # Batch-level: save snapshot + log + termination check
            last_round = max(t.round_num for t in tasks)
            if _out_dir:
                _save_state_snapshot(
                    _out_dir, state, skipped,
                    evaluator.rng.getstate(), np_rng.bit_generator.state,
                    last_round, t_start, prev_elapsed, exp_seed,
                )

            if last_round % 10 < actual_size or last_round == budget:
                logger.info(
                    "Round %d/%d: edge_cases=%d, skipped=%d, logits=%s",
                    last_round, budget, len(state.edge_cases), skipped,
                    [f"{l:.2f}" for l in state.skill_logits],
                )

            if stop_event is not None and stop_event.is_set():
                logger.info(
                    "Stop event set; exiting after batch snapshot "
                    "(round=%d, edge_cases=%d)",
                    last_round, len(state.edge_cases),
                )
                terminated = True

            if len(state.edge_cases) >= target_k:
                logger.info("Target edge cases reached (%d)", target_k)
                terminated = True
            elif state.rounds_since_last_edge >= dim_returns_r:
                logger.info(
                    "Diminishing returns (%d rounds without edge case)",
                    dim_returns_r,
                )
                terminated = True

        # --- Build result ---
        elapsed = time.time() - t_start + prev_elapsed
        stats = {
            "rounds_run": state.round_num,
            "edge_cases_found": len(state.edge_cases),
            "skipped_rounds": skipped,
            "final_logits": list(state.skill_logits),
            "total_time_seconds": round(elapsed, 1),
            "total_time_human": f"{elapsed / 60:.1f} minutes",
            "seed": exp_seed,
        }
        logger.info("SGBS v2 finished: %s", stats)
        return SearchResult(
            edge_cases=state.edge_cases,
            round_records=round_records,
            state=state,
            stats=stats,
            rng_state=evaluator.rng.getstate(),
            np_rng_state=np_rng.bit_generator.state,
            skipped=skipped,
        )

    def _build_composer(self) -> Composer | TemplateComposer:
        cfg = self.config.get("composer", {})
        composer_type = cfg.get("type", "llm")

        if composer_type == "template":
            seed = self.config.get("experiment", {}).get("seed", 42)
            return TemplateComposer(seed=seed)

        category = self.config.get("bare_requests", {}).get(
            "category", "harmful_refusal",
        )
        return Composer(
            model=cfg.get("model", "deepseek-chat"),
            api_key_env=cfg.get("api_key_env", "DEEPSEEK_API_TOKEN"),
            base_url=cfg.get("base_url", "https://api.deepseek.com"),
            max_retries=cfg.get("max_retries", 5),
            temperature=cfg.get("temperature", 0.7),
            max_tokens=cfg.get("max_tokens", 2048),
            category=category,
        )


# ---------------------------------------------------------------------------
# Batch helpers (extracted from main loop)
# ---------------------------------------------------------------------------

def _sample_follow_ups(
    pool: list[Skill],
    max_turns: int,
    np_rng: np.random.Generator,
) -> list[Skill] | None:
    """Sample follow-up skills using the seeded numpy RNG.

    Returns None when the pool is empty or max_turns <= 0 (single-turn run).
    Turn count is uniform in [1, min(max_turns, pool_size)] so each item gets
    variable pressure depth.

    Note: must use the seeded `np_rng` (not the global `random` module) so
    SR runs are reproducible across re-runs and resumes.
    """
    if not pool or max_turns <= 0:
        return None
    upper = min(max_turns, len(pool))
    n_fu = int(np_rng.integers(1, upper + 1))
    indices = np_rng.choice(len(pool), size=n_fu, replace=False)
    return [pool[int(i)] for i in indices]


def _sample(
    *,
    state: SearchState,
    round_num: int,
    br_lookup: dict[str, BareRequest],
    skills: dict[str, Skill],
    np_rng: np.random.Generator,
    c: float,
    skill_method: str,
    max_skip_retries: int,
    skip_penalty: float,
    mis_size: int,
    top_k: int,
    top_p: float,
    lower_bound: int,
    sc_alpha: float,
    sc_beta: float,
    follow_up_skill_pool: list[Skill] | None = None,
    max_follow_up_turns: int = 0,
    max_compositions: int = 3,
) -> SampledTask | BatchTask | None:
    """Sample bare request + skills for one round (no LLM call).

    Returns:
        BatchTask  — count=0, no compose needed (prompt = raw bare request).
        SampledTask — count>0, needs compose via _compose_task().
        None       — all retries exhausted (skip).

    Mutates state: tried_combinations, skill_logits, skill penalty on skip.
    """
    br_scores = ucb.thompson_scores(state.bare_request_profiles, np_rng)
    if not br_scores:
        return None
    q_key = max(br_scores, key=br_scores.get)
    q = br_lookup[q_key]

    selected_skills: list[Skill] = []
    for _retry in range(max_skip_retries):
        count = skill_count.sample_count(
            state.skill_logits, mis_size, top_k, top_p,
            lower_bound=lower_bound, rng=np_rng,
        )

        if count == 0:
            # Sample follow-up skills first so they participate in the dedup key.
            # Turn count is uniform in [1, min(max_follow_up_turns, pool_size)]
            # so each item gets variable pressure depth.
            fu_skills = _sample_follow_ups(
                follow_up_skill_pool, max_follow_up_turns, np_rng,
            )
            fu_part = tuple(s.name for s in (fu_skills or []))
            combo_key = (frozenset(), fu_part)
            if state.tried_combinations[q_key][combo_key] >= max_compositions:
                continue
            state.tried_combinations[q_key][combo_key] += 1

            # Always route through composer (rephrase + follow-ups)
            return SampledTask(
                round_num=round_num, q_key=q_key, q=q,
                count=0, selected_skills=[],
                fu_skills=fu_skills,
            )

        if skill_method == "thompson":
            scores = ucb.thompson_scores(state.skill_profiles, np_rng)
        else:
            scores = ucb.ucb_scores(state.skill_profiles, round_num, c)

        selected_skills = select_skills(count, scores, skills)
        if selected_skills is None:
            skill_count.update_logits(
                state.skill_logits, count, "no_combo", sc_alpha, sc_beta,
                lower_bound=lower_bound,
            )
            continue

        # Sample follow-up skills first so they participate in the dedup key.
        # Variable turn count: uniform in [1, min(max_follow_up_turns, pool_size)].
        fu_skills = _sample_follow_ups(
            follow_up_skill_pool, max_follow_up_turns, np_rng,
        )
        fu_part = tuple(s.name for s in (fu_skills or []))
        combo_key = (frozenset(s.name for s in selected_skills), fu_part)
        if state.tried_combinations[q_key][combo_key] >= max_compositions:
            for s in selected_skills:
                ucb.penalize_profile(state.skill_profiles[s.name], skip_penalty)
            continue

        state.tried_combinations[q_key][combo_key] += 1

        return SampledTask(
            round_num=round_num, q_key=q_key, q=q,
            count=count, selected_skills=selected_skills,
            fu_skills=fu_skills,
        )

    return None  # all retries exhausted


def _compose_task(sampled: SampledTask, composer) -> BatchTask | None:
    """Call composer to create the final prompt. Thread-safe.

    Returns BatchTask on success, None on compose failure.
    """
    try:
        compose_kwargs = {}
        if sampled.fu_skills:
            compose_kwargs["follow_up_skills"] = sampled.fu_skills
        prompt, compose_meta = composer.compose(
            sampled.q.prompt, sampled.selected_skills, **compose_kwargs,
        )
    except Exception as e:
        logger.warning("Compose failed for q='%s': %s", sampled.q.prompt[:60], e)
        return None

    # Token usage of the compose call(s), read on the same worker thread that
    # made them. Composers without the method (older siblings) yield None.
    compose_usage = getattr(composer, "get_last_usage", lambda: None)()

    metadata = {**sampled.q.metadata, **compose_meta}
    return BatchTask(
        round_num=sampled.round_num,
        q_key=sampled.q_key,
        q=sampled.q,
        count=sampled.count,
        selected_skills=sampled.selected_skills,
        prompt=prompt,
        metadata=metadata,
        seeds=[],
        fu_skills=sampled.fu_skills,
        compose_usage=compose_usage,
    )


def _process_result(
    *,
    state: SearchState,
    task: BatchTask,
    eval_result,
    round_records: list[RoundRecord],
    sc_alpha: float,
    sc_beta: float,
    lower_bound: int,
    out_dir: Path | None,
) -> None:
    """Update state after a single evaluation completes.

    Handles: profile updates, logit updates, edge case recording,
    incremental saves (trial_details + edge_cases).
    """
    result_type = classify_result(eval_result)
    reward = ucb.compute_reward(eval_result.p_hat)

    # Record round details
    follow_ups = task.metadata.get("follow_ups", [])
    is_multi_turn = bool(follow_ups)
    round_records.append(RoundRecord(
        round_num=task.round_num,
        bare_request=task.q_key,
        skill_count=task.count,
        skills=[s.name for s in task.selected_skills],
        prompt=task.prompt,
        p_hat=eval_result.p_hat,
        stage=eval_result.stage,
        result_type=result_type,
        is_edge_case=eval_result.is_edge_case,
        trials=[t.to_dict(is_multi_turn=is_multi_turn) for t in eval_result.trial_details],
        follow_ups=follow_ups,
        compose_usage=task.compose_usage,
    ))

    # Update skill_logits
    skill_count.update_logits(
        state.skill_logits, task.count, result_type, sc_alpha, sc_beta,
        lower_bound=lower_bound,
    )

    # Update bandit profiles
    ucb.update_profile(state.bare_request_profiles[task.q_key], reward)
    if task.count > 0 and task.selected_skills:
        ind_reward = ucb.compute_individual_reward(reward, task.count)
        for s in task.selected_skills:
            ucb.update_profile(state.skill_profiles[s.name], ind_reward)

    # Edge case handling
    if eval_result.is_edge_case:
        ec = EdgeCase(
            prompt=task.prompt,
            bare_request=task.q_key,
            skills=[s.name for s in task.selected_skills],
            p_hat=eval_result.p_hat,
            pass_count=eval_result.pass_count,
            total_trials=eval_result.total_trials,
            round_found=task.round_num,
            follow_ups=follow_ups,
            follow_up_skills=[s.name for s in (task.fu_skills or [])],
            metadata=task.metadata,
        )
        state.edge_cases.append(ec)
        state.rounds_since_last_edge = 0
        logger.info(
            "Edge case #%d found (round %d): p=%.3f, skills=%s",
            len(state.edge_cases), task.round_num, ec.p_hat, ec.skills,
        )
        if out_dir:
            with open(out_dir / "edge_cases.jsonl", "a") as f:
                f.write(json.dumps(asdict(ec), ensure_ascii=False) + "\n")
    else:
        state.rounds_since_last_edge += 1

    # Incremental save: trial details
    if out_dir:
        with open(out_dir / "trial_details.jsonl", "a") as f:
            f.write(json.dumps(asdict(round_records[-1]), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Incremental save (during search loop)
# ---------------------------------------------------------------------------

def _save_state_snapshot(
    out_dir: Path,
    state: SearchState,
    skipped: int,
    rng_state: tuple,
    np_rng_state: dict,
    round_num: int,
    t_start: float,
    prev_elapsed: float,
    seed: int,
) -> None:
    """Save search state snapshot for resume. Overwrites previous snapshot."""
    elapsed = time.time() - t_start + prev_elapsed
    state_dict = {
        "round_num": state.round_num,
        "rounds_since_last_edge": state.rounds_since_last_edge,
        "skill_logits": state.skill_logits,
        "skipped": skipped,
        "bare_request_profiles": {
            k: {"alpha": v.alpha, "beta": v.beta}
            for k, v in state.bare_request_profiles.items()
        },
        "skill_profiles": {
            k: {"alpha": v.alpha, "beta": v.beta}
            for k, v in state.skill_profiles.items()
        },
        "tried_combinations": _serialize_tried_combinations(state.tried_combinations),
        "rng_state": rng_state,
        "np_rng_state": _serialize_np_rng_state(np_rng_state),
    }
    with open(out_dir / "search_state.json", "w") as f:
        json.dump(state_dict, f, indent=2, ensure_ascii=False)

    stats = {
        "rounds_run": round_num,
        "edge_cases_found": len(state.edge_cases),
        "skipped_rounds": skipped,
        "final_logits": list(state.skill_logits),
        "total_time_seconds": round(elapsed, 1),
        "total_time_human": f"{elapsed / 60:.1f} minutes",
        "seed": seed,
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _serialize_tried_combinations(
    tried: dict[str, Counter],
) -> dict[str, list[list]]:
    """Serialize tried_combinations to JSON-safe nested lists.

    Each q_key maps to a list of [main_list, fu_list, n_tried] entries.
    """
    return {
        k: [
            [sorted(main), list(fu), n_tried]
            for (main, fu), n_tried in counter.items()
        ]
        for k, counter in tried.items()
    }


def _deserialize_tried_combinations(
    raw: dict[str, list],
) -> defaultdict[str, Counter]:
    """Reconstruct tried_combinations from JSON.

    Supports new format (3-tuple [main, fu, n_tried]) and legacy format
    (plain list[list[str]] of frozensets without fu/count). Legacy entries
    migrate to (frozenset(items), ()): 1.
    """
    out: defaultdict[str, Counter] = defaultdict(Counter)
    for k, entries in raw.items():
        counter: Counter = Counter()
        for entry in entries:
            if (
                len(entry) == 3
                and isinstance(entry[0], list)
                and isinstance(entry[1], list)
                and isinstance(entry[2], int)
            ):
                main, fu, n_tried = entry
                counter[(frozenset(main), tuple(fu))] = n_tried
            else:
                # Legacy: entry is a flat list of skill names (a frozenset).
                counter[(frozenset(entry), ())] = 1
        out[k] = counter
    return out


def _serialize_np_rng_state(state: dict | None) -> dict | None:
    """Convert np.random.Generator bit_generator state to JSON-safe dict."""
    if state is None:
        return None
    out = {"bit_generator": state["bit_generator"]}
    s = state["state"]
    out["state"] = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in s.items()
    }
    out["has_uint32"] = int(state["has_uint32"])
    out["uinteger"] = int(state["uinteger"])
    return out


def _deserialize_np_rng_state(data: dict | None) -> dict | None:
    """Convert JSON dict back to np.random.Generator bit_generator state."""
    if data is None:
        return None
    s = {}
    for k, v in data["state"].items():
        if isinstance(v, list):
            s[k] = np.array(v, dtype=np.uint64)
        else:
            s[k] = v
    return {
        "bit_generator": data["bit_generator"],
        "state": s,
        "has_uint32": data["has_uint32"],
        "uinteger": data["uinteger"],
    }


def save_results(result: SearchResult, output_dir: str, config: dict | None = None) -> None:
    """Save search results to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Edge cases — JSONL
    ec_path = out / "edge_cases.jsonl"
    with open(ec_path, "w") as f:
        for ec in result.edge_cases:
            f.write(json.dumps(asdict(ec), ensure_ascii=False) + "\n")
    logger.info("Saved %d edge cases to %s", len(result.edge_cases), ec_path)

    # Trial details — JSONL (one record per round, with full responses)
    details_path = out / "trial_details.jsonl"
    with open(details_path, "w") as f:
        for rr in result.round_records:
            f.write(json.dumps(asdict(rr), ensure_ascii=False) + "\n")
    logger.info("Saved %d round records to %s", len(result.round_records), details_path)

    # Stats — JSON
    stats_path = out / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(result.stats, f, indent=2, ensure_ascii=False)

    # Metadata — JSON (environment info)
    _save_metadata(out, result.stats, config)

    # Search state — JSON (for potential resumption)
    state_path = out / "search_state.json"
    state_dict = {
        "round_num": result.state.round_num,
        "rounds_since_last_edge": result.state.rounds_since_last_edge,
        "skill_logits": result.state.skill_logits,
        "skipped": result.skipped,
        "bare_request_profiles": {
            k: {"alpha": v.alpha, "beta": v.beta}
            for k, v in result.state.bare_request_profiles.items()
        },
        "skill_profiles": {
            k: {"alpha": v.alpha, "beta": v.beta}
            for k, v in result.state.skill_profiles.items()
        },
        "tried_combinations": _serialize_tried_combinations(result.state.tried_combinations),
        "rng_state": result.rng_state,
        "np_rng_state": _serialize_np_rng_state(result.np_rng_state),
    }
    with open(state_path, "w") as f:
        json.dump(state_dict, f, indent=2, ensure_ascii=False)

    logger.info("Search state saved to %s", state_path)


def _save_metadata(out: Path, stats: dict, config: dict | None) -> None:
    """Save experiment metadata (environment, timing, config summary)."""
    import torch
    import transformers

    env = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda or "unknown"
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": stats.get("total_time_seconds"),
        "total_time_human": stats.get("total_time_human"),
        "seed": stats.get("seed"),
        "environment": env,
    }
    if config:
        metadata["target_model"] = config.get("target_model", {}).get("name", "unknown")
        metadata["category"] = config.get("bare_requests", {}).get("category", "unknown")

    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Metadata saved to %s", out / "metadata.json")


# ---------------------------------------------------------------------------
# Resume / continue from previous run
# ---------------------------------------------------------------------------

def load_search_state(results_dir: str) -> dict:
    """Load saved search state from a previous run for resumption.

    Args:
        results_dir: Path to the results directory of a previous run.

    Returns:
        Dict with keys: 'config', 'state', 'rng_state', 'skipped',
        'round_records', 'prev_stats', 'prev_model_name'.
    """
    out = Path(results_dir)
    if not out.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # --- Load config ---
    config_path = out / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {results_dir}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # --- Load search_state.json ---
    state_path = out / "search_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"search_state.json not found in {results_dir}")
    with open(state_path) as f:
        raw = json.load(f)

    # Reconstruct bandit profiles (alpha/beta format)
    # Also supports legacy format (reward_sum/count) for backward compatibility
    def _load_profile(v: dict) -> ucb.ArmProfile:
        if "alpha" in v:
            return ucb.ArmProfile(alpha=v["alpha"], beta=v["beta"])
        # Legacy: convert reward_sum/count to alpha/beta
        rs, cnt = v["reward_sum"], v["count"]
        return ucb.ArmProfile(alpha=rs, beta=cnt - rs)

    bare_request_profiles = {
        k: _load_profile(v)
        for k, v in raw["bare_request_profiles"].items()
    }
    skill_profiles = {
        k: _load_profile(v)
        for k, v in raw["skill_profiles"].items()
    }

    # Reconstruct tried_combinations.
    # New format: {q_key: [[main_list, fu_list, n_tried], ...]}.
    # Legacy format: {q_key: [[skill_name, ...], ...]} (auto-migrated).
    tried_combinations = _deserialize_tried_combinations(
        raw.get("tried_combinations", {}),
    )

    # --- Load edge cases ---
    edge_cases = []
    ec_path = out / "edge_cases.jsonl"
    if ec_path.exists():
        with open(ec_path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    d.setdefault("follow_ups", [])  # backward compat
                    d.setdefault("metadata", {})    # backward compat
                    edge_cases.append(EdgeCase(**d))

    # --- Reconstruct SearchState ---
    state = SearchState(
        bare_request_profiles=bare_request_profiles,
        skill_profiles=skill_profiles,
        skill_logits=raw["skill_logits"],
        tried_combinations=tried_combinations,
        edge_cases=edge_cases,
        round_num=raw["round_num"],
        rounds_since_last_edge=raw.get("rounds_since_last_edge", 0),
    )

    # --- RNG state (may not exist in older saves) ---
    rng_state = None
    if "rng_state" in raw and raw["rng_state"] is not None:
        rs = raw["rng_state"]
        # JSON stores tuples as lists; random.Random.setstate() needs:
        # (version: int, internalstate: tuple[int, ...], gauss_next: float)
        rng_state = (rs[0], tuple(rs[1]), rs[2])
    else:
        logger.warning(
            "No rng_state in %s — trial seeds will diverge from original run",
            state_path,
        )

    # --- Load round records ---
    round_records = []
    details_path = out / "trial_details.jsonl"
    if details_path.exists():
        with open(details_path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    round_records.append(RoundRecord(**d))

    # --- Load previous stats ---
    prev_stats = {}
    stats_path = out / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            prev_stats = json.load(f)

    # --- Load metadata for model name validation ---
    prev_model_name = None
    meta_path = out / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            prev_model_name = meta.get("target_model")

    logger.info(
        "Loaded search state from %s: round=%d, edge_cases=%d, "
        "round_records=%d, rng_state=%s",
        results_dir, state.round_num, len(edge_cases),
        len(round_records), "present" if rng_state else "missing",
    )

    # --- Numpy RNG state (may not exist in older saves) ---
    np_rng_state = _deserialize_np_rng_state(raw.get("np_rng_state"))
    if np_rng_state is None:
        logger.warning(
            "No np_rng_state in %s — skill count sampling seeds will diverge",
            state_path,
        )

    return {
        "config": config,
        "state": state,
        "rng_state": rng_state,
        "np_rng_state": np_rng_state,
        "skipped": raw.get("skipped", 0),
        "round_records": round_records,
        "prev_stats": prev_stats,
        "prev_model_name": prev_model_name,
    }
