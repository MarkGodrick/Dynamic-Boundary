"""Uniform-random selector: the naive baseline SGBS is compared against.

`sample_uniform` is a drop-in sibling of `search.sgbs._sample`. It mirrors that
function step for step and replaces exactly one thing: the *scores* that drive
selection. Where SGBS draws them from the Beta posteriors (bare request, skill)
and the skill-count logit distribution, this module draws them uniformly.
Everything downstream is the same shared code — compatibility backtracking via
`select_skills`, the `tried_combinations` dedup budget, follow-up sampling, and
(outside the sampler) the composer, progressive evaluator, and stopping rule.

Two consequences worth stating explicitly:

1. The sampler never *reads* bandit state, but `sgbs._process_result` still
   *writes* it. The posteriors and skill logits in `search_state.json` are
   therefore live but unused under this selector; keeping the write path shared
   is what makes the two arms produce byte-compatible artifacts (snapshots,
   resume, trial_details schema), so the analysis code can parse both with one
   implementation. The run's selector is recorded in its saved `config.yaml`.

2. The signature is kept identical to `_sample` — including parameters this
   selector has no use for (`c`, `skill_method`, `top_k`, `top_p`, `sc_alpha`,
   `sc_beta`, `skip_penalty`). Callers dispatch between the two by name alone,
   so they must stay interchangeable; `tests/test_uniform_selector.py` asserts
   the parity.
"""

from typing import Iterable

import numpy as np

from logger import get_logger

from .sgbs import BareRequest, SampledTask, SearchState, _sample_follow_ups
from .skill_selector import Skill, select_skills

logger = get_logger(__name__)


def uniform_scores(
    keys: Iterable[str],
    np_rng: np.random.Generator,
) -> dict[str, float]:
    """Assign every key an independent U(0, 1) score.

    Taking the argmax of these scores is a uniform draw over the keys, and
    ranking by them is a uniform random permutation — which is what makes
    `select_skills` (a greedy scan down the ranked list) degrade into "shuffle,
    then greedily keep compatible skills".
    """
    return {key: float(np_rng.random()) for key in keys}


def count_support(
    skill_logits: list[float],
    mis_size: int,
    lower_bound: int,
) -> tuple[int, int]:
    """Inclusive [low, high] range of skill counts SGBS can reach.

    `skill_count.sample_count` samples inside a dynamic window
    `[lower_bound, min(peak + 3, mis_size, len(logits) - 1)]`; the widest that
    window ever gets — i.e. the support of the SGBS count distribution — is
    `[lower_bound, min(mis_size, len(logits) - 1)]`. The uniform selector draws
    over that same support so the two arms explore identical count ranges.
    """
    high = min(mis_size, len(skill_logits) - 1)
    if high < lower_bound:
        raise ValueError(
            f"empty skill-count support: lower_bound={lower_bound} > "
            f"min(mis_size={mis_size}, max_count-1={len(skill_logits) - 1})"
        )
    return lower_bound, high


def sample_uniform(
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
) -> SampledTask | None:
    """Uniformly sample bare request + skill count + skill subset for one round.

    Signature-compatible with `sgbs._sample`; `c`, `skill_method`, `top_k`,
    `top_p`, `sc_alpha`, `sc_beta` and `skip_penalty` are accepted for that
    parity and deliberately unused (they parameterise bandit behaviour that
    this selector does not have).

    Returns:
        SampledTask — ready for compose.
        None        — all retries exhausted (skip), same as `_sample`.

    Mutates state: `tried_combinations` only. No bandit posterior or logit is
    read or written here.
    """
    del c, skill_method, top_k, top_p, sc_alpha, sc_beta, skip_penalty  # parity

    br_scores = uniform_scores(state.bare_request_profiles, np_rng)
    if not br_scores:
        return None
    q_key = max(br_scores, key=br_scores.get)
    q = br_lookup[q_key]

    low, high = count_support(state.skill_logits, mis_size, lower_bound)

    for _retry in range(max_skip_retries):
        count = int(np_rng.integers(low, high + 1))

        if count == 0:
            selected_skills: list[Skill] = []
        else:
            selected_skills = select_skills(
                count, uniform_scores(skills, np_rng), skills,
            )
            if selected_skills is None:
                # No compatible combo of this size; redraw. SGBS instead
                # pushes the count logit down ("no_combo"); the uniform arm
                # has no such memory by construction.
                continue

        # Sample follow-up skills first so they participate in the dedup key
        # (mirrors _sample; no-op for single-turn categories).
        fu_skills = _sample_follow_ups(
            follow_up_skill_pool, max_follow_up_turns, np_rng,
        )
        fu_part = tuple(s.name for s in (fu_skills or []))
        combo_key = (frozenset(s.name for s in selected_skills), fu_part)
        if state.tried_combinations[q_key][combo_key] >= max_compositions:
            continue
        state.tried_combinations[q_key][combo_key] += 1

        return SampledTask(
            round_num=round_num, q_key=q_key, q=q,
            count=count, selected_skills=selected_skills,
            fu_skills=fu_skills,
        )

    return None  # all retries exhausted
