"""Bandit logic for bare request and skill selection.

Supports two selection strategies (configurable for ablation):
- UCB1: deterministic score = r_bar + c * sqrt(ln(t) / n)
- Thompson Sampling: stochastic score = sample from Beta(alpha, beta)

Both share the same ArmProfile(alpha, beta) and update rules.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ArmProfile:
    """Mutable state for a single bandit arm.

    alpha: success pseudo-count (prior + cumulative reward)
    beta:  failure pseudo-count (prior + cumulative 1-reward + penalties)
    """
    alpha: float
    beta: float


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def thompson_scores(
    profiles: dict[str, ArmProfile],
    rng: np.random.Generator,
    exclude: set[str] | None = None,
) -> dict[str, float]:
    """Sample a score for each arm from its Beta posterior."""
    exclude = exclude or set()
    scores = {}
    for key, arm in profiles.items():
        if key in exclude:
            continue
        a = max(arm.alpha, 1e-6)
        b = max(arm.beta, 1e-6)
        scores[key] = float(rng.beta(a, b))
    return scores


def ucb_scores(
    profiles: dict[str, ArmProfile],
    total_rounds: int,
    c: float = 1.0,
    exclude: set[str] | None = None,
) -> dict[str, float]:
    """Compute deterministic UCB1 score for each arm."""
    exclude = exclude or set()
    scores = {}
    for key, arm in profiles.items():
        if key in exclude:
            continue
        scores[key] = ucb_score(arm, total_rounds, c)
    return scores


def ucb_score(arm: ArmProfile, total_rounds: int, c: float = 1.0) -> float:
    """Compute UCB1 score: r_bar + c * sqrt(ln(t) / n)."""
    n = arm.alpha + arm.beta
    r_bar = arm.alpha / n
    if total_rounds <= 0:
        return r_bar
    exploration = c * math.sqrt(math.log(total_rounds) / n)
    return r_bar + exploration


def ucb_select(
    profiles: dict[str, ArmProfile],
    total_rounds: int,
    c: float = 1.0,
    exclude: set[str] | None = None,
) -> str | None:
    """Select the arm with the highest UCB score.

    Returns None if all arms are excluded.
    """
    scores = ucb_scores(profiles, total_rounds, c, exclude)
    if not scores:
        return None
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Reward and update
# ---------------------------------------------------------------------------

def compute_reward(p_hat: float) -> float:
    """Boundary-seeking reward: peaks at 1.0 when p_hat = 0.5, 0.0 at extremes."""
    return 1.0 - 2.0 * abs(p_hat - 0.5)


def compute_individual_reward(reward: float, skill_count: int) -> float:
    """Per-skill credit for a K-skill combination under aggregate feedback.

    Combinatorial bandit with aggregate feedback: a single joint reward r is
    observed for a K-hot action, but K bandit arms (one per skill) need to be
    updated. Under an additive linear model E[r | a] = (1/K) Σ_{i ∈ a} π_i,
    the unbiased per-skill posterior update is r/K; this is also the unique
    choice that conserves total posterior mass across the K participating
    arms (each round adds α + β = 1 in aggregate, regardless of K).
    """
    return reward / max(skill_count, 1)


def create_profiles(
    keys: list[str], prior_alpha: float = 0.3, prior_beta: float = 0.7,
) -> dict[str, ArmProfile]:
    """Create bandit profiles with Beta prior initialization."""
    return {key: ArmProfile(alpha=prior_alpha, beta=prior_beta) for key in keys}


def update_profile(profile: ArmProfile, reward: float) -> None:
    """Update an arm profile after evaluation.

    reward in [0, 1]: alpha gets reward, beta gets (1 - reward).
    """
    profile.alpha += reward
    profile.beta += (1.0 - reward)


def penalize_profile(profile: ArmProfile, penalty: float) -> None:
    """Penalize an arm for skip (adds to failure pseudo-count only)."""
    profile.beta += penalty
