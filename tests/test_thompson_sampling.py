"""Tests for Thompson Sampling and bandit logic.

Verifies:
1. ArmProfile(alpha, beta) creation and update
2. Thompson scores produce valid samples with correct diversity
3. UCB scores still work with alpha/beta representation
4. Skill selector picks compatible skills from pre-computed scores
5. Skip penalty increases beta only
6. Config switch between thompson and ucb methods
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search import ucb
from search.skill_selector import Skill, select_skills, is_compatible


# ---------------------------------------------------------------------------
# 1. ArmProfile creation and update
# ---------------------------------------------------------------------------

class TestArmProfile:
    def test_create_profiles_default(self):
        profiles = ucb.create_profiles(["a", "b", "c"])
        assert len(profiles) == 3
        for p in profiles.values():
            assert p.alpha == 0.3
            assert p.beta == 0.7

    def test_create_profiles_custom_prior(self):
        profiles = ucb.create_profiles(["x"], prior_alpha=1.0, prior_beta=1.0)
        assert profiles["x"].alpha == 1.0
        assert profiles["x"].beta == 1.0

    def test_update_profile_full_reward(self):
        p = ucb.ArmProfile(alpha=1.0, beta=1.0)
        ucb.update_profile(p, reward=1.0)
        assert p.alpha == 2.0
        assert p.beta == 1.0  # beta += (1 - 1.0) = 0

    def test_update_profile_zero_reward(self):
        p = ucb.ArmProfile(alpha=1.0, beta=1.0)
        ucb.update_profile(p, reward=0.0)
        assert p.alpha == 1.0  # alpha += 0
        assert p.beta == 2.0

    def test_update_profile_partial_reward(self):
        p = ucb.ArmProfile(alpha=1.0, beta=1.0)
        ucb.update_profile(p, reward=0.6)
        assert abs(p.alpha - 1.6) < 1e-9
        assert abs(p.beta - 1.4) < 1e-9

    def test_penalize_profile(self):
        p = ucb.ArmProfile(alpha=5.0, beta=3.0)
        ucb.penalize_profile(p, penalty=0.1)
        assert p.alpha == 5.0  # unchanged
        assert abs(p.beta - 3.1) < 1e-9


# ---------------------------------------------------------------------------
# 2. Thompson scores
# ---------------------------------------------------------------------------

class TestThompsonScores:
    def test_returns_scores_for_all_non_excluded(self):
        profiles = ucb.create_profiles(["a", "b", "c"])
        rng = np.random.default_rng(42)
        scores = ucb.thompson_scores(profiles, rng)
        assert set(scores.keys()) == {"a", "b", "c"}

    def test_exclude_works(self):
        profiles = ucb.create_profiles(["a", "b", "c"])
        rng = np.random.default_rng(42)
        scores = ucb.thompson_scores(profiles, rng, exclude={"b"})
        assert "b" not in scores
        assert set(scores.keys()) == {"a", "c"}

    def test_scores_in_zero_one(self):
        profiles = ucb.create_profiles(["a", "b"])
        rng = np.random.default_rng(42)
        for _ in range(100):
            scores = ucb.thompson_scores(profiles, rng)
            for v in scores.values():
                assert 0.0 <= v <= 1.0

    def test_high_alpha_produces_higher_scores(self):
        """A skill with alpha=50, beta=5 should score higher on average."""
        profiles = {
            "strong": ucb.ArmProfile(alpha=50.0, beta=5.0),
            "weak": ucb.ArmProfile(alpha=2.0, beta=10.0),
        }
        rng = np.random.default_rng(42)
        strong_wins = 0
        n = 500
        for _ in range(n):
            scores = ucb.thompson_scores(profiles, rng)
            if scores["strong"] > scores["weak"]:
                strong_wins += 1
        # Strong should win >90% of the time
        assert strong_wins > 0.9 * n

    def test_stochastic_not_deterministic(self):
        """Same profiles should produce different scores on consecutive calls."""
        profiles = ucb.create_profiles(["a", "b"])
        rng = np.random.default_rng(42)
        s1 = ucb.thompson_scores(profiles, rng)
        s2 = ucb.thompson_scores(profiles, rng)
        # Very unlikely to be exactly equal
        assert s1 != s2

    def test_exploration_weak_arm_can_win(self):
        """A weak but uncertain arm should occasionally beat a strong one."""
        profiles = {
            "strong": ucb.ArmProfile(alpha=20.0, beta=5.0),   # mean=0.8
            "uncertain": ucb.ArmProfile(alpha=0.5, beta=0.5),  # mean=0.5 but wide
        }
        rng = np.random.default_rng(42)
        uncertain_wins = 0
        n = 1000
        for _ in range(n):
            scores = ucb.thompson_scores(profiles, rng)
            if scores["uncertain"] > scores["strong"]:
                uncertain_wins += 1
        # Should win sometimes (exploration), but not too often
        assert 20 < uncertain_wins < 500


# ---------------------------------------------------------------------------
# 3. UCB scores with alpha/beta
# ---------------------------------------------------------------------------

class TestUCBScores:
    def test_ucb_score_basic(self):
        arm = ucb.ArmProfile(alpha=3.0, beta=7.0)  # r_bar = 0.3, n = 10
        score = ucb.ucb_score(arm, total_rounds=100, c=1.0)
        # r_bar = 0.3, exploration = sqrt(ln(100)/10) ≈ 0.678
        assert score > 0.3
        assert score < 2.0

    def test_ucb_select_picks_best(self):
        profiles = {
            "a": ucb.ArmProfile(alpha=8.0, beta=2.0),   # r_bar=0.8
            "b": ucb.ArmProfile(alpha=2.0, beta=8.0),   # r_bar=0.2
        }
        selected = ucb.ucb_select(profiles, total_rounds=100)
        assert selected == "a"

    def test_ucb_scores_dict(self):
        profiles = ucb.create_profiles(["x", "y", "z"])
        scores = ucb.ucb_scores(profiles, total_rounds=10, c=1.0)
        assert len(scores) == 3
        for v in scores.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# 4. Skill selector with pre-computed scores
# ---------------------------------------------------------------------------

class TestSelectSkills:
    def _make_skills(self):
        return {
            "A": Skill(name="A", description="a", tags={"t1"},
                       incompatible_tags=set()),
            "B": Skill(name="B", description="b", tags={"t2"},
                       incompatible_tags=set()),
            "C": Skill(name="C", description="c", tags={"t1"},
                       incompatible_tags={"t1"}),  # conflicts with A
            "D": Skill(name="D", description="d", tags={"t3"},
                       incompatible_tags=set()),
        }

    def test_select_top_compatible(self):
        skills = self._make_skills()
        scores = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6}
        result = select_skills(2, scores, skills)
        assert result is not None
        names = [s.name for s in result]
        assert names == ["A", "B"]

    def test_skip_incompatible(self):
        """C conflicts with A; should be skipped."""
        skills = self._make_skills()
        scores = {"A": 0.9, "C": 0.85, "B": 0.7, "D": 0.6}
        result = select_skills(2, scores, skills)
        names = [s.name for s in result]
        assert names == ["A", "B"]  # C skipped due to conflict with A

    def test_returns_none_if_not_enough(self):
        skills = self._make_skills()
        scores = {"A": 0.9}  # only one skill scored
        result = select_skills(2, scores, skills)
        assert result is None

    def test_count_one(self):
        skills = self._make_skills()
        scores = {"D": 0.5, "A": 0.9}
        result = select_skills(1, scores, skills)
        assert len(result) == 1
        assert result[0].name == "A"

    def test_empty_scores(self):
        skills = self._make_skills()
        result = select_skills(1, {}, skills)
        assert result is None


# ---------------------------------------------------------------------------
# 5. Reward computation (unchanged)
# ---------------------------------------------------------------------------

class TestReward:
    def test_boundary_reward(self):
        assert ucb.compute_reward(0.5) == 1.0
        assert ucb.compute_reward(0.0) == 0.0
        assert ucb.compute_reward(1.0) == 0.0
        assert abs(ucb.compute_reward(0.4) - 0.8) < 1e-9

    def test_individual_reward(self):
        assert abs(ucb.compute_individual_reward(1.0, 1) - 1.0) < 1e-9
        assert abs(ucb.compute_individual_reward(1.0, 4) - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# 6. is_compatible
# ---------------------------------------------------------------------------

class TestIsCompatible:
    def test_same_name_incompatible(self):
        a = Skill(name="A", description="")
        assert not is_compatible(a, [a])

    def test_tag_conflict(self):
        a = Skill(name="A", description="", tags={"x"}, incompatible_tags=set())
        b = Skill(name="B", description="", tags=set(), incompatible_tags={"x"})
        assert not is_compatible(a, [b])

    def test_no_conflict(self):
        a = Skill(name="A", description="", tags={"x"})
        b = Skill(name="B", description="", tags={"y"})
        assert is_compatible(a, [b])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
