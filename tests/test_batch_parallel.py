"""Tests for batch parallel evaluation.

Verifies:
1. Evaluator.evaluate() with pre-generated seeds produces same results
2. _sample returns SampledTask/BatchTask/None; _compose_task returns BatchTask/None
3. _process_result updates state correctly
4. Batch loop with batch_size=1 matches sequential behavior
"""

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search import ucb
from search.evaluator import Evaluator, EvalResult
from search.sgbs import (
    BatchTask, SampledTask, BareRequest, SearchState, EdgeCase, RoundRecord,
    _sample, _compose_task, _process_result,
)
from search.skill_selector import Skill


# ---------------------------------------------------------------------------
# 1. Evaluator with pre-generated seeds
# ---------------------------------------------------------------------------

class TestEvaluatorSeeds:
    def _make_evaluator(self, judge_fn, seed=42):
        target = MagicMock()
        target.generate = MagicMock(return_value="mock response")
        judge = MagicMock()
        judge.judge = judge_fn
        return Evaluator(
            target_model=target, judge=judge,
            coarse_n=3, fine_n=2,
            coarse_range=(0.1, 0.9), fine_range=(0.3, 0.7),
            seed=seed,
        )

    def test_seeds_none_uses_internal_rng(self):
        """When seeds=None, evaluate uses self.rng (original behavior)."""
        call_count = [0]
        def mock_judge(messages, metadata=None):
            call_count[0] += 1
            return call_count[0] % 2 == 0
        evaluator = self._make_evaluator(mock_judge)
        result = evaluator.evaluate("test", seeds=None)
        assert isinstance(result, EvalResult)
        assert result.total_trials >= 3

    def test_pregenerated_seeds_used(self):
        """Pre-generated seeds are passed to target_model.generate."""
        seeds_received = []
        target = MagicMock()
        def capture_generate(prompt, seed=None):
            seeds_received.append(seed)
            return "response"
        target.generate = capture_generate

        judge = MagicMock()
        judge.judge = MagicMock(return_value=True)

        evaluator = Evaluator(
            target_model=target, judge=judge,
            coarse_n=3, fine_n=2,
            coarse_range=(0.0, 1.0), fine_range=(0.0, 1.0),
            seed=42,
        )
        pre_seeds = [100, 200, 300, 400, 500]
        evaluator.evaluate("test", seeds=pre_seeds)
        # All 5 seeds should have been used (3 coarse + 2 fine, since all pass)
        assert seeds_received == [100, 200, 300, 400, 500]

    def test_pregenerated_seeds_coarse_reject(self):
        """When coarse rejects, only coarse seeds are consumed."""
        seeds_received = []
        target = MagicMock()
        def capture_generate(prompt, seed=None):
            seeds_received.append(seed)
            return "response"
        target.generate = capture_generate

        judge = MagicMock()
        judge.judge = MagicMock(return_value=False)  # all fail → p_hat=0.0

        evaluator = Evaluator(
            target_model=target, judge=judge,
            coarse_n=3, fine_n=2,
            coarse_range=(0.1, 0.9), fine_range=(0.3, 0.7),
            seed=42,
        )
        pre_seeds = [100, 200, 300, 400, 500]
        result = evaluator.evaluate("test", seeds=pre_seeds)
        # p_hat=0.0 → outside coarse_range → stage 1 only → 3 seeds used
        assert result.stage == 1
        assert len(seeds_received) == 3
        assert seeds_received == [100, 200, 300]

    def test_deterministic_with_same_seeds(self):
        """Same seeds → same results regardless of RNG state."""
        call_count = [0]
        def deterministic_judge(prompt, response, metadata=None):
            call_count[0] += 1
            return call_count[0] % 3 != 0  # deterministic pattern

        e1 = self._make_evaluator(deterministic_judge, seed=1)
        call_count[0] = 0
        r1 = e1.evaluate("test", seeds=[10, 20, 30, 40, 50])

        e2 = self._make_evaluator(deterministic_judge, seed=999)
        call_count[0] = 0
        r2 = e2.evaluate("test", seeds=[10, 20, 30, 40, 50])

        assert r1.p_hat == r2.p_hat
        assert r1.trial_results == r2.trial_results


# ---------------------------------------------------------------------------
# 2. _sample and _compose_task
# ---------------------------------------------------------------------------

class TestSampleAndCompose:
    def _make_state(self):
        return SearchState(
            bare_request_profiles=ucb.create_profiles(["q1", "q2"]),
            skill_profiles=ucb.create_profiles(["s1", "s2", "s3"]),
            skill_logits=[5.0, 3.0, 1.0],  # max_count=3
            tried_combinations=defaultdict(Counter),
            edge_cases=[],
        )

    def _make_skills(self):
        return {
            "s1": Skill(name="s1", description="", tags={"a"}),
            "s2": Skill(name="s2", description="", tags={"b"}),
            "s3": Skill(name="s3", description="", tags={"c"}),
        }

    def _make_composer(self):
        composer = MagicMock()
        composer.compose = MagicMock(return_value=("composed prompt", {"key": "val"}))
        return composer

    def test_sample_returns_sampled_or_batch_task(self):
        state = self._make_state()
        skills = self._make_skills()
        np_rng = np.random.default_rng(42)

        result = _sample(
            state=state, round_num=1, br_lookup={
                "q1": BareRequest("q1"), "q2": BareRequest("q2"),
            },
            skills=skills, np_rng=np_rng,
            c=1.0, skill_method="thompson",
            max_skip_retries=5, skip_penalty=0.1,
            mis_size=8, top_k=3, top_p=0.9,
            lower_bound=0, sc_alpha=0.05, sc_beta=0.1,
        )
        assert result is not None
        assert isinstance(result, SampledTask)
        # count=0 (rephrase-only) and count>0 (skill-composed) are both SampledTask
        if result.count > 0:
            assert len(result.selected_skills) > 0

    def test_compose_task_produces_batch_task(self):
        """_compose_task wraps SampledTask into BatchTask."""
        composer = self._make_composer()
        sampled = SampledTask(
            round_num=1, q_key="q1", q=BareRequest("q1", {"m": 1}),
            count=1,
            selected_skills=[Skill(name="s1", description="", tags={"a"})],
            fu_skills=None,
        )
        task = _compose_task(sampled, composer)
        assert task is not None
        assert isinstance(task, BatchTask)
        assert task.prompt == "composed prompt"
        assert task.metadata["key"] == "val"
        assert task.metadata["m"] == 1  # bare_request metadata merged

    def test_compose_task_returns_none_on_failure(self):
        """_compose_task returns None when composer raises."""
        composer = MagicMock()
        composer.compose = MagicMock(side_effect=RuntimeError("API error"))
        sampled = SampledTask(
            round_num=1, q_key="q1", q=BareRequest("q1"),
            count=1,
            selected_skills=[Skill(name="s1", description="", tags={"a"})],
            fu_skills=None,
        )
        task = _compose_task(sampled, composer)
        assert task is None

    def test_sample_and_compose_end_to_end(self):
        """Full flow: _sample → _compose_task → BatchTask."""
        state = self._make_state()
        skills = self._make_skills()
        composer = self._make_composer()
        np_rng = np.random.default_rng(42)

        result = _sample(
            state=state, round_num=1, br_lookup={
                "q1": BareRequest("q1"), "q2": BareRequest("q2"),
            },
            skills=skills, np_rng=np_rng,
            c=1.0, skill_method="thompson",
            max_skip_retries=5, skip_penalty=0.1,
            mis_size=8, top_k=3, top_p=0.9,
            lower_bound=0, sc_alpha=0.05, sc_beta=0.1,
        )
        assert result is not None
        if isinstance(result, SampledTask):
            task = _compose_task(result, composer)
            assert isinstance(task, BatchTask)
        else:
            task = result
        assert task.round_num == 1
        assert task.q_key in ("q1", "q2")

    def test_returns_none_when_all_combos_tried(self):
        """All retries exhausted → returns None (skip)."""
        state = self._make_state()
        skills = self._make_skills()
        np_rng = np.random.default_rng(42)

        # Pre-fill tried_combinations for all possible combos.
        # New combo_key shape: (frozenset(main_skills), tuple(fu_skill_names)).
        # With no fu pool, fu tuple is always (). Saturate count >= max_compositions.
        for q_key in ("q1", "q2"):
            state.tried_combinations[q_key][(frozenset(), ())] = 1
            for s in skills:
                state.tried_combinations[q_key][(frozenset([s]), ())] = 1
            for i, s1 in enumerate(skills):
                for s2 in list(skills)[i+1:]:
                    state.tried_combinations[q_key][(frozenset([s1, s2]), ())] = 1

        result = _sample(
            state=state, round_num=1, br_lookup={
                "q1": BareRequest("q1"), "q2": BareRequest("q2"),
            },
            skills=skills, np_rng=np_rng,
            c=1.0, skill_method="thompson",
            max_skip_retries=5, skip_penalty=0.1,
            mis_size=8, top_k=3, top_p=0.9,
            lower_bound=0, sc_alpha=0.05, sc_beta=0.1,
            max_compositions=1,
        )
        assert result is None


# ---------------------------------------------------------------------------
# 3. _process_result
# ---------------------------------------------------------------------------

class TestProcessResult:
    def test_edge_case_recorded(self):
        state = SearchState(
            bare_request_profiles=ucb.create_profiles(["q1"]),
            skill_profiles=ucb.create_profiles(["s1"]),
            skill_logits=[5.0, 3.0],
            tried_combinations=defaultdict(Counter),
            edge_cases=[],
        )
        task = BatchTask(
            round_num=1, q_key="q1",
            q=BareRequest("q1"), count=1,
            selected_skills=[Skill(name="s1", description="")],
            prompt="test", metadata={}, seeds=[],
        )
        eval_result = EvalResult(
            p_hat=0.5, pass_count=5, total_trials=10,
            stage=2, is_edge_case=True,
        )
        records = []
        _process_result(
            state=state, task=task, eval_result=eval_result,
            round_records=records,
            sc_alpha=0.05, sc_beta=0.1, lower_bound=0, out_dir=None,
        )
        assert len(state.edge_cases) == 1
        assert state.edge_cases[0].p_hat == 0.5
        assert state.rounds_since_last_edge == 0
        assert len(records) == 1

    def test_non_edge_increments_counter(self):
        state = SearchState(
            bare_request_profiles=ucb.create_profiles(["q1"]),
            skill_profiles=ucb.create_profiles(["s1"]),
            skill_logits=[5.0, 3.0],
            tried_combinations=defaultdict(Counter),
            edge_cases=[],
            rounds_since_last_edge=5,
        )
        task = BatchTask(
            round_num=1, q_key="q1",
            q=BareRequest("q1"), count=0, selected_skills=[],
            prompt="test", metadata={}, seeds=[],
        )
        eval_result = EvalResult(
            p_hat=0.0, pass_count=0, total_trials=10,
            stage=1, is_edge_case=False,
        )
        records = []
        _process_result(
            state=state, task=task, eval_result=eval_result,
            round_records=records,
            sc_alpha=0.05, sc_beta=0.1, lower_bound=0, out_dir=None,
        )
        assert len(state.edge_cases) == 0
        assert state.rounds_since_last_edge == 6

    def test_profiles_updated(self):
        state = SearchState(
            bare_request_profiles=ucb.create_profiles(["q1"]),
            skill_profiles=ucb.create_profiles(["s1"]),
            skill_logits=[5.0, 3.0],
            tried_combinations=defaultdict(Counter),
            edge_cases=[],
        )
        old_alpha = state.skill_profiles["s1"].alpha
        task = BatchTask(
            round_num=1, q_key="q1",
            q=BareRequest("q1"), count=1,
            selected_skills=[Skill(name="s1", description="")],
            prompt="test", metadata={}, seeds=[],
        )
        eval_result = EvalResult(
            p_hat=0.5, pass_count=5, total_trials=10,
            stage=2, is_edge_case=True,
        )
        _process_result(
            state=state, task=task, eval_result=eval_result,
            round_records=[],
            sc_alpha=0.05, sc_beta=0.1, lower_bound=0, out_dir=None,
        )
        # reward=1.0 for p_hat=0.5, individual=1.0/1=1.0 (linear credit, k=1)
        assert state.skill_profiles["s1"].alpha == old_alpha + 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
