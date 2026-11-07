"""Tests for the unified metadata pipeline.

Verifies:
1. Skill.extra populated correctly by load_skills
2. Composer returns (prompt, metadata) uniformly
3. ConstraintJudge is stateless (reads from metadata, no leakage)
4. Evaluator passes metadata through to judge
5. SGBS main loop merges bare_request.metadata + compose metadata
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search.skill_selector import Skill, load_skills
from search.sgbs import BareRequest
from search.composer import Composer, TemplateComposer
from search.evaluator import Evaluator, EvalResult
from judges.constraint_judge import ConstraintJudge
from judges.keyword_judge import KeywordJudge


def _msgs(prompt: str, response: str) -> list[dict]:
    """Helper: build a 2-element messages list for single-turn judge calls."""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


# ---------------------------------------------------------------------------
# 1. Skill.extra
# ---------------------------------------------------------------------------

class TestSkillExtra:
    def test_custom_format_loads_extra(self, tmp_path):
        """Non-core keys in skill dictionary go into skill.extra."""
        skills_data = [
            {
                "name": "keywords:existence",
                "description": "Must contain keywords",
                "tags": ["keywords"],
                "category": "constrained_completion",
                "incompatible_tags": [],
                "param_keys": ["keywords"],
                "param_pool": {"keywords": [["AI", "ML"], ["cat", "dog"]]},
                "count_in_ifeval": 42,
            }
        ]
        path = tmp_path / "skills.json"
        path.write_text(json.dumps(skills_data))

        skills = load_skills(str(path))
        s = skills["keywords:existence"]

        assert s.name == "keywords:existence"
        assert s.description == "Must contain keywords"
        assert "keywords" in s.tags
        assert s.extra["param_keys"] == ["keywords"]
        assert s.extra["param_pool"]["keywords"] == [["AI", "ML"], ["cat", "dog"]]
        assert s.extra["count_in_ifeval"] == 42
        # Core keys should NOT be in extra
        assert "name" not in s.extra
        assert "description" not in s.extra
        assert "tags" not in s.extra

    def test_harmful_refusal_format_loads_extra(self, tmp_path):
        """Harmful refusal format puts non-core fields into extra."""
        skills_data = [
            {
                "name": "narrative_reframing",
                "description": "Reframe narratives",
                "tags": ["reframing"],
                "category": "",
                "incompatible_tags": [],
                "example": "As part of a story...",
                "source": "dejavu",
                "source_meta": {"score": 0.9},
            }
        ]
        path = tmp_path / "skills.json"
        path.write_text(json.dumps(skills_data))

        skills = load_skills(str(path))
        s = skills["narrative_reframing"]

        assert s.name == "narrative_reframing"
        assert s.extra["example"] == "As part of a story..."
        assert s.extra["source"] == "dejavu"
        assert s.extra["source_meta"]["score"] == 0.9


# ---------------------------------------------------------------------------
# 2. Composer unified interface
# ---------------------------------------------------------------------------

class TestComposerInterface:
    def test_template_composer_returns_tuple(self):
        """TemplateComposer.compose() returns (prompt, metadata) tuple."""
        composer = TemplateComposer(seed=42)

        # With no skills: returns bare request + empty metadata
        prompt, meta = composer.compose("Write a poem", [])
        assert prompt == "Write a poem"
        assert meta == {}

    def test_template_composer_with_skills(self):
        """TemplateComposer with skills returns constraints in metadata."""
        composer = TemplateComposer(seed=42)

        skill = Skill(
            name="punctuation:no_comma",
            description="No commas allowed",
            extra={
                "param_keys": [],
                "param_pool": {},
            },
        )

        prompt, meta = composer.compose("Write a poem", [skill])
        assert isinstance(prompt, str)
        assert "constraints" in meta
        assert len(meta["constraints"]) == 1
        assert meta["constraints"][0]["type"] == "punctuation:no_comma"

    def test_template_composer_multi_skill(self):
        """Multiple skills produce multiple constraints in metadata."""
        composer = TemplateComposer(seed=42)

        skills = [
            Skill(name="punctuation:no_comma", description="No commas",
                  extra={"param_keys": [], "param_pool": {}}),
            Skill(name="change_case:english_lowercase", description="All lowercase",
                  extra={"param_keys": [], "param_pool": {}}),
        ]

        prompt, meta = composer.compose("Write a poem", skills)
        assert len(meta["constraints"]) == 2
        types = {c["type"] for c in meta["constraints"]}
        assert types == {"punctuation:no_comma", "change_case:english_lowercase"}


# ---------------------------------------------------------------------------
# 3. ConstraintJudge is stateless
# ---------------------------------------------------------------------------

class TestConstraintJudgeStateless:
    def test_no_metadata_returns_true(self):
        """With no metadata (count=0), judge returns True."""
        judge = ConstraintJudge()
        assert judge.judge(_msgs("prompt", "any response")) is True
        assert judge.judge(_msgs("prompt", "any response"), None) is True
        assert judge.judge(_msgs("prompt", "any response"), {}) is True

    def test_empty_constraints_returns_true(self):
        """Empty constraints list → True."""
        judge = ConstraintJudge()
        assert judge.judge(_msgs("p", "r"), {"constraints": []}) is True

    def test_reads_from_metadata(self):
        """Judge reads constraints from metadata, not internal state."""
        judge = ConstraintJudge()

        meta_comma = {"constraints": [{"type": "punctuation:no_comma", "kwargs": {}}]}
        meta_none = {}

        # Response with comma should fail with comma constraint
        assert judge.judge(_msgs("p", "hello, world"), meta_comma) is False
        # Same response should pass with no constraints
        assert judge.judge(_msgs("p", "hello, world"), meta_none) is True
        # And fail again with constraint
        assert judge.judge(_msgs("p", "hello, world"), meta_comma) is False

    def test_no_state_leakage_between_calls(self):
        """Constraints from one call do NOT leak into the next."""
        judge = ConstraintJudge()

        # Call 1: with no_comma constraint
        meta1 = {"constraints": [{"type": "punctuation:no_comma", "kwargs": {}}]}
        judge.judge(_msgs("p", "hello, world"), meta1)  # False

        # Call 2: no constraints — must return True regardless of call 1
        assert judge.judge(_msgs("p", "hello, world")) is True
        assert judge.judge(_msgs("p", "hello, world"), None) is True
        assert judge.judge(_msgs("p", "hello, world"), {}) is True

    def test_constraint_passes(self):
        """Response satisfying constraint returns True."""
        judge = ConstraintJudge()
        meta = {"constraints": [{"type": "punctuation:no_comma", "kwargs": {}}]}
        assert judge.judge(_msgs("p", "hello world")) is True  # no comma → pass
        assert judge.judge(_msgs("p", "hello world"), meta) is True

    def test_multiple_constraints_all_must_pass(self):
        """All constraints must pass for True."""
        judge = ConstraintJudge()
        meta = {"constraints": [
            {"type": "punctuation:no_comma", "kwargs": {}},
            {"type": "change_case:english_lowercase", "kwargs": {}},
        ]}
        # Both pass
        assert judge.judge(_msgs("p", "hello world"), meta) is True
        # Comma fails
        assert judge.judge(_msgs("p", "hello, world"), meta) is False
        # Uppercase fails
        assert judge.judge(_msgs("p", "Hello world"), meta) is False

    def test_judge_detailed_stateless(self):
        """judge_detailed also reads from metadata."""
        judge = ConstraintJudge()

        meta = {"constraints": [
            {"type": "punctuation:no_comma", "kwargs": {}},
            {"type": "change_case:english_lowercase", "kwargs": {}},
        ]}

        result = judge.judge_detailed(_msgs("p", "Hello, world"), meta)
        assert result["passed"] is False
        assert len(result["results"]) == 2

        # Without metadata: no constraints, passes
        result2 = judge.judge_detailed(_msgs("p", "Hello, world"))
        assert result2["passed"] is True
        assert result2["results"] == []


# ---------------------------------------------------------------------------
# 4. Evaluator passes metadata to judge
# ---------------------------------------------------------------------------

class TestEvaluatorMetadata:
    def _make_evaluator(self, judge_fn):
        """Create an Evaluator with mock target model and custom judge function."""
        target = MagicMock()
        target.generate = MagicMock(return_value="mock response")

        judge = MagicMock()
        judge.judge = judge_fn

        return Evaluator(
            target_model=target,
            judge=judge,
            coarse_n=3,
            fine_n=2,
            coarse_range=(0.1, 0.9),
            fine_range=(0.3, 0.7),
            seed=42,
        )

    def test_metadata_passed_to_judge(self):
        """Evaluator forwards metadata to each judge.judge() call."""
        received_metadata = []

        def mock_judge(messages, metadata=None):
            received_metadata.append(metadata)
            return True

        evaluator = self._make_evaluator(mock_judge)
        test_meta = {"constraints": [{"type": "test"}]}
        evaluator.evaluate("test prompt", metadata=test_meta)

        # All judge calls should have received the metadata
        assert len(received_metadata) >= 3  # at least coarse_n
        for m in received_metadata:
            assert m == test_meta

    def test_none_metadata_passed_through(self):
        """When metadata is None, judge receives None."""
        received_metadata = []

        def mock_judge(messages, metadata=None):
            received_metadata.append(metadata)
            return True

        evaluator = self._make_evaluator(mock_judge)
        evaluator.evaluate("test prompt")

        for m in received_metadata:
            assert m is None

    def test_metadata_does_not_affect_eval_logic(self):
        """Metadata is transparent to evaluation logic (coarse/fine ranges)."""
        call_count = [0]

        def alternating_judge(messages, metadata=None):
            call_count[0] += 1
            return call_count[0] % 2 == 0  # alternates True/False

        evaluator = self._make_evaluator(alternating_judge)
        result = evaluator.evaluate("p", metadata={"test": True})

        assert isinstance(result, EvalResult)
        assert result.total_trials >= 3


# ---------------------------------------------------------------------------
# 5. BareRequest metadata merge
# ---------------------------------------------------------------------------

class TestBareRequestMerge:
    def test_empty_metadata_merge(self):
        """BareRequest with empty metadata + empty compose → empty."""
        q = BareRequest(prompt="test", metadata={})
        compose_meta = {}
        merged = {**q.metadata, **compose_meta}
        assert merged == {}

    def test_bare_metadata_only(self):
        """Category D/E: bare_request has metadata, compose adds nothing."""
        q = BareRequest(prompt="What is the capital of France?",
                        metadata={"ground_truth": "Paris"})
        compose_meta = {}
        merged = {**q.metadata, **compose_meta}
        assert merged == {"ground_truth": "Paris"}

    def test_compose_metadata_only(self):
        """Category B: bare_request has no metadata, compose adds constraints."""
        q = BareRequest(prompt="Write a poem", metadata={})
        compose_meta = {"constraints": [{"type": "no_comma"}]}
        merged = {**q.metadata, **compose_meta}
        assert merged == {"constraints": [{"type": "no_comma"}]}

    def test_both_metadata_merge(self):
        """Both sources contribute, compose takes priority on overlap."""
        q = BareRequest(prompt="test",
                        metadata={"ground_truth": "Paris", "source": "original"})
        compose_meta = {"constraints": [{"type": "no_comma"}], "source": "composed"}
        merged = {**q.metadata, **compose_meta}
        assert merged["ground_truth"] == "Paris"
        assert merged["constraints"] == [{"type": "no_comma"}]
        assert merged["source"] == "composed"  # compose wins


# ---------------------------------------------------------------------------
# 6. KeywordJudge accepts metadata gracefully
# ---------------------------------------------------------------------------

class TestKeywordJudgeMetadata:
    def test_ignores_metadata(self):
        """KeywordJudge works the same with or without metadata."""
        judge = KeywordJudge()
        msgs = _msgs("harmful request", "I cannot help with that request.")

        assert judge.judge(msgs) is True
        assert judge.judge(msgs, None) is True
        assert judge.judge(msgs, {"constraints": []}) is True


# ---------------------------------------------------------------------------
# 7. Integration: full pipeline with ConstraintJudge
# ---------------------------------------------------------------------------

class TestIntegrationPipeline:
    def test_count_zero_gets_no_constraints(self):
        """Simulates count=0 path: bare request → empty metadata → judge returns True."""
        q = BareRequest(prompt="Write a blog post", metadata={})
        compose_meta = {}  # count=0, no composition
        metadata = {**q.metadata, **compose_meta}

        judge = ConstraintJudge()
        # Any response should pass — no constraints
        msgs = _msgs(q.prompt, "Here is a blog post with CAPS and, commas!")
        assert judge.judge(msgs, metadata) is True

    def test_count_nonzero_gets_constraints(self):
        """Simulates count>0: compose adds constraints, judge checks them."""
        q = BareRequest(prompt="Write a poem", metadata={})
        compose_meta = {"constraints": [
            {"type": "punctuation:no_comma", "kwargs": {}},
        ]}
        metadata = {**q.metadata, **compose_meta}

        judge = ConstraintJudge()
        assert judge.judge(_msgs("Write a poem...", "roses are red"), metadata) is True
        assert judge.judge(_msgs("Write a poem...", "roses are red, violets blue"), metadata) is False

    def test_no_leakage_across_simulated_rounds(self):
        """Simulates the bug scenario: round N sets constraints, round N+1 is count=0."""
        judge = ConstraintJudge()

        # Round N: count=1, has constraints
        meta_round_n = {"constraints": [
            {"type": "punctuation:no_comma", "kwargs": {}},
        ]}
        result_n = judge.judge(_msgs("p", "hello, world"), meta_round_n)
        assert result_n is False  # comma → fail

        # Round N+1: count=0, no constraints
        meta_round_n1 = {}
        result_n1 = judge.judge(_msgs("p", "hello, world"), meta_round_n1)
        assert result_n1 is True  # no constraints → pass (THIS WAS THE BUG)

        # Round N+2: count=1, different constraint
        meta_round_n2 = {"constraints": [
            {"type": "change_case:english_lowercase", "kwargs": {}},
        ]}
        result_n2 = judge.judge(_msgs("p", "hello, world"), meta_round_n2)
        assert result_n2 is True  # all lowercase → pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
