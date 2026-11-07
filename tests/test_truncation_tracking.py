"""Tests for truncation-tracking infrastructure.

Covers:
1. APIModel.get_last_meta() stashes finish_reason / completion_tokens /
   truncated flag after each generate call.
2. build_trial() accepts truncated and completion_tokens kwargs.
3. run_single_trial() reads meta via model.get_last_meta() and OR-aggregates
   across multi-turn.
4. aggregate_trials() reports n_truncated.
5. Backward compat: models without get_last_meta (HF, placeholder) → trial
   fields set to None, no crash.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.trial_utils import (
    _accumulate_judge_meta,
    _accumulate_meta,
    aggregate_trials,
    build_trial,
    run_single_trial,
    trial_exception_record,
)
from models.api_model import APIModel


# ---------------------------------------------------------------------------
# 1. APIModel.get_last_meta
# ---------------------------------------------------------------------------

def _mk_api_model_with_fake_client(finish_reason: str, content: str = "ok",
                                   completion_tokens: int = 42):
    """Build an APIModel with its OpenAI client replaced by a mock that
    returns a canned response."""
    m = APIModel(
        model="dummy", api_key_env="DUMMY_KEY", base_url=None,
        max_retries=1, temperature=0.0, max_tokens=128,
    )
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    usage = SimpleNamespace(completion_tokens=completion_tokens)
    fake_resp = SimpleNamespace(choices=[choice], usage=usage)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_resp
    m._client = fake_client
    import os
    os.environ["DUMMY_KEY"] = "dummy"
    return m


class TestAPIModelMeta:
    def test_meta_empty_before_generate(self):
        m = APIModel(model="dummy", api_key_env="DUMMY_KEY", base_url=None)
        assert m.get_last_meta() == {}

    def test_meta_populated_finish_stop(self):
        m = _mk_api_model_with_fake_client(finish_reason="stop", content="hi")
        out = m.generate("hello")
        assert out == "hi"
        meta = m.get_last_meta()
        assert meta["finish_reason"] == "stop"
        assert meta["truncated"] is False
        assert meta["completion_tokens"] == 42

    def test_meta_populated_finish_length(self):
        m = _mk_api_model_with_fake_client(finish_reason="length", content="cut")
        m.generate("hello")
        meta = m.get_last_meta()
        assert meta["finish_reason"] == "length"
        assert meta["truncated"] is True
        assert meta["completion_tokens"] == 42

    def test_meta_set_on_none_content(self):
        # content=None is a legitimate (if rare) API response. It returns
        # immediately via the `if content is None` branch; meta should be
        # stashed before that early return.
        m = _mk_api_model_with_fake_client(finish_reason="length", content=None)
        out = m.generate("hello")
        assert out == ""  # parsed empty string
        meta = m.get_last_meta()
        assert meta["truncated"] is True
        assert meta["finish_reason"] == "length"


# ---------------------------------------------------------------------------
# 2. build_trial kwargs
# ---------------------------------------------------------------------------

class TestBuildTrial:
    def test_defaults_none(self):
        t = build_trial(
            seed=0, response="x", responses=["x"], is_multi_turn=False,
            passed=True, failed_at_turn=0, judge_outcome="ok",
            judge_error=None,
        )
        assert t["truncated"] is None
        assert t["completion_tokens"] is None

    def test_truncated_true(self):
        t = build_trial(
            seed=0, response="x", responses=["x"], is_multi_turn=False,
            passed=True, failed_at_turn=0, judge_outcome="ok",
            judge_error=None, truncated=True, completion_tokens=1024,
        )
        assert t["truncated"] is True
        assert t["completion_tokens"] == 1024

    def test_exception_record_has_none_meta(self):
        t = trial_exception_record(42, RuntimeError("boom"))
        assert t["truncated"] is None
        assert t["completion_tokens"] is None


# ---------------------------------------------------------------------------
# 3. _accumulate_meta semantics
# ---------------------------------------------------------------------------

class TestAccumulateMeta:
    def test_no_get_last_meta_returns_prev_unchanged(self):
        class NoMeta:
            pass
        prev = {"truncated": False, "completion_tokens": 100}
        out = _accumulate_meta(NoMeta(), prev)
        assert out == prev

    def test_first_turn_populates(self):
        m = SimpleNamespace(get_last_meta=lambda: {"truncated": False, "completion_tokens": 50})
        out = _accumulate_meta(m, {"truncated": None, "completion_tokens": None})
        assert out["truncated"] is False
        assert out["completion_tokens"] == 50

    def test_truncated_is_sticky_or(self):
        # Turn 1: False, Turn 2: True → accumulator True
        m1 = SimpleNamespace(get_last_meta=lambda: {"truncated": False, "completion_tokens": 50})
        m2 = SimpleNamespace(get_last_meta=lambda: {"truncated": True, "completion_tokens": 30})
        out = _accumulate_meta(m1, {"truncated": None, "completion_tokens": None})
        out = _accumulate_meta(m2, out)
        assert out["truncated"] is True
        assert out["completion_tokens"] == 80

    def test_completion_tokens_sums(self):
        m = SimpleNamespace(get_last_meta=lambda: {"truncated": False, "completion_tokens": 25})
        out = _accumulate_meta(m, {"truncated": False, "completion_tokens": 75})
        assert out["completion_tokens"] == 100

    def test_none_completion_tokens_preserved_when_new_known(self):
        m = SimpleNamespace(get_last_meta=lambda: {"truncated": False, "completion_tokens": 25})
        out = _accumulate_meta(m, {"truncated": None, "completion_tokens": None})
        assert out["completion_tokens"] == 25


# ---------------------------------------------------------------------------
# 4. run_single_trial passes meta through
# ---------------------------------------------------------------------------

class _FakeModel:
    """Model with `get_last_meta`, returns canned finish_reason per call."""
    def __init__(self, sequence: list[dict]):
        self._seq = list(sequence)
        self._meta = {}

    def _pop(self):
        self._meta = self._seq.pop(0)

    def generate(self, prompt, seed=None):
        self._pop()
        return "RESP"

    def generate_conversation(self, messages, seed=None):
        self._pop()
        return "RESP"

    def get_last_meta(self):
        return self._meta


class _AlwaysPassJudge:
    def judge(self, messages, metadata):
        return True


class _AlwaysFailJudge:
    def judge(self, messages, metadata):
        return False


class TestRunSingleTrial:
    def test_single_turn_captures_truncated(self):
        model = _FakeModel([{"truncated": True, "completion_tokens": 1024}])
        trial = run_single_trial(
            model, _AlwaysPassJudge(),
            prompt="p", metadata=None, seed=0,
        )
        assert trial["truncated"] is True
        assert trial["completion_tokens"] == 1024

    def test_multi_turn_OR_aggregates(self):
        model = _FakeModel([
            {"truncated": False, "completion_tokens": 100},  # turn 1
            {"truncated": True, "completion_tokens": 50},   # turn 2
            {"truncated": False, "completion_tokens": 30},   # turn 3
        ])
        trial = run_single_trial(
            model, _AlwaysPassJudge(),
            prompt="p",
            metadata={"follow_ups": ["f1", "f2"]},
            seed=0,
        )
        assert trial["truncated"] is True
        assert trial["completion_tokens"] == 180

    def test_early_exit_on_fail_still_captures(self):
        model = _FakeModel([{"truncated": True, "completion_tokens": 1024}])
        trial = run_single_trial(
            model, _AlwaysFailJudge(),
            prompt="p", metadata=None, seed=0,
        )
        assert trial["passed"] is False
        assert trial["truncated"] is True
        assert trial["completion_tokens"] == 1024

    def test_model_without_meta_yields_none(self):
        class Dumb:
            def generate(self, prompt, seed=None): return "R"
            def generate_conversation(self, messages, seed=None): return "R"
        trial = run_single_trial(
            Dumb(), _AlwaysPassJudge(),
            prompt="p", metadata=None, seed=0,
        )
        assert trial["truncated"] is None
        assert trial["completion_tokens"] is None


# ---------------------------------------------------------------------------
# 5. aggregate_trials counts n_truncated
# ---------------------------------------------------------------------------

class TestAggregateTrials:
    def test_n_truncated(self):
        trials = [
            {"passed": True, "judge_outcome": "ok", "truncated": True},
            {"passed": True, "judge_outcome": "ok", "truncated": False},
            {"passed": False, "judge_outcome": "ok", "truncated": True},
            {"passed": None, "judge_outcome": "trial_exception", "truncated": None},
        ]
        agg = aggregate_trials(trials)
        assert agg["n_truncated"] == 2
        assert agg["n_attempted"] == 4
        assert agg["total_trials"] == 3
        assert agg["pass_count"] == 2

    def test_n_truncated_none_values_skipped(self):
        trials = [
            {"passed": True, "judge_outcome": "ok", "truncated": None},
            {"passed": True, "judge_outcome": "ok", "truncated": None},
        ]
        agg = aggregate_trials(trials)
        assert agg["n_truncated"] == 0

    def test_n_judge_truncated(self):
        # per_turn_judge_meta is a list; truncated=True in ANY turn counts.
        trials = [
            # judge truncated (single turn), model didn't
            {"passed": True, "judge_outcome": "ok", "truncated": False,
             "per_turn_judge_meta": [
                 {"truncated": True, "finish_reason": "length"},
             ]},
            # both fine
            {"passed": True, "judge_outcome": "ok", "truncated": False,
             "per_turn_judge_meta": [
                 {"truncated": False, "finish_reason": "stop"},
             ]},
            # model truncated, judge fine (single turn)
            {"passed": False, "judge_outcome": "ok", "truncated": True,
             "per_turn_judge_meta": [
                 {"truncated": False, "finish_reason": "stop"},
             ]},
            # rule-based judge (no meta ever)
            {"passed": True, "judge_outcome": "ok", "truncated": False,
             "per_turn_judge_meta": None},
        ]
        agg = aggregate_trials(trials)
        assert agg["n_truncated"] == 1
        assert agg["n_judge_truncated"] == 1

    def test_n_judge_truncated_multi_turn_any_turn_counts(self):
        # Middle-turn judge truncation should count, even if last turn was fine.
        trials = [
            {"passed": True, "judge_outcome": "ok", "truncated": False,
             "per_turn_judge_meta": [
                 {"truncated": False, "finish_reason": "stop"},
                 {"truncated": True, "finish_reason": "length"},  # middle turn cut
                 {"truncated": False, "finish_reason": "stop"},
             ]},
            {"passed": True, "judge_outcome": "ok", "truncated": False,
             "per_turn_judge_meta": [
                 {"truncated": False, "finish_reason": "stop"},
                 {"truncated": False, "finish_reason": "stop"},
             ]},
        ]
        agg = aggregate_trials(trials)
        assert agg["n_judge_truncated"] == 1


# ---------------------------------------------------------------------------
# 6. response_parser.parse_with_reasoning
# ---------------------------------------------------------------------------

class TestParseWithReasoning:
    def test_base_returns_none_reasoning(self):
        from models.response_parser import ResponseParser
        p = ResponseParser()
        parsed, reasoning = p.parse_with_reasoning("hello world")
        assert parsed == "hello world"
        assert reasoning is None

    def test_r1_extracts_reasoning(self):
        from models.response_parser import R1ResponseParser
        p = R1ResponseParser()
        text = "<think>reasoning step 1\nstep 2</think>\n\nfinal answer"
        parsed, reasoning = p.parse_with_reasoning(text)
        assert parsed == "final answer"
        assert reasoning == "reasoning step 1\nstep 2"

    def test_r1_reasoning_without_open_tag(self):
        # vLLM may not expose the `<think>` opener; rsplit on </think> still
        # works as the primary anchor.
        from models.response_parser import R1ResponseParser
        p = R1ResponseParser()
        text = "reasoning here</think>final"
        parsed, reasoning = p.parse_with_reasoning(text)
        assert parsed == "final"
        assert reasoning == "reasoning here"

    def test_r1_no_close_tag_returns_raw_no_reasoning(self):
        from models.response_parser import R1ResponseParser
        p = R1ResponseParser()
        text = "<think>truncated mid-reasoning"
        parsed, reasoning = p.parse_with_reasoning(text)
        # Matches existing parse() semantics: returns original text, no split
        assert parsed == text
        assert reasoning is None

    def test_r1_empty_post_think_returns_raw(self):
        from models.response_parser import R1ResponseParser
        p = R1ResponseParser()
        text = "<think>reasoning</think>"
        parsed, reasoning = p.parse_with_reasoning(text)
        # post-think empty → treat as "no clean split"
        assert parsed == text
        assert reasoning is None

    def test_harmony_extracts_analysis_channel(self):
        from models.response_parser import HarmonyResponseParser
        p = HarmonyResponseParser()
        text = (
            "<|channel|>analysis<|message|>User asked X. Plan: do Y.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Y is the answer.<|return|>"
        )
        parsed, reasoning = p.parse_with_reasoning(text)
        assert parsed == "Y is the answer."
        assert reasoning == "User asked X. Plan: do Y."

    def test_harmony_plain_text_variant(self):
        from models.response_parser import HarmonyResponseParser
        p = HarmonyResponseParser()
        text = "analysisthe user wants X. assistantfinalthe answer is Y"
        parsed, reasoning = p.parse_with_reasoning(text)
        assert parsed == "the answer is Y"
        # plain-text variant extracts the pre-"assistantfinal" chunk
        assert reasoning is not None
        assert "the user wants X" in reasoning


# ---------------------------------------------------------------------------
# 7. APIModel stashes reasoning_content + prompt_tokens
# ---------------------------------------------------------------------------

class TestAPIModelReasoningMeta:
    def test_reasoning_stashed_via_parser(self):
        from models.response_parser import R1ResponseParser
        m = APIModel(
            model="r1-test", api_key_env="DUMMY_KEY", base_url=None,
            max_retries=1, temperature=0.0, max_tokens=128,
            response_parser=R1ResponseParser(),
        )
        import os
        os.environ["DUMMY_KEY"] = "dummy"
        choice = SimpleNamespace(
            message=SimpleNamespace(
                content="<think>thinking hard</think>hi there"
            ),
            finish_reason="stop",
        )
        usage = SimpleNamespace(completion_tokens=50, prompt_tokens=200)
        fake_resp = SimpleNamespace(choices=[choice], usage=usage)
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_resp
        m._client = fake_client

        out = m.generate("prompt")
        assert out == "hi there"  # parser stripped CoT
        meta = m.get_last_meta()
        assert meta["reasoning_content"] == "thinking hard"
        assert meta["prompt_tokens"] == 200
        assert meta["completion_tokens"] == 50
        assert meta["finish_reason"] == "stop"
        assert meta["truncated"] is False

    def test_reasoning_none_for_base_parser(self):
        # Non-reasoning model (base ResponseParser) → reasoning_content=None
        m = _mk_api_model_with_fake_client(finish_reason="stop", content="hi")
        m.generate("prompt")
        meta = m.get_last_meta()
        assert meta["reasoning_content"] is None


# ---------------------------------------------------------------------------
# 8. LLMJudge.get_last_meta (via judge_and_capture)
# ---------------------------------------------------------------------------

class TestLLMJudgeMeta:
    def test_judge_meta_captured_on_success(self):
        from judges.llm_judge import LLMJudge
        from core.trial_utils import judge_and_capture

        j = LLMJudge(
            model="deepseek-chat", api_key_env="DUMMY_KEY",
            base_url=None, category="harmful_refusal",
            max_retries=1,
        )
        import os
        os.environ["DUMMY_KEY"] = "dummy"
        choice = SimpleNamespace(
            message=SimpleNamespace(
                content="Reasoning about X.\nAnswer: safe"
            ),
            finish_reason="stop",
        )
        usage = SimpleNamespace(completion_tokens=30, prompt_tokens=500)
        fake_resp = SimpleNamespace(choices=[choice], usage=usage)
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_resp
        j._client = fake_client

        passed, outcome, err, label, meta = judge_and_capture(
            j, [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "r"}],
            metadata=None,
        )
        assert passed is True
        assert outcome == "ok"
        assert meta is not None
        assert meta["finish_reason"] == "stop"
        assert meta["truncated"] is False
        assert meta["prompt_tokens"] == 500
        assert meta["completion_tokens"] == 30

    def test_judge_meta_detects_truncation(self):
        from judges.llm_judge import LLMJudge
        from core.trial_utils import judge_and_capture

        j = LLMJudge(
            model="deepseek-chat", api_key_env="DUMMY_KEY",
            base_url=None, category="harmful_refusal",
            max_retries=1,
        )
        import os
        os.environ["DUMMY_KEY"] = "dummy"
        # Truncated before emitting "Answer: ..." line → parse fails
        choice = SimpleNamespace(
            message=SimpleNamespace(
                content="Reasoning about X. But I need to consider Y and Z and"
            ),
            finish_reason="length",
        )
        usage = SimpleNamespace(completion_tokens=1024, prompt_tokens=500)
        fake_resp = SimpleNamespace(choices=[choice], usage=usage)
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_resp
        j._client = fake_client

        passed, outcome, err, label, meta = judge_and_capture(
            j, [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "r"}],
            metadata=None,
        )
        # Parse failed (no Answer line) → passed might be None/False depending
        # on category policy; the important assertion is meta flagged truncation.
        assert meta is not None
        assert meta["truncated"] is True
        assert meta["finish_reason"] == "length"


# ---------------------------------------------------------------------------
# 9. Per-turn meta list in run_single_trial
# ---------------------------------------------------------------------------

class TestPerTurnMeta:
    def test_single_turn_list_length_1(self):
        model = _FakeModel([
            {"truncated": False, "completion_tokens": 100,
             "finish_reason": "stop", "reasoning_content": None,
             "prompt_tokens": 50},
        ])
        trial = run_single_trial(
            model, _AlwaysPassJudge(),
            prompt="p", metadata=None, seed=0,
        )
        assert trial["per_turn_meta"] is not None
        assert len(trial["per_turn_meta"]) == 1
        assert trial["per_turn_meta"][0]["finish_reason"] == "stop"
        assert trial["per_turn_meta"][0]["prompt_tokens"] == 50

    def test_multi_turn_list_preserves_per_turn_detail(self):
        model = _FakeModel([
            {"truncated": False, "completion_tokens": 100,
             "finish_reason": "stop", "reasoning_content": "r1",
             "prompt_tokens": 30},
            {"truncated": True, "completion_tokens": 1024,
             "finish_reason": "length", "reasoning_content": None,
             "prompt_tokens": 80},
            {"truncated": False, "completion_tokens": 50,
             "finish_reason": "stop", "reasoning_content": "r3",
             "prompt_tokens": 150},
        ])
        trial = run_single_trial(
            model, _AlwaysPassJudge(),
            prompt="p",
            metadata={"follow_ups": ["f1", "f2"]},
            seed=0,
        )
        assert len(trial["per_turn_meta"]) == 3
        # Turn-level detail preserved
        assert trial["per_turn_meta"][1]["truncated"] is True
        assert trial["per_turn_meta"][1]["finish_reason"] == "length"
        assert trial["per_turn_meta"][2]["reasoning_content"] == "r3"
        # Trial-level aggregates still correct
        assert trial["truncated"] is True  # OR
        assert trial["completion_tokens"] == 1174  # sum

    def test_model_without_meta_no_per_turn(self):
        class Dumb:
            def generate(self, prompt, seed=None): return "R"
            def generate_conversation(self, messages, seed=None): return "R"
        trial = run_single_trial(
            Dumb(), _AlwaysPassJudge(),
            prompt="p", metadata=None, seed=0,
        )
        assert trial["per_turn_meta"] is None


# ---------------------------------------------------------------------------
# 10. Per-turn judge meta (judge_and_capture returns single dict;
#     run_single_trial accumulates into list)
# ---------------------------------------------------------------------------


class _MetaJudge:
    """Judge with `get_last_meta`, returns canned verdict + meta per call."""

    def __init__(self, verdicts: list[bool], metas: list[dict | None]):
        assert len(verdicts) == len(metas)
        self._verdicts = list(verdicts)
        self._metas = list(metas)
        self._meta_last = None

    def judge(self, messages, metadata):
        verdict = self._verdicts.pop(0)
        self._meta_last = self._metas.pop(0)
        return verdict

    def get_last_meta(self):
        # Return the current turn's meta (None is normalized downstream).
        return self._meta_last


class TestAccumulateJudgeMeta:
    def test_none_preserves_prev(self):
        out = _accumulate_judge_meta(None, None)
        assert out is None
        prev = [{"finish_reason": "stop"}]
        out = _accumulate_judge_meta(prev, None)
        # None meta must NOT wipe prior list
        assert out == prev

    def test_first_append_lazy_init(self):
        out = _accumulate_judge_meta(None, {"finish_reason": "stop", "truncated": False})
        assert out == [{"finish_reason": "stop", "truncated": False}]

    def test_append_preserves_order(self):
        prev = [{"finish_reason": "stop"}]
        out = _accumulate_judge_meta(prev, {"finish_reason": "length", "truncated": True})
        assert len(out) == 2
        assert out[0]["finish_reason"] == "stop"
        assert out[1]["truncated"] is True

    def test_does_not_mutate_prev(self):
        prev = [{"finish_reason": "stop"}]
        _ = _accumulate_judge_meta(prev, {"finish_reason": "length"})
        assert prev == [{"finish_reason": "stop"}]  # unchanged


class TestPerTurnJudgeMeta:
    def test_single_turn_judge_list_length_1(self):
        model = _FakeModel([{"truncated": False, "completion_tokens": 100}])
        judge = _MetaJudge(
            verdicts=[True],
            metas=[{"finish_reason": "stop", "truncated": False,
                    "completion_tokens": 20, "prompt_tokens": 300}],
        )
        trial = run_single_trial(model, judge, prompt="p", metadata=None, seed=0)
        assert trial["per_turn_judge_meta"] is not None
        assert len(trial["per_turn_judge_meta"]) == 1
        assert trial["per_turn_judge_meta"][0]["finish_reason"] == "stop"
        assert trial["per_turn_judge_meta"][0]["prompt_tokens"] == 300

    def test_multi_turn_judge_meta_per_turn_preserved(self):
        # Trial passes all 3 turns; judge truncates on turn 2 (middle).
        # With per-turn list, that truncation is visible — previously only
        # the last (turn 3) judge_meta was kept, hiding the middle-turn cut.
        model = _FakeModel([
            {"truncated": False, "completion_tokens": 100},
            {"truncated": False, "completion_tokens": 120},
            {"truncated": False, "completion_tokens": 110},
        ])
        judge = _MetaJudge(
            verdicts=[True, True, True],
            metas=[
                {"finish_reason": "stop",   "truncated": False, "completion_tokens": 30, "prompt_tokens": 100},
                {"finish_reason": "length", "truncated": True,  "completion_tokens": 1024, "prompt_tokens": 250},
                {"finish_reason": "stop",   "truncated": False, "completion_tokens": 25, "prompt_tokens": 400},
            ],
        )
        trial = run_single_trial(
            model, judge,
            prompt="p", metadata={"follow_ups": ["f1", "f2"]}, seed=0,
        )
        assert len(trial["per_turn_judge_meta"]) == 3
        assert trial["per_turn_judge_meta"][1]["truncated"] is True
        # aggregate_trials now catches middle-turn truncation
        agg = aggregate_trials([trial])
        assert agg["n_judge_truncated"] == 1

    def test_multi_turn_early_exit_on_fail_has_partial_list(self):
        # Trial fails at turn 2 (judge returns False); per-turn list has 2
        # entries (turn 1 pass, turn 2 fail). No turn-3 judge call was made.
        model = _FakeModel([
            {"truncated": False, "completion_tokens": 100},
            {"truncated": False, "completion_tokens": 120},
        ])
        judge = _MetaJudge(
            verdicts=[True, False],
            metas=[
                {"finish_reason": "stop", "truncated": False},
                {"finish_reason": "stop", "truncated": False},
            ],
        )
        trial = run_single_trial(
            model, judge,
            prompt="p", metadata={"follow_ups": ["f1", "f2"]}, seed=0,
        )
        assert trial["passed"] is False
        assert trial["failed_at_turn"] == 2
        assert len(trial["per_turn_judge_meta"]) == 2

    def test_rule_based_judge_yields_none(self):
        # _AlwaysPassJudge has no get_last_meta → per_turn_judge_meta stays None.
        model = _FakeModel([{"truncated": False, "completion_tokens": 100}])
        trial = run_single_trial(
            model, _AlwaysPassJudge(),
            prompt="p", metadata=None, seed=0,
        )
        assert trial["per_turn_judge_meta"] is None
