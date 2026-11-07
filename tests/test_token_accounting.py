"""Tests for composer / judge token accounting (core.usage + plumbing).

Covers:
  - extract_usage() against DeepSeek-shaped and OpenAI-shaped usage objects
  - Composer.get_last_usage() aggregation across the 1-N calls of one compose()
  - reset between consecutive compose() calls
  - thread isolation under a ThreadPoolExecutor
  - TemplateComposer.get_last_usage() is None (sibling parity)
  - _compose_task survives a composer without get_last_usage
  - RoundRecord round-trips compose_usage, and old files lacking the key load
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest

from core.usage import accumulate_usage, empty_usage, extract_usage
from search.composer import Composer, TemplateComposer
from search.sgbs import BareRequest, RoundRecord, SampledTask, _compose_task


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _Usage:
    """Bag-of-attributes stand-in for an SDK usage object."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Details:
    def __init__(self, cached_tokens):
        self.cached_tokens = cached_tokens


class _Response:
    def __init__(self, usage=None):
        self.usage = usage


def _deepseek_usage(prompt=100, completion=20, hit=60, miss=40):
    return _Response(_Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_cache_hit_tokens=hit,
        prompt_cache_miss_tokens=miss,
    ))


def _openai_usage(prompt=100, completion=20, cached=60):
    return _Response(_Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=_Details(cached),
    ))


def _plain_usage(prompt=100, completion=20):
    return _Response(_Usage(prompt_tokens=prompt, completion_tokens=completion))


# --- Full chat-completion stubs (for the judge's _call_api path) -----------

class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish_reason):
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Completion:
    def __init__(self, usage, content="Answer: safe", finish_reason="stop"):
        self.usage = usage
        self.choices = [_Choice(content, finish_reason)]


class _StubClient:
    """Stands in for an OpenAI client: .chat.completions.create(...)."""

    def __init__(self, response):
        self._response = response
        self.chat = self
        self.completions = self

    def create(self, **kw):
        return self._response


def _deepseek_completion(prompt=100, completion=20, hit=60, miss=40,
                         finish_reason="stop"):
    return _Completion(
        _Usage(prompt_tokens=prompt, completion_tokens=completion,
               prompt_cache_hit_tokens=hit, prompt_cache_miss_tokens=miss),
        finish_reason=finish_reason,
    )


def _openai_completion(prompt=100, completion=20, cached=60):
    return _Completion(
        _Usage(prompt_tokens=prompt, completion_tokens=completion,
               prompt_tokens_details=_Details(cached)),
    )


def _plain_completion(prompt=100, completion=20):
    return _Completion(
        _Usage(prompt_tokens=prompt, completion_tokens=completion),
    )


class _FakeSkill:
    def __init__(self, name):
        self.name = name
        self.description = f"desc of {name}"
        self.extra = {}


# ---------------------------------------------------------------------------
# extract_usage
# ---------------------------------------------------------------------------

class TestExtractUsage:

    def test_deepseek_flat_fields(self):
        u = extract_usage(_deepseek_usage())
        assert u == {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "cache_hit_tokens": 60,
            "cache_miss_tokens": 40,
        }

    def test_openai_nested_cached_tokens_derives_miss(self):
        u = extract_usage(_openai_usage(prompt=100, cached=60))
        assert u["cache_hit_tokens"] == 60
        assert u["cache_miss_tokens"] == 40  # derived: prompt - cached

    def test_openai_details_as_dict(self):
        resp = _Response(_Usage(
            prompt_tokens=100,
            completion_tokens=20,
            prompt_tokens_details={"cached_tokens": 25},
        ))
        u = extract_usage(resp)
        assert u["cache_hit_tokens"] == 25
        assert u["cache_miss_tokens"] == 75

    def test_no_cache_fields_stays_none(self):
        """A provider with no cache reporting must not be costed as all-miss."""
        u = extract_usage(_plain_usage())
        assert u["prompt_tokens"] == 100
        assert u["cache_hit_tokens"] is None
        assert u["cache_miss_tokens"] is None

    def test_missing_usage_object(self):
        assert extract_usage(_Response(None)) == empty_usage()
        assert extract_usage(object()) == empty_usage()

    def test_zero_cached_tokens_is_not_treated_as_missing(self):
        u = extract_usage(_openai_usage(prompt=100, cached=0))
        assert u["cache_hit_tokens"] == 0
        assert u["cache_miss_tokens"] == 100


class TestAccumulateUsage:

    def test_sums_and_counts(self):
        agg = accumulate_usage([
            extract_usage(_deepseek_usage(100, 20, 60, 40)),
            extract_usage(_deepseek_usage(200, 30, 150, 50)),
        ])
        assert agg == {
            "calls": 2,
            "prompt_tokens": 300,
            "completion_tokens": 50,
            "cache_hit_tokens": 210,
            "cache_miss_tokens": 90,
        }

    def test_empty_list(self):
        agg = accumulate_usage([])
        assert agg["calls"] == 0
        assert all(agg[k] is None for k in
                   ("prompt_tokens", "completion_tokens",
                    "cache_hit_tokens", "cache_miss_tokens"))

    def test_field_none_only_when_no_record_has_it(self):
        agg = accumulate_usage([
            extract_usage(_plain_usage(100, 20)),
            extract_usage(_deepseek_usage(50, 10, 30, 20)),
        ])
        assert agg["prompt_tokens"] == 150
        # only the second record reported a cache split
        assert agg["cache_hit_tokens"] == 30
        assert agg["cache_miss_tokens"] == 20


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

def _make_composer(monkeypatch, responses):
    """Composer whose _call_api records usage from a canned response list.

    `responses` is consumed round-robin so any number of calls works.
    """
    comp = Composer(api_key_env="UNUSED")
    monkeypatch.setattr(comp, "_ensure_client", lambda: None)
    monkeypatch.setattr(comp, "_ensure_templates", lambda: None)

    state = {"i": 0}

    def fake_call_api(messages):
        resp = responses[state["i"] % len(responses)]
        state["i"] += 1
        comp._record_usage(resp)
        return "composed text"

    monkeypatch.setattr(comp, "_call_api", fake_call_api)
    # Skip template machinery: go straight through _call_api.
    monkeypatch.setattr(comp, "_compose_prompt",
                        lambda br, sk: comp._call_api([]))
    monkeypatch.setattr(comp, "_rephrase", lambda br: comp._call_api([]))
    monkeypatch.setattr(
        comp, "_generate_follow_ups",
        lambda br, fus: [comp._call_api([]) for _ in fus],
    )
    return comp


class TestComposerUsage:

    def test_none_before_any_compose(self):
        comp = Composer(api_key_env="UNUSED")
        assert comp.get_last_usage() is None

    def test_single_call(self, monkeypatch):
        comp = _make_composer(monkeypatch, [_deepseek_usage(100, 20, 60, 40)])
        comp.compose("req", [_FakeSkill("s1")])
        assert comp.get_last_usage() == {
            "calls": 1,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "cache_hit_tokens": 60,
            "cache_miss_tokens": 40,
        }

    def test_follow_ups_add_calls(self, monkeypatch):
        """One compose() = turn-1 call + one call per follow-up skill."""
        comp = _make_composer(monkeypatch, [_deepseek_usage(100, 20, 60, 40)])
        comp.compose("req", [_FakeSkill("s1")],
                     follow_up_skills=[_FakeSkill("f1"), _FakeSkill("f2")])
        agg = comp.get_last_usage()
        assert agg["calls"] == 3
        assert agg["prompt_tokens"] == 300
        assert agg["completion_tokens"] == 60

    def test_reset_between_composes(self, monkeypatch):
        comp = _make_composer(monkeypatch, [_deepseek_usage(100, 20, 60, 40)])
        comp.compose("req", [_FakeSkill("s1")])
        comp.compose("req2", [_FakeSkill("s2")])
        agg = comp.get_last_usage()
        assert agg["calls"] == 1, "usage must not accumulate across compose()"
        assert agg["prompt_tokens"] == 100

    def test_no_skills_path_still_counted(self, monkeypatch):
        """Empty skill list takes the _rephrase branch, which also bills."""
        comp = _make_composer(monkeypatch, [_deepseek_usage(70, 10, 0, 70)])
        comp.compose("req", [])
        assert comp.get_last_usage()["calls"] == 1
        assert comp.get_last_usage()["prompt_tokens"] == 70

    def test_thread_isolation(self, monkeypatch):
        """4 workers composing concurrently must not see each other's tokens."""
        comp = Composer(api_key_env="UNUSED")
        monkeypatch.setattr(comp, "_ensure_client", lambda: None)
        monkeypatch.setattr(comp, "_ensure_templates", lambda: None)

        def make_prompt(bare_request, skills):
            # token count keyed to the request so each thread is identifiable
            n = int(bare_request)
            comp._record_usage(_deepseek_usage(n, n, n // 2, n - n // 2))
            return "composed"

        monkeypatch.setattr(comp, "_compose_prompt", make_prompt)

        def work(n):
            comp.compose(str(n), [_FakeSkill("s")])
            return n, comp.get_last_usage()

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(work, [100, 200, 300, 400] * 5))

        for n, agg in results:
            assert agg["calls"] == 1
            assert agg["prompt_tokens"] == n

    def test_record_usage_outside_compose(self, monkeypatch):
        """_call_api used directly (no compose) still tallies, not crashes."""
        comp = Composer(api_key_env="UNUSED")
        comp._record_usage(_plain_usage(10, 5))
        assert comp.get_last_usage()["calls"] == 1


class TestTemplateComposerParity:

    def test_get_last_usage_is_none(self):
        tc = TemplateComposer(seed=0)
        assert tc.get_last_usage() is None

    def test_signature_parity_with_composer(self):
        assert hasattr(TemplateComposer, "get_last_usage")
        assert hasattr(Composer, "get_last_usage")


# ---------------------------------------------------------------------------
# Plumbing: _compose_task -> BatchTask -> RoundRecord -> jsonl
# ---------------------------------------------------------------------------

class _StubComposer:
    """Minimal composer with usage."""

    def __init__(self, usage):
        self._usage = usage

    def compose(self, bare_request, skills, **kw):
        return "prompt", {}

    def get_last_usage(self):
        return self._usage


class _LegacyComposer:
    """Composer predating token accounting — no get_last_usage at all."""

    def compose(self, bare_request, skills, **kw):
        return "prompt", {}


def _sampled():
    return SampledTask(
        round_num=1,
        q_key="q1",
        q=BareRequest(prompt="q1", metadata={}),
        count=1,
        selected_skills=[_FakeSkill("s1")],
        fu_skills=None,
    )


class TestComposeTaskPlumbing:

    def test_usage_reaches_batch_task(self):
        usage = {"calls": 2, "prompt_tokens": 300, "completion_tokens": 40,
                 "cache_hit_tokens": 200, "cache_miss_tokens": 100}
        task = _compose_task(_sampled(), _StubComposer(usage))
        assert task is not None
        assert task.compose_usage == usage

    def test_composer_without_method(self):
        task = _compose_task(_sampled(), _LegacyComposer())
        assert task is not None
        assert task.compose_usage is None

    def test_template_composer_yields_none(self):
        task = _compose_task(_sampled(), _StubComposer(None))
        assert task.compose_usage is None


class TestRoundRecordSerialization:

    def test_round_trip(self):
        usage = {"calls": 1, "prompt_tokens": 10, "completion_tokens": 2,
                 "cache_hit_tokens": None, "cache_miss_tokens": None}
        rec = RoundRecord(
            round_num=1, bare_request="q", skill_count=1, skills=["s"],
            prompt="p", p_hat=0.5, stage=2, result_type="edge_case",
            is_edge_case=True, trials=[], compose_usage=usage,
        )
        d = json.loads(json.dumps(asdict(rec)))
        assert d["compose_usage"] == usage
        assert RoundRecord(**d).compose_usage == usage

    def test_old_file_without_key_loads(self):
        """trial_details.jsonl written before this change must still load."""
        d = {
            "round_num": 1, "bare_request": "q", "skill_count": 1,
            "skills": ["s"], "prompt": "p", "p_hat": 0.5, "stage": 2,
            "result_type": "too_easy", "is_edge_case": False, "trials": [],
            "follow_ups": [],
        }
        rec = RoundRecord(**d)
        assert rec.compose_usage is None

    def test_default_is_none(self):
        rec = RoundRecord(
            round_num=1, bare_request="q", skill_count=0, skills=[],
            prompt="p", p_hat=0.1, stage=1, result_type="too_hard",
            is_edge_case=False, trials=[],
        )
        assert asdict(rec)["compose_usage"] is None


# ---------------------------------------------------------------------------
# Judge meta
# ---------------------------------------------------------------------------

class TestJudgeMetaCacheFields:
    """Exercise LLMJudge._call_api's real meta-building path with a stub client."""

    @staticmethod
    def _judge_with_response(resp):
        from judges.llm_judge import LLMJudge

        judge = LLMJudge(api_key_env="UNUSED", category="harmful_refusal")
        judge._client = _StubClient(resp)
        return judge

    @pytest.mark.parametrize("resp,hit,miss", [
        (_deepseek_completion(100, 20, 60, 40), 60, 40),
        (_openai_completion(100, 20, 60), 60, 40),
        (_plain_completion(100, 20), None, None),
    ])
    def test_cache_split_recorded(self, resp, hit, miss):
        judge = self._judge_with_response(resp)
        judge._call_api([{"role": "user", "content": "x"}])
        meta = judge.get_last_meta()
        assert meta["prompt_tokens"] == 100
        assert meta["completion_tokens"] == 20
        assert meta["cache_hit_tokens"] == hit
        assert meta["cache_miss_tokens"] == miss

    def test_existing_meta_keys_preserved(self):
        """Truncation detection and raw_text must survive the refactor."""
        judge = self._judge_with_response(
            _deepseek_completion(100, 20, 60, 40, finish_reason="length"),
        )
        judge._call_api([{"role": "user", "content": "x"}])
        meta = judge.get_last_meta()
        assert meta["finish_reason"] == "length"
        assert meta["truncated"] is True
        assert meta["raw_text"] == "Answer: safe"
