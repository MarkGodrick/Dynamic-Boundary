"""Tests for LLMJudge error handling.

Covers:
  - Content-safety block maps to category-specific verdicts.
  - Transient errors are retried; exhausted retries return None.
  - Deterministic non-safety errors return None without retrying.
  - Thread-local `get_last_error()` survives concurrent judge() calls.
  - `core.trial_utils.aggregate_trials` excludes None from denominator.
"""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judges.llm_judge import (  # noqa: E402
    JudgeAPIError,
    JudgeSafetyBlockedError,
    LLMJudge,
    _classify_api_error,
    _extract_error_text,
)
from core.trial_utils import aggregate_trials, judge_and_capture  # noqa: E402


# ---------------------------------------------------------------------------
# Fake OpenAI exception classes (match attributes used by _classify_api_error)
# ---------------------------------------------------------------------------

def _make_fake_openai_error(cls_name: str, body: dict | str, status_code: int = 400):
    """Construct an object that quacks like the named openai exception."""
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        InternalServerError,
        PermissionDeniedError,
        RateLimitError,
    )
    exc_cls = {
        "BadRequestError": BadRequestError,
        "RateLimitError": RateLimitError,
        "APITimeoutError": APITimeoutError,
        "APIConnectionError": APIConnectionError,
        "AuthenticationError": AuthenticationError,
        "PermissionDeniedError": PermissionDeniedError,
        "InternalServerError": InternalServerError,
    }[cls_name]
    # Bypass the SDK's strict __init__ by instantiating manually.
    err = exc_cls.__new__(exc_cls)
    err.status_code = status_code
    err.body = body if isinstance(body, dict) else {"error": {"message": body}}
    # Exception init for __str__ fallback
    Exception.__init__(err, str(body))
    return err


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------

def test_classify_safety_blocked_new_body_shape():
    err = _make_fake_openai_error(
        "BadRequestError",
        {"error": {"message": "Content Exists Risk", "type": "invalid_request_error"}},
    )
    assert _classify_api_error(err) == "safety_blocked"


def test_classify_safety_blocked_legacy_body_shape():
    err = _make_fake_openai_error(
        "BadRequestError",
        {"detail": "Content Exists Risk"},
    )
    assert _classify_api_error(err) == "safety_blocked"


def test_classify_safety_blocked_case_insensitive():
    err = _make_fake_openai_error(
        "BadRequestError",
        {"error": {"message": "content exists risk"}},
    )
    assert _classify_api_error(err) == "safety_blocked"


def test_classify_deterministic_non_safety_bad_request():
    err = _make_fake_openai_error(
        "BadRequestError",
        {"error": {"message": "Invalid Format"}},
    )
    assert _classify_api_error(err) == "deterministic"


def test_classify_transient_errors():
    for name in ("RateLimitError", "APITimeoutError", "APIConnectionError",
                 "InternalServerError"):
        err = _make_fake_openai_error(name, {"error": {"message": "x"}}, status_code=500)
        assert _classify_api_error(err) == "transient", name


def test_classify_auth_deterministic():
    err = _make_fake_openai_error(
        "AuthenticationError",
        {"error": {"message": "bad key"}},
        status_code=401,
    )
    assert _classify_api_error(err) == "deterministic"


def test_extract_error_text_includes_both_shapes():
    err = _make_fake_openai_error(
        "BadRequestError",
        {"error": {"message": "Content Exists Risk"}, "detail": "fallback"},
    )
    text = _extract_error_text(err).lower()
    assert "content exists risk" in text
    assert "fallback" in text


# ---------------------------------------------------------------------------
# LLMJudge verdict mapping tests
# ---------------------------------------------------------------------------

@pytest.fixture
def judge_factory(tmp_path, monkeypatch):
    """Return a factory(category) that builds an LLMJudge with a mocked client
    and an isolated error log dir."""
    monkeypatch.setenv("DEEPSEEK_API_TOKEN", "test-key")

    def _make(category: str):
        j = LLMJudge(category=category, error_log_dir=tmp_path)
        # Bypass real HTTP: inject a mock client.
        j._client = MagicMock()
        return j

    return _make


def _mock_raise(exc):
    def _raiser(*args, **kwargs):
        raise exc
    return _raiser


def test_safety_block_harmful_refusal_returns_false(judge_factory):
    j = judge_factory("harmful_refusal")
    err = _make_fake_openai_error(
        "BadRequestError",
        {"error": {"message": "Content Exists Risk"}},
    )
    j._client.chat.completions.create.side_effect = _mock_raise(err)

    verdict = j.judge(
        [{"role": "user", "content": "p"},
         {"role": "assistant", "content": "r"}],
    )
    assert verdict is False
    last = j.get_last_error()
    assert last is not None
    assert last["classification"] == "safety_blocked"
    assert last["body"]["error"]["message"] == "Content Exists Risk"


def test_safety_block_over_refusal_returns_true(judge_factory):
    j = judge_factory("over_refusal")
    err = _make_fake_openai_error(
        "BadRequestError",
        {"error": {"message": "Content Exists Risk"}},
    )
    j._client.chat.completions.create.side_effect = _mock_raise(err)

    verdict = j.judge(
        [{"role": "user", "content": "p"},
         {"role": "assistant", "content": "r"}],
    )
    assert verdict is True


def test_safety_block_sycophancy_returns_none(judge_factory):
    j = judge_factory("sycophancy_resistance")
    err = _make_fake_openai_error(
        "BadRequestError",
        {"error": {"message": "Content Exists Risk"}},
    )
    j._client.chat.completions.create.side_effect = _mock_raise(err)

    verdict = j.judge(
        [{"role": "user", "content": "p"},
         {"role": "assistant", "content": "r"}],
        metadata={"correct_answer": "A", "wrong_answer": "B"},
    )
    assert verdict is None


def test_transient_exhausted_returns_none(judge_factory):
    j = judge_factory("harmful_refusal")
    j.max_retries = 3
    j.retry_delay = 0.0  # no sleep in test
    err = _make_fake_openai_error(
        "RateLimitError", {"error": {"message": "rate limited"}}, status_code=429,
    )
    j._client.chat.completions.create.side_effect = _mock_raise(err)

    verdict = j.judge(
        [{"role": "user", "content": "p"},
         {"role": "assistant", "content": "r"}],
    )
    assert verdict is None
    last = j.get_last_error()
    assert last["classification"] == "transient"
    assert last["attempts"] == 3


def test_deterministic_error_returns_none_without_retry(judge_factory):
    j = judge_factory("harmful_refusal")
    j.max_retries = 8
    j.retry_delay = 0.0
    err = _make_fake_openai_error(
        "BadRequestError", {"error": {"message": "Invalid Format"}},
    )
    j._client.chat.completions.create.side_effect = _mock_raise(err)

    verdict = j.judge(
        [{"role": "user", "content": "p"},
         {"role": "assistant", "content": "r"}],
    )
    assert verdict is None
    last = j.get_last_error()
    assert last["classification"] == "deterministic"
    assert last["attempts"] == 1  # deterministic → no retry
    # Mock should have been called exactly once.
    assert j._client.chat.completions.create.call_count == 1


def test_successful_verdict_clears_last_error(judge_factory):
    j = judge_factory("harmful_refusal")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Answer: safe"
    j._client.chat.completions.create.return_value = mock_response

    verdict = j.judge(
        [{"role": "user", "content": "p"},
         {"role": "assistant", "content": "r"}],
    )
    assert verdict is True
    assert j.get_last_error() is None


# ---------------------------------------------------------------------------
# Thread-local isolation
# ---------------------------------------------------------------------------

def test_last_error_is_thread_local(judge_factory):
    """Two concurrent judge calls on the same judge instance must not clobber
    each other's last_error context.
    """
    j = judge_factory("harmful_refusal")
    j.retry_delay = 0.0

    safety_err = _make_fake_openai_error(
        "BadRequestError",
        {"error": {"message": "Content Exists Risk"}},
    )

    mock_ok = MagicMock()
    mock_ok.choices = [MagicMock()]
    mock_ok.choices[0].message.content = "Answer: safe"

    # Alternate: safety err on odd calls, success on even (via a lock-guarded
    # counter so we can interleave calls from two threads).
    lock = threading.Lock()
    state = {"n": 0}

    def side_effect(*args, **kwargs):
        with lock:
            state["n"] += 1
            n = state["n"]
        if n % 2 == 1:
            raise safety_err
        return mock_ok

    j._client.chat.completions.create.side_effect = side_effect

    results: dict[str, tuple] = {}

    def worker(tag: str):
        v = j.judge(
            [{"role": "user", "content": "p"},
             {"role": "assistant", "content": "r"}],
        )
        results[tag] = (v, j.get_last_error())

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    # One thread saw the safety error, the other saw a success.
    outcomes = sorted(
        (("err" if err else "ok"), v) for v, err in results.values()
    )
    assert outcomes == [("err", False), ("ok", True)]
    # The successful thread's get_last_error is None (was cleared on entry).
    # The failing thread's get_last_error has classification=safety_blocked.
    # (Thread-local storage guarantees no crossover.)


# ---------------------------------------------------------------------------
# Trial aggregation
# ---------------------------------------------------------------------------

def test_aggregate_trials_excludes_none_from_denominator():
    trials = [
        {"seed": 1, "passed": True,  "judge_outcome": "ok"},
        {"seed": 2, "passed": False, "judge_outcome": "ok"},
        {"seed": 3, "passed": True,  "judge_outcome": "ok"},
        {"seed": 4, "passed": None,  "judge_outcome": "api_exhausted"},
        {"seed": 5, "passed": None,  "judge_outcome": "trial_exception"},
        {"seed": 6, "passed": False, "judge_outcome": "safety_classified"},
    ]
    stats = aggregate_trials(trials)
    assert stats["pass_count"] == 2
    assert stats["total_trials"] == 4  # excludes the two None trials
    assert stats["n_attempted"] == 6
    assert stats["n_safety_classified"] == 1
    assert stats["n_api_failed"] == 1
    assert stats["n_trial_failed"] == 1
    assert stats["p_hat"] == 0.5


def test_aggregate_trials_all_failed_yields_none_p_hat():
    trials = [
        {"seed": 1, "passed": None, "judge_outcome": "api_exhausted"},
        {"seed": 2, "passed": None, "judge_outcome": "api_exhausted"},
    ]
    stats = aggregate_trials(trials)
    assert stats["pass_count"] == 0
    assert stats["total_trials"] == 0
    assert stats["p_hat"] is None


# ---------------------------------------------------------------------------
# judge_and_capture integration
# ---------------------------------------------------------------------------

def test_judge_and_capture_reads_last_error(judge_factory):
    j = judge_factory("harmful_refusal")
    err = _make_fake_openai_error(
        "BadRequestError",
        {"error": {"message": "Content Exists Risk"}},
    )
    j._client.chat.completions.create.side_effect = _mock_raise(err)

    passed, outcome, edict, label, jmeta = judge_and_capture(
        j, [{"role": "user", "content": "p"},
            {"role": "assistant", "content": "r"}],
        metadata=None,
    )
    assert passed is False
    assert outcome == "safety_classified"
    assert edict is not None
    assert edict["classification"] == "safety_blocked"
    assert label == "safety_blocked"
    # Safety-blocked path never reaches a successful API response → no meta.
    assert jmeta is None


def test_judge_and_capture_without_get_last_error_method():
    """A rule-based judge without get_last_error/get_last_label should report
    outcome='ok' and label=None."""

    class DummyJudge:
        def judge(self, messages, metadata=None):
            return True

    passed, outcome, edict, label, jmeta = judge_and_capture(DummyJudge(), [], None)
    assert passed is True
    assert outcome == "ok"
    assert edict is None
    assert label is None
    # DummyJudge has no get_last_meta → meta is None.
    assert jmeta is None
