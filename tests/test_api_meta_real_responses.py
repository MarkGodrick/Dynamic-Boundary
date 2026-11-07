"""Validate meta-reading code against real OpenAI-compatible API responses.

The unit tests in test_truncation_tracking.py use `SimpleNamespace` / MagicMock
to fake response objects, which silently accept any attribute name. If we
typo `finish_reason` as `finishReason`, those mocks still pass — but a real
`openai.types.chat.ChatCompletion` (pydantic) would raise AttributeError.

These tests load captured JSON fixtures (see
scripts/debug/capture_api_response_fixtures.py), reconstruct a real
ChatCompletion via `model_validate`, mock the SDK client to return it, then
exercise the full code path in APIModel / LLMJudge and assert each meta
field matches the JSON source-of-truth exactly.

Add new fixtures by running the capture script and committing the JSON;
pytest.parametrize picks them up automatically.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletion

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judges.llm_judge import LLMJudge
from core.trial_utils import judge_and_capture
from models.api_model import APIModel

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "api_responses"


def _iter_fixtures():
    """Yield (fixture_name, json_dict) for every JSON under FIXTURE_DIR.

    Files are surfaced as pytest parametrize IDs (the stem), so test
    failures name the specific fixture that broke.
    """
    if not FIXTURE_DIR.exists():
        return
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        yield pytest.param(data, id=path.stem)


# Pre-resolve fixtures once so both test classes share the same IDs.
FIXTURES = list(_iter_fixtures())


@pytest.fixture
def dummy_env():
    """Set a dummy API key env var so APIModel / LLMJudge init succeeds."""
    os.environ["DUMMY_KEY"] = "dummy"
    yield
    # Leave DUMMY_KEY set — other tests in this module may rely on it.


def _mock_client(chat_completion: ChatCompletion) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = chat_completion
    return client


def _expected_meta(data: dict) -> dict:
    """Pull the ground-truth meta values straight from the fixture JSON."""
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return {
        "finish_reason": choice["finish_reason"],
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "truncated": choice["finish_reason"] == "length",
    }


@pytest.mark.skipif(not FIXTURES, reason="no fixtures captured yet — run scripts/debug/capture_api_response_fixtures.py")
class TestAPIModelRealResponse:
    """Run APIModel.generate() against reconstructed real ChatCompletion
    objects and assert `get_last_meta()` reads every field correctly."""

    @pytest.mark.parametrize("fixture_data", FIXTURES)
    def test_get_last_meta_matches_fixture(self, dummy_env, fixture_data):
        expected = _expected_meta(fixture_data)
        completion = ChatCompletion.model_validate(fixture_data)

        m = APIModel(
            model="dummy", api_key_env="DUMMY_KEY", base_url=None,
            max_retries=1, temperature=0.0, max_tokens=128,
        )
        m._client = _mock_client(completion)

        # Call generate; meta should be stashed regardless of content.
        m.generate("any prompt")
        meta = m.get_last_meta()

        assert meta["finish_reason"] == expected["finish_reason"], (
            f"finish_reason attribute path broken: got {meta['finish_reason']!r}, "
            f"expected {expected['finish_reason']!r}"
        )
        assert meta["truncated"] == expected["truncated"]
        assert meta["completion_tokens"] == expected["completion_tokens"], (
            "usage.completion_tokens attribute path broken"
        )
        assert meta["prompt_tokens"] == expected["prompt_tokens"], (
            "usage.prompt_tokens attribute path broken"
        )
        # reasoning_content is None for base parser — parser logic is
        # covered separately; here we just ensure the key exists.
        assert "reasoning_content" in meta

    @pytest.mark.parametrize("fixture_data", FIXTURES)
    def test_content_returned_is_string(self, dummy_env, fixture_data):
        completion = ChatCompletion.model_validate(fixture_data)
        m = APIModel(
            model="dummy", api_key_env="DUMMY_KEY", base_url=None,
            max_retries=1, temperature=0.0, max_tokens=128,
        )
        m._client = _mock_client(completion)

        out = m.generate("any prompt")
        expected_content = (fixture_data["choices"][0]["message"]["content"] or "")
        assert out == expected_content.strip()


@pytest.mark.skipif(not FIXTURES, reason="no fixtures captured yet — run scripts/debug/capture_api_response_fixtures.py")
class TestLLMJudgeRealResponse:
    """Run LLMJudge.judge() against reconstructed real ChatCompletion objects
    and assert the judge's `get_last_meta()` reads every field correctly.

    We do NOT assert on the verdict — fixture content is not a valid judge
    reply ('Answer: safe/unsafe'), so `passed` will typically be False or
    None. That is fine: meta is stashed in `_call_api` before parsing, so
    it must reflect the real response regardless."""

    @pytest.mark.parametrize("fixture_data", FIXTURES)
    def test_judge_get_last_meta_matches_fixture(self, dummy_env, fixture_data):
        expected = _expected_meta(fixture_data)
        completion = ChatCompletion.model_validate(fixture_data)

        j = LLMJudge(
            model="deepseek-chat", api_key_env="DUMMY_KEY",
            base_url=None, category="harmful_refusal", max_retries=1,
        )
        j._client = _mock_client(completion)

        passed, outcome, err, label, meta = judge_and_capture(
            j,
            [{"role": "user", "content": "q"},
             {"role": "assistant", "content": "r"}],
            metadata=None,
        )

        assert meta is not None, "judge_meta should be populated on API success"
        assert meta["finish_reason"] == expected["finish_reason"]
        assert meta["truncated"] == expected["truncated"]
        assert meta["completion_tokens"] == expected["completion_tokens"]
        assert meta["prompt_tokens"] == expected["prompt_tokens"]
