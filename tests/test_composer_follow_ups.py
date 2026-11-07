"""Tests for Composer._generate_follow_ups (sequential + style-variety context).

Covers the B-plan invariants:
  - One API call per fu_skill (N skills → N calls).
  - previous_user_messages placeholder is filled with sentinel on the first
    call and a numbered list of prior outputs on subsequent calls.
  - Empty LLM returns are skipped (realized < planned is tolerated).
  - Stray `===FOLLOW_UP===` markers leaked by the LLM are stripped.
  - `_format_prev_messages` helper returns the expected sentinel / list shape.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search.composer import Composer  # noqa: E402
from search.skill_selector import Skill  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill(name: str, description: str = "desc") -> Skill:
    return Skill(name=name, description=description)


def _make_composer() -> Composer:
    """Construct a Composer without doing any network setup. The LLM client
    is never touched because `_call_api` is patched in every test."""
    c = Composer(category="sycophancy_resistance")
    # _ensure_client is a no-op here — we bypass it by patching _call_api,
    # which is called AFTER _ensure_client. Mark as initialized so no API
    # key env var is needed at test time.
    c._client = object()  # sentinel; never used
    return c


# ---------------------------------------------------------------------------
# _format_prev_messages
# ---------------------------------------------------------------------------

def test_format_prev_messages_empty_returns_sentinel():
    assert Composer._format_prev_messages([]) == (
        "(none — this is the first follow-up)"
    )


def test_format_prev_messages_numbers_entries_from_one():
    out = Composer._format_prev_messages(["hello", "world"])
    assert out == "1. hello\n2. world"


# ---------------------------------------------------------------------------
# _generate_follow_ups: core invariants
# ---------------------------------------------------------------------------

def test_n_skills_produces_n_api_calls_and_n_messages():
    """Hard 1-to-1 count invariant: 3 skills → 3 API calls → 3 messages."""
    composer = _make_composer()
    skills = [_make_skill(f"s{i}") for i in range(3)]

    with patch.object(composer, "_call_api", side_effect=["m1", "m2", "m3"]) as mock:
        out = composer._generate_follow_ups("bare request", skills)

    assert mock.call_count == 3
    assert out == ["m1", "m2", "m3"]


def test_previous_user_messages_is_sentinel_on_first_call():
    composer = _make_composer()
    skills = [_make_skill("s0")]

    with patch.object(composer, "_call_api", return_value="only_msg") as mock:
        composer._generate_follow_ups("bare", skills)

    # Inspect the user-role message content sent to _call_api.
    messages = mock.call_args_list[0].args[0]
    user_content = messages[1]["content"]
    assert "(none — this is the first follow-up)" in user_content


def test_previous_user_messages_accumulates_prior_outputs():
    """Call i (0-indexed, i>=1) should see a numbered list of the first i
    successfully-generated messages."""
    composer = _make_composer()
    skills = [_make_skill(f"s{i}") for i in range(3)]

    with patch.object(
        composer, "_call_api", side_effect=["alpha", "beta", "gamma"],
    ) as mock:
        composer._generate_follow_ups("bare", skills)

    # Call 0: no prior.
    assert "(none — this is the first follow-up)" in mock.call_args_list[0].args[0][1]["content"]
    # Call 1: sees "1. alpha".
    assert "1. alpha" in mock.call_args_list[1].args[0][1]["content"]
    assert "2." not in mock.call_args_list[1].args[0][1]["content"]
    # Call 2: sees "1. alpha\n2. beta".
    c2_content = mock.call_args_list[2].args[0][1]["content"]
    assert "1. alpha" in c2_content
    assert "2. beta" in c2_content
    assert "3." not in c2_content


def test_empty_llm_return_is_skipped_not_recorded():
    """Realized < planned is tolerated when the LLM returns an empty string
    for a specific skill. The next call still proceeds."""
    composer = _make_composer()
    skills = [_make_skill(f"s{i}") for i in range(3)]

    with patch.object(
        composer, "_call_api", side_effect=["first", "   ", "third"],
    ) as mock:
        out = composer._generate_follow_ups("bare", skills)

    assert mock.call_count == 3              # all 3 calls fired
    assert out == ["first", "third"]         # empty one dropped
    # The 3rd call's prior-messages block should NOT include the empty slot;
    # only successfully-generated messages accumulate.
    c3_content = mock.call_args_list[2].args[0][1]["content"]
    assert "1. first" in c3_content
    assert "2." not in c3_content            # no blank "2." entry


def test_stray_delimiter_is_stripped():
    """If the LLM leaks the old batch marker, strip it rather than letting
    it corrupt the stored message."""
    composer = _make_composer()
    skills = [_make_skill("s0")]

    with patch.object(
        composer, "_call_api",
        return_value="Hello world ===FOLLOW_UP===",
    ):
        out = composer._generate_follow_ups("bare", skills)

    assert out == ["Hello world"]


def test_empty_skill_list_returns_empty_no_api_calls():
    composer = _make_composer()
    with patch.object(composer, "_call_api") as mock:
        out = composer._generate_follow_ups("bare", [])
    assert out == []
    assert mock.call_count == 0


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def test_each_call_renders_required_placeholders():
    """Ensure the template receives exactly the placeholders the composer
    fills — a mismatch would raise KeyError at format time."""
    composer = _make_composer()
    skills = [_make_skill("only_one", "one-line description")]

    with patch.object(composer, "_call_api", return_value="msg") as mock:
        composer._generate_follow_ups("THE_BARE", skills)

    content = mock.call_args_list[0].args[0][1]["content"]
    assert "THE_BARE" in content
    assert "only_one" in content
    assert "one-line description" in content
