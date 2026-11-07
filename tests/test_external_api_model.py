"""Tests for provider-native external target-model adapters."""

import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from models.external_api_model import (
    ExternalAPIModel,
    resolve_external_provider,
)
from models.loader import create_target_model


PROVIDERS = {
    "anthropic": {
        "protocol": "anthropic_messages",
        "base_url": "https://relay.test/anthropic",
    },
    "openai_responses": {
        "protocol": "openai_responses",
        "base_url": "https://relay.test/openai/v1",
    },
    "openrouter": {
        "protocol": "openai_chat_completions",
        "base_url": "https://openrouter.test/api/v1",
        "headers": {
            "HTTP-Referer": "https://boundary-search.local",
            "X-Title": "Dynamic-Boundary test",
        },
    },
}


def _model(monkeypatch, model: str, handler, **kwargs) -> ExternalAPIModel:
    monkeypatch.setenv("TEST_EXTERNAL_API_KEY", "secret")
    target = ExternalAPIModel(
        model=model,
        api_key_env="TEST_EXTERNAL_API_KEY",
        providers=PROVIDERS,
        max_retries=kwargs.pop("max_retries", 1),
        retry_delay=kwargs.pop("retry_delay", 0.0),
        max_tokens=kwargs.pop("max_tokens", 128),
        **kwargs,
    )
    target._client = httpx.Client(transport=httpx.MockTransport(handler))
    return target


class TestProviderResolution:
    def test_model_prefix_routes_to_protocol(self):
        name, profile = resolve_external_provider(
            "claude-sonnet-4-6", PROVIDERS,
        )
        assert name == "anthropic"
        assert profile["protocol"] == "anthropic_messages"

        name, profile = resolve_external_provider("gpt-5.4", PROVIDERS)
        assert name == "openai_responses"
        assert profile["protocol"] == "openai_responses"

    def test_explicit_profile_wins(self):
        name, _ = resolve_external_provider(
            "custom-deployment", PROVIDERS, "openai_responses",
        )
        assert name == "openai_responses"

    def test_openrouter_model_prefix_routes_to_chat_completions(self):
        name, profile = resolve_external_provider(
            "openai/gpt-5.4", PROVIDERS,
        )
        assert name == "openrouter"
        assert profile["protocol"] == "openai_chat_completions"

    def test_ambiguous_protocol_requires_explicit_profile(self):
        providers = dict(PROVIDERS)
        providers["second_openai"] = dict(PROVIDERS["openai_responses"])
        with pytest.raises(ValueError, match="exactly one provider"):
            resolve_external_provider("gpt-5.4", providers)


class TestAnthropicMessages:
    def test_single_turn_payload_and_meta(self, monkeypatch):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = request.headers
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": " OK "}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 11,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 3,
                        "cache_creation_input_tokens": 2,
                        "output_tokens_details": {"thinking_tokens": 0},
                    },
                },
            )

        target = _model(monkeypatch, "claude-sonnet-4-6", handler)
        assert target.generate("hello", seed=42) == "OK"

        assert captured["url"].endswith("/anthropic/v1/messages")
        assert captured["headers"]["x-api-key"] == "secret"
        assert captured["payload"] == {
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "hello"}],
        }
        meta = target.get_last_meta()
        assert meta["finish_reason"] == "end_turn"
        assert meta["prompt_tokens"] == 16
        assert meta["completion_tokens"] == 1
        assert meta["cache_hit_tokens"] == 3
        assert meta["cache_miss_tokens"] == 11
        assert meta["cache_write_tokens"] == 2
        assert meta["truncated"] is False
        assert meta["seed_supported"] is False

    def test_system_messages_are_hoisted_for_multi_turn(self, monkeypatch):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "second answer"}],
                    "stop_reason": "max_tokens",
                    "usage": {"input_tokens": 20, "output_tokens": 128},
                },
            )

        target = _model(monkeypatch, "claude-sonnet-4-6", handler)
        out = target.generate_conversation([
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second"},
        ])
        assert out == "second answer"
        assert captured["payload"]["system"] == "be concise"
        assert [m["role"] for m in captured["payload"]["messages"]] == [
            "user", "assistant", "user",
        ]
        assert target.get_last_meta()["truncated"] is True


class TestOpenAIResponses:
    def test_response_payload_text_and_cache_meta(self, monkeypatch):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = request.headers
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "model": "gpt-5.4",
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "O"},
                            {"type": "output_text", "text": "K"},
                        ],
                    }],
                    "usage": {
                        "input_tokens": 4391,
                        "input_tokens_details": {
                            "cached_tokens": 3840,
                            "cache_write_tokens": 0,
                        },
                        "output_tokens": 5,
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            )

        target = _model(monkeypatch, "gpt-5.4", handler)
        assert target.generate("hello", seed=7) == "OK"

        assert captured["url"].endswith("/openai/v1/responses")
        assert captured["headers"]["authorization"] == "Bearer secret"
        assert captured["payload"] == {
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": "hello"}],
            "max_output_tokens": 128,
        }
        meta = target.get_last_meta()
        assert meta["finish_reason"] == "completed"
        assert meta["prompt_tokens"] == 4391
        assert meta["completion_tokens"] == 5
        assert meta["cache_hit_tokens"] == 3840
        assert meta["cache_miss_tokens"] == 551
        assert meta["reasoning_tokens"] == 0
        assert meta["truncated"] is False

    def test_incomplete_response_is_truncated(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [{
                        "content": [{
                            "type": "output_text",
                            "text": "partial",
                        }],
                    }],
                    "usage": {"input_tokens": 10, "output_tokens": 128},
                },
            )

        target = _model(monkeypatch, "gpt-5.4", handler)
        assert target.generate("hello") == "partial"
        assert target.get_last_meta()["finish_reason"] == "max_output_tokens"
        assert target.get_last_meta()["truncated"] is True


class TestOpenAIChatCompletions:
    def test_payload_text_headers_and_meta(self, monkeypatch):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = request.headers
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "model": "openai/gpt-5.4",
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": " OK ",
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "cost": 0.0001,
                        "prompt_tokens_details": {
                            "cached_tokens": 2,
                            "cache_write_tokens": 1,
                        },
                        "completion_tokens_details": {
                            "reasoning_tokens": 3,
                        },
                    },
                },
            )

        target = _model(
            monkeypatch,
            "openai/gpt-5.4",
            handler,
            provider="openrouter",
            temperature=0.0,
        )
        assert target.generate("hello") == "OK"

        assert captured["url"].endswith("/api/v1/chat/completions")
        assert captured["headers"]["authorization"] == "Bearer secret"
        assert captured["headers"]["http-referer"] == (
            "https://boundary-search.local"
        )
        assert captured["payload"] == {
            "model": "openai/gpt-5.4",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 128,
            "temperature": 0.0,
        }
        meta = target.get_last_meta()
        assert meta["finish_reason"] == "stop"
        assert meta["prompt_tokens"] == 10
        assert meta["completion_tokens"] == 5
        assert meta["cache_hit_tokens"] == 2
        assert meta["cache_miss_tokens"] == 8
        assert meta["cache_write_tokens"] == 1
        assert meta["reasoning_tokens"] == 3
        assert meta["cost"] == 0.0001
        assert meta["truncated"] is False

    def test_length_with_reasoning_only_is_empty_and_truncated(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [{
                        "message": {
                            "content": None,
                            "reasoning": "internal reasoning",
                        },
                        "finish_reason": "length",
                    }],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 64,
                        "completion_tokens_details": {
                            "reasoning_tokens": 64,
                        },
                    },
                },
            )

        target = _model(
            monkeypatch,
            "openai/gpt-5.4",
            handler,
            provider="openrouter",
        )
        assert target.generate("hello") == ""
        meta = target.get_last_meta()
        assert meta["truncated"] is True
        assert meta["reasoning_content"] == "internal reasoning"

    def test_body_level_error_in_http_200_is_rejected(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "error": {
                        "message": "model unavailable",
                        "code": 403,
                    },
                },
            )

        target = _model(
            monkeypatch,
            "openai/gpt-5.4",
            handler,
            provider="openrouter",
        )
        with pytest.raises(RuntimeError, match="model unavailable"):
            target.generate("hello")


class TestRetriesAndLifecycle:
    def test_retries_rate_limit_then_succeeds(self, monkeypatch):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(
                    429, headers={"retry-after": "0"}, json={"error": "busy"},
                )
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output": [{"content": [
                        {"type": "output_text", "text": "ok"},
                    ]}],
                    "usage": {},
                },
            )

        target = _model(
            monkeypatch, "gpt-5.4", handler, max_retries=2,
        )
        assert target.generate("hello") == "ok"
        assert calls == 2

    def test_non_retryable_error_fails_once_and_clears_meta(self, monkeypatch):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(400, json={"error": "bad request"})

        target = _model(
            monkeypatch, "gpt-5.4", handler, max_retries=3,
        )
        target._local.meta = {"stale": True}
        with pytest.raises(RuntimeError, match="HTTP 400"):
            target.generate("hello")
        assert calls == 1
        assert target.get_last_meta() == {}

    def test_meta_is_thread_local(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            prompt = payload["input"][0]["content"]
            count = int(prompt)
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output": [{"content": [
                        {"type": "output_text", "text": prompt},
                    ]}],
                    "usage": {"input_tokens": count, "output_tokens": 1},
                },
            )

        target = _model(
            monkeypatch, "gpt-5.4", handler, max_concurrency=2,
        )

        def worker(value: int):
            output = target.generate(str(value))
            return output, target.get_last_meta()["prompt_tokens"]

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(worker, [10, 20]))
        assert sorted(results) == [("10", 10), ("20", 20)]

        target.shutdown()
        target.shutdown()
        assert target._client is None


def test_loader_creates_external_api_model(monkeypatch):
    monkeypatch.setenv("TEST_EXTERNAL_API_KEY", "secret")
    target = create_target_model({
        "type": "external_api",
        "name": "gpt-5.4",
        "api_key_env": "TEST_EXTERNAL_API_KEY",
        "providers": PROVIDERS,
        "max_new_tokens": 256,
        "max_concurrency": 2,
    })
    assert isinstance(target, ExternalAPIModel)
    assert target.name == "gpt-5.4"
    assert target.protocol == "openai_responses"
    assert target.max_tokens == 256
    assert target.supports_concurrent is True
    target.shutdown()
