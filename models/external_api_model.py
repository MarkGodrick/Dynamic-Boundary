"""Target models served by external, provider-native HTTP APIs.

This module is intentionally separate from :mod:`models.api_model`.
``APIModel`` is the OpenAI-compatible client used internally by
``VLLMModel``; external providers use different request and response
schemas and should not affect that local-server path.
"""

from __future__ import annotations

import email.utils
import os
import random
import threading
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from logger import get_logger
from .response_parser import ResponseParser

logger = get_logger(__name__)


_ANTHROPIC_PROTOCOL = "anthropic_messages"
_OPENAI_PROTOCOL = "openai_responses"
_OPENAI_CHAT_PROTOCOL = "openai_chat_completions"
_SUPPORTED_PROTOCOLS = frozenset({
    _ANTHROPIC_PROTOCOL,
    _OPENAI_PROTOCOL,
    _OPENAI_CHAT_PROTOCOL,
})
_RETRYABLE_STATUS = frozenset({408, 409, 429})


def resolve_external_provider(
    model: str,
    providers: dict[str, dict],
    explicit_provider: str | None = None,
) -> tuple[str, dict]:
    """Resolve an external model to one configured provider profile.

    Explicit configuration wins. Model-name routing is only a convenience
    fallback; the endpoint profile remains authoritative about the actual
    wire protocol.
    """
    if not providers:
        raise ValueError("external_api target requires a non-empty `providers` mapping")

    if explicit_provider is not None:
        if explicit_provider not in providers:
            raise ValueError(
                f"Unknown external API provider profile {explicit_provider!r}; "
                f"available profiles: {sorted(providers)}"
            )
        profile_name = explicit_provider
    else:
        wanted_protocol: str | None = None
        if model.startswith("claude-"):
            wanted_protocol = _ANTHROPIC_PROTOCOL
        elif model.startswith("gpt-"):
            wanted_protocol = _OPENAI_PROTOCOL
        elif model.startswith(("openai/", "anthropic/")):
            wanted_protocol = _OPENAI_CHAT_PROTOCOL
        if wanted_protocol is None:
            raise ValueError(
                f"Cannot infer external API provider for model {model!r}; "
                "set target_model.provider explicitly"
            )

        matches = [
            name
            for name, cfg in providers.items()
            if cfg.get("protocol") == wanted_protocol
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one provider with protocol {wanted_protocol!r} "
                f"for model {model!r}, found {matches}; set target_model.provider"
            )
        profile_name = matches[0]

    profile = dict(providers[profile_name] or {})
    protocol = profile.get("protocol")
    if protocol not in _SUPPORTED_PROTOCOLS:
        raise ValueError(
            f"Provider {profile_name!r} has unsupported protocol {protocol!r}; "
            f"expected one of {sorted(_SUPPORTED_PROTOCOLS)}"
        )
    if not profile.get("base_url"):
        raise ValueError(f"Provider {profile_name!r} is missing `base_url`")
    return profile_name, profile


def _message_text(content: Any) -> str:
    """Convert string or text-block message content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def _retry_after_seconds(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = email.utils.parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


class _NonRetryableAPIError(RuntimeError):
    """An external API rejected a request that retries cannot repair."""


class ExternalAPIModel:
    """Target model accessed through provider-native or OpenAI-compatible APIs.

    The class implements the same small target-model interface consumed by
    the trial runners: ``generate``, ``generate_conversation``,
    ``get_last_meta``, and ``shutdown``.
    """

    supports_concurrent = True

    def __init__(
        self,
        model: str,
        api_key_env: str,
        providers: dict[str, dict],
        provider: str | None = None,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        temperature: float | None = None,
        max_tokens: int = 2048,
        max_concurrency: int = 4,
        timeout: float = 600.0,
        response_parser: ResponseParser | None = None,
    ):
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

        self.name = model
        self.model = model
        self.api_key_env = api_key_env
        self.provider_name, self.provider = resolve_external_provider(
            model, providers, provider,
        )
        self.protocol = self.provider["protocol"]
        self.base_url = str(self.provider["base_url"]).rstrip("/")
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay)
        self.temperature = temperature
        self.max_tokens = int(max_tokens)
        self.max_concurrency = int(max_concurrency)
        self.timeout = float(timeout)
        self.response_parser = response_parser or ResponseParser()

        self._client: httpx.Client | None = None
        self._client_lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(self.max_concurrency)
        self._local = threading.local()

    def _ensure_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                limits = httpx.Limits(
                    max_connections=max(8, self.max_concurrency * 2),
                    max_keepalive_connections=max(4, self.max_concurrency),
                )
                self._client = httpx.Client(timeout=self.timeout, limits=limits)
                logger.info(
                    "External API client initialized "
                    "(model=%s, provider=%s, protocol=%s)",
                    self.model, self.provider_name, self.protocol,
                )
        return self._client

    def _api_key(self) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in env var {self.api_key_env!r}")
        return api_key

    def get_last_meta(self) -> dict:
        """Return normalized metadata for this thread's latest successful call."""
        return getattr(self._local, "meta", {})

    def generate(self, prompt: str, seed: int | None = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.generate_conversation(messages, seed=seed)

    def generate_conversation(
        self,
        messages: list[dict],
        seed: int | None = None,
    ) -> str:
        """Generate from an explicit conversation.

        External APIs used here do not expose a reproducible seed, so ``seed``
        is accepted for interface compatibility but intentionally not sent.
        """
        self._local.meta = {}
        if self.protocol == _ANTHROPIC_PROTOCOL:
            url, headers, payload = self._anthropic_request(messages)
        elif self.protocol == _OPENAI_PROTOCOL:
            url, headers, payload = self._openai_request(messages)
        elif self.protocol == _OPENAI_CHAT_PROTOCOL:
            url, headers, payload = self._openai_chat_request(messages)
        else:  # guarded by resolver; retained as a defensive assertion
            raise AssertionError(f"Unexpected external API protocol: {self.protocol}")

        data = self._post_with_retries(url, headers, payload)
        if self.protocol == _ANTHROPIC_PROTOCOL:
            return self._parse_anthropic(data)
        if self.protocol == _OPENAI_PROTOCOL:
            return self._parse_openai(data)
        return self._parse_openai_chat(data)

    def _anthropic_request(
        self, messages: list[dict],
    ) -> tuple[str, dict, dict]:
        system_parts: list[str] = []
        conversation: list[dict] = []
        for message in messages:
            role = message.get("role")
            if role in {"system", "developer"}:
                system_parts.append(_message_text(message.get("content", "")))
                continue
            if role not in {"user", "assistant"}:
                raise ValueError(f"Unsupported Anthropic message role: {role!r}")
            conversation.append({
                "role": role,
                "content": message.get("content", ""),
            })

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": conversation,
        }
        if system_parts:
            payload["system"] = "\n".join(system_parts)
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        return (
            f"{self.base_url}/v1/messages",
            {
                "x-api-key": self._api_key(),
                "anthropic-version": self.provider.get(
                    "anthropic_version", "2023-06-01",
                ),
                "content-type": "application/json",
            },
            payload,
        )

    def _openai_request(
        self, messages: list[dict],
    ) -> tuple[str, dict, dict]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "max_output_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return (
            f"{self.base_url}/responses",
            {
                "authorization": f"Bearer {self._api_key()}",
                "content-type": "application/json",
            },
            payload,
        )

    def _openai_chat_request(
        self, messages: list[dict],
    ) -> tuple[str, dict, dict]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        headers = {
            str(key): str(value)
            for key, value in (self.provider.get("headers") or {}).items()
        }
        headers.update({
            "authorization": f"Bearer {self._api_key()}",
            "content-type": "application/json",
        })
        return f"{self.base_url}/chat/completions", headers, payload

    def _post_with_retries(
        self,
        url: str,
        headers: dict,
        payload: dict,
    ) -> dict:
        client = self._ensure_client()
        delay = self.retry_delay
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            response: httpx.Response | None = None
            try:
                with self._semaphore:
                    response = client.post(url, headers=headers, json=payload)
                if 200 <= response.status_code < 300:
                    data = response.json()
                    body_error = data.get("error") if isinstance(data, dict) else None
                    if not body_error:
                        return data

                    error_code = (
                        body_error.get("code")
                        if isinstance(body_error, dict)
                        else None
                    )
                    retryable = (
                        error_code in _RETRYABLE_STATUS
                        or (
                            isinstance(error_code, int)
                            and error_code >= 500
                        )
                    )
                    error = RuntimeError(
                        f"API error from {self.provider_name}: {body_error}"
                    )
                    if not retryable:
                        raise _NonRetryableAPIError(str(error))
                    last_error = error
                    raise httpx.TransportError(str(error))

                body = response.text[:1000]
                retryable = (
                    response.status_code in _RETRYABLE_STATUS
                    or response.status_code >= 500
                )
                error = RuntimeError(
                    f"HTTP {response.status_code} from {self.provider_name}: {body}"
                )
                if not retryable:
                    raise _NonRetryableAPIError(str(error))
                last_error = error
            except _NonRetryableAPIError:
                raise
            except (httpx.TimeoutException, httpx.TransportError, ValueError) as exc:
                last_error = exc

            logger.warning(
                "External API error (model=%s, attempt=%d/%d): %s",
                self.model, attempt, self.max_retries, last_error,
            )
            if attempt == self.max_retries:
                break
            retry_after = _retry_after_seconds(response)
            sleep_for = retry_after if retry_after is not None else delay
            if sleep_for > 0:
                time.sleep(sleep_for + random.uniform(0.0, min(0.25, sleep_for / 4)))
            delay *= 2

        raise RuntimeError(
            f"ExternalAPIModel failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def _parse_anthropic(self, data: dict) -> str:
        raw_text = "".join(
            str(block.get("text", ""))
            for block in data.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        parsed, reasoning = self.response_parser.parse_with_reasoning(raw_text)

        usage = data.get("usage") or {}
        input_tokens = usage.get("input_tokens")
        cache_read = usage.get("cache_read_input_tokens")
        cache_write = usage.get("cache_creation_input_tokens")
        total_input = sum(
            value for value in (input_tokens, cache_read, cache_write)
            if isinstance(value, int)
        )
        if not any(
            isinstance(value, int)
            for value in (input_tokens, cache_read, cache_write)
        ):
            total_input = None

        finish_reason = data.get("stop_reason")
        self._local.meta = {
            "provider": self.provider_name,
            "protocol": self.protocol,
            "model": data.get("model") or self.model,
            "finish_reason": finish_reason,
            "prompt_tokens": total_input,
            "completion_tokens": usage.get("output_tokens"),
            "cache_hit_tokens": cache_read,
            "cache_miss_tokens": input_tokens,
            "cache_write_tokens": cache_write,
            "reasoning_tokens": (
                (usage.get("output_tokens_details") or {}).get("thinking_tokens")
            ),
            "truncated": finish_reason in {
                "max_tokens", "model_context_window_exceeded",
            },
            "reasoning_content": reasoning,
            "seed_supported": False,
        }
        return parsed

    def _parse_openai(self, data: dict) -> str:
        text_parts: list[str] = []
        for item in data.get("output", []):
            if not isinstance(item, dict):
                continue
            for part in item.get("content", []):
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text":
                    text_parts.append(str(part.get("text", "")))
                elif part.get("type") == "refusal":
                    text_parts.append(str(part.get("refusal", "")))
        raw_text = "".join(text_parts).strip()
        parsed, reasoning = self.response_parser.parse_with_reasoning(raw_text)

        usage = data.get("usage") or {}
        input_tokens = usage.get("input_tokens")
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        cache_hit = input_details.get("cached_tokens")
        cache_miss = (
            input_tokens - cache_hit
            if isinstance(input_tokens, int) and isinstance(cache_hit, int)
            else None
        )
        cache_write = input_details.get("cache_write_tokens")
        incomplete = data.get("incomplete_details") or {}
        status = data.get("status")
        finish_reason = incomplete.get("reason") or status

        self._local.meta = {
            "provider": self.provider_name,
            "protocol": self.protocol,
            "model": data.get("model") or self.model,
            "finish_reason": finish_reason,
            "prompt_tokens": input_tokens,
            "completion_tokens": usage.get("output_tokens"),
            "cache_hit_tokens": cache_hit,
            "cache_miss_tokens": cache_miss,
            "cache_write_tokens": cache_write,
            "reasoning_tokens": output_details.get("reasoning_tokens"),
            "truncated": (
                status == "incomplete"
                and finish_reason in {
                    "max_output_tokens", "max_tokens",
                    "model_context_window_exceeded",
                }
            ),
            "reasoning_content": reasoning,
            "seed_supported": False,
        }
        return parsed

    def _parse_openai_chat(self, data: dict) -> str:
        choices = data.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        content = message.get("content")
        raw_text = _message_text(content).strip() if content is not None else ""
        parsed, parsed_reasoning = self.response_parser.parse_with_reasoning(raw_text)

        usage = data.get("usage") or {}
        input_tokens = usage.get("prompt_tokens")
        input_details = usage.get("prompt_tokens_details") or {}
        output_details = usage.get("completion_tokens_details") or {}
        cache_hit = input_details.get("cached_tokens")
        cache_miss = (
            input_tokens - cache_hit
            if isinstance(input_tokens, int) and isinstance(cache_hit, int)
            else None
        )
        finish_reason = choice.get("finish_reason")
        reasoning = (
            message.get("reasoning")
            or message.get("reasoning_content")
            or parsed_reasoning
        )

        self._local.meta = {
            "provider": self.provider_name,
            "protocol": self.protocol,
            "model": data.get("model") or self.model,
            "finish_reason": finish_reason,
            "prompt_tokens": input_tokens,
            "completion_tokens": usage.get("completion_tokens"),
            "cache_hit_tokens": cache_hit,
            "cache_miss_tokens": cache_miss,
            "cache_write_tokens": input_details.get("cache_write_tokens"),
            "reasoning_tokens": output_details.get("reasoning_tokens"),
            "truncated": finish_reason == "length",
            "reasoning_content": reasoning,
            "cost": usage.get("cost"),
            "seed_supported": False,
        }
        return parsed

    def shutdown(self) -> None:
        """Close the provider HTTP connection pool; safe to call repeatedly."""
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
