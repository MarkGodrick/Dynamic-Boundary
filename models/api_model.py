"""API-based target model via OpenAI-compatible endpoint."""

import os
import threading
import time

import httpx

from logger import get_logger
from .errors import ContextOverflowError
from .response_parser import ResponseParser

logger = get_logger(__name__)


# Reserve some tokens for safety (chat template overhead, special tokens)
_TOKEN_SAFETY_MARGIN = 32
# Minimum max_tokens — below this we treat as context overflow
_MIN_MAX_TOKENS = 64


class APIModel:
    """Target model accessed via OpenAI-compatible API.

    Works with: Qwen API, Gemini (via OpenAI-compatible proxy), etc.
    """

    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        max_retries: int = 5,
        retry_delay: float = 0.5,
        temperature: float = 1.0,
        max_tokens: int = 512,
        response_parser: ResponseParser | None = None,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_parser = response_parser or ResponseParser()
        self._client = None
        self._tokenize_url = self._build_tokenize_url(base_url) if base_url else None
        # Thread-local stash for per-call metadata (finish_reason, token
        # counts). Read via `get_last_meta()` after each generate().
        self._local = threading.local()

    def get_last_meta(self) -> dict:
        """Return meta from the most recent successful generate() call on
        this thread.

        Keys (all may be absent/None if the API did not provide them):
          - "finish_reason": str, e.g. "stop" | "length" | "tool_calls"
          - "completion_tokens": int | None — output token count
          - "prompt_tokens":     int | None — input token count (detects
            context pressure on prompts near max_model_len)
          - "truncated":         bool — True iff finish_reason == "length"
          - "reasoning_content": str | None — CoT stripped by the response
            parser (R1 </think> prefix, Harmony analysis channel). None for
            non-reasoning models or when parser did not strip anything.

        Returns an empty dict if called before any generate on this thread,
        or if the last call raised.
        """
        return getattr(self._local, "meta", {})

    @staticmethod
    def _build_tokenize_url(base_url: str) -> str:
        """Strip OpenAI-compat suffix from base_url to reach vLLM /tokenize.

        e.g. http://localhost:8200/v1 -> http://localhost:8200/tokenize
        """
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        return f"{root}/tokenize"

    def _count_tokens(self, messages: list[dict]) -> tuple[int, int]:
        """Query vLLM /tokenize for exact token count and max_model_len.

        Returns (token_count, max_model_len). Raises on failure.
        """
        resp = httpx.post(
            self._tokenize_url,
            json={"model": self.model, "messages": messages},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["count"], data["max_model_len"]

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI
        from core.http_client import build_openai_http_client

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in env var '{self.api_key_env}'")

        kwargs = {"api_key": api_key, "http_client": build_openai_http_client()}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        logger.info("APIModel client initialized (model=%s)", self.model)

    def generate(self, prompt: str, seed: int | None = None) -> str:
        """Generate a response via API (single-turn).

        Args:
            prompt: User message text.
            seed: Random seed (passed to API if supported, not all APIs honor it).
        """
        messages = [{"role": "user", "content": prompt}]
        return self.generate_conversation(messages, seed=seed)

    def generate_conversation(self, messages: list[dict], seed: int | None = None) -> str:
        """Generate a response for a multi-turn conversation via API.

        For multi-turn (len(messages) > 1), queries vLLM /tokenize to get the
        exact token count and shrinks max_tokens to fit. Raises
        ContextOverflowError if there is not enough room.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."} dicts.
            seed: Random seed (passed to API if supported).
        """
        self._ensure_client()

        # Dynamic max_tokens for multi-turn (avoid context overflow).
        # Skip the tokenize call for single-turn — prompt is small enough.
        max_tokens = self.max_tokens
        if len(messages) > 1 and self._tokenize_url:
            try:
                n_tokens, max_model_len = self._count_tokens(messages)
            except Exception as e:
                logger.warning("Tokenize failed, using default max_tokens: %s", e)
            else:
                remaining = max_model_len - n_tokens - _TOKEN_SAFETY_MARGIN
                if remaining < _MIN_MAX_TOKENS:
                    raise ContextOverflowError(
                        f"Context full: {n_tokens}/{max_model_len} tokens, "
                        f"only {remaining} remaining (min={_MIN_MAX_TOKENS})"
                    )
                max_tokens = min(self.max_tokens, remaining)

        create_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            create_kwargs["seed"] = seed

        delay = self.retry_delay
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(**create_kwargs)
                choice = response.choices[0]
                content = choice.message.content
                usage = getattr(response, "usage", None)

                # Parse and extract reasoning (if any) in one pass; set meta
                # before returning so `get_last_meta` reflects this attempt
                # even on empty/None content branches.
                text_to_parse = content.strip() if content else ""
                parsed, reasoning = self.response_parser.parse_with_reasoning(
                    text_to_parse
                )
                self._local.meta = {
                    "finish_reason": choice.finish_reason,
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "truncated": choice.finish_reason == "length",
                    "reasoning_content": reasoning,
                }
                if content is None:
                    return parsed
                if content:
                    return parsed
                # Empty content (not None): symptom of broken vLLM state —
                # e.g. server emitting only special/<pad> tokens that the API
                # strips. Surface finish_reason + token counts so the cause is
                # visible in the log instead of needing a manual probe.
                logger.warning(
                    "Empty API response (attempt %d) finish=%s prompt_tok=%s "
                    "completion_tok=%s",
                    attempt + 1,
                    choice.finish_reason,
                    getattr(usage, "prompt_tokens", None),
                    getattr(usage, "completion_tokens", None),
                )
            except Exception as e:
                logger.warning(
                    "API error (attempt %d/%d): %s", attempt + 1, self.max_retries, e,
                )
            if attempt < self.max_retries - 1:
                time.sleep(delay)
                delay *= 2

        raise RuntimeError(f"APIModel failed after {self.max_retries} attempts")
