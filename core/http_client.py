"""Shared httpx.Client builder for OpenAI SDK calls.

Default `OpenAI()` uses `httpx.Limits(max_connections=100, max_keepalive=20)`,
which caps concurrency at 100 even when the ThreadPoolExecutor running the
trial loop is wider. With panel inference at max_workers=200 against a vLLM
server that has spare KV-cache, the 100-connection cap becomes the bottleneck
(observed empirically at 2026-04-28: vLLM running=~80 / waiting=0 / KV<50%).

This module returns an httpx.Client with raised limits, intended to be passed
as the `http_client=` kwarg to `OpenAI()`. Each OpenAI client gets its own
httpx.Client — pools are per-host, and a single shared client across the
vLLM endpoint and the DeepSeek endpoint would mix unrelated keepalive
connections.
"""

import httpx


def build_openai_http_client(
    max_connections: int = 300,
    max_keepalive: int | None = None,
    timeout: float = 600.0,
) -> httpx.Client:
    """Return an httpx.Client tuned for high-concurrency OpenAI SDK use.

    Args:
        max_connections: TCP connection pool cap. Match or exceed the
            ThreadPoolExecutor's max_workers to avoid pool-timeouts.
        max_keepalive: idle keepalive cap; defaults to ~2/3 of
            max_connections so most connections stay reusable while the
            remaining slots can absorb bursts of new connections.
        timeout: per-request timeout in seconds. vLLM long generations and
            occasional slow DeepSeek judge calls both need generous values
            (httpx default 5s is too tight).
    """
    if max_keepalive is None:
        max_keepalive = max(20, (max_connections * 2) // 3)
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive,
    )
    return httpx.Client(limits=limits, timeout=timeout)
