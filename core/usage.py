"""Token-usage extraction from OpenAI-compatible API responses.

Shared by the LLM composer (`search/composer.py`) and the LLM judge
(`judges/llm_judge.py`) so the cost ledger reads one field layout regardless
of which component made the call.

Providers report cached input tokens differently:

- DeepSeek: flat ``usage.prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``
  (the two sum to ``prompt_tokens``).
- OpenAI:   nested ``usage.prompt_tokens_details.cached_tokens`` only; the miss
  count has to be derived as ``prompt_tokens - cached_tokens``.

When a provider reports neither, both cache fields stay ``None``. Callers must
render that as "cache breakdown unavailable" rather than assuming a full miss —
cached input is roughly a quarter of the miss price, so silently defaulting
would skew the USD column by up to ~2x.
"""

from __future__ import annotations

USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
)


def empty_usage() -> dict:
    """A usage dict with every field unknown."""
    return {k: None for k in USAGE_FIELDS}


def extract_usage(response) -> dict:
    """Pull token counts off an OpenAI-compatible response.

    Args:
        response: The API response object (or anything with a ``usage``
            attribute). ``None`` / missing usage yields all-``None`` fields.

    Returns:
        dict with keys ``prompt_tokens``, ``completion_tokens``,
        ``cache_hit_tokens``, ``cache_miss_tokens``. Any field the provider
        did not report is ``None``.
    """
    usage = getattr(response, "usage", None)
    out = empty_usage()
    if usage is None:
        return out

    out["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
    out["completion_tokens"] = getattr(usage, "completion_tokens", None)

    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    miss = getattr(usage, "prompt_cache_miss_tokens", None)

    if hit is None:
        # OpenAI shape: usage.prompt_tokens_details.cached_tokens
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            hit = getattr(details, "cached_tokens", None)
            if hit is None and isinstance(details, dict):
                hit = details.get("cached_tokens")

    if miss is None and hit is not None and out["prompt_tokens"] is not None:
        miss = out["prompt_tokens"] - hit

    out["cache_hit_tokens"] = hit
    out["cache_miss_tokens"] = miss
    return out


def accumulate_usage(records: list[dict]) -> dict:
    """Sum a list of per-call usage dicts into one aggregate.

    Returns ``{calls, prompt_tokens, completion_tokens, cache_hit_tokens,
    cache_miss_tokens}``. A field stays ``None`` if *no* record reported it;
    records that reported it partially contribute their value (missing entries
    count as 0), which matches how a mixed-provider run should be billed.
    """
    agg: dict = {"calls": len(records)}
    for field in USAGE_FIELDS:
        values = [r.get(field) for r in records if r.get(field) is not None]
        agg[field] = sum(values) if values else None
    return agg
