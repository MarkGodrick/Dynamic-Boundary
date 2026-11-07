from scripts.analysis.estimate_judge_bias_cost import (
    price_tokens,
    select_item_ids,
)


def test_price_tokens_splits_cache_hit_and_miss_rates():
    result = price_tokens(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        prices={
            "input": 1.0,
            "input_cache_hit": 0.1,
            "input_cache_miss": 1.0,
            "output": 5.0,
        },
        cache_hit_rate=0.25,
    )

    assert result["input_cache_hit_tokens"] == 250_000
    assert result["input_cache_miss_tokens"] == 750_000
    assert result["input_cost_usd"] == 0.775
    assert result["output_cost_usd"] == 5.0
    assert result["total_cost_usd"] == 5.775


def test_select_item_ids_is_deterministic_and_full_when_nonpositive():
    item_ids = [f"item-{index}" for index in range(20)]

    first = select_item_ids(
        item_ids, 5, seed=7, category="harmful_refusal"
    )
    second = select_item_ids(
        list(reversed(item_ids)), 5, seed=7, category="harmful_refusal"
    )

    assert first == second
    assert len(first) == 5
    assert select_item_ids(
        item_ids, 0, seed=7, category="harmful_refusal"
    ) == sorted(item_ids)
