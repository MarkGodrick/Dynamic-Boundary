from scripts.analysis.rebuttal_cost_ledger import (
    Counter,
    aggregate_trial_records,
    price_usd,
    resolve_tokenizer,
)


def test_aggregate_trial_records_counts_multiturn_queries_and_tokens():
    records = [{
        "trials": [{
            "per_turn_meta": [
                {"prompt_tokens": 10, "completion_tokens": 3},
                {"prompt_tokens": 20, "completion_tokens": 4},
            ],
            "per_turn_judge_meta": [
                {"prompt_tokens": 30, "completion_tokens": 5},
                {"prompt_tokens": 40, "completion_tokens": 6},
            ],
        }]
    }]

    trial_units, target, judge = aggregate_trial_records(records)

    assert trial_units == 1
    assert (target.calls, target.prompt_tokens, target.completion_tokens) == (2, 30, 7)
    assert (judge.calls, judge.prompt_tokens, judge.completion_tokens) == (2, 70, 11)


def test_missing_meta_is_counted_not_silently_zero_filled():
    records = [{"trials": [{"per_turn_meta": None, "per_turn_judge_meta": [None]}]}]

    _, target, judge = aggregate_trial_records(records)

    assert target.calls == target.missing_token_calls == 1
    assert judge.calls == judge.missing_token_calls == 1


def test_price_uses_cache_miss_rate_when_split_is_absent():
    counter = Counter(calls=1, prompt_tokens=1_000_000, completion_tokens=1_000_000)

    assert price_usd(counter) == 3.0 / 7.15


def test_price_uses_reported_cache_split():
    counter = Counter(
        calls=1,
        completion_tokens=1_000_000,
        cache_hit_tokens=1_000_000,
        cache_miss_tokens=0,
    )

    assert price_usd(counter) == 2.02 / 7.15


def test_bundled_deepseek_v3_tokenizer_and_chat_framing():
    estimator = resolve_tokenizer(None)

    assert estimator.exact is True
    assert estimator.chat_format == "deepseek_v3"
    assert estimator.count("Hello!") == 2
    # BOS + "system" + User marker + "Hello!" (2) + Assistant marker.
    assert estimator.count_messages("system", "Hello!") == 6
