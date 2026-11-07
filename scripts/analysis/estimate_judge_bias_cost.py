"""Estimate inter-judge API cost from the stored response matrices.

The script performs no API calls. It reconstructs exactly the judge prompt
used by ``judges.llm_judge.JUDGE_TEMPLATES``, tokenizes that prompt with the
bundled DeepSeek V3/V4-compatible tokenizer, and applies the per-judge prices
from ``configs/rebuttal/judge_bias.yaml``.

Input tokens are counted directly. Output tokens cannot be known before a
judge call, so by default the script uses the observed mean completion length
for each category/judge from the completed judge-human runs. A fixed fallback
or scale factor can be supplied explicitly.

Recommended sampled estimate:

    conda run -n boundary --no-capture-output python \
      scripts/analysis/estimate_judge_bias_cost.py \
      --max-items 200 --trials-per-cell 2 --workers 9

Full 9 x 500 x 40 estimate (tokenization only; no network):

    conda run -n boundary --no-capture-output python \
      scripts/analysis/estimate_judge_bias_cost.py \
      --max-items 0 --trials-per-cell 40 \
      --workers 27 \
      --output results/rebuttal/judge_bias/inter_judge_cost_full.json

The DeepSeek token count is exact for the bundled tokenizer and DeepSeek chat
framing, but remains a proxy for OpenAI and Anthropic billing tokenizers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from judges.llm_judge import JUDGE_TEMPLATES  # noqa: E402
from scripts.analysis.rebuttal_cost_ledger import resolve_tokenizer  # noqa: E402


CATEGORY_INPUTS = {
    "harmful_refusal": {
        "matrix_root": "results/harmful_refusal/filter_ablation_raw_v2_judgefix",
        "item_bank": "benchmarks/harmful_refusal/filter_ablation_pool_k500.jsonl",
    },
    "over_refusal": {
        "matrix_root": "results/over_refusal/filter_ablation_raw_v2_judgefix",
        "item_bank": "benchmarks/over_refusal/filter_ablation_pool_k500.jsonl",
    },
    "sycophancy_resistance": {
        "matrix_root": "results/sycophancy_resistance/filter_ablation_raw_v2",
        "item_bank": (
            "benchmarks/sycophancy_resistance/filter_ablation_pool_k500.jsonl"
        ),
    },
}

DEFAULT_CONFIG = REPO_ROOT / "configs/rebuttal/judge_bias.yaml"
DEFAULT_JUDGE_HUMAN_ROOT = (
    REPO_ROOT / "results/rebuttal/judge_bias/judge_human"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "results/rebuttal/judge_bias/inter_judge_cost_estimate.json"
)


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("judges"), dict):
        raise ValueError(f"{path}: expected a top-level judges mapping")
    for name, profile in config["judges"].items():
        prices = profile.get("prices_usd_per_million")
        if not isinstance(prices, dict):
            raise ValueError(f"{path}: judge {name!r} has no price mapping")
        required = {"input", "output"}
        missing = required - set(prices)
        if missing:
            raise ValueError(
                f"{path}: judge {name!r} missing prices {sorted(missing)}"
            )
    return config


def select_item_ids(
    item_ids: list[str], max_items: int, *, seed: int, category: str
) -> list[str]:
    """Select a deterministic uniform prompt sample for cost estimation."""
    ordered = sorted(item_ids)
    if max_items <= 0 or max_items >= len(ordered):
        return ordered
    rng = random.Random(f"{seed}|{category}|cost-estimate")
    return sorted(rng.sample(ordered, max_items))


def trial_order(
    n_trials: int, *, seed: int, model: str, item_id: str
) -> list[int]:
    """Match the stable trial sampling used by run_judge_bias.py."""
    indices = list(range(n_trials))
    digest = hashlib.sha256(
        f"{seed}|{model}|{item_id}".encode("utf-8")
    ).hexdigest()
    random.Random(int(digest[:16], 16)).shuffle(indices)
    return indices


def reconstruct_trial_messages(
    item: dict[str, Any], trial: dict[str, Any]
) -> list[dict[str, str]]:
    prompt = item.get("prompt") or item.get("bare_request")
    if not isinstance(prompt, str):
        raise ValueError("item bank record lacks prompt/bare_request")
    responses = trial.get("responses")
    if not responses:
        response = trial.get("response")
        responses = [response] if isinstance(response, str) else []
    if not responses or any(not isinstance(value, str) for value in responses):
        raise ValueError("stored trial lacks response text")
    metadata = item.get("metadata") or {}
    followups = item.get("follow_ups") or metadata.get("follow_ups") or []
    if len(followups) < len(responses) - 1:
        raise ValueError(
            f"stored trial has {len(responses)} responses but only "
            f"{len(followups)} follow-ups"
        )
    messages: list[dict[str, str]] = []
    for index, response in enumerate(responses):
        user_text = prompt if index == 0 else followups[index - 1]
        messages.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": response},
            ]
        )
    return messages


def canonical_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    for key in ("correct_answer", "wrong_answer"):
        if key in record:
            metadata[key] = record[key]
    return metadata


def format_conversation(messages: list[dict[str, str]]) -> str:
    user_count = sum(message["role"] == "user" for message in messages)
    multi = user_count > 1
    blocks: list[str] = []
    user_index = 0
    assistant_index = 0
    for message in messages:
        if message["role"] == "user":
            user_index += 1
            tag = f"USER REQUEST {user_index}" if multi else "REQUEST"
        elif message["role"] == "assistant":
            assistant_index += 1
            tag = f"AI RESPONSE {assistant_index}" if multi else "RESPONSE"
        else:
            continue
        blocks.append(
            f"<BEGIN {tag}>\n{message['content']}\n<END {tag}>"
        )
    return "\n\n".join(blocks)


def build_judge_request(
    category: str,
    messages: list[dict[str, str]],
    metadata: dict[str, Any],
) -> tuple[str, str]:
    template = JUDGE_TEMPLATES[category]
    values = dict(metadata)
    values["conversation"] = format_conversation(messages)
    return template.get("system") or "", template["user"].format(**values)


def load_completion_means(
    root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Read empirical completion lengths keyed by (judge_name, category)."""
    values: dict[tuple[str, str], list[int]] = defaultdict(list)
    if not root.is_dir():
        return {}
    for path in sorted(root.glob("*/judgments.jsonl")):
        for row in load_jsonl(path):
            meta = row.get("judge_meta") or {}
            count = meta.get("completion_tokens")
            judge = row.get("judge_name")
            category = row.get("category")
            if (
                isinstance(judge, str)
                and isinstance(category, str)
                and isinstance(count, int)
                and count >= 0
            ):
                values[(judge, category)].append(count)
    return {
        key: {
            "mean": sum(counts) / len(counts),
            "n_observed": len(counts),
        }
        for key, counts in values.items()
    }


def price_tokens(
    *,
    input_tokens: float,
    output_tokens: float,
    prices: dict[str, float],
    cache_hit_rate: float,
) -> dict[str, float]:
    hit_tokens = input_tokens * cache_hit_rate
    miss_tokens = input_tokens - hit_tokens
    hit_rate = float(prices.get("input_cache_hit", prices["input"]))
    miss_rate = float(prices.get("input_cache_miss", prices["input"]))
    input_cost = (hit_tokens * hit_rate + miss_tokens * miss_rate) / 1_000_000
    output_cost = output_tokens * float(prices["output"]) / 1_000_000
    return {
        "input_cache_hit_tokens": hit_tokens,
        "input_cache_miss_tokens": miss_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + output_cost,
    }


def prepare_category(
    *,
    category: str,
    matrix_root: Path,
    item_bank_path: Path,
    max_items: int,
    seed: int,
) -> dict[str, Any]:
    item_records = list(load_jsonl(item_bank_path))
    item_bank = {
        str(record["prompt_id"]): record
        for record in item_records
        if "prompt_id" in record
    }
    if len(item_bank) != len(item_records):
        raise ValueError(f"{item_bank_path}: missing or duplicate prompt_id")
    selected_ids = select_item_ids(
        list(item_bank), max_items, seed=seed, category=category
    )

    matrix_paths = sorted(
        path
        for path in matrix_root.glob("*/response_matrix.jsonl")
        if not path.parent.name.startswith(".")
    )
    if not matrix_paths:
        raise ValueError(f"No response_matrix.jsonl under {matrix_root}")

    return {
        "matrix_root": str(matrix_root),
        "item_bank": str(item_bank_path),
        "n_available_items": len(item_bank),
        "n_selected_items": len(selected_ids),
        "selected_item_ids": selected_ids,
        "n_models": len(matrix_paths),
        "_matrix_paths": [str(path) for path in matrix_paths],
    }


def estimate_model_matrix(job: dict[str, Any]) -> dict[str, Any]:
    """Tokenize one category-model matrix shard in a worker process."""
    category = str(job["category"])
    matrix_path = Path(job["matrix_path"])
    item_bank_path = Path(job["item_bank_path"])
    selected = set(job["selected_item_ids"])
    trials_per_cell = int(job["trials_per_cell"])
    seed = int(job["seed"])
    tokenizer = resolve_tokenizer(str(job["tokenizer_path"]))

    item_bank = {
        str(record["prompt_id"]): record
        for record in load_jsonl(item_bank_path)
        if str(record.get("prompt_id")) in selected
    }
    missing_bank = selected - set(item_bank)
    if missing_bank:
        raise ValueError(
            f"{item_bank_path}: missing {len(missing_bank)} selected items"
        )

    metadata_path = matrix_path.parent / "metadata.json"
    model = matrix_path.parent.name
    if metadata_path.is_file():
        model = str(
            json.loads(metadata_path.read_text(encoding="utf-8")).get(
                "model", model
            )
        )

    seen: set[str] = set()
    input_tokens = 0
    response_count = 0
    for entry in load_jsonl(matrix_path):
        item_id = str(entry["prompt_id"])
        if item_id not in selected:
            continue
        seen.add(item_id)
        trials = entry.get("trials") or []
        order = trial_order(
            len(trials), seed=seed, model=model, item_id=item_id
        )
        chosen = order[:trials_per_cell]
        if len(chosen) != trials_per_cell:
            raise ValueError(
                f"{matrix_path}: {item_id} has {len(trials)} trials, "
                f"requested {trials_per_cell}"
            )
        item = item_bank[item_id]
        metadata = canonical_metadata(item)
        for trial_index in chosen:
            messages = reconstruct_trial_messages(item, trials[trial_index])
            system, user = build_judge_request(category, messages, metadata)
            input_tokens += tokenizer.count_messages(system, user)
            response_count += 1

    missing = selected - seen
    if missing:
        raise ValueError(
            f"{matrix_path}: missing {len(missing)} selected items"
        )
    return {
        "category": category,
        "model": model,
        "matrix_path": str(matrix_path),
        "n_model_item_cells": len(seen),
        "responses": response_count,
        "input_tokens": input_tokens,
    }


def estimate_categories(
    *,
    categories: list[str],
    max_items: int,
    trials_per_cell: int,
    seed: int,
    tokenizer_path: Path,
    workers: int,
) -> dict[str, dict[str, Any]]:
    """Run all category-model tokenizer shards and aggregate their counts."""
    prepared: dict[str, dict[str, Any]] = {}
    jobs: list[dict[str, Any]] = []
    for category in categories:
        paths = CATEGORY_INPUTS[category]
        category_info = prepare_category(
            category=category,
            matrix_root=(REPO_ROOT / paths["matrix_root"]).resolve(),
            item_bank_path=(REPO_ROOT / paths["item_bank"]).resolve(),
            max_items=max_items,
            seed=seed,
        )
        prepared[category] = category_info
        for matrix_path in category_info.pop("_matrix_paths"):
            jobs.append(
                {
                    "category": category,
                    "matrix_path": matrix_path,
                    "item_bank_path": category_info["item_bank"],
                    "selected_item_ids": category_info["selected_item_ids"],
                    "trials_per_cell": trials_per_cell,
                    "seed": seed,
                    "tokenizer_path": str(tokenizer_path),
                }
            )

    shard_results: list[dict[str, Any]] = []
    if workers == 1:
        for index, job in enumerate(jobs, start=1):
            result = estimate_model_matrix(job)
            shard_results.append(result)
            print(
                f"  tokenized {index}/{len(jobs)} "
                f"{result['category']} / {result['model']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(estimate_model_matrix, job): job
                for job in jobs
            }
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                shard_results.append(result)
                completed += 1
                print(
                    f"  tokenized {completed}/{len(jobs)} "
                    f"{result['category']} / {result['model']}",
                    flush=True,
                )

    for category, result in prepared.items():
        shards = [
            shard for shard in shard_results
            if shard["category"] == category
        ]
        if len(shards) != result["n_models"]:
            raise RuntimeError(
                f"{category}: expected {result['n_models']} model shards, "
                f"got {len(shards)}"
            )
        per_model = {
            shard["model"]: {
                "matrix_path": shard["matrix_path"],
                "responses": shard["responses"],
                "input_tokens": shard["input_tokens"],
            }
            for shard in sorted(shards, key=lambda value: value["model"])
        }
        response_count = sum(shard["responses"] for shard in shards)
        input_tokens = sum(shard["input_tokens"] for shard in shards)
        result.update(
            {
                "trials_per_cell": trials_per_cell,
                "n_model_item_cells": sum(
                    shard["n_model_item_cells"] for shard in shards
                ),
                "n_stored_responses": response_count,
                "input_tokens_deepseek_proxy": input_tokens,
                "mean_input_tokens_per_response": (
                    input_tokens / response_count if response_count else None
                ),
                "per_source_model": per_model,
            }
        )
    return prepared


def render_table(report: dict[str, Any]) -> str:
    lines = [
        "| Category | Judge | Responses | Input M | Output M | Cost USD |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for category, result in report["categories"].items():
        for judge_name, judge in result["judges"].items():
            lines.append(
                f"| {category} | {judge_name} | "
                f"{result['n_stored_responses']:,} | "
                f"{judge['input_tokens_estimate'] / 1_000_000:.3f} | "
                f"{judge['output_tokens_estimate'] / 1_000_000:.3f} | "
                f"{judge['cost']['total_cost_usd']:.2f} |"
            )
    totals = report["totals"]
    lines.append(
        f"| **TOTAL** | all | **{totals['stored_responses']:,}** | "
        f"**{totals['input_tokens_across_judges'] / 1_000_000:.3f}** | "
        f"**{totals['output_tokens_across_judges'] / 1_000_000:.3f}** | "
        f"**{totals['cost_usd']:.2f}** |"
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=sorted(CATEGORY_INPUTS),
        default=list(CATEGORY_INPUTS),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Judge profiles and USD-per-million prices",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        help="Judge profile names; default: every profile in --config",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=200,
        help="Prompts/category; <=0 means all available prompts (default: 200)",
    )
    parser.add_argument(
        "--trials-per-cell",
        type=int,
        default=2,
        help="Stored trials sampled per model-item cell (default: 2)",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--tokenizer",
        help="Optional tokenizer.json; default: bundled DeepSeek tokenizer",
    )
    parser.add_argument(
        "--judge-human-root",
        type=Path,
        default=DEFAULT_JUDGE_HUMAN_ROOT,
        help="Completed runs used for empirical output-token means",
    )
    parser.add_argument(
        "--fallback-completion-tokens",
        type=float,
        default=128.0,
        help="Per-call output tokens when no empirical mean exists",
    )
    parser.add_argument(
        "--completion-tokens-per-call",
        type=float,
        help=(
            "Override empirical output-token means with one fixed value for "
            "every category/judge"
        ),
    )
    parser.add_argument(
        "--completion-scale",
        type=float,
        default=1.0,
        help="Multiply every empirical/fallback output-token mean",
    )
    parser.add_argument(
        "--cache-hit-rate",
        type=float,
        default=0.0,
        help="Assumed fraction of input tokens billed at cache-hit price",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(9, os.cpu_count() or 1),
        help=(
            "Parallel category-model tokenizer processes "
            "(default: min(9, CPU count))"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.trials_per_cell < 1:
        raise SystemExit("--trials-per-cell must be >= 1")
    if args.fallback_completion_tokens < 0:
        raise SystemExit("--fallback-completion-tokens must be >= 0")
    if (
        args.completion_tokens_per_call is not None
        and args.completion_tokens_per_call < 0
    ):
        raise SystemExit("--completion-tokens-per-call must be >= 0")
    if args.completion_scale < 0:
        raise SystemExit("--completion-scale must be >= 0")
    if not 0 <= args.cache_hit_rate <= 1:
        raise SystemExit("--cache-hit-rate must be between 0 and 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    available_judges = config["judges"]
    judge_names = args.judges or list(available_judges)
    unknown = set(judge_names) - set(available_judges)
    if unknown:
        raise SystemExit(
            f"Unknown judges {sorted(unknown)}; "
            f"available={sorted(available_judges)}"
        )

    tokenizer = resolve_tokenizer(args.tokenizer)
    completion_means = load_completion_means(
        args.judge_human_root.expanduser().resolve()
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Exact stored-response prompt reconstruction; DeepSeek tokenizer "
            "input count; empirical judge-human completion-token mean"
        ),
        "config": str(config_path),
        "selection": {
            "categories": args.categories,
            "max_items": args.max_items,
            "trials_per_cell": args.trials_per_cell,
            "seed": args.seed,
            "item_sampling": "deterministic_uniform_without_replacement",
            "trial_sampling": "stable_seeded_without_replacement",
            "tokenizer_workers": args.workers,
        },
        "tokenizer": {
            "path": str(tokenizer.path),
            "model": tokenizer.model_label,
            "exact_for_deepseek": tokenizer.exact,
            "chat_format": tokenizer.chat_format,
            "warning": (
                "DeepSeek token counts are pricing proxies for OpenAI and "
                "Anthropic, whose billing tokenizers differ."
            ),
        },
        "assumptions": {
            "cache_hit_rate": args.cache_hit_rate,
            "completion_scale": args.completion_scale,
            "fallback_completion_tokens_per_call": (
                args.fallback_completion_tokens
            ),
            "fixed_completion_tokens_per_call": (
                args.completion_tokens_per_call
            ),
            "input_prices": "configs/rebuttal/judge_bias.yaml",
            "output_token_source": str(
                args.judge_human_root.expanduser().resolve()
            ),
        },
        "categories": {},
    }

    total_responses = 0
    total_calls = 0
    total_input_across_judges = 0.0
    total_output_across_judges = 0.0
    total_cost = 0.0
    category_estimates = estimate_categories(
        categories=args.categories,
        max_items=args.max_items,
        trials_per_cell=args.trials_per_cell,
        seed=args.seed,
        tokenizer_path=tokenizer.path,
        workers=args.workers,
    )
    for category in args.categories:
        result = category_estimates[category]
        result["judges"] = {}
        input_tokens = float(result["input_tokens_deepseek_proxy"])
        calls = int(result["n_stored_responses"])
        for judge_name in judge_names:
            empirical = completion_means.get((judge_name, category))
            if args.completion_tokens_per_call is not None:
                completion_per_call = args.completion_tokens_per_call
                completion_source = {
                    "kind": "fixed_cli_override",
                    "n_observed": 0,
                }
            elif empirical:
                completion_per_call = float(empirical["mean"])
                completion_source = {
                    "kind": "judge_human_empirical_mean",
                    "n_observed": empirical["n_observed"],
                }
            else:
                completion_per_call = args.fallback_completion_tokens
                completion_source = {
                    "kind": "fixed_fallback",
                    "n_observed": 0,
                }
            completion_per_call *= args.completion_scale
            output_tokens = calls * completion_per_call
            prices = available_judges[judge_name][
                "prices_usd_per_million"
            ]
            cost = price_tokens(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                prices=prices,
                cache_hit_rate=args.cache_hit_rate,
            )
            result["judges"][judge_name] = {
                "requested_model": available_judges[judge_name]["model"],
                "calls": calls,
                "input_tokens_estimate": input_tokens,
                "completion_tokens_per_call": completion_per_call,
                "completion_token_source": completion_source,
                "output_tokens_estimate": output_tokens,
                "prices_usd_per_million": prices,
                "cost": cost,
            }
            total_calls += calls
            total_input_across_judges += input_tokens
            total_output_across_judges += output_tokens
            total_cost += cost["total_cost_usd"]
        report["categories"][category] = result
        total_responses += calls
        print(
            f"{category}: tokenized {calls:,} stored responses "
            f"({input_tokens / 1_000_000:.3f}M input tokens)",
            flush=True,
        )

    report["totals"] = {
        "stored_responses": total_responses,
        "judge_calls": total_calls,
        "input_tokens_across_judges": total_input_across_judges,
        "output_tokens_across_judges": total_output_across_judges,
        "cost_usd": total_cost,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print()
    print(render_table(report))
    print()
    print(f"Wrote {output_path}")
    print(
        "Note: OpenAI/Anthropic input tokens use the DeepSeek tokenizer as "
        "a proxy; output tokens use empirical judge-human means."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
