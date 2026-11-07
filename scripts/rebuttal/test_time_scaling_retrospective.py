"""Retrospective test-time token-budget simulation on stored responses.

This script takes responses generated with a large ``max_tokens`` setting,
tokenizes each response with the target model's own tokenizer, truncates the
visible response prefix to smaller budgets, and re-runs the category judge.
It is intentionally labelled a retrospective simulation: it does not replace
an actual generation sweep.

The first supported category is constrained completion (CIF), whose
IFEval/IFBench judge is local.

Example:
    conda run -n boundary --no-capture-output python -u \
        scripts/rebuttal/test_time_scaling_retrospective.py \
        --category constrained_completion \
        --budgets 1024 2048 4096 8192
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adaptive.anchor_bundle import load_anchor_bundle
from adaptive.posterior import EAPEstimator
from judges.constraint_judge import ConstraintJudge


DEFAULT_MODELS = (
    "microsoft/Phi-4-reasoning-plus",
    "tiiuae/Falcon-H1R-7B",
)


def has_final_answer(response: str) -> bool:
    """Diagnostic only; judging still follows the existing CIF implementation."""
    if "<think>" not in response:
        return True
    if "</think>" not in response:
        return False
    return bool(response.rsplit("</think>", 1)[1].strip())


def safe_model_name(model: str) -> str:
    return model.replace("/", "_").replace("-", "_")


def resolve_local_snapshot(model: str) -> Path:
    """Resolve the locally cached main snapshot without any network access."""
    from huggingface_hub import constants

    repo_dir = Path(constants.HF_HUB_CACHE) / f"models--{model.replace('/', '--')}"
    main_ref = repo_dir / "refs" / "main"
    if main_ref.exists():
        revision = main_ref.read_text().strip()
        snapshot = repo_dir / "snapshots" / revision
        if snapshot.is_dir():
            return snapshot

    snapshots = sorted((repo_dir / "snapshots").glob("*"))
    snapshots = [p for p in snapshots if p.is_dir()]
    if len(snapshots) == 1:
        return snapshots[0]
    raise FileNotFoundError(
        f"Could not resolve one local tokenizer snapshot for {model}: {repo_dir}"
    )


def truncate_response(tokenizer, response: str, source_tokens: int,
                      source_truncated: bool, budget: int) -> tuple[str, int, bool, int]:
    """Return simulated text, actual token count, truncation flag, visible IDs.

    ``source_tokens`` comes from the inference engine and is authoritative for
    budget accounting. The re-encoded visible response can be shorter because
    generation metadata may count terminal special tokens that are omitted
    from decoded text.
    """
    token_ids = tokenizer.encode(response, add_special_tokens=False)
    visible_tokens = len(token_ids)
    was_truncated = source_tokens > budget or (
        source_tokens == budget and source_truncated
    )
    actual_tokens = min(source_tokens, budget)

    if source_tokens > budget and visible_tokens > budget:
        response = tokenizer.decode(
            token_ids[:budget],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return response, actual_tokens, was_truncated, visible_tokens


def summarize_ability(rows_by_prompt: dict[str, tuple[int, int]],
                      bundle, prior_sigma: float) -> tuple[float, float]:
    item_index = {item.item_id: item for item in bundle.items}
    missing = [item_id for item_id in item_index if item_id not in rows_by_prompt]
    extra = [item_id for item_id in rows_by_prompt if item_id not in item_index]
    if missing or extra:
        raise ValueError(
            f"Anchor mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    ordered = [item.item_id for item in bundle.items]
    k = np.asarray([rows_by_prompt[item_id][0] for item_id in ordered], dtype=float)
    n = np.asarray([rows_by_prompt[item_id][1] for item_id in ordered], dtype=float)
    beta = np.asarray([item_index[item_id].beta for item_id in ordered], dtype=float)
    return EAPEstimator(prior_mu=0.0, prior_sigma=prior_sigma).estimate(k, n, beta)


def write_summary_files(records: list[dict], output_root: Path) -> None:
    csv_path = output_root / "summary.csv"
    fields = [
        "category", "model", "max_tokens", "n_items", "n_trials",
        "mean_actual_tokens", "truncation_rate", "final_answer_rate",
        "mean_pass_rate", "pass_rate_given_final_answer",
        "theta", "theta_se", "delta_theta_from_1024",
        "delta_theta_from_largest", "label_flips_vs_source",
        "mean_abs_token_count_mismatch",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in records])

    lines = [
        "# Retrospective test-time token-budget simulation",
        "",
        "These results re-tokenize and truncate stored large-budget responses; "
        "they are not independent generation runs. Anchor item difficulties "
        "are frozen across budgets.",
        "",
        "| Model | max_tokens | Mean actual tokens | Truncation | Final answer | "
        "Pass rate | Pass given final | theta (SE) | Delta theta from 1,024 |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in records:
        delta = row["delta_theta_from_1024"]
        delta_text = "--" if delta is None else f"{delta:+.3f}"
        lines.append(
            f"| {row['model']} | {row['max_tokens']:,} | "
            f"{row['mean_actual_tokens']:.1f} | "
            f"{100 * row['truncation_rate']:.1f}% | "
            f"{100 * row['final_answer_rate']:.1f}% | "
            f"{100 * row['mean_pass_rate']:.1f}% | "
            f"{100 * row['pass_rate_given_final_answer']:.1f}% | "
            f"{row['theta']:+.3f} ({row['theta_se']:.3f}) | {delta_text} |"
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n")
    (output_root / "summary.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    )


def run_model(*, category: str, model: str, budgets: list[int],
              source_root: Path, output_root: Path, bundle,
              prior_sigma: float, overwrite: bool,
              reuse_source_labels_at_largest_budget: bool) -> list[dict]:
    from transformers import AutoTokenizer

    model_short = safe_model_name(model)
    source_dir = source_root / model_short
    source_path = source_dir / "response_matrix.jsonl"
    source_meta_path = source_dir / "metadata.json"
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    tokenizer_path = resolve_local_snapshot(model)
    print(f"[{model}] tokenizer={tokenizer_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    judge = ConstraintJudge()
    metadata_by_prompt = {item.item_id: item.metadata for item in bundle.items}
    largest_budget = max(budgets)

    model_root = output_root / model_short
    handles: dict[int, object] = {}
    temporary_paths: dict[int, Path] = {}
    final_paths: dict[int, Path] = {}
    for budget in budgets:
        budget_dir = model_root / f"max_tokens_{budget}"
        budget_dir.mkdir(parents=True, exist_ok=True)
        final_path = budget_dir / "response_matrix.jsonl"
        if final_path.exists() and not overwrite:
            raise FileExistsError(f"{final_path} exists; pass --overwrite to replace")
        temp_path = budget_dir / "response_matrix.jsonl.tmp"
        handles[budget] = temp_path.open("w")
        temporary_paths[budget] = temp_path
        final_paths[budget] = final_path

    aggregate = {
        budget: {
            "rows": {},
            "n_trials": 0,
            "actual_tokens": 0,
            "n_truncated": 0,
            "n_final_answer": 0,
            "n_pass": 0,
            "n_pass_with_final_answer": 0,
            "label_flips": 0,
            "abs_token_mismatch": 0,
        }
        for budget in budgets
    }
    seen_prompts: set[str] = set()

    try:
        with source_path.open() as source_file:
            for row_number, line in enumerate(source_file, start=1):
                if not line.strip():
                    continue
                source_row = json.loads(line)
                prompt_id = source_row["prompt_id"]
                if prompt_id not in metadata_by_prompt:
                    raise KeyError(f"{prompt_id} absent from anchor bundle")
                seen_prompts.add(prompt_id)
                prompt_metadata = metadata_by_prompt[prompt_id]
                output_trials = {budget: [] for budget in budgets}

                for source_trial in source_row.get("trials") or []:
                    response = source_trial.get("response", "")
                    encoded_ids = tokenizer.encode(response, add_special_tokens=False)
                    visible_tokens = len(encoded_ids)
                    stored_tokens = source_trial.get("completion_tokens")
                    source_tokens = (
                        int(stored_tokens) if stored_tokens is not None else visible_tokens
                    )
                    source_was_truncated = bool(source_trial.get("truncated"))

                    for budget in budgets:
                        if source_tokens > budget and visible_tokens > budget:
                            simulated_response = tokenizer.decode(
                                encoded_ids[:budget],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )
                        else:
                            simulated_response = response
                        actual_tokens = min(source_tokens, budget)
                        was_truncated = source_tokens > budget or (
                            source_tokens == budget and source_was_truncated
                        )
                        final_answer_present = has_final_answer(simulated_response)
                        reused_source_label = (
                            reuse_source_labels_at_largest_budget
                            and budget == largest_budget
                            and source_trial.get("passed") is not None
                        )
                        if reused_source_label:
                            passed = bool(source_trial["passed"])
                        else:
                            passed = judge.judge(
                                [{"role": "assistant", "content": simulated_response}],
                                prompt_metadata,
                            )
                        out_trial = dict(source_trial)
                        out_trial.update({
                            "response": simulated_response,
                            "source_passed": source_trial.get("passed"),
                            "passed": bool(passed),
                            "failed_at_turn": 0 if passed else 1,
                            "judge_outcome": "ok",
                            "judge_error": None,
                            "judge_label": None,
                            "truncated": was_truncated,
                            "completion_tokens": actual_tokens,
                            "per_turn_meta": [{
                                "finish_reason": "length" if was_truncated else "stop",
                                "completion_tokens": actual_tokens,
                                "truncated": was_truncated,
                                "retrospective": True,
                            }],
                            "per_turn_judge_meta": None,
                            "source_completion_tokens": stored_tokens,
                            "source_truncated": source_trial.get("truncated"),
                            "visible_token_count": visible_tokens,
                            "simulated_max_tokens": budget,
                            "final_answer_present": final_answer_present,
                            "reused_source_label": reused_source_label,
                        })
                        output_trials[budget].append(out_trial)

                        stats = aggregate[budget]
                        stats["n_trials"] += 1
                        stats["actual_tokens"] += actual_tokens
                        stats["n_truncated"] += int(was_truncated)
                        stats["n_final_answer"] += int(final_answer_present)
                        stats["n_pass"] += int(bool(passed))
                        stats["n_pass_with_final_answer"] += int(
                            bool(passed) and final_answer_present
                        )
                        stats["label_flips"] += int(
                            source_trial.get("passed") is not None
                            and bool(source_trial.get("passed")) != bool(passed)
                        )
                        stats["abs_token_mismatch"] += abs(source_tokens - visible_tokens)

                for budget in budgets:
                    trials = output_trials[budget]
                    pass_count = sum(int(trial["passed"]) for trial in trials)
                    total_trials = len(trials)
                    out_row = {
                        "prompt_id": prompt_id,
                        "model": model,
                        "trials": trials,
                        "pass_count": pass_count,
                        "total_trials": total_trials,
                        "n_attempted": total_trials,
                        "n_safety_classified": 0,
                        "n_api_failed": 0,
                        "n_trial_failed": 0,
                        "n_judge_label_unparsed": 0,
                        "n_truncated": sum(int(t["truncated"]) for t in trials),
                        "n_judge_truncated": 0,
                        "p_hat": pass_count / total_trials if total_trials else None,
                    }
                    handles[budget].write(
                        json.dumps(out_row, ensure_ascii=False) + "\n"
                    )
                    aggregate[budget]["rows"][prompt_id] = (
                        pass_count, total_trials
                    )
                if row_number % 25 == 0:
                    print(
                        f"[{model}] processed {row_number}/{bundle.J} items",
                        flush=True,
                    )
    finally:
        for handle in handles.values():
            handle.close()

    missing_prompts = set(metadata_by_prompt) - seen_prompts
    if missing_prompts:
        raise ValueError(f"Source matrix missing {len(missing_prompts)} anchor prompts")

    source_meta = (
        json.loads(source_meta_path.read_text()) if source_meta_path.exists() else {}
    )
    records = []
    for budget in budgets:
        os.replace(temporary_paths[budget], final_paths[budget])
        stats = aggregate[budget]
        theta, theta_se = summarize_ability(
            stats["rows"], bundle, prior_sigma=prior_sigma
        )
        n_trials = stats["n_trials"]
        record = {
            "category": category,
            "model": model,
            "max_tokens": budget,
            "n_items": len(stats["rows"]),
            "n_trials": n_trials,
            "mean_actual_tokens": stats["actual_tokens"] / n_trials,
            "truncation_rate": stats["n_truncated"] / n_trials,
            "final_answer_rate": stats["n_final_answer"] / n_trials,
            "mean_pass_rate": stats["n_pass"] / n_trials,
            "pass_rate_given_final_answer": (
                stats["n_pass_with_final_answer"] / stats["n_final_answer"]
                if stats["n_final_answer"] else 0.0
            ),
            "theta": theta,
            "theta_se": theta_se,
            "delta_theta_from_1024": None,
            "delta_theta_from_largest": None,
            "label_flips_vs_source": stats["label_flips"],
            "mean_abs_token_count_mismatch": (
                stats["abs_token_mismatch"] / n_trials
            ),
        }
        records.append(record)
        metadata = {
            "experiment": "retrospective_test_time_token_budget_simulation",
            "retrospective": True,
            "category": category,
            "model": model,
            "simulated_max_tokens": budget,
            "source_response_matrix": str(source_path),
            "source_metadata": source_meta,
            "tokenizer_snapshot": str(tokenizer_path),
            "judge": "ConstraintJudge",
            "judge_policy": (
                "existing repository behavior: strip closed <think> blocks; "
                "retain incomplete reasoning text"
            ),
            "reused_source_labels": (
                reuse_source_labels_at_largest_budget and budget == largest_budget
            ),
            "anchor_bundle": f"benchmarks/{category}/anchor_alo_bundle_M9.json",
            "anchor_beta_frozen": True,
            "ability_estimator": "EAP",
            "prior_mu": 0.0,
            "prior_sigma": prior_sigma,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "summary": record,
        }
        final_paths[budget].with_name("metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
        )
    return records


def add_deltas(records: list[dict]) -> None:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_model[record["model"]].append(record)
    for model_records in by_model.values():
        model_records.sort(key=lambda row: row["max_tokens"])
        baseline_row = next(
            (row for row in model_records if row["max_tokens"] == 1024),
            model_records[0],
        )
        baseline = baseline_row["theta"]
        largest = model_records[-1]["theta"]
        for row in model_records:
            row["delta_theta_from_1024"] = (
                None
                if row["max_tokens"] == baseline_row["max_tokens"]
                else row["theta"] - baseline
            )
            row["delta_theta_from_largest"] = row["theta"] - largest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="constrained_completion")
    parser.add_argument(
        "--models", nargs="+", default=list(DEFAULT_MODELS),
        help="Hugging Face model IDs whose anchor_set_alo matrices are reused.",
    )
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=[1024, 2048, 4096, 8192]
    )
    parser.add_argument("--prior-sigma", type=float, default=3.0)
    parser.add_argument(
        "--output-root",
        default="results/rebuttal/test_time_scaling/retrospective",
    )
    parser.add_argument(
        "--source-root", default=None,
        help="Directory containing model subdirectories with response matrices. "
             "Default: results/{category}/anchor_set_alo",
    )
    parser.add_argument(
        "--reuse-source-labels-at-largest-budget",
        action="store_true",
        help="Reuse stored trial labels for the largest requested budget. "
             "Useful for exact alignment with an existing source matrix.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.category != "constrained_completion":
        raise ValueError(
            "Only constrained_completion is supported without an external judge"
        )
    budgets = sorted(set(args.budgets))
    if not budgets or budgets[0] <= 0:
        raise ValueError("--budgets must contain positive integers")

    source_root = (
        Path(args.source_root)
        if args.source_root
        else Path("results") / args.category / "anchor_set_alo"
    )
    bundle_path = Path("benchmarks") / args.category / "anchor_alo_bundle_M9.json"
    output_root = Path(args.output_root) / args.category
    output_root.mkdir(parents=True, exist_ok=True)
    bundle = load_anchor_bundle(bundle_path)

    records = []
    for model in args.models:
        records.extend(run_model(
            category=args.category,
            model=model,
            budgets=budgets,
            source_root=source_root,
            output_root=output_root,
            bundle=bundle,
            prior_sigma=args.prior_sigma,
            overwrite=args.overwrite,
            reuse_source_labels_at_largest_budget=(
                args.reuse_source_labels_at_largest_budget
            ),
        ))
    add_deltas(records)
    records.sort(key=lambda row: (row["model"], row["max_tokens"]))
    write_summary_files(records, output_root)
    print(f"Summary: {output_root / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
