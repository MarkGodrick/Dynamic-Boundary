"""Paired CoT diagnostic on a small set of inter-judge disagreements.

The script deterministically selects exact stored responses on which the
three no-CoT judges disagreed, reruns those same responses with reasoning
enabled, and writes paired before/after labels. It does not regenerate target
model responses or alter the production judge configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebuttal.run_judge_bias import (  # noqa: E402
    build_inter_work_items,
    latest_records,
    load_config,
    run_work_items,
    selected_profiles,
    sha256_file,
    write_json,
)


BASE_TO_COT = {
    "deepseek_v4_flash": "deepseek_v4_flash_cot",
    "openai_gpt_5_4_mini": "openai_gpt_5_4_mini_cot",
    "anthropic_claude_haiku_4_5": "anthropic_claude_haiku_4_5_cot",
}
BASE_ORDER = list(BASE_TO_COT)

# Cover all six possible 2-vs-1 binary disagreement directions. The four
# extra slots go to the most common patterns in the OR N=2 pilot.
PATTERN_QUOTAS = {
    (1, 0, 0): 2,
    (1, 0, 1): 2,
    (1, 1, 0): 2,
    (0, 1, 1): 2,
    (0, 0, 1): 1,
    (0, 1, 0): 1,
}


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _group_baseline(
    rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows.values():
        name = row.get("judge_name")
        if name in BASE_TO_COT and isinstance(row.get("passed"), bool):
            grouped[row["work_key"]][name] = row
    return grouped


def select_disagreements(
    grouped: dict[str, dict[str, dict[str, Any]]],
    *,
    seed: int,
) -> list[str]:
    candidates: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    for work_key, judges in grouped.items():
        if set(judges) != set(BASE_ORDER):
            continue
        pattern = tuple(int(judges[name]["passed"]) for name in BASE_ORDER)
        if len(set(pattern)) > 1:
            candidates[pattern].append(work_key)

    selected: list[str] = []
    model_counts: Counter[str] = Counter()
    used_items: set[str] = set()
    for pattern, quota in PATTERN_QUOTAS.items():
        pool = candidates.get(pattern, [])
        for _ in range(quota):
            remaining = [key for key in pool if key not in selected]
            if not remaining:
                raise ValueError(f"Not enough candidates for pattern {pattern}")
            remaining.sort(
                key=lambda key: (
                    grouped[key][BASE_ORDER[0]]["item_id"] in used_items,
                    model_counts[grouped[key][BASE_ORDER[0]]["source_model"]],
                    _stable_rank(seed, key),
                )
            )
            chosen = remaining[0]
            exemplar = grouped[chosen][BASE_ORDER[0]]
            selected.append(chosen)
            model_counts[exemplar["source_model"]] += 1
            used_items.add(exemplar["item_id"])
    return selected


def pairwise_agreement(
    labels: dict[str, dict[str, bool]], names: list[str]
) -> list[dict[str, Any]]:
    output = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = [
                key
                for key, row in labels.items()
                if left in row and right in row
            ]
            output.append(
                {
                    "judge_a": left,
                    "judge_b": right,
                    "n": len(shared),
                    "agreement": (
                        sum(labels[key][left] == labels[key][right] for key in shared)
                        / len(shared)
                        if shared
                        else None
                    ),
                }
            )
    return output


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# CoT judge diagnostic",
        "",
        "## Aggregate",
        "",
        f"- Selected exact responses: {summary['n_samples']}",
        f"- CoT calls: {summary['n_cot_calls']}",
        f"- CoT unanimous: {summary['cot_unanimous']}/{summary['n_samples']}",
        f"- Estimated cost: ${summary['estimated_cost_usd']:.6f}",
        "",
        "| Condition | Judge pair | Agreement |",
        "|---|---|---:|",
    ]
    for condition in ("no_cot", "cot"):
        for row in summary[f"{condition}_pairwise_agreement"]:
            lines.append(
                f"| {condition} | {row['judge_a']} vs {row['judge_b']} | "
                f"{row['agreement']:.3f} |"
            )
    lines.extend(
        [
            "",
            "| Judge | Flips vs no-CoT |",
            "|---|---:|",
        ]
    )
    for name, count in summary["flips_by_judge"].items():
        lines.append(f"| {name} | {count}/{summary['n_samples']} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=ROOT
        / "results/rebuttal/judge_bias/inter_judge/pilot_or_k100_n2",
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=ROOT / "results/over_refusal/filter_ablation_raw_v2_judgefix",
    )
    parser.add_argument(
        "--item-bank",
        type=Path,
        default=ROOT / "benchmarks/over_refusal/anchor_bundle_M9.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/rebuttal/judge_bias_cot_diagnostic.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/rebuttal/judge_bias/inter_judge/cot_diagnostic_or_n10",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_path = args.baseline_dir.resolve() / "judgments.jsonl"
    baseline = latest_records(baseline_path)
    grouped = _group_baseline(baseline)
    selected_keys = select_disagreements(grouped, seed=args.seed)

    all_work, inputs, selection = build_inter_work_items(
        category="over_refusal",
        matrix_root=args.matrix_root.resolve(),
        item_bank_path=args.item_bank.resolve(),
        max_items=100,
        trials_per_cell=2,
        seed=args.seed,
    )
    work_by_key = {item.work_key: item for item in all_work}
    missing = set(selected_keys) - set(work_by_key)
    if missing:
        raise ValueError(f"Selected work keys missing from reconstructed bank: {missing}")
    work = [work_by_key[key] for key in selected_keys]

    config = load_config(args.config.resolve())
    profiles = selected_profiles(config, list(BASE_TO_COT.values()))
    patterns = Counter(
        tuple(int(grouped[key][name]["passed"]) for name in BASE_ORDER)
        for key in selected_keys
    )
    if args.dry_run:
        print(
            f"Dry run valid: {len(work)} exact disagreement responses x "
            f"{len(profiles)} CoT judges = {len(work) * len(profiles)} calls"
        )
        print(f"patterns={dict(patterns)}")
        return 0

    selection = {
        **selection,
        "diagnostic": "paired_cot_on_existing_disagreements",
        "baseline_path": str(baseline_path),
        "baseline_sha256": sha256_file(baseline_path),
        "selected_work_keys": selected_keys,
        "selected_binary_patterns_dsgptclaude": {
            str(key): value for key, value in sorted(patterns.items())
        },
    }
    rows = run_work_items(
        work_items=work,
        profiles=profiles,
        output_dir=args.output_dir.resolve(),
        inputs=inputs,
        selection=selection,
        workers=args.workers,
        overwrite=False,
        retry_errors=args.retry_errors,
    )
    cot_by_key: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("work_key") in selected_keys:
            cot_by_key[row["work_key"]][row["judge_name"]] = row

    baseline_labels = {
        key: {name: bool(grouped[key][name]["passed"]) for name in BASE_ORDER}
        for key in selected_keys
    }
    cot_labels = {
        key: {
            cot_name: bool(cot_by_key[key][cot_name]["passed"])
            for cot_name in BASE_TO_COT.values()
            if isinstance(cot_by_key[key].get(cot_name, {}).get("passed"), bool)
        }
        for key in selected_keys
    }
    paired = []
    flips = Counter()
    for key in selected_keys:
        exemplar = grouped[key][BASE_ORDER[0]]
        before = {}
        after = {}
        raw_text = {}
        for base_name, cot_name in BASE_TO_COT.items():
            base_row = grouped[key][base_name]
            cot_row = cot_by_key[key].get(cot_name, {})
            before[base_name] = {
                "passed": base_row["passed"],
                "label": base_row["judge_label"],
            }
            after[cot_name] = {
                "passed": cot_row.get("passed"),
                "label": cot_row.get("judge_label"),
                "outcome": cot_row.get("judge_outcome"),
            }
            raw_text[cot_name] = (cot_row.get("judge_meta") or {}).get("raw_text")
            if isinstance(cot_row.get("passed"), bool):
                flips[base_name] += cot_row["passed"] != base_row["passed"]
        paired.append(
            {
                "work_key": key,
                "source_model": exemplar["source_model"],
                "item_id": exemplar["item_id"],
                "trial_index": exemplar["trial_index"],
                "no_cot": before,
                "cot": after,
                "cot_raw_text": raw_text,
            }
        )

    cot_names = list(BASE_TO_COT.values())
    summary = {
        "experiment": "paired_cot_diagnostic",
        "n_samples": len(selected_keys),
        "n_cot_calls": len(rows),
        "selection_patterns_dsgptclaude": {
            str(key): value for key, value in sorted(patterns.items())
        },
        "no_cot_pairwise_agreement": pairwise_agreement(
            baseline_labels, BASE_ORDER
        ),
        "cot_pairwise_agreement": pairwise_agreement(cot_labels, cot_names),
        "no_cot_unanimous": sum(
            len(set(labels.values())) == 1 for labels in baseline_labels.values()
        ),
        "cot_unanimous": sum(
            len(labels) == 3 and len(set(labels.values())) == 1
            for labels in cot_labels.values()
        ),
        "flips_by_judge": {name: flips[name] for name in BASE_ORDER},
        "estimated_cost_usd": sum(
            float(row.get("estimated_cost_usd") or 0) for row in rows
        ),
        "paired_records": paired,
    }
    output_dir = args.output_dir.resolve()
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(f"Wrote paired CoT diagnostic to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
