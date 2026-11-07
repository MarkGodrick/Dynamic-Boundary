"""Compare N=40 cell pass rates across judges on a small OR pilot.

Ten model-prompt cells are selected from the existing N=2 inter-judge run,
prioritizing high trial-level disagreement while covering all nine reference
models. The existing two trials are reused and only the remaining 38 stored
responses per cell are rejudged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebuttal.run_judge_bias import (  # noqa: E402
    WorkItem,
    _trial_order,
    canonical_metadata,
    latest_records,
    load_config,
    load_jsonl,
    reconstruct_trial_messages,
    run_work_items,
    selected_profiles,
    sha256_file,
    write_json,
    write_jsonl,
)


JUDGES = [
    "deepseek_v4_flash",
    "openai_gpt_5_4_mini",
    "anthropic_claude_haiku_4_5",
]


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _baseline_cells(
    latest: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], dict[str, dict[str, dict[str, Any]]]]:
    cells: dict[
        tuple[str, str], dict[str, dict[str, dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(dict))
    for row in latest.values():
        if (
            row.get("judge_name") in JUDGES
            and isinstance(row.get("passed"), bool)
        ):
            cell = (row["source_model"], row["item_id"])
            cells[cell][row["work_key"]][row["judge_name"]] = row
    return cells


def _cell_disagreement(
    trials: dict[str, dict[str, dict[str, Any]]],
) -> float:
    values = []
    for judges in trials.values():
        if set(judges) != set(JUDGES):
            continue
        labels = [bool(judges[name]["passed"]) for name in JUDGES]
        values.extend(
            int(labels[left] != labels[right])
            for left in range(3)
            for right in range(left + 1, 3)
        )
    return float(np.mean(values)) if values else -1.0


def select_cells(
    cells: dict[tuple[str, str], dict[str, dict[str, dict[str, Any]]]],
    *,
    n_cells: int,
    seed: int,
) -> list[tuple[str, str]]:
    if n_cells < 1:
        raise ValueError("n_cells must be positive")
    scores = {cell: _cell_disagreement(trials) for cell, trials in cells.items()}
    models = sorted({model for model, _ in cells})
    selected: list[tuple[str, str]] = []
    used_items: set[str] = set()

    # First cover every panel model once when the requested size permits.
    for model in models[:n_cells]:
        candidates = [cell for cell in cells if cell[0] == model]
        candidates.sort(
            key=lambda cell: (
                -scores[cell],
                cell[1] in used_items,
                _stable_rank(seed, "|".join(cell)),
            )
        )
        chosen = candidates[0]
        selected.append(chosen)
        used_items.add(chosen[1])

    remaining = [cell for cell in cells if cell not in selected]
    remaining.sort(
        key=lambda cell: (
            -scores[cell],
            cell[1] in used_items,
            _stable_rank(seed, "|".join(cell)),
        )
    )
    selected.extend(remaining[: n_cells - len(selected)])
    if len(selected) != n_cells:
        raise ValueError(f"Could select only {len(selected)} of {n_cells} cells")
    return selected


def build_selected_work(
    *,
    selected: list[tuple[str, str]],
    cells: dict[tuple[str, str], dict[str, dict[str, dict[str, Any]]]],
    item_bank_path: Path,
    seed: int,
) -> tuple[list[WorkItem], list[dict[str, str]]]:
    item_bank = {
        str(row["prompt_id"]): row for row in load_jsonl(item_bank_path)
    }
    by_source: dict[str, set[str]] = defaultdict(set)
    source_model: dict[tuple[str, str], str] = {}
    for cell in selected:
        exemplar = next(iter(next(iter(cells[cell].values())).values()))
        by_source[exemplar["source_file"]].add(cell[1])
        source_model[(exemplar["source_file"], cell[1])] = cell[0]

    work: list[WorkItem] = []
    inputs = [{"path": str(item_bank_path), "sha256": sha256_file(item_bank_path)}]
    for raw_path, wanted_items in sorted(by_source.items()):
        matrix_path = Path(raw_path)
        inputs.append({"path": str(matrix_path), "sha256": sha256_file(matrix_path)})
        found: set[str] = set()
        with matrix_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                item_id = str(entry["prompt_id"])
                if item_id not in wanted_items:
                    continue
                found.add(item_id)
                model = source_model[(raw_path, item_id)]
                trials = entry.get("trials") or []
                order = _trial_order(
                    len(trials), seed=seed, model=model, item_id=item_id
                )
                if len(order) < 40:
                    raise ValueError(f"{model}|{item_id} has only {len(order)} trials")
                for trial_index in order[:40]:
                    messages = reconstruct_trial_messages(
                        item_bank[item_id], trials[trial_index]
                    )
                    work.append(
                        WorkItem(
                            experiment="inter_judge",
                            category="over_refusal",
                            sample_id=f"{model}|{item_id}|{trial_index}",
                            messages=messages,
                            metadata=canonical_metadata(item_bank[item_id]),
                            source_file=str(matrix_path),
                            source_model=model,
                            item_id=item_id,
                            trial_index=trial_index,
                        )
                    )
        missing = wanted_items - found
        if missing:
            raise ValueError(f"{matrix_path}: missing selected items {sorted(missing)}")
    if len(work) != len(selected) * 40:
        raise ValueError(f"Expected {len(selected) * 40} work items, got {len(work)}")
    return work, inputs


def _safe_spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    value = float(spearmanr(left, right)[0])
    return value if math.isfinite(value) else None


def analyze(
    rows: list[dict[str, Any]],
    selected: list[tuple[str, str]],
) -> dict[str, Any]:
    by_cell: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        cell = (row["source_model"], row["item_id"])
        if cell in selected and row.get("judge_name") in JUDGES:
            by_cell[cell][row["judge_name"]].append(row)

    cell_rows = []
    for model, item_id in selected:
        groups = by_cell[(model, item_id)]
        rates = {}
        for name in JUDGES:
            valid = [row for row in groups[name] if isinstance(row.get("passed"), bool)]
            rates[name] = sum(row["passed"] for row in valid) / len(valid)
        work_labels = {
            row["work_key"]: row["passed"]
            for name in JUDGES
            for row in groups[name]
            if isinstance(row.get("passed"), bool)
        }
        pair_agreement = {}
        for left_index, left in enumerate(JUDGES):
            left_map = {
                row["work_key"]: row["passed"]
                for row in groups[left]
                if isinstance(row.get("passed"), bool)
            }
            for right in JUDGES[left_index + 1 :]:
                right_map = {
                    row["work_key"]: row["passed"]
                    for row in groups[right]
                    if isinstance(row.get("passed"), bool)
                }
                shared = set(left_map) & set(right_map)
                pair_agreement[f"{left}__{right}"] = (
                    sum(left_map[key] == right_map[key] for key in shared)
                    / len(shared)
                )
        cell_rows.append(
            {
                "source_model": model,
                "item_id": item_id,
                "n_trials_per_judge": {
                    name: sum(
                        isinstance(row.get("passed"), bool) for row in groups[name]
                    )
                    for name in JUDGES
                },
                "pass_rate": rates,
                "pass_rate_range": max(rates.values()) - min(rates.values()),
                "pairwise_trial_agreement": pair_agreement,
            }
        )

    pairwise = []
    for left_index, left in enumerate(JUDGES):
        for right in JUDGES[left_index + 1 :]:
            left_rates = [row["pass_rate"][left] for row in cell_rows]
            right_rates = [row["pass_rate"][right] for row in cell_rows]
            deltas = np.asarray(right_rates) - np.asarray(left_rates)
            agreement_key = f"{left}__{right}"
            pairwise.append(
                {
                    "judge_a": left,
                    "judge_b": right,
                    "mean_trial_agreement": float(
                        np.mean(
                            [
                                row["pairwise_trial_agreement"][agreement_key]
                                for row in cell_rows
                            ]
                        )
                    ),
                    "mean_signed_pass_rate_delta_b_minus_a": float(np.mean(deltas)),
                    "mean_absolute_pass_rate_delta": float(np.mean(np.abs(deltas))),
                    "median_absolute_pass_rate_delta": float(
                        np.median(np.abs(deltas))
                    ),
                    "max_absolute_pass_rate_delta": float(np.max(np.abs(deltas))),
                    "spearman_pass_rate": _safe_spearman(left_rates, right_rates),
                    "n_cells_abs_delta_le_0_10": int(
                        np.sum(np.abs(deltas) <= 0.10 + 1e-12)
                    ),
                    "n_cells_same_majority_side": int(
                        np.sum(
                            (np.asarray(left_rates) >= 0.5)
                            == (np.asarray(right_rates) >= 0.5)
                        )
                    ),
                    "n_cells": len(cell_rows),
                }
            )

    latest_cost = sum(float(row.get("estimated_cost_usd") or 0) for row in rows)
    incremental_cost = sum(
        float(row.get("estimated_cost_usd") or 0)
        for row in rows
        if not row.get("reused_from")
    )
    return {
        "experiment": "judge_pass_rate_diagnostic",
        "n_cells": len(cell_rows),
        "n_trials_per_cell": 40,
        "judges": JUDGES,
        "pairwise": pairwise,
        "cells": cell_rows,
        "latest_record_cost_usd": latest_cost,
        "incremental_cost_usd": incremental_cost,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Judge pass-rate diagnostic",
        "",
        "| Judge pair | Trial agreement | Mean abs(Δp) | Median abs(Δp) | "
        "Max abs(Δp) | rho(p) | Cells abs(Δp)<=.10 | Same side of .5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["pairwise"]:
        rho = row["spearman_pass_rate"]
        rho_text = "NA" if rho is None else f"{rho:.3f}"
        lines.append(
            f"| {row['judge_a']} vs {row['judge_b']} | "
            f"{row['mean_trial_agreement']:.3f} | "
            f"{row['mean_absolute_pass_rate_delta']:.3f} | "
            f"{row['median_absolute_pass_rate_delta']:.3f} | "
            f"{row['max_absolute_pass_rate_delta']:.3f} | "
            f"{rho_text} | "
            f"{row['n_cells_abs_delta_le_0_10']}/{row['n_cells']} | "
            f"{row['n_cells_same_majority_side']}/{row['n_cells']} |"
        )
    lines.extend(
        [
            "",
            "| Model | Item | p(DeepSeek) | p(GPT) | p(Claude) | Range |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["cells"]:
        rates = row["pass_rate"]
        lines.append(
            f"| {row['source_model']} | {row['item_id']} | "
            f"{rates[JUDGES[0]]:.3f} | {rates[JUDGES[1]]:.3f} | "
            f"{rates[JUDGES[2]]:.3f} | {row['pass_rate_range']:.3f} |"
        )
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
        "--item-bank",
        type=Path,
        default=ROOT / "benchmarks/over_refusal/anchor_bundle_M9.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/rebuttal/judge_bias.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/rebuttal/judge_bias/inter_judge/pass_rate_or_cells10_n40",
    )
    parser.add_argument("--n-cells", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_path = args.baseline_dir.resolve() / "judgments.jsonl"
    baseline = latest_records(baseline_path)
    cells = _baseline_cells(baseline)
    selected = select_cells(cells, n_cells=args.n_cells, seed=args.seed)
    work, inputs = build_selected_work(
        selected=selected,
        cells=cells,
        item_bank_path=args.item_bank.resolve(),
        seed=args.seed,
    )
    work_keys = {item.work_key for item in work}
    seed_rows = [
        {**row, "reused_from": str(baseline_path)}
        for row in baseline.values()
        if row.get("work_key") in work_keys and row.get("judge_name") in JUDGES
    ]
    expected_seeded = args.n_cells * 2 * len(JUDGES)
    if len(seed_rows) != expected_seeded:
        raise ValueError(
            f"Expected {expected_seeded} reusable N=2 records, got {len(seed_rows)}"
        )
    if args.dry_run:
        print(
            f"Dry run valid: {args.n_cells} cells x 40 trials x 3 judges; "
            f"reused={len(seed_rows)}, new_calls={len(work) * 3 - len(seed_rows)}"
        )
        for cell in selected:
            print(cell, f"N2_disagreement={_cell_disagreement(cells[cell]):.3f}")
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = output_dir / "judgments.jsonl"
    if not judgments_path.exists():
        write_jsonl(judgments_path, seed_rows)

    config = load_config(args.config.resolve())
    profiles = selected_profiles(config, JUDGES)
    rows = run_work_items(
        work_items=work,
        profiles=profiles,
        output_dir=output_dir,
        inputs=[
            {
                "path": str(baseline_path),
                "sha256": sha256_file(baseline_path),
                "purpose": "reused first two trials",
            },
            *inputs,
        ],
        selection={
            "category": "over_refusal",
            "n_cells": args.n_cells,
            "n_trials_per_cell": 40,
            "seed": args.seed,
            "selected_cells": [
                {
                    "source_model": model,
                    "item_id": item_id,
                    "n2_trial_disagreement": _cell_disagreement(
                        cells[(model, item_id)]
                    ),
                }
                for model, item_id in selected
            ],
            "selection_method": (
                "highest N=2 trial disagreement with all panel models covered"
            ),
            "n_seeded_records": len(seed_rows),
        },
        workers=args.workers,
        overwrite=False,
        retry_errors=args.retry_errors,
    )
    summary = analyze(rows, selected)
    summary["n_seeded_records"] = len(seed_rows)
    summary["n_new_calls_planned"] = len(work) * len(JUDGES) - len(seed_rows)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(f"Wrote judge pass-rate diagnostic to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
