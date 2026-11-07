"""Build response-level human-reference datasets for judge-bias experiments.

This converter creates the canonical ``final_label`` convention consumed by
``scripts/rebuttal/run_judge_bias.py``:

* ``1``: the stored assistant response passes the category criterion;
* ``0``: the stored assistant response fails the category criterion.

The harmful-refusal source is JailbreakBench's judge-comparison split. The
over-refusal source is the original-English fields preserved in the public
WildGuardTestJP mirror; its labels and 1,725 source pairs come from
WildGuardTest, whose official repository is gated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, load_dataset


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE = ROOT / ".cache/huggingface"
DEFAULT_OUTPUT_ROOT = ROOT / "dataset/judge_bias"

JBB_DATASET = "JailbreakBench/JBB-Behaviors"
JBB_CONFIG = "judge_comparison"
JBB_SPLIT = "test"
JBB_REVISION = "886acc352a31533ffbcf4ef22c744658688086fc"

WILDGUARD_ORIGINAL_DATASET = "allenai/wildguardmix"
WILDGUARD_ORIGINAL_CONFIG = "wildguardtest"
WILDGUARD_ORIGINAL_SPLIT = "test"
WILDGUARD_MIRROR_DATASET = "sbintuitions/WildGuardTestJP"
WILDGUARD_MIRROR_SPLIT = "test"
WILDGUARD_MIRROR_REVISION = "2223033c49ef3f32724b6d4570543f2b7bc9fe51"


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(dataset: Dataset, required: set[str], source: str) -> None:
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def convert_jbb(dataset: Dataset) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Canonicalize JBB human jailbreak verdicts to harmful-refusal pass labels."""
    required = {
        "Index",
        "goal",
        "prompt",
        "target_response",
        "human1",
        "human2",
        "human3",
        "human_majority",
    }
    require_columns(dataset, required, JBB_DATASET)

    records: list[dict[str, Any]] = []
    vote_patterns: Counter[str] = Counter()
    for source_row, row in enumerate(dataset):
        votes = [int(row[f"human{i}"]) for i in range(1, 4)]
        majority = int(row["human_majority"])
        if any(vote not in (0, 1) for vote in votes) or majority not in (0, 1):
            raise ValueError(f"JBB row {source_row} has a non-binary human label")
        computed_majority = int(sum(votes) >= 2)
        if computed_majority != majority:
            raise ValueError(
                f"JBB row {source_row}: human_majority={majority} "
                f"but votes={votes}"
            )

        # JBB positive = jailbreak success/unsafe. The repository convention is
        # category pass, so an unsafe response must be inverted to zero.
        final_label = 1 - majority
        vote_patterns["".join(str(vote) for vote in votes)] += 1
        index = int(row["Index"])
        prompt = str(row["prompt"])
        response = str(row["target_response"])
        records.append(
            {
                "id": f"jbb-judge-comparison-{index:03d}",
                "category": "harmful_refusal",
                "source": JBB_DATASET,
                "source_config": JBB_CONFIG,
                "source_split": JBB_SPLIT,
                "source_revision": JBB_REVISION,
                "source_row": source_row,
                "source_index": index,
                "goal": str(row["goal"]),
                "prompt": prompt,
                "response": response,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "human_annotations": {
                    "human1_jailbreak_success": votes[0],
                    "human2_jailbreak_success": votes[1],
                    "human3_jailbreak_success": votes[2],
                    "human_majority_jailbreak_success": majority,
                },
                "human_label_source": "JBB three-human majority on exact response",
                "reference_raw_label": "unsafe" if majority else "safe",
                "final_label": final_label,
                "final_label_source": "JBB human_majority inverted to category pass",
                "metadata": {
                    "goal": str(row["goal"]),
                    "human_unsafe_votes": sum(votes),
                    "human_unanimous": len(set(votes)) == 1,
                },
            }
        )

    records.sort(key=lambda record: record["source_index"])
    summary = {
        "source_rows": len(dataset),
        "output_rows": len(records),
        "final_label_counts": dict(
            sorted(Counter(record["final_label"] for record in records).items())
        ),
        "human_majority_counts": dict(
            sorted(
                Counter(
                    record["human_annotations"][
                        "human_majority_jailbreak_success"
                    ]
                    for record in records
                ).items()
            )
        ),
        "unanimous_rows": sum(
            record["metadata"]["human_unanimous"] for record in records
        ),
        "vote_patterns": dict(sorted(vote_patterns.items())),
    }
    return records, summary


def convert_wildguard(
    dataset: Dataset,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select safe-prompt pairs and map WildGuard human labels to pass/fail."""
    required = {
        "prompt_original",
        "response_original",
        "adversarial",
        "prompt_harm_label",
        "response_refusal_label",
        "response_harm_label",
        "subcategory",
    }
    require_columns(dataset, required, WILDGUARD_MIRROR_DATASET)

    records: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    raw_labels: Counter[str] = Counter()
    for source_row, row in enumerate(dataset):
        if row["prompt_harm_label"] != "unharmful":
            excluded["prompt_not_human_labeled_unharmful"] += 1
            continue
        prompt = row["prompt_original"]
        response = row["response_original"]
        if not isinstance(prompt, str) or not prompt.strip():
            excluded["missing_prompt"] += 1
            continue
        if not isinstance(response, str) or not response.strip():
            excluded["missing_response"] += 1
            continue

        refusal = row["response_refusal_label"]
        harm = row["response_harm_label"]
        if refusal not in {"refusal", "compliance"}:
            excluded["missing_response_refusal_label"] += 1
            continue
        if harm not in {"harmful", "unharmful"}:
            excluded["missing_response_harm_label"] += 1
            continue

        # Match the production judge's precedence: harmful content is unsafe
        # even when the response also refuses; otherwise a refusal to a safe
        # prompt is over-refusal, and helpful safe compliance is appropriate.
        if harm == "harmful":
            raw_label = "unsafe"
            final_label = 0
        elif refusal == "refusal":
            raw_label = "over_refusal"
            final_label = 0
        else:
            raw_label = "appropriate"
            final_label = 1
        raw_labels[raw_label] += 1

        record = {
            "id": f"wildguardtest-safe-{source_row:04d}",
            "category": "over_refusal",
            "source": WILDGUARD_ORIGINAL_DATASET,
            "source_config": WILDGUARD_ORIGINAL_CONFIG,
            "source_split": WILDGUARD_ORIGINAL_SPLIT,
            "source_mirror": WILDGUARD_MIRROR_DATASET,
            "source_mirror_split": WILDGUARD_MIRROR_SPLIT,
            "source_mirror_revision": WILDGUARD_MIRROR_REVISION,
            "source_row": source_row,
            "prompt": prompt,
            "response": response,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "human_annotations": {
                "prompt_harm_label": row["prompt_harm_label"],
                "response_refusal_label": refusal,
                "response_harm_label": harm,
            },
            "human_label_source": (
                "WildGuardTest three-human response-level annotation"
            ),
            "reference_raw_label": raw_label,
            "final_label": final_label,
            "final_label_source": (
                "WildGuardTest safe-prompt refusal/harm labels canonicalized "
                "to category pass"
            ),
            "metadata": {
                "adversarial": bool(row["adversarial"]),
                "subcategory": row["subcategory"],
            },
        }
        for key in ("prompt_origin_model", "response_origin_model"):
            if key in dataset.column_names:
                record["metadata"][key] = row[key]
        records.append(record)

    summary = {
        "source_rows": len(dataset),
        "output_rows": len(records),
        "selection": (
            "prompt_harm_label=unharmful with non-empty original English "
            "prompt/response and complete human response labels"
        ),
        "final_label_counts": dict(
            sorted(Counter(record["final_label"] for record in records).items())
        ),
        "reference_raw_label_counts": dict(sorted(raw_labels.items())),
        "excluded_counts": dict(sorted(excluded.items())),
    }
    return records, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help="Hugging Face cache directory (default: project-local cache)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root containing harmful_refusal/ and over_refusal/",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow Hub access; default to cached/offline loading",
    )
    return parser.parse_args()


def load_sources(
    cache_dir: Path, *, online: bool
) -> tuple[Dataset, Dataset]:
    """Load pinned sources online or directly from their cached Arrow files."""
    if online:
        datasets_cache = cache_dir / "datasets"
        jbb = load_dataset(
            JBB_DATASET,
            JBB_CONFIG,
            split=JBB_SPLIT,
            revision=JBB_REVISION,
            cache_dir=str(datasets_cache),
        )
        wildguard = load_dataset(
            WILDGUARD_MIRROR_DATASET,
            split=WILDGUARD_MIRROR_SPLIT,
            revision=WILDGUARD_MIRROR_REVISION,
            cache_dir=str(datasets_cache),
        )
        return jbb, wildguard

    jbb_arrow = (
        cache_dir
        / "datasets/JailbreakBench___jbb-behaviors"
        / JBB_CONFIG
        / "0.0.0"
        / JBB_REVISION
        / "jbb-behaviors-test.arrow"
    )
    wildguard_arrow = (
        cache_dir
        / "datasets/sbintuitions___wild_guard_test_jp/default/0.0.0"
        / WILDGUARD_MIRROR_REVISION
        / "wild_guard_test_jp-test.arrow"
    )
    missing = [path for path in (jbb_arrow, wildguard_arrow) if not path.is_file()]
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Cached source data is missing. Re-run with --online first:\n" + paths
        )
    return Dataset.from_file(str(jbb_arrow)), Dataset.from_file(
        str(wildguard_arrow)
    )


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    jbb, wildguard = load_sources(cache_dir, online=args.online)

    jbb_records, jbb_summary = convert_jbb(jbb)
    wildguard_records, wildguard_summary = convert_wildguard(wildguard)
    outputs = {
        "harmful_refusal": (
            output_root / "harmful_refusal/final.jsonl",
            jbb_records,
            jbb_summary,
        ),
        "over_refusal": (
            output_root / "over_refusal/final.jsonl",
            wildguard_records,
            wildguard_summary,
        ),
    }
    combined_summary: dict[str, Any] = {
        "canonical_label_semantics": {
            "1": "response passes the category judge",
            "0": "response fails the category judge",
        },
        "datasets": {},
    }
    for category, (path, records, summary) in outputs.items():
        write_jsonl(path, records)
        summary["output_path"] = str(path)
        summary["output_sha256"] = sha256_file(path)
        combined_summary["datasets"][category] = summary
        print(
            f"{category}: wrote {len(records)} rows to {path} "
            f"(labels={summary['final_label_counts']})"
        )

    summary_path = output_root / "conversion_summary.json"
    write_json(summary_path, combined_summary)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
