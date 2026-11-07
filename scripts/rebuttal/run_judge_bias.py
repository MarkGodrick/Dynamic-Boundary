"""Run response-level judge-bias experiments without regenerating responses.

The runner imports the production prompt templates and parser from
``judges.llm_judge``. It supports two experiments:

* ``human``: compare one or more judges with canonical response-level human
  pass/fail labels.
* ``inter``: rejudge stored response matrices with the same judges, report
  pairwise agreement, and fit one Rasch scale per judge on the common
  successful verdicts.

All outputs are append/resume safe. ``smoke`` exercises both paths with fake
judges and tiny local fixtures; it never calls an API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.trial_utils import judge_and_capture  # noqa: E402
from difficulty.rasch import fit_mle, fit_ols  # noqa: E402
from judges.llm_judge import JUDGE_TEMPLATES, LLMJudge  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/rebuttal/judge_bias.yaml"
VALID_CATEGORIES = frozenset(JUDGE_TEMPLATES)


@dataclass(frozen=True)
class WorkItem:
    experiment: str
    category: str
    sample_id: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any]
    source_file: str
    human_pass: bool | None = None
    reference_label_source: str | None = None
    source_model: str | None = None
    item_id: str | None = None
    trial_index: int | None = None

    @property
    def work_key(self) -> str:
        if self.experiment == "judge_human":
            return f"human|{self.category}|{self.sample_id}"
        return (
            f"inter|{self.category}|{self.source_model}|{self.item_id}|"
            f"{self.trial_index}"
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(_json_safe(value), ensure_ascii=False) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def template_sha256(category: str) -> str:
    template = JUDGE_TEMPLATES[category]
    payload = json.dumps(template, ensure_ascii=False, sort_keys=True)
    return sha256_text(payload)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    judges = config.get("judges")
    if not isinstance(judges, dict) or not judges:
        raise ValueError(f"{path}: expected a non-empty 'judges' mapping")
    required = {"model", "api_key_env"}
    for name, profile in judges.items():
        missing = required - set(profile or {})
        if missing:
            raise ValueError(f"Judge {name!r} is missing {sorted(missing)}")
    return config


def selected_profiles(
    config: dict[str, Any], requested: list[str] | None
) -> dict[str, dict[str, Any]]:
    profiles = config["judges"]
    names = requested or list(profiles)
    unknown = set(names) - set(profiles)
    if unknown:
        raise ValueError(
            f"Unknown judges {sorted(unknown)}; available: {sorted(profiles)}"
        )
    return {name: dict(profiles[name]) for name in names}


def normalize_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages")
    if messages is not None:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        normalized = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                raise ValueError("messages require string user/assistant turns")
            normalized.append({"role": role, "content": content})
        if normalized[-1]["role"] != "assistant":
            raise ValueError("the final message must be an assistant response")
        return normalized

    prompt = record.get("prompt") or record.get("question")
    response = (
        record.get("response")
        or record.get("target_response")
        or record.get("last_response")
    )
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise ValueError("record needs messages or a prompt/response pair")
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def canonical_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    for key in ("correct_answer", "wrong_answer"):
        if key in record:
            metadata[key] = record[key]
    return metadata


def parse_dataset_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("--dataset must use CATEGORY=PATH")
    category, raw_path = spec.split("=", 1)
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Unsupported category {category!r}; expected {sorted(VALID_CATEGORIES)}"
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return category, path


def build_human_work_items(
    dataset_specs: list[str], label_field: str
) -> tuple[list[WorkItem], list[dict[str, str]]]:
    work: list[WorkItem] = []
    inputs: list[dict[str, str]] = []
    seen: set[str] = set()
    for spec in dataset_specs:
        category, path = parse_dataset_spec(spec)
        records = load_jsonl(path)
        inputs.append({"path": str(path), "sha256": sha256_file(path)})
        for index, record in enumerate(records):
            sample_id = str(record.get("id") or record.get("sample_id") or index)
            identity = f"{category}|{sample_id}"
            if identity in seen:
                raise ValueError(f"Duplicate human sample {identity}")
            seen.add(identity)
            label = record.get(label_field)
            if isinstance(label, bool):
                human_pass = label
            elif label in (0, 1):
                human_pass = bool(label)
            else:
                raise ValueError(
                    f"{path}:{index + 1}: {label_field!r} must be pass=1/fail=0"
                )
            work.append(
                WorkItem(
                    experiment="judge_human",
                    category=category,
                    sample_id=sample_id,
                    messages=normalize_messages(record),
                    metadata=canonical_metadata(record),
                    source_file=str(path),
                    human_pass=human_pass,
                    reference_label_source=str(
                        record.get("final_label_source")
                        or record.get("human_label_source")
                        or record.get("annotation_source")
                        or "unspecified"
                    ),
                    item_id=sample_id,
                )
            )
    return work, inputs


def _even_beta_sample(
    item_ids: list[str],
    item_bank: dict[str, dict[str, Any]],
    max_items: int | None,
) -> list[str]:
    if max_items is None or max_items <= 0 or max_items >= len(item_ids):
        return sorted(item_ids)
    ordered = sorted(
        item_ids,
        key=lambda item_id: (
            float(
                item_bank[item_id].get(
                    "beta",
                    (item_bank[item_id].get("metadata") or {}).get("beta", 0.0),
                )
            ),
            item_id,
        ),
    )
    if max_items == 1:
        return [ordered[len(ordered) // 2]]
    positions = {
        round(index * (len(ordered) - 1) / (max_items - 1))
        for index in range(max_items)
    }
    return [ordered[index] for index in sorted(positions)]


def _trial_order(
    n_trials: int, *, seed: int, model: str, item_id: str
) -> list[int]:
    indices = list(range(n_trials))
    stable_seed = int(
        sha256_text(f"{seed}|{model}|{item_id}")[:16],
        16,
    )
    random.Random(stable_seed).shuffle(indices)
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


def build_inter_work_items(
    *,
    category: str,
    matrix_root: Path,
    item_bank_path: Path,
    max_items: int | None,
    trials_per_cell: int,
    seed: int,
) -> tuple[list[WorkItem], list[dict[str, str]], dict[str, Any]]:
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Unsupported category {category!r}")
    if trials_per_cell < 1:
        raise ValueError("trials_per_cell must be >= 1")
    item_records = load_jsonl(item_bank_path)
    item_bank = {
        str(record["prompt_id"]): record
        for record in item_records
        if "prompt_id" in record
    }
    if len(item_bank) != len(item_records):
        raise ValueError("item bank has missing or duplicate prompt_id values")

    selected_ids = set(
        _even_beta_sample(sorted(item_bank), item_bank, max_items)
    )
    matrix_paths: list[Path] = []
    available_ids: set[str] = set()
    selected_seen: set[str] = set()
    work: list[WorkItem] = []
    skipped_trials = 0
    for model_dir in sorted(matrix_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        matrix_path = model_dir / "response_matrix.jsonl"
        if not matrix_path.is_file():
            continue
        matrix_paths.append(matrix_path)
        metadata_path = model_dir / "metadata.json"
        model_name = model_dir.name
        if metadata_path.is_file():
            model_name = str(
                json.loads(metadata_path.read_text(encoding="utf-8")).get(
                    "model", model_name
                )
            )
        with matrix_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                item_id = str(entry["prompt_id"])
                available_ids.add(item_id)
                if item_id not in selected_ids:
                    continue
                selected_seen.add(item_id)
                trials = entry.get("trials") or []
                order = _trial_order(
                    len(trials), seed=seed, model=model_name, item_id=item_id
                )
                for trial_index in order[:trials_per_cell]:
                    trial = trials[trial_index]
                    try:
                        messages = reconstruct_trial_messages(
                            item_bank[item_id], trial
                        )
                    except ValueError:
                        skipped_trials += 1
                        continue
                    work.append(
                        WorkItem(
                            experiment="inter_judge",
                            category=category,
                            sample_id=f"{model_name}|{item_id}|{trial_index}",
                            messages=messages,
                            metadata=canonical_metadata(item_bank[item_id]),
                            source_file=str(matrix_path),
                            source_model=model_name,
                            item_id=item_id,
                            trial_index=trial_index,
                        )
                    )
    if not matrix_paths:
        raise ValueError(f"No response_matrix.jsonl found under {matrix_root}")
    missing_selected = selected_ids - selected_seen
    if missing_selected:
        preview = sorted(missing_selected)[:5]
        raise ValueError(
            f"{len(missing_selected)} selected item IDs missing from matrices: "
            f"{preview}"
        )
    manifest_inputs = [
        {"path": str(item_bank_path), "sha256": sha256_file(item_bank_path)},
        {
            "path": str(matrix_root),
            "sha256": sha256_text(
                "\n".join(
                    sorted(
                        f"{path}:{sha256_file(path)}"
                        for path in matrix_paths
                    )
                )
            ),
        },
    ]
    selection = {
        "category": category,
        "n_available_items": len(available_ids),
        "n_selected_items": len(selected_ids),
        "selected_item_ids": sorted(selected_ids),
        "trials_per_cell": trials_per_cell,
        "seed": seed,
        "n_work_items": len(work),
        "n_skipped_trials_without_reconstructable_response": skipped_trials,
        "item_sampling": "evenly_spaced_on_stored_beta",
    }
    return work, manifest_inputs, selection


class FakeJudge:
    """Deterministic no-network judge used only by the smoke command/tests."""

    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self._last_label: str | None = None
        self._last_meta: dict[str, Any] = {}

    def judge(
        self, messages: list[dict[str, str]], metadata: dict[str, Any] | None
    ) -> bool:
        work_key = str((metadata or {}).get("_judge_bias_work_key", ""))
        base = int(sha256_text(work_key)[:8], 16) % 7
        offset = int(sha256_text(self.name)[:8], 16) % 3
        passed = (base + offset) % 7 not in {0, 1}
        labels = {
            "harmful_refusal": ("safe", "unsafe"),
            "over_refusal": ("appropriate", "over_refusal"),
            "sycophancy_resistance": ("pass", "fail"),
        }
        self._last_label = labels[self.category][0 if passed else 1]
        self._last_meta = {
            "finish_reason": "stop",
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 100,
            "truncated": False,
            "raw_text": f"Answer: {self._last_label}",
            "served_model": f"fake-{self.name}",
        }
        return passed

    def get_last_error(self) -> None:
        return None

    def get_last_label(self) -> str | None:
        return self._last_label

    def get_last_meta(self) -> dict[str, Any]:
        return self._last_meta


def make_judge(
    name: str,
    profile: dict[str, Any],
    category: str,
    *,
    fake: bool,
) -> Any:
    if fake:
        return FakeJudge(name, category)
    return LLMJudge(
        model=profile["model"],
        api_key_env=profile["api_key_env"],
        base_url=profile.get("base_url"),
        category=category,
        max_retries=int(profile.get("max_retries", 8)),
        retry_delay=float(profile.get("retry_delay", 0.5)),
        temperature=profile.get("temperature", 0.0),
        max_tokens=profile.get("max_tokens", 1024),
        extra_body=profile.get("extra_body"),
        request_kwargs=profile.get("request_kwargs"),
    )


def estimate_cost(
    meta: dict[str, Any] | None, profile: dict[str, Any]
) -> float | None:
    if not meta:
        return None
    prices = profile.get("prices_usd_per_million") or {}
    if not prices:
        return None
    prompt = meta.get("prompt_tokens")
    completion = meta.get("completion_tokens")
    if prompt is None or completion is None:
        return None
    cache_hit = meta.get("cache_hit_tokens")
    cache_miss = meta.get("cache_miss_tokens")
    if cache_hit is None or cache_miss is None:
        cache_hit = 0
        cache_miss = prompt
    return (
        float(cache_hit) * float(prices.get("input_cache_hit", prices["input"]))
        + float(cache_miss) * float(prices.get("input_cache_miss", prices["input"]))
        + float(completion) * float(prices["output"])
    ) / 1_000_000


def _judge_one(
    item: WorkItem,
    judge_name: str,
    judge: Any,
    profile: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    metadata = dict(item.metadata)
    metadata["_judge_bias_work_key"] = item.work_key
    try:
        passed, outcome, error, label, meta = judge_and_capture(
            judge, item.messages, metadata
        )
    except Exception as exc:
        passed, outcome, label, meta = None, "runner_exception", None, None
        error = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    record_key = f"{item.work_key}|{judge_name}"
    return {
        "record_key": record_key,
        "work_key": item.work_key,
        "experiment": item.experiment,
        "category": item.category,
        "sample_id": item.sample_id,
        "source_file": item.source_file,
        "source_model": item.source_model,
        "item_id": item.item_id,
        "trial_index": item.trial_index,
        "human_pass": item.human_pass,
        "reference_label_source": item.reference_label_source,
        "judge_name": judge_name,
        "requested_model": profile["model"],
        "template_sha256": template_sha256(item.category),
        "messages_sha256": sha256_text(
            json.dumps(item.messages, ensure_ascii=False, sort_keys=True)
        ),
        "passed": passed,
        "judge_label": label,
        "judge_outcome": outcome,
        "judge_error": error,
        "judge_meta": meta,
        "elapsed_seconds": time.perf_counter() - started,
        "estimated_cost_usd": estimate_cost(meta, profile),
        "judged_at": utc_now(),
    }


def latest_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        latest[record["record_key"]] = record
    return latest


def run_work_items(
    *,
    work_items: list[WorkItem],
    profiles: dict[str, dict[str, Any]],
    output_dir: Path,
    inputs: list[dict[str, str]],
    selection: dict[str, Any] | None,
    workers: int,
    overwrite: bool,
    retry_errors: bool,
    fake: bool = False,
) -> list[dict[str, Any]]:
    run_started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = output_dir / "judgments.jsonl"
    if overwrite and judgments_path.exists():
        judgments_path.unlink()
    existing = latest_records(judgments_path)
    for record in existing.values():
        profile = profiles.get(record["judge_name"])
        if profile and record.get("requested_model") != profile["model"]:
            raise ValueError(
                "Existing output uses a different requested model for "
                f"{record['judge_name']}; choose a fresh output directory"
            )

    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "template_source": "judges.llm_judge.JUDGE_TEMPLATES",
        "template_sha256": {
            category: template_sha256(category)
            for category in sorted({item.category for item in work_items})
        },
        "judges": profiles,
        "inputs": inputs,
        "selection": selection,
        "n_work_items": len(work_items),
        "n_requested_calls": len(work_items) * len(profiles),
        "resume": {
            "overwrite": overwrite,
            "retry_errors": retry_errors,
            "n_existing_records": len(existing),
        },
        "fake_smoke_run": fake,
    }
    write_json(output_dir / "manifest.json", manifest)

    judges = {
        (category, name): make_judge(
            name, profile, category, fake=fake
        )
        for category in sorted({item.category for item in work_items})
        for name, profile in profiles.items()
    }
    tasks: list[tuple[WorkItem, str]] = []
    for item in work_items:
        for judge_name in profiles:
            key = f"{item.work_key}|{judge_name}"
            old = existing.get(key)
            if old is None or (
                retry_errors and old.get("judge_outcome") != "ok"
            ):
                tasks.append((item, judge_name))

    print(
        f"Prepared {len(work_items)} stored responses x {len(profiles)} judges; "
        f"{len(tasks)} API calls remain"
    )
    with judgments_path.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _judge_one,
                    item,
                    judge_name,
                    judges[(item.category, judge_name)],
                    profiles[judge_name],
                ): (item, judge_name)
                for item, judge_name in tasks
            }
            completed = 0
            for future in as_completed(futures):
                record = future.result()
                output.write(
                    json.dumps(_json_safe(record), ensure_ascii=False) + "\n"
                )
                output.flush()
                existing[record["record_key"]] = record
                completed += 1
                if completed % 100 == 0 or completed == len(tasks):
                    print(f"  completed {completed}/{len(tasks)}")
    manifest["completed_at"] = utc_now()
    manifest["run_wall_seconds"] = time.perf_counter() - run_started
    manifest["n_new_calls"] = len(tasks)
    manifest["n_latest_records"] = len(existing)
    write_json(output_dir / "manifest.json", manifest)
    return list(existing.values())


def _counter_dict(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def usage_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "prompt_tokens",
        "completion_tokens",
        "cache_hit_tokens",
        "cache_miss_tokens",
    )
    output: dict[str, Any] = {"calls": len(group)}
    for field in fields:
        values = [
            row["judge_meta"].get(field)
            for row in group
            if row.get("judge_meta")
            and row["judge_meta"].get(field) is not None
        ]
        output[field] = sum(values) if values else None
        output[f"{field}_reported_calls"] = len(values)
    return output


def analyze_human(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("experiment") == "judge_human":
            groups[(row["category"], row["judge_name"])].append(row)
    results = []
    for (category, judge_name), group in sorted(groups.items()):
        valid = [
            row
            for row in group
            if isinstance(row.get("passed"), bool)
            and isinstance(row.get("human_pass"), bool)
        ]
        # Positive class is category failure, not pass.
        tp = sum(not row["human_pass"] and not row["passed"] for row in valid)
        fn = sum(not row["human_pass"] and row["passed"] for row in valid)
        fp = sum(row["human_pass"] and not row["passed"] for row in valid)
        tn = sum(row["human_pass"] and row["passed"] for row in valid)
        n = len(valid)
        results.append(
            {
                "category": category,
                "judge_name": judge_name,
                "requested_model": group[0]["requested_model"],
                "n_requested": len(group),
                "n_valid": n,
                "agreement": (tp + tn) / n if n else None,
                "fnr_failure_positive": fn / (tp + fn) if tp + fn else None,
                "fpr_failure_positive": fp / (fp + tn) if fp + tn else None,
                "confusion_failure_positive": {
                    "tp_human_fail_judge_fail": tp,
                    "fn_human_fail_judge_pass": fn,
                    "fp_human_pass_judge_fail": fp,
                    "tn_human_pass_judge_pass": tn,
                },
                "outcomes": _counter_dict(row["judge_outcome"] for row in group),
                "reference_label_sources": _counter_dict(
                    row.get("reference_label_source", "unspecified")
                    for row in group
                ),
                "usage": usage_summary(group),
                "estimated_cost_usd": sum(
                    row.get("estimated_cost_usd") or 0.0 for row in group
                ),
                "wall_seconds_sum": sum(
                    float(row.get("elapsed_seconds") or 0.0) for row in group
                ),
            }
        )
    return {
        "experiment": "judge_human",
        "metric_definition": (
            "failure is the positive class: FNR=P(judge pass|human fail), "
            "FPR=P(judge fail|human pass)"
        ),
        "results": results,
    }


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    value = float(spearmanr(left, right)[0])
    return value if math.isfinite(value) else None


def _pairwise_agreement(
    by_judge: dict[str, dict[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    output = []
    names = sorted(by_judge)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = by_judge[left_name]
            right = by_judge[right_name]
            shared = sorted(set(left) & set(right))
            pairs = [
                (left[key]["passed"], right[key]["passed"])
                for key in shared
                if isinstance(left[key].get("passed"), bool)
                and isinstance(right[key].get("passed"), bool)
            ]
            agreement = (
                sum(a == b for a, b in pairs) / len(pairs) if pairs else None
            )
            output.append(
                {
                    "judge_a": left_name,
                    "judge_b": right_name,
                    "n_shared_valid": len(pairs),
                    "agreement": agreement,
                }
            )
    return output


def fit_common_rasch(
    rows: list[dict[str, Any]],
    judge_names: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    judge_names = sorted(judge_names)
    by_judge: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in judge_names
    }
    for row in rows:
        name = row.get("judge_name")
        if (
            row.get("experiment") == "inter_judge"
            and name in by_judge
            and isinstance(row.get("passed"), bool)
        ):
            by_judge[name][row["work_key"]] = row
    common_keys = set.intersection(
        *(set(by_judge[name]) for name in judge_names)
    ) if judge_names else set()
    exemplar = by_judge[judge_names[0]] if judge_names else {}
    models = sorted({exemplar[key]["source_model"] for key in common_keys})
    items = sorted({exemplar[key]["item_id"] for key in common_keys})
    model_index = {value: index for index, value in enumerate(models)}
    item_index = {value: index for index, value in enumerate(items)}
    if len(models) < 2 or len(items) < 2:
        return (
            {
                "status": "skipped",
                "reason": "need at least two models and two items",
                "n_common_verdicts": len(common_keys),
            },
            _pairwise_agreement(by_judge),
        )

    scales: dict[str, dict[str, Any]] = {}
    for judge_name in judge_names:
        passed = np.zeros((len(models), len(items)), dtype=float)
        totals = np.zeros_like(passed)
        for key in common_keys:
            row = by_judge[judge_name][key]
            model_i = model_index[row["source_model"]]
            item_i = item_index[row["item_id"]]
            passed[model_i, item_i] += int(row["passed"])
            totals[model_i, item_i] += 1
        missing_cells = int(np.sum(totals == 0))
        if missing_cells:
            return (
                {
                    "status": "skipped",
                    "reason": "common verdicts do not form a complete model-item matrix",
                    "missing_cells": missing_cells,
                    "n_common_verdicts": len(common_keys),
                },
                _pairwise_agreement(by_judge),
            )
        p_hat = passed / totals
        theta0, beta0 = fit_ols(p_hat)
        theta, beta, nll = fit_mle(passed, totals, theta0, beta0)
        scales[judge_name] = {
            "theta": {model: theta[index] for index, model in enumerate(models)},
            "beta": {item: beta[index] for index, item in enumerate(items)},
            "neg_loglik": nll,
        }

    comparisons = []
    for left_index, left_name in enumerate(judge_names):
        for right_name in judge_names[left_index + 1 :]:
            left_theta = np.array(
                [scales[left_name]["theta"][model] for model in models]
            )
            right_theta = np.array(
                [scales[right_name]["theta"][model] for model in models]
            )
            left_beta = np.array(
                [scales[left_name]["beta"][item] for item in items]
            )
            right_beta = np.array(
                [scales[right_name]["beta"][item] for item in items]
            )
            delta_theta = right_theta - left_theta
            delta_beta = right_beta - left_beta
            comparisons.append(
                {
                    "judge_a": left_name,
                    "judge_b": right_name,
                    "spearman_rho_beta": _safe_spearman(left_beta, right_beta),
                    "spearman_rho_theta": _safe_spearman(
                        left_theta, right_theta
                    ),
                    "mean_beta_shift_b_minus_a": float(np.mean(delta_beta)),
                    "mean_theta_shift_b_minus_a": float(np.mean(delta_theta)),
                    "beta_nonuniform_residual_sd": float(
                        np.std(delta_beta - np.mean(delta_beta))
                    ),
                    "theta_nonuniform_residual_sd": float(
                        np.std(delta_theta - np.mean(delta_theta))
                    ),
                }
            )
    return (
        {
            "status": "ok",
            "gauge": "sum(theta)=0 fitted separately for every judge",
            "n_models": len(models),
            "n_items": len(items),
            "n_common_trial_verdicts": len(common_keys),
            "models": models,
            "items": items,
            "scales": scales,
            "comparisons": comparisons,
        },
        _pairwise_agreement(by_judge),
    )


def analyze_inter(
    rows: list[dict[str, Any]], judge_names: list[str]
) -> dict[str, Any]:
    rasch, pairwise = fit_common_rasch(rows, judge_names)
    selected = [row for row in rows if row.get("experiment") == "inter_judge"]
    per_judge = []
    for name in judge_names:
        group = [row for row in selected if row.get("judge_name") == name]
        per_judge.append(
            {
                "judge_name": name,
                "requested_model": group[0]["requested_model"] if group else None,
                "n_requested": len(group),
                "n_valid": sum(isinstance(row.get("passed"), bool) for row in group),
                "outcomes": _counter_dict(row["judge_outcome"] for row in group),
                "usage": usage_summary(group),
                "estimated_cost_usd": sum(
                    row.get("estimated_cost_usd") or 0.0 for row in group
                ),
                "wall_seconds_sum": sum(
                    float(row.get("elapsed_seconds") or 0.0) for row in group
                ),
            }
        )
    return {
        "experiment": "inter_judge",
        "pairwise_agreement": pairwise,
        "rasch": rasch,
        "per_judge": per_judge,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    if summary["experiment"] == "judge_human":
        lines = [
            "# Judge-human alignment",
            "",
            "Positive class: category failure. Therefore FNR is a human failure "
            "judged as pass, and FPR is a human pass judged as failure.",
            "",
            "| Category | Judge | n | Agreement | FNR | FPR |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for result in summary["results"]:
            fmt = lambda value: "NA" if value is None else f"{value:.4f}"
            lines.append(
                f"| {result['category']} | {result['judge_name']} | "
                f"{result['n_valid']} | {fmt(result['agreement'])} | "
                f"{fmt(result['fnr_failure_positive'])} | "
                f"{fmt(result['fpr_failure_positive'])} |"
            )
        return "\n".join(lines) + "\n"

    lines = [
        "# Inter-judge alignment and Rasch invariance",
        "",
        "| Judge pair | Agreement | rho(beta) | rho(theta) | "
        "beta shift | theta shift | beta residual SD | theta residual SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    scale_comparisons = {
        (row["judge_a"], row["judge_b"]): row
        for row in summary.get("rasch", {}).get("comparisons", [])
    }
    fmt = lambda value: "NA" if value is None else f"{value:.4f}"
    for agreement in summary["pairwise_agreement"]:
        key = (agreement["judge_a"], agreement["judge_b"])
        scale = scale_comparisons.get(key, {})
        lines.append(
            f"| {key[0]} vs {key[1]} | {fmt(agreement['agreement'])} | "
            f"{fmt(scale.get('spearman_rho_beta'))} | "
            f"{fmt(scale.get('spearman_rho_theta'))} | "
            f"{fmt(scale.get('mean_beta_shift_b_minus_a'))} | "
            f"{fmt(scale.get('mean_theta_shift_b_minus_a'))} | "
            f"{fmt(scale.get('beta_nonuniform_residual_sd'))} | "
            f"{fmt(scale.get('theta_nonuniform_residual_sd'))} |"
        )
    return "\n".join(lines) + "\n"


def save_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    summary["generated_at"] = utc_now()
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )


def build_smoke_fixtures(root: Path) -> tuple[Path, Path, Path]:
    fixture_dir = root / "fixtures"
    human_path = fixture_dir / "human.jsonl"
    human = []
    for index in range(6):
        human.append(
            {
                "id": f"human-{index}",
                "messages": [
                    {"role": "user", "content": f"Question {index}"},
                    {"role": "assistant", "content": f"Answer {index}"},
                ],
                "final_label": int(index % 3 != 0),
                "correct_answer": "correct",
                "wrong_answer": "wrong",
            }
        )
    write_jsonl(human_path, human)

    item_bank_path = fixture_dir / "items.jsonl"
    items = [
        {
            "prompt_id": f"item-{item_index}",
            "prompt": f"Prompt {item_index}",
            "beta": float(item_index - 2),
            "metadata": {"correct_answer": "correct", "wrong_answer": "wrong"},
        }
        for item_index in range(5)
    ]
    write_jsonl(item_bank_path, items)
    matrix_root = fixture_dir / "matrix"
    for model_index in range(3):
        model_dir = matrix_root / f"model-{model_index}"
        model_dir.mkdir(parents=True, exist_ok=True)
        write_json(model_dir / "metadata.json", {"model": f"model-{model_index}"})
        entries = []
        for item_index in range(5):
            entries.append(
                {
                    "prompt_id": f"item-{item_index}",
                    "trials": [
                        {"response": f"Response {model_index}-{item_index}-{trial}"}
                        for trial in range(2)
                    ],
                }
            )
        write_jsonl(model_dir / "response_matrix.jsonl", entries)
    return human_path, matrix_root, item_bank_path


def run_smoke(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if args.output_dir:
        root = args.output_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = Path(tempfile.mkdtemp(prefix="boundary_judge_bias_smoke_"))
    human_path, matrix_root, item_bank_path = build_smoke_fixtures(root)
    profiles = selected_profiles(config, list(config["judges"])[:3])

    human_work, human_inputs = build_human_work_items(
        [f"sycophancy_resistance={human_path}"], "final_label"
    )
    human_dir = root / "human"
    human_rows = run_work_items(
        work_items=human_work,
        profiles=profiles,
        output_dir=human_dir,
        inputs=human_inputs,
        selection=None,
        workers=3,
        overwrite=True,
        retry_errors=False,
        fake=True,
    )
    save_summary(human_dir, analyze_human(human_rows))

    inter_work, inter_inputs, selection = build_inter_work_items(
        category="sycophancy_resistance",
        matrix_root=matrix_root,
        item_bank_path=item_bank_path,
        max_items=5,
        trials_per_cell=2,
        seed=42,
    )
    inter_dir = root / "inter"
    inter_rows = run_work_items(
        work_items=inter_work,
        profiles=profiles,
        output_dir=inter_dir,
        inputs=inter_inputs,
        selection=selection,
        workers=3,
        overwrite=True,
        retry_errors=False,
        fake=True,
    )
    inter_summary = analyze_inter(inter_rows, list(profiles))
    save_summary(inter_dir, inter_summary)
    if inter_summary["rasch"].get("status") != "ok":
        raise RuntimeError(f"smoke Rasch fit failed: {inter_summary['rasch']}")
    print(f"Smoke test passed. Outputs: {root}")
    return 0


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--judges", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the call plan without using credentials",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry existing non-ok records; successful records are always resumed",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    human = subparsers.add_parser("human", help="Judge-human alignment")
    add_run_args(human)
    human.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Canonical response-level input as CATEGORY=PATH; repeatable",
    )
    human.add_argument(
        "--label-field",
        default="final_label",
        help="Field with canonical pass=1/fail=0 human verdict",
    )

    inter = subparsers.add_parser("inter", help="Inter-judge + Rasch")
    add_run_args(inter)
    inter.add_argument("--category", required=True, choices=sorted(VALID_CATEGORIES))
    inter.add_argument("--matrix-root", type=Path, required=True)
    inter.add_argument("--item-bank", type=Path, required=True)
    inter.add_argument("--max-items", type=int, default=100)
    inter.add_argument("--trials-per-cell", type=int, default=1)
    inter.add_argument("--seed", type=int, default=20260727)

    smoke = subparsers.add_parser("smoke", help="No-network end-to-end smoke test")
    smoke.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    smoke.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    if args.command == "smoke":
        return run_smoke(args, config)

    requested_judges = args.judges
    if args.command == "human" and requested_judges is None:
        requested_judges = [
            config.get("production_judge", next(iter(config["judges"])))
        ]
    profiles = selected_profiles(config, requested_judges)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    output_dir = args.output_dir.resolve()
    if args.command == "human":
        work, inputs = build_human_work_items(args.dataset, args.label_field)
        if args.dry_run:
            counts = Counter(item.category for item in work)
            print(
                f"Dry run valid: {len(work)} reference-labeled responses; "
                f"categories={dict(counts)}; judges={list(profiles)}; "
                f"planned_calls={len(work) * len(profiles)}"
            )
            return 0
        rows = run_work_items(
            work_items=work,
            profiles=profiles,
            output_dir=output_dir,
            inputs=inputs,
            selection=None,
            workers=args.workers,
            overwrite=args.overwrite,
            retry_errors=args.retry_errors,
        )
        save_summary(output_dir, analyze_human(rows))
    else:
        work, inputs, selection = build_inter_work_items(
            category=args.category,
            matrix_root=args.matrix_root.resolve(),
            item_bank_path=args.item_bank.resolve(),
            max_items=args.max_items,
            trials_per_cell=args.trials_per_cell,
            seed=args.seed,
        )
        if args.dry_run:
            models = sorted({item.source_model for item in work})
            print(
                f"Dry run valid: {len(work)} stored trial responses; "
                f"models={len(models)}; items={selection['n_selected_items']}; "
                f"judges={list(profiles)}; "
                f"planned_calls={len(work) * len(profiles)}"
            )
            return 0
        rows = run_work_items(
            work_items=work,
            profiles=profiles,
            output_dir=output_dir,
            inputs=inputs,
            selection=selection,
            workers=args.workers,
            overwrite=args.overwrite,
            retry_errors=args.retry_errors,
        )
        save_summary(output_dir, analyze_inter(rows, list(profiles)))
    print(f"Wrote judge-bias outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
