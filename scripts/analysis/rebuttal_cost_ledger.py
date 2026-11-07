"""Build the rebuttal usage/cost ledger from stored experiment artifacts.

The script performs no model inference.  It reads trial JSONL files, counts
target/judge calls and provider-reported tokens, estimates legacy composer
tokens when usage metadata was not yet recorded, and emits:

    results/rebuttal/cost_ledger/usage_ledger.json
    results/rebuttal/cost_ledger/usage_ledger.md

Measured, tokenizer-estimated, proxy, and lower-bound quantities are labelled
in the JSON ledger.  The Markdown report contains both a stage-level audit
table and the compact HiES W2 table.

Run from the repository root:

    /data/whx/anaconda3/envs/boundary/bin/python \
        scripts/analysis/rebuttal_cost_ledger.py

Pass ``--composer-tokenizer /path/to/tokenizer.json`` to override the bundled
tokenizer.  By default the script uses
``third_party/tokenizers/deepseek_v3_tokenizer/tokenizer.json`` and its
DeepSeek chat framing.  It falls back to the locally cached
DeepSeek-R1-Distill-Llama-8B tokenizer only when the bundled tokenizer is
absent.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


CATEGORY_LABELS = {
    "harmful_refusal": "A (HRR)",
    "over_refusal": "$A^\\prime$ (OR)",
    "constrained_completion": "B (CIF)",
    "sycophancy_resistance": "C (SR)",
}

# DeepSeek v4-flash, verified 2026-07-26. Rates are CNY / 1M tokens.
PRICING = {
    "input_cache_hit_per_m": 0.02,
    "input_cache_miss_per_m": 1.00,
    "output_per_m": 2.00,
    "cny_per_usd": 7.15,
    "verified": "2026-07-26",
    "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
}

REFERENCE_DIRS = {
    "harmful_refusal": "results/harmful_refusal/filter_ablation_raw_v2_judgefix",
    "over_refusal": "results/over_refusal/filter_ablation_raw_v2_judgefix",
    "constrained_completion": "results/constrained_completion/filter_ablation_raw_v2",
    "sycophancy_resistance": "results/sycophancy_resistance/filter_ablation_raw_v2",
}

SGBS_DIRS = {
    "harmful_refusal": (
        "results/harmful_refusal/edge_case_validation/"
        "tiiuae_Falcon_H1R_7B_harmful_refusal_20260512_074354"
    ),
    "over_refusal": (
        "results/over_refusal/edge_case_validation/"
        "tiiuae_Falcon_H1R_7B_over_refusal_20260512_074408"
    ),
    "constrained_completion": (
        "results/constrained_completion/edge_case_validation/"
        "tiiuae_Falcon_H1R_7B_constrained_completion_20260606_123614"
    ),
    "sycophancy_resistance": (
        "results/sycophancy_resistance/edge_case_validation/"
        "microsoft_Phi_4_reasoning_plus_sycophancy_resistance_20260628_045920"
    ),
}

STEP2_DIRS = {
    "harmful_refusal": "results/harmful_refusal/claim3_step2/tiiuae_Falcon_H1R_7B",
    "over_refusal": "results/over_refusal/claim3_step2/tiiuae_Falcon_H1R_7B",
    "constrained_completion": (
        "results/constrained_completion/claim3_step2/tiiuae_Falcon_H1R_7B"
    ),
    "sycophancy_resistance": (
        "results/sycophancy_resistance/claim3_step2/"
        "microsoft_Phi_4_reasoning_plus"
    ),
}

STEP2_START_LOGS = {
    "harmful_refusal": (
        "results/harmful_refusal/claim3_step2/tiiuae_Falcon_H1R_7B/"
        "stage2_collect.log"
    ),
    # The SR panel run was interrupted and resumed. This is the first log of
    # the successful collection window; the final timestamp comes from the
    # last completed matrix artifact.
    "sycophancy_resistance": (
        "logs/runs/collect_sr_phi4r_20260708_155444.log"
    ),
}

# Claim-3 models classified as in-panel.  Each entry is (adaptive run, matrix
# model directory).  The latter is used only when a legacy run lacks
# trials.jsonl; exact selected item IDs are read from result.json.
IN_PANEL_RUNS = {
    "harmful_refusal": [
        (
            "results/adaptive/harmful_refusal/"
            "allenai_Olmo_3_7B_Instruct_harmful_refusal_eap_20260506_110251",
            "allenai_Olmo_3_7B_Instruct",
        ),
        (
            "results/adaptive/harmful_refusal/"
            "ibm_granite_granite_4.1_8b_harmful_refusal_eap_20260506_112443",
            "ibm_granite_granite_4.1_8b",
        ),
    ],
    "over_refusal": [
        (
            "results/adaptive/over_refusal/"
            "allenai_Olmo_3_7B_Instruct_over_refusal_eap_20260506_120850",
            "allenai_Olmo_3_7B_Instruct",
        ),
        (
            "results/adaptive/over_refusal/"
            "ibm_granite_granite_4.1_8b_over_refusal_eap_20260506_122412",
            "ibm_granite_granite_4.1_8b",
        ),
        (
            "results/adaptive/over_refusal/"
            "microsoft_phi_4_over_refusal_eap_20260506_120639",
            "microsoft_phi_4",
        ),
        (
            "results/adaptive/over_refusal/"
            "microsoft_Phi_4_reasoning_plus_over_refusal_eap_20260506_124441",
            "microsoft_Phi_4_reasoning_plus",
        ),
    ],
    "constrained_completion": [
        (
            "results/adaptive/constrained_completion/"
            "allenai_Olmo_3_7B_Instruct_constrained_completion_eap_20260506_130100",
            "allenai_Olmo_3_7B_Instruct",
        ),
        (
            "results/adaptive/constrained_completion/"
            "ibm_granite_granite_4.1_8b_constrained_completion_eap_20260506_131328",
            "ibm_granite_granite_4.1_8b",
        ),
        (
            "results/adaptive/constrained_completion/"
            "microsoft_phi_4_constrained_completion_eap_20260506_125856",
            "microsoft_phi_4",
        ),
        (
            "results/adaptive/constrained_completion/"
            "microsoft_Phi_4_reasoning_plus_constrained_completion_eap_20260506_130802",
            "microsoft_Phi_4_reasoning_plus",
        ),
        (
            "results/adaptive/constrained_completion/"
            "tiiuae_Falcon_H1R_7B_constrained_completion_eap_20260506_132040",
            "tiiuae_Falcon_H1R_7B",
        ),
    ],
    "sycophancy_resistance": [
        (
            "results/adaptive/sycophancy_resistance/"
            "allenai_Olmo_3_7B_Instruct_sycophancy_resistance_eap_20260506_135749",
            "allenai_Olmo_3_7B_Instruct",
        ),
        (
            "results/adaptive/sycophancy_resistance/"
            "ibm_granite_granite_4.1_8b_sycophancy_resistance_eap_20260506_142717",
            "ibm_granite_granite_4.1_8b",
        ),
        (
            "results/adaptive/sycophancy_resistance/"
            "microsoft_phi_4_sycophancy_resistance_eap_20260506_135429",
            "microsoft_phi_4",
        ),
        (
            "results/adaptive/sycophancy_resistance/"
            "tiiuae_Falcon_H1R_7B_sycophancy_resistance_eap_20260506_145053",
            "tiiuae_Falcon_H1R_7B",
        ),
    ],
}

OUT_PANEL_ADAPTIVE = {
    "harmful_refusal": (
        "results/adaptive/harmful_refusal/"
        "tiiuae_Falcon_H1R_7B_harmful_refusal_eap_20260506_114519",
        "tiiuae_Falcon_H1R_7B",
    ),
    "over_refusal": (
        "results/adaptive/over_refusal/"
        "tiiuae_Falcon_H1R_7B_over_refusal_eap_20260506_123315",
        "tiiuae_Falcon_H1R_7B",
    ),
    "constrained_completion": (
        "results/adaptive/constrained_completion/"
        "tiiuae_Falcon_H1R_7B_constrained_completion_eap_20260506_132040",
        "tiiuae_Falcon_H1R_7B",
    ),
    "sycophancy_resistance": (
        "results/adaptive/sycophancy_resistance/"
        "microsoft_Phi_4_reasoning_plus_sycophancy_resistance_eap_20260506_141339",
        "microsoft_Phi_4_reasoning_plus",
    ),
}


@dataclass
class Counter:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int | None = None
    cache_miss_tokens: int | None = None
    missing_token_calls: int = 0
    provenance: str = "measured"
    note: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "Counter") -> None:
        for key in ("calls", "prompt_tokens", "completion_tokens", "missing_token_calls"):
            setattr(self, key, getattr(self, key) + getattr(other, key))
        for key in ("cache_hit_tokens", "cache_miss_tokens"):
            value = getattr(other, key)
            if value is not None:
                current = getattr(self, key)
                setattr(self, key, value if current is None else current + value)
        if other.provenance != self.provenance:
            self.provenance = "mixed"


@dataclass
class Stage:
    category: str
    name: str
    artifact_paths: list[str]
    trial_units: int = 0
    target: Counter = field(default_factory=Counter)
    judge: Counter = field(default_factory=Counter)
    composer: Counter = field(default_factory=Counter)
    wall_seconds: float | None = None
    wall_kind: str = "measured"
    wall_note: str = ""

    @property
    def model_queries(self) -> int:
        return self.target.calls + self.judge.calls + self.composer.calls

    @property
    def total_tokens(self) -> int:
        return self.target.total_tokens + self.judge.total_tokens + self.composer.total_tokens

    @property
    def api_cost_usd(self) -> float:
        return price_usd(self.judge) + price_usd(self.composer)

    def add(self, other: "Stage") -> None:
        self.artifact_paths.extend(other.artifact_paths)
        self.trial_units += other.trial_units
        self.target.add(other.target)
        self.judge.add(other.judge)
        self.composer.add(other.composer)
        if other.wall_seconds is not None:
            self.wall_seconds = (self.wall_seconds or 0.0) + other.wall_seconds

    def to_dict(self) -> dict:
        result = asdict(self)
        result.update(
            model_queries=self.model_queries,
            total_tokens=self.total_tokens,
            api_cost_usd=round(self.api_cost_usd, 6),
        )
        return result


class TokenEstimator:
    """Small wrapper around tokenizers.Tokenizer, avoiding transformers/torch."""

    def __init__(
        self,
        tokenizer_path: Path,
        exact: bool,
        model_label: str,
        chat_format: str = "approximate",
    ):
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.path = tokenizer_path
        self.exact = exact
        self.model_label = model_label
        self.chat_format = chat_format

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text or "", add_special_tokens=False).ids)

    def count_messages(self, system: str, user: str) -> int:
        if self.chat_format == "deepseek_v3":
            # Exact two-message rendering from the bundled chat_template with
            # add_generation_prompt=True. Composer always sends one system and
            # one user message, so none of the tool/multi-assistant branches
            # in the general template apply here.
            rendered = (
                "<｜begin▁of▁sentence｜>"
                + system
                + "<｜User｜>"
                + user
                + "<｜Assistant｜>"
            )
            return self.count(rendered)
        # Explicit fallback for tokenizers that lack the composer chat
        # template: two wrappers plus assistant priming.
        return self.count(system) + self.count(user) + 8


def resolve_tokenizer(explicit: str | None) -> TokenEstimator:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"composer tokenizer not found: {path}")
        config_path = path.with_name("tokenizer_config.json")
        chat_format = "approximate"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            template = config.get("chat_template") or ""
            if "<｜User｜>" in template and "<｜Assistant｜>" in template:
                chat_format = "deepseek_v3"
        return TokenEstimator(
            path,
            True,
            "user-supplied composer tokenizer",
            chat_format=chat_format,
        )

    bundled = (
        REPO_ROOT
        / "third_party/tokenizers/deepseek_v3_tokenizer/tokenizer.json"
    )
    if bundled.exists():
        return TokenEstimator(
            bundled,
            True,
            "DeepSeek V3 tokenizer from deepseek_v3_tokenizer.zip",
            chat_format="deepseek_v3",
        )

    candidates = [
        Path("/data/whx/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B"),
        Path("/data1/whx/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B"),
    ]
    for root in candidates:
        matches = sorted(root.glob("snapshots/*/tokenizer.json"))
        if matches:
            return TokenEstimator(
                matches[-1],
                False,
                "deepseek-ai/DeepSeek-R1-Distill-Llama-8B (fallback proxy)",
            )
    raise FileNotFoundError(
        "No composer tokenizer found. Pass --composer-tokenizer tokenizer.json."
    )


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _add_optional(counter: Counter, key: str, value: int | None) -> None:
    if value is None:
        return
    current = getattr(counter, key)
    setattr(counter, key, value if current is None else current + value)


def aggregate_trial_records(records: Iterable[dict]) -> tuple[int, Counter, Counter]:
    """Aggregate trial units, target turns, and judge calls."""
    trial_units = 0
    target = Counter()
    judge = Counter()
    for record in records:
        for trial in record.get("trials") or []:
            trial_units += 1
            target_meta = trial.get("per_turn_meta")
            if target_meta:
                for meta in target_meta:
                    target.calls += 1
                    if not meta or meta.get("prompt_tokens") is None or meta.get("completion_tokens") is None:
                        target.missing_token_calls += 1
                        continue
                    target.prompt_tokens += meta["prompt_tokens"]
                    target.completion_tokens += meta["completion_tokens"]
                    _add_optional(target, "cache_hit_tokens", meta.get("cache_hit_tokens"))
                    _add_optional(target, "cache_miss_tokens", meta.get("cache_miss_tokens"))
            else:
                target.calls += 1
                target.missing_token_calls += 1

            judge_meta = trial.get("per_turn_judge_meta")
            if judge_meta:
                for meta in judge_meta:
                    judge.calls += 1
                    if not meta or meta.get("prompt_tokens") is None or meta.get("completion_tokens") is None:
                        judge.missing_token_calls += 1
                        continue
                    judge.prompt_tokens += meta["prompt_tokens"]
                    judge.completion_tokens += meta["completion_tokens"]
                    _add_optional(judge, "cache_hit_tokens", meta.get("cache_hit_tokens"))
                    _add_optional(judge, "cache_miss_tokens", meta.get("cache_miss_tokens"))
    return trial_units, target, judge


def price_usd(counter: Counter) -> float:
    if counter.calls == 0:
        return 0.0
    if counter.cache_hit_tokens is None or counter.cache_miss_tokens is None:
        hit = 0
        miss = counter.prompt_tokens
    else:
        hit = counter.cache_hit_tokens
        miss = counter.cache_miss_tokens
    cny = (
        hit / 1e6 * PRICING["input_cache_hit_per_m"]
        + miss / 1e6 * PRICING["input_cache_miss_per_m"]
        + counter.completion_tokens / 1e6 * PRICING["output_per_m"]
    )
    return cny / PRICING["cny_per_usd"]


def _messages_tokens(estimator: TokenEstimator, system: str, user: str) -> int:
    return estimator.count_messages(system, user)


def _skills_by_name(category: str) -> dict[str, str]:
    paths = [REPO_ROOT / "skills" / category / "skill_dictionary.json"]
    if category == "sycophancy_resistance":
        paths.append(REPO_ROOT / "skills" / category / "follow_up_skills.json")
    result = {}
    for path in paths:
        for row in json.loads(path.read_text()):
            result[row["name"]] = row["description"]
    return result


def _few_shot_blocks(examples: list[dict]) -> list[str]:
    blocks = []
    for n in (2, 3):
        for chosen in itertools.combinations(examples, min(n, len(examples))):
            parts = []
            for i, ex in enumerate(chosen, 1):
                skills = "\n".join(f"- {skill}" for skill in ex["skills"])
                parts.append(
                    f"Example {i}:\nBase request: {ex['bare_request']}\n"
                    f"Skills:\n{skills}\nComposed prompt: {ex['composed']}"
                )
            blocks.append("\n\n".join(parts))
    return blocks or [""]


def estimate_composer(
    category: str,
    records: Iterable[dict],
    estimator: TokenEstimator,
) -> Counter:
    """Estimate legacy LLM composer tokens from saved inputs and outputs."""
    if category == "constrained_completion":
        return Counter(provenance="not_applicable", note="TemplateComposer makes no API calls")

    module = importlib.import_module(f"search.templates.{category}")
    templates = module.COMPOSE_TEMPLATES
    rephrase = getattr(module, "REPHRASE_TEMPLATE", None)
    follow_template = getattr(module, "FOLLOW_UP_TEMPLATE", None)
    few_shots = _few_shot_blocks(module.FEW_SHOT_EXAMPLES)
    descriptions = _skills_by_name(category)
    follow_descs = json.loads(
        (REPO_ROOT / "skills/sycophancy_resistance/follow_up_skills.json").read_text()
    ) if category == "sycophancy_resistance" else []
    representative_follow_desc = (
        sorted(follow_descs, key=lambda row: estimator.count(row["description"]))[
            len(follow_descs) // 2
        ]["description"]
        if follow_descs else ""
    )
    # Pre-tokenize all text that is independent of the saved record. Token
    # counts are nearly additive; formatting the variable fields separately
    # avoids re-encoding every template/few-shot combination for every SGBS
    # round (millions of redundant encodes on the full ledger). The fixed
    # portion includes the exact DeepSeek V3 two-message chat framing.
    mean_few_shot_tokens = round(
        statistics.mean(estimator.count(block) for block in few_shots)
    )
    compose_fixed_tokens = []
    for template in templates:
        empty_user = template["user"].format(
            bare_request="",
            skill_descriptions="",
            few_shot_block="",
        )
        compose_fixed_tokens.append(
            estimator.count_messages(template["system"], empty_user)
            + mean_few_shot_tokens
        )
    mean_compose_fixed = round(statistics.mean(compose_fixed_tokens))

    counter = Counter(
        provenance="tokenizer_estimated",
        note=(
            f"{estimator.model_label}; exact_tokenizer={estimator.exact}; "
            "saved output tokenized directly; unknown random compose template/few-shot "
            "choice represented by the mean fixed-token contribution over all available "
            f"choices; variable fields tokenized separately; chat_format={estimator.chat_format}"
        ),
    )
    for record in records:
        bare = record.get("bare_request") or ""
        skill_names = record.get("skills") or []
        output = record.get("prompt") or ""
        if skill_names:
            skill_text = "\n".join(
                f"- {name}: {descriptions.get(name, '')}" for name in skill_names
            )
            counter.prompt_tokens += (
                mean_compose_fixed
                + estimator.count(bare)
                + estimator.count(skill_text)
            )
            counter.calls += 1
            counter.completion_tokens += estimator.count(output)
        elif rephrase is not None:
            user = rephrase["user"].format(bare_request=bare)
            counter.prompt_tokens += _messages_tokens(estimator, rephrase["system"], user)
            counter.calls += 1
            counter.completion_tokens += estimator.count(output)

        follow_ups = (
            record.get("follow_ups")
            or (record.get("metadata") or {}).get("follow_ups")
            or []
        )
        follow_names = record.get("follow_up_skills") or []
        previous: list[str] = []
        for index, follow_output in enumerate(follow_ups):
            if follow_template is None:
                break
            name = follow_names[index] if index < len(follow_names) else None
            if name:
                desc_text = f"- {name}: {descriptions.get(name, '')}"
            else:
                # Old SGBS records saved outputs but not selected follow-up
                # skill names.  Use a median-length real description as a
                # visible approximation instead of treating it as zero.
                desc_text = f"- unknown: {representative_follow_desc}"
            prev_block = (
                "(none — this is the first follow-up)"
                if not previous
                else "\n".join(f"{i}. {msg}" for i, msg in enumerate(previous, 1))
            )
            user = follow_template["user"].format(
                bare_request=bare,
                follow_up_description=desc_text,
                previous_user_messages=prev_block,
            )
            counter.prompt_tokens += _messages_tokens(
                estimator, follow_template["system"], user
            )
            counter.completion_tokens += estimator.count(follow_output)
            counter.calls += 1
            previous.append(follow_output)
    return counter


_LOG_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_EDGE = re.compile(r"Edge case #(\d+) found")


def log_span(path: Path, stop_edge: int | None = None) -> float | None:
    first = None
    last = None
    with path.open(errors="replace") as handle:
        for line in handle:
            match = _LOG_TS.match(line)
            if not match:
                continue
            stamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            first = first or stamp
            last = stamp
            edge = _EDGE.search(line)
            if stop_edge is not None and edge and int(edge.group(1)) == stop_edge:
                return (stamp - first).total_seconds()
    return (last - first).total_seconds() if first and last else None


def metadata_span(matrix_dir: Path) -> float | None:
    timestamps = []
    for path in matrix_dir.glob("*/metadata.json"):
        data = json.loads(path.read_text())
        for key in ("timestamp", "last_updated", "completed_at"):
            value = data.get(key)
            if value:
                try:
                    timestamps.append(datetime.fromisoformat(value))
                except ValueError:
                    pass
    if len(timestamps) < 2:
        return None
    return (max(timestamps) - min(timestamps)).total_seconds()


def clock_log_to_metadata_span(log_path: Path, matrix_dir: Path) -> float | None:
    """Recover duration when a stage log stores HH:MM:SS without the date."""
    clock_pattern = re.compile(r"^\s*(\d{2}:\d{2}:\d{2})\s+\|")
    first_clock = None
    with log_path.open(errors="replace") as handle:
        for line in handle:
            match = clock_pattern.match(line)
            if match:
                first_clock = datetime.strptime(match.group(1), "%H:%M:%S").time()
                break
    completed = []
    for path in matrix_dir.glob("*/metadata.json"):
        value = json.loads(path.read_text()).get("last_updated")
        if value:
            completed.append(datetime.fromisoformat(value))
    if first_clock is None or not completed:
        return None
    end = max(completed)
    start = datetime.combine(end.date(), first_clock)
    if start > end:
        start -= timedelta(days=1)
    return (end - start).total_seconds()


def aggregate_matrix(category: str, matrix_dir: Path, name: str) -> Stage:
    stage = Stage(category, name, [str(matrix_dir.relative_to(REPO_ROOT))])
    for path in sorted(matrix_dir.glob("*/response_matrix.jsonl")):
        n, target, judge = aggregate_trial_records(read_jsonl(path))
        stage.trial_units += n
        stage.target.add(target)
        stage.judge.add(judge)
    stage.wall_seconds = metadata_span(matrix_dir)
    stage.wall_kind = "lower_bound"
    stage.wall_note = "span between model completion timestamps; first model startup is not recorded"
    return stage


def reference_stage(category: str, estimator: TokenEstimator) -> Stage:
    matrix = REPO_ROOT / REFERENCE_DIRS[category]
    stage = aggregate_matrix(category, matrix, "reference_panel")
    source = REPO_ROOT / f"benchmarks/{category}/random_baseline_k2000.jsonl"
    first_matrix = next(iter(sorted(matrix.glob("*/response_matrix.jsonl"))))
    prompt_ids = {row["prompt_id"] for row in read_jsonl(first_matrix)}
    records = [
        row for row in read_jsonl(source) if row.get("prompt_id") in prompt_ids
    ]
    if len(records) != len(prompt_ids):
        raise ValueError(
            f"{source}: found {len(records)}/{len(prompt_ids)} reference prompts"
        )
    stage.composer = estimate_composer(category, records, estimator)
    stage.artifact_paths.append(str(source.relative_to(REPO_ROOT)))
    return stage


def sgbs_stage(category: str, estimator: TokenEstimator, k: int = 50) -> Stage:
    run_dir = REPO_ROOT / SGBS_DIRS[category]
    edges = sorted(read_jsonl(run_dir / "edge_cases.jsonl"), key=lambda row: row["round_found"])
    if len(edges) < k:
        raise ValueError(f"{run_dir}: only {len(edges)} edge cases, need {k}")
    round_k = edges[k - 1]["round_found"]
    records = [
        row for row in read_jsonl(run_dir / "trial_details.jsonl")
        if row["round_num"] <= round_k
    ]
    n, target, judge = aggregate_trial_records(records)
    stage = Stage(
        category,
        "sgbs_to_k50",
        [str(run_dir.relative_to(REPO_ROOT))],
        trial_units=n,
        target=target,
        judge=judge,
        composer=estimate_composer(category, records, estimator),
        wall_seconds=log_span(run_dir / "search.log", stop_edge=k),
        wall_kind="measured",
        wall_note=f"search.log first timestamp through edge #{k}; R_K={round_k}",
    )
    return stage


def _selected_ids(result: dict) -> set[str]:
    return set(result.get("pilot_item_ids") or []) | {
        row["item_id"] for row in result.get("history") or []
    }


def adaptive_stage(category: str, run: str, matrix_model: str) -> Stage:
    run_dir = REPO_ROOT / run
    trial_path = run_dir / "trials.jsonl"
    provenance = "measured"
    if trial_path.exists():
        records = read_jsonl(trial_path)
        artifacts = [str(trial_path.relative_to(REPO_ROOT))]
    else:
        result_path = run_dir / "result.json"
        result = json.loads(result_path.read_text())
        wanted = _selected_ids(result)
        proxy_path = (
            REPO_ROOT / f"results/{category}/anchor_set/{matrix_model}/response_matrix.jsonl"
        )
        records = [
            row for row in read_jsonl(proxy_path) if row.get("prompt_id") in wanted
        ]
        if len(records) != len(wanted):
            raise ValueError(
                f"{proxy_path}: found {len(records)}/{len(wanted)} selected adaptive items"
            )
        artifacts = [
            str(result_path.relative_to(REPO_ROOT)),
            str(proxy_path.relative_to(REPO_ROOT)),
        ]
        provenance = "proxy_same_model_item"
    n, target, judge = aggregate_trial_records(records)
    target.provenance = provenance
    judge.provenance = provenance
    stage = Stage(
        category,
        "adaptive_cat",
        artifacts,
        trial_units=n,
        target=target,
        judge=judge,
        composer=Counter(provenance="not_applicable"),
        wall_seconds=log_span(run_dir / "adaptive.log"),
        wall_kind="measured",
        wall_note=(
            "adaptive.log first-to-last timestamp"
            if provenance == "measured"
            else "wall measured from adaptive.log; tokens recovered from exact selected model-item matrix cells"
        ),
    )
    return stage


def step2_stage(category: str) -> Stage:
    root = REPO_ROOT / STEP2_DIRS[category]
    stage = aggregate_matrix(category, root / "matrix", "step2_panel")
    relative_log = STEP2_START_LOGS.get(category)
    log_path = REPO_ROOT / relative_log if relative_log else None
    if log_path is not None and log_path.exists():
        stage.wall_seconds = log_span(log_path)
        if stage.wall_seconds is None:
            stage.wall_seconds = clock_log_to_metadata_span(log_path, root / "matrix")
        stage.wall_kind = "measured"
        stage.wall_note = (
            "collection-log start through final matrix last_updated timestamp; "
            "includes interruptions/resumes"
        )
        stage.artifact_paths.append(str(log_path.relative_to(REPO_ROOT)))
    return stage


def combine(category: str, name: str, stages: Iterable[Stage]) -> Stage:
    result = Stage(category, name, [])
    stages = list(stages)
    for stage in stages:
        result.add(stage)
    if any(stage.wall_seconds is None for stage in stages):
        result.wall_seconds = None
        result.wall_kind = "incomplete"
    elif any(stage.wall_kind == "lower_bound" for stage in stages):
        result.wall_kind = "lower_bound_sum"
    else:
        result.wall_kind = "sum_of_stage_wall_times"
    result.wall_note = "sum of adaptive CAT, SGBS-to-K=50, and Step-2 panel stages"
    return result


def median_stage(category: str, stages: list[Stage]) -> dict:
    def median_int(values: Iterable[int]) -> int:
        return round(statistics.median(values))

    wall = [stage.wall_seconds for stage in stages if stage.wall_seconds is not None]
    return {
        "category": category,
        "name": "in_panel_per_model_median",
        "n_models": len(stages),
        "trial_units": median_int(stage.trial_units for stage in stages),
        "target_queries": median_int(stage.target.calls for stage in stages),
        "judge_queries": median_int(stage.judge.calls for stage in stages),
        "composer_queries": median_int(stage.composer.calls for stage in stages),
        "model_queries": median_int(stage.model_queries for stage in stages),
        "total_tokens": median_int(stage.total_tokens for stage in stages),
        "api_cost_usd": round(statistics.median(stage.api_cost_usd for stage in stages), 6),
        "wall_seconds": statistics.median(wall) if wall else None,
        "wall_range_seconds": [min(wall), max(wall)] if wall else None,
        "members": [stage.to_dict() for stage in stages],
        "note": "median across Claim-3 models classified as in-panel; wall range shown separately",
    }


def _fmt_queries(value: int) -> str:
    return f"{value / 1000:.1f}k"


def _fmt_tokens(value: int) -> str:
    return f"{value / 1e6:.2f}M"


def _fmt_hours(seconds: float | None, kind: str | None = None) -> str:
    if seconds is None:
        return "n/a"
    prefix = "≥" if kind in {"lower_bound", "lower_bound_sum"} else ""
    return f"{prefix}{seconds / 3600:.1f}h"


def _compact_cell(row: dict) -> str:
    return (
        f"{_fmt_queries(row['model_queries'])} queries; "
        f"{_fmt_tokens(row['total_tokens'])} tok; "
        f"{_fmt_hours(row['wall_seconds'], row.get('wall_kind'))}; "
        f"\\${row['api_cost_usd']:.2f}"
    )


def _queries_tjc(row: dict) -> str:
    if "target" in row:
        values = (
            row["target"]["calls"],
            row["judge"]["calls"],
            row["composer"]["calls"],
        )
    else:
        values = (
            row["target_queries"],
            row["judge_queries"],
            row["composer_queries"],
        )
    return " / ".join(f"{value:,}" for value in values)


def _estimated_marker(row: dict) -> str:
    if "target" not in row:
        return "≈" if row["category"] == "harmful_refusal" else ""
    if row["category"] == "constrained_completion":
        return ""
    provenances = {
        row["target"]["provenance"],
        row["judge"]["provenance"],
        row["composer"]["provenance"],
    }
    return "≈" if provenances & {"tokenizer_estimated", "proxy_same_model_item", "mixed"} else ""


def _half_up(value: float) -> float:
    return math.floor(value * 10 + 0.5) / 10


def _in_panel_wall(row: dict) -> str:
    median_minutes = _half_up(row["wall_seconds"] / 60)
    low, high = (_half_up(value / 60) for value in row["wall_range_seconds"])
    return f"{median_minutes:.1f} min [{low:.1f}, {high:.1f}]"


def _out_panel_wall(stages: list[dict]) -> str:
    by_name = {stage["name"]: stage for stage in stages}
    adaptive = by_name["adaptive_cat"]
    sgbs = by_name["sgbs_to_k50"]
    panel = by_name["step2_panel"]
    panel_prefix = "≥" if panel["wall_kind"] == "lower_bound" else ""
    return (
        f"CAT {_half_up(adaptive['wall_seconds'] / 60):.1f}m + "
        f"SGBS {_half_up(sgbs['wall_seconds'] / 3600):.1f}h + "
        f"panel {panel_prefix}{_half_up(panel['wall_seconds'] / 3600):.1f}h"
    )


def render_markdown(ledger: dict) -> str:
    lines = [
        "# Rebuttal usage/cost ledger",
        "",
        "All token counts come from stored provider metadata unless a row is marked "
        "`tokenizer_estimated` or `proxy_same_model_item`. API cost includes only "
        "DeepSeek composer/judge calls; target and reference-panel inference ran "
        "locally. When cache accounting is absent, input is conservatively priced "
        "at the cache-miss rate.",
        "",
        "## Stage-level audit",
        "",
        "| Category | Stage | Trial units | Model queries (T/J/C) | Total tokens | Wall-clock | API cost | Token provenance |",
        "|:---:|:---|---:|---:|---:|---:|---:|:---|",
    ]
    for category, payload in ledger["categories"].items():
        for stage in payload["stages"]:
            if "target" in stage:
                provenance = "/".join(
                    sorted({
                        stage["target"]["provenance"],
                        stage["judge"]["provenance"],
                        stage["composer"]["provenance"],
                    })
                )
            else:
                member_sources = set()
                for member in stage["members"]:
                    member_sources.update({
                        member["target"]["provenance"],
                        member["judge"]["provenance"],
                    })
                provenance = "median[" + "/".join(sorted(member_sources)) + "]"
            lines.append(
                f"| {CATEGORY_LABELS[category]} | {stage['name']} | "
                f"{stage['trial_units']:,} | {_queries_tjc(stage)} | "
                f"{stage['total_tokens']:,} | "
                f"{_fmt_hours(stage['wall_seconds'], stage.get('wall_kind'))} | "
                f"${stage['api_cost_usd']:.2f} | {provenance} |"
            )
    lines.extend([
        "",
        "## HiES W2 main table",
        "",
        "| Category | Stage | Model queries (T/J/C) | LLM tokens | Wall-clock | API cost |",
        "|:---:|:---|---:|---:|:---|---:|",
    ])
    for category, payload in ledger["categories"].items():
        reference = payload["reference"]
        in_panel = payload["in_panel"]
        out_panel = payload["out_of_panel"]
        ref_marker = _estimated_marker(reference)
        in_marker = _estimated_marker(in_panel)
        out_marker = _estimated_marker(out_panel)
        lines.append(
            f"| {CATEGORY_LABELS[category]} | One-time anchor build | "
            f"{_queries_tjc(reference)} | {ref_marker}{_fmt_tokens(reference['total_tokens'])} | "
            f"{_fmt_hours(reference['wall_seconds'], reference['wall_kind'])} artifact span | "
            f"{ref_marker}\\${reference['api_cost_usd']:.2f} |"
        )
        lines.append(
            f"| {CATEGORY_LABELS[category]} | In-panel CAT, per target | "
            f"{_queries_tjc(in_panel)} | {in_marker}{_fmt_tokens(in_panel['total_tokens'])} median | "
            f"{_in_panel_wall(in_panel)} | {in_marker}\\${in_panel['api_cost_usd']:.2f} |"
        )
        lines.append(
            f"| {CATEGORY_LABELS[category]} | Out-of-panel trigger | "
            f"{_queries_tjc(out_panel)} | {out_marker}{_fmt_tokens(out_panel['total_tokens'])} | "
            f"{_out_panel_wall(payload['stages'])} | "
            f"{out_marker}\\${out_panel['api_cost_usd']:.2f} |"
        )
    lines.extend([
        "",
        "`T/J/C` are target, judge, and composer endpoint calls. SR counts actual "
        "conversation turns, not trial cells. `≈` means that the recorded target/"
        "judge usage is combined with tokenizer-estimated legacy composer usage "
        "(or, for early HR CAT, exact selected cells recovered from the response "
        "matrix). `≥` is a wall-clock lower bound caused by a missing start timestamp. "
        "CIF judging and composition are rule-based and therefore incur zero API cost.",
        "",
        "API cost includes DeepSeek composer/judge usage only, not local target/panel "
        "GPU inference. It is a reproduction-cost estimate using the pricing snapshot "
        "below; inputs without a recorded cache split are conservatively priced as "
        "cache misses.",
        "",
        "CIF and SR SGBS took a long time in ordinary elapsed time: both targets "
        "could generate up to 8,192 tokens, and the SR run was interrupted and "
        "resumed. The values are measured elapsed wall-clock, not pure compute time.",
        "",
        "## Out-of-panel stage split",
        "",
        "| Category | SGBS tokens / API cost | Step-2 panel tokens / API cost |",
        "|:---:|---:|---:|",
    ])
    for category, payload in ledger["categories"].items():
        by_name = {stage["name"]: stage for stage in payload["stages"]}
        sgbs = by_name["sgbs_to_k50"]
        panel = by_name["step2_panel"]
        sgbs_marker = _estimated_marker(sgbs)
        lines.append(
            f"| {CATEGORY_LABELS[category]} | "
            f"{sgbs_marker}{_fmt_tokens(sgbs['total_tokens'])} / "
            f"{sgbs_marker}\\${sgbs['api_cost_usd']:.2f} | "
            f"{_fmt_tokens(panel['total_tokens'])} / \\${panel['api_cost_usd']:.2f} |"
        )
    lines.extend([
        "",
        "## Estimation assumptions",
        "",
        f"- Composer tokenizer: `{ledger['tokenizer']['model']}` "
        f"(exact={str(ledger['tokenizer']['exact']).lower()}).",
        "- Legacy composer outputs are stored and tokenized directly. Missing random "
        "template/few-shot choices are averaged over the repository templates.",
        "- Old SR SGBS records omit follow-up skill names; each missing description "
        "is represented by a median-length entry from the follow-up skill dictionary.",
        "- Matrix wall-clock values based only on completion metadata are lower bounds; "
        "the first model's startup and inference began before the first timestamp.",
        f"- Pricing: DeepSeek v4-flash rates verified {PRICING['verified']}; "
        f"{PRICING['cny_per_usd']} CNY/USD.",
        "",
    ])
    return "\n".join(lines)


def build_ledger(estimator: TokenEstimator) -> dict:
    categories = {}
    for category in CATEGORY_LABELS:
        reference = reference_stage(category, estimator)
        in_members = [
            adaptive_stage(category, run, model)
            for run, model in IN_PANEL_RUNS[category]
        ]
        in_panel = median_stage(category, in_members)
        adaptive = adaptive_stage(category, *OUT_PANEL_ADAPTIVE[category])
        sgbs = sgbs_stage(category, estimator)
        step2 = step2_stage(category)
        out_total = combine(category, "out_of_panel_total", [adaptive, sgbs, step2])
        categories[category] = {
            "reference": reference.to_dict(),
            "in_panel": in_panel,
            "out_of_panel": out_total.to_dict(),
            "stages": [
                reference.to_dict(),
                in_panel,
                adaptive.to_dict(),
                sgbs.to_dict(),
                step2.to_dict(),
                out_total.to_dict(),
            ],
        }
    return {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "pricing": PRICING,
        "tokenizer": {
            "path": str(estimator.path),
            "model": estimator.model_label,
            "exact": estimator.exact,
            "chat_format": estimator.chat_format,
        },
        "categories": categories,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composer-tokenizer", help="exact tokenizer.json for composer model")
    parser.add_argument(
        "--output-dir",
        default="results/rebuttal/cost_ledger",
        help="output directory, relative to repository root by default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = REPO_ROOT / output
    estimator = resolve_tokenizer(args.composer_tokenizer)
    ledger = build_ledger(estimator)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "usage_ledger.json"
    md_path = output / "usage_ledger.md"
    json_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")
    md_path.write_text(render_markdown(ledger))
    print(md_path.read_text())
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
