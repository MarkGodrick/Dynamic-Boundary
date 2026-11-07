from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from judges.llm_judge import LLMJudge


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts/rebuttal/run_judge_bias.py"
SPEC = importlib.util.spec_from_file_location("run_judge_bias", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_human_metrics_use_failure_as_positive_class():
    rows = []
    # human pass/judge pass: TN; human fail/judge fail: TP;
    # human fail/judge pass: FN; human pass/judge fail: FP.
    for index, (human_pass, passed) in enumerate(
        [(True, True), (False, False), (False, True), (True, False)]
    ):
        rows.append(
            {
                "experiment": "judge_human",
                "category": "harmful_refusal",
                "judge_name": "j",
                "requested_model": "m",
                "record_key": str(index),
                "human_pass": human_pass,
                "passed": passed,
                "judge_outcome": "ok",
                "estimated_cost_usd": 0,
                "elapsed_seconds": 0,
            }
        )
    result = MODULE.analyze_human(rows)["results"][0]
    assert result["agreement"] == 0.5
    assert result["fnr_failure_positive"] == 0.5
    assert result["fpr_failure_positive"] == 0.5
    assert result["confusion_failure_positive"] == {
        "tp_human_fail_judge_fail": 1,
        "fn_human_fail_judge_pass": 1,
        "fp_human_pass_judge_fail": 1,
        "tn_human_pass_judge_pass": 1,
    }


def test_inter_loader_streams_selected_trials_and_skips_hidden_dirs(tmp_path):
    item_bank = tmp_path / "items.jsonl"
    _write_jsonl(
        item_bank,
        [
            {"prompt_id": "a", "prompt": "pa", "beta": -1, "metadata": {}},
            {"prompt_id": "b", "prompt": "pb", "beta": 1, "metadata": {}},
        ],
    )
    matrix = tmp_path / "matrix"
    for model in ("m1", "m2"):
        _write_jsonl(
            matrix / model / "response_matrix.jsonl",
            [
                {
                    "prompt_id": item,
                    "trials": [{"response": f"{model}-{item}-0"}],
                }
                for item in ("a", "b")
            ],
        )
    _write_jsonl(
        matrix / ".backup" / "response_matrix.jsonl",
        [{"prompt_id": "a", "trials": [{"response": "ignored"}]}],
    )

    work, _inputs, selection = MODULE.build_inter_work_items(
        category="harmful_refusal",
        matrix_root=matrix,
        item_bank_path=item_bank,
        max_items=2,
        trials_per_cell=1,
        seed=7,
    )
    assert len(work) == 4
    assert {item.source_model for item in work} == {"m1", "m2"}
    assert selection["n_selected_items"] == 2


def test_fake_run_resumes_and_fits_common_rasch(tmp_path):
    item_bank = tmp_path / "items.jsonl"
    _write_jsonl(
        item_bank,
        [
            {"prompt_id": f"i{i}", "prompt": f"p{i}", "beta": i, "metadata": {}}
            for i in range(4)
        ],
    )
    matrix = tmp_path / "matrix"
    for model in ("m1", "m2", "m3"):
        _write_jsonl(
            matrix / model / "response_matrix.jsonl",
            [
                {
                    "prompt_id": f"i{i}",
                    "trials": [{"response": f"{model}-{i}-{trial}"} for trial in range(2)],
                }
                for i in range(4)
            ],
        )
    work, inputs, selection = MODULE.build_inter_work_items(
        category="harmful_refusal",
        matrix_root=matrix,
        item_bank_path=item_bank,
        max_items=4,
        trials_per_cell=2,
        seed=9,
    )
    profiles = {
        name: {
            "model": name,
            "api_key_env": "UNUSED",
            "prices_usd_per_million": {
                "input": 1,
                "output": 1,
            },
        }
        for name in ("judge_b", "judge_a", "judge_c")
    }
    output = tmp_path / "out"
    first = MODULE.run_work_items(
        work_items=work,
        profiles=profiles,
        output_dir=output,
        inputs=inputs,
        selection=selection,
        workers=3,
        overwrite=True,
        retry_errors=False,
        fake=True,
    )
    second = MODULE.run_work_items(
        work_items=work,
        profiles=profiles,
        output_dir=output,
        inputs=inputs,
        selection=selection,
        workers=3,
        overwrite=False,
        retry_errors=False,
        fake=True,
    )
    assert len(first) == len(second) == len(work) * 3
    assert sum(1 for _ in (output / "judgments.jsonl").open()) == len(first)
    summary = MODULE.analyze_inter(second, list(profiles))
    assert summary["rasch"]["status"] == "ok"
    assert len(summary["pairwise_agreement"]) == 3
    assert len(summary["rasch"]["comparisons"]) == 3
    assert all(
        row["judge_a"] < row["judge_b"]
        for row in summary["rasch"]["comparisons"]
    )


def test_llm_judge_request_options_and_response_identity():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="response-1",
            model="served-model",
            created=123,
            system_fingerprint="fp",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="Answer: safe"),
                )
            ],
        )

    judge = LLMJudge(
        model="requested-model",
        api_key_env="UNUSED",
        category="harmful_refusal",
        temperature=None,
        max_tokens=99,
        extra_body={"thinking": {"type": "disabled"}},
        request_kwargs={"max_completion_tokens": 77},
    )
    judge._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    passed = judge.judge(
        [
            {"role": "user", "content": "p"},
            {"role": "assistant", "content": "r"},
        ]
    )
    assert passed is True
    assert "temperature" not in captured
    assert "max_tokens" not in captured
    assert captured["max_completion_tokens"] == 77
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    meta = judge.get_last_meta()
    assert meta["response_id"] == "response-1"
    assert meta["served_model"] == "served-model"
    assert meta["system_fingerprint"] == "fp"
