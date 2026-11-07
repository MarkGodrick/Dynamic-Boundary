from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from datasets import Dataset


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = (
    ROOT / "scripts/data_prep/build_judge_bias_human_datasets.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_judge_bias_human_datasets", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_jbb_inverts_jailbreak_success_to_category_pass():
    dataset = Dataset.from_list(
        [
            {
                "Index": 0,
                "goal": "g0",
                "prompt": "p0",
                "target_response": "r0",
                "human1": 0,
                "human2": 0,
                "human3": 1,
                "human_majority": 0,
            },
            {
                "Index": 1,
                "goal": "g1",
                "prompt": "p1",
                "target_response": "r1",
                "human1": 1,
                "human2": 1,
                "human3": 0,
                "human_majority": 1,
            },
        ]
    )
    records, summary = MODULE.convert_jbb(dataset)
    assert [record["final_label"] for record in records] == [1, 0]
    assert summary["final_label_counts"] == {0: 1, 1: 1}


def test_wildguard_uses_unsafe_precedence_and_excludes_missing_labels():
    base = {
        "prompt_original": "safe prompt",
        "response_original": "response",
        "adversarial": False,
        "prompt_harm_label": "unharmful",
        "subcategory": "benign",
    }
    dataset = Dataset.from_list(
        [
            {
                **base,
                "response_refusal_label": "compliance",
                "response_harm_label": "unharmful",
            },
            {
                **base,
                "response_refusal_label": "refusal",
                "response_harm_label": "unharmful",
            },
            {
                **base,
                "response_refusal_label": "refusal",
                "response_harm_label": "harmful",
            },
            {
                **base,
                "response_refusal_label": None,
                "response_harm_label": "unharmful",
            },
            {
                **base,
                "prompt_harm_label": "harmful",
                "response_refusal_label": "refusal",
                "response_harm_label": "unharmful",
            },
        ]
    )
    records, summary = MODULE.convert_wildguard(dataset)
    assert [record["reference_raw_label"] for record in records] == [
        "appropriate",
        "over_refusal",
        "unsafe",
    ]
    assert [record["final_label"] for record in records] == [1, 0, 0]
    assert summary["excluded_counts"] == {
        "missing_response_refusal_label": 1,
        "prompt_not_human_labeled_unharmful": 1,
    }
