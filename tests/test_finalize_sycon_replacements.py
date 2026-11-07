from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts/data_prep/finalize_sycon_replacements.py"
SPEC = importlib.util.spec_from_file_location(
    "finalize_sycon_replacements", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _base(record_id, model, row, gpt4o_label):
    return {
        "id": record_id,
        "source_model": model,
        "source_row": row,
        "gpt4o_label": gpt4o_label,
    }


def test_finalize_applies_human_labels_and_replaces_rejected_items(monkeypatch):
    specs = (
        {
            "replacement_for_id": "reject",
            "source_model": "model-a",
            "source_row": 2,
        },
    )
    monkeypatch.setattr(MODULE, "REPLACEMENT_SPECS", specs)
    base = [
        _base("reject", "model-a", 1, 0),
        _base("reviewed", "model-b", 3, 0),
        _base("automatic", "model-b", 4, 1),
    ]
    reviewed = [
        {
            **base[0],
            "human_reviewed": True,
            "human_label": -1,
            "human_reviewed_at": "now",
        },
        {
            **base[1],
            "human_reviewed": True,
            "human_label": 1,
            "human_reviewed_at": "now",
        },
    ]
    replacement = {
        "id": "replacement",
        "replacement_for_id": "reject",
        "source_model": "model-a",
        "source_row": 2,
        "gpt4o_label": 1,
        "gpt4o_judge": {"error": None},
    }

    final, summary = MODULE.finalize_records(
        base_records=base,
        reviewed_records=reviewed,
        annotated_replacements=[replacement],
    )

    by_id = {record["id"]: record for record in final}
    assert set(by_id) == {"reviewed", "automatic", "replacement"}
    assert by_id["reviewed"]["final_label"] == 1
    assert by_id["reviewed"]["final_label_source"] == "user_review"
    assert by_id["automatic"]["final_label_source"] == "gpt4o_auto"
    assert by_id["replacement"]["final_label_source"] == "gpt4o_auto_replacement"
    assert summary["source_model_counts"] == {"model-a": 1, "model-b": 2}
