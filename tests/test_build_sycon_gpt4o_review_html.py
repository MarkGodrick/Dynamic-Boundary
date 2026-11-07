from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts/data_prep/build_sycon_gpt4o_review_html.py"
SPEC = importlib.util.spec_from_file_location(
    "build_sycon_gpt4o_review_html", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_render_html_embeds_only_requested_records():
    template_path = (
        ROOT
        / "scripts/data_prep/templates"
        / "sycon_review.html"
    )
    records = [
        {
            "id": "review-1",
            "confidence_label": 0,
            "sycophancy_label": 1,
            "gpt4o_label": 0,
            "last_response": "line one\nline two",
            "review_reasons": ["gpt4o_codex_disagreement"],
        },
        {
            "id": "review-2",
            "confidence_label": -1,
            "sycophancy_label": 1,
            "gpt4o_label": 1,
            "review_reasons": ["item_or_ground_truth_quality"],
        },
    ]

    html = MODULE.render_html(template_path.read_text(encoding="utf-8"), records)

    assert "SYCON GPT-4o 人工复核 · 2 items" in html
    assert '"id": "review-1"' in html
    assert '"id": "review-2"' in html
    assert "sycon_gpt4o_reviewed_2.jsonl" in html
    assert "record.gpt4o_label" in html
    assert '"last_response": "line one\\nline two"' in html
    assert "sycon-fp-audit-001" not in html
