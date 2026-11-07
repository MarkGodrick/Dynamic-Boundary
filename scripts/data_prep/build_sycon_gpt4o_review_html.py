"""Build a self-contained review HTML for the GPT-4o SYCON review subset.

The existing validated review page supplies the UI template. This generator
replaces its embedded records and updates the initial-label/provenance text so
only GPT-4o/Codex disagreements and item-quality flags require human review.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "dataset/judge_bias/sycophancy_resistance"
DEFAULT_TEMPLATE = Path(__file__).parent / "templates/sycon_review.html"
DEFAULT_INPUT = DATA_DIR / "sycon_false_presupposition_gpt4o_review.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "sycon_false_presupposition_gpt4o_review.html"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def replace_exact(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}, found {count}")
    return text.replace(old, new)


def render_html(template: str, records: list[dict[str, Any]]) -> str:
    if not records:
        raise ValueError("Review subset is empty")
    count = len(records)
    embedded = json.dumps(records, ensure_ascii=False)
    storage_key = "sycon-review-gpt4o-final-v1-9cd35a0"

    pattern = re.compile(
        r"    const DATA = .*?;\n"
        r"    const STORAGE_KEY = '[^']+';",
        flags=re.DOTALL,
    )
    replacement = (
        f"    const DATA = {embedded};\n"
        f"    const STORAGE_KEY = '{storage_key}';"
    )
    # Use a replacement callback so backslash escapes inside JSON strings
    # (especially ``\n``) are not interpreted by ``re.sub``.
    html, replacements = pattern.subn(lambda _: replacement, template, count=1)
    if replacements != 1:
        raise ValueError(f"Expected one embedded DATA block, found {replacements}")

    substitutions = [
        (
            "<title>SYCON 人工复核 · 24 items</title>",
            f"<title>SYCON GPT-4o 人工复核 · {count} items</title>",
        ),
        (
            "<p>仅包含 24 条低置信度或不适合样例</p>",
            f"<p>仅包含 {count} 条 GPT-4o/初标分歧或题目质量可疑样例</p>",
        ),
        (
            '<div class="progress-meta"><span id="progressText">'
            "已确认 0 / 24</span>",
            '<div class="progress-meta"><span id="progressText">'
            f"已确认 0 / {count}</span>",
        ),
        (
            '<div class="stat"><strong id="statPending">24</strong>',
            f'<div class="stat"><strong id="statPending">{count}</strong>',
        ),
        (
            '<div class="stat"><strong id="statConf0">10</strong>'
            "<span>原置信 0</span>",
            '<div class="stat"><strong id="statConf0">0</strong>'
            "<span>GPT/初标分歧</span>",
        ),
        (
            '<div class="stat"><strong id="statConfN1">14</strong>'
            "<span>原置信 -1</span>",
            '<div class="stat"><strong id="statConfN1">0</strong>'
            "<span>题目质量</span>",
        ),
        (
            '<option value="all">全部 24 条</option>',
            f'<option value="all">全部 {count} 条</option>',
        ),
        (
            "const initialLabel = record.confidence_label === -1 ? -1 : "
            "record.sycophancy_label;",
            "const initialLabel = record.confidence_label === -1 ? -1 : "
            "record.gpt4o_label;",
        ),
        (
            "$('#statConf0').textContent = "
            "DATA.filter(r => r.confidence_label === 0).length;",
            "$('#statConf0').textContent = DATA.filter(r => "
            "r.review_reasons.includes('gpt4o_codex_disagreement')).length;",
        ),
        (
            "$('#statConfN1').textContent = "
            "DATA.filter(r => r.confidence_label === -1).length;",
            "$('#statConfN1').textContent = DATA.filter(r => "
            "r.review_reasons.includes('item_or_ground_truth_quality')).length;",
        ),
        (
            '<span class="chip warn">Codex 初标 '
            "${record.sycophancy_label} · 原置信度 "
            "${record.confidence_label}</span>",
            '<span class="chip warn">GPT-4o ${record.gpt4o_label} · '
            "Codex 初标 ${record.sycophancy_label}</span>"
            '<span class="chip warn">${escapeHtml(record.review_reasons.join(" · "))}'
            "</span>",
        ),
        (
            ": `Codex 初标 ${record.sycophancy_label}，原置信度 "
            "${record.confidence_label}；点击任一标签完成确认`;",
            ": `GPT-4o 标签 ${record.gpt4o_label}，Codex 初标 "
            "${record.sycophancy_label}；点击任一标签完成确认`;",
        ),
        (
            "human_label_source: s.reviewed ? 'user_review' : 'codex_initial',",
            "human_label_source: s.reviewed ? 'user_review' : "
            "(record.confidence_label === -1 ? 'quality_flag' : 'gpt4o_auto'),",
        ),
        (
            "download('sycon_reviewed_24.jsonl',",
            f"download('sycon_gpt4o_reviewed_{count}.jsonl',",
        ),
        (
            "download('sycon_reviewed_24.csv',",
            f"download('sycon_gpt4o_reviewed_{count}.csv',",
        ),
        (
            "const fields = ['id','source_model','source_row','question',"
            "'wrong_answer','correct_answer','last_user_message','last_response',"
            "'sycophancy_label','confidence_label','human_label','human_reviewed',"
            "'human_label_source','human_review_note','human_reviewed_at'];",
            "const fields = ['id','source_model','source_row','question',"
            "'wrong_answer','correct_answer','last_user_message','last_response',"
            "'sycophancy_label','confidence_label','gpt4o_label','gpt4o_judge',"
            "'review_reasons','human_label','human_reviewed','human_label_source',"
            "'human_review_note','human_reviewed_at'];",
        ),
        (
            "if (!confirm('确定清除浏览器中这 24 条的人工确认和备注吗？"
            "导出的文件不会被删除。')) return;",
            f"if (!confirm('确定清除浏览器中这 {count} 条的人工确认和备注吗？"
            "导出的文件不会被删除。')) return;",
        ),
    ]
    for old, new in substitutions:
        html = replace_exact(html, old, new)
    return html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_jsonl(args.input)
    html = render_html(args.template.read_text(encoding="utf-8"), records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output} with {len(records)} review items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
