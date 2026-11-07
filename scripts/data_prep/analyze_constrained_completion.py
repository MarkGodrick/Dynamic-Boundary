"""Distribution analysis for constrained_completion (post-merge, pre-gap-fill).

Outputs:
  - bare_distribution.json   : bare pool by source, topic, length
  - val_coverage.json        : validation per-constraint-type coverage (83 types)
  - distribution_summary.md  : human-readable summary

Used as a Phase-A checkpoint before deciding whether to hand-craft gap-fill items.
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BARE = ROOT / "data" / "constrained_completion" / "bare_requests.jsonl"
VAL_IFBENCH = ROOT / "data" / "constrained_completion" / "validation" / "ifbench.jsonl"
VAL_MIE = ROOT / "data" / "constrained_completion" / "validation" / "manyifeval.jsonl"
SKILL = ROOT / "skills" / "constrained_completion" / "skill_dictionary.json"
OUT_DIR = ROOT / "results" / "data_prep" / "constrained_completion"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def analyze_bare() -> dict:
    items = load_jsonl(BARE)
    by_source = Counter(i["source"] for i in items)
    # InFoBench subcategory (topical)
    infobench = [i for i in items if i["source"] == "infobench"]
    topical = Counter(i.get("source_meta", {}).get("category", "unknown") for i in infobench)
    subsets = Counter(i.get("source_meta", {}).get("subset", "unknown") for i in infobench)
    with_context = sum(1 for i in infobench if i.get("source_meta", {}).get("has_context_input"))
    # Prompt length
    lens = [len(i["prompt"].split()) for i in items]
    return {
        "total": len(items),
        "by_source": dict(by_source),
        "prompt_word_len": {
            "mean": round(sum(lens) / len(lens), 1),
            "max": max(lens),
            "min": min(lens),
            "p95": sorted(lens)[int(0.95 * len(lens))],
        },
        "infobench_topical_top20": topical.most_common(20),
        "infobench_subsets": dict(subsets),
        "infobench_with_context_fraction": round(with_context / max(len(infobench), 1), 3),
    }


def analyze_val_coverage() -> dict:
    skills = json.loads(SKILL.read_text())
    all_types = {s["name"] for s in skills}
    ifeval_types = {s["name"] for s in skills if s["source"] == "ifeval"}
    ifbench_types = {s["name"] for s in skills if s["source"] == "ifbench"}

    ifbench_val = load_jsonl(VAL_IFBENCH)
    mie_val = load_jsonl(VAL_MIE)

    # Count per-type applications in each tier
    ifbench_counts = Counter()
    for it in ifbench_val:
        for t in it["instruction_id_list"]:
            ifbench_counts[t] += 1

    mie_counts = Counter()
    for it in mie_val:
        for t in it["instruction_id_list"]:
            mie_counts[t] += 1

    covered = set(ifbench_counts) | set(mie_counts)
    uncovered = all_types - covered

    return {
        "total_types_in_skill_dict": len(all_types),
        "types_covered_by_validation": len(covered),
        "coverage_ratio": round(len(covered) / len(all_types), 3),
        "uncovered_types": sorted(uncovered),
        "ifbench_per_type_counts": dict(sorted(ifbench_counts.items(), key=lambda x: -x[1])),
        "manyifeval_per_type_counts": dict(sorted(mie_counts.items(), key=lambda x: -x[1])),
        "type_coverage_by_source": {
            "ifeval_types_covered": len(covered & ifeval_types),
            "ifeval_types_total": len(ifeval_types),
            "ifbench_types_covered": len(covered & ifbench_types),
            "ifbench_types_total": len(ifbench_types),
        },
        "val_num_instructions_distribution": {
            "ifbench": dict(sorted(Counter(i["num_instructions"] for i in ifbench_val).items())),
            "manyifeval": dict(sorted(Counter(i["num_instructions"] for i in mie_val).items())),
        },
    }


def write_summary(bare: dict, val: dict) -> str:
    lines = []
    lines.append("# Category B — Distribution Summary (pre-gap-fill)")
    lines.append("")
    lines.append("## Bare Pool")
    lines.append(f"- Total: **{bare['total']}** items")
    lines.append(f"- By source: {bare['by_source']}")
    lines.append(f"- Prompt length (words): mean {bare['prompt_word_len']['mean']}, "
                 f"p95 {bare['prompt_word_len']['p95']}, max {bare['prompt_word_len']['max']}")
    lines.append(f"- InFoBench Easy/Hard split: {bare['infobench_subsets']}")
    lines.append(f"- InFoBench items with context input: {bare['infobench_with_context_fraction']*100:.0f}%")
    lines.append("- InFoBench top-20 topical categories:")
    for k, v in bare["infobench_topical_top20"]:
        lines.append(f"    - {k}: {v}")
    lines.append("")
    lines.append("## Validation Coverage (per 83 skill types)")
    lines.append(f"- Types covered: **{val['types_covered_by_validation']}/83** "
                 f"({val['coverage_ratio']*100:.1f}%)")
    cbs = val["type_coverage_by_source"]
    lines.append(f"- IFEval types covered: {cbs['ifeval_types_covered']}/{cbs['ifeval_types_total']}")
    lines.append(f"- IFBench types covered: {cbs['ifbench_types_covered']}/{cbs['ifbench_types_total']}")
    lines.append(f"- Uncovered types ({len(val['uncovered_types'])}):")
    for t in val["uncovered_types"]:
        lines.append(f"    - {t}")
    lines.append("")
    lines.append("## Validation num_instructions distribution")
    lines.append(f"- IFBench: {val['val_num_instructions_distribution']['ifbench']}")
    lines.append(f"- ManyIFEval: {val['val_num_instructions_distribution']['manyifeval']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bare = analyze_bare()
    val = analyze_val_coverage()

    (OUT_DIR / "bare_distribution.json").write_text(json.dumps(bare, indent=2, ensure_ascii=False))
    (OUT_DIR / "val_coverage.json").write_text(json.dumps(val, indent=2, ensure_ascii=False))

    summary = write_summary(bare, val)
    (OUT_DIR / "distribution_summary.md").write_text(summary)

    print(summary)


if __name__ == "__main__":
    main()
