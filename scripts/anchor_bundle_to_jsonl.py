"""Convert anchor_bundle_M9.json into a jsonl dataset that
collect_response_matrix.py can consume.

Each output row carries the fields collect_response_matrix.py / panel_inference
expect (prompt_id, prompt, category, metadata) plus beta for downstream
analysis. No source_model field is emitted, so all dispatched models will
evaluate every anchor item (own_entries stays empty).

Usage:
    python scripts/anchor_bundle_to_jsonl.py \\
        benchmarks/harmful_refusal/anchor_bundle_M9.json \\
        benchmarks/harmful_refusal/anchor_bundle_M9.jsonl
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="path to anchor_bundle_M9.json")
    ap.add_argument("output", help="path to anchor_bundle_M9.jsonl")
    args = ap.parse_args()

    bundle = json.loads(Path(args.input).read_text())
    items = bundle["items"]
    cat = bundle["category"]

    seen: set[str] = set()
    rows = []
    for it in items:
        pid = it["item_id"]
        if pid in seen:
            raise ValueError(f"duplicate prompt_id in anchor bundle: {pid}")
        seen.add(pid)
        rows.append({
            "prompt_id": pid,
            "prompt": it["prompt"],
            "category": cat,
            "metadata": it.get("metadata", {}),
            "beta": it["beta"],
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows -> {out} (category={cat}, M={bundle['M']}, N={bundle['N']})")


if __name__ == "__main__":
    main()
