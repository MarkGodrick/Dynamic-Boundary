"""Build validation benchmark for constrained_completion (Option Δ′).

Validation pool = IFBench 300 + ManyIFEval 2160 = 2460 items (mixed pool, no tiers).
Per-item metadata: `source` and `num_instructions` (axes for Rasch β analysis).

Inputs:
  - allenai/IFBench_test HF dataset                                (300 items)
  - /tmp/probe_manyifeval/.../data/manyifeval.jsonl                 (2160 items)

Outputs:
  - data/constrained_completion/validation/ifbench.jsonl            (300)
  - data/constrained_completion/validation/manyifeval.jsonl         (2160)
  - data/constrained_completion/validation/all.jsonl                (2460 merged, id-assigned)
"""

import json
from collections import Counter
from datetime import date
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent.parent
VAL_DIR = ROOT / "data" / "constrained_completion" / "validation"
MANYIFEVAL_LOCAL = Path("/tmp/probe_manyifeval/Multiple-Instructions-Following/data/manyifeval.jsonl")

IFBENCH_OUT = VAL_DIR / "ifbench.jsonl"
MIE_OUT = VAL_DIR / "manyifeval.jsonl"
ALL_OUT = VAL_DIR / "all.jsonl"

IFBENCH_VERSION = "allenai/IFBench_test HF dataset (Pyatkin et al. 2025, NeurIPS 2025)"
MIE_VERSION = "kenoharada/Multiple-Instructions-Following@main (Kato et al. 2025)"
DOWNLOAD_DATE = str(date.today())


def build_ifbench() -> list[dict]:
    ds = load_dataset("allenai/IFBench_test", split="train")
    print(f"IFBench: {len(ds)} items")

    items = []
    for row in ds:
        # Filter null/empty kwargs per constraint
        filtered_kwargs = []
        for kw in row["kwargs"]:
            filtered = {k: v for k, v in kw.items() if v is not None and v != "" and v != []}
            filtered_kwargs.append(filtered)

        items.append({
            "prompt": row["prompt"],
            "source": "ifbench",
            "source_id": row["key"],
            "source_version": IFBENCH_VERSION,
            "download_date": DOWNLOAD_DATE,
            "instruction_id_list": row["instruction_id_list"],
            "kwargs": filtered_kwargs,
            "num_instructions": len(row["instruction_id_list"]),
            "source_meta": {
                "origin": "wildchat_held_out_user_queries",
            },
        })
    return items


def build_manyifeval() -> list[dict]:
    if not MANYIFEVAL_LOCAL.exists():
        raise FileNotFoundError(f"{MANYIFEVAL_LOCAL} missing — clone ManyIFEval first")
    items = []
    with MANYIFEVAL_LOCAL.open() as f:
        raw = [json.loads(l) for l in f if l.strip()]
    print(f"ManyIFEval: {len(raw)} items")

    for row in raw:
        # Drop kwarg nulls per constraint to match IFBench shape
        filtered_kwargs = []
        for kw in row["kwargs"]:
            if isinstance(kw, str):
                kw = json.loads(kw)
            filtered = {k: v for k, v in kw.items() if v is not None and v != "" and v != []}
            filtered_kwargs.append(filtered)

        items.append({
            "prompt": row["prompt"],
            "source": "manyifeval",
            "source_id": f"{row['key']}_n{len(row['instruction_id_list'])}",
            "source_version": MIE_VERSION,
            "download_date": DOWNLOAD_DATE,
            "instruction_id_list": row["instruction_id_list"],
            "kwargs": filtered_kwargs,
            "num_instructions": len(row["instruction_id_list"]),
            "source_meta": {
                "task_prompt": row["task_prompt"],
                "ifeval_key": row["key"],
            },
        })
    return items


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main() -> None:
    ifbench = build_ifbench()
    manyifeval = build_manyifeval()

    write_jsonl(IFBENCH_OUT, ifbench)
    print(f"Wrote {IFBENCH_OUT}: {len(ifbench)}")

    write_jsonl(MIE_OUT, manyifeval)
    print(f"Wrote {MIE_OUT}: {len(manyifeval)}")

    # Merged pool with global ids
    merged = []
    for idx, it in enumerate(ifbench + manyifeval):
        it = {**it, "id": idx}
        merged.append(it)
    write_jsonl(ALL_OUT, merged)
    print(f"Wrote {ALL_OUT}: {len(merged)}")

    # Summary
    print("\n--- Validation pool summary ---")
    print(f"Total: {len(merged)}")
    print(f"By source: {dict(Counter(i['source'] for i in merged))}")
    print(f"num_instructions distribution: {dict(sorted(Counter(i['num_instructions'] for i in merged).items()))}")


if __name__ == "__main__":
    main()
