"""Merge AVG-era validation responses + ALO-only new responses → full ALO matrix.

For each (panel model, category):
  - Read mapping benchmarks/{cat}/validation_sample_alo_mapping.json
    (built by `build_validation_sample_alo_only.py`)
  - For each ALO item (in validation_sample_alo.jsonl order):
      - overlap → pull the row from results/{cat}/validation/validation_benchmark_response/
                  <panel_model>/response_matrix.jsonl, keyed by the mapped AVG prompt_id,
                  then re-key it to the ALO prompt_id.
      - alo_only → pull the row from results/{cat}/validation/
                   validation_benchmark_response_alo_only/<panel_model>/response_matrix.jsonl.

Output:
  results/{cat}/validation/validation_benchmark_response_alo/<panel_model>/response_matrix.jsonl
  results/{cat}/validation/validation_benchmark_response_alo/<panel_model>/metadata.json
    (records dual-source provenance + content-mapping reference)

Refuses to run on any cell where the required overlap or alo_only source is missing;
abort with a clear error so the user can finish inference first.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent.parent

CATS = ["harmful_refusal", "over_refusal", "constrained_completion", "sycophancy_resistance"]
PANEL_MODELS = [
    "Qwen_Qwen2.5_72B_Instruct_AWQ",
    "Qwen_Qwen3_30B_A3B_Instruct_2507",
    "google_gemma_3_27b_it",
    "Qwen_Qwen3_8B",
    "google_gemma_3_12b_it",
    "mistralai_Mistral_Small_3.1_24B_Instruct_2503",
    "google_gemma_3_4b_it",
    "mistralai_Mistral_Nemo_Instruct_2407",
    "deepseek_ai_DeepSeek_R1_Distill_Llama_8B",
]


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_jsonl_by_pid(path: Path) -> dict[str, dict]:
    return {row["prompt_id"]: row for row in load_jsonl(path)}


def merge_one(cat: str, model: str) -> dict:
    mapping_path = ROOT / "benchmarks" / cat / "validation_sample_alo_mapping.json"
    alo_bundle = ROOT / "benchmarks" / cat / "validation_sample_alo.jsonl"
    avg_resp = ROOT / "results" / cat / "validation" / "validation_benchmark_response" / model / "response_matrix.jsonl"
    only_resp = ROOT / "results" / cat / "validation" / "validation_benchmark_response_alo_only" / model / "response_matrix.jsonl"
    out_dir = ROOT / "results" / cat / "validation" / "validation_benchmark_response_alo" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_resp = out_dir / "response_matrix.jsonl"
    out_meta = out_dir / "metadata.json"

    with open(mapping_path) as f:
        mapping_payload = json.load(f)
    mapping: dict[str, str | None] = mapping_payload["mapping"]
    alo_items = load_jsonl(alo_bundle)
    n_alo_only = mapping_payload["n_alo_only"]

    if not avg_resp.exists():
        raise FileNotFoundError(f"[{cat}/{model}] missing AVG response matrix: {avg_resp}")
    avg_rows = load_jsonl_by_pid(avg_resp)
    if n_alo_only > 0:
        if not only_resp.exists():
            raise FileNotFoundError(
                f"[{cat}/{model}] missing alo_only response matrix: {only_resp} "
                f"(run inference on validation_sample_alo_only.jsonl first)"
            )
        only_rows = load_jsonl_by_pid(only_resp)
    else:
        only_rows = {}

    missing_overlap: list[str] = []
    missing_only: list[str] = []
    n_overlap_used = 0
    n_alo_only_used = 0
    with open(out_resp, "w") as f:
        for it in alo_items:
            alo_pid = it["prompt_id"]
            avg_pid = mapping.get(alo_pid)
            if avg_pid is None:
                # alo_only
                row = only_rows.get(alo_pid)
                if row is None:
                    missing_only.append(alo_pid)
                    continue
                n_alo_only_used += 1
            else:
                # overlap
                row = avg_rows.get(avg_pid)
                if row is None:
                    missing_overlap.append(f"{alo_pid}←{avg_pid}")
                    continue
                row = dict(row)  # shallow copy
                row["prompt_id"] = alo_pid  # re-key
                row.setdefault("metadata", {})
                row["metadata"] = {**row.get("metadata", {}),
                                   "source_response_prompt_id": avg_pid,
                                   "source_response_dataset": "validation_sample.jsonl"}
                n_overlap_used += 1
            f.write(json.dumps(row) + "\n")

    if missing_overlap or missing_only:
        raise RuntimeError(
            f"[{cat}/{model}] merge incomplete: "
            f"overlap missing={len(missing_overlap)} alo_only missing={len(missing_only)} "
            f"first_overlap_missing={missing_overlap[:3]} "
            f"first_alo_only_missing={missing_only[:3]}"
        )

    # Provenance metadata
    avg_meta_path = ROOT / "results" / cat / "validation" / "validation_benchmark_response" / model / "metadata.json"
    only_meta_path = ROOT / "results" / cat / "validation" / "validation_benchmark_response_alo_only" / model / "metadata.json"
    avg_meta = json.loads(avg_meta_path.read_text()) if avg_meta_path.exists() else {}
    only_meta = json.loads(only_meta_path.read_text()) if only_meta_path.exists() else {}

    meta = {
        "model": avg_meta.get("model") or only_meta.get("model"),
        "category": cat,
        "n_trials": avg_meta.get("n_trials") or only_meta.get("n_trials"),
        "n_prompts": len(alo_items),
        "n_overlap_items": n_overlap_used,
        "n_alo_only_items": n_alo_only_used,
        "dataset_path": f"benchmarks/{cat}/validation_sample_alo.jsonl",
        "content_mapping": f"benchmarks/{cat}/validation_sample_alo_mapping.json",
        "built_at": datetime.utcnow().isoformat(timespec="seconds"),
        "sources": {
            "overlap": {
                "response_matrix": str(avg_resp.relative_to(ROOT)),
                "metadata": str(avg_meta_path.relative_to(ROOT)) if avg_meta_path.exists() else None,
                "dataset_hash": avg_meta.get("dataset_hash"),
                "dataset_path": avg_meta.get("dataset_path"),
                "last_updated": avg_meta.get("last_updated"),
            },
            "alo_only": {
                "response_matrix": str(only_resp.relative_to(ROOT)) if n_alo_only > 0 else None,
                "metadata": str(only_meta_path.relative_to(ROOT)) if only_meta_path.exists() else None,
                "dataset_hash": only_meta.get("dataset_hash"),
                "dataset_path": only_meta.get("dataset_path"),
                "last_updated": only_meta.get("last_updated"),
            },
        },
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "cat": cat, "model": model,
        "n_overlap": n_overlap_used, "n_alo_only": n_alo_only_used,
        "n_total": len(alo_items),
        "out": str(out_resp.relative_to(ROOT)),
    }


def main() -> None:
    print(f"{'category':<26s} {'model':<48s} {'overlap':>8s} {'alo_only':>9s} {'total':>6s}")
    print("-" * 105)
    for cat in CATS:
        for m in PANEL_MODELS:
            r = merge_one(cat, m)
            print(f"{r['cat']:<26s} {r['model']:<48s} "
                  f"{r['n_overlap']:>8d} {r['n_alo_only']:>9d} {r['n_total']:>6d}")
        print()


if __name__ == "__main__":
    main()
