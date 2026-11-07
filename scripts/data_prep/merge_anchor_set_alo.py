"""Merge AVG-era overlap responses + new ALO-only responses → full ALO matrix.

For each (V model, category):
  - overlap items  = rows in results/{cat}/anchor_set/<model>/response_matrix.jsonl
                     whose prompt_id is in benchmarks/{cat}/anchor_alo_bundle_M9.jsonl
                     (= ALO ∩ AVG, since anchor_set was built on the AVG anchor)
  - alo_only items = rows in results/{cat}/anchor_set_alo_only/<model>/response_matrix.jsonl
                     (already scoped to anchor_alo_only_bundle_M9.jsonl)
  - merged matrix  = overlap + alo_only, ordered to match the ALO bundle's prompt_id sequence

Output:
  results/{cat}/anchor_set_alo/<model>/response_matrix.jsonl
  results/{cat}/anchor_set_alo/<model>/metadata.json   (provenance pointers)

Run:
  conda run -n boundary --no-capture-output python -u scripts/data_prep/merge_anchor_set_alo.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

CATS = ["harmful_refusal", "over_refusal", "constrained_completion", "sycophancy_resistance"]
V_MODELS = [
    "allenai_Olmo_3_7B_Instruct",
    "ibm_granite_granite_4.1_8b",
    "microsoft_phi_4",
    "microsoft_Phi_4_reasoning_plus",
    "tiiuae_Falcon_H1R_7B",
]


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_bundle_ids(path: Path) -> list[str]:
    return [row["prompt_id"] for row in load_jsonl(path)]


def merge_one(cat: str, model_key: str) -> dict:
    bundle_path = ROOT / "benchmarks" / cat / "anchor_alo_bundle_M9.jsonl"
    alo_only_path = ROOT / "benchmarks" / cat / "anchor_alo_only_bundle_M9.jsonl"
    src_overlap = ROOT / "results" / cat / "anchor_set" / model_key / "response_matrix.jsonl"
    src_alo_only = ROOT / "results" / cat / "anchor_set_alo_only" / model_key / "response_matrix.jsonl"
    out_dir = ROOT / "results" / cat / "anchor_set_alo" / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_resp = out_dir / "response_matrix.jsonl"
    out_meta = out_dir / "metadata.json"

    alo_ids = load_bundle_ids(bundle_path)
    alo_only_ids = set(load_bundle_ids(alo_only_path))
    overlap_ids = [pid for pid in alo_ids if pid not in alo_only_ids]

    overlap_rows = {row["prompt_id"]: row for row in load_jsonl(src_overlap)}
    alo_only_rows = {row["prompt_id"]: row for row in load_jsonl(src_alo_only)}

    missing_overlap = [pid for pid in overlap_ids if pid not in overlap_rows]
    missing_alo_only = [pid for pid in alo_only_ids if pid not in alo_only_rows]
    if missing_overlap or missing_alo_only:
        raise RuntimeError(
            f"[{cat}/{model_key}] missing rows: "
            f"overlap={len(missing_overlap)} alo_only={len(missing_alo_only)} "
            f"first_overlap_missing={missing_overlap[:3]} "
            f"first_alo_only_missing={missing_alo_only[:3]}"
        )

    # Write merged matrix ordered by anchor_alo_bundle prompt_id sequence.
    n_overlap_used = 0
    n_alo_only_used = 0
    with open(out_resp, "w") as f:
        for pid in alo_ids:
            if pid in alo_only_ids:
                row = alo_only_rows[pid]
                n_alo_only_used += 1
            else:
                row = overlap_rows[pid]
                n_overlap_used += 1
            f.write(json.dumps(row) + "\n")

    assert n_overlap_used == len(overlap_ids)
    assert n_alo_only_used == len(alo_only_ids)
    assert n_overlap_used + n_alo_only_used == len(alo_ids)

    src_meta_overlap = ROOT / "results" / cat / "anchor_set" / model_key / "metadata.json"
    src_meta_alo_only = ROOT / "results" / cat / "anchor_set_alo_only" / model_key / "metadata.json"
    with open(src_meta_overlap) as f:
        meta_overlap = json.load(f)
    with open(src_meta_alo_only) as f:
        meta_alo_only = json.load(f)

    meta = {
        "model": meta_alo_only["model"],
        "category": cat,
        "n_trials": meta_alo_only["n_trials"],
        "n_prompts": len(alo_ids),
        "n_overlap_items": n_overlap_used,
        "n_alo_only_items": n_alo_only_used,
        "dataset_path": f"benchmarks/{cat}/anchor_alo_bundle_M9.jsonl",
        "built_at": datetime.utcnow().isoformat(timespec="seconds"),
        "sources": {
            "overlap": {
                "response_matrix": str(src_overlap.relative_to(ROOT)),
                "metadata": str(src_meta_overlap.relative_to(ROOT)),
                "dataset_hash": meta_overlap.get("dataset_hash"),
                "dataset_path": meta_overlap.get("dataset_path"),
                "last_updated": meta_overlap.get("last_updated"),
            },
            "alo_only": {
                "response_matrix": str(src_alo_only.relative_to(ROOT)),
                "metadata": str(src_meta_alo_only.relative_to(ROOT)),
                "dataset_hash": meta_alo_only.get("dataset_hash"),
                "dataset_path": meta_alo_only.get("dataset_path"),
                "last_updated": meta_alo_only.get("last_updated"),
            },
        },
        "judge_revisions_inherited": {
            "overlap": meta_overlap.get("judge_revision"),
            "alo_only": meta_alo_only.get("judge_revision"),
        },
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "cat": cat,
        "model": model_key,
        "n_overlap": n_overlap_used,
        "n_alo_only": n_alo_only_used,
        "n_total": len(alo_ids),
        "out": str(out_resp.relative_to(ROOT)),
    }


def main() -> None:
    print(f"{'category':<26s} {'model':<32s} {'overlap':>8s} {'alo_only':>9s} {'total':>6s}")
    print("-" * 90)
    for cat in CATS:
        for m in V_MODELS:
            r = merge_one(cat, m)
            print(f"{r['cat']:<26s} {r['model']:<32s} {r['n_overlap']:>8d} {r['n_alo_only']:>9d} {r['n_total']:>6d}")
        print()


if __name__ == "__main__":
    main()
