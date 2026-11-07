"""Build unified over_refusal bare request pool with full provenance metadata.

Pillar-1 of the data sourcing pipeline for Category A' (over_refusal).
See docs/data_sourcing_principles.md §3.1.

Sources (all participate in one global embedding dedup pass):
  1. XSTest safe (250) — Röttger et al. NAACL 2024, hand-crafted over-refusal
     test prompts across 10 trigger types.  We filter label=="safe" only.
  2. OR-Bench-Hard-1K (~1320) — Cui et al. ICML 2025, LLM-rewritten prompts
     that are hard even for SOTA models to not over-refuse.
  3. Author-handwritten gap-fill (~30) — professional/technical dual-use
     edge cases absent from the two benchmarks.

FalseReject (Zhang et al. COLM 2025) is intentionally NOT in the bare pool.
It is reserved for the Pillar-3 validation benchmark (V3 disjoint).

Output schema (per line of bare_requests.jsonl):
    id              : int   global id after dedup
    prompt          : str
    source          : str   xstest / orbench_hard / author_handwritten
    source_id       : int   index in the original dataset
    source_version  : str   HF dataset path or version marker
    download_date   : str   YYYY-MM-DD
    source_method   : str   human-written / llm-rewritten
    category        : str   XSTest type or OR-Bench rejection category
    source_meta     : dict  extra fields from source (focus, note, etc.)

Usage:
    conda run --no-capture-output -n boundary python -u \
        scripts/data_prep/build_bare_over_refusal.py
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


DOWNLOAD_DATE = date.today().isoformat()


def load_xstest_safe() -> list[dict]:
    """Load XSTest safe prompts (label == 'safe')."""
    ds = load_dataset("Paul/XSTest", split="train")
    out = []
    for i, row in enumerate(ds):
        if row.get("label") != "safe":
            continue
        out.append({
            "prompt": row["prompt"],
            "source": "xstest",
            "source_id": row.get("id", i),
            "source_version": "Paul/XSTest (Röttger et al. NAACL 2024)",
            "download_date": DOWNLOAD_DATE,
            "source_method": "human-written",
            "category": row.get("type", ""),
            "source_meta": {
                "focus": row.get("focus", ""),
                "note": row.get("note", ""),
            },
        })
    return out


def load_orbench_hard() -> list[dict]:
    """Load OR-Bench-Hard-1K subset."""
    ds = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
    return [
        {
            "prompt": row["prompt"],
            "source": "orbench_hard",
            "source_id": i,
            "source_version": "bench-llm/or-bench:or-bench-hard-1k (Cui et al. ICML 2025)",
            "download_date": DOWNLOAD_DATE,
            "source_method": "llm-rewritten",
            "category": row.get("category", ""),
            "source_meta": {},
        }
        for i, row in enumerate(ds)
    ]


def load_handwritten(path: Path) -> list[dict]:
    """Optional gap-fill items.  Returns [] if file absent."""
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            out.append({
                "prompt": raw["prompt"],
                "source": raw.get("source", "author_handwritten"),
                "source_id": raw.get("source_id", i),
                "source_version": raw.get("source_version", "gap_fill_v1"),
                "download_date": raw.get("download_date", DOWNLOAD_DATE),
                "source_method": raw.get("source_method", "human-written"),
                "category": raw.get("category", ""),
                "source_meta": raw.get("source_meta", {}),
            })
    return out


def compute_embeddings(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return np.array(model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    ))


def deduplicate(
    existing: list[dict],
    candidates: list[dict],
    threshold: float,
    model_name: str,
) -> tuple[list[dict], list[dict]]:
    """Return (kept, dropped).  Dropped items get a _dedup_against field."""
    if not candidates:
        return [], []
    if not existing:
        return list(candidates), []

    existing_texts = [e["prompt"] for e in existing]
    candidate_texts = [c["prompt"] for c in candidates]
    all_embs = compute_embeddings(existing_texts + candidate_texts, model_name)
    existing_embs = all_embs[:len(existing_texts)]
    candidate_embs = all_embs[len(existing_texts):]
    sim_matrix = candidate_embs @ existing_embs.T

    kept, kept_embs, dropped = [], [], []
    for i, cand in enumerate(tqdm(candidates, desc="Dedup")):
        sims_existing = sim_matrix[i]
        max_idx = int(sims_existing.argmax())
        max_sim_existing = float(sims_existing[max_idx])
        if max_sim_existing >= threshold:
            cand = dict(cand)
            cand["_dedup_against"] = {
                "source": existing[max_idx]["source"],
                "source_id": existing[max_idx]["source_id"],
                "sim": round(max_sim_existing, 4),
            }
            dropped.append(cand)
            continue

        if kept_embs:
            kept_arr = np.stack(kept_embs)
            sims_kept = candidate_embs[i] @ kept_arr.T
            max_idx_k = int(sims_kept.argmax())
            max_sim_kept = float(sims_kept[max_idx_k])
            if max_sim_kept >= threshold:
                cand = dict(cand)
                cand["_dedup_against"] = {
                    "source": kept[max_idx_k]["source"],
                    "source_id": kept[max_idx_k]["source_id"],
                    "sim": round(max_sim_kept, 4),
                }
                dropped.append(cand)
                continue

        kept.append(cand)
        kept_embs.append(candidate_embs[i])

    return kept, dropped


def main():
    parser = argparse.ArgumentParser(description="Build unified over_refusal bare request pool")
    parser.add_argument("--sim-threshold", type=float, default=0.85)
    parser.add_argument("--output", type=str,
                        default="data/over_refusal/bare_requests.jsonl")
    parser.add_argument("--dedup-report", type=str,
                        default="results/data_prep/over_refusal/bare_dedup_report.json")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--handwritten", type=str,
                        default="data/over_refusal/gap_fill_items.jsonl")
    args = parser.parse_args()

    print("Loading sources...")
    xstest = load_xstest_safe()
    orbench = load_orbench_hard()
    handwritten = load_handwritten(Path(args.handwritten))
    print(f"  XSTest safe:        {len(xstest)}")
    print(f"  OR-Bench-Hard-1K:   {len(orbench)}")
    print(f"  Handwritten:        {len(handwritten)}" + (
        f" (from {args.handwritten})" if handwritten else " (absent)"))
    raw_total = len(xstest) + len(orbench) + len(handwritten)
    print(f"  Raw total:          {raw_total}")

    # XSTest as anchor (canonical, hand-crafted, smaller)
    pool = list(xstest)
    all_dropped = []

    print(f"\nDedup OR-Bench-Hard against XSTest...")
    kept, dropped = deduplicate(pool, orbench, args.sim_threshold, args.model)
    print(f"  Kept {len(kept)} / {len(orbench)} from OR-Bench-Hard (dropped {len(dropped)})")
    pool.extend(kept)
    all_dropped.extend(dropped)

    if handwritten:
        print(f"\nDedup handwritten against pool...")
        kept, dropped = deduplicate(pool, handwritten, args.sim_threshold, args.model)
        print(f"  Kept {len(kept)} / {len(handwritten)} from handwritten (dropped {len(dropped)})")
        pool.extend(kept)
        all_dropped.extend(dropped)

    from collections import Counter
    source_counts = Counter(item["source"] for item in pool)
    print(f"\n=== Final pool: {len(pool)} ({len(all_dropped)} removed by dedup) ===")
    for src, cnt in source_counts.most_common():
        print(f"  {src}: {cnt}")

    method_counts = Counter(item["source_method"] for item in pool)
    print(f"\nGeneration method distribution:")
    for m, cnt in method_counts.most_common():
        print(f"  {m}: {cnt}")

    cat_counts = Counter(item["category"] for item in pool)
    print(f"\nCategory distribution:")
    for cat, cnt in cat_counts.most_common():
        print(f"  {cat}: {cnt}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, item in enumerate(pool):
            item["id"] = i
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nSaved bare pool to {out_path}")

    report_path = Path(args.dedup_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "sim_threshold": args.sim_threshold,
            "embedding_model": args.model,
            "raw_total": raw_total,
            "final_total": len(pool),
            "removed": len(all_dropped),
            "by_source": dict(source_counts),
            "by_method": dict(method_counts),
            "by_category": dict(cat_counts),
            "dropped_items": all_dropped,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved dedup report to {report_path}")


if __name__ == "__main__":
    main()
