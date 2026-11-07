"""V3 leakage check: bare pool ↔ each validation tier (item-level disjoint).

Compares the bare request pool against each validation tier using both:
  (a) exact normalized string match (lowercased, whitespace collapsed)
  (b) sentence-transformer cosine similarity above a threshold (default 0.85)

Per principle V3, the bare pool and each validation tier should be disjoint.
This script outputs a leakage_report.json under results/data_prep/harmful_refusal/
listing any duplicates found, with sims and originating IDs, so the report can
ship as paper-appendix evidence that V3 is satisfied.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset  # noqa: F401  (kept for parity with other scripts)


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def compute_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return np.array(model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ))


def check_pair(
    bare: list[dict],
    val: list[dict],
    tier_name: str,
    sim_threshold: float,
    model_name: str,
) -> dict:
    bare_norm = {normalize(b["prompt"]): i for i, b in enumerate(bare)}
    exact_hits = []
    for j, v in enumerate(val):
        n = normalize(v["prompt"])
        if n in bare_norm:
            i = bare_norm[n]
            exact_hits.append({
                "tier": tier_name,
                "val_id": v["id"], "val_source": v["source"], "val_prompt": v["prompt"][:200],
                "bare_id": bare[i]["id"], "bare_source": bare[i]["source"],
            })

    bare_texts = [b["prompt"] for b in bare]
    val_texts = [v["prompt"] for v in val]
    print(f"  [{tier_name}] computing embeddings ({len(bare_texts)} bare + {len(val_texts)} val)...")
    embs = compute_embeddings(bare_texts + val_texts, model_name)
    bare_embs, val_embs = embs[:len(bare)], embs[len(bare):]
    sim_matrix = val_embs @ bare_embs.T

    semantic_hits = []
    for j in range(len(val)):
        max_idx = int(sim_matrix[j].argmax())
        max_sim = float(sim_matrix[j, max_idx])
        if max_sim >= sim_threshold:
            semantic_hits.append({
                "tier": tier_name,
                "sim": round(max_sim, 4),
                "val_id": val[j]["id"], "val_source": val[j]["source"],
                "val_prompt": val[j]["prompt"][:200],
                "bare_id": bare[max_idx]["id"], "bare_source": bare[max_idx]["source"],
                "bare_prompt": bare[max_idx]["prompt"][:200],
            })

    return {
        "tier": tier_name,
        "n_val": len(val),
        "n_bare": len(bare),
        "exact_match_count": len(exact_hits),
        "semantic_match_count": len(semantic_hits),
        "sim_threshold": sim_threshold,
        "exact_hits": exact_hits,
        "semantic_hits": semantic_hits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bare", type=str, default="data/harmful_refusal/bare_requests.jsonl")
    parser.add_argument("--val-dir", type=str, default="data/harmful_refusal/validation")
    parser.add_argument("--report", type=str,
                        default="results/data_prep/harmful_refusal/leakage_report.json")
    parser.add_argument("--sim-threshold", type=float, default=0.85)
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    bare = load_jsonl(Path(args.bare))
    print(f"Loaded bare pool: {len(bare)} items from {args.bare}")

    val_dir = Path(args.val_dir)
    tiers = {}
    for tier in ("easy", "medium", "hard"):
        tiers[tier] = load_jsonl(val_dir / f"{tier}.jsonl")
        print(f"Loaded {tier}: {len(tiers[tier])} items")

    print()
    results = {}
    for tier_name, val in tiers.items():
        print(f"Checking bare ↔ {tier_name}...")
        results[tier_name] = check_pair(bare, val, tier_name, args.sim_threshold, args.model)
        r = results[tier_name]
        print(f"  exact:    {r['exact_match_count']}")
        print(f"  semantic: {r['semantic_match_count']} (sim ≥ {args.sim_threshold})")
        print()

    total_exact = sum(r["exact_match_count"] for r in results.values())
    total_semantic = sum(r["semantic_match_count"] for r in results.values())
    v3_exact_satisfied = total_exact == 0
    v3_semantic_satisfied = total_semantic == 0

    summary = {
        "bare_pool_size": len(bare),
        "validation_tier_sizes": {k: len(v) for k, v in tiers.items()},
        "sim_threshold": args.sim_threshold,
        "embedding_model": args.model,
        "total_exact_matches": total_exact,
        "total_semantic_matches": total_semantic,
        "v3_exact_disjoint": v3_exact_satisfied,
        "v3_semantic_disjoint": v3_semantic_satisfied,
        "tiers": results,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"=== V3 leakage check summary ===")
    print(f"  exact disjoint:    {v3_exact_satisfied}  ({total_exact} hits)")
    print(f"  semantic disjoint: {v3_semantic_satisfied}  ({total_semantic} hits at sim ≥ {args.sim_threshold})")
    print(f"\nFull report → {report_path}")


if __name__ == "__main__":
    main()
