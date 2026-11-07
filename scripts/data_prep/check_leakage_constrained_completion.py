"""V3 leakage check: bare × validation for constrained_completion.

Checks:
  1. Exact string match (case-normalized)
  2. Embedding cosine similarity at threshold 0.85 (MiniLM-L6-v2)

Outputs:
  results/data_prep/constrained_completion/leakage_report.json

Expected result (Option Δ′):
  - bare 325 IFEval ∩ IFBench 300 validation: 0 (different prompt sources)
  - bare 325 IFEval ∩ ManyIFEval 2160 validation: 0 (216 overlap keys removed)
  - bare ~480 InFoBench ∩ both validation tiers: should be 0 (different provenance)
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "constrained_completion"
RESULTS_DIR = ROOT / "results" / "data_prep" / "constrained_completion"

BARE = DATA_DIR / "bare_requests.jsonl"
VAL_IFBENCH = DATA_DIR / "validation" / "ifbench.jsonl"
VAL_MIE = DATA_DIR / "validation" / "manyifeval.jsonl"
REPORT = RESULTS_DIR / "leakage_report.json"

EMBED_THRESHOLD = 0.85


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


def exact_overlap(bare: list[dict], val: list[dict]) -> list[dict]:
    bare_norm = {normalize(b["prompt"]): b for b in bare}
    hits = []
    for v in val:
        n = normalize(v["prompt"])
        if n in bare_norm:
            b = bare_norm[n]
            hits.append({
                "bare_id": b["id"],
                "bare_source": b["source"],
                "bare_source_id": b["source_id"],
                "val_source": v["source"],
                "val_source_id": v["source_id"],
            })
    return hits


def embed_overlap(bare: list[dict], val: list[dict], threshold: float) -> list[dict]:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    bare_emb = model.encode([b["prompt"] for b in bare], normalize_embeddings=True, batch_size=64)
    val_emb = model.encode([v["prompt"] for v in val], normalize_embeddings=True, batch_size=64)
    sim = np.asarray(val_emb) @ np.asarray(bare_emb).T  # (V, B)
    hits = []
    for vi in range(len(val)):
        for bi in range(len(bare)):
            s = float(sim[vi, bi])
            if s >= threshold:
                hits.append({
                    "bare_id": bare[bi]["id"],
                    "bare_source": bare[bi]["source"],
                    "bare_source_id": bare[bi]["source_id"],
                    "val_source": val[vi]["source"],
                    "val_source_id": val[vi]["source_id"],
                    "cosine": s,
                })
    return hits


def main() -> None:
    bare = load_jsonl(BARE)
    val_ifbench = load_jsonl(VAL_IFBENCH)
    val_mie = load_jsonl(VAL_MIE)
    print(f"Bare: {len(bare)} ({dict(Counter(b['source'] for b in bare))})")
    print(f"Val IFBench: {len(val_ifbench)}")
    print(f"Val ManyIFEval: {len(val_mie)}")

    report = {"threshold": EMBED_THRESHOLD}

    for vname, val in [("ifbench", val_ifbench), ("manyifeval", val_mie)]:
        print(f"\n--- bare × {vname} ---")
        exact = exact_overlap(bare, val)
        print(f"  exact match: {len(exact)}")
        embed = embed_overlap(bare, val, EMBED_THRESHOLD)
        print(f"  embed cos≥{EMBED_THRESHOLD}: {len(embed)}")
        report[vname] = {
            "n_val": len(val),
            "exact_hits": exact,
            "embed_hits": embed,
            "exact_count": len(exact),
            "embed_count": len(embed),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {REPORT}")

    # Fail hard if any exact leakage
    total_exact = sum(report[k]["exact_count"] for k in ("ifbench", "manyifeval"))
    if total_exact > 0:
        print(f"\n⚠  TOTAL EXACT LEAKS: {total_exact} — V3 VIOLATION")
    else:
        print(f"\n✓  V3 exact-disjoint: 0 leaks across both validation tiers")


if __name__ == "__main__":
    main()
