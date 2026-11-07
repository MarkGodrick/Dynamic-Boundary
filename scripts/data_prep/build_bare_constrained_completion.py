"""Build unified bare request pool for constrained_completion (Option Δ′).

Pipeline:
  Step 1 (extract-ifeval):  google/IFEval 541 → parallel DeepSeek base-task
                            extraction → 541 extracted → filter 216 ManyIFEval
                            overlap → 325 IFEval items
  Step 2 (extract-infobench): kqsong/InFoBench 500 → parallel DeepSeek
                              base-task extraction → ~500 raw
  Step 3 (merge-dedup):     325 + ~500 → embedding dedup at cos≥0.85 → ~823

All LLM extraction runs in parallel via ThreadPoolExecutor (workers=20).
Inputs come from HuggingFace directly — no local .bak dependencies.

Outputs:
  - data/constrained_completion/bare_requests_raw.jsonl               (pre-dedup merged)
  - data/constrained_completion/bare_requests.jsonl                   (post-dedup final)
  - results/data_prep/constrained_completion/bare_dedup_report.json   (dedup audit)

Usage:
  python scripts/data_prep/build_bare_constrained_completion.py --step all
  python scripts/data_prep/build_bare_constrained_completion.py --step extract-ifeval
  python scripts/data_prep/build_bare_constrained_completion.py --step extract-infobench
  python scripts/data_prep/build_bare_constrained_completion.py --step merge-dedup
"""

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "data_prep"))

# Reuse the LLM extraction prompt from the existing script
from build_constrained_completion import EXTRACT_PROMPT  # noqa: E402

DATA_DIR = ROOT / "data" / "constrained_completion"
RESULTS_DIR = ROOT / "results" / "data_prep" / "constrained_completion"
MANYIFEVAL_LOCAL = Path("/tmp/probe_manyifeval/Multiple-Instructions-Following/data/manyifeval.jsonl")

IFEVAL_325_OUT = DATA_DIR / "bare_requests_ifeval_filtered.jsonl"
INFOBENCH_RAW_OUT = DATA_DIR / "bare_requests_infobench_raw.jsonl"
MERGED_RAW_OUT = DATA_DIR / "bare_requests_raw.jsonl"
FINAL_OUT = DATA_DIR / "bare_requests.jsonl"
DEDUP_REPORT = RESULTS_DIR / "bare_dedup_report.json"

IFEVAL_SOURCE_VERSION = "google/IFEval HF dataset (Zhou et al. 2023, NeurIPS 2024)"
INFOBENCH_SOURCE_VERSION = "kqsong/InFoBench HF dataset (Qin et al. 2024, ACL Findings)"
DOWNLOAD_DATE = str(date.today())
WORKERS = 20


def _make_deepseek_client():
    from openai import OpenAI
    api_key = os.environ.get("DEEPSEEK_API_TOKEN")
    if not api_key:
        raise ValueError("DEEPSEEK_API_TOKEN not set")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def _extract_base_task(prompt_text: str, client) -> str:
    """Call DeepSeek to strip constraints from a prompt and return the base task."""
    messages = [{"role": "user", "content": EXTRACT_PROMPT.format(prompt=prompt_text)}]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            base = resp.choices[0].message.content.strip()
            if base.startswith('"') and base.endswith('"'):
                base = base[1:-1]
            return base
        except Exception as e:
            print(f"  API error attempt {attempt+1}: {e}", file=sys.stderr, flush=True)
            time.sleep(2)
    return prompt_text  # fallback to original


# ---------------------------------------------------------------------------
# Step 1: Extract IFEval base tasks (parallel) + filter 216 ManyIFEval overlap
# ---------------------------------------------------------------------------

def get_manyifeval_keys() -> set[int]:
    """Return the 216 IFEval `key` values used by ManyIFEval."""
    if not MANYIFEVAL_LOCAL.exists():
        raise FileNotFoundError(
            f"{MANYIFEVAL_LOCAL} not found. Clone ManyIFEval first:\n"
            "  git clone https://github.com/kenoharada/Multiple-Instructions-Following.git "
            "/tmp/probe_manyifeval/Multiple-Instructions-Following"
        )
    keys = set()
    with MANYIFEVAL_LOCAL.open() as f:
        for line in f:
            if line.strip():
                keys.add(json.loads(line)["key"])
    return keys


def extract_ifeval() -> None:
    """Load google/IFEval, parallel-extract base tasks, filter to 325."""
    from datasets import load_dataset

    mie_keys = get_manyifeval_keys()
    print(f"ManyIFEval uses {len(mie_keys)} IFEval keys", flush=True)

    ds = load_dataset("google/IFEval", split="train")
    rows = [r for r in ds if r["key"] not in mie_keys]
    print(f"IFEval 541 → {len(rows)} after filtering overlap", flush=True)
    assert len(rows) == 325, f"expected 325 after filter, got {len(rows)}"

    client = _make_deepseek_client()
    results: dict[int, str] = {}
    lock = threading.Lock()
    done = [0]

    def worker(idx: int, row: dict) -> tuple[int, str]:
        return idx, _extract_base_task(row["prompt"], client)

    print(f"Launching {WORKERS} workers for {len(rows)} IFEval extractions...", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(worker, i, r): i for i, r in enumerate(rows)}
        for fut in as_completed(futures):
            idx, base = fut.result()
            results[idx] = base
            with lock:
                done[0] += 1
                k = done[0]
            if k % 25 == 0 or k == len(rows):
                print(f"  [{k}/{len(rows)}] extracted", flush=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with IFEVAL_325_OUT.open("w") as f:
        for i, row in enumerate(rows):
            entry = {
                "prompt": results[i],
                "source": "ifeval",
                "source_id": row["key"],
                "source_version": IFEVAL_SOURCE_VERSION,
                "download_date": DOWNLOAD_DATE,
                "source_method": "author-written",
                "source_meta": {
                    "original_prompt": row["prompt"],
                    "instruction_id_list": row["instruction_id_list"],
                    "manyifeval_filter": True,
                },
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {IFEVAL_325_OUT}", flush=True)


# ---------------------------------------------------------------------------
# Step 2: LLM extract base tasks from InFoBench 500
# ---------------------------------------------------------------------------

def build_infobench_input(row: dict) -> str:
    """Concat context + instruction; DeepSeek strips constraints next."""
    instruction = row["instruction"]
    context = row.get("input") or ""
    if context.strip():
        return f"{context.strip()}\n\n{instruction.strip()}"
    return instruction.strip()


def extract_infobench() -> None:
    """Load kqsong/InFoBench, parallel-extract base tasks, write raw jsonl."""
    from datasets import load_dataset

    ds = load_dataset("kqsong/InFoBench", split="train")
    print(f"InFoBench total: {len(ds)}", flush=True)

    client = _make_deepseek_client()
    results: dict[int, str] = {}
    lock = threading.Lock()
    done = [0]

    rows = list(ds)

    def worker(idx: int, row: dict) -> tuple[int, str]:
        raw_input = build_infobench_input(row)
        return idx, _extract_base_task(raw_input, client)

    print(f"Launching {WORKERS} workers for {len(rows)} InFoBench extractions...", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(worker, i, r): i for i, r in enumerate(rows)}
        for fut in as_completed(futures):
            idx, base = fut.result()
            results[idx] = base
            with lock:
                done[0] += 1
                k = done[0]
            if k % 25 == 0 or k == len(rows):
                print(f"  [{k}/{len(rows)}] extracted", flush=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with INFOBENCH_RAW_OUT.open("w") as f:
        for i, row in enumerate(rows):
            raw_input = build_infobench_input(row)
            entry = {
                "prompt": results[i],
                "source": "infobench",
                "source_id": row["id"],
                "source_version": INFOBENCH_SOURCE_VERSION,
                "download_date": DOWNLOAD_DATE,
                "source_method": "author-written",
                "source_meta": {
                    "original_prompt": raw_input,
                    "category": row["category"],
                    "subset": row["subset"],
                    "has_context_input": bool(row.get("input", "").strip()),
                },
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"InFoBench raw extracted: {len(rows)} → {INFOBENCH_RAW_OUT}", flush=True)


# ---------------------------------------------------------------------------
# Step 3: Merge + embedding dedup
# ---------------------------------------------------------------------------

def merge_and_dedup(threshold: float = 0.85) -> None:
    all_items = []
    for path in [IFEVAL_325_OUT, INFOBENCH_RAW_OUT]:
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run earlier step first")
        with path.open() as f:
            for line in f:
                if line.strip():
                    all_items.append(json.loads(line))
    print(f"Merged raw: {len(all_items)} items")
    print(f"  by source: {Counter(i['source'] for i in all_items)}")

    with MERGED_RAW_OUT.open("w") as f:
        for it in all_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Embedding dedup
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [it["prompt"] for it in all_items]
    print(f"Computing embeddings for {len(texts)} prompts...")
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    emb = np.asarray(emb)

    sim = emb @ emb.T
    kept_idx: list[int] = []
    dropped_log: list[dict] = []
    for i in range(len(all_items)):
        collision = None
        for j in kept_idx:
            if sim[i, j] > threshold:
                collision = (j, float(sim[i, j]))
                break
        if collision is None:
            kept_idx.append(i)
        else:
            dropped_log.append({
                "dropped_index": i,
                "dropped_source": all_items[i]["source"],
                "dropped_source_id": all_items[i]["source_id"],
                "collides_with_index": collision[0],
                "collides_with_source": all_items[collision[0]]["source"],
                "collides_with_source_id": all_items[collision[0]]["source_id"],
                "cosine": collision[1],
            })

    kept = [all_items[i] for i in kept_idx]
    print(f"Post-dedup: {len(kept)} (dropped {len(dropped_log)})")
    print(f"  by source: {Counter(i['source'] for i in kept)}")

    with FINAL_OUT.open("w") as f:
        for new_id, item in enumerate(kept):
            item["id"] = new_id
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DEDUP_REPORT.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "raw_count": len(all_items),
                "final_count": len(kept),
                "dropped_count": len(dropped_log),
                "by_source_raw": dict(Counter(i["source"] for i in all_items)),
                "by_source_final": dict(Counter(i["source"] for i in kept)),
                "dropped": dropped_log,
            },
            indent=2,
        )
    )
    print(f"Wrote {FINAL_OUT}")
    print(f"Wrote {DEDUP_REPORT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        required=True,
        choices=["extract-ifeval", "extract-infobench", "merge-dedup", "all"],
    )
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    if args.step in ("extract-ifeval", "all"):
        print("=== Step 1: Extract IFEval + filter to 325 ===")
        extract_ifeval()

    if args.step in ("extract-infobench", "all"):
        print("\n=== Step 2: Extract InFoBench 500 via DeepSeek ===")
        extract_infobench()

    if args.step in ("merge-dedup", "all"):
        print("\n=== Step 3: Merge + dedup ===")
        merge_and_dedup(threshold=args.threshold)


if __name__ == "__main__":
    main()
