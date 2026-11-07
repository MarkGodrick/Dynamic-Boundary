"""Annotate bare_requests.jsonl with a super_category per item (parallel).

Super-categories (8):
  Natural Sciences, Social Sciences, Arts, Business and Economics,
  Engineering and Technology, Lifestyle and Leisure, Office and Work, Platform

InFoBench → parse from source_meta.category (colon prefix, or "Platform"), no API.
IFEval    → DeepSeek classification via ThreadPoolExecutor (workers=20).

Writes in-place: data/constrained_completion/bare_requests.jsonl
  (adds `super_category` field at top level per item)
"""

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BARE = ROOT / "data" / "constrained_completion" / "bare_requests.jsonl"

SUPERCATS = [
    "Natural Sciences",
    "Social Sciences",
    "Arts",
    "Business and Economics",
    "Engineering and Technology",
    "Lifestyle and Leisure",
    "Office and Work",
    "Platform",
]

CLASSIFY_PROMPT = """Classify the following task description into exactly ONE of these 8 categories:

1. Natural Sciences — mathematics, physics, chemistry, biology, astronomy, medical/earth sciences
2. Social Sciences — history, psychology, education, linguistics, political science, law, sociology
3. Arts — literature, film, music, painting, drama, architecture
4. Business and Economics — marketing, finance, management, real estate, entrepreneurship
5. Engineering and Technology — computer science, software engineering, AI, electrical/mechanical engineering
6. Lifestyle and Leisure — family, travel, sports, hobbies, food/cooking, fashion, games
7. Office and Work — document formatting, job recruitment, event planning, calendar
8. Platform — tasks tied to a specific website/service (email, social media, e-commerce, search, etc.) rather than a subject matter

Task description:
{task}

Respond with ONLY the category name (one of the 8 above), nothing else.
"""


def infobench_to_supercat(raw_category: str) -> str:
    if ":" in raw_category:
        prefix = raw_category.split(":", 1)[0].strip()
        if prefix in SUPERCATS:
            return prefix
    return "Platform"


def classify_ifeval(task: str, client) -> str:
    messages = [{"role": "user", "content": CLASSIFY_PROMPT.format(task=task)}]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.0,
                max_tokens=32,
            )
            raw = resp.choices[0].message.content.strip().strip('"').strip("'")
            for c in SUPERCATS:
                if c.lower() in raw.lower():
                    return c
            return "Platform"  # fallback
        except Exception as e:
            print(f"  API error attempt {attempt+1}: {e}", file=sys.stderr, flush=True)
            time.sleep(2)
    return "Platform"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()

    with BARE.open() as f:
        items = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(items)} bare items", flush=True)

    # InFoBench: zero-cost mapping
    for it in items:
        if "super_category" in it:
            continue
        if it["source"] == "infobench":
            raw = (it.get("source_meta") or {}).get("category", "")
            it["super_category"] = infobench_to_supercat(raw)

    # IFEval: parallel DeepSeek classification
    ife_idx = [
        i for i, it in enumerate(items)
        if it["source"] == "ifeval" and "super_category" not in it
    ]
    print(f"IFEval items to classify via DeepSeek: {len(ife_idx)}", flush=True)

    if ife_idx:
        from openai import OpenAI
        api_key = os.environ.get("DEEPSEEK_API_TOKEN")
        if not api_key:
            raise ValueError("DEEPSEEK_API_TOKEN not set")
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        out_lock = threading.Lock()
        done = [0]

        def classify(idx: int) -> tuple[int, str]:
            cat = classify_ifeval(items[idx]["prompt"], client)
            return idx, cat

        print(f"Launching {args.workers} workers for {len(ife_idx)} items...", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(classify, idx): idx for idx in ife_idx}
            for fut in as_completed(futures):
                idx, cat = fut.result()
                items[idx]["super_category"] = cat
                with out_lock:
                    done[0] += 1
                    k = done[0]
                if k % 25 == 0 or k == len(ife_idx):
                    print(f"  [{k}/{len(ife_idx)}] classified", flush=True)
                if k % args.checkpoint_every == 0 or k == len(ife_idx):
                    with out_lock, BARE.open("w") as f:
                        for it in items:
                            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Final write (ensures InFoBench-only case also writes)
    with BARE.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Summary
    by_cat_source: dict[str, Counter] = {}
    for it in items:
        c = it.get("super_category", "MISSING")
        by_cat_source.setdefault(c, Counter())[it["source"]] += 1

    print("\n--- super_category × source ---", flush=True)
    totals = {c: sum(by_cat_source[c].values()) for c in by_cat_source}
    for c in sorted(totals, key=lambda x: -totals[x]):
        ife = by_cat_source[c].get("ifeval", 0)
        ifb = by_cat_source[c].get("infobench", 0)
        print(f"  {c:30s} total={totals[c]:4d}  (IFEval={ife}, InFoBench={ifb})", flush=True)


if __name__ == "__main__":
    main()
