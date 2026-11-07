"""Build skill dictionary and bare requests for constrained_completion category.

Step 1: Extract 25 constraint types from IFEval → skill_dictionary.json
Step 2: Use LLM to extract base tasks from IFEval prompts → raw bare requests
Step 3: Embedding-based deduplication → bare_requests.jsonl

Usage:
    # Step 1 only (no API cost):
    python scripts/build_constrained_completion.py --step skills

    # Step 2 (requires DeepSeek API):
    python scripts/build_constrained_completion.py --step bare-requests

    # Step 3 (requires sentence-transformers):
    python scripts/build_constrained_completion.py --step deduplicate

    # All steps:
    python scripts/build_constrained_completion.py --step all
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ---------------------------------------------------------------------------
# Constraint type descriptions (manually curated from IFEval paper)
# ---------------------------------------------------------------------------

CONSTRAINT_DESCRIPTIONS = {
    "keywords:existence": "The response must contain ALL of the specified keywords.",
    "keywords:frequency": "A specific keyword must appear at least/at most N times.",
    "keywords:forbidden_words": "The response must NOT contain any of the specified forbidden words.",
    "keywords:letter_frequency": "A specific letter or character must appear at least/at most N times.",
    "language:response_language": "The entire response must be written in the specified language.",
    "length_constraints:number_words": "The response must contain at least/at most N words.",
    "length_constraints:number_sentences": "The response must contain at least/at most N sentences.",
    "length_constraints:number_paragraphs": "The response must contain exactly N paragraphs, separated by blank lines.",
    "length_constraints:nth_paragraph_first_word": "The Nth paragraph must start with a specific word.",
    "detectable_format:number_bullet_lists": "The response must contain at least N bullet point lists.",
    "detectable_format:number_highlighted_sections": "The response must contain at least N highlighted sections (using *...*) .",
    "detectable_format:title": "The response must include a title wrapped in <<...>>.",
    "detectable_format:json_format": "The entire response must be valid JSON.",
    "detectable_format:multiple_sections": "The response must be divided into N sections with specified section headers.",
    "detectable_format:constrained_response": "The response must be one of the allowed predefined choices.",
    "detectable_content:number_placeholders": "The response must contain at least N placeholder markers like [...].",
    "detectable_content:postscript": "The response must end with a postscript starting with the specified marker.",
    "change_case:english_lowercase": "The entire response must be in lowercase.",
    "change_case:english_capital": "The entire response must be in UPPERCASE.",
    "change_case:capital_word_frequency": "The response must contain at least/less than N fully capitalized words.",
    "startend:end_checker": "The response must end with the exact specified phrase.",
    "startend:quotation": "The response must be wrapped in double quotation marks.",
    "combination:two_responses": "Give two different responses, separated by 6 asterisks ******.",
    "combination:repeat_prompt": "The response must first repeat the original prompt before answering.",
    "punctuation:no_comma": "The response must not contain any commas.",
}

# Incompatible pairs: constraints that cannot coexist
INCOMPATIBLE_PAIRS = [
    ("change_case:english_lowercase", "change_case:english_capital"),
    ("change_case:english_lowercase", "change_case:capital_word_frequency"),
    ("detectable_format:json_format", "detectable_format:number_bullet_lists"),
    ("detectable_format:json_format", "detectable_format:number_highlighted_sections"),
    ("detectable_format:json_format", "detectable_format:title"),
    ("detectable_format:json_format", "detectable_format:multiple_sections"),
    ("detectable_format:constrained_response", "length_constraints:number_words"),
    ("detectable_format:constrained_response", "length_constraints:number_paragraphs"),
    ("detectable_format:constrained_response", "detectable_format:number_bullet_lists"),
    ("combination:repeat_prompt", "change_case:english_lowercase"),
    ("combination:repeat_prompt", "change_case:english_capital"),
]


# ---------------------------------------------------------------------------
# Step 1: Build skill dictionary
# ---------------------------------------------------------------------------

def build_skill_dictionary(output_dir: Path) -> None:
    """Extract constraint types from IFEval and save as skill dictionary."""
    from datasets import load_dataset
    ds = load_dataset("google/IFEval", split="train")

    # Collect parameter schemas per constraint type
    type_params = defaultdict(list)
    for row in ds:
        for iid, kw in zip(row["instruction_id_list"], row["kwargs"]):
            non_null = {k: v for k, v in kw.items() if v is not None and v != "" and v != []}
            type_params[iid].append(non_null)

    # Build skill dictionary
    skills = []
    # Build incompatibility lookup
    incompat_map = defaultdict(set)
    for a, b in INCOMPATIBLE_PAIRS:
        incompat_map[a].add(b)
        incompat_map[b].add(a)

    for constraint_type in sorted(CONSTRAINT_DESCRIPTIONS.keys()):
        description = CONSTRAINT_DESCRIPTIONS[constraint_type]
        all_params = type_params.get(constraint_type, [])
        count = len(all_params)

        # Extract parameter keys
        param_keys = set()
        for ex in all_params:
            param_keys.update(ex.keys())

        # Build param_pool: for each key, collect all unique values seen
        param_pool = {}
        for key in sorted(param_keys):
            values = []
            for ex in all_params:
                v = ex.get(key)
                if v is not None and v not in values:
                    values.append(v)
            param_pool[key] = values

        # Tags: include both category prefix and full name
        # Full name in tags enables incompatibility matching via is_compatible()
        category_tag = constraint_type.split(":")[0]
        tags = [category_tag, constraint_type]

        # Incompatible tags
        incompatible_tags = list(incompat_map.get(constraint_type, set()))

        skills.append({
            "name": constraint_type,
            "description": description,
            "tags": tags,
            "category": "constrained_completion",
            "incompatible_tags": incompatible_tags,
            "param_keys": sorted(param_keys),
            "param_pool": param_pool,
            "count_in_ifeval": count,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "skill_dictionary.json"
    with open(out_path, "w") as f:
        json.dump(skills, f, indent=2, ensure_ascii=False)

    print(f"Skill dictionary: {len(skills)} constraint types → {out_path}")


# ---------------------------------------------------------------------------
# Step 2: Extract bare requests via LLM
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """You are given an instruction-following prompt that contains a base task mixed with formatting/structural constraints.

Your job: extract ONLY the base task, removing all constraints about format, length, keywords, language, case, etc.

Examples:
Input: "Write a 300+ word summary of the wikipedia page about Raymond III. Do not use any commas and highlight at least 3 sections."
Output: "Write a summary of the wikipedia page about Raymond III."

Input: "Write a letter to a friend in all lowercase letters ask them to go and vote."
Output: "Write a letter to a friend asking them to go and vote."

Input: "Write a long email template that invites a group of participants to a meeting, with at least 500 words. The email must include the keywords 'correlated' and 'experiencing' and should not use any commas."
Output: "Write an email template that invites a group of participants to a meeting."

Now extract the base task from:
Input: "{prompt}"
Output:"""


def extract_bare_requests(output_dir: Path) -> None:
    """Use LLM to extract base tasks from IFEval prompts."""
    from datasets import load_dataset
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_TOKEN")
    if not api_key:
        raise ValueError("DEEPSEEK_API_TOKEN not set")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    ds = load_dataset("google/IFEval", split="train")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "bare_requests_raw.jsonl"

    # Resume support: skip already processed
    done_keys = set()
    if raw_path.exists():
        with open(raw_path) as f:
            for line in f:
                if line.strip():
                    done_keys.add(json.loads(line)["key"])
        print(f"Resuming: {len(done_keys)} already processed")

    with open(raw_path, "a") as f:
        for i, row in enumerate(ds):
            if row["key"] in done_keys:
                continue

            prompt = row["prompt"]
            messages = [
                {"role": "user", "content": EXTRACT_PROMPT.format(prompt=prompt)},
            ]

            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                        temperature=0.0,
                        max_tokens=512,
                    )
                    base_task = response.choices[0].message.content.strip()
                    # Strip quotes if LLM wrapped it
                    if base_task.startswith('"') and base_task.endswith('"'):
                        base_task = base_task[1:-1]
                    break
                except Exception as e:
                    print(f"  API error (attempt {attempt+1}): {e}")
                    time.sleep(2)
                    base_task = prompt  # fallback to original

            entry = {
                "key": row["key"],
                "original_prompt": prompt,
                "base_task": base_task,
                "instruction_ids": row["instruction_id_list"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(ds)}] processed")

    print(f"Raw bare requests: {len(ds)} → {raw_path}")


# ---------------------------------------------------------------------------
# Step 3: Deduplicate via embedding similarity
# ---------------------------------------------------------------------------

def deduplicate_bare_requests(
    output_dir: Path,
    threshold: float = 0.85,
) -> None:
    """Deduplicate bare requests using embedding similarity."""
    raw_path = output_dir / "bare_requests_raw.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Run --step bare-requests first.")

    with open(raw_path) as f:
        raw = [json.loads(l) for l in f if l.strip()]

    texts = [r["base_task"] for r in raw]
    print(f"Computing embeddings for {len(texts)} bare requests...")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    except ImportError:
        print("sentence-transformers not installed, falling back to exact dedup only")
        # Exact string dedup
        seen = set()
        unique = []
        for r in raw:
            t = r["base_task"].strip().lower()
            if t not in seen:
                seen.add(t)
                unique.append(r)
        embeddings = None

    if embeddings is not None:
        # Greedy dedup: iterate, skip if too similar to any kept
        sim_matrix = embeddings @ embeddings.T
        keep = []
        kept_indices = []
        for i in range(len(raw)):
            if any(sim_matrix[i, j] > threshold for j in kept_indices):
                continue
            keep.append(raw[i])
            kept_indices.append(i)
        unique = keep

    # Save
    out_path = output_dir / "bare_requests.jsonl"
    with open(out_path, "w") as f:
        for i, entry in enumerate(unique):
            out = {
                "prompt": entry["base_task"],
                "original_prompt": entry["original_prompt"],
                "key": entry["key"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Deduplicated: {len(raw)} → {len(unique)} bare requests (threshold={threshold})")
    print(f"Saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build constrained_completion skill dictionary and bare requests",
    )
    parser.add_argument(
        "--step", required=True,
        choices=["skills", "bare-requests", "deduplicate", "all"],
        help="Which step to run",
    )
    parser.add_argument(
        "--skills-dir", default="skills/constrained_completion",
        help="Output dir for skill dictionary",
    )
    parser.add_argument(
        "--data-dir", default="data/constrained_completion",
        help="Output dir for bare requests",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Embedding similarity threshold for dedup",
    )
    args = parser.parse_args()

    skills_dir = Path(args.skills_dir)
    data_dir = Path(args.data_dir)

    if args.step in ("skills", "all"):
        print("Step 1: Building skill dictionary...")
        build_skill_dictionary(skills_dir)

    if args.step in ("bare-requests", "all"):
        print("\nStep 2: Extracting bare requests via LLM...")
        extract_bare_requests(data_dir)

    if args.step in ("deduplicate", "all"):
        print("\nStep 3: Deduplicating bare requests...")
        deduplicate_bare_requests(data_dir, threshold=args.threshold)


if __name__ == "__main__":
    main()
