"""Analyze jailbreak skill dictionary for contradictory skill pairs.

Approach:
  1. Embed all skill definitions with sentence-transformer
  2. Extract functional keywords from definitions
  3. Check known antonym patterns
  4. Cluster skills by theme, look for intra-cluster contradictions
  5. Find skills whose embeddings point in "opposite directions" relative to cluster center

Usage:
    python scripts/analyze_skill_contradictions.py
"""

import json
import re
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer


def load_skills(path="skills/harmful_refusal/jailbreak_dictionary_closed.json"):
    with open(path) as f:
        data = json.load(f)
    return {s["skill_name"]: s["definition"] for s in data}


# ---------------------------------------------------------------------------
# 1. Keyword-based antonym detection
# ---------------------------------------------------------------------------

ANTONYM_PAIRS = [
    # directness
    ({"indirect", "disguise", "mask", "camouflage", "pretext", "reframe", "fictional",
      "narrative", "euphemistic", "hypothetical", "benign", "academic", "scholarly",
      "roleplay", "story", "creative"},
     {"direct", "explicit", "blunt", "unambiguous", "overt", "straightforward"}),
    # tone
    ({"formal", "academic", "authoritative", "professional", "institutional",
      "commanding", "imperative", "coercive", "threatening", "aggressive"},
     {"casual", "informal", "playful", "polite", "deferential", "humor", "irony",
      "friendly", "conversational"}),
    # length
    ({"brief", "short", "concise", "minimal", "terse", "compact", "succinct",
      "low-context", "omit"},
     {"verbose", "lengthy", "extended", "exhaustive", "detailed", "comprehensive",
      "overload", "amplif", "elaborate", "procedural"}),
    # output structure
    ({"single", "one channel", "lock", "restrict", "confine", "rigid",
      "one output", "enforc"},
     {"dual", "two", "parallel", "bifurcat", "split", "multiple channel",
      "both.*response"}),
    # reasoning
    ({"suppress.*reason", "hide.*reason", "conceal.*reason", "suppress.*logic",
      "suppress.*explain", "withhold.*reasoning", "suppress.*disclosure",
      "meta.*suppress"},
     {"stepwise", "step-by-step", "chain.*thought", "elicit.*reason",
      "reasoning.*chain", "elaborate.*step", "multi-step reasoning"}),
    # obfuscation vs clarity
    ({"obfuscat", "fragment", "encrypt", "encoded", "cipher", "scrambl",
      "dispers", "noise", "sharding"},
     {"clarity", "specific", "precise", "explicit detail", "clear",
      "completeness", "actionable detail"}),
    # single vs multi persona
    ({"single persona", "one persona", "identity detach", "self-concept override"},
     {"dual persona", "two persona", "multiple persona", "bifurcat.*persona",
      "split.*persona", "two character"}),
    # compliance vs refusal manipulation
    ({"suppress refusal", "ban refusal", "blacklist refusal",
      "suppress.*safety", "suppress.*disclaimer", "suppress.*warning",
      "block refusal"},
     {"trigger refusal", "induce refusal", "elicit refusal",
      "refusal reversal", "leverage refusal"}),
    # fragmented vs unified
    ({"fragment", "segment", "shard", "split.*input", "chunk", "piece",
      "partial", "incremental"},
     {"unified", "bundl", "batch", "aggregate", "consolidat", "end-to-end",
      "comprehensive.*single"}),
    # repetition vs novelty
    ({"repetition", "repeat", "persist", "reinforce.*same", "reiterat"},
     {"variation", "rotat", "identity variation", "diversif", "alternate"}),
    # code/technical vs natural language
    ({"code", "program", "script", "function", "executable", "syntax",
      "json", "schema", "data structure"},
     {"natural language", "prose", "narrative", "story", "essay", "free text",
      "conversational"}),
]

ANTONYM_LABELS = [
    "request_directness",
    "tone",
    "length_complexity",
    "output_channels",
    "reasoning_visibility",
    "obfuscation_vs_clarity",
    "persona_count",
    "refusal_handling",
    "input_granularity",
    "repetition_vs_variation",
    "modality_code_vs_text",
]


def keyword_match(text: str, keywords: set[str]) -> list[str]:
    """Check which keywords appear in text (case-insensitive, supports regex)."""
    text_lower = text.lower()
    matches = []
    for kw in keywords:
        if ".*" in kw or "(" in kw:
            if re.search(kw, text_lower):
                matches.append(kw)
        elif kw in text_lower:
            matches.append(kw)
    return matches


def find_antonym_matches(skills: dict[str, str]):
    """For each skill, check which antonym groups it matches."""
    results = defaultdict(lambda: defaultdict(list))  # dim -> side -> [skill_names]

    for name, defn in skills.items():
        for idx, (side_a_kw, side_b_kw) in enumerate(ANTONYM_PAIRS):
            label = ANTONYM_LABELS[idx]
            matches_a = keyword_match(defn, side_a_kw)
            matches_b = keyword_match(defn, side_b_kw)
            # Also check skill name
            matches_a += keyword_match(name, side_a_kw)
            matches_b += keyword_match(name, side_b_kw)

            if matches_a and not matches_b:
                results[label]["side_a"].append(name)
            elif matches_b and not matches_a:
                results[label]["side_b"].append(name)
            elif matches_a and matches_b:
                # Both sides matched — flag for manual review
                results[label]["ambiguous"].append(name)

    return results


# ---------------------------------------------------------------------------
# 2. Embedding-based analysis
# ---------------------------------------------------------------------------

def find_embedding_opposites(skills: dict[str, str], model_name="all-MiniLM-L6-v2"):
    """Find skills that are semantically opposite within functional clusters."""
    names = list(skills.keys())
    texts = [f"{n}: {skills[n]}" for n in names]

    print("Computing embeddings...")
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    # Cosine similarity matrix
    sim = embs @ embs.T

    # Find most DISSIMILAR pairs (lowest cosine similarity)
    # These might represent contradictions
    print("\n=== Most dissimilar skill pairs (potential new contradictions) ===")
    # Get lower triangle indices
    n = len(names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((sim[i, j], names[i], names[j]))

    pairs.sort()  # ascending similarity

    # Print bottom 30 (most dissimilar)
    seen_printed = 0
    for sim_val, a, b in pairs[:50]:
        # Skip pairs we already know about
        print(f"  sim={sim_val:.3f}  {a}  <->  {b}")
        seen_printed += 1
        if seen_printed >= 30:
            break

    # Also: cluster and find intra-cluster outlier pairs
    print("\n=== Clustering for intra-cluster contradictions ===")
    from sklearn.cluster import KMeans

    n_clusters = 25
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embs)

    cluster_groups = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_groups[lab].append(i)

    # For each cluster, find the pair with lowest internal similarity
    intra_contradictions = []
    for cid, members in cluster_groups.items():
        if len(members) < 3:
            continue
        min_sim = 1.0
        min_pair = None
        for i_idx in range(len(members)):
            for j_idx in range(i_idx + 1, len(members)):
                ii, jj = members[i_idx], members[j_idx]
                s = sim[ii, jj]
                if s < min_sim:
                    min_sim = s
                    min_pair = (names[ii], names[jj])
        if min_pair and min_sim < 0.3:
            intra_contradictions.append((cid, min_sim, min_pair, len(members)))

    intra_contradictions.sort(key=lambda x: x[1])
    for cid, s, (a, b), size in intra_contradictions[:15]:
        print(f"  Cluster {cid} (size={size}): sim={s:.3f}  {a}  <->  {b}")
        da = skills[a].split('.')[0]
        db = skills[b].split('.')[0]
        print(f"    A: {da}")
        print(f"    B: {db}")
        print()


def main():
    skills = load_skills()
    print(f"Loaded {len(skills)} skills\n")

    # Phase 1: Keyword antonym detection
    print("=" * 60)
    print("PHASE 1: Keyword-based antonym detection")
    print("=" * 60)
    results = find_antonym_matches(skills)

    total_tagged = set()
    for dim in ANTONYM_LABELS:
        if dim not in results:
            print(f"\n{dim}: no matches")
            continue
        r = results[dim]
        a_list = r.get("side_a", [])
        b_list = r.get("side_b", [])
        amb = r.get("ambiguous", [])
        total_tagged.update(a_list)
        total_tagged.update(b_list)

        if not a_list and not b_list:
            continue

        print(f"\n{dim}:")
        print(f"  Side A ({len(a_list)}):")
        for s in sorted(a_list):
            print(f"    - {s}")
        print(f"  Side B ({len(b_list)}):")
        for s in sorted(b_list):
            print(f"    - {s}")
        if amb:
            print(f"  Ambiguous ({len(amb)}):")
            for s in sorted(amb):
                print(f"    - {s}")
        print(f"  Contradiction pairs: {len(a_list) * len(b_list)}")

    print(f"\nTotal skills with dimension tags: {len(total_tagged)} / {len(skills)}")

    # Phase 2: Embedding-based
    print()
    print("=" * 60)
    print("PHASE 2: Embedding-based contradiction detection")
    print("=" * 60)
    find_embedding_opposites(skills)


if __name__ == "__main__":
    main()
