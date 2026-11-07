"""Stratified downsample of fixed validation benchmarks.

Produces an anchor-set-matched subsample for Claim 3 (resolution comparison
of DBE anchor vs. fixed benchmarks at matched prompt budget). Output schema
is identical to `random_compose_baseline.py` so that
`collect_response_matrix.py` / `fit_irt.py` consume it without modification.

Usage:
    python scripts/sample_validation_benchmarks.py --category harmful_refusal
    python scripts/sample_validation_benchmarks.py --category over_refusal
    python scripts/sample_validation_benchmarks.py --category constrained_completion
    python scripts/sample_validation_benchmarks.py --category sycophancy_resistance

Default target sizes match the active anchor-set filter sizes (per filter
ablation). Use --filter-rule to switch between AVG and ALO:
  --filter-rule avg  (default for back-compat): A=300, A'=308, B=175, C=414
  --filter-rule alo:                            A=293, A'=299, B=183, C=416
Override either with --target-size.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Per-category stratification specs (anchor-matched targets per Claim 3)
# ---------------------------------------------------------------------------

_AVG_SPECS: dict[str, dict] = {
    "harmful_refusal": {
        # AVG anchor 306 → target 300 across 3 tiers, 100 each
        "default_target": 300,
        "tiers": [
            ("easy",   "data/harmful_refusal/validation/easy.jsonl",   100),
            ("medium", "data/harmful_refusal/validation/medium.jsonl", 100),
            ("hard",   "data/harmful_refusal/validation/hard.jsonl",   100),
        ],
        # Within easy: prefer JBB-Originals (55) then DoNotAnswer
        "easy_substratify": {
            "primary_source": "jailbreakbench",
            "primary_quota": 55,
        },
    },
    "over_refusal": {
        # AVG anchor 308 → target 308 from FalseReject single pool, stratified by category_id (46 cats)
        "default_target": 308,
        "tiers": [
            ("all", "data/over_refusal/validation/all.jsonl", 308),
        ],
        "stratify_key": ("metadata", "category_id"),
    },
    "constrained_completion": {
        # AVG anchor 175 → target 175: ifbench 25 + manyifeval 150 (15 per num_instructions 1..10)
        "default_target": 175,
        "tiers": [
            ("ifbench",    "data/constrained_completion/validation/ifbench.jsonl",     25),
            ("manyifeval", "data/constrained_completion/validation/manyifeval.jsonl", 150),
        ],
        # Within manyifeval: stratify by num_instructions (1..10), 15 each
        "manyifeval_stratify_key": "num_instructions",
        "manyifeval_per_bucket": 15,
    },
    "sycophancy_resistance": {
        # AVG anchor 414 → target 414 across 3 tiers, 138 each
        "default_target": 414,
        "tiers": [
            ("easy",   "data/sycophancy_resistance/validation/easy.jsonl",   138),
            ("medium", "data/sycophancy_resistance/validation/medium.jsonl", 138),
            ("hard",   "data/sycophancy_resistance/validation/hard.jsonl",   138),
        ],
    },
}

# ALO[0.20, 0.80] anchor sizes: A=293, A'=299, B=183, C=416. Tier quotas
# scaled from AVG (largest tier absorbs the residual; substratification
# proportions preserved).
_ALO_SPECS: dict[str, dict] = {
    "harmful_refusal": {
        # ALO anchor 293 → 98/98/97 (slight imbalance absorbs the −7 vs AVG)
        "default_target": 293,
        "tiers": [
            ("easy",   "data/harmful_refusal/validation/easy.jsonl",   98),
            ("medium", "data/harmful_refusal/validation/medium.jsonl", 98),
            ("hard",   "data/harmful_refusal/validation/hard.jsonl",   97),
        ],
        # Within easy: keep JBB-Originals primary share at 55 (43 DoNotAnswer residual)
        "easy_substratify": {
            "primary_source": "jailbreakbench",
            "primary_quota": 55,
        },
    },
    "over_refusal": {
        # ALO anchor 299 → 299 single tier
        "default_target": 299,
        "tiers": [
            ("all", "data/over_refusal/validation/all.jsonl", 299),
        ],
        "stratify_key": ("metadata", "category_id"),
    },
    "constrained_completion": {
        # ALO anchor 183 → 25 ifbench + 158 manyifeval (per-bucket varies 15/16
        # via _stratified_sample largest-remaining; +8 vs AVG absorbed by manyifeval)
        "default_target": 183,
        "tiers": [
            ("ifbench",    "data/constrained_completion/validation/ifbench.jsonl",     25),
            ("manyifeval", "data/constrained_completion/validation/manyifeval.jsonl", 158),
        ],
        "manyifeval_stratify_key": "num_instructions",
        "manyifeval_per_bucket": 16,  # 8 buckets at 16, 2 at 15 (largest-remaining)
    },
    "sycophancy_resistance": {
        # ALO anchor 416 → 139/139/138 (+2 vs AVG split into easy/medium)
        "default_target": 416,
        "tiers": [
            ("easy",   "data/sycophancy_resistance/validation/easy.jsonl",   139),
            ("medium", "data/sycophancy_resistance/validation/medium.jsonl", 139),
            ("hard",   "data/sycophancy_resistance/validation/hard.jsonl",   138),
        ],
    },
}

FILTER_SPECS = {"avg": _AVG_SPECS, "alo": _ALO_SPECS}
# Back-compat alias: existing callers expect CATEGORY_SAMPLE_SPECS.
# Default exposure = AVG, matching the prior --filter-rule=avg default.
CATEGORY_SAMPLE_SPECS = _AVG_SPECS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _stratified_sample(items: list[dict], k: int, key_fn, rng: random.Random) -> list[dict]:
    """Equal-quota stratified sample across distinct key values.

    Distributes k samples evenly across buckets defined by key_fn. Buckets
    smaller than their quota are exhausted; the residual is redistributed
    to the largest remaining buckets.
    """
    if k >= len(items):
        return list(items)

    buckets: dict = defaultdict(list)
    for it in items:
        buckets[key_fn(it)].append(it)

    n_buckets = len(buckets)
    base_quota = k // n_buckets
    quotas = {bk: base_quota for bk in buckets}
    leftover = k - base_quota * n_buckets
    # Round-robin distribute leftover to buckets in size order (largest first)
    for bk in sorted(buckets, key=lambda b: -len(buckets[b])):
        if leftover <= 0:
            break
        quotas[bk] += 1
        leftover -= 1

    sampled: list[dict] = []
    deficit = 0
    for bk, q in quotas.items():
        bucket = buckets[bk]
        rng.shuffle(bucket)
        take = min(q, len(bucket))
        sampled.extend(bucket[:take])
        if take < q:
            deficit += q - take

    # Backfill from remaining items if any bucket was exhausted below quota
    if deficit > 0:
        remaining = [it for bk, bucket in buckets.items()
                     for it in bucket[quotas[bk]:]]
        rng.shuffle(remaining)
        sampled.extend(remaining[:deficit])

    return sampled


def _random_sample(items: list[dict], k: int, rng: random.Random) -> list[dict]:
    if k >= len(items):
        return list(items)
    return rng.sample(items, k)


# ---------------------------------------------------------------------------
# Per-category schema converters: validation item → unified output schema
# ---------------------------------------------------------------------------

def _convert_a(item: dict, idx: int, tier: str) -> dict:
    """Harmful refusal: prompt is a string; preserve full source metadata."""
    return _build_entry(
        prompt_id=f"val_A_{tier}_{idx:04d}",
        prompt=item["prompt"],
        category="harmful_refusal",
        metadata={
            **{k: v for k, v in item.items() if k not in ("prompt", "id")},
            "tier": tier,
        },
    )


def _convert_aprime(item: dict, idx: int, tier: str) -> dict:
    """Over-refusal: prompt is a string; metadata already has source info."""
    md = dict(item.get("metadata", {}))
    md["tier"] = tier
    md["validation_id"] = item.get("id")
    return _build_entry(
        prompt_id=f"val_Aprime_{idx:04d}",
        prompt=item["prompt"],
        category="over_refusal",
        metadata=md,
    )


def _convert_b(item: dict, idx: int, tier: str) -> dict:
    """Constrained completion: build metadata.constraints from instruction_id_list+kwargs.

    The validation jsonl stores constraints at the top level as parallel arrays
    `instruction_id_list` and `kwargs`. ConstraintJudge expects them packed into
    `metadata.constraints` as a list of {type, kwargs} objects.
    """
    types = item.get("instruction_id_list") or []
    kwargs_list = item.get("kwargs") or [{} for _ in types]
    if len(kwargs_list) < len(types):
        kwargs_list = list(kwargs_list) + [{}] * (len(types) - len(kwargs_list))

    constraints = [
        {"type": t, "kwargs": (kw or {})}
        for t, kw in zip(types, kwargs_list)
    ]

    md = {
        "constraints": constraints,
        "source": item.get("source"),
        "source_id": item.get("source_id"),
        "source_version": item.get("source_version"),
        "num_instructions": item.get("num_instructions"),
        "tier": tier,
    }
    if "source_meta" in item:
        md["source_meta"] = item["source_meta"]

    return _build_entry(
        prompt_id=f"val_B_{tier}_{idx:04d}",
        prompt=item["prompt"],
        category="constrained_completion",
        metadata=md,
    )


def _convert_c(item: dict, idx: int, tier: str) -> dict:
    """Sycophancy resistance: convert list[dict] prompt → (prompt_str, follow_ups).

    Tier semantics:
      easy   — Sharma `answer` template 3, single-turn. metadata has correct_answer,
               incorrect_answer (renamed to wrong_answer for judge template).
      medium — Sharma `are_you_sure`, 2-turn. Drop the `ai` prefix message; treat
               the human question as turn 1 and metadata.pushback as the only follow-up.
               correct_answer derived from correct_letter; wrong_answer left empty
               (the user's pushback is generic, not a specific wrong claim).
      hard   — SYCON 5-turn false-presupposition. metadata.pushbacks is a 4-element list;
               correct_answer and wrong_answer are not provided (the construct is
               "did the model hold the correct factual position"); both default to
               empty string so the judge template formats without KeyError.
    """
    prompt_msgs = item.get("prompt", [])
    md_in = item.get("metadata", {})

    # Extract the initial human message
    initial_human = next(
        (m["content"] for m in prompt_msgs if m.get("type") == "human"),
        "",
    )

    follow_ups: list[str] = []
    if tier == "medium":
        if "pushback" in md_in:
            follow_ups = [md_in["pushback"]]
    elif tier == "hard":
        follow_ups = list(md_in.get("pushbacks", []))

    # Judge template requires correct_answer + wrong_answer keys to format
    correct_answer = (
        md_in.get("correct_answer")
        or md_in.get("correct_letter")
        or ""
    )
    wrong_answer = (
        md_in.get("wrong_answer")
        or md_in.get("incorrect_answer")
        or ""
    )

    md = {
        **md_in,
        "correct_answer": correct_answer,
        "wrong_answer": wrong_answer,
        "follow_ups": follow_ups,
        "tier": tier,
        "validation_id": item.get("id"),
    }
    # Keep incorrect_answer alias too for back-compat readers
    if "incorrect_answer" not in md and wrong_answer:
        md["incorrect_answer"] = wrong_answer

    return _build_entry(
        prompt_id=f"val_C_{tier}_{idx:04d}",
        prompt=initial_human,
        category="sycophancy_resistance",
        metadata=md,
    )


def _build_entry(prompt_id: str, prompt: str, category: str, metadata: dict) -> dict:
    """Common output schema, mirroring random_compose_baseline.py output."""
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "bare_request": None,
        "skills": [],
        "skill_count": 0,
        "follow_up_skills": [],
        "source_model": "validation",
        "p_hat": None,
        "pass_count": None,
        "total_trials": None,
        "round_found": None,
        "metadata": metadata,
        "category": category,
    }


# ---------------------------------------------------------------------------
# Per-category samplers
# ---------------------------------------------------------------------------

def _sample_a(rng: random.Random, target: int, project_root: Path) -> list[dict]:
    spec = CATEGORY_SAMPLE_SPECS["harmful_refusal"]
    out: list[dict] = []
    idx = 0
    for tier_name, rel_path, quota in spec["tiers"]:
        items = _load_jsonl(str(project_root / rel_path))

        if tier_name == "easy":
            sub = spec["easy_substratify"]
            primary = [it for it in items if it.get("source") == sub["primary_source"]]
            others  = [it for it in items if it.get("source") != sub["primary_source"]]
            n_primary = min(sub["primary_quota"], len(primary), quota)
            n_other = quota - n_primary
            sampled = _random_sample(primary, n_primary, rng) + _random_sample(others, n_other, rng)
        elif tier_name == "medium":
            # Stratify AIR-Bench by L2 category for diversity across hazard families
            def air_l2(it: dict):
                return (it.get("source_meta") or {}).get("l2_name", "_unknown")
            sampled = _stratified_sample(items, quota, air_l2, rng)
        else:  # hard: WildJailbreak — single split, random
            sampled = _random_sample(items, quota, rng)

        for it in sampled:
            out.append(_convert_a(it, idx, tier_name))
            idx += 1
    if len(out) != target:
        print(f"[note] A target {target}, produced {len(out)}", flush=True)
    return out


def _sample_aprime(rng: random.Random, target: int, project_root: Path) -> list[dict]:
    spec = CATEGORY_SAMPLE_SPECS["over_refusal"]
    rel_path = spec["tiers"][0][1]
    items = _load_jsonl(str(project_root / rel_path))

    def cat_id(it: dict):
        return (it.get("metadata") or {}).get("category_id", -1)

    sampled = _stratified_sample(items, target, cat_id, rng)
    return [_convert_aprime(it, i, "all") for i, it in enumerate(sampled)]


def _sample_b(rng: random.Random, target: int, project_root: Path) -> list[dict]:
    spec = CATEGORY_SAMPLE_SPECS["constrained_completion"]
    quota_sum = sum(q for _, _, q in spec["tiers"])
    if quota_sum != target:
        print(f"[note] B tier quotas sum to {quota_sum}, target {target}", flush=True)
    out: list[dict] = []
    idx = 0
    for tier_name, rel_path, quota in spec["tiers"]:
        items = _load_jsonl(str(project_root / rel_path))

        if tier_name == "manyifeval":
            # Stratify by num_instructions ∈ {1..10}, equal quota per bucket
            def by_ninst(it: dict):
                return it.get("num_instructions", 0)
            sampled = _stratified_sample(items, quota, by_ninst, rng)
        else:
            # ifbench: random subsample
            sampled = _random_sample(items, quota, rng)

        for it in sampled:
            out.append(_convert_b(it, idx, tier_name))
            idx += 1
    return out


def _sample_c(rng: random.Random, target: int, project_root: Path) -> list[dict]:
    spec = CATEGORY_SAMPLE_SPECS["sycophancy_resistance"]
    quota_sum = sum(q for _, _, q in spec["tiers"])
    if quota_sum != target:
        print(f"[note] C tier quotas sum to {quota_sum}, target {target}", flush=True)
    out: list[dict] = []
    idx = 0
    for tier_name, rel_path, quota in spec["tiers"]:
        items = _load_jsonl(str(project_root / rel_path))
        sampled = _random_sample(items, quota, rng)
        for it in sampled:
            out.append(_convert_c(it, idx, tier_name))
            idx += 1
    return out


SAMPLERS = {
    "harmful_refusal": _sample_a,
    "over_refusal": _sample_aprime,
    "constrained_completion": _sample_b,
    "sycophancy_resistance": _sample_c,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global CATEGORY_SAMPLE_SPECS
    parser = argparse.ArgumentParser(
        description="Stratified downsample of fixed validation benchmarks "
                    "to the schema consumed by collect_response_matrix.py.",
    )
    parser.add_argument("--category", required=True,
                        choices=list(_AVG_SPECS))
    parser.add_argument("--filter-rule", choices=list(FILTER_SPECS), default="avg",
                        help="Anchor-set filter to size-match: avg (default, back-compat) or alo.")
    parser.add_argument("--target-size", type=int, default=None,
                        help="Override default anchor-matched target.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Output path (default: benchmarks/{category}/validation_sample[_alo].jsonl)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    CATEGORY_SAMPLE_SPECS = FILTER_SPECS[args.filter_rule]
    spec = CATEGORY_SAMPLE_SPECS[args.category]
    target = args.target_size if args.target_size is not None else spec["default_target"]
    suffix = "" if args.filter_rule == "avg" else f"_{args.filter_rule}"
    output = args.output or f"benchmarks/{args.category}/validation_sample{suffix}.jsonl"
    output_path = project_root / output

    rng = random.Random(args.seed)
    sampler = SAMPLERS[args.category]

    print(f"Category:        {args.category}")
    print(f"Filter rule:     {args.filter_rule}")
    print(f"Target size:     {target} (anchor-matched)")
    print(f"Seed:            {args.seed}")
    print(f"Output:          {output_path}", flush=True)

    items = sampler(rng, target, project_root)

    # Schema sanity
    required = {"prompt_id", "prompt", "metadata", "category"}
    for it in items:
        if not required.issubset(it):
            sys.exit(f"[fatal] missing required keys in item {it.get('prompt_id')}: "
                     f"{required - set(it)}")

    # Diagnostic: tier distribution
    tier_dist = Counter(it.get("metadata", {}).get("tier", "?") for it in items)
    print("\nTier distribution:")
    for t, c in sorted(tier_dist.items()):
        print(f"  {t}: {c}")

    # Category-specific diagnostic
    if args.category == "constrained_completion":
        n_constraints = Counter(len(it["metadata"].get("constraints", [])) for it in items)
        print("\nConstraints-per-item distribution:")
        for k, c in sorted(n_constraints.items()):
            print(f"  {k}: {c}")
    elif args.category == "sycophancy_resistance":
        fu_dist = Counter(len(it["metadata"].get("follow_ups", [])) for it in items)
        print("\nFollow-up count distribution:")
        for k, c in sorted(fu_dist.items()):
            print(f"  {k} follow-ups: {c}")
    elif args.category == "over_refusal":
        cat_dist = Counter(it["metadata"].get("category_id", -1) for it in items)
        print(f"\nFalseReject category coverage: {len(cat_dist)} of 46 categories represented")

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(items)} items → {output_path}")


if __name__ == "__main__":
    main()
