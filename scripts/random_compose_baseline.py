"""Generate a random-composition baseline dataset.

For each item: randomly pick a bare request, randomly pick skill_count skills
(respecting incompatibility), compose via the category-appropriate composer,
and save in the same JSONL format as edge_case_dataset.jsonl so that
collect_response_matrix.py / filter_random_compose.py can consume it directly.

Composer selection:
  - constrained_completion → TemplateComposer (mechanical constraint appending)
  - harmful_refusal / sycophancy_resistance / over_refusal → Composer (LLM)

Parallelization:
  Sampling decisions (bare request + skill combo) run sequentially against a
  seeded RNG so the dataset is reproducible. Only the LLM compose API calls
  are parallelized via ThreadPoolExecutor (default 20 workers).

Usage:
    python scripts/random_compose_baseline.py \
        --category harmful_refusal \
        --n 2000 --seed 42

    # Then run two-stage panel filter:
    python scripts/filter_random_compose.py \
        --dataset benchmarks/harmful_refusal/random_baseline_k2000.jsonl \
        --output-dir results/harmful_refusal/random_filter \
        --category harmful_refusal
"""

import argparse
import json
import random
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search.skill_selector import load_skills, is_compatible, Skill
from search.composer import Composer, TemplateComposer


# ---------------------------------------------------------------------------
# Composer factory
# ---------------------------------------------------------------------------

LLM_COMPOSER_CATEGORIES = {"harmful_refusal", "sycophancy_resistance", "over_refusal"}


def make_composer(category: str, seed: int | None = None):
    """Return the category-appropriate composer.

    - TemplateComposer: constrained_completion (mechanical, no LLM calls).
    - Composer (LLM):   harmful_refusal / sycophancy_resistance / over_refusal.
    """
    if category == "constrained_completion":
        return TemplateComposer(seed=seed)
    if category in LLM_COMPOSER_CATEGORIES:
        return Composer(category=category)
    raise ValueError(f"No composer defined for category '{category}'")


# ---------------------------------------------------------------------------
# Category configs: paths + defaults
# ---------------------------------------------------------------------------

CATEGORY_CONFIGS = {
    "constrained_completion": {
        "bare_requests": "data/constrained_completion/bare_requests.jsonl",
        "skill_dict": "skills/constrained_completion/skill_dictionary.json",
        "skill_count_range": (1, 3),
    },
    "harmful_refusal": {
        "bare_requests": "data/harmful_refusal/bare_requests.jsonl",
        "skill_dict": "skills/harmful_refusal/skill_dictionary.json",
        "skill_count_range": (0, 3),
    },
    "over_refusal": {
        "bare_requests": "data/over_refusal/bare_requests.jsonl",
        "skill_dict": "skills/over_refusal/skill_dictionary.json",
        "skill_count_range": (0, 3),
    },
    "sycophancy_resistance": {
        "bare_requests": "data/sycophancy_resistance/bare_requests.jsonl",
        "skill_dict": "skills/sycophancy_resistance/skill_dictionary.json",
        "follow_up_skill_dict": "skills/sycophancy_resistance/follow_up_skills.json",
        "max_follow_up_turns": 3,
        "skill_count_range": (0, 3),
    },
}


def load_bare_requests(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def build_skill_count_sequence(
    n: int,
    sc_min: int,
    sc_max: int,
    weights_str: str | None,
    rng: random.Random,
) -> list[int]:
    """Return a length-n list of skill counts, one per item.

    If `weights_str` is None, sample uniformly from [sc_min, sc_max] using
    `rng` (consumes n draws). Otherwise parse weights, normalize, convert to
    exact bucket sizes summing to n, build a flat sequence, and shuffle it
    with `rng` so the bucket order is not contiguous.
    """
    if weights_str is None:
        return [rng.randint(sc_min, sc_max) for _ in range(n)]

    raw_weights = [float(w.strip()) for w in weights_str.split(",") if w.strip()]
    if not raw_weights:
        raise ValueError("--skill-count-weights is empty")
    if len(raw_weights) != sc_max + 1:
        raise ValueError(
            f"--skill-count-weights expects {sc_max + 1} values (positions 0..{sc_max}), "
            f"got {len(raw_weights)}"
        )
    # Zero out counts below sc_min so they never appear
    for sc in range(sc_min):
        raw_weights[sc] = 0.0
    total_w = sum(raw_weights)
    if total_w <= 0:
        raise ValueError("--skill-count-weights: all weights are zero after sc_min cutoff")

    # Normalize and round to exact counts
    exact_counts = [round(n * w / total_w) for w in raw_weights]
    diff = n - sum(exact_counts)
    if diff != 0:
        # Fix rounding drift on the largest non-zero bucket
        largest_idx = max(
            (i for i, c in enumerate(exact_counts) if c > 0),
            key=lambda i: exact_counts[i],
        )
        exact_counts[largest_idx] += diff

    sequence: list[int] = []
    for sc, cnt in enumerate(exact_counts):
        sequence.extend([sc] * cnt)
    rng.shuffle(sequence)
    return sequence


def sample_compatible_skills(
    all_skills: list[Skill],
    count: int,
    rng: random.Random,
    max_retries: int = 5,
) -> list[Skill]:
    """Sample `count` mutually compatible skills uniformly at random.

    A single greedy shuffle-and-pick can fail when the first few picks have
    wide incompatibility tag sets, causing the selection to terminate with
    fewer than `count` skills even though a valid `count`-sized combo exists
    elsewhere in the pool. We retry up to `max_retries` times and return the
    best attempt (most skills found).

    Reproducibility: the main `rng` advances by exactly one randint call, so
    downstream sampling decisions are not perturbed by retry counts. All
    shuffles use a sub-rng seeded from that single draw.
    """
    if count == 0:
        return []

    sub_seed = rng.randint(0, 2**31 - 1)
    sub_rng = random.Random(sub_seed)

    best: list[Skill] = []
    for _ in range(max_retries):
        shuffled = list(all_skills)
        sub_rng.shuffle(shuffled)
        selected: list[Skill] = []
        for skill in shuffled:
            if len(selected) >= count:
                break
            if is_compatible(skill, selected):
                selected.append(skill)
        if len(selected) == count:
            return selected
        if len(selected) > len(best):
            best = selected
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Generate random-composition baseline dataset",
    )
    parser.add_argument("--category", required=True, choices=list(CATEGORY_CONFIGS))
    parser.add_argument("--n", type=int, default=500, help="Number of items to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skill-count-min", type=int, default=None)
    parser.add_argument("--skill-count-max", type=int, default=None)
    parser.add_argument(
        "--skill-count-weights", type=str, default=None,
        help="Comma-separated weights for each skill count, starting at 0. "
             "e.g. '1,6,6,6' for counts [0,1,2,3] suppresses count=0 to ~5%% of N "
             "while keeping 1/2/3 roughly balanced. Weights are normalized and "
             "converted to exact bucket sizes that sum to --n. "
             "Default: uniform over [skill_count_min, skill_count_max].",
    )
    parser.add_argument(
        "--max-follow-up-turns", type=int, default=None,
        help="Upper bound on follow-up turns for multi-turn categories "
             "(e.g. sycophancy_resistance). Each item's turn count is sampled "
             "uniformly from [1, min(this, pool_size)]. Default: category config.",
    )
    parser.add_argument("--output", default=None,
                        help="Output path (default: benchmarks/{category}/random_baseline_k{n}.jsonl)")
    parser.add_argument("--workers", type=int, default=20,
                        help="Concurrent LLM compose workers (ignored by TemplateComposer)")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    cfg = CATEGORY_CONFIGS[args.category]
    sc_min = args.skill_count_min or cfg["skill_count_range"][0]
    sc_max = args.skill_count_max or cfg["skill_count_range"][1]
    output = args.output or f"benchmarks/{args.category}/random_baseline_k{args.n}.jsonl"

    rng = random.Random(args.seed)

    # Load data. Do NOT pass category= to load_skills: files live under
    # skills/{category}/ so the partition is already implicit, and a file's
    # `category` field often holds a SUB-category (e.g. "authority",
    # "challenge"). Passing category=args.category would silently drop every
    # skill whose sub-category label doesn't match the top-level name.
    bare_requests = load_bare_requests(cfg["bare_requests"])
    skill_dict = load_skills(cfg["skill_dict"])
    all_skills = list(skill_dict.values())
    composer = make_composer(args.category, seed=args.seed)

    if not all_skills:
        sys.exit(
            f"[fatal] Main skill pool is empty for category '{args.category}' "
            f"({cfg['skill_dict']}). Refusing to generate items."
        )

    # Optional follow-up skill pool for multi-turn categories
    fu_dict_path = cfg.get("follow_up_skill_dict")
    max_fu = (
        args.max_follow_up_turns
        if args.max_follow_up_turns is not None
        else cfg.get("max_follow_up_turns", 0)
    )
    fu_pool: list[Skill] = []
    if fu_dict_path and max_fu > 0:
        fu_pool = list(load_skills(fu_dict_path).values())
    fu_upper = min(max_fu, len(fu_pool)) if fu_pool else 0

    # Fail-fast: if the category declares a follow-up pool, follow-ups MUST be
    # active. Silently falling back to single-turn for a multi-turn category
    # would produce a benchmark incomparable to SGBS output.
    if fu_dict_path and fu_upper == 0:
        sys.exit(
            f"[fatal] Category '{args.category}' declares follow_up_skill_dict "
            f"but follow-ups are disabled (pool={len(fu_pool)}, max_fu={max_fu}). "
            f"Check {fu_dict_path} and --max-follow-up-turns."
        )

    print(f"Category:       {args.category}")
    print(f"Composer:       {type(composer).__name__}")
    print(f"Bare requests:  {len(bare_requests)}")
    print(f"Skills:         {len(all_skills)}")
    if args.skill_count_weights:
        print(f"Skill count:    Weighted [{sc_min}..{sc_max}] weights={args.skill_count_weights}")
    else:
        print(f"Skill count:    Uniform [{sc_min}, {sc_max}]  (0 = pure rephrase baseline)")
    if fu_upper > 0:
        print(f"Follow-ups:     {len(fu_pool)} skills, turns uniform in [1, {fu_upper}]")
    else:
        print(f"Follow-ups:     disabled")
    print(f"Items to gen:   {args.n}")
    print(f"Workers:        {args.workers}")
    print(f"Output:         {output}", flush=True)

    # ------------------------------------------------------------------
    # Step 1: sampling decisions (sequential, seeded RNG for reproducibility)
    # ------------------------------------------------------------------
    count_sequence = build_skill_count_sequence(
        n=args.n,
        sc_min=sc_min,
        sc_max=sc_max,
        weights_str=args.skill_count_weights,
        rng=rng,
    )
    planned_dist = Counter(count_sequence)
    print("Planned skill count distribution:")
    for k in sorted(planned_dist):
        print(f"  count={k}: {planned_dist[k]}")

    decisions: list[tuple[int, dict, list[Skill], list[Skill]]] = []
    shortfall_count = 0
    fu_turn_counts: list[int] = []
    for i in range(args.n):
        br = rng.choice(bare_requests)
        requested = count_sequence[i]
        selected = sample_compatible_skills(all_skills, requested, rng)
        if len(selected) < requested:
            shortfall_count += 1
        fu_skills: list[Skill] = []
        if fu_upper > 0:
            n_fu = rng.randint(1, fu_upper)
            fu_skills = rng.sample(fu_pool, n_fu)
            fu_turn_counts.append(n_fu)
        decisions.append((i, br, selected, fu_skills))

    if shortfall_count > 0:
        print(f"  [note] {shortfall_count}/{args.n} items got fewer skills than "
              f"requested after retries (compatibility constraints)", flush=True)

    if fu_turn_counts:
        fu_dist = Counter(fu_turn_counts)
        print("Planned follow-up turn distribution:")
        for k in sorted(fu_dist):
            print(f"  turns={k}: {fu_dist[k]}")

    # ------------------------------------------------------------------
    # Step 2: composition (parallel for LLM composer, sequential-equivalent
    # for TemplateComposer — ThreadPoolExecutor still works correctly for both)
    # ------------------------------------------------------------------
    items: list[dict | None] = [None] * args.n
    done_lock = threading.Lock()
    done_count = [0]
    fail_count = [0]

    def worker(decision: tuple[int, dict, list[Skill], list[Skill]]) -> tuple[int, dict | None]:
        i, br, selected, fu_skills = decision
        compose_kwargs = {"bare_request": br["prompt"], "skills": selected}
        if fu_skills:
            compose_kwargs["follow_up_skills"] = fu_skills
        try:
            prompt, compose_meta = composer.compose(**compose_kwargs)
        except Exception as e:
            print(f"[error] item {i}: {type(e).__name__}: {e}", flush=True)
            return i, None
        merged_meta = {**br.get("metadata", {}), **compose_meta}
        return i, {
            "prompt_id": f"rand_{i+1:04d}",
            "prompt": prompt,
            "bare_request": br["prompt"],
            "skills": [s.name for s in selected],
            "skill_count": len(selected),
            "follow_up_skills": [s.name for s in fu_skills],
            "source_model": "random",
            "p_hat": None,
            "pass_count": None,
            "total_trials": None,
            "round_found": None,
            "metadata": merged_meta,
            "category": args.category,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, d) for d in decisions]
        for fut in as_completed(futures):
            i, item = fut.result()
            items[i] = item
            with done_lock:
                done_count[0] += 1
                if item is None:
                    fail_count[0] += 1
                k = done_count[0]
                n_fail = fail_count[0]
            if k % args.progress_every == 0 or k == args.n:
                print(f"  [{k}/{args.n}] composed  (fails so far: {n_fail})", flush=True)

    # Drop failed items before writing
    successful_items = [it for it in items if it is not None]
    n_success = len(successful_items)
    n_fail = args.n - n_success

    # ------------------------------------------------------------------
    # Duplicate check
    #
    # TemplateComposer (constrained_completion) is deterministic in
    # (bare_request, skill_set) → prompt; LLM Composer (HR/OR/SR) randomizes
    # via rephrasing so prompt-level collisions are near-zero even when the
    # underlying tuple repeats. We report both granularities so the warning
    # threshold reflects which composer is in use.
    # ------------------------------------------------------------------
    prompts = [it["prompt"] for it in successful_items]
    n_unique_prompt = len(set(prompts))
    tuple_keys = [
        (it["bare_request"],
         tuple(sorted(it.get("skills") or [])),
         tuple(sorted(it.get("follow_up_skills") or [])))
        for it in successful_items
    ]
    n_unique_tuple = len(set(tuple_keys))
    n_dup_prompt = n_success - n_unique_prompt
    n_dup_tuple = n_success - n_unique_tuple
    print(f"\nDuplicate check:")
    print(f"  unique prompts:            {n_unique_prompt}/{n_success} "
          f"({100*n_unique_prompt/n_success:.2f}% unique, {n_dup_prompt} dup)")
    print(f"  unique (bare, skill_set):  {n_unique_tuple}/{n_success} "
          f"({100*n_unique_tuple/n_success:.2f}% unique, {n_dup_tuple} dup)")
    if n_success and n_dup_prompt / n_success > 0.01:
        print(f"  [warn] prompt-level duplicate rate > 1% — consider larger --n "
              f"or richer bare_request / skill pool")

    # ------------------------------------------------------------------
    # Step 3: save
    # ------------------------------------------------------------------
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for item in successful_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    sc_dist = Counter(it["skill_count"] for it in successful_items)
    print(f"\nGenerated {n_success} items  (failed: {n_fail})")
    print("Skill count distribution:")
    for k in sorted(sc_dist):
        print(f"  count={k}: {sc_dist[k]}")
    if fu_upper > 0:
        fu_realized = Counter(
            len(it.get("metadata", {}).get("follow_ups", []))
            for it in successful_items
        )
        print("Realized follow-up message count distribution:")
        for k in sorted(fu_realized):
            print(f"  follow_ups={k}: {fu_realized[k]}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
