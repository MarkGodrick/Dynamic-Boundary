"""Probe whether a closed-API target has CIF edge cases at small skill counts.

HiES W1 needs a closed-weight target to run the query-only DBE pipeline end
to end. The harmful_refusal Step-2 SGBS run found 0 edge cases in 64 rounds,
so before spending budget on a full SGBS run we measure the raw hit rate of
constrained_completion at fixed skill counts.

For each requested skill count k, sample `combos` random (bare request,
compatible k-skill subset) pairs, run `n_trials` trials per pair through the
same TemplateComposer + ConstraintJudge used in the real experiment, and
report the pass-rate distribution. A pair is "coarse-gate" if p_hat lands in
`coarse_range` and "edge" if it lands in `fine_range`.

Usage:
    python scripts/rebuttal/probe_cif_closed_api.py \\
        --config configs/constrained_completion/adaptive_closed_api.yaml \\
        --model openai/gpt-5.4-mini \\
        --skill-counts 2,3 --combos 8 --n-trials 15
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from judges import create_judge
from judges.constraint_judge import ConstraintJudge
from logger import setup_logging, get_logger
from models import create_target_model, unload_model
from search.composer import TemplateComposer
from search.skill_selector import Skill, is_compatible, load_skills

logger = get_logger(__name__)

def load_bare_requests(path: str, limit: int | None = None) -> list[dict]:
    """Read bare requests as {prompt, metadata} dicts."""
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            out.append({
                "prompt": entry.get("prompt", entry.get("request", "")),
                "metadata": entry.get("metadata", {}),
            })
    if limit:
        out = out[:limit]
    return out


def sample_compatible(
    skills: dict[str, Skill], count: int, rng: random.Random,
    max_attempts: int = 200,
) -> list[Skill] | None:
    """Draw `count` mutually compatible skills uniformly at random."""
    names = list(skills)
    for _ in range(max_attempts):
        selected: list[Skill] = []
        for name in rng.sample(names, len(names)):
            cand = skills[name]
            if is_compatible(cand, selected):
                selected.append(cand)
            if len(selected) == count:
                return selected
    return None


def run_one_combo(
    target_model, judge, composer: TemplateComposer,
    bare: dict, skill_set: list[Skill],
    n_trials: int, seeds: list[int], max_workers: int,
) -> dict:
    """Compose one prompt and run n_trials independent trials on it."""
    prompt, compose_meta = composer.compose(bare["prompt"], skill_set)
    metadata = {**bare["metadata"], **compose_meta}

    def _trial(seed: int) -> dict:
        response = target_model.generate(prompt, seed=seed)
        meta = {}
        get_meta = getattr(target_model, "get_last_meta", None)
        if callable(get_meta):
            meta = get_meta() or {}
        detail = judge.judge_detailed([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ], metadata)
        return {
            "seed": seed,
            "passed": bool(detail["passed"]),
            "per_constraint": detail["results"],
            "response_chars": len(response),
            "finish_reason": meta.get("finish_reason"),
            "completion_tokens": meta.get("completion_tokens"),
            "cost": meta.get("cost"),
            "response": response,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        trials = list(pool.map(_trial, seeds[:n_trials]))

    n_pass = sum(1 for t in trials if t["passed"])
    # Per-constraint failure counts localize which constraint the model misses.
    per_type_fail: dict[str, int] = {}
    for t in trials:
        for r in t["per_constraint"]:
            if not r["passed"]:
                per_type_fail[r["type"]] = per_type_fail.get(r["type"], 0) + 1

    return {
        "prompt": prompt,
        "bare_request": bare["prompt"],
        "skills": [s.name for s in skill_set],
        "constraints": compose_meta.get("constraints", []),
        "n_trials": len(trials),
        "pass_count": n_pass,
        "p_hat": n_pass / len(trials) if trials else 0.0,
        "per_type_fail": per_type_fail,
        "trials": trials,
    }
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sgbs-config",
                    default="configs/constrained_completion/search_constrained.yaml",
                    help="source of skills / bare_requests / composer settings")
    ap.add_argument("--skill-counts", default="2,3")
    ap.add_argument("--combos", type=int, default=8)
    ap.add_argument("--n-trials", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--output-dir", default="results/rebuttal/closed_models/probe_cif")
    args = ap.parse_args()

    setup_logging()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    with open(args.sgbs_config) as f:
        sgbs_cfg = yaml.safe_load(f) or {}

    coarse_range = tuple(sgbs_cfg["evaluator"]["coarse_range"])
    fine_range = tuple(sgbs_cfg["evaluator"]["fine_range"])
    skill_counts = [int(x) for x in args.skill_counts.split(",") if x.strip()]

    skills_cfg = sgbs_cfg.get("skills", {}) or {}
    skills = load_skills(
        skills_cfg["dictionary_path"],
        category=skills_cfg.get("category"),
    )
    br_cfg = sgbs_cfg.get("bare_requests", {}) or {}
    bare_requests = load_bare_requests(br_cfg["data_path"], br_cfg.get("limit"))
    logger.info("Loaded %d skills, %d bare requests", len(skills), len(bare_requests))

    judge = create_judge(sgbs_cfg.get("judge", {}), category="constrained_completion")
    if not isinstance(judge, ConstraintJudge):
        raise SystemExit(
            f"expected the rule-based ConstraintJudge, got {type(judge).__name__}"
        )
    composer = TemplateComposer(seed=args.seed)

    target_cfg = dict(cfg.get("target_model", {}) or {})
    target_cfg["name"] = args.model
    target_model = create_target_model(target_cfg)

    rng = random.Random(args.seed)
    trial_rng = random.Random(args.seed + 1)
    seeds = [trial_rng.randint(0, 2**31 - 1) for _ in range(args.n_trials)]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"{args.model.replace('/', '_')}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    try:
        for k in skill_counts:
            for i in range(args.combos):
                skill_set = sample_compatible(skills, k, rng)
                if skill_set is None:
                    logger.warning("No compatible %d-skill subset found; skipping", k)
                    continue
                bare = rng.choice(bare_requests)
                rec = run_one_combo(
                    target_model, judge, composer, bare, skill_set,
                    args.n_trials, seeds, args.max_workers,
                )
                rec["skill_count"] = k
                rec["combo_index"] = i
                rec["coarse_gate"] = coarse_range[0] <= rec["p_hat"] <= coarse_range[1]
                rec["edge_at_fine"] = fine_range[0] <= rec["p_hat"] <= fine_range[1]
                records.append(rec)
                logger.info(
                    "k=%d combo=%d p_hat=%.3f (%d/%d) coarse=%s skills=%s",
                    k, i, rec["p_hat"], rec["pass_count"], rec["n_trials"],
                    rec["coarse_gate"], rec["skills"],
                )
                with open(out_dir / "probe_records.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
    finally:
        unload_model(target_model)

    summary = {
        "model": args.model,
        "n_trials": args.n_trials,
        "coarse_range": list(coarse_range),
        "fine_range": list(fine_range),
        "total_target_calls": sum(r["n_trials"] for r in records),
        "total_cost": sum(
            t.get("cost") or 0.0 for r in records for t in r["trials"]
        ),
        "by_skill_count": {},
    }
    for k in skill_counts:
        sub = [r for r in records if r["skill_count"] == k]
        if not sub:
            continue
        ps = sorted(r["p_hat"] for r in sub)
        summary["by_skill_count"][str(k)] = {
            "combos": len(sub),
            "mean_p_hat": sum(ps) / len(ps),
            "min_p_hat": ps[0],
            "median_p_hat": ps[len(ps) // 2],
            "max_p_hat": ps[-1],
            "coarse_gate_hits": sum(1 for r in sub if r["coarse_gate"]),
            "edge_at_fine_hits": sum(1 for r in sub if r["edge_at_fine"]),
        }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    logger.info("Wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
