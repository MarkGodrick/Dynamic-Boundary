"""Consolidate per-model SGBS edge cases into a single dataset.

Reads each experiment's edge_cases.jsonl + metadata.json, assigns unique IDs,
and writes a flat JSONL to <output-dir>/edge_case_dataset.jsonl. Cross-checks
each run's metadata.category against --category to avoid silent mismatches,
and refuses to merge multiple runs with the same target_model.

Usage:
    python scripts/data_prep/build_edge_case_dataset.py \
        --results-dir results/over_refusal/edge_case \
        --output-dir benchmarks/over_refusal \
        --category over_refusal
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path


def load_edge_cases_from_experiment(exp_dir: Path) -> tuple[list[dict], str | None]:
    """Load edge cases from one experiment dir.

    Returns (edge_cases, exp_category). exp_category is the run's recorded
    category (or None if absent) so the caller can cross-check it against
    --category.
    """
    ec_path = exp_dir / "edge_cases.jsonl"
    meta_path = exp_dir / "metadata.json"

    if not ec_path.exists() or not meta_path.exists():
        return [], None

    with open(meta_path) as f:
        meta = json.load(f)
    source_model = meta.get("target_model", "unknown")
    exp_category = meta.get("category")

    edge_cases = []
    with open(ec_path) as f:
        for line in f:
            if not line.strip():
                continue
            ec = json.loads(line)
            ec["source_model"] = source_model
            edge_cases.append(ec)

    return edge_cases, exp_category


def _reconstruct_constraints(prompt: str, skill_names: list[str],
                              skill_dict: dict) -> list[dict]:
    """Reconstruct constraint metadata by brute-force matching param_pool
    combinations against constraint descriptions found in the prompt text.

    Only called when metadata is missing from older runs.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from search.templates.constrained_completion import generate_constraint_description

    constraints = []
    for skill_name in skill_names:
        skill_entry = skill_dict.get(skill_name)
        if skill_entry is None:
            constraints.append({"type": skill_name, "kwargs": {}, "description": ""})
            continue

        pool = skill_entry.get("param_pool", {})
        if not pool:
            kwargs = {}
            desc = generate_constraint_description(skill_name, kwargs)
            constraints.append({"type": skill_name, "kwargs": kwargs, "description": desc})
            continue

        # Brute-force: try all combinations, find the one whose description
        # appears in the prompt.
        keys = list(pool.keys())
        values = [pool[k] for k in keys]

        matched_kwargs = {}
        matched_desc = ""
        for combo in product(*values):
            kwargs = dict(zip(keys, combo))
            desc = generate_constraint_description(skill_name, kwargs)
            # Also try without trailing period (inline template strips it)
            if desc in prompt or desc.rstrip(".") in prompt:
                matched_kwargs = kwargs
                matched_desc = desc
                break

        if not matched_desc:
            # Fallback: use default kwargs (first value of each param)
            matched_kwargs = {k: v[0] for k, v in pool.items() if v}
            matched_desc = generate_constraint_description(skill_name, matched_kwargs)

        constraints.append({
            "type": skill_name,
            "kwargs": matched_kwargs,
            "description": matched_desc,
        })

    return constraints


def _load_skill_dict(category: str) -> dict:
    """Load skill dictionary as {name: entry} lookup."""
    path = Path(__file__).resolve().parent.parent.parent / "skills" / category / "skill_dictionary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {s["name"]: s for s in raw}
    return raw


def build_dataset(results_dir: str, category: str,
                   slots: int | None = None) -> list[dict]:
    """Collect all edge cases across experiments, assign prompt IDs.

    Order is deterministic:
      - source_models iterated in alphabetical (string sort) order;
      - within each model, ecs sorted by (round_found, prompt) when
        round_found is present on every ec; otherwise the model's file
        order is preserved as a fallback (older runs may lack the field).

    If `slots` is not None, keep at most `slots` edge cases per source_model
    (truncated AFTER sorting, so the earliest-found rounds win).
    """
    results_path = Path(results_dir)
    by_model: dict[str, list[dict]] = {}
    seen_models: dict[str, Path] = {}

    for exp_dir in sorted(results_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        ecs, exp_category = load_edge_cases_from_experiment(exp_dir)
        if not ecs:
            continue

        # Cross-check category to avoid silent mismatch (cf. 2026-04-20
        # sycophancy incident): a run with a different category would otherwise
        # be quietly relabeled here and produce wrong judgments downstream.
        if exp_category is not None and exp_category != category:
            print(f"ERROR: {exp_dir.name} has category={exp_category!r} but "
                  f"--category={category!r}. Refusing to merge mismatched runs.")
            sys.exit(1)
        if exp_category is None:
            print(f"  [warn] {exp_dir.name} has no metadata.category; "
                  f"skipping cross-check")

        # Refuse to silently double-count: if the same target_model appears in
        # two run dirs (e.g. an aborted + retried run), make the user pick.
        src = ecs[0]["source_model"]
        if src in seen_models:
            print(f"ERROR: duplicate target_model {src!r} found in:\n"
                  f"  {seen_models[src]}\n  {exp_dir}\n"
                  f"Move/delete the stale run before re-building.")
            sys.exit(1)
        seen_models[src] = exp_dir
        by_model[src] = ecs

    all_edge_cases = []
    for src in sorted(by_model.keys()):
        ecs = by_model[src]

        # Per-model sort: (round_found, prompt) when round_found is present
        # everywhere; else keep file order as fallback.
        if all(ec.get("round_found") is not None for ec in ecs):
            ecs = sorted(
                ecs,
                key=lambda ec: (ec["round_found"], ec.get("prompt", "")),
            )
        else:
            print(f"  [warn] {src}: some ecs lack round_found, "
                  f"keeping file order")

        if slots is not None and len(ecs) > slots:
            print(f"  {src}: truncating {len(ecs)} → {slots} edge cases")
            ecs = ecs[:slots]

        all_edge_cases.extend(ecs)

    # Backfill is constrained_completion-specific: it reconstructs IFEval
    # {type, kwargs, description} from skill_name + param_pool. Other
    # categories' `skills` are jailbreak / persuasion / perturbation names
    # with completely different semantics, and their judges don't need
    # constraint metadata — leave their empty metadata alone.
    needs_backfill = (
        category == "constrained_completion"
        and any(
            not ec.get("metadata") and ec.get("skills") for ec in all_edge_cases
        )
    )
    skill_dict = _load_skill_dict(category) if needs_backfill else {}
    if needs_backfill:
        n_missing = sum(
            1 for ec in all_edge_cases
            if not ec.get("metadata") and ec.get("skills")
        )
        print(f"  Backfilling metadata for {n_missing}/{len(all_edge_cases)} edge cases...")

    # Assign unique prompt IDs
    dataset = []
    for i, ec in enumerate(all_edge_cases, 1):
        metadata = ec.get("metadata", {})

        # Backfill: reconstruct constraint metadata from prompt + skills
        if not metadata and skill_dict and ec.get("skills"):
            constraints = _reconstruct_constraints(
                ec["prompt"], ec["skills"], skill_dict,
            )
            metadata = {"constraints": constraints}

        entry = {
            "prompt_id": f"ec_{i:03d}",
            "prompt": ec["prompt"],
            "bare_request": ec["bare_request"],
            "skills": ec.get("skills", []),
            "skill_count": len(ec.get("skills", [])),
            "follow_up_skills": ec.get("follow_up_skills", []),
            "source_model": ec["source_model"],
            "p_hat": ec["p_hat"],
            "pass_count": ec.get("pass_count"),
            "total_trials": ec.get("total_trials"),
            "round_found": ec.get("round_found"),
            "metadata": metadata,
            "category": category,
        }
        dataset.append(entry)

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build edge case dataset from SGBS experiment results",
    )
    parser.add_argument(
        "--results-dir", default="results/smoke_test",
        help="Directory containing experiment result subdirectories",
    )
    parser.add_argument(
        "--output-dir", default="benchmarks/harmful_refusal",
        help="Output directory for the consolidated dataset",
    )
    parser.add_argument(
        "--category", default="harmful_refusal",
        help="Safety category label",
    )
    parser.add_argument(
        "--slots", type=int, default=100,
        help="Max edge cases to keep per source_model (file-order truncation). "
             "Note: changing this changes prompt_id assignment, so outputs "
             "with different --slots are not mergeable by prompt_id.",
    )
    args = parser.parse_args()

    dataset = build_dataset(args.results_dir, args.category, args.slots)

    # Write dataset
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"edge_case_dataset_slot{args.slots}.jsonl"
    with open(out_path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Summary
    models = {}
    for entry in dataset:
        m = entry["source_model"]
        models[m] = models.get(m, 0) + 1

    print(f"Dataset written to {out_path}")
    print(f"Total: {len(dataset)} edge cases from {len(models)} models")
    for m, n in sorted(models.items()):
        print(f"  {m}: {n}")


if __name__ == "__main__":
    main()
