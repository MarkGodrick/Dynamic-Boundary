"""Re-judge CIF response matrices after the ConstraintJudge CoT-strip fix.

ConstraintJudge previously fed the raw assistant content (including any
<think>...</think> reasoning block) to IFEval/IFBench checkers, which
counted reasoning tokens / matched keywords inside the reasoning / failed
JSON validation on the prepended <think> block. Reasoning-emitting models
(Falcon-H1R-7B, Phi-4-reasoning-plus) thus produced systematically wrong
`passed` labels in CIF.

After judges/constraint_judge.py was patched (2026-05-16) to strip the
leading <think>...</think> block before feeding the checkers, every
already-collected CIF response_matrix.jsonl for affected models must have
its `passed` / `pass_count` / `total_trials` / `p_hat` recomputed in place.
The raw `response` strings are NOT modified — only the judge labels change.

Usage:
    conda run -n boundary --no-capture-output python -u \\
        scripts/rejudge_cif.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judges.constraint_judge import ConstraintJudge


TARGETS = [
    "results/constrained_completion/anchor_set/microsoft_Phi_4_reasoning_plus",
    "results/constrained_completion/anchor_set/tiiuae_Falcon_H1R_7B",
    "results/constrained_completion/anchor_set_alo_only/microsoft_Phi_4_reasoning_plus",
    "results/constrained_completion/anchor_set_alo_only/tiiuae_Falcon_H1R_7B",
]

JUDGE_REVISION = "2026-05-16_cot_strip"


def load_constraints_index() -> dict[str, list[dict]]:
    """Build prompt_id -> constraints from both anchor bundles (covers AVG ∪ ALO)."""
    index: dict[str, list[dict]] = {}
    for bundle_path in (
        Path("benchmarks/constrained_completion/anchor_bundle_M9.jsonl"),
        Path("benchmarks/constrained_completion/anchor_alo_only_bundle_M9.jsonl"),
    ):
        if not bundle_path.exists():
            raise FileNotFoundError(bundle_path)
        with open(bundle_path) as f:
            for line in f:
                row = json.loads(line)
                pid = row["prompt_id"]
                constraints = (row.get("metadata") or {}).get("constraints", [])
                index[pid] = constraints
    return index


def rejudge_file(model_dir: Path, judge: ConstraintJudge,
                 constraints_index: dict[str, list[dict]]) -> dict:
    jsonl_path = model_dir / "response_matrix.jsonl"
    meta_path = model_dir / "metadata.json"
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    rows = [json.loads(line) for line in open(jsonl_path)]

    n_trials_total = 0
    n_flipped_to_pass = 0
    n_flipped_to_fail = 0
    p_hat_old_sum = 0.0
    p_hat_new_sum = 0.0
    missing_constraints = []

    for row in rows:
        pid = row["prompt_id"]
        if pid not in constraints_index:
            missing_constraints.append(pid)
            continue
        constraints = constraints_index[pid]
        metadata = {"constraints": constraints}

        new_pass = 0
        for trial in row.get("trials") or []:
            old_passed = bool(trial.get("passed"))
            messages = [{"role": "assistant", "content": trial.get("response", "")}]
            new_passed = judge.judge(messages, metadata)
            trial["passed"] = new_passed
            n_trials_total += 1
            if new_passed and not old_passed:
                n_flipped_to_pass += 1
            elif (not new_passed) and old_passed:
                n_flipped_to_fail += 1
            if new_passed:
                new_pass += 1

        old_p_hat = row.get("p_hat") or 0.0
        p_hat_old_sum += old_p_hat
        total = row.get("total_trials") or len(row.get("trials") or [])
        row["pass_count"] = new_pass
        row["p_hat"] = (new_pass / total) if total else None
        p_hat_new_sum += (row["p_hat"] or 0.0)

    with open(jsonl_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["judge_revision"] = JUDGE_REVISION
    meta["judge_rejudge_at"] = datetime.now().isoformat()
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    return {
        "n_rows": len(rows),
        "n_trials": n_trials_total,
        "n_flipped_to_pass": n_flipped_to_pass,
        "n_flipped_to_fail": n_flipped_to_fail,
        "mean_p_hat_old": p_hat_old_sum / len(rows) if rows else float("nan"),
        "mean_p_hat_new": p_hat_new_sum / len(rows) if rows else float("nan"),
        "missing_constraints": missing_constraints,
    }


def main() -> None:
    constraints_index = load_constraints_index()
    judge = ConstraintJudge()

    print(f"Constraints index: {len(constraints_index)} prompt_ids")
    print()
    header = (f"{'target':<82s} {'rows':>5s} {'trials':>6s} "
              f"{'→pass':>6s} {'→fail':>6s} {'p̂_old':>7s} {'p̂_new':>7s} {'Δp̂':>8s}")
    print(header)
    print("-" * 130)

    for target in TARGETS:
        d = Path(target)
        if not d.exists():
            print(f"{target:<82s}  MISSING"); continue
        result = rejudge_file(d, judge, constraints_index)
        delta = result["mean_p_hat_new"] - result["mean_p_hat_old"]
        short = "/".join(d.parts[-3:])
        print(f"{short:<82s} {result['n_rows']:>5d} {result['n_trials']:>6d} "
              f"{result['n_flipped_to_pass']:>6d} {result['n_flipped_to_fail']:>6d} "
              f"{result['mean_p_hat_old']:>7.4f} {result['mean_p_hat_new']:>7.4f} "
              f"{delta:>+8.4f}")
        if result["missing_constraints"]:
            print(f"  WARN: {len(result['missing_constraints'])} prompt_ids "
                  f"missing constraints in index")


if __name__ == "__main__":
    main()
