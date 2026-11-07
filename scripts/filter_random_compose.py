"""Two-stage panel filter for a random-compose baseline dataset.

Takes a JSONL produced by `random_compose_baseline.py` (K candidate items) and
filters it down to an anchor set via:

  Stage 1 — COARSE
    Run all K items × PANEL_MODELS × N_coarse trials.
    Compute panel-mean p̄_x = (1/M) · Σ_m p̂_{m,x}.
    Keep items with p̄ ∈ [coarse_lo, coarse_hi] (loose filter, default [0.0159, 0.9841]).

  Stage 2 — FINE
    Run kept items × PANEL_MODELS × N_fine additional trials.
    Combine with coarse trials → total N = N_coarse + N_fine per (item, model).
    Recompute panel-mean p̄ and keep items with p̄ ∈ [fine_lo, fine_hi]
    (strict filter, default [0.05, 0.95]).

Seed handling: coarse trials use base seed; fine trials use base seed + 10000
so the two sets of trials never collide.

Output layout:
    {output_dir}/
      stage1_coarse/
        {model_short}/response_matrix.jsonl   # N_c trials per item
        {model_short}/metadata.json
      stage2_fine/
        {model_short}/response_matrix.jsonl   # N_f additional trials
        {model_short}/metadata.json
      combined/
        {model_short}/response_matrix.jsonl   # merged N=N_c+N_f for kept items
      coarse_filter_report.json               # p̄ and pass/drop per item after stage 1
      fine_filter_report.json                 # p̄ and pass/drop per item after stage 2
      anchor_set.jsonl                        # final kept items (subset of input dataset)

Usage:
    python scripts/filter_random_compose.py \\
        --dataset benchmarks/harmful_refusal/random_baseline.jsonl \\
        --output-dir results/harmful_refusal/random_filter \\
        --category harmful_refusal \\
        --n-coarse 7 --n-fine 33 \\
        --coarse-lo 0.0159 --coarse-hi 0.9841 \\
        --fine-lo 0.05 --fine-hi 0.95
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stdout block-buffers when redirected to a file; only the explicit
# flush=True calls inside run_jobs_on_model land in real time, so per-model
# "Done:" / "Starting:" boundaries are invisible until the process exits.
# Switch to line-buffered so every print() flushes on newline.
sys.stdout.reconfigure(line_buffering=True)

from core.panel_config import PANEL_MODELS
from core.judge_factory import create_judge
from core.io import load_dataset
from core.panel_inference import run_model_on_items
from core.trial_utils import aggregate_trials
from core.inference_budgets import get_max_tokens, get_max_model_len


COARSE_SEED_BASE = 42
FINE_SEED_OFFSET = 10000  # coarse uses seeds in [0, 2^31); fine uses seeds + offset


def model_short(name: str) -> str:
    return name.replace("/", "_").replace("-", "_")


def load_per_model_response_matrix(path: Path) -> dict[str, dict]:
    """Return {prompt_id: entry} for a per-model response_matrix.jsonl."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            out[e["prompt_id"]] = e
    return out


def compute_panel_mean_p(
    stage_dir: Path,
    items: list[dict],
    panel_models: list[dict],
    *,
    require_all: bool = False,
) -> dict[str, dict]:
    """For each item, compute panel-mean p̄ from per-model response matrices
    written under {stage_dir}/{model_short}/response_matrix.jsonl.

    If require_all is True, exits with error when any panel model's per-model
    file is missing — used to guarantee panel-mean is computed over the full
    panel (not a --models subset).

    Returns {prompt_id: {
        "panel_mean_p": float,
        "per_model_p_hat": {model_name: p_hat},
        "per_model_pass_count": {model_name: int},
        "per_model_total_trials": {model_name: int},
    }}.
    """
    by_item: dict[str, dict] = {}
    for it in items:
        by_item[it["prompt_id"]] = {
            "panel_mean_p": None,
            "per_model_p_hat": {},
            "per_model_pass_count": {},
            "per_model_total_trials": {},
        }

    for mcfg in panel_models:
        name = mcfg["name"]
        path = stage_dir / model_short(name) / "response_matrix.jsonl"
        if not path.exists():
            if require_all:
                print(f"[error] required panel data missing: {path}")
                print(f"        compute_panel_mean_p was called with require_all=True "
                      f"(full-panel computation); cannot proceed without all models.")
                sys.exit(1)
            print(f"[warn] missing: {path}")
            continue
        entries = load_per_model_response_matrix(path)
        for pid, e in entries.items():
            if pid not in by_item:
                continue
            by_item[pid]["per_model_p_hat"][name] = e["p_hat"]
            by_item[pid]["per_model_pass_count"][name] = e["pass_count"]
            by_item[pid]["per_model_total_trials"][name] = e["total_trials"]

    for pid, info in by_item.items():
        # Filter out None p_hat (cell had zero valid trials — all judges failed).
        p_hats = [p for p in info["per_model_p_hat"].values() if p is not None]
        info["panel_mean_p"] = sum(p_hats) / len(p_hats) if p_hats else None
    return by_item


def merge_coarse_fine_trials(
    coarse_dir: Path,
    fine_dir: Path,
    kept_ids: set[str],
    panel_models: list[dict],
    combined_dir: Path,
) -> None:
    """For each panel model, concatenate coarse+fine trials for kept items and
    write combined/{model_short}/response_matrix.jsonl."""
    for mcfg in panel_models:
        name = mcfg["name"]
        ms = model_short(name)
        coarse_path = coarse_dir / ms / "response_matrix.jsonl"
        fine_path = fine_dir / ms / "response_matrix.jsonl"
        if not coarse_path.exists() or not fine_path.exists():
            print(f"[warn] skip merge for {name}: missing coarse or fine file")
            continue
        coarse_map = load_per_model_response_matrix(coarse_path)
        fine_map = load_per_model_response_matrix(fine_path)

        merged_entries = []
        for pid in sorted(kept_ids):
            c = coarse_map.get(pid)
            f_ = fine_map.get(pid)
            if c is None or f_ is None:
                print(f"[warn] {name}/{pid}: missing coarse or fine entry, skipping")
                continue
            trials = list(c["trials"]) + list(f_["trials"])
            stats = aggregate_trials(trials)
            merged_entries.append({
                "prompt_id": pid,
                "model": name,
                "trials": trials,
                **stats,
            })

        out_path = combined_dir / ms / "response_matrix.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for e in merged_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"  merged → {out_path}  ({len(merged_entries)} items)")


def save_filter_report(
    path: Path,
    stage: str,
    items: list[dict],
    panel_info: dict[str, dict],
    lo: float,
    hi: float,
    kept_ids: set[str],
) -> None:
    p_bar_list = [info["panel_mean_p"] for info in panel_info.values()
                  if info["panel_mean_p"] is not None]
    report = {
        "stage": stage,
        "n_items_input": len(items),
        "n_items_kept": len(kept_ids),
        "pass_rate": len(kept_ids) / len(items) if items else 0.0,
        "filter_range": [lo, hi],
        "panel_mean_p_stats": {
            "mean": sum(p_bar_list) / len(p_bar_list) if p_bar_list else None,
            "min": min(p_bar_list) if p_bar_list else None,
            "max": max(p_bar_list) if p_bar_list else None,
        },
        "per_item": [
            {
                "prompt_id": it["prompt_id"],
                "panel_mean_p": panel_info[it["prompt_id"]]["panel_mean_p"],
                "per_model_p_hat": panel_info[it["prompt_id"]]["per_model_p_hat"],
                "kept": it["prompt_id"] in kept_ids,
            }
            for it in items
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  report → {path}")


def main():
    parser = argparse.ArgumentParser(description="Two-stage panel filter for random-compose baseline")
    parser.add_argument("--dataset", required=True,
                        help="Input JSONL from random_compose_baseline.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--category", required=True)
    # Project standard: N_coarse + N_fine = 40 total trials per cell.
    # Split defaults locked to docs/filter_method_spec.md §F (2026-04-24):
    #   α=5% FNR tolerance at p₀=0.05 (psychometric convention)
    #   N_c=7  (closed form: ⌈ln(α)/ln(1−p₀)/M⌉ at M=9; smallest feasible)
    #   N_f=33
    #   coarse [1/63, 62/63] ≈ [0.0159, 0.9841]  (k_min=1 at M·N_c=63)
    #   fine   [0.05, 0.95]                       (AVG winner, spec §E)
    # Exact FNR(p=0.05) = 0.95^63 ≈ 3.95% ≤ α.
    # Empirical 4-cat mean savings 20.3% (B 44.7%, A 29.1%, A' 2.7%, D 4.8%);
    # universal argmax-mean across 4 cats. See
    # scripts/analysis/empirical_pool_costs.py for the sweep.
    parser.add_argument("--n-coarse", type=int, default=7)
    parser.add_argument("--n-fine", type=int, default=33)
    parser.add_argument("--coarse-lo", type=float, default=0.0159)
    parser.add_argument("--coarse-hi", type=float, default=0.9841)
    parser.add_argument("--fine-lo", type=float, default=0.05)
    parser.add_argument("--fine-hi", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=COARSE_SEED_BASE)
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--max-workers", type=int, default=100)
    parser.add_argument("--server-timeout", type=int, default=900)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to run (default: all panel models)")
    parser.add_argument("--skip-coarse", action="store_true",
                        help="Skip stage 1 (assume coarse outputs already on disk)")
    parser.add_argument("--skip-fine", action="store_true",
                        help="Skip stage 2 (assume fine outputs already on disk)")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip post-fine merge + final filter + anchor_set write. "
                             "Use when running fine inference for a --models subset in parallel; "
                             "afterwards do a separate full-panel run with --skip-coarse "
                             "--skip-fine (no --skip-merge) to merge and write the anchor set.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a model's stage if its response_matrix.jsonl already exists")
    parser.add_argument("--cuda-devices", default=None,
                        help="Override panel_config cuda_devices for ALL models in this run "
                             "(e.g. '2,3'). For running multiple instances in parallel on "
                             "disjoint GPUs without modifying panel_config.py. GPU count must "
                             "match the model's --tensor-parallel-size in extra_args.")
    args = parser.parse_args()

    all_models = list(PANEL_MODELS)
    if args.models:
        wanted = {n.strip() for n in args.models.split(",")}
        all_models = [m for m in all_models if m["name"] in wanted]
        if not all_models:
            print(f"No matching panel models for: {args.models}")
            sys.exit(1)
    if args.cuda_devices:
        all_models = [{**m, "cuda_devices": args.cuda_devices} for m in all_models]

    dataset = load_dataset(args.dataset)
    print(f"Dataset: {len(dataset)} items  ({args.dataset})")
    print(f"Panel: {[m['name'] for m in all_models]}")
    print(f"N_coarse = {args.n_coarse}, N_fine = {args.n_fine}, total N = {args.n_coarse + args.n_fine}")
    print(f"Coarse filter: [{args.coarse_lo}, {args.coarse_hi}]")
    print(f"Fine   filter: [{args.fine_lo}, {args.fine_hi}]")

    out_dir = Path(args.output_dir)
    coarse_dir = out_dir / "stage1_coarse"
    fine_dir = out_dir / "stage2_fine"
    combined_dir = out_dir / "combined"

    # -----------------------------------------------------------
    # Shared judge
    # -----------------------------------------------------------
    judge_config = {
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_TOKEN",
        "base_url": "https://api.deepseek.com",
    }
    judge = create_judge(args.category, judge_config)

    # -----------------------------------------------------------
    # STAGE 1: COARSE
    # -----------------------------------------------------------
    print(f"\n{'#'*60}\n# STAGE 1: COARSE (N={args.n_coarse} per item)\n{'#'*60}")
    if not args.skip_coarse:
        for mcfg in all_models:
            ms = model_short(mcfg["name"])
            out_path = coarse_dir / ms / "response_matrix.jsonl"
            if args.skip_existing and out_path.exists():
                print(f"  SKIP (existing): {mcfg['name']}")
                continue
            ok, msg = run_model_on_items(
                items=dataset,
                model_cfg=mcfg,
                judge=judge,
                n_trials=args.n_coarse,
                seed=args.seed,
                port=args.port,
                max_workers=args.max_workers,
                server_timeout=args.server_timeout,
                out_path=out_path,
                max_tokens=get_max_tokens(args.category, mcfg["name"]),
                max_model_len=get_max_model_len(args.category, mcfg["name"]),
            )
            if not ok:
                print(f"  [stop] stage 1 failed for {mcfg['name']}: {msg}")
                sys.exit(1)
    else:
        print("  --skip-coarse: assuming stage1_coarse outputs already on disk")

    # Compute panel mean + coarse filter.
    # Always use the FULL panel (not the --models subset) so kept_items is
    # consistent across parallel runs that filter --models for fine inference.
    coarse_info = compute_panel_mean_p(
        coarse_dir, dataset, list(PANEL_MODELS), require_all=True
    )
    coarse_kept_ids = {
        pid for pid, info in coarse_info.items()
        if info["panel_mean_p"] is not None
        and args.coarse_lo <= info["panel_mean_p"] <= args.coarse_hi
    }
    kept_items = [it for it in dataset if it["prompt_id"] in coarse_kept_ids]
    print(f"\n  Coarse filter: {len(kept_items)} / {len(dataset)} kept "
          f"({100*len(kept_items)/len(dataset):.1f}%)")

    save_filter_report(
        out_dir / "coarse_filter_report.json",
        stage="coarse",
        items=dataset,
        panel_info=coarse_info,
        lo=args.coarse_lo,
        hi=args.coarse_hi,
        kept_ids=coarse_kept_ids,
    )

    if not kept_items:
        print("  No items kept after coarse filter — aborting.")
        sys.exit(2)

    # -----------------------------------------------------------
    # STAGE 2: FINE
    # -----------------------------------------------------------
    print(f"\n{'#'*60}\n# STAGE 2: FINE (+{args.n_fine} additional trials per kept item)\n{'#'*60}")
    if not args.skip_fine:
        for mcfg in all_models:
            ms = model_short(mcfg["name"])
            out_path = fine_dir / ms / "response_matrix.jsonl"
            if args.skip_existing and out_path.exists():
                print(f"  SKIP (existing): {mcfg['name']}")
                continue
            ok, msg = run_model_on_items(
                items=kept_items,
                model_cfg=mcfg,
                judge=judge,
                n_trials=args.n_fine,
                seed=args.seed + FINE_SEED_OFFSET,
                port=args.port,
                max_workers=args.max_workers,
                server_timeout=args.server_timeout,
                out_path=out_path,
                max_tokens=get_max_tokens(args.category, mcfg["name"]),
                max_model_len=get_max_model_len(args.category, mcfg["name"]),
            )
            if not ok:
                print(f"  [stop] stage 2 failed for {mcfg['name']}: {msg}")
                sys.exit(1)
    else:
        print("  --skip-fine: assuming stage2_fine outputs already on disk")

    # -----------------------------------------------------------
    # MERGE + FINAL FILTER
    # -----------------------------------------------------------
    if args.skip_merge:
        print(f"\n  --skip-merge: skipping merge + final filter + anchor_set write.")
        print(f"\n{'='*60}\n  Filter summary (partial — merge skipped)\n{'='*60}")
        print(f"  input (raw compose)     : {len(dataset)}")
        print(f"  after coarse [{args.coarse_lo}, {args.coarse_hi}] : {len(kept_items)} ({100*len(kept_items)/len(dataset):.1f}%)")
        print(f"  output dir              : {out_dir}")
        return

    # Verify every panel model has both coarse + fine on disk before merging —
    # merge / final filter / anchor_set must use the FULL panel, regardless of
    # whether --models was filtered for inference.
    missing_files: list[Path] = []
    for m in PANEL_MODELS:
        ms = model_short(m["name"])
        for stage_d in (coarse_dir, fine_dir):
            p = stage_d / ms / "response_matrix.jsonl"
            if not p.exists():
                missing_files.append(p)
    if missing_files:
        print(f"\n[error] merge requires full-panel coarse + fine data on disk; "
              f"{len(missing_files)} files missing:")
        for p in missing_files[:10]:
            print(f"   {p}")
        if len(missing_files) > 10:
            print(f"   ... ({len(missing_files) - 10} more)")
        print(f"  Run the missing models' fine stage first, or pass --skip-merge "
              f"to defer merging to a later full-panel invocation.")
        sys.exit(1)

    print(f"\n{'#'*60}\n# MERGE coarse + fine  →  combined  (N={args.n_coarse + args.n_fine})\n{'#'*60}")
    merge_coarse_fine_trials(
        coarse_dir=coarse_dir,
        fine_dir=fine_dir,
        kept_ids=coarse_kept_ids,
        panel_models=list(PANEL_MODELS),
        combined_dir=combined_dir,
    )

    combined_info = compute_panel_mean_p(
        combined_dir, kept_items, list(PANEL_MODELS), require_all=True
    )
    anchor_ids = {
        pid for pid, info in combined_info.items()
        if info["panel_mean_p"] is not None
        and args.fine_lo <= info["panel_mean_p"] <= args.fine_hi
    }
    anchor_set = [it for it in kept_items if it["prompt_id"] in anchor_ids]

    print(f"\n  Fine filter: {len(anchor_set)} / {len(kept_items)} kept "
          f"({100*len(anchor_set)/len(kept_items):.1f}% of post-coarse, "
          f"{100*len(anchor_set)/len(dataset):.1f}% of input)")

    save_filter_report(
        out_dir / "fine_filter_report.json",
        stage="fine",
        items=kept_items,
        panel_info=combined_info,
        lo=args.fine_lo,
        hi=args.fine_hi,
        kept_ids=anchor_ids,
    )

    anchor_path = out_dir / "anchor_set.jsonl"
    with open(anchor_path, "w") as f:
        for it in anchor_set:
            enriched = dict(it)
            info = combined_info[it["prompt_id"]]
            enriched["panel_mean_p"] = info["panel_mean_p"]
            enriched["per_model_p_hat"] = info["per_model_p_hat"]
            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    print(f"\n  anchor_set → {anchor_path}  ({len(anchor_set)} items)")

    # -----------------------------------------------------------
    # Summary
    # -----------------------------------------------------------
    print(f"\n{'='*60}\n  Filter summary\n{'='*60}")
    print(f"  input (raw compose)     : {len(dataset)}")
    print(f"  after coarse [{args.coarse_lo}, {args.coarse_hi}] : {len(kept_items)} ({100*len(kept_items)/len(dataset):.1f}%)")
    print(f"  after fine   [{args.fine_lo}, {args.fine_hi}] : {len(anchor_set)} ({100*len(anchor_set)/len(dataset):.1f}%)")
    print(f"  output dir              : {out_dir}")


if __name__ == "__main__":
    main()
