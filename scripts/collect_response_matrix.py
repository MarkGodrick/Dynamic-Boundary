"""Collect response matrix: run all edge case prompts on panel/validation models.

For each model:
  1. Start VLLMModel (auto-manages server lifecycle)
  2. Run prompts × N trials concurrently
  3. Judge each response (ConstraintJudge or LLMJudge per category)
  4. Skip source_model's own edge cases (reuse search-phase data)
  5. Save results to {output_dir}/{model_short}/response_matrix.jsonl

Three model lists are defined at the top of this file:
  - PANEL_MODELS:      locked M=9 reference panel (default target).
                       Canonical spec: docs/filter_method_spec.md §G.
  - VALIDATION_MODELS: held-out validation models used to test that panel-
                       fitted β generalizes beyond the panel (--include-validation).
  - CANDIDATE_MODELS:  reserved for future panel re-selection (panel currently
                       locked; --candidates-only is a no-op until refilled).

Usage:
    # Collect response matrix on the full panel
    python scripts/collect_response_matrix.py \\
        --dataset benchmarks/constrained_completion/edge_case_dataset.jsonl \\
        --output-dir results/response_matrix/constrained_completion \\
        --category constrained_completion \\
        --n-trials 20

    # Single model
    python scripts/collect_response_matrix.py \\
        --dataset benchmarks/constrained_completion/edge_case_dataset.jsonl \\
        --category constrained_completion \\
        --models "google/gemma-3-4b-it"

    # Panel re-selection profiling: run only CANDIDATE_MODELS on an existing
    # edge_case dataset, then jointly IRT-fit with the existing panel response
    # matrix to get candidate θ values.
    python scripts/collect_response_matrix.py \\
        --dataset benchmarks/constrained_completion/edge_case_dataset.jsonl \\
        --output-dir results/constrained_completion/panel_profiling \\
        --category constrained_completion \\
        --n-trials 40 \\
        --candidates-only
"""

import argparse
import hashlib
import json
import random
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# When stdout is redirected to a file (e.g. nohup or `>run.log`), Python
# block-buffers it. Only the explicit `flush=True` calls inside
# run_jobs_on_model would land in real time, so the per-model "Done:" /
# "Starting:" boundaries stayed invisible until process exit. Switch to
# line-buffered so every print() flushes on newline.
sys.stdout.reconfigure(line_buffering=True)

from core.panel_inference import run_jobs_on_model
from core.panel_config import PANEL_MODELS, VALIDATION_MODELS, CANDIDATE_MODELS
from core.judge_factory import create_judge
from core.io import load_dataset
from core.trial_utils import aggregate_trials

from core.inference_budgets import get_max_tokens, get_max_model_len


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dataset_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def load_prior_results(
    result_path: Path, meta_path: Path, ds_hash: str, n_trials: int, model_name: str,
) -> dict[str, list[dict]]:
    """Read existing response_matrix.jsonl + metadata.json. Returns
    `prior_completed` (pid → list[trial], drop trial_exception). Hard-fails
    on dataset_hash / n_trials mismatch so seed regen lines up.
    """
    if not result_path.exists():
        return {}
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text())
        except Exception:
            prev = {}
        prev_hash = prev.get("dataset_hash")
        if prev_hash and prev_hash != ds_hash:
            print(
                f"ERROR: dataset hash mismatch for {model_name}: "
                f"prev={prev_hash}, now={ds_hash}. Refusing to resume — "
                f"seeds would no longer line up. Delete the directory or "
                f"use a fresh --output-dir."
            )
            sys.exit(1)
        prev_n = prev.get("n_trials")
        if prev_n is not None and prev_n != n_trials:
            print(
                f"ERROR: n_trials mismatch for {model_name}: "
                f"prev={prev_n}, now={n_trials}. Item-major RNG would shift "
                f"seeds; refusing to resume. Delete the directory or use "
                f"a fresh --output-dir."
            )
            sys.exit(1)

    prior_completed: dict[str, list[dict]] = {}
    with open(result_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") == "search_phase":
                continue
            kept = [
                t for t in (row.get("trials") or [])
                if t.get("judge_outcome") != "trial_exception"
            ]
            if kept:
                prior_completed[row["prompt_id"]] = kept
    return prior_completed


def try_full_prefill(
    items: list[dict],
    n_trials: int,
    seed: int,
    model_name: str,
    prior_completed: dict[str, list[dict]],
) -> list[dict] | None:
    """Check whether prior_completed already covers every (item, trial_seed)
    for `items`. If so, return result rows in run_jobs_on_model's schema so
    the caller can skip vLLM startup entirely. Returns None on any miss.

    Seed regeneration mirrors run_jobs_on_model: master seed → item-major
    rng.randint draws. dataset_hash + n_trials are already validated by
    load_prior_results, so a full match here implies the seeds line up.
    """
    if not prior_completed:
        return None

    rng = random.Random(seed)
    rows: list[dict] = []
    for item in items:
        item_seeds = [rng.randint(0, 2**31 - 1) for _ in range(n_trials)]
        seed_map = {t["seed"]: t for t in prior_completed.get(item["prompt_id"], [])}
        trials = [seed_map.get(s) for s in item_seeds]
        if any(t is None for t in trials):
            return None
        rows.append({
            "prompt_id": item["prompt_id"],
            "model": model_name,
            "trials": trials,
            **aggregate_trials(trials),
        })
    return rows


def make_save_callback(
    model_name: str,
    model_out_dir: Path,
    own_results: list[dict],
    args,
    ds_hash: str,
    model_cfg: dict,
):
    """Build a save_callback closure for run_jobs_on_model. The callback
    merges own_results (search_phase, frozen) with the partial inference
    rows it receives, sorts by prompt_id, and atomically writes both
    response_matrix.jsonl and metadata.json.
    """
    result_path = model_out_dir / "response_matrix.jsonl"
    meta_path = model_out_dir / "metadata.json"
    tmp_path = result_path.with_suffix(".jsonl.tmp")

    def _save(partial_inference: list[dict]) -> None:
        all_rows = list(own_results)
        all_rows.extend(partial_inference)
        all_rows.sort(key=lambda r: r["prompt_id"])

        with open(tmp_path, "w") as f:
            for r in all_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp_path.replace(result_path)

        n_done_trials = sum(len(r.get("trials") or []) for r in partial_inference)
        meta = {
            "model": model_name,
            "category": args.category,
            "n_trials": args.n_trials,
            "seed": args.seed,
            "n_prompts": len(all_rows),
            "n_skipped_own": len(own_results),
            "n_evaluated_target": len(partial_inference),
            "n_evaluated_done_trials": n_done_trials,
            "dataset_hash": ds_hash,
            "dataset_path": args.dataset,
            "last_updated": datetime.now().isoformat(),
            "vllm_config": model_cfg,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    return _save


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect response matrix on panel/validation models",
    )
    parser.add_argument(
        "--dataset", default="benchmarks/constrained_completion/edge_case_dataset.jsonl",
    )
    parser.add_argument("--output-dir", default="results/response_matrix")
    parser.add_argument("--category", default="constrained_completion")
    parser.add_argument("--n-trials", type=int, default=40,
                        help="Repeated target trials per model–item pair (default: 40).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--max-workers", type=int, default=100)
    parser.add_argument("--server-timeout", type=int, default=900)
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model names to run (default: all panel models)",
    )
    parser.add_argument("--include-validation", action="store_true")
    parser.add_argument(
        "--candidates-only", action="store_true",
        help="Profile mode: run ONLY CANDIDATE_MODELS (panel/validation ignored). "
             "Use with --dataset pointing at edge_case_dataset.jsonl for panel "
             "re-selection profiling.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a model entirely if its response_matrix.jsonl exists "
             "(legacy behavior). Default behavior is auto-resume from "
             "partial state.",
    )
    parser.add_argument(
        "--save-interval", type=float, default=60.0,
        help="Seconds between periodic saves of partial inference state.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--cuda-devices", default=None,
        help="Override panel_config cuda_devices for ALL models in this run "
             "(e.g. '2,3'). For running multiple collectors in parallel on "
             "disjoint GPUs without modifying panel_config.py. GPU count must "
             "match the model's --tensor-parallel-size in extra_args.",
    )
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset)

    # Match the category to the dataset so dispatch selects its intended judge.
    # Datasets without a category field receive a warning.
    ds_categories = {e["category"] for e in dataset if "category" in e}
    if not ds_categories:
        print("  [warn] dataset has no top-level 'category' field; "
              "skipping --category cross-check")
    elif ds_categories != {args.category}:
        print(f"ERROR: --category={args.category!r} but dataset contains "
              f"{ds_categories}. Refusing to run (silent category mismatch "
              f"produces wrong judgments; cf. 2026-04-20 sycophancy incident).")
        sys.exit(1)

    # Build model list
    if args.candidates_only:
        if not CANDIDATE_MODELS:
            print("ERROR: --candidates-only but CANDIDATE_MODELS is empty. "
                  "Panel is locked (see docs/filter_method_spec.md §G); refill "
                  "CANDIDATE_MODELS only for new panel re-selection work.")
            sys.exit(1)
        all_models = list(CANDIDATE_MODELS)
        if args.include_validation:
            print("NOTE: --include-validation ignored in --candidates-only mode")
    else:
        all_models = list(PANEL_MODELS)
        if args.include_validation:
            all_models.extend(VALIDATION_MODELS)

    if args.models:
        filter_names = {n.strip() for n in args.models.split(",")}
        all_models = [m for m in all_models if m["name"] in filter_names]
        if not all_models:
            print(f"No matching models found for: {args.models}")
            sys.exit(1)

    # Dataset hash: stamped in metadata.json and verified on resume so that
    # changes to dataset content/order don't silently misalign seeds.
    ds_hash = dataset_hash(args.dataset)

    # SIGINT/SIGTERM → graceful drain. The handler only sets the event;
    # run_jobs_on_model checks it and finishes in-flight work, then returns
    # the partial result list. The outer model loop then breaks.
    stop_event = threading.Event()

    def _on_signal(sig, _frame):
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] caught signal {sig}, "
            f"will save and exit after current model drains",
            flush=True,
        )
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print(f"Dataset: {len(dataset)} prompts (hash={ds_hash})")
    print(f"Trials per prompt: {args.n_trials}")
    print(f"Category: {args.category}")
    print(f"Models to run: {len(all_models)}")

    # Judge config (shared across all models, only used for LLM-based judges)
    judge_config = {
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_TOKEN",
        "base_url": "https://api.deepseek.com",
    }
    judge = create_judge(args.category, judge_config)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for model_cfg in all_models:
        model_name = model_cfg["name"]
        model_short = model_name.replace("/", "_").replace("-", "_")
        model_out_dir = out_dir / model_short
        result_path = model_out_dir / "response_matrix.jsonl"
        meta_path = model_out_dir / "metadata.json"

        if args.skip_existing and result_path.exists():
            print(f"\n{'='*60}")
            print(f"  SKIP (existing): {model_name}")
            print(f"{'='*60}")
            summary[model_name] = "SKIPPED"
            continue

        if stop_event.is_set():
            print(f"  [skip] {model_name} (stop_event set before model start)")
            summary[model_name] = "SKIPPED (stop)"
            continue

        print(f"\n{'='*60}")
        print(f"  {datetime.now().strftime('%H:%M:%S')} | Starting: {model_name}")
        print(f"{'='*60}")

        # Split dataset: own edge cases vs others
        own_entries = [e for e in dataset if e.get("source_model") == model_name]
        other_entries = [e for e in dataset if e.get("source_model") != model_name]
        print(f"  {len(own_entries)} prompts from search phase (skip inference)")
        print(f"  {len(other_entries)} prompts to evaluate")

        # Own (search-phase) results — reused verbatim, never re-inferred
        own_results = []
        for e in own_entries:
            total = e.get("total_trials")
            own_results.append({
                "prompt_id": e["prompt_id"],
                "model": model_name,
                "trials": [],
                "pass_count": e.get("pass_count"),
                "total_trials": total,
                "n_attempted": total,
                "n_safety_classified": 0,
                "n_api_failed": 0,
                "n_trial_failed": 0,
                "p_hat": e.get("p_hat"),
                "source": "search_phase",
            })

        # Resume: load any prior partial state from disk. Hash + n_trials
        # checks happen inside load_prior_results (hard-fail on mismatch).
        prior_completed = load_prior_results(
            result_path, meta_path, ds_hash, args.n_trials, model_name,
        )
        n_prior = sum(len(v) for v in prior_completed.values())
        if n_prior:
            print(
                f"  [resume] {n_prior} prior trials loaded "
                f"({len(prior_completed)} prompts)"
            )

        model_out_dir.mkdir(parents=True, exist_ok=True)
        save_cb = make_save_callback(
            model_name, model_out_dir, own_results, args, ds_hash, model_cfg,
        )

        run_results: list[dict] = []
        prefilled = try_full_prefill(
            other_entries, args.n_trials, args.seed, model_name, prior_completed,
        )
        if prefilled is not None:
            print(
                f"  [skip-server] all {len(other_entries)} prompts × "
                f"{args.n_trials} trials already in prior state; "
                f"skipping vLLM startup"
            )
            run_results = prefilled
        elif other_entries:
            from models.vllm_model import VLLMModel

            try:
                vllm = VLLMModel(
                    model_name=model_name,
                    conda_env=model_cfg["conda_env"],
                    cuda_devices=args.cuda_devices or model_cfg["cuda_devices"],
                    port=args.port,
                    dtype=model_cfg.get("dtype", "bfloat16"),
                    max_model_len=get_max_model_len(args.category, model_name),
                    gpu_memory_utilization=model_cfg["gpu_mem_util"],
                    extra_args=model_cfg.get("extra_args", []),
                    server_timeout=args.server_timeout,
                    max_tokens=get_max_tokens(args.category, model_name),
                )
            except RuntimeError as e:
                print(f"  FAILED: {e}")
                summary[model_name] = "FAILED (server)"
                continue

            try:
                run_results = run_jobs_on_model(
                    items=other_entries,
                    model=vllm,
                    judge=judge,
                    n_trials=args.n_trials,
                    seed=args.seed,
                    max_workers=args.max_workers,
                    model_name=model_name,
                    prior_completed=prior_completed,
                    save_callback=save_cb,
                    save_interval=args.save_interval,
                    stop_event=stop_event,
                )
            finally:
                vllm.shutdown()
                time.sleep(3)

        # Final save: covers (a) the no-inference case where all entries are
        # own_entries, and (b) corrects any timing-edge state from the last
        # periodic save during run_jobs_on_model.
        save_cb(run_results)

        # Stats over the merged result set
        results = list(own_results) + list(run_results)
        results.sort(key=lambda r: r["prompt_id"])
        p_hats = [r["p_hat"] for r in results if r.get("p_hat") is not None]
        avg_p = sum(p_hats) / len(p_hats) if p_hats else float("nan")
        n_attempted = sum(r.get("n_attempted", r["total_trials"]) for r in results)
        n_safety = sum(r.get("n_safety_classified", 0) for r in results)
        n_api = sum(r.get("n_api_failed", 0) for r in results)
        n_trial = sum(r.get("n_trial_failed", 0) for r in results)
        print(
            f"  Done: avg p_hat={avg_p:.3f} | "
            f"attempted={n_attempted} safety_classified={n_safety} "
            f"api_failed={n_api} trial_failed={n_trial}"
        )
        summary[model_name] = (
            "STOPPED" if stop_event.is_set() else "OK"
        )

        if stop_event.is_set():
            print(
                f"\n[{datetime.now().strftime('%H:%M:%S')}] stop_event set, "
                f"skipping remaining models"
            )
            break

    # Print summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for model, status in summary.items():
        print(f"  {status:8s} | {model}")


if __name__ == "__main__":
    main()
