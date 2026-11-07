"""Direct entry point for SGBS v2 boundary search (no adaptive evaluation).

Runs standalone SGBS for reference-panel models or other targets. The
adaptive-evaluation entry point (main.py) also calls SGBS during Step 2.

Per-model dispatch is idempotent: for each name in the model list, the
script looks under experiment.output_dir for an existing matching run dir
({model_short}_{category}_<timestamp>) and decides:
  - skip: existing dir has edge_cases.jsonl count >= target_edge_cases
  - resume: existing dir is resumable but EC count < target
  - fresh: no matching dir, or matching dir is unresumable
The resumable check requires search_state.json + config.yaml; missing
either triggers a fresh run with a new timestamp (the broken dir is left
untouched).

Usage:
    # Fresh / auto-resume / auto-skip (per model)
    python scripts/run_sgbs.py --config configs/<cat>/search_<name>.yaml \\
        --models name1,name2,name3
    # Or put `models: [n1, n2, ...]` in the yaml
    python scripts/run_sgbs.py --config configs/<cat>/search_<name>.yaml

    # Force fresh (ignore any matching existing dir; new timestamp always)
    python scripts/run_sgbs.py --config <yaml> --models <name> --force-fresh
"""

import argparse
import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

# Allow running as `python scripts/run_sgbs.py ...` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from logger import setup_logging, add_file_handler, get_logger
from search import SGBSAsync, SGBSv2, BareRequest, save_results
from search.sgbs import load_search_state
from search.skill_selector import load_skills
from models import create_target_model, unload_model
from judges import create_judge
from core.model_dispatch import (
    resolve_target_model,
    model_short as _model_short,
    find_existing_run_dir,
)
from core.signal_utils import (
    install_two_stage_handlers,
    set_active_model,
    clear_active_model,
)


logger = get_logger(__name__)


def create_sgbs_runner(config: dict) -> SGBSv2:
    """Select the SGBS implementation from config."""
    mode = (config.get("search") or {}).get("mode", "sync")
    if mode == "async":
        return SGBSAsync(config)
    if mode in ("sync", "batch"):
        return SGBSv2(config)
    raise ValueError(f"Unsupported search.mode: {mode!r}")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(
    config: dict,
    args: argparse.Namespace,
    model_name: str | None = None,
) -> dict:
    config["target_model"] = resolve_target_model(
        config.get("target_model", {}) or {},
        model_name,
    )
    if args.budget:
        config.setdefault("search", {})["budget"] = args.budget
    if args.category:
        config.setdefault("bare_requests", {})["category"] = args.category
        config.setdefault("skills", {})["category"] = args.category
    if args.output_dir:
        config.setdefault("experiment", {})["output_dir"] = args.output_dir
    return config


def load_bare_requests(config: dict) -> list[BareRequest]:
    """Load bare requests from JSONL file specified in config."""
    br_cfg = config.get("bare_requests", {})
    data_path = br_cfg.get("data_path")
    limit = br_cfg.get("limit")

    if not data_path:
        raise ValueError("bare_requests.data_path must be set in config")

    requests = []
    with open(data_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            prompt = entry.get("prompt", entry.get("request", ""))
            metadata = entry.get("metadata", {})
            requests.append(BareRequest(prompt=prompt, metadata=metadata))
    if limit:
        requests = requests[:limit]
    logger.info("Loaded %d bare requests from %s", len(requests), data_path)
    return requests


# ---------------------------------------------------------------------------
# Idempotent dispatch: skip / resume / fresh per model
# ---------------------------------------------------------------------------

def _count_edge_cases(run_dir: Path) -> int:
    ec_path = run_dir / "edge_cases.jsonl"
    if not ec_path.exists():
        return 0
    with ec_path.open() as f:
        return sum(1 for line in f if line.strip())


def _can_resume(run_dir: Path) -> bool:
    return (
        (run_dir / "search_state.json").exists()
        and (run_dir / "config.yaml").exists()
    )


def _resolve_items(
    args: argparse.Namespace, parser: argparse.ArgumentParser,
) -> tuple[list[tuple], int]:
    """Resolve the list of (kind, ...) work items from args + yaml.

    Each item is one of:
      - ("fresh", model_name)
      - ("resume", run_dir, model_name, ec_count)
      - ("skip",  run_dir, model_name, ec_count)
    """
    if not args.config:
        parser.error("--config is required")
    yaml_cfg = load_config(args.config) or {}

    # Model list: CLI > yaml.models > legacy yaml.target_model.name (single).
    yaml_models = yaml_cfg.get("models")
    if not yaml_models:
        legacy_name = (yaml_cfg.get("target_model") or {}).get("name")
        if legacy_name:
            yaml_models = [legacy_name]
    cli_models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models else None
    )
    names = cli_models or yaml_models
    if not names:
        parser.error(
            "no models to run: specify --models or yaml `models` field"
        )

    exp_cfg = yaml_cfg.get("experiment") or {}
    base_dir = Path(exp_cfg.get("output_dir", "results/"))
    if args.output_dir:
        base_dir = Path(args.output_dir)
    category = (yaml_cfg.get("bare_requests") or {}).get("category", "")
    target_k = (yaml_cfg.get("search") or {}).get("target_edge_cases", 10)

    items: list[tuple] = []
    for name in names:
        if args.force_fresh:
            items.append(("fresh", name))
            continue
        matched = find_existing_run_dir(base_dir, f"{_model_short(name)}_{category}_")
        if matched is None:
            items.append(("fresh", name))
            continue
        if not _can_resume(matched):
            logger.info(
                "Existing dir %s lacks search_state.json/config.yaml; "
                "starting fresh (new timestamp)", matched.name,
            )
            items.append(("fresh", name))
            continue
        ec = _count_edge_cases(matched)
        if ec >= target_k:
            items.append(("skip", matched, name, ec))
        else:
            items.append(("resume", matched, name, ec))
    return items, target_k


def main():
    parser = argparse.ArgumentParser(description="SGBS v2 Boundary Search")
    parser.add_argument("--config", help="Path to YAML config")
    parser.add_argument(
        "--models",
        help=(
            "Comma-separated list of model names. "
            "Overrides yaml `models` field. Single model = single-element list."
        ),
    )
    parser.add_argument(
        "--force-fresh", action="store_true",
        help="Ignore matching existing run dirs; always create a new timestamped dir.",
    )
    parser.add_argument("--budget", type=int, help="Override search budget")
    parser.add_argument(
        "--category",
        help="Override category (harmful_refusal/constrained_completion/sycophancy_resistance)",
    )
    parser.add_argument("--output-dir", help="Override output base directory")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    stop_event = install_two_stage_handlers()

    items, target_k = _resolve_items(args, parser)
    _run_sequential(args, items, target_k, stop_event)


def _run_sequential(
    args: argparse.Namespace,
    items: list[tuple],
    target_k: int,
    stop_event: threading.Event,
) -> None:
    """Execute items in order. Per-iteration log handler isolation +
    inter-iteration sleep for vLLM port release. Failures are logged and
    the loop continues; non-zero exit if any item failed. If stop_event is
    set (Ctrl-C from two-stage handler), abort the remaining items.
    """
    project_logger = logging.getLogger("BoundarySearch")
    n_skip = sum(1 for it in items if it[0] == "skip")
    n_run = len(items) - n_skip
    logger.info(
        "Sequential dispatch: %d items (%d to run, %d to skip)",
        len(items), n_run, n_skip,
    )

    failures: list[tuple] = []
    for i, item in enumerate(items, 1):
        if stop_event.is_set():
            logger.info(
                "Stop event set; skipping remaining %d item(s)",
                len(items) - i + 1,
            )
            break
        kind = item[0]
        per_args = argparse.Namespace(**vars(args))
        per_args.models = None
        handlers_before = list(project_logger.handlers)

        logger.info("=" * 70)
        try:
            if kind == "skip":
                _, run_dir, name, ec = item
                logger.info(
                    "[%d/%d] SKIP %s: %s has %d EC >= target_K=%d",
                    i, len(items), name, run_dir.name, ec, target_k,
                )
            elif kind == "fresh":
                _, name = item
                logger.info("[%d/%d] FRESH %s", i, len(items), name)
                _run_fresh(per_args, model_name=name, stop_event=stop_event)
            elif kind == "resume":
                _, run_dir, name, ec = item
                logger.info(
                    "[%d/%d] RESUME %s from %s (%d/%d EC)",
                    i, len(items), name, run_dir.name, ec, target_k,
                )
                _run_resume(per_args, run_dir, stop_event=stop_event)
        except Exception as e:  # noqa: BLE001
            logger.exception("[%d/%d] %s failed: %s", i, len(items), kind, e)
            failures.append((kind, item, str(e)))
        finally:
            for h in list(project_logger.handlers):
                if h not in handlers_before:
                    project_logger.removeHandler(h)
                    h.close()
            if i < len(items) and kind != "skip" and not stop_event.is_set():
                time.sleep(5)  # let vLLM port fully release

    logger.info("=" * 70)
    logger.info(
        "Sequential dispatch complete: %d skipped, %d run, %d failed",
        n_skip, n_run - len(failures), len(failures),
    )
    if failures:
        logger.warning("Failures:")
        for failed in failures:
            logger.warning("  %s: %s", failed[1], failed[2])
        sys.exit(1)


def _run_fresh(
    args: argparse.Namespace,
    *,
    model_name: str,
    stop_event: threading.Event,
) -> None:
    """Fresh run for a single model. Always creates a new timestamped dir.

    Target-model lifecycle in try/finally: register with signal_utils so the
    hard-exit path can SIGKILL the vllm pgroup; unload_model is invoked from
    the finally branch so vllm shutdown runs on success and on error alike.
    """
    config = load_config(args.config)
    config = apply_overrides(config, args, model_name=model_name)

    exp_cfg = config.get("experiment", {})
    base_dir = exp_cfg.get("output_dir", "results/")
    if exp_cfg.get("name"):
        exp_name = exp_cfg["name"]
    else:
        category = config.get("bare_requests", {}).get("category", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{_model_short(model_name)}_{category}_{timestamp}"
    output_dir = Path(base_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(str(output_dir / "search.log"))

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.info("Output directory: %s", output_dir)

    bare_requests = load_bare_requests(config)
    skills_cfg = config.get("skills", {})
    skills = load_skills(
        skills_cfg.get("dictionary_path", "skills/skill_dictionary.json"),
        category=skills_cfg.get("category"),
    )

    target_model = create_target_model(config.get("target_model", {}))
    set_active_model(target_model)
    try:
        category = config.get("bare_requests", {}).get("category", "harmful_refusal")
        judge = create_judge(config.get("judge", {}), category=category)

        sgbs = create_sgbs_runner(config)
        result = sgbs.run(
            target_model=target_model,
            judge=judge,
            bare_requests=bare_requests,
            skills=skills,
            output_dir=str(output_dir),
            stop_event=stop_event,
        )

        save_results(result, str(output_dir), config=config)

        print(f"\nSearch complete: {result.stats}")
        print(f"Results saved to: {output_dir}")
    finally:
        clear_active_model()
        unload_model(target_model)


def _run_resume(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    stop_event: threading.Event,
) -> None:
    """Resume from an existing run dir. Reuses saved config (target_model,
    skills, judge); only budget/category/output_dir from CLI override.
    """
    prev = load_search_state(str(output_dir))

    config = prev["config"]
    config = apply_overrides(config, args, model_name=None)

    add_file_handler(str(output_dir / "search.log"))

    prev_model = prev["prev_model_name"]
    curr_model = config.get("target_model", {}).get("name")
    if prev_model and curr_model and prev_model != curr_model:
        logger.warning(
            "Model mismatch: previous=%s, current=%s", prev_model, curr_model,
        )

    prev_round = prev["state"].round_num
    budget = config.get("search", {}).get("budget", 100)
    if budget <= prev_round:
        logger.warning(
            "Budget (%d) <= completed rounds (%d), nothing to do for %s",
            budget, prev_round, output_dir.name,
        )
        return

    logger.info(
        "Resuming from %s: round %d -> budget %d",
        output_dir, prev_round, budget,
    )

    bare_requests = load_bare_requests(config)
    skills_cfg = config.get("skills", {})
    skills = load_skills(
        skills_cfg.get("dictionary_path", "skills/skill_dictionary.json"),
        category=skills_cfg.get("category"),
    )

    target_model = create_target_model(config.get("target_model", {}))
    set_active_model(target_model)
    try:
        category = config.get("bare_requests", {}).get("category", "harmful_refusal")
        judge = create_judge(config.get("judge", {}), category=category)

        sgbs = create_sgbs_runner(config)
        result = sgbs.run(
            target_model=target_model,
            judge=judge,
            bare_requests=bare_requests,
            skills=skills,
            initial_state=prev["state"],
            initial_rng_state=prev["rng_state"],
            initial_np_rng_state=prev["np_rng_state"],
            initial_skipped=prev["skipped"],
            initial_round_records=prev["round_records"],
            prev_elapsed=prev["prev_stats"].get("total_time_seconds", 0),
            output_dir=str(output_dir),
            stop_event=stop_event,
        )

        save_results(result, str(output_dir), config=config)

        print(f"\nSearch complete (resumed): {result.stats}")
        print(f"Results saved to: {output_dir}")
    finally:
        clear_active_model()
        unload_model(target_model)


if __name__ == "__main__":
    main()
