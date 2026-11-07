"""Entry point for adaptive evaluation on the calibrated logit scale.

This is Algorithm 2 of the paper (App. D.2): Fisher-info CAT on the panel
anchor set, with optional Step-2 SGBS routing when the target model's θ̂
falls outside the panel's θ̂ range. Step 2 performs SGBS and panel calibration; see adaptive/calibrate.py.

For pure SGBS runs (per-panel-model boundary set generation, etc.), use
scripts/run_sgbs.py.

Per-model dispatch is idempotent: for each name in the model list, the
script looks under experiment.output_dir for an existing matching run dir
({model_short}_{category}_{estimator}_<timestamp>) and decides:
  - skip:  matching dir contains a complete result.json
  - fresh: no matching dir, or matching dir lacks result.json
--force-fresh creates a new timestamped directory. An interrupted adaptive
run starts again on the next invocation.

Usage:
    # Single model
    python main.py \\
        --config configs/harmful_refusal/adaptive.yaml \\
        --anchor-bundle benchmarks/harmful_refusal/anchor_alo_bundle_M9.json \\
        --models deepseek-ai/DeepSeek-R1-Distill-Llama-70B

    # Multiple models, sequential
    python main.py \\
        --config configs/harmful_refusal/adaptive.yaml \\
        --anchor-bundle benchmarks/harmful_refusal/anchor_alo_bundle_M9.json \\
        --models n1,n2,n3
"""

import argparse
import copy
import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

from logger import setup_logging, add_file_handler, get_logger
from models import create_target_model, unload_model
from judges import create_judge

from adaptive.anchor_bundle import load_anchor_bundle
from adaptive.posterior import build_estimator
from adaptive.adaptive_eval import run_adaptive_eval
from core.model_dispatch import (
    resolve_target_model,
    model_short,
    find_existing_run_dir,
)
from core.signal_utils import (
    install_two_stage_handlers,
    set_active_model,
    clear_active_model,
)


logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Per-model dispatch
# ---------------------------------------------------------------------------

def _resolve_model_names(args: argparse.Namespace, yaml_cfg: dict) -> list[str]:
    """CLI > yaml.models > legacy yaml.target_model.name (single)."""
    cli_models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models else None
    )
    yaml_models = yaml_cfg.get("models")
    if not yaml_models:
        legacy_name = (yaml_cfg.get("target_model") or {}).get("name")
        if legacy_name:
            yaml_models = [legacy_name]
    return cli_models or yaml_models or []


# A run is considered "done" only if its terminal state is one of these.
# step2_failed / interrupted / missing → not done, eligible for reuse.
_COMPLETED_STOPPED_REASONS = frozenset({
    "stability", "max_iters", "anchor_exhausted", "step2_done",
})


def _completion_state(run_dir: Path) -> tuple[str, str | None]:
    """Inspect a run dir and classify it.

    Returns (state, stopped_reason) where state ∈ {"done", "partial", "empty"}.
      done    : result.json present with terminal stopped_reason
      partial : result.json missing or stopped_reason not terminal,
                AND there's some incremental data (history / step2 dirs)
      empty   : nothing useful to resume from
    """
    result_path = run_dir / "result.json"
    if result_path.exists():
        try:
            with open(result_path) as f:
                payload = json.load(f)
            sr = payload.get("stopped_reason")
            if sr in _COMPLETED_STOPPED_REASONS:
                return "done", sr
            return "partial", sr
        except (json.JSONDecodeError, OSError):
            pass  # fall through to incremental-data check
    has_partial = (
        (run_dir / "history.jsonl").exists()
        or (run_dir / "step2_sgbs").is_dir()
        or (run_dir / "step2_panel").is_dir()
    )
    return ("partial" if has_partial else "empty"), None


def _resolve_items(
    args: argparse.Namespace,
    yaml_cfg: dict,
    bundle_category: str,
    estimator_name: str,
) -> list[tuple]:
    """Per-model dispatch decisions:
        ("skip",  matched_dir, name)
        ("reuse", matched_dir, name, prior_stopped_reason | None)
        ("fresh", name)
    """
    names = _resolve_model_names(args, yaml_cfg)
    if not names:
        sys.exit(
            "no models to run: specify --models, yaml `models`, or "
            "yaml target_model.name"
        )

    exp_cfg = yaml_cfg.get("experiment") or {}
    base_dir = Path(args.output_dir or exp_cfg.get("output_dir", "results/"))

    items: list[tuple] = []
    for name in names:
        if args.force_fresh:
            items.append(("fresh", name))
            continue
        prefix = f"{model_short(name)}_{bundle_category}_{estimator_name}_"
        matched = find_existing_run_dir(base_dir, prefix)
        if matched is None:
            items.append(("fresh", name))
            continue
        state, prior_sr = _completion_state(matched)
        if state == "done":
            items.append(("skip", matched, name))
        elif state == "partial":
            items.append(("reuse", matched, name, prior_sr))
        else:
            items.append(("fresh", name))
    return items


def _run_one(
    args: argparse.Namespace,
    yaml_cfg: dict,
    bundle,
    estimator_name: str,
    model_name: str,
    stop_event: threading.Event,
    *,
    reuse_dir: Path | None = None,
) -> None:
    """Run adaptive evaluation for a single target model.

    Handles target_model lifecycle in a try/finally so vllm shutdown is
    deterministic: register with signal_utils for hard-exit reachability,
    transparent shutdown via unload_model on the way out (success or error).

    reuse_dir: if given, runs into that existing directory instead of a
    fresh-timestamped one. adaptive_eval itself does not yet resume Step 1
    (history.jsonl is overwritten), but Step 2 panel inference resumes
    automatically by reading the existing step2_panel/<model>.jsonl files.
    """
    config = copy.deepcopy(yaml_cfg)
    # CLI overrides on yaml.target_model BEFORE resolve_target_model so
    # panel_config still merges normally and TP-consistency check sees
    # the overridden cuda_devices.
    if args.cuda_devices:
        config.setdefault("target_model", {})["cuda_devices"] = args.cuda_devices
    config["target_model"] = resolve_target_model(
        config.get("target_model", {}) or {}, model_name,
    )
    if args.estimator:
        config.setdefault("adaptive", {})["estimator"] = args.estimator

    if reuse_dir is not None:
        output_dir = reuse_dir
    else:
        exp_cfg = config.get("experiment", {}) or {}
        base_dir = Path(args.output_dir or exp_cfg.get("output_dir", "results/"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = (
            exp_cfg.get("name")
            or f"{model_short(model_name)}_{bundle.category}_{estimator_name}_{timestamp}"
        )
        output_dir = base_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(str(output_dir / "adaptive.log"))

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.info("Output directory: %s", output_dir)
    logger.info("Target model: %s   Estimator: %s", model_name, estimator_name)

    target_model = create_target_model(config.get("target_model", {}))
    set_active_model(target_model)
    try:
        judge = create_judge(config.get("judge", {}), category=bundle.category)

        adaptive_cfg = dict(config.get("adaptive", {}))
        adaptive_cfg.setdefault("estimator_name", estimator_name)
        adaptive_cfg["step2"] = config.get("step2", {}) or {}
        estimator = build_estimator(estimator_name, adaptive_cfg)

        result = run_adaptive_eval(
            target_model=target_model,
            judge=judge,
            anchor_bundle=bundle,
            estimator=estimator,
            config=adaptive_cfg,
            output_dir=str(output_dir),
            stop_event=stop_event,
        )

        print()
        print(f"Adaptive evaluation complete  ({result.stopped_reason})")
        print(f"  target            : {result.target_model}")
        print(f"  category          : {result.category}")
        print(f"  estimator         : {result.estimator_name}")
        print(f"  iterations used   : pilot={len(result.pilot_item_ids)} + "
              f"loop={len([s for s in result.history if not s.triggered])}")
        print(f"  θ̂ final          : {result.theta_hat_final:+.3f}")
        print(f"  SE final          : {result.theta_se_final:.3f}")
        print(f"  converged         : {result.converged}")
        if result.step2:
            print(f"  step2 fired       : K={len(result.step2['new_items'])} "
                  f"survivors (dropped {result.step2['n_dropped_saturated']}), "
                  f"panel inferences={result.step2['n_panel_inferences']}")
        print(f"  results dir       : {output_dir}")
    finally:
        clear_active_model()
        unload_model(target_model)


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive evaluation on calibrated logit scale (Algorithm 2)",
    )
    parser.add_argument("--config", required=True, help="YAML config (target_model, judge, adaptive)")
    parser.add_argument("--anchor-bundle", required=True, help="Path to anchor_alo_bundle_M9.json")
    parser.add_argument(
        "--models",
        help=(
            "Comma-separated model names. Overrides yaml `models` or "
            "target_model.name. Single model = single-element list."
        ),
    )
    parser.add_argument("--estimator", choices=["eap", "laplace"], default=None,
                        help="Override adaptive.estimator (eap or laplace)")
    parser.add_argument("--output-dir", help="Override output base directory")
    parser.add_argument(
        "--cuda-devices",
        help="Override yaml target_model.cuda_devices (comma-separated GPU "
             "ids, e.g. '0,1'). Applied to all dispatched models in the run; "
             "panel_config's extra_args/TP are still authoritative, so the "
             "GPU count must match --tensor-parallel-size.",
    )
    parser.add_argument(
        "--force-fresh", action="store_true",
        help="Ignore existing complete run dirs; always create a new timestamped dir.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    stop_event = install_two_stage_handlers()

    # ------------------------------------------------------------------
    # Load config + bundle (bundle is shared across all target models)
    # ------------------------------------------------------------------
    yaml_cfg = load_config(args.config) or {}
    bundle = load_anchor_bundle(args.anchor_bundle)
    logger.info("Anchor bundle: category=%s, M=%d, J=%d, N=%d, filter=%s",
                bundle.category, bundle.M, bundle.J, bundle.N, bundle.filter_spec)

    estimator_name = (
        args.estimator
        or (yaml_cfg.get("adaptive") or {}).get("estimator", "eap")
    )

    # ------------------------------------------------------------------
    # Per-model dispatch
    # ------------------------------------------------------------------
    items = _resolve_items(args, yaml_cfg, bundle.category, estimator_name)
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
        handlers_before = list(project_logger.handlers)
        logger.info("=" * 70)
        try:
            if kind == "skip":
                _, run_dir, name = item
                logger.info(
                    "[%d/%d] SKIP %s: %s already complete",
                    i, len(items), name, run_dir.name,
                )
            elif kind == "reuse":
                _, run_dir, name, prior_sr = item
                logger.info(
                    "[%d/%d] REUSE %s in %s (prior stopped_reason=%s); "
                    "Step-1 will rerun, Step-2 panel inference resumes from "
                    "existing step2_panel/*.jsonl",
                    i, len(items), name, run_dir.name, prior_sr,
                )
                _run_one(args, yaml_cfg, bundle, estimator_name, name,
                         stop_event, reuse_dir=run_dir)
            else:
                _, name = item
                logger.info("[%d/%d] FRESH %s", i, len(items), name)
                _run_one(args, yaml_cfg, bundle, estimator_name, name, stop_event)
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
        for f in failures:
            logger.warning("  %s: %s", f[1], f[2])
        sys.exit(1)


if __name__ == "__main__":
    main()
