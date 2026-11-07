"""Algorithm 2 (paper App. D.2): adaptive evaluation of a target model on the
calibrated logit scale.

Pilot → Fisher-info CAT loop with EAP / MAP+Laplace θ updates; termination
on θ̂ stability (|Δθ̂_t| < ε for K consecutive iters, gated by t ≥ t_min).
Coverage trigger fires when θ̂ drifts outside the panel θ̂ range (extended
by τ_pad), gated by the same iteration guard and L-hysteresis. On trigger,
Step-2 (`adaptive.calibrate.step2_calibrate`) runs SGBS for K new items,
evaluates the panel + target on those items, fits per-item β̂ via 1-D MLE,
and re-runs EAP over the union (anchor + new). The loop then exits with
stopped_reason='step2_done'.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from logger import get_logger
from core.panel_inference import run_jobs_on_model

from .anchor_bundle import AnchorBundle, AnchorItem
from .cat_select import fisher_info_select, pilot_indices
from .coverage import CoverageState, coverage_failed, should_trigger
from .posterior import PosteriorEstimator
from .calibrate import step2_calibrate, Step2Result


logger = get_logger(__name__)


@dataclass
class AdaptiveStep:
    t: int
    item_id: str
    beta_x: float
    theta_hat_pre: float
    theta_se_pre: float
    coverage_failed: bool
    consecutive_failures: int
    triggered: bool
    n_trials: int
    n_pass: int
    p_hat: float
    theta_hat_post: float
    theta_se_post: float


@dataclass
class AdaptiveResult:
    target_model: str
    category: str
    estimator_name: str
    theta_hat_final: float
    theta_se_final: float
    converged: bool
    stopped_reason: str                  # "stability" | "max_iters" | "step2_done" | "step2_failed" | "anchor_exhausted" | "interrupted"
    history: list[AdaptiveStep] = field(default_factory=list)
    pilot_item_ids: list[str] = field(default_factory=list)
    panel_theta: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    step2: dict[str, Any] | None = None  # Step2Result.to_dict() if Step-2 fired


def _run_n_trials_on_items(
    target_model,
    judge,
    items: list[AnchorItem],
    N: int,
    seed: int,
    target_name: str,
    max_workers: int,
    stop_event=None,
) -> tuple[dict[str, tuple[int, int]], list[dict]]:
    """Run N trials per item against (target_model, judge) via the panel FSM.

    Wraps `core.panel_inference.run_jobs_on_model` so adaptive Step-1 reuses
    the same model/judge two-pool concurrent runner that panel inference and
    Step-2 use — no progressive filtering, no range gates, just N trials per
    item.

    Returns (summary, raw_rows):
      summary  = {item_id: (n_pass, n_valid)} — kept for the existing
                 inference/posterior-update code path.
      raw_rows = the runner's per-item result list (each entry has
                 `prompt_id`, `model`, `trials` [N dicts with seed/response/
                 passed/judge_outcome/per_turn_meta/...], plus aggregated
                 `pass_count`/`total_trials`). Returned alongside so the
                 caller can persist trial-level detail.
    """
    panel_items = [
        {"prompt": it.prompt, "prompt_id": it.item_id, "metadata": it.metadata}
        for it in items
    ]
    results = run_jobs_on_model(
        items=panel_items,
        model=target_model,
        judge=judge,
        n_trials=N,
        seed=seed,
        max_workers=max_workers,
        model_name=target_name,
        stop_event=stop_event,
    )
    summary = {r["prompt_id"]: (r["pass_count"], r["total_trials"]) for r in results}
    return summary, results


def run_adaptive_eval(
    target_model,
    judge,
    anchor_bundle: AnchorBundle,
    estimator: PosteriorEstimator,
    config: dict,
    output_dir: str,
    stop_event=None,
) -> AdaptiveResult:
    """Run Algorithm 2 (Step 1 + Step 2 trigger).

    Required config keys (with defaults):
        N                       (default: bundle.N)
        pilot_quantiles         (default: [0.10, 0.30, 0.50, 0.70, 0.90])
        stab_eps                (default: 0.02) — |Δθ̂_t| stability threshold
        stab_K                  (default: 3)    — consecutive stable iters required
        tau_pad                 (default: 0.5)  — coverage pad on panel θ̂ range
        t_min                   (default: 2)    — iteration guard before any termination/trigger
        hysteresis_L            (default: 2)    — consecutive coverage failures required
        max_iters               (default: 50)
        concurrent              (default: True)
        trial_seed              (default: 42)
        estimator_name          (for logging only)
        step1_workers           (default: N if concurrent else 1)
                                — pool size for the Step-1 trial runner.
    """
    N: int = int(config.get("N", anchor_bundle.N))
    pilot_quantiles = config.get("pilot_quantiles", [0.10, 0.30, 0.50, 0.70, 0.90])
    stab_eps: float = float(config.get("stab_eps", 0.02))
    stab_K: int = int(config.get("stab_K", 3))
    tau_pad: float = float(config.get("tau_pad", 0.5))
    t_min: int = int(config.get("t_min", 2))
    hysteresis_L: int = int(config.get("hysteresis_L", 2))
    max_iters: int = int(config.get("max_iters", 50))
    concurrent: bool = bool(config.get("concurrent", True))
    trial_seed: int = int(config.get("trial_seed", 42))
    estimator_name: str = str(config.get("estimator_name", "?"))
    step1_workers: int = int(config.get("step1_workers", N) if concurrent else 1)

    panel_theta_values = list(anchor_bundle.panel_theta.values())
    if not panel_theta_values:
        raise ValueError("anchor_bundle.panel_theta is empty; coverage trigger needs the panel θ̂ range")
    panel_theta_min = float(min(panel_theta_values))
    panel_theta_max = float(max(panel_theta_values))

    target_name = getattr(target_model, "name", "<unknown>")
    items = anchor_bundle.items
    J = len(items)
    beta = np.array([it.beta for it in items], dtype=np.float64)
    used = np.zeros(J, dtype=bool)
    pass_counts = np.zeros(J, dtype=np.float64)
    trial_counts = np.zeros(J, dtype=np.float64)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history_path = output_path / "history.jsonl"
    history_fp = open(history_path, "w")
    # trials.jsonl: one row per item-evaluation (1 row × pilot_size from the
    # pilot batch, then 1 row per main-loop iter). Each row embeds all N
    # trials with the full schema from core.trial_utils.build_trial
    # (seed, response, passed, judge_outcome, judge_label, per_turn_meta,
    # completion_tokens, ...). Flushed after every write so a kill mid-run
    # leaves a usable partial file.
    trials_path = output_path / "trials.jsonl"
    trials_fp = open(trials_path, "w")

    def _write_trial_rows(iter_tag, beta_lookup, raw_rows):
        for r in raw_rows:
            row = {
                "iter": iter_tag,
                "item_id": r["prompt_id"],
                "beta": beta_lookup.get(r["prompt_id"]),
                "model": r["model"],
                "pass_count": r.get("pass_count"),
                "total_trials": r.get("total_trials"),
                "trials": r.get("trials", []),
            }
            trials_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        trials_fp.flush()

    history: list[AdaptiveStep] = []

    # ------------------------------------------------------------------
    # Pilot: pilot_size items at the prescribed β̂ quantiles
    # ------------------------------------------------------------------
    pilot_idx = pilot_indices(beta, pilot_quantiles)
    pilot_ids: list[str] = []
    logger.info("Pilot (n=%d, quantiles=%s):", len(pilot_idx), pilot_quantiles)
    pilot_items_obj = [items[j] for j in pilot_idx]
    pilot_trial_results, pilot_raw_rows = _run_n_trials_on_items(
        target_model=target_model, judge=judge,
        items=pilot_items_obj, N=N, seed=trial_seed,
        target_name=target_name, max_workers=step1_workers,
        stop_event=stop_event,
    )
    _write_trial_rows(
        iter_tag="pilot",
        beta_lookup={it.item_id: float(it.beta) for it in pilot_items_obj},
        raw_rows=pilot_raw_rows,
    )
    for j in pilot_idx:
        item = items[j]
        n_pass, n_valid = pilot_trial_results[item.item_id]
        used[j] = True
        pass_counts[j] = n_pass
        trial_counts[j] = n_valid
        pilot_ids.append(item.item_id)
        logger.info("  pilot j=%-3d  β=%+.3f  k=%2d/%-2d  p̂=%.3f  %s",
                    j, beta[j], n_pass, n_valid, n_pass / max(n_valid, 1), item.item_id)

    # Initialize θ̂ from pilot
    theta_hat, theta_se = estimator.estimate(
        pass_counts[used], trial_counts[used], beta[used],
    )
    logger.info("Post-pilot: θ̂=%+.3f  SE=%.3f  (n_used=%d)", theta_hat, theta_se, used.sum())

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    cov = CoverageState()
    delta_history: deque[float] = deque(maxlen=stab_K)
    stopped_reason = "max_iters"
    converged = False
    step2_payload: dict[str, Any] | None = None

    # If stop_event was set during pilot (panel_inference drained gracefully
    # and returned), skip the main loop — otherwise we'd dispatch one more
    # CAT-selected trial before the in-loop check fires.
    skip_main_loop = stop_event is not None and stop_event.is_set()
    if skip_main_loop:
        logger.info("Stop event set during pilot; skipping main loop")
        stopped_reason = "interrupted"

    for t in range(1, max_iters + 1) if not skip_main_loop else []:
        if used.all():
            stopped_reason = "anchor_exhausted"
            break

        # Pick next item
        x_idx = fisher_info_select(theta_hat, beta, used, N=N)
        item = items[x_idx]
        beta_x = float(beta[x_idx])

        # Coverage check: θ̂ outside panel θ̂ range (extended by τ_pad)
        failed = coverage_failed(theta_hat, panel_theta_min, panel_theta_max, tau_pad)
        cov.record(failed)
        triggered = False
        if failed and should_trigger(t, t_min, cov.consecutive_failures, hysteresis_L):
            logger.warning(
                "[t=%d] coverage trigger fired (θ̂=%.2f outside [%.2f, %.2f] ± %.2f); "
                "invoking Step-2 calibration",
                t, theta_hat, panel_theta_min, panel_theta_max, tau_pad,
            )
            triggered = True
            try:
                step2_res: Step2Result = step2_calibrate(
                    target_model=target_model,
                    judge=judge,
                    anchor_bundle=anchor_bundle,
                    theta_hat=theta_hat,
                    pass_counts=pass_counts,
                    trial_counts=trial_counts,
                    used=used,
                    estimator=estimator,
                    config=config,
                    output_dir=output_dir,
                    stop_event=stop_event,
                )
            except Exception as e:
                logger.exception(
                    "[t=%d] Step-2 calibration failed: %s; recording trigger and exiting", t, e,
                )
                if stop_event is not None and stop_event.is_set():
                    # Step-2 raised because panel inference was drained mid-run
                    # (e.g. ctrl-c) — record as interrupted, not algorithmic failure.
                    stopped_reason = "interrupted"
                else:
                    stopped_reason = "step2_failed"
                step = AdaptiveStep(
                    t=t, item_id=item.item_id, beta_x=beta_x,
                    theta_hat_pre=theta_hat, theta_se_pre=theta_se,
                    coverage_failed=True, consecutive_failures=cov.consecutive_failures,
                    triggered=True, n_trials=0, n_pass=0, p_hat=float("nan"),
                    theta_hat_post=theta_hat, theta_se_post=theta_se,
                )
                history.append(step)
                history_fp.write(json.dumps(asdict(step)) + "\n")
                break

            # Step-2 succeeded: post-trigger θ̂ comes from EAP over union.
            theta_hat = step2_res.theta_hat_new
            theta_se = step2_res.theta_se_new
            if stop_event is not None and stop_event.is_set():
                # Step-2 drained internally but the run was interrupted —
                # don't claim "done", record as interrupted.
                stopped_reason = "interrupted"
            else:
                stopped_reason = "step2_done"
            # Step-2 by design re-fits θ̂ over a richer cell pool; treat it
            # as a converged exit (the trigger replaced the in-loop convergence test).
            converged = True
            step2_payload = step2_res.to_dict()
            step = AdaptiveStep(
                t=t, item_id=item.item_id, beta_x=beta_x,
                theta_hat_pre=step2_payload["theta_hat_new"],  # post-trigger value
                theta_se_pre=step2_payload["theta_se_new"],
                coverage_failed=True, consecutive_failures=cov.consecutive_failures,
                triggered=True, n_trials=0, n_pass=0, p_hat=float("nan"),
                theta_hat_post=theta_hat, theta_se_post=theta_se,
            )
            history.append(step)
            history_fp.write(json.dumps(asdict(step)) + "\n"); history_fp.flush()
            logger.info(
                "[t=%d] Step-2 done: %d new items (β̂ panel-fitted), "
                "θ̂_new=%+.3f SE=%.3f over %d anchor + %d new items",
                t, len(step2_res.new_items),
                theta_hat, theta_se,
                int(used.sum()), len(step2_res.new_items),
            )
            break

        # Step 1: run N trials and update θ̂. Per-iter seed offset prevents
        # the same N trial seeds being reused across iters (vllm temp=1.0
        # + identical seeds collapses methodological independence).
        trial_results, raw_rows = _run_n_trials_on_items(
            target_model=target_model, judge=judge,
            items=[item], N=N, seed=trial_seed + t,
            target_name=target_name, max_workers=step1_workers,
            stop_event=stop_event,
        )
        _write_trial_rows(
            iter_tag=t,
            beta_lookup={item.item_id: beta_x},
            raw_rows=raw_rows,
        )
        n_pass, n_valid = trial_results[item.item_id]
        used[x_idx] = True
        pass_counts[x_idx] = n_pass
        trial_counts[x_idx] = n_valid

        theta_hat_post, theta_se_post = estimator.estimate(
            pass_counts[used], trial_counts[used], beta[used],
        )

        step = AdaptiveStep(
            t=t, item_id=item.item_id, beta_x=beta_x,
            theta_hat_pre=theta_hat, theta_se_pre=theta_se,
            coverage_failed=failed, consecutive_failures=cov.consecutive_failures,
            triggered=False, n_trials=n_valid, n_pass=n_pass,
            p_hat=(n_pass / n_valid if n_valid else float("nan")),
            theta_hat_post=theta_hat_post, theta_se_post=theta_se_post,
        )
        history.append(step)
        history_fp.write(json.dumps(asdict(step)) + "\n"); history_fp.flush()
        logger.info("[t=%2d] x=%s  β=%+.3f  k=%2d/%-2d  p̂=%.3f  θ̂: %+.3f→%+.3f  SE: %.3f→%.3f%s",
                    t, item.item_id, beta_x, n_pass, n_valid,
                    step.p_hat, theta_hat, theta_hat_post,
                    theta_se, theta_se_post,
                    "  [cov-fail]" if failed else "")

        theta_hat, theta_se = theta_hat_post, theta_se_post

        if stop_event is not None and stop_event.is_set():
            logger.info(
                "Stop event set; exiting at iter t=%d (history.jsonl flushed)",
                t,
            )
            stopped_reason = "interrupted"
            break

        # Stability termination (T2c, paper App. D.2): exit when |Δθ̂_t|
        # has stayed below stab_eps for stab_K consecutive iters, gated by
        # t ≥ t_min so the pilot doesn't trivially satisfy it. Use the
        # pre/post values from the AdaptiveStep we just recorded — by this
        # point `theta_hat` has already been reassigned to theta_hat_post.
        delta_history.append(abs(step.theta_hat_post - step.theta_hat_pre))
        if t >= t_min and len(delta_history) >= stab_K \
                and all(d < stab_eps for d in delta_history):
            stopped_reason = "stability"
            converged = True
            break

    history_fp.close()
    trials_fp.close()

    result = AdaptiveResult(
        target_model=target_name,
        category=anchor_bundle.category,
        estimator_name=estimator_name,
        theta_hat_final=theta_hat,
        theta_se_final=theta_se,
        converged=converged,
        stopped_reason=stopped_reason,
        history=history,
        pilot_item_ids=pilot_ids,
        panel_theta=dict(anchor_bundle.panel_theta),
        config={
            "N": N, "pilot_quantiles": pilot_quantiles,
            "stab_eps": stab_eps, "stab_K": stab_K, "tau_pad": tau_pad,
            "panel_theta_min": panel_theta_min, "panel_theta_max": panel_theta_max,
            "t_min": t_min, "hysteresis_L": hysteresis_L,
            "max_iters": max_iters, "trial_seed": trial_seed,
            "estimator_name": estimator_name,
            "step1_workers": step1_workers,
        },
        step2=step2_payload,
    )
    save_adaptive_result(result, output_dir)
    return result


def save_adaptive_result(result: AdaptiveResult, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "result.json", "w") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
