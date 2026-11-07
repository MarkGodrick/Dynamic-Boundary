"""Step-2 calibration after a triggered SGBS run (paper App. D.2 §Step 2).

Trigger path: adaptive eval detects coverage failure → calls step2_calibrate
with the current state. Pipeline:

  (1) SGBS sub-loop on m_new → K boundary items E_new (m_new responses included).
  (2) unload m_new (free GPU).
  (3) Sequential panel inference: each PANEL_MODEL evaluates E_new × N trials.
  (4) 1-D MLE β̂_x per new item with θ̂_panel frozen
      (saturated items, where Σ_m k_m ∈ {0, Σ_m n_m}, are dropped).
  (5) EAP θ̂_new over X_anchor ∪ E_new with β̂_anchor + β̂_new frozen.

Single-trigger semantics: caller terminates the adaptive loop after this
returns (stopped_reason="step2_done"). m_new is *not* restarted.

K defaults to 50 and is configured by cfg["step2"]["K"].
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import brentq

from logger import get_logger

from .anchor_bundle import AnchorBundle, AnchorItem


logger = get_logger(__name__)


# Compatibility exception for callers of earlier versions.
class Step2NotImplemented(NotImplementedError):
    """Compatibility exception retained for existing imports."""


@dataclass
class Step2Result:
    """Outcome of one Step-2 trigger.

    Caller (run_adaptive_eval) writes this into AdaptiveResult and exits the
    main loop with stopped_reason='step2_done'.
    """
    new_items: list[AnchorItem]      # E_new survivors with calibrated β̂_x
    new_pass_counts: list[int]       # m_new pass counts on each survivor (from SGBS)
    new_trial_counts: list[int]      # m_new trial counts (from SGBS)
    n_panel_inferences: int          # ΣΣ n_m over surviving items × M panel models
    n_dropped_saturated: int         # E_new entries discarded for panel saturation
    theta_hat_new: float             # EAP over X_anchor ∪ E_new
    theta_se_new: float
    sgbs_sub_dir: str
    panel_responses_path: str        # dir containing per-panel-model jsonl matrices

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["new_items"] = [asdict(it) for it in self.new_items]
        return d


# ---------------------------------------------------------------------------
# 1-D MLE β̂ for a single Rasch item with frozen panel θ
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _fit_beta_1d(
    panel_k: np.ndarray, panel_n: np.ndarray, panel_theta: np.ndarray,
) -> float | None:
    """1-D MLE for β in Rasch model given M panel cells with frozen θ_m.

    Score equation (monotone decreasing in β):
        Σ_m n_m σ(θ_m - β) - Σ_m k_m == 0

    Returns None if the data is saturated (Σk = 0 or Σk = Σn) — β̂ then
    diverges to ±∞ and is unidentifiable.
    """
    sum_k = float(panel_k.sum())
    sum_n = float(panel_n.sum())
    if sum_k == 0.0 or sum_k == sum_n:
        return None  # saturation → β̂ diverges

    def f(beta: float) -> float:
        return float((panel_n * _sigmoid(panel_theta - beta)).sum() - sum_k)

    # f is monotone decreasing in β. Search a wide bracket; if even ±30 logit
    # doesn't bracket a root, treat as effectively saturated.
    lo, hi = -30.0, 30.0
    if f(lo) <= 0 or f(hi) >= 0:
        return None
    return float(brentq(f, lo, hi, xtol=1e-6))


# ---------------------------------------------------------------------------
# (1) SGBS sub-loop on m_new
# ---------------------------------------------------------------------------

def _run_sgbs_subloop(
    target_model, judge, bundle: AnchorBundle, K: int,
    sgbs_cfg: dict, output_dir: Path, stop_event=None,
) -> tuple[list, str]:
    """Run a single SGBS run on m_new with target_edge_cases = K.

    Returns (edge_cases, sub_dir_str). edge_cases: list[EdgeCase].
    """
    from search import SGBSv2, BareRequest
    from search.skill_selector import load_skills

    # Don't mutate caller's config; override target_edge_cases for this run.
    sub_cfg = json.loads(json.dumps(sgbs_cfg))  # deep copy via json
    sub_cfg.setdefault("search", {})["target_edge_cases"] = K

    skills_cfg = sub_cfg.get("skills", {}) or {}
    skills = load_skills(
        skills_cfg.get("dictionary_path", "skills/skill_dictionary.json"),
        category=skills_cfg.get("category", bundle.category),
    )

    br_cfg = sub_cfg.get("bare_requests", {}) or {}
    data_path = br_cfg.get("data_path")
    if not data_path:
        raise ValueError(
            "step2: sgbs_config.bare_requests.data_path is required"
        )
    bare_requests = []
    with open(data_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            bare_requests.append(BareRequest(
                prompt=entry.get("prompt", entry.get("request", "")),
                metadata=entry.get("metadata", {}),
            ))
    limit = br_cfg.get("limit")
    if limit:
        bare_requests = bare_requests[:limit]
    logger.info("Step2 SGBS: %d bare_requests, K=%d", len(bare_requests), K)

    sub_dir = output_dir / "step2_sgbs"
    sub_dir.mkdir(parents=True, exist_ok=True)

    sgbs = SGBSv2(sub_cfg)
    result = sgbs.run(
        target_model=target_model,
        judge=judge,
        bare_requests=bare_requests,
        skills=skills,
        output_dir=str(sub_dir),
        stop_event=stop_event,
    )
    return list(result.edge_cases), str(sub_dir)


# ---------------------------------------------------------------------------
# (3) Panel inference on E_new
# ---------------------------------------------------------------------------

def _panel_evaluate(
    items: list[dict], panel_models: list[dict], judge,
    N: int, max_tokens: int, panel_workers: int,
    output_dir: Path, category: str, stop_event=None,
) -> dict[str, dict[str, tuple[int, int]]]:
    """Sequentially evaluate every panel model on `items`.

    Returns {model_name: {prompt_id: (pass_count, total_trials)}}. Models
    that fail to start contribute no entries; saturated items will then be
    flagged downstream (missing-cell branch in step2_calibrate).
    """
    from core.panel_inference import run_jobs_on_model
    from core.inference_budgets import get_max_model_len
    from core.signal_utils import set_active_model, clear_active_model
    from models.vllm_model import VLLMModel

    panel_dir = output_dir / "step2_panel"
    panel_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, dict[str, tuple[int, int]]] = {}
    expected_n_jobs = len(items) * N
    for i, mcfg in enumerate(panel_models, 1):
        if stop_event is not None and stop_event.is_set():
            logger.info(
                "Stop event set; aborting panel inference at model %d/%d",
                i, len(panel_models),
            )
            break
        name = mcfg["name"]
        short = name.replace("/", "_").replace("-", "_")
        out_path = panel_dir / f"{short}.jsonl"
        logger.info("=" * 70)
        logger.info("[panel %d/%d] %s", i, len(panel_models), name)

        # Resume: load partial results from prior run if file exists.
        # `run_jobs_on_model` reuses trials whose (prompt_id, seed) match
        # the regenerated seed table — so reusing seed=42 below is critical.
        prior_completed = _load_prior_panel_results(out_path)
        prior_n = sum(len(v) for v in prior_completed.values())
        if prior_completed:
            if prior_n >= expected_n_jobs:
                logger.info(
                    "  ↻ resume: %s already has %d/%d trials → skipping panel start",
                    out_path.name, prior_n, expected_n_jobs,
                )
                # Still build the per-prompt summary for downstream β̂ fit.
                out[name] = _summarize_panel_jsonl(out_path)
                continue
            logger.info(
                "  ↻ resume: %s has %d/%d trials, will continue",
                out_path.name, prior_n, expected_n_jobs,
            )

        try:
            vllm = VLLMModel(
                model_name=name,
                conda_env=mcfg["conda_env"],
                cuda_devices=mcfg["cuda_devices"],
                port=mcfg.get("port", 8100),
                dtype=mcfg.get("dtype", "bfloat16"),
                max_model_len=get_max_model_len(category, name),
                gpu_memory_utilization=mcfg["gpu_mem_util"],
                extra_args=mcfg.get("extra_args", []),
                max_tokens=max_tokens,
            )
        except RuntimeError as e:
            # Fail fast: a partial panel can't calibrate (downstream would
            # drop every E_new item for a missing cell anyway). Earlier
            # panel models' jsonl in step2_panel/ persist for resume.
            raise RuntimeError(
                f"Panel model {name} failed to start: {e}; "
                f"calibration aborted (completed panel jsonls in "
                f"{panel_dir} are kept for resume)."
            ) from e

        set_active_model(vllm)
        try:
            def _save_atomic(rows: list[dict], _path=out_path) -> None:
                tmp = _path.with_suffix(".jsonl.tmp")
                with open(tmp, "w") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                tmp.replace(_path)

            results = run_jobs_on_model(
                items=items, model=vllm, judge=judge, n_trials=N,
                seed=42, max_workers=panel_workers,
                model_name=name,
                prior_completed=prior_completed or None,
                save_callback=_save_atomic,
                save_interval=30.0,
                stop_event=stop_event,
            )
            _save_atomic(results)
            out[name] = {
                r["prompt_id"]: (int(r["pass_count"]), int(r["total_trials"]))
                for r in results
                if r.get("pass_count") is not None
            }
            logger.info("  → saved %d items to %s", len(results), out_path.name)
        finally:
            clear_active_model()
            vllm.shutdown()
            time.sleep(3)  # let vLLM port fully release before next panel model

    return out


def _load_prior_panel_results(path: Path) -> dict[str, list[dict]]:
    """Load a previous panel-inference jsonl into prior_completed format.

    Returns {prompt_id: [trial,...]} suitable for run_jobs_on_model's
    prior_completed kwarg. Drops trial_exception entries so they get retried
    (mirrors online_response_matrix.py's resume policy).
    """
    if not path.exists():
        return {}
    out: dict[str, list[dict]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = row.get("prompt_id")
            trials = row.get("trials") or []
            kept = [
                t for t in trials
                if t.get("judge_outcome") != "trial_exception"
            ]
            if kept:
                out[pid] = kept
    return out


def _summarize_panel_jsonl(path: Path) -> dict[str, tuple[int, int]]:
    """Read a finished panel jsonl into {prompt_id: (pass_count, total_trials)}."""
    out: dict[str, tuple[int, int]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("pass_count") is None:
                continue
            out[row["prompt_id"]] = (int(row["pass_count"]), int(row["total_trials"]))
    return out


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def step2_calibrate(
    *,
    target_model,
    judge,
    anchor_bundle: AnchorBundle,
    theta_hat: float,
    pass_counts: np.ndarray,
    trial_counts: np.ndarray,
    used: np.ndarray,
    estimator,
    config: dict,
    output_dir: str,
    stop_event=None,
) -> Step2Result:
    """Run Step-2: SGBS triggered → panel inference → β̂ fit → EAP θ̂_new.

    Args follow Algorithm 2 (paper App. D.2). `config` reads:
        step2.K                  default 50 — SGBS target_edge_cases
        step2.sgbs_config        path to sgbs yaml (skills + bare_requests)
        step2.panel_workers      default 50 — vllm concurrency per panel model
        step2.max_tokens         default from get_max_tokens(anchor_bundle.category)
        step2.panel_models       optional override of core.panel_config.PANEL_MODELS
    """
    from models import unload_model
    from core.inference_budgets import get_max_tokens
    from core.panel_config import PANEL_MODELS

    step2_cfg = config.get("step2", {}) or {}
    K = int(step2_cfg.get("K", 50))
    sgbs_config_path = step2_cfg.get("sgbs_config")
    if not sgbs_config_path:
        raise ValueError(
            "step2.sgbs_config not set: provide a sgbs yaml path "
            "(needed for skills + bare_requests + composer config)"
        )
    panel_workers = int(step2_cfg.get("panel_workers", 50))
    max_tokens = int(step2_cfg.get(
        "max_tokens",
        get_max_tokens(anchor_bundle.category),
    ))
    panel_models = step2_cfg.get("panel_models") or list(PANEL_MODELS)

    # Fail fast: calibration zero-point is set by the anchor's panel set, so
    # any deviation (subset or extra) makes the new β̂ incomparable to anchor β̂
    # on the same logit scale.
    panel_names_set = {m["name"] for m in panel_models}
    anchor_set = set(anchor_bundle.panel_models)
    if panel_names_set != anchor_set:
        raise RuntimeError(
            f"step2.panel_models must equal anchor_bundle.panel_models "
            f"(missing={sorted(anchor_set - panel_names_set)}, "
            f"extra={sorted(panel_names_set - anchor_set)})"
        )

    with open(sgbs_config_path) as f:
        sgbs_cfg = yaml.safe_load(f) or {}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Step-2 trigger: K=%d, sgbs_config=%s", K, sgbs_config_path)
    logger.info("Pre-trigger θ̂_new = %+.3f", theta_hat)

    # ------------------------------------------------------------------
    # (1) SGBS sub-loop on m_new
    # ------------------------------------------------------------------
    edge_cases, sgbs_sub_dir = _run_sgbs_subloop(
        target_model, judge, anchor_bundle, K, sgbs_cfg, out_dir, stop_event,
    )
    logger.info("SGBS sub-loop returned %d edge cases (target K=%d)",
                len(edge_cases), K)
    if not edge_cases:
        raise RuntimeError(
            "SGBS sub-loop returned 0 edge cases; cannot Step-2 calibrate"
        )

    # Build panel-inference items: stable prompt_id from index + content hash.
    import hashlib
    items: list[dict] = []
    for i, ec in enumerate(edge_cases):
        h = hashlib.sha256(ec.prompt.encode()).hexdigest()[:8]
        pid = f"step2_{i:03d}_{h}"
        items.append({
            "prompt_id": pid,
            "prompt": ec.prompt,
            "metadata": dict(ec.metadata or {}),
        })

    # ------------------------------------------------------------------
    # (2) Free GPU: m_new must shut down before panel models start
    # ------------------------------------------------------------------
    logger.info("Unloading m_new before panel inference")
    unload_model(target_model)
    time.sleep(3)

    # ------------------------------------------------------------------
    # (3) Panel evaluation on E_new (sequential, M=9 models)
    # ------------------------------------------------------------------
    panel_responses = _panel_evaluate(
        items=items, panel_models=panel_models, judge=judge,
        N=anchor_bundle.N, max_tokens=max_tokens,
        panel_workers=panel_workers, output_dir=out_dir,
        category=anchor_bundle.category,
        stop_event=stop_event,
    )

    # ------------------------------------------------------------------
    # (4) 1-D MLE β̂_x per new item, with θ̂_panel frozen
    # ------------------------------------------------------------------
    panel_names = [m["name"] for m in panel_models]
    panel_theta_arr = np.array(
        [anchor_bundle.panel_theta[n] for n in panel_names], dtype=np.float64,
    )

    new_items: list[AnchorItem] = []
    new_pass: list[int] = []
    new_trial: list[int] = []
    n_dropped = 0
    n_panel_inferences = 0
    fit_records: list[dict] = []

    for ec, item in zip(edge_cases, items):
        pid = item["prompt_id"]
        cells_k, cells_n = [], []
        miss = []
        for name in panel_names:
            row = panel_responses.get(name, {}).get(pid)
            if row is None:
                miss.append(name)
                continue
            k_m, n_m = row
            cells_k.append(k_m)
            cells_n.append(n_m)
        if miss:
            logger.warning(
                "item %s missing %d/%d panel cells (%s); dropped",
                pid, len(miss), len(panel_names), miss,
            )
            n_dropped += 1
            continue

        panel_k = np.asarray(cells_k, dtype=np.float64)
        panel_n = np.asarray(cells_n, dtype=np.float64)
        n_panel_inferences += int(panel_n.sum())

        beta_x = _fit_beta_1d(panel_k, panel_n, panel_theta_arr)
        if beta_x is None:
            logger.warning(
                "item %s saturated on panel (k=%s, n=%s); dropped",
                pid, panel_k.astype(int).tolist(), panel_n.astype(int).tolist(),
            )
            n_dropped += 1
            continue

        new_items.append(AnchorItem(
            item_id=pid,
            prompt=ec.prompt,
            metadata=dict(ec.metadata or {}),
            beta=beta_x,
        ))
        new_pass.append(int(ec.pass_count))
        new_trial.append(int(ec.total_trials))
        fit_records.append({
            "prompt_id": pid,
            "beta": beta_x,
            "panel_k": panel_k.astype(int).tolist(),
            "panel_n": panel_n.astype(int).tolist(),
            "m_new_k": int(ec.pass_count),
            "m_new_n": int(ec.total_trials),
        })

    logger.info(
        "β̂ fit: %d items survived, %d dropped (saturated/missing)",
        len(new_items), n_dropped,
    )
    if not new_items:
        raise RuntimeError(
            "All E_new items dropped (saturation/missing); cannot calibrate"
        )

    with open(out_dir / "step2_calibration.json", "w") as f:
        json.dump({
            "K": K,
            "n_dropped_saturated": n_dropped,
            "n_panel_inferences": n_panel_inferences,
            "panel_theta": {n: float(anchor_bundle.panel_theta[n]) for n in panel_names},
            "items": fit_records,
        }, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # (5) EAP θ̂_new over X_anchor ∪ E_new with all β frozen
    # ------------------------------------------------------------------
    anchor_betas = np.array(
        [it.beta for it in anchor_bundle.items], dtype=np.float64,
    )
    union_pass = np.concatenate([
        pass_counts[used], np.array(new_pass, dtype=np.float64),
    ])
    union_trial = np.concatenate([
        trial_counts[used], np.array(new_trial, dtype=np.float64),
    ])
    union_beta = np.concatenate([
        anchor_betas[used],
        np.array([it.beta for it in new_items], dtype=np.float64),
    ])
    theta_hat_new, theta_se_new = estimator.estimate(
        union_pass, union_trial, union_beta,
    )
    logger.info(
        "Post Step-2: θ̂_new: %+.3f → %+.3f, SE: %.3f",
        theta_hat, theta_hat_new, theta_se_new,
    )

    return Step2Result(
        new_items=new_items,
        new_pass_counts=new_pass,
        new_trial_counts=new_trial,
        n_panel_inferences=n_panel_inferences,
        n_dropped_saturated=n_dropped,
        theta_hat_new=theta_hat_new,
        theta_se_new=theta_se_new,
        sgbs_sub_dir=sgbs_sub_dir,
        panel_responses_path=str(out_dir / "step2_panel"),
    )
