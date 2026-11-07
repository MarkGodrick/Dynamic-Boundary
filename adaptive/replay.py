"""Offline replay of the adaptive (Step-1) CAT evaluation on a frozen anchor
response matrix.

Extracted from ``scripts/analysis/claim3_ablation_simulation.py`` so that both
the ablation simulator and the Claim-3 Step-2 calibration driver
(``scripts/analysis/calibrate_claim3_step2.py``) share one implementation
instead of importing across scripts.

Pure replay: ``simulate`` consumes pre-collected ``(pass_count, total_trials)``
cells from a response matrix via lookup — no model inference. The estimator is
the same adaptive-GH EAP used live (``adaptive.posterior.EAPEstimator``); its
Newton step already clamps θ to ±10 logits, so no separate "robust" subclass is
needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .anchor_bundle import AnchorBundle, load_anchor_bundle
from .cat_select import fisher_info_select, pilot_indices
from .posterior import EAPEstimator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = [
    "harmful_refusal", "over_refusal",
    "constrained_completion", "sycophancy_resistance",
]

T_MAX = 50
PILOT_QUANTILES = [0.10, 0.30, 0.50, 0.70, 0.90]
TAU_COV = 1.0   # for cov_v1 (current paper)
TAU_PAD = 0.5   # for cov_v2 (panel θ̂ range margin)

# Paper's primary adaptive config (T2c: stability termination, ε=0.02, K=3,
# prior σ=3.0). Used both by the ablation simulator's CONFIGS table and by the
# Step-2 calibration driver, which must replay Step-1 under the same config.
T2C_CONFIG: dict = dict(
    t_min=2, prior_sigma=3.0, apply_tmin_gate=True, force_no_early=False,
    termination="stability", stab_eps=0.02, stab_K=3,
)


def load_response_matrix(path: Path) -> Dict[str, Tuple[int, int]]:
    lookup: Dict[str, Tuple[int, int]] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            lookup[row['prompt_id']] = (int(row['pass_count']), int(row['total_trials']))
    return lookup


def bundle_path_for(category: str, filter_rule: str) -> Path:
    name = "anchor_alo_bundle_M9.json" if filter_rule == "alo" else "anchor_bundle_M9.json"
    return PROJECT_ROOT / "benchmarks" / category / name


def response_matrix_path_for(category: str, model: str, filter_rule: str) -> Path:
    dirname = "anchor_set_alo" if filter_rule == "alo" else "anchor_set"
    return PROJECT_ROOT / "results" / category / dirname / model / "response_matrix.jsonl"


def find_full_anchor_pairs(categories: List[str], filter_rule: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for cat in categories:
        dirname = "anchor_set_alo" if filter_rule == "alo" else "anchor_set"
        anchor_dir = PROJECT_ROOT / "results" / cat / dirname
        if not anchor_dir.exists():
            continue
        for model_dir in sorted(anchor_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            rm_path = model_dir / "response_matrix.jsonl"
            if rm_path.exists():
                pairs.append((cat, model_dir.name))
    return pairs


def likelihood_only_se(theta_hat: float, k: np.ndarray, N: np.ndarray,
                       beta: np.ndarray) -> float:
    """SE based purely on observed Fisher info, ignoring prior.

    SE_lik(θ) = 1 / sqrt(Σ N_j σ(θ-β_j)(1-σ(θ-β_j)))

    With saturated cells (p≈0 or p≈1), info contribution per cell collapses
    to 0, so SE_lik stays large — exactly the diagnostic signal posterior SE
    masks via prior shrinkage.
    """
    diff = theta_hat - beta
    p = np.where(diff >= 0, 1.0 / (1.0 + np.exp(-diff)),
                 np.exp(diff) / (1.0 + np.exp(diff)))
    info = float((N * p * (1.0 - p)).sum())
    if info <= 1e-10:
        return float('inf')
    return float(1.0 / np.sqrt(info))


def coverage_v1(theta_hat: float, theta_se: float, beta_x: float,
                tau_se_thr: float, tau_cov: float = TAU_COV) -> bool:
    """Current paper coverage check (anchor β proximity + SE gate)."""
    return abs(beta_x - theta_hat) > tau_cov and theta_se > tau_se_thr


def coverage_v2(theta_hat: float, theta_se: float, panel_min: float, panel_max: float,
                tau_se_thr: float, tau_pad: float = TAU_PAD) -> bool:
    """Panel θ̂ range based + SE gate (B2 coverage variant)."""
    out_of_range = theta_hat < panel_min - tau_pad or theta_hat > panel_max + tau_pad
    return out_of_range and theta_se > tau_se_thr


def simulate(bundle: AnchorBundle,
             response_lookup: Dict[str, Tuple[int, int]],
             config: dict,
             T_max: int = T_MAX) -> dict:
    """Replay Step-1 adaptive CAT on a frozen anchor matrix.

    Returns a dict with the trajectory, the full-anchor reference θ̂, and the
    terminal CAT state (``used`` mask + per-item pass/trial counts) so callers
    can reconstruct the CAT-used anchor subset for Step-2 augmentation.
    """
    items = bundle.items
    J = len(items)
    beta = np.array([it.beta for it in items], dtype=np.float64)
    panel_values = list(bundle.panel_theta.values())
    panel_min, panel_max = float(min(panel_values)), float(max(panel_values))

    missing = [it.item_id for it in items if it.item_id not in response_lookup]
    if missing:
        return {"error": f"missing {len(missing)} items in response_lookup"}

    estimator = EAPEstimator(prior_mu=0.0, prior_sigma=config['prior_sigma'], n_quad=21)

    used = np.zeros(J, dtype=bool)
    pass_counts = np.zeros(J, dtype=np.float64)
    trial_counts = np.zeros(J, dtype=np.float64)

    # Pilot
    pilot_idx = pilot_indices(beta, PILOT_QUANTILES)
    for j in pilot_idx:
        k, n = response_lookup[items[j].item_id]
        used[j] = True
        pass_counts[j] = k
        trial_counts[j] = n

    theta_hat, theta_se = estimator.estimate(
        pass_counts[used], trial_counts[used], beta[used]
    )

    # Initial likelihood-only SE on pilot
    se_lik_pilot = likelihood_only_se(theta_hat, pass_counts[used], trial_counts[used], beta[used])

    trajectory: List[dict] = [{
        't': 0,
        'item_id': '<pilot>',
        'beta_x': float(np.mean(beta[pilot_idx])),
        'theta_hat': float(theta_hat),
        'theta_se': float(theta_se),
        'theta_se_lik': se_lik_pilot,
        'delta_theta': 0.0,
        'cov_v1': False,
        'cov_v2': coverage_v2(theta_hat, theta_se, panel_min, panel_max,
                              tau_se_thr=config.get('tau_se', 0.30)),
        'phase': 'pilot',
    }]

    termination = config.get('termination', 'se_post')
    stab_eps = config.get('stab_eps', 0.05)
    stab_K = config.get('stab_K', 3)
    stab_streak = 0  # consecutive |Δθ̂_t| < eps count

    stopped_reason = "max_iters"
    for t in range(1, T_max + 1):
        if used.all():
            stopped_reason = "anchor_exhausted"
            break

        x_idx = fisher_info_select(theta_hat, beta, used, N=40)
        item = items[x_idx]
        beta_x = float(beta[x_idx])

        # Coverage checks BEFORE the trial (matches algorithm's semantics)
        cov_se_thr = config.get('tau_se', 0.30)
        c_v1 = coverage_v1(theta_hat, theta_se, beta_x, tau_se_thr=cov_se_thr)
        c_v2 = coverage_v2(theta_hat, theta_se, panel_min, panel_max,
                           tau_se_thr=cov_se_thr)

        # "Run trial" via lookup
        k, n = response_lookup[item.item_id]
        used[x_idx] = True
        pass_counts[x_idx] = k
        trial_counts[x_idx] = n

        theta_hat_post, theta_se_post = estimator.estimate(
            pass_counts[used], trial_counts[used], beta[used]
        )

        # Likelihood-only SE at post-update θ̂
        se_lik_post = likelihood_only_se(
            theta_hat_post, pass_counts[used], trial_counts[used], beta[used]
        )

        delta_theta = abs(theta_hat_post - theta_hat)

        trajectory.append({
            't': t,
            'item_id': item.item_id,
            'beta_x': beta_x,
            'theta_hat': float(theta_hat_post),
            'theta_se': float(theta_se_post),
            'theta_se_lik': se_lik_post,
            'delta_theta': delta_theta,
            'cov_v1': c_v1,
            'cov_v2': c_v2,
            'phase': 'main',
        })

        theta_hat = float(theta_hat_post)
        theta_se = float(theta_se_post)

        # Stability tracker update
        if delta_theta < stab_eps:
            stab_streak += 1
        else:
            stab_streak = 0

        if not config['force_no_early']:
            tmin_ok = (t >= config['t_min']) if config['apply_tmin_gate'] else True
            should_stop = False

            if termination == "se_post":
                should_stop = theta_se <= config['tau_se']
            elif termination == "se_lik":
                should_stop = se_lik_post <= config['tau_se']
            elif termination == "stability":
                should_stop = stab_streak >= stab_K
            elif termination == "stability_and_lik":
                # Both stability AND likelihood SE must satisfy
                should_stop = (stab_streak >= stab_K) and (se_lik_post <= config['tau_se'])

            if should_stop and tmin_ok:
                stopped_reason = termination
                break

    # Reference: θ̂_full from same estimator config
    all_k = np.array([response_lookup[it.item_id][0] for it in items], dtype=np.float64)
    all_n = np.array([response_lookup[it.item_id][1] for it in items], dtype=np.float64)
    theta_full, se_full = estimator.estimate(all_k, all_n, beta)

    return {
        "trajectory": trajectory,
        "theta_full": float(theta_full),
        "se_full": float(se_full),
        "stopped_reason": stopped_reason,
        "panel_min": panel_min,
        "panel_max": panel_max,
        # Terminal CAT state for Step-2 augmentation (CAT-used anchor subset).
        "used": used.copy(),
        "pass_counts": pass_counts.copy(),
        "trial_counts": trial_counts.copy(),
    }
