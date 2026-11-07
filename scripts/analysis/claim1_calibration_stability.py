"""Claim 1: empirical calibration sensitivity via panel-LOO.

The default anchor set is ALO[0.20, 0.80]; AVG[0.05, 0.95] is available
for comparison. Anchor membership is fixed from the observed full panel,
including in the parametric bootstrap. Each LOO fit is translated to match
the mean ability of the surviving models in the full fit. Reports item-rank
correlations, item-difficulty displacement, and held-out ability shifts.
These are sensitivity diagnostics, not tests against external DIF cutoffs
or proof of calibration validity on arbitrary new models.

Usage:
    conda run -n boundary --no-capture-output python \\
        scripts/analysis/claim1_calibration_stability.py \\
        --anchor-filter alo --with-null-sim --n-bootstrap 100 --seed 42 \\
        --output results/analysis/claim1_calibration_stability_alo_with_null.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difficulty.rasch import load_response_matrix, fit_ols, fit_mle


CATEGORIES = {
    "harmful_refusal":        "filter_ablation_raw_v2_judgefix",
    "over_refusal":           "filter_ablation_raw_v2_judgefix",
    "constrained_completion": "filter_ablation_raw_v2",
    "sycophancy_resistance":  "filter_ablation_raw_v2",
}

AVG_LO, AVG_HI = 0.05, 0.95
FILTER_LABELS = {"alo": "ALO[0.20, 0.80]", "avg": "AVG[0.05, 0.95]"}


def avg_filter_mask(p_hat: np.ndarray,
                    lo: float = AVG_LO, hi: float = AVG_HI) -> np.ndarray:
    panel_mean = p_hat.mean(axis=0)
    return (panel_mean >= lo) & (panel_mean <= hi)


def alo_filter_mask(p_hat: np.ndarray) -> np.ndarray:
    """Keep items with at least one observed pass rate in [0.20, 0.80]."""
    return ((p_hat >= 0.20) & (p_hat <= 0.80)).any(axis=0)


def fit_theta_fixed_beta(pass_counts_row: np.ndarray,
                         N_trials_row: np.ndarray,
                         beta_fixed: np.ndarray,
                         theta_init: float = 0.0,
                         max_iter: int = 200,
                         tol: float = 1e-7) -> float:
    """1PL MLE for one subject's theta with item difficulties beta held fixed.

    Score equation: sum_x [k_x - N_x * sigma(theta - beta_x)] = 0. Newton
    iterations on a single variable; the score is monotone decreasing in
    theta so convergence is global.
    """
    theta = float(theta_init)
    for _ in range(max_iter):
        z = np.clip(theta - beta_fixed, -40, 40)
        p = 1.0 / (1.0 + np.exp(-z))
        score = (pass_counts_row - N_trials_row * p).sum()
        info = (N_trials_row * p * (1 - p)).sum()
        if info < 1e-9:
            break
        step = score / info
        theta += step
        if abs(step) < tol:
            break
    return theta


def _panel_loo_inner(pc: np.ndarray,
                     Nt: np.ndarray,
                     p: np.ndarray,
                     models: list[str]) -> dict:
    """Pure-computation LOO over a (M, J) response submatrix.

    Reusable both for the real-data Claim 1 main run and for the
    parametric null bootstrap. Returns the per-LOO + pooled summary.
    """
    M = pc.shape[0]
    th_init, b_init = fit_ols(p)
    theta_full, beta_full, _ = fit_mle(pc, Nt, th_init, b_init)

    per_loo = []
    pooled_abs_dbeta = []
    for m_drop in range(M):
        keep = [i for i in range(M) if i != m_drop]
        pc_l, Nt_l, p_l = pc[keep], Nt[keep], p[keep]
        th_init_l, b_init_l = fit_ols(p_l)
        theta_loo, beta_loo, _ = fit_mle(pc_l, Nt_l, th_init_l, b_init_l)

        # 1PL: identifiable up to translation. Re-anchor LOO to full-fit
        # coordinates by matching mean(theta) on surviving 8 models.
        shift = float(theta_full[keep].mean() - theta_loo.mean())
        beta_loo_aligned = beta_loo + shift
        theta_loo_aligned = theta_loo + shift

        delta_beta = beta_loo_aligned - beta_full
        abs_dbeta = np.abs(delta_beta)
        sp_rho, _ = spearmanr(beta_full, beta_loo_aligned)
        pe_r, _ = pearsonr(beta_full, beta_loo_aligned)
        delta_theta_keep = theta_loo_aligned - theta_full[keep]

        # Accuracy-space MAE: held-out model's theta is refit using FIXED
        # beta_loo_aligned (the calibration any new external model would
        # see); predict its own cell pass rates and compare against the
        # observed Binomial p_hat.
        theta_held = fit_theta_fixed_beta(pc[m_drop], Nt[m_drop],
                                          beta_loo_aligned,
                                          theta_init=float(theta_full[m_drop]))
        p_pred = 1.0 / (1.0 + np.exp(
            -np.clip(theta_held - beta_loo_aligned, -40, 40)))
        p_obs = pc[m_drop] / np.clip(Nt[m_drop], 1, None)
        # Cell-level MAE (mixes 1PL link mis-specification with calibration
        # drift; reported for completeness).
        acc_mae_cell = float(np.mean(np.abs(p_pred - p_obs)))
        acc_rmse_cell = float(np.sqrt(np.mean((p_pred - p_obs) ** 2)))
        # In-sample mean score residual; not a held-out prediction metric.
        # With equal trial counts, this is zero by the MLE score equation.
        acc_score_err = float(abs(p_pred.mean() - p_obs.mean()))

        pooled_abs_dbeta.append(abs_dbeta)
        per_loo.append({
            "held_out": models[m_drop],
            "shift": shift,
            "spearman_rho": float(sp_rho),
            "pearson_r": float(pe_r),
            "median_abs_delta_beta": float(np.median(abs_dbeta)),
            "p95_abs_delta_beta": float(np.quantile(abs_dbeta, 0.95)),
            "max_abs_delta_beta": float(abs_dbeta.max()),
            "median_abs_delta_theta_keep": float(np.median(np.abs(delta_theta_keep))),
            "max_abs_delta_theta_keep": float(np.abs(delta_theta_keep).max()),
            "theta_held_fixed_beta": float(theta_held),
            "theta_held_full_fit": float(theta_full[m_drop]),
            "abs_delta_theta_held": float(abs(theta_held - theta_full[m_drop])),
            "accuracy_mae_cell": acc_mae_cell,
            "accuracy_rmse_cell": acc_rmse_cell,
            "accuracy_score_error": acc_score_err,
        })

    pooled = np.concatenate(pooled_abs_dbeta)
    sp = [r["spearman_rho"] for r in per_loo]
    pe = [r["pearson_r"] for r in per_loo]
    mae_cell = [r["accuracy_mae_cell"] for r in per_loo]
    score_err = [r["accuracy_score_error"] for r in per_loo]
    dth = [r["abs_delta_theta_held"] for r in per_loo]
    keep_dth_med = [r["median_abs_delta_theta_keep"] for r in per_loo]

    return {
        "n_panel": M,
        "spearman_rho_mean": float(np.mean(sp)),
        "spearman_rho_min":  float(np.min(sp)),
        "pearson_r_mean":    float(np.mean(pe)),
        "pearson_r_min":     float(np.min(pe)),
        "median_abs_delta_beta_pooled": float(np.median(pooled)),
        "p95_abs_delta_beta_pooled":    float(np.quantile(pooled, 0.95)),
        "max_abs_delta_beta_pooled":    float(pooled.max()),
        "median_abs_delta_theta_keep_mean": float(np.mean(keep_dth_med)),
        "abs_delta_theta_held_mean": float(np.mean(dth)),
        "abs_delta_theta_held_max":  float(np.max(dth)),
        "accuracy_mae_cell_mean": float(np.mean(mae_cell)),
        "accuracy_mae_cell_max":  float(np.max(mae_cell)),
        "accuracy_score_error_mean": float(np.mean(score_err)),
        "accuracy_score_error_max":  float(np.max(score_err)),
        "theta_full": theta_full.tolist(),
        "beta_full": beta_full.tolist(),
        "panel": models,
        "per_loo": per_loo,
    }


def panel_loo(category: str, response_dir: str, anchor_filter: str = "alo") -> dict:
    rm = load_response_matrix(response_dir)
    pass_counts = rm["pass_counts"]
    N_trials = rm["total_trials"]
    p_hat = rm["p_hat"]
    models = rm["models"]
    M, J_pool = pass_counts.shape

    masks = {"alo": alo_filter_mask, "avg": avg_filter_mask}
    mask = masks[anchor_filter](p_hat)
    pc = pass_counts[:, mask]
    Nt = N_trials[:, mask]
    p = p_hat[:, mask]
    J = int(mask.sum())

    inner = _panel_loo_inner(pc, Nt, p, models)
    inner.update({
        "category": category,
        "response_dir": response_dir,
        "n_items_pool": int(J_pool),
        "filter": FILTER_LABELS[anchor_filter],
        "n_items_anchor": J,
        "anchor_items": [item for item, kept in zip(rm["items"], mask) if kept],
        "input_sha256": {
            str(path.relative_to(response_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(response_dir).glob("*/*.json*"))
            if path.name in {"response_matrix.jsonl", "metadata.json"}
        },
        "_N_trials_anchor": Nt.tolist(),
    })
    return inner


def null_bootstrap(theta_full: np.ndarray,
                   beta_full: np.ndarray,
                   N_trials: np.ndarray,
                   models: list[str],
                   n_bootstrap: int = 100,
                   seed: int = 42) -> dict:
    """Parametric null distribution of LOO stability metrics.

    Generative null: under correct 1PL Rasch with parameters fixed at
    (theta_full, beta_full), simulate Binomial responses and re-run the
    full LOO pipeline. Question answered: "if Rasch is correctly
    specified at this panel size + N_trials, how large would the
    observed |Delta beta|, Spearman rho, etc. be from sampling noise
    alone?"

    Returns empirical null distributions and their 5th/95th percentiles.
    These are conditional on fitted parameters and fixed anchor membership.
    """
    rng = np.random.default_rng(seed)
    M, J = N_trials.shape
    z = np.clip(theta_full[:, None] - beta_full[None, :], -40, 40)
    p_true = 1.0 / (1.0 + np.exp(-z))
    Nt_int = N_trials.astype(int)

    samples = {
        "spearman_rho_min": [],
        "spearman_rho_mean": [],
        "pearson_r_min": [],
        "pearson_r_mean": [],
        "median_abs_delta_beta_pooled": [],
        "p95_abs_delta_beta_pooled": [],
        "max_abs_delta_beta_pooled": [],
        "abs_delta_theta_held_max": [],
    }
    for b in range(n_bootstrap):
        pc_sim = rng.binomial(Nt_int, p_true)
        p_sim = pc_sim / np.clip(Nt_int, 1, None)
        res = _panel_loo_inner(pc_sim, N_trials, p_sim, models)
        for k in samples:
            if not np.isfinite(res[k]):
                raise ValueError(f"Non-finite {k} in bootstrap replicate {b}")
            samples[k].append(res[k])
        if (b + 1) % 10 == 0:
            print(f"  [null-sim] completed {b + 1}/{n_bootstrap}", flush=True)

    def quantiles(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        a = np.asarray(xs)
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "p05": float(np.quantile(a, 0.05)),
            "p50": float(np.quantile(a, 0.50)),
            "p95": float(np.quantile(a, 0.95)),
        }

    return {
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "anchor_selection": "fixed observed anchor set; no reselection in bootstrap",
        "null": {k: quantiles(v) for k, v in samples.items()},
        "samples": samples,
    }


def null_verdict(observed: dict, null: dict) -> dict:
    """Directional comparison with empirical null 5th/95th percentiles.

    Spearman / Pearson are correlations -> upper bound is 1, we test
    observed >= null_p05 (correlations should not be LOWER than
    sampling-noise-only). |Delta beta|-style metrics test
    observed <= null_p95 (drift should not be LARGER than
    sampling-noise-only). These comparisons do not establish model validity.
    """
    n = null["null"]
    out: dict[str, dict] = {}

    for k in ("spearman_rho_min", "spearman_rho_mean",
              "pearson_r_min", "pearson_r_mean"):
        if not n[k] or n[k].get("n", 0) == 0:
            continue
        obs = observed[k]
        thresh = n[k]["p05"]
        out[k] = {
            "observed": obs,
            "null_p05": thresh,
            "consistent_with_null": bool(obs >= thresh),
        }
    for k in ("median_abs_delta_beta_pooled",
              "p95_abs_delta_beta_pooled",
              "max_abs_delta_beta_pooled",
              "abs_delta_theta_held_max"):
        if not n[k] or n[k].get("n", 0) == 0:
            continue
        obs = observed[k]
        thresh = n[k]["p95"]
        out[k] = {
            "observed": obs,
            "null_p95": thresh,
            "consistent_with_null": bool(obs <= thresh),
        }
    return out


def print_summary(s: dict) -> None:
    print(f"  n_panel={s['n_panel']}  J_pool={s['n_items_pool']}  "
          f"J_anchor({s['filter']})={s['n_items_anchor']}")
    print(f"  Spearman rho(beta)   mean={s['spearman_rho_mean']:.4f}  "
          f"min={s['spearman_rho_min']:.4f}")
    print(f"  Pearson  r(beta)     mean={s['pearson_r_mean']:.4f}  "
          f"min={s['pearson_r_min']:.4f}")
    print(f"  |Delta beta| pooled  median={s['median_abs_delta_beta_pooled']:.3f}  "
          f"p95={s['p95_abs_delta_beta_pooled']:.3f}  "
          f"max={s['max_abs_delta_beta_pooled']:.3f}  (logit)")
    print(f"  |Delta theta_keep|   median(per-LOO mean)={s['median_abs_delta_theta_keep_mean']:.3f}  (logit)")
    print(f"  |Delta theta_held|   mean={s['abs_delta_theta_held_mean']:.3f}  "
          f"max={s['abs_delta_theta_held_max']:.3f}  (logit)")
    # NOTE: cell-MAE and score-error logged to JSON only:
    #   - score-error is mathematically zero (1PL MLE constraint),
    #     not informative as a stability indicator;
    #   - cell-MAE measures 1PL link goodness-of-fit per model
    #     (Claim 2 territory), not panel-LOO stability.
    # Growing Pains "MAE 2-3pp" alignment requires anchor/held-out item
    # split, which is the Claim 3 experiment.


def main():
    ap = argparse.ArgumentParser(description="Claim 1 calibration stability via panel-LOO.")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--output",
                    default="results/analysis/claim1_calibration_stability_alo.json")
    ap.add_argument("--anchor-filter", choices=FILTER_LABELS, default="alo")
    ap.add_argument("--categories", nargs="+", default=None,
                    help="Subset of categories to run (default: all 4).")
    ap.add_argument("--with-null-sim", action="store_true",
                    help="Run parametric null bootstrap on each category.")
    ap.add_argument("--n-bootstrap", type=int, default=100,
                    help="Bootstrap samples for null simulation.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.with_null_sim and args.n_bootstrap < 1:
        ap.error("--n-bootstrap must be positive when --with-null-sim is set")

    cats = args.categories or list(CATEGORIES.keys())
    root = Path(args.results_root)
    summaries: dict[str, dict] = {}

    for cat in cats:
        if cat not in CATEGORIES:
            print(f"  [skip] unknown category {cat}")
            continue
        rdir = root / cat / CATEGORIES[cat]
        if not rdir.exists():
            print(f"  [skip] {cat}: response_dir missing -> {rdir}")
            continue
        print(f"\n=== {cat}   (response_dir: {rdir.name}) ===")
        s = panel_loo(cat, str(rdir), args.anchor_filter)
        summaries[cat] = s
        print_summary(s)

        if args.with_null_sim:
            print(f"  [null-sim] running {args.n_bootstrap} bootstrap "
                  f"replicates ...", flush=True)
            null = null_bootstrap(
                theta_full=np.asarray(s["theta_full"]),
                beta_full=np.asarray(s["beta_full"]),
                N_trials=np.asarray(s["_N_trials_anchor"]),
                models=s["panel"],
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            verdict = null_verdict(s, null)
            s["null_bootstrap"] = null
            s["null_verdict"] = verdict
            print("  Observed vs empirical null percentiles (p95 for drift; p05 for rho/r):")
            for k, v in verdict.items():
                bound_label = "p05" if "rho" in k or k.startswith("pearson") else "p95"
                bound_val = v.get(f"null_{bound_label}", float("nan"))
                tag = "✓" if v["consistent_with_null"] else "✗"
                print(f"    {tag} {k:42s}  obs={v['observed']:.4f}  "
                      f"null_{bound_label}={bound_val:.4f}")

    # Cross-category aggregation
    if len(summaries) >= 2:
        print("\n=== Cross-category aggregate ===")
        for k, label in [
            ("spearman_rho_min",        "Spearman min"),
            ("pearson_r_min",           "Pearson  min"),
            ("median_abs_delta_beta_pooled", "median |Delta beta|"),
            ("p95_abs_delta_beta_pooled",    "p95    |Delta beta|"),
            ("abs_delta_theta_held_max", "max  |Delta theta_held|"),
        ]:
            vals = [s[k] for s in summaries.values()]
            is_pp = ("MAE" in label) or ("Acc" in label)
            unit = "pp" if is_pp else ("logit" if "Delta" in label else "")
            scale = 100.0 if is_pp else 1.0
            print(f"  {label:30s}: " + "  ".join(
                f"{c[:4]}={v*scale:.3f}{unit}" for c, v in zip(summaries.keys(), vals)))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "filter": FILTER_LABELS[args.anchor_filter],
            "irt_model": "1pl",
            "alignment": "translation-only (mean theta_keep matched)",
            "categories": summaries,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nJSON -> {args.output}")


if __name__ == "__main__":
    main()
