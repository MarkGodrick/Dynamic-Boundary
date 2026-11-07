"""Joint Rasch fit on SGBS + Random union, then compare on the SAME β scale.

Two analyses:
  (A) Joint fit: concatenate 600 SGBS items + 600 random items, run MLE on
      the 6×1200 response matrix. Now both sets have β on the same scale.
      Generate a dual histogram (SGBS vs random) on the unified axis.

  (B) Random + filtering: apply panel-based filters to the 600 random items
      (drop items where panel mean p̂ is too extreme), and see what the
      filtered subset's quality looks like vs SGBS.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difficulty.rasch import load_response_matrix


EC_DIR = "results/constrained_completion/validation/edge_case"
RB_DIR = "results/constrained_completion/validation/random_baseline"


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def neg_loglik(params, K, N, M, J):
    theta = params[:M]
    beta = params[M:]
    diff = theta[:, None] - beta[None, :]
    p = np.clip(sigmoid(diff), 1e-10, 1 - 1e-10)
    return -(K * np.log(p) + (N - K) * np.log(1 - p)).sum()


def neg_loglik_grad(params, K, N, M, J):
    theta = params[:M]
    beta = params[M:]
    diff = theta[:, None] - beta[None, :]
    p = np.clip(sigmoid(diff), 1e-10, 1 - 1e-10)
    residual = K - N * p
    return np.concatenate([-residual.sum(axis=1), residual.sum(axis=0)])


def joint_mle(K, N):
    M, J = K.shape
    # OLS init
    p_smooth = np.clip(K / N, 0.025, 0.975)
    y = np.log(p_smooth / (1 - p_smooth))
    theta_init = y.mean(axis=1) - y.mean()
    beta_init = -(y.mean(axis=0) - y.mean())

    x0 = np.concatenate([theta_init, beta_init])
    result = minimize(
        neg_loglik, x0, args=(K, N, M, J),
        jac=neg_loglik_grad, method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    theta = result.x[:M]
    beta = result.x[M:]
    shift = theta.mean()
    return theta - shift, beta - shift


def main():
    # Load both
    ec_rm = load_response_matrix(EC_DIR)
    rb_rm = load_response_matrix(RB_DIR)

    # Verify model order matches
    assert ec_rm["models"] == rb_rm["models"], "Model lists differ!"
    models = ec_rm["models"]
    M = len(models)

    # Concatenate items (SGBS first, then random)
    K_ec = ec_rm["pass_counts"]
    N_ec = ec_rm["total_trials"]
    K_rb = rb_rm["pass_counts"]
    N_rb = rb_rm["total_trials"]

    K_joint = np.concatenate([K_ec, K_rb], axis=1)
    N_joint = np.concatenate([N_ec, N_rb], axis=1)
    p_joint = K_joint / np.maximum(N_joint, 1)

    n_ec = K_ec.shape[1]
    n_rb = K_rb.shape[1]
    print(f"Joint fit: {M} models × {n_ec + n_rb} items ({n_ec} SGBS + {n_rb} random)")

    # Joint MLE
    theta_joint, beta_joint = joint_mle(K_joint, N_joint)

    beta_ec = beta_joint[:n_ec]
    beta_rb = beta_joint[n_ec:]

    print("\n" + "=" * 78)
    print("  JOINT FIT — model abilities on unified β scale")
    print("=" * 78)
    short = lambda m: m.split("/")[-1].replace("DeepSeek-R1-Distill-Llama", "R1") \
                                       .replace("Qwen3-30B-A3B-Instruct-2507", "Qwen-30B") \
                                       .replace("Mistral-Small-3.1-24B-Instruct-2503", "Mistral-24B") \
                                       .replace("gemma-3-", "Gemma3-").replace("-it", "")[:18]
    for i in np.argsort(-theta_joint):
        print(f"  {short(models[i]):<20} θ = {theta_joint[i]:+.3f}")

    # ----- (A) Histogram comparison on unified scale -----
    print("\n" + "=" * 78)
    print("  (A) β distribution on UNIFIED scale")
    print("=" * 78)
    print(f"  {'percentile':<14} {'SGBS':>12} {'Random':>12}")
    print(f"  {'-'*14} {'-'*12} {'-'*12}")
    for label, q in [("p05", 5), ("p25", 25), ("p50 (median)", 50),
                     ("p75", 75), ("p95", 95)]:
        bs = np.percentile(beta_ec, q)
        br = np.percentile(beta_rb, q)
        print(f"  {label:<14} {bs:>+12.3f} {br:>+12.3f}")
    print(f"  {'IQR':<14} "
          f"{np.percentile(beta_ec, 75) - np.percentile(beta_ec, 25):>+12.3f} "
          f"{np.percentile(beta_rb, 75) - np.percentile(beta_rb, 25):>+12.3f}")

    # Histogram bins (focus on -3 to +3, lump extremes)
    bins = np.linspace(-3, 3, 25)   # 24 bins of width 0.25
    hist_ec, _ = np.histogram(np.clip(beta_ec, -3, 3), bins=bins)
    hist_rb, _ = np.histogram(np.clip(beta_rb, -3, 3), bins=bins)

    n_ec_below = (beta_ec < -3).sum()
    n_ec_above = (beta_ec > 3).sum()
    n_rb_below = (beta_rb < -3).sum()
    n_rb_above = (beta_rb > 3).sum()
    print(f"\n  β < -3:  SGBS {n_ec_below}, Random {n_rb_below}")
    print(f"  β > +3:  SGBS {n_ec_above}, Random {n_rb_above}")
    print(f"  Items inside [-3, +3]: SGBS {hist_ec.sum()}, Random {hist_rb.sum()}")

    # ----- (B) Random + filtering analysis -----
    print("\n" + "=" * 78)
    print("  (B) Random + post-hoc filtering by panel mean p̂")
    print("=" * 78)
    print(f"  Each random item has 6 cells (one per panel model). Compute the")
    print(f"  per-item panel-mean p̂, then filter by various thresholds.\n")

    p_rb = K_rb / N_rb              # (M, J)
    item_mean = p_rb.mean(axis=0)   # per-item mean across 6 models

    print(f"  {'filter':<32} {'kept':>8} {'TIF mean':>10} {'TIF min':>10} {'useful%':>10}")
    print(f"  {'-'*32} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    # SGBS reference
    p_ec = K_ec / N_ec
    tif_ec = []
    for t in theta_joint:
        p_th = sigmoid(t - beta_ec)
        tif_ec.append((p_th * (1 - p_th)).sum())
    useful_ec = ((p_ec >= 0.2) & (p_ec <= 0.8)).sum() / p_ec.size * 100
    print(f"  {'SGBS (reference)':<32} {n_ec:>8} {np.mean(tif_ec):>10.2f} "
          f"{np.min(tif_ec):>10.2f} {useful_ec:>9.1f}%")

    # Random unfiltered
    tif_rb = []
    for t in theta_joint:
        p_th = sigmoid(t - beta_rb)
        tif_rb.append((p_th * (1 - p_th)).sum())
    useful_rb = ((p_rb >= 0.2) & (p_rb <= 0.8)).sum() / p_rb.size * 100
    print(f"  {'Random (unfiltered)':<32} {n_rb:>8} {np.mean(tif_rb):>10.2f} "
          f"{np.min(tif_rb):>10.2f} {useful_rb:>9.1f}%")

    # Random with various filters
    filters = [
        ("Random filter [0.05, 0.95]", 0.05, 0.95),
        ("Random filter [0.1, 0.9]",   0.10, 0.90),
        ("Random filter [0.2, 0.8]",   0.20, 0.80),
        ("Random filter [0.3, 0.7]",   0.30, 0.70),
        ("Random filter [0.4, 0.6]",   0.40, 0.60),
    ]
    for label, lo, hi in filters:
        mask = (item_mean >= lo) & (item_mean <= hi)
        n_kept = mask.sum()
        if n_kept == 0:
            print(f"  {label:<32} {0:>8}")
            continue
        beta_filt = beta_rb[mask]
        p_filt = p_rb[:, mask]
        tif_filt = []
        for t in theta_joint:
            p_th = sigmoid(t - beta_filt)
            tif_filt.append((p_th * (1 - p_th)).sum())
        useful_filt = ((p_filt >= 0.2) & (p_filt <= 0.8)).sum() / p_filt.size * 100
        print(f"  {label:<32} {n_kept:>8} {np.mean(tif_filt):>10.2f} "
              f"{np.min(tif_filt):>10.2f} {useful_filt:>9.1f}%")

    # ----- (C) Same comparison, but adjusting for budget -----
    print("\n" + "=" * 78)
    print("  (C) Budget-normalized comparison")
    print("=" * 78)
    print(f"  Random total inference: {n_rb} items × 6 models × 20 trials = "
          f"{n_rb * 6 * 20} trials")
    print(f"  SGBS total inference:   {n_ec} items × 6 models × 20 trials = "
          f"{n_ec * 6 * 20} trials")
    print(f"  (NB: SGBS additionally spent search budget — see ToDo for budget-fair analysis)")

    # Save data for HTML viz
    out = {
        "models": models,
        "theta_joint": theta_joint.tolist(),
        "beta_ec": beta_ec.tolist(),
        "beta_rb": beta_rb.tolist(),
        "hist_ec": hist_ec.tolist(),
        "hist_rb": hist_rb.tolist(),
        "bins": bins.tolist(),
        "n_ec_outside": [int(n_ec_below), int(n_ec_above)],
        "n_rb_outside": [int(n_rb_below), int(n_rb_above)],
    }
    Path("results/constrained_completion").mkdir(exist_ok=True, parents=True)
    with open("results/constrained_completion/joint_fit_data.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Joint fit data saved to results/constrained_completion/joint_fit_data.json")


if __name__ == "__main__":
    main()
