"""HiES Q2.1: end-to-end pipeline sensitivity to trials per cell.

The experiment reuses the stored 40 trial records for every reference-panel
model/item cell. For each smaller N it rebuilds pass-count matrices, reapplies
the ALO filter, and refits the joint 1PL Rasch model. It measures:

1. filter-set transitions relative to N=40 (retained, dropped, newly admitted,
   and Jaccard overlap);
2. translation-invariant fitted-scale summaries (robust/full beta span, theta
   span, and median theta standard error).

By default, the analysis is repeated for 20 reproducible random orderings of
each cell's 40 trials. Within one ordering, the first 10 trials are also part
of the first 20 and 30 trials. Prefix truncation remains available as a
single-order diagnostic.

Run from the repository root:

    conda run -n boundary --no-capture-output python -u \
        scripts/analysis/rebuttal_n_sensitivity.py

Single-prefix diagnostic:

    conda run -n boundary --no-capture-output python -u \
        scripts/analysis/rebuttal_n_sensitivity.py \
        --sampling prefix --seeds 42
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from core.panel_config import PANEL_MODELS  # noqa: E402
from difficulty.rasch import _sigmoid, fit_mle, fit_ols  # noqa: E402


CATEGORY_SOURCES = {
    "harmful_refusal": "filter_ablation_raw_v2_judgefix",
    "over_refusal": "filter_ablation_raw_v2_judgefix",
    "constrained_completion": "filter_ablation_raw_v2",
    "sycophancy_resistance": "filter_ablation_raw_v2",
}
DEFAULT_N_VALUES = (10, 20, 30)
DEFAULT_SEEDS = tuple(range(20))
REFERENCE_N = 40
ALO_LO = 0.20
ALO_HI = 0.80


@dataclass(frozen=True)
class PanelTrials:
    """Stored trial records aligned as model -> item -> trials."""

    models: list[str]
    items: list[str]
    trials: dict[str, dict[str, list[dict]]]
    source_dir: Path


def model_dir_name(model: str) -> str:
    return model.replace("/", "_").replace("-", "_")


def parse_int_list(value: str, *, label: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError(f"{label} cannot be empty")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError(f"{label} contains duplicate values")
    return parsed


def load_panel_trials(source_dir: Path, models: Sequence[str]) -> PanelTrials:
    """Load cells with at least REFERENCE_N stored trial attempts.

    Only prompt IDs present with 40 stored attempts for every panel model are
    retained. Missing/incomplete cells are reported by the caller via the
    resulting item count and the source manifest.
    """
    per_model: dict[str, dict[str, list[dict]]] = {}
    for model in models:
        path = source_dir / model_dir_name(model) / "response_matrix.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing panel response matrix: {path}")
        cells: dict[str, list[dict]] = {}
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt_id = row["prompt_id"]
                trials = row.get("trials") or []
                if prompt_id in cells:
                    raise ValueError(f"Duplicate prompt_id {prompt_id!r} in {path}:{line_number}")
                if len(trials) >= REFERENCE_N:
                    cells[prompt_id] = trials[:REFERENCE_N]
        per_model[model] = cells

    common_items = sorted(set.intersection(*(set(cells) for cells in per_model.values())))
    if not common_items:
        raise ValueError(f"No complete N={REFERENCE_N} cells shared by all models in {source_dir}")
    return PanelTrials(
        models=list(models),
        items=common_items,
        trials=per_model,
        source_dir=source_dir,
    )


def _stable_permutation(length: int, seed: int, *cell_key: str) -> np.ndarray:
    """Return a process-independent random permutation for one cell."""
    payload = "\0".join((str(seed), *cell_key)).encode()
    digest = hashlib.sha256(payload).digest()
    cell_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return np.random.default_rng(cell_seed).permutation(length)


def select_trials(
    trials: Sequence[dict],
    n: int,
    sampling: str,
    seed: int,
    *cell_key: str,
) -> list[dict]:
    """Select N attempted trials; random selections are nested across N."""
    if not 0 < n <= len(trials):
        raise ValueError(f"Requested N={n} from a cell with {len(trials)} trials")
    if sampling == "prefix":
        return list(trials[:n])
    if sampling == "subsample":
        order = _stable_permutation(len(trials), seed, *cell_key)
        return [trials[int(index)] for index in order[:n]]
    raise ValueError(f"Unknown sampling mode: {sampling!r}")


def build_count_matrices(
    data: PanelTrials,
    n: int,
    sampling: str,
    seed: int,
    category: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Reaggregate selected trial records using the production denominator."""
    shape = (len(data.models), len(data.items))
    pass_counts = np.zeros(shape, dtype=np.float64)
    total_trials = np.zeros(shape, dtype=np.float64)
    for model_index, model in enumerate(data.models):
        for item_index, item in enumerate(data.items):
            selected = select_trials(
                data.trials[model][item],
                n,
                sampling,
                seed,
                category,
                model,
                item,
            )
            valid = [trial for trial in selected if trial.get("passed") is not None]
            pass_counts[model_index, item_index] = sum(
                trial.get("passed") is True for trial in valid
            )
            total_trials[model_index, item_index] = len(valid)
    return pass_counts, total_trials


def alo_mask(
    pass_counts: np.ndarray,
    total_trials: np.ndarray,
    lo: float = ALO_LO,
    hi: float = ALO_HI,
) -> np.ndarray:
    """Keep an item when at least one observed panel rate is in [lo, hi]."""
    p_hat = np.divide(
        pass_counts,
        total_trials,
        out=np.full_like(pass_counts, np.nan),
        where=total_trials > 0,
    )
    return np.any((p_hat >= lo) & (p_hat <= hi), axis=0)


def fit_filtered_rasch(
    pass_counts: np.ndarray,
    total_trials: np.ndarray,
    keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit Rasch on one condition and return theta, beta, and theta SE."""
    if not np.any(keep):
        raise ValueError("ALO filter retained no items")
    counts = pass_counts[:, keep]
    totals = total_trials[:, keep]
    if np.any(totals <= 0):
        raise ValueError("A retained model/item cell has no valid trials")
    p_hat = counts / totals
    theta_init, beta_init = fit_ols(p_hat)
    theta, beta, _ = fit_mle(counts, totals, theta_init, beta_init)
    theta_se = theta_standard_errors(theta, beta, totals)
    return theta, beta, theta_se


def theta_standard_errors(
    theta: np.ndarray,
    beta: np.ndarray,
    total_trials: np.ndarray,
) -> np.ndarray:
    """Efficient joint-MLE theta SE under the sum(theta)=0 constraint.

    Profiling the item parameters out of the Rasch Fisher matrix leaves an
    M x M information matrix for theta. Parameterizing the last theta as the
    negative sum of the first M-1 enforces the same gauge as ``fit_mle`` and
    reduces the required inverse to only (M-1) x (M-1).
    """
    n_models = len(theta)
    probabilities = _sigmoid(theta[:, None] - beta[None, :])
    weights = total_trials * probabilities * (1.0 - probabilities)
    item_information = weights.sum(axis=0)
    if np.any(item_information <= 0):
        raise ValueError("Cannot compute theta SE with a zero-information item")

    profile_information = np.diag(weights.sum(axis=1))
    profile_information -= (weights / item_information) @ weights.T

    constraint_map = np.zeros((n_models, n_models - 1))
    constraint_map[:-1, :] = np.eye(n_models - 1)
    constraint_map[-1, :] = -1.0
    free_information = constraint_map.T @ profile_information @ constraint_map
    free_covariance = np.linalg.pinv(free_information, rcond=1e-10)
    theta_covariance = constraint_map @ free_covariance @ constraint_map.T
    return np.sqrt(np.maximum(np.diag(theta_covariance), 0.0))


def filter_transition_metrics(
    keep: np.ndarray,
    reference_keep: np.ndarray,
) -> dict[str, float | int]:
    """Summarize filter membership changes using N=40 as the reference."""
    if keep.shape != reference_keep.shape:
        raise ValueError("Filter masks must have the same shape")
    retained = int(keep.sum())
    reference_retained = int(reference_keep.sum())
    common = int(np.logical_and(keep, reference_keep).sum())
    union = int(np.logical_or(keep, reference_keep).sum())
    pool = len(keep)
    dropped = reference_retained - common
    newly_admitted = retained - common
    reference_excluded = pool - reference_retained
    return {
        "n_pool_items": pool,
        "n_retained": retained,
        "n_excluded": pool - retained,
        "retention_rate": retained / pool,
        "n_reference_retained": reference_retained,
        "n_common_retained": common,
        "n_dropped_from_reference": dropped,
        "dropped_from_reference_rate": (
            dropped / reference_retained if reference_retained else float("nan")
        ),
        "n_newly_admitted": newly_admitted,
        "newly_admitted_rate": (
            newly_admitted / reference_excluded
            if reference_excluded
            else float("nan")
        ),
        "retained_jaccard": common / union if union else 1.0,
    }


def pipeline_metrics(
    *,
    category: str,
    n: int,
    seed: int,
    sampling: str,
    total_trials: np.ndarray,
    keep: np.ndarray,
    theta: np.ndarray,
    beta: np.ndarray,
    theta_se: np.ndarray,
    reference_keep: np.ndarray,
    reference_scale: dict[str, float],
) -> dict:
    """Compute filter-transition and fitted-scale metrics for one condition."""
    beta_q90_span = float(np.quantile(beta, 0.95) - np.quantile(beta, 0.05))
    beta_full_span = float(beta.max() - beta.min())
    theta_span = float(theta.max() - theta.min())
    median_theta_se = float(np.median(theta_se))
    row = {
        "category": category,
        "sampling": sampling,
        "seed": seed,
        "n": n,
        "n_cells_below_requested_valid_n": int(np.sum(total_trials < n)),
        "min_valid_trials": int(total_trials.min()),
        "beta_q90_span": beta_q90_span,
        "delta_beta_q90_span": beta_q90_span - reference_scale["beta_q90_span"],
        "beta_full_span": beta_full_span,
        "delta_beta_full_span": beta_full_span - reference_scale["beta_full_span"],
        "theta_span": theta_span,
        "delta_theta_span": theta_span - reference_scale["theta_span"],
        "median_theta_se": median_theta_se,
        "delta_median_theta_se": (
            median_theta_se - reference_scale["median_theta_se"]
        ),
    }
    row.update(filter_transition_metrics(keep, reference_keep))
    return row


def summarize_metrics(rows: Sequence[dict]) -> list[dict]:
    metric_names = (
        "n_retained",
        "n_excluded",
        "retention_rate",
        "n_common_retained",
        "n_dropped_from_reference",
        "dropped_from_reference_rate",
        "n_newly_admitted",
        "newly_admitted_rate",
        "retained_jaccard",
        "beta_q90_span",
        "delta_beta_q90_span",
        "beta_full_span",
        "delta_beta_full_span",
        "theta_span",
        "delta_theta_span",
        "median_theta_se",
        "delta_median_theta_se",
    )
    groups: dict[tuple[str, int, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["category"], row["n"], row["sampling"]), []).append(row)

    summaries = []
    for (category, n, sampling), group in sorted(groups.items()):
        summary = {
            "category": category,
            "n": n,
            "sampling": sampling,
            "n_replicates": len(group),
        }
        for metric in metric_names:
            values = np.asarray([row[metric] for row in group], dtype=float)
            summary[f"{metric}_mean"] = float(np.nanmean(values))
            summary[f"{metric}_std"] = (
                float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            )
            summary[f"{metric}_min"] = float(np.nanmin(values))
            summary[f"{metric}_max"] = float(np.nanmax(values))
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: Sequence[dict]) -> None:
    print(
        "\ncategory                 N    kept  dropped  added  Jaccard"
        "  beta90 span (delta)  theta span (delta)  median theta SE"
    )
    print("-" * 125)
    for row in rows:
        print(
            f"{row['category']:<24} {row['n']:>2}  "
            f"{row['n_retained_mean']:>7.1f}  "
            f"{row['n_dropped_from_reference_mean']:>7.1f}  "
            f"{row['n_newly_admitted_mean']:>5.1f}  "
            f"{row['retained_jaccard_mean']:>7.4f}  "
            f"{row['beta_q90_span_mean']:>8.3f} "
            f"({row['delta_beta_q90_span_mean']:+.3f})  "
            f"{row['theta_span_mean']:>8.3f} "
            f"({row['delta_theta_span_mean']:+.3f})  "
            f"{row['median_theta_se_mean']:>8.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--categories",
        default=",".join(CATEGORY_SOURCES),
        help="Comma-separated category names (default: all four).",
    )
    parser.add_argument(
        "--n-values",
        default=",".join(map(str, DEFAULT_N_VALUES)),
        help="Comma-separated smaller trial budgets (default: 10,20,30).",
    )
    parser.add_argument(
        "--sampling",
        choices=("prefix", "subsample"),
        default="subsample",
        help="Use reproducible random trial orderings (default) or take the first N trials.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(map(str, DEFAULT_SEEDS)),
        help="Comma-separated seeds (default: 0,...,19).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results" / "rebuttal" / "budget_sensitivity" / "repeated_trials_n",
    )
    args = parser.parse_args()

    categories = [part.strip() for part in args.categories.split(",") if part.strip()]
    unknown = sorted(set(categories) - set(CATEGORY_SOURCES))
    if unknown:
        parser.error(f"Unknown categories: {', '.join(unknown)}")
    n_values = parse_int_list(args.n_values, label="--n-values")
    if any(n <= 0 or n >= REFERENCE_N for n in n_values):
        parser.error(f"--n-values must be between 1 and {REFERENCE_N - 1}")
    seeds = parse_int_list(args.seeds, label="--seeds")
    if args.sampling == "prefix" and len(seeds) != 1:
        parser.error("Prefix sampling is deterministic; pass exactly one seed")

    panel = [entry["name"] for entry in PANEL_MODELS]
    metric_rows: list[dict] = []
    references: dict[str, dict] = {}

    for category in categories:
        source_dir = REPO / "results" / category / CATEGORY_SOURCES[category]
        print(f"\n[{category}] loading {source_dir.relative_to(REPO)}", flush=True)
        data = load_panel_trials(source_dir, panel)

        reference_counts, reference_totals = build_count_matrices(
            data, REFERENCE_N, "prefix", 0, category
        )
        reference_keep = alo_mask(reference_counts, reference_totals)
        reference_theta, reference_beta, reference_theta_se = fit_filtered_rasch(
            reference_counts, reference_totals, reference_keep
        )
        reference_scale = {
            "beta_q90_span": float(
                np.quantile(reference_beta, 0.95)
                - np.quantile(reference_beta, 0.05)
            ),
            "beta_full_span": float(reference_beta.max() - reference_beta.min()),
            "theta_span": float(reference_theta.max() - reference_theta.min()),
            "median_theta_se": float(np.median(reference_theta_se)),
        }
        references[category] = {
            "source_dir": str(source_dir.relative_to(REPO)),
            "n_pool_items": len(data.items),
            "n_reference_retained": int(reference_keep.sum()),
            "n_cells_below_reference_valid_n": int(
                np.sum(reference_totals < REFERENCE_N)
            ),
            "min_reference_valid_trials": int(reference_totals.min()),
            **reference_scale,
        }
        metric_rows.append(
            pipeline_metrics(
                category=category,
                n=REFERENCE_N,
                seed=-1,
                sampling=args.sampling,
                total_trials=reference_totals,
                keep=reference_keep,
                theta=reference_theta,
                beta=reference_beta,
                theta_se=reference_theta_se,
                reference_keep=reference_keep,
                reference_scale=reference_scale,
            )
        )
        print(
            f"  N=40 reference: pool={len(data.items)}, "
            f"ALO-kept={reference_keep.sum()}, "
            f"cells with <40 valid verdicts={np.sum(reference_totals < REFERENCE_N)}",
            flush=True,
        )

        for seed in seeds:
            for n in sorted(n_values):
                counts, totals = build_count_matrices(
                    data, n, args.sampling, seed, category
                )
                keep = alo_mask(counts, totals)
                theta, beta, theta_se = fit_filtered_rasch(counts, totals, keep)
                row = pipeline_metrics(
                    category=category,
                    n=n,
                    seed=seed,
                    sampling=args.sampling,
                    total_trials=totals,
                    keep=keep,
                    theta=theta,
                    beta=beta,
                    theta_se=theta_se,
                    reference_keep=reference_keep,
                    reference_scale=reference_scale,
                )
                metric_rows.append(row)
                print(
                    f"  seed={seed} N={n}: kept={row['n_retained']}, "
                    f"dropped={row['n_dropped_from_reference']}, "
                    f"added={row['n_newly_admitted']}, "
                    f"beta90_span={row['beta_q90_span']:.3f}, "
                    f"theta_span={row['theta_span']:.3f}",
                    flush=True,
                )

    summaries = summarize_metrics(metric_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics_by_seed.csv", metric_rows)
    write_csv(args.output_dir / "summary_by_category.csv", summaries)
    manifest = {
        "experiment": "HiES Q2.1 repeated-trials-per-cell sensitivity",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling": args.sampling,
        "seeds": seeds,
        "n_values": sorted(n_values),
        "reference_n": REFERENCE_N,
        "filter": {"rule": "alo", "lo": ALO_LO, "hi": ALO_HI},
        "estimand": "end-to-end ALO-filter plus Rasch-calibration sensitivity",
        "gauge": "sum(panel_theta)=0; all reported spans are translation invariant",
        "beta_span": {
            "primary": "95th percentile minus 5th percentile",
            "secondary": "maximum minus minimum",
        },
        "filter_reference": "N=40 retained set",
        "panel_models": panel,
        "categories": references,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print_summary(summaries)
    print(f"\nWrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
