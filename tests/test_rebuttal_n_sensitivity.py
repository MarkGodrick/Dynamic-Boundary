from __future__ import annotations

import numpy as np

from scripts.analysis.rebuttal_n_sensitivity import (
    alo_mask,
    filter_transition_metrics,
    select_trials,
    summarize_metrics,
    theta_standard_errors,
)


def test_prefix_selection_uses_first_n_trials():
    trials = [{"passed": value} for value in (True, False, None, True)]
    assert select_trials(trials, 2, "prefix", 42, "cell") == trials[:2]


def test_seeded_subsamples_are_deterministic_and_nested():
    trials = [{"passed": bool(index % 2), "index": index} for index in range(40)]
    first_10 = select_trials(trials, 10, "subsample", 7, "category", "model", "item")
    first_20 = select_trials(trials, 20, "subsample", 7, "category", "model", "item")
    repeated = select_trials(trials, 10, "subsample", 7, "category", "model", "item")

    assert first_10 == repeated
    assert first_10 == first_20[:10]
    assert len({trial["index"] for trial in first_20}) == 20


def test_alo_filter_requires_an_observed_rate_in_band():
    pass_counts = np.asarray([[0, 5, 0], [10, 10, 0]], dtype=float)
    total_trials = np.asarray([[10, 10, 0], [10, 10, 0]], dtype=float)
    assert alo_mask(pass_counts, total_trials).tolist() == [False, True, False]


def test_filter_transition_metrics_distinguish_dropped_and_new_items():
    reference = np.asarray([True, True, False, False])
    current = np.asarray([True, False, True, False])
    metrics = filter_transition_metrics(current, reference)

    assert metrics["n_retained"] == 2
    assert metrics["n_excluded"] == 2
    assert metrics["n_dropped_from_reference"] == 1
    assert metrics["n_newly_admitted"] == 1
    assert metrics["retained_jaccard"] == 1 / 3


def test_theta_standard_errors_scale_with_trial_information():
    theta = np.asarray([-0.5, 0.0, 0.5])
    beta = np.asarray([-1.0, -0.25, 0.25, 1.0])
    n10 = np.full((3, 4), 10.0)
    n40 = np.full((3, 4), 40.0)

    se10 = theta_standard_errors(theta, beta, n10)
    se40 = theta_standard_errors(theta, beta, n40)

    assert np.all(se10 > 0)
    assert np.allclose(se40, se10 / 2)


def test_summary_reports_cross_seed_mean_and_sample_std():
    base = {
        "category": "test",
        "n": 10,
        "sampling": "subsample",
        "n_retained": 2,
        "n_excluded": 2,
        "retention_rate": 0.5,
        "n_common_retained": 2,
        "n_dropped_from_reference": 0,
        "dropped_from_reference_rate": 0.0,
        "n_newly_admitted": 0,
        "newly_admitted_rate": 0.0,
        "retained_jaccard": 1.0,
        "delta_beta_q90_span": 0.2,
        "beta_full_span": 3.0,
        "delta_beta_full_span": 0.3,
        "theta_span": 2.0,
        "delta_theta_span": 0.1,
        "median_theta_se": 0.2,
        "delta_median_theta_se": 0.1,
    }
    rows = [
        {**base, "beta_q90_span": 1.0},
        {**base, "beta_q90_span": 1.4},
    ]
    summary = summarize_metrics(rows)[0]
    assert summary["beta_q90_span_mean"] == np.float64(1.2)
    assert np.isclose(
        summary["beta_q90_span_std"],
        np.std([1.0, 1.4], ddof=1),
    )
