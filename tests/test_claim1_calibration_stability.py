"""Regression checks for Claim 1 anchor selection and null simulation."""

import numpy as np

from scripts.analysis.claim1_calibration_stability import (
    alo_filter_mask,
    avg_filter_mask,
    null_bootstrap,
)


def test_alo_uses_individual_rates_and_includes_window_endpoints():
    rates = np.array([[0.0, 0.2, 0.8, 0.81], [1.0, 1.0, 0.0, 1.0]])
    # A split all-fail/all-pass item has an in-window mean but no in-window model.
    np.testing.assert_array_equal(alo_filter_mask(rates), [False, True, True, False])
    np.testing.assert_array_equal(avg_filter_mask(rates), [True, True, True, True])


def test_null_bootstrap_completes_all_metrics_with_fixed_anchor_dimensions():
    theta = np.array([-0.8, -0.2, 0.3, 0.7])
    beta = np.linspace(-1.0, 1.0, 12)
    trials = np.full((len(theta), len(beta)), 40)
    result = null_bootstrap(theta, beta, trials, ["a", "b", "c", "d"],
                            n_bootstrap=3, seed=17)
    assert result["seed"] == 17
    assert result["n_bootstrap"] == 3
    for key, summary in result["null"].items():
        assert summary["n"] == 3
        assert len(result["samples"][key]) == 3
        assert np.isfinite(result["samples"][key]).all()
        assert summary["p05"] <= summary["p50"] <= summary["p95"]
    assert result["null"]["median_abs_delta_beta_pooled"]["mean"] > 0
    assert result["null"]["abs_delta_theta_held_max"]["mean"] > 0
