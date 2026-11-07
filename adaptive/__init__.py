"""Adaptive evaluation on a calibrated logit scale.

Step 1 uses Fisher-information CAT and EAP or MAP/Laplace ability updates.
Step 2 uses SGBS and reference-panel calibration to augment anchor coverage.
"""

from .anchor_bundle import (
    AnchorBundle, AnchorItem,
    load_anchor_bundle, save_anchor_bundle,
)
from .posterior import (
    PosteriorEstimator,
    EAPEstimator, MAPLaplaceEstimator,
    build_estimator,
)
from .cat_select import fisher_info_select, pilot_indices
from .coverage import CoverageState, coverage_failed, should_trigger
from .calibrate import Step2NotImplemented, step2_calibrate
from .adaptive_eval import (
    AdaptiveStep, AdaptiveResult,
    run_adaptive_eval, save_adaptive_result,
)

__all__ = [
    # anchor bundle
    "AnchorBundle", "AnchorItem", "load_anchor_bundle", "save_anchor_bundle",
    # posterior
    "PosteriorEstimator", "EAPEstimator", "MAPLaplaceEstimator", "build_estimator",
    # CAT + coverage
    "fisher_info_select", "pilot_indices",
    "CoverageState", "coverage_failed", "should_trigger",
    # Step-2 calibration
    "Step2NotImplemented", "step2_calibrate",
    # main loop
    "AdaptiveStep", "AdaptiveResult", "run_adaptive_eval", "save_adaptive_result",
]
