"""Resolve the canonical filter_ablation results directory for a category.

After 2026-04-26 judge rejudge, HR/OR data lives in `filter_ablation_raw_v2_judgefix/`
(post-fix), while CC (rule-based, no LLM judge) and SR (not yet collected at v2)
remain at `filter_ablation_raw_v2/`.

Use this helper from any analysis script that consumes per-category response
matrices. Hardcoding paths in scripts leads to silent drift when directory
conventions change.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"

_V2_JUDGEFIX = "filter_ablation_raw_v2_judgefix"
_V2 = "filter_ablation_raw_v2"

_CAT_TO_SUBDIR = {
    "harmful_refusal":         _V2_JUDGEFIX,
    "over_refusal":            _V2_JUDGEFIX,
    "constrained_completion":  _V2,
    "sycophancy_resistance":   _V2,
}


def filter_ablation_dir(category: str) -> Path:
    """Return absolute path to the canonical filter_ablation dir for `category`."""
    if category not in _CAT_TO_SUBDIR:
        raise KeyError(f"unknown category: {category!r}")
    return RESULTS / category / _CAT_TO_SUBDIR[category]
