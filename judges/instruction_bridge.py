"""Unified bridge for IFEval + IFBench verifier classes.

Both IFEval (25 types, Zhou et al. 2023) and IFBench (58 types, Pyatkin et al.
2025) ship Python Instruction subclasses with identical interface:

    instance = Cls(instruction_id=...)
    description = instance.build_description(**kwargs)   # sets state, returns text
    passed     = instance.check_following(response)      # reads state, returns bool

This module lazy-loads both registries from
skills/constrained_completion/verifiers/ and exposes a single uniform API:

    describe(type_name, kwargs) -> str
    check(type_name, kwargs, response) -> bool
    type_names() -> frozenset[str]   # all 83 types across both sources

Load order matters: IFEval must be loaded as a proper Python package via the
`instruction_following_eval` symlink BEFORE we inject the ifbench/ directory
into sys.path, because IFBench's instructions_registry.py does `import
instructions` at top level and would otherwise shadow IFEval's internal
submodule.
"""

from __future__ import annotations

import sys
import warnings
from functools import lru_cache
from pathlib import Path

from logger import get_logger

logger = get_logger(__name__)

VERIFIERS_DIR = (
    Path(__file__).resolve().parent.parent
    / "skills"
    / "constrained_completion"
    / "verifiers"
)
IFBENCH_DIR = VERIFIERS_DIR / "ifbench"


@lru_cache(maxsize=1)
def _load_registries() -> tuple[dict, dict]:
    """Return (IFEVAL_DICT, IFBENCH_DICT), loading both modules once."""
    # syllapy (imported transitively by ifbench/instructions.py) uses the
    # deprecated pkg_resources API and emits a UserWarning. Suppress only
    # during the one-shot upstream load; warning scope stays narrow.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*pkg_resources is deprecated.*",
            category=UserWarning,
        )
        # 1) IFEval via package form (symlink verifiers/instruction_following_eval → ifeval)
        if str(VERIFIERS_DIR) not in sys.path:
            sys.path.insert(0, str(VERIFIERS_DIR))
        from instruction_following_eval.instructions_registry import (  # type: ignore[import-not-found]
            INSTRUCTION_DICT as IFEVAL_DICT,
        )

        # 2) IFBench via top-level sys.path injection
        if str(IFBENCH_DIR) not in sys.path:
            sys.path.insert(0, str(IFBENCH_DIR))
        import instructions  # type: ignore[import-not-found]  # noqa: F401
        from instructions_registry import INSTRUCTION_DICT as IFBENCH_DICT  # type: ignore[import-not-found]

    return IFEVAL_DICT, IFBENCH_DICT


@lru_cache(maxsize=1)
def type_names() -> frozenset[str]:
    ife, ifb = _load_registries()
    return frozenset(ife.keys()) | frozenset(ifb.keys())


def _lookup_class(type_name: str):
    ife, ifb = _load_registries()
    if type_name in ife:
        return ife[type_name]
    if type_name in ifb:
        return ifb[type_name]
    raise KeyError(f"Unknown constraint type: {type_name!r}")


def _instantiate(type_name: str, kwargs: dict):
    """Build an Instruction instance and prime it via build_description.

    Filters None/empty values out of kwargs (both IFEval and IFBench dataset
    schemas store `None` for unused fields). Falls back to `build_description()`
    with no args if the filtered kwargs are rejected.
    """
    cls = _lookup_class(type_name)
    instance = cls(instruction_id=type_name)
    filtered = {
        k: v
        for k, v in (kwargs or {}).items()
        if v is not None and v != "" and v != []
    }
    try:
        desc = instance.build_description(**filtered)
    except TypeError as e:
        logger.warning(
            "%s build_description rejected kwargs %s: %s", type_name, filtered, e
        )
        desc = instance.build_description()
    return instance, desc


def describe(type_name: str, kwargs: dict | None = None) -> str:
    """Return the natural-language constraint description for any type (IFEval or IFBench)."""
    _, desc = _instantiate(type_name, kwargs or {})
    return desc


def check(type_name: str, kwargs: dict | None, response: str) -> bool:
    """Evaluate whether `response` satisfies a constraint of the given type."""
    instance, _ = _instantiate(type_name, kwargs or {})
    try:
        return bool(instance.check_following(response))
    except Exception as e:
        logger.warning("%s check_following raised %s, treating as FAIL", type_name, e)
        return False
