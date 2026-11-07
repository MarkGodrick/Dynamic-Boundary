"""Shared per-model dispatch helpers for entry points (main.py, run_sgbs.py).

Resolves vllm fields from PANEL/VALIDATION_MODELS, validates GPU/TP
consistency, and locates existing run dirs for idempotent skip/resume
decisions. The "is this run complete?" predicate is left to each entry
point — main.py looks for result.json, run_sgbs.py looks for
search_state.json + edge_cases.jsonl count.
"""

from __future__ import annotations

from pathlib import Path

from logger import get_logger
from .panel_config import PANEL_MODELS, VALIDATION_MODELS


logger = get_logger(__name__)


# panel_config key -> yaml target_model key
PANEL_FIELDS = {
    "cuda_devices": "cuda_devices",
    "conda_env": "conda_env",
    "extra_args": "extra_args",
    "gpu_mem_util": "gpu_memory_utilization",
    "dtype": "dtype",
}

# panel_config wins (yaml ignored) for these when the model has a panel entry.
# cuda_devices is intentionally not authoritative — it's a per-experiment GPU
# placement choice the user sets in yaml without editing panel_config. The
# TP-consistency check catches cuda count vs --tensor-parallel-size mismatches.
PANEL_AUTHORITATIVE = {"extra_args", "conda_env", "gpu_memory_utilization", "dtype"}


def model_short(name: str) -> str:
    return name.replace("/", "_").replace("-", "_")


def check_tp_consistency(target: dict) -> None:
    """Raise if cuda_devices count != TP × DP × PP product.

    vLLM needs CUDA_VISIBLE_DEVICES to expose exactly TP×DP×PP GPUs. Yaml's
    cuda_devices commonly desyncs from panel_config's extra_args parallelism
    settings; vLLM's own error ("world size mismatch") is opaque. Catch it here.

    Recognized flags (long + short forms):
      --tensor-parallel-size / -tp
      --data-parallel-size / -dp
      --pipeline-parallel-size / -pp
    """
    cuda = target.get("cuda_devices")
    extra = target.get("extra_args") or []

    n_gpus = (
        len([x for x in cuda.split(",") if x.strip()])
        if isinstance(cuda, str) and cuda.strip()
        else None
    )

    def _flag_int(*names: str, default: int = 1) -> int:
        for i, tok in enumerate(extra):
            if tok in names and i + 1 < len(extra):
                try:
                    return int(extra[i + 1])
                except (TypeError, ValueError):
                    pass
        return default

    tp = _flag_int("--tensor-parallel-size", "-tp")
    dp = _flag_int("--data-parallel-size", "-dp")
    pp = _flag_int("--pipeline-parallel-size", "-pp")
    expected = tp * dp * pp

    if expected == 1 and n_gpus in (None, 1):
        return
    parallel_str = f"TP={tp}×DP={dp}×PP={pp}={expected}"
    if n_gpus is None:
        raise ValueError(
            f"target_model has {parallel_str} GPUs in extra_args but no "
            f"cuda_devices set. Set target_model.cuda_devices to a comma-separated "
            f"list of {expected} GPU IDs."
        )
    if n_gpus != expected:
        raise ValueError(
            f"target_model parallel mismatch: cuda_devices='{cuda}' has {n_gpus} "
            f"GPU(s) but extra_args specify {parallel_str}. These must match. "
            f"Either fix yaml's cuda_devices to list {expected} GPU IDs, or remove "
            f"cuda_devices from yaml to inherit panel_config's default."
        )


def resolve_target_model(yaml_target: dict, model_name: str | None) -> dict:
    """Build the effective target_model config.

    Merge order:
        1. panel_config defaults for `model_name` (if in PANEL/VALIDATION)
        2. Yaml fields, with one exception:
           Fields in PANEL_AUTHORITATIVE (extra_args, conda_env,
           gpu_memory_utilization, dtype) are panel-only when the model has a
           panel_config entry — yaml's value is ignored. This prevents a yaml
           configured for one model from leaking incompatible flags to other
           models in a sequential run. cuda_devices is yaml-controlled.
        3. `model_name` overrides `name`.
    """
    name = model_name or yaml_target.get("name")
    target: dict = {}
    if not name:
        target.update(yaml_target)
        return target

    entry = next(
        (m for m in (*PANEL_MODELS, *VALIDATION_MODELS) if m.get("name") == name),
        None,
    )
    if entry is not None:
        target["name"] = name
        for src_key, dst_key in PANEL_FIELDS.items():
            if src_key in entry:
                target[dst_key] = entry[src_key]
    else:
        logger.info(
            "Target %s not in PANEL/VALIDATION; relying on yaml target_model only.",
            name,
        )

    ignored: list[str] = []
    for k, v in yaml_target.items():
        if entry is not None and k in PANEL_AUTHORITATIVE:
            ignored.append(k)
            continue
        target[k] = v
    if ignored:
        logger.info(
            "Yaml field(s) %s ignored for %s (panel_config is authoritative)",
            sorted(ignored), name,
        )
    target["name"] = name

    if entry is not None:
        logger.info(
            "Effective target_model for %s: %s",
            name,
            ", ".join(f"{k}={target[k]!r}" for k in PANEL_FIELDS.values() if k in target),
        )

    if target.get("type") == "vllm":
        check_tp_consistency(target)
    return target


def find_existing_run_dir(base_dir: Path, prefix: str) -> Path | None:
    """Return the latest run dir under base_dir whose name starts with prefix.

    Convention: run dirs end in `_YYYYMMDD_HHMMSS`, so lexicographic sort over
    same-prefix matches is chronological. Caller decides what constitutes a
    "complete" or "resumable" run by inspecting the returned dir.
    """
    if not base_dir.is_dir():
        return None
    matches = [
        d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)
    ]
    if not matches:
        return None
    matches.sort(key=lambda p: p.name)
    return matches[-1]
