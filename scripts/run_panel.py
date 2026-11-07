"""Run SGBS v2 across multiple models sequentially.

For each model, generates a temporary config with model-specific vLLM
parameters, then calls main.py.

Usage:
    python scripts/run_panel.py --config configs/constrained/search_constrained.yaml --verbose
    python scripts/run_panel.py --config configs/constrained/search_constrained.yaml --include-validation
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Model configurations with vLLM-specific parameters
# ---------------------------------------------------------------------------

PANEL_MODELS = [
    # {
    #     "name": "google/gemma-3-4b-it",
    #     "cuda_devices": "4,5",
    #     "port": 8200,
    #     "gpu_memory_utilization": 0.9,
    #     "extra_args": ["--tensor-parallel-size", "2"],
    # },
    # {
    #     "name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    #     "cuda_devices": "4,5",
    #     "port": 8200,
    #     "gpu_memory_utilization": 0.9,
    #     "extra_args": ["--tensor-parallel-size", "2"],
    # },
    # {
    #     "name": "google/gemma-3-12b-it",
    #     "cuda_devices": "4,5",
    #     "port": 8200,
    #     "gpu_memory_utilization": 0.9,
    #     "extra_args": ["--tensor-parallel-size", "2"],
    # },
    # {
    #     "name": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    #     "cuda_devices": "4,5",
    #     "port": 8200,
    #     "gpu_memory_utilization": 0.9,
    #     "extra_args": ["--enforce-eager", "--tensor-parallel-size", "2"],
    # },
    # {
    #     "name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    #     "cuda_devices": "4,5",
    #     "port": 8200,
    #     "gpu_memory_utilization": 0.9,
    #     "extra_args": ["--enforce-eager", "--tensor-parallel-size", "2"],
    # },
    {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "cuda_devices": "0,1",
        "port": 8200,
        "gpu_memory_utilization": 0.95,
        "server_timeout": 1200,
        "extra_args": ["--enforce-eager", "--tensor-parallel-size", "2"],
    },
]

VALIDATION_MODELS = [
    {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "cuda_devices": "0",
        "port": 8100,
        "gpu_memory_utilization": 0.9,
        "extra_args": ["--enforce-eager"],
    },
    {
        "name": "Qwen/Qwen3-8B",
        "cuda_devices": "0",
        "port": 8100,
        "gpu_memory_utilization": 0.9,
        "extra_args": [],
    },
    {
        "name": "mistralai/Mistral-Nemo-Instruct-2407",
        "cuda_devices": "0",
        "port": 8100,
        "gpu_memory_utilization": 0.9,
        "extra_args": [],
    },
    {
        "name": "google/gemma-3-27b-it",
        "cuda_devices": "0",
        "port": 8100,
        "gpu_memory_utilization": 0.85,
        "extra_args": ["--enforce-eager"],
    },
]


# ---------------------------------------------------------------------------
# vLLM cleanup helpers
# ---------------------------------------------------------------------------

def kill_vllm_on_port(port: int, timeout: int = 10) -> None:
    """Find and kill any vLLM processes listening on the given port."""
    try:
        result = subprocess.run(
            ["lsof", "-t", "-i", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [int(p) for p in result.stdout.strip().split() if p.strip()]
    except (subprocess.TimeoutExpired, ValueError):
        pids = []

    if not pids:
        return

    print(f"  Cleaning up {len(pids)} residual process(es) on port {port}: {pids}")

    # SIGTERM first
    for pid in pids:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

    # Wait, then SIGKILL stragglers
    time.sleep(min(timeout, 5))
    for pid in pids:
        try:
            os.kill(pid, 0)  # check if still alive
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def make_model_config(base_config: dict, model_cfg: dict) -> dict:
    """Create a per-model config by merging base config with model overrides."""
    config = {k: dict(v) if isinstance(v, dict) else v for k, v in base_config.items()}

    tm = dict(config.get("target_model", {}))
    tm["name"] = model_cfg["name"]
    tm["cuda_devices"] = model_cfg.get("cuda_devices", "0")
    tm["gpu_memory_utilization"] = model_cfg.get("gpu_memory_utilization", 0.9)
    tm["extra_args"] = model_cfg.get("extra_args", [])

    if "port" in model_cfg:
        tm["port"] = model_cfg["port"]
    if "conda_env" in model_cfg:
        tm["conda_env"] = model_cfg["conda_env"]
    if "server_timeout" in model_cfg:
        tm["server_timeout"] = model_cfg["server_timeout"]

    config["target_model"] = tm
    return config


def find_resumable_dir(
    base_dir: str, model_name: str, category: str, budget: int,
) -> Path | None:
    """Find the latest existing results directory that can be resumed.

    Returns the directory Path if found and not yet finished, else None.
    """
    model_short = model_name.replace("/", "_").replace("-", "_")
    pattern = f"{model_short}_{category}_*"
    candidates = sorted(Path(base_dir).glob(pattern), reverse=True)  # newest first

    for d in candidates:
        state_file = d / "search_state.json"
        if not state_file.exists():
            continue
        with open(state_file) as f:
            state = json.load(f)
        completed_round = state.get("round_num", 0)
        if completed_round < budget:
            return d
        # Already finished — skip
    return None


def run_model(
    model_cfg: dict, base_config: dict, verbose: bool, resume: bool = False,
) -> bool:
    """Run SGBS for a single model via subprocess with a temp config."""
    model_name = model_cfg["name"]
    config = make_model_config(base_config, model_cfg)

    port = model_cfg.get("port", 8100)

    # Check for resumable run
    resume_dir = None
    if resume:
        base_dir = config.get("experiment", {}).get("output_dir", "results/")
        category = config.get("bare_requests", {}).get("category", "")
        budget = config.get("search", {}).get("budget", 100)
        resume_dir = find_resumable_dir(base_dir, model_name, category, budget)

    if resume_dir:
        # Resume: main.py reads config from saved results dir
        with open(resume_dir / "search_state.json") as f:
            prev_round = json.load(f)["round_num"]
        tmp = None
        cmd = [sys.executable, "main.py", "--continue-from", str(resume_dir)]
        mode_str = f"RESUME from round {prev_round}"
    else:
        # Fresh start: write temp config with model-specific overrides
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="sgbs_", delete=False,
        )
        yaml.dump(config, tmp, default_flow_style=False, allow_unicode=True)
        tmp.close()
        cmd = [sys.executable, "main.py", "--config", tmp.name]
        mode_str = "FRESH"

    if verbose:
        cmd.append("--verbose")

    print(f"\n{'='*60}")
    print(f"  {datetime.now().strftime('%H:%M:%S')} | {mode_str}: {model_name}")
    print(f"  vLLM: cuda={model_cfg.get('cuda_devices', '0')}, "
          f"mem={model_cfg.get('gpu_memory_utilization', 0.9)}, "
          f"extra={model_cfg.get('extra_args', [])}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=False)
    except KeyboardInterrupt:
        print(f"  Interrupted: {model_name}")
        result = None
    finally:
        # Clean up temp config (None when resuming)
        if tmp is not None:
            Path(tmp.name).unlink(missing_ok=True)
        # Defensive cleanup: kill any vLLM left on port
        kill_vllm_on_port(port)
        time.sleep(3)  # brief cooldown for GPU memory release

    if result is None:
        raise KeyboardInterrupt  # re-raise after cleanup

    if result.returncode != 0:
        print(f"  FAILED: {model_name} (exit code {result.returncode})")
        return False

    print(f"  {datetime.now().strftime('%H:%M:%S')} | Completed: {model_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run SGBS across panel models")
    parser.add_argument("--config", required=True, help="Base config YAML")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--include-validation", action="store_true",
                        help="Also run validation models")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to run (default: all panel)")
    parser.add_argument("--resume", action="store_true",
                        help="Auto-detect and resume from previous incomplete runs")
    args = parser.parse_args()

    base_config = yaml.safe_load(open(args.config))

    all_models = list(PANEL_MODELS)
    if args.include_validation:
        all_models.extend(VALIDATION_MODELS)

    if args.models:
        filter_names = {n.strip() for n in args.models.split(",")}
        all_models = [m for m in all_models if m["name"] in filter_names]

    print(f"Running SGBS on {len(all_models)} models")
    print(f"Base config: {args.config}")

    results = {}
    try:
        for model_cfg in all_models:
            ok = run_model(model_cfg, base_config, args.verbose, args.resume)
            results[model_cfg["name"]] = "OK" if ok else "FAILED"
    except KeyboardInterrupt:
        print(f"\n  Interrupted by user. Cleaning up...")
        # kill_vllm_on_port already ran in run_model's finally block
        results[model_cfg["name"]] = "INTERRUPTED"

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for model, status in results.items():
        print(f"  {status:6s} | {model}")


if __name__ == "__main__":
    main()
