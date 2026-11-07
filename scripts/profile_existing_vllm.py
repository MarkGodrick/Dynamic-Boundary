"""Profile a candidate model using an ALREADY-RUNNING vllm server.

Use case: when collect_response_matrix.py fails to start vllm for a certain
model (e.g., due to dtype/quantization flag incompatibility), start vllm
manually in another tmux pane and point this script at it.

It reuses collect_response_matrix's collect_responses() + create_judge() and
connects directly via APIModel, so the output format is identical and can
be consumed by fit_irt.py without any modification.

Usage:
    # Step 1: in another tmux pane, start vllm manually:
    CUDA_VISIBLE_DEVICES=0,1 HF_HOME=/data1/whx/.cache/huggingface \
        /data/whx/anaconda3/envs/vllm085/bin/vllm serve \
        Qwen/Qwen2.5-72B-Instruct-AWQ \
        --port 8300 --tensor-parallel-size 2

    # Step 2 (after vllm shows "Application startup complete"):
    python scripts/profile_existing_vllm.py \
        --model-name Qwen/Qwen2.5-72B-Instruct-AWQ \
        --server-url http://localhost:8300/v1 \
        --dataset benchmarks/constrained_completion/edge_case_dataset.jsonl \
        --output-dir results/constrained_completion/panel_profiling \
        --category constrained_completion \
        --n-trials 40
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.api_model import APIModel
from models.response_parser import get_parser

# Reuse logic from collect_response_matrix.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_response_matrix import (  # noqa: E402
    create_judge, load_dataset, collect_responses,
)


def main():
    parser = argparse.ArgumentParser(
        description="Profile a candidate model on an existing vllm server"
    )
    parser.add_argument("--model-name", required=True,
                        help="HF model id (used for API model field + output dir name)")
    parser.add_argument("--server-url", default="http://localhost:8300/v1",
                        help="Running vllm OpenAI-compatible endpoint")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--category", default="constrained_completion")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max new tokens per generation")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    # Health check the vllm server first (fail fast)
    try:
        import httpx
        health_url = args.server_url.rstrip("/").removesuffix("/v1") + "/health"
        r = httpx.get(health_url, timeout=5)
        if r.status_code != 200:
            print(f"ERROR: vllm health check failed ({health_url} → HTTP {r.status_code})")
            sys.exit(1)
        print(f"[OK] vllm server healthy at {args.server_url}")
    except Exception as e:
        print(f"ERROR: cannot reach vllm at {args.server_url}: {e}")
        print("  Make sure you started vllm manually in another pane and it printed")
        print("  'Application startup complete.' before running this script.")
        sys.exit(1)

    # Build APIModel pointing at the running vllm server
    os.environ.setdefault("VLLM_API_KEY", "dummy")
    api_model = APIModel(
        model=args.model_name,
        api_key_env="VLLM_API_KEY",
        base_url=args.server_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        response_parser=get_parser(args.model_name),
    )

    # Judge (same logic as collect_response_matrix)
    judge_config = {
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_TOKEN",
        "base_url": "https://api.deepseek.com",
    }
    judge = create_judge(args.category, judge_config)

    # Load dataset
    dataset = load_dataset(args.dataset)
    print(f"Dataset:           {len(dataset)} prompts")
    print(f"Trials per prompt: {args.n_trials}")
    print(f"Category:          {args.category}")
    print(f"Model:             {args.model_name}")
    print(f"Server:            {args.server_url}")

    # Output directory (same convention as collect_response_matrix)
    model_short = args.model_name.replace("/", "_").replace("-", "_")
    out_dir = Path(args.output_dir) / model_short
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "response_matrix.jsonl"

    print(f"\n{'='*60}")
    print(f"  {datetime.now().strftime('%H:%M:%S')} | Starting profiling")
    print(f"{'='*60}")

    # NOTE: for candidate profiling, source_model in dataset is one of the
    # existing panel models (never the candidate), so we evaluate ALL items
    # unconditionally — no own_entries / other_entries split needed.
    results = collect_responses(
        dataset=dataset,
        model=api_model,
        judge=judge,
        n_trials=args.n_trials,
        seed=args.seed,
        max_workers=args.max_workers,
    )

    # Fix model field to be the evaluating model (not source_model)
    for r in results:
        r["model"] = args.model_name

    # Sort by prompt_id for consistent output
    results.sort(key=lambda r: r["prompt_id"])

    # Save
    with open(result_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    meta = {
        "model": args.model_name,
        "category": args.category,
        "n_trials": args.n_trials,
        "seed": args.seed,
        "n_prompts": len(dataset),
        "server_url": args.server_url,
        "timestamp": datetime.now().isoformat(),
        "note": "collected via profile_existing_vllm.py (manual vllm server)",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    avg_p = sum(r["p_hat"] for r in results) / len(results)
    print(f"\n{'='*60}")
    print(f"  Done: avg p_hat={avg_p:.3f}")
    print(f"  Saved to {result_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
