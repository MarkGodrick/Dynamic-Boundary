"""Comprehensive truncation / length / vLLM-slot audit.

Outputs four sections:
  [1] Truncation rate × pass rate per (category, model) from filter_ablation
  [2] Random-composed prompt token-length distribution per category (Qwen2.5
      tokenizer as reference) — table + PNG histogram
  [3] target-model completion_tokens distribution per (category, model) —
      table + PNG histogram
  [4] Live vLLM server slot/KV/preemption stats — discovered by scanning
      processes and querying /metrics

Run:
  conda run -n boundary --no-capture-output python -u scripts/analysis/truncation_audit.py

Outputs:
  stdout: tables + summary
  results/analysis/truncation_audit/prompt_lengths_hist.png
  results/analysis/truncation_audit/completion_tokens_hist.png
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.result_paths import filter_ablation_dir

CATEGORIES = [
    "harmful_refusal",
    "over_refusal",
    "constrained_completion",
    "sycophancy_resistance",
]

PANEL_DIR_NAMES = [
    "Qwen_Qwen2.5_72B_Instruct_AWQ",
    "Qwen_Qwen3_30B_A3B_Instruct_2507",
    "google_gemma_3_27b_it",
    "Qwen_Qwen3_8B",
    "google_gemma_3_12b_it",
    "mistralai_Mistral_Small_3.1_24B_Instruct_2503",
    "google_gemma_3_4b_it",
    "mistralai_Mistral_Nemo_Instruct_2407",
    "deepseek_ai_DeepSeek_R1_Distill_Llama_8B",
]

# Reference tokenizer for prompt length stats. Qwen tokenizers are close to
# the median across our panel; results should be within ±10% of any other
# panel model's tokenizer.
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
BENCH_ROOT = REPO_ROOT / "benchmarks"
OUT_DIR = RESULTS_ROOT / "analysis" / "truncation_audit"


# ---------------------------------------------------------------------------
# [1] Truncation × pass-rate
# ---------------------------------------------------------------------------


def section1_truncation_pass() -> None:
    print("\n" + "=" * 80)
    print("[1] Truncation rate & pass rate (truncated vs non-truncated)")
    print("=" * 80)
    print(
        "Note: SR data has no per-trial 'truncated' field (older schema); "
        "we approximate via response length > 4000 chars (≈ near 1024-token cap)."
    )
    print()

    header = (
        f"{'category':<25}{'model':<48}"
        f"{'n_trials':>10}{'trunc%':>8}"
        f"{'pass(t)':>9}{'pass(nt)':>10}{'Δ_pass':>9}"
    )

    for cat in CATEGORIES:
        print(f"\n--- {cat} ---")
        print(header)
        print("-" * len(header))
        cat_dir = filter_ablation_dir(cat)
        if not cat_dir.exists():
            print(f"  (missing dir: {cat_dir})")
            continue
        sub_dirs = sorted([d for d in cat_dir.iterdir() if d.is_dir()])
        for sub in sub_dirs:
            f = sub / "response_matrix.jsonl"
            if not f.exists():
                continue
            n_trials = 0
            n_trunc = 0
            n_pass_t = 0
            n_pass_nt = 0
            n_t = 0
            n_nt = 0
            for line in f.open():
                r = json.loads(line)
                for t in r["trials"]:
                    n_trials += 1
                    if "truncated" in t and t["truncated"] is not None:
                        # New-schema trials have an explicit field
                        is_trunc = bool(t["truncated"])
                    elif "responses" in t and t["responses"]:
                        # SR multi-turn: any per-turn response > 4000 chars
                        # → almost certainly hit max_tokens=1024
                        is_trunc = any(len(x) > 4000 for x in t["responses"])
                    elif isinstance(t.get("response"), str):
                        # HR/OR/CC old schema: single response, char heuristic
                        is_trunc = len(t["response"]) > 4000
                    else:
                        is_trunc = False

                    is_pass = bool(t.get("passed"))
                    if is_trunc:
                        n_trunc += 1
                        n_t += 1
                        if is_pass:
                            n_pass_t += 1
                    else:
                        n_nt += 1
                        if is_pass:
                            n_pass_nt += 1
            trunc_pct = 100.0 * n_trunc / n_trials if n_trials else 0.0
            pt = (n_pass_t / n_t) if n_t else float("nan")
            pnt = (n_pass_nt / n_nt) if n_nt else float("nan")
            delta = (pt - pnt) if (n_t and n_nt) else float("nan")
            print(
                f"{cat:<25}{sub.name:<48}"
                f"{n_trials:>10}{trunc_pct:>7.1f}%"
                f"{pt:>9.3f}{pnt:>10.3f}{delta:>+9.3f}"
            )


# ---------------------------------------------------------------------------
# [2] Prompt length distribution
# ---------------------------------------------------------------------------


def section2_prompt_lengths() -> dict:
    print("\n" + "=" * 80)
    print(f"[2] Prompt token-length distribution (tokenizer: {TOKENIZER_NAME})")
    print("=" * 80)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

    # Find a random-composed pool file per category.
    pool_files = {}
    for cat in CATEGORIES:
        d = BENCH_ROOT / cat
        if not d.exists():
            continue
        # Prefer filter_ablation_pool_k500 (current spec); fall back gracefully.
        cands = [
            "filter_ablation_pool_k500.jsonl",
            "random_baseline_k500.jsonl",
            "random_baseline.jsonl",
            "random_baseline_k2000.jsonl",
        ]
        for c in cands:
            p = d / c
            if p.exists():
                pool_files[cat] = p
                break

    print(
        f"{'category':<25}{'pool_file':<40}"
        f"{'n':>6}{'P50':>7}{'P95':>7}{'P99':>7}{'max':>7}"
    )
    print("-" * 99)
    cat_lens: dict[str, list[int]] = {}
    for cat in CATEGORIES:
        if cat not in pool_files:
            print(f"{cat:<25}{'(no pool file)':<40}")
            cat_lens[cat] = []
            continue
        f = pool_files[cat]
        prompts = [json.loads(line)["prompt"] for line in f.open()]
        toks = [len(tok.encode(p, add_special_tokens=False)) for p in prompts]
        toks_sorted = sorted(toks)
        n = len(toks_sorted)

        def pct(p: float) -> int:
            return toks_sorted[min(n - 1, int(n * p))]

        cat_lens[cat] = toks
        print(
            f"{cat:<25}{f.name:<40}"
            f"{n:>6}{pct(0.50):>7}{pct(0.95):>7}{pct(0.99):>7}{toks_sorted[-1]:>7}"
        )

    # Plot 4 histograms in one figure
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, cat in zip(axes.flat, CATEGORIES):
        toks = cat_lens.get(cat, [])
        if not toks:
            ax.set_title(f"{cat}\n(no data)")
            continue
        ax.hist(toks, bins=40, color="steelblue", edgecolor="black", linewidth=0.3)
        ax.axvline(np.percentile(toks, 95), color="orange", linestyle="--",
                   linewidth=1, label=f"P95={int(np.percentile(toks, 95))}")
        ax.axvline(np.percentile(toks, 99), color="red", linestyle="--",
                   linewidth=1, label=f"P99={int(np.percentile(toks, 99))}")
        ax.set_title(f"{cat} (n={len(toks)})")
        ax.set_xlabel("prompt tokens")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle(f"Random-composed prompt length (tokenizer: {TOKENIZER_NAME})")
    fig.tight_layout()
    out = OUT_DIR / "prompt_lengths_hist.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\nSaved: {out}")
    return cat_lens


# ---------------------------------------------------------------------------
# [3] Completion-token distribution
# ---------------------------------------------------------------------------


def _extract_completion_tokens(t: dict) -> list[int]:
    """Pull all available completion_token counts from a trial.

    HR/OR/CC new schema: top-level 't.completion_tokens' is the per-trial total
    (single-turn). SR (and any multi-turn): per-turn counts in
    'per_turn_meta'. Returns [] if neither available."""
    out: list[int] = []
    if isinstance(t.get("completion_tokens"), int):
        out.append(t["completion_tokens"])
    for m in t.get("per_turn_meta") or []:
        if isinstance(m.get("completion_tokens"), int):
            out.append(m["completion_tokens"])
    return out


def section3_completion_tokens() -> dict:
    print("\n" + "=" * 80)
    print("[3] Completion-token distribution (target model output)")
    print("=" * 80)
    print("Note: many older runs have no completion_tokens field — those models "
          "show n=0 and are skipped.")

    print(
        f"\n{'category':<25}{'model':<48}"
        f"{'n':>8}{'P50':>7}{'P95':>7}{'P99':>7}{'max':>7}"
    )
    print("-" * 110)

    cat_model_tokens: dict[tuple[str, str], list[int]] = {}
    for cat in CATEGORIES:
        cat_dir = filter_ablation_dir(cat)
        if not cat_dir.exists():
            continue
        for sub in sorted(cat_dir.iterdir()):
            f = sub / "response_matrix.jsonl"
            if not f.exists():
                continue
            tokens: list[int] = []
            for line in f.open():
                r = json.loads(line)
                for t in r["trials"]:
                    tokens.extend(_extract_completion_tokens(t))
            if not tokens:
                print(f"{cat:<25}{sub.name:<48}{0:>8}{'-':>7}{'-':>7}{'-':>7}{'-':>7}")
                continue
            tokens_sorted = sorted(tokens)
            n = len(tokens_sorted)

            def pct(p):
                return tokens_sorted[min(n - 1, int(n * p))]

            cat_model_tokens[(cat, sub.name)] = tokens
            print(
                f"{cat:<25}{sub.name:<48}"
                f"{n:>8}{pct(0.50):>7}{pct(0.95):>7}{pct(0.99):>7}{tokens_sorted[-1]:>7}"
            )

    # Plot: one subplot per category showing histograms of *all* models with
    # data, color-coded.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, cat in zip(axes.flat, CATEGORIES):
        models_with_data = [
            (m, cat_model_tokens[(cat, m)])
            for m in PANEL_DIR_NAMES
            if (cat, m) in cat_model_tokens
        ]
        if not models_with_data:
            ax.set_title(f"{cat}\n(no models with completion_tokens)")
            continue
        cmap = plt.colormaps["tab10"]
        bins = np.linspace(0, max(max(v) for _, v in models_with_data), 50)
        for i, (m, vals) in enumerate(models_with_data):
            ax.hist(
                vals, bins=bins, alpha=0.45,
                label=m.replace("_", " ")[:30],
                color=cmap(i % 10), edgecolor="none",
            )
        ax.set_title(f"{cat}")
        ax.set_xlabel("completion tokens")
        ax.set_ylabel("count")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Target-model completion_tokens distribution per (category, model)")
    fig.tight_layout()
    out = OUT_DIR / "completion_tokens_hist.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\nSaved: {out}")
    return cat_model_tokens


# ---------------------------------------------------------------------------
# [4] Live vLLM server status
# ---------------------------------------------------------------------------


def _discover_vllm_ports() -> list[tuple[int, str, dict]]:
    """Scan running processes for `vllm serve` and extract (port, model, args).

    Returns list of (port, model_name, parsed_args_dict) tuples.
    """
    out = subprocess.run(
        ["ps", "-eo", "pid,args"], capture_output=True, text=True, check=False,
    ).stdout
    procs = []
    for line in out.splitlines():
        if "vllm serve" not in line or "conda run" in line:
            continue  # skip the conda-wrapper PID; want the real python child
        m_port = re.search(r"--port\s+(\d+)", line)
        m_model = re.search(r"vllm serve\s+(\S+)", line)
        if not (m_port and m_model):
            continue
        args = {}
        for key in ("max-model-len", "gpu-memory-utilization", "tensor-parallel-size",
                    "dtype", "quantization"):
            mm = re.search(rf"--{key}\s+(\S+)", line)
            if mm:
                args[key] = mm.group(1)
        procs.append((int(m_port.group(1)), m_model.group(1), args))
    return procs


def _get_metric(metrics_text: str, name: str) -> str | None:
    """Extract first numeric value for a Prometheus gauge `name`."""
    for line in metrics_text.splitlines():
        if line.startswith(name) and not line.startswith(name + "_"):
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 2:
                return parts[1]
    return None


def section4_vllm_status() -> None:
    print("\n" + "=" * 80)
    print("[4] Live vLLM server status (via /metrics)")
    print("=" * 80)
    procs = _discover_vllm_ports()
    if not procs:
        print("No running vLLM server processes found.")
        return

    header = (
        f"{'port':>5}  {'model':<42}{'mml':>6}{'util':>6}{'tp':>4}"
        f"{'run':>6}{'wait':>6}{'kv%':>7}{'preempt':>10}"
    )
    print(header)
    print("-" * len(header))
    for port, model, args in sorted(procs):
        mml = args.get("max-model-len", "-")
        util = args.get("gpu-memory-utilization", "-")
        tp = args.get("tensor-parallel-size", "1")

        running = waiting = kv = preempt = "?"
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/metrics", timeout=5
            ) as r:
                metrics = r.read().decode()
            r_v = _get_metric(metrics, "vllm:num_requests_running")
            w_v = _get_metric(metrics, "vllm:num_requests_waiting")
            k_v = _get_metric(metrics, "vllm:gpu_cache_usage_perc")
            p_v = _get_metric(metrics, "vllm:num_preemptions_total")
            if r_v: running = f"{float(r_v):.0f}"
            if w_v: waiting = f"{float(w_v):.0f}"
            if k_v: kv = f"{float(k_v) * 100:.1f}"
            if p_v: preempt = f"{float(p_v):.0f}"
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            kv = preempt = "DEAD"
            running = waiting = f"({type(e).__name__})"

        print(
            f"{port:>5}  {model[:42]:<42}{mml:>6}{util:>6}{tp:>4}"
            f"{running:>6}{waiting:>6}{kv:>7}{preempt:>10}"
        )
    print()
    print("Legend: mml=max-model-len, util=gpu-memory-utilization, "
          "tp=tensor-parallel-size,")
    print("        run=concurrent in flight, wait=queued, kv%=KV cache usage, "
          "preempt=cumulative preemptions since start.")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    section1_truncation_pass()
    section2_prompt_lengths()
    section3_completion_tokens()
    section4_vllm_status()
    print("\n" + "=" * 80)
    print(f"PNG outputs in: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
