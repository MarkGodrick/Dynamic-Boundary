"""SGBS vs. naive uniform search — paired-run comparison.

Compares a Thompson-sampling SGBS run against a uniform-selector run that
shares every other mechanism (composer, compatibility constraints, evaluator,
progressive N, stopping rule, seed policy, target K). Reads only run
artifacts; runs no inference.

Usage
-----
    python scripts/analysis/sgbs_vs_uniform.py \
        --root results/rebuttal/sgbs_vs_uniform [--k 50]

Expected layout under --root:

    <root>/<category>/<arm_dir>/<run_dir>/{config.yaml,stats.json,
                                           edge_cases.jsonl,
                                           trial_details.jsonl,search.log}

`<arm_dir>` names starting with "sgbs" / "uniform" pick the arm; the choice is
cross-checked against each run's own `config.yaml:search.selector` and a
mismatch is a hard error. Cells pair by (category, target model).

Metric definitions (K = --k, default 50)
----------------------------------------
R_K      Round number of the K-th edge case: `round_found` of the K-th entry
         of edge_cases.jsonl sorted ascending. "Rounds spent to collect K."
Q_K      Sum of len(trials) over all rounds with round_num <= R_K. This is the
         number of target-model generate() calls (one trial = one prompt
         evaluated at one seed). For multi-turn categories a trial spans
         several turns, so `target_turns` is reported alongside.
hit rate K / R_K — edge cases per round.
T_K      Wall-clock seconds from the first search.log line to the
         "Edge case #K found" line. This is end-to-end and therefore INCLUDES
         vLLM server startup, which is why T_K can slightly exceed
         stats.json's total_time_seconds (the search loop's own timer, started
         after the model is up). Both are reported so the gap is visible.
         stats.json's figure additionally covers draining in-flight
         evaluations and shutdown, so neither is a subset of the other.
tokens   target: sum of per_turn_meta[].prompt_tokens / completion_tokens.
         judge:  sum of per_turn_judge_meta[].* (0 calls for rule-based judges).
         composer: sum of compose_usage.* (absent in runs predating token
         accounting, and always absent for TemplateComposer -> reported as
         null, never as 0).
         Turns whose meta is missing are counted separately
         (`turns_missing_meta`) so gaps never masquerade as zeros.
GPU-h    T_K / 3600 * (number of CUDA devices in config.target_model).
cost     composer + judge tokens priced with PRICING_SNAPSHOT, in CNY (the
         currency DeepSeek quotes; a USD conversion at CNY_PER_USD is emitted
         alongside). Cached and uncached input are priced separately -- cache
         hits are 50x cheaper than misses, so ignoring the split would be a
         large error. Input with no cache split recorded is priced at the MISS
         rate and flagged, never silently assumed cached. Target inference is
         local, so its cost is reported as GPU-hours instead. JSON keys named
         `usd`/`cost_usd` hold CNY values (kept for backward compatibility).
equal-budget
         Q* = min over the two arms of Q_K. For each arm: how many edge cases
         were found within the first Q* target calls (rounds are taken in
         round_num order, and a round is included only if it fits entirely).
ratio    uniform / SGBS, per cell. Aggregated across cells with a GEOMETRIC
         mean, because absolute cost is category-dominated (R_K spans 3x and
         target input tokens 9x across categories for one model).

Every counter is emitted twice: truncated at R_K, and over the full run
(draining in-flight evaluations can overshoot K by 1-2 items).

Outputs
-------
    <root>/summary.json          all cells + aggregate ratios
    <root>/summary.md            human-readable tables
    <root>/curves/<cell>__<arm>.csv
                                 cumulative edge cases vs cumulative target
                                 calls vs elapsed seconds (the Q3 figure data)
    <run_dir>/cell_summary.json  per-arm counters + audit fingerprint
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

# Only the composer and the LLM judge are billed; the target model runs locally
# (its cost is reported as GPU-hours instead).
# Verified 2026-07-26 against https://api-docs.deepseek.com/zh-cn/quick_start/pricing
# Rates are quoted by DeepSeek in CNY per 1M tokens, flat around the clock (the
# page lists no off-peak/discount window). USD figures are derived with
# CNY_PER_USD below, so the conversion is auditable rather than baked into the
# rates. `deepseek-chat` is a legacy alias for deepseek-v4-flash non-thinking
# mode and is deprecated 2026-07-24; it is priced identically here so that runs
# whose config still names it are costed correctly.
CNY_PER_USD = 7.15  # informational only; report CNY as the primary currency

PRICING_SNAPSHOT = {
    "verified": True,
    "snapshot_date": "2026-07-26",
    "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing",
    "currency": "CNY",
    "cny_per_usd": CNY_PER_USD,
    "models": {
        "deepseek-v4-flash": {
            "provider": "DeepSeek",
            "input_cache_hit_per_m": 0.02,
            "input_cache_miss_per_m": 1.00,
            "output_per_m": 2.00,
        },
        "deepseek-v4-pro": {
            "provider": "DeepSeek",
            "input_cache_hit_per_m": 0.025,
            "input_cache_miss_per_m": 3.00,
            "output_per_m": 6.00,
        },
        "deepseek-chat": {  # legacy alias -> flash non-thinking
            "provider": "DeepSeek",
            "input_cache_hit_per_m": 0.02,
            "input_cache_miss_per_m": 1.00,
            "output_per_m": 2.00,
        },
        "deepseek-reasoner": {  # legacy alias -> flash thinking
            "provider": "DeepSeek",
            "input_cache_hit_per_m": 0.02,
            "input_cache_miss_per_m": 1.00,
            "output_per_m": 2.00,
        },
    },
}

PRICING_WARNING = (
    "UNVERIFIED PRICING: PRICING_SNAPSHOT['verified'] is False. "
    "Confirm the per-M rates before using any cost figure."
)

# Rates are CNY; the `cost_usd`/`usd` keys are retained for backward
# compatibility with existing summary.json consumers, but the VALUES are CNY.
COST_CURRENCY = PRICING_SNAPSHOT["currency"]

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
_TS_FMT = "%Y-%m-%d %H:%M:%S"

RE_LINE_TS = re.compile(rf"^{_TS}\s*\|")
RE_EDGE = re.compile(rf"^{_TS}.*Edge case #(\d+) found \(round (\d+)\)")
RE_ROUND = re.compile(rf"^{_TS}.*\bRound (\d+)/\d+:")
RE_OUTPUT_DIR = re.compile(r"Output directory:")


def parse_log_timeline(log_path: Path) -> dict:
    """Extract wall-clock landmarks from search.log.

    Returns:
        {
          "t0": iso str | None,
          "edge_seconds": {edge_index: seconds_since_t0},
          "edge_rounds":  {edge_index: round_num},
          "round_seconds": [(round_num, seconds_since_t0), ...],
          "launches": int,   # >1 means the log spans a resume
        }
    """
    t0 = None
    edge_seconds: dict[int, float] = {}
    edge_rounds: dict[int, int] = {}
    round_seconds: list[tuple[int, float]] = []
    launches = 0

    if not log_path.exists():
        return {"t0": None, "edge_seconds": {}, "edge_rounds": {},
                "round_seconds": [], "launches": 0}

    with open(log_path, errors="replace") as f:
        for line in f:
            m_ts = RE_LINE_TS.match(line)
            if m_ts and t0 is None:
                t0 = datetime.strptime(m_ts.group(1), _TS_FMT)
            if RE_OUTPUT_DIR.search(line):
                launches += 1

            m = RE_EDGE.match(line)
            if m:
                ts = datetime.strptime(m.group(1), _TS_FMT)
                idx = int(m.group(2))
                edge_seconds[idx] = (ts - t0).total_seconds()
                edge_rounds[idx] = int(m.group(3))
                continue

            m = RE_ROUND.match(line)
            if m:
                ts = datetime.strptime(m.group(1), _TS_FMT)
                round_seconds.append((int(m.group(2)), (ts - t0).total_seconds()))

    return {
        "t0": t0.isoformat() if t0 else None,
        "edge_seconds": edge_seconds,
        "edge_rounds": edge_rounds,
        "round_seconds": round_seconds,
        "launches": launches,
    }


def elapsed_at_round(round_seconds: list[tuple[int, float]], round_num: int):
    """Wall-clock seconds at `round_num`, from the last logged round <= it.

    "Round N/M" lines appear roughly every 10 rounds, so this is ~10-round
    granular. Returns None if no logged round precedes `round_num`.
    """
    best = None
    for r, s in round_seconds:
        if r <= round_num:
            best = s
        else:
            break
    return best


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _zero_tokens() -> dict:
    return {
        "calls": 0,
        "turns": 0,
        "turns_missing_meta": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_hit_tokens": None,
        "cache_miss_tokens": None,
    }


def _add_cache(acc: dict, key: str, value) -> None:
    """Accumulate a cache field that may legitimately be unreported.

    None stays None until some record reports a number; from then on it is a
    running sum. This keeps "provider gave no cache split" distinguishable
    from "zero cached tokens".
    """
    if value is None:
        return
    acc[key] = value if acc[key] is None else acc[key] + value


def count_round_tokens(rounds: list[dict]) -> dict:
    """Aggregate target / judge / composer token counters over `rounds`."""
    target = _zero_tokens()
    judge = _zero_tokens()
    composer = _zero_tokens()
    composer_seen = False

    n_trials = 0
    n_truncated = 0
    stages: dict[str, int] = {}
    result_types: dict[str, int] = {}

    for rec in rounds:
        stages[str(rec.get("stage"))] = stages.get(str(rec.get("stage")), 0) + 1
        rt = str(rec.get("result_type"))
        result_types[rt] = result_types.get(rt, 0) + 1

        cu = rec.get("compose_usage")
        if cu:
            composer_seen = True
            composer["calls"] += cu.get("calls") or 0
            composer["prompt_tokens"] += cu.get("prompt_tokens") or 0
            composer["completion_tokens"] += cu.get("completion_tokens") or 0
            _add_cache(composer, "cache_hit_tokens", cu.get("cache_hit_tokens"))
            _add_cache(composer, "cache_miss_tokens", cu.get("cache_miss_tokens"))

        for trial in rec.get("trials") or []:
            n_trials += 1
            target["calls"] += 1
            if trial.get("truncated"):
                n_truncated += 1

            ptm = trial.get("per_turn_meta")
            if ptm:
                for turn in ptm:
                    target["turns"] += 1
                    if turn is None:
                        target["turns_missing_meta"] += 1
                        continue
                    target["prompt_tokens"] += turn.get("prompt_tokens") or 0
                    target["completion_tokens"] += turn.get("completion_tokens") or 0
                    _add_cache(target, "cache_hit_tokens", turn.get("cache_hit_tokens"))
                    _add_cache(target, "cache_miss_tokens", turn.get("cache_miss_tokens"))
            else:
                # No per-turn meta at all: we know a call happened but not its
                # token counts. Recorded, never silently zero-filled.
                target["turns"] += 1
                target["turns_missing_meta"] += 1

            ptjm = trial.get("per_turn_judge_meta")
            if ptjm:
                for turn in ptjm:
                    judge["calls"] += 1
                    judge["turns"] += 1
                    if turn is None:
                        judge["turns_missing_meta"] += 1
                        continue
                    judge["prompt_tokens"] += turn.get("prompt_tokens") or 0
                    judge["completion_tokens"] += turn.get("completion_tokens") or 0
                    _add_cache(judge, "cache_hit_tokens", turn.get("cache_hit_tokens"))
                    _add_cache(judge, "cache_miss_tokens", turn.get("cache_miss_tokens"))

    return {
        "rounds": len(rounds),
        "trials": n_trials,
        "truncated_trials": n_truncated,
        "truncation_rate": (n_truncated / n_trials) if n_trials else None,
        "turns_per_trial": (target["turns"] / n_trials) if n_trials else None,
        "target": target,
        "judge": judge,
        # A composer that never reported usage (TemplateComposer, or a run
        # predating token accounting) yields null rather than a zeroed dict.
        "composer": composer if composer_seen else None,
        "stage_counts": stages,
        "result_type_counts": result_types,
    }


def price_usd(tokens: dict | None, model_id: str | None) -> dict | None:
    """Price one token counter with PRICING_SNAPSHOT.

    Returns None when there is nothing to price or no rate for `model_id`.
    Input tokens without a cache split are priced at the MISS rate and flagged,
    since assuming hits would understate cost.

    NOTE: the returned `usd` key holds a value in PRICING_SNAPSHOT['currency']
    (CNY for DeepSeek). The key name is kept for backward compatibility; a
    `cny` alias and a derived `usd_converted` are included for clarity.
    """
    if not tokens or not model_id:
        return None
    rates = PRICING_SNAPSHOT["models"].get(model_id)
    if rates is None:
        return {"usd": None, "note": f"no rate for model_id={model_id!r}"}

    hit = tokens.get("cache_hit_tokens")
    miss = tokens.get("cache_miss_tokens")
    note = None
    if hit is None or miss is None:
        hit, miss = 0, tokens.get("prompt_tokens") or 0
        note = "cache breakdown unavailable; all input priced at miss rate"

    cost = (
        hit / 1e6 * rates["input_cache_hit_per_m"]
        + miss / 1e6 * rates["input_cache_miss_per_m"]
        + (tokens.get("completion_tokens") or 0) / 1e6 * rates["output_per_m"]
    )
    out = {
        "usd": round(cost, 4),  # value is in COST_CURRENCY (CNY); key kept for compat
        "currency": COST_CURRENCY,
        "cny": round(cost, 4),
        "usd_converted": round(cost / CNY_PER_USD, 4),
        "pricing_verified": PRICING_SNAPSHOT["verified"],
    }
    if note:
        out["note"] = note
    return out


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def _gpu_names(cuda_devices) -> tuple[list | None, str]:
    """Resolve CUDA indices to device names on THIS host.

    metadata.json's environment.gpu_name is the *main* process's device 0 —
    the target model runs in a vLLM subprocess with CUDA_VISIBLE_DEVICES set,
    so that field does not identify the GPU the run actually used. Resolve
    from config.target_model.cuda_devices instead.
    """
    if cuda_devices is None:
        return None, "cuda_devices absent from config"
    idxs = [s.strip() for s in str(cuda_devices).split(",") if s.strip()]
    try:
        import torch
        names = [torch.cuda.get_device_properties(int(i)).name for i in idxs]
        return names, "torch.cuda.get_device_properties at analysis time"
    except Exception as e:  # pragma: no cover - host dependent
        return None, f"unresolved ({type(e).__name__}: {e})"


def build_fingerprint(cfg: dict, meta: dict, stats: dict) -> dict:
    """Audit block: everything that must match across the two arms."""
    search = cfg.get("search") or {}
    ev = cfg.get("evaluator") or {}
    sc = cfg.get("skill_count") or {}
    tm = cfg.get("target_model") or {}
    comp = cfg.get("composer") or {}
    judge = cfg.get("judge") or {}
    env = (meta or {}).get("environment") or {}

    gpu_names, gpu_src = _gpu_names(tm.get("cuda_devices"))
    cuda_devices = str(tm.get("cuda_devices")) if tm.get("cuda_devices") is not None else None
    n_gpus = len([s for s in (cuda_devices or "").split(",") if s.strip()]) or None

    return {
        "selector": search.get("selector", "thompson"),
        "seed": (cfg.get("experiment") or {}).get("seed", stats.get("seed")),
        "target_edge_cases": search.get("target_edge_cases"),
        "budget": search.get("budget"),
        "diminishing_returns": search.get("diminishing_returns"),
        "max_compositions": search.get("max_compositions"),
        "batch_size": search.get("batch_size"),
        "coarse_n": ev.get("coarse_n"),
        "coarse_range": ev.get("coarse_range"),
        "fine_n": ev.get("fine_n"),
        "fine_range": ev.get("fine_range"),
        "lower_bound": sc.get("lower_bound"),
        "max_count": sc.get("max_count"),
        "mis_size": (cfg.get("skills") or {}).get("mis_size"),
        "composer_type": comp.get("type"),
        "composer_model": comp.get("model"),
        "judge_type": judge.get("type"),
        "judge_model": judge.get("model"),
        "target_model": tm.get("name"),
        "max_new_tokens": tm.get("max_new_tokens"),
        "temperature": tm.get("temperature"),
        "max_model_len": tm.get("max_model_len"),
        "cuda_devices": cuda_devices,
        "n_gpus": n_gpus,
        "gpu_names": gpu_names,
        "gpu_names_source": gpu_src,
        # Kept only to document the discrepancy; NOT the run's GPU.
        "gpu_name_main_process": env.get("gpu_name"),
        "conda_env": tm.get("conda_env"),
        "pytorch_version": env.get("pytorch_version"),
        "python_version": env.get("python_version"),
    }


# ---------------------------------------------------------------------------
# Per-arm summary
# ---------------------------------------------------------------------------

def rolling_hit_rate(rounds_sorted: list[dict], window: int) -> list[dict]:
    """Edge cases per round in consecutive round windows.

    This is the quantitative form of the central claim: SGBS's hit rate climbs
    as its posteriors sharpen, while the uniform arm's is flat by construction.
    A cumulative ratio therefore understates the asymptotic advantage, and a
    mid-run ratio understates it further — which is why the headline comparison
    is only read at K.
    """
    if not rounds_sorted:
        return []
    max_round = rounds_sorted[-1]["round_num"]
    out = []
    lo = 0
    while lo < max_round:
        hi = lo + window
        chunk = [r for r in rounds_sorted if lo < r["round_num"] <= hi]
        if chunk:
            edges = sum(1 for r in chunk if r.get("is_edge_case"))
            out.append({
                "lo": lo, "hi": hi,
                "rounds": len(chunk),
                "edge_cases": edges,
                "hit_rate": edges / len(chunk),
            })
        lo = hi
    return out


def summarize_arm(run_dir: Path, k: int, window: int = 200) -> dict:
    """Compute all counters for one run directory."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    stats = json.loads((run_dir / "stats.json").read_text()) if (run_dir / "stats.json").exists() else {}
    meta = json.loads((run_dir / "metadata.json").read_text()) if (run_dir / "metadata.json").exists() else {}

    edges = _read_jsonl(run_dir / "edge_cases.jsonl")
    rounds = _read_jsonl(run_dir / "trial_details.jsonl")
    timeline = parse_log_timeline(run_dir / "search.log")
    fp = build_fingerprint(cfg, meta, stats)

    edge_rounds = sorted(e["round_found"] for e in edges)
    n_edges = len(edge_rounds)
    complete = n_edges >= k
    r_k = edge_rounds[k - 1] if complete else None

    rounds_sorted = sorted(rounds, key=lambda r: r["round_num"])
    full = count_round_tokens(rounds_sorted)
    at_k = count_round_tokens([r for r in rounds_sorted if r["round_num"] <= r_k]) if complete else None

    # Wall clock: prefer the log landmark for the K-th edge case.
    t_k = timeline["edge_seconds"].get(k) if complete else None
    if complete and t_k is None:
        # log line missing (e.g. truncated log) -> fall back, and say so
        t_k = elapsed_at_round(timeline["round_seconds"], r_k)

    n_gpus = fp["n_gpus"] or 1
    gpu_hours_k = (t_k / 3600 * n_gpus) if t_k else None
    total_time = stats.get("total_time_seconds")
    gpu_hours_full = (total_time / 3600 * n_gpus) if total_time else None

    def _costs(block):
        if block is None:
            return None
        return {
            "composer": price_usd(block["composer"], fp["composer_model"]),
            "judge": price_usd(block["judge"] if block["judge"]["calls"] else None,
                               fp["judge_model"]),
        }

    return {
        "run_dir": str(run_dir),
        "selector": fp["selector"],
        "target_model": fp["target_model"],
        "category": (cfg.get("bare_requests") or {}).get("category"),
        "k": k,
        "complete": complete,
        "edge_cases_found": n_edges,
        "rounds_run": stats.get("rounds_run", full["rounds"]),
        "rounds_in_trial_details": full["rounds"],
        "skipped_rounds": stats.get("skipped_rounds"),
        # --- truncated at R_K ---
        "at_k": None if not complete else {
            "R_k": r_k,
            "Q_k": at_k["trials"],
            "hit_rate": k / r_k,
            "T_k_seconds": t_k,
            "T_k_source": ("search.log 'Edge case #%d found'" % k
                           if timeline["edge_seconds"].get(k) is not None
                           else "fallback: last 'Round N/M' line <= R_k"),
            "gpu_hours": gpu_hours_k,
            "tokens": at_k,
            "cost_usd": _costs(at_k),
        },
        # --- full run ---
        "full_run": {
            "rounds": full["rounds"],
            "queries": full["trials"],
            "edge_cases": n_edges,
            "hit_rate": (n_edges / full["rounds"]) if full["rounds"] else None,
            "total_time_seconds": total_time,
            "gpu_hours": gpu_hours_full,
            "tokens": full,
            "cost_usd": _costs(full),
        },
        "log": {
            "t0": timeline["t0"],
            "launches": timeline["launches"],
            "edge_seconds": {str(i): s for i, s in sorted(timeline["edge_seconds"].items())},
        },
        "rolling_hit_rate": {
            "window": window,
            "windows": rolling_hit_rate(rounds_sorted, window),
        },
        "fingerprint": fp,
    }


def curve_rows(run_dir: Path, k: int) -> list[dict]:
    """Cumulative edge cases vs cumulative target calls vs elapsed seconds."""
    rounds = sorted(_read_jsonl(run_dir / "trial_details.jsonl"),
                    key=lambda r: r["round_num"])
    timeline = parse_log_timeline(run_dir / "search.log")
    rows = []
    cum_q = 0
    cum_e = 0
    cum_out = 0
    for rec in rounds:
        cum_q += len(rec.get("trials") or [])
        cum_e += 1 if rec.get("is_edge_case") else 0
        for t in rec.get("trials") or []:
            for turn in (t.get("per_turn_meta") or []):
                if turn:
                    cum_out += turn.get("completion_tokens") or 0
        rows.append({
            "round_num": rec["round_num"],
            "cum_queries": cum_q,
            "cum_edge_cases": cum_e,
            "cum_target_output_tokens": cum_out,
            "elapsed_s": elapsed_at_round(timeline["round_seconds"], rec["round_num"]),
        })
    return rows


def edges_within_budget(run_dir: Path, q_star: int) -> dict:
    """Edge cases found within the first `q_star` target calls.

    Rounds are consumed in round_num order; a round is included only if it
    fits entirely within the budget.
    """
    rounds = sorted(_read_jsonl(run_dir / "trial_details.jsonl"),
                    key=lambda r: r["round_num"])
    cum_q = 0
    cum_e = 0
    last_round = 0
    for rec in rounds:
        n = len(rec.get("trials") or [])
        if cum_q + n > q_star:
            break
        cum_q += n
        cum_e += 1 if rec.get("is_edge_case") else 0
        last_round = rec["round_num"]
    return {"budget_queries": q_star, "queries_used": cum_q,
            "edge_cases": cum_e, "rounds": last_round}


# ---------------------------------------------------------------------------
# Cell discovery and pairing
# ---------------------------------------------------------------------------

REQUIRED_FILES = ("config.yaml", "stats.json", "edge_cases.jsonl",
                  "trial_details.jsonl")


def discover_runs(root: Path) -> tuple[dict, list[str]]:
    """Map (category, arm, model) -> run_dir. Returns (runs, warnings).

    Raises SystemExit on an ambiguity (two run dirs for the same key) or on an
    arm-label mismatch between the directory name and config.search.selector.
    """
    runs: dict[tuple[str, str, str], Path] = {}
    warnings: list[str] = []

    for cat_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if cat_dir.name in {"curves"}:
            continue
        for arm_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            name = arm_dir.name.lower()
            if name.startswith("sgbs"):
                arm = "sgbs"
            elif name.startswith("uniform"):
                arm = "uniform"
            else:
                warnings.append(f"skipped unrecognized arm dir: {arm_dir}")
                continue

            for run_dir in sorted(p for p in arm_dir.iterdir() if p.is_dir()):
                missing = [f for f in REQUIRED_FILES if not (run_dir / f).exists()]
                if missing:
                    warnings.append(
                        f"skipped incomplete run dir (missing {', '.join(missing)}): {run_dir}"
                    )
                    continue

                cfg = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
                selector = (cfg.get("search") or {}).get("selector", "thompson")
                expected = {"sgbs": "thompson", "uniform": "uniform"}[arm]
                if selector != expected:
                    sys.exit(
                        f"ERROR arm mismatch: {run_dir}\n"
                        f"  directory implies arm={arm} (selector={expected!r}) "
                        f"but config.yaml has search.selector={selector!r}"
                    )

                model = (cfg.get("target_model") or {}).get("name") or "unknown"
                key = (cat_dir.name, arm, model)
                if key in runs:
                    sys.exit(
                        f"ERROR ambiguous cell {key}: two run dirs\n"
                        f"  {runs[key]}\n  {run_dir}\n"
                        "Remove or move the stale one; this script will not guess."
                    )
                runs[key] = run_dir

    return runs, warnings


# ---------------------------------------------------------------------------
# Ratios and aggregation
# ---------------------------------------------------------------------------

RATIO_KEYS = ("rounds", "queries", "wall_clock", "gpu_hours",
              "target_output_tokens", "target_prompt_tokens",
              "judge_tokens", "usd")


def _safe_ratio(num, den):
    if num is None or den in (None, 0):
        return None
    return num / den


def cell_ratios(sgbs: dict, uniform: dict) -> dict | None:
    """uniform / SGBS at K. None if either arm has not reached K."""
    if not (sgbs["complete"] and uniform["complete"]):
        return None
    s, u = sgbs["at_k"], uniform["at_k"]

    def _judge_tokens(a):
        j = a["tokens"]["judge"]
        tot = (j["prompt_tokens"] or 0) + (j["completion_tokens"] or 0)
        return tot or None

    def _usd(a):
        vals = [c["usd"] for c in (a["cost_usd"] or {}).values()
                if c and c.get("usd") is not None]
        return sum(vals) if vals else None

    return {
        "rounds": _safe_ratio(u["R_k"], s["R_k"]),
        "queries": _safe_ratio(u["Q_k"], s["Q_k"]),
        "wall_clock": _safe_ratio(u["T_k_seconds"], s["T_k_seconds"]),
        "gpu_hours": _safe_ratio(u["gpu_hours"], s["gpu_hours"]),
        "target_output_tokens": _safe_ratio(
            u["tokens"]["target"]["completion_tokens"],
            s["tokens"]["target"]["completion_tokens"]),
        "target_prompt_tokens": _safe_ratio(
            u["tokens"]["target"]["prompt_tokens"],
            s["tokens"]["target"]["prompt_tokens"]),
        "judge_tokens": _safe_ratio(_judge_tokens(u), _judge_tokens(s)),
        "usd": _safe_ratio(_usd(u), _usd(s)),
    }


def geometric_mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and v > 0 and math.isfinite(v)]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


FINGERPRINT_MUST_MATCH = (
    "seed", "target_edge_cases", "budget", "diminishing_returns",
    "max_compositions", "batch_size", "coarse_n", "coarse_range",
    "fine_n", "fine_range", "lower_bound", "max_count", "mis_size",
    "composer_type", "judge_type", "target_model", "max_new_tokens",
    "temperature", "max_model_len",
)


def fingerprint_diff(sgbs: dict, uniform: dict) -> list[dict]:
    """Fields that must be identical across arms but are not."""
    out = []
    for key in FINGERPRINT_MUST_MATCH:
        a, b = sgbs["fingerprint"].get(key), uniform["fingerprint"].get(key)
        if a != b:
            out.append({"field": key, "sgbs": a, "uniform": b})
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(v, spec=""):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return format(v, spec) if spec else f"{v:.3f}"
    return str(v)


def _m(v):
    """Format a token count in millions."""
    return "n/a" if v is None else f"{v / 1e6:.2f}M"


def render_markdown(report: dict) -> str:
    L: list[str] = []
    L.append("# SGBS vs. uniform search")
    L.append("")
    L.append(f"K = {report['k']}. Generated from run artifacts only; no inference re-run.")
    L.append("")
    if not PRICING_SNAPSHOT["verified"]:
        L.append(f"> **{PRICING_WARNING}**")
        L.append("")

    L.append("## Metric definitions")
    L.append("")
    L.append("| symbol | definition |")
    L.append("|---|---|")
    L.append(f"| R_K | round number of the {report['k']}-th edge case |")
    L.append("| Q_K | target-model calls in rounds <= R_K (one trial = one call) |")
    L.append("| hit rate | K / R_K |")
    L.append("| T_K | seconds from first log line to the \"Edge case #K found\" line |")
    L.append("| GPU-h | T_K / 3600 x number of CUDA devices |")
    L.append("| ratio | uniform / SGBS; aggregated across cells with a geometric mean |")
    L.append("")

    L.append("## Cells")
    L.append("")
    L.append(f"| cell | arm | status | R_K | Q_K | hit rate | T_K (h) | GPU-h "
             f"| target out | judge tok | cost ({COST_CURRENCY}) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for cell in report["cells"]:
        for arm in ("sgbs", "uniform"):
            a = cell["arms"].get(arm)
            if a is None:
                continue
            if a["complete"]:
                k = a["at_k"]
                status = "complete"
                r, q = k["R_k"], k["Q_k"]
                hr = f"{k['hit_rate'] * 100:.2f}%"
                th = _fmt(k["T_k_seconds"] / 3600 if k["T_k_seconds"] else None, ".2f")
                gh = _fmt(k["gpu_hours"], ".2f")
                out_tok = _m(k["tokens"]["target"]["completion_tokens"])
                jt = _m((k["tokens"]["judge"]["prompt_tokens"] or 0)
                        + (k["tokens"]["judge"]["completion_tokens"] or 0))
                usd_vals = [c["usd"] for c in (k["cost_usd"] or {}).values()
                            if c and c.get("usd") is not None]
                usd = f"{sum(usd_vals):.2f}" if usd_vals else "0.00"
            else:
                f = a["full_run"]
                status = f"INCOMPLETE ({a['edge_cases_found']}/{report['k']})"
                r, q = f["rounds"], f["queries"]
                hr = f"{f['hit_rate'] * 100:.2f}%" if f["hit_rate"] else "n/a"
                th = _fmt(f["total_time_seconds"] / 3600 if f["total_time_seconds"] else None, ".2f")
                gh = _fmt(f["gpu_hours"], ".2f")
                out_tok = _m(f["tokens"]["target"]["completion_tokens"])
                jt = _m((f["tokens"]["judge"]["prompt_tokens"] or 0)
                        + (f["tokens"]["judge"]["completion_tokens"] or 0))
                usd = "n/a"
            L.append(f"| {cell['cell']} | {arm} | {status} | {r} | {q} | {hr} "
                     f"| {th} | {gh} | {out_tok} | {jt} | {usd} |")
    L.append("")
    L.append("Rows marked INCOMPLETE report full-run counters at the current "
             "state, not counters at K; no ratio is computed for them.")
    L.append("")

    L.append("## Ratios (uniform / SGBS)")
    L.append("")
    rated = [c for c in report["cells"] if c["ratios"]]
    if not rated:
        L.append("_No cell has both arms at K yet._")
    else:
        L.append("| cell | " + " | ".join(RATIO_KEYS) + " |")
        L.append("|---" * (len(RATIO_KEYS) + 1) + "|")
        for c in rated:
            L.append(f"| {c['cell']} | "
                     + " | ".join(_fmt(c["ratios"][k], ".2f") for k in RATIO_KEYS) + " |")
        L.append(f"| **geo. mean** | "
                 + " | ".join(_fmt(report["aggregate"][k], ".2f") for k in RATIO_KEYS) + " |")
    L.append("")

    L.append("## Rolling hit rate (edge cases per round, by round window)")
    L.append("")
    L.append("SGBS's hit rate climbs as its posteriors sharpen; the uniform "
             "arm's is flat by construction. A mid-run ratio therefore "
             "understates the final one.")
    L.append("")
    for cell in report["cells"]:
        for arm, a in cell["arms"].items():
            rh = a.get("rolling_hit_rate") or {}
            wins = rh.get("windows") or []
            if not wins:
                continue
            L.append(f"**{cell['cell']} — {arm}** (window = {rh['window']} rounds)")
            L.append("")
            L.append("| rounds | " + " | ".join(f"{w['lo']}-{w['hi']}" for w in wins) + " |")
            L.append("|---" * (len(wins) + 1) + "|")
            L.append("| hit rate | " + " | ".join(f"{w['hit_rate'] * 100:.2f}%" for w in wins) + " |")
            L.append("")

    L.append("## Equal-budget comparison")
    L.append("")
    eb = [c for c in report["cells"] if c["equal_budget"]]
    if not eb:
        L.append("_Needs at least one arm's Q_K; unavailable._")
    else:
        L.append("| cell | budget Q* (target calls) | SGBS edge cases | uniform edge cases |")
        L.append("|---|---|---|---|")
        for c in eb:
            e = c["equal_budget"]
            L.append(f"| {c['cell']} | {e['budget_queries']} | "
                     f"{e['sgbs']['edge_cases']} | {e['uniform']['edge_cases']} |")
    L.append("")

    if any(c["fingerprint_diff"] for c in report["cells"]):
        L.append("## Fingerprint mismatches (arms should differ ONLY in selector)")
        L.append("")
        for c in report["cells"]:
            for d in c["fingerprint_diff"]:
                L.append(f"- **{c['cell']}** `{d['field']}`: "
                         f"sgbs={d['sgbs']!r} uniform={d['uniform']!r}")
        L.append("")

    if report["warnings"]:
        L.append("## Warnings")
        L.append("")
        for w in report["warnings"]:
            L.append(f"- {w}")
        L.append("")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/rebuttal/sgbs_vs_uniform",
                    help="root containing <category>/<arm>_k*/<run_dir>/")
    ap.add_argument("--k", type=int, default=50,
                    help="target edge-case count at which arms are compared")
    ap.add_argument("--no-curves", action="store_true",
                    help="skip writing the per-arm curve CSVs")
    ap.add_argument("--window", type=int, default=200,
                    help="round-window size for the rolling hit-rate table")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"ERROR: --root not a directory: {root}")

    runs, warnings = discover_runs(root)
    if not runs:
        sys.exit(f"ERROR: no usable run dirs under {root}")

    # Pair by (category, model)
    keys = sorted({(cat, model) for (cat, _arm, model) in runs})
    cells = []
    for cat, model in keys:
        arms = {}
        for arm in ("sgbs", "uniform"):
            rd = runs.get((cat, arm, model))
            if rd is not None:
                arms[arm] = summarize_arm(rd, args.k, args.window)
        cell_name = f"{cat} x {model}"
        if len(arms) < 2:
            warnings.append(
                f"cell '{cell_name}' has only the {list(arms)[0]} arm; "
                "no comparison possible"
            )
        for arm, a in arms.items():
            # stats.json and trial_details.jsonl are written by separate code
            # paths. A mismatch means either the run is still in progress
            # (expected) or a snapshot was interrupted (audit this).
            if a["rounds_run"] != a["rounds_in_trial_details"]:
                warnings.append(
                    f"cell '{cell_name}' {arm}: stats.json rounds_run="
                    f"{a['rounds_run']} but trial_details.jsonl has "
                    f"{a['rounds_in_trial_details']} rounds "
                    "(normal while the run is live; otherwise investigate)"
                )

        ratios = None
        eq = None
        fp_diff = []
        if len(arms) == 2:
            fp_diff = fingerprint_diff(arms["sgbs"], arms["uniform"])
            ratios = cell_ratios(arms["sgbs"], arms["uniform"])
            q_list = [a["at_k"]["Q_k"] for a in arms.values() if a["complete"]]
            if q_list:
                # If only one arm reached K, its Q_K is still a legitimate
                # equal-budget yardstick for the other arm.
                q_star = min(q_list)
                eq = {
                    "budget_queries": q_star,
                    "sgbs": edges_within_budget(Path(arms["sgbs"]["run_dir"]), q_star),
                    "uniform": edges_within_budget(Path(arms["uniform"]["run_dir"]), q_star),
                    "both_arms_complete": all(a["complete"] for a in arms.values()),
                }

        cells.append({
            "cell": cell_name,
            "category": cat,
            "target_model": model,
            "arms": arms,
            "ratios": ratios,
            "equal_budget": eq,
            "fingerprint_diff": fp_diff,
        })

        # Per-arm artifacts
        for arm, a in arms.items():
            rd = Path(a["run_dir"])
            (rd / "cell_summary.json").write_text(
                json.dumps(a, indent=2, ensure_ascii=False) + "\n")
            if not args.no_curves:
                cdir = root / "curves"
                cdir.mkdir(parents=True, exist_ok=True)
                safe = f"{cat}__{model.replace('/', '_')}__{arm}"
                rows = curve_rows(rd, args.k)
                if rows:
                    with open(cdir / f"{safe}.csv", "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0]))
                        w.writeheader()
                        w.writerows(rows)

    aggregate = {
        key: geometric_mean([c["ratios"][key] for c in cells if c["ratios"]])
        for key in RATIO_KEYS
    }

    report = {
        "k": args.k,
        "root": str(root),
        "pricing_snapshot": PRICING_SNAPSHOT,
        "cells": cells,
        "aggregate": aggregate,
        "warnings": warnings,
    }

    (root / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    md = render_markdown(report)
    (root / "summary.md").write_text(md + "\n")

    print(md)
    print(f"\nWrote {root / 'summary.json'} and {root / 'summary.md'}")
    if not PRICING_SNAPSHOT["verified"]:
        print(f"WARNING: {PRICING_WARNING}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
