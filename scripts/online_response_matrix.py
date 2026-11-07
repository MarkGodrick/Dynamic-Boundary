"""Collect response matrix against ALREADY-RUNNING vLLM servers.

Differs from `scripts/collect_response_matrix.py` in that this script does NOT
spawn vLLM — it discovers live servers (port scan + /v1/models match), routes
trials to them with adaptive concurrency (TCP-AIMD-style cwnd), persists
partial state continuously, and reacts to servers coming up / going down
mid-run. Output schema is identical to collect_response_matrix so downstream
IRT / analysis code consumes both interchangeably.

Usage:
    # 1) Start one or more vLLM servers manually (any model from PANEL_MODELS),
    #    e.g. on ports 8200, 8210
    # 2) Run this script — it auto-discovers and dispatches
    python scripts/online_response_matrix.py \\
        --dataset benchmarks/over_refusal/edge_case_dataset.jsonl \\
        --output-dir results/online_matrix/over_refusal \\
        --category over_refusal \\
        --n-trials 40 \\
        --ports 8200-8400

Resume: re-run with same --output-dir; existing per-prompt trials are kept,
seeds are deterministically re-derived from --seed + dataset order. A
dataset-content hash is stamped in metadata.json and verified on resume.
"""

import argparse
import hashlib
import json
import queue
import random
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Line-buffer stdout so progress prints land in real time when redirected.
sys.stdout.reconfigure(line_buffering=True)

import os

from core.inference_budgets import get_max_tokens
from core.io import load_dataset
from core.judge_factory import create_judge
from core.panel_config import CANDIDATE_MODELS, PANEL_MODELS, VALIDATION_MODELS
from core.trial_utils import (
    _accumulate_judge_meta,
    _accumulate_meta_from_dict,
    aggregate_trials,
    build_trial,
    do_model_step,
    judge_and_capture,
    trial_exception_record,
)
from models.api_model import APIModel
from models.response_parser import get_parser


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_ports(spec: str) -> list[int]:
    """`8100-8200,8300` → sorted unique list of ports."""
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(chunk))
    return sorted(out)


def parse_hosts(spec: str) -> list[str]:
    return [h.strip() for h in spec.split(",") if h.strip()]


def model_short(name: str) -> str:
    return name.replace("/", "_").replace("-", "_")


def prompt_seeds(master_seed: int, prompt_id: str, n: int) -> list[int]:
    """Per-prompt deterministic seed stream.

    Unlike collect_response_matrix.py (which uses an item-major RNG so seed
    assignment depends on both dataset order and n_trials), each prompt here
    gets its own RNG keyed on (master_seed, prompt_id). This means:
      - growing n_trials reuses existing seeds (just appends new ones), so
        partial resume works across n_trials changes;
      - reordering / inserting dataset entries doesn't shift seeds for
        unaffected prompts.
    """
    h = hashlib.sha256(f"{master_seed}:{prompt_id}".encode()).digest()
    rng = random.Random(int.from_bytes(h[:8], "big"))
    return [rng.randint(0, 2**31 - 1) for _ in range(n)]


def dataset_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Server probe
# ---------------------------------------------------------------------------

def probe_server(host: str, port: int, timeout: float = 3.0) -> str | None:
    """Return the served model id if (host, port) is a healthy vLLM, else None.

    vLLM servers started with --api-key (or VLLM_API_KEY env on client) reject
    /v1/* without a bearer token, so we send the same token APIModel uses.
    /health itself is unauthenticated.
    """
    base = f"http://{host}:{port}"
    auth_token = os.environ.get("VLLM_API_KEY", "dummy")
    headers = {"Authorization": f"Bearer {auth_token}"}
    try:
        req = urllib.request.Request(f"{base}/health", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return None
        req = urllib.request.Request(f"{base}/v1/models", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        models = data.get("data") or []
        if not models:
            return None
        return models[0].get("id")
    except (urllib.error.URLError, ConnectionError, OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Per-server state (cwnd, executor, APIModel)
# ---------------------------------------------------------------------------

@dataclass
class ServerInfo:
    host: str
    port: int
    model_name: str
    api_model: APIModel
    executor: ThreadPoolExecutor
    cwnd_min: float
    cwnd_max: float
    cwnd: float
    md_hysteresis_s: float
    ai_coeff: float = 1.0           # k in CA: cwnd += k/cwnd per success
    ssthresh: float = float("inf")  # set on first MD; before that → slow start
    error_threshold: int = 3        # consecutive errors before marking unhealthy
    in_flight: int = 0
    last_md_time: float = 0.0
    healthy: bool = True
    n_success: int = 0
    n_error: int = 0
    consecutive_errors: int = 0     # reset on success; bumps unhealthy when ≥ threshold
    latencies: deque = field(default_factory=lambda: deque(maxlen=20))

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def key(self) -> str:
        return f"{self.host}:{self.port}"

    def on_success(self, latency: float) -> None:
        if self.cwnd < self.ssthresh:
            # Slow start: +1 per success → cwnd ~doubles per RTT-worth.
            self.cwnd = min(self.cwnd_max, self.cwnd + 1.0)
        else:
            # Congestion avoidance: +k/cwnd per success → +k per RTT-worth.
            self.cwnd = min(
                self.cwnd_max,
                self.cwnd + self.ai_coeff / max(self.cwnd, 1.0),
            )
        self.latencies.append(latency)
        self.n_success += 1
        self.consecutive_errors = 0  # success resets the run
        # Latency-based MD: only enabled in CA. During SS, cwnd doubles per RTT,
        # so latency naturally rises (Little's Law) and would spuriously trigger.
        if (
            self.cwnd >= self.ssthresh
            and len(self.latencies) == self.latencies.maxlen
        ):
            sorted_lat = sorted(self.latencies)
            med = sorted_lat[len(sorted_lat) // 2]
            head = self.latencies[0]
            if head > 0 and med > 3.0 * head:
                self._maybe_md()

    def on_error(self) -> None:
        self.n_error += 1
        self.consecutive_errors += 1
        self._maybe_md()
        # Active health gating: don't wait for the probe (≤ poll_interval lag).
        # N consecutive errors without an interleaved success ≈ server gone.
        if self.healthy and self.consecutive_errors >= self.error_threshold:
            self.healthy = False
            print(
                f"  [server-unhealthy] {self.key} → {self.consecutive_errors} "
                f"consecutive errors (threshold={self.error_threshold}); "
                f"draining and removing",
                flush=True,
            )

    def _maybe_md(self) -> None:
        now = time.time()
        if now - self.last_md_time < self.md_hysteresis_s:
            return
        old = self.cwnd
        self.ssthresh = max(self.cwnd_min, old / 2.0)
        self.cwnd = max(self.cwnd_min, self.cwnd * 0.5)
        self.last_md_time = now
        if old > self.cwnd:
            print(
                f"  [cwnd-md] {self.key} cwnd {old:.1f} → {self.cwnd:.1f} "
                f"(ssthresh={self.ssthresh:.1f}, "
                f"succ={self.n_success}, err={self.n_error})",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Trial FSM
# ---------------------------------------------------------------------------

@dataclass
class TrialJob:
    """Per-trial state machine; mirrors panel_inference._TrialJob."""
    model_name: str
    prompt_id: str
    trial_idx: int
    seed: int
    item: dict

    stage: str = "model"  # "model" | "judge" | "done"
    turn: int = 1
    messages: list[dict] = field(default_factory=list)

    responses: list[str] = field(default_factory=list)
    meta_accum: dict = field(default_factory=lambda: {
        "truncated": None, "completion_tokens": None, "per_turn_meta": None,
    })
    judge_meta_list: list | None = None

    last_response: str = ""
    last_passed: bool | None = None
    last_outcome: str = "ok"
    last_err: dict | None = None
    last_label: str | None = None

    final_trial: dict | None = None
    n_server_retries: int = 0   # bumped on server-side failure → requeue
    server_key: str | None = None  # currently dispatched server (for accounting)

    @property
    def follow_ups(self) -> list[str]:
        return (self.item.get("metadata") or {}).get("follow_ups") or []

    @property
    def is_multi_turn(self) -> bool:
        return bool(self.follow_ups)


# ---------------------------------------------------------------------------
# Per-model state
# ---------------------------------------------------------------------------

@dataclass
class ModelState:
    name: str
    short: str
    out_dir: Path
    items_by_id: dict[str, dict]   # other entries (need inference)
    own_results: list[dict]        # search-phase data (inherited from edge_case_dataset)
    expected_seeds: dict[str, list[int]]
    completed_trials: dict[str, list[dict]]  # prompt_id → finalized trials
    pending: deque                 # FIFO[TrialJob]
    in_flight: int = 0             # jobs in FSM (model or judge stage)
    rr_idx: int = 0                # round-robin index across servers
    total_target_trials: int = 0   # for progress reporting
    dirty: bool = True             # save needed


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class OnlineMatrixCollector:
    def __init__(self, args, dataset: list[dict], judge):
        self.args = args
        self.dataset = dataset
        self.judge = judge
        self.dataset_hash = dataset_hash(args.dataset)
        self.target_models = self._resolve_target_models()
        self.target_names = {m["name"] for m in self.target_models}

        # per-model state
        self.models: dict[str, ModelState] = {}
        self._init_model_states()

        # server registry
        self.servers: dict[str, ServerInfo] = {}
        self.servers_lock = threading.Lock()

        # event queue
        self.events_q: queue.Queue = queue.Queue()

        # judge pool (shared)
        self.judge_pool = ThreadPoolExecutor(
            max_workers=args.judge_workers, thread_name_prefix="judge",
        )

        # control
        self.stop_event = threading.Event()
        self.last_save_t = 0.0
        self.last_progress_t = 0.0
        self.start_t = time.time()

    # ----- target model list resolution -----

    def _resolve_target_models(self) -> list[dict]:
        if self.args.candidates_only:
            if not CANDIDATE_MODELS:
                print("ERROR: --candidates-only but CANDIDATE_MODELS empty.")
                sys.exit(1)
            base = list(CANDIDATE_MODELS)
        else:
            base = list(PANEL_MODELS)
            if self.args.include_validation:
                base.extend(VALIDATION_MODELS)
        if self.args.models:
            wanted = {n.strip() for n in self.args.models.split(",")}
            base = [m for m in base if m["name"] in wanted]
            if not base:
                print(f"ERROR: --models filter matched nothing: {self.args.models}")
                sys.exit(1)
        return base

    # ----- per-model init / resume -----

    def _init_model_states(self) -> None:
        out_root = Path(self.args.output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        rng_master_seed = self.args.seed
        for m in self.target_models:
            name = m["name"]
            short = model_short(name)
            out_dir = out_root / short
            out_dir.mkdir(parents=True, exist_ok=True)

            own_entries = [e for e in self.dataset if e.get("source_model") == name]
            other_entries = [e for e in self.dataset if e.get("source_model") != name]

            # Own (search-phase) results — copied verbatim, never re-inferred.
            own_results = []
            for e in own_entries:
                total = e.get("total_trials")
                own_results.append({
                    "prompt_id": e["prompt_id"],
                    "model": name,
                    "trials": [],
                    "pass_count": e.get("pass_count"),
                    "total_trials": total,
                    "n_attempted": total,
                    "n_safety_classified": 0,
                    "n_api_failed": 0,
                    "n_trial_failed": 0,
                    "p_hat": e.get("p_hat"),
                    "source": "search_phase",
                })

            # Deterministic seed table — per-prompt RNG keyed on prompt_id so
            # growing n_trials or reordering the dataset doesn't reshuffle
            # already-collected seeds (see `prompt_seeds` docstring).
            expected_seeds: dict[str, list[int]] = {
                e["prompt_id"]: prompt_seeds(
                    rng_master_seed, e["prompt_id"], self.args.n_trials,
                )
                for e in other_entries
            }

            # Resume
            completed_trials: dict[str, list[dict]] = {}
            self._maybe_resume(out_dir, completed_trials)

            # Build pending: missing seeds per prompt
            pending: deque = deque()
            items_by_id = {e["prompt_id"]: e for e in other_entries}
            for pid, item in items_by_id.items():
                done_seeds = {
                    t["seed"] for t in completed_trials.get(pid, [])
                    # trial_exception → retryable; drop from done set
                    if t.get("judge_outcome") != "trial_exception"
                }
                # Rebuild the keep-list filtering trial_exception out so they
                # don't pollute final aggregation either.
                kept = [
                    t for t in completed_trials.get(pid, [])
                    if t.get("judge_outcome") != "trial_exception"
                ]
                completed_trials[pid] = kept

                for t_idx, s in enumerate(expected_seeds[pid]):
                    if s in done_seeds:
                        continue
                    pending.append(TrialJob(
                        model_name=name, prompt_id=pid, trial_idx=t_idx, seed=s,
                        item=item,
                        messages=[{"role": "user", "content": item["prompt"]}],
                    ))

            total_target = len(other_entries) * self.args.n_trials
            self.models[name] = ModelState(
                name=name, short=short, out_dir=out_dir,
                items_by_id=items_by_id, own_results=own_results,
                expected_seeds=expected_seeds, completed_trials=completed_trials,
                pending=pending, total_target_trials=total_target,
            )

            done_n = sum(len(v) for v in completed_trials.values())
            print(f"  [{short}] resume: {done_n}/{total_target} trials done "
                  f"(pending={len(pending)}, own_search={len(own_entries)})")

    def _maybe_resume(self, out_dir: Path, completed: dict[str, list[dict]]) -> None:
        path = out_dir / "response_matrix.jsonl"
        meta_path = out_dir / "metadata.json"
        if not path.exists():
            return
        # Hash check: refuse if dataset changed
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
            prev_hash = meta.get("dataset_hash")
            if prev_hash and prev_hash != self.dataset_hash:
                print(
                    f"ERROR: dataset hash mismatch in {meta_path}. "
                    f"Previous run used hash={prev_hash}, current dataset hash="
                    f"{self.dataset_hash}. Refusing to resume — seeds would no "
                    f"longer line up. Delete the directory or use a fresh "
                    f"--output-dir."
                )
                sys.exit(1)
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("source") == "search_phase":
                    continue  # own_results re-derived from dataset
                pid = row["prompt_id"]
                completed[pid] = list(row.get("trials") or [])

    # ----- server discovery -----

    def _probe_loop(self) -> None:
        """Background thread: every poll-interval, probe configured (host, port)
        pairs and push up/down events to the main loop.
        """
        hosts = parse_hosts(self.args.hosts)
        ports = parse_ports(self.args.ports)
        # Deduplicated list of (host, port) probe targets.
        targets = [(h, p) for h in hosts for p in ports]
        while not self.stop_event.is_set():
            seen_keys: set[str] = set()
            for host, port in targets:
                if self.stop_event.is_set():
                    return
                key = f"{host}:{port}"
                model_id = probe_server(host, port, timeout=2.0)
                if model_id is None:
                    if key in self.servers and self.servers[key].healthy:
                        self.events_q.put({"kind": "server_down", "key": key})
                    continue
                seen_keys.add(key)
                if model_id not in self.target_names:
                    # Server up but its model isn't in our target list — silently skip
                    # except for one WARN per session.
                    continue
                if key not in self.servers:
                    self.events_q.put({
                        "kind": "server_up",
                        "host": host, "port": port, "model_name": model_id,
                    })
            self.stop_event.wait(self.args.server_poll_interval)

    def _on_server_up(self, host: str, port: int, model_name: str) -> None:
        key = f"{host}:{port}"
        if key in self.servers:
            return
        try:
            os.environ.setdefault("VLLM_API_KEY", "dummy")
            api = APIModel(
                model=model_name,
                api_key_env="VLLM_API_KEY",
                base_url=f"http://{host}:{port}/v1",
                temperature=1.0,
                max_tokens=get_max_tokens(self.args.category, model_name),
                response_parser=get_parser(model_name),
            )
            executor = ThreadPoolExecutor(
                max_workers=int(self.args.cwnd_max) + 8,
                thread_name_prefix=f"model-{port}",
            )
            srv = ServerInfo(
                host=host, port=port, model_name=model_name,
                api_model=api, executor=executor,
                cwnd_min=float(self.args.cwnd_min),
                cwnd_max=float(self.args.cwnd_max),
                cwnd=float(self.args.cwnd_init),
                md_hysteresis_s=float(self.args.md_hysteresis),
                ai_coeff=float(self.args.ai_coeff),
                error_threshold=int(self.args.server_error_threshold),
            )
            with self.servers_lock:
                self.servers[key] = srv
            print(
                f"  [server-up] {key} → {model_name}  "
                f"(cwnd={srv.cwnd:.0f}, max_workers={int(srv.cwnd_max)+8})",
                flush=True,
            )
        except Exception as e:
            print(f"  [server-up FAIL] {key}: {e}", flush=True)

    def _on_server_down(self, key: str) -> None:
        with self.servers_lock:
            srv = self.servers.get(key)
            if srv is None or not srv.healthy:
                return
            srv.healthy = False
        print(
            f"  [server-down] {key} (in_flight={srv.in_flight} will drain or fail)",
            flush=True,
        )

    def _maybe_finalize_server(self, srv: ServerInfo) -> None:
        if srv.healthy or srv.in_flight > 0:
            return
        # Final teardown
        with self.servers_lock:
            if srv.key in self.servers:
                del self.servers[srv.key]
        try:
            srv.executor.shutdown(wait=False)
        except Exception:
            pass
        print(f"  [server-removed] {srv.key} (drained)", flush=True)

    # ----- dispatch -----

    def _healthy_servers_for(self, model_name: str) -> list[ServerInfo]:
        with self.servers_lock:
            return [s for s in self.servers.values()
                    if s.healthy and s.model_name == model_name]

    def _dispatch(self) -> None:
        """For each model with pending work, fill its servers up to cwnd."""
        for state in self.models.values():
            if not state.pending:
                continue
            servers = self._healthy_servers_for(state.name)
            if not servers:
                continue
            # Round-robin across servers; keep dispatching while there's room.
            n = len(servers)
            consecutive_full = 0
            i = state.rr_idx
            while state.pending and consecutive_full < n:
                s = servers[i % n]
                i += 1
                if s.in_flight < int(s.cwnd):
                    job = state.pending.popleft()
                    state.in_flight += 1
                    self._submit_model(job, s)
                    consecutive_full = 0
                else:
                    consecutive_full += 1
            state.rr_idx = i

    def _submit_model(self, job: TrialJob, server: ServerInfo) -> None:
        server.in_flight += 1
        job.server_key = server.key
        is_conv = job.turn > 1
        arg = job.messages if is_conv else job.item["prompt"]
        start = time.time()
        try:
            fut = server.executor.submit(
                do_model_step, server.api_model, arg, job.seed,
                is_conversation=is_conv,
            )
        except RuntimeError:
            # Executor already shut down (e.g. server torn down concurrently)
            server.in_flight -= 1
            self._handle_model_step_failure(job, RuntimeError("executor shut down"))
            return
        fut.add_done_callback(lambda f, _j=job, _s=start, _srv=server:
                              self.events_q.put({
                                  "kind": "model_done", "job": _j,
                                  "fut": f, "start": _s, "server_key": _srv.key,
                              }))

    def _submit_judge(self, job: TrialJob) -> None:
        metadata = job.item.get("metadata")
        try:
            fut = self.judge_pool.submit(
                judge_and_capture, self.judge, list(job.messages), metadata,
            )
        except RuntimeError:
            # Judge pool shut down (only happens during stop)
            job.final_trial = trial_exception_record(
                job.seed, RuntimeError("judge pool shut down"),
            )
            self._finalize_trial(job)
            return
        fut.add_done_callback(lambda f, _j=job: self.events_q.put({
            "kind": "judge_done", "job": _j, "fut": f,
        }))

    # ----- event handlers -----

    def _handle_model_done(self, ev: dict) -> None:
        job: TrialJob = ev["job"]
        fut = ev["fut"]
        start = ev["start"]
        server_key = ev["server_key"]

        # Decrement server in_flight (server may have been removed — be defensive)
        with self.servers_lock:
            srv = self.servers.get(server_key)
        if srv is not None:
            srv.in_flight -= 1

        try:
            response, meta = fut.result()
        except Exception as e:
            if srv is not None:
                srv.on_error()
            self._handle_model_step_failure(job, e)
            if srv is not None and not srv.healthy:
                self._maybe_finalize_server(srv)
            return

        if srv is not None:
            srv.on_success(time.time() - start)
            if not srv.healthy:
                # Server marked down by probe but request still succeeded —
                # tear down once drained.
                self._maybe_finalize_server(srv)

        # Advance: model → judge
        job.last_response = response
        job.responses.append(response)
        job.messages.append({"role": "assistant", "content": response})
        job.meta_accum = _accumulate_meta_from_dict(job.meta_accum, meta)
        job.stage = "judge"
        self._submit_judge(job)

    def _handle_model_step_failure(
        self, job: TrialJob, exc: Exception,
    ) -> None:
        """Server-side exception path. Requeue up to N times, then record as
        trial_exception so the run isn't held hostage by a permanently broken
        prompt.
        """
        state = self.models[job.model_name]
        if job.n_server_retries < self.args.max_retries_per_trial:
            job.n_server_retries += 1
            # Reset FSM to clean turn-1 state for retry
            job.stage = "model"
            job.turn = 1
            job.messages = [{"role": "user", "content": job.item["prompt"]}]
            job.responses = []
            job.meta_accum = {
                "truncated": None, "completion_tokens": None, "per_turn_meta": None,
            }
            job.judge_meta_list = None
            state.pending.append(job)
            state.in_flight -= 1
        else:
            job.final_trial = trial_exception_record(job.seed, exc)
            self._finalize_trial(job)

    def _handle_judge_done(self, ev: dict) -> None:
        job: TrialJob = ev["job"]
        fut = ev["fut"]
        try:
            result = fut.result()
        except Exception as e:
            # Judge failure: not server's fault. Record as trial_exception.
            job.final_trial = trial_exception_record(job.seed, e)
            self._finalize_trial(job)
            return

        passed, outcome, err, label, judge_meta = result
        job.last_passed = passed
        job.last_outcome = outcome
        job.last_err = err
        job.last_label = label
        job.judge_meta_list = _accumulate_judge_meta(job.judge_meta_list, judge_meta)

        next_idx = job.turn - 1  # follow_ups[0] feeds turn 2
        follow_ups = job.follow_ups
        if passed is None or not passed:
            self._finalize_trial(job, failed_at_turn=job.turn)
            return
        if next_idx >= len(follow_ups):
            self._finalize_trial(job, failed_at_turn=0)
            return

        # Next multi-turn turn → back to model stage. Re-dispatch via pending so
        # we respect cwnd / server availability uniformly with new trials.
        job.messages.append({"role": "user", "content": follow_ups[next_idx]})
        job.turn += 1
        job.stage = "model"
        state = self.models[job.model_name]
        state.pending.appendleft(job)
        state.in_flight -= 1

    def _finalize_trial(
        self, job: TrialJob, *, failed_at_turn: int | None = None,
    ) -> None:
        state = self.models[job.model_name]
        if job.final_trial is None:
            job.final_trial = build_trial(
                seed=job.seed,
                response=job.last_response,
                responses=job.responses,
                is_multi_turn=job.is_multi_turn,
                passed=job.last_passed,
                failed_at_turn=failed_at_turn if failed_at_turn is not None else 0,
                judge_outcome=job.last_outcome,
                judge_error=job.last_err,
                judge_label=job.last_label,
                truncated=job.meta_accum["truncated"],
                completion_tokens=job.meta_accum["completion_tokens"],
                per_turn_meta=job.meta_accum["per_turn_meta"],
                per_turn_judge_meta=job.judge_meta_list,
            )
        job.stage = "done"
        state.completed_trials.setdefault(job.prompt_id, []).append(job.final_trial)
        state.in_flight -= 1
        state.dirty = True

    # ----- save -----

    def _save_all(self) -> None:
        for state in self.models.values():
            if state.dirty:
                self._save_model(state)
                state.dirty = False
        self.last_save_t = time.time()

    def _save_model(self, state: ModelState) -> None:
        rows: list[dict] = list(state.own_results)  # search-phase entries
        for pid in state.items_by_id:
            trials = state.completed_trials.get(pid, [])
            stats = aggregate_trials(trials)
            rows.append({
                "prompt_id": pid,
                "model": state.name,
                "trials": trials,
                **stats,
            })
        rows.sort(key=lambda r: r["prompt_id"])

        out_path = state.out_dir / "response_matrix.jsonl"
        tmp = out_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(out_path)

        done_n = sum(len(v) for v in state.completed_trials.values())
        meta = {
            "model": state.name,
            "category": self.args.category,
            "n_trials": self.args.n_trials,
            "seed": self.args.seed,
            "n_prompts": len(state.items_by_id) + len(state.own_results),
            "n_skipped_own": len(state.own_results),
            "n_evaluated_target": state.total_target_trials,
            "n_evaluated_done": done_n,
            "dataset_hash": self.dataset_hash,
            "dataset_path": self.args.dataset,
            "last_updated": datetime.now().isoformat(),
            "via": "online_response_matrix.py",
        }
        with open(state.out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    # ----- progress -----

    def _print_progress(self) -> None:
        with self.servers_lock:
            srv_lines = [
                f"{s.key}={s.model_name.split('/')[-1]}|cwnd={s.cwnd:.0f}|"
                f"if={s.in_flight}|ok={s.n_success}|err={s.n_error}"
                for s in self.servers.values()
            ]
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] "
            f"servers={len(srv_lines)}: " + ("; ".join(srv_lines) or "(none)"),
            flush=True,
        )
        for state in self.models.values():
            done = sum(len(v) for v in state.completed_trials.values())
            print(
                f"  [{state.short}] "
                f"done={done}/{state.total_target_trials} "
                f"pending={len(state.pending)} in_flight={state.in_flight}",
                flush=True,
            )

    # ----- main loop -----

    def run(self) -> None:
        # Signal handlers — two-stage shutdown:
        #   1st signal: set stop_event, let main loop break and _shutdown drain
        #   2nd signal: hard-exit (in-flight httpx calls have 600s timeout and
        #               the model executors are non-daemon, so an unforced exit
        #               can hang the process for minutes after Ctrl-C).
        def _on_sig(sig, _frame):
            if self.stop_event.is_set():
                print(
                    f"\n[{datetime.now().strftime('%H:%M:%S')}] second signal "
                    f"{sig} → hard exit",
                    flush=True,
                )
                os._exit(130)
            print(
                f"\n[{datetime.now().strftime('%H:%M:%S')}] caught signal "
                f"{sig}, draining… (Ctrl-C again to force exit)",
                flush=True,
            )
            self.stop_event.set()
        signal.signal(signal.SIGINT, _on_sig)
        signal.signal(signal.SIGTERM, _on_sig)

        # Probe thread
        probe_th = threading.Thread(target=self._probe_loop, name="probe", daemon=True)
        probe_th.start()

        # Initial save (resume can shrink trial_exception entries)
        self._save_all()

        # Main loop: drain events, dispatch, save, progress
        while True:
            # Termination check
            all_done = all(
                not s.pending and s.in_flight == 0 for s in self.models.values()
            )
            if self.stop_event.is_set() or all_done:
                break

            # Block on next event with a small timeout so we can re-dispatch
            # when a new server appears or cwnd grows.
            try:
                ev = self.events_q.get(timeout=0.5)
            except queue.Empty:
                ev = None

            if ev is not None:
                k = ev["kind"]
                if k == "model_done":
                    self._handle_model_done(ev)
                elif k == "judge_done":
                    self._handle_judge_done(ev)
                elif k == "server_up":
                    self._on_server_up(ev["host"], ev["port"], ev["model_name"])
                elif k == "server_down":
                    self._on_server_down(ev["key"])

            # Dispatch fresh work (cheap when nothing to dispatch)
            self._dispatch()

            # Periodic save
            now = time.time()
            if now - self.last_save_t >= self.args.save_interval:
                self._save_all()
            if now - self.last_progress_t >= self.args.progress_interval:
                self._print_progress()
                self.last_progress_t = now

        # ---- shutdown ----
        self._shutdown()

    def _shutdown(self) -> None:
        # Stop signaling probe thread to exit
        self.stop_event.set()

        # Drain in-flight with timeout
        deadline = time.time() + self.args.drain_timeout
        while time.time() < deadline:
            in_flight = sum(s.in_flight for s in self.models.values())
            if in_flight == 0:
                break
            try:
                ev = self.events_q.get(timeout=0.5)
            except queue.Empty:
                continue
            k = ev["kind"]
            if k == "model_done":
                self._handle_model_done(ev)
            elif k == "judge_done":
                self._handle_judge_done(ev)

        # Final save
        for state in self.models.values():
            state.dirty = True
        self._save_all()

        # Tear down judge + server executors
        try:
            self.judge_pool.shutdown(wait=False)
        except Exception:
            pass
        with self.servers_lock:
            servers = list(self.servers.values())
        for s in servers:
            try:
                s.executor.shutdown(wait=False)
            except Exception:
                pass

        self._print_final_summary()

        # Hard exit: ThreadPoolExecutor worker threads are non-daemon and our
        # httpx client uses a 600s per-request timeout, so an ordinary
        # interpreter exit would block in `_python_exit`'s thread-join until
        # every in-flight request hits its timeout. We've already saved + the
        # final summary is printed, so bypass at-exit handlers entirely.
        in_flight = sum(s.in_flight for s in self.models.values())
        if in_flight > 0:
            print(
                f"  [hard-exit] {in_flight} requests still in-flight after "
                f"drain_timeout; exiting without waiting for httpx timeout",
                flush=True,
            )
        os._exit(0)

    def _print_final_summary(self) -> None:
        print(f"\n{'='*60}\n  Summary\n{'='*60}", flush=True)
        for state in self.models.values():
            done = sum(len(v) for v in state.completed_trials.values())
            p_hats = []
            for trials in state.completed_trials.values():
                stats = aggregate_trials(trials)
                if stats["p_hat"] is not None:
                    p_hats.append(stats["p_hat"])
            avg = sum(p_hats) / len(p_hats) if p_hats else float("nan")
            print(
                f"  [{state.short}] {done}/{state.total_target_trials} "
                f"trials | avg p_hat={avg:.3f} | pending={len(state.pending)}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect response matrix against already-running vLLM servers "
            "(adaptive cwnd, auto-resume, save on signal)."
        ),
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--category", default="constrained_completion")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models", default=None,
        help="Comma-separated model names (default: PANEL_MODELS)",
    )
    parser.add_argument("--include-validation", action="store_true")
    parser.add_argument("--candidates-only", action="store_true")
    parser.add_argument(
        "--ports", default="8100-9000",
        help="Port spec to probe, e.g. '8100-8200,8300'.",
    )
    parser.add_argument(
        "--hosts", default="127.0.0.1",
        help="Comma-separated hosts/IPs to probe (default: localhost only).",
    )
    parser.add_argument("--server-poll-interval", type=float, default=10.0)
    parser.add_argument("--save-interval", type=float, default=30.0)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument(
        "--cwnd-init", type=float, default=8.0,
        help="Initial concurrency window per server (slow-start starting point).",
    )
    parser.add_argument("--cwnd-min", type=float, default=4.0)
    parser.add_argument("--cwnd-max", type=float, default=100.0)
    parser.add_argument(
        "--md-hysteresis", type=float, default=5.0,
        help="Min seconds between consecutive cwnd halvings.",
    )
    parser.add_argument(
        "--ai-coeff", type=float, default=3.0,
        help="Additive-increase coefficient k in congestion-avoidance phase: "
             "cwnd += k/cwnd per success (k=1.0 is canonical Reno; default 3.0 "
             "grows ~3× faster after the first congestion event).",
    )
    parser.add_argument(
        "--server-error-threshold", type=int, default=3,
        help="Consecutive error count (with no interleaved success) at which "
             "a server is actively marked unhealthy and removed, without "
             "waiting for the periodic probe to detect shutdown.",
    )
    parser.add_argument("--judge-workers", type=int, default=100)
    parser.add_argument("--max-retries-per-trial", type=int, default=3)
    parser.add_argument("--drain-timeout", type=float, default=30.0)

    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    # Category cross-check (same guard as collect_response_matrix.py)
    ds_categories = {e["category"] for e in dataset if "category" in e}
    if not ds_categories:
        print("  [warn] dataset has no top-level 'category' field; skipping cross-check")
    elif ds_categories != {args.category}:
        print(
            f"ERROR: --category={args.category!r} but dataset contains "
            f"{ds_categories}. Refusing to run."
        )
        sys.exit(1)

    print(f"Dataset:   {len(dataset)} prompts")
    print(f"Category:  {args.category}")
    print(f"Trials:    {args.n_trials} (seed={args.seed})")
    print(f"Hosts:     {parse_hosts(args.hosts)}")
    print(f"Ports:     {parse_ports(args.ports)[:10]}{'...' if len(parse_ports(args.ports))>10 else ''} "
          f"(total {len(parse_ports(args.ports))})")

    judge_config = {
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_TOKEN",
        "base_url": "https://api.deepseek.com",
    }
    judge = create_judge(args.category, judge_config)

    collector = OnlineMatrixCollector(args, dataset, judge)
    print(f"Models:    {[m['name'] for m in collector.target_models]}")
    print()
    collector.run()


if __name__ == "__main__":
    main()
