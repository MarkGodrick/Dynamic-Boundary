"""Logic-level smoke test for adaptive evaluation (no vllm, no judge).

Monkey-patches `adaptive.adaptive_eval._run_n_trials_on_items` with a Rasch
sampler (Bernoulli on σ(θ_truth - β)) so the entire CAT loop, pilot,
stop_event handling, and Step-2 input validation can be exercised
deterministically. Complements `smoke_test_adaptive.py`, which is the
integration-level smoke test that does spin up a real vllm + judge.

Run from anywhere:
    python scripts/smoke_test_adaptive_fake_llm.py

Coverage:
  [1] convergence on Rasch truth (θ̂ accuracy via stability termination)
  [2] anchor exhaustion (large tau_pad suppresses trigger; tight stab_eps never met)
  [3] max_iters termination (tight stab_eps never met within max_iters)
  [4] stop_event during pilot → main loop skipped, "interrupted"
  [5] stop_event during main loop → "interrupted" with partial history
  [6] Step-2 panel-set subset → fail-fast before SGBS
  [7] Step-2 panel-set superset → fail-fast before SGBS
"""

import logging
import math
import random
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING, format='%(message)s')

# Project root = parent of scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive import adaptive_eval
from adaptive.anchor_bundle import AnchorBundle, AnchorItem
from adaptive.posterior import EAPEstimator


PASS, FAIL = [], []


def check(label, cond, detail=""):
    (PASS if cond else FAIL).append(label)
    sym = "✓" if cond else "✗"
    print(f"  {sym} {label}" + (f" — {detail}" if detail else ""))


def make_bundle(panel, J=20, beta_lo=-2.0, beta_hi=2.0, N=40):
    betas = np.linspace(beta_lo, beta_hi, J)
    return AnchorBundle(
        category="harmful_refusal",
        panel_models=list(panel),
        panel_theta={m: 0.0 for m in panel},
        items=[
            AnchorItem(item_id=f"smoke_{i:03d}", prompt=f"item-{i}",
                       metadata={}, beta=float(b))
            for i, b in enumerate(betas)
        ],
        filter_spec="smoke", M=len(panel), N=N,
    )


def fake_runner_rasch(theta_truth):
    """Replacement for _run_n_trials_on_items: Rasch p=σ(θ-β) Bernoulli.

    Returns (summary, raw_rows) per the production signature. raw_rows
    are stub dicts with empty trials list — the smoke test only exercises
    the posterior/CAT logic, not trial-detail persistence.
    """
    def _fn(target_model, judge, items, N, seed, target_name, max_workers,
            stop_event=None):
        out = {}
        rows = []
        for it in items:
            rng = random.Random(f"{seed}_{it.item_id}")
            p = 1.0 / (1.0 + math.exp(-(theta_truth - it.beta)))
            n_pass = sum(1 for _ in range(N) if rng.random() < p)
            out[it.item_id] = (n_pass, N)
            rows.append({
                "prompt_id": it.item_id, "model": target_name,
                "pass_count": n_pass, "total_trials": N, "trials": [],
            })
        return out, rows
    return _fn


class FakeModel:
    name = "fake-llm"


def run_eval(bundle, *, theta_truth=0.0, max_iters=15, stab_eps=0.05,
             stab_K=3, tau_pad=2.0, stop_event=None, runner=None):
    estimator = EAPEstimator(prior_mu=0.0, prior_sigma=3.0, n_quad=21)
    cfg = {
        "N": 40, "stab_eps": stab_eps, "stab_K": stab_K, "tau_pad": tau_pad,
        "t_min": 2, "hysteresis_L": 2,
        "max_iters": max_iters, "trial_seed": 42,
        "concurrent": False, "estimator_name": "eap",
    }
    orig = adaptive_eval._run_n_trials_on_items
    adaptive_eval._run_n_trials_on_items = runner or fake_runner_rasch(theta_truth)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            return adaptive_eval.run_adaptive_eval(
                target_model=FakeModel(), judge=None,
                anchor_bundle=bundle, estimator=estimator,
                config=cfg, output_dir=tmp, stop_event=stop_event,
            )
    finally:
        adaptive_eval._run_n_trials_on_items = orig


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def case_convergence():
    print("\n[1] CONVERGENCE  θ_truth=+0.5, J=20  (stab_eps=0.05, K=3)")
    bundle = make_bundle(["m1", "m2", "m3"], J=20)
    res = run_eval(bundle, theta_truth=0.5, max_iters=20, stab_eps=0.05)
    print(f"     θ̂={res.theta_hat_final:+.3f}  SE={res.theta_se_final:.3f}  "
          f"stop={res.stopped_reason}  iters={len(res.history)}")
    check("θ̂ within ±0.4 of truth=0.5",
          abs(res.theta_hat_final - 0.5) < 0.4,
          f"θ̂={res.theta_hat_final:+.3f}")
    check("converged via stability or max_iters",
          res.stopped_reason in ("stability", "max_iters"),
          res.stopped_reason)


def case_anchor_exhausted():
    print("\n[2] ANCHOR_EXHAUSTED  J=8, stab_eps=1e-6, tau_pad=100 (suppress trigger)")
    bundle = make_bundle(["m1"], J=8)
    res = run_eval(bundle, theta_truth=0.0, max_iters=50, stab_eps=1e-6,
                   tau_pad=100.0)
    print(f"     stop={res.stopped_reason}  iters={len(res.history)}")
    check("stopped_reason == anchor_exhausted",
          res.stopped_reason == "anchor_exhausted", res.stopped_reason)


def case_max_iters():
    print("\n[3] MAX_ITERS  max_iters=5, stab_eps=1e-6")
    bundle = make_bundle(["m1"], J=30)
    res = run_eval(bundle, theta_truth=0.0, max_iters=5, stab_eps=1e-6,
                   tau_pad=100.0)
    print(f"     stop={res.stopped_reason}  iters={len(res.history)}")
    check("stopped_reason == max_iters",
          res.stopped_reason == "max_iters", res.stopped_reason)


def case_pilot_stop():
    print("\n[4] STOP_EVENT during PILOT → skip main loop")
    bundle = make_bundle(["m1"], J=20)
    stop_event = threading.Event()

    def runner(target_model, judge, items, N, seed, target_name, max_workers,
               stop_event=None):
        # Simulate panel_inference draining: set stop_event before returning
        if stop_event is not None:
            stop_event.set()
        summary = {it.item_id: (N // 2, N) for it in items}
        rows = [{"prompt_id": it.item_id, "model": target_name,
                 "pass_count": N // 2, "total_trials": N, "trials": []}
                for it in items]
        return summary, rows

    res = run_eval(bundle, stop_event=stop_event, runner=runner)
    print(f"     stop={res.stopped_reason}  iters={len(res.history)}")
    check("stopped_reason == interrupted",
          res.stopped_reason == "interrupted", res.stopped_reason)
    check("main loop skipped (history empty)",
          len(res.history) == 0, f"len={len(res.history)}")


def case_main_loop_stop():
    print("\n[5] STOP_EVENT during MAIN LOOP")
    bundle = make_bundle(["m1"], J=20)
    stop_event = threading.Event()
    call_count = [0]

    def runner(target_model, judge, items, N, seed, target_name, max_workers,
               stop_event=None):
        call_count[0] += 1
        # 1st call = pilot, 2nd+ = main iters; set stop on the 3rd call
        if call_count[0] == 3 and stop_event is not None:
            stop_event.set()
        summary = {it.item_id: (N // 2, N) for it in items}
        rows = [{"prompt_id": it.item_id, "model": target_name,
                 "pass_count": N // 2, "total_trials": N, "trials": []}
                for it in items]
        return summary, rows

    # stab_eps=1e-6 + tau_pad=100 → no stability / no coverage trigger;
    # only stop_event can break the loop.
    res = run_eval(bundle, stop_event=stop_event, runner=runner,
                   stab_eps=1e-6, tau_pad=100.0, max_iters=20)
    print(f"     stop={res.stopped_reason}  iters={len(res.history)}  calls={call_count[0]}")
    check("stopped_reason == interrupted",
          res.stopped_reason == "interrupted", res.stopped_reason)
    check("main loop ran 1+ iters before stop",
          len(res.history) >= 1, f"len={len(res.history)}")


def case_a3_subset():
    print("\n[6] A3 fail-fast — step2.panel_models is SUBSET of anchor")
    from adaptive.calibrate import step2_calibrate
    bundle = make_bundle(["m1", "m2", "m3"], J=10)
    cfg = {"step2": {
        "K": 5, "sgbs_config": "/nonexistent",
        "panel_models": [{"name": "m1"}, {"name": "m2"}],
    }}
    raised, err = False, ""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            step2_calibrate(
                target_model=None, judge=None, anchor_bundle=bundle,
                theta_hat=0.0, pass_counts=np.zeros(10),
                trial_counts=np.zeros(10), used=np.zeros(10, dtype=bool),
                estimator=EAPEstimator(), config=cfg, output_dir=tmp,
            )
    except RuntimeError as e:
        raised, err = True, str(e)
    print(f"     err={err[:120]}")
    check("raised RuntimeError", raised)
    check("error mentions 'must equal'", "must equal" in err)
    check("error names missing m3", "m3" in err)


def case_a3_extra():
    print("\n[7] A3 fail-fast — step2.panel_models is SUPERSET of anchor")
    from adaptive.calibrate import step2_calibrate
    bundle = make_bundle(["m1", "m2", "m3"], J=10)
    cfg = {"step2": {
        "K": 5, "sgbs_config": "/nonexistent",
        "panel_models": [{"name": n} for n in ["m1", "m2", "m3", "m_extra"]],
    }}
    raised, err = False, ""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            step2_calibrate(
                target_model=None, judge=None, anchor_bundle=bundle,
                theta_hat=0.0, pass_counts=np.zeros(10),
                trial_counts=np.zeros(10), used=np.zeros(10, dtype=bool),
                estimator=EAPEstimator(), config=cfg, output_dir=tmp,
            )
    except RuntimeError as e:
        raised, err = True, str(e)
    print(f"     err={err[:120]}")
    check("raised RuntimeError", raised)
    check("error names extra m_extra", "m_extra" in err)


def main():
    print("=" * 64)
    print("Adaptive eval smoke test (fake-LLM Rasch sim)")
    print("=" * 64)
    case_convergence()
    case_anchor_exhausted()
    case_max_iters()
    case_pilot_stop()
    case_main_loop_stop()
    case_a3_subset()
    case_a3_extra()
    print("\n" + "=" * 64)
    print(f"PASS: {len(PASS)}   FAIL: {len(FAIL)}")
    if FAIL:
        print("\nFAILURES:")
        for f in FAIL:
            print(f"  ✗ {f}")
        sys.exit(1)
    print("✓ all checks pass")


if __name__ == "__main__":
    main()
