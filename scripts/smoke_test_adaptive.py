"""Synthetic smoke test for the adaptive evaluation loop.

Mocks target_model.generate and judge.judge to draw Bernoulli(σ(θ_true - β_x))
for each trial, where β_x is looked up from the anchor bundle by prompt
identity. This validates the full Algorithm 2 plumbing (pilot → CAT → EAP
update → SE convergence) without touching GPU or the LLM judge API.

Runs both EAP and MAP+Laplace and reports θ̂ vs θ_true, SE, iters used.

Usage:
    python scripts/smoke_test_adaptive.py \\
        --anchor-bundle benchmarks/harmful_refusal/anchor_bundle_M9.json \\
        --theta-true -0.16
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from logger import setup_logging, get_logger

from adaptive.anchor_bundle import load_anchor_bundle
from adaptive.posterior import build_estimator, _sigmoid
from adaptive.adaptive_eval import run_adaptive_eval


logger = get_logger(__name__)


class MockTargetModel:
    """Returns a token unique per (prompt, seed) — judge will look at prompt only."""

    name = "mock://synthetic"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str, seed: int | None = None) -> str:
        self.calls += 1
        return f"<mock response for seed={seed}>"

    def generate_conversation(self, messages, seed=None):
        return self.generate(messages[-1].get("content", ""), seed=seed)


class MockJudge:
    """Bernoulli(σ(θ_true − β_x)) where β_x is looked up by prompt → β."""

    def __init__(self, theta_true: float, prompt_to_beta: dict, rng_seed: int = 0):
        self.theta_true = float(theta_true)
        self.prompt_to_beta = prompt_to_beta
        self.rng = np.random.default_rng(rng_seed)
        self.calls = 0

    def judge(self, messages, metadata=None):
        self.calls += 1
        # First user message carries the anchor item prompt
        prompt = messages[0]["content"] if messages else ""
        beta = self.prompt_to_beta.get(prompt)
        if beta is None:
            raise RuntimeError(f"prompt not in bundle: {prompt[:40]}...")
        p = float(_sigmoid(self.theta_true - beta))
        return bool(self.rng.random() < p)

    def get_last_error(self):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-bundle", required=True)
    parser.add_argument("--theta-true", type=float, default=-0.16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--stab-eps", type=float, default=0.02)
    parser.add_argument("--stab-K", type=int, default=3)
    parser.add_argument("--tau-pad", type=float, default=0.5)
    parser.add_argument("--N", type=int, default=40)
    args = parser.parse_args()

    setup_logging(verbose=False)

    bundle = load_anchor_bundle(args.anchor_bundle)
    print(f"Loaded bundle: cat={bundle.category}  J={bundle.J}  M={bundle.M}  N={bundle.N}")
    betas = np.array(bundle.betas())
    print(f"  β̂ range: [{betas.min():+.2f}, {betas.max():+.2f}]   median {np.median(betas):+.2f}")
    print(f"  panel θ̂ range: [{min(bundle.panel_theta.values()):+.2f}, "
          f"{max(bundle.panel_theta.values()):+.2f}]")
    print(f"  θ_true (synthetic target): {args.theta_true:+.3f}")
    print()

    prompt_to_beta = {it.prompt: it.beta for it in bundle.items}

    config_base = {
        "N": args.N,
        "pilot_quantiles": [0.10, 0.30, 0.50, 0.70, 0.90],
        "stab_eps": args.stab_eps,
        "stab_K": args.stab_K,
        "tau_pad": args.tau_pad,
        "t_min": 2,
        "hysteresis_L": 2,
        "max_iters": args.max_iters,
        "trial_seed": args.seed,
        "concurrent": False,                # serial → deterministic for mock
        "prior_mu": 0.0,
        "prior_sigma": 3.0,
        "n_quad": 21,
    }

    for est_name in ("eap", "laplace"):
        target = MockTargetModel()
        judge = MockJudge(args.theta_true, prompt_to_beta, rng_seed=args.seed)
        estimator = build_estimator(est_name, config_base)
        cfg = dict(config_base, estimator_name=est_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_adaptive_eval(
                target_model=target, judge=judge, anchor_bundle=bundle,
                estimator=estimator, config=cfg, output_dir=tmpdir,
            )

        n_loop = len([s for s in result.history if not s.triggered])
        n_pilot = len(result.pilot_item_ids)
        err = result.theta_hat_final - args.theta_true
        print(f"=== {est_name.upper():<7}  ===")
        print(f"  pilot iters    : {n_pilot}")
        print(f"  loop iters     : {n_loop}")
        print(f"  total target calls: {target.calls}  (= (pilot+loop) × N = {(n_pilot+n_loop)*args.N})")
        print(f"  judge calls    : {judge.calls}")
        print(f"  θ̂ final       : {result.theta_hat_final:+.3f}  (true {args.theta_true:+.3f}, err {err:+.3f})")
        print(f"  SE final       : {result.theta_se_final:.3f}")
        print(f"  converged      : {result.converged}  (stopped: {result.stopped_reason})")
        print()


if __name__ == "__main__":
    main()
