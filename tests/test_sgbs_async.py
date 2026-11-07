"""Tests for asynchronous SGBS scheduling."""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search.evaluator import EvalResult
from search.sgbs import BareRequest, BatchTask, SampledTask
from search.sgbs_async import SGBSAsync


class _FakeEvaluator:
    def __init__(self, *args, seed=None, **kwargs):
        self.coarse_n = 1
        self.fine_n = 1
        self.rng = random.Random(seed)

    def evaluate(self, prompt, metadata=None, seeds=None):
        time.sleep((metadata or {}).get("delay", 0.0))
        p_hat = (metadata or {}).get("p_hat", 1.0)
        return EvalResult(
            p_hat=p_hat,
            pass_count=int(p_hat * 10),
            total_trials=10,
            stage=2,
            is_edge_case=0.3 <= p_hat <= 0.7,
        )


def _config(**search_overrides):
    search = {
        "mode": "async",
        "budget": 3,
        "target_edge_cases": 10,
        "diminishing_returns": 100,
        "max_in_flight": 2,
        "compose_workers": 2,
        "eval_workers": 2,
    }
    search.update(search_overrides)
    return {
        "search": search,
        "ucb": {"prior_alpha": 0.3, "prior_beta": 0.7},
        "skill_count": {
            "max_count": 1,
            "bare_request_bias": 3.0,
            "alpha": 0.05,
            "beta": 0.1,
        },
        "skills": {"mis_size": 8},
        "composer": {"type": "template"},
        "evaluator": {"coarse_n": 1, "fine_n": 1},
        "experiment": {"seed": 123},
    }


def test_async_run_processes_submitted_tasks(monkeypatch):
    """Async scheduler composes/evaluates tasks and updates SGBS records."""

    def fake_sample(**kwargs):
        round_num = kwargs["round_num"]
        return SampledTask(
            round_num=round_num,
            q_key=f"q{round_num}",
            q=BareRequest(f"q{round_num}"),
            count=0,
            selected_skills=[],
            fu_skills=None,
        )

    def fake_compose(sampled, composer):
        return BatchTask(
            round_num=sampled.round_num,
            q_key=sampled.q_key,
            q=sampled.q,
            count=sampled.count,
            selected_skills=[],
            prompt=f"prompt-{sampled.round_num}",
            metadata={
                "p_hat": 0.5 if sampled.round_num == 2 else 1.0,
                "delay": 0.02 if sampled.round_num == 1 else 0.0,
            },
            seeds=[],
            fu_skills=None,
        )

    monkeypatch.setattr("search.sgbs_async.Evaluator", _FakeEvaluator)
    monkeypatch.setattr("search.sgbs_async._compose_task", fake_compose)

    runner = SGBSAsync(_config())
    runner._sample_fn = fake_sample  # selector is resolved per-instance
    result = runner.run(
        target_model=object(),
        judge=object(),
        bare_requests=[BareRequest("q1"), BareRequest("q2"), BareRequest("q3")],
        skills={},
    )

    assert result.stats["mode"] == "async"
    assert result.stats["rounds_run"] == 3
    assert len(result.round_records) == 3
    assert [r.round_num for r in result.round_records] == [1, 2, 3]
    assert len(result.edge_cases) == 1
    assert result.edge_cases[0].round_found == 2


def test_run_sgbs_selects_async_runner():
    from scripts.run_sgbs import create_sgbs_runner
    from search import SGBSv2

    async_runner = create_sgbs_runner(_config())
    sync_runner = create_sgbs_runner({"search": {"mode": "sync"}})

    assert isinstance(async_runner, SGBSAsync)
    assert isinstance(sync_runner, SGBSv2)
    assert not isinstance(sync_runner, SGBSAsync)
