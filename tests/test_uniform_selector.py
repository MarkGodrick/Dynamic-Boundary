"""Tests for the uniform-random selector (SGBS vs. naive uniform baseline).

Verifies:
1. Signature parity with `sgbs._sample` (the two must be interchangeable)
2. Skill-count support matches SGBS's, and each count is reachable
3. Marginals are uniform over bare requests / skills / counts
4. Returned skill subsets are mutually compatible and of the requested size
5. Dedup budget (`max_compositions`) and retry exhaustion behave like `_sample`
6. The sampler reads/writes no bandit state
7. Determinism under a fixed seed
8. Selector dispatch from config, and that it actually changes behaviour
"""

import inspect
import json
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search import skill_count, ucb
from search.sgbs import (
    BareRequest,
    SGBSv2,
    SearchState,
    _resolve_selector,
    _sample,
    load_search_state,
    save_results,
)
from search.skill_selector import Skill, is_compatible, load_skills
from search.uniform import count_support, sample_uniform, uniform_scores


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _skills(n: int = 6, conflicting: bool = False) -> dict[str, Skill]:
    """n independent skills; with conflicting=True they all mutually exclude."""
    out = {}
    for i in range(n):
        name = f"s{i}"
        out[name] = Skill(
            name=name,
            description=f"skill {i}",
            tags={"x"} if conflicting else {f"t{i}"},
            incompatible_tags={"x"} if conflicting else set(),
        )
    return out


def _bare_requests(n: int = 4) -> list[BareRequest]:
    return [BareRequest(prompt=f"q{i}", metadata={"i": i}) for i in range(n)]


def _state(
    bare_requests: list[BareRequest],
    skills: dict[str, Skill],
    max_count: int = 4,
) -> SearchState:
    return SearchState(
        bare_request_profiles=ucb.create_profiles(
            [br.prompt for br in bare_requests],
        ),
        skill_profiles=ucb.create_profiles(list(skills.keys())),
        skill_logits=skill_count.init_skill_logits(max_count, 3.0),
        tried_combinations=defaultdict(Counter),
        edge_cases=[],
    )


def _kwargs(state, bare_requests, skills, **overrides) -> dict:
    """Full keyword set accepted by both samplers, with SGBS-like defaults."""
    kw = dict(
        state=state,
        round_num=1,
        br_lookup={br.prompt: br for br in bare_requests},
        skills=skills,
        np_rng=np.random.default_rng(0),
        c=1.0,
        skill_method="thompson",
        max_skip_retries=5,
        skip_penalty=0.1,
        mis_size=8,
        top_k=3,
        top_p=0.9,
        lower_bound=1,
        sc_alpha=0.05,
        sc_beta=0.1,
        follow_up_skill_pool=None,
        max_follow_up_turns=0,
        max_compositions=10**6,  # effectively disable dedup unless overridden
    )
    kw.update(overrides)
    return kw


# ---------------------------------------------------------------------------
# 1. Signature parity  (CLAUDE.md: sibling implementations must stay callable
#    through the same keyword set)
# ---------------------------------------------------------------------------

class TestSignatureParity:
    def test_same_parameter_names_in_same_order(self):
        a = list(inspect.signature(_sample).parameters)
        b = list(inspect.signature(sample_uniform).parameters)
        assert a == b

    def test_same_kinds_and_defaults(self):
        pa = inspect.signature(_sample).parameters
        pb = inspect.signature(sample_uniform).parameters
        for name in pa:
            assert pa[name].kind == pb[name].kind, name
            assert pa[name].default == pb[name].default, name

    def test_all_keyword_only(self):
        for p in inspect.signature(sample_uniform).parameters.values():
            assert p.kind is inspect.Parameter.KEYWORD_ONLY


# ---------------------------------------------------------------------------
# 2. Skill-count support
# ---------------------------------------------------------------------------

class TestCountSupport:
    def test_matches_sgbs_widest_window(self):
        # max_count=4 -> logits length 4 -> counts 0..3; lower_bound clips low.
        assert count_support([0.0] * 4, mis_size=8, lower_bound=1) == (1, 3)
        assert count_support([0.0] * 4, mis_size=8, lower_bound=0) == (0, 3)

    def test_mis_size_caps_the_top(self):
        assert count_support([0.0] * 6, mis_size=2, lower_bound=0) == (0, 2)

    def test_empty_support_raises(self):
        with pytest.raises(ValueError, match="empty skill-count support"):
            count_support([0.0] * 2, mis_size=8, lower_bound=3)

    def test_sampled_counts_stay_in_support_and_cover_it(self):
        brs, sk = _bare_requests(), _skills(6)
        st = _state(brs, sk)
        rng = np.random.default_rng(7)
        seen = set()
        for r in range(400):
            task = sample_uniform(**_kwargs(st, brs, sk, round_num=r, np_rng=rng))
            assert task is not None
            assert 1 <= task.count <= 3
            assert len(task.selected_skills) == task.count
            seen.add(task.count)
        assert seen == {1, 2, 3}

    def test_count_zero_reachable_and_yields_bare_request(self):
        brs, sk = _bare_requests(), _skills(6)
        st = _state(brs, sk)
        rng = np.random.default_rng(3)
        zeros = 0
        for r in range(400):
            task = sample_uniform(
                **_kwargs(st, brs, sk, round_num=r, np_rng=rng, lower_bound=0)
            )
            if task.count == 0:
                zeros += 1
                assert task.selected_skills == []
        assert zeros > 0


# ---------------------------------------------------------------------------
# 3. Uniform marginals
# ---------------------------------------------------------------------------

class TestUniformMarginals:
    def test_uniform_scores_cover_all_keys_in_unit_interval(self):
        rng = np.random.default_rng(0)
        scores = uniform_scores(["a", "b", "c"], rng)
        assert set(scores) == {"a", "b", "c"}
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_bare_requests_are_uniform(self):
        brs, sk = _bare_requests(4), _skills(6)
        st = _state(brs, sk)
        rng = np.random.default_rng(11)
        n = 4000
        picks = Counter(
            sample_uniform(**_kwargs(st, brs, sk, round_num=r, np_rng=rng)).q_key
            for r in range(n)
        )
        assert len(picks) == 4
        for v in picks.values():
            assert 0.8 * n / 4 < v < 1.2 * n / 4

    def test_counts_are_uniform(self):
        brs, sk = _bare_requests(), _skills(6)
        st = _state(brs, sk)
        rng = np.random.default_rng(13)
        n = 3000
        counts = Counter(
            sample_uniform(**_kwargs(st, brs, sk, round_num=r, np_rng=rng)).count
            for r in range(n)
        )
        for c in (1, 2, 3):
            assert 0.85 * n / 3 < counts[c] < 1.15 * n / 3

    def test_skills_are_uniform(self):
        """With count fixed at 1, every skill should be picked ~equally."""
        brs, sk = _bare_requests(), _skills(6)
        st = _state(brs, sk, max_count=2)  # support = {1}
        rng = np.random.default_rng(17)
        n = 3000
        picks = Counter()
        for r in range(n):
            task = sample_uniform(**_kwargs(st, brs, sk, round_num=r, np_rng=rng))
            assert task.count == 1
            picks[task.selected_skills[0].name] += 1
        assert len(picks) == 6
        for v in picks.values():
            assert 0.8 * n / 6 < v < 1.2 * n / 6


# ---------------------------------------------------------------------------
# 4. Compatibility handling
# ---------------------------------------------------------------------------

class TestCompatibility:
    def test_selected_skills_are_mutually_compatible(self):
        # Half the pool mutually conflicts, so a naive pick would violate.
        sk = _skills(4)
        for name in ("s0", "s1"):
            sk[name].tags = {"x"}
            sk[name].incompatible_tags = {"x"}
        brs = _bare_requests()
        st = _state(brs, sk)
        rng = np.random.default_rng(5)
        for r in range(300):
            task = sample_uniform(**_kwargs(st, brs, sk, round_num=r, np_rng=rng))
            assert task is not None
            chosen: list[Skill] = []
            for s in task.selected_skills:
                assert is_compatible(s, chosen)
                chosen.append(s)

    def test_returns_none_when_no_compatible_combo_exists(self):
        sk = _skills(4, conflicting=True)  # at most 1 skill can ever be chosen
        brs = _bare_requests()
        st = _state(brs, sk, max_count=3)
        rng = np.random.default_rng(2)
        # lower_bound=2 forces count=2, which is unsatisfiable.
        task = sample_uniform(
            **_kwargs(st, brs, sk, np_rng=rng, lower_bound=2)
        )
        assert task is None


# ---------------------------------------------------------------------------
# 5. Dedup budget
# ---------------------------------------------------------------------------

class TestDedup:
    def _single_combo_setup(self):
        sk = _skills(1)
        brs = [BareRequest(prompt="only-q")]
        st = _state(brs, sk, max_count=2)  # support = {1}, single skill
        return brs, sk, st

    def test_combo_reused_up_to_max_compositions_then_exhausts(self):
        brs, sk, st = self._single_combo_setup()
        rng = np.random.default_rng(1)
        kw = dict(max_compositions=2, np_rng=rng)
        assert sample_uniform(**_kwargs(st, brs, sk, round_num=1, **kw)) is not None
        assert sample_uniform(**_kwargs(st, brs, sk, round_num=2, **kw)) is not None
        # Only combo is used up -> every retry collides -> skip.
        assert sample_uniform(**_kwargs(st, brs, sk, round_num=3, **kw)) is None

    def test_tried_combinations_counter_is_updated(self):
        brs, sk, st = self._single_combo_setup()
        rng = np.random.default_rng(1)
        sample_uniform(**_kwargs(st, brs, sk, np_rng=rng, max_compositions=3))
        counter = st.tried_combinations["only-q"]
        assert counter[(frozenset({"s0"}), ())] == 1


# ---------------------------------------------------------------------------
# 6. No bandit state is touched
# ---------------------------------------------------------------------------

class TestNoBanditStateMutation:
    def test_profiles_and_logits_unchanged(self):
        brs, sk = _bare_requests(), _skills(6)
        st = _state(brs, sk)
        logits_before = list(st.skill_logits)
        rng = np.random.default_rng(21)
        for r in range(50):
            sample_uniform(**_kwargs(st, brs, sk, round_num=r, np_rng=rng))
        assert st.skill_logits == logits_before
        for p in list(st.bare_request_profiles.values()) + list(
            st.skill_profiles.values()
        ):
            assert (p.alpha, p.beta) == (0.3, 0.7)

    def test_skip_does_not_penalize_skills(self):
        """`_sample` penalizes skills on a dedup collision; uniform must not."""
        brs, sk, st = TestDedup()._single_combo_setup()
        rng = np.random.default_rng(1)
        kw = dict(max_compositions=1, np_rng=rng, skip_penalty=0.5)
        sample_uniform(**_kwargs(st, brs, sk, round_num=1, **kw))
        assert sample_uniform(**_kwargs(st, brs, sk, round_num=2, **kw)) is None
        assert st.skill_profiles["s0"].beta == 0.7


# ---------------------------------------------------------------------------
# 7. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def _trace(self, seed: int) -> list[tuple]:
        brs, sk = _bare_requests(), _skills(6)
        st = _state(brs, sk)
        rng = np.random.default_rng(seed)
        out = []
        for r in range(30):
            t = sample_uniform(**_kwargs(st, brs, sk, round_num=r, np_rng=rng))
            out.append((t.q_key, t.count, tuple(s.name for s in t.selected_skills)))
        return out

    def test_same_seed_same_trace(self):
        assert self._trace(99) == self._trace(99)

    def test_different_seed_different_trace(self):
        assert self._trace(99) != self._trace(100)


# ---------------------------------------------------------------------------
# 8. Selector dispatch
# ---------------------------------------------------------------------------

class TestSelectorDispatch:
    def test_default_is_thompson(self):
        name, fn = _resolve_selector({})
        assert name == "thompson"
        assert fn is _sample
        assert SGBSv2({}).selector == "thompson"

    def test_uniform_selected_from_config(self):
        cfg = {"search": {"selector": "uniform"}}
        name, fn = _resolve_selector(cfg)
        assert name == "uniform"
        assert fn is sample_uniform
        runner = SGBSv2(cfg)
        assert runner.selector == "uniform"
        assert runner._sample_fn is sample_uniform

    def test_unknown_selector_raises(self):
        with pytest.raises(ValueError, match="Unsupported search.selector"):
            _resolve_selector({"search": {"selector": "epsilon-greedy"}})

    def test_uniform_ignores_a_skewed_posterior(self):
        """The behavioural difference: Thompson chases the strong arm,
        uniform does not even look at it."""
        brs, sk = _bare_requests(), _skills(6)
        st_t = _state(brs, sk, max_count=2)  # count fixed at 1
        st_u = _state(brs, sk, max_count=2)
        for st in (st_t, st_u):
            st.skill_profiles["s0"] = ucb.ArmProfile(alpha=200.0, beta=1.0)

        n = 300
        t_hits = u_hits = 0
        rng_t = np.random.default_rng(31)
        rng_u = np.random.default_rng(31)
        for r in range(n):
            t = _sample(**_kwargs(st_t, brs, sk, round_num=r, np_rng=rng_t))
            u = sample_uniform(**_kwargs(st_u, brs, sk, round_num=r, np_rng=rng_u))
            t_hits += t.selected_skills[0].name == "s0"
            u_hits += u.selected_skills[0].name == "s0"
        assert t_hits > 0.9 * n           # exploitation
        assert 0.08 * n < u_hits < 0.28 * n  # ~1/6, no exploitation


# ---------------------------------------------------------------------------
# 9. End-to-end through the real search loop
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_CIF_SKILLS = _ROOT / "skills/constrained_completion/skill_dictionary.json"
_CIF_REQUESTS = _ROOT / "data/constrained_completion/bare_requests.jsonl"


def _fake_target():
    """Target model stub: deterministic per (prompt, seed)."""
    class _Target:
        def generate(self, prompt, seed=None):
            return f"response<{seed}>"
    return _Target()


def _coin_judge(p_pass: float = 0.5):
    """Judge stub: verdict from a stable hash of the response, so p_hat lands
    near p_pass and some rounds reach the fine stage / become edge cases.

    A plain class, not MagicMock: `judge_and_capture` probes the judge for
    optional `get_last_error` / `get_last_label` / `get_last_meta`, and a mock
    would auto-create them and return unserializable objects.
    """
    class _Judge:
        def judge(self, messages, metadata=None):
            h = zlib.crc32(messages[-1]["content"].encode()) % 1000
            return h < p_pass * 1000
    return _Judge()


def _loop_config(selector: str, budget: int = 40) -> dict:
    return {
        "experiment": {"seed": 42},
        "search": {
            "selector": selector,
            "budget": budget,
            "target_edge_cases": 10**6,   # never stop early
            "diminishing_returns": 10**6,
            "batch_size": 4,
            "max_compositions": 3,
        },
        "ucb": {"prior_alpha": 0.3, "prior_beta": 0.7, "c": 1.0,
                "skill_method": "thompson"},
        "skill_count": {"max_count": 4, "bare_request_bias": 3.0,
                        "lower_bound": 1, "top_k": 3, "top_p": 0.9,
                        "alpha": 0.05, "beta": 0.1},
        "skills": {"mis_size": 8},
        "composer": {"type": "template"},
        "evaluator": {"coarse_n": 3, "fine_n": 2,
                      "coarse_range": [0.3, 0.7], "fine_range": [0.45, 0.55]},
        "bare_requests": {"category": "constrained_completion"},
    }


@pytest.mark.skipif(
    not (_CIF_SKILLS.exists() and _CIF_REQUESTS.exists()),
    reason="constrained_completion skill/bare-request data not available",
)
class TestEndToEndLoop:
    """Drive the real SGBSv2 loop with the real CIF skill dictionary and
    TemplateComposer, stubbing only the target model and judge. Catches wiring
    bugs the unit tests can't: composer kwargs, metadata merge, serialization,
    resume."""

    def _run(self, selector: str, out_dir: Path, budget: int = 40):
        out_dir.mkdir(parents=True, exist_ok=True)  # run_sgbs.py does this
        skills = load_skills(str(_CIF_SKILLS))
        bare_requests = []
        with open(_CIF_REQUESTS) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    bare_requests.append(
                        BareRequest(prompt=e["prompt"],
                                    metadata=e.get("metadata", {})),
                    )
                if len(bare_requests) >= 20:
                    break
        runner = SGBSv2(_loop_config(selector, budget))
        return runner.run(
            target_model=_fake_target(),
            judge=_coin_judge(),
            bare_requests=bare_requests,
            skills=skills,
            output_dir=str(out_dir),
        ), skills

    def test_uniform_run_completes_and_serializes(self, tmp_path):
        result, skills = self._run("uniform", tmp_path)

        assert result.stats["rounds_run"] == 40
        assert len(result.round_records) + result.skipped == 40
        for rec in result.round_records:
            assert 1 <= rec.skill_count <= 3
            assert len(rec.skills) == rec.skill_count
            assert set(rec.skills) <= set(skills)
            assert rec.prompt  # composer produced something

        save_results(result, str(tmp_path), config=_loop_config("uniform"))
        for name in ("trial_details.jsonl", "search_state.json", "stats.json"):
            assert (tmp_path / name).exists()

        # Resume path must accept a uniform run dir unchanged.
        with open(tmp_path / "config.yaml", "w") as f:
            yaml.dump(_loop_config("uniform"), f)
        prev = load_search_state(str(tmp_path))
        assert prev["state"].round_num == 40
        assert prev["config"]["search"]["selector"] == "uniform"

    def test_thompson_run_still_works(self, tmp_path):
        """Regression: the default path must be unaffected by the dispatch."""
        result, _ = self._run("thompson", tmp_path)
        assert result.stats["rounds_run"] == 40
        assert len(result.round_records) + result.skipped == 40

    def test_arms_explore_differently(self, tmp_path):
        """Same seed, same data: the two selectors must not produce the same
        trajectory (otherwise the dispatch silently fell through)."""
        uni, _ = self._run("uniform", tmp_path / "u")
        tho, _ = self._run("thompson", tmp_path / "t")
        uni_trace = [(r.bare_request, tuple(r.skills)) for r in uni.round_records]
        tho_trace = [(r.bare_request, tuple(r.skills)) for r in tho.round_records]
        assert uni_trace != tho_trace


if __name__ == "__main__":
    pytest.main([__file__, "-v"])