"""Rule-based judge for constrained_completion category.

Dispatches all 83 constraint types (25 IFEval + 58 IFBench) to the bundled
upstream Python verifier classes via `judges.instruction_bridge`. No behavior
is re-implemented here — every check is delegated to the canonical
`check_following` method from IFEval (Zhou et al. 2023) or IFBench
(Pyatkin et al. 2025), which minimizes drift from the published benchmarks.

References:
  - skills/constrained_completion/verifiers/ifeval/instructions.py   (IFEval)
  - skills/constrained_completion/verifiers/ifbench/instructions.py  (IFBench)
"""

from logger import get_logger

from .instruction_bridge import check as _check, type_names

logger = get_logger(__name__)


_THINK_CLOSER = "</think>"


def _strip_reasoning(text: str) -> str:
    """Strip leading <think>...</think> reasoning block if present.

    CIF-specific: rule-based constraint checkers (IFEval/IFBench) score the
    full response string — word counts, keyword matches, JSON validation,
    section counters. Embedded reasoning blocks pollute every such check.

    Split on the LAST </think>; content after is the final answer. If
    </think> is absent (truncated mid-CoT or model emitted no reasoning) or
    the post-</think> region is empty, return the original text unchanged so
    nothing is silently lost.
    """
    if _THINK_CLOSER not in text:
        return text
    post = text.rsplit(_THINK_CLOSER, 1)[1].lstrip()
    return post if post else text


class ConstraintJudge:
    """Rule-based judge for constrained_completion.

    Stateless: reads constraints from metadata["constraints"] each call.
    Implements the Judge protocol: judge(messages, metadata) -> bool.
    """

    def judge(
        self,
        messages: list[dict],
        metadata: dict | None = None,
    ) -> bool:
        """Return True iff the last assistant response satisfies ALL constraints."""
        response = self._extract_last_response(messages)
        constraints = (metadata or {}).get("constraints", [])
        if not constraints:
            return True

        known = type_names()
        for constraint in constraints:
            ctype = constraint["type"]
            kwargs = constraint.get("kwargs", {})

            if ctype not in known:
                logger.warning("No checker for constraint type: %s", ctype)
                continue
            if not _check(ctype, kwargs, response):
                return False

        return True

    def judge_detailed(
        self,
        messages: list[dict],
        metadata: dict | None = None,
    ) -> dict:
        """Return per-constraint pass/fail plus overall pass.

        Shape: {"passed": bool, "results": [{"type": str, "passed": bool}, ...]}
        """
        response = self._extract_last_response(messages)
        constraints = (metadata or {}).get("constraints", [])
        if not constraints:
            return {"passed": True, "results": []}

        known = type_names()
        results = []
        all_passed = True
        for constraint in constraints:
            ctype = constraint["type"]
            kwargs = constraint.get("kwargs", {})

            if ctype not in known:
                results.append({"type": ctype, "passed": True, "note": "no checker"})
                continue

            passed = _check(ctype, kwargs, response)
            results.append({"type": ctype, "passed": passed})
            if not passed:
                all_passed = False

        return {"passed": all_passed, "results": results}

    @staticmethod
    def _extract_last_response(messages: list[dict]) -> str:
        """Return the content of the last assistant message, or empty string.

        Strips a leading <think>...</think> reasoning block defensively so the
        constraint checkers only see the final answer; see _strip_reasoning.
        """
        for m in reversed(messages):
            if m["role"] == "assistant":
                return _strip_reasoning(m["content"])
        return ""
