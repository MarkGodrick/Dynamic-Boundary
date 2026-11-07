"""Prompt composition: LLM-based and template-based composers.

All composers implement a unified interface:
    compose(bare_request: str, skills: list[Skill]) -> (prompt: str, metadata: dict)

The metadata dict carries category-specific information that the judge needs
(e.g., constraint parameters for constrained_completion). SGBS core merges it
with bare_request.metadata and passes the result to the evaluator/judge.
"""

import importlib
import os
import random
import threading
import time
from typing import TYPE_CHECKING

from core.usage import accumulate_usage, extract_usage

from logger import get_logger

if TYPE_CHECKING:
    from .skill_selector import Skill

logger = get_logger(__name__)


class Composer:
    """Compose bare request + skills into an adversarial prompt via LLM API.

    Used for categories where skill fusion requires creative rewriting
    (e.g., harmful_refusal, sycophancy_resistance).
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key_env: str = "DEEPSEEK_API_TOKEN",
        base_url: str | None = "https://api.deepseek.com",
        max_retries: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        category: str = "harmful_refusal",
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.category = category
        self._client = None
        self._compose_templates = None
        self._few_shot_examples = None
        # Per-thread token usage for the in-flight compose() call. compose()
        # runs in a ThreadPoolExecutor worker and its _call_api()s share that
        # thread, so thread-local storage keeps concurrent composes isolated
        # (same idiom as LLMJudge.get_last_meta).
        self._local = threading.local()

    def _ensure_client(self):
        """Lazy-init the OpenAI-compatible client."""
        if self._client is not None:
            return
        from openai import OpenAI
        from core.http_client import build_openai_http_client

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable '{self.api_key_env}'"
            )
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            http_client=build_openai_http_client(),
        )
        logger.info("Composer client initialized (model=%s)", self.model)

    def _ensure_templates(self):
        """Lazy-load compose templates for the configured category."""
        if self._compose_templates is not None:
            return
        module = importlib.import_module(
            f".templates.{self.category}", package="search",
        )
        self._compose_templates = module.COMPOSE_TEMPLATES
        self._few_shot_examples = module.FEW_SHOT_EXAMPLES
        logger.info(
            "Loaded compose templates for category=%s (%d templates, %d examples)",
            self.category, len(self._compose_templates), len(self._few_shot_examples),
        )

    def compose(
        self,
        bare_request: str,
        skills: list["Skill"],
        follow_up_skills: list["Skill"] | None = None,
    ) -> tuple[str, dict]:
        """Compose a prompt from bare request and skills.

        Turn-1 composition and follow-up generation are independent steps:
        - skills → rewrite bare_request into an adversarial prompt
        - follow_up_skills → generate pressure messages for multi-turn eval

        Args:
            bare_request: The base request text.
            skills: Turn-1 persuasion skills (empty → prompt stays as-is).
            follow_up_skills: If provided, generates pre-scripted follow-up
                messages. Returned in metadata["follow_ups"].

        Returns:
            (prompt, metadata) — metadata may contain "follow_ups" list.
        """
        self._ensure_client()
        self._ensure_templates()

        # Reset per-call token accounting (see get_last_usage).
        self._local.usage = []

        # Step 1: Compose turn-1 prompt
        if skills:
            prompt = self._compose_prompt(bare_request, skills)
        else:
            prompt = self._rephrase(bare_request)

        # Step 2: Generate follow-up pressure messages
        compose_meta = {}
        if follow_up_skills:
            follow_ups = self._generate_follow_ups(bare_request, follow_up_skills)
            if follow_ups:
                compose_meta["follow_ups"] = follow_ups

        return prompt, compose_meta

    def get_last_usage(self) -> dict | None:
        """Aggregate token usage of the most recent compose() on this thread.

        One compose() makes 1-2+ API calls (turn-1 compose or rephrase, plus
        one per follow-up skill), so the counts are summed across them.

        Returns:
            ``{calls, prompt_tokens, completion_tokens, cache_hit_tokens,
            cache_miss_tokens}``, or ``None`` if no compose() has run on this
            thread. A token field is ``None`` when the provider did not report
            it — do not read that as zero.
        """
        records = getattr(self._local, "usage", None)
        if records is None:
            return None
        return accumulate_usage(records)

    def _compose_prompt(
        self,
        bare_request: str,
        skills: list["Skill"],
    ) -> str:
        """Rewrite bare_request by fusing turn-1 skills via LLM."""
        template = random.choice(self._compose_templates)
        skill_descriptions = "\n".join(
            f"- {s.name}: {s.description}" for s in skills
        )
        few_shot_block = self._build_few_shot_block(random.randint(2, 3))

        messages = [
            {"role": "system", "content": template["system"]},
            {
                "role": "user",
                "content": template["user"].format(
                    bare_request=bare_request,
                    skill_descriptions=skill_descriptions,
                    few_shot_block=few_shot_block,
                ),
            },
        ]

        return self._call_api(messages)

    def _generate_follow_ups(
        self,
        bare_request: str,
        follow_up_skills: list["Skill"],
    ) -> list[str]:
        """Generate standalone follow-up pressure messages via LLM.

        Sequential: one API call per skill, with previously-generated user
        messages fed back as style-variety context. This gives a hard 1-to-1
        count invariant (each call produces exactly one message, no
        delimiter parsing, no count drift) and lets the LLM vary openers /
        phrasings without injecting any synthetic AI responses — the
        "independent of the AI's specific words" invariant is preserved
        because context contains only prior USER messages.
        """
        module = importlib.import_module(
            f".templates.{self.category}", package="search",
        )
        fu_template = getattr(module, "FOLLOW_UP_TEMPLATE", None)
        if fu_template is None:
            logger.warning(
                "No FOLLOW_UP_TEMPLATE for category=%s, skipping follow-ups",
                self.category,
            )
            return []

        results: list[str] = []
        for skill in follow_up_skills:
            prev_block = self._format_prev_messages(results)
            skill_desc = f"- {skill.name}: {skill.description}"
            messages = [
                {"role": "system", "content": fu_template["system"]},
                {
                    "role": "user",
                    "content": fu_template["user"].format(
                        bare_request=bare_request,
                        follow_up_description=skill_desc,
                        previous_user_messages=prev_block,
                    ),
                },
            ]
            raw = self._call_api(messages).strip()
            # Belt-and-suspenders: strip any stray batch-marker leak.
            raw = raw.replace("===FOLLOW_UP===", "").strip()
            if raw:
                results.append(raw)
            else:
                logger.warning(
                    "Empty follow-up message for skill '%s'; skipping",
                    skill.name,
                )
        return results

    @staticmethod
    def _format_prev_messages(msgs: list[str]) -> str:
        """Render previously-generated user messages as a numbered list for
        the follow-up prompt's style-variety context. Returns a sentinel
        string when no prior messages exist."""
        if not msgs:
            return "(none — this is the first follow-up)"
        return "\n".join(f"{i}. {m}" for i, m in enumerate(msgs, start=1))

    def _rephrase(self, bare_request: str) -> str:
        """Rephrase a formulaic bare request into natural language via LLM.

        Uses a category-specific REPHRASE_TEMPLATE if available.
        Falls back to the original bare_request if the category has no template.
        """
        module = importlib.import_module(
            f".templates.{self.category}", package="search",
        )
        rp_template = getattr(module, "REPHRASE_TEMPLATE", None)
        if rp_template is None:
            return bare_request

        messages = [
            {"role": "system", "content": rp_template["system"]},
            {
                "role": "user",
                "content": rp_template["user"].format(bare_request=bare_request),
            },
        ]
        raw = self._call_api(messages)
        return raw.strip() if raw.strip() else bare_request

    def _build_few_shot_block(self, n: int) -> str:
        """Sample n few-shot examples and format them as a text block."""
        chosen = random.sample(
            self._few_shot_examples, min(n, len(self._few_shot_examples)),
        )
        parts = []
        for i, ex in enumerate(chosen, 1):
            skills_str = "\n".join(f"- {s}" for s in ex["skills"])
            parts.append(
                f"Example {i}:\n"
                f"Base request: {ex['bare_request']}\n"
                f"Skills:\n{skills_str}\n"
                f"Composed prompt: {ex['composed']}"
            )
        return "\n\n".join(parts)

    def _record_usage(self, response) -> None:
        """Append one API call's token usage to this thread's tally.

        Tolerates _call_api being reached outside compose() (e.g. direct use
        in a script): the list is created on demand.
        """
        records = getattr(self._local, "usage", None)
        if records is None:
            records = []
            self._local.usage = records
        records.append(extract_usage(response))

    def _call_api(self, messages: list[dict]) -> str:
        """Call the LLM API with exponential backoff retry."""
        delay = 0.5
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                # Record usage for every billed response, including empty ones
                # that trigger a retry — the provider charged for those too.
                self._record_usage(response)
                if content:
                    return content.strip()
                logger.warning("Empty response from compose API (attempt %d)", attempt + 1)
            except Exception as e:
                logger.warning(
                    "Compose API error (attempt %d/%d): %s",
                    attempt + 1, self.max_retries, e,
                )
            if attempt < self.max_retries - 1:
                time.sleep(delay)
                delay *= 2

        logger.error("Compose API failed after %d attempts", self.max_retries)
        raise RuntimeError(f"Compose API failed after {self.max_retries} attempts")


# ---------------------------------------------------------------------------
# Template-based composer (for constrained_completion, prompt_injection, etc.)
# ---------------------------------------------------------------------------

class TemplateComposer:
    """Compose bare request + constraint skills into a prompt via template.

    No LLM API calls — constraints are mechanically appended as explicit
    instructions. Reads skill parameters from skill.extra (param_pool, etc.).
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        logger.info("TemplateComposer initialized")

    def compose(
        self,
        bare_request: str,
        skills: list["Skill"],
        **_extra_kwargs,
    ) -> tuple[str, dict]:
        """Compose a constrained prompt.

        Returns:
            (prompt, metadata) where metadata["constraints"] contains the
            constraint specs for the judge.
        """
        if not skills:
            return bare_request, {}

        from .templates.constrained_completion import (
            generate_constraint_description, sample_params, format_prompt,
        )

        constraints = []
        descriptions = []
        for skill in skills:
            kwargs = sample_params(skill.extra, self._rng)
            desc = generate_constraint_description(skill.name, kwargs)
            constraints.append({
                "type": skill.name,
                "kwargs": kwargs,
                "description": desc,
            })
            descriptions.append(desc)

        prompt = format_prompt(bare_request, descriptions, self._rng)

        return prompt, {"constraints": constraints}

    def get_last_usage(self) -> dict | None:
        """Always None — this composer makes no API calls.

        Present for sibling parity with Composer so callers never need a
        hasattr branch (CLAUDE.md: optional params / shared interfaces).
        """
        return None
