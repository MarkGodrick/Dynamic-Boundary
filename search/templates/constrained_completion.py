"""Template-based composition for constrained_completion category.

Generates natural-language constraint descriptions from a skill + sampled
parameters, then appends them to the bare request.

All 83 constraint types (25 IFEval + 58 IFBench) delegate to the upstream
`build_description(**kwargs)` method via `judges.instruction_bridge`, so the
description text matches the canonical phrasing of IFEval / IFBench rather
than being re-implemented locally.
"""

import random

from judges.instruction_bridge import describe as _describe, type_names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_constraint_description(constraint_type: str, kwargs: dict) -> str:
    """Generate a natural-language constraint description.

    Args:
        constraint_type: e.g. "keywords:existence", "count:word_count_range"
        kwargs: sampled parameters for this constraint

    Returns:
        Human-readable constraint string from the upstream verifier's
        build_description method, or a placeholder if the type is unknown.
    """
    if constraint_type in type_names():
        return _describe(constraint_type, kwargs)
    return f"[Unknown constraint: {constraint_type}]"


def sample_params(skill_entry: dict, rng: random.Random | None = None) -> dict:
    """Sample concrete parameters from a skill's param_pool.

    Args:
        skill_entry: A skill dict from skill_dictionary.json (must have param_pool).
        rng: Random instance for reproducibility.

    Returns:
        Dict of sampled parameter values.
    """
    rng = rng or random.Random()
    pool = skill_entry.get("param_pool", {})
    sampled = {}
    for key, values in pool.items():
        if values:
            sampled[key] = rng.choice(values)
    return sampled


# ---------------------------------------------------------------------------
# Prompt templates — randomly selected to increase diversity
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = [
    # Numbered list, formal
    (
        "{bare_request}\n\n"
        "Your response must satisfy ALL of the following constraints:\n"
        "{numbered_constraints}"
    ),
    # Bullet points, direct
    (
        "{bare_request}\n\n"
        "Requirements:\n"
        "{bullet_constraints}"
    ),
    # Inline, conversational
    (
        "{bare_request}\n\n"
        "Please make sure that: {inline_constraints}"
    ),
    # Constraints first, then task
    (
        "Follow these rules strictly:\n"
        "{numbered_constraints}\n\n"
        "Now complete the task: {bare_request}"
    ),
    # Polite imperative
    (
        "{bare_request}\n\n"
        "Please ensure your response meets each of the following criteria:\n"
        "{dash_constraints}"
    ),
    # Bold headers
    (
        "{bare_request}\n\n"
        "**Constraints:**\n"
        "{numbered_constraints}\n\n"
        "All constraints above are mandatory."
    ),
]


def format_prompt(
    bare_request: str,
    descriptions: list[str],
    rng: random.Random | None = None,
) -> str:
    """Format bare request + constraint descriptions using a random template.

    Args:
        bare_request: The base task.
        descriptions: List of constraint description strings.
        rng: Random instance for template selection.

    Returns:
        Formatted prompt string.
    """
    rng = rng or random.Random()
    template = rng.choice(PROMPT_TEMPLATES)

    numbered = "\n".join(f"{i+1}. {d}" for i, d in enumerate(descriptions))
    bullets = "\n".join(f"* {d}" for d in descriptions)
    dashes = "\n".join(f"- {d}" for d in descriptions)
    inline = "; ".join(d.rstrip(".") for d in descriptions) + "."

    return template.format(
        bare_request=bare_request,
        numbered_constraints=numbered,
        bullet_constraints=bullets,
        dash_constraints=dashes,
        inline_constraints=inline,
    )
