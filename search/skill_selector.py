"""Skill selection with compatibility filtering.

Takes pre-computed scores (from Thompson Sampling or UCB) and greedily
picks the top-k compatible skills.
"""

import json
from dataclasses import dataclass, field

from logger import get_logger

logger = get_logger(__name__)


@dataclass
class Skill:
    """A single skill from the dictionary."""
    name: str
    description: str
    tags: set[str] = field(default_factory=set)
    category: str = ""
    incompatible_tags: set[str] = field(default_factory=set)
    extra: dict = field(default_factory=dict)


def is_compatible(candidate: Skill, selected: list[Skill]) -> bool:
    """Check if candidate skill is compatible with all already-selected skills.

    Two skills conflict if either's tags intersect the other's incompatible_tags.
    """
    for s in selected:
        if candidate.name == s.name:
            return False
        if candidate.tags & s.incompatible_tags:
            return False
        if s.tags & candidate.incompatible_tags:
            return False
    return True


def select_skills(
    count: int,
    scores: dict[str, float],
    skills: dict[str, Skill],
) -> list[Skill] | None:
    """Select `count` compatible skills by greedily picking from ranked scores.

    Args:
        count: Number of skills to select.
        scores: Pre-computed score for each skill key (from thompson_scores or
                ucb_scores).  Higher is better.
        skills: Skill dictionary (name -> Skill).

    Returns:
        List of Skill objects, or None if not enough compatible skills found.
    """
    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    selected: list[Skill] = []

    for key in sorted_keys:
        if key not in skills:
            continue
        candidate = skills[key]
        if is_compatible(candidate, selected):
            selected.append(candidate)
            if len(selected) == count:
                return selected

    return None


_CORE_KEYS = {"name", "description", "tags", "category", "incompatible_tags"}


def load_skills(
    path: str, category: str | None = None
) -> dict[str, Skill]:
    """Load skill dictionary from a JSON file.

    Expected format: list of dicts with at least "name" and "description".
    Core keys (name, description, tags, category, incompatible_tags) map to
    Skill fields; all other keys go into skill.extra.

    Args:
        path: Path to the JSON file.
        category: If provided, filter skills to this category.

    Returns:
        Dict mapping skill name to Skill object.
    """
    with open(path, "r") as f:
        raw = json.load(f)

    skills = {}
    for entry in raw:
        extra = {k: v for k, v in entry.items() if k not in _CORE_KEYS}
        skill = Skill(
            name=entry["name"],
            description=entry.get("description", ""),
            tags=set(entry.get("tags", [])),
            category=entry.get("category", ""),
            incompatible_tags=set(entry.get("incompatible_tags", [])),
            extra=extra,
        )
        if category and skill.category and skill.category != category:
            continue
        skills[skill.name] = skill

    logger.info("Loaded %d skills from %s (category=%s)", len(skills), path, category)
    return skills
