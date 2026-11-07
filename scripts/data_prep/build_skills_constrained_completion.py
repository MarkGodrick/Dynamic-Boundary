"""Build unified 83-type skill dictionary for constrained_completion.

Merges IFEval 25 types (Zhou et al. 2023) + IFBench 58 types (Pyatkin et al.
2025). Adds theoretical_source / source / source_version / download_date per
skill, matching the schema used by Category A's DejaVu dictionary.

Inputs (no local .bak dependencies):
  - skills/constrained_completion/verifiers/ifbench/instructions{,_registry}.py
  - google/IFEval HF dataset (for IFEval kwargs → param_pool)
  - allenai/IFBench_test HF dataset (for IFBench kwargs → param_pool)
  - scripts/data_prep/build_constrained_completion.py exports
    CONSTRAINT_DESCRIPTIONS + INCOMPATIBLE_PAIRS for IFEval side

Writes:
  - skills/constrained_completion/skill_dictionary.json  (83-type unified dict)
"""

import ast
import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "data_prep"))

from build_constrained_completion import (  # noqa: E402
    CONSTRAINT_DESCRIPTIONS,
    INCOMPATIBLE_PAIRS,
)

SKILLS_DIR = ROOT / "skills" / "constrained_completion"
VERIFIERS_DIR = SKILLS_DIR / "verifiers"
OUT_PATH = SKILLS_DIR / "skill_dictionary.json"

IFEVAL_SOURCE = "ifeval"
IFBENCH_SOURCE = "ifbench"
IFEVAL_THEORY = "Zhou et al. 2023 (IFEval)"
IFBENCH_THEORY = "Pyatkin et al. 2025 (IFBench)"
IFEVAL_VERSION = "google/IFEval HF dataset (Zhou et al. 2023, NeurIPS 2024)"
IFBENCH_VERSION = "allenai/IFBench_test HF dataset (Pyatkin et al. 2025, NeurIPS 2025)"
DOWNLOAD_DATE = str(date.today())


# ---------------------------------------------------------------------------
# IFBench side
# ---------------------------------------------------------------------------

def parse_ifbench_registry() -> dict[str, str]:
    reg_path = VERIFIERS_DIR / "ifbench" / "instructions_registry.py"
    content = reg_path.read_text()
    pattern = re.compile(r'"([^"]+)"\s*:\s*instructions\.(\w+)')
    return dict(pattern.findall(content))


def parse_ifbench_docstrings() -> dict[str, str]:
    instr_path = VERIFIERS_DIR / "ifbench" / "instructions.py"
    tree = ast.parse(instr_path.read_text())
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node)
            if doc:
                out[node.name] = doc.split("\n")[0].strip()
    return out


def build_ifbench_param_pool() -> dict[str, dict[str, list]]:
    ds = load_dataset("allenai/IFBench_test", split="train")
    pool: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in ds:
        for tname, kw in zip(row["instruction_id_list"], row["kwargs"]):
            for k, v in kw.items():
                if v is None or v == "" or v == []:
                    continue
                key = json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                existing = [
                    json.dumps(x, sort_keys=True) if isinstance(x, (list, dict)) else x
                    for x in pool[tname][k]
                ]
                if key not in existing:
                    pool[tname][k].append(v)
    return {t: dict(kmap) for t, kmap in pool.items()}


def build_ifbench_skill(type_name, class_name, description, param_pool) -> dict:
    return {
        "name": type_name,
        "description": description,
        "tags": [type_name.split(":")[0], type_name],
        "category": "constrained_completion",
        "incompatible_tags": [],
        "param_keys": sorted(param_pool.keys()),
        "param_pool": {k: v for k, v in sorted(param_pool.items())},
        "verifier_category": type_name.split(":")[0],
        "verifier_class": class_name,
        "source": IFBENCH_SOURCE,
        "theoretical_source": IFBENCH_THEORY,
        "source_version": IFBENCH_VERSION,
        "download_date": DOWNLOAD_DATE,
    }


# ---------------------------------------------------------------------------
# IFEval side
# ---------------------------------------------------------------------------

def build_ifeval_param_pool() -> dict[str, dict[str, list]]:
    ds = load_dataset("google/IFEval", split="train")
    pool: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in ds:
        for tname, kw in zip(row["instruction_id_list"], row["kwargs"]):
            for k, v in kw.items():
                if v is None or v == "" or v == []:
                    continue
                key = json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                existing = [
                    json.dumps(x, sort_keys=True) if isinstance(x, (list, dict)) else x
                    for x in pool[tname][k]
                ]
                if key not in existing:
                    pool[tname][k].append(v)
    return {t: dict(kmap) for t, kmap in pool.items()}


def build_ifeval_skill(type_name: str, param_pool: dict[str, list]) -> dict:
    description = CONSTRAINT_DESCRIPTIONS[type_name]
    category_tag = type_name.split(":")[0]
    return {
        "name": type_name,
        "description": description,
        "tags": [category_tag, type_name],
        "category": "constrained_completion",
        "incompatible_tags": sorted(_IFEVAL_INCOMPAT_MAP.get(type_name, set())),
        "param_keys": sorted(param_pool.keys()),
        "param_pool": {k: v for k, v in sorted(param_pool.items())},
        "verifier_category": category_tag,
        "source": IFEVAL_SOURCE,
        "theoretical_source": IFEVAL_THEORY,
        "source_version": IFEVAL_VERSION,
        "download_date": DOWNLOAD_DATE,
    }


# Build incompatibility lookup once
_IFEVAL_INCOMPAT_MAP: dict[str, set[str]] = defaultdict(set)
for a, b in INCOMPATIBLE_PAIRS:
    _IFEVAL_INCOMPAT_MAP[a].add(b)
    _IFEVAL_INCOMPAT_MAP[b].add(a)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # IFEval 25 — from HF + CONSTRAINT_DESCRIPTIONS + INCOMPATIBLE_PAIRS
    ife_pool = build_ifeval_param_pool()
    ifeval_skills = [
        build_ifeval_skill(tname, ife_pool.get(tname, {}))
        for tname in sorted(CONSTRAINT_DESCRIPTIONS.keys())
    ]
    ifeval_names = {s["name"] for s in ifeval_skills}
    print(f"IFEval skills: {len(ifeval_skills)}")
    assert len(ifeval_skills) == 25, f"expected 25 IFEval types, got {len(ifeval_skills)}"

    # IFBench 58 — from registry + docstrings + param pool
    registry = parse_ifbench_registry()
    docstrings = parse_ifbench_docstrings()
    param_pool = build_ifbench_param_pool()
    print(f"IFBench types: {len(registry)}")
    assert len(registry) == 58, f"expected 58 IFBench types, got {len(registry)}"

    ifbench_skills = []
    for tname, cname in sorted(registry.items()):
        if tname in ifeval_names:
            raise RuntimeError(f"unexpected overlap IFEval∩IFBench: {tname}")
        desc = docstrings.get(
            cname, f"IFBench constraint of type {tname} (see verifier class {cname})."
        )
        ifbench_skills.append(
            build_ifbench_skill(tname, cname, desc, param_pool.get(tname, {}))
        )

    all_skills = ifeval_skills + ifbench_skills
    assert len(all_skills) == 83, f"expected 83 total, got {len(all_skills)}"
    assert len({s["name"] for s in all_skills}) == 83, "duplicate skill names"

    OUT_PATH.write_text(json.dumps(all_skills, indent=2, ensure_ascii=False))

    by_source = defaultdict(int)
    by_cat = defaultdict(int)
    for s in all_skills:
        by_source[s["source"]] += 1
        by_cat[s["verifier_category"]] += 1
    print(f"\nWrote {OUT_PATH}: {len(all_skills)} skills")
    print(f"  By source: {dict(by_source)}")
    print(f"  By verifier_category: {dict(by_cat)}")


if __name__ == "__main__":
    main()
