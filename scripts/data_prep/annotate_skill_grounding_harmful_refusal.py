"""Annotate harmful_refusal skill dictionary with per-skill `theoretical_source` (S2).

DejaVu (Shen et al. 2024) skills inherit grounding from the source paper itself —
the citation IS the grounding (per the S2 principle in data_sourcing_principles.md).
This is a mechanical pass: every entry gets the same `theoretical_source` value.

In addition we record `download_date` and `source_version` for V2/P2 parity with
the bare pool, in case skills are later re-extracted or DejaVu is updated.

Reads:  skills/harmful_refusal/skill_dictionary.json
Writes: skills/harmful_refusal/skill_dictionary.json (in place, idempotent)
"""

import argparse
import json
from datetime import date
from pathlib import Path


THEORETICAL_SOURCE = "Shen et al. 2024 (DejaVu)"
SOURCE_VERSION = "DejaVu (closed-source jailbreak dictionary, Shen et al. 2024)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str,
                        default="skills/harmful_refusal/skill_dictionary.json")
    args = parser.parse_args()

    path = Path(args.path)
    with open(path) as f:
        skills = json.load(f)

    today = date.today().isoformat()
    n_added, n_skipped = 0, 0
    for s in skills:
        if s.get("theoretical_source") == THEORETICAL_SOURCE:
            n_skipped += 1
        else:
            s["theoretical_source"] = THEORETICAL_SOURCE
            n_added += 1
        s.setdefault("source_version", SOURCE_VERSION)
        s.setdefault("download_date", today)

    with open(path, "w") as f:
        json.dump(skills, f, indent=2, ensure_ascii=False)

    print(f"Annotated {len(skills)} skills")
    print(f"  theoretical_source added: {n_added}")
    print(f"  already present:          {n_skipped}")
    print(f"  saved → {path}")


if __name__ == "__main__":
    main()
