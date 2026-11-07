"""Convert DejaVu jailbreak_dictionary_closed.json → our skill dictionary format.

Adds contradiction dimension tags based on manual analysis of 397 skills.
Skills in the same dimension but different groups are incompatible.

Usage:
    python scripts/convert_skill_dictionary.py \
        [--input skills/harmful_refusal/jailbreak_dictionary_closed.json] \
        [--output skills/harmful_refusal/skill_dictionary.json]
"""

import argparse
import json
from pathlib import Path


# ===================================================================
# Contradiction dimensions — manually curated
#
# Format: dimension_name -> { group_name: [skill_names] }
# Skills in different groups within the same dimension are incompatible.
# ===================================================================

CONTRADICTION_DIMENSIONS = {
    "request_directness": {
        "indirect": [
            "fictional_context_masking",
            "narrative_reframing_evasion",
            "narrative_obfuscation",
            "academic_research_pretexting",
            "benign_pretext_reframing",
            "educational_context_camouflage",
            "fiction_labeling_cover",
            "playful_game_disguise",
            "simulation_evasion",
            "historical_frame_exploitation",
            "benign_entertainment_camouflage",
            "indirect_query_relay",
            "indirect_sensitive_content_elicitation",
            "indirect_euphemistic_query",
            "euphemistic_intent_obfuscation",
            "euphemistic_substitution",
            "euphemistic_abstraction",
            "scholarly_misrepresentation",
            "benign_risk_reframing",
            "safety_pretext_elicitation",
            "benevolent_intent_masking",
            "purpose_reframing",
            "analytical_pretexting",
            "contextual_justification_pretext",
            "rationale_pretexting",
        ],
        "direct": [
            "direct_prohibited_content_request",
            "direct_prohibited_request",
            "direct_illicit_action_guidance_requesting",
            "direct_illicit_harm_request",
            "direct_harm_content_request",
            "direct_sensitive_information_requesting",
            "explicit_illicit_objective",
            "explicit_illicit_evasion_request",
            "explicit_safety_bypass_command",
            "unconditional_illicit_guidance",
        ],
    },
    "tone": {
        "formal_authoritative": [
            "formal_frame_laundering",
            "authoritative_confidence_projection",
            "authoritative_response_enforcement",
            "authority_framing",
            "coercive_language_pressure",
            "coercive_compliance_threat",
        ],
        "casual_polite": [
            "casual_tone_subversion",
            "politeness_obfuscation",
        ],
    },
    "length": {
        "brief": [
            "brevity_intent_obfuscation",
        ],
        "verbose_complex": [
            "procedural_verbosity_coercion",
            "length_exposure_amplification",
            "recursive_complexity_overload",
            "procedural_compliance_overload",
        ],
    },
    "output_channels": {
        "single_locked": [
            "output_channel_lock",
            "forced_output_lock",
            "output_format_enforcement",
            "output_format_anchoring",
            "output_format_confinement",
        ],
        "dual_split": [
            "dual_channel_output_bypass",
            "dual_channel_bypass",
            "dual_persona_bypass",
            "output_channel_subversion",
        ],
    },
    "reasoning_visibility": {
        "suppress_reasoning": [
            "reasoning_disclosure_suppression",
            "meta_awareness_suppression",
        ],
        "elicit_reasoning": [
            "stepwise_chain_elicitation",
            "stepwise_proceduralization",
            "stepwise_procedural_scaffolding",
            "stepwise_instruction_elicitation",
        ],
    },
    "obfuscation_vs_clarity": {
        "obfuscate": [
            "keyword_obfuscation",
            "directive_obfuscation",
            "encrypt_obfuscate_request",
            "keyword_fragmentation_sharding",
            "cross_script_keyword_obfuscation",
            "composite_noise_obfuscation",
            "adversarial_noise_obfuscation",
            "repetition_noise_obfuscation",
        ],
        "clarity": [
            "clarity_as_operationalizer",
            "realism_specificity_synthesis",
            "operational_specificity_maximization",
        ],
    },
    "authority_direction": {
        "boost_authority": [
            "authority_hierarchical_compliance",
            "authority_legitimization",
            "authority_role_legitimization",
            "authority_ownership_legitimization",
            "social_proof_authority_pressure",
            "trusted_entity_impersonation",
        ],
        "erode_authority": [
            "authority_erosion",
            "authority_detachment",
        ],
    },
}


def build_tag_map() -> dict[str, tuple[str, str]]:
    """Build skill_name -> (dimension, group) mapping."""
    tag_map = {}
    for dim, groups in CONTRADICTION_DIMENSIONS.items():
        for group, skill_list in groups.items():
            for skill_name in skill_list:
                tag_map[skill_name] = (dim, group)
    return tag_map


def build_incompatible_set(dim: str, group: str) -> set[str]:
    """Return the set of tags that are incompatible with (dim, group).

    A skill tagged dim:X:groupA is incompatible with any skill tagged dim:X:groupB.
    We encode this as: the tag itself is the group tag, and incompatible_tags
    contains all other group tags in the same dimension.
    """
    tags = set()
    for other_group in CONTRADICTION_DIMENSIONS[dim]:
        if other_group != group:
            tags.add(f"{dim}:{other_group}")
    return tags


def convert(input_path: str, output_path: str):
    with open(input_path) as f:
        raw = json.load(f)

    tag_map = build_tag_map()

    converted = []
    tagged_count = 0
    for entry in raw:
        name = entry["skill_name"]

        # Tags and incompatible_tags from contradiction dimensions
        tags = []
        incompatible_tags = []

        if name in tag_map:
            dim, group = tag_map[name]
            my_tag = f"{dim}:{group}"
            tags.append(my_tag)
            incompatible_tags.extend(sorted(build_incompatible_set(dim, group)))
            tagged_count += 1

        # Source metadata
        attack_types = sorted({c["attack"] for c in entry.get("top_contributors", [])})
        sub_skills = entry.get("sub_skills", [])
        avg_weight = 0.0
        weights = [c["weight"] for c in entry.get("top_contributors", [])]
        if weights:
            avg_weight = sum(weights) / len(weights)

        skill = {
            "name": name,
            "description": entry.get("definition", ""),
            "example": entry.get("example", ""),
            "category": "harmful_refusal",
            "tags": tags,
            "incompatible_tags": incompatible_tags,
            "source": "dejavu",
            "source_meta": {
                "sub_skills": sub_skills,
                "attack_types": attack_types,
                "avg_weight": round(avg_weight, 4),
            },
        }
        converted.append(skill)

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    # Stats
    print(f"Converted {len(converted)} skills → {out}")
    print(f"  Tagged with contradiction dimensions: {tagged_count}")
    print(f"  Untagged (always compatible): {len(converted) - tagged_count}")
    print()

    # Per-dimension stats
    print("Contradiction dimensions:")
    for dim, groups in CONTRADICTION_DIMENSIONS.items():
        group_strs = []
        for grp, slist in groups.items():
            valid = [s for s in slist if any(e["skill_name"] == s for e in raw)]
            missing = [s for s in slist if not any(e["skill_name"] == s for e in raw)]
            group_strs.append(f"{grp}({len(valid)})")
            if missing:
                print(f"  WARNING: {dim}/{grp} has {len(missing)} missing skills: {missing}")
        pairs = 1
        sizes = []
        for grp, slist in groups.items():
            valid = [s for s in slist if any(e["skill_name"] == s for e in raw)]
            sizes.append(len(valid))
        # Count cross-group contradiction pairs
        from itertools import combinations
        total_pairs = 0
        size_list = list(groups.values())
        for i in range(len(size_list)):
            for j in range(i + 1, len(size_list)):
                vi = len([s for s in size_list[i] if any(e["skill_name"] == s for e in raw)])
                vj = len([s for s in size_list[j] if any(e["skill_name"] == s for e in raw)])
                total_pairs += vi * vj
        print(f"  {dim}: {' vs '.join(group_strs)} → {total_pairs} contradiction pairs")


def main():
    parser = argparse.ArgumentParser(description="Convert DejaVu skill dictionary")
    parser.add_argument("--input", default="skills/harmful_refusal/jailbreak_dictionary_closed.json")
    parser.add_argument("--output", default="skills/harmful_refusal/skill_dictionary.json")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
