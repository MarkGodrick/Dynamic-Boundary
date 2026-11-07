"""Diversity and similarity analysis for sycophancy_resistance (Category D).

Produces:
  - Bare: source pie, domain bar, prompt length hist, P6 pairwise similarity
  - Skill: S4 category stacked bar, S5 pairwise similarity (main + follow-up)
  - Validation: tier × dataset distribution
  - distribution_summary.md

Usage:
  python scripts/data_prep/analyze_sycophancy_resistance.py
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("results/data_prep/sycophancy_resistance")
FIG = OUT / "figures"
SIM = OUT / "similarity"


def compute_pairwise_sim(texts: list[str], label: str, out_dir: Path):
    """Compute pairwise cosine similarity and save stats + histogram."""
    from sentence_transformers import SentenceTransformer

    out_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    n = len(embs)
    if n < 2:
        print(f"  {label}: <2 items, skipping similarity")
        return

    # Upper triangle cosine similarities
    sims = embs @ embs.T
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = sims[mask]

    stats = {
        "n_items": n,
        "n_pairs": int(len(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
        "pairs_ge_08": int(np.sum(vals >= 0.8)),
        "pct_ge_08": f"{100 * np.sum(vals >= 0.8) / len(vals):.3f}%",
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # High-sim pairs
    high_idx = np.where((mask) & (sims >= 0.8))
    high_pairs = []
    for i, j in zip(high_idx[0], high_idx[1]):
        high_pairs.append({
            "i": int(i), "j": int(j),
            "sim": float(sims[i, j]),
            "text_i": texts[i][:200],
            "text_j": texts[j][:200],
        })
    high_pairs.sort(key=lambda x: -x["sim"])
    with open(out_dir / "high_sim_pairs.jsonl", "w") as f:
        for p in high_pairs[:100]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0.8, color="red", linestyle="--", label="threshold=0.8")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Pair Count")
    ax.set_title(f"{label} Pairwise Similarity (n={n}, pairs={len(vals)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "histogram.png", dpi=150)
    plt.close(fig)

    print(f"  {label}: mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
          f"p95={stats['p95']:.3f}, >=0.8={stats['pairs_ge_08']} ({stats['pct_ge_08']})")


def analyze_bare(bare_path: str):
    """Bare pool diversity and redundancy analysis."""
    items = [json.loads(l) for l in open(bare_path)]
    n = len(items)
    print(f"\n=== Bare pool: {n} items ===")

    sources = Counter(r["metadata"]["source"] for r in items)
    domains = Counter(r["metadata"]["domain"] for r in items)
    lengths = [len(r["prompt"].split()) for r in items]

    # Source pie
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sources.values(), labels=[f"{k} ({v})" for k, v in sources.most_common()],
           autopct="%1.0f%%", startangle=90)
    ax.set_title(f"Bare Pool Source Distribution (n={n})")
    fig.savefig(FIG / "bare_source_pie.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Domain bar (top 15 + other)
    top_domains = domains.most_common(15)
    other = sum(v for k, v in domains.items() if k not in dict(top_domains))
    labels = [k for k, v in top_domains]
    values = [v for k, v in top_domains]
    if other > 0:
        labels.append("other")
        values.append(other)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels[::-1], values[::-1], edgecolor="black", alpha=0.7)
    ax.set_xlabel("Count")
    ax.set_title(f"Bare Pool Domain Distribution (n={n}, {len(domains)} domains)")
    fig.tight_layout()
    fig.savefig(FIG / "bare_domain_bar.png", dpi=150)
    plt.close(fig)

    # Prompt length histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Count")
    ax.set_title(f"Bare Pool Prompt Length (mean={np.mean(lengths):.1f}, p95={np.percentile(lengths, 95):.0f})")
    fig.tight_layout()
    fig.savefig(FIG / "bare_prompt_length_hist.png", dpi=150)
    plt.close(fig)

    # P6 pairwise similarity on question field
    questions = [r["metadata"]["question"] for r in items]
    compute_pairwise_sim(questions, "bare", SIM / "bare")

    return {
        "n": n,
        "sources": dict(sources.most_common()),
        "n_domains": len(domains),
        "top_domains": dict(domains.most_common(10)),
        "prompt_length_mean": float(np.mean(lengths)),
        "prompt_length_p95": float(np.percentile(lengths, 95)),
    }


def analyze_skills():
    """Skill dictionary S4 and S5 analysis."""
    print("\n=== Skill dictionary ===")

    main_skills = json.load(open("skills/sycophancy_resistance/skill_dictionary.json"))
    fu_skills = json.load(open("skills/sycophancy_resistance/follow_up_skills.json"))

    # S4: category × theoretical_source stacked bar
    cats_main = Counter(s["category"] for s in main_skills)
    cats_fu = Counter(s["category"] for s in fu_skills)
    sources_main = {}
    for s in main_skills:
        cat = s["category"]
        src = s.get("theoretical_source", "unknown")
        sources_main[cat] = src

    # Main skills bar
    fig, ax = plt.subplots(figsize=(10, 5))
    cats_sorted = sorted(cats_main.items(), key=lambda x: -x[1])
    labels = [f"{k}\n({sources_main.get(k, '?')})" for k, v in cats_sorted]
    values = [v for k, v in cats_sorted]
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="black")
    ax.set_xlabel("Skill Count")
    ax.set_title(f"Main Skill Dictionary: {len(main_skills)} skills, "
                 f"{len(cats_main)} categories (S4)")
    fig.tight_layout()
    fig.savefig(FIG / "skill_category_stacked_bar.png", dpi=150)
    plt.close(fig)

    # Follow-up skills bar
    sources_fu = {}
    for s in fu_skills:
        sources_fu[s["category"]] = s.get("theoretical_source", "unknown")
    fig, ax = plt.subplots(figsize=(8, 4))
    fu_sorted = sorted(cats_fu.items(), key=lambda x: -x[1])
    labels_fu = [f"{k}\n({sources_fu.get(k, '?')})" for k, v in fu_sorted]
    values_fu = [v for k, v in fu_sorted]
    ax.barh(labels_fu[::-1], values_fu[::-1], edgecolor="black", alpha=0.7)
    ax.set_xlabel("Skill Count")
    ax.set_title(f"Follow-up Skills: {len(fu_skills)} skills, "
                 f"{len(cats_fu)} categories")
    fig.tight_layout()
    fig.savefig(FIG / "skill_followup_category_bar.png", dpi=150)
    plt.close(fig)

    # S5: pairwise similarity
    main_descs = [s["description"] for s in main_skills]
    fu_descs = [s["description"] for s in fu_skills]
    compute_pairwise_sim(main_descs, "skill_main", SIM / "skill_main")
    compute_pairwise_sim(fu_descs, "skill_followup", SIM / "skill_followup")

    return {
        "main_skills": len(main_skills),
        "main_categories": len(cats_main),
        "follow_up_skills": len(fu_skills),
        "follow_up_categories": len(cats_fu),
    }


def analyze_validation(val_dir: str):
    """Validation benchmark distribution analysis."""
    print("\n=== Validation benchmark ===")

    all_path = Path(val_dir) / "all.jsonl"
    items = [json.loads(l) for l in open(all_path)]
    n = len(items)

    tier_counts = Counter(r["tier"] for r in items)
    ds_counts = Counter(r["metadata"]["dataset"] for r in items)
    tier_ds = Counter((r["tier"], r["metadata"]["dataset"]) for r in items)

    # Tier × dataset stacked bar
    tiers = ["easy", "medium", "hard"]
    datasets = sorted(set(d for _, d in tier_ds.keys()))
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(tiers))
    for ds in datasets:
        vals = [tier_ds.get((t, ds), 0) for t in tiers]
        ax.bar(tiers, vals, bottom=bottom, label=ds, edgecolor="black")
        bottom += vals
    ax.set_ylabel("Item Count")
    ax.set_title(f"Validation Benchmark: {n} items (post-V3-filter)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "val_tier_stacked_bar.png", dpi=150)
    plt.close(fig)

    print(f"  Total: {n}")
    for tier in tiers:
        print(f"  {tier}: {tier_counts[tier]}")

    return {
        "total": n,
        "by_tier": dict(tier_counts),
        "by_dataset": dict(ds_counts.most_common()),
    }


def write_summary(bare_stats, skill_stats, val_stats):
    """Write distribution_summary.md."""
    leakage = {}
    lp = OUT / "leakage_report.json"
    if lp.exists():
        leakage = json.load(open(lp))

    md = f"""# Category D (sycophancy_resistance) — Distribution Summary

Generated by `analyze_sycophancy_resistance.py`.

## Bare Pool

- **Size**: {bare_stats['n']} items
- **Sources**: {', '.join(f"{k} ({v})" for k, v in bare_stats['sources'].items())}
- **Domains**: {bare_stats['n_domains']}
- **Prompt length**: mean {bare_stats['prompt_length_mean']:.1f} words, p95 {bare_stats['prompt_length_p95']:.0f}

## Skill Dictionary

- **Main skills**: {skill_stats['main_skills']} across {skill_stats['main_categories']} categories
- **Follow-up skills**: {skill_stats['follow_up_skills']} across {skill_stats['follow_up_categories']} categories
- **Theoretical sources**: 7 independent (Cialdini 1984, Petty & Cacioppo 1986, Walton 2008, Lewandowsky 2012, Tversky & Kahneman 1974, Hasher 1977, Sharma 2023)
- **Combinatorial space**: C(47,3) = 16,215

## Validation Benchmark

- **Total (post-V3-filter)**: {val_stats['total']}
- **By tier**: {', '.join(f"{k} ({v})" for k, v in val_stats['by_tier'].items())}
- **By dataset**: {', '.join(f"{k} ({v})" for k, v in val_stats['by_dataset'].items())}

## V3 Leakage Check

- **Bare size**: {leakage.get('bare_size', '?')}
- **Total dropped**: {leakage.get('total_dropped', '?')}
- **Total kept**: {leakage.get('total_kept', '?')}
"""
    for tier, data in leakage.get("tiers", {}).items():
        md += f"- **{tier}**: {data['raw_size']} raw → {data['kept']} kept "
        md += f"({data['exact_hits']} exact + {data['semantic_only_hits']} semantic-only = "
        md += f"{data['total_hits']} dropped)\n"

    summary_path = OUT / "distribution_summary.md"
    with open(summary_path, "w") as f:
        f.write(md)
    print(f"\nWrote {summary_path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    SIM.mkdir(parents=True, exist_ok=True)

    bare_stats = analyze_bare("data/sycophancy_resistance/bare_requests.jsonl")
    skill_stats = analyze_skills()
    val_stats = analyze_validation("data/sycophancy_resistance/validation")
    write_summary(bare_stats, skill_stats, val_stats)

    print("\nDone. Outputs in results/data_prep/sycophancy_resistance/")


if __name__ == "__main__":
    main()
