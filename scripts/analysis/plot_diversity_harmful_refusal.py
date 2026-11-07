"""Diversity figures for harmful_refusal (P3 / S4).

Produces three figures under results/data_prep/harmful_refusal/figures/:

  1. bare_source_pie.png        — bare pool composition by source (P3)
  2. bare_category_bar.png      — harm category distribution across the bare pool (P3)
                                  AdvBench items have no native category; they are
                                  bucketed as "uncategorized" — disclosed in caption
  3. skill_attack_family_bar.png — DejaVu attack_type family coverage (S4)
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def plot_bare_source_pie(bare: list[dict], out_path: Path) -> None:
    counts = Counter(b["source"] for b in bare)
    labels = list(counts.keys())
    sizes = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.Set2(range(len(labels)))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=[f"{k}\n(n={v})" for k, v in zip(labels, sizes)],
        autopct="%1.1f%%", colors=colors, startangle=90,
    )
    ax.set_title(f"Bare Request Pool — Source Composition (post-dedup, n={len(bare)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_bare_canonical_bar(bare: list[dict], out_path: Path) -> None:
    """Top-level canonical category bar (paper main figure)."""
    cat_counts = Counter(b.get("canonical_category") or "MISSING" for b in bare)
    items = sorted(cat_counts.items(), key=lambda x: -x[1])
    labels, sizes = zip(*items)

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.45)))
    bars = ax.barh(range(len(labels)), sizes, color="steelblue")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Item count")
    ax.set_title(
        f"Bare Pool — Canonical Harm Category Distribution (n={len(bare)})\n"
        "Top-level taxonomy from MLCommons AILuminate v1.0 (pruned, 9 categories)"
    )
    for i, (_, size) in enumerate(items):
        ax.text(size + max(sizes) * 0.005, i, str(size), va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_bare_subcategory_smallmultiples(bare: list[dict], out_path: Path) -> None:
    """One small-multiple subplot per canonical category, each showing that
    canonical's subcategory horizontal bar chart. Avoids the text-overlap problem
    of a single stacked bar."""
    by_cat: dict[str, Counter] = {}
    for b in bare:
        cc = b.get("canonical_category") or "MISSING"
        sub = b.get("subcategory") or "ambiguous"
        by_cat.setdefault(cc, Counter())[sub] += 1

    canon_order = sorted(by_cat.keys(), key=lambda c: -sum(by_cat[c].values()))
    n_cat = len(canon_order)
    ncols = 3
    nrows = (n_cat + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.0, nrows * 2.6))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax_idx, cc in enumerate(canon_order):
        ax = axes[ax_idx]
        sub_items = sorted(by_cat[cc].items(), key=lambda x: -x[1])
        labels = [s for s, _ in sub_items]
        sizes = [n for _, n in sub_items]
        total = sum(sizes)

        ax.barh(range(len(labels)), sizes, color="steelblue", edgecolor="white")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{cc} (n={total})", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlim(0, max(sizes) * 1.18)
        for i, n in enumerate(sizes):
            ax.text(n + max(sizes) * 0.02, i, str(n), va="center", fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for k in range(n_cat, len(axes)):
        axes[k].axis("off")

    fig.suptitle(
        f"Bare Pool — Subcategory Breakdown within each Canonical Category (n={len(bare)})",
        fontsize=12, y=1.005,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def plot_skill_attack_family(skills: list[dict], out_path: Path) -> None:
    fam_counts = Counter()
    for s in skills:
        for at in s.get("source_meta", {}).get("attack_types", []):
            fam_counts[at] += 1
    items = sorted(fam_counts.items(), key=lambda x: -x[1])
    labels, sizes = zip(*items)

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.3)))
    bars = ax.barh(range(len(labels)), sizes, color="darkorange")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Skill count (a skill may belong to multiple families)")
    ax.set_title(
        f"DejaVu Skill Dictionary — Attack Family Coverage (n_skills={len(skills)}, n_families={len(labels)})"
    )
    for i, (label, size) in enumerate(items):
        ax.text(size + max(sizes) * 0.005, i, str(size), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bare", type=str, default="data/harmful_refusal/bare_requests.jsonl")
    parser.add_argument("--skills", type=str,
                        default="skills/harmful_refusal/skill_dictionary.json")
    parser.add_argument("--out-dir", type=str,
                        default="results/data_prep/harmful_refusal/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bare = load_jsonl(Path(args.bare))
    print(f"Loaded {len(bare)} bare requests")
    with open(args.skills) as f:
        skills = json.load(f)
    print(f"Loaded {len(skills)} skills")

    print()
    plot_bare_source_pie(bare, out_dir / "bare_source_pie.png")
    plot_bare_canonical_bar(bare, out_dir / "bare_canonical_category_bar.png")
    plot_bare_subcategory_smallmultiples(bare, out_dir / "bare_subcategory_smallmultiples.png")
    plot_skill_attack_family(skills, out_dir / "skill_attack_family_bar.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
