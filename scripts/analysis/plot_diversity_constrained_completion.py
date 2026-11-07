"""Diversity figures for constrained_completion (P3 / S4).

Produces figures under results/data_prep/constrained_completion/figures/:

  1. bare_source_pie.png                — bare pool by source (P3)
  2. bare_infobench_topical_bar.png     — InFoBench topical categories, top 20 (P3)
  3. bare_prompt_length_hist.png        — bare prompt length histogram (P3)
  4. skill_verifier_category_bar.png    — skills by verifier_category (S4)
  5. skill_by_source_bar.png            — skills by source IFEval vs IFBench (S4)
  6. val_num_instructions_stacked.png   — validation num_instructions by source
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "constrained_completion"
SKILL = ROOT / "skills" / "constrained_completion" / "skill_dictionary.json"
OUT_DIR = ROOT / "results" / "data_prep" / "constrained_completion" / "figures"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def plot_bare_source_pie(bare: list[dict]) -> None:
    counts = Counter(b["source"] for b in bare)
    labels = list(counts.keys())
    sizes = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.Set2(range(len(labels)))
    ax.pie(
        sizes,
        labels=[f"{k}\n(n={v})" for k, v in zip(labels, sizes)],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    ax.set_title(
        f"Bare Request Pool — Source Composition (post-dedup, n={len(bare)})"
    )
    fig.tight_layout()
    out = OUT_DIR / "bare_source_pie.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def plot_bare_super_category(bare: list[dict]) -> None:
    """Stacked super-category bar across both sources (P3 main figure)."""
    by_cat_source: dict[str, Counter] = {}
    for b in bare:
        c = b.get("super_category", "Unknown")
        s = b["source"]
        by_cat_source.setdefault(c, Counter())[s] += 1

    cats_sorted = sorted(
        by_cat_source.keys(),
        key=lambda c: -sum(by_cat_source[c].values()),
    )
    ife_counts = [by_cat_source[c].get("ifeval", 0) for c in cats_sorted]
    ifb_counts = [by_cat_source[c].get("infobench", 0) for c in cats_sorted]

    fig, ax = plt.subplots(figsize=(10, max(4, len(cats_sorted) * 0.55)))
    y = range(len(cats_sorted))
    ax.barh(y, ife_counts, color="steelblue", label="IFEval (n=325)")
    ax.barh(y, ifb_counts, left=ife_counts, color="darkorange", label="InFoBench (n=498)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(cats_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("Item count")
    ax.set_title(
        f"Bare Pool — Super-Category Distribution (n={len(bare)})\n"
        "InFoBench categories mapped via colon-prefix; IFEval via DeepSeek classification"
    )
    for i, (a, b) in enumerate(zip(ife_counts, ifb_counts)):
        total = a + b
        if total > 0:
            ax.text(total + 2, i, str(total), va="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "bare_super_category_stacked.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def plot_infobench_topical(bare: list[dict]) -> None:
    """InFoBench fine-grained top-20 categories (appendix figure)."""
    infob = [b for b in bare if b["source"] == "infobench"]
    cats = Counter(
        (b.get("source_meta") or {}).get("category", "unknown") for b in infob
    )
    top = cats.most_common(20)
    other = sum(v for k, v in cats.items() if k not in dict(top))
    labels = [k for k, _ in top] + ([f"(other, n={other}, {len(cats)-20} cats)"] if other else [])
    sizes = [v for _, v in top] + ([other] if other else [])

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.42)))
    ax.barh(range(len(labels)), sizes, color="steelblue")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Item count")
    ax.set_title(
        f"InFoBench fine-grained topical categories — top 20 (appendix)\n"
        f"n={len(infob)} items across {len(cats)} distinct categories"
    )
    for i, s in enumerate(sizes):
        ax.text(s + max(sizes) * 0.005, i, str(s), va="center", fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "bare_infobench_topical_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def plot_prompt_length(bare: list[dict]) -> None:
    lens = [len(b["prompt"].split()) for b in bare]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lens, bins=40, color="steelblue", edgecolor="white")
    mean = sum(lens) / len(lens)
    ax.axvline(mean, color="red", linestyle="--", label=f"mean={mean:.1f}")
    ax.set_xlabel("Prompt length (words)")
    ax.set_ylabel("Count")
    ax.set_title(f"Bare Pool — Prompt Length (n={len(lens)})")
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "bare_prompt_length_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def plot_skill_verifier_category(skills: list[dict]) -> None:
    by_cat_source = {}
    for s in skills:
        cat = s["verifier_category"]
        src = s["source"]
        by_cat_source.setdefault(cat, Counter())[src] += 1
    cats_sorted = sorted(by_cat_source.keys(), key=lambda c: -sum(by_cat_source[c].values()))
    ife_counts = [by_cat_source[c].get("ifeval", 0) for c in cats_sorted]
    ifb_counts = [by_cat_source[c].get("ifbench", 0) for c in cats_sorted]

    fig, ax = plt.subplots(figsize=(10, max(4, len(cats_sorted) * 0.4)))
    y = range(len(cats_sorted))
    ax.barh(y, ife_counts, color="steelblue", label="IFEval")
    ax.barh(y, ifb_counts, left=ife_counts, color="darkorange", label="IFBench")
    ax.set_yticks(list(y))
    ax.set_yticklabels(cats_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("Skill count")
    ax.set_title(
        f"Skill Dictionary — Verifier Category Coverage (n={len(skills)})\n"
        "Grouped by verifier namespace (IFEval 25 + IFBench 58)"
    )
    for i, (a, b) in enumerate(zip(ife_counts, ifb_counts)):
        total = a + b
        if total > 0:
            ax.text(total + 0.2, i, str(total), va="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "skill_verifier_category_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def plot_skill_by_source(skills: list[dict]) -> None:
    counts = Counter(s["source"] for s in skills)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(counts.keys()), list(counts.values()), color=["steelblue", "darkorange"])
    for i, (k, v) in enumerate(counts.items()):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=11)
    ax.set_ylabel("Skill count")
    ax.set_title(f"Skill Dictionary — By Source (total {len(skills)})")
    fig.tight_layout()
    out = OUT_DIR / "skill_by_source_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def plot_val_num_instructions(val: list[dict]) -> None:
    by_n_source = {}
    for v in val:
        n = v["num_instructions"]
        src = v["source"]
        by_n_source.setdefault(n, Counter())[src] += 1
    ns = sorted(by_n_source.keys())
    ifb_counts = [by_n_source[n].get("ifbench", 0) for n in ns]
    mie_counts = [by_n_source[n].get("manyifeval", 0) for n in ns]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(ns))
    ax.bar(x, ifb_counts, color="steelblue", label="IFBench")
    ax.bar(x, mie_counts, bottom=ifb_counts, color="darkorange", label="ManyIFEval")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("num_instructions (constraint count per item)")
    ax.set_ylabel("Validation item count")
    ax.set_title(
        f"Validation Pool — num_instructions Distribution (n={len(val)})\n"
        "IFBench contributes the OOD type axis (num=1,2); "
        "ManyIFEval contributes the depth axis (num=1..10)"
    )
    ax.legend()
    for i, (a, b) in enumerate(zip(ifb_counts, mie_counts)):
        total = a + b
        ax.text(i, total + 20, str(total), ha="center", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "val_num_instructions_stacked.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bare = load_jsonl(DATA / "bare_requests.jsonl")
    skills = json.loads(SKILL.read_text())
    val = load_jsonl(DATA / "validation" / "all.jsonl")

    print(f"Loaded bare={len(bare)}, skills={len(skills)}, val={len(val)}")
    plot_bare_source_pie(bare)
    plot_bare_super_category(bare)
    plot_infobench_topical(bare)
    plot_prompt_length(bare)
    plot_skill_verifier_category(skills)
    plot_skill_by_source(skills)
    plot_val_num_instructions(val)


if __name__ == "__main__":
    main()
