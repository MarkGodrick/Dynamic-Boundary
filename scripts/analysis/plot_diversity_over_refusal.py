"""Generate diversity figures for over_refusal bare pool (P3).

Outputs:
  results/data_prep/over_refusal/figures/bare_source_pie.png
  results/data_prep/over_refusal/figures/bare_category_bar.png
  results/data_prep/over_refusal/figures/bare_prompt_length_hist.png

Usage:
  conda run --no-capture-output -n boundary python -u \
      scripts/analysis/plot_diversity_over_refusal.py
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    bare_path = "data/over_refusal/bare_requests.jsonl"
    fig_dir = Path("results/data_prep/over_refusal/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    items = [json.loads(line) for line in open(bare_path)]
    print(f"Loaded {len(items)} bare prompts")

    # Source pie
    source_counts = Counter(item["source"] for item in items)
    labels = list(source_counts.keys())
    sizes = list(source_counts.values())
    label_map = {
        "xstest": f"XSTest ({source_counts['xstest']})",
        "orbench_hard": f"OR-Bench-Hard ({source_counts['orbench_hard']})",
        "author_handwritten": f"Handcrafted ({source_counts['author_handwritten']})",
    }
    labels = [label_map.get(l, l) for l in labels]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors[:len(sizes)],
           startangle=90)
    ax.set_title(f"Over-Refusal Bare Pool Source Distribution (n={len(items)})")
    fig.tight_layout()
    fig.savefig(fig_dir / "bare_source_pie.png", dpi=150)
    plt.close()

    # Category bar — group: XSTest types vs OR-Bench categories vs handcrafted
    xstest_types = [
        "homonyms", "figurative_language", "safe_targets", "safe_contexts",
        "definitions", "nons_group_real_discr", "real_group_nons_discr",
        "historical_events", "privacy_public", "privacy_fictional"
    ]
    orbench_cats = [
        "illegal", "privacy", "unethical", "harmful", "deception",
        "sexual", "violence", "self-harm", "hate", "harassment"
    ]

    cat_counts = Counter(item["category"] for item in items)

    # Stacked horizontal bar: XSTest types, OR-Bench categories, handcrafted
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [2, 2, 3]})

    # XSTest types
    xs_cats = [(c, cat_counts.get(c, 0)) for c in xstest_types]
    xs_cats.sort(key=lambda x: -x[1])
    ax = axes[0]
    ax.barh([c for c, _ in xs_cats], [n for _, n in xs_cats], color="#4C72B0")
    ax.set_title("XSTest Safe: 10 Trigger Types")
    ax.set_xlabel("Count")

    # OR-Bench categories
    ob_cats = [(c, cat_counts.get(c, 0)) for c in orbench_cats]
    ob_cats.sort(key=lambda x: -x[1])
    ax = axes[1]
    ax.barh([c for c, _ in ob_cats], [n for _, n in ob_cats], color="#DD8452")
    ax.set_title("OR-Bench-Hard: 10 Rejection Categories")
    ax.set_xlabel("Count")

    # Handcrafted
    hw_cats = [(c, n) for c, n in cat_counts.items()
               if c not in xstest_types and c not in orbench_cats]
    hw_cats.sort(key=lambda x: -x[1])
    ax = axes[2]
    ax.barh([c for c, _ in hw_cats], [n for _, n in hw_cats], color="#55A868")
    ax.set_title("Handcrafted: Professional/Technical Domains")
    ax.set_xlabel("Count")

    fig.suptitle(f"Over-Refusal Bare Pool Category Distribution (n={len(items)})",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "bare_category_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Prompt length histogram
    lengths = [len(item["prompt"].split()) for item in items]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=50, edgecolor="none", alpha=0.8, color="#4C72B0")
    ax.axvline(np.mean(lengths), color="red", linestyle="--", alpha=0.7,
               label=f"Mean: {np.mean(lengths):.1f}")
    ax.axvline(np.percentile(lengths, 95), color="orange", linestyle="--", alpha=0.7,
               label=f"P95: {np.percentile(lengths, 95):.0f}")
    ax.set_xlabel("Prompt Length (words)")
    ax.set_ylabel("Count")
    ax.set_title(f"Over-Refusal Bare Pool Prompt Length (n={len(items)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "bare_prompt_length_hist.png", dpi=150)
    plt.close()

    print(f"Saved 3 figures to {fig_dir}")
    print(f"  Prompt length: mean={np.mean(lengths):.1f}, "
          f"median={np.median(lengths):.1f}, p95={np.percentile(lengths, 95):.0f}")


if __name__ == "__main__":
    main()
