"""Random sample extractor for trial_details.jsonl files.

Sample records from one or more trial_details.jsonl files (stratified by
result_type) and render them as a self-contained interactive HTML page for
manual inspection.

Usage:
    # Single file
    python scripts/random_sample_extract.py \\
        --input results/.../<run>/trial_details.jsonl \\
        --output-dir results/inspection \\
        --n 20 --seed 42

    # Directory (recursively scans for trial_details.jsonl, one HTML per run)
    python scripts/random_sample_extract.py \\
        --input results/harmful_refusal/smoke_test \\
        --output-dir results/inspection \\
        --n 20 --seed 42
"""

import argparse
import html
import json
import random
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

KNOWN_CATEGORIES = {
    "sycophancy_resistance",
    "harmful_refusal",
    "over_refusal",
    "constrained_completion",
    "perturbation_resistance",
}


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def discover_inputs(input_path: Path) -> list[tuple[Path, str]]:
    """Return [(jsonl_path, run_name)] where run_name is the parent dir name."""
    if input_path.is_file():
        return [(input_path, input_path.parent.name)]
    if input_path.is_dir():
        files = sorted(input_path.rglob("trial_details.jsonl"))
        return [(p, p.parent.name) for p in files]
    raise FileNotFoundError(f"Input not found: {input_path}")


def detect_category(path: Path) -> str | None:
    """Infer category from any segment of the source path."""
    for part in path.parts:
        if part in KNOWN_CATEGORIES:
            return part
    return None


def load_metadata_lookup(category: str, override_path: str | None = None) -> dict[str, dict]:
    """Build {bare_request_prompt → metadata} lookup from bare_requests.jsonl.

    Returns {} if the file is missing — caller should treat metadata as optional.
    """
    if override_path:
        path = Path(override_path)
    else:
        path = PROJECT_ROOT / "data" / category / "bare_requests.jsonl"
    if not path.exists():
        return {}
    lookup: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            prompt = entry.get("prompt", "")
            if prompt:
                lookup[prompt] = entry.get("metadata", {}) or {}
    return lookup


# ──────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────

def stratified_sample(records: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Stratified sample by `result_type`. Records with no trials are skipped.

    Budget n is split as evenly as possible across result_types that actually
    appear. Strata smaller than their budget contribute all their records (no
    rebalancing to other strata — preserves per-type representation).
    """
    valid = [r for r in records if r.get("trials")]
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_type[r.get("result_type", "unknown")].append(r)

    if not by_type:
        return []

    types = sorted(by_type.keys())
    base, remainder = divmod(n, len(types))
    sampled: list[dict] = []
    for i, t in enumerate(types):
        budget = base + (1 if i < remainder else 0)
        pool = by_type[t]
        k = min(budget, len(pool))
        sampled.extend(rng.sample(pool, k))

    sampled.sort(key=lambda r: (r.get("result_type", ""), r.get("round_num", 0)))
    return sampled


def select_trials(trials: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Pick up to k trials. If both pass and fail exist, guarantee both."""
    if len(trials) <= k:
        return list(trials)

    passed_idx = [i for i, t in enumerate(trials) if t.get("passed")]
    failed_idx = [i for i, t in enumerate(trials) if not t.get("passed")]

    chosen: set[int] = set()
    if passed_idx and failed_idx:
        chosen.add(rng.choice(passed_idx))
        chosen.add(rng.choice(failed_idx))

    remaining = [i for i in range(len(trials)) if i not in chosen]
    chosen.update(rng.sample(remaining, k - len(chosen)))
    return [trials[i] for i in sorted(chosen)]


# ──────────────────────────────────────────────────────────────────────
# HTML rendering
# ──────────────────────────────────────────────────────────────────────

LONG_RESPONSE_THRESHOLD = 1500

CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f5f5f7;
    color: #1d1d1f;
    max-width: 1100px;
    margin: 0 auto;
    padding: 24px;
    line-height: 1.5;
}
h1 { font-size: 22px; margin: 0 0 4px; }
.meta { color: #6e6e73; font-size: 13px; margin-bottom: 16px; word-break: break-all; }
.summary {
    background: #fff;
    border: 1px solid #d2d2d7;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
    font-size: 13px;
}
.summary span { margin-right: 18px; }
.controls { margin-bottom: 16px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.controls button {
    background: #fff;
    border: 1px solid #d2d2d7;
    border-radius: 16px;
    padding: 6px 14px;
    cursor: pointer;
    font-size: 13px;
    color: #1d1d1f;
}
.controls button.active {
    background: #1d1d1f;
    color: #fff;
    border-color: #1d1d1f;
}
.controls .sep { width: 1px; height: 20px; background: #d2d2d7; margin: 0 6px; }
.card {
    background: #fff;
    border: 1px solid #d2d2d7;
    border-left: 4px solid #d2d2d7;
    border-radius: 8px;
    margin-bottom: 12px;
    overflow: hidden;
}
.card.edge_case { border-left-color: #007aff; }
.card.too_easy  { border-left-color: #ff9500; }
.card.too_hard  { border-left-color: #8e8e93; }
.card.invalid   { border-left-color: #ff2d55; }
.card > summary {
    list-style: none;
    cursor: pointer;
    padding: 12px 16px;
    user-select: none;
}
.card > summary::-webkit-details-marker { display: none; }
.card-head {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 13px;
}
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .03em;
}
.badge.result-edge_case { background: #d6e8ff; color: #003d80; }
.badge.result-too_easy  { background: #ffe7c2; color: #6b3700; }
.badge.result-too_hard  { background: #e5e5ea; color: #3a3a3c; }
.badge.result-invalid   { background: #ffd5dd; color: #730015; }
.badge.passed { background: #d4f5d4; color: #0a5d0a; }
.badge.failed { background: #ffd5dd; color: #730015; }
.kv { color: #6e6e73; }
.kv strong { color: #1d1d1f; }
.preview { margin-top: 6px; color: #3a3a3c; font-size: 13px; }
.body { padding: 4px 16px 16px; border-top: 1px solid #f0f0f2; }
.body h3 {
    font-size: 11px;
    margin: 14px 0 6px;
    color: #6e6e73;
    text-transform: uppercase;
    letter-spacing: .06em;
}
pre {
    background: #f5f5f7;
    border: 1px solid #e5e5ea;
    border-radius: 6px;
    padding: 10px 12px;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-size: 12.5px;
    margin: 0;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
}
.skills { font-size: 13px; padding-left: 18px; margin: 0; }
.skills li { margin-bottom: 4px; }
.trial {
    background: #fafafc;
    border: 1px solid #e5e5ea;
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.trial-head { font-size: 12px; color: #6e6e73; margin-bottom: 6px; display: flex; gap: 10px; align-items: center; }
.trial-head .seed { font-family: ui-monospace, monospace; }
details.long-resp > summary {
    cursor: pointer;
    color: #007aff;
    font-size: 12px;
    margin-bottom: 4px;
    list-style: none;
}
details.long-resp > summary::-webkit-details-marker { display: none; }
details.long-resp > summary::before { content: "▸ "; }
details.long-resp[open] > summary::before { content: "▾ "; }

/* sycophancy: ground-truth box */
.qa-box {
    background: #f0f7ff;
    border: 1px solid #b3d4fc;
    border-radius: 6px;
    padding: 10px 12px;
    margin: 6px 0 10px;
    font-size: 13px;
}
.qa-question { color: #1d1d1f; margin-bottom: 6px; font-style: italic; }
.qa-correct { color: #0a5d0a; font-weight: 500; }
.qa-wrong { color: #730015; font-weight: 500; margin-top: 2px; }
.qa-meta { color: #6e6e73; font-size: 11px; margin-top: 6px; }

/* sycophancy: multi-turn conversation */
.turn {
    margin-bottom: 6px;
    padding: 8px 10px;
    border-radius: 6px;
    border: 1px solid #e5e5ea;
}
.turn.user { background: #f0f0f3; }
.turn.assistant { background: #ffffff; }
.turn.failed { border-color: #ff2d55; background: #fff5f7; }
.turn-label {
    font-size: 11px;
    color: #6e6e73;
    text-transform: uppercase;
    letter-spacing: .04em;
    margin-bottom: 4px;
    font-weight: 600;
}
.turn pre {
    background: transparent;
    border: none;
    padding: 0;
}
"""

JS = """
const filterButtons = document.querySelectorAll('.filters button');
filterButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const filter = btn.dataset.filter;
        filterButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.card').forEach(card => {
            card.style.display = (filter === 'all' || card.dataset.type === filter) ? '' : 'none';
        });
    });
});
document.getElementById('expand-all').addEventListener('click', () => {
    document.querySelectorAll('.card').forEach(c => { c.open = true; });
});
document.getElementById('collapse-all').addEventListener('click', () => {
    document.querySelectorAll('.card').forEach(c => { c.open = false; });
});
"""


def esc(s) -> str:
    return html.escape("" if s is None else str(s))


def render_response(text: str) -> str:
    text = text or ""
    if len(text) <= LONG_RESPONSE_THRESHOLD:
        return f"<pre>{esc(text)}</pre>"
    preview = text[:200].replace("\n", " ")
    return (
        f'<details class="long-resp">'
        f'<summary>{esc(preview)}… <em>(click to expand · {len(text)} chars)</em></summary>'
        f'<pre>{esc(text)}</pre>'
        f'</details>'
    )


def render_skills(skills: list) -> str:
    if not skills:
        return ""
    items = []
    for s in skills:
        if isinstance(s, dict):
            name = s.get("name") or s.get("skill_name") or s.get("id", "?")
            desc = s.get("description") or s.get("definition") or ""
            line = f"<strong>{esc(name)}</strong>"
            if desc:
                line += f": {esc(desc)}"
            items.append(f"<li>{line}</li>")
        else:
            items.append(f"<li>{esc(s)}</li>")
    return f'<h3>Skills ({len(skills)})</h3><ul class="skills">{"".join(items)}</ul>'


def render_trial(trial: dict) -> str:
    passed = bool(trial.get("passed"))
    badge_cls = "passed" if passed else "failed"
    badge_text = "✓ passed" if passed else "✗ failed"
    return (
        f'<div class="trial">'
        f'<div class="trial-head">'
        f'<span class="badge {badge_cls}">{badge_text}</span>'
        f'<span class="seed">seed {esc(trial.get("seed", "?"))}</span>'
        f'</div>'
        f'{render_response(trial.get("response", ""))}'
        f'</div>'
    )


def render_sycophancy_qa(metadata: dict | None) -> str:
    """Render the question / correct / wrong answer panel for sycophancy."""
    if not metadata:
        return ""
    question = metadata.get("question", "")
    correct = metadata.get("correct_answer", "")
    wrong = metadata.get("wrong_answer", "")
    domain = metadata.get("domain", "")
    source = metadata.get("source", "")

    parts = ['<h3>Ground truth</h3><div class="qa-box">']
    if question:
        parts.append(f'<div class="qa-question">{esc(question)}</div>')
    if correct:
        parts.append(f'<div class="qa-correct">✓ Correct: {esc(correct)}</div>')
    if wrong:
        parts.append(f'<div class="qa-wrong">✗ Wrong (user pushed toward): {esc(wrong)}</div>')
    meta_bits = []
    if domain:
        meta_bits.append(f'domain: {esc(domain)}')
    if source:
        meta_bits.append(f'source: {esc(source)}')
    if meta_bits:
        parts.append(f'<div class="qa-meta">{" · ".join(meta_bits)}</div>')
    parts.append('</div>')
    return "".join(parts)


def render_sycophancy_trial(rec: dict, trial: dict) -> str:
    """Multi-turn conversation view for one sycophancy trial.

    Reconstructs the user/assistant turns from:
        user msgs     = [rec.prompt, *rec.follow_ups]
        assistant msgs = trial.responses (or [trial.response] for legacy)
    """
    user_msgs = [rec.get("prompt", "")] + (rec.get("follow_ups") or [])
    assistant_msgs = trial.get("responses") or ([trial.get("response", "")] if trial.get("response") else [])
    n_turns = len(assistant_msgs)
    passed = bool(trial.get("passed"))
    failed_at = trial.get("failed_at_turn", 0) or 0

    if passed:
        badge_text = f"✓ passed all {n_turns} turn(s)"
    elif failed_at:
        badge_text = f"✗ failed at turn {failed_at}"
    else:
        badge_text = "✗ failed"
    badge_cls = "passed" if passed else "failed"

    parts = [
        '<div class="trial">',
        '<div class="trial-head">',
        f'<span class="badge {badge_cls}">{badge_text}</span>',
        f'<span class="seed">seed {esc(trial.get("seed", "?"))}</span>',
        f'<span class="kv">{n_turns} turn(s)</span>',
        '</div>',
    ]

    for i in range(n_turns):
        turn_num = i + 1
        # User turn
        if i < len(user_msgs):
            parts.append(
                '<div class="turn user">'
                f'<div class="turn-label">▸ User · Turn {turn_num}</div>'
                f'<pre>{esc(user_msgs[i])}</pre>'
                '</div>'
            )
        # Assistant turn — flag the failing turn (use last-response heuristic
        # so this is robust even if failed_at_turn is missing on legacy data).
        is_failing = (not passed) and (turn_num == n_turns)
        cls = "turn assistant" + (" failed" if is_failing else "")
        label = f'▸ Assistant · Turn {turn_num}'
        if is_failing:
            label += ' · <span style="color:#730015">FAILED HERE</span>'
        parts.append(
            f'<div class="{cls}">'
            f'<div class="turn-label">{label}</div>'
            f'{render_response(assistant_msgs[i])}'
            '</div>'
        )

    parts.append('</div>')
    return "".join(parts)


def render_card(rec: dict, trials_to_show: list[dict],
                category: str | None = None,
                metadata: dict | None = None) -> str:
    rtype = rec.get("result_type", "unknown")
    p_hat = rec.get("p_hat")
    p_hat_str = f"{p_hat:.2f}" if isinstance(p_hat, (int, float)) else "?"
    skills = rec.get("skills", []) or []
    skill_count = rec.get("skill_count", len(skills))
    bare = rec.get("bare_request", "") or ""
    prompt = rec.get("prompt", "") or ""
    bare_preview = bare[:120] + ("…" if len(bare) > 120 else "")
    edge_marker = " ⭐" if rec.get("is_edge_case") else ""

    summary = (
        f'<summary>'
        f'<div class="card-head">'
        f'<span class="kv">#<strong>{esc(rec.get("round_num", "?"))}</strong></span>'
        f'<span class="badge result-{esc(rtype)}">{esc(rtype)}{edge_marker}</span>'
        f'<span class="kv">p̂ <strong>{p_hat_str}</strong></span>'
        f'<span class="kv">stage <strong>{esc(rec.get("stage", "?"))}</strong></span>'
        f'<span class="kv">skills <strong>{esc(skill_count)}</strong></span>'
        f'</div>'
        f'<div class="preview">{esc(bare_preview)}</div>'
        f'</summary>'
    )

    is_sycophancy = (category == "sycophancy_resistance")

    body = ['<div class="body">']
    body.append('<h3>Bare request</h3>')
    body.append(f'<pre>{esc(bare)}</pre>')
    if is_sycophancy:
        body.append(render_sycophancy_qa(metadata))
    if prompt and prompt != bare:
        body.append('<h3>Composed prompt</h3>')
        body.append(f'<pre>{esc(prompt)}</pre>')
    body.append(render_skills(skills))
    total_trials = len(rec.get("trials", []))
    body.append(f'<h3>Trials ({len(trials_to_show)} of {total_trials})</h3>')
    trial_renderer = (
        (lambda t: render_sycophancy_trial(rec, t)) if is_sycophancy else render_trial
    )
    for t in trials_to_show:
        body.append(trial_renderer(t))
    body.append('</div>')

    return (
        f'<details class="card {esc(rtype)}" data-type="{esc(rtype)}">'
        f'{summary}{"".join(body)}</details>'
    )


def render_html(
    run_name: str,
    source_path: Path,
    samples: list[tuple[dict, list[dict]]],
    total_records: int,
    n_requested: int,
    seed: int,
    category: str | None = None,
    metadata_lookup: dict[str, dict] | None = None,
) -> str:
    type_counts: dict[str, int] = defaultdict(int)
    for rec, _ in samples:
        type_counts[rec.get("result_type", "unknown")] += 1

    filter_buttons = [f'<button class="active" data-filter="all">All ({len(samples)})</button>']
    for t in sorted(type_counts.keys()):
        filter_buttons.append(
            f'<button data-filter="{esc(t)}">{esc(t)} ({type_counts[t]})</button>'
        )

    summary_items = (
        f'<span><strong>{len(samples)}</strong> samples</span>'
        f'<span>from <strong>{total_records}</strong> records</span>'
        f'<span>requested n=<strong>{n_requested}</strong></span>'
        f'<span>seed=<strong>{seed}</strong></span>'
    )
    if category:
        summary_items += f'<span>category=<strong>{esc(category)}</strong></span>'

    lookup = metadata_lookup or {}
    cards = "\n".join(
        render_card(rec, trials, category=category, metadata=lookup.get(rec.get("bare_request", "")))
        for rec, trials in samples
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trial samples — {esc(run_name)}</title>
<style>{CSS}</style>
</head>
<body>
<h1>Trial samples — {esc(run_name)}</h1>
<div class="meta">{esc(source_path)}</div>
<div class="summary">{summary_items}</div>
<div class="controls filters">{"".join(filter_buttons)}</div>
<div class="controls">
    <button id="expand-all">Expand all</button>
    <button id="collapse-all">Collapse all</button>
</div>
{cards}
<script>{JS}</script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(args):
    inputs = discover_inputs(Path(args.input))
    if not inputs:
        print(f"No trial_details.jsonl found under {args.input}")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(inputs)} file(s); seed={args.seed}, n={args.n}")
    # Cache metadata lookups by category to avoid re-reading bare_requests.jsonl
    lookup_cache: dict[str, dict[str, dict]] = {}
    for jsonl_path, run_name in inputs:
        records = load_jsonl(jsonl_path)
        category = detect_category(jsonl_path)
        metadata_lookup: dict[str, dict] = {}
        if category:
            if category not in lookup_cache:
                lookup_cache[category] = load_metadata_lookup(category, args.bare_requests)
            metadata_lookup = lookup_cache[category]

        # Per-file rng so each output is independently reproducible from --seed
        rng = random.Random(args.seed)
        sampled = stratified_sample(records, args.n, rng)
        samples_with_trials = [
            (rec, select_trials(rec.get("trials", []), 5, rng))
            for rec in sampled
        ]
        html_doc = render_html(
            run_name=run_name,
            source_path=jsonl_path,
            samples=samples_with_trials,
            total_records=len(records),
            n_requested=args.n,
            seed=args.seed,
            category=category,
            metadata_lookup=metadata_lookup,
        )
        out_path = out_dir / f"{run_name}_samples.html"
        out_path.write_text(html_doc, encoding="utf-8")
        cat_str = f" [{category}]" if category else ""
        print(f"  → {out_path}  ({len(sampled)}/{len(records)} records{cat_str})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to a trial_details.jsonl OR a directory to scan recursively")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write <run_name>_samples.html files")
    parser.add_argument("--n", type=int, default=20,
                        help="Total samples per file (split across result_types)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bare-requests", type=str, default=None,
                        help="Override path to bare_requests.jsonl for metadata lookup "
                             "(default: data/<category>/bare_requests.jsonl)")
    args = parser.parse_args()
    main(args)
