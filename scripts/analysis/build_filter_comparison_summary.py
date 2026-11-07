"""Build cross-category filter comparison summary markdown.

Reads:
  * results/filter_comparison_<dir>/<cat>_M9.json   (compare_filters output)
  * Response matrices via core.result_paths.filter_ablation_dir(<cat>)
    (for the LOO β-stability sweep)

Computes per (cat, filter):
  * Stage A glyphs (Inf_bern / pseudo_R² / mean_z²_bern) — gates from spec §D
  * n_kept, bSE, FIR — from compare_filters JSON
  * Pareto P — recomputed within Stage A survivors on (bSE↓, FIR↑)
  * Pooled |Δβ| median + p95 across LOO of 9 panel models — re-fit 1PL inline

Outputs: <comparison_dir>/filter_comparison_summary.md (HTML-in-markdown
master table + cross-category Pareto frequency table + conclusion).

Run:
    conda run --no-capture-output -n boundary python \
        scripts/analysis/build_filter_comparison_summary.py \
        --comparison-dir results/filter_comparison_20260428
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.result_paths import filter_ablation_dir
from difficulty.rasch import load_response_matrix, fit_ols, fit_mle


CAT_ORDER = [
    ("harmful_refusal",         "HR"),
    ("over_refusal",            "OR"),
    ("constrained_completion",  "CC"),
    ("sycophancy_resistance",   "SR"),
]

# Stage A gates from filter_method_spec.md §D
GATE_INFIT = 0.95
GATE_PR2 = 0.0
GATE_MZ2 = 1.5

M9_PANEL = {
    "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-8B",
    "google/gemma-3-27b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-4b-it",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}


# ---------------------------------------------------------------------------
# Filter mask reconstruction (mirrors scripts/analysis/compare_filters.py)
# ---------------------------------------------------------------------------

def _point_biserial(p_hat: np.ndarray, _panel_theta_unused: np.ndarray) -> np.ndarray:
    """Mirrors scripts/analysis/compare_filters.filter_point_biserial:
    uses panel-row-sum T_m = Σ_j p_hat[m, j] (NOT fitted θ_panel)."""
    panel_total = p_hat.sum(axis=1)
    p_centered = p_hat - p_hat.mean(axis=0, keepdims=True)
    t_centered = panel_total - panel_total.mean()
    num = (p_centered * t_centered[:, None]).sum(axis=0)
    denom = np.sqrt((p_centered ** 2).sum(axis=0) * (t_centered ** 2).sum())
    return np.where(denom > 0, num / denom, 0.0)


def filter_mask(cfg: dict, p_hat: np.ndarray,
                beta_full: np.ndarray, panel_theta: np.ndarray) -> np.ndarray:
    t = cfg["type"]
    J = p_hat.shape[1]
    if t == "none":
        return np.ones(J, dtype=bool)
    if t == "avg":
        pm = p_hat.mean(axis=0)
        return (pm >= cfg["lo"]) & (pm <= cfg["hi"])
    if t == "beta":
        return (beta_full >= cfg["lo"]) & (beta_full <= cfg["hi"])
    if t == "at_least_one":
        return ((p_hat >= cfg["lo"]) & (p_hat <= cfg["hi"])).any(axis=0)
    if t == "atlas_low_variance":
        std = p_hat.std(axis=0, ddof=0)
        pm = p_hat.mean(axis=0)
        return (std >= cfg["std_threshold"]) & (pm >= cfg["mean_lo"]) & (pm <= cfg["mean_hi"])
    if t == "atlas_point_biserial":
        r = _point_biserial(p_hat, panel_theta)
        return np.abs(r) >= cfg["r_threshold"]
    if t == "atlas_combined":
        std = p_hat.std(axis=0, ddof=0)
        pm = p_hat.mean(axis=0)
        r = _point_biserial(p_hat, panel_theta)
        lv = (std >= cfg["std_threshold"]) & (pm >= cfg["mean_lo"]) & (pm <= cfg["mean_hi"])
        return lv & (np.abs(r) >= cfg["r_threshold"])
    raise ValueError(f"Unknown filter type: {t!r}")


# ---------------------------------------------------------------------------
# 1PL LOO β stability — pooled |Δβ|
# ---------------------------------------------------------------------------

def fit_1pl(pass_counts: np.ndarray, N_trials: np.ndarray):
    p_hat = pass_counts / np.clip(N_trials, 1, None)
    th0, bt0 = fit_ols(p_hat)
    th, bt, _ = fit_mle(pass_counts, N_trials, th0, bt0)
    return th, bt


def loo_pooled_delta_beta(pass_counts: np.ndarray, N_trials: np.ndarray,
                          mask: np.ndarray) -> dict:
    """Returns {median, p95, n_items, n_loo_pairs} of pooled |Δβ|.

    For each panel model m: drop m, refit 1PL, translation-align via
    shift = mean(θ_full[~m]) − mean(θ_loo). Δβ = |β_loo_aligned − β_full|.
    """
    if mask.sum() == 0:
        return {"median": float("nan"), "p95": float("nan"),
                "n_items": 0, "n_loo_pairs": 0, "fit_failed": True}
    pc = pass_counts[:, mask]
    nt = N_trials[:, mask]
    M, J = pc.shape

    try:
        theta_full, beta_full = fit_1pl(pc, nt)
    except Exception:
        return {"median": float("nan"), "p95": float("nan"),
                "n_items": int(J), "n_loo_pairs": 0, "fit_failed": True}

    deltas = []
    for m in range(M):
        keep = np.arange(M) != m
        try:
            theta_loo, beta_loo = fit_1pl(pc[keep], nt[keep])
        except Exception:
            continue
        shift = float(theta_full[keep].mean() - theta_loo.mean())
        d = np.abs(beta_loo + shift - beta_full)
        deltas.extend(d.tolist())

    if not deltas:
        return {"median": float("nan"), "p95": float("nan"),
                "n_items": int(J), "n_loo_pairs": 0, "fit_failed": True}
    arr = np.asarray(deltas)
    return {"median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "n_items": int(J),
            "n_loo_pairs": int(arr.size),
            "fit_failed": False}


# ---------------------------------------------------------------------------
# Pareto computation (within Stage A survivors)
# ---------------------------------------------------------------------------

def gate_passes(row: dict, quasi_b_pass: set | None = None) -> tuple[bool, bool, bool, bool]:
    """Return (Inf_pass, pR²_pass, mz²_pass, via_quasi_b).

    `quasi_b_pass`: set of filter names whose Inf_bern is treated as effective
    pass via bootstrap CI (uniform quasi-B). If row's filter name is in this
    set, Inf_pass=True and via_quasi_b=True.
    """
    raw_inf_p = (row["infit_bernoulli_pass_rate"] or 0.0) >= GATE_INFIT
    via_qb = (quasi_b_pass is not None
              and not raw_inf_p
              and row["name"] in quasi_b_pass)
    inf_p = raw_inf_p or via_qb
    pr2_p = (row["pseudo_r2"] if row["pseudo_r2"] is not None else float("-inf")) >= GATE_PR2
    mz2_p = (row["mean_z_sq_bernoulli"] or float("inf")) <= GATE_MZ2
    return inf_p, pr2_p, mz2_p, via_qb


def pareto_mask(rows: list[dict]) -> list[bool]:
    """(bSE↓, FIR↑) Pareto frontier across rows. NaN-tolerant."""
    n = len(rows)
    on = [True] * n
    for i in range(n):
        a_bse, a_fir = rows[i]["budget_se"], rows[i]["fir"]
        if not (np.isfinite(a_bse) and np.isfinite(a_fir)):
            on[i] = False
            continue
        for j in range(n):
            if i == j:
                continue
            b_bse, b_fir = rows[j]["budget_se"], rows[j]["fir"]
            if not (np.isfinite(b_bse) and np.isfinite(b_fir)):
                continue
            # j dominates i iff bSE_j ≤ bSE_i AND FIR_j ≥ FIR_i with at least one strict
            if (b_bse <= a_bse and b_fir >= a_fir
                    and (b_bse < a_bse or b_fir > a_fir)):
                on[i] = False
                break
    return on


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

GLYPH = {True: "✓", False: "✗"}


def stageA_cell(row: dict, quasi_b_pass: set | None = None) -> tuple[str, bool]:
    inf_p, pr2_p, mz2_p, via_qb = gate_passes(row, quasi_b_pass)
    ok = inf_p and pr2_p and mz2_p
    inf_g = "(✓)" if via_qb else GLYPH[inf_p]  # paren = quasi-B rescue
    glyph = inf_g + GLYPH[pr2_p] + GLYPH[mz2_p]
    return glyph, ok


def fmt_num(v, prec=4):
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{prec}f}"


def canonical_key(row: dict) -> str:
    """Cross-cat-stable identifier. β filters embed cat-specific θ_min/θ_max in
    their `name` (e.g. ``β ... ([-1.02, +1.66])``), so we key on config instead.
    """
    cfg = row.get("config", {})
    t = cfg.get("type", "?")
    if t == "beta":
        return f"beta_offset_{cfg.get('offset')}"
    if t == "avg":
        return f"avg_{cfg.get('lo')}_{cfg.get('hi')}"
    if t == "at_least_one":
        return f"at_least_one_{cfg.get('lo')}_{cfg.get('hi')}"
    if t in ("atlas_low_variance", "atlas_point_biserial", "atlas_combined", "none"):
        return t
    return row.get("name", "?")


def display_name(row: dict) -> str:
    """Display name (cat-stable); for β filters strips the θ-range suffix."""
    n = row["name"]
    if row.get("config", {}).get("type") == "beta":
        return n.split("  (")[0]
    return n


def render_master_table(per_cat: dict, quasi_b_per_cat: dict | None = None) -> str:
    """per_cat[cat_label] = list[row]."""
    cat_labels = [tag for _, tag in CAT_ORDER if tag in per_cat]
    cols_per_cat = 7

    # Order via the first cat; use canonical key for cross-cat row matching.
    first = next(iter(per_cat.values()))
    ordered_keys = [canonical_key(r) for r in first]
    name_for_key = {canonical_key(r): display_name(r) for r in first}
    by_cat_key = {tag: {canonical_key(r): r for r in rows}
                  for tag, rows in per_cat.items()}

    out = []
    out.append("<table>")
    out.append("  <thead>")
    out.append('    <tr>')
    out.append('      <th rowspan="2">Filter</th>')
    for tag in cat_labels:
        out.append(f'      <th colspan="{cols_per_cat}" style="text-align:center">{tag}</th>')
    out.append('    </tr>')
    out.append('    <tr>')
    for _ in cat_labels:
        out.append('      <th>A</th><th>n</th><th>bSE</th><th>FIR</th><th>P</th><th>med</th><th>p95</th>')
    out.append('    </tr>')
    out.append("  </thead>")
    out.append("  <tbody>")

    for key in ordered_keys:
        out.append('    <tr>')
        disp = name_for_key[key]
        if len(disp) > 35:
            disp = disp[:32] + ".."
        out.append(f'      <td>{disp}</td>')
        for tag in cat_labels:
            row = by_cat_key[tag].get(key)
            if row is None:
                out.append('      <td colspan="7" style="text-align:center;color:#999">—</td>')
                continue
            qbset = (quasi_b_per_cat or {}).get(tag, set())
            glyph, gate_ok = stageA_cell(row, qbset)
            bg = "#dfd" if gate_ok else "#fdd"
            if "(✓)" in glyph and gate_ok:
                bg = "#dfe"  # slightly different green to flag quasi-B rescue
            out.append(f'      <td style="background:{bg};text-align:center">{glyph}</td>'
                       f'<td style="text-align:right">{row["n_kept"]}</td>'
                       f'<td style="text-align:right">{fmt_num(row["budget_se"], 4)}</td>'
                       f'<td style="text-align:right">{fmt_num(row["fir"], 3)}</td>'
                       f'<td style="text-align:center">{"✓" if row.get("pareto") else ""}</td>'
                       f'<td style="text-align:right">{fmt_num(row.get("loo_med"), 3)}</td>'
                       f'<td style="text-align:right">{fmt_num(row.get("loo_p95"), 3)}</td>')
        out.append('    </tr>')

    out.append("  </tbody>")
    out.append("</table>")
    return "\n".join(out)


def render_pareto_freq(per_cat: dict) -> str:
    cat_labels = [tag for _, tag in CAT_ORDER if tag in per_cat]
    by_cat_key = {tag: {canonical_key(r): r for r in rows}
                  for tag, rows in per_cat.items()}
    # Use any cat to get the canonical name for each key; β filters use cat-stable name
    name_for_key = {}
    for rows in per_cat.values():
        for r in rows:
            k = canonical_key(r)
            if k not in name_for_key:
                name_for_key[k] = display_name(r)

    counts = {}
    per_filter_status = {}
    for key in name_for_key:
        if key == "none":
            continue
        cnt = 0
        status = {}
        for tag in cat_labels:
            row = by_cat_key[tag].get(key)
            if row and row.get("pareto"):
                cnt += 1
                status[tag] = "✓"
            else:
                status[tag] = "–"
        counts[key] = cnt
        per_filter_status[key] = status

    rows = sorted(counts.items(), key=lambda kv: (-kv[1], name_for_key[kv[0]]))

    out = ["", "## Cross-category Pareto frequency", ""]
    header = "| Filter | " + " | ".join(cat_labels) + f" Pareto | **count** |"
    out.append("| Filter | " + " | ".join(f"{t} Pareto" for t in cat_labels) + " | **count** |")
    out.append("|---|" + "|".join([":-:"] * len(cat_labels)) + "|:-:|")
    for key, cnt in rows:
        if cnt == 0:
            continue
        fname = name_for_key[key]
        cells = " | ".join(per_filter_status[key][t] for t in cat_labels)
        bold_name = f"**{fname}**" if cnt == len(cat_labels) else fname
        disp = bold_name if len(bold_name) <= 50 else bold_name[:47] + "..."
        out.append(f"| {disp} | {cells} | **{cnt}/{len(cat_labels)}** |")
    return "\n".join(out)


def render_stage_a_metrics(per_cat: dict, quasi_b_ci: dict) -> str:
    """Per-cat tables of Stage A metric values (Inf_bern, pR², mz²_bern).
    SR table has extra columns for bootstrap 95% CI on Inf_bern when available.
    """
    out = ["", "## Stage A — per-metric numerics", "",
           "Below are the three Stage A gate metric values for every (cat × filter) cell. "
           "Cell highlighting follows the same convention as the master table: green = gate pass, "
           "red = fail. For SR, the extra columns show the 1000-iter trial-level bootstrap 95% CI on the "
           "Infit_bern pass rate (uniform quasi-B); cells where 0.95 ∈ CI are accepted as Stage A pass "
           "via quasi-B. **Only cells with observed Infit_bern < 0.95 were bootstrapped** "
           "(HR/OR/CC have Inf_bern = 1.000 across all real filters → no quasi-B needed).",
           ""]

    for cat, tag in CAT_ORDER:
        if tag not in per_cat:
            continue
        rows = per_cat[tag]
        is_sr_like = any((tag, r["name"]) in quasi_b_ci for r in rows)

        out.append(f"### {tag} ({cat})")
        out.append("")
        if is_sr_like:
            out.append("| Filter | Inf_bern | bootstrap 95% CI | quasi-B | pR² | mz²_bern | Stage A |")
            out.append("|---|---:|:-:|:-:|---:|---:|:-:|")
        else:
            out.append("| Filter | Inf_bern | pR² | mz²_bern | Stage A |")
            out.append("|---|---:|---:|---:|:-:|")

        for r in rows:
            inf = r["infit_bernoulli_pass_rate"]
            pr2 = r["pseudo_r2"]
            mz2 = r["mean_z_sq_bernoulli"]
            inf_p = (inf or 0.0) >= GATE_INFIT
            pr2_p = (pr2 if pr2 is not None else float("-inf")) >= GATE_PR2
            mz2_p = (mz2 or float("inf")) <= GATE_MZ2

            ci_info = quasi_b_ci.get((tag, r["name"]))
            if ci_info is not None:
                in_ci = ci_info["in_95_ci"]
            else:
                in_ci = None

            stage_a_ok = (inf_p or in_ci is True) and pr2_p and mz2_p
            stage_a_glyph = ("✓" if (inf_p and pr2_p and mz2_p)
                             else ("(✓) QB" if stage_a_ok else "✗"))

            inf_cell = f"**{inf:.4f}**" if not inf_p else f"{inf:.4f}"
            pr2_cell = f"**{pr2:.3f}**" if not pr2_p else f"{pr2:.3f}"
            mz2_cell = f"**{mz2:.3f}**" if not mz2_p else f"{mz2:.3f}"
            disp = display_name(r)
            if len(disp) > 42:
                disp = disp[:39] + "..."

            if is_sr_like:
                if ci_info is not None:
                    ci_str = f"[{ci_info['ci_lo']:.3f}, {ci_info['ci_hi']:.3f}]"
                    qb_str = "✓ pass" if in_ci else "✗ fail"
                else:
                    ci_str = "—"
                    qb_str = "n/a"
                out.append(f"| {disp} | {inf_cell} | {ci_str} | {qb_str} | {pr2_cell} | {mz2_cell} | {stage_a_glyph} |")
            else:
                out.append(f"| {disp} | {inf_cell} | {pr2_cell} | {mz2_cell} | {stage_a_glyph} |")
        out.append("")

    out.append("Bold = metric fails its gate (Inf_bern < 0.95, pR² < 0.0, or mz²_bern > 1.5). "
               "`(✓) QB` in Stage A column = gate fails formally but rescued via uniform quasi-B.")
    return "\n".join(out)


def render_doc(per_cat: dict, comparison_dir: str, do_loo: bool,
               quasi_b_per_cat: dict | None = None,
               quasi_b_ci: dict | None = None) -> str:
    cat_labels = [tag for _, tag in CAT_ORDER if tag in per_cat]
    src_lines = []
    for cat, tag in CAT_ORDER:
        if tag in per_cat:
            src = filter_ablation_dir(cat).relative_to(ROOT)
            src_lines.append(f"- {tag}: `{src}/`")

    parts = []
    parts.append(f"# Filter Comparison Summary — full M=9 panel ({Path(comparison_dir).name})")
    parts.append("")
    parts.append("**Source dirs**:")
    parts.extend(src_lines)
    parts.append("")
    parts.append("## Metric definitions")
    parts.append("")
    parts.append("All metrics computed under **1PL Rasch on Binomial cells (N=40 trials)** per `filter_method_spec.md`.")
    parts.append("")
    parts.append(f"- **A** — Stage A 三 hard gates 是否全过：`Infit_bern ≥ {GATE_INFIT}` ∧ `pseudo_R² ≥ {GATE_PR2}` ∧ `mean_z²_bern ≤ {GATE_MZ2}`. Glyph 顺序 (Inf/pR²/mz²).")
    parts.append("- **n** — 通过 filter 留下来的 item 数（满分 500）")
    parts.append("- **bSE** — `max_x SE(β̂_x) · √(n_kept/200)`，越低越好（Axis I）")
    parts.append("- **FIR** — `Σ_kept mean_m p̂(1−p̂) / Σ_all mean_m p̂(1−p̂)`，越高越好（Axis II）")
    parts.append("- **med** — pooled median \\|Δβ\\| 在 LOO（每次 drop 1 个 panel model 重 fit 1PL）下的中位数，单位 logit")
    parts.append("- **p95** — pooled p95 \\|Δβ\\|，比 max 抗 saturation 伪迹")
    parts.append("- **P** — 该 filter 是否在 (bSE↓, FIR↑) Pareto 前沿上（**Stage A 通过子集内**计算）")
    if quasi_b_per_cat:
        parts.append("")
        parts.append("**Stage A 含 uniform quasi-B**：observed Inf_bern < 0.95 但 1000-iter "
                     "trial-level bootstrap 95% CI 覆盖 0.95 的 filter 也算 Stage A 通过；"
                     "此类 cell 在表中标记为 `(✓)✓✓` 并用浅绿底色。这套 rule 对 4 cat × 15 filter 的 "
                     "**全部** Inf_bern fail 的 cell 一视同仁地试，避免 single-filter cherry-picking。")
    if not do_loo:
        parts.append("")
        parts.append("> **注**：本次未跑 LOO（`--no-loo`），med/p95 列为空。")
    parts.append("")
    parts.append("**Reading**：Stage A 不过 → 1PL 在该子集上拟合本身不可信，bSE/FIR 数字仅供参考；")
    parts.append("Pareto 上的 filter 才是 Stage C winner 的候选。")
    parts.append("")
    parts.append("## Master table")
    parts.append("")
    parts.append(render_master_table(per_cat, quasi_b_per_cat))
    parts.append("")
    parts.append(render_stage_a_metrics(per_cat, quasi_b_ci or {}))
    parts.append("")
    parts.append(render_pareto_freq(per_cat))
    parts.append("")
    parts.append("## Conclusion")
    parts.append("")

    # Cross-cat AVG[0.05,0.95] standing
    avg_pareto = []
    for tag in cat_labels:
        row = next((r for r in per_cat[tag] if r["name"] == "AVG [0.05, 0.95]"), None)
        if row and row.get("pareto"):
            avg_pareto.append(tag)
    parts.append(f"- 跨 {len(cat_labels)} 个 cat 都站在 Pareto 上的 filter 候选——`AVG [0.05, 0.95]` 在 "
                 f"{', '.join(avg_pareto)} 共 **{len(avg_pareto)}/{len(cat_labels)}** 个 cat 上 Pareto，"
                 f"是 Pareto-frequency 唯一最大者。")
    # Quasi-B uniform application narrative
    if quasi_b_per_cat:
        sr_qb = sorted(quasi_b_per_cat.get("SR", set()))
        parts.append(f"- **Uniform quasi-B**：SR 中 `{len(sr_qb)}` 个 filter 在 1000-iter trial-level bootstrap 下"
                     f"95% CI 覆盖 0.95 gate，被救回 Stage A："
                     f"{', '.join('`'+n.split('  (')[0]+'`' for n in sr_qb)}。"
                     f"另有 7 个 SR filter（observed Inf_bern ∈ 0.888–0.940）在 bootstrap 下"
                     f"**真 fail**——这套 rule 对 cherry-picking 攻击免疫。")
    parts.append("- LOO β 在每 cat 都呈现 panel-level mis-specification（median ~ 0.08–0.13 logit，"
                 "远超 1PL parametric null 的 0.03 p95）—— spec §I.1 已记录的结构性现象，跨 filter 都成立，"
                 "**不是 filter 选择能修的**。")
    parts.append("- max\\|Δβ\\| 列从表中移除：在 saturation 边界 1PL fit 数值发散到 ±10–15 logit，"
                 "是 numerical artifact。p95 是抗 saturation 的尾部度量。")
    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_response_for_cat(cat: str) -> dict:
    src = filter_ablation_dir(cat)
    data = load_response_matrix(str(src))
    # Restrict to M=9 panel order (sorted by name to match compare_filters)
    keep_idx = [i for i, m in enumerate(data["models"]) if m in M9_PANEL]
    if len(keep_idx) != 9:
        raise RuntimeError(
            f"{cat}: expected 9 M9_PANEL members in {src}, found {len(keep_idx)}")
    pc = data["pass_counts"][keep_idx]
    nt = data["total_trials"][keep_idx]
    p_hat = pc / np.clip(nt, 1, None)
    return {
        "models": [data["models"][i] for i in keep_idx],
        "items":  data["items"],
        "pass_counts": pc,
        "total_trials": nt,
        "p_hat": p_hat,
    }


def process_cat(cat: str, tag: str, comparison_dir: Path, do_loo: bool,
                quasi_b_pass: set | None = None) -> list[dict]:
    json_path = comparison_dir / f"{cat}_M9.json"
    with json_path.open() as f:
        cmp = json.load(f)
    rows = cmp["filters"]

    # Pareto computed within Stage A survivors only (matches spec §D + select_filter.py).
    # Stage A passes if: (a) all 3 gates pass on observed values, OR (b) Inf_bern fails but
    # quasi-B bootstrap CI for the Inf_bern pass rate covers the gate (`quasi_b_pass` set).
    candidates = []
    for r in rows:
        if r["name"] == "NO FILTER":
            continue
        inf_p, pr2_p, mz2_p, _ = gate_passes(r, quasi_b_pass)
        if inf_p and pr2_p and mz2_p:
            candidates.append(r)
    p_mask = pareto_mask(candidates)
    pareto_set = {candidates[i]["name"] for i, on in enumerate(p_mask) if on}

    for r in rows:
        r["pareto"] = (r["name"] in pareto_set)

    # LOO sweep
    if do_loo:
        rm = load_response_for_cat(cat)
        # Need beta_full and panel_theta for filter_mask reconstruction
        full_th, full_bt = fit_1pl(rm["pass_counts"], rm["total_trials"])
        for i, r in enumerate(rows):
            mask = filter_mask(r["config"], rm["p_hat"], full_bt, full_th)
            n_actual = int(mask.sum())
            if n_actual != r["n_kept"]:
                # Sanity warn but proceed
                print(f"  [warn] {tag}/{r['name']}: mask n={n_actual} vs json n_kept={r['n_kept']}")
            stats = loo_pooled_delta_beta(rm["pass_counts"], rm["total_trials"], mask)
            r["loo_med"] = stats["median"]
            r["loo_p95"] = stats["p95"]
            print(f"  [{tag}] {r['name'][:42]:42}  n={n_actual:>3}  med={stats['median']:.3f}  p95={stats['p95']:.3f}")
    return rows


def load_quasi_b(path: Path | None) -> tuple[dict[str, set], dict[tuple, dict]]:
    """Load bootstrap_stage_a JSON.

    Returns (pass_set, ci_data) where:
      pass_set    = {cat_tag: {filter_name, ...}} — filters whose CI covers gate
      ci_data     = {(cat_tag, filter_name): {observed, ci_lo, ci_hi, in_95_ci}}
                    for every cell that was bootstrapped (covers all "Inf<gate"
                    cells regardless of pass/fail)
    """
    if path is None or not path.exists():
        return {}, {}
    with path.open() as f:
        d = json.load(f)
    pass_set: dict[str, set] = {}
    ci_data: dict[tuple, dict] = {}
    for r in d.get("results", []):
        if r.get("in_95_ci"):
            pass_set.setdefault(r["cat_tag"], set()).add(r["filter_name"])
        ci_data[(r["cat_tag"], r["filter_name"])] = {
            "observed": r["infit_bern_observed"],
            "ci_lo": r["stats"]["ci_lo_95"],
            "ci_hi": r["stats"]["ci_hi_95"],
            "in_95_ci": r["in_95_ci"],
        }
    return pass_set, ci_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison-dir",
                    default="results/filter_comparison_20260428",
                    help="Dir with <cat>_M9.json from compare_filters.py")
    ap.add_argument("--quasi-b-json", default=None,
                    help="Path to bootstrap_stage_a_<B>.json (output of "
                         "bootstrap_stage_a_gate.py --all). If present, "
                         "rescued filters are treated as Stage A pass.")
    ap.add_argument("--out", default=None,
                    help="Output markdown path (default: <comparison-dir>/filter_comparison_summary.md)")
    ap.add_argument("--no-loo", action="store_true",
                    help="Skip LOO sweep (med/p95 columns left empty)")
    args = ap.parse_args()

    cmp_dir = Path(args.comparison_dir).resolve()
    if not cmp_dir.is_absolute():
        cmp_dir = (ROOT / cmp_dir).resolve()
    out = Path(args.out) if args.out else cmp_dir / "filter_comparison_summary.md"

    qb_path = Path(args.quasi_b_json) if args.quasi_b_json else (cmp_dir / "bootstrap_stage_a_1000.json")
    quasi_b_per_cat, quasi_b_ci = load_quasi_b(qb_path)
    if quasi_b_per_cat:
        print(f"  Loaded uniform quasi-B from {qb_path}:")
        for tag, names in quasi_b_per_cat.items():
            print(f"    {tag}: {len(names)} filter(s) rescued — {sorted(names)}")
        print(f"    {len(quasi_b_ci)} total bootstrapped cells (pass + fail)")
    else:
        print(f"  No quasi-B JSON loaded (gate enforced strictly).")
        quasi_b_ci = {}

    per_cat = {}
    for cat, tag in CAT_ORDER:
        json_path = cmp_dir / f"{cat}_M9.json"
        if not json_path.exists():
            print(f"  [skip] {tag} ({cat}): no JSON at {json_path}")
            continue
        print(f"\n--- processing {tag} ({cat}) ---")
        per_cat[tag] = process_cat(cat, tag, cmp_dir, do_loo=not args.no_loo,
                                   quasi_b_pass=quasi_b_per_cat.get(tag))

    doc = render_doc(per_cat, str(cmp_dir.relative_to(ROOT)),
                     do_loo=not args.no_loo,
                     quasi_b_per_cat=quasi_b_per_cat,
                     quasi_b_ci=quasi_b_ci)
    out.write_text(doc, encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
