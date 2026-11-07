#!/usr/bin/env python3
"""Empirical pool cost projection under ALO[0.20, 0.80] coarse-fine protocol.

Per-cell labeling (paper §coarse-fine):
    K^(c)_{m,x} ~ Bin(N_c, p_hat_{m,x})       (p_hat from N=40 response matrix)
    Label_{m,x} = L   if K^(c) ≤ k_lo
                  H   if K^(c) ≥ N_c - k_lo
                  M   otherwise
Item passes coarse iff ∃m: Label_{m,x} = M
       <=> NOT (∀m: Label_{m,x} ∈ {L, H}).

Item-level coarse-pass probability (with cells independent given p_hat):
    P(coarse-pass | x) = 1 - ∏_m [P(L | p_hat_{m,x}) + P(H | p_hat_{m,x})]
    r_c = mean_x P(coarse-pass | x)

FNR at ALO inner boundary p_0 = 0.20:
    Worst-case anchor configuration is "exactly one model at p_0 = 0.20,
    all others at p = 0" — saturated cells get labeled L w.p. ≈ 1; the
    boundary model is mis-labeled L iff K^(c) ≤ k_lo:
        FNR(p_0=0.20; N_c, k_lo) = P(Bin(N_c, 0.20) ≤ k_lo)

Companion / replacement of empirical_pool_costs.py (which assumes
AVG[0.05, 0.95] panel-sum saturation rule with FNR = (1 - p_0)^{M·N_c}).

Inputs:
    results/{cat}/filter_ablation/{model_dir}/response_matrix.jsonl
    (resolved via core.result_paths.filter_ablation_dir)

Outputs:
    stdout — per-cat r_c, total cost/cell, savings vs flat N=40 (= 360)
           + (N_c, k_lo) FNR-feasible sweep
    --output-txt: optional Chinese summary file for thesis use

Used by:
    paper/neurips_2026.tex §sec:coarse-fine (tab:c3)
    毕业论文_附录_coarse_fine_ALO.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from core.result_paths import filter_ablation_dir

TARGET_M = 9
N_TOTAL = 40
P_BOUNDARY = 0.20      # ALO inner boundary
ALPHA = 0.05           # FNR tolerance

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

CATEGORIES = {
    "constrained_completion": filter_ablation_dir("constrained_completion"),
    "harmful_refusal":        filter_ablation_dir("harmful_refusal"),
    "over_refusal":           filter_ablation_dir("over_refusal"),
    "sycophancy_resistance":  filter_ablation_dir("sycophancy_resistance"),
}

CAT_SHORT = {"constrained_completion": "B (CIF)",
             "harmful_refusal":        "A (HRR)",
             "over_refusal":           "A' (OR)",
             "sycophancy_resistance":  "C (SR)"}
CAT_ORDER = ["constrained_completion", "harmful_refusal",
             "over_refusal", "sycophancy_resistance"]


def load_per_cell_p_hat(response_dir: Path, keep_only_m9: bool = True):
    """Load (M, K) p_hat matrix from filter_ablation dir.

    p_hat[m, x] = pass_count[m, x] / total_trials[m, x]   (≈ p_{m,x})
    """
    model_dict = {}
    for mdir in sorted(response_dir.iterdir()):
        if not mdir.is_dir():
            continue
        meta_path = mdir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            name = json.load(f).get("model", mdir.name)
        if keep_only_m9 and name not in M9_PANEL:
            continue
        rm_path = mdir / "response_matrix.jsonl"
        if not rm_path.exists():
            continue
        items = {}
        with open(rm_path) as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                n = e.get("total_trials", 0)
                if n and n > 0:
                    items[e["prompt_id"]] = e["pass_count"] / n
        model_dict[name] = items

    if not model_dict:
        return np.array([]), []

    models = sorted(model_dict.keys())
    common = sorted(set.intersection(*[set(d.keys()) for d in model_dict.values()]))
    if not common:
        return np.array([]), models
    p_hat = np.array([
        [model_dict[m][pid] for m in models] for pid in common
    ]).T  # shape (M, K)
    return p_hat, models


def cell_extreme_prob(p_hat, N_c, k_lo):
    """Per-cell P(L) + P(H) given p_hat as true rate. Shape == p_hat.shape."""
    P_L = binom.cdf(k_lo, N_c, p_hat)
    P_H = 1.0 - binom.cdf(N_c - k_lo - 1, N_c, p_hat)
    return P_L + P_H


def item_coarse_pass_prob(p_hat_mat, N_c, k_lo):
    """P(item passes coarse | p_hat matrix). Shape (K,)."""
    P_extreme = cell_extreme_prob(p_hat_mat, N_c, k_lo)  # (M, K)
    return 1.0 - P_extreme.prod(axis=0)


def fnr_alo(N_c, k_lo, p_0=P_BOUNDARY):
    """FNR at ALO inner boundary under worst-case anchor (single-model)."""
    return float(binom.cdf(k_lo, N_c, p_0))


def report_locked(N_c, k_lo, p_hat_per_cat):
    print("=" * 100)
    print(f"  ALO[{P_BOUNDARY:.2f}, {1-P_BOUNDARY:.2f}] coarse-fine "
          f"projection — locked (N_c, k_lo) = ({N_c}, {k_lo})")
    print(f"  FNR(p_0={P_BOUNDARY:.2f}) under worst-case anchor = {fnr_alo(N_c, k_lo):.4f}")
    print(f"  Stage-1 cost = M·N_c = {TARGET_M * N_c}, "
          f"max stage-2 cost = M·(N - N_c) = {TARGET_M * (N_TOTAL - N_c)}, "
          f"flat baseline M·N = {TARGET_M * N_TOTAL}")
    print("=" * 100)
    print(f"  {'cat':<26} {'K':>5} {'r_c':>8} {'stage1':>8} {'stage2':>8} "
          f"{'total':>8} {'baseline':>9} {'savings':>9}")
    print("  " + "-" * 96)
    rows = []
    for cat in CAT_ORDER:
        if cat not in p_hat_per_cat:
            print(f"  {cat:<26} (missing)")
            continue
        p_hat, _ = p_hat_per_cat[cat]
        K = p_hat.shape[1]
        prob = item_coarse_pass_prob(p_hat, N_c, k_lo)
        r_c = float(prob.mean())
        stage1 = TARGET_M * N_c
        stage2 = r_c * TARGET_M * (N_TOTAL - N_c)
        total = stage1 + stage2
        baseline = TARGET_M * N_TOTAL
        sav = 1 - total / baseline
        rows.append((cat, K, r_c, stage1, stage2, total, baseline, sav))
        print(f"  {CAT_SHORT[cat]:<26} {K:>5} {r_c:>8.4f} "
              f"{stage1:>8.1f} {stage2:>8.1f} {total:>8.1f} {baseline:>9d} {sav:>8.1%}")
    mean_sav = float(np.mean([r[7] for r in rows])) if rows else float("nan")
    print("  " + "-" * 96)
    print(f"  {'mean savings':<26} {'':>5} {'':>8} {'':>8} {'':>8} {'':>8} {'':>9} {mean_sav:>8.1%}")
    return rows, mean_sav


def sweep(p_hat_per_cat, N_c_range, k_lo_range):
    print("\n" + "=" * 110)
    print(f"  FNR-feasible (N_c, k_lo) sweep at p_0 = {P_BOUNDARY}, α = {ALPHA}")
    print(f"  savings = 1 - (M·N_c + r_c·M·(N - N_c)) / (M·N)")
    print("=" * 110)
    cats_short = {c: CAT_SHORT[c].split()[0] for c in CAT_ORDER}
    print(f"  {'N_c':>4} {'k_lo':>5} {'FNR':>7} {'M·N_c':>7} "
          + "  ".join(f"{cats_short[c]:>8}" for c in CAT_ORDER)
          + "    mean")
    print("  " + "-" * 92)
    rows = []
    for N_c in N_c_range:
        for k_lo in k_lo_range:
            if k_lo >= N_c // 2:
                continue
            fnr = fnr_alo(N_c, k_lo)
            if fnr > ALPHA:
                continue
            savings = []
            for cat in CAT_ORDER:
                if cat not in p_hat_per_cat:
                    savings.append(None); continue
                p_hat, _ = p_hat_per_cat[cat]
                prob = item_coarse_pass_prob(p_hat, N_c, k_lo)
                r_c = float(prob.mean())
                total = TARGET_M * N_c + r_c * TARGET_M * (N_TOTAL - N_c)
                base = TARGET_M * N_TOTAL
                savings.append(1 - total / base)
            mean_sav = np.mean([s for s in savings if s is not None])
            rows.append((N_c, k_lo, fnr, savings, mean_sav))
            sav_str = "  ".join(f"{s*100:>7.1f}%" if s is not None else "    —"
                                for s in savings)
            print(f"  {N_c:>4} {k_lo:>5} {fnr:>6.2%} {TARGET_M*N_c:>7} {sav_str}    "
                  f"{mean_sav*100:>5.1f}%")
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-txt", type=str, default=None,
                    help="Write a Chinese summary to this file in addition to stdout")
    args = ap.parse_args()

    p_hat_per_cat = {}
    for cat in CAT_ORDER:
        p_hat, models = load_per_cell_p_hat(CATEGORIES[cat], keep_only_m9=True)
        if p_hat.size > 0 and p_hat.shape[0] == TARGET_M:
            p_hat_per_cat[cat] = (p_hat, models)
        else:
            print(f"WARN: {cat} loaded shape={p_hat.shape}, expected M=9 panel; skipped")

    # Primary: locked (14, 0)
    rows_14_0, mean_14_0 = report_locked(14, 0, p_hat_per_cat)
    # Comparison: (22, 1)
    rows_22_1, mean_22_1 = report_locked(22, 1, p_hat_per_cat)
    # Failing baseline (AVG era): (7, 0) — FNR exceeds α
    print(f"\n  Note: (N_c=7, k_lo=0) has FNR({P_BOUNDARY}) = {fnr_alo(7,0):.3f} > α — INFEASIBLE under ALO")
    rows_7_0, mean_7_0 = report_locked(7, 0, p_hat_per_cat)

    # Sweep
    sweep_rows = sweep(p_hat_per_cat, range(8, 26), range(0, 4))

    if args.output_txt:
        write_thesis_txt(args.output_txt, rows_14_0, mean_14_0,
                         rows_22_1, mean_22_1, rows_7_0, mean_7_0,
                         sweep_rows, p_hat_per_cat)
        print(f"\n  → wrote {args.output_txt}")


def write_thesis_txt(path, r14, m14, r22, m22, r7, m7, sweep_rows, p_hat_per_cat):
    """Write the Chinese thesis txt update file."""
    lines = []
    lines.append("================================================================")
    lines.append("  毕业论文 — 附录: ALO[0.20, 0.80] 下 coarse-fine 协议重算")
    lines.append(f"  生成自 scripts/analysis/empirical_pool_costs_alo.py")
    lines.append(f"  生成日期 2026-05-17 / 输入：results/{{cat}}/filter_ablation/{{model}}/response_matrix.jsonl")
    lines.append("  独立于 paper/ 与 docs/，与 NeurIPS 稿件保持隔离")
    lines.append("================================================================")
    lines.append("")
    lines.append("背景：过往附录的 coarse-fine 分析（毕业论文_实验数据汇总.txt §2）")
    lines.append("基于 AVG[0.05, 0.95] 协议下的面板汇总规则；切换 ALO 后舍弃语义、")
    lines.append("最差锚定形态、FNR 推导都发生变化，需重跑。本文件取代该 §2。")
    lines.append("")
    lines.append("─────────────────────────────────────────────────────────────")
    lines.append("1. ALO 协议下的逐模型粗筛规则（与 paper §coarse-fine 对应）")
    lines.append("─────────────────────────────────────────────────────────────")
    lines.append("")
    lines.append("  每个 (m, x) 单元独立分类：")
    lines.append("    K^(c)_{m,x} ~ Bin(N_c, p_{m,x})")
    lines.append("    Label_{m,x} = L   if K^(c) ≤ k_lo")
    lines.append("                  H   if K^(c) ≥ N_c - k_lo")
    lines.append("                  M   otherwise (潜在锚定)")
    lines.append("  题 x 粗筛通过当且仅当存在某 m 使 Label_{m,x} = M。")
    lines.append("")
    lines.append("  对比 AVG 时代规则：")
    lines.append("    S = Σ_m K^(c)_{m,x}, S ∈ [k_min, M·N_c - k_min] → keep")
    lines.append("    （面板汇总规则；只识别全面板同方向饱和）")
    lines.append("  ALO 规则新增了「面板分裂饱和」识别（如 5 个 L + 4 个 H, 任无 M）。")
    lines.append("")
    lines.append("─────────────────────────────────────────────────────────────")
    lines.append("2. FNR 闭式推导（ALO 最差锚定形态）")
    lines.append("─────────────────────────────────────────────────────────────")
    lines.append("")
    lines.append("  最差锚定形态：恰有一个模型 m* 在 ALO 下边界 p_0 = 0.20，")
    lines.append("  其余 M - 1 个在 p = 0（饱和零端）。此时 ALO 应保留该题，")
    lines.append("  而粗筛错误剔除 ⇔ m* 被误分类为 L：")
    lines.append("")
    lines.append("    FNR(p_0 = 0.20; N_c, k_lo) = P(Bin(N_c, 0.20) ≤ k_lo) ≤ α")
    lines.append("")
    lines.append("  设 α = 0.05、k_lo = 0：")
    lines.append("    0.80^{N_c} ≤ 0.05  ⇔  N_c ≥ ⌈ln 0.05 / ln 0.80⌉ = 14")
    lines.append(f"    FNR(0.20; 14, 0) = 0.80^14 = {fnr_alo(14, 0):.4f} (满足 α)")
    lines.append("")
    lines.append("  备选 k_lo = 1：")
    lines.append("    需要 P(Bin(N_c, 0.20) ≤ 1) ≤ 0.05")
    lines.append(f"    → N_c = 22 时 FNR = {fnr_alo(22, 1):.4f}，N_c = 21 时 FNR = {fnr_alo(21, 1):.4f}")
    lines.append(f"    阶段一代价 M·N_c = {TARGET_M*22} (+{(22-14)/14*100:.0f}% vs 14)")
    lines.append("")
    lines.append("  AVG 时代 (7, 0)/(7, 1) 复用到 ALO 下：")
    lines.append(f"    FNR(0.20; 7, 0) = 0.80^7 = {fnr_alo(7, 0):.4f}  → 4 倍超容忍，INFEASIBLE")
    lines.append("")
    lines.append("  锁定 (N_c, k_lo) = (14, 0)。")
    lines.append("")
    lines.append("─────────────────────────────────────────────────────────────")
    lines.append("3. 经验代价节省（M = 9, N_total = 40, baseline = 360）")
    lines.append("─────────────────────────────────────────────────────────────")
    lines.append("")
    lines.append("  3.1 锁定配置 (N_c, k_lo) = (14, 0)")
    lines.append("")
    lines.append("    cat              K     r_c    stage1  stage2  total   savings")
    lines.append("    ───────────────────────────────────────────────────────────")
    for row in r14:
        cat, K, r_c, s1, s2, tot, base, sav = row
        lines.append(f"    {CAT_SHORT[cat]:<14}  {K:>3}  {r_c:>6.4f}  "
                     f"{s1:>5.1f}  {s2:>6.1f}  {tot:>6.1f}  {sav*100:>5.1f}%")
    lines.append(f"    四类别均值                                              {m14*100:>5.1f}%")
    lines.append("")
    lines.append("  3.2 备选 (N_c, k_lo) = (22, 1)")
    lines.append("")
    lines.append("    cat              K     r_c    stage1  stage2  total   savings")
    lines.append("    ───────────────────────────────────────────────────────────")
    for row in r22:
        cat, K, r_c, s1, s2, tot, base, sav = row
        lines.append(f"    {CAT_SHORT[cat]:<14}  {K:>3}  {r_c:>6.4f}  "
                     f"{s1:>5.1f}  {s2:>6.1f}  {tot:>6.1f}  {sav*100:>5.1f}%")
    lines.append(f"    四类别均值                                              {m22*100:>5.1f}%")
    lines.append("")
    lines.append(f"    Δ vs (14, 0) = {(m22 - m14)*100:+.1f} pp 平均节省")
    lines.append("    阶段一代价从 126 升至 198 (+57%) 未被节省提升抵消。")
    lines.append("")
    lines.append("  3.3 失败基线 (N_c, k_lo) = (7, 0)  [FNR 不达标，仅供数值对比]")
    lines.append("")
    lines.append("    cat              K     r_c    stage1  stage2  total   savings")
    lines.append("    ───────────────────────────────────────────────────────────")
    for row in r7:
        cat, K, r_c, s1, s2, tot, base, sav = row
        lines.append(f"    {CAT_SHORT[cat]:<14}  {K:>3}  {r_c:>6.4f}  "
                     f"{s1:>5.1f}  {s2:>6.1f}  {tot:>6.1f}  {sav*100:>5.1f}%")
    lines.append(f"    四类别均值                                              {m7*100:>5.1f}%")
    lines.append("    (FNR=0.21 远超 α=0.05；锚定召回不可接受。)")
    lines.append("")
    lines.append("─────────────────────────────────────────────────────────────")
    lines.append("4. FNR-feasible (N_c, k_lo) 完整扫描表")
    lines.append("─────────────────────────────────────────────────────────────")
    lines.append("")
    lines.append("  约束：FNR(0.20; N_c, k_lo) = P(Bin(N_c, 0.2) ≤ k_lo) ≤ 0.05")
    lines.append("")
    lines.append("  N_c  k_lo   FNR    M·N_c   B(CIF)   A(HRR)   A'(OR)   C(SR)   mean")
    lines.append("  ──────────────────────────────────────────────────────────────────")
    for N_c, k_lo, fnr, savings, mean_sav in sweep_rows:
        sav_str = "  ".join(f"{s*100:>6.1f}%" if s is not None else "    —"
                            for s in savings)
        lines.append(f"  {N_c:>3}  {k_lo:>4}  {fnr*100:>4.2f}% {TARGET_M*N_c:>5}  "
                     f"{sav_str}  {mean_sav*100:>5.1f}%")
    lines.append("")
    lines.append("─────────────────────────────────────────────────────────────")
    lines.append("5. 定性观察 & paper §coarse-fine tab:c3 填表（locked (14, 0)）")
    lines.append("─────────────────────────────────────────────────────────────")
    lines.append("")
    lines.append("  paper tab:c3 应填入：")
    lines.append("")
    pool_shape = {
        "constrained_completion": "heavy U",
        "harmful_refusal": "moderate U",
        "over_refusal": "moderate U",
        "sycophancy_resistance": "near-uniform",
    }
    for row in r14:
        cat, K, r_c, s1, s2, tot, base, sav = row
        lines.append(f"  {CAT_SHORT[cat]:<14}  pool={pool_shape[cat]:<14}  "
                     f"r_c={r_c:.4f}  total/cell={tot:.1f}  vs 360  → {sav*100:.1f}% 节省")
    lines.append(f"  四类别均值节省 = {m14*100:.1f}%")
    lines.append("")
    lines.append("文件结束。")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
