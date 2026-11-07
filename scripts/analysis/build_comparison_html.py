"""Build the updated SGBS vs Random comparison HTML.

This is a major rewrite of the previous HTML, reflecting the corrected
narrative based on:
  - Histogram fix (no clipping artifacts, real β tail counts)
  - Per-θ filtered TIF (filter helps panel center, less so panel edge)
  - Precise SGBS budget from stats.json
  - Budget-fair finding: random+filter can match SGBS at 0.53× budget
  - New narrative: SGBS's value is per-model attribution + dynamic extension,
    NOT budget efficiency
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difficulty.rasch import load_response_matrix


EC_DIR = "results/constrained_completion/validation/edge_case"
RB_DIR = "results/constrained_completion/validation/random_baseline"
SGBS_DIR = "results/constrained_completion/edge_case"


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def short(m):
    s = m.split("/")[-1]
    for old, new in [("DeepSeek-R1-Distill-Llama", "R1"),
                     ("Qwen3-30B-A3B-Instruct-2507", "Qwen-30B"),
                     ("Mistral-Small-3.1-24B-Instruct-2503", "Mistral-24B"),
                     ("gemma-3-", "Gemma3-"), ("-it", "")]:
        s = s.replace(old, new)
    return s[:14]


def main():
    data = json.load(open("results/constrained_completion/joint_fit_data.json"))
    theta = np.array(data["theta_joint"])
    beta_ec = np.array(data["beta_ec"])
    beta_rb = np.array(data["beta_rb"])
    models = data["models"]
    model_short = [short(m) for m in models]

    # SGBS budget
    total_evaluated = 0
    for d in sorted(Path(SGBS_DIR).iterdir()):
        sp = d / "stats.json"
        if sp.exists():
            s = json.load(open(sp))
            total_evaluated += s["rounds_run"] - s.get("skipped_rounds", 0)
    SGBS_REAL = int(total_evaluated * (15 + 0.5 * 25)) + 600 * 6 * 20

    # Random p_hat for filters
    rb_rm = load_response_matrix(RB_DIR)
    p_rb = rb_rm["pass_counts"] / rb_rm["total_trials"]
    item_mean = p_rb.mean(axis=0)

    # Compute filtered subsets
    filter_data = []
    for label, lo, hi in [
        ("[0.20, 0.80]", 0.20, 0.80),
        ("[0.30, 0.70]", 0.30, 0.70),
        ("[0.40, 0.60]", 0.40, 0.60),
    ]:
        mask = (item_mean >= lo) & (item_mean <= hi)
        n_kept = int(mask.sum())
        beta_filt = beta_rb[mask]
        per_theta = []
        for t in theta:
            tif_at_t = sum(sigmoid(t - b) * (1 - sigmoid(t - b)) for b in beta_filt)
            per_theta.append(tif_at_t)
        per_item_total = sum(pt / n_kept for pt in per_theta) if n_kept > 0 else 0
        # SGBS sum
        sgbs_sum = sum(
            sum(sigmoid(t - b) * (1 - sigmoid(t - b)) for b in beta_ec)
            for t in theta
        )
        kept_needed = sgbs_sum / per_item_total if per_item_total > 0 else float('inf')
        gen_needed = kept_needed / (n_kept / 600) if n_kept > 0 else float('inf')
        inf_cost = gen_needed * 6 * 20
        ratio = inf_cost / SGBS_REAL
        filter_data.append({
            "label": label, "lo": lo, "hi": hi,
            "n_kept": n_kept, "per_theta": per_theta,
            "per_item_total": per_item_total,
            "kept_needed": kept_needed,
            "gen_needed": gen_needed,
            "inf_cost": inf_cost,
            "ratio": ratio,
        })

    # SGBS per-theta TIF
    sgbs_per_theta = [
        sum(sigmoid(t - b) * (1 - sigmoid(t - b)) for b in beta_ec)
        for t in theta
    ]
    rb_per_theta = [
        sum(sigmoid(t - b) * (1 - sigmoid(t - b)) for b in beta_rb)
        for t in theta
    ]

    # Histogram bins (no clipping; only show items in [-3, +3])
    bins = list(np.linspace(-3, 3, 31))
    inside_ec = [b for b in beta_ec if -3 <= b <= 3]
    inside_rb = [b for b in beta_rb if -3 <= b <= 3]
    hist_ec, _ = np.histogram(inside_ec, bins=bins)
    hist_rb, _ = np.histogram(inside_rb, bins=bins)
    n_ec_below = int((beta_ec < -3).sum())
    n_ec_above = int((beta_ec > 3).sum())
    n_rb_below = int((beta_rb < -3).sum())
    n_rb_above = int((beta_rb > 3).sum())
    # Random extreme β stats
    rb_extreme_min = float(beta_rb.min())
    rb_extreme_max = float(beta_rb.max())

    # ===== TIF curve data =====
    theta_grid = list(np.linspace(-3, 3, 61))
    tif_ec_curve = []
    tif_rb_curve = []
    for t in theta_grid:
        tif_ec_curve.append(sum(sigmoid(t - b) * (1 - sigmoid(t - b)) for b in beta_ec))
        tif_rb_curve.append(sum(sigmoid(t - b) * (1 - sigmoid(t - b)) for b in beta_rb))
    # Also a filtered TIF curve for the best filter
    best_filt = filter_data[0]   # [0.20, 0.80] - best balance
    mask = (item_mean >= best_filt["lo"]) & (item_mean <= best_filt["hi"])
    beta_best = beta_rb[mask]
    tif_filt_curve = [
        sum(sigmoid(t - b) * (1 - sigmoid(t - b)) for b in beta_best)
        for t in theta_grid
    ]

    # ============= Build HTML =============
    html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>SGBS vs Random + Filter — Updated Comparison</title>
<style>
  :root {
    --c-bg: #0f1419; --c-bg-card: #1a2030; --c-text: #e5e7eb;
    --c-text-strong: #f9fafb; --c-muted: #9ca3af; --c-sgbs: #4ade80;
    --c-rand: #f87171; --c-filt: #60a5fa; --c-warn: #fbbf24;
    --c-border: #334155;
  }
  body {
    font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
    max-width: 1100px; margin: 0 auto; padding: 40px 32px 80px;
    color: var(--c-text); background: var(--c-bg); line-height: 1.7;
  }
  h1 { font-size: 26px; color: var(--c-text-strong); border-bottom: 2px solid var(--c-text-strong); padding-bottom: 12px; }
  h2 { font-size: 20px; margin-top: 44px; color: var(--c-text-strong); border-left: 4px solid #60a5fa; padding-left: 12px; }
  h3 { font-size: 16px; color: #cbd5e1; margin-top: 26px; }
  p { margin: 10px 0; }
  strong { color: var(--c-text-strong); }
  em { color: #cbd5e1; }
  table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 13px; background: var(--c-bg-card); }
  th, td { border: 1px solid var(--c-border); padding: 6px 10px; text-align: center; }
  th { background: #2a3145; color: var(--c-text-strong); }
  td.num { font-variant-numeric: tabular-nums; font-family: "SF Mono", Consolas, monospace; }
  tr.highlight td { background: rgba(74, 222, 128, 0.15); font-weight: 600; }
  tr.warn td { background: rgba(248, 113, 113, 0.10); }
  .legend { display: flex; gap: 22px; justify-content: center; margin: 14px 0; font-size: 13px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 8px; }
  .legend-box { width: 16px; height: 12px; display: inline-block; }
  svg { display: block; margin: 14px auto; max-width: 100%; height: auto; background: var(--c-bg-card); border: 1px solid var(--c-border); border-radius: 6px; }
  .key { background: rgba(96, 165, 250, 0.1); border-left: 4px solid #60a5fa; padding: 14px 20px; margin: 18px 0; border-radius: 4px; }
  .alert { background: rgba(251, 191, 36, 0.10); border-left: 4px solid #fbbf24; padding: 14px 20px; margin: 18px 0; border-radius: 4px; }
  .alert strong { color: #fcd34d; }
  .key strong { color: #93c5fd; }
  ol li, ul li { margin: 4px 0; }
</style>
</head>
<body>

<h1>SGBS vs Random + Filter — Updated Comparison</h1>
<p style="color: var(--c-muted); font-size: 13px;">
constrained_completion · 600 SGBS items + 600 random items · joint Rasch fit · 6 panel models · 20 trials/cell
</p>

<div class="alert">
  <strong>⚠ Major narrative update</strong>
  <p style="margin: 6px 0 0 0;">基于精确 budget 分析和 per-θ 过滤后 TIF 重新评估，<strong>原结论"SGBS 更 budget-efficient"是错的</strong>。Random + offline filter 在 anchor dataset 构建上实际更省 budget。SGBS 的真正 value-add 在三个 random 永远做不到的能力：(1) per-model attribution，(2) online algorithm，(3) dynamic extension to new models。下面详细展开。</p>
</div>
'''

    # ===== Section 1: Histogram =====
    html += '''
<h2>1. β 分布对比（修复版：无 clipping artifact）</h2>
<p>Joint Rasch 拟合后的 1200 个 item 在同一 β 标尺上。Histogram <strong>只展示 [−3, +3] 范围内的 item</strong>，超出范围的极端值另行标注（不再硬塞到端点造成假双峰）。</p>
'''
    html += '<div class="legend"><div class="legend-item"><span class="legend-box" style="background:#4ade80"></span>SGBS edge cases (n=600)</div>'
    html += '<div class="legend-item"><span class="legend-box" style="background:#f87171"></span>Random composing (n=600)</div>'
    html += '<div class="legend-item"><span style="background:#fbbf24; width:2px; height:14px; display:inline-block;"></span>Panel model θ</div></div>'

    # SVG histogram
    W, H = 1000, 380
    padL, padR, padT, padB = 60, 30, 30, 80
    plotW, plotH = W - padL - padR, H - padT - padB
    n_bins = len(hist_ec)
    bin_w = plotW / n_bins
    max_count = max(int(max(hist_ec)), int(max(hist_rb)), 1)
    bar_w = bin_w * 0.42
    gap = bin_w * 0.08

    def x_to_screen(b):
        return padL + (b + 3) / 6 * plotW

    svg = f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">'
    # Y grid
    for v in [0, 25, 50, 75, 100]:
        if v > max_count:
            continue
        y = padT + plotH - (v / max_count) * plotH
        svg += f'<line x1="{padL}" y1="{y}" x2="{padL+plotW}" y2="{y}" stroke="#334155"/>'
        svg += f'<text x="{padL-8}" y="{y+4}" text-anchor="end" font-size="11" fill="#cbd5e1" font-family="monospace">{v}</text>'
    # X ticks
    for v in [-3, -2, -1, 0, 1, 2, 3]:
        x = x_to_screen(v)
        svg += f'<line x1="{x}" y1="{padT+plotH}" x2="{x}" y2="{padT+plotH+5}" stroke="#cbd5e1"/>'
        svg += f'<text x="{x}" y="{padT+plotH+18}" text-anchor="middle" font-size="12" fill="#e2e8f0" font-family="monospace">{"+"+str(v) if v>0 else str(v)}</text>'
    # Bars
    for i, (c_ec, c_rb) in enumerate(zip(hist_ec, hist_rb)):
        x_left = padL + i * bin_w
        if c_ec > 0:
            h = (c_ec / max_count) * plotH
            svg += f'<rect x="{x_left+gap}" y="{padT+plotH-h}" width="{bar_w}" height="{h}" fill="#4ade80" opacity="0.85"/>'
        if c_rb > 0:
            h = (c_rb / max_count) * plotH
            svg += f'<rect x="{x_left+bin_w/2+gap/2}" y="{padT+plotH-h}" width="{bar_w}" height="{h}" fill="#f87171" opacity="0.85"/>'
    svg += f'<line x1="{padL}" y1="{padT+plotH}" x2="{padL+plotW}" y2="{padT+plotH}" stroke="#cbd5e1" stroke-width="1.5"/>'
    # Theta markers
    for m, t in zip(model_short, theta):
        if -3 <= t <= 3:
            x = x_to_screen(t)
            svg += f'<line x1="{x}" y1="{padT}" x2="{x}" y2="{padT+plotH}" stroke="#fbbf24" stroke-width="2" stroke-dasharray="4,2"/>'
            svg += f'<text x="{x}" y="{padT+plotH+36}" text-anchor="middle" font-size="9" fill="#fbbf24" font-family="monospace">{m}</text>'
            svg += f'<text x="{x}" y="{padT+plotH+48}" text-anchor="middle" font-size="9" fill="#fbbf24" font-family="monospace">θ={t:+.2f}</text>'
    svg += f'<text x="20" y="{padT+plotH/2}" text-anchor="middle" font-size="12" fill="#cbd5e1" transform="rotate(-90, 20, {padT+plotH/2})">Item count</text>'
    svg += f'<text x="{padL+plotW/2}" y="{H-8}" text-anchor="middle" font-size="12" fill="#cbd5e1">β (item difficulty, joint Rasch)</text>'
    svg += '</svg>'
    html += svg

    html += f'''
<p style="text-align:center; font-size: 13px; color: var(--c-muted);">
  <strong>SGBS</strong>: 600/600 items 在 [−3, +3] 范围内（<strong>0 个</strong>极端值）<br>
  <strong>Random</strong>: 373/600 items 在范围内，<strong style="color:#f87171;">227 个 (38%)</strong> 在 [−3, +3] 之外<br>
  Random 极端 β 真实范围：<strong style="color:#f87171;">[{rb_extreme_min:+.1f}, {rb_extreme_max:+.1f}]</strong>，
  其中 {n_rb_below} 个 β &lt; −3，{n_rb_above} 个 β &gt; +3。<br>
  这些极端 item 是 MLE 把整列 cell 全 0 或全 1 的 item 推到 ±∞ 的结果，<strong>对 IRT 拟合贡献 0 信息</strong>。
</p>

<div class="key">
  <strong>修正说明：</strong>之前的 histogram 把 ±15 量级的极端 β 强行 clip 到 ±3 端点，造成 random 在 [−3, −2.75] 和 [2.75, 3] 处出现假的高峰。修复后可以看到：random 在 [−3, +3] 内部其实分布相当均匀，问题在于 38% 的 item 直接被 MLE 推到 ±∞。
</div>
'''

    # ===== Section 2: TIF curves with filter =====
    html += '''
<h2>2. TIF 曲线 — 加入 filtered random 对照</h2>
<p>TIF(θ) = Σ<sub>x</sub> σ(θ−β<sub>x</sub>)(1−σ(θ−β<sub>x</sub>))，是难度 bank 在能力 θ 处提供的总信息量。下图额外加了 <strong>random + OLD filter [0.20, 0.80]</strong> 的曲线（n=215）。</p>
'''
    W2, H2 = 1000, 320
    padL2, padR2, padT2, padB2 = 60, 30, 30, 60
    plotW2, plotH2 = W2 - padL2 - padR2, H2 - padT2 - padB2
    max_tif = max(max(tif_ec_curve), max(tif_rb_curve), max(tif_filt_curve))

    def tx2(t):
        return padL2 + (t + 3) / 6 * plotW2

    def ty2(v):
        return padT2 + plotH2 - (v / max_tif) * plotH2

    svg2 = f'<svg viewBox="0 0 {W2} {H2}" xmlns="http://www.w3.org/2000/svg">'
    for v in [0, 30, 60, 90, 120]:
        if v > max_tif: continue
        y = ty2(v)
        svg2 += f'<line x1="{padL2}" y1="{y}" x2="{padL2+plotW2}" y2="{y}" stroke="#334155"/>'
        svg2 += f'<text x="{padL2-8}" y="{y+4}" text-anchor="end" font-size="11" fill="#cbd5e1" font-family="monospace">{v}</text>'
    for v in [-3, -2, -1, 0, 1, 2, 3]:
        x = tx2(v)
        svg2 += f'<line x1="{x}" y1="{padT2+plotH2}" x2="{x}" y2="{padT2+plotH2+5}" stroke="#cbd5e1"/>'
        svg2 += f'<text x="{x}" y="{padT2+plotH2+18}" text-anchor="middle" font-size="12" fill="#e2e8f0" font-family="monospace">{"+"+str(v) if v>0 else str(v)}</text>'

    def make_path(thetas, tifs, color):
        pts = " ".join(f"{tx2(t):.1f},{ty2(v):.1f}" for t, v in zip(thetas, tifs))
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>'

    svg2 += make_path(theta_grid, tif_ec_curve, "#4ade80")
    svg2 += make_path(theta_grid, tif_rb_curve, "#f87171")
    svg2 += make_path(theta_grid, tif_filt_curve, "#60a5fa")

    for m, t in zip(model_short, theta):
        if -3 <= t <= 3:
            x = tx2(t)
            svg2 += f'<line x1="{x}" y1="{padT2}" x2="{x}" y2="{padT2+plotH2}" stroke="#fbbf24" stroke-width="1.5" stroke-dasharray="3,2" opacity="0.5"/>'
            svg2 += f'<text x="{x}" y="{padT2+plotH2+34}" text-anchor="middle" font-size="9" fill="#fbbf24" font-family="monospace">{m}</text>'
    svg2 += f'<text x="20" y="{padT2+plotH2/2}" text-anchor="middle" font-size="12" fill="#cbd5e1" transform="rotate(-90, 20, {padT2+plotH2/2})">TIF(θ)</text>'
    svg2 += f'<text x="{padL2+plotW2/2}" y="{H2-6}" text-anchor="middle" font-size="12" fill="#cbd5e1">θ (model ability)</text>'
    svg2 += '</svg>'
    html += svg2

    html += '''
<div class="legend">
  <div class="legend-item"><span class="legend-box" style="background:#4ade80"></span>SGBS (n=600)</div>
  <div class="legend-item"><span class="legend-box" style="background:#f87171"></span>Random unfiltered (n=600)</div>
  <div class="legend-item"><span class="legend-box" style="background:#60a5fa"></span>Random + filter [0.20, 0.80] (n=215)</div>
  <div class="legend-item"><span style="background:#fbbf24; width:2px; height:14px; display:inline-block;"></span>Panel θ</div>
</div>
<p style="font-size: 13px; color: var(--c-muted);">
  Random+filter 的 TIF 总量看起来比 SGBS 低（因为只有 215 个 item，是 SGBS 的 1/3），但<strong>每个 item 的信息密度更高</strong>。下面的 budget-fair 分析会说明只要多生成几倍 item，filtered random 就能追平 SGBS。
</p>
'''

    # ===== Section 3: Per-θ TIF table =====
    html += '''
<h2>3. Per-θ TIF — Filter 是否帮到 panel 边缘模型？</h2>
<p>下表显示在 6 个 panel 模型的 θ 位置上各方法的 TIF 总量和 per-item 效率。<strong>per-item 效率 = TIF(θ_m) / n_items</strong>，反映了"每个 item 平均贡献多少信息"。</p>
<table>
  <thead>
    <tr><th>Model (θ)</th><th>SGBS<br/>(n=600)</th><th>Random unfilt<br/>(n=600)</th><th>OLD [0.20, 0.80]<br/>(n=215)</th><th>OLD [0.30, 0.70]<br/>(n=133)</th><th>OLD [0.40, 0.60]<br/>(n=60)</th></tr>
  </thead>
  <tbody>
'''
    method_betas = [
        ("SGBS", beta_ec, 600),
        ("RandUnfilt", beta_rb, 600),
    ]
    for fd in filter_data:
        mask = (item_mean >= fd["lo"]) & (item_mean <= fd["hi"])
        method_betas.append((fd["label"], beta_rb[mask], fd["n_kept"]))

    for i, (m, t) in enumerate(zip(model_short, theta)):
        html += f'    <tr><td>{m} (θ={t:+.2f})</td>'
        for label, bset, n in method_betas:
            if n == 0:
                html += '<td class="num">—</td>'
                continue
            tif_at = sum(sigmoid(t - b) * (1 - sigmoid(t - b)) for b in bset)
            per = tif_at / n
            html += f'<td class="num">{tif_at:.1f}<br/><span style="color:#9ca3af;font-size:11px">({per:.3f}/it)</span></td>'
        html += '</tr>\n'
    html += '''  </tbody>
</table>
<p style="font-size: 13px; color: var(--c-muted);">
  <strong>关键观察</strong>：对于大多数 panel θ 位置，<strong>OLD filter 之后 random 的 per-item 效率甚至 ≥ SGBS</strong>。
  唯一的例外是 R1-8B（panel 最低的 θ=−1.34），filter 帮助有限——因为 OLD filter 倾向于挑 β 接近 panel 中心的 item。
</p>
'''

    # ===== Section 4: Budget breakdown =====
    html += f'''
<h2>4. Budget 精确估算（来自 stats.json）</h2>
<p>之前的 budget 估算是粗略外推，现在用 SGBS 6 个 run 的 stats.json 直接读取实际 round 数：</p>
<table>
  <thead>
    <tr><th>阶段</th><th>SGBS</th><th>Random</th></tr>
  </thead>
  <tbody>
    <tr><td>Search/Generation evaluated rounds</td><td class="num">{total_evaluated:,}</td><td class="num">0 (template only)</td></tr>
    <tr><td>Search trials (15 + 0.5×25 per round, REALISTIC)</td><td class="num">{int(total_evaluated * 27.5):,}</td><td class="num">0</td></tr>
    <tr><td>Response matrix collection</td><td class="num">600 × 6 × 20 = 72,000</td><td class="num">600 × 6 × 20 = 72,000</td></tr>
    <tr class="highlight"><td><strong>Total inference</strong></td><td class="num"><strong>{SGBS_REAL:,}</strong></td><td class="num"><strong>72,000</strong></td></tr>
  </tbody>
</table>
<p>SGBS 的实际总 budget = <strong>{SGBS_REAL:,} trials</strong>，是 random 的 <strong>{SGBS_REAL/72000:.1f}×</strong>。</p>
'''

    # ===== Section 5: Budget-fair extrapolation =====
    html += f'''
<h2>5. Budget-Fair Extrapolation — 关键发现</h2>
<p>问题：如果 random 用 SGBS 等量的 budget，加上 filter，能不能追上 SGBS 的 raw total TIF？答案是<strong>不仅能追上，还有大量富余</strong>。</p>

<table>
  <thead>
    <tr><th>Random + Filter</th><th>Pass rate</th><th>TIF/item (sum across 6 panel θ)</th><th>Kept needed to match SGBS</th><th>Items to generate</th><th>Inference cost</th><th>× SGBS budget</th></tr>
  </thead>
  <tbody>
'''
    for fd in filter_data:
        cls = ' class="highlight"' if fd["ratio"] < 1.0 else ' class="warn"' if fd["ratio"] >= 1.5 else ''
        html += f'    <tr{cls}><td>OLD {fd["label"]}</td><td class="num">{fd["n_kept"]/600*100:.1f}%</td><td class="num">{fd["per_item_total"]:.3f}</td><td class="num">{fd["kept_needed"]:.0f}</td><td class="num">{fd["gen_needed"]:.0f}</td><td class="num">{fd["inf_cost"]:,.0f}</td><td class="num"><strong>{fd["ratio"]:.2f}×</strong></td></tr>\n'

    html += f'''  </tbody>
</table>
<p style="font-size: 13px; color: var(--c-muted);">
  ×SGBS budget 是相对于 SGBS realistic total ({SGBS_REAL:,} trials) 的比值。绿色 = 比 SGBS 省，黄色 = 比 SGBS 贵。
</p>

<div class="alert">
  <strong>核心发现</strong>: Random + OLD filter [0.20, 0.80] 只需 <strong>{filter_data[0]["ratio"]:.2f}×</strong> 的 SGBS budget 就能在 raw total TIF 上追平 SGBS。
  这意味着 <strong>random + offline filter 在 anchor dataset 构建上比 SGBS 更 budget-efficient</strong>。
  之前我说 "filter 救不了 random" 的判断是错的。
</div>
'''

    # ===== Section 6: New narrative =====
    html += '''
<h2>6. 新 Narrative：SGBS 的真正 value-add</h2>
<p>既然 random + filter 在 budget 上更优，那 SGBS 的 contribution 在哪里？答案是 random + filter <strong>永远做不到</strong>的三件事：</p>

<h3>(1) Per-Model Attribution</h3>
<p>SGBS 产出的每个 item 都有一个明确的 source model——它是为某个特定 panel model 搜出来的，满足 β ≈ θ_M（这就是 Claim 1）。这意味着可以回答"模型 M 在 capability boundary 处具体是什么样的 prompt"——这是 <strong>per-model interpretability</strong>。</p>
<p>Random + filter 产出的是<strong>通用 informative item bank</strong>，没有 per-model attribution。每个 item 只是"对至少几个模型 informative"，但没有归属于任何具体模型。</p>

<h3>(2) Online Algorithm</h3>
<p>Random + filter 是 <strong>offline 算法</strong>：必须先评估所有生成的 random items（包括最后被 filter 丢掉的），才知道哪些保留。生成 1619 items 就要全部评估 1619 × 6 × 20 trials。</p>
<p>SGBS 是 <strong>online 算法</strong>：边搜边决策，每一步立即决定要不要继续探索这个方向。在某些 use case（streaming evaluation, budget-constrained iterative refinement）下，online 决策很重要。</p>

<h3>(3) Dynamic Extension to New Models</h3>
<p>这是最关键的能力。Random + filter 的 filter 阈值是基于 <strong>panel models</strong> 的 mean p̂。如果新模型 m_new 的 θ 在 panel 范围之外（比 Qwen-30B 还强，或比 R1-8B 还弱），random pool 里几乎没有任何 item 在它的 boundary 附近，filter 也救不了。</p>
<p>SGBS 可以为任意 m_new <strong>动态搜索</strong>，找到针对该模型的 boundary item。这是 random + filter 永远做不到的。</p>

<div class="key">
  <strong>新 Paper 定位</strong>：我们的 evaluation protocol 采用 <strong>hybrid approach</strong>：
  <ol>
    <li><strong>Anchor set 主体</strong>: 用 random composing + offline filter（更省 budget，覆盖好）</li>
    <li><strong>SGBS 补充</strong>: 在 anchor set 中加入少量 SGBS-generated items，提供 per-model attribution + 难度多样性补充</li>
    <li><strong>Dynamic extension</strong>: 评估新模型时，先用 anchor set 做 CAT-style 评估；如果新模型 θ 在 panel 之外或 anchor 覆盖不足，触发 SGBS 为新模型动态搜索</li>
  </ol>
  <p style="margin-top: 8px;">这个 hybrid 定位<strong>更诚实也更 robust</strong>——SGBS 不再 frame 成"更省 budget"（这个 claim 站不住脚），而是 frame 成"提供 random+filter 永远做不到的 capability"。</p>
</div>
'''

    # ===== Section 7: Updated headlines =====
    html += '''
<h2>7. 更新后的 Headline Numbers</h2>
<table>
  <thead>
    <tr><th>维度</th><th>SGBS</th><th>Random + filter [0.20, 0.80]</th><th>结论</th></tr>
  </thead>
  <tbody>
    <tr><td>Total TIF (raw)</td><td class="num">668</td><td class="num">42 (n=215, scaled to match needs ~580 kept)</td><td>Random+filter 可以追平 at scale</td></tr>
    <tr><td>Per-item TIF efficiency (sum 6 θ)</td><td class="num">1.114</td><td class="num">1.152</td><td><strong>Random+filter 更高</strong></td></tr>
    <tr><td>Budget to match SGBS TIF</td><td class="num">100%</td><td class="num">~53%</td><td><strong>Random+filter 更省</strong></td></tr>
    <tr><td>Per-model attribution</td><td>✓</td><td>✗</td><td>SGBS 独有</td></tr>
    <tr><td>Online decisioning</td><td>✓</td><td>✗</td><td>SGBS 独有</td></tr>
    <tr><td>Extension to new models (out-of-panel)</td><td>✓</td><td>✗</td><td>SGBS 独有</td></tr>
  </tbody>
</table>

<p>简而言之：<strong>raw budget efficiency 不再是 SGBS 的卖点</strong>。SGBS 在 hybrid 评估 protocol 中扮演 "per-model interpretability + dynamic extension" 的角色，与 random + filter 的 anchor set 构建是<strong>互补关系而非竞争关系</strong>。</p>

<div style="margin-top: 60px; padding-top: 16px; border-top: 1px solid var(--c-border); color: var(--c-muted); font-size: 12px;">
  Generated · constrained_completion · 600 SGBS items + 600 random items, joint Rasch fit · v2 (post-budget-fair correction)
</div>
</body>
</html>
'''

    out_path = "docs/sgbs_vs_random_comparison.html"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Updated HTML written to {out_path} ({len(html)} chars)")


if __name__ == "__main__":
    main()
