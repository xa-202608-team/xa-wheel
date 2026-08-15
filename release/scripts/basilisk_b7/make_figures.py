#!/usr/bin/env python
"""scripts/basilisk_b7/make_figures.py

BASILISK-B7 §6..§11 —— 生成最终技术报告用图。

**只读**。数据来源仅限:
  - checkpoints/basilisk_b7/frozen_result_index.json   (指标)
  - checkpoints/basilisk_b6/all_raw.npz                (冻结逐点预测)
  - data/features/wheel/basilisk_b19/target_features.h5 (冻结观测特征)
  - checkpoints/basilisk_b18/dataset_audit.json         (冻结 EOL 索引)
  - checkpoints/basilisk_b2/split.json                  (B2 划分, 仅 Fig5 诊断)
  - docs/basilisk_b21/split_manifest.json               (B2.1 划分, 仅 Fig5)

禁止训练、禁止重新评估、禁止重新 bootstrap。轨迹与 seed 的选择规则全部写死在
本脚本里且完全 deterministic —— 不允许人工挑好看的曲线。

输出 (PNG + SVG, dpi >= 300, 全英文标注):
  docs/figures/basilisk_final/fig1_basilisk_health_trajectory.{png,svg}
  docs/figures/basilisk_final/fig2_transfer_gain_vs_labels.{png,svg}
  docs/figures/basilisk_final/fig3_primary_method_comparison.{png,svg}
  docs/figures/basilisk_final/fig4_health_management_flow.{png,svg}
  docs/figures/basilisk_final/fig5_split_coverage_diagnostic.{png,svg}
  docs/basilisk_b7/figure_manifest.md
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches            # noqa: E402
import matplotlib.pyplot as plt                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "docs" / "figures" / "basilisk_final"
MANIFEST = ROOT / "docs" / "basilisk_b7" / "figure_manifest.md"

INDEX = ROOT / "checkpoints" / "basilisk_b7" / "frozen_result_index.json"
RAW = ROOT / "checkpoints" / "basilisk_b6" / "all_raw.npz"
FEAT = ROOT / "data" / "features" / "wheel" / "basilisk_b19" / \
    "target_features.h5"
B18_AUDIT = ROOT / "checkpoints" / "basilisk_b18" / "dataset_audit.json"
B2_SPLIT = ROOT / "checkpoints" / "basilisk_b2" / "split.json"
B21_SPLIT = ROOT / "docs" / "basilisk_b21" / "split_manifest.json"

DPI = 320
PRIMARY_LEVEL = 5
BINS = ["short", "medium", "long"]
LEVELS = [3, 5, 10, 21]

# 单位换算 —— 与 B1.8/B1.9 冻结参数一致, 不引入新定义
DT_S = 1800.0                     # B1.9 退化聚合窗口 30 min
SEC_PER_YEAR = 31557600.0
YR_PER_SAMPLE = DT_S / SEC_PER_YEAR

# 论文风格: 无花哨主题, 无渐变, 无 3D
plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
    "figure.autolayout": False,
})

COLOR = {
    "damage_extrapolation": "#1b1b1b",
    "target_only": "#1f6fb4",
    "source_finetune": "#d95f02",
    "source_mmd_finetune": "#118a5a",
    "const_mean_info": "#8c8c8c",
    "true": "#b3261e",
}
LABEL = {
    "damage_extrapolation": "damage_extrapolation (physics baseline)",
    "target_only": "target_only",
    "source_finetune": "source_finetune",
    "source_mmd_finetune": "source_mmd_finetune",
    "const_mean_info": "const_mean_info",
}

_saved: list[tuple[str, str]] = []


def save(fig, stem: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        p = FIGDIR / f"{stem}.{ext}"
        fig.savefig(p, format=ext, bbox_inches="tight")
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        _saved.append((str(p.relative_to(ROOT)).replace("\\", "/"), sha))
        print(f">> {p.relative_to(ROOT)}  sha256={sha[:16]}...")
    plt.close(fig)


# ==========================================================================
# Figure 1 —— §6 单条冻结 test 轨迹的健康与 RUL
# ==========================================================================
def select_seed(idx: dict) -> int:
    """确定性 seed 选择: PRIMARY 档 target_only per-seed RMSE 的**中位数** seed。

    取中位数而非最优 —— 避免展示"最好看的一次训练"。
    """
    lv = idx["facts"]["C_b6_label_scarcity_matrix"]["by_level"]["5"]
    seeds = idx["facts"]["C_b6_label_scarcity_matrix"]["formal_seeds"]
    rmse = lv["per_seed_info_macro_rmse"]["target_only"]
    order = sorted(range(len(seeds)), key=lambda i: rmse[i])
    return int(seeds[order[len(order) // 2]])


def select_trajectory(tids: np.ndarray, eol_of: dict) -> tuple[int, dict]:
    """确定性轨迹选择规则 (§6, 禁止人工挑选):

    1. 只在 PRIMARY test set 中选;
    2. 只取 event-observed 轨迹 (right-censored 没有真 EOL, 不得伪造);
    3. 取 EOL 最接近 test event EOL **中位数**的那一条;
    4. 若并列, 取 traj_id 最小者。
    """
    cand = sorted(int(t) for t in np.unique(tids)
                  if eol_of[int(t)]["event_observed"])
    eols = [float(eol_of[t]["eol_idx"]) for t in cand]
    med = float(np.median(eols))
    best = min(cand, key=lambda t: (abs(float(eol_of[t]["eol_idx"]) - med), t))
    return best, {
        "rule": ("primary test set, event-observed only, EOL closest to the "
                 "median event EOL of the test set, ties broken by smallest "
                 "traj_id"),
        "n_event_candidates": len(cand),
        "median_event_eol_idx": med,
        "selected_eol_idx": float(eol_of[best]["eol_idx"]),
        "selected_eol_years": float(eol_of[best]["eol_years"]),
        "manual_selection": False,
    }


def figure1(idx: dict) -> dict:
    z = np.load(RAW, allow_pickle=True)
    audit = json.loads(B18_AUDIT.read_text(encoding="utf-8"))
    eol_of = {int(r["traj_id"]): r for r in audit["per_trajectory"]}
    seed = select_seed(idx)
    base = f"n{PRIMARY_LEVEL}_target_only_s{seed}__"
    tids = z[base + "tids"]
    tid, rule = select_trajectory(tids, eol_of)
    rule["seed"] = seed
    rule["seed_rule"] = ("median seed by PRIMARY-level target_only "
                        "info macro RMSE (not the best seed)")
    rule["traj_id"] = tid

    m = tids == tid
    tau = z[base + "tau"][m]
    eol_idx = float(eol_of[tid]["eol_idx"])
    rul_scale = 18408.599609375        # B6 冻结 rul_scale (samples)
    smp = tau * eol_idx                                  # 采样点索引
    t_yr = smp * YR_PER_SAMPLE
    eol_yr = eol_idx * YR_PER_SAMPLE

    with h5py.File(FEAT, "r") as f:
        g = f[f"traj_{tid:03d}"]
        hi = g["hi_damage_obs"][:]
        n_h5 = len(hi)
    si = np.clip(np.round(smp).astype(int), 0, n_h5 - 1)
    hi_s = hi[si]

    true_rul = z[base + "true"][m] * rul_scale * YR_PER_SAMPLE
    preds = {}
    for grp in ("damage_extrapolation", "target_only", "source_mmd_finetune"):
        key = f"n{PRIMARY_LEVEL}_{grp}_s{seed}__"
        if key + "pred" not in z.files:
            continue                       # 没有冻结逐点预测就略去, 绝不重训
        gt = z[key + "tids"]
        preds[grp] = z[key + "pred"][gt == tid] * rul_scale * YR_PER_SAMPLE
    rule["methods_plotted"] = sorted(preds)
    rule["methods_skipped_no_frozen_pointwise_prediction"] = [
        g for g in ("source_finetune",) if g not in preds]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.0, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.25], "hspace": 0.12})

    ax1.plot(t_yr, hi_s, color="#3b3b3b", lw=1.5,
             label="observable health index  $HI_{D,obs}$ (telemetry-derived)")
    ax1.axhline(1.0, color=COLOR["true"], ls=":", lw=1.1,
                label="project-defined failure state  $D \\geq 1$")
    ax1.axvline(eol_yr, color=COLOR["true"], ls="--", lw=1.1)
    ax1.annotate(f"EOL = {eol_yr:.2f} y", xy=(eol_yr, 0.06),
                 xytext=(-6, 0), textcoords="offset points",
                 ha="right", va="bottom", fontsize=8, color=COLOR["true"])
    ax1.set_ylabel("health index / damage proxy [-]")
    ax1.set_ylim(-0.03, 1.18)
    ax1.set_title(
        f"Figure 1  Frozen test trajectory traj_{tid:03d} "
        f"(PRIMARY level $n_{{event}}=5$, seed {seed})", loc="left")
    ax1.legend(loc="upper left", frameon=False)

    ax2.plot(t_yr, true_rul, color=COLOR["true"], lw=1.8,
             label="true RUL (right-censored points excluded where undefined)")
    for grp in ("damage_extrapolation", "target_only", "source_mmd_finetune"):
        if grp not in preds:
            continue
        ax2.plot(t_yr, preds[grp], color=COLOR[grp], lw=1.2,
                 ls="-" if grp == "damage_extrapolation" else "--",
                 label=LABEL[grp])
    ax2.axvline(eol_yr, color=COLOR["true"], ls="--", lw=1.1)
    ax2.set_xlabel("mission time [years]")
    ax2.set_ylabel("remaining useful life [years]")
    ax2.legend(loc="upper right", frameon=False)
    ax2.set_xlim(0, t_yr.max())

    cap = ("Basilisk provides the mission / load profile (attitude dynamics, "
           "controller, wheel speed, command torque, mission mode). "
           "The cumulative degradation model $dD/dt = g_{duty}\\,a_T(T)/L_{ref}$, "
           "the failure state $D \\geq 1$ and all RUL labels are defined by "
           "this project, not by Basilisk. $D \\geq 1$ is a project-defined "
           "simulated failure state, not a manufacturer hardware failure "
           "specification. Curves are read from frozen B6 predictions; "
           "no model was retrained for this figure.")
    fig.text(0.005, -0.055, cap, fontsize=7.0, va="top", wrap=True)
    save(fig, "fig1_basilisk_health_trajectory")
    return rule


# ==========================================================================
# Figure 2 —— §7 配对增益 vs 失效标签预算
# ==========================================================================
def figure2(idx: dict) -> dict:
    fc = idx["facts"]["C_b6_label_scarcity_matrix"]
    x = np.array(LEVELS, dtype=float)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    off = {"gain_ft": -0.06, "gain_mmd": 0.06}
    for key, grp in (("gain_ft", "source_finetune"),
                     ("gain_mmd", "source_mmd_finetune")):
        mean, lo, hi = [], [], []
        for n in LEVELS:
            gi = fc["by_level"][str(n)]["gain"][key]
            mean.append(gi["mean"])
            lo.append(gi["mean"] - gi["ci95_lower"])
            hi.append(gi["ci95_upper"] - gi["mean"])
        xs = x * (1.0 + off[key])
        ax.errorbar(xs, mean, yerr=[lo, hi], color=COLOR[grp], marker="o",
                    ms=4.5, lw=1.4, capsize=3.0, elinewidth=1.0,
                    label=f"{grp}  (frozen 95% paired bootstrap CI)")
    ax.axhline(0.0, color="#444444", lw=1.0, ls="-")
    ax.axvline(PRIMARY_LEVEL, color="#7a3f9d", lw=1.1, ls="-.")
    ax.annotate("PRIMARY\n$n_{event}=5$", xy=(PRIMARY_LEVEL, ax.get_ylim()[1]),
                xytext=(4, -4), textcoords="offset points", ha="left",
                va="top", fontsize=8, color="#7a3f9d")
    ax.set_xscale("log")
    ax.set_xticks(LEVELS)
    ax.set_xticklabels([str(n) for n in LEVELS])
    ax.set_xlabel("number of event-observed failure-labelled training "
                  "trajectories  $n_{event}$")
    ax.set_ylabel("paired RMSE gain\n(target_only $-$ transfer)")
    ax.set_title("Figure 2  Paired transfer gain versus failure-label budget",
                 loc="left")
    ax.legend(loc="lower left", frameon=False)

    cap = ("Positive value favors transfer. Error bars are the 95% paired "
           "bootstrap intervals frozen in BASILISK-B6 (one paired difference "
           "per seed, never per time point); no bootstrap was re-run in B7. "
           "PRIMARY was fixed at $n_{event}=5$ before any B6 result was seen; "
           "the other three budgets are SECONDARY and descriptive only, and "
           "no trend line is fitted. Although the MMD curve is above zero at "
           "$n_{event}=5$, no primary positive-transfer verdict was obtained: "
           "only 3/5 seeds improved and the pre-registered gate set was not "
           "fully passed.")
    fig.text(0.005, -0.10, cap, fontsize=7.0, va="top", wrap=True)
    save(fig, "fig2_transfer_gain_vs_labels")
    return {"x_values": LEVELS, "ci_source": "frozen B6 bootstrap",
            "bootstrap_rerun": False, "trend_line_fitted": False}


# ==========================================================================
# Figure 3 —— §8 PRIMARY 档方法 × 寿命分箱
# ==========================================================================
def figure3(idx: dict) -> dict:
    fe = idx["facts"]["E_b6_lifetime_bin_metrics"]
    lb = fe["by_level"][str(PRIMARY_LEVEL)]["gain_by_bin"]
    methods = ["damage_extrapolation", "target_only",
               "source_mmd_finetune", "source_finetune"]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(8.2, 3.9), gridspec_kw={"width_ratios": [1.55, 1.0]})
    w = 0.2
    xs = np.arange(len(BINS))
    for i, g in enumerate(methods):
        vals = [lb[b].get(g) for b in BINS]
        ax.bar(xs + (i - 1.5) * w, vals, w, color=COLOR[g], label=LABEL[g],
               edgecolor="none")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{b}\n(n={lb[b]['n_traj_in_bin']})" for b in BINS])
    ax.set_ylabel("test info-zone trajectory-macro RMSE")
    ax.set_xlabel("lifetime bin (frozen B2.1 event-EOL tertiles)")
    ax.set_ylim(0, None)                 # 完整比例, 绝不截断 y 轴
    ax.set_title("(a) full scale — all four methods", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=7.2)

    learn = ["target_only", "source_mmd_finetune", "source_finetune"]
    for i, g in enumerate(learn):
        vals = [lb[b].get(g) for b in BINS]
        ax2.bar(xs + (i - 1.0) * 0.26, vals, 0.26, color=COLOR[g],
                label=LABEL[g], edgecolor="none")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(BINS)
    ax2.set_xlabel("lifetime bin")
    ax2.set_ylim(0, None)
    ax2.set_title("(b) learning methods only — auxiliary view", loc="left")
    ax2.legend(loc="lower right", frameon=False, fontsize=7.2)

    fig.suptitle("Figure 3  PRIMARY level ($n_{event}=5$): method comparison "
                 "resolved by lifetime bin", x=0.005, ha="left", y=1.04,
                 fontsize=10)
    cap = ("Panel (a) keeps the full y-axis scale: the physics baseline "
           "damage_extrapolation is roughly six times more accurate than every "
           "learning method, and that gap is shown honestly rather than hidden "
           "by axis truncation. Panel (b) is an auxiliary view of the learning "
           "methods alone; it also starts at zero. The differences among the "
           "three learning methods are small relative to their distance from "
           "the physics baseline.")
    fig.text(0.005, -0.10, cap, fontsize=7.0, va="top", wrap=True)
    save(fig, "fig3_primary_method_comparison")
    return {"y_axis_truncated": False, "bins": BINS,
            "methods": methods, "auxiliary_panel_starts_at_zero": True}


# ==========================================================================
# Figure 4 —— §9 健康管理流程
# ==========================================================================
def _box(ax, x, y, w, h, text, fc="#eef3f8", ec="#3a5a78", fs=7.2, bold=False):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.9, facecolor=fc, edgecolor=ec))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2, ls="-", color="#3a5a78"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9,
                                linestyle=ls, shrinkA=0, shrinkB=0))


def figure4() -> dict:
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    chain = [
        "Basilisk mission modes\n(5 modes: nadir / slew / safe / ...)",
        "reaction-wheel speed $\\omega$ and command torque $T_{cmd}$\n"
        "(Basilisk, 1 s resolution)",
        "30-min aggregation window\n(degradation time base)",
        "telemetry: motor current, wheel speed,\ntemperature, command",
        "self-calibrated observable features $x_T$\n"
        "(no hidden $D$, $b_{true}$ or EOL)",
        "observable degradation health index $HI_{D,obs}$",
        "censor-aware RUL estimation\n(right-censored trajectories kept)",
        "uncertainty and prognostic warning metrics\n"
        "(coverage, miss rate, false alarm, PH)",
        "maintenance / health-management decision",
    ]
    x0, w = 0.045, 0.50
    h, gap = 0.083, 0.026
    top = 0.955
    ys = []
    for i, txt in enumerate(chain):
        y = top - (i + 1) * h - i * gap
        ys.append(y)
        fc = "#eef3f8"
        if i == 0:
            fc = "#e6efe6"
        if i == len(chain) - 1:
            fc = "#f6ecd9"
        _box(ax, x0, y, w, h, txt, fc=fc)
        if i:
            _arrow(ax, x0 + w / 2, ys[i - 1], x0 + w / 2, y + h)

    ax.text(x0, top + 0.012, "TARGET DOMAIN  —  LEO reaction wheel",
            fontsize=8.4, fontweight="bold", color="#24384a")

    # 独立的源域逻辑框
    sx, sw = 0.625, 0.335
    src = [
        "SOURCE DOMAIN\nXJTU-SY bearing degradation",
        "encoder pretraining\n(source degradation dynamics)",
        "target fine-tuning /\nMMD adaptation",
    ]
    sy = []
    for i, txt in enumerate(src):
        y = top - (i + 1) * 0.10 - i * 0.035
        sy.append(y)
        _box(ax, sx, y, sw, 0.10, txt, fc="#f2eef7", ec="#6b4a86",
             bold=(i == 0))
        if i:
            _arrow(ax, sx + sw / 2, sy[i - 1], sx + sw / 2, y + 0.10,
                   color="#6b4a86")
    # 迁移只作用在 HI / 退化动力学层
    _arrow(ax, sx, sy[-1] + 0.05, x0 + w + 0.004, ys[6] + h / 2,
           ls="--", color="#6b4a86")
    ax.text(sx - 0.012, sy[-1] - 0.028,
            "transfer acts on the health-index /\n"
            "degradation-dynamics level only,\nnever on raw waveforms",
            fontsize=7.0, ha="left", va="top", color="#6b4a86")

    ax.add_patch(mpatches.FancyBboxPatch(
        (sx - 0.012, 0.245), sw + 0.024, 0.155,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=0.9, facecolor="#fbf0ef", edgecolor="#b3261e"))
    ax.text(sx + sw / 2, 0.3225,
            "The formal study did not establish\npositive transfer.\n"
            "Engineering recommendation:\ndamage_extrapolation.",
            ha="center", va="center", fontsize=7.4, color="#8a1f18",
            linespacing=1.4)

    ax.text(0.005, 0.055,
            "Figure 4  Flywheel health-management pipeline. Basilisk supplies "
            "the mission and load profile only; the cumulative degradation "
            "model, the failure state $D \\geq 1$ and all RUL labels are "
            "project-defined. The source-domain branch is drawn as a separate "
            "logical block because it was evaluated but did not improve "
            "accuracy under the frozen confirmatory protocol. Deployment does "
            "not require Basilisk at runtime: a committed mission-profile "
            "library is used instead.",
            fontsize=7.0, va="top", ha="left", wrap=True)
    save(fig, "fig4_health_management_flow")
    return {"claims_transfer_improves_accuracy": False,
            "source_domain_drawn_as_separate_block": True}


# ==========================================================================
# Figure 5 —— §10 (可选) 划分覆盖诊断
# ==========================================================================
def _split_eols(splits: dict, eol_of: dict) -> dict:
    out = {}
    for name in ("train", "val", "test"):
        tids = [int(t.split("_")[1]) for t in splits[name]["tids"]]
        out[name] = sorted(float(eol_of[t]["eol_idx"]) * YR_PER_SAMPLE
                           for t in tids if eol_of[t]["event_observed"])
    return out


def figure5(idx: dict) -> dict:
    audit = json.loads(B18_AUDIT.read_text(encoding="utf-8"))
    eol_of = {int(r["traj_id"]): r for r in audit["per_trajectory"]}
    b2 = json.loads(B2_SPLIT.read_text(encoding="utf-8"))
    b21 = json.loads(B21_SPLIT.read_text(encoding="utf-8"))
    e2 = _split_eols(b2["splits"], eol_of)
    e21 = _split_eols(b21["splits"], eol_of)
    edges = [float(x) * YR_PER_SAMPLE
             for x in b21["lifetime_bins"]["event"]["edges"]]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
    names = ["train", "val", "test"]
    ypos = {"train": 2.0, "val": 1.0, "test": 0.0}
    for ax, data, ttl, verdict in (
            (axes[0], e2, "(a) B2 random split", "B2_GENERALIZATION_FAIL"),
            (axes[1], e21, "(b) B2.1 coverage-aware split",
             "B21_GENERALIZATION_PASS")):
        for nm in names:
            v = data[nm]
            ax.scatter(v, np.full(len(v), ypos[nm]), s=13, alpha=0.8,
                       color={"train": "#1f6fb4", "val": "#d95f02",
                              "test": "#118a5a"}[nm], edgecolors="none")
            ax.scatter([min(v)], [ypos[nm]], s=52, facecolors="none",
                       edgecolors="#b3261e", linewidths=1.1)
        for e in edges:
            ax.axvline(e, color="#8c8c8c", ls=":", lw=0.9)
        tr_min, te_min = min(data["train"]), min(data["test"])
        if te_min < tr_min:
            ax.axvspan(te_min, tr_min, color="#b3261e", alpha=0.12)
            ax.annotate("test short-life tail\nnot covered by train",
                        xy=(te_min, 0.45), xytext=(6, 0),
                        textcoords="offset points", fontsize=7.0,
                        color="#8a1f18", va="center")
        ax.set_yticks([ypos[n] for n in names])
        ax.set_yticklabels(names)
        ax.set_xlabel("event-observed EOL [years]")
        ax.set_title(f"{ttl}\n{verdict}", loc="left", fontsize=9)
    axes[0].set_ylim(-0.6, 2.6)
    fig.suptitle("Figure 5  Split coverage diagnostic (optional): "
                 "event-EOL support per split",
                 x=0.005, ha="left", y=1.05, fontsize=10)
    cap = ("Red circles mark the minimum event-observed EOL of each split; "
           "dotted lines are the frozen B2.1 event-EOL tertile edges. In B2 "
           "the test split reaches below the shortest training life, so the "
           "short-life tail is unsupported — that is the diagnostic content of "
           "B2_GENERALIZATION_FAIL. B2.1 covers short / medium / long in every "
           "split and is stable. The two panels use different test splits, so "
           "their RMSE values are NOT a fair algorithmic comparison and none "
           "is shown here.")
    fig.text(0.005, -0.14, cap, fontsize=7.0, va="top", wrap=True)
    save(fig, "fig5_split_coverage_diagnostic")
    return {
        "b2_min_event_eol_years": {k: min(v) for k, v in e2.items()},
        "b21_min_event_eol_years": {k: min(v) for k, v in e21.items()},
        "rmse_compared_across_splits": False,
    }


def main() -> int:
    if not INDEX.exists():
        raise SystemExit("!! 先运行 collect_frozen_results.py")
    idx = json.loads(INDEX.read_text(encoding="utf-8"))
    print("== §6 Figure 1")
    r1 = figure1(idx)
    print("== §7 Figure 2")
    r2 = figure2(idx)
    print("== §8 Figure 3")
    r3 = figure3(idx)
    print("== §9 Figure 4")
    r4 = figure4()
    print("== §10 Figure 5 (optional)")
    r5 = figure5(idx)

    lines = [
        "# B7 figure manifest（§6–§11）",
        "",
        "全部图由 `scripts/basilisk_b7/make_figures.py` 生成，"
        "数据只读自冻结产物，未训练、未重新评估、未重新 bootstrap。",
        "",
        "## 风格约定（§11）",
        "",
        f"- dpi = {DPI}（≥ 300），PNG + SVG 成对输出",
        "- 坐标轴 / 图例 / 图注全英文，论文风格，无花哨主题、无 3D、无渐变背景",
        "- y 轴不截断（Figure 3 主图与辅图都从 0 起）",
        "- 置信区间一律标明 95%，且来自 B6 冻结 bootstrap",
        "- exploratory（B3X / B4X）结果不进入任何最终图",
        "",
        "## 产物与哈希",
        "",
        "| file | sha256 |",
        "|---|---|",
    ]
    for p, s in _saved:
        lines.append(f"| `{p}` | `{s[:16]}…` |")
    lines += [
        "",
        "## Figure 1 —— 轨迹与 seed 的确定性选择规则（§6）",
        "",
        f"- seed 规则：{r1['seed_rule']} → seed = **{r1['seed']}**",
        f"- 轨迹规则：{r1['rule']} → **traj_{r1['traj_id']:03d}**",
        f"- test 中 event-observed 候选数 = {r1['n_event_candidates']}，"
        f"其 EOL 中位数 = {r1['median_event_eol_idx']:.0f} samples，"
        f"所选轨迹 EOL = {r1['selected_eol_idx']:.0f} samples "
        f"（{r1['selected_eol_years']:.3f} 年）",
        f"- `manual_selection = {str(r1['manual_selection']).lower()}`",
        f"- 绘出的方法：{r1['methods_plotted']}",
        f"- 因无冻结逐点预测而略去的方法："
        f"{r1['methods_skipped_no_frozen_pointwise_prediction']}"
        "（**不为出图而重训**）",
        "",
        "## Figure 2（§7）",
        "",
        f"- x = {r2['x_values']}（event-observed 失效标签轨迹数）",
        "- y = paired RMSE gain = target_only − transfer，"
        "图注写明 “Positive value favors transfer.”",
        f"- 误差棒来源：{r2['ci_source']}，"
        f"`bootstrap_rerun = {str(r2['bootstrap_rerun']).lower()}`",
        f"- `trend_line_fitted = {str(r2['trend_line_fitted']).lower()}`",
        "- 画 y=0 参考线，并用垂直线 + 注释突出 n=5 PRIMARY",
        "- 图注明确写出 “no primary positive-transfer verdict was obtained.”",
        "",
        "## Figure 3（§8）",
        "",
        f"- 方法：{r3['methods']}；分箱：{r3['bins']}",
        f"- `y_axis_truncated = {str(r3['y_axis_truncated']).lower()}` "
        "——主图保留完整比例，诚实显示 physics baseline 与学习方法的差距",
        f"- 辅图（learning methods only）同样从 0 起："
        f"{str(r3['auxiliary_panel_starts_at_zero']).lower()}",
        "",
        "## Figure 4（§9）",
        "",
        "- 目标域主链：Basilisk mission modes → wheel speed / command torque "
        "→ 30-min aggregation → telemetry → self-calibrated features → "
        "observable HI → censor-aware RUL → warning metrics → "
        "maintenance decision",
        "- 源域为**独立逻辑框**：XJTU-SY → encoder pretraining → "
        "fine-tuning / MMD adaptation",
        f"- `claims_transfer_improves_accuracy = "
        f"{str(r4['claims_transfer_improves_accuracy']).lower()}`；"
        "输出处标注 “The formal study did not establish positive transfer.”",
        "- 标注部署不需要 Basilisk runtime，使用已提交的 mission-profile library",
        "",
        "## Figure 5（§10，可选但已实现）",
        "",
        f"- B2 各 split 最小 event EOL（年）：{r5['b2_min_event_eol_years']}",
        f"- B2.1 各 split 最小 event EOL（年）：{r5['b21_min_event_eol_years']}",
        f"- `rmse_compared_across_splits = "
        f"{str(r5['rmse_compared_across_splits']).lower()}` "
        "—— B2 与 B2.1 的 test split 不同，图中不做 RMSE 公平比较",
        "",
    ]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("\n".join(lines), encoding="utf-8")
    print(f">> 已写 {MANIFEST.relative_to(ROOT)}")
    print(f">> §6..§11: 共 {len(_saved)} 个图文件 (PNG+SVG 成对)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
