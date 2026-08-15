#!/usr/bin/env python
"""scripts/basilisk_b6/summarize_matrix.py

BASILISK-B6 §14/§17/§18/§24-7 —— 四档主结果表 + gain-vs-label 趋势 + 文档产出。

产出:
  * checkpoints/basilisk_b6/summary.json —— 四档主表 (五行齐备)、趋势、damage 比较;
  * docs/basilisk_b6/results.md            —— 四档主结果表 (§14 十三项指标 + 分 bin);
  * docs/basilisk_b6/transfer_gain_vs_labels.md —— §17 FT / MMD 趋势表;
  * docs/basilisk_b6/lifetime_bin_results.md    —— §14 short/medium/long;
  * docs/basilisk_b6/warning_results.md         —— §14 warning / PH / alpha-lambda。

§18 纪律: damage_extrapolation **必须在主表内**, 不得降级成脚注; 每档都要报
target−damage / ft−damage / mmd−damage。若 damage 仍明显领先, 原样写出:
"学习模型未超过基于已知累计损伤结构的物理外推基线。"

§17 纪律: 只有 4 个采样点, **不拟合趋势线**, 只描述"是否出现标签越少增益越大"。

本脚本不下 Gate 结论 —— 判定在 final_transfer_verdict.py。

用法:
    python scripts/basilisk_b6/summarize_matrix.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.basilisk_b11.calibrate_degradation import (  # noqa: E402
    load_b11_config as load_cfg,
)

NAN = float("nan")
INVALID = "B6_INVALID"

MAIN_ROWS = ("target_only", "source_finetune", "source_mmd_finetune",
             "const_mean_info", "damage_extrapolation")
TRANSFER_ROWS = ("source_finetune", "source_mmd_finetune")
SHORT = {"target_only": "target_only", "source_finetune": "source_ft",
         "source_mmd_finetune": "source_mmd", "const_mean_info": "const_mean",
         "damage_extrapolation": "damage_extrap"}

SUMMARY_PURPOSE = (
    "B6 §14: 在 B2.1 冻结划分上, 沿 **失效标签稀缺** 轴 "
    "(n_event_labeled ∈ {3,5,10,21}, 每档始终保留全部 24 条 censored) "
    "给出四档正式主结果表, 并描述 paired gain 随标签数量的变化。"
    "稀缺轴是 event-observed failure label 数量, **不是 target 轨迹总数** —— "
    "真实航天约束是失效标签稀缺, 退化 / 未失效运行数据仍可大量存在。"
)

DAMAGE_PHRASING = "学习模型未超过基于已知累计损伤结构的物理外推基线。"

DAMAGE_DISCIPLINE = (
    "§18: damage_extrapolation 留在主表内, 不降级为脚注。"
    "不得因为赛题主题叫\"迁移学习\"就隐藏物理外推基线领先这一事实。"
)

TREND_DISCIPLINE = (
    "§17: x = n_event_labeled (3,5,10,21), y = paired gain, FT 与 MMD 分开。"
    "只有 4 个采样点 —— **不拟合趋势线**, 只做描述性判断: "
    "是否出现\"标签越少增益越大\"的迹象。"
    "四档之间的 gain 不可直接比较显著性 (训练集不同, 尺度不同), "
    "档间只看方向与量级。"
)

NESTED_NOTE = (
    "四档标签子集是**嵌套**的 (n=3 ⊂ n=5 ⊂ n=10 ⊂ n=21): 每个寿命 bin 只洗牌一次, "
    "各档取前 k 条。这样趋势不会被\"换了一组不同轨迹\"混淆。"
)

SUBSET_NOTE = (
    "移除 event 轨迹会同时改变该档的 rul_scale 与特征 z-score (二者都由 train 行统计"
    "得出) —— 这是标签预算的直接后果, 不是公平性破坏: "
    "同一 cell 内三种方法共享同一份 rul_scale / mu / sd。"
)

CONST_NOTE = (
    "const_mean_info 是标尺不是候选方法: 它只回答\"这个模型是否学到了任何东西\"。"
    "它被列入 excluded_oracle_methods, 不参与工程推荐排名。"
)


def _f(x) -> float:
    if x is None:
        return NAN
    x = float(x)
    return x if np.isfinite(x) else NAN


def _fmt(x, nd: int = 6, sign: bool = False) -> str:
    v = _f(x)
    if not np.isfinite(v):
        return "NaN"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _int(x) -> str:
    return "—" if x is None else str(int(x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel_basilisk_b6.yaml")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b6 = cfg["b6"]
    P = cfg["paths"]
    proto = json.loads((ROOT / cfg["protocol"]["hash_path"]).read_text("utf-8"))

    M = json.loads((ROOT / P["metrics_json"]).read_text("utf-8"))
    S = json.loads((ROOT / P["paired_json"]).read_text("utf-8"))
    L = json.loads((ROOT / P["lifetime_json"]).read_text("utf-8"))
    W = json.loads((ROOT / P["warning_json"]).read_text("utf-8"))
    if bool(M.get("fast")):
        raise SystemExit("!! metrics 来自 --fast 冒烟跑, 不得进入正式汇总")
    for nm, blk in (("paired", S), ("lifetime", L), ("warning", W)):
        for k in ("protocol_sha256", "subset_manifest_sha256", "split_sha256"):
            if str(blk[k]) != str(M[k]):
                raise SystemExit(f"!! {nm} 的 {k} 与 metrics 不一致 —— {INVALID}")

    levels = [int(x) for x in M["label_levels"]]
    primary = int(M["primary_level"])
    if primary != int(proto["primary"]["n_event_labeled"]):
        raise SystemExit(f"!! primary 档与冻结件不符 —— {INVALID}")
    bins = list(L["bins"]["names"])

    # ---------------- §14 四档主表 (五行齐备) ----------------
    tables: dict[str, dict] = {}
    for n in levels:
        s = S["by_level"][str(n)]
        wbg = W["by_level"][str(n)]["by_group"]
        lbg = L["by_level"][str(n)]["by_group"]
        agg = s["aggregate_rmse"]
        rows: dict[str, dict] = {}
        for g in MAIN_ROWS:
            rows[g] = {
                "info_macro_rmse": _f(agg[g]["mean"]),
                "info_macro_rmse_std": _f(agg[g]["std"]),
                "info_macro_rmse_per_seed":
                    [_f(p[g]) for p in s["per_seed"]],
                "info_pooled_rmse":
                    _f(np.mean([_f(p["info_pooled_rmse"][g])
                                for p in s["per_seed"]])),
                "full_macro_rmse":
                    _f(np.nanmean([_f(p["full_macro_rmse"][g])
                                   for p in s["per_seed"]])
                       if any(np.isfinite(_f(p["full_macro_rmse"][g]))
                              for p in s["per_seed"]) else NAN),
                "mae": _f(np.mean([_f(p["mae"][g]) for p in s["per_seed"]])),
                "macro_corr": _f(s["corr_mean_by_group"][g]),
                "psr": _f(s["output_stability"][g]["psr_mean"]),
                "pred_std": _f(s["output_stability"][g]["pred_std_mean"]),
                "true_std": _f(s["output_stability"][g]["true_std_mean"]),
                "high_variance_warning":
                    bool(s["output_stability"][g]["high_variance_warning"]),
                "catastrophic_rate":
                    _f(s["catastrophic_rate_mean_by_group"][g]),
                "warning_coverage": _f(wbg[g]["warning_coverage"]["mean"]),
                "miss_rate": _f(wbg[g]["miss_rate"]["mean"]),
                "false_alarm_rate": _f(wbg[g]["false_alarm_rate"]["mean"]),
                "ph": _f(wbg[g]["prognostic_horizon"]["mean"]),
                "ph_n_seeds_evaluable":
                    int(wbg[g]["prognostic_horizon"]["n_seeds_evaluable"]),
                "alpha_lambda": {k: _f(v["accuracy_mean"])
                                 for k, v in wbg[g]["alpha_lambda"].items()},
                "convergence": _f(wbg[g]["convergence"]["macro_mean"]),
                "convergence_late": _f(wbg[g]["convergence"]["late_mean"]),
                "bin_rmse": {b: _f(lbg[g][b]["rmse_mean"]) for b in bins},
                "bin_n_seeds_evaluable":
                    {b: int(lbg[g][b]["n_seeds_evaluable"]) for b in bins},
            }
        dmg = rows["damage_extrapolation"]["info_macro_rmse"]
        vs_damage = {
            "target_minus_damage": _f(agg["target_minus_damage"]["mean"]),
            "ft_minus_damage": _f(agg["ft_minus_damage"]["mean"]),
            "mmd_minus_damage": _f(agg["mmd_minus_damage"]["mean"]),
            "definition": "正数 = 该学习方法 RMSE 更高, 即不如物理外推基线",
            "damage_leads_all_learned": bool(
                all(np.isfinite(_f(agg[k]["mean"])) and _f(agg[k]["mean"]) > 0
                    for k in ("target_minus_damage", "ft_minus_damage",
                              "mmd_minus_damage"))),
            "rmse_damage": dmg,
        }
        beat_const = {
            "ft_minus_const": _f(agg["ft_minus_const"]["mean"]),
            "mmd_minus_const": _f(agg["mmd_minus_const"]["mean"]),
            "ft_beats_const": bool(np.isfinite(_f(agg["ft_minus_const"]["mean"]))
                                   and _f(agg["ft_minus_const"]["mean"]) < 0),
            "mmd_beats_const":
                bool(np.isfinite(_f(agg["mmd_minus_const"]["mean"]))
                     and _f(agg["mmd_minus_const"]["mean"]) < 0),
            "definition": "负数 = 该方法 RMSE 低于 const_mean_info, 即打赢了常数标尺",
        }
        tables[str(n)] = {
            "n_event_labeled": n,
            "role": str(s["role"]),
            "is_primary": bool(n == primary),
            "train_composition": dict(s["train_composition"]),
            "event_bin_counts": dict(s["event_bin_counts"]),
            "formal_seeds": list(s["formal_seeds"]),
            "rows": rows,
            "row_order": list(MAIN_ROWS),
            "gain": {k: {kk: _f(v[kk]) if kk != "improve_count" else int(v[kk])
                         for kk in ("mean", "median", "std", "ci95_lower",
                                    "ci95_upper", "improve_count")}
                     for k, v in s["gain"].items()},
            "vs_damage": vs_damage,
            "vs_const": beat_const,
            "fairness_pass": bool(s["fairness_pass"]),
            "checkpoint_validation_only": bool(s["checkpoint_validation_only"]),
            "catastrophic_threshold_per_seed":
                [_f(p["catastrophic_threshold"]) for p in s["per_seed"]],
            "rul_scale_per_seed": [_f(p["rul_scale"]) for p in s["per_seed"]],
        }

    damage_leads_everywhere = all(
        tables[str(n)]["vs_damage"]["damage_leads_all_learned"] for n in levels)

    # ---------------- §17 趋势 (描述性, 不拟合) ----------------
    trend: dict[str, dict] = {}
    for meth, key in (("source_finetune", "gain_ft"),
                      ("source_mmd_finetune", "gain_mmd")):
        pts = [{"n_event_labeled": n,
                "role": tables[str(n)]["role"],
                "is_primary": tables[str(n)]["is_primary"],
                **tables[str(n)]["gain"][key]} for n in levels]
        means = [p["mean"] for p in pts]
        fin = [m for m in means if np.isfinite(m)]
        asc = [p["mean"] for p in sorted(pts, key=lambda q: q["n_event_labeled"])]
        monotone_dec = bool(len(asc) == len(levels)
                            and all(np.isfinite(x) for x in asc)
                            and all(asc[i] >= asc[i + 1]
                                    for i in range(len(asc) - 1)))
        argmax_n = (int(pts[int(np.argmax(means))]["n_event_labeled"])
                    if fin and len(fin) == len(means) else None)
        trend[meth] = {
            "points": pts,
            "n_levels_mean_positive": int(sum(1 for m in means
                                              if np.isfinite(m) and m > 0)),
            "n_levels": len(levels),
            "largest_mean_gain_at_n": argmax_n,
            "monotone_decreasing_in_labels": monotone_dec,
            "fewer_labels_larger_gain_sign":
                bool(argmax_n is not None and argmax_n == min(levels)),
            "descriptive_only": True,
            "trend_line_fitted": False,
        }

    out = {
        "stage": "BASILISK_B6",
        "label": str(b6["label"]),
        "formal_stage": True,
        "purpose": SUMMARY_PURPOSE,
        "question": str(b6["question"]),
        "scarcity_axis": str(b6["scarcity_axis"]),
        "scarcity_axis_is_not": str(b6["scarcity_axis_is_not"]),
        "nested_subsets_note": NESTED_NOTE,
        "subset_note": SUBSET_NOTE,
        "const_note": CONST_NOTE,
        "damage_discipline": DAMAGE_DISCIPLINE,
        "trend_discipline": TREND_DISCIPLINE,
        "protocol_sha256": str(M["protocol_sha256"]),
        "config_sha256": str(M["config_sha256"]),
        "subset_manifest_sha256": str(M["subset_manifest_sha256"]),
        "split_sha256": str(M["split_sha256"]),
        "input_schema": str(M["input_schema"]),
        "n_features": int(M["n_features"]),
        "formal_seeds": [int(s) for s in M["formal_seeds"]],
        "label_levels": levels,
        "primary_level": primary,
        "secondary_levels": [int(x) for x in M["secondary_levels"]],
        "primary_metric": str(b6["primary_metric"]),
        "lifetime_bins": {"names": bins, "edges": list(L["bins"]["edges"]),
                          "source": str(L["bins"]["source"]),
                          "recomputed_from_b6_test": False,
                          "counts_test": dict(L["bins"]["counts_test"])},
        "main_tables": tables,
        "main_table_rows": list(MAIN_ROWS),
        "damage_in_main_table": True,
        "damage_demoted_to_footnote": False,
        "damage_leads_all_learned_at_every_level": damage_leads_everywhere,
        "mandatory_statement_when_damage_leads":
            (DAMAGE_PHRASING if damage_leads_everywhere else None),
        "gain_vs_labels": trend,
        "frozen_facts": dict(b6["frozen_facts"]),
        "b5_must_not_be_overwritten": True,
        "verdict_not_decided_here":
            "Gate 判定与最终结论在 final_transfer_verdict.py。",
    }
    op = ROOT / P["summary_json"]
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                             default=float) + "\n", encoding="utf-8")
    print(f">> 已写 {op.relative_to(ROOT)}")

    doc = ROOT / P["doc_dir"]
    doc.mkdir(parents=True, exist_ok=True)

    hdr = (f"protocol_sha256 = `{M['protocol_sha256']}`  \n"
           f"subset_manifest_sha256 = `{M['subset_manifest_sha256']}`  \n"
           f"split_sha256 = `{M['split_sha256']}` (B2.1, 未重划)  \n"
           f"输入 schema = `{M['input_schema']}`, n_features = "
           f"{M['n_features']}  \n"
           f"正式 seed = {M['formal_seeds']}  \n"
           f"主指标 = `{b6['primary_metric']}`\n")

    # ---------------- results.md ----------------
    lines = ["# BASILISK-B6 正式结果: 失效标签稀缺矩阵", "", hdr, "",
             f"> {SUMMARY_PURPOSE}", "",
             f"> {NESTED_NOTE}", "",
             f"> {SUBSET_NOTE}", "",
             f"> {DAMAGE_DISCIPLINE}", "",
             f"> {CONST_NOTE}", ""]
    for n in levels:
        T = tables[str(n)]
        tc = T["train_composition"]
        tag = "**PRIMARY**" if T["is_primary"] else "SECONDARY"
        lines += [f"## n_event_labeled = {n} ({tag})", "",
                  f"train = {tc['total']} (event {tc['event']} + censored "
                  f"{tc['censored']}) ｜ val = 31 ｜ test = 74 (val/test 未动)  ",
                  f"event 分 bin = " + ", ".join(
                      f"{b} {T['event_bin_counts'].get(b, 0)}" for b in bins)
                  + f" ｜ role = `{T['role']}`  ",
                  f"rul_scale (逐 seed) = "
                  f"{[round(x, 1) for x in T['rul_scale_per_seed']]}  ",
                  f"catastrophic 阈值 (逐 seed, 该档 target_only validation "
                  f"中位轨迹 RMSE × 2.0) = "
                  f"{[round(x, 6) for x in T['catastrophic_threshold_per_seed']]}",
                  "",
                  "| 方法 | info macro RMSE | ±std | info pooled | full macro | "
                  "MAE | macro corr | PSR | catastrophic | warn cov | miss | "
                  "false alarm | PH | convergence |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for g in MAIN_ROWS:
            r = T["rows"][g]
            lines.append(
                f"| `{SHORT[g]}` | {_fmt(r['info_macro_rmse'])} | "
                f"{_fmt(r['info_macro_rmse_std'])} | "
                f"{_fmt(r['info_pooled_rmse'])} | {_fmt(r['full_macro_rmse'])} | "
                f"{_fmt(r['mae'])} | {_fmt(r['macro_corr'], 4)} | "
                f"{_fmt(r['psr'], 3)} | {_fmt(r['catastrophic_rate'], 4)} | "
                f"{_fmt(r['warning_coverage'], 4)} | {_fmt(r['miss_rate'], 4)} | "
                f"{_fmt(r['false_alarm_rate'], 4)} | {_fmt(r['ph'], 2)} | "
                f"{_fmt(r['convergence'], 4)} |")
        lines += ["",
                  "alpha-lambda accuracy (右删失轨迹不参与, 保持 NaN):", "",
                  "| 方法 | " + " | ".join(
                      f"λ={k}" for k in
                      T["rows"]["target_only"]["alpha_lambda"]) + " |",
                  "|---|" + "---|" * len(T["rows"]["target_only"]["alpha_lambda"])]
        for g in MAIN_ROWS:
            al = T["rows"][g]["alpha_lambda"]
            lines.append(f"| `{SHORT[g]}` | " + " | ".join(
                _fmt(v, 4) for v in al.values()) + " |")
        lines += ["", "分 bin info macro RMSE (bin 取自 B2.1, 未按 B6 test 重算):",
                  "", "| 方法 | " + " | ".join(bins) + " |",
                  "|---|" + "---|" * len(bins)]
        for g in MAIN_ROWS:
            lines.append(f"| `{SHORT[g]}` | " + " | ".join(
                _fmt(T["rows"][g]["bin_rmse"][b]) for b in bins) + " |")
        gf, gm = T["gain"]["gain_ft"], T["gain"]["gain_mmd"]
        lines += ["", "配对增益 (gain = target_only − source 方法, 正数 = source 更好):",
                  "",
                  "| 方法 | mean | median | std | CI95 下界 | CI95 上界 | improve |",
                  "|---|---|---|---|---|---|---|",
                  f"| `source_ft` | {_fmt(gf['mean'], 6, True)} | "
                  f"{_fmt(gf['median'], 6, True)} | {_fmt(gf['std'])} | "
                  f"{_fmt(gf['ci95_lower'], 6, True)} | "
                  f"{_fmt(gf['ci95_upper'], 6, True)} | "
                  f"{gf['improve_count']}/5 |",
                  f"| `source_mmd` | {_fmt(gm['mean'], 6, True)} | "
                  f"{_fmt(gm['median'], 6, True)} | {_fmt(gm['std'])} | "
                  f"{_fmt(gm['ci95_lower'], 6, True)} | "
                  f"{_fmt(gm['ci95_upper'], 6, True)} | "
                  f"{gm['improve_count']}/5 |", ""]
        vd, vc = T["vs_damage"], T["vs_const"]
        lines += ["与物理外推基线 / 常数标尺的比较 (§18, 主表内, 非脚注):", "",
                  "| 比较 | 值 | 含义 |", "|---|---|---|",
                  f"| target − damage | {_fmt(vd['target_minus_damage'], 6, True)} "
                  f"| {'target_only 不如物理外推' if vd['target_minus_damage'] > 0 else 'target_only 优于物理外推'} |",
                  f"| ft − damage | {_fmt(vd['ft_minus_damage'], 6, True)} | "
                  f"{'source_ft 不如物理外推' if vd['ft_minus_damage'] > 0 else 'source_ft 优于物理外推'} |",
                  f"| mmd − damage | {_fmt(vd['mmd_minus_damage'], 6, True)} | "
                  f"{'source_mmd 不如物理外推' if vd['mmd_minus_damage'] > 0 else 'source_mmd 优于物理外推'} |",
                  f"| ft − const | {_fmt(vc['ft_minus_const'], 6, True)} | "
                  f"{'打赢常数标尺' if vc['ft_beats_const'] else '未打赢常数标尺'} |",
                  f"| mmd − const | {_fmt(vc['mmd_minus_const'], 6, True)} | "
                  f"{'打赢常数标尺' if vc['mmd_beats_const'] else '未打赢常数标尺'} |",
                  "",
                  f"公平性 (同 cell 内 train/val/test IDs、batch 顺序、初始权重同一): "
                  f"**{'PASS' if T['fairness_pass'] else 'FAIL'}** ｜ "
                  f"checkpoint 选择 = "
                  f"{'validation_only' if T['checkpoint_validation_only'] else 'INVALID'}",
                  ""]
    if damage_leads_everywhere:
        lines += ["## §18 必须原样写出的结论", "", f"> **{DAMAGE_PHRASING}**", "",
                  "四个标签档下 target_only / source_ft / source_mmd 的 test "
                  "information-zone trajectory-macro RMSE 均高于 "
                  "`damage_extrapolation`。这是本阶段的事实陈述, 不因赛题主题是"
                  "迁移学习而被隐藏或弱化。", ""]
    (doc / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f">> 已写 {(doc / 'results.md').relative_to(ROOT)}")

    # ---------------- transfer_gain_vs_labels.md ----------------
    lines = ["# BASILISK-B6 §17 迁移增益 vs 失效标签数量", "", hdr, "",
             f"> {TREND_DISCIPLINE}", "", f"> {NESTED_NOTE}", ""]
    for meth, disp in (("source_finetune", "source_finetune (FT)"),
                       ("source_mmd_finetune", "source_mmd_finetune (MMD)")):
        t = trend[meth]
        lines += [f"## {disp}", "",
                  "| n_event_labeled | role | mean gain | median | std | "
                  "CI95 下界 | CI95 上界 | improve |",
                  "|---|---|---|---|---|---|---|---|"]
        for p in t["points"]:
            tag = "**PRIMARY**" if p["is_primary"] else "secondary"
            lines.append(
                f"| {p['n_event_labeled']} | {tag} | "
                f"{_fmt(p['mean'], 6, True)} | {_fmt(p['median'], 6, True)} | "
                f"{_fmt(p['std'])} | {_fmt(p['ci95_lower'], 6, True)} | "
                f"{_fmt(p['ci95_upper'], 6, True)} | {p['improve_count']}/5 |")
        lines += ["",
                  f"- mean gain > 0 的档数: {t['n_levels_mean_positive']}/"
                  f"{t['n_levels']}",
                  f"- mean gain 最大的档: n = {t['largest_mean_gain_at_n']}",
                  f"- 随标签数量单调递减: {t['monotone_decreasing_in_labels']}",
                  f"- 出现\"标签越少增益越大\"迹象: "
                  f"{t['fewer_labels_larger_gain_sign']}",
                  "- 趋势线拟合: **未做** (只有 4 个点, 描述性分析)", ""]
    lines += ["## 解读纪律", "",
              "- 四档的 train 不同, 因此 gain 的**档间比较只看方向与量级**, "
              "不做档间显著性检验。",
              "- SECONDARY 档 (n=3/10/21) 只提供趋势, **不得单独推翻 PRIMARY "
              "(n=5) 判定** (§16)。",
              "- 每档的 CI 都是 5 个 seed 级配对差值的 bootstrap, 只描述训练"
              "随机性下差值方向是否稳定, **不是泛化误差置信区间**。", ""]
    (doc / "transfer_gain_vs_labels.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f">> 已写 {(doc / 'transfer_gain_vs_labels.md').relative_to(ROOT)}")

    # ---------------- lifetime_bin_results.md ----------------
    lines = ["# BASILISK-B6 §14 寿命分 bin 结果", "", hdr, "",
             f"> {L['bin_source_note']}", "", f"> {L['nan_discipline']}", "",
             f"bin 边界 = {L['bins']['edges']} ｜ test 侧计数 = "
             f"{L['bins']['counts_test']} ｜ 四档完全相同 (B6 只改 train)",
             "",
             f"右删失 bin 退化 = {L['censored_bins']['degenerate']} "
             f"(有效 bin 数 {L['censored_bins']['n_effective_bins']}) —— "
             f"如实报告, 不为凑三个 bin 编造边界。", ""]
    for n in levels:
        LB = L["by_level"][str(n)]
        tag = "**PRIMARY**" if LB["is_primary"] else "SECONDARY"
        lines += [f"## n_event_labeled = {n} ({tag})", "",
                  "| 方法 | " + " | ".join(
                      f"{b} RMSE (n_seed)" for b in bins) + " |",
                  "|---|" + "---|" * len(bins)]
        for g in MAIN_ROWS:
            cells = [f"{_fmt(LB['by_group'][g][b]['rmse_mean'])} "
                     f"({LB['by_group'][g][b]['n_seeds_evaluable']})"
                     for b in bins]
            lines.append(f"| `{SHORT[g]}` | " + " | ".join(cells) + " |")
        lines += ["", "逐 bin 配对增益 (正数 = source 方法更好):", "",
                  "| bin | n_traj | gain_ft | ≥0 | gain_mmd | ≥0 | "
                  "damage RMSE |", "|---|---|---|---|---|---|---|"]
        for b in bins:
            gb = LB["gain_by_bin"][b]
            lines.append(
                f"| {b} | {gb['n_traj_in_bin']} | "
                f"{_fmt(gb['gain_ft'], 6, True)} | "
                f"{'是' if gb['gain_ft_nonneg'] else '否'} | "
                f"{_fmt(gb['gain_mmd'], 6, True)} | "
                f"{'是' if gb['gain_mmd_nonneg'] else '否'} | "
                f"{_fmt(gb['damage_extrapolation'])} |")
        nb = LB["n_bins_gain_nonneg"]
        lt = LB["localized_transfer_benefit"]
        lines += ["",
                  f"- gain ≥ 0 的 bin 数: FT {nb['ft']}/{nb['n_bins']} ｜ "
                  f"MMD {nb['mmd']}/{nb['n_bins']} "
                  f"(§15 条件 5 要求 ≥ "
                  f"{proto['gate_thresholds']['min_lifetime_bins_gain_nonneg']})",
                  f"- `{lt['label']}` (仅短寿命 bin 获益): FT {lt['ft']} ｜ "
                  f"MMD {lt['mmd']}",
                  f"- catastrophic 计数 (逐 bin 逐 seed 之和): " + " ｜ ".join(
                      f"{SHORT[g]} " + "/".join(
                          _int(LB['by_group'][g][b]['catastrophic_count_total'])
                          for b in bins) for g in MAIN_ROWS),
                  ""]
    (doc / "lifetime_bin_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f">> 已写 {(doc / 'lifetime_bin_results.md').relative_to(ROOT)}")

    # ---------------- warning_results.md ----------------
    pc = W["prognostics_cfg"]
    lines = ["# BASILISK-B6 §14 报警 / 预测视界结果", "", hdr, "",
             f"> {W['miss_discipline']}", "", f"> {W['nan_discipline']}", "",
             f"evaluator = {W['evaluator']}  ",
             f"rul_threshold = {pc['rul_threshold']} ｜ persistence = "
             f"{pc['persistence']} ｜ alpha = {pc['alpha']} ｜ lambdas = "
             f"{pc['lambdas']} ｜ late_from = {pc['late_from']}", ""]
    for n in levels:
        WB = W["by_level"][str(n)]
        tag = "**PRIMARY**" if WB["is_primary"] else "SECONDARY"
        gi = WB["gate_inputs"]
        lines += [f"## n_event_labeled = {n} ({tag})", "",
                  "| 方法 | warn coverage | miss rate | coverage before EOL | "
                  "miss before EOL | false alarm | PH (n_seed) | "
                  "convergence | late conv | warning_valid |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for g in MAIN_ROWS:
            b = WB["by_group"][g]
            lines.append(
                f"| `{SHORT[g]}` | {_fmt(b['warning_coverage']['mean'], 4)} | "
                f"{_fmt(b['miss_rate']['mean'], 4)} | "
                f"{_fmt(b['coverage_before_eol']['mean'], 4)} | "
                f"{_fmt(b['miss_rate_before_eol']['mean'], 4)} | "
                f"{_fmt(b['false_alarm_rate']['mean'], 4)} | "
                f"{_fmt(b['prognostic_horizon']['mean'], 2)} "
                f"({b['prognostic_horizon']['n_seeds_evaluable']}) | "
                f"{_fmt(b['convergence']['macro_mean'], 4)} | "
                f"{_fmt(b['convergence']['late_mean'], 4)} | "
                f"{b['warning_valid_count']}/{b['n_seeds']} |")
        lines += ["",
                  f"- §15 条件 8 (miss rate 不高于 target_only + "
                  f"{gi['warning_miss_tol']}): "
                  f"target {_fmt(gi['target_only_miss_mean'], 4)} ｜ "
                  f"FT {_fmt(gi['ft_miss_mean'], 4)} "
                  f"({'PASS' if gi['ft_within_tol'] else 'FAIL'}) ｜ "
                  f"MMD {_fmt(gi['mmd_miss_mean'], 4)} "
                  f"({'PASS' if gi['mmd_within_tol'] else 'FAIL'})",
                  "- PH / alpha-lambda 的分母只含 event-observed 轨迹; "
                  "右删失轨迹保持 NaN, 不伪造 EOL。", ""]
    (doc / "warning_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f">> 已写 {(doc / 'warning_results.md').relative_to(ROOT)}")

    # ---------------- 控制台摘要 ----------------
    print(f"\n{'=' * 78}")
    for n in levels:
        T = tables[str(n)]
        tag = "PRIMARY  " if T["is_primary"] else "secondary"
        print(f"[n={n:<2} {tag}] " + "  ".join(
            f"{SHORT[g]} {T['rows'][g]['info_macro_rmse']:.6f}"
            for g in MAIN_ROWS))
        gf, gm = T["gain"]["gain_ft"], T["gain"]["gain_mmd"]
        print(f"{'':>16}gain_ft {gf['mean']:+.6f} (imp {gf['improve_count']}/5) "
              f"｜ gain_mmd {gm['mean']:+.6f} (imp {gm['improve_count']}/5) "
              f"｜ damage {T['vs_damage']['rmse_damage']:.6f}")
    print(f"{'=' * 78}")
    for meth in ("source_finetune", "source_mmd_finetune"):
        t = trend[meth]
        print(f">> §17 {meth}: mean gain > 0 的档 "
              f"{t['n_levels_mean_positive']}/{t['n_levels']}, "
              f"最大增益在 n={t['largest_mean_gain_at_n']}, "
              f"\"标签越少增益越大\"迹象 = {t['fewer_labels_larger_gain_sign']}")
    if damage_leads_everywhere:
        print(f">> §18 必须原样写出: {DAMAGE_PHRASING}")
    print(">> Gate 判定与最终结论在 final_transfer_verdict.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
