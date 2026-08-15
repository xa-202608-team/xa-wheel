#!/usr/bin/env python
"""scripts/basilisk_b7/build_final_tables.py

BASILISK-B7 §5 —— 从 frozen result index 构建四张主表。

**只读**。唯一输入是 checkpoints/basilisk_b7/frozen_result_index.json,
不再触碰任何原始 JSON / 训练器 / evaluator。

Table A: B5 全标签确认性对比
Table B: B6 失效标签稀缺矩阵
Table C: 配对迁移增益 (FT / MMD 分别)
Table D: PRIMARY 档寿命分箱 + 告警指标

输出:
  checkpoints/basilisk_b7/final_tables.json   (机器可读, 供审计脚本比对)
  docs/basilisk_b7/final_tables.md            (markdown 片段, 供 results.md 复用)
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "checkpoints" / "basilisk_b7" / "frozen_result_index.json"
OUT_JSON = ROOT / "checkpoints" / "basilisk_b7" / "final_tables.json"
OUT_MD = ROOT / "docs" / "basilisk_b7" / "final_tables.md"

LEVELS = [3, 5, 10, 21]
PRIMARY_LEVEL = 5
BINS = ["short", "medium", "long"]

# §5 Table A 明确要求的方法顺序
TABLE_A_ORDER = ["damage_extrapolation", "target_only",
                 "source_mmd_finetune", "source_finetune"]
TABLE_B_COLS = ["target_only", "source_finetune", "source_mmd_finetune",
                "const_mean_info", "damage_extrapolation"]
TABLE_D_ROWS = ["damage_extrapolation", "target_only",
                "source_mmd_finetune", "source_finetune"]

DISPLAY = {
    "damage_extrapolation": "damage_extrapolation (physics)",
    "target_only": "target_only",
    "source_mmd_finetune": "source_mmd_finetune",
    "source_finetune": "source_finetune",
    "const_mean_info": "const_mean_info (reference)",
}


def _fmt(v, nd=6, sign=False):
    """NaN / None 一律显示 n/a —— 不伪造 0。"""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def build_table_a(fb: dict) -> dict:
    rows = []
    for g in TABLE_A_ORDER:
        r = fb["rows"][g]
        rows.append({
            "method": g,
            "display": DISPLAY[g],
            "info_macro_rmse": r["info_macro_rmse"],
            "info_macro_rmse_std": r["info_macro_rmse_std"],
            "macro_corr": r["macro_corr"],
            "warning_coverage": r["warning_coverage"],
            "miss_rate": r["miss_rate"],
            "catastrophic_rate": r["catastrophic_rate"],
        })
    return {
        "id": "TableA",
        "title": ("Table A — B5 full-label confirmatory comparison "
                  "(target train labels: 21 event + 24 censored)"),
        "verdict": fb["verdict"],
        "verdict_statement": (
            f"`{fb['verdict']}` —— 在全部目标域训练标签可用的确认性协议下, "
            "两个迁移分组都未通过预登记门槛。"),
        "primary_metric": fb["primary_metric"],
        "formal_seeds": fb["formal_seeds"],
        "n_seeds": len(fb["formal_seeds"]),
        "columns": ["method", "info_macro_rmse", "info_macro_rmse_std",
                    "macro_corr", "warning_coverage", "miss_rate",
                    "catastrophic_rate"],
        "method_order_frozen": TABLE_A_ORDER,
        "rows": rows,
        "gates_passed": {
            m: f"{fb['decision'][m]['n_passed']}/"
               f"{fb['decision'][m]['n_conditions']}"
            for m in ("source_finetune", "source_mmd_finetune")},
        "note": ("const_mean_info 作为参考下界列出, 不参与工程推荐候选集; "
                 "coverage 与 miss 必须成对阅读。"),
    }


def build_table_b(fc: dict) -> dict:
    rows = []
    for n in LEVELS:
        lv = fc["by_level"][str(n)]
        m, s = lv["mean_info_macro_rmse"], lv["std_info_macro_rmse"]
        rows.append({
            "n_event_labeled": n,
            "role": lv["role"],
            "is_primary": lv["is_primary"],
            "train_composition": lv["train_composition"],
            "event_bin_counts": lv["event_bin_counts"],
            "info_macro_rmse": {g: m[g] for g in TABLE_B_COLS},
            "info_macro_rmse_std": {g: s[g] for g in TABLE_B_COLS},
            "best_method": min(TABLE_B_COLS, key=lambda g: m[g]),
        })
    return {
        "id": "TableB",
        "title": ("Table B — B6 failure-label scarcity matrix "
                  "(primary metric: test info-trajectory macro RMSE)"),
        "primary_verdict": fc["primary_verdict"],
        "primary_level": fc["primary_level"],
        "scarcity_axis": fc["scarcity_axis"],
        "scarcity_axis_is_not": fc["scarcity_axis_is_not"],
        "always_keep_all_censored_train": fc["always_keep_all_censored_train"],
        "n_censored_train_always": fc["n_censored_train_always"],
        "formal_seeds": fc["formal_seeds"],
        "columns": TABLE_B_COLS,
        "rows": rows,
        "note": ("稀缺轴是 event-observed 失效标签轨迹数, 不是目标域总轨迹数; "
                 "24 条 right-censored 训练轨迹在每一档都全部保留。"
                 "damage_extrapolation 在四档全部领先所有学习方法。"),
    }


def build_table_c(fc: dict) -> dict:
    blocks = {}
    for key, method in (("gain_ft", "source_finetune"),
                        ("gain_mmd", "source_mmd_finetune")):
        rows = []
        for n in LEVELS:
            lv = fc["by_level"][str(n)]
            gi = lv["gain"][key]
            rows.append({
                "n_event_labeled": n,
                "role": lv["role"],
                "is_primary": lv["is_primary"],
                "mean_gain": gi["mean"],
                "median_gain": gi["median"],
                "improve_seeds": f"{gi['improve_count']}/{gi['n_seeds']}",
                "improve_count": gi["improve_count"],
                "n_seeds": gi["n_seeds"],
                "ci95_lower": gi["ci95_lower"],
                "ci95_upper": gi["ci95_upper"],
                "ci_lower_positive": gi["ci_lower_positive"],
            })
        blocks[method] = {
            "method": method,
            "gain_key": key,
            "rows": rows,
        }
    p5 = fc["by_level"]["5"]["gain"]["gain_mmd"]
    return {
        "id": "TableC",
        "title": ("Table C — paired transfer gain vs failure-label budget "
                  "(gain = target_only RMSE − transfer RMSE; "
                  "positive value favors transfer)"),
        "gain_definition": fc["gain_definition"],
        "ci_level": "95%",
        "ci_unit": "one paired difference per seed (never per time point)",
        "bootstrap_reused_from_frozen_b6": True,
        "bootstrap_rerun_in_b7": False,
        "primary_level": PRIMARY_LEVEL,
        "primary_must_be_visually_highlighted": True,
        "blocks": blocks,
        "primary_mmd_qualified_signal": {
            "n_event_labeled": PRIMARY_LEVEL,
            "method": "source_mmd_finetune",
            "mean_gain": p5["mean"],
            "median_gain": p5["median"],
            "ci95_lower": p5["ci95_lower"],
            "improve_seeds": f"{p5['improve_count']}/{p5['n_seeds']}",
            "qualifier": ("mean / median / CI95-lower 均为正, 但仅 "
                          f"{p5['improve_count']}/{p5['n_seeds']} 个 seed 改善, "
                          "且预登记门槛未全部通过。"),
            "must_not_be_called_significant_positive_transfer": True,
            "gates_passed": (
                f"{fc['primary_decision']['source_mmd_finetune']['n_passed']}/"
                f"{fc['primary_decision']['source_mmd_finetune']['n_conditions']}"
            ),
            "failed_conditions": fc["primary_decision"][
                "source_mmd_finetune"]["failed_conditions"],
        },
        "note": ("PRIMARY = n_event_labeled 5, 在看到任何 B6 结果之前固定; "
                 "其余三档为 SECONDARY, 只用于趋势描述, 不能覆盖 PRIMARY 判定。"),
    }


def build_table_d(fe: dict, fd: dict, fc: dict) -> dict:
    lb = fe["by_level"][str(PRIMARY_LEVEL)]["gain_by_bin"]
    wg = fd["by_level"][str(PRIMARY_LEVEL)]["by_group"]
    rows = []
    for g in TABLE_D_ROWS:
        w = wg[g]
        rows.append({
            "method": g,
            "display": DISPLAY[g],
            "bin_rmse": {b: lb[b].get(g) for b in BINS},
            "bin_n_traj": {b: lb[b]["n_traj_in_bin"] for b in BINS},
            "warning_coverage": w["warning_coverage"],
            "miss_rate": w["miss_rate"],
            "false_alarm_rate": w["false_alarm_rate"],
            "prognostic_horizon": w["prognostic_horizon"],
            "n_seeds_evaluable": w["n_seeds_evaluable"],
        })
    return {
        "id": "TableD",
        "title": ("Table D — PRIMARY level (n_event_labeled = 5): "
                  "lifetime-bin RMSE and prognostic warning metrics"),
        "primary_level": PRIMARY_LEVEL,
        "bin_names": BINS,
        "bin_edges": fe["bin_edges"],
        "bin_source": fe["bin_source"],
        "bin_recomputed_from_b6_test": fe["recomputed_from_b6_test"],
        "bin_counts_test": fe["counts_test"],
        "damage_baseline_in_table": True,
        "damage_baseline_in_footnote_only": False,
        "row_order_frozen": TABLE_D_ROWS,
        "rows": rows,
        "gain_by_bin": {
            b: {"gain_ft": lb[b]["gain_ft"], "gain_mmd": lb[b]["gain_mmd"],
                "gain_ft_nonneg": lb[b]["gain_ft_nonneg"],
                "gain_mmd_nonneg": lb[b]["gain_mmd_nonneg"],
                "n_traj_in_bin": lb[b]["n_traj_in_bin"]}
            for b in BINS},
        "n_bins_gain_nonneg": fe["by_level"][
            str(PRIMARY_LEVEL)]["n_bins_gain_nonneg"],
        "empty_bin_rule": fe["empty_bin_rule"],
        "deployability_note": fd["deployability_note"],
        "note": ("const_mean_info 不参与本表 —— 它没有随时间变化的告警行为; "
                 "PH 只在 event-observed 轨迹上计算, right-censored 轨迹排除, "
                 "不伪造 EOL。damage_extrapolation 必须留在正表内。"),
    }


def md_table_a(t: dict) -> list[str]:
    L = [f"### {t['title']}", "",
         f"判定：**`{t['verdict']}`**。"
         f"主指标 = `{t['primary_metric']}`，"
         f"{t['n_seeds']} 个确认性 seed {t['formal_seeds']}。", "",
         "| method | info macro RMSE | std | macro corr | warning coverage "
         "| miss rate | catastrophic rate |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for r in t["rows"]:
        L.append(
            f"| `{r['display']}` | {_fmt(r['info_macro_rmse'])} | "
            f"{_fmt(r['info_macro_rmse_std'])} | {_fmt(r['macro_corr'], 4)} | "
            f"{_fmt(r['warning_coverage'], 4)} | {_fmt(r['miss_rate'], 4)} | "
            f"{_fmt(r['catastrophic_rate'], 4)} |")
    L += ["",
          f"预登记门槛通过数：`source_finetune` "
          f"{t['gates_passed']['source_finetune']}，`source_mmd_finetune` "
          f"{t['gates_passed']['source_mmd_finetune']}。",
          "",
          f"> {t['note']}", ""]
    return L


def md_table_b(t: dict) -> list[str]:
    L = [f"### {t['title']}", "",
         f"PRIMARY 判定：**`{t['primary_verdict']}`**"
         f"（PRIMARY = n_event_labeled {t['primary_level']}）。", "",
         "| n_event | role | target_only | source_finetune "
         "| source_mmd_finetune | const_mean_info | damage_extrapolation |",
         "|---:|---|---:|---:|---:|---:|---:|"]
    for r in t["rows"]:
        m = r["info_macro_rmse"]
        n = (f"**{r['n_event_labeled']}**" if r["is_primary"]
             else str(r["n_event_labeled"]))
        role = f"**{r['role']}**" if r["is_primary"] else r["role"]
        L.append(
            f"| {n} | {role} | {_fmt(m['target_only'])} | "
            f"{_fmt(m['source_finetune'])} | "
            f"{_fmt(m['source_mmd_finetune'])} | "
            f"{_fmt(m['const_mean_info'])} | "
            f"{_fmt(m['damage_extrapolation'])} |")
    L += ["",
          f"稀缺轴 = `{t['scarcity_axis']}`，"
          f"**不是** `{t['scarcity_axis_is_not']}`；"
          f"每一档都保留全部 {t['n_censored_train_always']} 条 right-censored "
          "训练轨迹。",
          "",
          f"> {t['note']}", ""]
    return L


def md_table_c(t: dict) -> list[str]:
    L = [f"### {t['title']}", "",
         "`gain = RMSE_target_only − RMSE_source_method`，**正数 = 迁移方法更好**。"
         "符号在 protocol 冻结时已固定，不得反转；配对只在**同一标签档、"
         "同一 seed** 内进行。误差区间为 B6 已冻结的 "
         f"{t['ci_level']} 配对 bootstrap（{t['ci_unit']}），"
         "B7 未重新 bootstrap。", ""]
    for method, blk in t["blocks"].items():
        L += [f"**{method}**", "",
              "| n_event | role | mean gain | median gain | improve seeds "
              "| CI95 |",
              "|---:|---|---:|---:|---:|---|"]
        for r in blk["rows"]:
            n = (f"**{r['n_event_labeled']}**" if r["is_primary"]
                 else str(r["n_event_labeled"]))
            role = (f"◀ **{r['role']}**" if r["is_primary"] else r["role"])
            L.append(
                f"| {n} | {role} | {_fmt(r['mean_gain'], 6, True)} "
                f"| {_fmt(r['median_gain'], 6, True)} | {r['improve_seeds']} | "
                f"[{_fmt(r['ci95_lower'], 6, True)}, "
                f"{_fmt(r['ci95_upper'], 6, True)}] |")
        L.append("")
    q = t["primary_mmd_qualified_signal"]
    L += [
        "**PRIMARY 档 MMD 的条件性信号（必须带限定词阅读）**",
        "",
        f"在 n_event_labeled = {q['n_event_labeled']} 这一 PRIMARY 档，"
        f"`source_mmd_finetune` 的 mean gain = {_fmt(q['mean_gain'], 6, True)}、"
        f"median gain = {_fmt(q['median_gain'], 6, True)}、"
        f"CI95 lower = {_fmt(q['ci95_lower'], 6, True)} 均为正；"
        f"**但** 仅 {q['improve_seeds']} 个 seed 改善，"
        f"预登记门槛通过 {q['gates_passed']}"
        f"（未通过条件编号 {q['failed_conditions']}）。"
        "该结果只能表述为条件性低标签信号，"
        "**不得**表述为显著正迁移。",
        "",
        f"> {t['note']}", ""]
    return L


def md_table_d(t: dict) -> list[str]:
    L = [f"### {t['title']}", "",
         f"寿命分箱边界 `{t['bin_edges']}` 来自 `{t['bin_source']}`，"
         f"`recomputed_from_b6_test = "
         f"{str(t['bin_recomputed_from_b6_test']).lower()}`；"
         f"test 各箱轨迹数 {t['bin_counts_test']}。", "",
         "| method | RMSE short | RMSE medium | RMSE long "
         "| warning coverage | miss rate | false alarm | PH |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in t["rows"]:
        b = r["bin_rmse"]
        L.append(
            f"| `{r['display']}` | {_fmt(b['short'])} | {_fmt(b['medium'])} | "
            f"{_fmt(b['long'])} | {_fmt(r['warning_coverage'], 4)} | "
            f"{_fmt(r['miss_rate'], 4)} | "
            f"{_fmt(r['false_alarm_rate'], 4)} | "
            f"{_fmt(r['prognostic_horizon'], 4)} |")
    nb = t["n_bins_gain_nonneg"]
    L += ["",
          f"分箱增益非负的箱数：`source_finetune` {nb['ft']}/{nb['n_bins']}，"
          f"`source_mmd_finetune` {nb['mmd']}/{nb['n_bins']}。",
          "",
          f"> {t['deployability_note']}",
          f">",
          f"> {t['note']}", ""]
    return L


def main() -> int:
    if not INDEX.exists():
        raise SystemExit("!! 先运行 collect_frozen_results.py")
    idx = json.loads(INDEX.read_text(encoding="utf-8"))
    f = idx["facts"]
    ta = build_table_a(f["B_b5_full_label_formal"])
    tb = build_table_b(f["C_b6_label_scarcity_matrix"])
    tc = build_table_c(f["C_b6_label_scarcity_matrix"])
    td = build_table_d(f["E_b6_lifetime_bin_metrics"],
                       f["D_b6_warning_metrics"],
                       f["C_b6_label_scarcity_matrix"])

    tables = {
        "stage": "BASILISK_B7",
        "section": "§5 final tables",
        "built_from": "checkpoints/basilisk_b7/frozen_result_index.json",
        "frozen_result_index_sha256": hashlib.sha256(
            INDEX.read_bytes()).hexdigest(),
        "read_only": True,
        "recomputed_any_metric": False,
        "all_numbers_from_frozen_json": True,
        "FINAL_TRANSFER_CONCLUSION": f["G_final_transfer_conclusion"][
            "FINAL_TRANSFER_CONCLUSION"],
        "ENGINEERING_RECOMMENDATION": f["F_final_engineering_recommendation"][
            "ENGINEERING_RECOMMENDATION"],
        "tables": {"TableA": ta, "TableB": tb, "TableC": tc, "TableD": td},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(tables, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f">> 已写 {OUT_JSON.relative_to(ROOT)}")
    print(">> final_tables_sha256 = "
          f"{hashlib.sha256(OUT_JSON.read_bytes()).hexdigest()}")

    lines = [
        "# B7 四张主表（§5）",
        "",
        "所有数字来自 "
        "`checkpoints/basilisk_b7/frozen_result_index.json`"
        f"（`sha256 = {tables['frozen_result_index_sha256'][:16]}…`），"
        "B7 未重算任何指标。",
        "",
        f"- `FINAL_TRANSFER_CONCLUSION = "
        f"{tables['FINAL_TRANSFER_CONCLUSION']}`",
        f"- `ENGINEERING_RECOMMENDATION = "
        f"{tables['ENGINEERING_RECOMMENDATION']}`",
        "",
    ]
    lines += md_table_a(ta) + md_table_b(tb) + md_table_c(tc) + md_table_d(td)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f">> 已写 {OUT_MD.relative_to(ROOT)}")
    print(">> §5: Table A/B/C/D 构建完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
