#!/usr/bin/env python
"""scripts/basilisk_b5/summarize_b5.py

BASILISK-B5 §9/§14/§15/§18/§19/§20 —— 正式 Gate 判定与最终结论。

**FT 与 MMD 各自独立判定, 八条全部满足才算正转移** (§9):
  1 ≥4/5 seed gain > 0 ｜ 2 mean(gain) > 0 ｜ 3 median(gain) > 0
  4 paired seed bootstrap 95% CI 下界 > 0
  5 corr 无系统性退化 (source corr mean >= target corr mean - 0.02)
  6 catastrophic error rate 不高于 target_only
  7 warning miss rate 不高于 target_only + 0.05
  8 short/medium/long 中至少 2/3 个 bin 的 gain > 0

阈值全部从 protocol_hash.json 读 —— **不从 config 现读**, 因为 config 可能在出
数字之后被改; protocol_hash 是出数字之前冻结的那一份。两边不一致即 B5_INVALID。

§19 最终判定必须是四种组合之一 (A/B/C/D), 外加**独立**的
ENGINEERING_RECOMMENDATION (§15), 其依据必须是正式 test 表的主指标。

§20 B5 完成后停止, 不自动跑 B6, 不自动跑低数据矩阵。

用法:
    python scripts/basilisk_b5/summarize_b5.py
"""
from __future__ import annotations

import argparse
import hashlib
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
from scripts.basilisk_b2.baselines_b2 import EXCLUDED_ORACLE_METHODS  # noqa: E402

NAN = float("nan")

MAIN_TABLE_ROWS = ("target_only", "source_finetune", "source_mmd_finetune",
                   "const_mean_info", "damage_extrapolation")
TRANSFER_ROWS = ("source_finetune", "source_mmd_finetune")

SUMMARY_PURPOSE = (
    "B5 §19: 在 B2.1 冻结划分 (寿命覆盖完整) 上用全新正式 seed 判定 "
    "source_finetune 与 source_mmd_finetune 是否相对 Target-only 构成正式正转移, "
    "并单独给出工程推荐。"
    "这是第一次允许产生正式迁移结论的阶段。"
    "B4X_TRANSFER_STABILIZATION_SIGNAL 是探索性信号, 未被继承为本阶段结论。"
    "B2_GENERALIZATION_FAIL 在其自身划分上保持终局, 本阶段不追溯改写。"
)

PASS_MEANING = (
    "B5_FT_POSITIVE_TRANSFER / B5_MMD_POSITIVE_TRANSFER 只意味着: "
    "在 B2.1 划分、CORE_ONLY 输入、五个正式 seed 下, 该迁移方法的 test "
    "information-zone trajectory-macro RMSE 相对 Target-only 稳定更低, "
    "且未在 corr / catastrophic / warning miss / 寿命分 bin 上出现系统性退化。"
    "它**不意味着**该方法是最佳方法, 更不意味着它优于物理 damage extrapolation "
    "基线 —— 后者是独立命题, 见 transfer_vs_damage。"
    "它也不意味着结论可外推到其它划分、其它输入 schema 或低数据场景。"
)

FAIL_MEANING = (
    "B5_FT_NO_POSITIVE_TRANSFER / B5_MMD_NO_POSITIVE_TRANSFER 意味着: "
    "在消除寿命覆盖不匹配这一已知缺陷之后, 该迁移方法在预登记的八条门槛下"
    "没能证明相对 Target-only 的稳定增益。"
    "这是一个正当的负面结论, 不是实验失败, **不得通过降低门槛、换 seed、"
    "改超参或事后挑子集来翻转**。"
)

RECOMMENDATION_BASIS = (
    "§15: ENGINEERING_RECOMMENDATION 与迁移判定分开。"
    "推荐依据只能是正式 test 表的主指标 (info macro RMSE 的 seed 均值), "
    "不得依据机理性 / 叙事性解释。"
    "允许出现 B5_FT_POSITIVE_TRANSFER = true 同时推荐 damage_extrapolation —— "
    "因为两者回答的问题不同: 前者问预训练是否帮到了这个模型, "
    "后者问现在该部署什么。"
    "persistence 与 true_rul_lookup 是 oracle, 排除在候选之外。"
)

DAMAGE_PHRASING_WHEN_NEGATIVE = (
    "迁移模型可能相对 Target-only 有增益, 但未达到物理 damage extrapolation 基线。")

EXIT_DISCIPLINE = (
    "§20: B5 完成后停止。不自动跑低数据矩阵, 不自动执行 B6。"
    "若成立正式正转移, 下一阶段建议 B6: low-data / truncation transfer matrix; "
    "若不成立, 下一阶段建议 B6: baseline formal matrix + negative-transfer "
    "conclusion。两者都只写入建议, 不自动执行。"
)

GATE_BAR_DISCIPLINE = (
    "阈值全部从 protocol_hash.json 读取 —— 那是在任何 B5 数字之前冻结的一份。"
    "不从 config 现读, 因为 config 可能在出数字之后被改。"
    "两边不一致即 B5_INVALID。禁止降低门槛。"
)


def _cmp_le(a: float, b: float) -> bool:
    """a <= b, 且两者都必须有限。NaN 一律判不通过, 不当作通过。"""
    return bool(np.isfinite(a) and np.isfinite(b) and a <= b)


def _cmp_ge(a: float, b: float) -> bool:
    return bool(np.isfinite(a) and np.isfinite(b) and a >= b)


def _pos(a: float) -> bool:
    return bool(np.isfinite(a) and a > 0)


def decide_one(name: str, gi: dict, wi: dict, lb: dict, thr: dict,
               labels: dict) -> dict:
    """§9 八条 —— 对单一迁移方法判定。任一条不满足即 NO_POSITIVE_TRANSFER。"""
    key = "ft" if name == "source_finetune" else "mmd"
    g = gi[key]
    n_bins_pos = int(lb["n_bins_gain_positive"][key])
    conds = [
        {"id": 1, "name": "improve_count",
         "detail": f"{g['improve_count']}/{thr['n_seeds']} seed gain > 0, "
                   f"要求 ≥ {thr['min_improve_count']}",
         "passed": bool(g["improve_count"] >= thr["min_improve_count"])},
        {"id": 2, "name": "mean_positive",
         "detail": f"mean(gain) = {g['mean']:+.6f} > 0",
         "passed": _pos(g["mean"]) if thr["require_mean_positive"] else True},
        {"id": 3, "name": "median_positive",
         "detail": f"median(gain) = {g['median']:+.6f} > 0",
         "passed": _pos(g["median"]) if thr["require_median_positive"] else True},
        {"id": 4, "name": "bootstrap_ci_lower_positive",
         "detail": f"paired seed bootstrap CI95 下界 = {g['ci95_lower']:+.6f} > 0",
         "passed": (_pos(g["ci95_lower"])
                    if thr["require_ci_lower_positive"] else True)},
        {"id": 5, "name": "no_corr_degradation",
         "detail": (f"corr {g['corr_mean']:.4f} >= "
                    f"target {g['corr_mean_target']:.4f} - {thr['corr_tol']}"),
         "passed": _cmp_ge(g["corr_mean"],
                           g["corr_mean_target"] - thr["corr_tol"])},
        {"id": 6, "name": "catastrophic_not_worse",
         "detail": (f"catastrophic rate {g['catastrophic_mean']:.6f} <= "
                    f"target {g['catastrophic_mean_target']:.6f}"),
         "passed": (_cmp_le(g["catastrophic_mean"],
                            g["catastrophic_mean_target"])
                    if thr["require_catastrophic_not_worse"] else True)},
        {"id": 7, "name": "warning_miss_not_worse",
         "detail": (f"miss {wi[f'{key}_miss_mean']:.6f} <= target "
                    f"{wi['target_only_miss_mean']:.6f} + "
                    f"{thr['warning_miss_tol']}"),
         "passed": bool(wi[f"{key}_within_tol"])},
        {"id": 8, "name": "lifetime_bins_gain_positive",
         "detail": (f"{n_bins_pos}/{thr['n_lifetime_bins']} 个 bin gain > 0, "
                    f"要求 ≥ {thr['min_lifetime_bins_gain_positive']}"),
         "passed": bool(n_bins_pos >= thr["min_lifetime_bins_gain_positive"])},
    ]
    n_passed = sum(1 for c in conds if c["passed"])
    all_ok = n_passed == len(conds)
    verdict = labels[f"{key}_pass"] if all_ok else labels[f"{key}_fail"]
    return {
        "method": name,
        "conditions": conds,
        "n_conditions": len(conds),
        "n_passed": n_passed,
        "all_required": True,
        "verdict": verdict,
        "positive_transfer": bool(all_ok),
        "failed_conditions": [c["id"] for c in conds if not c["passed"]],
        "thresholds_frozen_before_run": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel_basilisk_b5.yaml")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b5 = cfg["b5"]
    P = cfg["paths"]

    # --- 阈值只从冻结的 protocol_hash 读 ---
    proto = json.loads((ROOT / cfg["protocol"]["hash_path"]).read_text("utf-8"))
    thr = dict(proto["gate_thresholds"])
    labels = dict(proto["gate_labels"])
    now = hashlib.sha256(
        (ROOT / cfg["protocol"]["path"]).read_bytes()).hexdigest()
    if now != str(proto["protocol_sha256"]):
        raise SystemExit("!! protocol.md 在冻结之后被改动 —— B5_INVALID")
    # config 现值必须与冻结值一致, 否则说明有人事后调了门槛
    live = {"min_improve_count": int(b5["gate"]["min_improve_count"]),
            "corr_tol": float(b5["gate"]["corr_tol"]),
            "warning_miss_tol": float(b5["gate"]["warning_miss_tol"]),
            "min_lifetime_bins_gain_positive":
                int(b5["gate"]["min_lifetime_bins_gain_positive"])}
    for k, v in live.items():
        if thr[k] != v:
            raise SystemExit(
                f"!! B5_INVALID: gate.{k} config={v} != 冻结值 {thr[k]} —— "
                f"禁止降低门槛")

    M = json.loads((ROOT / P["metrics_json"]).read_text("utf-8"))
    G = json.loads((ROOT / P["gain_json"]).read_text("utf-8"))
    L = json.loads((ROOT / P["lifetime_json"]).read_text("utf-8"))
    W = json.loads((ROOT / P["warning_json"]).read_text("utf-8"))
    if bool(M.get("fast")):
        raise SystemExit("!! metrics 来自 --fast 冒烟跑, 不得下正式结论")
    for nm, blk in (("gain", G), ("lifetime", L), ("warning", W)):
        if str(blk["split_sha256"]) != str(M["split_sha256"]):
            raise SystemExit(f"!! {nm} 的划分哈希与 metrics 不一致 —— B5_INVALID")

    dec_ft = decide_one("source_finetune", G["gate_inputs"], W["gate_inputs"],
                        L, thr, labels)
    dec_mmd = decide_one("source_mmd_finetune", G["gate_inputs"],
                         W["gate_inputs"], L, thr, labels)

    # §19 四种组合
    ft_ok, mmd_ok = dec_ft["positive_transfer"], dec_mmd["positive_transfer"]
    if ft_ok and mmd_ok:
        combo, combo_desc = "A", "FT positive + MMD positive"
    elif ft_ok and not mmd_ok:
        combo, combo_desc = "B", "FT positive + MMD negative"
    elif not ft_ok and mmd_ok:
        combo, combo_desc = "C", "FT negative + MMD positive"
    else:
        combo, combo_desc = "D", labels["both_fail"]
    overall = (labels["both_fail"] if combo == "D"
               else f"{dec_ft['verdict']} + {dec_mmd['verdict']}")

    # --- §18 主结果表 (五行齐备, damage 不得降级成脚注) ---
    agg = G["aggregate_rmse"]
    wbg = W["by_group"]
    sbg = G["output_stability"]["by_group"]
    cbg = G["catastrophic_rate_mean_by_group"]
    corr = G["corr_mean_by_group"]
    rows_per_seed = M["per_seed"]

    def _pooled(g: str) -> float:
        v = [float(r[g]["info_pooled_rmse"]) for r in rows_per_seed
             if np.isfinite(float(r[g]["info_pooled_rmse"]))]
        return float(np.mean(v)) if v else NAN

    def _mae(g: str) -> float:
        # MAE 由寿命分 bin 表的三 bin 均值给出 (同一 evaluator, 同一取点)
        vals = [L["by_group"][g][n]["mae_mean"] for n in L["bins"]["names"]]
        vals = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
        return float(np.mean(vals)) if vals else NAN

    main_table = {}
    for g in MAIN_TABLE_ROWS:
        main_table[g] = {
            "info_macro_rmse": agg[g]["mean"],
            "info_macro_rmse_std": agg[g]["std"],
            "info_pooled_rmse": _pooled(g),
            "mae": _mae(g),
            "corr": corr[g],
            "psr": sbg[g]["psr_mean"],
            "pred_std": sbg[g]["pred_std_mean"],
            "true_std": sbg[g]["true_std_mean"],
            "high_variance_warning": bool(sbg[g]["high_variance_warning"]),
            "warning_coverage": wbg[g]["warning_coverage"]["mean"],
            "miss_rate": wbg[g]["miss_rate"]["mean"],
            "ph": wbg[g]["prognostic_horizon"]["mean"],
            "ph_n_evaluable": wbg[g]["prognostic_horizon"]["n_seeds_evaluable"],
            "alpha_lambda": {k: v["accuracy_mean"]
                             for k, v in wbg[g]["alpha_lambda"].items()},
            "catastrophic_rate": cbg[g],
        }

    # --- §14 与物理基线的比较 ---
    dmg = agg["damage_extrapolation"]["mean"]
    tvd = {}
    for g in TRANSFER_ROWS:
        d = dmg - agg[g]["mean"]
        tvd[g] = {"transfer_vs_damage": d,
                  "transfer_worse_than_physics": bool(d < 0),
                  "rmse_transfer": agg[g]["mean"], "rmse_damage": dmg}
    any_worse = any(v["transfer_worse_than_physics"] for v in tvd.values())
    damage_block = {
        "kept_in_main_table": True,
        "demoted_to_footnote": False,
        "comparison": tvd,
        "definition": "transfer_vs_damage = RMSE_damage - RMSE_transfer; "
                      "负数 = 迁移不如物理基线",
        "mandatory_statement": (DAMAGE_PHRASING_WHEN_NEGATIVE if any_worse
                                else None),
        "forbid_calling_positive_transfer_the_best_method": True,
    }

    # --- §15 工程推荐: 只按正式 test 表主指标, 与迁移判定分开 ---
    cands = [g for g in b5["engineering_recommendation"]["candidates"]
             if g not in EXCLUDED_ORACLE_METHODS]
    scored = sorted(((g, float(agg[g]["mean"])) for g in cands
                     if np.isfinite(float(agg[g]["mean"]))),
                    key=lambda kv: kv[1])
    if not scored:
        raise SystemExit("!! 无候选可评分 —— 拒绝给推荐")
    rec, rec_val = scored[0]
    recommendation = {
        "recommendation": rec,
        "basis": "formal_test_table_primary_metric",
        "primary_metric": str(b5["primary_metric"]),
        "ranking": [{"method": g, "info_macro_rmse": v} for g, v in scored],
        "value": rec_val,
        "separate_from_transfer_verdict": True,
        "meaning": RECOMMENDATION_BASIS,
        "excluded_oracle_methods": list(EXCLUDED_ORACLE_METHODS),
        "note_if_diverges": (
            "迁移判定与工程推荐可以指向不同对象; 两者回答的问题不同。"
            if (ft_ok or mmd_ok) and rec != "source_finetune"
               and rec != "source_mmd_finetune" else None),
    }

    # --- §20 退出 ---
    er = b5["exit_rules"]
    next_stage = (str(er["next_stage_on_positive"]) if (ft_ok or mmd_ok)
                  else str(er["next_stage_on_negative"]))

    out = {
        "stage": "BASILISK_B5",
        "label": str(b5["label"]),
        "formal_stage": True,
        "first_stage_allowed_to_conclude_transfer": True,
        "purpose": SUMMARY_PURPOSE,
        "pass_meaning": PASS_MEANING,
        "fail_meaning": FAIL_MEANING,
        "gate_bar_discipline": GATE_BAR_DISCIPLINE,
        "exit_discipline": EXIT_DISCIPLINE,
        "questions": dict(b5["questions"]),
        "frozen_facts": dict(b5["frozen_facts"]),
        "b4x_must_not_be_promoted": True,
        "b2_verdict_status": "B2_GENERALIZATION_FAIL_REMAINS_FINAL",
        "protocol_sha256": str(proto["protocol_sha256"]),
        "config_sha256": str(proto["config_sha256"]),
        "split_sha256": str(M["split_sha256"]),
        "input_schema": str(M["input_schema"]),
        "formal_seeds": [int(s) for s in M["formal_seeds"]],
        "seed_role": str(M["seed_role"]),
        "architecture_note": str(M["architecture_note"]),
        "primary_metric": str(b5["primary_metric"]),
        "gate_thresholds": thr,
        "decision": {"source_finetune": dec_ft, "source_mmd_finetune": dec_mmd},
        "verdict_combination": {"combination": combo,
                               "description": combo_desc,
                               "allowed": ["A", "B", "C", "D"]},
        "verdict": overall,
        "ft_verdict": dec_ft["verdict"],
        "mmd_verdict": dec_mmd["verdict"],
        "positive_transfer": {"ft": ft_ok, "mmd": mmd_ok},
        "main_table": main_table,
        "main_table_rows": list(MAIN_TABLE_ROWS),
        "per_seed_detail_kept_separately": str(P["metrics_json"]),
        "lifetime_bins": {"gain_by_bin": L["gain_by_bin"],
                          "n_bins_gain_positive": L["n_bins_gain_positive"],
                          "short_life": L["short_life"],
                          "localized_transfer_benefit":
                              L["localized_transfer_benefit"]},
        "warning": {"by_group": {g: {
            "warning_coverage": wbg[g]["warning_coverage"]["mean"],
            "miss_rate": wbg[g]["miss_rate"]["mean"],
            "miss_rate_before_eol": wbg[g]["miss_rate_before_eol"]["mean"],
            "false_alarm_rate": wbg[g]["false_alarm_rate"]["mean"],
            "ph": wbg[g]["prognostic_horizon"]["mean"],
            "ph_n_evaluable": wbg[g]["prognostic_horizon"]["n_evaluable_traj"],
            "convergence": wbg[g]["convergence"]["macro_mean"],
            "warning_valid_count": wbg[g]["warning_valid_count"],
        } for g in MAIN_TABLE_ROWS}},
        "output_stability": G["output_stability"],
        "damage_baseline": damage_block,
        "engineering_recommendation": recommendation,
        "ENGINEERING_RECOMMENDATION": rec,
        "next_stage": next_stage,
        "b6_auto_run": False,
        "low_data_matrix_auto_run": False,
        "forbid_auto_run_b6": bool(er["forbid_auto_run_b6"]),
        "stop_after_b5": bool(er["stop_after_b5"]),
    }

    # 从 conditions 重算一次, 与写入的 verdict 对不上就报错 —— 防止手改结论
    for nm, dec in (("ft", dec_ft), ("mmd", dec_mmd)):
        recomputed = all(c["passed"] for c in dec["conditions"])
        if recomputed != dec["positive_transfer"]:
            raise SystemExit(f"!! {nm} 判定与 conditions 不一致 —— 拒绝写出")

    op = ROOT / P["summary_json"]
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                             default=float) + "\n", encoding="utf-8")
    print(f">> 已写 {op.relative_to(ROOT)}")
    print(f"\n{'=' * 70}")
    print(f"{'方法':24s} {'RMSE':>9s} {'corr':>7s} {'PSR':>7s} {'miss':>7s} "
          f"{'cat':>7s}")
    for g in MAIN_TABLE_ROWS:
        t = main_table[g]
        print(f"{g:24s} {t['info_macro_rmse']:9.6f} {t['corr']:7.4f} "
              f"{t['psr']:7.3f} {t['miss_rate']:7.4f} "
              f"{t['catastrophic_rate']:7.4f}")
    print(f"{'=' * 70}")
    for nm, dec in (("FT ", dec_ft), ("MMD", dec_mmd)):
        print(f"[{nm}] {dec['verdict']}  ({dec['n_passed']}/8 条通过)")
        for c in dec["conditions"]:
            print(f"    {'PASS' if c['passed'] else 'FAIL'} {c['id']}. "
                  f"{c['name']}: {c['detail']}")
    print(f"\n>> §19 组合 {combo}: {combo_desc}")
    print(f">> 最终判定: {overall}")
    for g, v in tvd.items():
        print(f">> transfer_vs_damage[{g}] = {v['transfer_vs_damage']:+.6f} "
              f"({'迁移不如物理基线' if v['transfer_worse_than_physics'] else '优于物理基线'})")
    if any_worse:
        print(f">> 必须原样写出: {DAMAGE_PHRASING_WHEN_NEGATIVE}")
    print(f">> ENGINEERING_RECOMMENDATION = {rec} "
          f"(info macro RMSE {rec_val:.6f}, 依据正式 test 表)")
    print(f">> 下一阶段建议: {next_stage} —— 不自动执行 (§20)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
