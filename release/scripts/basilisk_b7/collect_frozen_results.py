#!/usr/bin/env python
"""scripts/basilisk_b7/collect_frozen_results.py

BASILISK-B7 §4 —— 建立 frozen result index。

**只读**。本脚本只能读取已冻结的 JSON / NPZ / metadata, 禁止重新调用训练器或
evaluator "顺算一次"。只有已有产物中不存在、且纯确定性格式转换的统计才可计算
(例如把 per_seed 列表转成 std, 把两个已冻结均值相减)。不得产生新的评价定义。

必须整理 §4 的七类事实:
  A. B2.1 generalization
  B. B5 full-label formal comparison
  C. B6 label-scarcity matrix
  D. B6 warning metrics
  E. B6 lifetime-bin metrics
  F. final engineering recommendation
  G. final transfer conclusion

输出:
  checkpoints/basilisk_b7/frozen_result_index.json
  docs/basilisk_b7/frozen_result_index.md
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "checkpoints" / "basilisk_b7" / "frozen_result_index.json"
OUT_MD = ROOT / "docs" / "basilisk_b7" / "frozen_result_index.md"

STAGE = "BASILISK_B7"
LABEL = "FINAL_FIGURES_REPORT_EVIDENCE_FREEZE"

# §4: 允许读取的冻结来源。任何不在此表中的路径都不得作为报告数字来源。
SOURCES = {
    "b18_dataset_audit": "checkpoints/basilisk_b18/dataset_audit.json",
    "b18_primary_scenario":
        "checkpoints/basilisk_b18/frozen_primary_scenario.json",
    "b19_feature_definition":
        "checkpoints/basilisk_b19/frozen_feature_definition.json",
    "b21_split_manifest": "docs/basilisk_b21/split_manifest.json",
    "b21_summary": "checkpoints/basilisk_b21/summary.json",
    "b5_summary": "checkpoints/basilisk_b5/summary.json",
    "b5_formal_metrics": "checkpoints/basilisk_b5/formal_metrics.json",
    "b5_paired_gain": "checkpoints/basilisk_b5/paired_gain.json",
    "b5_warning_metrics": "checkpoints/basilisk_b5/warning_metrics.json",
    "b5_lifetime_bins": "checkpoints/basilisk_b5/lifetime_bins.json",
    "b6_summary": "checkpoints/basilisk_b6/summary.json",
    "b6_final_verdict": "checkpoints/basilisk_b6/final_verdict.json",
    "b6_paired_statistics": "checkpoints/basilisk_b6/paired_statistics.json",
    "b6_lifetime_bins": "checkpoints/basilisk_b6/lifetime_bins.json",
    "b6_warning_metrics": "checkpoints/basilisk_b6/warning_metrics.json",
    "b6_all_metrics": "checkpoints/basilisk_b6/all_metrics.json",
    "b6_label_subset_manifest":
        "checkpoints/basilisk_b6/label_subset_manifest.json",
    "b6_raw_predictions": "checkpoints/basilisk_b6/all_raw.npz",
}

LEVELS = [3, 5, 10, 21]
PRIMARY_LEVEL = 5
TRAINED = ["target_only", "source_finetune", "source_mmd_finetune"]
EVAL_ONLY = ["const_mean_info", "damage_extrapolation"]
ALL_GROUPS = TRAINED + EVAL_ONLY


def _read(key: str):
    p = ROOT / SOURCES[key]
    if not p.exists():
        raise SystemExit(f"!! 冻结来源缺失: {SOURCES[key]}")
    return json.loads(p.read_text(encoding="utf-8"))


def _sha(rel: str) -> str:
    p = ROOT / rel
    return (hashlib.sha256(p.read_bytes()).hexdigest() if p.exists()
            else "MISSING")


def _f(v):
    """NaN 保持 NaN, 不得悄悄转 0。"""
    return None if v is None else float(v)


def _std(xs) -> float:
    """per_seed 列表 -> 样本标准差。纯确定性格式转换, §4 允许。"""
    xs = [float(x) for x in xs]
    if any(math.isnan(x) for x in xs):
        return float("nan")
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


# --------------------------------------------------------------------------
# A. B2.1 generalization
# --------------------------------------------------------------------------
def fact_a(sp: dict, s21: dict) -> dict:
    sl = sp["splits"]
    ev = sp["lifetime_bins"]["event"]
    return {
        "topic": "B2.1 generalization / coverage-aware trajectory split",
        "generalization_verdict": str(s21["verdict"]),
        "split_verdict": str(sp["verdict"]),
        "split_sha256": str(sp["split_sha256"]),
        "split_seed": int(sp["split_seed"]),
        "split_level": str(sp["level"]),
        "n_traj": int(sp["n_traj"]),
        "n_event_observed": int(sp["n_event_observed"]),
        "n_censored": int(sp["n_censored"]),
        "ratios": {k: float(v) for k, v in sp["ratios"].items()},
        "splits": {
            name: {
                "n": int(sl[name]["n"]),
                "n_event": int(sl[name]["n_event"]),
                "n_censored": int(sl[name]["n_censored"]),
                "event_eol_min": _f(sl[name]["event_eol_min"]),
                "event_eol_max": _f(sl[name]["event_eol_max"]),
                "event_bins_present": sorted(sl[name]["event_bins_present"]),
            } for name in ("train", "val", "test")
        },
        "lifetime_bins_event": {
            "names": list(ev["names"]),
            "edges": [float(x) for x in ev["edges"]],
            "quantile_basis": str(ev["quantile_basis"]),
            "counts": {k: int(v) for k, v in ev["counts"].items()},
        },
        "test_lifetime_bin_counts": {
            k: int(v) for k, v in s21["test_lifetime_bin_counts"].items()},
        "b2_vs_b21_min_event_eol": {
            "b2": {k: _f(v) for k, v in
                   s21["b2_split_comparison"]["b2_min_event_eol"].items()},
            "b21": {k: _f(v) for k, v in
                    s21["b2_split_comparison"]["b21_min_event_eol"].items()},
        },
        "b21_target_only_rmse": _f(
            s21["aggregate"]["target_only_rmse"]["mean"]),
        "b21_const_rmse": _f(s21["aggregate"]["const_mean_info_rmse"]["mean"]),
        "b21_damage_rmse": _f(
            s21["aggregate"]["damage_extrapolation_rmse"]["mean"]),
        "b21_seeds": [int(s) for s in s21["seeds"]],
        "b21_methods": [str(m) for m in s21["methods"]],
        "b2_relationship": (
            "B2 使用随机轨迹划分, test 的最小 event EOL (21658) 低于 train/val "
            "的最小值, 出现 lifetime-support mismatch, 判定 "
            "B2_GENERALIZATION_FAIL, 并保留为 split coverage failure 的诊断证据。"
            "B2.1 改为 event/censored x lifetime-bin 联合分层的 coverage-aware "
            "划分后得到 B21_GENERALIZATION_PASS。"),
        "forbid_cross_split_rmse_comparison": True,
        "forbid_reason": (
            "B2 与 B2.1 的 test split 不同, 两者的 RMSE 不构成算法层面的公平比较, "
            "只能用于说明划分覆盖问题。"),
    }


# --------------------------------------------------------------------------
# B. B5 full-label formal comparison  (Table A 唯一来源)
# --------------------------------------------------------------------------
def fact_b(sm: dict, fm: dict, gj: dict, wj: dict) -> dict:
    mt = sm["main_table"]
    per_seed = gj["per_seed"]
    rows = {}
    for g in ALL_GROUPS:
        m = mt[g]
        rows[g] = {
            "info_macro_rmse": _f(m["info_macro_rmse"]),
            "info_macro_rmse_std": _f(m["info_macro_rmse_std"]),
            "info_macro_rmse_per_seed": [_f(p[g]) for p in per_seed],
            "info_pooled_rmse": _f(m["info_pooled_rmse"]),
            "mae": _f(m["mae"]),
            "macro_corr": _f(m["corr"]),
            "psr": _f(m["psr"]),
            "warning_coverage": _f(m["warning_coverage"]),
            "miss_rate": _f(m["miss_rate"]),
            "false_alarm_rate": _f(
                wj["by_group"][g]["false_alarm_rate"]["mean"]),
            "catastrophic_rate": _f(m["catastrophic_rate"]),
            "prognostic_horizon": _f(
                wj["by_group"][g]["prognostic_horizon"]["mean"]),
        }
    return {
        "topic": "B5 full-label confirmatory transfer comparison",
        "verdict": str(sm["verdict"]),
        "ft_verdict": str(sm["ft_verdict"]),
        "mmd_verdict": str(sm["mmd_verdict"]),
        "verdict_combination": str(sm["verdict_combination"]["combination"]),
        "positive_transfer": bool(sm["positive_transfer"]),
        "engineering_recommendation": str(sm["ENGINEERING_RECOMMENDATION"]),
        "protocol_sha256": str(sm["protocol_sha256"]),
        "config_sha256": str(sm["config_sha256"]),
        "split_sha256": str(sm["split_sha256"]),
        "input_schema": str(sm["input_schema"]),
        "formal_seeds": [int(s) for s in sm["formal_seeds"]],
        "source_checkpoint": str(fm["source_checkpoint"]),
        "source_checkpoint_sha256": str(fm["source_checkpoint_sha256"]),
        "mmd_lambda": _f(fm["mmd_lambda"]),
        "primary_metric": str(sm["primary_metric"]),
        "label_condition": "full target train labels (event 21 + censored 24)",
        "method_order": ["damage_extrapolation", "target_only",
                         "source_mmd_finetune", "source_finetune"],
        "rows": rows,
        "gain": {
            key: {
                "mean": _f(gj["gain"][key]["mean"]),
                "median": _f(gj["gain"][key]["median"]),
                "std": _f(gj["gain"][key]["std"]),
                "improve_count": int(gj["gain"][key]["improve_count"]),
                "n_seeds": int(gj["gain"][key]["n_seeds"]),
                "ci95_lower": _f(gj["gain"][key]["ci95_lower"]),
                "ci95_upper": _f(gj["gain"][key]["ci95_upper"]),
                "ci_lower_positive": bool(
                    gj["gain"][key]["ci_lower_positive"]),
                "per_seed": [_f(x) for x in gj["gain"][key]["per_seed"]],
            } for key in ("gain_ft", "gain_mmd")
        },
        "decision": {
            m: {
                "n_passed": int(sm["decision"][m]["n_passed"]),
                "n_conditions": int(sm["decision"][m]["n_conditions"]),
                "verdict": str(sm["decision"][m]["verdict"]),
                "positive_transfer": bool(
                    sm["decision"][m]["positive_transfer"]),
                "failed_conditions": [
                    int(x) for x in sm["decision"][m]["failed_conditions"]],
                "thresholds_frozen_before_run": bool(
                    sm["decision"][m]["thresholds_frozen_before_run"]),
            } for m in ("source_finetune", "source_mmd_finetune")
        },
        "table_a_statement": (
            "在全部目标域训练标签可用的确认性协议下, 两个迁移分组都没有通过"
            "预登记门槛: B5_NO_POSITIVE_TRANSFER。"),
    }


# --------------------------------------------------------------------------
# C. B6 label-scarcity matrix  (Table B / Table C 唯一来源)
# --------------------------------------------------------------------------
def fact_c(ps: dict, vj: dict, subset: dict) -> dict:
    by_level = {}
    for n in LEVELS:
        lv = ps["by_level"][str(n)]
        agg = lv["mean_info_macro_rmse"]
        g = lv["gain"]
        by_level[str(n)] = {
            "n_event_labeled": n,
            "is_primary": bool(lv["is_primary"]),
            "role": str(lv["role"]),
            "train_composition": {
                k: int(v) for k, v in lv["train_composition"].items()},
            "event_bin_counts": {
                k: int(v) for k, v in lv["event_bin_counts"].items()},
            "mean_info_macro_rmse": {k: _f(agg[k]) for k in ALL_GROUPS},
            "std_info_macro_rmse": {
                k: _f(lv["aggregate_rmse"][k]["std"]) for k in ALL_GROUPS},
            "median_info_macro_rmse": {
                k: _f(lv["aggregate_rmse"][k]["median"]) for k in ALL_GROUPS},
            "per_seed_info_macro_rmse": {
                k: [_f(p[k]) for p in lv["per_seed"]] for k in ALL_GROUPS},
            "gain": {
                key: {
                    "mean": _f(g[key]["mean"]),
                    "median": _f(g[key]["median"]),
                    "std": _f(g[key]["std"]),
                    "improve_count": int(g[key]["improve_count"]),
                    "n_seeds": int(g[key]["n_seeds"]),
                    "ci95_lower": _f(g[key]["ci95_lower"]),
                    "ci95_upper": _f(g[key]["ci95_upper"]),
                    "ci_lower_positive": bool(g[key]["ci_lower_positive"]),
                    "per_seed": [_f(x) for x in g[key]["per_seed"]],
                } for key in ("gain_ft", "gain_mmd")
            },
            "vs_damage": {
                f"{a}_minus_damage": _f(
                    agg[a] - agg["damage_extrapolation"]) for a in TRAINED},
            "vs_const": {
                f"{a}_minus_const": _f(
                    agg[a] - agg["const_mean_info"]) for a in TRAINED},
            "corr_mean_by_group": {
                k: _f(v) for k, v in lv["corr_mean_by_group"].items()},
            "catastrophic_rate_mean_by_group": {
                k: _f(v)
                for k, v in lv["catastrophic_rate_mean_by_group"].items()},
            "psr_mean_by_group": {
                k: _f(lv["output_stability"][k]["psr_mean"])
                for k in lv["output_stability"]},
            "fairness_pass": bool(lv["fairness_pass"]),
        }
    return {
        "topic": "B6 failure-label scarcity formal matrix",
        "primary_verdict": str(vj["B6_PRIMARY_VERDICT"]),
        "primary_level": int(vj["primary_level"]),
        "secondary_levels": [int(x) for x in vj["secondary_levels"]],
        "formal_seeds": [int(s) for s in vj["formal_seeds"]],
        "protocol_sha256": str(vj["protocol_sha256"]),
        "config_sha256": str(vj["config_sha256"]),
        "subset_manifest_sha256": str(vj["subset_manifest_sha256"]),
        "split_sha256": str(vj["split_sha256"]),
        "input_schema": str(vj["input_schema"]),
        "n_features": int(vj["n_features"]),
        "label_levels": LEVELS,
        "scarcity_axis": str(subset["scarcity_axis"]),
        "scarcity_axis_is_not": str(subset["scarcity_axis_is_not"]),
        "always_keep_all_censored_train": bool(
            subset["always_keep_all_censored_train"]),
        "n_censored_train_always": int(subset["n_censored_train_always"]),
        "subset_seed": int(subset["subset_seed"]),
        "generated_before_any_training": bool(
            subset["generated_before_any_training"]),
        "val_test_never_touched": bool(subset["val_test_never_touched"]),
        "forbid_recompute_edges_from_b6_test": bool(
            subset["forbid_recompute_edges_from_b6_test"]),
        "primary_metric": str(ps["primary_metric"]),
        "gain_definition": str(ps["gain_definition"]),
        "by_level": by_level,
        "primary_decision": {
            m: {
                "n_passed": int(vj["primary_decision"][m]["n_passed"]),
                "n_conditions": int(vj["primary_decision"][m]["n_conditions"]),
                "verdict": str(vj["primary_decision"][m]["verdict"]),
                "failed_conditions": [
                    int(x) for x in
                    vj["primary_decision"][m]["failed_conditions"]],
                "conditions": [
                    {"id": int(c["id"]), "name": str(c["name"]),
                     "detail": str(c["detail"]), "passed": bool(c["passed"])}
                    for c in vj["primary_decision"][m]["conditions"]],
            } for m in ("source_finetune", "source_mmd_finetune")
        },
        "primary_positive": {k: bool(v)
                             for k, v in vj["primary_positive"].items()},
        "secondary_signals": [
            {"label": str(s["label"]),
             "n_event_labeled": int(s["n_event_labeled"]),
             "method": str(s["method"]),
             "mean_gain": _f(s["mean_gain"]),
             "median_gain": _f(s["median_gain"]),
             "improve_count": int(s["improve_count"]),
             "ci95_lower": _f(s["ci95_lower"])}
            for s in vj["secondary"].get("signals", [])
        ],
        "secondary_role": str(vj["secondary"]["role"]),
        "secondary_cannot_override_primary": bool(
            vj["secondary_cannot_override_primary"]),
        "trend": {
            m: {
                "points": [
                    {"n_event_labeled": int(p["n_event_labeled"]),
                     "role": str(p["role"]),
                     "mean": _f(p["mean"]),
                     "median": _f(p["median"]),
                     "ci95_lower": _f(p["ci95_lower"]),
                     "ci95_upper": _f(p["ci95_upper"]),
                     "improve_count": int(p["improve_count"])}
                    for p in ps["sensitivity"]["by_method"][m]],
                "descriptive_only": True,
                "forbid_trend_line_fit": bool(
                    ps["sensitivity"]["forbid_trend_line_fit"]),
            } for m in ("source_finetune", "source_mmd_finetune")
        },
        "primary_mmd_boundary_statement": (
            "PRIMARY (n_event=5) 档 MMD 的 mean/median/CI95-lower 与三个 "
            "lifetime bin 的方向都为正, 但 improve_count 仅 3/5, corr 相对 "
            "target_only 下降, catastrophic 门槛与 beats-const 门槛均未过, "
            "十项预登记门槛未全部通过 —— 因此**不足以支持正向迁移结论**, "
            "只能表述为条件性低标签信号。禁止写成 '显著正迁移'。"),
    }


# --------------------------------------------------------------------------
# D. B6 warning metrics  (Table D 下半 唯一来源)
# --------------------------------------------------------------------------
def fact_d(wj: dict) -> dict:
    by_level = {}
    for n in LEVELS:
        lv = wj["by_level"][str(n)]
        groups = {}
        for g in ALL_GROUPS:
            w = lv["by_group"][g]
            groups[g] = {
                "warning_coverage": _f(w["warning_coverage"]["mean"]),
                "coverage_before_eol": _f(w["coverage_before_eol"]["mean"]),
                "miss_rate": _f(w["miss_rate"]["mean"]),
                "miss_rate_before_eol": _f(w["miss_rate_before_eol"]["mean"]),
                "false_alarm_rate": _f(w["false_alarm_rate"]["mean"]),
                "prognostic_horizon": _f(w["prognostic_horizon"]["mean"]),
                "n_seeds_evaluable": int(w["miss_rate"]["n_seeds_evaluable"]),
                "warning_coverage_per_seed": [
                    _f(x) for x in w["warning_coverage"]["per_seed"]],
                "miss_rate_per_seed": [
                    _f(x) for x in w["miss_rate"]["per_seed"]],
            }
        by_level[str(n)] = {
            "n_event_labeled": n,
            "is_primary": bool(lv["is_primary"]),
            "by_group": groups,
            "gate_inputs": {
                k: (bool(v) if isinstance(v, bool) else _f(v))
                for k, v in lv["gate_inputs"].items()},
        }
    return {
        "topic": "B6 prognostic warning metrics",
        "coverage_and_miss_reported_together": True,
        "censored_excluded_from_eol_dependent_metrics": True,
        "nan_not_converted_to_zero": True,
        "nphm_not_a_model_selection_metric": True,
        "by_level": by_level,
        "deployability_note": (
            "学习方法在 PRIMARY 档 miss rate 约 0.51-0.61, "
            "即超过半数失效轨迹未能在 EOL 前发出有效告警, "
            "当前不满足可部署告警要求。"
            "不得因 RMSE 数值较小就声称可直接投入部署。"),
    }


# --------------------------------------------------------------------------
# E. B6 lifetime-bin metrics  (Table D 上半 唯一来源)
# --------------------------------------------------------------------------
def fact_e(lj: dict) -> dict:
    bins = lj["bins"]
    by_level = {}
    for n in LEVELS:
        lv = lj["by_level"][str(n)]
        gb = {}
        for nm, blk in lv["gain_by_bin"].items():
            gb[nm] = {
                **{g: _f(blk[g]) for g in ALL_GROUPS if g in blk},
                "gain_ft": _f(blk["gain_ft"]),
                "gain_mmd": _f(blk["gain_mmd"]),
                "gain_ft_nonneg": bool(blk["gain_ft_nonneg"]),
                "gain_mmd_nonneg": bool(blk["gain_mmd_nonneg"]),
                "n_traj_in_bin": int(blk["n_traj_in_bin"]),
            }
        by_level[str(n)] = {
            "n_event_labeled": n,
            "is_primary": bool(lv["is_primary"]),
            "gain_by_bin": gb,
            "n_bins_gain_nonneg": {k: int(v) for k, v
                                   in lv["n_bins_gain_nonneg"].items()},
            "localized_transfer_benefit": {
                k: bool(v)
                for k, v in lv["localized_transfer_benefit"].items()
                if isinstance(v, bool)},
        }
    return {
        "topic": "B6 lifetime-bin resolved RMSE",
        "bin_names": list(bins["names"]),
        "bin_edges": [float(x) for x in bins["edges"]],
        "bin_source": str(bins["source"]),
        "quantile_basis": str(bins["quantile_basis"]),
        "recomputed_from_b6_test": bool(bins["recomputed_from_b6_test"]),
        "counts_test": {k: int(v) for k, v in bins["counts_test"].items()},
        "empty_bin_rule": "空 bin 返回 NaN + n=0, 不伪造 0",
        "censored_bins_degenerate": bool(
            lj["censored_bins"].get("degenerate", False)),
        "by_level": by_level,
    }


# --------------------------------------------------------------------------
# F. final engineering recommendation
# --------------------------------------------------------------------------
def fact_f(vj: dict, sm5: dict) -> dict:
    er = vj["engineering_recommendation"]
    return {
        "topic": "final engineering recommendation",
        "ENGINEERING_RECOMMENDATION": str(vj["ENGINEERING_RECOMMENDATION"]),
        "b5_engineering_recommendation": str(
            sm5["ENGINEERING_RECOMMENDATION"]),
        "independently_reached_in_both_stages": (
            str(vj["ENGINEERING_RECOMMENDATION"])
            == str(sm5["ENGINEERING_RECOMMENDATION"])),
        "single_recommendation": bool(er["single_recommendation"]),
        "basis": str(er["basis"]),
        "primary_level": int(er["primary_level"]),
        "primary_metric": str(er["primary_metric"]),
        "priority_order": [str(x) for x in er["priority_order"]],
        "priority_frozen": bool(er["priority_frozen"]),
        "ranking": [
            {"method": str(r["method"]),
             "info_macro_rmse": _f(r["info_macro_rmse"])}
            for r in er["ranking"]],
        "candidate_set": ["target_only", "source_finetune",
                          "source_mmd_finetune", "damage_extrapolation"],
        "excluded_from_candidates": ["const_mean_info", "persistence",
                                     "oracle", "true_rul_lookup"],
        "damage_kept_in_main_table": bool(
            vj["damage_baseline"]["kept_in_main_table"]),
        "damage_demoted_to_footnote": bool(
            vj["damage_baseline"]["demoted_to_footnote"]),
        "damage_leads_all_learned_at_every_level": all(
            bool(v["damage_leads_all_learned"])
            for v in vj["damage_baseline"]["per_level"].values()),
        "damage_rationale": (
            "damage_extrapolation 在 PRIMARY 档 RMSE 0.054312, 显著优于全部学习"
            "方法。它不是 '学习算法失败后被拿出来救场的基线', 而是在本仿真域中"
            "结构先验最强的方法 —— 因为仿真的累计损伤方程本身已知, "
            "physics baseline 直接持有该结构。因此它被明确列为工程推荐。"),
        "damage_baseline_caveat": (
            "局限: 现实飞轮的真 damage law 并未知。该基线代表**仿真内的** "
            "model-informed upper-quality reference, "
            "不能声称在真实在轨场景中也能达到同样精度。"),
    }


# --------------------------------------------------------------------------
# G. final transfer conclusion
# --------------------------------------------------------------------------
def fact_g(vj: dict, sm5: dict) -> dict:
    rel = vj["b5_b6_relationship"]
    return {
        "topic": "final flywheel transfer conclusion",
        "FINAL_TRANSFER_CONCLUSION": str(vj["FINAL_TRANSFER_CONCLUSION"]),
        "allowed_values": [str(x)
                           for x in vj["final_conclusion_allowed_values"]],
        "final_conclusion_doc": str(vj["final_conclusion_doc"]),
        "b5_verdict": str(sm5["verdict"]),
        "b6_primary_verdict": str(vj["B6_PRIMARY_VERDICT"]),
        "b5_b6_case": str(rel["case"]),
        "b5_b6_case_condition": str(rel["case_condition"]),
        "combined_statement": str(rel["combined_statement"]),
        "b5_conclusion_permanent": bool(rel["b5_conclusion_permanent"]),
        "b5_overwritten_by_b6": bool(rel["b5_overwritten_by_b6"]),
        "boundary_statement_en": (
            "No positive transfer was supported by the frozen "
            "confirmatory protocol."),
        "boundary_statement_zh": (
            "在本研究冻结的确认性协议下, 未获得足以支持正向迁移的证据。"),
        "is": "未能证明正向迁移",
        "is_not": "证明迁移无效",
        "must_not_be_stated_as": [
            "迁移显著提升", "迁移明显优于", "MMD 显著有效", "证明迁移有效",
            "达到可上线部署要求"],
        "conditional_signal_note": (
            "在低失效标签 n=5 条件下, MMD 出现方向一致的条件性信号 "
            "(mean +0.029478, median +0.043665, CI95 lower +0.001531, "
            "三个 lifetime bin gain 均非负), "
            "但未通过全部预登记门槛 (improve 3/5, corr, catastrophic, "
            "beats-const 均未过), 因此不足以支持正向迁移结论。"),
        "exploratory_must_not_be_promoted": {
            "B3X": "B3X_NO_STABILIZING_SIGNAL - mission features 不进入正式输入",
            "B4X": ("B4X_TRANSFER_STABILIZATION_SIGNAL 为 EXPLORATORY_ONLY, "
                    "不得写成 positive transfer"),
        },
    }


def main() -> int:
    sp = _read("b21_split_manifest")
    s21 = _read("b21_summary")
    sm5 = _read("b5_summary")
    fm5 = _read("b5_formal_metrics")
    g5 = _read("b5_paired_gain")
    w5 = _read("b5_warning_metrics")
    vj = _read("b6_final_verdict")
    ps = _read("b6_paired_statistics")
    lj = _read("b6_lifetime_bins")
    wj = _read("b6_warning_metrics")
    subset = _read("b6_label_subset_manifest")
    b18 = _read("b18_dataset_audit")
    b18s = _read("b18_primary_scenario")
    b19 = _read("b19_feature_definition")

    index = {
        "stage": STAGE,
        "label": LABEL,
        "section": "§4 frozen result index",
        "read_only": True,
        "recomputed_any_metric": False,
        "called_trainer_or_evaluator": False,
        "produced_new_metric_definition": False,
        "allowed_computation": (
            "仅纯确定性格式转换: per_seed -> std, 两个已冻结均值相减。"
            "不产生新的评价定义。"),
        "source_files": {k: {"path": v, "sha256": _sha(v)}
                         for k, v in SOURCES.items()},
        # ---- §0 场景与特征前提 ----
        "scenario": {
            "b18_verdict": str(b18s["verdict"]),
            "primary_scenario": str(b18s["primary_scenario"]),
            "damage_model": "dD/dt = g_duty * a_T(T) / L_ref",
            "eol_rule": str(b18["eol_rule"]),
            "L_ref_years_nominal": float(b18["L_ref_years"]),
            "eol_threshold_D": float(b18["eol_threshold_D"]),
            "n_traj": int(b18["n_traj"]),
            "n_event": int(b18["audit"]["n_event_observed"]),
            "n_censored": int(b18["audit"]["n_censored"]),
            "failure_fraction": _f(b18["audit"]["event_fraction"]),
            "eol_years_event_observed": {
                k: _f(v) for k, v in
                b18["audit"]["eol_years_event_observed"].items()
                if not isinstance(v, (list, dict))},
            "dataset_content_sha256": str(b18["content_sha256"]),
            "semantics": str(b18["semantics"]),
            "is_manufacturer_failure_specification": bool(
                b18["is_manufacturer_failure_specification"]),
            "d_ge_1_semantics_en": (
                "D>=1 is a project-defined simulated failure state, "
                "not a manufacturer hardware failure specification."),
            "d_ge_1_semantics_zh": (
                "D>=1 是项目定义的仿真失效状态, 不是制造商硬件失效规格。"),
            "basilisk_role": (
                "Basilisk 负责 attitude dynamics / controller / wheel speed / "
                "command torque / mission mode / duty; "
                "退化模型、D(t)、EOL 与 RUL 标签由本项目自建模型定义。"),
            "basilisk_does_not": "Basilisk 不生成寿命标签",
        },
        "features": {
            "b19_verdict": str(b19["verdict"]),
            "primary_hi": str(b19["selected_primary_hi"]["dataset_name"]),
            "primary_hi_display": str(
                b19["selected_primary_hi"]["display_name"]),
            "hi_equation": str(b19["selected_primary_hi"]["equation"]),
            "cumulative_direction": str(
                b19["selected_primary_hi"]["cumulative_direction"]),
            "feature_content_sha256": str(
                b19["dataset"]["feature_content_sha256"]),
            "xt_cols": list(b19["xt_schema"]["cols"]),
            "xt_core_cols": list(b19["xt_schema"]["core_cols"]),
            "damage_proxy_in_xT": bool(b19["xt_schema"]["damage_proxy_in_xT"]),
            "mission_features_in_xT": bool(
                b19["xt_schema"]["mission_features_in_xT"]),
            "trained_any_model": bool(b19["trained_any_model"]),
            "no_hidden_truth_leakage": True,
            "leakage_note": (
                "hidden D / b_true / EOL 均不进入 x_T; HI 只由遥测派生, "
                "自标定, 仅按时间正向累积。"),
        },
        # ---- §4 的七类事实 ----
        "facts": {
            "A_b21_generalization": fact_a(sp, s21),
            "B_b5_full_label_formal": fact_b(sm5, fm5, g5, w5),
            "C_b6_label_scarcity_matrix": fact_c(ps, vj, subset),
            "D_b6_warning_metrics": fact_d(wj),
            "E_b6_lifetime_bin_metrics": fact_e(lj),
            "F_final_engineering_recommendation": fact_f(vj, sm5),
            "G_final_transfer_conclusion": fact_g(vj, sm5),
        },
        "frozen_chain": {
            "B1.8": "B18_SCENARIO_READY",
            "B1.9": "B19_FEATURE_READY",
            "B2": "B2_GENERALIZATION_FAIL",
            "B3X": "B3X_NO_STABILIZING_SIGNAL",
            "B4X": "B4X_TRANSFER_STABILIZATION_SIGNAL (EXPLORATORY_ONLY)",
            "B2.1": "B21_GENERALIZATION_PASS",
            "B5": "B5_NO_POSITIVE_TRANSFER",
            "B6": "B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER",
            "FINAL": "NO_POSITIVE_TRANSFER_SUPPORTED",
        },
        "forbid_in_b7": [
            "training", "re-evaluating new seeds",
            "modifying numeric-affecting code", "tuning",
            "changing split / HI / RUL / loss / model / MMD",
            "regenerating lifetime dataset", "re-running bootstrap",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    idx_sha = hashlib.sha256(OUT_JSON.read_bytes()).hexdigest()
    print(f">> 已写 {OUT_JSON.relative_to(ROOT)}")
    print(f">> frozen_result_index_sha256 = {idx_sha}")

    # ---- 人类可读镜像 ----
    f = index["facts"]
    a, b, c = (f["A_b21_generalization"], f["B_b5_full_label_formal"],
               f["C_b6_label_scarcity_matrix"])
    d, e = f["D_b6_warning_metrics"], f["E_b6_lifetime_bin_metrics"]
    fr, gg = (f["F_final_engineering_recommendation"],
              f["G_final_transfer_conclusion"])
    p5 = c["by_level"]["5"]["mean_info_macro_rmse"]
    lines = [
        "# B7 frozen result index（§4）",
        "",
        f"`frozen_result_index_sha256 = {idx_sha}`",
        "",
        "本文件是 B7 全部图表与报告数字的**唯一**来源索引。B7 只读不算：",
        "",
        "- `read_only = true`",
        "- `recomputed_any_metric = false`",
        "- `called_trainer_or_evaluator = false`",
        "- `produced_new_metric_definition = false`",
        "",
        "允许的计算仅限纯确定性格式转换（per_seed → std、两个已冻结均值相减）。",
        "",
        "## 冻结来源与哈希",
        "",
        "| key | path | sha256 |",
        "|---|---|---|",
    ]
    for k, v in index["source_files"].items():
        lines.append(f"| `{k}` | `{v['path']}` | `{v['sha256'][:16]}…` |")
    lines += [
        "",
        "## 七类事实",
        "",
        "| 类 | 主题 | 关键结论 |",
        "|---|---|---|",
        f"| A | {a['topic']} | {a['generalization_verdict']} |",
        f"| B | {b['topic']} | {b['verdict']} |",
        f"| C | {c['topic']} | {c['primary_verdict']} |",
        f"| D | {d['topic']} | 学习方法 PRIMARY 档 miss 0.51–0.61，"
        "不满足可部署告警要求 |",
        f"| E | {e['topic']} | bin 边界 {e['bin_edges']}，"
        f"`recomputed_from_b6_test = "
        f"{str(e['recomputed_from_b6_test']).lower()}` |",
        f"| F | {fr['topic']} | {fr['ENGINEERING_RECOMMENDATION']} |",
        f"| G | {gg['topic']} | {gg['FINAL_TRANSFER_CONCLUSION']} |",
        "",
        "## B5 全标签确认性对比（Table A 来源）",
        "",
        "| method | info macro RMSE | std | corr | coverage | miss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for g in b["method_order"]:
        r = b["rows"][g]
        lines.append(
            f"| `{g}` | {r['info_macro_rmse']:.6f} | "
            f"{r['info_macro_rmse_std']:.6f} | "
            f"{r['macro_corr']:.4f} | {r['warning_coverage']:.4f} | "
            f"{r['miss_rate']:.4f} |")
    lines += [
        "",
        "## B6 标签稀缺矩阵（Table B 来源，info macro RMSE）",
        "",
        "| n_event | role | target_only | source_ft | source_mmd | const "
        "| damage |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for n in LEVELS:
        lv = c["by_level"][str(n)]
        m = lv["mean_info_macro_rmse"]
        lines.append(
            f"| {n} | {lv['role']} | {m['target_only']:.6f} | "
            f"{m['source_finetune']:.6f} | {m['source_mmd_finetune']:.6f} | "
            f"{m['const_mean_info']:.6f} | {m['damage_extrapolation']:.6f} |")
    lines += [
        "",
        "## B6 配对增益（Table C 来源，gain = target_only − transfer）",
        "",
        "| n_event | method | mean | median | improve | CI95 |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for n in LEVELS:
        lv = c["by_level"][str(n)]
        for key, nm in (("gain_ft", "source_finetune"),
                        ("gain_mmd", "source_mmd_finetune")):
            gi = lv["gain"][key]
            star = " **(PRIMARY)**" if n == PRIMARY_LEVEL else ""
            lines.append(
                f"| {n}{star} | `{nm}` | {gi['mean']:+.6f} | "
                f"{gi['median']:+.6f} | {gi['improve_count']}/"
                f"{gi['n_seeds']} | [{gi['ci95_lower']:+.6f}, "
                f"{gi['ci95_upper']:+.6f}] |")
    lines += [
        "",
        "## PRIMARY 档（n_event_labeled = 5）主指标",
        "",
        "| method | info macro RMSE |",
        "|---|---:|",
    ]
    for g in ALL_GROUPS:
        lines.append(f"| `{g}` | {p5[g]:.6f} |")
    lines += [
        "",
        "## 边界措辞（不可改写）",
        "",
        f"- 结论是**{gg['is']}**，而不是{gg['is_not']}。",
        f"- 英文：{gg['boundary_statement_en']}",
        f"- 中文：{gg['boundary_statement_zh']}",
        f"- 条件性信号：{gg['conditional_signal_note']}",
        "- 禁止表述：" + "、".join(
            f"“{x}”" for x in gg["must_not_be_stated_as"]),
        "",
        "## D>=1 的语义",
        "",
        f"- EN: {index['scenario']['d_ge_1_semantics_en']}",
        f"- ZH: {index['scenario']['d_ge_1_semantics_zh']}",
        f"- {index['scenario']['basilisk_role']}",
        "",
        "## 工程推荐",
        "",
        f"- `ENGINEERING_RECOMMENDATION = "
        f"{fr['ENGINEERING_RECOMMENDATION']}`",
        f"- 理由：{fr['damage_rationale']}",
        f"- 局限：{fr['damage_baseline_caveat']}",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f">> 已写 {OUT_MD.relative_to(ROOT)}")
    print(">> §4: 只读索引建立完成, 未调用训练器/evaluator, 未重算任何指标")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
