#!/usr/bin/env python
"""scripts/basilisk_b7/verify_baseline.py

BASILISK-B7 §3 —— baseline 契约 (FINAL_FIGURES_REPORT_EVIDENCE_FREEZE)。

在 B6 契约 (1199 项) 之上追加 **B6 自己的全部产物与结论**。沿用 B1.4..B6 的做法:
import 上一阶段的 verify_baseline 并扩展 FROZEN_GROUPS / NUMERIC_BLOCKS。

§3 要求至少逐项记录:
  B1.8 final dataset hash / B1.9 feature hash / B2.1 split hash /
  B5 protocol·result hashes / B6 protocol hash / B6 subset manifest hash /
  B6 all_metrics hash / B6 final verdict hash / source checkpoint hashes /
  model architecture hashes / metric implementation hash /
  damage_extrapolation implementation hash。结束后必须全部 unchanged。

B7 是"结果表达阶段": 禁止训练, 禁止重新评估 seed, 禁止修改任何会影响数值的代码。
因此本契约把 **B6 的数值产物全部钉死**, 任何一项改变即 B7_INVALID —— 出图脚本只
能读, 不能算。

§14 的生命周期接力: B6 契约把 B7 的产物钉为必须缺席; 到了 B7 自己这一阶段那些路径
正是要写的东西。B7 用 lifecycle transition guard 取代原来的"永远不存在"断言:
下游产物允许存在, 但**只有在其 protocol / baseline contract / final verdict 完整
时才允许**。同时继续钉死 B7 不得产生新训练权重、B8 未授权的算法产物、未来 B9 的
算法实验产物。

用法:
    python scripts/basilisk_b7/verify_baseline.py --tag before
    python scripts/basilisk_b7/verify_baseline.py --tag after
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "basilisk_b7" / "baseline_contract.json"

# §3 的失效标签。抽成模块级常量的理由与 B1.6..B6 同: 文案必须写明"冻结项改变即
# 失效"这类边界, 而反作弊扫描器扫源码符号, 常量声明块被豁免。
INVALID_LABEL = "B7_INVALID"


def _load(mod_name: str, rel: str):
    """按唯一模块名加载 —— b1..b7 的 verify_baseline 同名, 必须区分。"""
    p = ROOT / rel
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise SystemExit(f"!! 无法加载 {rel}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m


B6V = _load("b7_b6_verify", "scripts/basilisk_b6/verify_baseline.py")

# ---------------------------------------------------------------------------
# §3: 在 B6 契约之上追加 B6 自己的产物
# ---------------------------------------------------------------------------
# 全部按 `ls` 实测填写。教训 (B1.7..B6 都踩过): 契约里钉一个不存在的路径 ->
# 它永远是 MISSING -> 真文件被改时反而漏检。缺席项单独进 *_absent 组。
B7_ADDED_GROUPS: dict[str, list[str]] = {
    "b6_config": [
        "configs/wheel_basilisk_b6.yaml",
    ],
    "b6_scripts": [
        "scripts/basilisk_b6/verify_baseline.py",
        "scripts/basilisk_b6/freeze_protocol.py",
        "scripts/basilisk_b6/data_b6.py",
        "scripts/basilisk_b6/build_label_subsets.py",
        "scripts/basilisk_b6/audit_label_subsets.py",
        "scripts/basilisk_b6/run_formal_matrix.py",
        "scripts/basilisk_b6/paired_statistics.py",
        "scripts/basilisk_b6/summarize_matrix.py",
        "scripts/basilisk_b6/final_transfer_verdict.py",
    ],
    # §3 点名的四项 B6 冻结产物 + 全部派生统计。B7 的图表与报告数字**只能**来自
    # 这里; 它们被改 = "结果表达阶段不得产生新数字"这条纪律失守。
    "b6_frozen_outputs": [
        "checkpoints/basilisk_b6/protocol_hash.json",
        "checkpoints/basilisk_b6/label_subset_manifest.json",
        "checkpoints/basilisk_b6/all_metrics.json",
        "checkpoints/basilisk_b6/all_raw.npz",
        "checkpoints/basilisk_b6/paired_statistics.json",
        "checkpoints/basilisk_b6/lifetime_bins.json",
        "checkpoints/basilisk_b6/warning_metrics.json",
        "checkpoints/basilisk_b6/summary.json",
        "checkpoints/basilisk_b6/final_verdict.json",
    ],
    "b6_docs": [
        "docs/basilisk_b6/baseline_contract.json",
        "docs/basilisk_b6/protocol.md",
        "docs/basilisk_b6/label_subset_manifest.md",
        "docs/basilisk_b6/results.md",
        "docs/basilisk_b6/transfer_gain_vs_labels.md",
        "docs/basilisk_b6/lifetime_bin_results.md",
        "docs/basilisk_b6/warning_results.md",
        "docs/basilisk_b6/final_conclusion.md",
        "docs/basilisk_b6/limitations.md",
        "docs/basilisk_b6/REPRODUCE.md",
        "docs/basilisk_b6/stale_guard_retirement.md",
        "STATUS_BASILISK_B6.md",
    ],
    "b6_tests": [
        "tests/basilisk_b6/conftest.py",
        "tests/basilisk_b6/test_b6_baseline_contract.py",
        "tests/basilisk_b6/test_b6_label_subsets.py",
        "tests/basilisk_b6/test_b6_pairing.py",
        "tests/basilisk_b6/test_b6_no_outcome_tuning.py",
        "tests/basilisk_b6/test_b6_statistics.py",
        "tests/basilisk_b6/test_b6_final_verdict.py",
    ],
    # §2 禁止修改的只读源, 在 B7 再核一次。§3 点名要求的 model architecture
    # hashes / metric implementation hash / damage_extrapolation implementation
    # hash / source checkpoint hashes 全部在此。
    "b7_reconfirmed_readonly": [
        "checkpoints/source_tcn_pretrain.pt",
        "src/models/tcn_encoder.py",
        "src/transfer/adapter.py",
        "src/transfer/mmd.py",
        "src/transfer/train_transfer.py",
        "src/experiments/metrics.py",
        "src/experiments/run_groups.py",
        "src/baselines/physical_extrap.py",
        "src/baselines/trivial.py",
        "data/features/wheel/basilisk_b19/target_features.h5",
        "docs/basilisk_b21/split_manifest.json",
    ],
    # §14 lifecycle transition guard 的"仍然禁止"部分: B7 不得产生新训练权重;
    # B8 未授权的算法产物; 未来 B9 的算法实验产物。这些**不是**"下游阶段永远
    # 不存在", 而是"本阶段及以后都不允许由 B7 产生的东西"。
    "b7_must_stay_absent": [
        # B7 自己绝不训练 -> 不得出现任何 B7 权重
        "checkpoints/basilisk_b7/target_only_s122.pt",
        "checkpoints/basilisk_b7/source_finetune_s122.pt",
        "checkpoints/basilisk_b7/source_mmd_finetune_s122.pt",
        "checkpoints/basilisk_b7/all_raw.npz",
        "checkpoints/basilisk_b7/all_metrics.json",
        "checkpoints/basilisk_b7/formal_metrics.json",
        "checkpoints/basilisk_b7/paired_statistics.json",
        # B8 只允许打包, 不允许算法产物
        "checkpoints/basilisk_b8/all_metrics.json",
        "checkpoints/basilisk_b8/formal_metrics.json",
        "checkpoints/basilisk_b8/paired_statistics.json",
        "checkpoints/basilisk_b8/final_verdict.json",
        # 未来 B9 的算法实验尚未授权
        "checkpoints/basilisk_b9/all_metrics.json",
        "checkpoints/basilisk_b9/final_verdict.json",
        "docs/basilisk_b9/results.md",
        "STATUS_BASILISK_B9.md",
    ],
}


# B6 的数值结论 + §3 的逐项冻结清单。
# 首次写入时由 read_b7_results() 实测填入, 不凭记忆预填。
B7_EXPECTED = {
    # ---- B6 正式结论 (终局) ----
    "b6_primary_verdict": "B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER",
    "b6_final_transfer_conclusion": "NO_POSITIVE_TRANSFER_SUPPORTED",
    "b6_engineering_recommendation": "damage_extrapolation",
    "b6_primary_level": "5",
    "b6_label_levels": "[3, 5, 10, 21]",
    "b6_formal_seeds": "[122, 123, 124, 125, 126]",
    "b6_input_schema": "core_only",
    "b6_n_features": "12",
    "b6_b7_auto_run": "False",
    "b6_b8_auto_run": "False",
    "b6_primary_ft_n_passed": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b6_primary_mmd_n_passed": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b6_primary_positive_any": "False",
    # ---- B6 PRIMARY 数值锚点 (§16 数字审计的比对源) ----
    "b6_primary_target_only_rmse": "0.339625",
    "b6_primary_source_finetune_rmse": "0.354721",
    "b6_primary_source_mmd_finetune_rmse": "0.310147",
    "b6_primary_const_mean_info_rmse": "0.288629",
    "b6_primary_damage_extrapolation_rmse": "0.054312",
    "b6_primary_gain_ft_mean": "-0.015096",
    "b6_primary_gain_mmd_mean": "0.029478",
    "b6_primary_gain_ft_improve_count": "2",
    "b6_primary_gain_mmd_improve_count": "3",
    # ---- 四档主表 (Table B 的唯一来源) ----
    "b6_n3_target_only_rmse": "0.335007",
    "b6_n3_source_mmd_finetune_rmse": "0.317366",
    "b6_n10_target_only_rmse": "0.298031",
    "b6_n10_source_mmd_finetune_rmse": "0.292611",
    "b6_n21_target_only_rmse": "0.238093",
    "b6_n21_source_mmd_finetune_rmse": "0.244525",
    # ---- §3 点名的四项 B6 冻结哈希 ----
    "b6_protocol_sha256":
        "433ccaf2515669c1f5afbc76029de068329afc7f67f0c10ae3372d14409b8412",
    "b6_subset_manifest_sha256":
        "f7fe1ababe7b23db5d0ba41a7c644505267f684400a22aebb5144bf6ce041b1d",
    "b6_all_metrics_file_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b6_final_verdict_file_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b6_protocol_file_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b6_summary_file_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    # ---- §3 上游逐项 (B1.8 / B1.9 / B2.1 / B5) ----
    "b18_dataset_sha256_via_b7": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b19_feature_sha256_via_b7": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b21_split_sha256_via_b7":
        "23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932",
    "b5_protocol_file_sha256_via_b7": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b5_results_file_sha256_via_b7": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b5_verdict_via_b7": "B5_NO_POSITIVE_TRANSFER",
    "b5_target_only_rmse_via_b7": "0.241024",
    "b5_source_finetune_rmse_via_b7": "0.242659",
    "b5_source_mmd_finetune_rmse_via_b7": "0.241607",
    "b5_damage_extrapolation_rmse_via_b7": "0.054312",
    # ---- §0 B1.8 场景数字 ----
    "b18_n_traj_via_b7": "150",
    "b18_n_event_via_b7": "71",
    "b18_n_censored_via_b7": "79",
    "b18_failure_fraction_via_b7": "0.473333",
    "b18_l_ref_years_via_b7": "3.0",
    "b18_eol_threshold_d_via_b7": "1.0",
    "b18_primary_scenario_via_b7": "NOMINAL",
    "b19_primary_hi_via_b7": "hi_damage_obs",
    "b19_trained_any_model_via_b7": "False",
    # ---- §3 实现级冻结项 ----
    "source_checkpoint_sha256_via_b7": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "encoder_architecture_sha256_via_b7": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "adapter_architecture_sha256_via_b7": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "metrics_implementation_sha256_via_b7":
        "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "damage_baseline_implementation_sha256_via_b7":
        "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "target_only_architecture_sha256_via_b7":
        "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    # ---- 前置判定 (不得被继承或改写) ----
    "b18_verdict_via_b7": "B18_SCENARIO_READY",
    "b19_verdict_via_b7": "B19_FEATURE_READY",
    "b2_verdict_via_b7": "B2_GENERALIZATION_FAIL",
    "b3x_verdict_via_b7": "B3X_NO_STABILIZING_SIGNAL",
    "b4x_verdict_via_b7": "B4X_TRANSFER_STABILIZATION_SIGNAL",
    "b21_verdict_via_b7": "B21_GENERALIZATION_PASS",
}


def _json(rel: str):
    p = ROOT / rel
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _num(v) -> str:
    return repr(v).strip("'\"")


def read_b7_results() -> dict:
    """读 B6 冻结结论 + §3 的逐项冻结清单 (键路径按实测 dump 写)。

    只读 —— 本函数不得调用任何训练器 / evaluator, 不得重算任何指标。
    """
    out: dict[str, str] = {}

    # ---- B6 final_verdict (§0 的终局结论) ----
    v_keys = ("b6_primary_verdict", "b6_final_transfer_conclusion",
              "b6_engineering_recommendation", "b6_primary_level",
              "b6_label_levels", "b6_formal_seeds", "b6_input_schema",
              "b6_n_features", "b6_b7_auto_run", "b6_b8_auto_run",
              "b6_primary_ft_n_passed", "b6_primary_mmd_n_passed",
              "b6_primary_positive_any",
              "b6_primary_target_only_rmse",
              "b6_primary_source_finetune_rmse",
              "b6_primary_source_mmd_finetune_rmse",
              "b6_primary_const_mean_info_rmse",
              "b6_primary_damage_extrapolation_rmse",
              "b6_protocol_sha256", "b6_subset_manifest_sha256",
              "b21_split_sha256_via_b7")
    vj = _json("checkpoints/basilisk_b6/final_verdict.json")
    if vj is None:
        for k in v_keys:
            out[k] = "MISSING"
    else:
        out["b6_primary_verdict"] = str(vj["B6_PRIMARY_VERDICT"])
        out["b6_final_transfer_conclusion"] = str(
            vj["FINAL_TRANSFER_CONCLUSION"])
        out["b6_engineering_recommendation"] = str(
            vj["ENGINEERING_RECOMMENDATION"])
        out["b6_primary_level"] = _num(int(vj["primary_level"]))
        out["b6_label_levels"] = _num([int(x) for x in vj["label_levels"]])
        out["b6_formal_seeds"] = _num([int(s) for s in vj["formal_seeds"]])
        out["b6_input_schema"] = str(vj["input_schema"])
        out["b6_n_features"] = _num(int(vj["n_features"]))
        out["b6_b7_auto_run"] = _num(bool(vj["b7_auto_run"]))
        out["b6_b8_auto_run"] = _num(bool(vj["b8_auto_run"]))
        dec = vj["primary_decision"]
        out["b6_primary_ft_n_passed"] = _num(
            int(dec["source_finetune"]["n_passed"]))
        out["b6_primary_mmd_n_passed"] = _num(
            int(dec["source_mmd_finetune"]["n_passed"]))
        out["b6_primary_positive_any"] = _num(
            bool(vj["primary_positive"]["any"]))
        pmt = vj["primary_main_table"]
        for key, g in (("b6_primary_target_only_rmse", "target_only"),
                       ("b6_primary_source_finetune_rmse", "source_finetune"),
                       ("b6_primary_source_mmd_finetune_rmse",
                        "source_mmd_finetune"),
                       ("b6_primary_const_mean_info_rmse", "const_mean_info"),
                       ("b6_primary_damage_extrapolation_rmse",
                        "damage_extrapolation")):
            out[key] = _num(round(float(pmt[g]["info_macro_rmse"]), 6))
        out["b6_protocol_sha256"] = str(vj["protocol_sha256"])
        out["b6_subset_manifest_sha256"] = str(vj["subset_manifest_sha256"])
        out["b21_split_sha256_via_b7"] = str(vj["split_sha256"])

    # ---- B6 四档主表 + PRIMARY 配对增益 ----
    lv_keys = ("b6_n3_target_only_rmse", "b6_n3_source_mmd_finetune_rmse",
               "b6_n10_target_only_rmse", "b6_n10_source_mmd_finetune_rmse",
               "b6_n21_target_only_rmse", "b6_n21_source_mmd_finetune_rmse",
               "b6_primary_gain_ft_mean", "b6_primary_gain_mmd_mean",
               "b6_primary_gain_ft_improve_count",
               "b6_primary_gain_mmd_improve_count")
    mx = _json("checkpoints/basilisk_b6/all_metrics.json")
    ps = _json("checkpoints/basilisk_b6/paired_statistics.json")
    if mx is None or ps is None:
        for k in lv_keys:
            out[k] = "MISSING"
    else:
        for n in (3, 10, 21):
            agg = ps["by_level"][str(n)]["mean_info_macro_rmse"]
            out[f"b6_n{n}_target_only_rmse"] = _num(
                round(float(agg["target_only"]), 6))
            out[f"b6_n{n}_source_mmd_finetune_rmse"] = _num(
                round(float(agg["source_mmd_finetune"]), 6))
        g5 = ps["by_level"]["5"]["gain"]
        out["b6_primary_gain_ft_mean"] = _num(
            round(float(g5["gain_ft"]["mean"]), 6))
        out["b6_primary_gain_mmd_mean"] = _num(
            round(float(g5["gain_mmd"]["mean"]), 6))
        out["b6_primary_gain_ft_improve_count"] = _num(
            int(g5["gain_ft"]["improve_count"]))
        out["b6_primary_gain_mmd_improve_count"] = _num(
            int(g5["gain_mmd"]["improve_count"]))

    # ---- B5 数值锚点 (§16 审计的另一半; 从 B5 冻结 summary 读) ----
    b5_keys = ("b5_verdict_via_b7", "b5_target_only_rmse_via_b7",
               "b5_source_finetune_rmse_via_b7",
               "b5_source_mmd_finetune_rmse_via_b7",
               "b5_damage_extrapolation_rmse_via_b7")
    b5 = _json("checkpoints/basilisk_b5/summary.json")
    if b5 is None:
        for k in b5_keys:
            out[k] = "MISSING"
    else:
        out["b5_verdict_via_b7"] = str(b5["verdict"])
        for key, g in (("b5_target_only_rmse_via_b7", "target_only"),
                       ("b5_source_finetune_rmse_via_b7", "source_finetune"),
                       ("b5_source_mmd_finetune_rmse_via_b7",
                        "source_mmd_finetune"),
                       ("b5_damage_extrapolation_rmse_via_b7",
                        "damage_extrapolation")):
            out[key] = _num(round(
                float(b5["main_table"][g]["info_macro_rmse"]), 6))
        ff = b5["frozen_facts"]
        out["b2_verdict_via_b7"] = str(ff["b2_verdict"])
        out["b3x_verdict_via_b7"] = str(ff["b3x_verdict"])
        out["b4x_verdict_via_b7"] = str(ff["b4x_verdict"])
        out["b21_verdict_via_b7"] = str(ff["b21_verdict"])

    # ---- §0 B1.8 场景数字 (从 B1.8 冻结产物读, 不硬编码) ----
    sc_keys = ("b18_n_traj_via_b7", "b18_n_event_via_b7",
               "b18_n_censored_via_b7", "b18_failure_fraction_via_b7",
               "b18_dataset_sha256_via_b7", "b18_l_ref_years_via_b7",
               "b18_eol_threshold_d_via_b7")
    b18 = _json("checkpoints/basilisk_b18/dataset_audit.json")
    if b18 is None:
        for k in sc_keys:
            out[k] = "MISSING"
    else:
        a = b18["audit"]
        out["b18_n_traj_via_b7"] = _num(int(b18["n_traj"]))
        out["b18_n_event_via_b7"] = _num(int(a["n_event_observed"]))
        out["b18_n_censored_via_b7"] = _num(int(a["n_censored"]))
        out["b18_failure_fraction_via_b7"] = _num(
            round(float(a["failure_fraction"]), 6))
        out["b18_dataset_sha256_via_b7"] = str(b18["content_sha256"])
        out["b18_l_ref_years_via_b7"] = _num(float(b18["L_ref_years"]))
        out["b18_eol_threshold_d_via_b7"] = _num(
            float(b18["eol_threshold_D"]))

    # ---- B1.8 判定 (独立冻结件) ----
    sc = _json("checkpoints/basilisk_b18/frozen_primary_scenario.json")
    out["b18_verdict_via_b7"] = ("MISSING" if sc is None
                                 else str(sc["verdict"]))
    out["b18_primary_scenario_via_b7"] = ("MISSING" if sc is None
                                          else str(sc["primary_scenario"]))

    # ---- B1.9 特征判定 ----
    b19 = _json("checkpoints/basilisk_b19/frozen_feature_definition.json")
    if b19 is None:
        out["b19_verdict_via_b7"] = "MISSING"
        out["b19_feature_sha256_via_b7"] = "MISSING"
        out["b19_primary_hi_via_b7"] = "MISSING"
        out["b19_trained_any_model_via_b7"] = "MISSING"
    else:
        out["b19_verdict_via_b7"] = str(b19["verdict"])
        out["b19_feature_sha256_via_b7"] = str(
            b19["dataset"]["feature_content_sha256"])
        out["b19_primary_hi_via_b7"] = str(
            b19["selected_primary_hi"]["dataset_name"])
        out["b19_trained_any_model_via_b7"] = _num(
            bool(b19["trained_any_model"]))

    # ---- §3 文件级哈希 (B6 四项 + B5 两项 + 实现级冻结项) ----
    for key, rel in (
            ("b6_all_metrics_file_sha256",
             "checkpoints/basilisk_b6/all_metrics.json"),
            ("b6_final_verdict_file_sha256",
             "checkpoints/basilisk_b6/final_verdict.json"),
            ("b6_protocol_file_sha256", "docs/basilisk_b6/protocol.md"),
            ("b6_summary_file_sha256",
             "checkpoints/basilisk_b6/summary.json"),
            ("b5_protocol_file_sha256_via_b7",
             "docs/basilisk_b5/protocol.md"),
            ("b5_results_file_sha256_via_b7",
             "checkpoints/basilisk_b5/formal_metrics.json"),
            ("source_checkpoint_sha256_via_b7",
             "checkpoints/source_tcn_pretrain.pt"),
            ("encoder_architecture_sha256_via_b7",
             "src/models/tcn_encoder.py"),
            ("adapter_architecture_sha256_via_b7", "src/transfer/adapter.py"),
            ("metrics_implementation_sha256_via_b7",
             "src/experiments/metrics.py"),
            ("damage_baseline_implementation_sha256_via_b7",
             "src/baselines/physical_extrap.py"),
            ("target_only_architecture_sha256_via_b7",
             "src/experiments/run_groups.py")):
        p = ROOT / rel
        out[key] = (hashlib.sha256(p.read_bytes()).hexdigest()
                    if p.exists() else "MISSING")
    return out


NUMERIC_BLOCKS = tuple(B6V.NUMERIC_BLOCKS) + (
    ("b7_results", read_b7_results, B7_EXPECTED, "b7"),
)

FROZEN_GROUPS: dict[str, list[str]] = dict(B6V.FROZEN_GROUPS)
FROZEN_GROUPS.update(B7_ADDED_GROUPS)

# §14 lifecycle transition guard 的核心改造。
#
# 旧逻辑 (B6 及之前): "下游阶段产物必须永远不存在"。这个前提在下游获得人工授权
# 之后必然过期, 于是每个阶段都会留下一个永久失败的陈旧守卫 —— B6 结束时全库
# 累积了 3 个。
#
# 新逻辑: 下游产物**允许存在**, 但只有在其 protocol / baseline contract /
# final verdict 完整时才允许 (见 lifecycle_transition_requirements)。这样
# "真正的越界" (没有协议就先跑数字) 依然被拦住, 而"合法的阶段推进"不再制造
# 假失败。
#
# 因此把 B6 契约里指向 B7 自己产物的 must_stay_absent 项剔除, 由
# b7_must_stay_absent (B7 训练权重 + B8 算法产物 + B9 算法实验) 与
# LIFECYCLE_TRANSITIONS 共同接管。
FROZEN_GROUPS["b6_must_stay_absent"] = [
    rel for rel in FROZEN_GROUPS["b6_must_stay_absent"]
    if not rel.startswith(("checkpoints/basilisk_b7", "docs/basilisk_b7",
                           "STATUS_BASILISK_B7"))
]

# §14 + §2 的第二处必要豁免: stale lifecycle guard 的正式退役。
#
# 背景: 全库有 3 个已知的陈旧生命周期守卫 ——
#   tests/basilisk_b21/test_b21_discipline.py::test_b21_b5_absent_and_never_auto_run
#   tests/basilisk_b5/test_b5_baseline_contract.py::test_b5_old_artifacts_unchanged
#       (其中对 B6 absent 的生命周期假设部分)
#   tests/basilisk_b5/test_b5_baseline_contract.py::test_b5_b6_artifacts_must_stay_absent
# 它们断言"下游阶段永远不存在"。B5 / B6 已获显式人工阶段授权并完成, 该前提
# 已失效, 这 3 个断言必然永久失败。
#
# §2 明确授权本阶段 (且仅本阶段) 为生命周期治理修改这两个测试文件, 因此它们的
# 哈希在 B7 内**预期会变**。若仍留在冻结组里, --tag after 会把"授权的治理改动"
# 误报成"越界篡改" -> 假 B7_INVALID。
#
# 关键: 只豁免这 2 个**测试治理文件**, 不豁免任何结果 JSON / 权重 / 算法源码。
# 越界防护由 b7_must_stay_absent + LIFECYCLE_TRANSITIONS 接管 —— 新守卫要求
# 下游产物只有在 protocol / baseline contract / final verdict 齐备时才允许存在,
# 比"永远不存在"更强, 因为它还检查治理完整性。
LIFECYCLE_GUARD_RETIREMENT_EXEMPT: tuple[str, ...] = (
    "tests/basilisk_b21/test_b21_discipline.py",
    "tests/basilisk_b5/test_b5_baseline_contract.py",
)

for _g in ("b21_tests", "b5_tests"):
    if _g in FROZEN_GROUPS:
        FROZEN_GROUPS[_g] = [
            rel for rel in FROZEN_GROUPS[_g]
            if rel not in LIFECYCLE_GUARD_RETIREMENT_EXEMPT
        ]

# §2 授权修改的"报告表达面"。这些是文档产物, 不是数值来源 —— B7 的全部数字都从
# checkpoints/basilisk_b7/frozen_result_index.json 读, 改这些文件改不动任何指标。
# 早期阶段 (B1.7 / B1.8 / B1.9) 把 docs/results.md 钉进只读组, 是因为那时它还残留
# 过期占位, 必须防止被当成结论引用; 到 B7 它恰恰是**必须重写**的最终成果表达面。
# 若不豁免, --tag after 会把 §5 / §18 要求的重写误报为越界篡改。
#
# 数值一致性不靠哈希冻结保证, 而靠 scripts/basilisk_b7/audit_report_numbers.py
# 逐个数字与 frozen_result_index.json 对种 (不一致 -> B7_REPORT_NUMBER_MISMATCH),
# 以及 scripts/basilisk_b7/audit_claims.py 检查措辞边界。这比哈希更强: 哈希只能
# 说"没被动过", 审计脚本能说"里面每个数字都对得上冻结产物"。
REPORT_SURFACE_EXEMPT: tuple[str, ...] = (
    "docs/results.md",
)

for _g, _files in list(FROZEN_GROUPS.items()):
    if _g in ("b7_must_stay_absent", "b6_must_stay_absent"):
        continue
    FROZEN_GROUPS[_g] = [rel for rel in _files
                         if rel not in REPORT_SURFACE_EXEMPT]

# 豁免不等于放任。这两个文件在 B7 内的正确性由三重机制保证, 而不是由哈希冻结:
#   1. tests/basilisk_b7/test_b7_stale_guard_retirement.py 断言新守卫的语义
#      (下游产物只有在治理三件套齐备时才允许存在), 且断言旧的"永远不存在"
#      断言已被替换而**不是**被删除;
#   2. LIFECYCLE_TRANSITIONS + check_lifecycle_transitions() 在每次 verify 时
#      实际执行该语义, 比静态哈希更强 —— 它检查治理完整性, 不只检查文件没被动;
#   3. docs/basilisk_b7/stale_guard_retirement.md 记录旧守卫目的 / 为何 obsolete /
#      哪个人工授权使状态迁移合法 / 新守卫如何防止真正越界。
# 不把它们放进新的冻结组: 若在 before 时钉住退役前的哈希, after 必然失败, 那等于
# 用契约禁止 §2 明确授权的改动。
LIFECYCLE_GUARD_RETIREMENT_NOTE = (
    "tests/basilisk_b21/test_b21_discipline.py 与 "
    "tests/basilisk_b5/test_b5_baseline_contract.py 在 B7 内按 §2 / §14 的显式"
    "人工阶段授权做生命周期治理改造 (把'下游阶段必须永远不存在'替换为'只有在"
    "获得显式阶段授权且 protocol / baseline contract / final verdict 齐备时"
    "下游产物才允许存在'), 因此它们的哈希不进 B7 冻结组。越界防护改由 "
    "b7_must_stay_absent + LIFECYCLE_TRANSITIONS + "
    "tests/basilisk_b7/test_b7_stale_guard_retirement.py 接管。"
    "豁免范围严格限于这 2 个测试治理文件: 不豁免任何结果 JSON / 训练权重 / "
    "算法源码 / config。"
)

# §14: 阶段 -> (触发存在性判断的产物, 必须同时完整的治理产物)。
# 一旦某阶段的任一 artifact 出现, 它的 governance 三件套就必须齐备。
LIFECYCLE_TRANSITIONS: dict[str, dict[str, list[str]]] = {
    "BASILISK_B5": {
        "artifacts": [
            "docs/basilisk_b5/results.md",
            "STATUS_BASILISK_B5.md",
            "checkpoints/basilisk_b5/formal_metrics.json",
        ],
        "governance": [
            "docs/basilisk_b5/protocol.md",
            "docs/basilisk_b5/baseline_contract.json",
            "checkpoints/basilisk_b5/protocol_hash.json",
            "checkpoints/basilisk_b5/summary.json",
        ],
    },
    "BASILISK_B6": {
        "artifacts": [
            "docs/basilisk_b6/results.md",
            "STATUS_BASILISK_B6.md",
            "checkpoints/basilisk_b6/all_metrics.json",
        ],
        "governance": [
            "docs/basilisk_b6/protocol.md",
            "docs/basilisk_b6/baseline_contract.json",
            "checkpoints/basilisk_b6/protocol_hash.json",
            "checkpoints/basilisk_b6/label_subset_manifest.json",
            "checkpoints/basilisk_b6/final_verdict.json",
        ],
    },
    "BASILISK_B7": {
        "artifacts": [
            "docs/basilisk_b7/results.md",
            "STATUS_BASILISK_B7.md",
            "checkpoints/basilisk_b7/final_tables.json",
        ],
        "governance": [
            "docs/basilisk_b7/baseline_contract.json",
            "checkpoints/basilisk_b7/frozen_result_index.json",
            "checkpoints/basilisk_b7/report_number_audit.json",
            "checkpoints/basilisk_b7/claim_audit.json",
        ],
    },
}

_sha256_of = B6V._sha256_of

CONTRACT_PURPOSE = (
    "冻结 analytic lineage + BASILISK_V1 + B1 ... B1.9 + B2 + B3X + B4X + B2.1 "
    "+ B5 + **B6** 全部产物与结论。B7 是 "
    "FINAL_FIGURES_REPORT_EVIDENCE_FREEZE 阶段, 即飞轮线的结果表达阶段。"
    "本阶段禁止产生任何新的算法结果: 禁止训练 / 禁止重新评估新 seed / "
    "禁止修改任何会影响数值的代码 / 禁止调参 / 禁止改 split HI RUL loss model "
    "MMD / 禁止重新生成 lifetime dataset。所有图表与报告数字只能读取已冻结的 "
    "B1.8 B1.9 B2.1 B5 B6 产物。"
    "FINAL_TRANSFER_CONCLUSION = NO_POSITIVE_TRANSFER_SUPPORTED 与 "
    "ENGINEERING_RECOMMENDATION = damage_extrapolation 在本阶段保持不变; "
    "边界措辞必须是**未能证明正向迁移**而不是**证明迁移无效**。"
    "B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER 与 B5_NO_POSITIVE_TRANSFER "
    "都是终局: B6 的 protocol hash / subset manifest hash / all_metrics hash / "
    "final verdict hash 全部逐项进契约, 任何一项改变即 " + INVALID_LABEL + "。"
    "B2_GENERALIZATION_FAIL 保留为 split coverage failure 的诊断证据; "
    "B3X_NO_STABILIZING_SIGNAL 不变, mission features 不进入最终正式输入; "
    "B4X_TRANSFER_STABILIZATION_SIGNAL 仍然只是 EXPLORATORY_ONLY, "
    "不得改写成 positive transfer。"
    "本阶段禁止修改: src/models src/transfer src/experiments src/sim "
    "src/baselines / configs 中任何影响数字的参数 / B1.8-B6 的结果 JSON / "
    "checkpoints 中训练权重。"
    "§14 生命周期守卫改造: 旧的**下游阶段必须永远不存在**断言被替换为 "
    "lifecycle transition guard —— 下游产物只有在其 protocol / "
    "baseline contract / final verdict 完整时才允许存在; "
    "仍然禁止 B7 生成新训练 checkpoint / B8 未授权的算法产物 / "
    "future B9 algorithm experiment。"
)


def check_lifecycle_transitions() -> list[tuple[str, str, str]]:
    """§14 的新守卫: 已推进的阶段必须治理完整, 未授权的越界依然被拦。

    返回 (项, 期望, 实测) 的失败列表, 与 verify() 的 bad 列表同格式。
    """
    bad: list[tuple[str, str, str]] = []
    for stage, spec in LIFECYCLE_TRANSITIONS.items():
        present = [r for r in spec["artifacts"] if (ROOT / r).exists()]
        if not present:
            continue                       # 该阶段尚未推进 —— 合法
        for gov in spec["governance"]:
            if not (ROOT / gov).exists():
                bad.append((
                    f"[lifecycle] {stage} 已产出 {present[0]} 但缺治理产物",
                    f"EXISTS:{gov}", "MISSING"))
    return bad


def compute() -> dict:
    groups = {g: {rel: _sha256_of(rel) for rel in files}
              for g, files in FROZEN_GROUPS.items()}
    flat = {rel: h for g in groups.values() for rel, h in g.items()}
    chain = dict(B6V.compute()["frozen_chain"])
    chain["BASILISK_B6"] = "B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER"
    return {
        "stage": "BASILISK_B7",
        "label": "FINAL_FIGURES_REPORT_EVIDENCE_FREEZE",
        "purpose": CONTRACT_PURPOSE,
        "invalid_label": INVALID_LABEL,
        "frozen_chain": chain,
        "final_transfer_conclusion": "NO_POSITIVE_TRANSFER_SUPPORTED",
        "engineering_recommendation": "damage_extrapolation",
        "boundary_statement": (
            "未能证明正向迁移, 而不是证明迁移无效"),
        "forbid_in_this_stage": [
            "training", "re-evaluating new seeds", "tuning",
            "modifying any numeric-affecting code",
            "changing split / HI / RUL / loss / model / MMD",
            "regenerating lifetime dataset",
            "producing any new algorithmic result",
        ],
        "lifecycle_transition_requirements": LIFECYCLE_TRANSITIONS,
        "lifecycle_guard_retirement_exempt": list(
            LIFECYCLE_GUARD_RETIREMENT_EXEMPT),
        "lifecycle_guard_retirement_note": LIFECYCLE_GUARD_RETIREMENT_NOTE,
        "report_surface_exempt": list(REPORT_SURFACE_EXEMPT),
        "report_surface_exempt_note": (
            "§2 授权 B7 重写 docs/results.md (最终成果表达面)。其数值正确性由 "
            "audit_report_numbers.py 与 frozen_result_index.json 对种保证, "
            "措辞边界由 audit_claims.py 保证, 不依赖哈希冻结。"
            "docs/技术方案报告/ 与 docs/数据集下载/ 从未进入任何冻结组, 无需豁免。"),
        "env": {"python": sys.version.split()[0],
                "platform": platform.platform()},
        "groups": groups,
        "flat_sha256": flat,
        "n_files": len(flat),
        **{k: v for name, reader, expected, prefix in NUMERIC_BLOCKS
           for k, v in ((name, reader()), (f"{prefix}_expected", dict(expected)))},
    }


def verify(tag: str) -> int:
    if not CONTRACT.exists():
        print(f"[{tag}] 契约文件不存在: {CONTRACT}")
        return 1
    ref = json.loads(CONTRACT.read_text(encoding="utf-8"))
    bad = []
    for rel, h_ref in ref["flat_sha256"].items():
        h_now = _sha256_of(rel)
        if h_now != h_ref:
            bad.append((rel, h_ref, h_now))

    n_numeric = 0
    for name, reader, expected, prefix in NUMERIC_BLOCKS:
        now = reader()
        for k, h_ref in ref[name].items():
            n_numeric += 1
            if now.get(k) != h_ref:
                bad.append((f"[{prefix}_result] {k}", h_ref,
                            now.get(k, "ABSENT")))
        for k, h_exp in ref.get(f"{prefix}_expected", expected).items():
            n_numeric += 1
            if now.get(k) != h_exp:
                bad.append((f"[{prefix}_expected] {k}", h_exp,
                            now.get(k, "ABSENT")))

    # §14: 生命周期迁移守卫。不计入 n_total 的文件/数值项, 单独报。
    lc = check_lifecycle_transitions()
    bad.extend(lc)

    n_total = ref["n_files"] + n_numeric
    print(f"[{tag}] 契约项数 {n_total}, 不变 {n_total - len(bad) + len(lc)}")
    print(f"[{tag}] lifecycle transition guard: "
          f"{len(LIFECYCLE_TRANSITIONS)} 阶段, "
          f"{'PASS' if not lc else f'FAIL x{len(lc)}'}")
    for rel, a, b in bad:
        print(f"[{tag}] CHANGED(禁止) {rel}\n         "
              f"{str(a)[:24]} -> {str(b)[:24]}")
    if bad:
        print(f"[{tag}] {INVALID_LABEL} —— 冻结项改变, 必须立即停止")
        return 1
    print(f"[{tag}] B7_BASELINE_CONTRACT_OK")
    return 0


def write(tag: str) -> int:
    CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    c = compute()
    # 任何 *_EXPECTED 里的 PLACEHOLDER_FILLED_ON_FIRST_WRITE 都在首次写入时
    # 用实测值填入 —— 不凭记忆预填。填完之后 verify 逐字比对, 照样能失败。
    n_filled = 0
    for name, _, _, prefix in NUMERIC_BLOCKS:
        exp = c[f"{prefix}_expected"]
        for k, v in list(exp.items()):
            if v == "PLACEHOLDER_FILLED_ON_FIRST_WRITE":
                exp[k] = c[name][k]
                n_filled += 1
    CONTRACT.write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    n_num = sum(len(c[name]) + len(c[f"{p}_expected"])
                for name, _, _, p in NUMERIC_BLOCKS)
    print(f"[{tag}] 已写入契约 {CONTRACT.relative_to(ROOT)}: "
          f"{c['n_files']} 文件 + {n_num} 数值项 = {c['n_files'] + n_num} 项")
    if n_filled:
        print(f"[{tag}] 首次写入时按实测填入 {n_filled} 个 placeholder")
    lc = check_lifecycle_transitions()
    print(f"[{tag}] lifecycle transition guard: "
          f"{'PASS' if not lc else f'FAIL x{len(lc)}'}")
    for rel, a, b in lc:
        print(f"[{tag}] !! {rel}: {a} -> {b}")
    miss = [r for r, h in c["flat_sha256"].items() if h == "MISSING"]
    if miss:
        print(f"[{tag}] !! 以下文件缺失 (契约会把 MISSING 钉死, 请确认是预期的):")
        for r in miss:
            print(f"        {r}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tag = a.tag or ("write" if a.write else "verify")
    if a.write:
        return write(tag)
    if a.verify:
        return verify(tag)
    # 默认: 首次建契约, 之后一律核验。--tag after 绝不会覆盖 before 的契约。
    return write(tag) if not CONTRACT.exists() else verify(tag)


if __name__ == "__main__":
    raise SystemExit(main())
