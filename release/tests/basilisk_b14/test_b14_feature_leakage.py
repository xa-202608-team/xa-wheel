# -*- coding: utf-8 -*-
"""BASILISK-B1.4 §17-§20: 数据/特征门与真值泄漏的测试。

核心命名测试:
  - test_feature_no_truth_leakage

本阶段 verdict = FAIL 时 §16/§18 禁止生成数据集与特征, 因此这些测试大多以
skip 呈现 —— 但 "FAIL 却存在产物" 必须是硬失败, 不能靠 skip 掩过去。
"""
import io
import json
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

XT_COLS = ["I_m", "omega", "T", "T_cmd", "sigma_Im", "b_hat_selfcal",
           "dT", "omega_err"]
TRUTH_FIELDS = ["b_true", "Kt", "Tc", "b0", "omega0"]


def _p(*parts):
    return os.path.join(ROOT, *parts)


# --------------------------------------------------------------------------
# §16/§17: 数据集只有在 PASS 后才允许存在
# --------------------------------------------------------------------------
def test_dataset_only_exists_after_pass(verdict_rec, dataset_rec):
    """§16: FAIL 时不得有 150 轨迹数据集; PASS 时必须有 Final Data Gate 记录。"""
    if verdict_rec is None:
        pytest.skip("calibration_verdict.json 未生成")
    final_h5 = _p("data", "sim", "wheel_basilisk_b14", "final", "wheel_all.h5")
    if verdict_rec["verdict"] == "B14_CALIBRATION_FAIL":
        assert dataset_rec is None, "FAIL 却生成了 dataset_summary.json"
        assert not os.path.exists(final_h5), "FAIL 却写出了正式数据集"
        return
    assert dataset_rec is not None
    assert dataset_rec["n_traj"] == 150
    assert dataset_rec["verdict"] in ("B14_DATA_READY", "B14_DATA_NOT_READY")


def test_features_only_exist_after_data_ready(dataset_rec, feature_rec):
    """§18: 只有 B14_DATA_READY 才允许 build features。"""
    feat_h5 = _p("data", "features", "wheel", "basilisk_b14",
                 "target_features.h5")
    if dataset_rec is None or dataset_rec.get("verdict") != "B14_DATA_READY":
        assert feature_rec is None, "数据未 READY 却生成了特征汇总"
        assert not os.path.exists(feat_h5), "数据未 READY 却写出了特征 h5"
        pytest.skip("数据未 READY, §18 禁止构建特征")
    assert feature_rec is not None


# --------------------------------------------------------------------------
# §19/§21: test_feature_no_truth_leakage
# --------------------------------------------------------------------------
def test_feature_no_truth_leakage(feature_rec, b14_cfg):
    """§19: x_T 核心 8 列不变; 真值字段与 mission_features 不得进入 x_T。"""
    # config 层面先钉死 (即使特征还没生成, 这条也必须成立)
    feats = b14_cfg["features"]
    assert list(feats["xt_cols"]) == XT_COLS, "x_T 核心 8 列被改动"
    assert feats["mission_features_in_xt"] is False
    assert list(feats["truth_fields"]) == TRUTH_FIELDS
    # 真值字段绝不能出现在 x_T 里
    for t in TRUTH_FIELDS:
        assert t not in XT_COLS, "真值字段 %s 进入了 x_T" % t
    if feature_rec is None:
        pytest.skip("feature_summary.json 未生成 (标定未 PASS)")
    # 特征已生成时, 逐列复核实际写出的列
    assert list(feature_rec["xt_cols"]) == XT_COLS
    for t in TRUTH_FIELDS:
        assert t not in feature_rec["xt_cols"]
    assert feature_rec["mission_features_in_xt"] is False
    assert feature_rec["nan_count"] == 0
    assert feature_rec["inf_count"] == 0


def test_b_hat_selfcal_is_observable_not_truth(b14_cfg):
    """`b_hat_selfcal` 必须是遥测派生的自校准估计, 不是 b_true 的别名。"""
    feats = b14_cfg["features"]
    assert "b_hat_selfcal" in feats["xt_cols"]
    assert "b_true" not in feats["xt_cols"]
    # 特征构建脚本若存在, 不得直接把 b_true 赋给 b_hat_selfcal
    src = _p("scripts", "basilisk_b14", "build_features.py")
    if os.path.exists(src):
        with io.open(src, encoding="utf-8") as fh:
            txt = fh.read()
        assert "b_hat_selfcal\"] = b_true" not in txt
        assert "b_hat_selfcal'] = b_true" not in txt


# --------------------------------------------------------------------------
# §20: 本阶段禁止训练
# --------------------------------------------------------------------------
def test_no_training_artifacts_in_b14():
    """§20: 即使全过也禁止训练。b14 命名空间不得出现权重/训练日志。"""
    ck = _p("checkpoints", "basilisk_b14")
    if not os.path.isdir(ck):
        pytest.skip("checkpoints/basilisk_b14 不存在")
    for fn in os.listdir(ck):
        low = fn.lower()
        for ext in (".pt", ".pth", ".ckpt", ".onnx", ".safetensors"):
            assert not low.endswith(ext), "出现训练权重: %s" % fn
        for tok in ("train_log", "metrics", "rul_", "transfer_"):
            assert tok not in low, "出现训练/迁移产物: %s" % fn


def test_stage_stops_at_declared_boundary(verdict_rec):
    """FAIL 时必须显式记录停止范围 —— 不得留下"可以接着往下走"的暗示。"""
    if verdict_rec is None:
        pytest.skip("calibration_verdict.json 未生成")
    if verdict_rec["verdict"] != "B14_CALIBRATION_FAIL":
        pytest.skip("非 FAIL 路径")
    note = verdict_rec["note"]
    for token in ("§15", "§11", "§16", "§18", "§20"):
        assert token in note, "停止说明缺少 %s" % token
    assert verdict_rec["frozen_calibration_written"] is False
    assert verdict_rec["q0_modified_after_probe"] is False
    assert verdict_rec["probe_reran"] is False
    assert len(verdict_rec["failed_gates"]) == 14 - verdict_rec["n_pass"]
