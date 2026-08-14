"""tests/basilisk_b13/test_b13_feature_leakage.py

§16/§17/§19 —— 特征无真值泄漏。

§19 命名测试:
  * test_feature_no_truth_leakage

B1.3 verdict = B13_CALIBRATION_FAIL, 因此 `build_features.py` **未运行**
(§16 明令仅 DATA_READY 才跑)。特征产物不存在时这些测试 skip —— 但
**契约类断言 (config 级) 仍然必须执行**, 因为那是"如果将来跑, 也不会泄漏"
的静态保证, 与产物是否存在无关。
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# §16 冻结的 x_T 列 (顺序也冻结 —— 换序会让 checkpoint 权重错位)
XT_EXPECTED = ["I_m", "omega", "T", "T_cmd",
               "sigma_Im", "b_hat_selfcal", "dT", "omega_err"]

# 绝不允许进入 x_T 的真值 / 标签 / 未来信息字段
TRUTH_FORBIDDEN = {
    "b_true", "b", "b0", "Kt", "Tc", "omega0",          # 物理真参数
    "q", "q_true", "q0", "hi", "HI", "soh", "SOH",      # 健康指标真值
    "rul", "RUL", "eol", "eol_idx", "failed",           # 标签
    "t_fail", "time_to_fail", "degradation_fraction",
    "g_duty", "_b1_g_duty", "_b1_speed_util_mean",      # 标定内部量
    "mode", "mission_mode",                             # mission_features
}


def test_feature_no_truth_leakage(b13_cfg, feature_rec):
    """§16: x_T 只含遥测派生量, 真值字段一律不得进入。

    两层检查:
      (1) config 层 —— xt_cols 与冻结清单逐位相同, 且与 truth_fields 无交集;
      (2) 产物层 —— 若 feature h5 摘要存在, 实际写入的列也要满足同样约束。
    只做 (1) 不够: config 对了但 build 脚本自己塞了一列, 静态检查看不见。
    只做 (2) 也不够: 产物不存在时就完全失去保护。
    """
    f = b13_cfg["features"]
    assert list(f["xt_cols"]) == XT_EXPECTED, \
        f"x_T 列被改动: {list(f['xt_cols'])} != {XT_EXPECTED}"
    assert f["mission_features_in_xt"] is False, \
        "§16: mission_features 从不进入 x_T"

    truth = set(map(str, f["truth_fields"]))
    leak = truth & set(f["xt_cols"])
    assert not leak, f"truth_fields 与 x_T 交集非空: {sorted(leak)}"

    bad = [c for c in f["xt_cols"] if c in TRUTH_FORBIDDEN]
    assert not bad, f"x_T 含禁止字段: {bad}"

    if feature_rec is None:
        pytest.skip("B13_CALIBRATION_FAIL -> build_features.py 未运行 (§16 合规)")

    # ---- 产物层 ----
    cols = list(feature_rec["xt_cols"])
    assert cols == XT_EXPECTED, f"产物 x_T 列 {cols} != {XT_EXPECTED}"
    bad = [c for c in cols if c in TRUTH_FORBIDDEN]
    assert not bad, f"产物 x_T 含禁止字段: {bad}"


def test_feature_artifacts_absent_on_calibration_fail(verdict_rec, feature_rec,
                                                      dataset_rec):
    """§13/§14/§16: FAIL 时不得存在 150 条数据集与特征产物。

    这条测试的意义是"证明流程真的停下来了"—— 比在报告里写一句"我停了"更硬。
    """
    if verdict_rec is None:
        pytest.skip("需先跑 freeze_calibration.py")
    if verdict_rec["verdict"] == "B13_CALIBRATION_PASS":
        pytest.skip("PASS 分支不适用")
    assert verdict_rec["verdict"] == "B13_CALIBRATION_FAIL"
    assert verdict_rec["frozen_calibration_written"] is False
    assert dataset_rec is None, "FAIL 却存在 150 条数据集摘要 —— 违反 §14"
    assert feature_rec is None, "FAIL 却存在特征产物 —— 违反 §16"
    # 目录本身也不该有 final 数据
    final = ROOT / "data" / "sim" / "wheel_basilisk_b13" / "final"
    if final.exists():
        assert not list(final.glob("*.h5")) and not list(final.glob("*.npz")), \
            f"final 目录存在轨迹文件: {list(final.iterdir())}"


def test_build_hi_reused_not_reimplemented():
    """§16: HI 构造必须复用无泄漏的 `build_hi.py`, 不得另写一份。

    另写一份的风险是它会悄悄用上真值 —— 所以这里检查 b13 脚本目录里
    没有自己的 HI 构造实现, 且 build_features 走的是共享模块。
    """
    d = ROOT / "scripts" / "basilisk_b13"
    bf = d / "build_features.py"
    if not bf.exists():
        pytest.skip("build_features.py 未创建 (FAIL 分支, §16 合规)")
    src = bf.read_text(encoding="utf-8")
    assert "build_hi" in src, "build_features.py 未引用 build_hi"
    # 不得自行定义 HI 公式
    for pat in ("def build_hi", "def compute_hi", "def make_hi"):
        assert pat not in src, f"b13 自行实现了 HI 构造 ({pat}) —— 必须复用"


def test_no_model_or_transfer_run(verdict_rec):
    """§18: 即使全过也不得跑 target_only / finetune / mmd / PF / S6。"""
    for rel in ("checkpoints/basilisk_b13/target_only",
                "checkpoints/basilisk_b13/source_finetune",
                "checkpoints/basilisk_b13/source_mmd",
                "checkpoints/basilisk_b13/rate_model",
                "checkpoints/basilisk_b13/wiener_pf",
                "docs/basilisk_b13/s6_report.md",
                "docs/basilisk_b13/rul_results.md"):
        p = ROOT / rel
        assert not p.exists(), f"§18 违规: 存在 {rel}"


def test_next_split_declared_but_not_executed(b13_cfg):
    """`next_split` 只是声明下一阶段的划分口径, 本阶段不得据此训练。

    个体划分 (按机台/器件) 的声明要存在 —— 否则下一阶段容易退化成随机划分,
    造成同一退化轨迹跨 train/val 泄漏。
    """
    # 实测字段名是 `level` (不是我最初猜的 `group_by`/`by`) ——
    # 键名必须照实读, 猜错会让这条保护形同虚设 (assert 永远命中缺键分支)。
    ns = b13_cfg["next_split"]
    key = str(ns["level"])
    assert key in ("trajectory", "unit", "device"), \
        f"next_split.level = {key!r} 不是个体级 —— 有轨迹泄漏风险"
    # 划分比例必须齐全且归一, 否则下一阶段会静默丢样本
    fr = [float(ns[k]) for k in ("train_frac", "val_frac", "test_frac")]
    assert abs(sum(fr) - 1.0) < 1e-12, f"划分比例不归一: {fr}"
    # 分层键只能用 event/censored 状态, 不得按标签值分层 (会泄漏 RUL 尺度)
    assert set(ns["stratify"]) == {"event_observed", "censored"}, \
        f"stratify 键异常: {ns['stratify']}"
