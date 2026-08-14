"""tests/basilisk_b18/test_b18_feature_leakage.py —— §19/§20 特征无泄漏与可复现。

含 §22 要求的 `test_damage_not_in_xt`、`test_truth_not_in_xt`、
`test_feature_hash_reproducible`。
"""
from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEAT = ROOT / "data/features/wheel/basilisk_b18/target_features.h5"
CORE_XT_COLS = ("I_m", "omega", "T", "T_cmd", "sigma_Im", "b_hat", "dT",
                "omega_err")


def _feat_or_skip():
    if not FEAT.exists():
        pytest.skip("特征文件未生成 (需先跑 build_features.py)")
    return FEAT


def test_damage_not_in_xt(b18_feature_stats):
    """§19/§20 Gate 4: 累计损伤 D 绝不进入 x_T。"""
    fs = b18_feature_stats
    assert fs["D_in_xt"] is False
    cols = [c.lower() for c in fs["xt_cols"]]
    assert "d" not in cols
    assert not any("damage" in c for c in cols)
    # 特征文件里不得残留 truth/ 组 (D 的载体)
    assert not any("truth" in s for s in fs["leaked_datasets"])
    import h5py
    with h5py.File(_feat_or_skip(), "r") as f:
        assert bool(f.attrs["damage_in_xT"]) is False
        for key in sorted(k for k in f.keys() if k.startswith("traj_"))[:20]:
            assert "truth" not in f[key], f"{key} 残留 truth 组"
            assert "D" not in f[key], f"{key} 残留 D"


def test_truth_not_in_xt(b18_feature_stats):
    """§20 Gate 3/5/6/7: 真 EOL / b_true / Kt / Tc / b0 / omega0 都不在 x_T。"""
    fs = b18_feature_stats
    assert fs["eol_in_xt"] is False
    assert fs["b_true_in_xt"] is False
    assert fs["physparam_in_xt"] == []
    assert fs["leaked_datasets"] == [], f"额外 dataset: {fs['leaked_datasets']}"
    assert fs["leaked_attrs"] == [], f"真值 attrs 残留: {fs['leaked_attrs']}"
    # 结构性保证: build_features 签名不含 params
    from src.sim.build_hi import build_features
    sig = list(inspect.signature(build_features).parameters)
    assert "params" not in sig, "build_features 出现 params 入参 => 泄漏风险"
    assert sig == fs["build_features_signature"]


def test_feature_hash_reproducible(b18_feature_stats):
    """§20 Gate 10: content hash 口径可复现 (不含 HDF5 容器元数据)。"""
    fs = b18_feature_stats
    assert fs["feature_content_sha256"] == fs[
        "feature_content_sha256_recomputed"]
    # 独立重算一次
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "b18t_bf", ROOT / "scripts/basilisk_b18/build_features.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["b18t_bf"] = m
    spec.loader.exec_module(m)
    again = m.feature_content_hash(_feat_or_skip())
    assert again == fs["feature_content_sha256"], "content hash 不可复现"
    # file hash 也应与磁盘一致 (容器元数据虽含时间戳, 但文件未被改动)
    assert hashlib.sha256(FEAT.read_bytes()).hexdigest() == \
        fs["feature_file_sha256"]


def test_hi_formula_unmodified(b18_feature_stats, b18_config):
    """§19: HI 公式一字不改, 直接复用 build_hi。"""
    fs = b18_feature_stats
    assert fs["hi_formula_modified"] is False
    assert fs["reuse_build_hi"] is True
    assert b18_config["features"]["reuse_build_hi"] is True
    assert b18_config["features"]["hi_formula_modified"] is False


def test_core_xt_columns_pinned(b18_feature_stats):
    """§19: 核心 8 列语义与位置不变, x_T 共 10 列。"""
    fs = b18_feature_stats
    assert len(fs["xt_cols"]) == 10
    assert tuple(fs["xt_cols"][:8]) == CORE_XT_COLS
    assert tuple(fs["core_xt_cols"]) == CORE_XT_COLS


def test_mission_features_not_in_xt(b18_feature_stats, b18_config):
    """§19: mission_features 只进辅助组, 从不进 x_T。"""
    fs = b18_feature_stats
    assert fs["mission_features_in_xT"] is False
    assert b18_config["features"]["mission_features_in_xt"] is False
    assert fs["n_mission_features_copied"] == fs["n_traj"]
    import h5py
    with h5py.File(_feat_or_skip(), "r") as f:
        assert bool(f.attrs["mission_features_in_xT"]) is False
        for key in sorted(k for k in f.keys() if k.startswith("traj_"))[:10]:
            assert "mission_features" in f[key]


def test_feature_gate_verdict_consistent(b18_feature_stats):
    """§20: verdict 必须与 10 条 Gate 的实际结果一致, 且 FAIL 项必须被文档记录。

    本测试**不**要求 Gate 全过 —— B1.8 的实际结论是 Gate 9 FAIL /
    `B18_FEATURE_NOT_READY`。它要求的是"结论不可与证据脱节":
    不允许 Gate 有 FAIL 却报 READY, 也不允许悄悄把某条 Gate 从名单里删掉,
    更不允许 FAIL 了却不在 limitations.md 里写明。
    """
    fs = b18_feature_stats
    assert fs["n_gates"] == 10
    assert sorted(g["no"] for g in fs["gates"]) == list(range(1, 11))
    failed = [g["no"] for g in fs["gates"] if not g["passed"]]
    assert fs["n_passed"] == 10 - len(failed)
    assert all(c["passed"] for c in fs["structural_checks"])

    expected = "B18_FEATURE_READY" if not failed else "B18_FEATURE_NOT_READY"
    assert fs["verdict"] == expected, \
        f"verdict {fs['verdict']} 与 FAIL 名单 {failed} 不一致"

    if failed:
        lim = ROOT / "docs" / "basilisk_b18" / "limitations.md"
        assert lim.exists(), "有 Gate FAIL 却没有 limitations.md"
        txt = lim.read_text(encoding="utf-8")
        for no in failed:
            name = next(g["name"] for g in fs["gates"] if g["no"] == no)
            assert name in txt, f"Gate {no} ({name}) FAIL 却未写入 limitations.md"
        assert "B18_FEATURE_NOT_READY" in txt


@pytest.mark.xfail(strict=True, reason=(
    "B1.8 实际结论: Gate 9 FAIL —— 监督 HI_B 的归一化分母仍锚在已废止的摩擦阈值 "
    "b_fail 上, 与新的 D>=1 失效定义不同源, 18/71 条 event 轨迹在 D=1 时 "
    "b_true < b_fail, 故 p95(HI_B) 最小仅 0.1257。协议禁止改 HI 公式(§19)、"
    "禁止降门限(§20)、禁止调 L_ref(§14), 故如实标为 xfail(strict) —— "
    "一旦转 PASS 会报 XPASS 强制复审, 不会静默变绿。"))
def test_feature_gates_all_passed(b18_feature_stats):
    """§20 的目标态: 10 条 Gate 全过 -> B18_FEATURE_READY。"""
    fs = b18_feature_stats
    failed = [g["no"] for g in fs["gates"] if not g["passed"]]
    assert not failed, f"未过的 Gate: {failed}"
    assert fs["verdict"] == "B18_FEATURE_READY"


def test_no_nan_inf_in_xt(b18_feature_stats):
    """§20 Gate 1/2: x_T 无 NaN/Inf。删失 rul 的 NaN 是设计, 分开统计。"""
    fs = b18_feature_stats
    assert fs["xt_nan"] == 0
    assert fs["xt_inf"] == 0
    assert fs["censored_rul_all_nan"] is True, \
        "删失轨迹的 rul 必须全为 NaN —— 不得伪造 EOL"
    import h5py
    with h5py.File(_feat_or_skip(), "r") as f:
        for key in sorted(k for k in f.keys() if k.startswith("traj_"))[:20]:
            x = f[key]["x_T"][:]
            assert np.isfinite(x).all(), f"{key} 的 x_T 含非有限值"
            if not int(f[key].attrs["event_observed"]):
                assert np.all(np.isnan(f[key]["rul"][:])), \
                    f"{key} 删失却有 rul 数值"


def test_b_hat_accuracy(b18_feature_stats):
    """§20 Gate 8: b_hat p95 相对误差 < 10% (实际 PASS)。"""
    fs = b18_feature_stats
    assert fs["b_hat_rel_err_p95"] < fs["b_hat_p95_rel_err_max"], \
        f"b_hat p95 rel err = {fs['b_hat_rel_err_p95']:.5f}"


def test_hi_coverage_shortfall_recorded(b18_feature_stats):
    """§20 Gate 9 未达标 —— 断言"缺口被如实记录", 而不是断言它达标。

    这里刻意不放宽门限: `hi_p95_min` 必须仍是 0.8 (config 冻结值), 缺口必须真实
    存在于统计里, 且必须在 Gate 9 的 passed=False 中体现。若哪天真的达标了,
    第二段断言会失败, 强制回来把本测试改回正向断言。
    """
    fs = b18_feature_stats
    assert fs["hi_p95_min"] == 0.8, "门限被改动 —— 这是 outcome tuning"
    g9 = next(g for g in fs["gates"] if g["no"] == 9)
    assert g9["name"] == "event_observed_hi_p95_above_min"
    shortfall = fs["hi_p95_event_min"] <= fs["hi_p95_min"]
    assert g9["passed"] is (not shortfall), \
        "Gate 9 的 passed 标记与统计量不一致"
    assert shortfall, (
        "Gate 9 现已达标 (event-observed HI p95 最小值 = "
        f"{fs['hi_p95_event_min']:.4f}) —— 请把本测试改回正向断言并复审 verdict")


def test_feature_lineage_traceable(b18_feature_stats, b18_dataset_audit):
    """特征文件必须能追溯到正式数据集与协议哈希。"""
    fs = b18_feature_stats
    assert fs["source_dataset_content_sha256"] == \
        b18_dataset_audit["content_sha256"]
    assert fs["protocol_sha256"] == b18_dataset_audit["protocol_sha256"]
    assert fs["scenario"] == "NOMINAL"
    assert float(fs["L_ref_years"]) == 3.0
    assert fs["trained_any_model"] is False
    import h5py
    with h5py.File(_feat_or_skip(), "r") as f:
        assert str(f.attrs["lineage"]) == "basilisk_b18"
        assert bool(f.attrs["is_manufacturer_failure_specification"]) is False
