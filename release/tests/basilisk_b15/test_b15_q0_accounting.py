"""tests/basilisk_b15/test_b15_q0_accounting.py

Basilisk-B1.5 §3 —— q0 记账口径修正的测试。

§3 要求: "新增测试证明该修正只影响审计量, 不改变任何 trajectory。"
本文件同时钉死修正的**成因解释**, 以免将来有人把 after 口径的成功
误当成"容差被放宽了"。
"""
from __future__ import annotations

import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints" / "basilisk_b15"
AUDIT = CKPT / "q0_accounting_audit.json"

B14_H5_REL = "data/sim/wheel_basilisk_b14/probe/wheel_all.h5"
B14_PAIRED_HASH = (
    "ff1de559b2746dcb394c4de6d70bda4d96ca594d1b953f58a4fa390366f41cf6")
B14_PROBE_CONTENT = (
    "424bcc5e0a99ff28471afb592c0aa606862815e2016f615d1d747887e0552625")


def _load() -> dict:
    assert AUDIT.exists(), f"缺少 {AUDIT}"
    return json.loads(io.open(AUDIT, encoding="utf-8").read())


def test_verdict_is_fixed():
    r = _load()
    assert r["verdict"] == "B15_Q0_ACCOUNTING_FIXED"
    assert r["n_traj"] == 60


def test_before_caliber_is_the_wrong_one():
    """before 口径必须 0/60 —— 与 B1.4 Gate 13 记录一致。"""
    r = _load()
    b = r["before"]
    assert b["definition"] == "q0 = b_true[0] / b_fail_f2"
    assert b["n_mapping_ok"] == 0
    assert b["verdict"] == "WRONG_CALIBER"
    # 与 B1.4 记录的 max_mapping_rel_err 吻合
    d = r["defect_explanation"]
    assert d["matches_b14_record"] is True
    assert abs(b["max_mapping_rel_err"]
               - d["b14_gate13_max_mapping_rel_err_recorded"]) < 5e-6


def test_after_caliber_passes_at_unchanged_tolerance():
    """关键: after 口径 60/60, 且容差与 B1.4 完全相同, **未放宽**。"""
    r = _load()
    assert r["tolerance_rel"] == 1e-12, "容差被改动了 —— §3 不允许放宽"
    a = r["after"]
    assert a["definition"] == "q0 = b0 / b_fail_f2"
    assert a["n_mapping_ok"] == r["n_traj"] == 60
    assert a["verdict"] == "CORRECT_CALIBER"
    # 纯除法 + 线性映射, 偏差应在 ULP 量级, 远严于 1e-12
    assert a["max_mapping_rel_err"] < 1e-15
    assert a["max_ulp"] <= 2


def test_defect_explanation_is_systematic_not_noise():
    """偏差必须全为正: 首样本已含一步退化 ⇒ before 口径系统性偏大。"""
    r = _load()
    d = r["defect_explanation"]
    assert d["all_excess_positive"] is True
    assert d["excess_equals_rel_err_before"] is True
    assert d["excess_min"] > 0.0
    for rec in r["per_trajectory"]:
        # 逐条: before 恒大于 after, 且 after 恒等于 u 的线性映射
        assert rec["q0_before_b_true0_over_bfail"] > \
            rec["q0_after_b0_over_bfail"], f"traj {rec['traj_id']} 偏差为负"
        assert rec["mapping_ok_after"] is True
        assert rec["mapping_ok_before"] is False


def test_b0_comes_from_sampling_time_params():
    """b0 必须取自 group attrs (sample_params 原始输出), 不是任何派生列。"""
    r = _load()
    for rec in r["per_trajectory"]:
        assert rec["b0_from_group_attrs"] is True


def test_trajectories_bitwise_unchanged():
    """§3 硬要求: 修正只影响审计量, 不改变任何 trajectory。"""
    r = _load()
    t = r["trajectories_unchanged"]
    assert t["bitwise_identical"] is True
    assert t["h5_sha256_before"] == t["h5_sha256_after"]
    assert t["h5_written_by_this_script"] is False
    assert t["paired_trajectory_hash_b14"] == B14_PAIRED_HASH
    assert t["probe_content_sha256_b14"] == B14_PROBE_CONTENT
    assert (ROOT / B14_H5_REL).exists()


def test_audit_script_never_opens_h5_for_write():
    """源码级证明: 只读打开, 无任何写路径。"""
    src = io.open(ROOT / "scripts" / "basilisk_b15" / "audit_q0_accounting.py",
                  encoding="utf-8").read()
    assert 'h5py.File(str(B14_H5), "r")' in src
    for bad in ('h5py.File(str(B14_H5), "w")', 'mode="a"', 'mode="r+"',
                "to_hdf", "create_dataset"):
        assert bad not in src, f"审计脚本出现写 h5 的痕迹: {bad}"


def test_b14_verdict_not_retroactively_revised():
    """B1.5 不得追认 B1.4 的 Gate 13 为 PASS。"""
    r = _load()
    nr = r["b14_verdict_not_revised"]
    assert nr["b14_verdict"] == "B14_CALIBRATION_FAIL"
    assert nr["b14_gate13_pass"] is False
    # B1.4 的产物必须原样存在, 未被本阶段改写
    b14 = json.loads(io.open(
        ROOT / "checkpoints" / "basilisk_b14" / "calibration_verdict.json",
        encoding="utf-8").read())
    assert b14["verdict"] == "B14_CALIBRATION_FAIL"
    assert b14["n_pass"] == 8


def test_no_rul_or_model_metric():
    r = _load()
    assert r["used_rul_metric"] is False
    assert r["used_model_metric"] is False
