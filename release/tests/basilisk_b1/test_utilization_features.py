"""tests/basilisk_b1/test_utilization_features.py

B1 §16 —— 无量纲轮利用率的量纲正确性测试。

覆盖:
  * test_hr16_rated_values_from_model  —— rated 值必须来自 Basilisk rwFactory,
    而不是为了失效率手填 (§2 铁律)
  * test_speed_util_bounded            —— speed_util ∈ [0, 1] (物理上不可超额定)
  * test_torque_util_bounded_or_flagged —— torque_util 允许瞬时 >1 (指令饱和),
    但必须被显式统计出来, 不允许悄悄超界
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "basilisk_b1"))

import calibrate_degradation as cal                      # noqa: E402

CFG = cal.load_b1_config("configs/wheel_basilisk_b1.yaml")
PHYS_JSON = ROOT / "checkpoints/basilisk_b1/physics_audit.json"


def _phys() -> dict:
    if not PHYS_JSON.exists():
        pytest.skip("physics_audit.json 未生成 (先跑 audit_physics.py)")
    return json.loads(PHYS_JSON.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# §2/§3 rated 值溯源
# --------------------------------------------------------------------------
def test_hr16_rated_values_from_model():
    """omega_rated / torque_rated / Js 必须与 Basilisk 官方 HR16 定义逐位一致。

    这是 §2 "rated 值必须来自 Basilisk 使用的 HR16 配置, 而不是为了失效率手工凑数"
    的直接检验: 现场调 rwFactory 拿读数, 与 config 记录的期望值按 tol_rel 比对。
    任何手改 config 里的 rated 数字都会让这个测试失败。
    """
    pytest.importorskip("Basilisk", reason="Basilisk 未安装 (不进 requirements)")
    mm = float(CFG["sim"]["profile"]["spacecraft"]["rw"]["max_momentum_Nms"])
    rated = cal.rated_values_from_basilisk(mm)
    checks = cal.check_rated_against_config(rated, CFG)
    bad = [(n, d) for n, ok, d in checks if not ok]
    assert not bad, f"rated 值与 Basilisk 读数不一致 (禁止手工凑数): {bad}"
    # 具体数值锚点: HR16 = 6000 RPM / 0.2 N·m, 防 config 与测试同时被改坏
    assert abs(rated["omega_rated_rad_s"] - 6000.0 * 2.0 * np.pi / 60.0) < 1e-9
    assert abs(rated["torque_rated_Nm"] - 0.2) < 1e-12
    # Js 必须与 maxMomentum/Omega_max 自洽 (说明读到的是同一个模型对象)
    assert abs(rated["rotor_inertia_Js_kgm2"]
               - mm / rated["omega_rated_rad_s"]) < 1e-12


def test_rated_source_is_declared():
    """config 必须显式声明 rated 的 provenance, 便于答辩时溯源。"""
    assert CFG["rated"]["source"] == "basilisk_rwFactory_Honeywell_HR16"


def test_b1_does_not_reuse_analytic_omega0_scale():
    """§4 明令: 不得把 analytic 的绝对轮速范围 (1000-2000) 直接套在 HR16 上。

    检验方式: b0 标定后的等效工作转速必须落在 HR16 实测利用率对应的量级,
    与 analytic 的 omega_ref=1500 相差一个显著倍数 (即真的重标定过)。
    """
    p = _phys()
    b = p["b0_calibration"]
    assert b["omega_operating_rad_s"] < 0.5 * b["omega_ref_design_rad_s"], \
        "标定后工作转速仍接近 analytic 的 1500 rad/s, 说明没有真正重标定"
    assert b["b0_scale"] > 1.0, "HR16 转速更低 -> b0 必须上调才能保持摩擦裕度"


# --------------------------------------------------------------------------
# §16 utilization 有界性
# --------------------------------------------------------------------------
def test_speed_util_bounded():
    """speed_util = |ω|/Ω_rated 必须 ∈ [0, 1]。

    转速超额定在物理上意味着轮子超速保护失效, Basilisk 的 HR16 模型不会给出
    这种工况; 若测到 >1, 说明 rated 值取错了 (量纲标定的地基就塌了)。
    """
    p = _phys()
    for m, s in p["pointwise_per_mode"].items():
        # 逐点统计只记 mean/rms/p95/max (非负性由 |ω| 保证), 这里查上界
        assert s["speed_util"]["max"] <= 1.0 + 1e-12, \
            f"{m} speed_util max = {s['speed_util']['max']} > 1 -> rated 取错"
        assert s["speed_util"]["mean"] >= 0.0, f"{m} speed_util mean 出现负值"
    for m, s in p["window_per_mode"].items():
        assert 0.0 <= s["speed_util"] <= 1.0, f"窗级 {m} speed_util 越界 {s}"
    r = p["window_reference_duty"]
    assert 0.0 < r["speed_util"] < 1.0, f"reference speed_util 越界 {r}"


def test_torque_util_bounded_or_flagged():
    """torque_util 允许触及 1.0 (指令饱和是真实工况), 但必须被显式统计。

    与 speed_util 不同, 力矩指令被 u_max 饱和限幅是反作用轮的正常行为, 所以
    这里**不**断言 <1; 断言的是: (a) 非负; (b) 不得离谱地超界 (数值发散);
    (c) 高利用率占比 (tu>0.5 / >0.8) 必须出现在审计报告里 —— 不允许悄悄超界。
    """
    p = _phys()
    for m, s in p["pointwise_per_mode"].items():
        tu = s["torque_util"]
        assert tu["mean"] >= 0.0, f"{m} torque_util mean 出现负值"
        assert tu["max"] <= 1.0 + 1e-9, \
            f"{m} torque_util max = {tu['max']} 明显超饱和界 -> 数值发散"
    for m, s in p["pointwise_per_mode"].items():
        frac = s["high_util_fraction"]
        for key in ("torque_util_gt_0.5", "torque_util_gt_0.8",
                    "speed_util_gt_0.5", "speed_util_gt_0.8"):
            assert key in frac, f"{m} 缺高利用率占比 {key} (必须显式统计)"
            assert 0.0 <= float(frac[key]) <= 1.0
    # 饱和确实发生过 (imaging torque_util max = 1.0), 所以"不断言 <1"不是空话
    assert max(s["torque_util"]["max"] for s in p["pointwise_per_mode"].values()) \
        >= 1.0 - 1e-9, "无任何模式触及力矩饱和, 本测试的宽松上界失去意义"


def test_utilization_uses_window_rms_not_mean():
    """窗级 speed_util 必须用 rms 口径 (与喂给退化模型的 omega_cmd 同源)。

    摩擦功耗 ∝ b·ω², 用 mean 会系统性低估双向工况 (desat/nadir 有穿零),
    进而低估磨损 —— 这正是 B1 要修的那类量纲错误。
    """
    rated = {"omega_rated_rad_s": 100.0, "torque_rated_Nm": 1.0}
    n = 8
    duty = {"stats": {
        "omega_rms": np.array([50.0]), "omega_mean": np.array([0.0]),
        "torque_rms": np.array([0.5]), "zero_crossing_count": np.array([4.0]),
        "maneuver_fraction": np.array([0.25]),
    }}
    d = cal.utilization_from_duty(duty, rated, n)
    assert d["speed_util"][0] == pytest.approx(0.5), "未使用 omega_rms 口径"
    assert d["torque_util"][0] == pytest.approx(0.5)
    assert d["zero_crossing_rate"][0] == pytest.approx(0.5)


def test_reference_duty_is_computed_not_handwritten():
    """reference duty 只能从冻结 profiles.h5 算出, config 里不得手填数值。"""
    txt = (ROOT / "configs/wheel_basilisk_b1.yaml").read_text(encoding="utf-8")
    wd = CFG["wear_drive"]
    assert wd["reference"] == "all_mode_ensemble_mean"
    for k in cal.DRIVE_KEYS:
        assert f"{k}_reference" not in txt and f"{k}_ref:" not in txt, \
            f"config 手填了 reference {k} —— reference 必须运行时从 profile 库算"
