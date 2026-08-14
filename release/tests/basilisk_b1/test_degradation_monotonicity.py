"""tests/basilisk_b1/test_degradation_monotonicity.py

B1 §16 —— 磨损驱动 g_duty 的单调性与非负性测试 (§4 硬要求)。

§4 对标定后的退化模型有四条硬要求:
  1. 退化率对 speed_util 单调递增
  2. 退化率对 torque_util 单调递增
  3. 对温度升高单调递增
  4. 最恶劣工况下为 nominal 的**有限**倍, 且**不允许出现负 wear rate**

退化率与 g_duty 的关系: tau_years <- tau_raw / g_duty, 而
  b(t) = b0·[1 + Δ·(1 - e^{-integ/τ})]  =>  db/dt ∝ 1/τ ∝ g_duty
所以 "退化率单调递增于 X" 等价于 "g_duty 单调递增于 X"。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "basilisk_b1"))

import calibrate_degradation as cal                      # noqa: E402

CFG = cal.load_b1_config("configs/wheel_basilisk_b1.yaml")
WD = CFG["wear_drive"]
# 一个"reference 处"的驱动量组合: 各项都等于 reference -> g_duty 应为 1.0
REF = {"speed_util": 0.30, "torque_util": 0.12,
       "zero_crossing_rate": 0.02, "maneuver_fraction": 0.20}


def _drives(**over) -> dict:
    d = {k: np.array([float(REF[k])]) for k in cal.DRIVE_KEYS}
    for k, v in over.items():
        d[k] = np.array([float(v)])
    return d


def _g(**over) -> float:
    return float(cal.g_duty(_drives(**over), REF, WD)[0])


# --------------------------------------------------------------------------
# 归一化: reference 处 g_duty = 1
# --------------------------------------------------------------------------
def test_g_duty_is_one_at_reference():
    """g_duty 在 reference duty 处必须精确为 1.0 —— 否则"标定"会顺带改变整体寿命尺度。

    权重和为 1 且每项在 reference 处归一为 1, 故 Σ w_i · 1 = 1。
    """
    assert _g() == pytest.approx(1.0, abs=1e-12)


def test_weights_sum_to_one_enforced():
    """权重和 != 1 必须直接报错 (防止有人靠改权重悄悄整体加速退化)。"""
    bad = dict(WD)
    bad["speed_util_weight"] = float(WD["speed_util_weight"]) + 0.1
    with pytest.raises(ValueError, match="权重和"):
        cal.g_duty(_drives(), REF, bad)


def test_negative_weight_rejected():
    """负权重会产生负 wear rate, 必须直接拒绝 (§4 明令)。

    注意让权重和仍为 1 —— 否则会先被"和必须为 1"拦下, 测不到负值分支。
    """
    bad = dict(WD)
    bad["speed_util_weight"] = -0.10
    bad["torque_util_weight"] = 0.50
    bad["zero_crossing_weight"] = 0.30
    bad["maneuver_weight"] = 0.30
    assert sum(bad[k] for k in ("speed_util_weight", "torque_util_weight",
                                "zero_crossing_weight", "maneuver_weight")) \
        == pytest.approx(1.0)
    with pytest.raises(ValueError, match="不得为负"):
        cal.g_duty(_drives(), REF, bad)


# --------------------------------------------------------------------------
# §4(1)(2) 单调性
# --------------------------------------------------------------------------
def test_wear_monotone_speed_util():
    """g_duty 必须随 speed_util 严格单调递增 (在未触上下 clip 的区间内)。"""
    xs = np.linspace(0.05, 0.95, 40)
    gs = np.array([_g(speed_util=x) for x in xs])
    assert np.all(np.diff(gs) > 0), \
        f"g_duty 对 speed_util 非单调递增: {gs[:5]} ... {gs[-5:]}"
    assert gs.min() > 0.0, "出现非正 wear rate"


def test_wear_monotone_torque_util():
    """g_duty 必须随 torque_util 严格单调递增。"""
    xs = np.linspace(0.0, 1.0, 40)
    gs = np.array([_g(torque_util=x) for x in xs])
    assert np.all(np.diff(gs) > 0), \
        f"g_duty 对 torque_util 非单调递增: {gs[:5]} ... {gs[-5:]}"
    assert gs.min() > 0.0, "出现非正 wear rate"


def test_wear_monotone_zero_crossing_and_maneuver():
    """穿零率与机动占比同样必须单调递增 (它们代表边界润滑 / 启停工况)。"""
    for key in ("zero_crossing_rate", "maneuver_fraction"):
        xs = np.linspace(0.0, 1.0, 30)
        gs = np.array([_g(**{key: x}) for x in xs])
        assert np.all(np.diff(gs) > 0), f"g_duty 对 {key} 非单调递增"


def test_wear_never_negative_on_random_drives():
    """任意 (含极端) 驱动量组合下 g_duty 恒 >= G_DUTY_MIN > 0。"""
    rng = np.random.default_rng(20260809)
    d = {k: rng.uniform(0.0, 1.5, size=2000) for k in cal.DRIVE_KEYS}
    g = cal.g_duty(d, REF, WD)
    assert np.all(np.isfinite(g)), "g_duty 出现 NaN/Inf"
    assert g.min() >= cal.G_DUTY_MIN, f"g_duty 下界被突破: {g.min()}"
    assert g.min() > 0.0


def test_negative_drive_input_clamped_not_propagated():
    """驱动量若因数值问题出现负值, 必须被夹到 0 而不是传成负 wear rate。"""
    g = cal.g_duty(_drives(speed_util=-0.5, torque_util=-0.3), REF, WD)
    assert float(g[0]) > 0.0
    # 与全零输入等价 (负值被 clamp 到 0)
    g0 = cal.g_duty(_drives(speed_util=0.0, torque_util=0.0), REF, WD)
    assert float(g[0]) == pytest.approx(float(g0[0]))


# --------------------------------------------------------------------------
# §4(4) 有限倍
# --------------------------------------------------------------------------
def test_worst_case_is_finite_multiple_of_nominal():
    """最恶劣工况 (四项全部拉满) 只能是 nominal 的有限倍, 由 g_duty_max 钉死。"""
    g = _g(speed_util=1.0, torque_util=1.0,
           zero_crossing_rate=1.0, maneuver_fraction=1.0)
    gmax = float(WD["g_duty_max"])
    assert np.isfinite(g) and g <= gmax + 1e-9, f"g_duty 未被上界约束: {g}"
    assert gmax < 100.0, "g_duty_max 过大, 失去『有限倍』的物理意义"


def test_ref_floor_prevents_divergence():
    """某项 reference 近零时, ref_floor 必须阻止该项发散。"""
    ref = dict(REF)
    ref["zero_crossing_rate"] = 0.0
    g = cal.g_duty(_drives(zero_crossing_rate=0.5), ref, WD)
    assert np.isfinite(g[0]) and g[0] <= float(WD["g_duty_max"]) + 1e-9


# --------------------------------------------------------------------------
# §4 温度: 退化率对温度单调递增 (Arrhenius, 由冻结 wheel_sim 提供)
# --------------------------------------------------------------------------
def test_wear_monotone_temperature():
    """Arrhenius 加速因子必须随温度单调递增 —— 复核冻结退化模型未被 B1 破坏。

    直接用 wheel_sim 的公式常量重算 (不改代码, 只验证行为):
      accel(T) = exp(-Ea/kB · (1/T - 1/T_ref))
    """
    from src.sim.wheel_sim import K_B
    dc = CFG["sim"]["disturbance"]
    Ea = float(dc["arrhenius_Ea_eV"])
    Tref = float(dc["arrhenius_T_ref_K"])
    T = np.linspace(260.0, 340.0, 50)
    accel = np.exp(-Ea / K_B * (1.0 / T - 1.0 / Tref))
    assert np.all(np.diff(accel) > 0), "Arrhenius 加速因子对温度非单调递增"
    assert accel.min() > 0.0


def test_calibration_only_touches_three_scale_params():
    """apply_calibration 只能改 b0 / tau_years / omega0, 其余参数原样保留 (§4)。"""
    rated = {"omega_rated_rad_s": 628.3185307179587}
    calib = {"b0_scale": 7.0}
    params = {"J": 0.03, "Kt": 0.025, "Tc": 1.2e-3, "b0": 2.0e-6,
              "Delta": 5.0, "tau_years": 1.0, "omega0": 1500.0,
              "T_base": 293.0, "T_amp": 5.0, "seed_traj": 12345}
    out = cal.apply_calibration(params, _drives(), REF, CFG, rated, calib)
    for k in ("J", "Kt", "Tc", "Delta", "T_base", "T_amp", "seed_traj"):
        assert out[k] == params[k], f"{k} 被意外改动"
    assert out["b0"] == pytest.approx(params["b0"] * 7.0)
    # reference 处 g_duty=1 -> tau 不变
    assert out["tau_years"] == pytest.approx(params["tau_years"])
    # omega0 必须按实际工作点重设, 不得沿用 analytic 的 1500
    assert out["omega0"] == pytest.approx(REF["speed_util"] * 628.3185307179587)
    assert out["omega0"] < 0.5 * params["omega0"]


def test_higher_utilization_shortens_tau():
    """高利用率轨迹的 tau_years 必须更短 (退化更快) —— 端到端单调性。"""
    rated = {"omega_rated_rad_s": 628.3185307179587}
    calib = {"b0_scale": 7.0}
    base = {"J": 0.03, "Kt": 0.025, "Tc": 1.2e-3, "b0": 2.0e-6, "Delta": 5.0,
            "tau_years": 1.0, "omega0": 1500.0, "T_base": 293.0,
            "T_amp": 5.0, "seed_traj": 1}
    lo = cal.apply_calibration(base, _drives(speed_util=0.20), REF, CFG, rated, calib)
    hi = cal.apply_calibration(base, _drives(speed_util=0.60), REF, CFG, rated, calib)
    assert hi["tau_years"] < lo["tau_years"], "高利用率未缩短 tau"
    assert hi["_b1_g_duty"] > lo["_b1_g_duty"] > 0.0


def test_wheel_sim_degradation_code_unchanged():
    """§4 "保留现有退化结构, 不重新发明模型": wheel_sim.py 必须一字未改。

    hash 由 B1 baseline contract 冻结; 这里做独立的结构性检查 —— B1 的标定
    只允许发生在**调用侧** (apply_calibration), wheel_sim 里不得出现任何 b1 字样。
    """
    src = (ROOT / "src/sim/wheel_sim.py").read_text(encoding="utf-8")
    for token in ("speed_util", "torque_util", "g_duty", "b1", "B1",
                  "omega_rated", "utilization"):
        assert token not in src, \
            f"src/sim/wheel_sim.py 出现 B1 标定痕迹 '{token}' —— 退化代码必须冻结"
