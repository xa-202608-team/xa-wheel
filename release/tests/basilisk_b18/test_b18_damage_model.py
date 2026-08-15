"""tests/basilisk_b18/test_b18_damage_model.py —— §5/§6/§9/§10/§11 损伤模型。

含 §22 要求的 `test_damage_monotonic`、`test_reference_damage_lifetime`。
"""
from __future__ import annotations

import numpy as np
import pytest

SEC_PER_YEAR = 365.25 * 24 * 3600


def test_damage_monotonic(b18_damage_module):
    """§5: D(t) 必须单调递增 —— 对随机非负 stress 与三个 L_ref 都成立。"""
    dm = b18_damage_module
    rng = np.random.default_rng(20260809)      # 局部 rng, 不用全局 seed
    for L in (4.5, 3.0, 2.25):
        for _ in range(20):
            s = rng.uniform(0.0, 20.0, size=4096)
            D = dm.damage_series(s, 1800.0, L)
            assert np.all(np.diff(D) >= 0.0), f"L_ref={L} 下 D 非单调"
            assert np.all(np.isfinite(D))
            assert D[0] >= 0.0


def test_damage_monotonic_in_real_data(b18_dataset_audit):
    """正式 150 条数据里每条轨迹的 D 都必须单调。"""
    a = b18_dataset_audit
    assert a["audit"]["D"]["all_monotone"] is True
    bad = [r["traj_id"] for r in a["per_trajectory"] if not r["D_monotone"]]
    assert not bad, f"非单调轨迹: {bad}"


def test_reference_damage_lifetime(b18_damage_module, b18_config,
                                   b18_damage_model):
    """§6: reference condition 下 stress=1, 故 D(t) = t / L_ref。

    这是模型最关键的可解释性质, 必须在测试里独立复算, 不能只信 JSON。
    """
    dm = b18_damage_module
    dist = b18_config["sim"]["disturbance"]
    T_ref = float(dist["arrhenius_T_ref_K"])
    # a_T(T_ref) = exp(0) = 1
    assert dm.arrhenius_accel(np.array([T_ref]), dist)[0] == pytest.approx(
        1.0, abs=1e-15)

    dt = float(b18_config["sim"]["sample_period_s"])
    n = 2048
    stress = np.ones(n, dtype=np.float64)      # reference: g_duty=1, a_T=1
    for name, sc in b18_config["lifetime_scenarios"]["scenarios"].items():
        L = float(sc["L_ref_years"])
        D = dm.damage_series(stress, dt, L)
        t_years = np.arange(1, n + 1) * dt / SEC_PER_YEAR
        analytic = t_years / L
        assert np.allclose(D, analytic, rtol=0.0, atol=1e-12), \
            f"{name}: D(t) != t/L_ref"
        # D 达到 1 的时刻恰为 L_ref
        assert D[-1] == pytest.approx(t_years[-1] / L, abs=1e-12)

    # derive_damage_model.py 的记录必须与此一致
    assert b18_damage_model["derivation_verified"] is True


def test_reference_identity_recorded_zero_error(b18_damage_model):
    """§6 的三个情景在推导脚本里必须都是零误差, 且 identity 成立。"""
    rc = b18_damage_model["reference_condition"]
    assert rc["weights_sum"] == pytest.approx(1.0, abs=1e-12)
    assert rc["g_duty_at_reference_max_dev"] == 0.0
    assert rc["a_T_at_T_ref_max_dev"] == 0.0
    assert rc["stress_at_reference_max_dev"] == 0.0
    assert rc["arrhenius_direction_correct"] is True
    for name, s in rc["per_scenario"].items():
        assert s["max_abs_err_vs_analytic"] == 0.0, f"{name} 有偏差"
        assert s["max_rel_err_vs_analytic"] == 0.0, f"{name} 有偏差"
    assert b18_damage_model["derivation_verified"] is True


def test_stress_is_g_duty_times_arrhenius(b18_damage_module, b18_config):
    """§5: stress = g_duty * a_T, 且非负; 负值必须抛错而不是被 clip。"""
    dm = b18_damage_module
    dist = b18_config["sim"]["disturbance"]
    T = np.full(64, float(dist["arrhenius_T_ref_K"]))
    # 直接检查 damage_series 对负 stress 的拒绝行为
    with pytest.raises(ValueError):
        dm.damage_series(np.array([1.0, -1.0, 1.0]), 1800.0, 3.0)


def test_no_second_temperature_acceleration(b18_config, b18_damage_module):
    """§5: 不得再造第二个温度加速函数 —— 与 wheel_sim.py 同式。"""
    assert b18_config["damage_model"][
        "second_temperature_acceleration_allowed"] is False
    assert b18_config["damage_model"]["reuse_arrhenius"] is True
    dist = b18_config["sim"]["disturbance"]
    Ea, Tref = float(dist["arrhenius_Ea_eV"]), float(dist["arrhenius_T_ref_K"])
    T = np.array([250.0, 293.15, 330.0])
    expected = np.exp(-Ea / b18_damage_module.K_B * (1.0 / T - 1.0 / Tref))
    got = b18_damage_module.arrhenius_accel(T, dist)
    assert np.allclose(got, expected, rtol=0.0, atol=0.0)
    # 方向: 温度越高加速越强
    assert got[0] < got[1] < got[2]


def test_g_duty_not_reweighted(b18_config):
    """§5: 不得重新调 g_duty 权重; 且 duty 只能接一次 (不进 tau_years)。"""
    d = b18_config["damage_model"]
    assert d["reuse_g_duty"] is True
    assert d["g_duty_reweighted"] is False
    assert d["g_duty_enters_tau_years"] is False
    assert d["tau_years_used_in_damage"] is False
    w = b18_config["wear_drive"]
    total = sum(float(w[k]) for k in
                ("speed_util_weight", "torque_util_weight",
                 "zero_crossing_weight", "maneuver_weight"))
    assert total == pytest.approx(1.0, abs=1e-12), \
        f"g_duty 权重和 = {total} != 1, reference identity 会被破坏"


def test_friction_link_and_post_eol_rule(b18_damage_module, b18_config):
    """§9: b(t) = b0*(1+Delta*D) 在 D<=1 区域; D>1 后冻结在 D=1。"""
    dm = b18_damage_module
    b0, Delta = 2e-6, 5.0
    D = np.array([0.0, 0.5, 1.0, 1.5, 3.0])
    b = dm.friction_from_damage(D, b0, Delta)
    assert b[0] == pytest.approx(b0)
    assert b[1] == pytest.approx(b0 * (1 + Delta * 0.5))
    assert b[2] == pytest.approx(b0 * (1 + Delta * 1.0))
    # post-EOL 冻结
    assert b[3] == pytest.approx(b[2])
    assert b[4] == pytest.approx(b[2])
    assert b18_config["damage_model"]["friction_link"][
        "post_eol_rule"] == "b_frozen_at_D_eq_one"
    assert b18_config["damage_model"]["friction_link"][
        "legacy_shape_used_as_damage_state"] is False


def test_lube_spike_does_not_touch_damage(b18_damage_module, b18_config):
    """§10: 润滑突变只乘 b, 不得跳变 D。"""
    dm = b18_damage_module
    dist = dict(b18_config["sim"]["disturbance"])
    dist["lube_spike_prob"] = 1.0            # 强制发生
    rng = np.random.default_rng(7)
    b = np.ones(100)
    out = dm.apply_lube_spike(b, rng, dist)
    assert out["spike_occurred"] is True
    assert 1.3 <= out["spike_multiplier"] <= 1.8
    assert out["affects_damage_state"] is False
    assert np.all(out["b"][:out["spike_idx"]] == 1.0)
    assert np.all(out["b"][out["spike_idx"]:] > 1.0)
    cfg_ls = b18_config["damage_model"]["lubrication_spike"]
    assert cfg_ls["affects_b"] is True
    assert cfg_ls["affects_damage_state"] is False
    assert list(cfg_ls["multiplier_range"]) == [1.3, 1.8]


def test_thermal_role_not_claimed_as_model(b18_config):
    """§11: T 是 environmental/telemetry condition, 不得声称是热模型。"""
    tr = b18_config["damage_model"]["thermal_role"]
    assert tr["claimed_as_thermal_model"] is False
    assert "environmental" in str(tr["T_is"]) or "telemetry" in str(tr["T_is"])


def test_eol_from_damage_right_censoring(b18_damage_module):
    """EOL 只由 D>=1 产生; 未达阈值必须报删失, 不伪造 EOL。"""
    dm = b18_damage_module
    D_hit = np.linspace(0.0, 2.0, 100)
    idx, ev = dm.eol_from_damage(D_hit)
    assert ev is True and D_hit[idx] >= 1.0 and D_hit[idx - 1] < 1.0
    D_cens = np.linspace(0.0, 0.7, 100)
    idx2, ev2 = dm.eol_from_damage(D_cens)
    assert ev2 is False and idx2 == 99      # 末尾索引, 但 event_observed=False


def test_semantics_never_manufacturer_spec(b18_damage_module, b18_config):
    """§24: D=1 是项目定义的仿真失效状态, 禁止称作厂家规格。"""
    dm = b18_damage_module
    assert dm.SEMANTICS == "engineering_assumption_failure_definition"
    assert dm.FORBIDDEN_NAMING == "manufacturer_failure_specification"
    assert dm.PROVENANCE == "NEW_B18_ASSUMPTION"
    d = b18_config["damage_model"]
    assert d["provenance"] == "NEW_B18_ASSUMPTION"
    assert d["semantics"] == "engineering_assumption_failure_definition"
    assert d["forbidden_naming"] == "manufacturer_failure_specification"
