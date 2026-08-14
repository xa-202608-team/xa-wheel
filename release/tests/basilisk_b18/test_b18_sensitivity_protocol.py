"""tests/basilisk_b18/test_b18_sensitivity_protocol.py —— §12/§15 CRN 与方向性。

含 §22 要求的 `test_fast_damage_not_slower`、`test_slow_damage_not_faster`、
`test_crn_aligned`。
"""
from __future__ import annotations

import numpy as np
import pytest

ORDER = ("SLOW", "NOMINAL", "FAST")


def test_crn_aligned(b18_sensitivity, b18_config, b18_audit):
    """§12: 三情景共用同一份随机输入, 唯一差异 = L_ref。"""
    s = b18_sensitivity
    assert s["paired_trajectory_hash"] == \
        s["paired_trajectory_hash_expected"], "CRN 断裂"
    assert s["paired_trajectory_hash"] == str(
        b18_config["sensitivity"]["expected_paired_trajectory_hash"])
    assert s["crn_preserved"] is True
    assert s["only_difference"] == "L_ref"
    assert s["shared_core_computed_once_for_all_scenarios"] is True
    assert s["crn_shared_unit_normals"] is True

    # 独立复核 (audit_sensitivity 的 crn_recheck)
    c = b18_audit["crn_recheck"]
    assert c["paired_hash_matches"] is True
    assert c["reproducible"] is True, "同输入两遍重算结果不同"
    assert c["scenarios_do_not_mutate_shared_core"] is True, \
        "build_scenario 污染了共享输入, 三情景差异不止 L_ref"


def test_crn_shared_params_identical_across_scenarios(b18_sensitivity):
    """逐轨迹核对: 三情景的物理参数与 duty 统计必须逐值相同。"""
    ps = b18_sensitivity["per_scenario"]
    ref = ps["NOMINAL"]
    for name in ("SLOW", "FAST"):
        for a, b in zip(ref, ps[name]):
            assert a["traj_id"] == b["traj_id"]
            for k in ("b0", "Delta", "tau_years", "omega0", "g_duty_mean",
                      "a_T_mean", "stress_mean", "n_samples"):
                assert a[k] == pytest.approx(b[k], rel=0.0, abs=0.0), \
                    f"traj {a['traj_id']} 的 {k} 在 {name} 与 NOMINAL 不同"
            assert a["mode_hist"] == b["mode_hist"]
            assert a["spike_occurred"] == b["spike_occurred"]


def test_damage_scales_exactly_as_inverse_lref(b18_sensitivity):
    """§12 的数学后果: D_end 严格按 1/L_ref 缩放 (逐轨迹核对)。

    这比"方向对"强得多 —— 它证明 L_ref 只是一个后置除数。
    """
    ps = b18_sensitivity["per_scenario"]
    L = {k: float(ps[k][0]["L_ref_years"]) for k in ORDER}
    for name in ("SLOW", "FAST"):
        for a, b in zip(ps["NOMINAL"], ps[name]):
            expected = a["D_end"] * L["NOMINAL"] / L[name]
            assert b["D_end"] == pytest.approx(expected, rel=1e-9), \
                f"traj {a['traj_id']}: {name} 的 D_end 不等于 NOMINAL·L_N/L_{name}"


def test_fast_damage_not_slower(b18_audit):
    """§15: FAST 的 event fraction 不得低于 NOMINAL, median EOL 不得更长。"""
    sc = b18_audit["scenarios"]
    assert sc["FAST"]["event_fraction"] >= sc["NOMINAL"]["event_fraction"], \
        "FAST 损伤更快, event fraction 却更低 => L_ref 实现错误"
    assert (sc["FAST"]["eol_years_event_observed"]["p50"]
            <= sc["NOMINAL"]["eol_years_event_observed"]["p50"]), \
        "FAST 的 median EOL 更长 => L_ref 实现错误"


def test_slow_damage_not_faster(b18_audit):
    """§15: SLOW 的 event fraction 不得高于 NOMINAL, median EOL 不得更短。"""
    sc = b18_audit["scenarios"]
    assert sc["SLOW"]["event_fraction"] <= sc["NOMINAL"]["event_fraction"], \
        "SLOW 损伤更慢, event fraction 却更高 => L_ref 实现错误"
    assert (sc["SLOW"]["eol_years_event_observed"]["p50"]
            >= sc["NOMINAL"]["eol_years_event_observed"]["p50"]), \
        "SLOW 的 median EOL 更短 => L_ref 实现错误"


def test_ordering_verdict_recorded(b18_audit, b18_config):
    """§15 判定结果与失败 verdict 名称必须落档。"""
    o = b18_audit["ordering"]
    assert o["passed"] is True
    assert len(o["checks"]) == 2
    assert all(c["passed"] for c in o["checks"])
    assert str(b18_config["sensitivity_ordering"]["fail_verdict"]) == \
        "B18_DAMAGE_MODEL_INVALID"


def test_sensitivity_n_traj_and_no_telemetry(b18_sensitivity, b18_config):
    """§12/§13: 每情景 60 条; sensitivity 不落 telemetry。"""
    assert b18_sensitivity["n_traj"] == 60 == int(
        b18_config["sensitivity"]["n_traj"])
    for k in ORDER:
        assert len(b18_sensitivity["per_scenario"][k]) == 60
    assert b18_sensitivity["persist_telemetry"] is False


def test_nominal_gates_all_passed(b18_audit):
    """§14: NOMINAL 的 14 条 Gate 全过, 且 gate 编号完整。"""
    g = b18_audit["nominal_gate"]
    assert g["n_gates"] == 14, f"应有 14 条 Gate, 实为 {g['n_gates']}"
    assert sorted(x["no"] for x in g["gates"]) == list(range(1, 15))
    assert g["all_passed"] is True, f"未过: {g['failed_gates']}"
    assert b18_audit["verdict"] == "B18_SENSITIVITY_OK"


def test_high_load_modes_damage_faster(b18_audit):
    """§14 Gate 4 的独立复核: imaging/nadir/desat 的损伤速率高于 cruise/safe。"""
    pm = b18_audit["scenarios"]["NOMINAL"]["per_mode_damage_rate"]
    hi = [pm[m]["mean_damage_rate_per_year"]
          for m in ("imaging", "nadir", "desat") if pm[m]["n_windows"] > 0]
    lo = [pm[m]["mean_damage_rate_per_year"]
          for m in ("cruise", "safe") if pm[m]["n_windows"] > 0]
    assert hi and lo
    assert min(hi) > max(lo), f"high {hi} 未全部高于 low {lo}"
    # 空 bin 必须是 NaN + n=0, 不是伪造的 0
    for m, v in pm.items():
        if v["n_windows"] == 0:
            assert np.isnan(v["mean_damage_rate_per_year"]), \
                f"{m} 空 bin 应为 NaN 而非 {v['mean_damage_rate_per_year']}"


def test_censored_not_counted_as_eol(b18_sensitivity):
    """右删失轨迹不得伪造 EOL: D 未达阈值时 event_observed 必须为 False。"""
    for name in ORDER:
        for r in b18_sensitivity["per_scenario"][name]:
            if not r["event_observed"]:
                assert r["D_max"] < 1.0, \
                    f"{name} traj {r['traj_id']}: 删失却 D_max>=1"
                assert r["n_post_eol_rows"] == 0
            else:
                assert r["D_at_eol"] >= 1.0


def test_eol_iqr_denominators(b18_audit):
    """分位数/IQR 只在 event-observed 上算; early/late 分母口径与 B1.2 一致。"""
    for name in ORDER:
        s = b18_audit["scenarios"][name]
        assert s["eol_years_event_observed"]["n"] == s["n_event_observed"]
        assert s["early_eol"]["denominator"] == "all_trajectories"
        assert s["late_eol"]["denominator"] == "event_observed_only"
