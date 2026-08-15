# -*- coding: utf-8 -*-
"""BASILISK-B1.4 §4-§7: healthy-entry q0 先验推导的测试。

所有键名照 checkpoints/basilisk_b14/healthy_entry_prior.json 实际结构。
"""
import io
import json
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

B_FAIL_F2 = 0.0005586102246421416
OMEGA_REF = 253.95995891950514
T_F_FAIL = 0.14311462970213382
TC_NOM = 0.00125
B0_DOC = (1e-06, 1e-05)
Q0_MIN_NEW = 0.0017901569929920683
Q0_MAX_NEW = 0.017901569929920685
Q0_OLD = (0.04137931034482758, 0.41379310344827586)
B0_SCALE_B13 = 23.114905847261028


def _p(*parts):
    return os.path.join(ROOT, *parts)


# --------------------------------------------------------------------------
# §4 推导本体
# --------------------------------------------------------------------------
def test_q0_support_equals_documented_b0_over_b_fail(prior_rec):
    """q0 = b0_documented / b_fail_f2, 逐位可复算。"""
    if prior_rec is None:
        pytest.skip("healthy_entry_prior.json 未生成")
    d = prior_rec["derivation"]
    assert d["rule"] == "documented_absolute_b0_over_frozen_b_fail_f2"
    assert d["distribution"] == "uniform"
    assert d["b0_documented_range_Nms_per_rad"] == list(B0_DOC)
    assert d["b_fail_f2_Nms_per_rad"] == B_FAIL_F2
    # 独立复算, 要求逐位相同 (不给容差 —— 这是纯除法)
    assert B0_DOC[0] / B_FAIL_F2 == d["q0_min"]
    assert B0_DOC[1] / B_FAIL_F2 == d["q0_max"]
    assert d["q0_range"] == [d["q0_min"], d["q0_max"]]
    assert d["q0_min"] == Q0_MIN_NEW
    assert d["q0_max"] == Q0_MAX_NEW
    assert d["q0_min"] < d["q0_max"], "区间不得倒挂"


def test_b0_restored_to_documented_absolute_magnitude(prior_rec):
    """新先验的物理意义: b0 = q0 * b_fail_f2 精确回到文档绝对量级。"""
    if prior_rec is None:
        pytest.skip("healthy_entry_prior.json 未生成")
    d = prior_rec["derivation"]
    assert d["b0_equals_documented_exactly"] is True
    assert d["b0_range_new_Nms_per_rad"] == list(B0_DOC)
    # 收缩因子恰为 B1.3 b0_scale 的倒数 —— 说明这就是「把 b0 还原到文档值」
    shrink = d["shrink_factor_vs_old"]
    assert abs(shrink * B0_SCALE_B13 - 1.0) < 1e-12
    assert abs(Q0_MAX_NEW / Q0_OLD[1] - shrink) < 1e-15


def test_u_initial_is_conversion_not_independent_input(prior_rec):
    """§4 的 utilization 是等价换算, 反解必须自洽 (不是另选的一个数字)。"""
    if prior_rec is None:
        pytest.skip("healthy_entry_prior.json 未生成")
    u = prior_rec["derivation"]["utilization"]
    assert u["formula"] == "(Tc_nom + b0 * omega_ref) / T_f_fail"
    assert u["linear_relation_consistent"] is True
    # 独立复算 u_initial_max
    recomputed = (TC_NOM + B0_DOC[1] * OMEGA_REF) / T_F_FAIL
    assert abs(recomputed - u["u_initial_max"]) < 1e-15
    # c = Tc/T_f_fail, 且 u = c + q0*(1-c)
    c = TC_NOM / T_F_FAIL
    assert abs(c - u["coulomb_share_c"]) < 1e-15
    assert abs(c + Q0_MAX_NEW * (1.0 - c) - u["u_initial_max"]) < 1e-15
    bs = prior_rec["derivation"]["back_solve_self_consistency"]
    assert bs["consistent"] is True
    assert bs["rel_err_b0"] < 1e-12
    assert bs["rel_err_q0"] == 0.0


# --------------------------------------------------------------------------
# §5 q0_min 重定义的例外条件
# --------------------------------------------------------------------------
def test_q0_min_redefinition_is_justified(prior_rec):
    """§5: 默认保留旧 q0_min; 只有在证明其同样无依据、且保留会导致区间倒挂时才可重定义。"""
    if prior_rec is None:
        pytest.skip("healthy_entry_prior.json 未生成")
    d = prior_rec["derivation"]
    assert d["q0_min_redefined"] is True
    assert d["q0_min_old"] == Q0_OLD[0]
    assert d["q0_max_old"] == Q0_OLD[1]
    # 倒挂事实必须成立: 旧 min 0.0414 > 新 max 0.0179
    assert d["interval_would_invert_if_min_kept"] is True
    assert Q0_OLD[0] > Q0_MAX_NEW
    reason = d["q0_min_redefinition_reason"]
    assert isinstance(reason, str) and len(reason) > 20
    assert "Im_rated" in reason


# --------------------------------------------------------------------------
# §6 Δ / tau / 润滑突变必须保持 B1.3
# --------------------------------------------------------------------------
def test_delta_unchanged(prior_rec):
    """§6/§21: Δ ∈ [3,10] 不变, 且不得与 q0 同时改动。"""
    if prior_rec is None:
        pytest.skip("healthy_entry_prior.json 未生成")
    d = prior_rec["derivation"]
    assert d["Delta_range"] == [3.0, 10.0]
    assert d["Delta_unchanged"] is True
    assert prior_rec["frozen_inputs"]["Delta_range"] == [3.0, 10.0]
    assert prior_rec["Delta_touched"] is False


def test_tau_unchanged(prior_rec):
    """§6/§21: tau ∈ [0.5,2.0] 年不变。"""
    if prior_rec is None:
        pytest.skip("healthy_entry_prior.json 未生成")
    d = prior_rec["derivation"]
    assert d["tau_years_range"] == [0.5, 2.0]
    assert d["tau_unchanged"] is True
    assert prior_rec["frozen_inputs"]["tau_years_range"] == [0.5, 2.0]
    assert prior_rec["tau_touched"] is False
    assert "1.3" in d["lube_spike_law"] and "1.8" in d["lube_spike_law"]


# --------------------------------------------------------------------------
# §7 b0 映射 / 阈值不得改动
# --------------------------------------------------------------------------
def test_f2_threshold_unchanged(prior_rec, protocol_rec):
    """§21: F2 阈值不得调整。"""
    if prior_rec is None:
        pytest.skip("healthy_entry_prior.json 未生成")
    assert prior_rec["f2_threshold_touched"] is False
    assert prior_rec["frozen_inputs"]["T_f_fail_Nm"] == T_F_FAIL
    if protocol_rec is not None:
        fv = protocol_rec["frozen_values"]
        assert fv["failure_threshold_Nm"] == T_F_FAIL
        assert fv["failure_definition"] == "F2_FRICTION_TORQUE_P95"
        assert fv["failure_window_samples"] == 48
        assert fv["failure_quantile"] == 0.95
        assert fv["persistence_windows"] == 4


def test_bfail_f2_unchanged(prior_rec):
    """§21: b_fail_f2 不得调整, 且必须与 (T_f_fail - Tc)/omega_ref 逐位一致。"""
    if prior_rec is None:
        pytest.skip("healthy_entry_prior.json 未生成")
    fi = prior_rec["frozen_inputs"]
    assert prior_rec["b_fail_f2_touched"] is False
    assert fi["b_fail_f2_Nms_per_rad"] == B_FAIL_F2
    assert fi["b_fail_f2_verified"] is True
    assert (T_F_FAIL - TC_NOM) / OMEGA_REF == fi["b_fail_f2_recomputed"]
    assert fi["b_fail_f2_recomputed"] == B_FAIL_F2
    # b0 = q0 * b_fail_f2 的单调映射公式本身不得改
    assert prior_rec["b0_mapping_formula_touched"] is False


# --------------------------------------------------------------------------
# §4 结构可达性 (诚实落盘的推论, 不是 Gate)
# --------------------------------------------------------------------------
def test_reachability_algebra_self_consistent(prior_rec):
    """q 上确界 = q0_max*(1+Δ_max)*spike_max, 与 verdict 一致; 只用冻结量推出。"""
    if prior_rec is None:
        pytest.skip("healthy_entry_prior.json 未生成")
    r = prior_rec["reachability"]
    q_sup = Q0_MAX_NEW * (1.0 + 10.0) * 1.8
    assert abs(q_sup - r["q_supremum"]) < 1e-15
    assert r["failure_requires_q_ge"] == 1.0
    # verdict 必须由代数决定, 不能与数值矛盾
    if r["q_supremum"] < 1.0:
        assert r["f2_failure_reachable"] is False
        assert r["verdict"] == "STRUCTURALLY_UNREACHABLE"
    else:
        assert r["f2_failure_reachable"] is True
    # 力矩域交叉验证: T_f_sup = Tc + q_sup*b_fail*omega_ref
    tf_sup = TC_NOM + r["q_supremum"] * B_FAIL_F2 * OMEGA_REF
    assert abs(tf_sup - r["Tf_supremum_Nm"]) < 1e-15
    assert r["Tf_supremum_Nm"] < r["T_f_fail_Nm"]
    assert abs(r["Tf_supremum_Nm"] / T_F_FAIL - r["Tf_sup_over_threshold"]) < 1e-12
    # 要达到 q=1 所需的 Δ 远超冻结上界 —— 说明不能靠改 Δ 逃避 (而 §6 也禁止改)
    assert r["Delta_needed_for_q_eq_1_with_max_spike"] > r["Delta_max_available"]
    assert r["Delta_max_available"] == 10.0
