"""tests/basilisk_b13/test_b13_degradation_mapping.py

§3/§4/§5/§7/§19 —— 退化方程审计与力矩域映射的正确性。

§19 命名测试:
  * test_degradation_equation_audited
  * test_bfail_f2_derived
  * test_q0_mapping
  * test_normalized_degradation_mapping
  * test_no_im_rated_in_failure_definition
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import CFG_REL  # noqa: F401

# B1.2 冻结值 (只读; 任何一处对不上就说明上游被改了)
T_F_FAIL_NM = 0.14311462970213382
TC_NOM_NM = 0.00125
OMEGA_RATED = 628.3185307179587
SU_REF_P95 = 0.4041898280945392
B_FAIL_F2 = 0.0005586102246421416
B_FAIL_OLD = 2.416666666666667e-05
Q0_OLD_RANGE = (0.04137931034482758, 0.41379310344827586)


def test_degradation_equation_audited(eq_audit_rec):
    """§3: 方程结构由**读源码**确定, 8 项检查全过, 分支落在乘性。"""
    if eq_audit_rec is None:
        pytest.skip("需先跑 audit_degradation_equation.py")
    assert eq_audit_rec["all_checks_pass"], \
        f"审计未全过: {[k for k, v in eq_audit_rec['checks'].items() if not v]}"
    assert eq_audit_rec["resolved_branch"] == "MULTIPLICATIVE_b0_PREFACTOR"
    # 方程原文必须真的含 b0 前因子与 exp 饱和
    src = eq_audit_rec["sources"]["wheel_sim"]["b_assign"]["source"]
    assert "b0" in src and "exp" in src and "Delta" in src, f"方程原文异常: {src}"
    # 审计必须指向 simulate 里的**主**赋值, 不是润滑跳变那行
    assert "*=" not in src, "抓到的是润滑跳变行 (b[spike:] *= ...), 不是主退化方程"


def test_degradation_equation_reflects_current_source(eq_audit_rec, b13_verify):
    """审计记录的 wheel_sim.py hash 必须与当下文件一致 —— 否则审计已过期。"""
    if eq_audit_rec is None:
        pytest.skip("需先跑 audit_degradation_equation.py")
    ws = eq_audit_rec["sources"]["wheel_sim"]
    assert b13_verify.sha256_of(ws["rel"]) == ws["sha256"], \
        "src/sim/wheel_sim.py 在审计后被改动 —— §0 明令不得修改"


def test_multiplicative_branch_means_only_b0_rescaled(eq_audit_rec, scaling_rec):
    """§7 的分叉: 乘性 -> 只缩放 b0, Delta/tau 必须保持不变。"""
    if eq_audit_rec is None or scaling_rec is None:
        pytest.skip("需先跑 §3 / §4 脚本")
    assert eq_audit_rec["rescale_params"] == ["b0"]
    assert set(eq_audit_rec["invariant_params"]) == {"Delta", "tau_years"}
    m = scaling_rec["mapping"]
    assert m["Delta_rescaled"] is False, "Delta 被缩放了 —— 乘性分支下这是错的"
    assert m["tau_rescaled"] is False, "tau_years 被缩放了 —— 它是时间尺度, 与 b_fail 无关"


def test_no_im_rated_in_failure_definition(scaling_rec, b13_cfg):
    """§4: b_fail_f2 的推导链**不得**含 Im_rated (F1 已被 provenance 否决)。"""
    if scaling_rec is None:
        pytest.skip("需先跑 derive_torque_scaling.py")
    f2 = scaling_rec["f2_torque_domain"]
    assert f2["uses_Im_rated"] is False
    assert f2["uses_Kt"] is False
    assert b13_cfg["failure_definition"]["depends_on_im_rated"] is False
    # 数值层面: b_fail_f2 必须能在**不知道** Im_rated 的情况下复算出来
    recomputed = (T_F_FAIL_NM - TC_NOM_NM) / (SU_REF_P95 * OMEGA_RATED)
    assert recomputed == B_FAIL_F2, \
        f"b_fail_f2 复算不一致: {recomputed!r} != {B_FAIL_F2!r}"
    assert scaling_rec["mapping"]["im_rated_touched"] is False


def test_bfail_f2_derived(scaling_rec):
    """§4: b_fail_f2 = (T_f_fail − Tc_nom) / omega_ref, 逐位可复现。"""
    if scaling_rec is None:
        pytest.skip("需先跑 derive_torque_scaling.py")
    f2 = scaling_rec["f2_torque_domain"]
    assert f2["T_f_fail_Nm"] == T_F_FAIL_NM, "F2 阈值被改了 —— §禁止"
    assert f2["Tc_nom_Nm"] == TC_NOM_NM
    omega_ref = SU_REF_P95 * OMEGA_RATED
    assert f2["omega_ref_rad_s"] == omega_ref
    assert f2["b_fail_f2_Nms_per_rad"] == B_FAIL_F2
    # 与 B1.2 独立记录的值交叉核对
    assert scaling_rec["b12_frozen_values_readback"]["T_f_fail_Nm"] == T_F_FAIL_NM


def test_q0_mapping(scaling_rec):
    """§5/§12: q0 = b0/b_fail 在新旧口径下**逐位相同**的区间。"""
    if scaling_rec is None:
        pytest.skip("需先跑 derive_torque_scaling.py")
    m = scaling_rec["mapping"]
    assert tuple(m["q0_old_range"]) == Q0_OLD_RANGE
    # 新 b0 区间 / b_fail_f2 必须回到同一个 q0 区间
    for b0_new, q_expect in zip(m["b0_range_new"], Q0_OLD_RANGE):
        assert b0_new / B_FAIL_F2 == pytest.approx(q_expect, rel=1e-12), \
            f"b0_new={b0_new!r} 归一化后不等于 q0_old={q_expect!r}"
    # 旧 b0 区间 / b_fail_old 也应给出同一 q0 区间 (这是映射的出发点)
    for b0_old, q_expect in zip(m["b0_range_old"], Q0_OLD_RANGE):
        assert b0_old / B_FAIL_OLD == pytest.approx(q_expect, rel=1e-12)


def test_q0_old_equals_b11_beta_range_design(scaling_rec):
    """独立交叉验证: q0_old 与 B1.1 记录的 beta_range_design 逐位相同。

    这不是巧合 —— q = b/b_fail **就是**无量纲摩擦裕度消耗率 beta。
    它是 §5 要求"用 normalized degradation state 表达映射"的物理依据。
    """
    if scaling_rec is None:
        pytest.skip("需先跑 derive_torque_scaling.py")
    import json
    from pathlib import Path
    p = (Path(__file__).resolve().parents[2] / "checkpoints" / "basilisk_b11"
         / "candidates.json")
    if not p.exists():
        pytest.skip("缺 B1.1 candidates.json")
    beta = json.loads(p.read_text(encoding="utf-8"))["b0_calibration"][
        "beta_range_design"]
    assert tuple(float(v) for v in beta) == Q0_OLD_RANGE, \
        f"q0_old {Q0_OLD_RANGE} != B1.1 beta_range_design {beta}"


def test_normalized_degradation_mapping(scaling_rec):
    """§5: 映射通过归一化状态成立 —— b0_scale 恰等于两个 b_fail 之比。

    注意这**不是**在验证"硬编码了比值", 而是在验证从 q0 守恒推出的 b0_scale
    与代数上应有的值一致。区别在于 b0_scale 的**来源**: 它由
    `q0_old · b_fail_f2 / b0_range_old` 运行时算出 (见 test_no_hardcoded_scale_ratio
    对源码的 AST 扫描), 这里只检查算出来的数对不对。
    """
    if scaling_rec is None:
        pytest.skip("需先跑 derive_torque_scaling.py")
    m = scaling_rec["mapping"]
    s = float(m["b0_scale"])
    assert s == B_FAIL_F2 / B_FAIL_OLD, \
        f"b0_scale {s!r} != b_fail_f2/b_fail_old {B_FAIL_F2 / B_FAIL_OLD!r}"
    # 两端点独立算出的 scale 必须一致 (否则映射不是纯乘性)
    assert m["scale_consistent_across_range"] is True
    assert float(m["b0_scale_lo_endpoint"]) == pytest.approx(
        float(m["b0_scale_hi_endpoint"]), rel=1e-12)


def test_normalized_trajectory_invariance_algebra():
    """§7 代数: q(t) 在联合缩放下不变 (b0→s·b0, b_fail→s·b_fail)。

    直接对着 §3 审定的方程算, 不依赖任何产物文件 —— 这是映射的数学根据。
    容差用 ULP 口径 (浮点乘除顺序不同会有 1-2 ULP 舍入, 见 protocol Gate 14)。
    """
    rng = np.random.default_rng(20260809)
    s = B_FAIL_F2 / B_FAIL_OLD
    worst_ulp = 0.0
    for _ in range(500):
        b0 = float(rng.uniform(*[1e-06, 1e-05]))
        D = float(rng.uniform(3.0, 10.0))
        tau = float(rng.uniform(0.5, 2.0))
        integ = np.linspace(0.0, 3.0, 256)
        shape = 1.0 + D * (1.0 - np.exp(-integ / tau))
        q_old = (b0 * shape) / B_FAIL_OLD
        q_new = ((b0 * s) * shape) / (B_FAIL_OLD * s)
        d = np.max(np.abs(q_new - q_old))
        worst_ulp = max(worst_ulp, d / np.spacing(float(np.max(q_old))))
    assert worst_ulp <= 8.0, f"归一化不变性超出 8 ULP: {worst_ulp:.2f} ULP"
