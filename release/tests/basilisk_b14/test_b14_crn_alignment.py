# -*- coding: utf-8 -*-
"""BASILISK-B1.4 §10/§13/§14: CRN 对齐与动力学乘子恒等的测试。

核心命名测试:
  - test_crn_uniforms_preserved
  - test_gate13_new_prior_mapping
  - test_gate14_dynamics_multiplier_identity

**重要**: probe 实测 Gate 13 的 (d) 子判据 **未通过**
(`n_u_bitwise_equal = 54/60`, `n_mapping_ok = 0/60`)。本文件的职责是如实固定
这一事实并锁住它的成因, **不是**把判据放宽到能过。因此:
  * 断言 Gate 13 在 audit 中确实被判为 FAIL (若哪天变成 PASS 而容差未变,
    说明有人动了口径, 测试必须炸);
  * 断言两处偏差的成因与量级被钉死 —— u 的偏差是 B1.3 `b0_effective`
    一乘一除的 ≤4 ULP 往返误差; mapping 的偏差是 `q0` 记录口径取自
    `q[0] = b_true[0]/b_fail` 而非 `b0/b_fail`, 首样本已含一步退化,
    是**系统性正偏**而非随机噪声;
  * 断言真正与 q0 无关的部分 (排序、Gate 14 乘子) 仍然精确成立。
"""
import os
import struct

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CRN_HASH = "ff1de559b2746dcb394c4de6d70bda4d96ca594d1b953f58a4fa390366f41cf6"
Q0_MIN_NEW = 0.0017901569929920683
Q0_MAX_NEW = 0.017901569929920685
B_FAIL_F2 = 0.0005586102246421416
B0_SCALE_B13 = 23.114905847261028


def _ulp_diff(a: float, b: float) -> int:
    ia = struct.unpack("<q", struct.pack("<d", a))[0]
    ib = struct.unpack("<q", struct.pack("<d", b))[0]
    return abs(ia - ib)


# --------------------------------------------------------------------------
# §10: CRN 复用
# --------------------------------------------------------------------------
def test_crn_paired_trajectory_hash_reused(probe_rec, b14_cfg):
    """§10: 必须复用 B1.3 的 paired_trajectory_hash, 且实测 hash 与之相同。"""
    assert b14_cfg["probe"]["expected_paired_trajectory_hash"] == CRN_HASH
    if probe_rec is None:
        pytest.skip("probe_summary.json 未生成")
    assert probe_rec["paired_trajectory_hash_expected"] == CRN_HASH
    assert probe_rec["paired_trajectory_hash"] == CRN_HASH
    assert probe_rec["crn_preserved"] is True
    assert probe_rec["seed"] == 20260809


def test_crn_uniforms_preserved(probe_rec):
    """§13(d): uniform 的**排序**必须严格保持; 数值偏差只允许 ULP 级往返误差。

    实测 54/60 逐位相同, 6 条差 ≤4 ULP —— 成因是 B1.3 侧的 `b0_effective`
    是 `b0_raw * 23.114905847261028`, 这里为反解 u 又除回去, 一乘一除引入
    float64 往返误差。CRN **本身**未被破坏 (paired_trajectory_hash 一致,
    argsort 一致), 但 §13(d) 要求"逐位", 故该子判据如实判 FAIL。
    """
    if probe_rec is None:
        pytest.skip("probe_summary.json 未生成")
    ev = probe_rec["crn_uniform_evidence"]
    assert ev["n_traj"] == 60
    assert ev["b0_scale_b13_readonly"] == B0_SCALE_B13
    # 排序必须严格保持 —— 这才是 "同一组随机数" 的实质
    assert ev["rank_order_preserved"] is True
    # 如实固定实测值, 不放宽也不粉饰
    assert ev["n_u_bitwise_equal"] == 54
    assert ev["all_u_bitwise_equal"] is False
    # 不逐位相同的那几条, 偏差必须只在 ULP 量级 (证明是往返误差而非换了随机流)
    bad = [r for r in ev["per_trajectory"] if not r["u_bitwise_equal"]]
    assert len(bad) == 6
    for r in bad:
        d = _ulp_diff(r["u_b14"], r["u_b13"])
        assert d <= 4, "traj %d 的 u 偏差 %d ULP 超出往返误差量级" % (r["traj_id"], d)
    for r in ev["per_trajectory"]:
        assert 0.0 <= r["u_b14"] <= 1.0


def test_b0_scale_is_exact_identity(probe_rec, prior_rec):
    """本先验的实现恰为 b0_scale = 1.0 —— b14 侧的 u 没有任何缩放往返误差。"""
    if probe_rec is None:
        pytest.skip("probe_summary.json 未生成")
    assert probe_rec["b0_scale_b14"] == 1.0
    assert probe_rec["b0_scale_b11_replaced"] != 1.0
    if prior_rec is not None:
        b0_hi = prior_rec["derivation"]["b0_documented_range_Nms_per_rad"][1]
        assert abs(Q0_MAX_NEW * B_FAIL_F2 / b0_hi - 1.0) < 1e-12


# --------------------------------------------------------------------------
# §13: 先验符合性
# --------------------------------------------------------------------------
def test_gate13_new_prior_mapping(probe_rec, protocol_rec):
    """§13(a)(b)(c) 成立, (d) 的 mapping 偏差成因必须被钉死。

    `q0` 的记录口径是 `q[0] = b_true[0]/b_fail`, 而 `b_true[0]` 已含首样本的
    一步退化, 故 `q0_observed / (b0/b_fail) > 1` 是**系统性正偏**, 与
    `q0_min + u*(q0_max - q0_min)` 的纯线性映射存在 ~1e-4 级差, 远超冻结的
    1e-12 相对容差。这是记账口径问题, 不是先验实现错误 —— 但 §13 的容差在
    probe 前已冻结, 不得事后放宽, 因此 (d) 如实判 FAIL。
    """
    if probe_rec is None:
        pytest.skip("probe_summary.json 未生成")
    ev = probe_rec["crn_uniform_evidence"]
    tol = 1e-12
    if protocol_rec is not None:
        tol = protocol_rec["frozen_values"]["gate13_q0_support_tol_rel"]
        assert tol == 1e-12, "Gate 13 容差被改动"
        assert protocol_rec["frozen_values"]["gate13_semantics"] == \
            "PRIOR_CONFORMANCE_AND_CRN_UNIFORM_IDENTITY"
        assert protocol_rec["frozen_values"]["q0_distribution"] == "uniform"
    # (a)(b): 观测 q0 整体落在新 support 内 —— 先验实现本身是对的
    assert probe_rec["q0_range_new"] == [Q0_MIN_NEW, Q0_MAX_NEW]
    assert probe_rec["q0_observed_min"] >= Q0_MIN_NEW * (1.0 - 1e-12)
    assert probe_rec["q0_observed_max"] <= Q0_MAX_NEW * (1.0 + 1e-12)
    # (d): 如实固定 —— 全部 60 条都超出 1e-12, 且偏差全为正 (系统性, 非随机)
    assert ev["all_mapping_ok"] is False
    assert ev["n_mapping_ok"] == 0
    assert ev["max_mapping_rel_err"] > tol
    assert ev["max_mapping_rel_err"] < 1e-2, "偏差量级异常, 不像首样本退化所致"
    for r in ev["per_trajectory"]:
        assert r["q0_observed"] > r["q0_expected"], (
            "traj %d 偏差为负 —— 与'首样本已含一步退化'的解释矛盾" % r["traj_id"])


def test_gate13_mapping_error_explained_by_first_sample_degradation(probe_rec):
    """把 (d) 的偏差成因锁死: q0_observed/(b0/b_fail) - 1 == mapping 相对误差。"""
    if probe_rec is None:
        pytest.skip("probe_summary.json 未生成")
    ev = probe_rec["crn_uniform_evidence"]
    recs = {int(r["traj_id"]): r for r in probe_rec["per_trajectory"]}
    worst = 0.0
    for r in ev["per_trajectory"]:
        b0 = recs[r["traj_id"]]["b0_effective"]
        # b0/b_fail 是"未退化"的 q0; 观测 q0 取自 q[0], 已含一步退化
        undegraded = b0 / B_FAIL_F2
        excess = r["q0_observed"] / undegraded - 1.0
        assert excess > 0.0
        # 该正偏必须与 audit 记录的 mapping 相对误差同量级 (同一个原因)
        assert abs(excess - r["rel_err"]) < 1e-6, (
            "traj %d: 首样本退化 %.3e 与 mapping 误差 %.3e 不同源"
            % (r["traj_id"], excess, r["rel_err"]))
        worst = max(worst, excess)
    assert abs(worst - ev["max_mapping_rel_err"]) < 5e-6


def test_gate13_reported_as_fail_not_relaxed(probe_audit_rec, protocol_rec):
    """Gate 13 必须在 audit 中如实判 FAIL, 且容差未被事后放宽。"""
    if probe_audit_rec is None:
        pytest.skip("probe_audit.json 未生成")
    g13 = [g for g in probe_audit_rec["gates"] if g["gate"] == 13]
    assert len(g13) == 1, "未找到唯一的 Gate 13"
    val = g13[0]["value"]
    assert g13[0]["name"] == "q0_prior_conformance_and_crn_identity"
    assert val["semantics"] == "PRIOR_CONFORMANCE_AND_CRN_UNIFORM_IDENTITY"
    # (a)(b)(c) 成立, (d) 不成立 -> 整条 FAIL
    assert val["a_sampled_min_within_support"] is True
    assert val["b_sampled_max_not_exceeding"] is True
    assert val["c_distribution_form_matches"] is True
    assert val["d_crn_uniform_bitwise_and_mapping"] is False
    assert g13[0]["pass"] is False, "Gate 13 在 (d) 不成立的情况下被判 PASS"
    assert val["support_tol_rel"] == 1e-12
    # KS 只是诊断, 不得成为 PASS 依据 (它恰好 consistent, 更不能拿来救 Gate)
    ks = val["diagnostic_ks_uniformity"]
    assert "仅诊断" in ks["note"]
    assert ks["consistent"] is True and g13[0]["pass"] is False


# --------------------------------------------------------------------------
# §14: 动力学乘子恒等
# --------------------------------------------------------------------------
def test_gate14_dynamics_multiplier_identity(probe_rec, protocol_rec):
    """§14: 归一化 q(t)/q0 与 B1.3 逐点一致 (仅 ULP 级差异) —— 实测 2 ULP PASS。"""
    if probe_rec is None:
        pytest.skip("probe_summary.json 未生成")
    q = probe_rec["q_multiplier_reference"]
    tol_ulp, tol_abs = 8.0, 1e-12
    if protocol_rec is not None:
        fv = protocol_rec["frozen_values"]
        assert fv["gate14_tolerance_mode"] == "ulp"
        tol_ulp = fv["gate14_tolerance_ulp"]
        tol_abs = fv["gate14_abs_backstop"]
        assert tol_ulp == 8.0 and tol_abs == 1e-12, "Gate 14 容差被改动"
    assert q["max_abs_diff"] <= tol_abs, "max_abs_diff = %.3e" % q["max_abs_diff"]
    assert q["max_abs_diff_in_ulp"] <= tol_ulp
    # b0 区间故意取不同 —— 正是这一点证明乘子与 b0/q0 无关
    assert q["b0_range_b14"] != q["b0_range_b13"]
    assert q["n_total"] > 0 and q["n_bitwise_equal"] <= q["n_total"]


def test_gate14_pass_in_audit(probe_audit_rec):
    """Gate 14 应 PASS —— q(t)/q0 与 q0 无关, 这是 §14 的结构性结论。"""
    if probe_audit_rec is None:
        pytest.skip("probe_audit.json 未生成")
    g14 = [g for g in probe_audit_rec["gates"] if g["gate"] == 14][0]
    assert g14["pass"] is True
    assert g14["value"]["max_abs_diff_in_ulp"] <= g14["value"]["tolerance_ulp"]


def test_only_b0_rescaled_in_probe(probe_rec):
    """§10: probe 自报的变更项只有 b0 support; 关键不变项必须齐备。"""
    if probe_rec is None:
        pytest.skip("probe_summary.json 未生成")
    assert probe_rec["only_q0_prior_changed"] is True
    changed = probe_rec["changed_params"]
    assert len(changed) == 1 and "b0" in changed[0]
    unchanged = set(probe_rec["unchanged"])
    for must in ("g_duty", "mode_weights", "horizon", "Im_rated",
                 "F2 threshold", "Delta_range", "tau_years_range",
                 "lubrication spike law", "noise seed"):
        assert must in unchanged, "未声明不变: %s" % must
    assert probe_rec["failure_threshold_Nm"] == 0.14311462970213382
    assert probe_rec["b_fail_f2_Nms_per_rad"] == B_FAIL_F2
    assert probe_rec["window_samples"] == 48
    assert probe_rec["window_quantile"] == 0.95
    assert probe_rec["persistence_windows"] == 4
