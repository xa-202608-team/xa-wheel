# -*- coding: utf-8 -*-
"""BASILISK-B1.4 §2/§3: q0 先验来源审计的测试。

键名一律照 checkpoints/basilisk_b14/q0_provenance_audit.json 实际结构读取
(q1_physical_meaning / q2_old_q0_arithmetic / q3_documented_b0 / q4_healthy_entry /
 hr16_official / provenance_gate)。猜键名会让 assert 永远命中 KeyError 分支,
保护形同虚设 —— 这是本项目已犯过两次的错, 此处显式钉死。
"""
import io
import json
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _p(*parts):
    return os.path.join(ROOT, *parts)


def _read_json(rel):
    path = _p(*rel.split("/"))
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# §3 provenance gate
# --------------------------------------------------------------------------
def test_provenance_gate_verdict_sufficient(prov_rec):
    """§3: 只有 SUFFICIENT 才允许继续生成 probe。"""
    if prov_rec is None:
        pytest.skip("q0_provenance_audit.json 未生成")
    gate = prov_rec["provenance_gate"]
    assert gate["verdict"] == "B14_Q0_PROVENANCE_SUFFICIENT"
    assert gate["pass"] is True
    # 两条路径至少一条成立 (path_A 已有明确依据 / path_B 唯一可推导)
    assert gate["path_A_explicit_existing_basis"] is True
    assert gate["path_B_unique_derivation"] is True


def test_provenance_only_allowed_sources_used(prov_rec):
    """§2: 只查允许来源; B1.3 结果类文件必须登记为「未读」。"""
    if prov_rec is None:
        pytest.skip("q0_provenance_audit.json 未生成")
    allowed = prov_rec["allowed_sources"]
    # 允许来源必须是需求/设计/config/官方器件参数, 不含任何 b13 结果
    for key, val in allowed.items():
        assert "basilisk_b13" not in val, "允许来源里混入了 B1.3 产物: %s=%s" % (key, val)
        assert "results" not in val.lower()
    forbidden = prov_rec["forbidden_sources_not_read"]
    assert isinstance(forbidden, list) and len(forbidden) >= 3
    # 明确登记 B1.3 的 probe 结果未被读取
    joined = " ".join(forbidden)
    assert "basilisk_b13" in joined
    assert prov_rec["ran_before_any_new_probe"] is True


def test_no_b13_outcome_used_in_provenance(prov_rec):
    """§4: 绝对禁止用 B1.3 的结果反推 q0。审计自身必须记录五个 False。"""
    if prov_rec is None:
        pytest.skip("q0_provenance_audit.json 未生成")
    assert prov_rec["used_rul_metric"] is False
    assert prov_rec["used_model_metric"] is False
    assert prov_rec["used_b13_failure_fraction"] is False
    assert prov_rec["used_b13_early_eol_fraction"] is False
    assert prov_rec["provenance_gate"]["formula_contains_b13_outcome"] is False


# --------------------------------------------------------------------------
# §2 五问: q 的物理含义
# --------------------------------------------------------------------------
def test_q1_q_is_derived_dimensionless_margin(prov_rec):
    """q = b/b_fail 是派生无量纲量, u_initial 与 q0 一一线性对应 (非独立自由度)。"""
    if prov_rec is None:
        pytest.skip("q0_provenance_audit.json 未生成")
    q1 = prov_rec["q1_physical_meaning"]
    assert q1["definition"] == "q = b(t) / b_fail"
    assert q1["old_q0_is_derived_not_prior"] is True
    eq = q1["equivalent_utilization"]
    # c = Tc_nom / T_f_fail, 与冻结值一致
    assert abs(eq["c_coulomb_share"] - 0.00873425730550147) < 1e-15
    # 线性映射自洽: u = c + q0*(1-c)
    c = eq["c_coulomb_share"]
    b_fail = q1["b_fail_f2_Nms_per_rad"]
    omega_ref = q1["omega_ref_rad_s"]
    Tc = 0.00125
    T_f_fail = 0.14311462970213382
    assert abs((T_f_fail - Tc) / omega_ref - b_fail) < 1e-18
    assert abs(Tc / T_f_fail - c) < 1e-15


# --------------------------------------------------------------------------
# §2 五问: 旧 q0 的算术来源
# --------------------------------------------------------------------------
def test_q2_old_q0_is_bitwise_quotient_of_documented_b0(prov_rec):
    """旧 q0 区间逐位等于 b0_documented / b_fail_old —— 它不是先验, 是派生。"""
    if prov_rec is None:
        pytest.skip("q0_provenance_audit.json 未生成")
    q2 = prov_rec["q2_old_q0_arithmetic"]
    assert q2["hypothesis_confirmed"] is True
    assert q2["bitwise_identical"] == [True, True]
    assert q2["q0_old_range"] == q2["recomputed_q0"]
    # 独立复算一遍, 不信任 JSON 里的 recomputed 字段
    lo, hi = q2["b0_documented_range_Nms_per_rad"]
    b_fail_old = q2["b_fail_old_Nms_per_rad"]
    assert lo / b_fail_old == q2["q0_old_range"][0]
    assert hi / b_fail_old == q2["q0_old_range"][1]


def test_q2_old_bound_basis_depends_on_rejected_Im_rated(prov_rec):
    """b_fail_old 依赖 B1.2 已否决的 Im_rated => 旧上下界同时失去依据 (§5 例外前提)。"""
    if prov_rec is None:
        pytest.skip("q0_provenance_audit.json 未生成")
    con = prov_rec["q2_old_q0_arithmetic"]["b_fail_old_construction"]
    assert con["depends_on_Im_rated"] is True
    assert con["matches_frozen"] is True
    # 复算 b_fail_old = (Kt*Im - Tc)/omega_ref_design
    recomputed = (con["Kt_nom"] * con["Im_rated_A"] - con["Tc_nom"]) / con["omega_ref_design"]
    assert recomputed == con["recomputed"]
    assert recomputed == prov_rec["q2_old_q0_arithmetic"]["b_fail_old_Nms_per_rad"]


# --------------------------------------------------------------------------
# §2 五问: 唯一可采信的独立依据 = 文档化 b0 绝对量级
# --------------------------------------------------------------------------
def test_q3_documented_b0_is_absolute_and_three_way_consistent(prov_rec):
    """b0 ∈ [1e-6, 1e-5] 是有量纲绝对量, 技术报告 / wheel.yaml / b14 config 三方一致。"""
    if prov_rec is None:
        pytest.skip("q0_provenance_audit.json 未生成")
    q3 = prov_rec["q3_documented_b0"]
    assert q3["consistent"] is True
    assert q3["is_absolute_physical_quantity"] is True
    assert q3["unit"] == "N*m*s/rad"
    assert q3["parsed_range_Nms_per_rad"] == [1e-06, 1e-05]
    assert q3["declared_in_b14_config"] == q3["parsed_range_Nms_per_rad"]
    assert q3["analytic_config_b0_range"] == q3["parsed_range_Nms_per_rad"]
    # 来源文件被指纹固定, 便于事后核对没有被改写
    assert len(q3["source_sha256"]) == 64
    assert len(q3["matched_lines"]) >= 1


# --------------------------------------------------------------------------
# §2 五问: HR16 官方 / healthy-entry 定量限值 —— 两条都不成立
# --------------------------------------------------------------------------
def test_q4_no_independent_healthy_entry_limit_exists(prov_rec):
    """允许来源不存在 healthy-entry/BOL 摩擦裕度定量限值; HI 阶段表被显式拒绝。"""
    if prov_rec is None:
        pytest.skip("q0_provenance_audit.json 未生成")
    q4 = prov_rec["q4_healthy_entry"]
    assert q4["quantitative_healthy_entry_limit_found"] is False
    # HI「健康期 0-0.2」不是同一物理量, 必须给出拒绝理由而不是悄悄拿来用
    assert q4["hi_stage_table_matches"] == []
    reason = q4["why_hi_stage_table_is_not_usable"]
    assert isinstance(reason, str) and len(reason) > 20
    hr16 = prov_rec["hr16_official"]
    assert hr16["queried"] is True
    assert hr16["error"] is None
    # 官方 HR16 三个摩擦系数全为 0 —— 提供不了摩擦依据
    assert hr16["cViscous"] == 0.0
    assert hr16["fCoulomb"] == 0.0
    assert hr16["fStatic"] == 0.0
    assert hr16["provides_viscous_friction"] is False
    assert hr16["provides_coulomb_friction"] is False
    # U_s/U_d 是不平衡力矩, 不能当摩擦用
    assert hr16["U_s"] == 2.8e-06
    assert hr16["U_d"] == 7.7e-07
    # path_A 的三个细分依据里, 只有 documented_absolute_b0 成立
    detail = prov_rec["provenance_gate"]["path_A_detail"]
    assert detail["hr16_official"] is False
    assert detail["healthy_entry_limit"] is False
    assert detail["documented_absolute_b0"] is True


# --------------------------------------------------------------------------
# §21 命名测试: provenance 必须先于 protocol 冻结
# --------------------------------------------------------------------------
def test_q0_provenance_precedes_protocol(prov_rec, prior_rec, protocol_rec, b14_derive):
    """§3→§4→§8 的时序: 审计 → 推导 → 冻结 protocol, 且不得倒挂。"""
    if prov_rec is None or prior_rec is None:
        pytest.skip("前置产物未生成")
    prov_path = _p("checkpoints", "basilisk_b14", "q0_provenance_audit.json")
    prior_path = _p("checkpoints", "basilisk_b14", "healthy_entry_prior.json")
    assert os.path.getmtime(prov_path) <= os.path.getmtime(prior_path)
    if protocol_rec is not None:
        proto_path = _p("checkpoints", "basilisk_b14", "protocol_hash.json")
        assert os.path.getmtime(prior_path) <= os.path.getmtime(proto_path)
    # 依赖链在数据上也必须成立: 推导脚本必须有硬门禁, 且只在 gate pass 时继续。
    # healthy_entry_prior.json 本身不落 provenance_verdict 字段 (照实读, 不臆造),
    # 所以这里检查推导模块的 require_provenance_pass 门禁函数确实存在, 且它读的是
    # q0_provenance_audit.json 的 provenance_gate.pass。
    if b14_derive is not None:
        assert hasattr(b14_derive, "require_provenance_pass")
        src_path = _p("scripts", "basilisk_b14", "derive_healthy_entry_prior.py")
        with io.open(src_path, encoding="utf-8") as fh:
            src = fh.read()
        assert "q0_provenance_audit.json" in src
        assert "provenance_gate" in src
        # 门禁必须在 gate 未过时抛出, 而不是打印警告继续算
        assert 'raise SystemExit' in src
        # 现场调用一次: 当前 gate 已过, 应返回同一份审计记录
        rec = b14_derive.require_provenance_pass(
            {"paths": {"ckpt_dir": "checkpoints/basilisk_b14"}})
        assert rec["provenance_gate"]["verdict"] == "B14_Q0_PROVENANCE_SUFFICIENT"
    assert prior_rec["candidate"] == "HE1_HEALTHY_ENTRY_Q0_PRIOR"
    assert prior_rec["n_candidates"] == 1
    # probe 产物若已存在, 必须晚于 provenance (§2: 审计先于 probe)
    probe_path = _p("checkpoints", "basilisk_b14", "probe_summary.json")
    if os.path.exists(probe_path):
        assert os.path.getmtime(prov_path) < os.path.getmtime(probe_path)
