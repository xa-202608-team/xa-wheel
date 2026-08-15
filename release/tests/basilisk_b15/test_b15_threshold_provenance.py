"""tests/basilisk_b15/test_b15_threshold_provenance.py

Basilisk-B1.5 §4/§5/§7-§10 —— F2 threshold 溯源审计的测试。

本文件的职责是**如实钉死** `B15_FAILURE_SEMANTICS_MISMATCH` 这一结论及其
每一条支撑证据, 而不是把判据放宽到能过。
"""
from __future__ import annotations

import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints" / "basilisk_b15"
AUDIT = CKPT / "failure_threshold_audit.json"

# B1.2 冻结值 —— 本阶段只读, 不得改动
ETA_MARGIN = 0.715573148510669
TORQUE_UTIL_REF_P95 = 0.284426851489331
U_MAX = 0.2
T_F_FAIL = 0.14311462970213382
HR16_FCOULOMB = 0.0005


def _load() -> dict:
    assert AUDIT.exists(), f"缺少 {AUDIT}"
    return json.loads(io.open(AUDIT, encoding="utf-8").read())


def test_verdict_is_semantics_mismatch():
    """§9: 证明 F2 本质是控制裕度阈值 ⇒ verdict 必须是 MISMATCH。"""
    r = _load()
    assert r["verdict"] == "B15_FAILURE_SEMANTICS_MISMATCH"
    assert r["verdict"] != "B15_THRESHOLD_PROVENANCE_SUFFICIENT"


def test_threshold_formula_unchanged():
    """§6: 禁止修改 threshold。逐位核对三个构成量与结果。"""
    r = _load()
    f = r["current_formula"]
    assert f["eta_margin_value"] == ETA_MARGIN
    assert f["torque_util_ref_p95_value"] == TORQUE_UTIL_REF_P95
    assert f["u_max_Nm"] == U_MAX
    assert f["T_f_fail_Nm"] == T_F_FAIL
    # 公式自洽: eta = 1 - tu, thr = eta * umax
    assert 1.0 - TORQUE_UTIL_REF_P95 == ETA_MARGIN
    assert ETA_MARGIN * U_MAX == T_F_FAIL
    assert r["anti_cheat"]["threshold_modified"] is False


def test_every_item_has_a_confidence_label():
    """§4: 每一项必须标注 DIRECT / DERIVED / ASSUMED / UNAVAILABLE 之一。"""
    r = _load()
    allowed = {"DIRECT", "DERIVED", "ASSUMED", "UNAVAILABLE"}
    table = r["provenance_table"]
    assert len(table) >= 6
    for row in table:
        assert row["confidence"] in allowed, \
            f"{row['item']} 的置信标签 {row['confidence']!r} 不在四类之内"
        assert row["source"], f"{row['item']} 没有 source"
        assert "is_failure_mechanism_parameter" in row
    c = r["confidence_counts"]
    assert sum(c.values()) == len(table)
    # 关键: 必须存在 UNAVAILABLE 项, 否则说明审计没找到真正的缺口
    assert c["UNAVAILABLE"] >= 2


def test_eta_margin_labeled_assumed():
    """eta_margin 无独立依据 ⇒ 必须是 ASSUMED, 不得标成 DERIVED 掩饰。"""
    r = _load()
    row = next(x for x in r["provenance_table"] if x["item"] == "eta_margin")
    assert row["confidence"] == "ASSUMED"
    assert row["independent_mechanical_or_control_basis"] is False
    assert row["is_failure_mechanism_parameter"] is False
    assert len(row["searched_for_basis"]) >= 3


def test_allowable_friction_limit_is_unavailable():
    """核心缺口: 许用摩擦限值在本项目可及范围内不存在。"""
    r = _load()
    row = next(x for x in r["provenance_table"]
               if "ALLOWABLE LIMIT" in x["item"])
    assert row["confidence"] == "UNAVAILABLE"
    assert row["value"] is None
    assert row["is_failure_mechanism_parameter"] is True
    assert r["independent_friction_failure_limit_exists"] is False


def test_hr16_cviscous_unavailable_even_with_friction_on():
    """本项目退化变量 b 的同类量, HR16 完全没有定义。"""
    r = _load()
    row = next(x for x in r["provenance_table"] if "cViscous" in x["item"])
    assert row["confidence"] == "UNAVAILABLE"
    assert row["value"] == 0.0
    live = r["hr16_live_read"]
    assert live["useRWfriction_True"]["cViscous"] == 0.0
    assert live["useRWfriction_False"]["cViscous"] == 0.0


def test_hr16_fcoulomb_correction_to_b14():
    """B1.4 记录 fCoulomb = 0.0 是 useRWfriction 默认 False 的假象。

    这是本阶段发现的一处**事实性更正**, 必须钉死: 数值、成因、以及
    "更正后结论不变"这三件事。
    """
    r = _load()
    live = r["hr16_live_read"]
    assert live["useRWfriction_False"]["fCoulomb"] == 0.0
    assert live["useRWfriction_True"]["fCoulomb"] == HR16_FCOULOMB
    # 源码级证据
    se = live["source_evidence"]
    assert se["hr16_definition_declares_fCoulomb"] is True
    assert any("varUseRWfriction" in l for l in se["zero_out_branch"])
    # 更正记录在溯源表里
    row = next(x for x in r["provenance_table"] if "fCoulomb" in x["item"])
    corr = row["correction_to_b14"]
    assert corr["b14_recorded"] == 0.0
    assert corr["actual_when_friction_enabled"] == HR16_FCOULOMB
    # 但它仍不是许用上限 —— 结论不变
    assert row["is_allowable_limit"] is False
    assert row["confidence"] == "DIRECT"


def test_q1_torque_util_is_duty_statistic():
    """§5 Q1: 必须判定为任务工况统计量, 且给出可验证的论据。"""
    r = _load()
    q = r["q5_answers"]["q1_is_torque_util_ref_p95_duty_or_mechanism"]
    assert q["answer"] == "MISSION_DUTY_STATISTIC"
    assert q["is_failure_mechanism_parameter"] is False
    row = next(x for x in r["provenance_table"]
               if x["item"] == "torque_util_ref_p95")
    pm = row["evidence_it_is_duty_not_mechanism"]["per_mode_torque_util"]
    # 论据必须成立: per-mode 跨度足够大, 才能排除"含机理信息"
    assert max(pm.values()) / min(pm.values()) > 50.0


def test_q3_semantics_is_control_authority_not_bearing():
    """§5 Q3 / §9: F2 是控制权限耗尽, 不是轴承机械失效。"""
    r = _load()
    q = r["q5_answers"]["q3_what_does_T_f_fail_represent"]
    assert q["answer"] == "ACTUATOR_CONTROL_AUTHORITY_EXHAUSTION"
    assert q["is_bearing_mechanical_failure"] is False
    qc = q["quantitative_corroboration"]
    # 定量佐证必须自洽可复算
    assert abs(qc["T_f_fail_over_fCoulomb"] - T_F_FAIL / HR16_FCOULOMB) < 1e-9
    assert qc["T_f_fail_over_fCoulomb"] > 100.0
    fs = r["failure_semantics"]
    assert fs["should_serve_as_bearing_degradation_EOL"] is False


def test_probe_not_allowed_and_no_dataset_generated():
    """§8: provenance 不足 ⇒ 不生成新 failure dataset, 不做第二次 probe。"""
    r = _load()
    assert r["probe_allowed"] is False
    assert r["anti_cheat"]["new_failure_dataset_generated"] is False
    # 磁盘上不得出现 b15 的仿真数据
    simdir = ROOT / "data" / "sim" / "wheel_basilisk_b15"
    if simdir.exists():
        assert not any(simdir.rglob("*.h5")), "§8 禁止生成新 failure dataset"


def test_two_routes_planned_not_executed():
    """§10: 两条路线只写计划, 本阶段不执行。"""
    r = _load()
    ro = r["routes"]
    assert ro["executed_in_this_stage"] is False
    assert ro["ROUTE_A"]["name"] == "RIGHT_CENSORED_SURVIVAL"
    assert ro["ROUTE_B"]["name"] == "DOCUMENTED_FAILURE_WHEEL"
    for k in ("ROUTE_A", "ROUTE_B"):
        assert len(ro[k]["plan"]) >= 4
        assert ro[k]["risk"]
        assert ro[k]["premise"]


def test_anti_cheat_clean_with_boundary_matching():
    """§6 反作弊: 无禁止依据; 且检测器必须按完整数字字面量匹配。"""
    r = _load()
    ac = r["anti_cheat"]
    assert ac["clean"] is True
    assert ac["forbidden_as_basis_hits"] == []
    for k in ("used_b13_b14_failure_fraction_as_basis", "used_rul_metric",
              "used_model_metric", "used_target_only_or_transfer",
              "delta_touched", "q0_touched", "horizon_touched",
              "im_rated_touched", "new_safety_factor_invented"):
        assert ac[k] is False, f"{k} 不为 False"
    # 检测器实现: 数字 token 必须有边界断言, 否则 0.3 会误伤
    src = io.open(ROOT / "scripts" / "basilisk_b15"
                  / "audit_failure_threshold.py", encoding="utf-8").read()
    assert r"(?<![\d.])" in src and r"(?![\d])" in src, \
        "数字 token 缺少边界断言, 会把 0.3141… / 0.3600… 误判为 B1.3 的 0.3"


def test_no_training_or_transfer_in_b15():
    """本阶段不得运行任何 RUL / transfer / S6。"""
    sdir = ROOT / "scripts" / "basilisk_b15"
    for p in sdir.glob("*.py"):
        src = io.open(p, encoding="utf-8").read()
        for bad in ("import torch", "from src.train", "from src.transfer",
                    "src.models", "particle_filter", "wiener"):
            assert bad not in src, f"{p.name} 出现训练相关引用: {bad}"
    for p in (ROOT / "checkpoints" / "basilisk_b15").glob("*.json"):
        txt = io.open(p, encoding="utf-8").read().lower()
        for bad in ("rmse", "phm_score", "nphm", "mae_days", "transfer_gain"):
            assert bad not in txt, f"{p.name} 出现 RUL/transfer 指标: {bad}"
    for pat in ("*.pt", "*.pth", "*.ckpt", "*.onnx", "*.safetensors"):
        assert not list((ROOT / "checkpoints" / "basilisk_b15").glob(pat))
