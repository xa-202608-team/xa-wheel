# -*- coding: utf-8 -*-
"""BASILISK-B1.4 §9/§12/§15: 14-Gate 审计与 verdict 的测试。

核心命名测试:
  - test_gate4_strict_less_than
  - (tail diagnostic 不得成为第 15 个 Gate)
"""
import io
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

GATE_NAMES = {
    1: "failure_fraction",
    2: "censored_fraction",
    3: "n_event_observed",
    4: "early_eol_fraction",
    5: "late_eol_fraction",
    6: "eol_iqr_samples",
    7: "no_nan_inf",
    8: "physical_hard_limits",
    9: "mode_wear_spearman",
    10: "no_rul_or_model_metric",
    11: "degradation_fraction_at_eol_median",
    12: "healthy_window_false_trigger",
    13: "q0_prior_conformance_and_crn_identity",
    14: "normalized_degradation_multiplier_identity",
}


def _p(*parts):
    return os.path.join(ROOT, *parts)


# --------------------------------------------------------------------------
# §12: 逐字复用 B1.3 的 14 条 Gate
# --------------------------------------------------------------------------
def test_exactly_fourteen_gates(probe_audit_rec, protocol_rec):
    """§9/§12: 恰好 14 条 Gate, 编号与名称与 B1.3 一致, 不得增删。"""
    if probe_audit_rec is None:
        pytest.skip("probe_audit.json 未生成")
    gates = probe_audit_rec["gates"]
    assert probe_audit_rec["n_gate"] == 14
    assert len(gates) == 14
    assert [g["gate"] for g in gates] == list(range(1, 15))
    for g in gates:
        assert g["name"] == GATE_NAMES[g["gate"]], (
            "Gate %d 名称变了: %s" % (g["gate"], g["name"]))
    if protocol_rec is not None:
        assert protocol_rec["frozen_values"]["gate_count_expected"] == 14
    assert probe_audit_rec["n_pass"] == sum(1 for g in gates if g["pass"])
    assert probe_audit_rec["all_pass"] == (probe_audit_rec["n_pass"] == 14)


def test_gate4_strict_less_than(probe_audit_rec, b14_cfg):
    """§12/§21: Gate 4 判据必须是严格 `< 0.30`, 不得改为 `<=`, 不得放宽阈值。"""
    if probe_audit_rec is None:
        pytest.skip("probe_audit.json 未生成")
    g4 = [g for g in probe_audit_rec["gates"] if g["gate"] == 4][0]
    assert g4["name"] == "early_eol_fraction"
    rule = g4["rule"]
    assert rule.strip().startswith("<"), rule
    assert "<=" not in rule, "Gate 4 被改成了非严格不等号: %r" % rule
    # 阈值本身不得放宽 (config 段名为 gate, 不是 gates —— 照实读)
    thr = float(b14_cfg["gate"]["early_eol_frac_max"])
    assert thr == 0.30
    assert "0.3" in rule
    # 判定逻辑自洽: value < 0.30 <=> pass (边界等于必须 FAIL)
    frac = float(g4["value"])
    assert g4["pass"] == (frac < thr)


def test_gate1_range_unchanged(probe_audit_rec, b14_cfg):
    """Gate 1 failure_fraction 区间 [0.35, 0.80] 不得放宽。"""
    if probe_audit_rec is None:
        pytest.skip("probe_audit.json 未生成")
    g = b14_cfg["gate"]
    assert float(g["failure_fraction_min"]) == 0.35
    assert float(g["failure_fraction_max"]) == 0.80
    g1 = [x for x in probe_audit_rec["gates"] if x["gate"] == 1][0]
    assert g1["name"] == "failure_fraction"
    assert "0.35" in g1["rule"] and "0.8" in g1["rule"]
    # 判定自洽: 区间内才 PASS
    ff = float(g1["value"])
    assert g1["pass"] == (0.35 <= ff <= 0.80)


def test_gate10_no_rul_or_model_metric(probe_audit_rec):
    """§20: Gate 10 必须 PASS —— 本阶段不得出现任何 RUL/模型指标。"""
    if probe_audit_rec is None:
        pytest.skip("probe_audit.json 未生成")
    g10 = [x for x in probe_audit_rec["gates"] if x["gate"] == 10][0]
    assert g10["pass"] is True
    assert probe_audit_rec["used_rul_metric"] is False
    assert probe_audit_rec["used_model_metric"] is False
    assert probe_audit_rec["used_failure_fraction_for_calibration"] is False


# --------------------------------------------------------------------------
# §9: tail diagnostic 只报告, 不作 Gate
# --------------------------------------------------------------------------
def test_tail_diagnostic_is_not_a_gate(probe_audit_rec, protocol_rec, b14_cfg):
    """§9: tail 只报 median/p90/p95/max, 绝不成为第 15 个 PASS/FAIL Gate。"""
    if probe_audit_rec is None:
        pytest.skip("probe_audit.json 未生成")
    tail = probe_audit_rec["tail_diagnostic"]
    assert tail["is_pass_fail_gate"] is False
    assert b14_cfg["tail_diagnostic"]["add_as_pass_fail_gate"] is False
    assert int(b14_cfg["tail_diagnostic"]["gate_count_expected"]) == 14
    if protocol_rec is not None:
        assert protocol_rec["frozen_values"]["tail_diagnostic_is_gate"] is False
        assert protocol_rec["frozen_values"]["tail_diagnostic_quantiles"] == \
            [0.5, 0.9, 0.95, 1.0]
    # tail 里不得出现 pass 字段 (出现即意味着它在被当作判据)
    assert "pass" not in tail


# --------------------------------------------------------------------------
# §4/§15: probe 结果与预注册预测的核对; FAIL 必须如实上报
# --------------------------------------------------------------------------
def test_observed_matches_pre_registered_prediction(probe_rec, prior_math_rec):
    """结构不可达的推论必须被实测证实 —— 这是反推禁令的正面证据。"""
    if probe_rec is None:
        pytest.skip("probe_summary.json 未生成")
    assert probe_rec["reachability_predicted"] == "STRUCTURALLY_UNREACHABLE"
    assert probe_rec["predicted_failure_fraction"] == 0.0
    obs = probe_rec["observed_failure_fraction"]
    assert probe_rec["prediction_matched"] == (obs == 0.0)
    if prior_math_rec is not None:
        pred = prior_math_rec["pre_registered_prediction"]
        # 预测与实测一致 => Gate 1 必然 FAIL (0.0 < 0.35)
        if obs == 0.0:
            assert pred["gate1_pass"] is False
            assert pred["verdict"] == "B14_CALIBRATION_FAIL"


def test_verdict_consistent_with_gates(verdict_rec, probe_audit_rec):
    """§15: 14/14 才 PASS; 否则 FAIL, 且不得因 FAIL 而改 q0 重跑。"""
    if verdict_rec is None or probe_audit_rec is None:
        pytest.skip("calibration_verdict.json / probe_audit.json 未生成")
    n_pass = probe_audit_rec["n_pass"]
    if n_pass == 14:
        assert verdict_rec["verdict"] == "B14_CALIBRATION_PASS"
    else:
        assert verdict_rec["verdict"] == "B14_CALIBRATION_FAIL"
    assert verdict_rec["n_pass"] == n_pass
    assert verdict_rec["n_gate"] == 14
    # FAIL 时禁止写入冻结标定 (否则等于把不合格标定发布出去)
    if verdict_rec["verdict"] == "B14_CALIBRATION_FAIL":
        assert verdict_rec["frozen_calibration_written"] is False
        assert not os.path.exists(
            _p("checkpoints", "basilisk_b14", "frozen_calibration.json"))
        assert verdict_rec["q0_modified_after_probe"] is False
        assert verdict_rec["probe_reran"] is False


def test_no_dataset_when_calibration_fails(verdict_rec, dataset_rec):
    """§16: 只有 PASS 才允许生成 150 轨迹数据集。"""
    if verdict_rec is None:
        pytest.skip("calibration_verdict.json 未生成")
    if verdict_rec["verdict"] == "B14_CALIBRATION_FAIL":
        assert dataset_rec is None, "FAIL 却生成了 150 轨迹数据集 (§16 违规)"
        h5 = _p("data", "sim", "wheel_basilisk_b14", "wheel_all.h5")
        assert not os.path.exists(h5), "FAIL 却写出了正式数据集 h5"


def test_report_md_written(probe_audit_rec):
    """probe 报告必须落盘, 且含尾部诊断的「为何不是 Gate」说明。"""
    if probe_audit_rec is None:
        pytest.skip("probe_audit.json 未生成")
    md = _p("docs", "basilisk_b14", "probe_report.md")
    assert os.path.exists(md)
    with io.open(md, encoding="utf-8") as fh:
        txt = fh.read()
    assert "14" in txt
    assert "尾部" in txt or "Tail diagnostic" in txt
    # 报告必须写明这不是 Gate, 防止事后被当作判据引用 (照实读报告原话)
    assert "非 Gate" in txt
    assert "PASS/FAIL Gate: **False**" in txt
