"""B1.9 §8: Gate 判定与 verdict 一致性 + 反调参。

含 §10 要求的以下函数名 (字节一致):
  test_feature_gate_verdict_consistent
"""
from __future__ import annotations

import re

import pytest

from conftest import CK, DOCS, ROOT, SCRIPTS

GATE_NAMES = [
    "hi_damage_obs_finite",
    "hi_in_unit_interval",
    "chronological_only",
    "no_hidden_damage_input",
    "no_b_true_input",
    "no_eol_or_rul_normalization",
    "event_eol_median_hi_above_min",
    "event_eol_p10_hi_above_min",
    "censored_not_forced_to_one",
    "implementation_audit_corr_above_min",
    "same_prefix_invariance",
    "feature_content_hash_reproducible",
    "xt_truth_leakage_zero",
    "mission_features_not_in_xt",
]


def test_feature_gate_verdict_consistent(b19_feature_stats, b19_feature_audit):
    """verdict 必须与 14 条 Gate 的实际结果一致, 两侧结论必须相同。

    本测试不预设 READY —— 它要求"结论不可与证据脱节": 不允许有 FAIL 却报 READY,
    不允许悄悄删掉某条 Gate, 不允许构建期与独立复核给出不同结论。
    """
    for fs in (b19_feature_stats, b19_feature_audit):
        assert fs["n_gates"] == 14
        assert sorted(g["no"] for g in fs["gates"]) == list(range(1, 15))
        assert [g["name"] for g in sorted(fs["gates"], key=lambda x: x["no"])] \
            == GATE_NAMES
        failed = [g["no"] for g in fs["gates"] if not g["passed"]]
        assert fs["n_passed"] == 14 - len(failed)
        expected = "B19_FEATURE_READY" if not failed else "B19_FEATURE_NOT_READY"
        if fs is b19_feature_audit and not fs["cross_check_consistent"]:
            expected = "B19_AUDIT_MISMATCH"
        assert fs["verdict"] == expected, \
            f"verdict {fs['verdict']} 与 FAIL 名单 {failed} 不一致"

    assert b19_feature_audit["cross_check_consistent"] is True
    assert b19_feature_audit["disagreeing_gates"] == []
    assert b19_feature_stats["verdict"] == b19_feature_audit["verdict"]


def test_status_file_matches_verdict(b19_feature_audit):
    """STATUS 文件里的判定必须与产物一致 —— 不允许文档写得比证据好。"""
    txt = (ROOT / "STATUS_BASILISK_B19.md").read_text(encoding="utf-8")
    assert b19_feature_audit["verdict"] in txt
    other = ("B19_FEATURE_NOT_READY"
             if b19_feature_audit["verdict"] == "B19_FEATURE_READY"
             else "B19_FEATURE_READY")
    # 反向词若出现, 必须是在否定/对照语境里 (同行含否定标记)
    for i, line in enumerate(txt.splitlines(), 1):
        if other in line:
            assert any(m in line for m in ("不是", "而非", "否则", "若", "非",
                                           "对照", "~~", "FAIL 名单")), \
                f"STATUS_BASILISK_B19.md:{i} 出现无语境的 {other}"


def test_frozen_definition_only_when_ready(b19_frozen, b19_feature_audit):
    """冻结件只在双侧 READY 且一致时才应存在。"""
    assert b19_frozen["verdict"] == "B19_FEATURE_READY"
    assert b19_frozen["gate_results"]["build_verdict"] == "B19_FEATURE_READY"
    assert b19_frozen["gate_results"]["audit_verdict"] == "B19_FEATURE_READY"
    assert b19_frozen["gate_results"]["cross_check_consistent"] is True
    assert b19_frozen["gate_results"]["n_passed"] == 14


def test_thresholds_not_tuned(b19_config, b19_config_raw, b19_frozen):
    """§8: 禁止调阈值降门槛 —— 三处 (config / 继承解析 / 冻结件) 必须一致且为原值。"""
    want = {"event_eol_median_hi_min": 0.90, "event_eol_p10_hi_min": 0.75,
            "audit_corr_min": 0.95, "prefix_invariance_atol": 0.0}
    for k, v in want.items():
        assert float(b19_config["feature_gate"][k]) == v, k
        assert float(b19_config_raw["feature_gate"][k]) == v, f"raw {k}"
        assert float(b19_frozen["gate_results"]["thresholds"][k]) == v, f"frozen {k}"


def test_no_outcome_selection_in_scripts():
    """脚本不得含"按结果挑主 HI"的分支 —— HI 在 protocol 单点冻结。"""
    for name in ("derive_damage_proxy.py", "build_features.py",
                 "audit_features.py", "freeze_feature_definition.py"):
        txt = (SCRIPTS / name).read_text(encoding="utf-8")
        for pat in ("argmax", "best_hi", "sorted(candidates", "max(candidates"):
            assert pat not in txt, f"{name} 含选择逻辑 {pat!r}"
        # 逐行、行首锚定: 只看可执行赋值, 不看常量字符串里的说明散文
        for i, line in enumerate(txt.splitlines(), 1):
            m = re.match(r"\s*(primary_hi|selected_hi)\s*=\s*(.+)$", line)
            if m is None:
                continue
            rhs = m.group(2)
            assert ("cfg[" in rhs or "f.attrs[" in rhs or "build[" in rhs
                    or "str(" in rhs), f"{name}:{i} 可疑赋值 {line.strip()}"


def test_no_training_artifacts_produced(b19_feature_stats, b19_frozen):
    """§12: 本阶段不训练 RUL —— 不得出现训练产物。"""
    assert b19_feature_stats["trained_any_model"] is False
    assert b19_frozen["trained_any_model"] is False
    for rel in ("checkpoints/basilisk_b19/target_only",
                "checkpoints/basilisk_b19/source_finetune",
                "checkpoints/basilisk_b19/metrics.json",
                "checkpoints/basilisk_b19/results.json"):
        assert not (ROOT / rel).exists(), f"出现训练产物 {rel}"


def test_gate10_marked_as_audit_only(b19_feature_stats):
    """Gate 10 的说明必须显式声明它只是实现审计, 不是能力证据。"""
    g10 = next(g for g in b19_feature_stats["gates"] if g["no"] == 10)
    assert g10["name"] == "implementation_audit_corr_above_min"
    d = g10["detail"]
    assert "IMPLEMENTATION_AUDIT" in d
    assert "recomputation" in d
