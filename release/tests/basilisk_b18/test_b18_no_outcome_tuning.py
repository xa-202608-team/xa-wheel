"""tests/basilisk_b18/test_b18_no_outcome_tuning.py —— §8/§13/§14 反结果调参。

含 §22 要求的 `test_primary_nominal_preselected`、`test_no_failure_fraction_selection`、
`test_no_rul_metric_selection`。

这些测试的目标是: 即使有人事后想把 primary 改成 FAST (因为它 event 更多),
或者拿 RUL 指标去挑 L_ref, 也会被 CI 拦住。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "basilisk_b18"

# §21 本阶段禁止存在的训练产物
FORBIDDEN_TRAINING = (
    "checkpoints/basilisk_b18/target_only",
    "checkpoints/basilisk_b18/source_finetune",
    "checkpoints/basilisk_b18/source_mmd",
    "checkpoints/basilisk_b18/rate_model",
    "checkpoints/basilisk_b18/wiener_pf",
    "checkpoints/basilisk_b18/s6",
    "docs/basilisk_b18/rul_results.md",
)


def test_primary_nominal_preselected(b18_config, b18_protocol_hash,
                                     b18_frozen_primary, b18_audit):
    """§8: primary=NOMINAL 在任何 lifetime 数据之前冻结, 四处记录一致。"""
    assert b18_config["primary_scenario"]["name"] == "NOMINAL"
    assert b18_config["primary_scenario"]["selected_before_results"] is True
    assert b18_protocol_hash["primary_scenario_at_freeze"] == "NOMINAL"
    assert b18_frozen_primary["primary_scenario"] == "NOMINAL"
    assert b18_frozen_primary["selected_before_results"] is True
    assert float(b18_frozen_primary["L_ref_years"]) == 3.0
    assert b18_frozen_primary["provenance"] == "NEW_B18_ASSUMPTION"
    assert b18_audit["primary_scenario_frozen_before_results"] == "NOMINAL"
    # 冻结时刻的记录必须早于数据: protocol_hash 记录了 7 个产物当时全部缺失
    assert b18_protocol_hash["frozen_before_all_absent_at_freeze_time"] is True
    assert b18_protocol_hash["primary_selected_before_results"] is True


def test_no_failure_fraction_selection(b18_config, b18_frozen_primary,
                                       b18_audit, b18_sensitivity):
    """§8/§13: 禁止用 failure fraction / event count 作为选择理由或依据。"""
    reason = str(b18_config["primary_scenario"]["reason"])
    banned = ("failure fraction", "failure_fraction", "event fraction",
              "event count", "event_count", "更好", "最好", "最优", "效果最佳")
    low = reason.lower()
    hits = [b for b in banned if b.lower() in low]
    assert not hits, f"选择理由含禁止措辞 {hits}: {reason}"
    assert "horizon" in low or "设计寿命" in reason

    assert b18_frozen_primary["selection_used_failure_fraction"] is False
    assert b18_frozen_primary["selection_used_event_count"] is False
    assert b18_sensitivity["used_failure_fraction_for_calibration"] is False
    assert b18_audit["primary_changed_by_this_audit"] is False
    assert b18_audit["lref_changed_by_this_audit"] is False

    # FAST 的 event 更多, 但 primary 仍是 NOMINAL —— 正是本条要守住的
    sc = b18_audit["scenarios"]
    if sc["FAST"]["event_fraction"] > sc["NOMINAL"]["event_fraction"]:
        assert b18_frozen_primary["primary_scenario"] == "NOMINAL", \
            "FAST 的 event fraction 更高, 但 primary 绝不允许因此改变"


def test_no_rul_metric_selection(b18_frozen_primary, b18_audit,
                                 b18_sensitivity, b18_dataset_audit):
    """§13/§14 Gate 6-7: 场景选择与 EOL 定义均不得读取模型/测试指标。"""
    assert b18_frozen_primary["selection_used_rul_metric"] is False
    assert b18_frozen_primary["selection_used_model_prediction"] is False
    for d in (b18_audit, b18_sensitivity, b18_dataset_audit):
        assert d["used_rul_metric"] is False
        assert d["used_model_metric"] is False


def test_no_training_artifacts_exist():
    """§21: 本阶段禁止训练任何 RUL 模型, 相关产物必须不存在。"""
    present = [p for p in FORBIDDEN_TRAINING if (ROOT / p).exists()]
    assert not present, f"§21 禁止的训练产物出现了: {present}"


def test_scripts_do_not_import_torch():
    """§21 的结构性保证: b18 脚本一律不 import torch / src.models / src.transfer。"""
    offenders = []
    for p in sorted(SCRIPTS.glob("*.py")):
        txt = p.read_text(encoding="utf-8")
        # 去掉字符串与注释里的提及, 只看真正的 import 语句
        for m in re.finditer(r"^\s*(?:import|from)\s+([\w\.]+)", txt, re.M):
            mod = m.group(1)
            if (mod == "torch" or mod.startswith("torch.")
                    or mod.startswith("src.models")
                    or mod.startswith("src.transfer")
                    or mod.startswith("src.experiments")):
                offenders.append(f"{p.name}: {mod}")
    assert not offenders, f"§21 违规 import: {offenders}"


def test_lref_not_adjusted_after_results(b18_dataset_audit, b18_config):
    """§14/§18: 看过结果后禁止回头改 L_ref 或模型。"""
    a = b18_dataset_audit
    assert a["lref_adjusted_after_seeing_results"] is False
    assert a["model_adjusted_after_seeing_results"] is False
    assert a["resample_param_distribution"] is False
    assert float(a["L_ref_years"]) == float(
        b18_config["lifetime_scenarios"]["scenarios"]["NOMINAL"]["L_ref_years"])


def test_audit_script_has_no_selection_branch():
    """审计脚本里不得出现"按统计量改 primary"的代码路径。

    检查方式: 审计脚本不得对 primary 做赋值式改写 —— 只允许从冻结记录读出核对。
    """
    txt = (SCRIPTS / "audit_sensitivity.py").read_text(encoding="utf-8")
    # 逐行、行首锚定 —— 只看可执行赋值语句, 不看字符串字面量里的说明文字
    # (AUDIT_PURPOSE 常量里写着 "primary=NOMINAL 已在……之前冻结", 那是声明
    #  本脚本不选主模型的散文, 不是代码路径)。
    bad = []
    for i, line in enumerate(txt.splitlines(), 1):
        m = re.match(r"\s*primary\s*=\s*(.+)$", line)
        if m is None:
            continue
        rhs = m.group(1)
        # 允许从冻结记录 / 配置读出核对
        if "rec[" in rhs or "cfg[" in rhs or "str(rec" in rhs:
            continue
        bad.append(f"{i}: {line.strip()}")
    assert not bad, f"审计脚本出现可疑的 primary 赋值: {bad}"
    # 不得按 event_fraction 排序取最大者
    for pat in ("argmax", "max(stats", "sorted(stats", "best_scenario"):
        assert pat not in txt, f"审计脚本含选择逻辑 {pat!r}"


def test_freeze_script_refuses_outcome_reason():
    """冻结脚本必须内置对事后合理化措辞的拒绝。"""
    txt = (SCRIPTS / "freeze_primary_scenario.py").read_text(encoding="utf-8")
    assert "FORBIDDEN_REASON_TOKENS" in txt
    assert "B18_OUTCOME_TUNED_REASON" in txt
    for tok in ("failure fraction", "更好"):
        assert tok in txt, f"拒绝清单缺 {tok!r}"
