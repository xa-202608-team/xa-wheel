"""tests/basilisk_b18/test_b18_documentation_honesty.py —— §24 文档诚实性。

B1.8 的核心授权条件是"不再把项目假设包装成厂家规格"。这一条如果只写在文档里,
下一次改写就会悄悄丢掉; 所以在这里用测试钉死。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "basilisk_b18"

# §24 必须出现的两条声明 (按语义匹配, 允许措辞微调但关键要素必须齐)
REQUIRED_IN_LIMITATIONS = (
    # 1. D=1 不是厂家阈值
    (r"D\s*=\s*1", r"(项目定义|项目自建|engineering.?assumption)",
     r"(不是|非).*(Honeywell|BCT|NanoAvionics|Basilisk|厂家|厂商)"),
    # 2. Basilisk 只负责工况, 失效定义由项目自建
    (r"Basilisk", r"(任务|姿态控制|工况)", r"(项目自建|项目定义|由项目)"),
)

# 全部 b18 文档中禁止出现的措辞
FORBIDDEN_PHRASES = (
    "manufacturer failure specification",
    "厂家失效阈值",
    "厂商失效阈值",
    "厂家失效规格",
    "vendor failure threshold",
    "hardware failure limit",
    "厂家给出的失效",
)


# 判定"这一行是在否认而非主张"的标记。
# 否认句是允许的 —— B1.8 的全部要点就是反复声明"这不是厂家规格"。
NEGATION_MARKERS = (
    "不是", "不能", "不得", "不可", "不再", "不存在", "非", "禁止", "禁",
    "无外部来源", "must_not", "forbidden", "not ", "never", "no ", "false",
    "❌",
    "~~",     # markdown 删除线 —— 禁用词清单本身就是否认形式
)


def _docs():
    if not DOCS.exists():
        pytest.skip("docs/basilisk_b18 不存在")
    return sorted(DOCS.glob("*.md"))


def test_limitations_exists_and_states_assumption():
    """§24: limitations.md 必须明确声明 D=1 是项目定义的仿真失效状态。"""
    p = DOCS / "limitations.md"
    if not p.exists():
        pytest.skip("limitations.md 未生成")
    txt = p.read_text(encoding="utf-8")
    for group in REQUIRED_IN_LIMITATIONS:
        for pat in group:
            assert re.search(pat, txt, re.S), \
                f"limitations.md 缺必需声明要素: {pat!r}"
    # 三家厂商与 Basilisk 都要点名否认
    for vendor in ("Honeywell", "BCT", "NanoAvionics", "Basilisk"):
        assert vendor in txt, f"limitations.md 未点名 {vendor}"


def test_no_manufacturer_claim_in_any_doc():
    """§24: 任何 b18 文档都不得把项目假设**主张**为厂家规格。

    否认句是允许且必需的 —— B1.8 的要点就是反复声明"这不是厂家规格"。因此只有
    在**缺少任何否认标记**时才判为违规。为避免跨行否认被漏判 (例如上一行写
    `must_not_be_called`), 判定窗口取该行及其前后各一行。
    """
    offenders = []
    for p in _docs():
        lines = p.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            low = line.lower()
            for bad in FORBIDDEN_PHRASES:
                if bad.lower() not in low:
                    continue
                window = "\n".join(lines[max(0, i - 1):i + 2])
                wlow = window.lower()
                if not any((mk in window) or (mk in wlow)
                           for mk in NEGATION_MARKERS):
                    offenders.append(f"{p.name}:{i + 1}: {line.strip()[:80]}")
    assert not offenders, "文档出现未否认的厂家规格措辞:\n" + "\n".join(offenders)


def test_docs_call_it_simulation_scenario():
    """§24: 应称 simulation degradation scenario / 工程假设失效定义。"""
    p = DOCS / "limitations.md"
    if not p.exists():
        pytest.skip("limitations.md 未生成")
    txt = p.read_text(encoding="utf-8")
    ok = ("simulation degradation scenario" in txt
          or "simulation_degradation_scenario" in txt
          or "仿真退化场景" in txt)
    assert ok, "limitations.md 未使用 simulation degradation scenario 表述"
    assert ("engineering-assumption failure definition" in txt
            or "engineering_assumption_failure_definition" in txt
            or "工程假设失效定义" in txt)


def test_primary_scenario_doc_forbids_outcome_reason():
    """§8: primary_scenario.md 必须显式列出"不是选择依据"的三条。"""
    p = DOCS / "primary_scenario.md"
    if not p.exists():
        pytest.skip("primary_scenario.md 未生成")
    txt = p.read_text(encoding="utf-8")
    assert "failure fraction" in txt.lower()
    assert "selected_before_results" in txt or "结果之前" in txt
    assert "NOMINAL" in txt
    # 必须出现否定式表述
    assert any(t in txt for t in ("❌", "不因为", "不是选择依据", "禁止"))


def test_sensitivity_report_declares_report_only():
    """§13: sensitivity_report.md 必须声明"只报告, 不选主模型"。"""
    p = DOCS / "sensitivity_report.md"
    if not p.exists():
        pytest.skip("sensitivity_report.md 未生成")
    txt = p.read_text(encoding="utf-8")
    assert "只报告" in txt or "不得依据" in txt
    assert "NOMINAL" in txt
    # 空 bin 口径必须写明
    assert "NaN" in txt


def test_docs_disclose_censoring_handling():
    """右删失处理必须在文档中写明, 不得默默丢掉或伪造。"""
    found = False
    for p in _docs():
        txt = p.read_text(encoding="utf-8")
        if ("删失" in txt and ("不伪造" in txt or "不混入" in txt)):
            found = True
            break
    assert found, "没有任何 b18 文档说明右删失的处理方式"


def test_no_rul_results_claimed():
    """§21: 本阶段不训练 RUL, 文档不得出现 RUL 精度类结论。"""
    offenders = []
    for p in _docs():
        txt = p.read_text(encoding="utf-8")
        for pat in (r"RMSE\s*=\s*[\d.]", r"nPHM\s*=\s*[\d.]",
                    r"MAE\s*=\s*[\d.]"):
            for m in re.finditer(pat, txt):
                offenders.append(f"{p.name}: {m.group(0)}")
    assert not offenders, f"§21 阶段不应有 RUL 指标结论: {offenders}"


def test_status_file_records_verdict():
    """STATUS_BASILISK_B18.md 必须记录最终 verdict, 便于接续。"""
    p = ROOT / "STATUS_BASILISK_B18.md"
    if not p.exists():
        pytest.skip("STATUS_BASILISK_B18.md 未生成")
    txt = p.read_text(encoding="utf-8")
    assert "B18" in txt
    assert any(v in txt for v in ("B18_FEATURE_READY", "B18_FEATURE_NOT_READY",
                                  "B18_SCENARIO_NOT_INFORMATIVE",
                                  "B18_DAMAGE_MODEL_INVALID"))
    # 前序失败结论不得被改写成成功
    for v in ("B1_CALIBRATION_FAIL", "B17_THERMAL_PROVENANCE_INSUFFICIENT"):
        assert v in txt, f"STATUS 未保留前序结论 {v}"
