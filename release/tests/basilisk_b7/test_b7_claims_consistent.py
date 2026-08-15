"""Test B7 所有文档的语义主张一致且符合 §17 纪律。

- 不得无限定出现正迁移主张
- 必须出现最终结论 (NO_POSITIVE_TRANSFER_SUPPORTED)
- 必须出现工程推荐 (damage_extrapolation)
- 必须披露局限性 (miss rate ~0.6, 不可部署)
"""
from __future__ import annotations

import json
import re

import pytest

from conftest import ROOT


def test_claim_audit_exists_and_passes():
    """claim_audit.json 必须存在且 verdict 为 PASS。"""
    audit_p = ROOT / "checkpoints" / "basilisk_b7" / "claim_audit.json"
    assert audit_p.exists(), "claim_audit.json 缺失 (先运行 audit_claims.py)"

    audit = json.loads(audit_p.read_text(encoding="utf-8"))
    assert audit["verdict"] == "CLAIM_AUDIT_PASS", f"主张审计未通过: {audit['verdict']}"
    assert len(audit["forbidden_unqualified_hits"]) == 0
    assert len(audit["missing_required_claims"]) == 0


def test_final_transfer_conclusion_appears_in_primary_docs():
    """最终结论 "未能证明正向迁移" 必须出现在 results.md 和技术方案报告中。"""
    for doc in ["docs/results.md", "docs/技术方案报告/技术方案报告.md"]:
        txt = (ROOT / doc).read_text(encoding="utf-8")
        assert "未能证明正向迁移" in txt or "no positive transfer was supported" in txt.lower(), \
            f"{doc} 缺少最终迁移结论"


def test_engineering_recommendation_appears_in_primary_docs():
    """工程推荐 damage_extrapolation 必须出现在主要文档中。"""
    for doc in ["docs/results.md", "docs/技术方案报告/技术方案报告.md"]:
        txt = (ROOT / doc).read_text(encoding="utf-8")
        assert "damage_extrapolation" in txt, f"{doc} 缺少工程推荐"


def test_no_unqualified_positive_transfer_claims():
    """整个项目的主要文档中不得出现无限定的正迁移主张。"""
    forbidden_patterns = [
        r"迁移.*显著.*提升",
        r"迁移.*明显.*优于",
        r"MMD.*显著.*有效",
        r"证明.*迁移.*有效",
    ]

    # 免责标记 — 带有这些标记的上下文允许出现禁止短语 (用来引用/批判)
    disclaimer_markers = ["不得", "禁止", "不允许", "不能", "不应", "不是",
                          "historical", "rejected", "invalidated", "已作废", "已推翻"]

    docs = [
        ROOT / "docs/results.md",
        ROOT / "docs/技术方案报告/技术方案报告.md",
        ROOT / "docs/basilisk_b7/final_tables.md",
        ROOT / "docs/basilisk_b7/frozen_result_index.md",
    ]

    hits = []
    for doc in docs:
        if not doc.exists():
            continue
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for pat in forbidden_patterns:
                if re.search(pat, line):
                    # 检查是否带有免责标记
                    has_disclaimer = any(m in line for m in disclaimer_markers)
                    if not has_disclaimer:
                        hits.append((doc.name, lineno, line.strip()[:80]))

    assert not hits, f"发现无限定的正迁移主张:\n" + "\n".join(
        f"  {d}:{l} | {t}" for d, l, t in hits
    )


def test_deployability_disclaimer_present():
    """必须明确声明当前模型不可直接部署 (miss rate 太高)。"""
    results_md = (ROOT / "docs/results.md").read_text(encoding="utf-8")
    assert "不满足可部署" in results_md or "不得直接投入部署" in results_md, \
        "缺少不可部署声明"
    # 必须提到 miss rate 的真实范围
    assert "0.51" in results_md or "0.52" in results_md or "0.61" in results_md, \
        "缺少 miss rate 真实数值披露"


def test_d_baseline_is_simulated_disclaimer_present():
    """必须明确声明 D≥1 是项目定义的仿真失效状态, 不是厂商规格。"""
    results_md = (ROOT / "docs/results.md").read_text(encoding="utf-8")
    tech_md = (ROOT / "docs/技术方案报告/技术方案报告.md").read_text(encoding="utf-8")

    for txt in [results_md, tech_md]:
        assert "simulated failure state" in txt.lower() or "仿真失效状态" in txt, \
            "缺少 D≥1 是仿真失效状态声明"


def test_verdict_consistent_across_documents():
    """所有文档的最终结论必须一致: B5 未通过, B6 PRIMARY 未通过。"""
    docs = [
        ROOT / "docs/results.md",
        ROOT / "docs/技术方案报告/技术方案报告.md",
        ROOT / "docs/basilisk_b7/frozen_result_index.md",
    ]

    for doc in docs:
        if not doc.exists():
            continue
        txt = doc.read_text(encoding="utf-8")
        # 不得出现 "B5 通过" / "B6 PRIMARY 通过" 等错误结论
        assert "B5 通过" not in txt, f"{doc.name} 错误声明 B5 通过"
        assert "B6 通过" not in txt, f"{doc.name} 错误声明 B6 通过"
        # B5/B6 的正确结论: 未通过 / NO_POSITIVE_TRANSFER
        assert "NO_POSITIVE_TRANSFER" in txt or "未能证明正向迁移" in txt, \
            f"{doc.name} 缺少正确结论"
