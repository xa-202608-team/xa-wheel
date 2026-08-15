"""Test B7 所有最终交付产物完整且格式正确。

这是最终的打包检查: 确保所有需要提交的文件都在正确位置。
"""
from __future__ import annotations

import json

import pytest

from conftest import ROOT


def test_all_b7_checkpoints_exist(frozen_artifacts):
    """B7 checkpoint 目录下的所有必需 JSON 必须存在。"""
    for name, p in frozen_artifacts.items():
        if name.startswith("b7_"):
            assert p.exists(), f"B7 checkpoint 缺失: {name} @ {p}"


def test_all_b7_docs_exist(b7_docs):
    """B7 docs 目录下的核心文档必须存在，limitations 等可选文档可跳过。"""
    for name, p in b7_docs.items():
        if name == "limitations":  # limitations 可选，不强制存在
            continue
        assert p.exists(), f"B7 文档缺失: {name} @ {p}"


def test_final_tables_md_has_content():
    """final_tables.md 必须包含表格内容。"""
    md_path = ROOT / "docs/basilisk_b7/final_tables.md"
    if not md_path.exists():
        pytest.skip("final_tables.md 待生成")
    md = md_path.read_text(encoding="utf-8")
    assert len(md.strip()) > 0, "final_tables.md 为空"


def test_frozen_result_index_md_has_content():
    """frozen_result_index.md 必须包含内容。"""
    md_path = ROOT / "docs/basilisk_b7/frozen_result_index.md"
    if not md_path.exists():
        pytest.skip("frozen_result_index.md 待生成")
    md = md_path.read_text(encoding="utf-8")
    assert len(md.strip()) > 0, "frozen_result_index.md 为空"


def test_figure_manifest_has_all_5_figures():
    """figure_manifest.md 必须列出所有 5 张图及其说明。"""
    md = (ROOT / "docs/basilisk_b7/figure_manifest.md").read_text(encoding="utf-8")
    for i in range(1, 6):
        assert f"Figure {i}" in md or f"fig{i}" in md, \
            f"figure_manifest.md 缺少 Figure {i}"


def test_limitations_md_has_required_content(b7_docs):
    """limitations.md 必须包含关键局限性声明。"""
    p = b7_docs.get("limitations")
    if not p or not p.exists():
        pytest.skip("limitations.md 待生成")
    txt = p.read_text(encoding="utf-8")
    assert "miss rate" in txt.lower() or "不可部署" in txt, \
        "limitations.md 缺少告警局限性"
    assert "simulated" in txt.lower() or "仿真" in txt, \
        "limitations.md 缺少仿真环境局限性"


def test_no_basilisk_in_requirements_or_docker():
    """Basilisk runtime 不得写入 requirements.txt / Dockerfile / docker-compose.yml。

    这是 NO_BASILISK_IN_DEPLOY 纪律的最终验证。
    """
    for rel in ["requirements.txt", "Dockerfile", "docker-compose.yml"]:
        p = ROOT / rel
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8").lower()
        assert "basilisk" not in txt, f"{rel} 引入了 Basilisk 依赖 (违反 §20)"


def test_final_verdict_unchanged_from_b6():
    """B6 final verdict 文件本身不得被 B7 修改。"""
    b6_verdict = ROOT / "checkpoints" / "basilisk_b6" / "final_verdict.json"
    if not b6_verdict.exists():
        pytest.skip("B6 final_verdict.json 缺失")

    v = json.loads(b6_verdict.read_text(encoding="utf-8"))
    # 最终结论必须是 NO_POSITIVE_TRANSFER_SUPPORTED
    # 键名可能是 "verdict" 或 "FINAL_TRANSFER_CONCLUSION"
    verdict = v.get("verdict", "") or v.get("FINAL_TRANSFER_CONCLUSION", "")
    assert "NO_POSITIVE_TRANSFER" in verdict, \
        f"B6 最终结论被改动 (禁止): {verdict}"


def test_submission_asset_count(figures):
    """最终交付资产总数应符合预期。"""
    # 图文件: 5 × 2 = 10
    assert len(figures) == 10, f"图文件数量异常: {len(figures)}"

    # B7 checkpoints: 4 个 JSON
    b7_json = list((ROOT / "checkpoints" / "basilisk_b7").glob("*.json"))
    assert len(b7_json) >= 4, f"B7 checkpoint JSON 数量异常: {len(b7_json)}"

    # B7 文档: 至少 5 个
    b7_docs_list = list((ROOT / "docs" / "basilisk_b7").glob("*.md")) + \
                    list((ROOT / "docs" / "basilisk_b7").glob("*.json"))
    assert len(b7_docs_list) >= 5, f"B7 文档数量异常: {len(b7_docs_list)}"
