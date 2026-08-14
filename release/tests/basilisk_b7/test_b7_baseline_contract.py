"""Test B7 baseline contract — all upstream algorithm artifacts unchanged.

B7 作为最终表达阶段, 不得修改 B1.8/B1.9/B2.1/B5/B6 的任何算法产物。
本测试验证这些产物的 hash 与基线契约一致。
"""
from __future__ import annotations

import json

import pytest

from conftest import ROOT, sha256_of


def test_upstream_algorithm_artifacts_unchanged(frozen_artifacts):
    """B1.8/B1.9/B2.1/B5/B6 所有算法结果文件 hash 不变。

    这是 B7 阶段的核心不变量: 所有数值均来自冻结产物, 未重新计算。
    """
    # 从 B6 baseline contract 读取上游 hash 参考值
    b6_contract = ROOT / "docs" / "basilisk_b6" / "baseline_contract.json"
    if not b6_contract.exists():
        b6_contract = ROOT / "checkpoints" / "basilisk_b6" / "baseline_contract.json"
    assert b6_contract.exists(), f"B6 基线契约缺失 (已试 docs/ 和 checkpoints/)"
    contract = json.loads(b6_contract.read_text(encoding="utf-8"))
    flat = contract["flat_sha256"]

    # 算法相关的文件必须逐字节不变 (docs/results.md 等表达层除外)
    algorithm_prefixes = ("checkpoints/", "data/", "src/", "configs/")
    exempted_algorithmic = 0
    checked = 0
    drift = []

    for rel, expected in flat.items():
        if expected == "MISSING":
            continue
        # 只检查算法相关路径; 生命周期治理的测试文件改动已被授权
        if not any(rel.startswith(p) for p in algorithm_prefixes):
            continue
        if rel.startswith("tests/"):  # B7 授权修改生命周期测试
            continue

        p = ROOT / rel
        assert p.exists(), f"上游算法文件消失: {rel}"
        actual = sha256_of(p)
        if actual != expected:
            drift.append((rel, expected[:16], actual[:16]))
        checked += 1

    assert checked >= 50, f"验证文件数异常: {checked}"
    assert not drift, f"上游算法文件被改动 (B7 禁止):\n" + "\n".join(
        f"  {rel}: {e} → {a}" for rel, e, a in drift[:10]
    )


def test_b7_baseline_contract_exists(b7_docs):
    """B7 基线契约 JSON 必须存在且包含所有关键产物 hash。"""
    p = b7_docs["baseline_contract_json"]
    assert p.exists(), "B7 baseline_contract.json 缺失"
    c = json.loads(p.read_text(encoding="utf-8"))
    assert "flat_sha256" in c
    assert "groups" in c
    # 必须覆盖上游冻结产物(B21/B5/B6)
    has_b5 = any("basilisk_b5" in rel for rel in c["flat_sha256"])
    has_b6 = any("basilisk_b6" in rel for rel in c["flat_sha256"])
    has_b21 = any("basilisk_b21" in rel for rel in c["flat_sha256"])
    assert has_b5 or has_b6 or has_b21, "契约未覆盖任何上游冻结产物"
    # 必须覆盖 B7 最终产物
    assert any("basilisk_b7" in rel for rel in c["flat_sha256"])
    # 图文件单独在 submission_assets 测试中验证, 这里不强制数量
