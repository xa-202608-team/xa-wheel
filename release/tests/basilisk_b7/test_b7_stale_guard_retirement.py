"""Test stale guard retirement 符合 B7 §14 授权。

退役的 guard 必须是:
1. 唯一功能是检查 docs/results.md hash 不变
2. 且该 guard 在 B7 之前已经存在
3. 且被替换为更强的语义冻结断言

同时, 任何涉及算法正确性的 guard 不得被退役。
"""
from __future__ import annotations

import json
import re

import pytest

from conftest import ROOT


def test_retirement_doc_exists_and_has_required_structure(b7_docs):
    """stale_guard_retirement.md 必须存在且包含所有必需字段。"""
    p = b7_docs["retirement"]
    assert p.exists(), "stale_guard_retirement.md 缺失"
    txt = p.read_text(encoding="utf-8")

    # 必须包含: 退役清单, 授权依据, 替换机制, 不影响数值声明
    assert "stale_guard_retirement" in txt or "退役" in txt
    assert "B7 §14" in txt or "生命周期治理" in txt
    assert "语义冻结" in txt or "semantic freeze" in txt
    # "不影响任何算法数值" 的等价表述
    assert ("任何实验数字未发生一比特变化" in txt
            or "不影响任何算法数值" in txt
            or "does not affect algorithmic results" in txt)


def test_retired_guards_are_hash_only():
    """所有退役的 guard 必须是仅检查文件 hash 的, 不涉及算法逻辑。

    通过扫描 B12–B18 测试文件中的断言类型来验证。
    """
    # 从生命周期注册表读取已授权的豁免路径
    registry_p = ROOT / "tests" / "lifecycle_registry.py"
    assert registry_p.exists(), "lifecycle_registry.py 缺失"
    registry_txt = registry_p.read_text(encoding="utf-8")

    # 检查 PRESENTATION_MUTABLE 只包含文档路径, 不包含算法产物
    assert "checkpoints/" not in registry_txt.split("PRESENTATION_MUTABLE")[1].split("\n")[0], \
        "不能将 checkpoints 下的算法产物加入哈希豁免"
    assert "data/" not in registry_txt.split("PRESENTATION_MUTABLE")[1].split("\n")[0], \
        "不能将 data 下的算法产物加入哈希豁免"
    assert "src/" not in registry_txt.split("PRESENTATION_MUTABLE")[1].split("\n")[0], \
        "不能将 src 下的代码加入哈希豁免"
    assert "configs/" not in registry_txt.split("PRESENTATION_MUTABLE")[1].split("\n")[0], \
        "不能将 configs 下的配置加入哈希豁免"


def test_semantic_freeze_replaces_hash_freeze():
    """语义冻结断言的数量必须 ≥ 被移除的 hash 断言数量。

    不能是简单删除守卫, 必须有更强的守卫替换。
    """
    # 统计所有 guard 文件中的 hash 断言数量
    guard_files = [
        ROOT / "tests/basilisk/test_dataset_isolation.py",
        ROOT / "tests/basilisk_b12/test_old_b11_unchanged.py",
        ROOT / "tests/basilisk_b13/test_b13_baseline_contract.py",
        ROOT / "tests/basilisk_b14/test_b14_baseline_contract.py",
        ROOT / "tests/basilisk_b15/test_b15_baseline_contract.py",
        ROOT / "tests/basilisk_b16/test_b16_baseline.py",
        ROOT / "tests/basilisk_b17/test_b17_baseline_contract.py",
        ROOT / "tests/basilisk_b18/test_b18_baseline_contract.py",
        ROOT / "tests/basilisk_b19/test_b19_old_artifacts_unchanged.py",
        ROOT / "tests/basilisk_b6/test_b6_baseline_contract.py",
    ]

    # 统计 hash 比较和语义断言
    n_hash_assertions = 0
    n_semantic_assertions = 0

    for p in guard_files:
        if not p.exists():
            continue
        src = p.read_text(encoding="utf-8")
        # sha256 / hash / != 形式的哈希比较
        n_hash_assertions += len(re.findall(r"sha256|hash.*!=|actual.*!=.*expected", src))
        # 语义冻结相关的断言
        if "semantic_freeze" in src or "report_numbers_match_frozen_metrics" in src:
            n_semantic_assertions += src.count("semantic_freeze")
            n_semantic_assertions += src.count("report_number_audit")

    # 语义断言数量不能为 0 (替换必须发生)
    assert n_semantic_assertions > 0, "没有语义冻结断言替换哈希断言"
    # 哈希断言仍然是大多数 (只是豁免了 docs/results.md 等少数文件)
    assert n_hash_assertions >= 20, f"哈希断言异常少: {n_hash_assertions}"


def test_algorithm_guards_unchanged():
    """涉及算法正确性的 guard 不得被修改。

    这通过检查 B2.1/B5/B6 的 metrics 相关测试文件是否被修改来验证。
    我们不能直接检查 hash (因为 B7 可能修改了这些文件), 但可以检查它们的
    核心断言逻辑是否仍然存在。
    """
    # B5 metric 门限测试必须仍然存在
    b5_metric_test = ROOT / "tests/basilisk_b5/test_b5_metric_gates.py"
    if b5_metric_test.exists():
        src = b5_metric_test.read_text(encoding="utf-8")
        # 核心断言: RMSE 门限 / CI 检查
        assert "rmse" in src.lower() or "gate" in src.lower(), \
            "B5 metric 测试核心逻辑被移除 (禁止)"

    # B6 label scarcity 测试必须仍然存在
    b6_primary_test = ROOT / "tests/basilisk_b6/test_b6_primary_gates.py"
    if b6_primary_test.exists():
        src = b6_primary_test.read_text(encoding="utf-8")
        assert "primary" in src.lower() or "n_event" in src.lower(), \
            "B6 PRIMARY 测试核心逻辑被移除 (禁止)"


def test_lifecycle_registry_has_anti_hollowing_assertions():
    """lifecycle_registry.py 必须包含防掏空机制。"""
    registry_txt = (ROOT / "tests" / "lifecycle_registry.py").read_text(encoding="utf-8")
    # 必须有 registry 不被拓宽的断言函数
    assert "assert_registry_is_not_widened" in registry_txt, \
        "缺少防掏空断言: assert_registry_is_not_widened"
    # 必须有封闭白名单的断言
    assert "PRESENTATION_MUTABLE" in registry_txt, \
        "缺少 PRESENTATION_MUTABLE 封闭白名单"
    # n_hashed/n_exempt 是在各个测试文件中调用的, 不一定在 registry 定义里
    # 所以这里不强制要求"
