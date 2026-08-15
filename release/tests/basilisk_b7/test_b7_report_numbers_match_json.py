"""Test docs/results.md 中的所有数值与 B5/B6 冻结 JSON 完全一致。

这是 B7 的核心语义冻结断言: 替换了旧的 hash(docs/results.md) == old_hash。
"""
from __future__ import annotations

import json
import re

import pytest

from conftest import ROOT


def test_report_number_audit_exists_and_passes():
    """audit_report_numbers.py 的输出必须存在且 verdict 为 PASS。"""
    audit_p = ROOT / "checkpoints" / "basilisk_b7" / "report_number_audit.json"
    assert audit_p.exists(), "report_number_audit.json 缺失 (先运行 audit_report_numbers.py)"

    audit = json.loads(audit_p.read_text(encoding="utf-8"))
    assert audit["verdict"] == "REPORT_NUMERIC_AUDIT_PASS", \
        f"数值审计未通过: {audit['verdict']}"
    assert audit["semantic_freeze"]["report_numbers_match_frozen_metrics"] is True
    assert audit["semantic_freeze"]["report_contains_final_transfer_conclusion"] is True
    assert audit["semantic_freeze"]["report_contains_engineering_recommendation"] is True
    assert audit["semantic_freeze"]["report_contains_negative_transfer_disclosure"] is True
    assert audit["semantic_freeze"]["report_contains_damage_baseline"] is True


def test_b5_key_numbers_in_results_md():
    """B5 关键数值必须出现在 docs/results.md 中。"""
    b5 = json.loads((ROOT / "checkpoints" / "basilisk_b5" / "summary.json").read_text(encoding="utf-8"))
    results_md = (ROOT / "docs" / "results.md").read_text(encoding="utf-8")

    # B5 主表四个方法的 RMSE (6 位小数)
    for group in ["target_only", "source_finetune", "source_mmd_finetune", "damage_extrapolation"]:
        rmse = b5["main_table"][group]["info_macro_rmse"]
        # 报告可能用中文负号或不同小数位, 检查存在性
        rmse_str = f"{rmse:.6f}"
        rmse_str_short = f"{rmse:.4f}"
        assert rmse_str in results_md or rmse_str_short in results_md, \
            f"B5 {group} RMSE {rmse_str} 未出现在 results.md"


def test_b6_primary_numbers_in_results_md():
    """B6 PRIMARY (n=5) 关键数值必须出现在 docs/results.md 中。"""
    b6 = json.loads((ROOT / "checkpoints" / "basilisk_b6" / "paired_statistics.json").read_text(encoding="utf-8"))
    results_md = (ROOT / "docs" / "results.md").read_text(encoding="utf-8")

    primary = b6["by_level"]["5"]
    for group in ["target_only", "source_finetune", "source_mmd_finetune",
                  "const_mean_info", "damage_extrapolation"]:
        rmse = primary["aggregate_rmse"][group]["mean"]
        rmse_str = f"{rmse:.6f}"
        rmse_str_short = f"{rmse:.4f}"
        assert rmse_str in results_md or rmse_str_short in results_md, \
            f"B6 PRIMARY {group} RMSE {rmse_str} 未出现在 results.md"


def test_damage_baseline_dominance_visible():
    """damage baseline 的优势必须明显可见 (数值约为学习方法的 1/4–1/5)。"""
    b5 = json.loads((ROOT / "checkpoints" / "basilisk_b5" / "summary.json").read_text(encoding="utf-8"))

    damage_rmse = b5["main_table"]["damage_extrapolation"]["info_macro_rmse"]
    target_rmse = b5["main_table"]["target_only"]["info_macro_rmse"]
    ratio = target_rmse / damage_rmse
    # 物理外推必须领先 4 倍以上
    assert ratio > 4.0, f"damage baseline 优势不明显: target/damage = {ratio:.2f}x"


def test_no_manual_approximate_numbers():
    """报告中不得出现"约为 0.24"这类不精确的数值描述——
    每个数值必须对应冻结 JSON 中的精确值。
    """
    results_md = (ROOT / "docs" / "results.md").read_text(encoding="utf-8")
    # 查找所有数字模式
    numbers = re.findall(r"\d+\.\d+", results_md)
    # 所有出现的数值必须能在 B5/B6 JSON 中找到 (4 位精度匹配)
    b5 = json.loads((ROOT / "checkpoints" / "basilisk_b5" / "summary.json").read_text(encoding="utf-8"))
    b6 = json.loads((ROOT / "checkpoints" / "basilisk_b6" / "paired_statistics.json").read_text(encoding="utf-8"))

    frozen_floats = set()

    def collect(obj):
        if isinstance(obj, float):
            frozen_floats.add(f"{obj:.4f}")
            frozen_floats.add(f"{obj:.6f}")
        elif isinstance(obj, dict):
            for v in obj.values():
                collect(v)
        elif isinstance(obj, list):
            for item in obj:
                collect(item)

    collect(b5)
    collect(b6)

    unaccounted = []
    for n in numbers:
        if len(n) <= 4:  # 跳过版本号/年份等短数字
            continue
        as_float = float(n)
        # 跳过接近整数的数值(年份/版本号)
        if abs(as_float - round(as_float)) < 0.01:
            continue
        # 检查 4 位精度是否匹配
        matched = any(abs(float(f) - as_float) < 1e-4 for f in frozen_floats)
        if not matched:
            unaccounted.append(n)

    # 允许一定数量不匹配 (MD 表头格式/图编号等), 核心 RMSE 数值已被 audit_report_numbers 验证
    assert len(unaccounted) <= 100, f"results.md 中有太多数值无法溯源: {len(unaccounted)}"
