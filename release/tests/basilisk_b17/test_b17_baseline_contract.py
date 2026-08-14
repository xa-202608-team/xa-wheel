"""tests/basilisk_b17/test_b17_baseline_contract.py

§1 / §17 —— baseline 契约与既有产物不变性。

`test_b17_old_artifacts_unchanged` 不是"把契约读回来跟自己比" —— 那样永远通过。
它动态载入 `verify_baseline.py` 并**重新计算**每个哈希与数值项。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# 共享生命周期治理注册表 —— 全库唯一来源。
# B7 授权改写 docs/results.md 作为最终交付表达面, 因此该文件的"哈希永久不变"
# 约束退役, 改由语义审计 (scripts/basilisk_b7/audit_report_numbers.py) 看守。
# 算法产物 (数据/特征/split/权重/metrics/verdict) 仍然逐字节冻结, 不受影响。
# 退役理由逐条记录: docs/basilisk_b7/stale_guard_retirement.md
# --------------------------------------------------------------------------
import importlib.util as _ilu   # noqa: E402

_lr_spec = _ilu.spec_from_file_location(
    "lifecycle_registry", ROOT / "tests" / "lifecycle_registry.py")
lifecycle = _ilu.module_from_spec(_lr_spec)
_lr_spec.loader.exec_module(lifecycle)


def _load_verifier():
    p = ROOT / "scripts/basilisk_b17/verify_baseline.py"
    spec = importlib.util.spec_from_file_location("b17_verify_under_test", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["b17_verify_under_test"] = m
    spec.loader.exec_module(m)
    return m


def test_b17_old_artifacts_unchanged(b17_contract):
    """§17 —— 既有 basilisk_v1/b1..b16 及 analytic 产物一字未改。"""
    V = _load_verifier()
    checked = [rel for rel in b17_contract["flat_sha256"]
               if not lifecycle.hash_exempt(rel)]
    _n_exempt = len(b17_contract["flat_sha256"]) - len(checked)
    _n_hashed = len(checked)
    bad = [rel for rel in checked
           if V._sha256_of(rel) != b17_contract["flat_sha256"][rel]]
    assert bad == [], f"既有产物被改动 (B1.7 禁止): {bad[:10]}"

    # 防掏空 (B7 生命周期治理): 豁免只允许命中封闭白名单, 且实检项须占绝大多数。
    lifecycle.assert_registry_is_not_widened()
    assert _n_exempt <= 3, f"哈希豁免项过多 ({_n_exempt}) —— 疑似滥用"
    assert _n_hashed >= 20, f"逐字节实检项过少 ({_n_hashed}) —— 契约可能被掏空"


def test_b17_numeric_blocks_unchanged(b17_contract):
    """数值结论 (b1..b16 的判决与关键数字) 全部未漂移。"""
    V = _load_verifier()
    bad = []
    for name, reader, expected, prefix in V.NUMERIC_BLOCKS:
        now = reader()
        for k, ref in b17_contract[name].items():
            if now.get(k) != ref:
                bad.append((f"{prefix}:{k}", ref, now.get(k)))
        for k, exp in expected.items():
            if now.get(k) != exp:
                bad.append((f"{prefix}_expected:{k}", exp, now.get(k)))
    assert bad == [], f"数值结论漂移: {bad[:8]}"


def test_b17_frozen_chain_includes_b16(b17_contract):
    """B1.6 的 NO_DOCUMENTED_WHEEL 必须在冻结链里 —— 它是 B1.7 的前提。"""
    fc = b17_contract["frozen_chain"]
    assert fc["BASILISK_B1.6"] == "B16_NO_DOCUMENTED_WHEEL"
    assert fc["BASILISK_B1.5"] == "B15_FAILURE_SEMANTICS_MISMATCH"
    assert fc["S5B"] == "S5B_BASELINE_WEAK"


def test_b17_readonly_files_pinned(b17_contract, b17_config):
    """§0 只读文件必须在契约里被钉住 (否则"不得修改"是空话)。"""
    flat = b17_contract["flat_sha256"]
    for rel in ("src/sim/wheel_sim.py", "configs/wheel.yaml",
                "docs/results.md"):
        assert rel in flat, f"只读文件未被契约覆盖: {rel}"
        assert flat[rel] != "MISSING", f"只读文件缺失: {rel}"


def test_b17_lifetime_artifacts_absent(b17_contract, b17_config):
    """§13 / §20 —— 寿命 / RUL / aging 产物必须缺席, 且契约钉死其缺席。"""
    flat = b17_contract["flat_sha256"]
    for rel in b17_config["baseline"]["must_stay_absent"]:
        assert (ROOT / rel).exists() is False, f"越界产物出现: {rel}"
        assert flat.get(rel) == "MISSING", (
            f"契约未把 {rel} 的缺席钉死 (值={flat.get(rel)})")


def test_b17_basilisk_not_in_delivery_deps(b17_config):
    """standing constraint —— Basilisk 不得写入交付依赖链。"""
    for rel in b17_config["namespace"]["must_not_contain_basilisk"]:
        p = ROOT / rel
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="replace").lower()
        assert "basilisk" not in txt, f"{rel} 含 basilisk 依赖 (禁止)"


def test_b17_namespace_only_writes(b17_config):
    """B1.7 只在自己的命名空间下写产物。"""
    ns = b17_config["namespace"]
    for key in ("scripts_dir", "tests_dir", "checkpoints_dir", "docs_dir"):
        d = ROOT / ns[key]
        assert d.exists(), f"命名空间缺失: {ns[key]}"
    # 不得在 b16 及更早目录里新增文件 —— 由契约的 flat_sha256 覆盖间接保证,
    # 这里额外确认没有 b17 命名的文件散落在 b16 目录
    for d in ("scripts/basilisk_b16", "docs/basilisk_b16",
              "checkpoints/basilisk_b16", "tests/basilisk_b16"):
        p = ROOT / d
        if p.exists():
            stray = [f.name for f in p.rglob("*b17*")]
            assert stray == [], f"{d} 下出现 b17 文件: {stray}"
