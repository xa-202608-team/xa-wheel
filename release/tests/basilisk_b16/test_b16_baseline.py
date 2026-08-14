"""tests/basilisk_b16/test_b16_baseline.py —— §15: baseline 与协议冻结顺序。"""
from __future__ import annotations

import hashlib
import importlib.util
import io
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
DOC = ROOT / "docs" / "basilisk_b16"


def _load_verifier():
    p = ROOT / "scripts" / "basilisk_b16" / "verify_baseline.py"
    spec = importlib.util.spec_from_file_location("b16_verify_for_test", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_b16_baseline_unchanged(b16_contract):
    """逐项重算契约里的文件哈希与数值块, 必须与冻结值完全一致。

    这不是"读一遍契约自证", 而是**重新计算**当前磁盘状态再比对 —— 与
    `verify_baseline.py --tag after` 同一判据, 只是搬进 pytest 里长期看守。
    """
    V = _load_verifier()
    checked = [rel for rel in b16_contract["flat_sha256"]
               if not lifecycle.hash_exempt(rel)]
    _n_exempt = len(b16_contract["flat_sha256"]) - len(checked)
    _n_hashed = len(checked)
    bad = [rel for rel in checked
           if V._sha256_of(rel) != b16_contract["flat_sha256"][rel]]
    assert bad == [], f"既有产物被改动 (B1.6 禁止): {bad[:10]}"

    # 防掏空 (B7 生命周期治理): 豁免只允许命中封闭白名单, 且实检项须占绝大多数。
    lifecycle.assert_registry_is_not_widened()
    assert _n_exempt <= 3, f"哈希豁免项过多 ({_n_exempt}) —— 疑似滥用"
    assert _n_hashed >= 20, f"逐字节实检项过少 ({_n_hashed}) —— 契约可能被掏空"

    for name, reader, expected, prefix in V.NUMERIC_BLOCKS:
        now = reader()
        for k, h in b16_contract[name].items():
            assert now.get(k) == h, f"[{prefix}] {k} 变了: {h} -> {now.get(k)}"
        for k, h in expected.items():
            assert now.get(k) == h, f"[{prefix}_expected] {k} 与脚本内冻结值不符"


def test_b16_contract_has_no_mutable_group():
    """B1.6 是 selection gate —— 契约里不允许出现任何 mutable 组。"""
    V = _load_verifier()
    assert not any("mutable" in g.lower() for g in V.FROZEN_GROUPS), \
        "B1.6 不得声明 mutable 组"


def test_b16_must_stay_absent_items_are_absent(b16_contract):
    """§13: 寿命数据类产物必须缺席, 且契约把这件事钉死为 MISSING。"""
    grp = b16_contract["groups"].get("b16_must_stay_absent")
    assert grp, "契约里没有 b16_must_stay_absent 组"
    for rel, h in grp.items():
        assert h == "MISSING", f"{rel} 不应存在, 契约记为 {h}"
        assert not (ROOT / rel).exists(), f"{rel} 竟然存在 —— 越过了 §13"


def test_protocol_frozen_before_selection(b16_protocol_hash, b16_registry,
                                          b16_scores, b16_verdict):
    """协议必须在候选注册/打分/裁决之前冻结, 且下游全部引用同一 hash。"""
    assert b16_protocol_hash["frozen_before"] == "candidate_registry"
    h = b16_protocol_hash["protocol_sha256"]
    src = io.open(DOC / "protocol.md", "rb").read()
    assert hashlib.sha256(src).hexdigest() == h, \
        "protocol.md 在冻结之后被改动过"
    for art in (b16_registry, b16_scores, b16_verdict):
        assert art["protocol_sha256"] == h


def test_protocol_discloses_post_hoc_rule():
    """§1 的 bol_attainment_check 后加披露必须留在协议里, 不得被删。

    这条规则是在观察到已冻结 profile 极值之后写入的, 属于必须主动交代的
    方法学调整。删掉披露 = 把后加规则伪装成预先规则。
    """
    txt = io.open(DOC / "protocol.md", encoding="utf-8").read()
    assert "bol_attainment_check" in txt
    assert "例外必须披露" in txt
    assert "只可能让候选更难通过" in txt
