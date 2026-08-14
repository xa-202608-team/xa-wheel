"""B1.9 §1/§6: 旧产物逐字不变。

含 §10 要求的以下函数名 (字节一致):
  test_b19_old_artifacts_unchanged
"""
from __future__ import annotations

import hashlib

import pytest

from conftest import CK, DOCS, ROOT

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

# B1.9 绝不可改动的文件 (契约会逐个核验; 这里做独立抽样复核)
READONLY = (
    "src/sim/wheel_sim.py",
    "src/sim/build_hi.py",
    "src/sim/damage_model.py",
    "configs/wheel.yaml",
    "configs/wheel_basilisk.yaml",
    "configs/wheel_basilisk_b18.yaml",
    "scripts/basilisk_b1/calibrate_degradation.py",
    "scripts/basilisk_b11/calibrate_degradation.py",
    "scripts/basilisk_b18/build_features.py",
    "data/sim/wheel_basilisk_b18/final/wheel_all.h5",
    "data/features/wheel/basilisk_b18/target_features.h5",
    "STATUS_BASILISK_B18.md",
)


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_b19_old_artifacts_unchanged(b19_baseline):
    """契约里记录的每个只读文件, 现场重算哈希必须逐字相同。"""
    files = b19_baseline["flat_sha256"]
    checked = 0
    drift = []
    _n_hashed = _n_exempt = 0
    for rel, want in files.items():
        p = ROOT / rel
        if want == "MISSING":
            assert not p.exists(), f"契约记为 MISSING 但文件出现了: {rel}"
            continue
        if lifecycle.hash_exempt(rel):
            assert p.exists(), f"{rel} 已豁免哈希但文件消失"
            _n_exempt += 1
            checked += 1   # 计入总账, 只跳过逐字节比对
            continue
        if not p.exists():
            drift.append(f"{rel}: 已消失")
            continue
        got = _sha(p)
        _n_hashed += 1
        if got != want:
            drift.append(f"{rel}: {want[:12]}… -> {got[:12]}…")
        checked += 1
    assert checked > 0, "契约里一个文件都没核到 —— 测试等于空转"
    assert not drift, "只读产物被改动:\n" + "\n".join(drift[:10])

    # 防掏空 (B7 生命周期治理): 豁免只允许命中封闭白名单, 且实检项须占绝大多数。
    lifecycle.assert_registry_is_not_widened()
    assert _n_exempt <= 3, f"哈希豁免项过多 ({_n_exempt}) —— 疑似滥用"
    assert _n_hashed >= 20, f"逐字节实检项过少 ({_n_hashed}) —— 契约可能被掏空"


def test_readonly_files_are_in_contract(b19_baseline):
    """关键只读文件必须真的被契约覆盖, 否则"不变"无从核验。"""
    files = b19_baseline["flat_sha256"]
    missing = [r for r in READONLY if r not in files]
    assert not missing, f"契约未覆盖 {missing}"


def test_b19_wrote_only_its_own_namespace(b19_feature_stats):
    """B1.9 的输出必须落在自己的命名空间内。"""
    out = b19_feature_stats["feature_h5"]
    assert out.startswith("data/features/wheel/basilisk_b19/")
    for forbidden in ("basilisk_b18", "basilisk_b1/", "basilisk_v1",
                      "schema_v1"):
        assert forbidden not in out


def test_b18_feature_file_bit_identical(b19_baseline):
    """§6 明令禁止覆盖旧 feature 文件 —— 单独再核一次。"""
    rel = "data/features/wheel/basilisk_b18/target_features.h5"
    want = b19_baseline["flat_sha256"][rel]
    assert want != "MISSING"
    assert _sha(ROOT / rel) == want


def test_b18_dataset_not_regenerated(b19_baseline, b19_feature_stats):
    """§9: 不得重新生成 lifetime dataset —— 源数据哈希与内容哈希双重核验。"""
    rel = "data/sim/wheel_basilisk_b18/final/wheel_all.h5"
    assert _sha(ROOT / rel) == b19_baseline["flat_sha256"][rel]
    assert b19_feature_stats["source_dataset"] == "data/sim/wheel_basilisk_b18/final"
    assert b19_feature_stats["n_traj"] == 150
    assert b19_feature_stats["n_event_observed"] == 71
    assert b19_feature_stats["n_censored"] == 79


def test_s5b_stage_not_reopened():
    """长期约束: S5B 阶段已关闭, 不得在 B1.9 里重开或重新解释。"""
    for doc in sorted(DOCS.glob("*.md")):
        txt = doc.read_text(encoding="utf-8")
        for bad in ("重开 S5B", "S5B 重启", "reopen S5B"):
            assert bad not in txt, f"{doc.name} 试图重开 S5B"
