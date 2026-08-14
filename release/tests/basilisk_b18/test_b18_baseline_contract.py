"""tests/basilisk_b18/test_b18_baseline_contract.py —— §2 冻结 baseline 契约。

含 §22 要求的 `test_b18_old_artifacts_unchanged`。
"""
from __future__ import annotations

import hashlib
import json
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


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def test_b18_old_artifacts_unchanged(b18_baseline_contract):
    """§2/§22: 契约里记录的每个旧产物文件哈希必须与磁盘现状一致。"""
    c = b18_baseline_contract
    changed = []
    _n_hashed = _n_exempt = 0
    for rel, expected in c["flat_sha256"].items():
        if lifecycle.hash_exempt(rel):
            _n_exempt += 1
            continue
        actual = _sha(ROOT / rel)
        _n_hashed += 1
        if actual != expected:
            changed.append((rel, expected[:12], actual[:12]))
    assert not changed, f"旧产物被改动: {changed[:10]}"

    # 防掏空 (B7 生命周期治理): 豁免只允许命中封闭白名单, 且实检项须占绝大多数。
    lifecycle.assert_registry_is_not_widened()
    assert _n_exempt <= 3, f"哈希豁免项过多 ({_n_exempt}) —— 疑似滥用"
    assert _n_hashed >= 20, f"逐字节实检项过少 ({_n_hashed}) —— 契约可能被掏空"


def test_b18_contract_covers_all_prior_stages(b18_baseline_contract):
    """契约必须覆盖 v1 + b1..b17 的数值结论, 缺一路就是漏冻结。"""
    c = b18_baseline_contract
    for key in ("v1_expected", "b1_expected", "b11_expected", "b12_expected",
                "b13_expected", "b14_expected", "b15_expected",
                "b16_expected", "b17_expected"):
        assert key in c and c[key], f"契约缺 {key}"
    assert c["n_files"] == len(c["flat_sha256"])
    assert c["n_files"] >= 251


def test_b18_frozen_verdicts_intact(b18_baseline_contract):
    """B1..B1.7 的失败结论是终局, 不得被改写或重新解释。"""
    c = b18_baseline_contract
    pins = {
        "b13_expected": ("b13_verdict", "B13_CALIBRATION_FAIL"),
        "b14_expected": ("b14_verdict", "B14_CALIBRATION_FAIL"),
        "b16_expected": ("b16_verdict", "B16_NO_DOCUMENTED_WHEEL"),
        "b17_expected": ("b17_verdict", "B17_THERMAL_PROVENANCE_INSUFFICIENT"),
    }
    for grp, (k, v) in pins.items():
        assert str(c[grp][k]) == v, f"{grp}.{k} 应为 {v}, 实为 {c[grp][k]}"


def test_b18_expected_matches_results(b18_baseline_contract):
    """契约自身的 expected 与 results 两侧必须一致 (verify_baseline 的判据)。"""
    c = b18_baseline_contract
    diffs = []
    for grp in ("v1", "b1", "b11", "b12", "b13", "b14", "b15", "b16", "b17"):
        exp, res = c.get(f"{grp}_expected"), c.get(f"{grp}_results")
        if not exp or not res:
            continue
        for k, v in exp.items():
            if k in res and str(res[k]) != str(v):
                diffs.append((grp, k, str(v)[:20], str(res[k])[:20]))
    assert not diffs, f"契约内部不一致: {diffs[:10]}"


def test_b18_readonly_sources_untouched(b18_baseline_contract):
    """§5/§19 只读依赖: wheel_sim.py / build_hi.py / wheel.yaml / results.md。"""
    c = b18_baseline_contract
    for rel in ("src/sim/wheel_sim.py", "src/sim/build_hi.py",
                "configs/wheel.yaml", "docs/results.md"):
        assert rel in c["flat_sha256"], f"{rel} 未进入契约"
        if lifecycle.hash_exempt(rel):
            # docs/results.md: B7 授权改写为最终交付表达面, 哈希冻结退役。
            # 替代守卫 = 语义冻结 (每个数字必须溯源到冻结的 B5/B6 产物)。
            # 这里断言那份机器审计确实存在且通过, 而不是"删掉了检查"。
            import json as _json
            rec_p = lifecycle.REPORT_NUMBER_AUDIT
            assert rec_p.exists(), \
                f"{rel} 已豁免哈希, 但语义审计记录缺失: {rec_p}"
            rec = _json.loads(rec_p.read_text(encoding="utf-8"))
            assert rec["verdict"] == "REPORT_NUMERIC_AUDIT_PASS", \
                f"{rel} 的语义审计未通过: {rec['verdict']}"
            for k in lifecycle.SEMANTIC_FREEZE_KEYS:
                assert rec["semantic_freeze"][k] is True, \
                    f"{rel} 语义冻结项未通过: {k}"
            continue
        assert _sha(ROOT / rel) == c["flat_sha256"][rel], f"{rel} 被修改"


def test_b18_namespace_isolated():
    """B1.8 的产物一律在 b18 命名空间, 不得污染旧目录。"""
    for rel in ("data/sim/wheel_basilisk_b18", "checkpoints/basilisk_b18",
                "docs/basilisk_b18", "scripts/basilisk_b18",
                "configs/wheel_basilisk_b18.yaml"):
        assert "b18" in rel
    # 旧命名空间下不得出现 b18 字样的新文件
    strays = []
    for old in ("data/sim/wheel_basilisk_v1", "data/sim/wheel_basilisk_b1",
                "checkpoints/basilisk_b1", "checkpoints/basilisk_b17"):
        d = ROOT / old
        if d.exists():
            strays += [p.name for p in d.rglob("*b18*")]
    assert not strays, f"旧命名空间被污染: {strays}"


def test_b18_basilisk_not_in_deploy_manifests():
    """Basilisk 不得写入 requirements.txt / Dockerfile / docker-compose.yml。"""
    for rel in ("requirements.txt", "Dockerfile", "docker-compose.yml"):
        p = ROOT / rel
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="replace").lower()
        assert "basilisk" not in txt, f"{rel} 含 basilisk 依赖 (禁止)"
