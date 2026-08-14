"""tests/basilisk_b2/test_b2_baseline_contract.py —— B2 契约与只读边界。

B2 会训练模型, 但**不得**改动 B1.9 及以前的任何产物, 也不得把 Basilisk 依赖
写进部署清单。这两件事由本文件守。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _numeric_items(c: dict) -> int:
    """契约里的数值项 = 除结构性字段外的所有 dict 块条目数。"""
    skip = {"groups", "env", "frozen_chain", "flat_sha256"}
    return sum(len(v) for k, v in c.items()
               if isinstance(v, dict) and k not in skip)


def test_contract_exists_and_nonempty(b2_baseline):
    c = b2_baseline
    assert c["stage"] == "BASILISK_B2", c["stage"]
    assert c["n_files"] == len(c["flat_sha256"])
    assert c["n_files"] > 0
    assert _numeric_items(c) > 0


def test_contract_pins_b19_verdict_and_hashes(b2_baseline):
    """B1.9 的结论与哈希必须在契约里, 且是实测值 (非 placeholder)。"""
    r = b2_baseline["b19_results"]
    e = b2_baseline["b19_expected"]
    assert r["b19_feature_verdict"] == "B19_FEATURE_READY"
    assert r["b19_audit_verdict"] == "B19_FEATURE_READY"
    assert r["b19_observability_verdict"] == "B19_OBSERVABILITY_OK"
    assert r["b19_primary_hi"] == "hi_damage_obs"
    assert r["b19_n_event_observed"] == "71"
    assert r["b19_n_censored"] == "79"
    assert r["b19_n_gates_passed"] == "14"
    for k, v in e.items():
        assert v != "PLACEHOLDER_FILLED_ON_FIRST_WRITE", k
        assert r[k] == v, (k, r[k], v)


def test_contract_pins_feature_h5(b2_baseline, b2_config):
    """§1: 唯一输入的 feature 文件必须在契约里 (逐字不变)。"""
    rel = b2_config["transfer"]["target_feature_path"]
    assert rel in b2_baseline["flat_sha256"]
    h = b2_baseline["flat_sha256"][rel]
    assert h != "MISSING" and len(h) == 64
    actual = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    assert actual == h, "B1.9 feature 文件已被改动 —— B2 的输入不再是冻结数据"


def test_contract_pins_frozen_code_b2_reuses(b2_baseline):
    """B2 import 复用的实现必须钉死: 它们变了, B2 的训练语义就变了。"""
    for rel in ("src/transfer/train_transfer.py",
                "src/transfer/adapter.py",
                "src/experiments/run_groups.py",
                "src/baselines/physical_extrap.py",
                "src/baselines/trivial.py"):
        assert rel in b2_baseline["flat_sha256"], rel
        h = b2_baseline["flat_sha256"][rel]
        assert h != "MISSING", rel
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == h, rel


def test_forbidden_artifacts_stay_absent(b2_baseline):
    """§6/§3 禁止的产物必须缺席 (契约把 MISSING 钉死)。"""
    grp = b2_baseline["groups"]["b2_must_stay_absent"]
    assert len(grp) > 0
    for rel, h in grp.items():
        assert h == "MISSING", f"{rel} 不该存在 (§6 不跑迁移 / §3 不截断)"
        assert not (ROOT / rel).exists(), rel


def test_frozen_chain_records_b19(b2_baseline):
    assert b2_baseline["frozen_chain"]["BASILISK_B1.9"] == "B19_FEATURE_READY"


def test_b19_docs_unchanged(b2_baseline):
    """B1.9 的文档与 checkpoint 逐字未变。"""
    for grp in ("b19_docs", "b19_checkpoints", "b19_scripts", "b19_config"):
        for rel, h in b2_baseline["groups"][grp].items():
            p = ROOT / rel
            if h == "MISSING":
                assert not p.exists(), rel
                continue
            assert p.exists(), rel
            assert hashlib.sha256(p.read_bytes()).hexdigest() == h, rel


# --------------------- 部署清单不得被 Basilisk 污染 ---------------------

@pytest.mark.parametrize("rel", ["requirements.txt", "Dockerfile",
                                 "docker-compose.yml"])
def test_basilisk_absent_from_deploy_files(rel):
    """Basilisk 是本地研究依赖, 不得进入可复现部署清单。"""
    p = ROOT / rel
    if not p.exists():
        pytest.skip(f"缺 {rel}")
    txt = p.read_text(encoding="utf-8", errors="ignore").lower()
    for bad in ("basilisk", "bskexamples", "bsk_"):
        assert bad not in txt, f"{rel} 含 {bad}"
