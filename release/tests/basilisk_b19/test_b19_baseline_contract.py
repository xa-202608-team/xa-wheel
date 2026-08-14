"""B1.9 §1: baseline 契约测试 —— 前置产物不变。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from conftest import CK, DOCS, ROOT, SCRIPTS, load_script


def _numeric_items(c: dict) -> int:
    """契约里的数值项 = 除结构性字段外的所有 dict 块条目数。

    注意: 契约文件本身只存"被冻结的哈希与数值", 不存 verdict ——
    verdict 由 verify_baseline.py --tag before/after 实跑时给出,
    不能预写进契约, 否则等于自证。
    """
    skip = {"groups", "env", "frozen_chain", "flat_sha256"}
    return sum(len(v) for k, v in c.items()
               if isinstance(v, dict) and k not in skip)


def test_baseline_contract_exists_and_ok(b19_baseline):
    """契约文件存在, 文件项与数值项都不为空, 且 n_files 与实际条目一致。"""
    c = b19_baseline
    assert c["stage"] == "BASILISK_B1.9", c["stage"]
    assert c["n_files"] == len(c["flat_sha256"])
    assert c["n_files"] > 0
    assert _numeric_items(c) > 0
    # B1.8 的双结论必须原样出现在冻结链上
    assert c["frozen_chain"]["BASILISK_B1.8"] == \
        "B18_SCENARIO_READY / B18_FEATURE_NOT_READY"


def test_baseline_contract_pins_b18_frozen_data(b19_baseline):
    """§9 明令不得重新生成 lifetime dataset -> 契约必须钉死 B1.8 的数据与特征。"""
    files = b19_baseline["flat_sha256"]
    for need in ("data/sim/wheel_basilisk_b18/final/wheel_all.h5",
                 "data/sim/wheel_basilisk_b18/final/params.json",
                 "data/features/wheel/basilisk_b18/target_features.h5"):
        assert need in files, f"契约未钉 {need}"
        assert files[need] not in ("MISSING", ""), f"{need} 哈希缺失"


def test_baseline_pins_b18_verdict_and_gate9(b19_baseline):
    """B1.8 的结论必须原样携带 —— 不允许在 B1.9 里被改写成 READY。"""
    exp = b19_baseline["b18_expected"]
    assert exp["b18_feature_verdict"] == "B18_FEATURE_NOT_READY"
    assert exp["b18_failed_gate_9_name"] == "event_observed_hi_p95_above_min"
    assert float(exp["b18_hi_p95_min_threshold"]) == 0.8
    assert float(exp["b18_hi_p95_event_min"]) < 0.8
    assert int(exp["b18_n_event_observed"]) == 71
    assert int(exp["b18_n_censored"]) == 79


def test_b19_l_ref_equals_b18_nominal(b19_config, b19_baseline):
    """§2/§10: L_ref 必须等于 B1.8 NOMINAL, B1.9 不得改。"""
    assert float(b19_config["damage_proxy"]["L_ref_years"]) == 3.0
    assert float(b19_baseline["b18_expected"]["b18_L_ref_years"]) == 3.0


def test_b19_config_forbids_training(b19_config_raw):
    """§12: 本阶段禁止训练, config 必须显式写死。"""
    tf = b19_config_raw["training_forbidden"]
    assert tf["target_only"] is False
    assert tf["transfer"] is False


def test_b19_namespace_is_isolated():
    """§0: B1.9 只写自己的命名空间。"""
    for p in (SCRIPTS, ROOT / "tests" / "basilisk_b19", CK, DOCS,
              ROOT / "data" / "features" / "wheel" / "basilisk_b19"):
        assert p.exists(), f"缺 {p}"
    assert (ROOT / "configs" / "wheel_basilisk_b19.yaml").exists()
    assert (ROOT / "STATUS_BASILISK_B19.md").exists()


def test_b19_not_in_requirements_or_docker():
    """长期约束: Basilisk 不得写入 requirements/Dockerfile/compose。"""
    for rel in ("requirements.txt", "Dockerfile", "docker-compose.yml"):
        p = ROOT / rel
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8").lower()
        assert "basilisk" not in txt, f"{rel} 出现 basilisk"
