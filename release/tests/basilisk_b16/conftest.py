"""tests/basilisk_b16/conftest.py —— B1.6 测试共享夹具。

`tests/` 无 `__init__.py`, 模块 basename 必须在八路命名空间
(basilisk/b1/b11/b12/b13/b14/b15/b16) 内唯一 —— 本目录全部文件带 `b16_` 前缀。
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CK = ROOT / "checkpoints" / "basilisk_b16"
DOC = ROOT / "docs" / "basilisk_b16"
SCR = ROOT / "scripts" / "basilisk_b16"


def _j(p: Path):
    if not p.exists():
        pytest.skip(f"缺少 {p.relative_to(ROOT)} —— 先跑 §16 流程")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def b16_cfg():
    return yaml.safe_load(
        io.open(ROOT / "configs" / "wheel_basilisk_b16.yaml",
                encoding="utf-8").read())


@pytest.fixture(scope="session")
def b16_contract():
    return _j(DOC / "baseline_contract.json")


@pytest.fixture(scope="session")
def b16_inventory():
    return _j(CK / "wheel_inventory.json")


@pytest.fixture(scope="session")
def b16_fields():
    return _j(CK / "field_audit.json")


@pytest.fixture(scope="session")
def b16_sources():
    return _j(CK / "datasource_audit.json")


@pytest.fixture(scope="session")
def b16_registry():
    return _j(CK / "candidate_registry.json")


@pytest.fixture(scope="session")
def b16_scores():
    return _j(CK / "provenance_scores.json")


@pytest.fixture(scope="session")
def b16_verdict():
    return _j(CK / "b16_verdict.json")


@pytest.fixture(scope="session")
def b16_protocol_hash():
    return _j(CK / "protocol_hash.json")
