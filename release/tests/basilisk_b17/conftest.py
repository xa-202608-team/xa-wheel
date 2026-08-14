"""tests/basilisk_b17/conftest.py —— B1.7 测试夹具。

命名纪律: `tests/` 无 `__init__.py`, 模块 basename 在九路命名空间
(basilisk / b1 / b11 / b12 / b13 / b14 / b15 / b16 / b17) 中必须唯一,
故本目录所有文件一律 `b17_` 前缀。
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CK = ROOT / "checkpoints" / "basilisk_b17"
DOCS = ROOT / "docs" / "basilisk_b17"


def _load(p: Path):
    if not p.exists():
        pytest.skip(f"产物不存在 (需先跑 §18 步骤): {p.relative_to(ROOT)}")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def b17_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def b17_config() -> dict:
    return yaml.safe_load(
        io.open(ROOT / "configs/wheel_basilisk_b17.yaml", encoding="utf-8"))


@pytest.fixture(scope="session")
def b17_protocol_hash() -> dict:
    return _load(CK / "protocol_hash.json")


@pytest.fixture(scope="session")
def b17_temperature() -> dict:
    return _load(CK / "temperature_provenance.json")


@pytest.fixture(scope="session")
def b17_registry() -> dict:
    return _load(CK / "thermal_parameter_registry.json")


@pytest.fixture(scope="session")
def b17_power() -> dict:
    return _load(CK / "mission_power_audit.json")


@pytest.fixture(scope="session")
def b17_feasibility() -> dict:
    return _load(CK / "thermal_feasibility.json")


@pytest.fixture(scope="session")
def b17_verdict() -> dict:
    return _load(CK / "b17_verdict.json")


@pytest.fixture(scope="session")
def b17_contract() -> dict:
    return _load(DOCS / "baseline_contract.json")
