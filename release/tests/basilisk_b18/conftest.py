"""tests/basilisk_b18/conftest.py —— B1.8 测试夹具。

命名纪律: `tests/` 无 `__init__.py`, 模块 basename 在十路命名空间
(basilisk / b1 / b11 / b12 / b13 / b14 / b15 / b16 / b17 / b18) 中必须唯一,
故本目录所有**文件**一律 `b18_` 前缀; §22 要求的**函数名**保持字节一致, 不加前缀。
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CK = ROOT / "checkpoints" / "basilisk_b18"
DOCS = ROOT / "docs" / "basilisk_b18"
SCRIPTS = ROOT / "scripts" / "basilisk_b18"


def _load(p: Path):
    if not p.exists():
        pytest.skip(f"产物不存在 (需先跑 §23 步骤): {p.relative_to(ROOT)}")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def b18_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def b18_config() -> dict:
    """B1.8 config —— 走 base_config 继承链解析, 与所有脚本看到的一致。

    脚本一律用 `CAL.load_b11_config`, 测试若只 `yaml.safe_load` 原文件, 会看不到
    从 `configs/wheel_basilisk.yaml` 继承来的 `sim:` 等块, 断言的就不是运行时
    真正生效的配置。
    """
    import importlib.util
    import sys
    name = "b18_conftest_calibrate"
    if name in sys.modules:
        cal = sys.modules[name]
    else:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "scripts/basilisk_b11/calibrate_degradation.py")
        cal = importlib.util.module_from_spec(spec)
        sys.modules[name] = cal
        spec.loader.exec_module(cal)
    return cal.load_b11_config("configs/wheel_basilisk_b18.yaml")


@pytest.fixture(scope="session")
def b18_config_raw() -> dict:
    """未解析继承链的原始 YAML —— 用于断言"某键确实写在 b18 文件里"。"""
    return yaml.safe_load(
        io.open(ROOT / "configs/wheel_basilisk_b18.yaml", encoding="utf-8"))


@pytest.fixture(scope="session")
def b18_protocol_hash() -> dict:
    return _load(CK / "protocol_hash.json")


@pytest.fixture(scope="session")
def b18_registry() -> dict:
    return _load(CK / "assumption_registry.json")


@pytest.fixture(scope="session")
def b18_damage_model() -> dict:
    return _load(CK / "damage_model.json")


@pytest.fixture(scope="session")
def b18_sensitivity() -> dict:
    return _load(CK / "sensitivity_summary.json")


@pytest.fixture(scope="session")
def b18_audit() -> dict:
    return _load(CK / "sensitivity_audit.json")


@pytest.fixture(scope="session")
def b18_frozen_primary() -> dict:
    return _load(CK / "frozen_primary_scenario.json")


@pytest.fixture(scope="session")
def b18_dataset_audit() -> dict:
    return _load(CK / "dataset_audit.json")


@pytest.fixture(scope="session")
def b18_feature_stats() -> dict:
    return _load(CK / "feature_stats.json")


@pytest.fixture(scope="session")
def b18_baseline_contract() -> dict:
    return _load(DOCS / "baseline_contract.json")


@pytest.fixture(scope="session")
def b18_damage_module():
    """import src.sim.damage_model (纯函数模块, 不触发仿真)。"""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.sim import damage_model
    return damage_model
