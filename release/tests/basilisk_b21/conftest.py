"""tests/basilisk_b21/conftest.py —— B2.1 测试夹具。

`SPLIT_COVERAGE_ROBUSTNESS`

命名纪律: `tests/` 无 `__init__.py`, 模块 basename 在全部命名空间
(basilisk / b1 / b11 .. b19 / b2 / b3x / b4x / b21) 中必须唯一, 故本目录所有
**文件**一律 `b21_` 前缀, `sys.modules` 键前缀 `b21t_`。
"""
from __future__ import annotations

import importlib.util as iu
import io
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CK = ROOT / "checkpoints" / "basilisk_b21"
CK_B2 = ROOT / "checkpoints" / "basilisk_b2"
CK_B4X = ROOT / "checkpoints" / "basilisk_b4x"
DOCS = ROOT / "docs" / "basilisk_b21"
CONFIG_REL = "configs/wheel_basilisk_b21.yaml"


def _load(p: Path):
    if not p.exists():
        pytest.skip(f"产物不存在 (需先跑 B2.1 流程): {p.relative_to(ROOT)}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_script(name: str, rel: str):
    key = f"b21t_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = iu.spec_from_file_location(key, ROOT / rel)
    assert spec is not None and spec.loader is not None
    m = iu.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="session")
def b21_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def b21_config() -> dict:
    cal = load_script("cal11b21",
                      "scripts/basilisk_b11/calibrate_degradation.py")
    return cal.load_b11_config(CONFIG_REL)


@pytest.fixture(scope="session")
def b21_config_raw() -> dict:
    return yaml.safe_load(io.open(ROOT / CONFIG_REL, encoding="utf-8"))


@pytest.fixture(scope="session")
def b21_split() -> dict:
    return _load(DOCS / "split_manifest.json")


@pytest.fixture(scope="session")
def b21_gate() -> dict:
    return _load(CK / "gate_metrics.json")


@pytest.fixture(scope="session")
def b21_summary() -> dict:
    return _load(CK / "summary.json")


@pytest.fixture(scope="session")
def b21_contract() -> dict:
    return _load(DOCS / "baseline_contract.json")


@pytest.fixture(scope="session")
def b4x_summary_ro() -> dict:
    """B4X 的冻结结论 —— 本阶段只读, 不得改。"""
    return _load(CK_B4X / "summary.json")


@pytest.fixture(scope="session")
def b2_split_ro() -> dict:
    """B2 的冻结划分 —— 本阶段只读, 用于对照 lifetime-support mismatch。"""
    return _load(CK_B2 / "split.json")


@pytest.fixture(scope="session")
def b21_splitter():
    return load_script("build_b21", "scripts/basilisk_b21/build_b21_split.py")


@pytest.fixture(scope="session")
def b21_gate_mod():
    return load_script("gate_b21", "scripts/basilisk_b21/run_b21_gate.py")


@pytest.fixture(scope="session")
def b21_summarizer():
    return load_script("sum_b21", "scripts/basilisk_b21/summarize_b21.py")
