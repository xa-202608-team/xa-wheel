"""tests/basilisk_b4x/conftest.py —— B4X 测试夹具。

`POST_B2_FAIL_EXPLORATORY` / `NOT_FORMAL_EVIDENCE`

命名纪律: `tests/` 无 `__init__.py`, 模块 basename 在全部命名空间
(basilisk / b1 / b11 .. b19 / b2 / b3x / b4x) 中必须唯一, 故本目录所有**文件**
一律 `b4x_` 前缀。
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
CK = ROOT / "checkpoints" / "basilisk_b4x"
CK_B2 = ROOT / "checkpoints" / "basilisk_b2"
CK_B3X = ROOT / "checkpoints" / "basilisk_b3x"
DOCS = ROOT / "docs" / "basilisk_b4x"
CONFIG_REL = "configs/wheel_basilisk_b4x.yaml"


def _load(p: Path):
    if not p.exists():
        pytest.skip(f"产物不存在 (需先跑 B4X 流程): {p.relative_to(ROOT)}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_script(name: str, rel: str):
    key = f"b4xt_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = iu.spec_from_file_location(key, ROOT / rel)
    assert spec is not None and spec.loader is not None
    m = iu.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="session")
def b4x_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def b4x_config() -> dict:
    cal = load_script("cal11b4", "scripts/basilisk_b11/calibrate_degradation.py")
    return cal.load_b11_config(CONFIG_REL)


@pytest.fixture(scope="session")
def b4x_config_raw() -> dict:
    return yaml.safe_load(io.open(ROOT / CONFIG_REL, encoding="utf-8"))


@pytest.fixture(scope="session")
def b4x_baseline() -> dict:
    return _load(DOCS / "baseline_contract.json")


@pytest.fixture(scope="session")
def b4x_b2_split() -> dict:
    return _load(CK_B2 / "split.json")


@pytest.fixture(scope="session")
def b4x_b2_metrics() -> dict:
    return _load(CK_B2 / "metrics.json")


@pytest.fixture(scope="session")
def b4x_b3x_summary() -> dict:
    return _load(CK_B3X / "summary.json")


@pytest.fixture(scope="session")
def b4x_transfer() -> dict:
    return _load(CK / "transfer_metrics.json")


@pytest.fixture(scope="session")
def b4x_stability() -> dict:
    return _load(CK / "transfer_stability.json")


@pytest.fixture(scope="session")
def b4x_summary() -> dict:
    return _load(CK / "summary.json")


@pytest.fixture(scope="session")
def b4x_data_mod():
    return load_script("data_b4x", "scripts/basilisk_b4x/data_b4x.py")


@pytest.fixture(scope="session")
def b4x_run_mod():
    return load_script("run_b4x",
                       "scripts/basilisk_b4x/run_transfer_diagnostic.py")


@pytest.fixture(scope="session")
def b4x_stab_mod():
    return load_script("stab_b4x",
                       "scripts/basilisk_b4x/diagnose_transfer_stability.py")


@pytest.fixture(scope="session")
def b4x_summ_mod():
    return load_script("summ_b4x", "scripts/basilisk_b4x/summarize_b4x.py")


@pytest.fixture(scope="session")
def b4x_feature_h5(b4x_config):
    p = ROOT / b4x_config["transfer"]["target_feature_path"]
    if not p.exists():
        pytest.skip(f"缺 B1.9 特征文件 {p}")
    return p
