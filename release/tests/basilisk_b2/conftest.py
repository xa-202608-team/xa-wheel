"""tests/basilisk_b2/conftest.py —— B2 测试夹具。

命名纪律: `tests/` 无 `__init__.py`, 模块 basename 在全部命名空间
(basilisk / b1 / b11 .. b19 / b2) 中必须唯一, 故本目录所有**文件**一律
`b2_` 前缀。
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
CK = ROOT / "checkpoints" / "basilisk_b2"
DOCS = ROOT / "docs" / "basilisk_b2"
SCRIPTS = ROOT / "scripts" / "basilisk_b2"
CONFIG_REL = "configs/wheel_basilisk_b2.yaml"


def _load(p: Path):
    if not p.exists():
        pytest.skip(f"产物不存在 (需先跑 B2 流程): {p.relative_to(ROOT)}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_script(name: str, rel: str):
    """按路径加载脚本模块 (脚本目录不是 package)。"""
    key = f"b2t_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = iu.spec_from_file_location(key, ROOT / rel)
    assert spec is not None and spec.loader is not None
    m = iu.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="session")
def b2_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def b2_config() -> dict:
    """走 base_config 继承链解析 —— 与脚本运行时看到的配置一致。"""
    cal = load_script("cal11", "scripts/basilisk_b11/calibrate_degradation.py")
    return cal.load_b11_config(CONFIG_REL)


@pytest.fixture(scope="session")
def b2_config_raw() -> dict:
    """未解析继承链的原始 YAML —— 断言"某键确实写在 b2 文件里"。"""
    return yaml.safe_load(io.open(ROOT / CONFIG_REL, encoding="utf-8"))


@pytest.fixture(scope="session")
def b2_baseline() -> dict:
    return _load(DOCS / "baseline_contract.json")


@pytest.fixture(scope="session")
def b2_split() -> dict:
    return _load(CK / "split.json")


@pytest.fixture(scope="session")
def b2_metrics() -> dict:
    return _load(CK / "metrics.json")


@pytest.fixture(scope="session")
def b2_data_mod():
    return load_script("data_b2", "scripts/basilisk_b2/data_b2.py")


@pytest.fixture(scope="session")
def b2_eval_mod():
    return load_script("eval_b2", "scripts/basilisk_b2/eval_b2.py")


@pytest.fixture(scope="session")
def b2_gate_mod():
    return load_script("run_gate", "scripts/basilisk_b2/run_gate.py")


@pytest.fixture(scope="session")
def b2_split_mod():
    return load_script("build_split", "scripts/basilisk_b2/build_split.py")


@pytest.fixture(scope="session")
def b2_baselines_mod():
    return load_script("baselines_b2", "scripts/basilisk_b2/baselines_b2.py")


@pytest.fixture(scope="session")
def b2_train_mod():
    return load_script("train_b2", "scripts/basilisk_b2/train_b2.py")


@pytest.fixture(scope="session")
def b2_feature_h5(b2_config):
    p = ROOT / b2_config["transfer"]["target_feature_path"]
    if not p.exists():
        pytest.skip(f"缺 B1.9 特征文件 {p}")
    return p
