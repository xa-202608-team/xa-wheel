"""tests/basilisk_b19/conftest.py —— B1.9 测试夹具。

命名纪律: `tests/` 无 `__init__.py`, 模块 basename 在十一路命名空间
(basilisk / b1 / b11 / b12 / b13 / b14 / b15 / b16 / b17 / b18 / b19) 中必须唯一,
故本目录所有**文件**一律 `b19_` 前缀; §10 要求的**函数名**保持字节一致, 不加前缀。
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
CK = ROOT / "checkpoints" / "basilisk_b19"
DOCS = ROOT / "docs" / "basilisk_b19"
SCRIPTS = ROOT / "scripts" / "basilisk_b19"
CONFIG_REL = "configs/wheel_basilisk_b19.yaml"


def _load(p: Path):
    if not p.exists():
        pytest.skip(f"产物不存在 (需先跑 §11 步骤): {p.relative_to(ROOT)}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_script(name: str, rel: str):
    """按路径加载脚本模块 (脚本目录不是 package)。"""
    key = f"b19t_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = iu.spec_from_file_location(key, ROOT / rel)
    assert spec is not None and spec.loader is not None
    m = iu.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="session")
def b19_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def b19_config() -> dict:
    """走 base_config 继承链解析 —— 与所有脚本看到的配置一致。

    只 `yaml.safe_load` 原文件会看不到继承来的 `sim:` / `rated:` / `wear_drive:`,
    断言的就不是运行时真正生效的配置。
    """
    cal = load_script("cal11", "scripts/basilisk_b11/calibrate_degradation.py")
    return cal.load_b11_config(CONFIG_REL)


@pytest.fixture(scope="session")
def b19_config_raw() -> dict:
    """未解析继承链的原始 YAML —— 用于断言"某键确实写在 b19 文件里"。"""
    return yaml.safe_load(io.open(ROOT / CONFIG_REL, encoding="utf-8"))


@pytest.fixture(scope="session")
def b19_baseline() -> dict:
    return _load(DOCS / "baseline_contract.json")


@pytest.fixture(scope="session")
def b19_observable_audit() -> dict:
    return _load(CK / "observable_input_audit.json")


@pytest.fixture(scope="session")
def b19_damage_proxy() -> dict:
    return _load(CK / "damage_proxy.json")


@pytest.fixture(scope="session")
def b19_feature_stats() -> dict:
    return _load(CK / "feature_stats.json")


@pytest.fixture(scope="session")
def b19_feature_audit() -> dict:
    return _load(CK / "feature_audit.json")


@pytest.fixture(scope="session")
def b19_frozen() -> dict:
    return _load(CK / "frozen_feature_definition.json")


@pytest.fixture(scope="session")
def b19_proxy_module():
    """加载损伤代理模块 —— 供 prefix/monotone/bounded 测试真跑而非读 JSON。"""
    return load_script("proxy", "scripts/basilisk_b19/derive_damage_proxy.py")


@pytest.fixture(scope="session")
def b19_reference(b19_config):
    """reference duty —— 测试侧独立重算, 不读任何脚本产物。"""
    from src.sim.basilisk_bridge import load_profile_library
    cal = load_script("cal11", "scripts/basilisk_b11/calibrate_degradation.py")
    cfg = b19_config
    p = ROOT / cfg["paths"]["profile_h5"]
    if not p.exists():
        pytest.skip(f"缺 profile 库 {p}")
    lib = load_profile_library(
        p, wheel_index=int(cfg["sim"]["profile"]["bridge"]["wheel_index"]))
    n_per_window = int(round(float(cfg["sim"]["sample_period_s"])
                             / float(cfg["sim"]["profile"]["dt_s"])))
    return cal.reference_duty(
        lib, cfg["rated"], n_per_window,
        float(cfg["sim"]["profile"]["duty"]["maneuver_torque_frac"]), cfg)


@pytest.fixture(scope="session")
def b19_source_h5(b19_config):
    p = ROOT / b19_config["paths"]["b18_final_dir"] / "wheel_all.h5"
    if not p.exists():
        pytest.skip(f"缺 B1.8 冻结数据 {p}")
    return p


@pytest.fixture(scope="session")
def b19_feature_h5(b19_config):
    p = ROOT / b19_config["paths"]["feature_h5"]
    if not p.exists():
        pytest.skip(f"缺 B1.9 特征文件 {p} (需先跑 build_features.py)")
    return p
