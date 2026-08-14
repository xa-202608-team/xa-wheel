"""tests/basilisk_b11/conftest.py

B1.1 测试的模块加载夹具。

**为什么需要它**: `build_features.py` 现在是**三处同名** ——
`scripts/basilisk/` (v1)、`scripts/basilisk_b1/` (B1)、`scripts/basilisk_b11/`;
`calibrate_degradation.py` 是两处同名 (B1 / B1.1)。三个目录都会被各自的测试
`sys.path.insert` 进来, `importlib.import_module("build_features")` 会命中
`sys.modules` 里**先到的那一份** —— 全套 pytest 一起跑时先导入的那份会赢, B1.1
测试就可能拿到 v1 或 B1 的模块。

正确做法: 按**文件路径**显式加载, 注册成独立模块名, 不与任何同名脚本抢 key。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_by_path(mod_name: str, rel: str):
    """按路径加载模块, 注册为唯一 mod_name (避免与 v1 / B1 同名脚本冲突)。"""
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    p = ROOT / rel
    if not p.exists():
        raise ImportError(f"缺 {rel}")
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {rel}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def b11_cal():
    """B1.1 的 calibrate_degradation (它自身 re-export B1 的 wear 结构)。"""
    mod = load_by_path("b11_calibrate_degradation",
                       "scripts/basilisk_b11/calibrate_degradation.py")
    assert hasattr(mod, "load_b11_config"), "加载到的不是 B1.1 calibrate_degradation"
    assert hasattr(mod, "initial_margin_ratio"), "缺 Gate 12 所需的 initial_margin_ratio"
    return mod


@pytest.fixture(scope="session")
def b1_cal(b11_cal):
    """B1.1 **实际 import 的那一份** B1 calibrate_degradation。

    注意不能在这里按路径重新加载一次 B1 的文件: 那会得到第二个独立的 module
    对象, 里面的函数与 B1.1 re-export 的函数虽同源却不是同一对象, 于是
    `is` 比较必然失败 —— 那是测试夹具的假失败, 不是实现问题。

    正确做法: 取 B1.1 自己注册的模块 (`b1_calibrate_degradation`), 并断言它的
    `__file__` 确实指向 `scripts/basilisk_b1/` —— 这样 "同一实现" 的判定既真实
    又能证明来源。
    """
    mod = sys.modules.get("b1_calibrate_degradation")
    assert mod is not None, \
        "B1.1 未通过 b1_calibrate_degradation 这一模块名加载 B1 实现"
    f = Path(mod.__file__).resolve()
    assert f == (ROOT / "scripts/basilisk_b1/calibrate_degradation.py").resolve(), \
        f"b1_calibrate_degradation 来源不对: {f}"
    assert hasattr(mod, "load_b1_config")
    return mod


@pytest.fixture(scope="session")
def b11_build_features():
    """B1.1 的 build_features 模块 (保证不是 v1 / B1 的同名脚本)。"""
    mod = load_by_path("b11_build_features",
                       "scripts/basilisk_b11/build_features.py")
    assert hasattr(mod, "feature_content_hash")
    assert hasattr(mod, "CORE_XT_COLS")
    src = (ROOT / "scripts/basilisk_b11/build_features.py").read_text(
        encoding="utf-8")
    assert "basilisk_b11" in src, "加载到的不是 B1.1 build_features"
    return mod


@pytest.fixture(scope="session")
def b11_cfg(b11_cal):
    return b11_cal.load_b11_config("configs/wheel_basilisk_b11.yaml")
