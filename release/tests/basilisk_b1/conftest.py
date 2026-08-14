"""tests/basilisk_b1/conftest.py

B1 测试的模块加载夹具。

**为什么需要它**: `scripts/basilisk/build_features.py` (v1) 与
`scripts/basilisk_b1/build_features.py` (B1) 同名。两个目录都被各自的测试
`sys.path.insert` 进来, `importlib.import_module("build_features")` 会命中
`sys.modules` 里**先到的那一份** —— 全套 pytest 一起跑时 v1 先被导入, B1 测试
就会拿到 v1 的模块 (缺 `feature_content_hash` 而报 AttributeError)。

正确做法: 按**文件路径**显式加载, 并注册成独立模块名, 不与 v1 抢同一个 key。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_by_path(mod_name: str, rel: str):
    """按路径加载模块, 注册为唯一 mod_name (避免与同名 v1 脚本冲突)。"""
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    p = ROOT / rel
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {rel}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def b1_build_features():
    """B1 的 build_features 模块 (保证不是 v1 的同名脚本)。"""
    mod = _load_by_path("b1_build_features",
                        "scripts/basilisk_b1/build_features.py")
    assert hasattr(mod, "feature_content_hash"), \
        "加载到的不是 B1 build_features (可能与 v1 同名脚本混淆)"
    assert hasattr(mod, "CORE_XT_COLS")
    return mod
