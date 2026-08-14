"""tests/basilisk_b12/conftest.py

B1.2 测试的模块加载夹具。

**为什么需要它**: 同名脚本现在是**四处** ——
`scripts/basilisk/` (v1)、`scripts/basilisk_b1/`、`scripts/basilisk_b11/`、
`scripts/basilisk_b12/` 各有 `verify_baseline.py` / `build_features.py`;
`generate_*.py` 亦然。四个目录都会被各自的测试 `sys.path.insert` 进来,
`importlib.import_module("build_features")` 会命中 `sys.modules` 里**先到的那一份**
—— 全套 pytest 一起跑时, B1.2 测试可能拿到 v1 / B1 / B1.1 的模块。

正确做法: 按**文件路径**显式加载, 注册成独立模块名 (`b12_*`), 不与任何同名脚本
抢 key; 并对加载结果断言 B1.2 独有的符号, 使"加载错文件"立刻暴露。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints" / "basilisk_b12"
DOCS = ROOT / "docs" / "basilisk_b12"


def load_by_path(mod_name: str, rel: str):
    """按路径加载模块, 注册为唯一 mod_name (避免与 v1 / B1 / B1.1 同名脚本冲突)。"""
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


def read_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def b12_gen():
    """B1.2 的 generate_failure_candidates (含窗级判定与阈值推导)。"""
    mod = load_by_path("b12_generate_failure_candidates",
                       "scripts/basilisk_b12/generate_failure_candidates.py")
    for fn in ("derive_thresholds", "redecide_eol", "candidate_statistic",
               "window_view", "degradation_fraction_at_eol"):
        assert hasattr(mod, fn), f"加载到的不是 B1.2 generate_failure_candidates (缺 {fn})"
    return mod


@pytest.fixture(scope="session")
def b12_mech():
    mod = load_by_path("b12_audit_failure_mechanisms",
                       "scripts/basilisk_b12/audit_failure_mechanisms.py")
    assert hasattr(mod, "decompose") and hasattr(mod, "frac_series")
    return mod


@pytest.fixture(scope="session")
def b12_prov():
    mod = load_by_path("b12_audit_electrical_provenance",
                       "scripts/basilisk_b12/audit_electrical_provenance.py")
    assert hasattr(mod, "decide_verdict") and hasattr(mod, "consistency_matrix")
    return mod


@pytest.fixture(scope="session")
def b12_verify():
    mod = load_by_path("b12_verify_baseline",
                       "scripts/basilisk_b12/verify_baseline.py")
    assert hasattr(mod, "B11_EXPECTED"), "加载到的不是 B1.2 verify_baseline"
    return mod


@pytest.fixture(scope="session")
def b12_cal():
    """B1.2 复用的 B1.1 calibrate_degradation (CRN / wear 层)。"""
    mod = load_by_path("b11_calibrate_degradation",
                       "scripts/basilisk_b11/calibrate_degradation.py")
    assert hasattr(mod, "load_b11_config")
    return mod


@pytest.fixture(scope="session")
def b12_cfg(b12_cal):
    return b12_cal.load_b11_config("configs/wheel_basilisk_b12.yaml")


@pytest.fixture(scope="session")
def prov_rec():
    return read_json(CKPT / "electrical_provenance.json")


@pytest.fixture(scope="session")
def mech_rec():
    return read_json(CKPT / "failure_mechanism_audit.json")


@pytest.fixture(scope="session")
def cand_rec():
    return read_json(CKPT / "candidates.json")


@pytest.fixture(scope="session")
def audit_rec():
    return read_json(CKPT / "candidate_audit.json")


@pytest.fixture(scope="session")
def sel_rec():
    return read_json(CKPT / "selection_result.json")
