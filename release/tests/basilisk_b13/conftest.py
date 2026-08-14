"""tests/basilisk_b13/conftest.py

B1.3 测试的模块加载夹具。

**为什么需要它**: 同名脚本现在是**五处** —— `scripts/basilisk/` (v1)、
`scripts/basilisk_b1/`、`scripts/basilisk_b11/`、`scripts/basilisk_b12/`、
`scripts/basilisk_b13/` 各有 `verify_baseline.py` / `generate_*.py` /
`build_features.py`。五个目录都会被各自的测试 `sys.path.insert` 进来,
`importlib.import_module("verify_baseline")` 会命中 `sys.modules` 里**先到的那一份**
—— 全套 pytest 一起跑时, B1.3 测试可能拿到 v1 / B1 / B1.1 / B1.2 的模块。

正确做法: 按**文件路径**显式加载, 注册成独立模块名 (`b13_*`), 不与任何同名脚本
抢 key; 并对加载结果断言 B1.3 独有的符号, 使"加载错文件"立刻暴露。

另: `tests/` 下没有 `__init__.py`, pytest 的模块名取自**文件basename**, 因此本目录
所有测试文件名都带 `b13` 前缀/后缀, 避免与前四个阶段的测试文件同名 (B1.2 阶段
已因此被迫改名两个文件)。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints" / "basilisk_b13"
DOCS = ROOT / "docs" / "basilisk_b13"
CFG_REL = "configs/wheel_basilisk_b13.yaml"


def load_by_path(mod_name: str, rel: str):
    """按路径加载模块, 注册为唯一 mod_name (避免与 v1/B1/B1.1/B1.2 同名脚本冲突)。"""
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
    """读 JSON; 不存在返回 None (让测试自己 skip, 而不是伪造空 dict)。"""
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# B1.3 自己的脚本
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def b13_verify():
    mod = load_by_path("b13_verify_baseline",
                       "scripts/basilisk_b13/verify_baseline.py")
    assert hasattr(mod, "B12_EXPECTED"), "加载到的不是 B1.3 verify_baseline"
    return mod


@pytest.fixture(scope="session")
def b13_audit_eq():
    mod = load_by_path("b13_audit_degradation_equation",
                       "scripts/basilisk_b13/audit_degradation_equation.py")
    assert hasattr(mod, "verify_multiplicative_structure"), \
        "加载到的不是 B1.3 audit_degradation_equation"
    return mod


@pytest.fixture(scope="session")
def b13_derive():
    mod = load_by_path("b13_derive_torque_scaling",
                       "scripts/basilisk_b13/derive_torque_scaling.py")
    assert hasattr(mod, "derive_b_fail_f2") and hasattr(mod, "derive_mapping"), \
        "加载到的不是 B1.3 derive_torque_scaling"
    return mod


@pytest.fixture(scope="session")
def b13_probe():
    mod = load_by_path("b13_generate_probe",
                       "scripts/basilisk_b13/generate_probe.py")
    assert hasattr(mod, "apply_b13_scaling"), "加载到的不是 B1.3 generate_probe"
    return mod


# ---------------------------------------------------------------------------
# 被复用的前阶段脚本 (只读, B1.3 未改写)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def b13_cal():
    """B1.1 的 calibrate_degradation (CRN / wear 层, B1.3 直接 import 复用)。"""
    mod = load_by_path("b11_calibrate_degradation",
                       "scripts/basilisk_b11/calibrate_degradation.py")
    assert hasattr(mod, "load_b11_config")
    return mod


@pytest.fixture(scope="session")
def b12_gen():
    """B1.2 的失效统计量 / 窗级判定 (B1.3 直接 import 复用)。"""
    mod = load_by_path("b12_generate_failure_candidates",
                       "scripts/basilisk_b12/generate_failure_candidates.py")
    for fn in ("candidate_statistic", "redecide_eol", "window_view"):
        assert hasattr(mod, fn), f"加载到的不是 B1.2 generate_failure_candidates (缺 {fn})"
    return mod


@pytest.fixture(scope="session")
def b13_cfg(b13_cal):
    return b13_cal.load_b11_config(CFG_REL)


# ---------------------------------------------------------------------------
# B1.3 产物
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def contract_rec():
    return read_json(DOCS / "baseline_contract.json")


@pytest.fixture(scope="session")
def eq_audit_rec():
    return read_json(CKPT / "degradation_equation_audit.json")


@pytest.fixture(scope="session")
def scaling_rec():
    return read_json(CKPT / "torque_scaling.json")


@pytest.fixture(scope="session")
def probe_rec():
    return read_json(CKPT / "probe_summary.json")


@pytest.fixture(scope="session")
def probe_audit_rec():
    return read_json(CKPT / "probe_audit.json")


@pytest.fixture(scope="session")
def frozen_rec():
    return read_json(CKPT / "frozen_calibration.json")


@pytest.fixture(scope="session")
def verdict_rec():
    return read_json(CKPT / "calibration_verdict.json")


@pytest.fixture(scope="session")
def dataset_rec():
    return read_json(CKPT / "dataset_summary.json")


@pytest.fixture(scope="session")
def feature_rec():
    return read_json(CKPT / "feature_summary.json")
