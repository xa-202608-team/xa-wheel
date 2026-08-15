"""tests/basilisk_b14/conftest.py

B1.4 测试的模块加载夹具。

**为什么需要它**: 同名脚本现在是**六处** —— `scripts/basilisk/` (v1)、
`scripts/basilisk_b1/`、`_b11/`、`_b12/`、`_b13/`、`_b14/` 各有
`verify_baseline.py` / `generate_probe.py` / `audit_probe.py` /
`build_features.py`。`importlib.import_module("audit_probe")` 会命中
`sys.modules` 里**先到的那一份** —— 全套 pytest 一起跑时, B1.4 测试可能拿到
前五个阶段的模块。

正确做法: 按**文件路径**显式加载, 注册成独立模块名 (`b14_*`), 并对加载结果
断言 B1.4 独有的符号, 使"加载错文件"立刻暴露。

另: `tests/` 下没有 `__init__.py`, pytest 的模块名取自**文件 basename**,
因此本目录所有测试文件名都带 `b14` 前缀, 避免与前五阶段测试文件同名。
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints" / "basilisk_b14"
DOCS = ROOT / "docs" / "basilisk_b14"
CFG_REL = "configs/wheel_basilisk_b14.yaml"


def load_by_path(mod_name: str, rel: str):
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
    """读 JSON; 不存在返回 None (让测试自己 skip, 而不是伪造空 dict)。

    必须显式 encoding='utf-8' —— Windows 默认 gbk 会在中文 JSON 上抛
    UnicodeDecodeError (PYTHONIOENCODING 只管 stdio, 不管 open())。
    """
    if not p.exists():
        return None
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------- B1.4 脚本
@pytest.fixture(scope="session")
def b14_verify():
    mod = load_by_path("b14_verify_baseline",
                       "scripts/basilisk_b14/verify_baseline.py")
    assert hasattr(mod, "B13_EXPECTED"), "加载到的不是 B1.4 verify_baseline"
    return mod


@pytest.fixture(scope="session")
def b14_prov():
    mod = load_by_path("b14_audit_q0_provenance",
                       "scripts/basilisk_b14/audit_q0_provenance.py")
    assert hasattr(mod, "provenance_gate"), "加载到的不是 B1.4 audit_q0_provenance"
    return mod


@pytest.fixture(scope="session")
def b14_derive():
    mod = load_by_path("b14_derive_healthy_entry_prior",
                       "scripts/basilisk_b14/derive_healthy_entry_prior.py")
    assert hasattr(mod, "derive_q0_support") and \
        hasattr(mod, "reachability_analysis"), \
        "加载到的不是 B1.4 derive_healthy_entry_prior"
    return mod


@pytest.fixture(scope="session")
def b14_validate():
    mod = load_by_path("b14_validate_prior_math",
                       "scripts/basilisk_b14/validate_prior_math.py")
    assert hasattr(mod, "v1_algebra") and hasattr(mod, "B13_OUTCOME_TOKENS"), \
        "加载到的不是 B1.4 validate_prior_math"
    return mod


@pytest.fixture(scope="session")
def b14_probe_mod():
    mod = load_by_path("b14_generate_probe",
                       "scripts/basilisk_b14/generate_probe.py")
    assert hasattr(mod, "apply_b14_prior") and \
        hasattr(mod, "crn_uniform_evidence"), \
        "加载到的不是 B1.4 generate_probe"
    return mod


# ------------------------------------------------- 被复用的前阶段脚本 (只读)
@pytest.fixture(scope="session")
def b14_cal():
    mod = load_by_path("b11_calibrate_degradation",
                       "scripts/basilisk_b11/calibrate_degradation.py")
    assert hasattr(mod, "load_b11_config")
    return mod


@pytest.fixture(scope="session")
def b12_gen():
    mod = load_by_path("b12_generate_failure_candidates",
                       "scripts/basilisk_b12/generate_failure_candidates.py")
    for fn in ("candidate_statistic", "redecide_eol", "window_view"):
        assert hasattr(mod, fn), f"加载到的不是 B1.2 generate_failure_candidates (缺 {fn})"
    return mod


@pytest.fixture(scope="session")
def b14_cfg(b14_cal):
    return b14_cal.load_b11_config(CFG_REL)


# --------------------------------------------------------------- B1.4 产物
@pytest.fixture(scope="session")
def contract_rec():
    return read_json(DOCS / "baseline_contract.json")


@pytest.fixture(scope="session")
def prov_rec():
    return read_json(CKPT / "q0_provenance_audit.json")


@pytest.fixture(scope="session")
def prior_rec():
    return read_json(CKPT / "healthy_entry_prior.json")


@pytest.fixture(scope="session")
def prior_math_rec():
    return read_json(CKPT / "prior_math_validation.json")


@pytest.fixture(scope="session")
def protocol_rec():
    return read_json(CKPT / "protocol_hash.json")


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


# --------------------------------------------------------- B1.3 只读参照
@pytest.fixture(scope="session")
def b13_probe_rec():
    return read_json(ROOT / "checkpoints" / "basilisk_b13" / "probe_summary.json")


@pytest.fixture(scope="session")
def b13_scaling_rec():
    return read_json(ROOT / "checkpoints" / "basilisk_b13" / "torque_scaling.json")
