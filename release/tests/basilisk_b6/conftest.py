"""tests/basilisk_b6/conftest.py —— B6 失效标签稀缺矩阵测试夹具。

`FAILURE_LABEL_SCARCITY_FORMAL_MATRIX`

命名纪律: `tests/` 无 `__init__.py`, 模块 basename 在全部命名空间
(basilisk / b1 / b11 .. b19 / b2 / b3x / b4x / b21 / b5) 中必须唯一, 故本目录
所有**文件**一律 `b6_` 前缀, `sys.modules` 键前缀 `b6t_`。
"""
from __future__ import annotations

import importlib.util as iu
import io
import json
import sys
import tokenize
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CK = ROOT / "checkpoints" / "basilisk_b6"
CK_B5 = ROOT / "checkpoints" / "basilisk_b5"
CK_B21 = ROOT / "checkpoints" / "basilisk_b21"
CK_B3X = ROOT / "checkpoints" / "basilisk_b3x"
CK_B4X = ROOT / "checkpoints" / "basilisk_b4x"
DOCS = ROOT / "docs" / "basilisk_b6"
DOCS_B21 = ROOT / "docs" / "basilisk_b21"
CONFIG_REL = "configs/wheel_basilisk_b6.yaml"


def _load(p: Path):
    if not p.exists():
        pytest.skip(f"产物不存在 (需先跑 B6 流程): {p.relative_to(ROOT)}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_script(name: str, rel: str):
    key = f"b6t_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = iu.spec_from_file_location(key, ROOT / rel)
    assert spec is not None and spec.loader is not None
    m = iu.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    return m


def code_only(rel: str) -> str:
    """剥掉注释与字符串后的源码。

    禁令类测试必须扫这个 —— 否则"注释里写明禁止使用 X"本身会被判成使用 X。
    """
    src = (ROOT / rel).read_text(encoding="utf-8")
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError):  # pragma: no cover
        return src
    return "\n".join(out)


def code_nospace(rel: str) -> str:
    """code_only 再去掉全部空白 —— 需要匹配带标点的写法时扫这个。"""
    return "".join(code_only(rel).split())


@pytest.fixture(scope="session")
def b6_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def b6_config() -> dict:
    cal = load_script("cal11b6", "scripts/basilisk_b11/calibrate_degradation.py")
    return cal.load_b11_config(CONFIG_REL)


@pytest.fixture(scope="session")
def b6_config_raw() -> dict:
    return yaml.safe_load(io.open(ROOT / CONFIG_REL, encoding="utf-8"))


@pytest.fixture(scope="session")
def b6_contract() -> dict:
    return _load(DOCS / "baseline_contract.json")


@pytest.fixture(scope="session")
def b6_protocol_hash() -> dict:
    return _load(CK / "protocol_hash.json")


@pytest.fixture(scope="session")
def b6_manifest() -> dict:
    return _load(CK / "label_subset_manifest.json")


@pytest.fixture(scope="session")
def b6_metrics() -> dict:
    return _load(CK / "all_metrics.json")


@pytest.fixture(scope="session")
def b6_paired() -> dict:
    return _load(CK / "paired_statistics.json")


@pytest.fixture(scope="session")
def b6_lifetime() -> dict:
    return _load(CK / "lifetime_bins.json")


@pytest.fixture(scope="session")
def b6_warning() -> dict:
    return _load(CK / "warning_metrics.json")


@pytest.fixture(scope="session")
def b6_summary() -> dict:
    return _load(CK / "summary.json")


@pytest.fixture(scope="session")
def b6_verdict() -> dict:
    return _load(CK / "final_verdict.json")


@pytest.fixture(scope="session")
def b21_split_ro() -> dict:
    """B2.1 冻结划分 —— B6 只读, 不得重划。"""
    return _load(DOCS_B21 / "split_manifest.json")


@pytest.fixture(scope="session")
def b5_summary_ro() -> dict:
    """B5 终局结论 —— 只读, 不得被 B6 覆盖。"""
    return _load(CK_B5 / "summary.json")


@pytest.fixture(scope="session")
def b5_protocol_ro() -> dict:
    return _load(CK_B5 / "protocol_hash.json")


@pytest.fixture(scope="session")
def b3x_summary_ro() -> dict:
    return _load(CK_B3X / "summary.json")


@pytest.fixture(scope="session")
def b4x_summary_ro() -> dict:
    return _load(CK_B4X / "summary.json")


@pytest.fixture(scope="session")
def b21_summary_ro() -> dict:
    return _load(CK_B21 / "summary.json")


@pytest.fixture(scope="session")
def b6_data_mod():
    return load_script("data_b6", "scripts/basilisk_b6/data_b6.py")


@pytest.fixture(scope="session")
def b6_subset_builder():
    return load_script("build_b6", "scripts/basilisk_b6/build_label_subsets.py")


@pytest.fixture(scope="session")
def b6_runner():
    return load_script("run_b6", "scripts/basilisk_b6/run_formal_matrix.py")


@pytest.fixture(scope="session")
def b6_stats_mod():
    return load_script("stats_b6", "scripts/basilisk_b6/paired_statistics.py")


@pytest.fixture(scope="session")
def b6_summarizer():
    return load_script("sum_b6", "scripts/basilisk_b6/summarize_matrix.py")


@pytest.fixture(scope="session")
def b6_verdict_mod():
    return load_script("verd_b6", "scripts/basilisk_b6/final_transfer_verdict.py")


@pytest.fixture(scope="session")
def b6_freezer():
    return load_script("freeze_b6", "scripts/basilisk_b6/freeze_protocol.py")
