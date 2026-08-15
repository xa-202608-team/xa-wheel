"""tests/basilisk_b5/conftest.py —— B5 正式迁移评估测试夹具。

`FORMAL_TRANSFER_EVALUATION`

命名纪律: `tests/` 无 `__init__.py`, 模块 basename 在全部命名空间
(basilisk / b1 / b11 .. b19 / b2 / b3x / b4x / b21) 中必须唯一, 故本目录所有
**文件**一律 `b5_` 前缀, `sys.modules` 键前缀 `b5t_`。
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
CK = ROOT / "checkpoints" / "basilisk_b5"
CK_B2 = ROOT / "checkpoints" / "basilisk_b2"
CK_B3X = ROOT / "checkpoints" / "basilisk_b3x"
CK_B4X = ROOT / "checkpoints" / "basilisk_b4x"
CK_B21 = ROOT / "checkpoints" / "basilisk_b21"
DOCS = ROOT / "docs" / "basilisk_b5"
DOCS_B21 = ROOT / "docs" / "basilisk_b21"
CONFIG_REL = "configs/wheel_basilisk_b5.yaml"


def _load(p: Path):
    if not p.exists():
        pytest.skip(f"产物不存在 (需先跑 B5 流程): {p.relative_to(ROOT)}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_script(name: str, rel: str):
    key = f"b5t_{name}"
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
    """code_only 再去掉全部空白。

    token 之间是用换行拼的, 所以 `torch.Generator` 在 code_only 里长成
    `torch\\n.\\nGenerator`, 直接子串匹配一定落空 —— 那样的禁令测试是空转的。
    需要匹配带标点的写法 (`torch.Generator`、`>0.02`) 时扫这个。
    """
    return "".join(code_only(rel).split())


@pytest.fixture(scope="session")
def b5_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def b5_config() -> dict:
    cal = load_script("cal11b5", "scripts/basilisk_b11/calibrate_degradation.py")
    return cal.load_b11_config(CONFIG_REL)


@pytest.fixture(scope="session")
def b5_config_raw() -> dict:
    return yaml.safe_load(io.open(ROOT / CONFIG_REL, encoding="utf-8"))


@pytest.fixture(scope="session")
def b5_contract() -> dict:
    return _load(DOCS / "baseline_contract.json")


@pytest.fixture(scope="session")
def b5_protocol_hash() -> dict:
    return _load(CK / "protocol_hash.json")


@pytest.fixture(scope="session")
def b5_metrics() -> dict:
    return _load(CK / "formal_metrics.json")


@pytest.fixture(scope="session")
def b5_gain() -> dict:
    return _load(CK / "gain_analysis.json")


@pytest.fixture(scope="session")
def b5_lifetime() -> dict:
    return _load(CK / "lifetime_bin_analysis.json")


@pytest.fixture(scope="session")
def b5_warning() -> dict:
    return _load(CK / "warning_analysis.json")


@pytest.fixture(scope="session")
def b5_summary() -> dict:
    return _load(CK / "summary.json")


@pytest.fixture(scope="session")
def b21_split_ro() -> dict:
    """B2.1 冻结划分 —— B5 只读, 不得重划。"""
    return _load(DOCS_B21 / "split_manifest.json")


@pytest.fixture(scope="session")
def b21_summary_ro() -> dict:
    return _load(CK_B21 / "summary.json")


@pytest.fixture(scope="session")
def b21_gate_ro() -> dict:
    return _load(CK_B21 / "gate_metrics.json")


@pytest.fixture(scope="session")
def b4x_summary_ro() -> dict:
    """B4X 探索性结论 —— 只读, 不得被提升为正式结论。"""
    return _load(CK_B4X / "summary.json")


@pytest.fixture(scope="session")
def b3x_summary_ro() -> dict:
    return _load(CK_B3X / "summary.json")


@pytest.fixture(scope="session")
def b5_data_mod():
    return load_script("data_b5", "scripts/basilisk_b5/data_b5.py")


@pytest.fixture(scope="session")
def b5_runner():
    return load_script("run_b5", "scripts/basilisk_b5/run_formal_transfer.py")


@pytest.fixture(scope="session")
def b5_gain_mod():
    return load_script("gain_b5", "scripts/basilisk_b5/analyze_paired_gain.py")


@pytest.fixture(scope="session")
def b5_summarizer():
    return load_script("sum_b5", "scripts/basilisk_b5/summarize_b5.py")


@pytest.fixture(scope="session")
def b5_freezer():
    return load_script("freeze_b5", "scripts/basilisk_b5/freeze_protocol.py")
