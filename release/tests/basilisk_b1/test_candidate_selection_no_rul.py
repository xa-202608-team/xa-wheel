"""tests/basilisk_b1/test_candidate_selection_no_rul.py

B1 §5 硬测试 (一) —— candidate 选择绝不允许使用 RUL 指标。

§5 明令 B1 的 candidate selection 只能看物理量 (failure fraction / EOL 分布 /
HI 动态范围 / friction growth / 温度电流范围 / mode-conditioned wear / censored
fraction / 数值稳定性), **绝对禁止**使用 RUL RMSE、correlation、PH、warning
lead、transfer gain 或任何神经网络指标。

实现方式: AST 静态分析 (不用子串匹配 —— 注释里出现 "rul" 不该导致误报, 而
`metrics["rul_rmse"]` 这种真实使用必须被抓到)。
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 参与 candidate 选择决策的脚本 (必须无 RUL 依赖)
SELECTION_SCRIPTS = ("scripts/basilisk_b1/generate_candidates.py",
                     "scripts/basilisk_b1/audit_candidates.py",
                     "scripts/basilisk_b1/freeze_candidate.py")

# RUL / 模型指标名 (作为标识符、属性名或字符串常量出现即违规)
RUL_TOKENS = {
    "rul_rmse", "rmse_rul", "rul_mae", "mae_rul", "nphm", "phm", "ph",
    "warning_lead", "lead_time", "transfer_gain", "rul_corr",
    "correlation_rul", "rul_pred", "pred_rul", "rul_hat", "rul_error",
    "test_rmse", "val_rmse", "rul_metric",
}
# 禁止 import 的模块前缀 (模型 / 迁移 / 实验编排)
FORBIDDEN_MODULES = ("torch", "src.models", "src.transfer", "src.experiments",
                     "src.baselines", "sklearn.metrics")
# 禁止读取的模型指标产物
FORBIDDEN_METRIC_FILES = ("s3_gate_metrics.json", "s4_metrics.json",
                          "s5_rate_metrics.json", "s5b_wiener_pf_metrics.json",
                          "results.md", "transfer_metrics.json")


def _tree(rel: str) -> tuple[ast.AST, str]:
    p = ROOT / rel
    assert p.exists(), f"缺 {rel}"
    src = p.read_text(encoding="utf-8")
    return ast.parse(src), src


def _iter_strings(tree: ast.AST):
    """遍历所有字符串常量, 但**跳过 docstring** (说明性文字不算使用)。"""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                            ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            yield node, node.value


@pytest.mark.parametrize("rel", SELECTION_SCRIPTS)
def test_candidate_selection_no_rul(rel):
    """选择链脚本中不得出现任何 RUL 指标的标识符 / 属性 / 键名。"""
    tree, _ = _tree(rel)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.lower() in RUL_TOKENS:
            bad.append(f"L{node.lineno}: 变量 {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.lower() in RUL_TOKENS:
            bad.append(f"L{node.lineno}: 属性 .{node.attr}")
        elif isinstance(node, ast.keyword) and node.arg \
                and node.arg.lower() in RUL_TOKENS:
            bad.append(f"L{node.lineno}: 关键字参数 {node.arg}")
    for node, s in _iter_strings(tree):
        key = s.strip().lower()
        if key in RUL_TOKENS:
            bad.append(f"L{node.lineno}: 字符串键 {s!r}")
    assert not bad, f"{rel} 使用了 RUL 指标 (违反 §5):\n" + "\n".join(bad)


@pytest.mark.parametrize("rel", SELECTION_SCRIPTS)
def test_candidate_selection_never_reads_metric_files(rel):
    """选择链脚本不得读取任何已冻结的模型指标产物。"""
    _, src = _tree(rel)
    bad = [f for f in FORBIDDEN_METRIC_FILES if f in src]
    assert not bad, f"{rel} 引用了模型指标文件 {bad} (违反 §5)"


def test_candidate_json_declares_no_rul_label():
    """candidate 数据集必须机器可读地声明"不含 RUL 标签"。"""
    import h5py
    from src.utils.config import PROJECT_ROOT
    import importlib
    sys.path.insert(0, str(ROOT / "scripts" / "basilisk_b1"))
    cal = importlib.import_module("calibrate_degradation")
    cfg = cal.load_b1_config("configs/wheel_basilisk_b1.yaml")
    root = PROJECT_ROOT / cfg["paths"]["candidate_dir"]
    if not root.exists():
        pytest.skip("candidate 数据未生成")
    found = 0
    for cname in cfg["candidates"]["registry"]:
        p = root / cname / "wheel_all.h5"
        if not p.exists():
            continue
        found += 1
        with h5py.File(p, "r") as f:
            assert bool(f.attrs["contains_rul_label"]) is False, \
                f"{cname} 声明含 RUL 标签"
            g = f[sorted(k for k in f.keys() if k.startswith("traj_"))[0]]
            for forbidden in ("rul", "rul_lower_bound", "hi_b", "hi_a", "x_T"):
                assert forbidden not in g, \
                    f"{cname} candidate 数据里出现 {forbidden} —— candidate 阶段" \
                    "不该有 HI/RUL 派生量"
    if found == 0:
        pytest.skip("candidate h5 尚未落盘")


def test_audit_report_records_no_rul_flag():
    """candidate_audit.json 必须留下 used_rul_metric=False 的可审计记录。"""
    p = ROOT / "checkpoints/basilisk_b1/candidate_audit.json"
    if not p.exists():
        pytest.skip("candidate_audit.json 未生成")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["used_rul_metric"] is False
    assert d["selected_before_any_RUL_training"] is True
