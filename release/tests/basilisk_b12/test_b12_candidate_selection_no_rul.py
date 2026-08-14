"""tests/basilisk_b12/test_b12_candidate_selection_no_rul.py

文件名带 b12_ 前缀: pytest 无 __init__.py 时按 basename 建模块名, 与
tests/basilisk_b1/test_candidate_selection_no_rul.py 同名会 collection error。
测试**函数名**仍按 §19 要求为 test_candidate_selection_no_rul。

§19 test_candidate_selection_no_rul —— 判据选择过程绝不可依赖 RUL 指标。

三层证据:
  1. **静态 (AST)**: freeze_failure_definition.py 的源码里不得出现任何 RUL 训练 /
     评估相关的符号 —— 不是查字符串, 而是查真实的 import 与属性访问链;
  2. **记录**: selection_result.json 声明 used_rul_metric=False, 且 selection_basis
     只写 Gate 表;
  3. **数据**: candidate h5 不得含任何 RUL 标签数据集。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# 判据选择阶段严禁触碰的模块 (§2: 不跑 target_only / source_finetune / MMD /
# rate model / Wiener PF)
FORBIDDEN_MODULES = (
    "src.train", "src.transfer", "src.baselines", "src.experiments",
    "torch", "sklearn.metrics",
)
FORBIDDEN_NAMES = (
    "rmse", "phm_score", "nphm", "predict_rul", "rul_pred",
    "target_only", "source_finetune", "source_mmd",
    "particle_filter", "wiener",
)

SELECT_SCRIPTS = (
    "scripts/basilisk_b12/freeze_failure_definition.py",
    "scripts/basilisk_b12/audit_failure_candidates.py",
    "scripts/basilisk_b12/generate_failure_candidates.py",
)


def _imports(tree) -> set[str]:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
    return out


def test_candidate_selection_no_rul(sel_rec, audit_rec, cand_rec):
    # ---- 1. AST: 选择链路上的脚本不得 import 训练/评估模块 ----
    for rel in SELECT_SCRIPTS:
        p = ROOT / rel
        assert p.exists(), f"缺 {rel}"
        tree = ast.parse(p.read_text(encoding="utf-8"))
        imps = _imports(tree)
        for bad in FORBIDDEN_MODULES:
            hit = [m for m in imps if m == bad or m.startswith(bad + ".")]
            assert not hit, f"{rel} import 了 {hit} —— 判据选择不得触碰 RUL 模型链路"
        # 属性/函数名层面: 排除注释与字符串, 只看真实标识符
        idents = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        idents |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        idents |= {n.name for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for bad in FORBIDDEN_NAMES:
            assert not any(bad in i.lower() for i in idents), \
                f"{rel} 出现 RUL 相关标识符含 '{bad}'"

    # ---- 2. 记录声明 ----
    for rec, tag in ((cand_rec, "candidates.json"),
                     (audit_rec, "candidate_audit.json"),
                     (sel_rec, "selection_result.json")):
        if rec is None:
            continue
        assert rec["used_rul_metric"] is False, f"{tag} used_rul_metric != False"

    if sel_rec is None:
        pytest.skip("selection_result.json 尚未生成")
    assert sel_rec["selected_before_any_RUL_training"] is True
    basis = sel_rec["selection_basis"]
    assert "Gate" in basis and "优先级" in basis, f"selection_basis 不合规: {basis}"
    assert sel_rec["verdict"] in {"B12_FAILURE_DEFINITION_READY",
                                 "B12_FAILURE_DEFINITION_FAIL"}

    # ---- 3. 数据: candidate h5 不得含 RUL 标签 ----
    if cand_rec is None:
        pytest.skip("candidates.json 尚未生成")
    import h5py
    for name, cd in cand_rec["per_candidate"].items():
        with h5py.File(Path(cd["h5"]), "r") as f:
            assert bool(f.attrs["contains_rul_label"]) is False, \
                f"{name} 声明含 RUL 标签"
            g = f[sorted(k for k in f.keys() if k.startswith("traj_"))[0]]
            for k in g.keys():
                assert "rul" not in k.lower(), f"{name} 含 RUL 数据集 {k}"


def test_b12_no_rul_artifacts_written():
    """§0/§22: B1.2 阶段不得产出任何 RUL 模型产物。"""
    ck = ROOT / "checkpoints" / "basilisk_b12"
    if not ck.exists():
        pytest.skip("checkpoints/basilisk_b12 尚未创建")
        return
    bad = [p.name for p in ck.rglob("*")
           if p.suffix.lower() in {".pt", ".pth", ".ckpt", ".onnx"}]
    assert not bad, f"B1.2 目录出现模型权重 {bad} —— §22 明令本阶段不跑模型"
    for p in ck.rglob("*.json"):
        low = p.read_text(encoding="utf-8").lower()
        for bad_k in ("val_loss", "test_rmse", "rul_rmse", "phm_score"):
            assert bad_k not in low, f"{p.name} 含模型指标 '{bad_k}'"
