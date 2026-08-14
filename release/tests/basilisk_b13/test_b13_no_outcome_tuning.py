"""tests/basilisk_b13/test_b13_no_outcome_tuning.py

§8/§10/§13/§19 —— 标定不得被"结果"反推。

§19 命名测试:
  * test_no_hardcoded_scale_ratio
  * test_single_candidate_only
  * test_no_failure_fraction_tuning
  * test_no_rul_metric
  * test_f2_threshold_frozen

做法: 对 `scripts/basilisk_b13/` 的标定链做 **AST 常量扫描** + config 逐值比对。
只看字符串是不够的 —— `0.65` 可以写成 `65/100`, 所以扫的是 AST 里的数值常量,
并把注释/文档字符串排除在外 (注释里提到禁令是**允许**的, 甚至是必要的)。
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from conftest import ROOT

# 标定链: 这些文件里出现"目标数值"就是反推标定
CALIBRATION_CHAIN = (
    "scripts/basilisk_b13/derive_torque_scaling.py",
    "scripts/basilisk_b13/generate_probe.py",
    "scripts/basilisk_b13/freeze_calibration.py",
)

# §8 禁止出现在标定代码里的数值常量
FORBIDDEN_CONSTANTS = {
    0.65: "B1 的候选失效率目标 (用失效率反推参数)",
    0.6: "候选失效率目标",
    0.7: "候选失效率目标",
    3.95: "b_fail_f2/b_fail_old 与 B1.1 b0_scale 的比值 (§5 明令不得硬编码)",
    23.114905847261028: "b0_scale 本身 (必须运行时推导, 不得写死)",
    5.906442914788147: "B1.1 的旧 b0_scale (不得写死)",
}
# 允许的物理常数 / 结构常数 (它们是输入, 不是被反推的目标)
ALLOWED = {
    0.14311462970213382,   # F2 阈值: 从 B1.2 只读
    0.00125,               # Tc_nom
    0.4041898280945392,    # speed_util_ref_p95
    628.3185307179587,     # Omega_rated
    0.0005586102246421416,  # b_fail_f2: 作为 readback 校验值
}


def numeric_constants(rel: str) -> list[tuple[int, float]]:
    """返回文件里所有数值常量 (行号, 值)。注释与 docstring 天然不在 AST 里。"""
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    out: list[tuple[int, float]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            out.append((node.lineno, float(node.value)))
    return out


def string_constants(rel: str) -> list[tuple[int, str]]:
    """AST 里的字符串常量, 但**排除** docstring (文档里写禁令是允许的)。"""
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            d = ast.get_docstring(node, clean=False)
            if d is not None:
                docstrings.add(d)
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value not in docstrings:
            out.append((node.lineno, node.value))
    return out


def test_no_hardcoded_scale_ratio():
    """§5/§8: 标定链里不得出现 3.95 / b0_scale 数值 / 失效率目标。"""
    hits = []
    for rel in CALIBRATION_CHAIN:
        for lineno, v in numeric_constants(rel):
            if v in ALLOWED:
                continue
            for bad, why in FORBIDDEN_CONSTANTS.items():
                if math.isclose(v, bad, rel_tol=1e-9, abs_tol=1e-12):
                    hits.append(f"{rel}:{lineno} 出现 {v!r} —— {why}")
    assert not hits, "标定代码里出现被禁的目标数值:\n" + "\n".join(hits)


def test_no_hardcoded_ratio_expression():
    """§5: 也不得写成 `0.143115 / 0.03625` 这种"看起来像推导"的除法字面量。"""
    hits = []
    for rel in CALIBRATION_CHAIN:
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) \
                    and isinstance(node.left, ast.Constant) \
                    and isinstance(node.right, ast.Constant):
                lo, ro = node.left.value, node.right.value
                if not isinstance(lo, (int, float)) or not isinstance(ro, (int, float)):
                    continue
                # 两个字面量相除且商接近 3.95 或 23.11 -> 就是硬编码比值换了个写法
                if ro == 0:
                    continue
                r = lo / ro
                for bad in (3.95, 23.114905847261028):
                    if math.isclose(r, bad, rel_tol=1e-3):
                        hits.append(f"{rel}:{node.lineno} 字面量相除 {lo}/{ro} ≈ {bad}")
    assert not hits, "出现硬编码比值 (换成除法写法):\n" + "\n".join(hits)


def test_single_candidate_only(b13_cfg, scaling_rec):
    """§8: 只允许一个 candidate `TQ1_TORQUE_NORMALIZED_MAPPING`。"""
    ds = b13_cfg["degradation_scaling"]
    assert ds["candidate"] == "TQ1_TORQUE_NORMALIZED_MAPPING"
    if scaling_rec is not None:
        m = scaling_rec["mapping"]
        assert int(m["n_candidates"]) == 1, \
            f"出现 {m['n_candidates']} 个 candidate —— §8 只允许一个"
        assert m["candidate"] == "TQ1_TORQUE_NORMALIZED_MAPPING"


def test_no_failure_fraction_tuning(b13_cfg, scaling_rec, probe_rec, frozen_rec):
    """§8: 四个反作弊标志位必须全 false, 且贯穿 config / 推导 / probe / 冻结。"""
    flags = ("hardcoded_target", "used_failure_fraction_for_calibration",
             "used_rul_metric", "used_model_metric")
    ds = b13_cfg["degradation_scaling"]
    for f in flags:
        assert ds[f] is False, f"config degradation_scaling.{f} 不是 false"
    for rec, name in ((scaling_rec, "torque_scaling"), (probe_rec, "probe_summary")):
        if rec is None:
            continue
        for f in flags:
            if f in rec:
                assert rec[f] is False, f"{name}.{f} 不是 false"
    if frozen_rec is not None:
        for f in flags:
            assert frozen_rec["protocol"][f] is False, \
                f"frozen_calibration.protocol.{f} 不是 false"
        assert frozen_rec["protocol"]["calibrated_before_any_rul_training"] is True


def test_no_rul_metric():
    """§8/§18: 标定链不得 import 任何模型/指标模块, 也不得读 metrics 文件。"""
    banned_mods = ("src.models", "src.train", "src.transfer", "src.baselines",
                   "src.experiments", "torch")
    hits = []
    for rel in CALIBRATION_CHAIN + ("scripts/basilisk_b13/audit_probe.py",):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for nm in names:
                for b in banned_mods:
                    if nm == b or nm.startswith(b + "."):
                        hits.append(f"{rel}:{node.lineno} import {nm}")
        for lineno, s in string_constants(rel):
            low = s.lower()
            if ("metrics" in low or "rmse" in low) and (".json" in low or "/" in low):
                # 允许 did_not_read 之类的**声明性**字符串, 但不允许真的路径拼接
                if "did_not_read" not in low and "checkpoints/*" not in low:
                    hits.append(f"{rel}:{lineno} 可疑指标路径 {s!r}")
    assert not hits, "标定/审计链触碰了模型或指标:\n" + "\n".join(hits)


def test_f2_threshold_frozen(b13_cfg, scaling_rec, probe_rec):
    """§禁止: F2 阈值只读, 全链路必须是同一个值。

    命名口径: B1.2 的**规范 candidate id** 是 `FRICTION_TORQUE_P95` (见
    checkpoints/basilisk_b12/candidates.json 的 per_candidate 键)。b13 config 里
    的 label 带 `F2_` 前缀便于阅读, 但真正用于查 B1.2 统计量的必须是规范 id ——
    否则 `candidate_statistic()` 会查不到而静默走错分支。所以这里分开断言:
      * config label 以规范 id 结尾 (允许 F2_ 前缀)
      * probe 实际使用的是规范 id
    """
    T = 0.14311462970213382
    CANON = "FRICTION_TORQUE_P95"
    assert float(b13_cfg["failure_definition"]["threshold_Nm"]) == T
    assert b13_cfg["failure_definition"]["name"].endswith(CANON), \
        f"config label {b13_cfg['failure_definition']['name']!r} 不含规范 id {CANON}"
    if scaling_rec is not None:
        assert scaling_rec["f2_torque_domain"]["T_f_fail_Nm"] == T
        assert scaling_rec["mapping"]["f2_threshold_touched"] is False
    if probe_rec is not None:
        assert float(probe_rec["failure_threshold_Nm"]) == T
        assert probe_rec["failure_definition"] == CANON, \
            "probe 用的不是 B1.2 规范 candidate id —— 统计量会查错"


def test_probe_generated_once(b13_cfg, probe_rec):
    """§10: probe 只允许生成一次; runs_allowed = 1, 且脚本有覆盖保护。"""
    assert int(b13_cfg["probe"]["runs_allowed"]) == 1
    src = (ROOT / "scripts/basilisk_b13/generate_probe.py").read_text(
        encoding="utf-8")
    assert "allow-overwrite" in src or "allow_overwrite" in src, \
        "generate_probe.py 缺少重复运行保护 (§10)"
    if probe_rec is not None:
        assert int(probe_rec["n_traj"]) == int(b13_cfg["probe"]["n_traj"]) == 60


def test_only_degradation_scaling_changed(probe_rec, b13_cfg):
    """§9: 相对 B1.2 只允许退化尺度变化, 其余全部逐值不变。"""
    if probe_rec is None:
        pytest.skip("需先跑 generate_probe.py")
    assert probe_rec["only_degradation_scaling_changed"] is True
    assert probe_rec["changed_params"] == ["b0 (via b0_scale)"]
    for must in ("g_duty", "mode_weights", "Im_rated", "F2 threshold",
                 "Delta_range", "tau_years_range"):
        assert any(must in u for u in probe_rec["unchanged"]), \
            f"未声明 {must} 保持不变"
    # wear_drive 权重逐值比对 B1.2 (禁止改)
    import yaml
    b12 = yaml.safe_load(
        (ROOT / "configs/wheel_basilisk_b12.yaml").read_text(encoding="utf-8"))
    if "wear_drive" in b12:
        assert b13_cfg["wear_drive"] == b12["wear_drive"], \
            "wear_drive 权重与 B1.2 不一致 —— §禁止修改"
