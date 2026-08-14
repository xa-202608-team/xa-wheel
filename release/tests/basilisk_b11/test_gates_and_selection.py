"""tests/basilisk_b11/test_gates_and_selection.py

B1.1 §18 —— Gate 与选择组硬测试:
  * `test_early_eol_gate`                    : Gate 11 实现正确 (含边界)
  * `test_initial_margin_gate`               : Gate 12 用逐轨迹 p95 再取轨迹间 p95,
                                               **不用单个极端时间点**
  * `test_candidate_selection_no_rul`        : 选择链不含任何 RUL 依赖
  * `test_candidate_selection_no_model_metric`: 不读任何模型指标产物
  * `test_selection_priority_is_pre_registered`
  * `test_fail_path_refuses_to_freeze`       : §11 全失败必须拒绝冻结
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CAND_JSON = ROOT / "checkpoints/basilisk_b11/candidates.json"
AUDIT_JSON = ROOT / "checkpoints/basilisk_b11/candidate_audit.json"
FROZEN_REGISTRY = ("P95_UTILIZATION_ONLY", "P95_UTILIZATION_DURATION_1P5",
                   "P95_UTILIZATION_DURATION_2P0")

SELECTION_SCRIPTS = ("scripts/basilisk_b11/generate_candidates.py",
                     "scripts/basilisk_b11/audit_candidates.py",
                     "scripts/basilisk_b11/freeze_candidate.py")
RUL_TOKENS = {
    "rul_rmse", "rmse_rul", "rul_mae", "mae_rul", "nphm", "phm", "ph",
    "warning_lead", "lead_time", "transfer_gain", "rul_corr",
    "correlation_rul", "rul_pred", "pred_rul", "rul_hat", "rul_error",
    "test_rmse", "val_rmse", "rul_metric",
}
FORBIDDEN_MODULES = ("torch", "src.models", "src.transfer", "src.experiments",
                     "src.baselines", "sklearn.metrics")
FORBIDDEN_METRIC_FILES = ("s3_gate_metrics.json", "s4_metrics.json",
                          "s5_rate_metrics.json", "s5b_wiener_pf_metrics.json",
                          "results.md", "transfer_metrics.json")


def _tree(rel: str):
    p = ROOT / rel
    assert p.exists(), f"缺 {rel}"
    src = p.read_text(encoding="utf-8")
    return ast.parse(src), src


def _iter_strings(tree):
    """遍历字符串常量, **跳过 docstring** (说明性文字不算使用)。"""
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in doc:
            yield node, node.value


def _audit() -> dict:
    if not AUDIT_JSON.exists():
        pytest.skip("candidate_audit.json 未生成")
    return json.loads(AUDIT_JSON.read_text(encoding="utf-8"))


def _cand() -> dict:
    if not CAND_JSON.exists():
        pytest.skip("candidates.json 未生成")
    return json.loads(CAND_JSON.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Gate 11 —— early-EOL
# --------------------------------------------------------------------------

def test_early_eol_gate(b11_cfg):
    """Gate 11 的判据与阈值必须走 config, 且 early 只能来自 event-observed。

    删失轨迹的 eol_idx = n-1 是**约定**而非观测; 若把它算进 early-EOL, 数字会
    在长 horizon 上莫名变好 —— 方向刚好相反的错误。
    """
    ef = float(b11_cfg["gate"]["early_eol_horizon_frac"])
    mx = float(b11_cfg["gate"]["early_eol_frac_max"])
    assert ef == 0.10 and mx == 0.30, \
        f"Gate 11 阈值被改动: frac={ef}, max={mx} (§6 预先登记 0.10 / 0.30)"
    d = _cand()
    for name in FROZEN_REGISTRY:
        c = d["candidates"][name]
        n = c["n_traj"]
        nw = c["n_windows"]
        rows = c["per_traj"]
        # 重算 early 标记, 与脚本记录逐条比对
        recomputed = sum(1 for r in rows
                         if r["failed"] and int(r["eol_idx"]) < ef * nw)
        assert recomputed == c["n_early_eol"], \
            f"{name} early-EOL 计数不可复算: {c['n_early_eol']} vs {recomputed}"
        assert abs(c["early_eol_fraction"] - recomputed / n) < 1e-12
        # 删失轨迹绝不能被标 early
        for r in rows:
            if not r["failed"]:
                assert r["early_eol"] is False, \
                    "删失轨迹被标成 early-EOL —— 右删失不得伪造 EOL"


def test_early_eol_gate_boundary(b11_cfg):
    """Gate 11 是严格小于 (`< max`), 边界值必须判 FAIL。"""
    src = (ROOT / "scripts/basilisk_b11/audit_candidates.py").read_text(
        encoding="utf-8")
    assert "early_eol_frac_max" in src, "Gate 11 未读取 config 阈值"
    # A11 实测 0.3333 > 0.30 必须 FAIL —— 这正是本阶段的真实结论
    a = _audit()["per_candidate"][FROZEN_REGISTRY[0]]
    g11 = [g for g in a["gates"] if g["name"].startswith("11")][0]
    assert g11["pass"] is False, "A11 的 early-EOL fraction 0.3333 应判 FAIL"


# --------------------------------------------------------------------------
# Gate 12 —— initial current margin
# --------------------------------------------------------------------------

def test_initial_margin_gate(b11_cfg, b11_cal):
    """Gate 12: 逐轨迹取初始健康窗的 p95, 再对轨迹级比值取 p95。

    §6 明令"不要使用单个极端时间点"。这里用合成信号验证:
    单点尖峰不应显著抬高 p95, 但整体抬升必须被抓到。
    """
    cal = b11_cal
    frac = float(b11_cfg["gate"]["initial_margin_healthy_frac"])
    im_rated = float(b11_cfg["sim"]["failure"]["Im_rated_A"])
    assert float(b11_cfg["gate"]["initial_margin_ratio_max"]) == 0.50, \
        "Gate 12 阈值被改动 (§6 预先登记 0.50)"
    assert frac == float(b11_cfg["sim"]["hi"]["healthy_frac"]), \
        "initial_margin_healthy_frac 必须与 sim.hi.healthy_frac 一致 —— " \
        "否则'初始健康段'在两处含义不同"

    n = 20000
    base = np.full(n, 0.30 * im_rated)
    # (a) 单点尖峰: 只有 1 个采样点跳到 10x, p95 几乎不动
    spike = base.copy()
    spike[5] = 10.0 * im_rated
    df_spike = pd.DataFrame({"I_m": spike})
    r_spike = cal.initial_margin_ratio(df_spike, b11_cfg)
    assert abs(r_spike - 0.30) < 0.02, \
        f"单个极端时间点显著影响了 Gate 12 ({r_spike:.4f}) —— 应为 p95 而非 max"
    # (b) 整体抬升: 必须被抓到
    df_hi = pd.DataFrame({"I_m": np.full(n, 0.80 * im_rated)})
    assert cal.initial_margin_ratio(df_hi, b11_cfg) > 0.75
    # (c) 只看初始健康段: 后段的高电流不得影响早期裕度
    late = base.copy()
    late[int(n * frac) + 10:] = 5.0 * im_rated
    df_late = pd.DataFrame({"I_m": late})
    assert abs(cal.initial_margin_ratio(df_late, b11_cfg) - 0.30) < 0.02, \
        "Gate 12 读到了健康段之外的电流"


def test_initial_margin_gate_uses_p95_of_trajectory_ratios():
    """汇总口径: 轨迹间必须取 p95, 而不是 max 或 mean。"""
    d = _cand()
    for name in FROZEN_REGISTRY:
        c = d["candidates"][name]
        vals = np.array([r["initial_margin_ratio"] for r in c["per_traj"]],
                        dtype=float)
        want = float(np.percentile(vals, 95))
        assert abs(c["initial_margin_ratio_p95"] - want) < 1e-9, \
            f"{name} 的汇总不是轨迹间 p95"
        assert c["initial_margin_ratio_p95"] <= c["initial_margin_ratio_max"] + 1e-12
        assert c["initial_margin_ratio_median"] <= c["initial_margin_ratio_p95"] + 1e-12


def test_initial_margin_gate_actually_fires():
    """本阶段的真实结论: 三个 candidate 的 Gate 12 全部 FAIL。

    这条测试把"发现"钉成回归基线 —— 如果有人事后调松阈值让它通过, 会失败。
    """
    a = _audit()
    for name in FROZEN_REGISTRY:
        g = [x for x in a["per_candidate"][name]["gates"]
             if x["name"].startswith("12")][0]
        assert g["pass"] is False, \
            f"{name} 的 Gate 12 变成 PASS —— 阈值或口径疑似被放宽"


# --------------------------------------------------------------------------
# 选择过程不得触碰 RUL / 模型指标
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rel", SELECTION_SCRIPTS)
def test_candidate_selection_no_rul(rel):
    """§9(10)/§17: 选择链脚本中不得出现任何 RUL 指标的标识符 / 属性 / 键名。"""
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
        if s.strip().lower() in RUL_TOKENS:
            bad.append(f"L{node.lineno}: 字符串键 {s!r}")
    assert not bad, f"{rel} 使用了 RUL 指标 (违反 §9/§17):\n" + "\n".join(bad)


@pytest.mark.parametrize("rel", SELECTION_SCRIPTS)
def test_candidate_selection_no_model_metric(rel):
    """选择链不得 import 模型 / 迁移模块, 也不得读取任何模型指标产物。"""
    tree, src = _tree(rel)
    bad_imp = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                if any(al.name == m or al.name.startswith(m + ".")
                       for m in FORBIDDEN_MODULES):
                    bad_imp.append(f"L{node.lineno}: import {al.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if any(node.module == m or node.module.startswith(m + ".")
                   for m in FORBIDDEN_MODULES):
                bad_imp.append(f"L{node.lineno}: from {node.module}")
    assert not bad_imp, f"{rel} 引入了模型/迁移模块:\n" + "\n".join(bad_imp)
    bad_files = [f for f in FORBIDDEN_METRIC_FILES if f in src]
    assert not bad_files, f"{rel} 引用了模型指标文件 {bad_files}"


def test_audit_records_no_rul_flags():
    """审计产物必须留下可机读的 used_rul_metric=False 记录。"""
    a = _audit()
    assert a["used_rul_metric"] is False
    assert a["used_model_metric"] is False
    assert a["selected_before_any_RUL_training"] is True
    c = _cand()
    assert c["used_rul_metric"] is False and c["used_model_metric"] is False


def test_selection_priority_is_pre_registered():
    """§10: 选择规则必须是预先登记的优先级, 不得依据 ff 距 65% 的远近。"""
    a = _audit()
    assert tuple(a["priority"]) == FROZEN_REGISTRY, f"优先级被改: {a['priority']}"
    rule = str(a["selection_rule"])
    assert "优先" in rule or "priority" in rule.lower()
    for bad in ("65", "RMSE", "rmse", "最接近"):
        assert bad not in rule, f"选择规则出现 {bad!r} —— §10 禁止此类依据"


def test_selection_ignores_failure_fraction_distance():
    """§10 反向证据: 审计脚本不得计算"到某个目标 ff 的距离"。"""
    src = (ROOT / "scripts/basilisk_b11/audit_candidates.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            if abs(node.value - 0.65) < 1e-12:
                bad.append(f"L{node.lineno}: 出现 0.65 (疑似 ff 目标值)")
    assert not bad, "\n".join(bad)


# --------------------------------------------------------------------------
# §11 失败路径
# --------------------------------------------------------------------------

def test_fail_path_refuses_to_freeze():
    """§11: 三个 candidate 全失败时必须判 FAIL 且**不得**生成冻结标定。"""
    a = _audit()
    if a["verdict"] != "B11_CALIBRATION_FAIL":
        pytest.skip("本次审计有 eligible candidate, 失败路径不适用")
    assert a["selected_candidate"] is None
    for name in FROZEN_REGISTRY:
        assert a["per_candidate"][name]["eligible"] is False
    frozen = ROOT / "checkpoints/basilisk_b11/frozen_calibration.json"
    assert not frozen.exists(), \
        "B11_CALIBRATION_FAIL 却生成了 frozen_calibration.json (§11 违规)"
    for p in ("docs/basilisk_b11/selected_calibration.md",
              "data/features/wheel/basilisk_b11/target_features.h5"):
        assert not (ROOT / p).exists(), f"FAIL 路径却产出了 {p}"


def test_no_official_dataset_on_fail(b11_cfg):
    """§13: FAIL 时不得生成正式 150 条数据集。"""
    a = _audit()
    if a["verdict"] != "B11_CALIBRATION_FAIL":
        pytest.skip("非失败路径")
    sim_dir = ROOT / b11_cfg["paths"]["sim_dir"]
    if not sim_dir.exists():
        return
    stray = [p.relative_to(ROOT).as_posix() for p in sim_dir.glob("seed_*")]
    assert not stray, f"FAIL 却生成了正式数据集目录: {stray}"


def test_gate_count_is_twelve():
    """§9: gate 必须恰好 12 条 (旧 10 + 新 2), 不得增删。"""
    import importlib.util
    p = ROOT / "scripts/basilisk_b11/audit_candidates.py"
    spec = importlib.util.spec_from_file_location("b11_ac_gatecount", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["b11_ac_gatecount"] = mod
    spec.loader.exec_module(mod)
    assert len(mod.GATE_NAMES) == 12, f"gate 数 {len(mod.GATE_NAMES)} != 12"
    a = _audit()
    for name in FROZEN_REGISTRY:
        assert len(a["per_candidate"][name]["gates"]) == 12


def test_gate_thresholds_from_config_not_hardcoded(b11_cfg):
    """跨阶段纪律: gate 阈值一律走 config。"""
    g = b11_cfg["gate"]
    for k in ("failure_fraction_min", "failure_fraction_max",
              "censored_fraction_min", "min_event_observed",
              "eol_at_horizon_end_frac_max", "eol_iqr_min",
              "early_eol_horizon_frac", "early_eol_frac_max",
              "initial_margin_healthy_frac", "initial_margin_ratio_max"):
        assert k in g, f"config 缺 gate.{k}"
    src = (ROOT / "scripts/basilisk_b11/audit_candidates.py").read_text(
        encoding="utf-8")
    for k in ("early_eol_frac_max", "initial_margin_ratio_max",
              "failure_fraction_min", "censored_fraction_min"):
        assert k in src, f"审计脚本未从 config 读取 {k}"


def test_empty_bin_returns_nan_not_zero():
    """跨阶段纪律: 无 event-observed 时 EOL 分位数必须是 None/NaN, 不伪造 0。"""
    d = _cand()
    for name in FROZEN_REGISTRY:
        c = d["candidates"][name]
        if c["n_event_observed"] == 0:
            assert c["eol_quantiles_observed"] is None
        else:
            assert c["eol_quantiles_observed"] is not None
    src = (ROOT / "scripts/basilisk_b11/audit_candidates.py").read_text(
        encoding="utf-8")
    assert "不伪造" in src, "审计脚本未声明空 bin 不伪造 0 的口径"
