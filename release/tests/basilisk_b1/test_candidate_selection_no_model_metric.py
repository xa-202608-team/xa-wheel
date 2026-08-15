"""tests/basilisk_b1/test_candidate_selection_no_model_metric.py

B1 §5 硬测试 (二) —— candidate 选择绝不允许触碰任何神经网络 / 模型指标。

与 test_candidate_selection_no_rul 互补:
  * 前者盯"RUL 指标名"
  * 本文件盯"模型这条路本身": import、checkpoint 读取、以及 gate 定义里
    只允许出现物理量白名单

另外钉死 §6 的 candidate 集合冻结 (只允许 A/B/C) 与 §8 的优先顺序规则。
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "basilisk_b1"))

import calibrate_degradation as cal                      # noqa: E402

CFG = cal.load_b1_config("configs/wheel_basilisk_b1.yaml")
B1_SCRIPTS = ("scripts/basilisk_b1/calibrate_degradation.py",
              "scripts/basilisk_b1/audit_physics.py",
              "scripts/basilisk_b1/generate_candidates.py",
              "scripts/basilisk_b1/audit_candidates.py",
              "scripts/basilisk_b1/freeze_candidate.py",
              "scripts/basilisk_b1/generate_dataset.py",
              "scripts/basilisk_b1/build_features.py")

FORBIDDEN_MODULES = ("torch", "src.models", "src.transfer", "src.experiments",
                     "src.baselines", "sklearn")
# §8 gate 只允许这些物理量作为判据 (与 config gate 段一一对应)
ALLOWED_GATE_KEYS = {
    "failure_fraction_min", "failure_fraction_max", "censored_fraction_min",
    "min_event_observed", "eol_at_horizon_end_frac_max", "eol_iqr_min",
    "b_monotone_frac_min", "temperature_min_K", "temperature_max_K",
    "current_absmax_A", "wear_ordering_min_spearman",
}
FROZEN_REGISTRY = ("UTILIZATION_ONLY", "UTILIZATION_PLUS_DURATION",
                   "UTILIZATION_PLUS_DURATION_LONG")


def _imports(rel: str) -> set[str]:
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


@pytest.mark.parametrize("rel", B1_SCRIPTS)
def test_candidate_selection_no_model_metric(rel):
    """B1 全链脚本都不得 import 任何模型 / 迁移 / 实验编排模块。

    这是比"不看指标"更强的保证: 连模型代码都进不来, 就不可能拿模型结果做选择。
    §14 同时要求 B1 阶段完全不训练模型, 本测试是它的静态守卫。
    """
    mods = _imports(rel)
    bad = [m for m in mods
           if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_MODULES)]
    assert not bad, f"{rel} import 了模型相关模块 {bad} (违反 §5/§14)"


@pytest.mark.parametrize("rel", B1_SCRIPTS)
def test_b1_scripts_never_load_checkpoints(rel):
    """不得从 checkpoints 加载任何模型权重 (torch.load / state_dict / .pt)。"""
    src = (ROOT / rel).read_text(encoding="utf-8")
    for token in ("torch.load", "state_dict", "load_state_dict",
                  ".pt\"", ".pt'", ".pth"):
        assert token not in src, f"{rel} 出现模型权重加载痕迹 {token!r}"


def test_gate_config_contains_only_physical_criteria():
    """config 的 gate 段只能含物理判据 —— 多出任何一个键都要人工复核。"""
    keys = set(CFG["gate"].keys())
    extra = keys - ALLOWED_GATE_KEYS
    assert not extra, f"gate 段出现非物理判据 {extra} (§5 只允许物理量)"
    missing = ALLOWED_GATE_KEYS - keys
    assert not missing, f"gate 段缺失判据 {missing} (§8 要求 10 条全查)"


def test_audit_gate_names_are_physical():
    """审计产物里逐条 gate 的名字不得含任何模型指标词。"""
    p = ROOT / "checkpoints/basilisk_b1/candidate_audit.json"
    if not p.exists():
        pytest.skip("candidate_audit.json 未生成")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["used_model_metric"] is False
    banned = ("rmse", "mae", "corr", "phm", "lead", "gain", "loss", "epoch")
    for cname, c in d["per_candidate"].items():
        assert len(c["gates"]) == 10, f"{cname} gate 条数 != 10"
        for g in c["gates"]:
            low = g["name"].lower()
            assert not any(b in low for b in banned), \
                f"{cname} gate 名含模型指标词: {g['name']}"


# --------------------------------------------------------------------------
# §6 candidate 集合冻结
# --------------------------------------------------------------------------
def test_candidate_set_frozen():
    """§6: registry 必须精确等于预先登记的三个, 且 spec 键集一致 —— 禁止 D/E。"""
    reg = tuple(CFG["candidates"]["registry"])
    assert reg == FROZEN_REGISTRY, f"candidate registry 被改动: {reg}"
    assert set(CFG["candidates"]["spec"].keys()) == set(FROZEN_REGISTRY)
    assert len(reg) == 3, "candidate 数量必须为 3"


def test_candidates_differ_only_in_horizon():
    """§6: A/B/C 退化尺度必须完全相同, 唯一差异是 horizon_scale。"""
    spec = CFG["candidates"]["spec"]
    scales = {c: float(spec[c]["horizon_scale"]) for c in FROZEN_REGISTRY}
    assert scales == {"UTILIZATION_ONLY": 1.0,
                      "UTILIZATION_PLUS_DURATION": 1.5,
                      "UTILIZATION_PLUS_DURATION_LONG": 2.0}, scales
    # 除 horizon_scale 外的键值必须逐一相同
    for c in FROZEN_REGISTRY:
        assert spec[c]["use_utilization_wear"] is True
        extra = set(spec[c].keys()) - {"horizon_scale", "use_utilization_wear"}
        assert not extra, f"{c} 含额外退化尺度参数 {extra} -> 三者不再同尺度"


def test_generate_candidates_hardcodes_frozen_registry():
    """生成脚本必须自带冻结 registry 常量并与 config 比对 (双保险)。"""
    src = (ROOT / "scripts/basilisk_b1/generate_candidates.py").read_text(
        encoding="utf-8")
    for name in FROZEN_REGISTRY:
        assert name in src, f"生成脚本未硬编码 {name}"
    assert "FROZEN_REGISTRY" in src and "!= FROZEN_REGISTRY" in src, \
        "生成脚本未校验 registry 是否被改动"


# --------------------------------------------------------------------------
# §8 优先顺序规则
# --------------------------------------------------------------------------
def test_candidate_priority_rule():
    """§8: 优先顺序必须是 A > B > C (尽量不靠延长任务期)。"""
    pr = list(CFG["candidates"]["priority"])
    assert pr == list(FROZEN_REGISTRY), f"优先顺序被改动: {pr}"
    # horizon 越短优先级越高 —— 这正是"不靠延长任务期"的形式化表达
    spec = CFG["candidates"]["spec"]
    hs = [float(spec[c]["horizon_scale"]) for c in pr]
    assert hs == sorted(hs), f"优先顺序未按 horizon 递增: {hs}"


def test_selection_takes_first_eligible_in_priority():
    """审计产物的 selected 必须是 priority 中第一个 eligible 者 (可复核)。"""
    p = ROOT / "checkpoints/basilisk_b1/candidate_audit.json"
    if not p.exists():
        pytest.skip("candidate_audit.json 未生成")
    d = json.loads(p.read_text(encoding="utf-8"))
    pr = d["priority"]
    expect = next((c for c in pr if d["per_candidate"][c]["eligible"]), None)
    assert d["selected_candidate"] == expect, \
        f"selected={d['selected_candidate']} 不是 priority 首个 eligible={expect}"
    if expect is None:
        assert d["verdict"] == "B1_CALIBRATION_FAIL"


def test_failure_fraction_band_is_35_to_80():
    """§9: 预先登记带宽必须是 35%-80%, 不得为了靠近 65% 被改窄。"""
    assert float(CFG["gate"]["failure_fraction_min"]) == 0.35
    assert float(CFG["gate"]["failure_fraction_max"]) == 0.80
    assert float(CFG["gate"]["censored_fraction_min"]) == 0.15
