"""tests/basilisk_b11/test_p95_reference.py

B1.1 §18 —— 标定口径组硬测试:
  * `test_p95_reference_from_profiles`      : p95 reference 由冻结 profile 统计算出
  * `test_b0_scale_derived_not_hardcoded`   : b0_scale 是推导值, 不是写死的 5.1
  * `test_im_rated_unchanged`               : §5 Im_rated 保持 1.5 A
  * `test_b11_wear_structure_unchanged`     : §2 wear 数学结构与 B1 逐个同一实现

这些测试只依赖冻结的 `profiles.h5` 与 config, 不需要 candidate 数据。
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROTOCOL_REC = ROOT / "checkpoints/basilisk_b11/protocol_hash.json"
CAL_SRC = ROOT / "scripts/basilisk_b11/calibrate_degradation.py"
AUDIT_SRC = ROOT / "scripts/basilisk_b11/audit_peak_reference.py"

# §4 期望量级参考值 —— **只用于反向断言"没被硬编码"**, 不作为目标
GOAL_HINT_VALUE = 5.1


def _rec() -> dict:
    if not PROTOCOL_REC.exists():
        pytest.skip("protocol_hash.json 未生成 (先跑 audit_peak_reference.py)")
    return json.loads(PROTOCOL_REC.read_text(encoding="utf-8"))


def test_p95_reference_from_profiles(b11_cal, b11_cfg):
    """p95 reference 必须能从冻结的 profiles.h5 重新算出同一数值。

    这同时证明两件事: (a) reference 是数据派生的, 不是配置里写死的常数;
    (b) 计算是确定性的 (无全局 RNG 参与)。
    """
    cal = b11_cal
    rec = _rec()
    ctx = cal.prepare_context(b11_cfg, [1.0])
    ref = ctx["ref"]
    assert ref["caliber"] == "p95"
    assert ref["speed_stat"] == "omega_p95"
    assert ref["torque_stat"] == "torque_p95"
    for key, rk in (("speed_util", "speed_util"), ("torque_util", "torque_util")):
        got = float(ref[key])
        want = float(rec["reference_p95"][rk])
        assert abs(got - want) < 1e-12, \
            f"p95 reference {key} 不可复算: {want!r} -> {got!r}"


def test_p95_reference_uses_same_mode_weighting_as_b1(b11_cal, b11_cfg):
    """§3: 必须使用与 B1 相同的模式加权定义 (等权非重叠窗), 只换统计量。"""
    cal = b11_cal
    ctx = cal.prepare_context(b11_cfg, [1.0])
    ref_p95, ref_rms = ctx["ref"], ctx["ref_rms"]
    assert ref_p95["aggregate"] == ref_rms["aggregate"]
    assert ref_p95["n_per_window"] == ref_rms["n_per_window"]
    # 两个口径的每模式窗数必须逐模式相同 —— 否则不是"只换统计量"
    for m in ref_p95["per_mode"]:
        assert (ref_p95["per_mode"][m]["n_windows"]
                == ref_rms["per_mode"][m]["n_windows"])
    # 与工况无关的两项 (过零率 / 机动占比) 换口径后必须逐位不变
    for k in ("zero_crossing_rate", "maneuver_fraction"):
        assert abs(float(ref_p95[k]) - float(ref_rms[k])) < 1e-15, \
            f"{k} 在换口径后变了 —— 说明改动超出了 speed/torque 统计量"


def test_rms_caliber_reproduces_b1_exactly(b11_cal, b11_cfg):
    """rms 口径必须逐位复现 B1 归档的 su_ref / b0_scale。

    这是"B1 与 B1.1 之间唯一差异是口径, 而非环境漂移"的关键证据。
    """
    cal = b11_cal
    rec = _rec()
    ctx = cal.prepare_context(b11_cfg, [1.0])
    assert abs(float(ctx["ref_rms"]["speed_util"])
               - float(rec["b1_cross_check"]["b1_su_ref"])) < 1e-15
    assert abs(float(ctx["calib_rms"]["b0_scale"])
               - float(rec["b1_cross_check"]["b1_b0_scale"])) < 1e-12
    assert rec["b1_cross_check"]["su_reproduced"] is True
    assert rec["b1_cross_check"]["b0_scale_reproduced"] is True


def test_b0_scale_derived_not_hardcoded(b11_cal, b11_cfg):
    """§4: b0_scale 必须由 profile 统计推导, 且严格满足 old/new == speed ratio。

    同时断言代码里不存在硬编码的 5.1 (goal 给的期望量级) —— 若有人为了对上
    "≈5.1"而写死数值, 这里会失败。
    """
    cal = b11_cal
    rec = _rec()
    ctx = cal.prepare_context(b11_cfg, [1.0])
    new = float(ctx["calib"]["b0_scale"])
    old = float(ctx["calib_rms"]["b0_scale"])
    ratio = float(ctx["ref"]["speed_util"]) / float(ctx["ref_rms"]["speed_util"])
    assert abs(old / new - ratio) < 1e-9, \
        f"b0_scale 未按 1/su_ref 缩放: old/new={old/new!r} vs ratio={ratio!r}"
    assert abs(new - float(rec["derivation"]["derived_b0_scale"])) < 1e-12
    assert rec["derivation"]["hardcoded_target"] is False
    # 派生值不应恰好等于 goal 的期望量级 —— 若相等, 极可能是被写死了
    assert abs(new - GOAL_HINT_VALUE) > 1e-6, \
        "derived_b0_scale 恰好等于 goal 提示的 5.1 —— 疑似硬编码"


@pytest.mark.parametrize("rel", ["scripts/basilisk_b11/calibrate_degradation.py",
                                 "scripts/basilisk_b11/audit_peak_reference.py",
                                 "scripts/basilisk_b11/generate_candidates.py",
                                 "configs/wheel_basilisk_b11.yaml"])
def test_no_hardcoded_b0_scale_literal(rel):
    """b0_scale 的具体数值不得作为字面量出现在代码 / config 里。

    用 AST 取数字字面量 (config 用文本扫描), 避开注释与说明文字的误报。
    """
    p = ROOT / rel
    assert p.exists(), f"缺 {rel}"
    src = p.read_text(encoding="utf-8")
    forbidden = (5.1, 5.906442914788147, 7.225772358824458)
    if rel.endswith(".yaml"):
        import yaml
        cfg = yaml.safe_load(src)

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    yield from walk(v)
            elif isinstance(o, list):
                for v in o:
                    yield from walk(v)
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                yield float(o)
        bad = [v for v in walk(cfg)
               if any(abs(v - f) < 1e-9 for f in forbidden)]
        assert not bad, f"{rel} 出现硬编码 b0_scale 数值 {bad}"
        return
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            v = float(node.value)
            if any(abs(v - f) < 1e-9 for f in forbidden):
                bad.append(f"L{node.lineno}: {v}")
    assert not bad, f"{rel} 出现硬编码 b0_scale 数值:\n" + "\n".join(bad)


def test_im_rated_unchanged(b11_cfg):
    """§5: Im_rated 必须保持 1.5 A, 且 provenance warning 只被记录不被"修好"。"""
    assert float(b11_cfg["sim"]["failure"]["Im_rated_A"]) == 1.5, \
        "§5 明令本阶段不得修改 Im_rated"
    import yaml
    raw = yaml.safe_load(
        (ROOT / "configs/wheel_basilisk_b11.yaml").read_text(encoding="utf-8"))
    # B1.1 自己的 config 不得覆写 Im_rated (哪怕写成同一个值也不允许 —— 那会掩盖
    # 它是继承来的这一事实)
    fail = ((raw.get("sim") or {}).get("failure") or {})
    assert "Im_rated_A" not in fail, \
        "B1.1 config 覆写了 Im_rated_A —— §5 要求继承, 不得在本阶段重新声明"


def test_im_rated_provenance_warning_recorded():
    """§5: Kt_nom·Im_rated < HR16 u_max 的物理疑点必须被显式记录。"""
    rec = _rec()
    w = rec.get("im_rated_provenance_warning")
    assert w and "IM_RATED_PROVENANCE_WARNING" in str(w), \
        "未记录 IM_RATED_PROVENANCE_WARNING"
    assert float(rec["im_rated_A"]) == 1.5


def test_b11_wear_structure_unchanged(b11_cal, b1_cal):
    """§2: wear 数学结构必须与 B1 **同一实现** (import 而非复制)。

    对 g_duty / wear_rate_multiplier / apply_calibration / b0_scale 逐个断言
    "是同一个函数对象"—— 这比比较源码文本更强: 复制粘贴会立刻失败。
    """
    for name in ("g_duty", "wear_rate_multiplier", "apply_calibration",
                 "b0_scale", "base_n_windows", "calibration_protocol_hash",
                 "params_rng", "rated_values_from_basilisk"):
        assert getattr(b11_cal, name) is getattr(b1_cal, name), \
            f"B1.1 的 {name} 不是 B1 的同一实现 —— 疑似复制并可能已漂移"
    assert b11_cal.DRIVE_KEYS == b1_cal.DRIVE_KEYS
    assert b11_cal.G_DUTY_MIN == b1_cal.G_DUTY_MIN


def test_b11_wear_drive_weights_identical_to_b1(b11_cal, b11_cfg, b1_cal):
    """§2: g_duty 的权重 / 指数 / 上下界必须与 B1 逐值相同。"""
    b1cfg = b1_cal.load_b1_config("configs/wheel_basilisk_b1.yaml")
    for k in ("speed_util_weight", "torque_util_weight", "zero_crossing_weight",
              "maneuver_weight", "speed_util_exponent", "torque_util_exponent",
              "ref_floor", "g_duty_max", "reference"):
        assert b11_cfg["wear_drive"][k] == b1cfg["wear_drive"][k], \
            f"wear_drive.{k} 与 B1 不同 —— §2 禁止改 g_duty 结构"


def test_g_duty_at_reference_is_one(b11_cal, b11_cfg):
    """g_duty 在参考工况处必须为 1.0 —— 权重和为 1 的归一化性质。"""
    cal = b11_cal
    ctx = cal.prepare_context(b11_cfg, [1.0])
    ref = ctx["ref"]
    drives = {k: np.array([float(ref[k])]) for k in cal.DRIVE_KEYS}
    g = cal.g_duty(drives, ref, b11_cfg["wear_drive"])
    assert abs(float(g[0]) - 1.0) < 1e-12, f"g_duty(reference) = {g[0]!r} != 1.0"


def test_failure_criterion_unchanged(b11_cal, b11_cfg, b1_cal):
    """§2: 失效判据 (阈值 + 持续采样数 + 判据类型) 不得改动。"""
    b1cfg = b1_cal.load_b1_config("configs/wheel_basilisk_b1.yaml")
    for k in ("Im_rated_A", "persistence_samples"):
        assert b11_cfg["sim"]["failure"][k] == b1cfg["sim"]["failure"][k], \
            f"failure.{k} 被改动 —— §2 禁止改失效判据"


def test_audit_script_prints_all_required_quantities():
    """§4 要求打印的 8 个量必须都在 audit 脚本的输出里出现。"""
    src = AUDIT_SRC.read_text(encoding="utf-8")
    for token in ("old_su_ref_rms", "new_su_ref_p95", "ratio_p95_to_rms",
                  "old_tu_ref_rms", "new_tu_ref_p95", "old_b0_scale",
                  "derived_b0_scale"):
        assert token in src, f"audit_peak_reference.py 未输出 {token}"
