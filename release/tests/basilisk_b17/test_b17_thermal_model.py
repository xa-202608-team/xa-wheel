"""tests/basilisk_b17/test_b17_thermal_model.py

§9 / §10 / §11 / §12 / §14 / §17 —— 热模型物理性、稳定性、功率映射 provenance、
以及"不得生成 RUL / 失效"边界。

关于 `test_thermal_model_energy_balance` / `test_thermal_model_stable`:
本阶段 Gate 未通过, 热模型**刻意不存在**。这两条测试因此写成"条件式契约":
  * 若模型存在 -> 必须满足能量平衡与稳定性;
  * 若模型不存在 -> 必须存在拒绝记录, 且拒绝理由是 provenance 不足,
    而不是"忘了跑"。
把它们写成无条件 skip 会让"模型缺席"这件事失去检验。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CK = ROOT / "checkpoints" / "basilisk_b17"

USABLE = ("DIRECT", "DERIVED_FROM_DIRECT", "LITERATURE_ANALOG")


def _model():
    p = CK / "thermal_model.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _refusal():
    p = CK / "thermal_model_refusal.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def test_power_mapping_has_provenance(b17_power, b17_registry):
    """§17 / §9 —— 所用功率映射必须有依据; 未用的映射必须说明原因。"""
    assert b17_power["mapping_used"] == \
        "mechanical_power_abs_torque_times_omega"
    # 机械功率只需已冻结遥测, 不需任何器件热参数 —— 这是它可用的理由
    assert "motor_torque_Nm" in b17_power["mapping_formula"]
    assert "wheel_speed_rad_s" in b17_power["mapping_formula"]
    assert b17_power["fabricated_parameters"] is False
    assert b17_power["profile_sha256"] not in ("", "MISSING")

    pr = b17_registry["parameter_registry"]
    # 未使用的两种映射: 其所需参数确实无来源, 且已逐条说明
    nu = b17_power["mappings_not_used"]
    assert "efficiency_derived_loss" in nu and "copper_loss_I2R" in nu
    for k, why in nu.items():
        assert why, f"{k} 未说明不使用的原因"
    if pr["torque_constant_Kt"]["provenance"] not in USABLE:
        assert "Kt" in nu["copper_loss_I2R"], "未说明缺 Kt"
    if pr["motor_winding_resistance"]["provenance"] not in USABLE:
        assert "R=" in nu["copper_loss_I2R"] or "R" in nu["copper_loss_I2R"]


def test_power_mapping_ordering_disclosed(b17_power):
    """§9 与 §18 的顺序不一致必须显式披露, 而不是悄悄按方便的那个执行。"""
    assert b17_power["ordering_disclosure"], "顺序冲突未披露"
    assert "§18" in b17_power["ordering_disclosure"]
    # Gate 未通过时运行本审计, 结果只能是诊断用途
    if not b17_power["provenance_gate_passed_at_run_time"]:
        assert b17_power["diagnostic_only"] is True
        assert b17_power["feeds_thermal_model"] is False


def test_power_stats_no_fake_zeros(b17_power):
    """空/非有限数据必须给 NaN + n=0, 不得伪造 0 (项目既有纪律)。"""
    for mode, v in b17_power["per_mode"].items():
        if not v.get("present"):
            continue
        s = v["hottest_wheel_power"]
        assert s["n"] > 0, f"{mode} 统计 n=0 却标 present"
        # 功率是 |torque*omega|, 必须非负且有限
        assert s["max"] >= s["p95"] >= 0.0
        assert s["mean"] >= 0.0 and s["rms"] >= 0.0


def test_high_load_modes_hotter(b17_power):
    """§9 —— imaging/desat 应比 cruise/safe 热负荷更高。

    若不成立, 说明任务谱无法体现模式差异, 那本身就是必须报告的问题。
    """
    c = b17_power["high_vs_low_load_check"]
    assert c["high_load_hotter_than_low_load"] is True, (
        f"高负载未更热: high={c['high_load_p95_W']} low={c['low_load_p95_W']}")


def test_thermal_model_energy_balance(b17_registry):
    """§17 —— 模型若存在, 必须是能量平衡形式且参数全部来自 registry。"""
    m = _model()
    if m is None:
        r = _refusal()
        assert r is not None, (
            "既无热模型也无拒绝记录 —— 无法区分「按协议停止」与「忘了跑」")
        assert r["action"] == "REFUSED"
        assert r["thermal_model_written"] is False
        assert r["parameters_assigned"] is False
        assert r["hardcoded_out_of_registry_params"] is False
        # 拒绝必须源于 Gate, 而非其它原因
        assert r["gate"]["all_pass"] is False
        assert r["any_model_identifiable"] is False
        # 每个候选模型都要列出缺哪些参数 (让结论具体可核)
        for name, v in r["per_model_identifiability"].items():
            assert v["missing_params"], f"{name} 判不可辨识却未列缺失参数"
        return
    # 模型存在的分支
    eq = m["equation"] or ""
    assert "dT/dt" in eq, "热模型必须是微分能量平衡形式"
    assert m["all_parameters_from_registry"] is True
    assert m["hardcoded_out_of_registry_params"] is False
    for k, v in m["parameters"].items():
        assert v["provenance"] in USABLE, f"{k} 参数来源不足却入模: {v}"


def test_thermal_model_stable(b17_registry):
    """§17 —— 模型若存在, RC 参数必须为正 (保证 tau>0, 指数收敛而非发散)。"""
    m = _model()
    if m is None:
        assert _refusal() is not None
        return
    for k, v in m["parameters"].items():
        val = v.get("value_used_in_model")
        if val is not None and any(t in k for t in ("resistance", "capacit",
                                                   "time_constant")):
            assert float(val) > 0.0, f"{k}={val} 非正 -> 热模型不稳定"


def test_reachability_before_timeseries(b17_feasibility, b17_config):
    """§11 —— 可达性解析必须先于任何时间序列仿真。"""
    assert b17_config["reachability"]["must_run_before_timeseries"] is True
    r = b17_feasibility["reachability"]
    assert "verdict" in r
    # 无模型时不得报出"结构上到不了"这种实质结论
    if not b17_feasibility["thermal_model_exists"]:
        assert r["verdict"] == "REACHABILITY_UNDECIDABLE_NO_MODEL", (
            "无热模型却给出了可达性实质判决 —— 这是伪造")
        assert r["why_undecidable"]
        assert r["why_not_labelled_unreachable"]
        # T_ss 一栏必须为空, 不能填数
        assert all(v is None for v in r["per_mode_T_ss_degC"].values())
        assert r["T_max_possible_degC"] is None
    else:
        assert r["verdict"] in (
            b17_config["reachability"]["verdicts"]["A"],
            b17_config["reachability"]["verdicts"]["B"],
            b17_config["reachability"]["verdicts"]["C"])


def test_feasibility_gated(b17_feasibility, b17_registry):
    """§14 —— 只有 Gate PASS 且 FEASIBLE 才允许跑 smoke。"""
    gate = b17_registry["gate"]["all_pass"]
    reach = b17_feasibility["reachability"]["verdict"]
    allowed = gate and reach == "THERMAL_LIMIT_FEASIBLE"
    assert b17_feasibility["feasibility_smoke_run"] is allowed, (
        f"smoke 运行状态与门控不符 (gate={gate}, reach={reach})")
    if not allowed:
        assert b17_feasibility["feasibility_smoke_refused_reason"]
        for k in ("T_start", "T_mean", "T_p95", "T_max", "tau_th",
                  "fraction_near_limit"):
            assert b17_feasibility[k] is None, f"未运行 smoke 却有 {k} 值"


def test_no_rul_or_failure_generation(b17_feasibility, b17_verdict, b17_power):
    """§17 —— 本阶段不得生成 EOL / RUL / 寿命数据 / 失效标签。"""
    for d in (b17_feasibility, b17_verdict, b17_power):
        for k in ("eol_generated", "rul_generated",
                  "lifetime_dataset_generated"):
            if k in d:
                assert d[k] is False, f"{k} 为真 —— 越界"
    v = b17_verdict
    assert v["thermal_aging_law_defined"] is False
    assert v["arrhenius_damage_accumulation_defined"] is False
    assert v["damage_state_defined"] is False
    assert v["target_only_run"] is False
    assert v["transfer_run"] is False
    assert v["s6_run"] is False
    # 文件层面复核: 不得存在任何寿命/RUL 产物
    for rel in ("data/sim/wheel_basilisk_b17",
                "checkpoints/basilisk_b17/lifetime_dataset.json",
                "checkpoints/basilisk_b17/rul_metrics.json",
                "checkpoints/basilisk_b17/eol_labels.json",
                "checkpoints/basilisk_b17/thermal_aging_law.json"):
        assert not (ROOT / rel).exists(), f"越界产物: {rel}"


def test_verdict_is_one_of_four(b17_verdict, b17_config):
    """§15 —— 判决必须是四选一, 且与 Gate/可达性自洽。"""
    assert b17_verdict["verdict"] in b17_config["verdicts"]
    g = b17_verdict["gate"]
    if not g["all_pass"]:
        # provenance 不足时只能是 B —— 不得报 C/D (那需要热模型才能观察到)
        assert b17_verdict["verdict"] == "B17_THERMAL_PROVENANCE_INSUFFICIENT"
    # 非 READY 时不得写 next_steps.md (§16)
    ready = b17_verdict["verdict"] == "B17_THERMAL_MODEL_READY"
    assert b17_verdict["next_steps_written"] is ready
    assert (ROOT / "docs/basilisk_b17/next_steps.md").exists() is ready
    assert b17_verdict["wheel_selected_for_lifetime_modelling"] is False
