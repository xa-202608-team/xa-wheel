"""tests/basilisk_b18/test_b18_assumption_registry.py —— §3/§7 假设台账与 L_ref 冻结。

含 §22 要求的 `test_lref_registry_frozen`。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LEVELS = ("DIRECT", "PROJECT_DOCUMENTED_ASSUMPTION", "NEW_B18_ASSUMPTION")


def test_lref_registry_frozen(b18_config, b18_protocol_hash, b18_damage_model):
    """§7: 三个 L_ref 与顺序在协议冻结时定死, 三处记录必须逐值一致。"""
    H = float(b18_config["lifetime_scenarios"]["horizon_years"])
    assert H == 3.0, "horizon 应为 Basilisk lifetime horizon 3.0 years"
    expect = {"SLOW": 1.5 * H, "NOMINAL": 1.0 * H, "FAST": 0.75 * H}
    scen = b18_config["lifetime_scenarios"]["scenarios"]
    for k, v in expect.items():
        assert float(scen[k]["L_ref_years"]) == pytest.approx(v, abs=1e-12), \
            f"{k} 的 L_ref 应为 {v}"
    assert float(scen["SLOW"]["multiplier"]) == 1.5
    assert float(scen["NOMINAL"]["multiplier"]) == 1.0
    assert float(scen["FAST"]["multiplier"]) == 0.75
    # 只允许三个情景 (§7: 不要增加第四个)
    assert len(scen) == 3 == int(b18_config["lifetime_scenarios"][
        "n_scenarios_allowed"])
    assert list(b18_config["lifetime_scenarios"]["order"]) == \
        ["SLOW", "NOMINAL", "FAST"]
    # 协议冻结记录
    frz = b18_protocol_hash["lifetime_scenarios_at_freeze"]
    assert {k: float(v) for k, v in frz.items()} == \
        {k: float(v["L_ref_years"]) for k, v in scen.items()}
    assert list(b18_protocol_hash["scenario_order_at_freeze"]) == \
        ["SLOW", "NOMINAL", "FAST"]
    # damage_model.json 记录
    assert {k: float(v) for k, v in
            b18_damage_model["lifetime_scenarios"].items()} == \
        {k: float(v["L_ref_years"]) for k, v in scen.items()}
    assert list(b18_damage_model["scenario_order"]) == ["SLOW", "NOMINAL", "FAST"]


def test_lref_not_derived_from_outcomes(b18_config):
    """§7: L_ref 不得由 event count / failure fraction / RUL metric 反推。"""
    ls = b18_config["lifetime_scenarios"]
    assert ls["derived_from_event_count"] is False
    assert ls["derived_from_failure_fraction"] is False
    assert ls["derived_from_rul_metric"] is False


def test_registry_three_levels_only(b18_registry):
    """§3: provenance 只有三级, 且每条都必须落在其中之一。"""
    assert tuple(b18_registry["levels"]) == LEVELS
    for item in b18_registry["registry"]:
        assert item["level"] in LEVELS, f"{item['name']} 的 level 非法"


def test_registry_direct_items_are_really_direct(b18_registry):
    """§3: 不得把 B/C 类写成 DIRECT。DIRECT 只能是 Basilisk profile / HR16 事实。"""
    direct = [i for i in b18_registry["registry"] if i["level"] == "DIRECT"]
    assert len(direct) == 5
    assert sorted(i["name"] for i in direct) == [
        "HR16_Omega_max", "HR16_u_max", "commanded_torque",
        "mission_mode_set", "wheel_speed"]
    banned = ("damage", "l_ref", "lref", "life_scale", "post_eol", "eol",
              "threshold", "failure")
    for i in direct:
        low = i["name"].lower()
        assert not any(b in low for b in banned), \
            f"DIRECT 项 {i['name']} 含失效/损伤类语义, 不可能是 DIRECT"
        assert i["source"] and i["verify"], f"{i['name']} 缺 source/verify"


def test_registry_new_b18_assumptions_declared(b18_registry):
    """§3 C 类: 累计损伤失效定义与 life-scale multiplier 必须显式标为新增假设。"""
    names = set(b18_registry["new_b18_names"])
    assert "cumulative_damage_failure_definition" in names
    assert "life_scale_multiplier" in names
    assert b18_registry["counts"]["NEW_B18_ASSUMPTION"] == len(names) == 4


def test_registry_never_claims_manufacturer_spec(b18_registry):
    """§24: 不得声称厂家失效规格, 也不得声称 DIRECT EOL provenance。"""
    assert b18_registry["manufacturer_failure_specification_claimed"] is False
    assert b18_registry["direct_eol_provenance_claimed"] is False


def test_abolished_thresholds_absent_from_eol_path(b18_config):
    """§4: F1 电流阈值 / F2 摩擦力矩阈值 / F3 b_true 阈值 / 85C 全不参与新 EOL。"""
    d = b18_config["damage_model"]
    assert d["eol_rule"] == "cumulative_damage_reaches_one"
    assert float(d["eol_threshold_D"]) == 1.0
    # 新 EOL 的唯一依据是 D, 配置里不得出现替代阈值键
    for bad in ("Im_threshold_A", "friction_torque_threshold",
                "b_true_threshold", "vendor_thermal_limit_C",
                "thermal_threshold_C"):
        assert bad not in d, f"{bad} 不应出现在 damage_model 中 (§4 已废止)"


def test_protocol_and_config_hashes_recorded(b18_protocol_hash, b18_registry,
                                             b18_damage_model):
    """协议与 config 的哈希必须被每个下游产物记录, 以便检测漂移。"""
    ph = b18_protocol_hash["protocol_sha256"]
    ch = b18_protocol_hash["config_sha256"]
    assert b18_registry["protocol_sha256"] == ph
    assert b18_registry["config_sha256"] == ch
    assert b18_damage_model["protocol_sha256"] == ph
    assert b18_damage_model["config_sha256"] == ch
    # 磁盘现状必须与冻结值一致
    p = ROOT / "docs/basilisk_b18/protocol.md"
    c = ROOT / "configs/wheel_basilisk_b18.yaml"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == ph, "protocol.md 漂移"
    assert hashlib.sha256(c.read_bytes()).hexdigest() == ch, "config 漂移"
