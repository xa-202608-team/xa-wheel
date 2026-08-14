"""tests/basilisk_b16/test_b16_provenance_scoring.py —— §15: §8/§9 判据与打分纪律。"""
from __future__ import annotations

import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCR = ROOT / "scripts" / "basilisk_b16"

C_KEYS = (
    "c1_constructible_in_basilisk",
    "c2_speed_torque_momentum_auditable",
    "c3_has_component_limit_threshold_direct_or_derived_from_direct",
    "c4_threshold_is_not_mission_statistic",
    "c5_threshold_not_selected_by_failure_outcome",
    "c6_threshold_maps_to_explicit_physical_quantity",
    "c7_quantity_observable_or_computable_in_current_sim",
    "c8_no_hidden_truth_needed_via_x_T",
)


def test_all_eight_conditions_evaluated(b16_scores):
    """§8 八条必须逐条落成结论, 不允许"综合判断"式黑箱。"""
    for r in b16_scores["rows"]:
        if not r["conditions"].get("c1_constructible_in_basilisk"):
            continue
        missing = [k for k in C_KEYS if k not in r["conditions"]]
        assert missing == [], f"{r['wheel_model']} 缺判据 {missing}"


def test_assumed_threshold_ineligible(b16_cfg, b16_scores):
    """§9: ASSUMED 阈值使候选自动不合格; 且合格候选的阈值必须 DIRECT/DERIVED。"""
    assert b16_cfg["assumed_threshold_makes_ineligible"] is True
    assert b16_scores["assumed_threshold_makes_ineligible"] is True
    for r in b16_scores["rows"]:
        for u in r.get("usable_limits", []):
            assert u["provenance"] in ("DIRECT", "DERIVED_FROM_DIRECT"), \
                f"{r['wheel_model']} 用了 {u['provenance']} 级阈值"


def test_score_never_overrides_eligibility(b16_scores):
    """§9 最容易犯的错: 分高但无失效阈值也放行。此处钉死不可能发生。"""
    for r in b16_scores["rows"]:
        if not r["eligible"]:
            assert r["score"] is None, \
                f"{r['wheel_model']} ineligible 却有分数 {r['score']}"
        else:
            assert r["score"] is not None and r["usable_limits"]


def test_failure_limit_is_component_limit(b16_cfg, b16_scores):
    """§2 C: 失效定义必须是 COMPONENT_LIMIT, 不得是任务/数据统计量。"""
    assert b16_cfg["failure_definition_required"] == "COMPONENT_LIMIT"
    allowed_kinds = {"mechanical_friction_bearing", "thermal",
                     "electrical_continuous_limit", "overspeed_momentum"}
    for r in b16_scores["rows"]:
        for u in r.get("usable_limits", []):
            assert u["kind"] in allowed_kinds, u["kind"]
            # control_saturation 已被 B1.5 否决, 绝不允许进入可用集
            assert u["kind"] != "control_saturation"


def test_mission_statistic_not_failure_limit(b16_cfg, b16_scores, b16_registry):
    """C4 / §4 E: 任务统计量与 BOL 即触及的限值都不得充当退化 EOL。"""
    forb = [s.lower() for s in b16_cfg["failure_definition_forbidden"]]
    assert any("percentile" in s or "duty" in s for s in forb), \
        "config 的禁止清单丢了 percentile / duty 项"
    for r in b16_scores["rows"]:
        for u in r.get("usable_limits", []):
            assert u["c4_not_mission_statistic"] is True
            # BOL 即触及者必须已被剔除
            assert u["bol_check"] is True, \
                f"{r['wheel_model']} 的可用阈值 bol_check={u['bol_check']}"
    # 反向: BOL 即触及的限值必须出现在 rejected 里并给出理由
    for c in b16_registry["candidates"]:
        s = c.get("direct_overspeed_limit", {})
        if s.get("attained_at_BOL") is True:
            row = next(r for r in b16_scores["rows"]
                       if r["wheel_model"] == c["wheel_model"])
            rj = [x for x in row["rejected_limits"]
                  if x["kind"] == "overspeed_momentum"]
            assert rj and rj[0]["reasons"], "BOL 触及却没有记录否决理由"


def test_unobservable_quantity_rejected(b16_scores, b16_fields):
    """C7: 当前仿真观测不到的物理量不得作为 EOL —— 即使其 provenance 是 DIRECT。"""
    for r in b16_scores["rows"]:
        for u in r.get("usable_limits", []):
            assert u["c7_observable_in_current_sim"] is True
    # 字段审计里不可观测的类别, 不得出现在任何 usable_limits
    cat_of = {"mechanical_friction_bearing": "mechanical", "thermal": "thermal",
              "electrical_continuous_limit": "electrical",
              "overspeed_momentum": "speed_momentum"}
    for r in b16_scores["rows"]:
        for u in r.get("usable_limits", []):
            cat = cat_of[u["kind"]]
            assert b16_fields["per_category"][cat][
                "category_observable_in_payload"] is True


def test_selection_rule_frozen(b16_cfg):
    """§10 冻结选择顺序 + 禁止的 tiebreak 必须原样在 config 里。"""
    assert b16_cfg["selection_order"] == [
        "mechanical_failure_threshold_available",
        "provenance_score_high",
        "basilisk_native_support",
        "minimal_model_change",
    ]
    tb = " ".join(b16_cfg["tiebreak_forbidden"]).lower()
    for k in ("failure_fraction", "rul", "data_availability"):
        assert k in tb, f"tiebreak_forbidden 少了 {k}"


def test_failure_mode_priority_frozen(b16_cfg):
    """§7 优先级必须预先冻结且机械排第一 (与源域轴承退化机理一致)。"""
    p = b16_cfg["failure_mode_priority"]
    assert p[0] == "mechanical_friction_bearing"
    assert p[-1] == "control_saturation"
    assert len(p) == 5


def test_score_weights_match_goal(b16_cfg):
    """§9 权重不得被悄悄调整。"""
    w = b16_cfg["score_weights"]
    assert w["direct_failure_threshold"] == 4
    assert w["derived_from_direct"] == 3
    assert w["direct_mechanical_threshold"] == 3
    assert w["direct_thermal_threshold"] == 2
    assert w["direct_electrical_threshold"] == 2
    assert w["basilisk_builtin_preset"] == 2
    assert w["public_official_datasheet"] == 2
