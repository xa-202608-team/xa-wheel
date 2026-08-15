"""tests/basilisk_b16/test_b16_direct_failure_semantics.py —— §15: 失效语义与裁决出口。"""
from __future__ import annotations

import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "basilisk_b16"
CK = ROOT / "checkpoints" / "basilisk_b16"

MANDATED_SENTENCE = (
    "当前 Basilisk + 可访问官方资料不足以支持具有直接机械失效阈值的反作用轮"
    "寿命数据构建。"
)


def test_frozen_choice_only_if_eligible(b16_verdict, b16_scores):
    """两条出口互斥, 且产物必须与出口严格一致。"""
    n = b16_scores["eligible_count"]
    assert b16_verdict["eligible_count"] == n
    art = b16_verdict["artifacts_written"]
    if n == 0:
        assert b16_verdict["verdict"] == "B16_NO_DOCUMENTED_WHEEL"
        assert b16_verdict["selected_wheel"] is None
        assert art["limitations_md"] is True
        # §11: 无 eligible 时绝不能留下"已选中"的产物
        assert art["frozen_wheel_choice_json"] is False
        assert art["selected_wheel_md"] is False
        assert not (CK / "frozen_wheel_choice.json").exists()
        assert not (DOC / "selected_wheel.md").exists()
        assert b16_verdict["awaiting_human_decision"] is True
    else:
        assert b16_verdict["verdict"] == "B16_DOCUMENTED_WHEEL_SELECTED"
        assert b16_verdict["selected_wheel"] in b16_scores["eligible_models"]
        assert art["frozen_wheel_choice_json"] is True
        assert art["selected_wheel_md"] is True
        choice = (CK / "frozen_wheel_choice.json")
        assert choice.exists()
        import json
        c = json.loads(choice.read_text(encoding="utf-8"))
        assert c["selected_before_any_new_lifetime_simulation"] is True
        for k in ("exact_model", "basilisk_construction",
                  "datasheet_or_document_source", "failure_mode",
                  "failure_threshold", "provenance_classification",
                  "observable_used_for_label"):
            assert k in c, f"§12 缺记录项 {k}"


def test_no_documented_wheel_verdict_is_a_valid_outcome(b16_verdict):
    """verdict 必须是 §9 预先冻结的两个之一, 不得出现第三种自造结论。"""
    assert b16_verdict["verdict"] in ("B16_DOCUMENTED_WHEEL_SELECTED",
                                     "B16_NO_DOCUMENTED_WHEEL")


def test_limitations_contains_mandated_sentence(b16_verdict):
    """§11 强制原句必须逐字出现, 不得改写成更好听的说法。"""
    if b16_verdict["verdict"] != "B16_NO_DOCUMENTED_WHEEL":
        return
    txt = io.open(DOC / "limitations.md", encoding="utf-8").read()
    assert MANDATED_SENTENCE in txt, "limitations.md 缺 §11 强制原句"


def test_no_automatic_fallback_taken(b16_verdict):
    """§11: 不得自动回退到 F2 / Im_rated / arbitrary threshold / 右删失路线。"""
    if b16_verdict["verdict"] != "B16_NO_DOCUMENTED_WHEEL":
        return
    fb = " ".join(b16_verdict["forbidden_fallbacks_not_taken"])
    for k in ("F2", "Im_rated", "arbitrary threshold", "right-censor"):
        assert k in fb, f"回退禁止记录里少了 {k}"
    txt = io.open(DOC / "limitations.md", encoding="utf-8").read()
    # limitations 必须把三条路线交给人工, 而不是自己选一条
    assert "等待人工" in txt or "人工授权" in txt


def test_direct_thermal_limit_not_silently_used(b16_registry, b16_scores):
    """RW0 那条 DIRECT 温度限存在但不可观测 —— 必须被记录且被否决, 两者都要。"""
    thermal_direct = [c["wheel_model"] for c in b16_registry["candidates"]
                      if c["direct_thermal_limit"]["provenance"] == "DIRECT"]
    for m in thermal_direct:
        row = next(r for r in b16_scores["rows"] if r["wheel_model"] == m)
        # 既然是 DIRECT, 就必须出现在 component_limit_candidates 里 (不许藏)
        kinds = [x["kind"] for x in row["component_limit_candidates"]]
        assert "thermal" in kinds, f"{m} 的 DIRECT 热限被漏记"
        # 但因不可观测, 必须落在 rejected 而非 usable
        assert "thermal" not in [x["kind"] for x in row["usable_limits"]], \
            f"{m} 用了不可观测的热限"
        rj = next(x for x in row["rejected_limits"] if x["kind"] == "thermal")
        assert rj["c7_observable_in_current_sim"] is False
        assert rj["reasons"], "否决必须给出理由"


def test_u_max_semantics_still_rejected(b16_scores, b16_fields):
    """B1.5 已判定 u_max 是控制饱和 —— B1.6 不得借它复活失效定义。"""
    assert "u_max" in b16_fields["only_active_limit_fields"]
    for r in b16_scores["rows"]:
        for u in r.get("usable_limits", []) + r.get("rejected_limits", []):
            assert u["kind"] != "control_saturation"
