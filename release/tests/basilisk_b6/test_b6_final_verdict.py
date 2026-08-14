"""tests/basilisk_b6/test_b6_final_verdict.py —— §15/§16/§18/§19/§20/§21 判定纪律。

- primary 判定必须是十条全过才 PASS, 任一条 NaN 一律计为 FAIL (§15)
- secondary 永远不能推翻 primary: 即使 n=3 全胜, primary 未过就必须是 NO (§16)
- damage 基线必须留在主表, 不得降级为脚注; 若领先必须原样写出定论句 (§18)
- 单一工程推荐只能从四个候选里选, 优先级冻结, oracle 方法排除 (§19)
- B5 结论不得被 B6 覆盖; 情况 B 绝不能写成"迁移学习总体显著优于 Target-only" (§20)
- 最终结论只能取两个允许值之一 (§21)
"""
from __future__ import annotations

import math

from conftest import ROOT, code_nospace

LEVELS = [3, 5, 10, 21]
TRAINED = ("target_only", "source_finetune", "source_mmd_finetune")
MAIN_ROWS = TRAINED + ("damage_extrapolation", "const_mean_info")
ALLOWED_FINAL = ["NO_POSITIVE_TRANSFER_SUPPORTED",
                 "CONDITIONAL_LOW_LABEL_TRANSFER_ONLY"]
FORBIDDEN_PHRASE = "迁移学习总体显著优于 Target-only"
VERDICT_SCRIPT = "scripts/basilisk_b6/final_transfer_verdict.py"
COND_NAMES = ["mean_gain_positive", "median_gain_positive", "improve_count",
              "bootstrap_ci_lower_positive", "lifetime_bins_gain_nonneg",
              "no_corr_degradation", "catastrophic_not_worse",
              "warning_miss_not_worse", "same_data_same_budget_fairness",
              "beats_const_mean_info"]


def test_b6_primary_gate_is_ten_conditions_all_required(b6_verdict):
    """§15: 十条门槛, 必须全部满足才算 PASS; 判定只在 primary 档做。"""
    assert int(b6_verdict["primary_level"]) == 5
    for meth in ("source_finetune", "source_mmd_finetune"):
        dec = b6_verdict["primary_decision"][meth]
        assert int(dec["level"]) == 5, f"{meth} 的判定不是在 primary 档做的"
        assert bool(dec["is_primary"]) is True
        assert int(dec["n_conditions"]) == 10
        assert bool(dec["all_required"]) is True
        assert [str(c["name"]) for c in dec["conditions"]] == COND_NAMES
        assert [int(c["id"]) for c in dec["conditions"]] == list(range(1, 11))
        n_pass = sum(1 for c in dec["conditions"] if bool(c["passed"]))
        assert int(dec["n_passed"]) == n_pass
        # PASS 当且仅当十条全过
        assert bool(dec["low_label_positive_transfer"]) is bool(n_pass == 10)
        assert bool(dec["thresholds_frozen_before_run"]) is True
        assert bool(dec["nan_counts_as_fail"]) is True
        assert sorted(int(i) for i in dec["failed_conditions"]) == \
            sorted(int(c["id"]) for c in dec["conditions"]
                   if not bool(c["passed"]))


def test_b6_verdict_label_matches_conditions(b6_verdict, b6_protocol_hash):
    """§15: 判定标签必须由 conditions 唯一决定, 不得手改。"""
    lab = b6_protocol_hash["gate_labels"]
    assert dict(b6_verdict["gate_labels"]) == dict(lab)
    any_ok = False
    for meth in ("source_finetune", "source_mmd_finetune"):
        dec = b6_verdict["primary_decision"][meth]
        ok = all(bool(c["passed"]) for c in dec["conditions"])
        exp = str(lab["primary_pass"]) if ok else str(lab["primary_fail"])
        assert str(dec["verdict"]) == exp, f"{meth} 的标签与 conditions 不符"
        any_ok = any_ok or ok
    # 两者皆 fail -> 整体 fail
    pv = str(b6_verdict["B6_PRIMARY_VERDICT"])
    assert pv == (str(lab["primary_pass"]) if any_ok
                  else str(lab["primary_fail"]))
    assert str(b6_verdict["primary_verdict"]) == pv
    assert bool(b6_verdict["primary_positive"]["any"]) is bool(any_ok)


def test_b6_nan_conditions_count_as_fail(b6_verdict):
    """§15: 任一条件取值为 NaN 时必须判 FAIL, 不得"缺数据就放过"。"""
    for meth in ("source_finetune", "source_mmd_finetune"):
        for c in b6_verdict["primary_decision"][meth]["conditions"]:
            v = c.get("value")
            if isinstance(v, (int, float)) and not isinstance(v, bool) \
                    and not math.isfinite(float(v)):
                assert bool(c["passed"]) is False, \
                    f"{meth} 条件 {c['name']} 取值为 NaN 却判成 passed"
    src = code_nospace(VERDICT_SCRIPT)
    # _pos/_le/_ge 必须先要求有限值
    assert "np.isfinite" in src, "判定脚本未对有限性做检查 —— NaN 可能被放过"


def test_b6_secondary_cannot_override_primary(b6_verdict, b6_config):
    """test_b6_secondary_cannot_override_primary (§23/§16)。"""
    sec = b6_verdict["secondary"]
    assert sorted(int(x) for x in sec["levels"]) == [3, 10, 21]
    assert 5 not in [int(x) for x in sec["levels"]]
    assert bool(sec["cannot_override_primary"]) is True
    assert bool(sec["override_attempted"]) is False
    assert bool(b6_verdict["secondary_cannot_override_primary"]) is True
    lab_fail = str(b6_verdict["gate_labels"]["primary_fail"])
    primary_ok = bool(b6_verdict["primary_positive"]["any"])
    # 核心不变量: 只要 primary 未过, 无论 secondary 出现多少信号, 正式判定仍是 NO
    if not primary_ok:
        assert str(b6_verdict["B6_PRIMARY_VERDICT"]) == lab_fail
        assert str(b6_verdict["FINAL_TRANSFER_CONCLUSION"]) == \
            "NO_POSITIVE_TRANSFER_SUPPORTED"
    # secondary 信号必须被打上"非 primary 确证证据"的标记
    for s in sec["signals"]:
        assert str(s["mark_as"]) == "NOT_PRIMARY_CONFIRMATORY_EVIDENCE"
        assert bool(s["cannot_override_primary"]) is True
        assert str(s["label"]).startswith("SECONDARY_LOW_LABEL_SIGNAL_AT_N")
        assert int(s["n_event_labeled"]) in (3, 10, 21)
        assert "不是 primary 的十条门槛" in str(s["criteria"])
    sig = b6_config["b6"]["secondary_signal"]
    assert bool(sig["cannot_override_primary"]) is True
    assert str(sig["mark_as"]) == "NOT_PRIMARY_CONFIRMATORY_EVIDENCE"


def test_b6_damage_in_main_table(b6_summary, b6_verdict, b6_config):
    """test_b6_damage_in_main_table (§23/§18) —— damage 必须在主表, 不是脚注。"""
    assert bool(b6_summary["damage_in_main_table"]) is True
    assert bool(b6_summary["damage_demoted_to_footnote"]) is False
    assert "damage_extrapolation" in list(b6_summary["main_table_rows"])
    db = b6_config["b6"]["damage_baseline"]
    assert bool(db["keep_in_main_table"]) is True
    assert bool(db["forbid_footnote_only"]) is True
    assert bool(db["forbid_hiding_because_theme_is_transfer_learning"]) is True
    # 每一档主表都要有 damage 行 + 三个差值
    for n in LEVELS:
        T = b6_summary["main_tables"][str(n)]
        assert "damage_extrapolation" in T["rows"], f"n={n} 主表缺 damage 行"
        vd = T["vs_damage"]
        for f in db["per_level_comparison"]:
            assert str(f) in vd, f"n={n} 缺差值 {f}"
    vb = b6_verdict["damage_baseline"]
    assert bool(vb["kept_in_main_table"]) is True
    assert bool(vb["demoted_to_footnote"]) is False
    assert sorted(int(k) for k in vb["per_level"]) == LEVELS


def test_b6_mandatory_damage_phrasing(b6_verdict, b6_summary, b6_config):
    """§18: damage 若仍明显领先, 必须原样写出那句定论。"""
    phrase = str(
        b6_config["b6"]["damage_baseline"]["mandatory_phrasing_when_damage_leads"])
    assert phrase == "学习模型未超过基于已知累计损伤结构的物理外推基线。"
    vb = b6_verdict["damage_baseline"]
    if bool(vb["damage_leads_at_primary"]):
        assert str(vb["mandatory_statement"]) == phrase
        doc = ROOT / "docs" / "basilisk_b6" / "results.md"
        assert doc.exists(), "缺 results.md"
        assert phrase in doc.read_text(encoding="utf-8"), \
            "damage 领先却未在 results.md 原样写出定论句"
    if bool(b6_summary["damage_leads_all_learned_at_every_level"]):
        assert str(b6_summary["mandatory_statement_when_damage_leads"]) == phrase


def test_b6_engineering_recommendation_valid(b6_verdict, b6_config):
    """§19: 单一推荐, 只从四个候选里选, oracle 方法排除, 优先级冻结。"""
    er = b6_verdict["engineering_recommendation"]
    erc = b6_config["b6"]["engineering_recommendation"]
    rec = str(b6_verdict["ENGINEERING_RECOMMENDATION"])
    assert rec == str(er["recommendation"])
    assert rec in list(erc["candidates"]), f"推荐 {rec} 不在候选内"
    ex = set(erc["excluded_oracle_methods"])
    assert rec not in ex, f"推荐了 oracle 方法 {rec}"
    assert bool(er["single_recommendation"]) is True
    assert bool(er["priority_frozen"]) is True
    assert list(er["priority_order"]) == list(erc["priority_order"])
    assert bool(er["separate_from_transfer_verdict"]) is True
    ranked = [str(d["method"]) for d in er["ranking"]]
    assert ranked and ranked[0] == rec, "推荐与排名首位不一致"
    assert not (set(ranked) & ex), "排名里混入了 oracle 方法"
    # 排名必须按 (RMSE, miss, catastrophic) 单调 —— 首位 RMSE 不得高于其他候选
    fin = [float(d["info_macro_rmse"]) for d in er["ranking"]
           if math.isfinite(float(d["info_macro_rmse"]))]
    assert fin == sorted(fin), "排名未按主指标升序 —— 优先级被改动"


def test_b6_b5_conclusion_not_overwritten(b6_verdict, b6_config, b5_summary_ro):
    """§0/§20: B5 的结论不得因 B6 被覆盖。"""
    ff = b6_config["b6"]["frozen_facts"]
    assert str(ff["b5_verdict"]) == "B5_NO_POSITIVE_TRANSFER"
    assert str(b5_summary_ro["verdict"]) == "B5_NO_POSITIVE_TRANSFER"
    rel = b6_verdict["b5_b6_relationship"]
    assert str(rel["b5_verdict"]) == "B5_NO_POSITIVE_TRANSFER"
    assert bool(rel["b5_overwritten_by_b6"]) is False
    assert bool(rel["b5_conclusion_permanent"]) is True
    assert bool(rel["forbid_writing_case_B_as_overall_superiority"]) is True
    assert str(rel["forbidden_phrasing"]) == FORBIDDEN_PHRASE


def test_b6_case_mapping_follows_s20(b6_verdict):
    """§20: 情况 A / B 的映射固定, 且 B 绝不可写成"总体优于"。"""
    primary_ok = bool(b6_verdict["primary_positive"]["any"])
    rel = b6_verdict["b5_b6_relationship"]
    case = str(rel["case"])
    assert case == ("B" if primary_ok else "A")
    final = str(b6_verdict["FINAL_TRANSFER_CONCLUSION"])
    if case == "A":
        assert final == "NO_POSITIVE_TRANSFER_SUPPORTED"
        assert "NO_POSITIVE_TRANSFER_SUPPORTED" in str(rel["combined_statement"])
    else:
        assert final == "CONDITIONAL_LOW_LABEL_TRANSFER_ONLY"
        desc = str(rel["combined_statement"])
        assert "NO_GENERAL_POSITIVE_TRANSFER" in desc
        assert "CONDITIONAL_BENEFIT_UNDER_FAILURE_LABEL_SCARCITY" in desc


def test_b6_forbidden_phrasing_absent_everywhere(b6_verdict):
    """§20 绝对禁令: 任何产物 / 文档里都不得出现那句话 (除禁令声明本身)。"""
    docs = sorted((ROOT / "docs" / "basilisk_b6").glob("*.md"))
    assert docs, "docs/basilisk_b6 下没有文档"
    for p in docs:
        txt = p.read_text(encoding="utf-8")
        for ln, line in enumerate(txt.splitlines(), 1):
            if FORBIDDEN_PHRASE in line:
                low = line
                assert ("禁" in low or "不得" in low or "forbidden" in low.lower()
                        or "绝不" in low), \
                    f"{p.name}:{ln} 出现被禁措辞且不是禁令声明: {line.strip()}"


def test_b6_final_conclusion_consistent(b6_verdict, b6_config):
    """test_b6_final_conclusion_consistent (§23/§21)。"""
    final = str(b6_verdict["FINAL_TRANSFER_CONCLUSION"])
    allowed = list(b6_config["b6"]["final_conclusion"]["allowed_values"])
    assert allowed == ALLOWED_FINAL
    assert final in allowed, f"最终结论 {final} 不在允许取值内"
    assert list(b6_verdict["final_conclusion_allowed_values"]) == allowed
    # 与 primary 判定一致
    primary_ok = bool(b6_verdict["primary_positive"]["any"])
    assert final == ("CONDITIONAL_LOW_LABEL_TRANSFER_ONLY" if primary_ok
                     else "NO_POSITIVE_TRANSFER_SUPPORTED")
    # 文档必须存在, 写明同一结论, 并分别解释五件事
    doc = ROOT / str(b6_config["b6"]["final_conclusion"]["path"])
    assert doc.exists(), f"缺最终结论文档 {doc}"
    txt = doc.read_text(encoding="utf-8")
    assert final in txt, "final_conclusion.md 未写出该结论"
    other = [a for a in allowed if a != final][0]
    for ln in txt.splitlines():
        if other in ln:
            assert ("未" in ln or "不是" in ln or "allowed" in ln.lower()
                    or "取值" in ln or "|" in ln), \
                f"final_conclusion.md 同时主张了另一个结论: {ln.strip()}"
    must = list(b6_config["b6"]["final_conclusion"]["must_explain_separately"])
    assert list(b6_verdict["final_conclusion_must_explain_separately"]) == must
    for topic, kw in (("b5_full_data_evidence", "B5"),
                      ("b6_low_label_evidence", "n=5"),
                      ("lifetime_bin_behavior", "short"),
                      ("warning_behavior", "告警"),
                      ("damage_baseline_comparison", "damage")):
        assert topic in must
        assert kw in txt, f"final_conclusion.md 未分别解释 {topic} (缺关键词 {kw})"


def test_b6_verdict_stops_here(b6_verdict, b6_config):
    """§26: B6 到此停止 —— 判定产物必须明确记录未自动跑 B7 / B8。"""
    assert bool(b6_verdict["stop_after_b6"]) is True
    assert bool(b6_verdict["forbid_auto_run_b7"]) is True
    assert bool(b6_verdict["forbid_auto_run_b8"]) is True
    assert bool(b6_verdict["b7_auto_run"]) is False
    assert bool(b6_verdict["b8_auto_run"]) is False
    assert list(b6_verdict["next_stage_allowed_only"]) == [
        "B7: figures + report", "B8: packaging + Docker"]


def test_b6_verdict_hashes_match_frozen(b6_verdict, b6_manifest,
                                        b6_protocol_hash):
    """判定必须挂在冻结协议 / 子集清单 / 划分之上。"""
    assert str(b6_verdict["protocol_sha256"]) == \
        str(b6_protocol_hash["protocol_sha256"])
    assert str(b6_verdict["subset_manifest_sha256"]) == \
        str(b6_manifest["subset_manifest_sha256"])
    assert str(b6_verdict["split_sha256"]) == str(b6_manifest["split_sha256"])
    # 产物里 schema 名沿用 config 的小写字面量 core_only, 与 §4 的 CORE_ONLY
    # 是同一个 schema —— 只比较大小写无关的标识, 不改冻结产物。
    assert str(b6_verdict["input_schema"]).upper() == "CORE_ONLY"
    assert int(b6_verdict["n_features"]) == 12
    assert [int(s) for s in b6_verdict["formal_seeds"]] == \
        [122, 123, 124, 125, 126]
    assert dict(b6_verdict["gate_thresholds"]) == \
        dict(b6_protocol_hash["gate_thresholds"])


def test_b6_trend_descriptive_only(b6_summary):
    """§17: 趋势只做描述, 不拟合趋势线。"""
    for meth in ("source_finetune", "source_mmd_finetune"):
        t = b6_summary["gain_vs_labels"][meth]
        assert bool(t["descriptive_only"]) is True
        assert bool(t["trend_line_fitted"]) is False
        assert [int(p["n_event_labeled"]) for p in t["points"]] == LEVELS
        assert int(t["n_levels"]) == 4
    src = code_nospace("scripts/basilisk_b6/summarize_matrix.py")
    for bad in ("np.polyfit", "polyfit(", "linregress", "curve_fit"):
        assert bad.replace(" ", "") not in src, f"汇总脚本拟合了趋势线: {bad}"
