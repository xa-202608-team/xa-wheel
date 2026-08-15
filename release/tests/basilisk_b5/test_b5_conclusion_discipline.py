"""tests/basilisk_b5/test_b5_conclusion_discipline.py —— §11..§15/§18/§19 结论纪律。

- damage_extrapolation 是强基线, 必须留在主结果表里, 不得降级成脚注 (§14)
- 工程推荐与迁移判定分开, 依据只能是正式 test 表 (§15)
- 主结果表五行齐备, 十列齐备 (§18)
- 短寿命优势若只在 short bin 出现, 必须报 localized transfer benefit (§12)
- 右删失轨迹不得伪造 EOL 指标; 空 bin 返回 NaN + n=0 (§13)
"""
from __future__ import annotations

import numpy as np

from conftest import ROOT

MAIN_ROWS = ("target_only", "source_finetune", "source_mmd_finetune",
             "const_mean_info", "damage_extrapolation")
MAIN_COLS = ("info_macro_rmse", "info_pooled_rmse", "mae", "corr", "psr",
             "warning_coverage", "miss_rate", "ph", "alpha_lambda",
             "catastrophic_rate")


def test_b5_damage_baseline_in_main_table(b5_summary):
    """test_b5_damage_baseline_in_main_table (§21/§14)。"""
    assert "damage_extrapolation" in b5_summary["main_table"]
    assert "damage_extrapolation" in b5_summary["main_table_rows"]
    d = b5_summary["damage_baseline"]
    assert bool(d["kept_in_main_table"]) is True
    assert bool(d["demoted_to_footnote"]) is False
    assert bool(d["forbid_calling_positive_transfer_the_best_method"]) is True


def test_b5_main_table_complete(b5_summary):
    """§18: 五行齐备, 十列齐备。"""
    mt = b5_summary["main_table"]
    assert tuple(b5_summary["main_table_rows"]) == MAIN_ROWS
    for g in MAIN_ROWS:
        for c in MAIN_COLS:
            assert c in mt[g], f"{g} 缺列 {c}"


def test_b5_per_seed_detail_kept_separately(b5_summary):
    """§18: 逐 seed 明细另存一表。"""
    rel = str(b5_summary["per_seed_detail_kept_separately"])
    assert (ROOT / rel).exists()


def test_b5_transfer_vs_damage_computed(b5_summary):
    """§14: transfer_vs_damage = RMSE_damage - RMSE_transfer, 逐方法算。"""
    mt = b5_summary["main_table"]
    dmg = float(mt["damage_extrapolation"]["info_macro_rmse"])
    cmp_ = b5_summary["damage_baseline"]["comparison"]
    for g in ("source_finetune", "source_mmd_finetune"):
        exp = dmg - float(mt[g]["info_macro_rmse"])
        assert abs(float(cmp_[g]["transfer_vs_damage"]) - exp) < 1e-9
        assert bool(cmp_[g]["transfer_worse_than_physics"]) == bool(exp < 0)


def test_b5_mandatory_sentence_when_transfer_loses_to_physics(b5_summary):
    """§14: 迁移不如物理基线时, 必须原样写出那句话。"""
    cmp_ = b5_summary["damage_baseline"]["comparison"]
    any_worse = any(bool(v["transfer_worse_than_physics"])
                    for v in cmp_.values())
    stmt = b5_summary["damage_baseline"]["mandatory_statement"]
    if any_worse:
        assert stmt is not None
        assert "未达到物理 damage extrapolation 基线" in str(stmt)
        # 文档里也必须出现这句
        txt = (ROOT / "docs/basilisk_b5/results.md").read_text(encoding="utf-8")
        assert "未达到物理 damage extrapolation 基线" in txt
    else:
        assert stmt is None


def test_b5_engineering_recommendation_separate(b5_summary):
    """test_b5_engineering_recommendation_separate (§21/§15)。"""
    rec = b5_summary["engineering_recommendation"]
    assert bool(rec["separate_from_transfer_verdict"]) is True
    assert str(rec["basis"]) == "formal_test_table_primary_metric"
    assert str(b5_summary["ENGINEERING_RECOMMENDATION"]) == \
        str(rec["recommendation"])
    allowed = {"target_only", "source_finetune", "source_mmd_finetune",
               "damage_extrapolation"}
    assert str(rec["recommendation"]) in allowed


def test_b5_recommendation_matches_test_table(b5_summary):
    """推荐必须就是正式 test 表主指标最优的那个 —— 不得凭机理性解释另选。"""
    mt = b5_summary["main_table"]
    rec = b5_summary["engineering_recommendation"]
    cands = [r["method"] for r in rec["ranking"]]
    best = min(cands, key=lambda g: float(mt[g]["info_macro_rmse"]))
    assert str(rec["recommendation"]) == best
    vals = [float(r["info_macro_rmse"]) for r in rec["ranking"]]
    assert vals == sorted(vals), "ranking 未按主指标升序"


def test_b5_lifetime_bin_table_complete(b5_lifetime):
    """§11: 每组每 bin 都要有 RMSE / MAE / corr / warning / catastrophic。"""
    names = list(b5_lifetime["bins"]["names"])
    for g, per in b5_lifetime["by_group"].items():
        for n in names:
            b = per[n]
            for k in ("rmse_mean", "mae_mean", "corr_mean",
                      "warning_coverage_mean", "warning_miss_mean",
                      "catastrophic_count_total", "n_traj_in_bin"):
                assert k in b, f"{g}/{n} 缺 {k}"


def test_b5_gain_by_bin_definition(b5_lifetime):
    """§11: 逐 bin 的 target - FT 与 target - MMD。"""
    for n, blk in b5_lifetime["gain_by_bin"].items():
        for key in ("gain_ft", "gain_mmd"):
            assert key in blk, f"{n} 缺 {key}"
    npos = b5_lifetime["n_bins_gain_positive"]
    for key in ("ft", "mmd"):
        cnt = sum(1 for blk in b5_lifetime["gain_by_bin"].values()
                  if blk[f"gain_{key}"] is not None
                  and np.isfinite(float(blk[f"gain_{key}"]))
                  and float(blk[f"gain_{key}"]) > 0)
        assert int(npos[key]) == cnt, f"{key} 的正 bin 计数不自洽"


def test_b5_short_life_five_quantities(b5_lifetime):
    """§12: short bin 必须报 mean RMSE / 逐 seed RMSE / catastrophic / 两个 std。"""
    for g, blk in b5_lifetime["short_life"]["by_group"].items():
        for k in ("mean_rmse", "per_seed_rmse", "catastrophic_count_total",
                  "pred_std", "true_std"):
            assert k in blk, f"short_life/{g} 缺 {k}"
        assert len(blk["per_seed_rmse"]) == 5


def test_b5_localized_benefit_flagged(b5_lifetime):
    """§12: 只在 short bin 有优势时, 必须标 localized transfer benefit。"""
    loc = b5_lifetime["localized_transfer_benefit"]
    gb = b5_lifetime["gain_by_bin"]
    for key in ("ft", "mmd"):
        def _pos(n):
            v = gb[n][f"gain_{key}"]
            return v is not None and np.isfinite(float(v)) and float(v) > 0
        exp = _pos("short") and not any(_pos(n) for n in ("medium", "long"))
        assert bool(loc[key]) == exp, f"{key} 的局部增益标记不自洽"


def test_b5_empty_bin_is_nan_not_zero(b5_lifetime):
    """空 bin 必须返回 NaN + n=0, 不得伪造 0。"""
    for g, per in b5_lifetime["by_group"].items():
        for n, b in per.items():
            if int(b["n_traj_in_bin"]) == 0:
                assert b["rmse_mean"] is None or \
                    not np.isfinite(float(b["rmse_mean"])), \
                    f"{g}/{n} 是空 bin 却给了有限 RMSE"


def test_b5_censored_bins_degeneracy_recorded(b5_lifetime):
    """删失侧 bin 已退化 (n_effective_bins=1), 必须如实记录, 不得当成三个 bin。"""
    c = b5_lifetime["censored_bins"]
    assert bool(c["degenerate"]) is True
    assert int(c["n_effective_bins"]) == 1


def test_b5_warning_miss_rate_reported(b5_summary, b5_warning):
    """不得只统计成功检测而隐藏 miss rate。"""
    for g in MAIN_ROWS:
        assert "miss_rate" in b5_summary["warning"]["by_group"][g]
        assert "miss_rate" in b5_warning["by_group"][g]


def test_b5_no_fake_eol_for_censored(b5_warning):
    """右删失轨迹不得伪造 EOL 指标 —— PH 必须记录被排除的删失数量。"""
    for g, blk in b5_warning["by_group"].items():
        ph = blk["prognostic_horizon"]
        assert "n_censored_excluded" in ph or "n_evaluable_traj" in ph, \
            f"{g} 未记录删失排除口径"


def test_b5_nphm_not_a_selection_metric(b5_summary):
    """nPHM 不得成为模型选择指标。"""
    assert str(b5_summary["engineering_recommendation"]["primary_metric"]) == \
        "test_info_trajectory_macro_rmse"
    for meth in ("source_finetune", "source_mmd_finetune"):
        names = [c["name"] for c in b5_summary["decision"][meth]["conditions"]]
        assert not any("nphm" in n.lower() for n in names)


def test_b5_verdict_reproducible_from_conditions(b5_summary):
    """结论必须能从 conditions 复算 —— 手改 verdict 会被这条抓住。"""
    for meth in ("source_finetune", "source_mmd_finetune"):
        d = b5_summary["decision"][meth]
        assert all(bool(c["passed"]) for c in d["conditions"]) == \
            bool(d["positive_transfer"])


def test_b5_next_stage_recorded_but_not_executed(b5_summary):
    """§20: 下一阶段只写建议。"""
    ns = str(b5_summary["next_stage"])
    assert "B6" in ns
    ft = bool(b5_summary["positive_transfer"]["ft"])
    mmd = bool(b5_summary["positive_transfer"]["mmd"])
    if ft or mmd:
        assert "low" in ns.lower() or "truncation" in ns.lower()
    else:
        assert "negative" in ns.lower() or "baseline" in ns.lower()
