"""tests/basilisk_b6/test_b6_statistics.py —— §8/§13/§14/§17 统计纪律。

- 配对统计单位是 seed 级配对增益, n=5; 禁止把端点数 / 轨迹条数当独立样本 (§13)
- gain 定义为 target_only − source_*, 正数 = source 更好 (§13)
- PRIMARY 永远是 n=5, 在看到任何结果之前就固定 (§8)
- 右删失轨迹无真 EOL 的指标保持 NaN, 不得转 0; 空 bin 返回 NaN + n=0 (§14)
- 敏感性只做描述性分析, 不得拟合趋势线 (§17)
"""
from __future__ import annotations

import math

import numpy as np

from conftest import code_nospace

LEVELS = [3, 5, 10, 21]
SEEDS = [122, 123, 124, 125, 126]
TRAINED = ("target_only", "source_finetune", "source_mmd_finetune")
ALL_GROUPS = TRAINED + ("damage_extrapolation", "const_mean_info")
STATS = "scripts/basilisk_b6/paired_statistics.py"


def test_b6_seed_level_bootstrap(b6_paired):
    """test_b6_seed_level_bootstrap (§23/§13) —— bootstrap 单位是 seed 级配对增益。"""
    bc = b6_paired["bootstrap_config"]
    assert str(bc["unit"]) == "seed_level_paired_gain"
    assert int(bc["n"]) == 5
    assert int(bc["samples"]) >= 5000
    assert bool(bc["forbid_endpoint_bootstrap"]) is True
    assert bool(bc["forbid_trajectory_count_significance"]) is True
    for k, lv in b6_paired["by_level"].items():
        for nm in ("gain_ft", "gain_mmd"):
            b = lv["gain"][nm]
            assert int(b["n_seeds"]) == 5, f"n={k} {nm} 的配对样本数不是 5"
            assert len(b["per_seed"]) == 5
            bs = b["bootstrap"]
            assert str(bs["unit"]) == "one_paired_difference_per_seed", \
                f"n={k} {nm} 的 bootstrap 单位错了: {bs['unit']}"
            assert int(bs["n"]) == 5
            assert int(bs["n_rep"]) >= 5000
            assert float(b["ci95_lower"]) == float(bs["ci_lower"])
            assert float(b["ci95_upper"]) == float(bs["ci_upper"])
            assert float(b["ci95_lower"]) <= float(b["ci95_upper"])


def test_b6_gain_definition_and_arithmetic(b6_paired, b6_metrics):
    """§13: gain = target_only − source_*, 且必须与 all_metrics 逐 seed 对得上。"""
    gd = str(b6_paired["gain_definition"])
    assert "target_only" in gd and "source_finetune" in gd
    assert "正数" in gd and "source" in gd
    for k, lv in b6_paired["by_level"].items():
        rows = {int(r["seed"]): r for r in b6_metrics["by_level"][k]["per_seed"]}
        for p in lv["per_seed"]:
            r = rows[int(p["seed"])]
            t0 = float(r["target_only"]["info_macro_rmse"])
            ft = float(r["source_finetune"]["info_macro_rmse"])
            md = float(r["source_mmd_finetune"]["info_macro_rmse"])
            assert math.isclose(float(p["gain_ft"]), t0 - ft, abs_tol=1e-9), \
                f"n={k} s{p['seed']} 的 gain_ft 与原始指标不符"
            assert math.isclose(float(p["gain_mmd"]), t0 - md, abs_tol=1e-9)
            assert bool(p["gain_ft_positive"]) is bool(t0 - ft > 0)


def test_b6_improve_count_matches_per_seed(b6_paired):
    """improve_count 必须是 per_seed 里严格为正的个数 —— 不得放宽成 >= 0。"""
    for k, lv in b6_paired["by_level"].items():
        for nm in ("gain_ft", "gain_mmd"):
            b = lv["gain"][nm]
            d = [float(x) for x in b["per_seed"]]
            assert int(b["improve_count"]) == sum(1 for x in d if x > 0), \
                f"n={k} {nm} 的 improve_count 与 per_seed 不符"
            assert math.isclose(float(b["mean"]), float(np.mean(d)), abs_tol=1e-8)
            assert math.isclose(float(b["median"]), float(np.median(d)),
                                abs_tol=1e-8)
            assert bool(b["ci_lower_positive"]) is bool(float(b["ci95_lower"]) > 0)


def test_b6_primary_is_n5(b6_paired, b6_config, b6_protocol_hash, b6_manifest):
    """test_b6_primary_is_n5 (§23/§8) —— PRIMARY 只能是 n=5。"""
    assert int(b6_paired["primary_level"]) == 5
    assert int(b6_manifest["primary_n_event_labeled"]) == 5
    pr = b6_config["b6"]["primary"]
    assert int(pr["n_event_labeled"]) == 5
    assert bool(pr["only_level_allowed_to_produce_formal_verdict"]) is True
    assert int(b6_protocol_hash["primary"]["n_event_labeled"]) == 5
    sec = b6_config["b6"]["secondary"]
    assert sorted(int(x) for x in sec["levels"]) == [3, 10, 21]
    assert 5 not in [int(x) for x in sec["levels"]]
    assert bool(sec["cannot_override_primary"]) is True
    assert sorted(int(x) for x in b6_paired["secondary_levels"]) == [3, 10, 21]
    assert bool(b6_paired["secondary_cannot_override_primary"]) is True


def test_b6_paired_statistics_covers_all_levels(b6_paired):
    """4 档全覆盖, seed 序列一致, 且非 fast 产物。"""
    assert sorted(int(k) for k in b6_paired["by_level"]) == LEVELS
    assert [int(s) for s in b6_paired["formal_seeds"]] == SEEDS
    for k, lv in b6_paired["by_level"].items():
        assert [int(p["seed"]) for p in lv["per_seed"]] == SEEDS, \
            f"n={k} 的 seed 序列不对"


def test_b6_report_metrics_present(b6_paired, b6_config):
    """§14: 每档必须报齐指标 —— 少一项就是漏报。

    per_seed 里主指标 (info macro RMSE) 直接以组名为键 (p["target_only"] 等),
    其余口径是组名字典 (info_pooled_rmse / full_macro_rmse / mae / corr /
    catastrophic_rate / warning_miss), psr 在 stability 子块内。
    """
    for k, lv in b6_paired["by_level"].items():
        p = lv["per_seed"][0]
        for field in ("info_pooled_rmse", "full_macro_rmse", "mae", "corr",
                      "stability", "catastrophic_rate", "warning_miss"):
            assert field in p, f"n={k} 缺指标 {field}"
        for g in TRAINED:
            assert g in p, f"n={k} 缺 {g} 的 info macro RMSE"
            assert g in p["info_pooled_rmse"], f"n={k} info_pooled_rmse 缺 {g}"
            assert "psr" in p["stability"][g], f"n={k} {g} 缺 psr"


def test_b6_nan_not_converted_to_zero(b6_paired, b6_lifetime):
    """§14 + 标准约束: 无真 EOL 的指标保持 NaN, 空 bin 返回 NaN + n=0。

    full caliber 含右删失点, 实践中恒为 NaN —— 若它变成 0.0, 说明有人把
    NaN 当 0 处理了, 那会把删失轨迹伪造成"预测完美"。
    """
    assert "nan_discipline" in b6_paired
    for k, lv in b6_paired["by_level"].items():
        for p in lv["per_seed"]:
            for g, v in p["full_macro_rmse"].items():
                assert v is None or math.isnan(float(v)) or float(v) > 0.0, \
                    f"n={k} {g} 的 full_macro_rmse = {v} —— 疑似 NaN 被转成 0"
    for k, lv in b6_lifetime["by_level"].items():
        for nm, blk in lv["gain_by_bin"].items():
            if int(blk["n_traj_in_bin"]) == 0:
                for g in TRAINED:
                    v = blk[g]
                    assert v is None or math.isnan(float(v)), \
                        f"n={k} bin {nm} 为空却给出了数值 {v} —— 伪造 0"
                assert bool(blk["gain_ft_nonneg"]) is False, \
                    f"n={k} bin {nm} 为空却判成 gain>=0"


def test_b6_no_nan_to_zero_in_stats_code():
    """静态: 统计脚本不得出现 NaN->0 的写法。"""
    src = code_nospace(STATS)
    for bad in ("np.nan_to_num", "nan_to_num(", "or 0.0", "fillna(0"):
        assert bad.replace(" ", "") not in src, f"统计脚本出现 NaN->0: {bad}"


def test_b6_no_endpoint_level_significance_in_code():
    """标准约束静态检查: 不得用端点 / 轨迹条数做显著性。"""
    src = code_nospace(STATS)
    for bad in ("ttest_ind", "ttest_rel", "mannwhitneyu", "wilcoxon",
                "n_points)", "bootstrap(traj"):
        assert bad.replace(" ", "") not in src, \
            f"统计脚本出现按端点/轨迹条数做检验: {bad}"


def test_b6_sensitivity_no_trend_line_fit(b6_paired):
    """test §17 —— 只有 4 个点, 只做描述性分析, 不得拟合趋势线。"""
    sens = b6_paired["sensitivity"]
    assert bool(sens["forbid_trend_line_fit"]) is True
    assert str(sens["x_axis"]) == "n_event_labeled"
    assert [int(x) for x in sens["x_values"]] == LEVELS
    for nm in ("source_finetune", "source_mmd_finetune"):
        rows = sens["by_method"][nm]
        assert [int(r["n_event_labeled"]) for r in rows] == LEVELS
        for r in rows:
            for f in ("mean", "median", "ci95_lower", "ci95_upper",
                      "improve_count"):
                assert f in r, f"{nm} 的敏感性行缺 {f}"
    src = code_nospace(STATS)
    for bad in ("np.polyfit", "polyfit(", "linregress", "LinearRegression",
                "curve_fit"):
        assert bad.replace(" ", "") not in src, f"统计脚本拟合了趋势线: {bad}"


def test_b6_lifetime_bins_from_frozen_edges(b6_lifetime, b6_manifest):
    """§14/§7: 寿命 bin 名与边界来自冻结件, 三处交叉一致。"""
    bins = b6_lifetime["bins"]
    assert list(bins["names"]) == ["short", "medium", "long"]
    assert list(bins["names"]) == list(b6_manifest["bin_names"])
    assert [float(x) for x in bins["edges"]] == \
        [float(x) for x in b6_manifest["bin_edges"]]
    assert bool(bins["recomputed_from_b6_test"]) is False, \
        "bin 边界被从 B6 test 重算 —— §7 禁止"
    assert "b21" in str(bins["source"]).lower()
    for k, lv in b6_lifetime["by_level"].items():
        assert sorted(lv["gain_by_bin"].keys()) == ["long", "medium", "short"]


def test_b6_warning_miss_reported_not_hidden(b6_warning):
    """标准约束: 不得只报成功检测而隐藏 miss rate。"""
    for k, lv in b6_warning["by_level"].items():
        for g in TRAINED:
            w = lv["by_group"][g]
            for f in ("warning_coverage", "miss_rate", "false_alarm_rate"):
                assert f in w, f"n={k} {g} 的告警指标缺 {f}"
            cov, miss = w["warning_coverage"]["mean"], w["miss_rate"]["mean"]
            if cov is not None and miss is not None and \
                    math.isfinite(float(cov)) and math.isfinite(float(miss)):
                assert float(miss) >= 0.0
        gi = lv["gate_inputs"]
        assert "ft_within_tol" in gi and "mmd_within_tol" in gi


def test_b6_hashes_consistent_across_artifacts(b6_paired, b6_lifetime,
                                               b6_warning, b6_metrics,
                                               b6_manifest, b6_protocol_hash):
    """所有统计产物必须挂在同一套冻结哈希上 —— 否则是混用了不同批次的数字。"""
    ph = str(b6_protocol_hash["protocol_sha256"])
    sm = str(b6_manifest["subset_manifest_sha256"])
    for art in (b6_paired, b6_lifetime, b6_warning, b6_metrics):
        assert str(art["protocol_sha256"]) == ph
        assert str(art["subset_manifest_sha256"]) == sm
        assert str(art["split_sha256"]) == str(b6_manifest["split_sha256"])
