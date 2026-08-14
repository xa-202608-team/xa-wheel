"""tests/basilisk_b5/test_b5_paired_statistics.py —— §8/§9/§10 配对统计与 Gate。

- gain 的符号定义固定为 target - source, 正数 = 迁移改进 (§8)
- bootstrap 的重抽单位必须是 **seed 级配对差**, n=5, ≥5000 次, 固定种子 (§10)
- 禁止用端点 bootstrap 替代 seed bootstrap; 禁止把时间点当独立样本;
  禁止用轨迹数放大显著性
- §9 八条门槛逐条核对, 且必须全部满足才算正转移
"""
from __future__ import annotations

import numpy as np

from conftest import code_nospace, code_only

ANALYSIS_SCRIPTS = ("scripts/basilisk_b5/analyze_paired_gain.py",
                    "scripts/basilisk_b5/summarize_b5.py")


def test_b5_gain_sign_definition(b5_metrics, b5_gain):
    """test_b5_gain_sign_definition (§21/§8) —— 逐 seed 重算 gain, 符号必须一致。"""
    for r in b5_metrics["per_seed"]:
        t0 = float(r["target_only"]["info_macro_rmse"])
        ft = float(r["source_finetune"]["info_macro_rmse"])
        md = float(r["source_mmd_finetune"]["info_macro_rmse"])
        g = r["gains"]
        assert abs(float(g["gain_ft"]) - (t0 - ft)) < 1e-12
        assert abs(float(g["gain_mmd"]) - (t0 - md)) < 1e-12
        assert "正数" in str(g["definition"]) or "positive" in str(g["definition"])
    # 分析脚本里的逐 seed 记录也必须与 metrics 对得上
    for r in b5_gain["per_seed"]:
        s = next(x for x in b5_metrics["per_seed"] if x["seed"] == r["seed"])
        assert abs(float(r["gain_ft"]) - float(s["gains"]["gain_ft"])) < 1e-12
        assert abs(float(r["gain_mmd"]) - float(s["gains"]["gain_mmd"])) < 1e-12


def test_b5_seed_level_bootstrap(b5_gain):
    """test_b5_seed_level_bootstrap (§21/§10) —— 重抽单位是 seed 级配对差。"""
    for key in ("ft", "mmd"):
        b = b5_gain["bootstrap"][key]
        assert int(b["n"]) == 5, "bootstrap 的 n 必须是 seed 数 5, 不是轨迹数"
        assert int(b["n_rep"]) >= 5000
        assert "seed" in str(b["unit"]).lower()
        assert int(b5_gain["bootstrap"]["seed"]) == 20260814
        for k in ("mean", "ci_lower", "ci_upper"):
            assert np.isfinite(float(b[k]))
        assert float(b["ci_lower"]) <= float(b["mean"]) <= float(b["ci_upper"])


def test_b5_bootstrap_unit_not_timepoints():
    """§10 禁令: 不得把时间点或轨迹当独立样本来放大显著性。"""
    for rel in ANALYSIS_SCRIPTS:
        src = code_only(rel)
        for bad in ("endpoint_bootstrap", "bootstrap_timepoints",
                    "n_traj_as_n", "pooled_bootstrap"):
            assert bad not in src, f"{rel} 出现被禁的 bootstrap 单位: {bad}"


def test_b5_gain_aggregates_consistent(b5_gain):
    """mean / median / std / improve_count 必须能由逐 seed gain 复算出来。"""
    for key, fld in (("ft", "gain_ft"), ("mmd", "gain_mmd")):
        vals = np.array([float(r[fld]) for r in b5_gain["per_seed"]], float)
        gi = b5_gain["gate_inputs"][key]
        assert len(vals) == 5
        assert abs(float(gi["mean"]) - float(np.mean(vals))) < 1e-9
        assert abs(float(gi["median"]) - float(np.median(vals))) < 1e-9
        assert int(gi["improve_count"]) == int(np.sum(vals > 0))


def test_b5_positive_transfer_gate_exact(b5_summary, b5_protocol_hash):
    """test_b5_positive_transfer_gate_exact (§21/§9) —— 八条, 全满足才算正转移。"""
    thr = b5_protocol_hash["gate_thresholds"]
    assert int(thr["min_improve_count"]) == 4
    assert int(thr["n_seeds"]) == 5
    assert float(thr["corr_tol"]) == 0.02
    assert float(thr["warning_miss_tol"]) == 0.05
    assert int(thr["min_lifetime_bins_gain_positive"]) == 2
    assert int(thr["n_lifetime_bins"]) == 3
    for k in ("require_mean_positive", "require_median_positive",
              "require_ci_lower_positive", "require_catastrophic_not_worse"):
        assert bool(thr[k]) is True

    for meth in ("source_finetune", "source_mmd_finetune"):
        d = b5_summary["decision"][meth]
        assert int(d["n_conditions"]) == 8, "§9 是八条, 不多不少"
        assert bool(d["all_required"]) is True
        assert [c["id"] for c in d["conditions"]] == [1, 2, 3, 4, 5, 6, 7, 8]
        # 判定必须与 conditions 逐条重算一致 —— 不允许手改结论
        recomputed = all(bool(c["passed"]) for c in d["conditions"])
        assert recomputed == bool(d["positive_transfer"])
        assert int(d["n_passed"]) == sum(1 for c in d["conditions"] if c["passed"])
        if not recomputed:
            assert "NO_POSITIVE_TRANSFER" in str(d["verdict"])
            assert d["failed_conditions"], "判负却没记下哪条不过"


def test_b5_gate_thresholds_not_lowered(b5_summary, b5_protocol_hash):
    """§9 禁止降低门槛 —— summary 用的阈值必须等于冻结的那份。"""
    assert b5_summary["gate_thresholds"] == b5_protocol_hash["gate_thresholds"]


def test_b5_verdict_is_one_of_four_combinations(b5_summary):
    """§19: 最终判定必须落在 A/B/C/D 四种组合之一。"""
    vc = b5_summary["verdict_combination"]
    assert str(vc["combination"]) in ("A", "B", "C", "D")
    ft = bool(b5_summary["positive_transfer"]["ft"])
    mmd = bool(b5_summary["positive_transfer"]["mmd"])
    exp = {(True, True): "A", (True, False): "B",
           (False, True): "C", (False, False): "D"}[(ft, mmd)]
    assert str(vc["combination"]) == exp
    if exp == "D":
        assert str(b5_summary["verdict"]) == "B5_NO_POSITIVE_TRANSFER"


def test_b5_ft_and_mmd_judged_independently(b5_summary):
    """§9: FT 与 MMD 各自独立判定, 不得一荣俱荣。"""
    a = b5_summary["decision"]["source_finetune"]
    b = b5_summary["decision"]["source_mmd_finetune"]
    assert a["method"] != b["method"]
    assert str(a["verdict"]) != "" and str(b["verdict"]) != ""
    assert "ft" in str(a["verdict"]).lower() or "FT" in str(a["verdict"])
    assert "mmd" in str(b["verdict"]).lower() or "MMD" in str(b["verdict"])


def test_b5_psr_reported_but_not_in_gate(b5_gain, b5_summary):
    """§17: PSR 是诊断项, 三个量一起报, 但不新增到正式 Gate。"""
    st = b5_gain["output_stability"]
    assert bool(st["in_gate"]) is False
    for g, blk in st["by_group"].items():
        for k in ("psr_mean", "pred_std_mean", "true_std_mean",
                  "high_variance_warning"):
            assert k in blk, f"{g} 缺 §17 必报项 {k}"
    ids = [c["name"] for c in
           b5_summary["decision"]["source_finetune"]["conditions"]]
    assert not any("psr" in n.lower() for n in ids), "PSR 不得进入正式 Gate"


def test_b5_no_nan_silently_zeroed(b5_gain, b5_warning):
    """NaN 必须保留为 NaN + n_evaluable, 不得转成 0。"""
    src = code_nospace("scripts/basilisk_b5/analyze_warning_metrics.py")
    for bad in ("nan_to_num", "fillna(0)", "or0.0"):
        assert bad not in src, f"出现把 NaN 变 0 的写法: {bad}"
    for g, blk in b5_warning["by_group"].items():
        ph = blk["prognostic_horizon"]
        assert "n_evaluable_traj" in ph or "n_evaluable" in ph, \
            f"{g} 的 PH 未记录可评估数量"


def test_b5_no_persistence_in_ranking(b5_summary):
    """persistence / true_rul_lookup 是 oracle, 不得进入可部署方法排名。"""
    excl = set(b5_summary["engineering_recommendation"]["excluded_oracle_methods"])
    assert {"persistence", "true_rul_lookup"} <= excl
    ranked = {r["method"] for r in
              b5_summary["engineering_recommendation"]["ranking"]}
    assert not (ranked & excl), f"排名里混进了 oracle: {ranked & excl}"
