"""tests/basilisk_b5/test_b5_no_posthoc_tuning.py —— §1/§7/§16 事后调参禁令。

本阶段禁止重新调任何模型或迁移超参。这些测试用 tokenize 剥掉注释与字符串后
扫源码 —— 否则"注释里写明禁止 X"本身会被判成使用了 X。
"""
from __future__ import annotations

import numpy as np

from conftest import ROOT, code_nospace, code_only

B5_SCRIPTS = (
    "scripts/basilisk_b5/verify_baseline.py",
    "scripts/basilisk_b5/freeze_protocol.py",
    "scripts/basilisk_b5/data_b5.py",
    "scripts/basilisk_b5/run_formal_transfer.py",
    "scripts/basilisk_b5/analyze_paired_gain.py",
    "scripts/basilisk_b5/analyze_lifetime_bins.py",
    "scripts/basilisk_b5/analyze_warning_metrics.py",
    "scripts/basilisk_b5/summarize_b5.py",
)


def test_b5_no_hyperparam_search():
    """test_b5_no_hyperparam_search (§21/§1) —— 无任何搜索/扫参结构。"""
    for rel in B5_SCRIPTS:
        src = code_nospace(rel)
        for bad in ("GridSearchCV", "optuna", "ray.tune", "hyperopt",
                    "random_search", "grid_search", "param_grid",
                    "learning_rates", "lr_candidates"):
            assert bad not in src, f"{rel} 出现调参搜索: {bad}"


def test_b5_no_hardcoded_thresholds():
    """阈值必须走 config / 冻结协议, 不硬编码在分析脚本里。"""
    for rel in ("scripts/basilisk_b5/summarize_b5.py",
                "scripts/basilisk_b5/analyze_paired_gain.py"):
        src = code_nospace(rel)
        # 门槛数字只允许经 thr[...] / cfg[...] / b5[...] 读入
        for bad in (">0.02", ">=0.02", "+0.05", ">0.05", "<0.02"):
            assert bad not in src, f"{rel} 硬编码了门槛数字: {bad}"


def test_b5_gate_reads_frozen_protocol_not_live_config():
    """§9: 门槛必须从冻结的 protocol_hash 读, 不从可事后修改的 config 现读。"""
    txt = (ROOT / "scripts/basilisk_b5/summarize_b5.py").read_text(
        encoding="utf-8")
    # 键名只出现在下标字符串里, 会被 code_only 剥掉, 故这里扫原文
    assert 'proto["gate_thresholds"]' in txt, \
        "summarize 未从冻结的 protocol_hash 读门槛"
    assert 'cfg["protocol"]["hash_path"]' in txt
    src = code_nospace("scripts/basilisk_b5/summarize_b5.py")
    # 冻结值与 config 现值必须交叉核对, 不一致即 B5_INVALID
    assert "B5_INVALID" in txt
    assert "thr[k]!=v" in src or "thr[k]!=v" in src.replace(" ", "")


def test_b5_no_test_based_selection():
    """§7: 禁止 test 选点 / test 早停 / 跨 seed 挑最优。"""
    for rel in B5_SCRIPTS:
        src = code_nospace(rel)
        for bad in ("best_on_test", "select_by_test", "argmin_test",
                    "test_early_stop", "pick_best_seed"):
            assert bad not in src, f"{rel} 出现 test 选点: {bad}"


def test_b5_no_seed_dropping(b5_metrics, b5_gain, b5_config):
    """五个 seed 全部入统计, 不得剔除"不好的" seed。"""
    seeds = [int(s) for s in b5_config["b5"]["seeds"]]
    assert [int(r["seed"]) for r in b5_metrics["per_seed"]] == seeds
    assert [int(r["seed"]) for r in b5_gain["per_seed"]] == seeds
    for key in ("ft", "mmd"):
        assert int(b5_gain["bootstrap"][key]["n"]) == len(seeds)


def test_b5_no_extra_epochs_for_failing_group(b5_metrics):
    """§6: 不得因为某组表现差而单独加 epoch。"""
    for r in b5_metrics["per_seed"]:
        budgets = {g: int(r[f"{g}_meta"]["max_epochs"])
                   for g in b5_metrics["trained_groups"]}
        assert len(set(budgets.values())) == 1, f"epoch 预算被单独放宽: {budgets}"


def test_b5_no_bar_lowering_language_in_code():
    """禁止出现"放宽门槛"式的降级逻辑。"""
    for rel in B5_SCRIPTS:
        src = code_nospace(rel)
        for bad in ("relax_gate", "lower_threshold", "soft_gate",
                    "min_improve_count=3"):
            assert bad not in src, f"{rel} 出现降低门槛的痕迹: {bad}"


def test_b5_prohibition_prose_lives_in_module_constants():
    """禁令说明写在模块级常量里 —— 便于扫描器豁免, 也便于落进 JSON。"""
    for rel, names in (
        ("scripts/basilisk_b5/analyze_paired_gain.py",
         ("GAIN_DEFINITION", "BOOTSTRAP_DISCIPLINE")),
        ("scripts/basilisk_b5/analyze_warning_metrics.py",
         ("NAN_DISCIPLINE", "MISS_DISCIPLINE")),
        ("scripts/basilisk_b5/summarize_b5.py",
         ("PASS_MEANING", "FAIL_MEANING", "EXIT_DISCIPLINE")),
    ):
        txt = (ROOT / rel).read_text(encoding="utf-8")
        for n in names:
            assert f"{n} = (" in txt or f"{n} = " in txt, f"{rel} 缺常量 {n}"


def test_b5_does_not_modify_frozen_source_modules():
    """§1: 禁止修改模型结构 / MMD / 优化器 / 损失权重 / 早停 等冻结实现。"""
    for rel in B5_SCRIPTS:
        src = code_nospace(rel)
        for bad in ("src.models.tcn_encoder.TCN=", "mmd.MMD_LAMBDA=",
                    "run_groups._train_with_early_stop="):
            assert bad not in src, f"{rel} 猴补丁了冻结模块: {bad}"


def test_b5_b4x_signal_not_promoted(b5_summary, b4x_summary_ro):
    """B4X 的稳定化信号是探索性的, 不得被继承为正式 positive-transfer 结论。"""
    assert str(b4x_summary_ro["verdict"]) == "B4X_TRANSFER_STABILIZATION_SIGNAL"
    assert bool(b4x_summary_ro["exploratory_only"]) is True
    assert bool(b5_summary["b4x_must_not_be_promoted"]) is True
    ff = b5_summary["frozen_facts"]
    assert "探索" in str(ff) or "exploratory" in str(ff).lower()


def test_b5_b2_verdict_not_retroactively_rewritten(b5_summary):
    """B2_GENERALIZATION_FAIL 在其自身划分上保持终局。"""
    assert str(b5_summary["b2_verdict_status"]) == \
        "B2_GENERALIZATION_FAIL_REMAINS_FINAL"


def test_b5_target_only_number_not_confused_with_b21(b5_summary, b5_metrics):
    """§6.3: B5 的 target_only 是重跑, 不得当成 B2.1 的 0.2441。"""
    note = str(b5_metrics["architecture_note"])
    assert "0.2441" in note or "重跑" in note
    t0 = float(b5_summary["main_table"]["target_only"]["info_macro_rmse"])
    assert np.isfinite(t0)


def test_b5_stops_after_b5(b5_summary):
    """§20: 不自动跑 B6, 不自动跑低数据矩阵。"""
    assert bool(b5_summary["b6_auto_run"]) is False
    assert bool(b5_summary["low_data_matrix_auto_run"]) is False
    assert bool(b5_summary["stop_after_b5"]) is True
    assert bool(b5_summary["forbid_auto_run_b6"]) is True
    for rel in ("checkpoints/basilisk_b5/low_data_matrix.json",
                "checkpoints/basilisk_b5/truncation_matrix.json",
                "checkpoints/basilisk_b6/metrics.json"):
        assert not (ROOT / rel).exists(), f"{rel} 被自动生成 —— 违反 §20"
