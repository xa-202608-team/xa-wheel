"""tests/basilisk_b2/test_b2_gate_discipline.py —— §6/§7/§8 闸门纪律。

守的是"闸门本身不被放水"这件事: 判定逻辑必须能判 FAIL、bootstrap 单位必须是
seed 而不是时间点、test 必须无法参与 checkpoint 选择、不得出现 source transfer。
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest


# --------------------------- §8 条件 7: 选型只看 val ---------------------------

def test_earlystop_signature_has_no_test_loader(b2_train_mod):
    """结构性保证: 训练入口拿不到 test loader。"""
    from src.experiments.run_groups import _train_with_early_stop
    sig = inspect.signature(_train_with_early_stop)
    for name in sig.parameters:
        assert "test" not in name.lower(), name


def test_train_b2_never_passes_test_loader(b2_train_mod):
    """train_target_only 的源码里不得出现 lte / test loader 的传递。"""
    src = inspect.getsource(b2_train_mod.train_target_only)
    assert "lte" not in src
    assert 'data["te"]' not in src


def test_history_contains_only_val_metrics(b2_metrics):
    """逐 epoch history 全是 val 量, 不含任何 test 量。"""
    for row in b2_metrics["per_seed"]:
        assert row["checkpoint_selection"] == "val_only"
        for h in row["history"]:
            for k in h:
                assert "test" not in k.lower(), k


def test_prepare_without_test_has_no_lte_key(b2_data_mod, b2_config, b2_split):
    """with_test=False 时结构上拿不到 test loader。"""
    d = b2_data_mod.prepare_b2(b2_config, b2_split, 72, with_test=False,
                               verbose=False)
    assert "lte" not in d
    d2 = b2_data_mod.prepare_b2(b2_config, b2_split, 72, with_test=True,
                                verbose=False)
    assert "lte" in d2


# --------------------------- §8 bootstrap 纪律 ---------------------------

def test_bootstrap_unit_is_seed_not_timepoint(b2_gate_mod, b2_metrics):
    """配对单位 = 每 seed 一个差值。n 必须等于 seed 数, 绝不是 endpoint 数。"""
    bs = b2_metrics["decision"]["bootstrap"]
    n_seeds = len(b2_metrics["seeds"])
    assert bs["n"] == n_seeds, (bs["n"], n_seeds)
    assert bs["unit"] == "one_paired_difference_per_seed"
    # endpoint 数是万级, 若 n 落到那个量级就是把时间点当独立样本了
    assert bs["n"] < 100


def test_bootstrap_is_deterministic(b2_gate_mod):
    """同 seed 同输入 -> 同 CI (可复现性要求)。"""
    d = [0.01, 0.02, -0.005, 0.03, 0.015]
    a = b2_gate_mod.paired_bootstrap(d, 500, 20260810)
    b = b2_gate_mod.paired_bootstrap(d, 500, 20260810)
    assert a == b


def test_bootstrap_ci_lower_negative_when_gain_negative(b2_gate_mod):
    """哨兵: 全负差值必须给出负的 CI 下界 (闸门能判 FAIL)。"""
    r = b2_gate_mod.paired_bootstrap([-0.5, -0.4, -0.6, -0.45, -0.55],
                                     1000, 20260810)
    assert r["ci_lower"] < 0
    assert r["mean"] < 0


def test_bootstrap_empty_gives_nan_not_zero(b2_gate_mod):
    """空输入 -> NaN, 不是 0。"""
    r = b2_gate_mod.paired_bootstrap([], 100, 1)
    assert r["n"] == 0
    assert not np.isfinite(r["mean"])
    assert not np.isfinite(r["ci_lower"])


# --------------------------- §8 判定逻辑 ---------------------------

def _row(gain, corr, psr, warn=True, sel="val_only"):
    return {"paired": {"gain_vs_const": gain, "better_than_const": gain > 0},
            "shape": {"macro_corr": corr, "psr": psr},
            "warning_valid": warn, "checkpoint_selection": sel}


@pytest.fixture()
def gc(b2_config):
    return b2_config["b2"]["gate"]


def test_decide_can_pass(b2_gate_mod, gc):
    """构造全部达标的输入 -> PASS。证明 PASS 分支可达 (不是死代码)。"""
    rows = [_row(0.05, 0.6, 0.8) for _ in range(5)]
    d = b2_gate_mod.decide(rows, gc)
    assert d["verdict"] == "B2_GENERALIZATION_PASS"
    assert d["n_passed"] == 7


def test_decide_fails_on_negative_gain(b2_gate_mod, gc):
    rows = [_row(-0.05, 0.6, 0.8) for _ in range(5)]
    d = b2_gate_mod.decide(rows, gc)
    assert d["verdict"] == "B2_GENERALIZATION_FAIL"


def test_decide_fails_on_low_psr(b2_gate_mod, gc):
    """PSR 低于 0.30 -> 条件 5 必须 FAIL (输出坍缩不得被放过)。"""
    rows = [_row(0.05, 0.6, 0.05) for _ in range(5)]
    d = b2_gate_mod.decide(rows, gc)
    assert d["verdict"] == "B2_GENERALIZATION_FAIL"
    assert not [c for c in d["conditions"] if c["id"] == 5][0]["pass"]


def test_decide_fails_on_bad_warning(b2_gate_mod, gc):
    rows = [_row(0.05, 0.6, 0.8, warn=False) for _ in range(5)]
    d = b2_gate_mod.decide(rows, gc)
    assert not [c for c in d["conditions"] if c["id"] == 6][0]["pass"]


def test_decide_fails_if_selection_not_val_only(b2_gate_mod, gc):
    rows = [_row(0.05, 0.6, 0.8, sel="test_peeked") for _ in range(5)]
    d = b2_gate_mod.decide(rows, gc)
    assert not [c for c in d["conditions"] if c["id"] == 7][0]["pass"]


def test_decide_requires_4_of_5(b2_gate_mod, gc):
    """3/5 优于基线 -> 条件 1 FAIL (阈值就是 4)。"""
    rows = [_row(0.05, 0.6, 0.8) for _ in range(3)] + \
           [_row(-0.05, 0.6, 0.8) for _ in range(2)]
    d = b2_gate_mod.decide(rows, gc)
    assert not [c for c in d["conditions"] if c["id"] == 1][0]["pass"]


def test_nan_psr_counts_as_fail_not_pass(b2_gate_mod, gc):
    """NaN 不得被当成达标 (NaN >= 0.3 在 numpy 里是 False, 这里钉死该行为)。"""
    rows = [_row(0.05, float("nan"), float("nan")) for _ in range(5)]
    d = b2_gate_mod.decide(rows, gc)
    for cid in (4, 5):
        assert not [c for c in d["conditions"] if c["id"] == cid][0]["pass"]


# --------------------------- §6/§7 边界 ---------------------------

def test_thresholds_come_from_config_not_hardcoded(b2_config_raw):
    """阈值走 config, 且是任务书给定值 —— 不得在代码里硬编码。"""
    g = b2_config_raw["b2"]["gate"]
    assert g["seeds"] == [72, 73, 74, 75, 76]
    assert g["min_improve_count"] == 4
    assert g["min_corr_count"] == 4
    assert g["min_psr_count"] == 4
    assert g["psr_min"] == 0.30
    assert g["metric"] == "info_macro_rmse"
    assert g["baseline"] == "const_mean_info"


def test_frozen_training_hyper_matches_s25(b2_config):
    """§7: 沿用 S2.5 已验证配置, 本阶段不调参。"""
    t = b2_config["training"]
    assert t["early_stop_metric"] == "info_macro_rmse"
    assert int(t["max_epochs"]) == 8
    assert int(t["early_stop_patience"]) == 2
    assert float(t["weight_decay"]) == 1e-3


def test_train_rejects_hyper_drift(b2_train_mod, b2_config, b2_split):
    """哨兵: 偷偷改超参必须被拒 (标 B2_CONFIG_INCOMPATIBLE 而非静默接受)。"""
    import copy
    c2 = copy.deepcopy(b2_config)
    c2["training"]["max_epochs"] = 40
    with pytest.raises(SystemExit, match="B2_CONFIG_INCOMPATIBLE"):
        b2_train_mod.train_target_only(c2, {"n_target": 10, "device": "cpu",
                                            "cap_eps": 1e-6, "censor_eta": 1.0,
                                            "ltr": None, "lva": None},
                                       72, "x")


def test_only_target_only_mode_allowed(b2_train_mod):
    """§6: 不跑 source transfer。传 source 模式必须直接报错。"""
    assert b2_train_mod.ALLOWED_MODES == ("target_only",)
    with pytest.raises(ValueError):
        b2_train_mod.train_target_only({}, {}, 1, "x", mode="source_finetune")


def test_no_source_transfer_in_results(b2_metrics):
    """产出里不得出现任何 source transfer 结果。"""
    dec = b2_metrics["decision"]
    assert dec["source_transfer_run"] is False
    assert dec["truncation_applied"] is False
    txt = str(b2_metrics["per_seed"])
    for bad in ("source_finetune", "source_mmd", "mmd_lambda"):
        assert bad not in txt, bad


def test_oracle_methods_excluded(b2_metrics, b2_baselines_mod):
    """persistence 类不可部署方法不得进入排名。"""
    assert "persistence" in b2_baselines_mod.EXCLUDED_ORACLE_METHODS
    assert "persistence" in b2_metrics["decision"]["excluded_oracle_methods"]
    for row in b2_metrics["per_seed"]:
        assert "persistence" not in row["paired"]


def test_no_truncation_applied(b2_config):
    """§3: 第一阶段用完整 train history, 不做 S3 truncation。"""
    assert b2_config["b2"]["use_full_train_history"] is True
    assert b2_config["b2"]["target_truncate_hi"] is None
