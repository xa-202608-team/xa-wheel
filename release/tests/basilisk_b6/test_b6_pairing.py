"""tests/basilisk_b6/test_b6_pairing.py —— §9/§11/§12 配对公平性与 checkpoint 纪律。

- 正式 seed 必须是 [122,123,124,125,126], 且不得复用历史阶段 seed (§9)
- 同 n_event + 同 seed 下三组的 train IDs / batch 顺序 / 初始权重必须同一,
  唯一允许的差别是 source 初始化与 MMD 项 (§11)
- checkpoint 选择全部 validation-only; 禁止用 test 早停 / 选点 / 换 seed 重跑 (§12)
"""
from __future__ import annotations

import pytest

from conftest import code_nospace

FORMAL_SEEDS = [122, 123, 124, 125, 126]
FORBIDDEN_SEEDS = [72, 73, 74, 75, 76, 92, 93, 94,
                   102, 103, 104, 105, 106, 112, 113, 114, 115, 116]
TRAINED = ("target_only", "source_finetune", "source_mmd_finetune")
LEVELS = (3, 5, 10, 21)
RUNNER = "scripts/basilisk_b6/run_formal_matrix.py"


def _cells(mx: dict):
    """遍历所有 (n_event, seed_row) —— 与 all_metrics.json 的结构解耦。"""
    for k, lv in mx["by_level"].items():
        for row in lv["per_seed"]:
            yield int(k), row


def test_b6_new_formal_seeds(b6_config, b6_protocol_hash):
    """test_b6_new_formal_seeds (§23/§9) —— 新正式 seed, 不复用历史 seed。"""
    assert [int(s) for s in b6_config["b6"]["seeds"]] == FORMAL_SEEDS
    assert [int(s) for s in b6_protocol_hash["formal_seeds"]] == FORMAL_SEEDS
    fb = [int(s) for s in b6_config["b6"]["forbid_reused_seeds"]]
    assert sorted(fb) == sorted(FORBIDDEN_SEEDS)
    assert not (set(FORMAL_SEEDS) & set(fb)), "正式 seed 与禁用 seed 有交叠"
    assert sorted(int(s) for s in b6_protocol_hash["forbidden_seeds"]) == \
        sorted(FORBIDDEN_SEEDS)


def test_b6_metrics_use_only_formal_seeds(b6_metrics):
    """跑出来的 seed 必须与冻结清单逐一相同, 不多不少。"""
    assert [int(s) for s in b6_metrics["formal_seeds"]] == FORMAL_SEEDS
    for n, row in _cells(b6_metrics):
        assert int(row["seed"]) in FORMAL_SEEDS, \
            f"n={n} 出现非正式 seed {row['seed']}"
    for k, lv in b6_metrics["by_level"].items():
        got = [int(r["seed"]) for r in lv["per_seed"]]
        assert got == FORMAL_SEEDS, f"n={k} 的 seed 序列 {got} 不是冻结序列"


def test_b6_matrix_is_complete_60_cells(b6_metrics):
    """§24-5: 4 档 × 3 方法 × 5 seed = 60 个训练 cell, 且非 fast 模式。"""
    assert bool(b6_metrics["fast"]) is False, "正式矩阵不得在 fast 模式下产出"
    assert int(b6_metrics["n_cells_trained"]) == 60
    assert sorted(int(k) for k in b6_metrics["by_level"].keys()) == sorted(LEVELS)
    n_seen = 0
    for n, row in _cells(b6_metrics):
        for g in TRAINED:
            assert g in row, f"n={n} seed {row['seed']} 缺 {g}"
            n_seen += 1
    assert n_seen == 60


def test_b6_same_batch_order_across_methods(b6_metrics):
    """test_b6_same_batch_order_across_methods (§23/§11)。"""
    for n, row in _cells(b6_metrics):
        sigs = {g: str(row[f"{g}_meta"]["batch_order_signature"])
                for g in TRAINED}
        assert len(set(sigs.values())) == 1, \
            f"n={n} seed {row['seed']} 三组 batch 顺序不同: {sigs}"
        seeds = {g: int(row[f"{g}_meta"]["batch_order_generator_seed"])
                 for g in TRAINED}
        assert len(set(seeds.values())) == 1, \
            f"n={n} seed {row['seed']} 三组 batch generator seed 不同: {seeds}"
        assert bool(row["fairness"]["same_batch_order"]) is True


def test_b6_same_init_signature(b6_metrics):
    """test_b6_same_init_signature (§23/§11) —— 同 cell 内初始权重同一。

    注意: source 组的初始权重来自源域 checkpoint, target_only 也必须从**同一**
    初始权重出发才是同一起跑线; 这正是 runner 的 fairness.same_init_weights。
    """
    for n, row in _cells(b6_metrics):
        sigs = {g: str(row[f"{g}_meta"]["init_weights_sha256"])
                for g in TRAINED}
        assert len(set(sigs.values())) == 1, \
            f"n={n} seed {row['seed']} 三组初始权重不同: {sigs}"
        assert bool(row["fairness"]["same_init_weights"]) is True


def test_b6_same_train_ids_across_methods(b6_metrics, b6_manifest):
    """§11: 同 cell 三组必须用**完全相同**的 train IDs, 且与冻结子集一致。"""
    for n, row in _cells(b6_metrics):
        sigs = {g: str(row[f"{g}_meta"]["train_ids_signature"])
                for g in TRAINED}
        assert len(set(sigs.values())) == 1, \
            f"n={n} seed {row['seed']} 三组 train IDs 不同"
        assert bool(row["fairness"]["same_train_ids"]) is True
        assert int(row["n_train"]) == int(
            b6_manifest["levels"][str(n)]["n_train_total"])


def test_b6_fairness_pass_every_cell(b6_metrics):
    """§15 条件 9 的输入: 每个 cell 的 fairness 必须 PASS。"""
    bad = [(n, row["seed"]) for n, row in _cells(b6_metrics)
           if not bool(row["fairness"]["pass"])]
    assert bad == [], f"公平性未通过的 cell: {bad}"


def test_b6_only_allowed_differences(b6_config):
    """§11: 唯一允许的差别是 source 初始化 与 (仅 MMD 组的) MMD 项。"""
    fair = b6_config["b6"]["fairness"]
    for k in ("same_train_ids", "same_val_ids", "same_test_ids",
              "same_batch_order_signature", "same_init_weight_signature",
              "same_optimizer", "same_training_budget", "same_early_stopping",
              "same_checkpoint_rule", "same_censoring_loss", "same_evaluator",
              "forbid_extra_epochs_for_failing_group"):
        assert bool(fair[k]) is True, f"fairness.{k} 必须为 true"
    ad = fair["allowed_differences"]
    assert set(ad.keys()) == {"source_finetune", "source_mmd_finetune"}
    assert str(ad["source_finetune"]) == "source_initialization"
    assert "mmd" in str(ad["source_mmd_finetune"]).lower()
    assert int(fair["shared_n_features"]) == 12


def test_b6_validation_only_checkpoint(b6_config, b6_metrics):
    """test_b6_validation_only_checkpoint (§23/§12)。"""
    ck = b6_config["b6"]["checkpoint"]
    assert str(ck["selection"]) == "validation_only"
    for k in ("forbid_test_based_early_stopping", "forbid_test_based_selection",
              "forbid_retrain_after_seeing_test",
              "forbid_rerun_failing_looking_seed", "forbid_auto_retry",
              "collapse_is_a_formal_result",
              "invalid_cell_only_on_code_error_or_nan",
              "forbid_seed_swap_on_invalid"):
        assert bool(ck[k]) is True, f"checkpoint.{k} 必须为 true"
    for n, row in _cells(b6_metrics):
        for g in TRAINED:
            meta = row[f"{g}_meta"]
            assert str(meta["val_metric"]["selection_basis"]) == "validation_only"
            assert str(meta["early_stop_metric"]) == "info_macro_rmse"
            assert int(meta["best_epoch"]) >= 0
            for rec in ck["record_per_group"]:
                assert str(rec) in meta, f"n={n} {g} 缺记录项 {rec}"


def test_b6_no_test_based_selection_in_code():
    """§12 静态禁令: runner 不得依据 test 做早停 / 选 checkpoint / 自动重试。"""
    src = code_nospace(RUNNER)
    for bad in ("best_test", "test_early_stop", "min(test", "argmin(test",
                "if test_rmse<best", "retry", "while attempt"):
        assert bad.replace(" ", "") not in src, f"runner 出现 test 选点/重试痕迹: {bad}"


def test_b6_no_extra_budget_for_source_groups(b6_metrics, b6_protocol_hash):
    """§11: 训练预算同一 —— 不得给表现差的组放宽 epoch 上限或 patience。

    注意: MMD 组是三阶段 (S1/S2/S3), 每阶段各自受 max_epochs 约束, 因此累计
    epochs_run 天然大于单阶段组, 不能拿总数做上限比较 —— 比的是**预算参数**。
    """
    hp = b6_protocol_hash["training_hyper_frozen"]
    cap, pat = int(hp["max_epochs"]), int(hp["early_stop_patience"])
    for n, row in _cells(b6_metrics):
        for g in TRAINED:
            meta = row[f"{g}_meta"]
            assert int(meta["max_epochs"]) == cap, \
                f"n={n} {g} s{row['seed']} 的 max_epochs 被改成 {meta['max_epochs']}"
            assert int(meta["early_stop_patience"]) == pat
            assert float(meta["lr"]) == float(hp["finetune_lr"])
            assert float(meta["weight_decay"]) == float(hp["weight_decay"])
            assert str(meta["optimizer"]) == "Adam"
            n_st = len(meta["stages"])
            assert int(meta["budget"]["epochs_run"]) <= cap * n_st, \
                f"n={n} {g} s{row['seed']} epoch 总数超出 {n_st} 阶段预算"


def test_b6_mmd_lambda_frozen_in_all_cells(b6_metrics, b6_protocol_hash):
    """§10/§26 禁令: MMD lambda 全程冻结, 且只在 MMD 组生效。"""
    lam = float(b6_protocol_hash["mmd_lambda_frozen"])
    assert lam == 1.0
    seen = set()
    for n, row in _cells(b6_metrics):
        for st in row["source_mmd_finetune_meta"]["stages"]:
            if bool(st.get("use_mmd")):
                seen.add(float(st["mmd_lambda"]))
                assert bool(st["mmd_lambda_frozen"]) is True
        assert bool(row["source_mmd_finetune_meta"]["use_mmd"]) is True
        for g in ("target_only", "source_finetune"):
            assert bool(row[f"{g}_meta"]["use_mmd"]) is False, \
                f"n={n} {g} 不应启用 MMD"
    assert seen == {lam}, f"MMD lambda 出现多个取值: {seen}"


def test_b6_source_init_only_for_source_groups(b6_metrics):
    """§11: 只有两个 source 组从源域 checkpoint 初始化。"""
    for n, row in _cells(b6_metrics):
        assert bool(row["target_only_meta"]["init_from_source_checkpoint"]) is False
        for g in ("source_finetune", "source_mmd_finetune"):
            assert bool(row[f"{g}_meta"]["init_from_source_checkpoint"]) is True


def test_b6_no_invalid_cells_or_documented(b6_metrics):
    """§12: cell 只在明确代码错误 / NaN 时标 INVALID, 且不得换 seed。"""
    inv = [(n, row["seed"]) for n, row in _cells(b6_metrics)
           if str(row.get("status", "OK")).upper() == "INVALID"]
    if inv:
        pytest.fail(f"存在 INVALID cell {inv} —— 必须在 limitations.md 中正式记录, "
                    "且不得换 seed 重跑")
