"""tests/basilisk_b5/test_b5_training_fairness.py —— §6/§7 严格配对与验证集选点。

三个学习组在同一个 seed 下必须完全配对: 同一份数据、同一 batch 顺序、
同一优化器、同一 epoch 预算、同一早停指标、同一选点规则、同一评估器。
**唯一允许的差异**: 初始化 (随机 / 源域预训练) 与 MMD 开关。

checkpoint 选点只允许基于验证集 (§7)。禁止 test 选点、test 早停、
跨 seed 挑最优 checkpoint、跨 seed 换组。

逐组的训练元数据落在 `per_seed[i]["<group>_meta"]` 里 (组块本身只放指标),
故下面一律走 `r[f"{g}_meta"]`。
"""
from __future__ import annotations

import numpy as np

from conftest import code_nospace

GROUPS = ("target_only", "source_finetune", "source_mmd_finetune")


def test_b5_formal_seeds_fresh(b5_metrics, b5_config, b5_protocol_hash):
    """test_b5_formal_seeds_fresh (§21/§5) —— 全新正式 seed, 不复用开发 seed。"""
    seeds = [int(s) for s in b5_metrics["formal_seeds"]]
    assert seeds == [112, 113, 114, 115, 116]
    assert len(set(seeds)) == len(seeds), "正式 seed 有重复"
    forbidden = set(int(x) for x in b5_config["b5"]["forbid_reused_seeds"])
    assert forbidden & {102, 103, 104, 105, 106}, "未把 B2.1 的 seed 列入禁用"
    assert forbidden & {92, 93, 94}, "未把 B4X 的 seed 列入禁用"
    assert not (set(seeds) & forbidden), "正式 seed 与被禁用的开发 seed 相交"
    assert [int(s) for s in b5_protocol_hash["formal_seeds"]] == seeds


def test_b5_three_groups_only(b5_metrics, b5_config):
    """§4: 只训三组, 不得加第四种迁移方法。"""
    assert tuple(b5_metrics["trained_groups"]) == GROUPS
    assert bool(b5_config["b5"]["forbid_new_transfer_methods"]) is True
    for r in b5_metrics["per_seed"]:
        assert set(g for g in GROUPS if f"{g}_meta" in r) == set(GROUPS)


def test_b5_same_target_batch_order(b5_metrics):
    """test_b5_same_target_batch_order (§21/§6.1) —— 三组 batch 顺序逐 seed 相同。

    generator 是有状态的, 每个 epoch 都会推进它。若不在每组开训前重置, 第二、
    第三组拿到的就是接着上一组走的顺序, RMSE 差里会混进 batch 顺序的差异。
    """
    for r in b5_metrics["per_seed"]:
        sigs = {g: r[f"{g}_meta"]["batch_order_signature"] for g in GROUPS}
        assert len(set(sigs.values())) == 1, \
            f"seed {r['seed']} 三组 batch 顺序不同: {sigs}"
        seeds = {g: r[f"{g}_meta"]["batch_order_generator_seed"] for g in GROUPS}
        assert len(set(seeds.values())) == 1
        assert int(next(iter(seeds.values()))) == int(r["seed"]) * 1000 + 7
    # 不同 seed 之间必须不同 —— 否则说明 generator 没按 seed 变
    all_sigs = {r["target_only_meta"]["batch_order_signature"]
                for r in b5_metrics["per_seed"]}
    assert len(all_sigs) == len(b5_metrics["per_seed"])


def test_b5_batch_order_generator_is_explicit(b5_data_mod):
    """target loader 必须用显式 generator, 且每组开训前重置。"""
    src = code_nospace("scripts/basilisk_b5/data_b5.py")
    assert "torch.Generator" in src and "generator=gtr" in src
    assert "reset_target_batch_order" in src
    assert b5_data_mod.TARGET_GEN_OFFSET == 7
    runner = code_nospace("scripts/basilisk_b5/run_formal_transfer.py")
    assert "reset_target_batch_order(data)" in runner, \
        "runner 未在每组开训前重置 batch 顺序 —— 三组不是严格配对"


def test_b5_identical_initial_weights(b5_metrics):
    """§6.2: 三组共享结构 (12, 10), 同一 seed 下初始权重必须逐位相同。"""
    for r in b5_metrics["per_seed"]:
        shas = {g: r[f"{g}_meta"]["init_weights_sha256"] for g in GROUPS}
        assert len(set(shas.values())) == 1, \
            f"seed {r['seed']} 三组初始权重不一致: {shas} —— §6.2 要求逐位相同"
    # 不同 seed 的初始权重必须不同
    per_seed = {r["target_only_meta"]["init_weights_sha256"]
                for r in b5_metrics["per_seed"]}
    assert len(per_seed) == len(b5_metrics["per_seed"])


def test_b5_shared_n_features(b5_metrics, b5_config):
    """§6.3: 三组共享 n_features=12, 因此 B5 的 target_only 是重跑, 不等于 0.2441。"""
    f = b5_config["b5"]["fairness"]
    assert int(f["shared_n_features"]) == 12
    assert bool(f["target_only_is_rerun_not_b21_number"]) is True
    note = str(b5_metrics["architecture_note"])
    assert "重跑" in note or "rerun" in note.lower()
    for r in b5_metrics["per_seed"]:
        for g in GROUPS:
            assert int(r[f"{g}_meta"]["n_features"]) == 12
            assert int(r[f"{g}_meta"]["n_target"]) == 10


def test_b5_same_epoch_budget(b5_metrics):
    """§6: 不得因为某组失败而单独加 epoch。"""
    for r in b5_metrics["per_seed"]:
        budgets = {g: int(r[f"{g}_meta"]["max_epochs"]) for g in GROUPS}
        assert len(set(budgets.values())) == 1, f"epoch 预算不一致: {budgets}"
        assert next(iter(budgets.values())) == 8
        pats = {g: int(r[f"{g}_meta"]["early_stop_patience"]) for g in GROUPS}
        assert len(set(pats.values())) == 1
        for g in GROUPS:
            assert int(r[f"{g}_meta"]["best_epoch"]) <= 8


def test_b5_same_optimizer_and_loss_weights(b5_metrics):
    for r in b5_metrics["per_seed"]:
        for key in ("optimizer", "lr", "weight_decay", "post_eol_weight",
                    "capped_weight", "early_stop_metric", "rul_scale",
                    "input_schema", "split_sha256"):
            vals = {g: r[f"{g}_meta"][key] for g in GROUPS}
            assert len(set(map(str, vals.values()))) == 1, \
                f"seed {r['seed']} 的 {key} 三组不一致: {vals}"


def test_b5_val_only_checkpoint(b5_metrics):
    """test_b5_val_only_checkpoint (§21/§7) —— 选点只依据验证集。"""
    for r in b5_metrics["per_seed"]:
        for g in GROUPS:
            m = r[f"{g}_meta"]
            vm = m["val_metric"]
            assert str(vm["selection_basis"]) == "validation_only"
            assert "test" not in str(vm["name"]).lower()
            assert np.isfinite(float(vm["value"]))
            assert len(str(m["checkpoint_sha256"])) == 64
            # best_epoch 是 0-based, 允许 0 (第一个 epoch 就是最好的)
            assert int(m["best_epoch"]) >= 0


def test_b5_no_test_in_training_path():
    """训练与选点代码路径里不允许出现 test 选点痕迹。"""
    src = code_nospace("scripts/basilisk_b5/run_formal_transfer.py")
    for bad in ("best_test", "test_best", "argmin_test"):
        assert bad not in src, f"训练路径出现 test 选点痕迹: {bad}"


def test_b5_no_cross_seed_checkpoint_picking(b5_metrics):
    """§7: 每个 (组, seed) 都必须有自己的 checkpoint, 不得跨 seed 复用。"""
    seen = {}
    for r in b5_metrics["per_seed"]:
        for g in GROUPS:
            m = r[f"{g}_meta"]
            sha = str(m["checkpoint_sha256"])
            assert (g, sha) not in seen, \
                f"{g} 的 checkpoint 在 seed {r['seed']} 与 {seen[(g, sha)]} 重复"
            seen[(g, sha)] = r["seed"]
            assert f"_s{r['seed']}" in str(m["checkpoint_path"]), \
                "checkpoint 路径与 seed 不对应, 疑似跨 seed 换组"
            assert str(m["group"]) == g and int(m["seed"]) == int(r["seed"])


def test_b5_catastrophic_threshold_from_val_only(b5_metrics):
    """灾难阈值必须来自 target_only 的验证集, 且五组共用同一个 —— 不能各自定标。"""
    for r in b5_metrics["per_seed"]:
        th = r["catastrophic_threshold"]
        assert str(th["source"]) == \
            "target_only_validation_median_trajectory_rmse"
        assert str(th["split"]) == "validation"
        assert str(th["group"]) == "target_only"
        assert bool(th["derived_from_test"]) is False
        assert np.isfinite(float(th["threshold"]))
        assert int(th["n_val_traj_evaluable"]) > 0
        # 三组 + 两个只评估组的灾难率都用这同一个阈值算
        assert set(r["catastrophic"]) >= set(GROUPS)
        assert set(r["catastrophic"]) >= {"const_mean_info",
                                          "damage_extrapolation"}
