"""tests/basilisk_b2/test_b2_censor_contract.py —— §4/§9 删失契约。

本文件是 B2 最重要的一组测试: 79 条右删失轨迹占数据的 53%, 一旦把它们的
NaN 真值当 0、或把观测截止当 EOL, 所有指标都会变好看而且**无法从数字上察觉**。
"""
from __future__ import annotations

import numpy as np


def test_censored_rul_is_nan_in_source_data(b2_feature_h5):
    """前置事实: 删失轨迹的 rul 在 h5 里就是 NaN, 且 rul_lower_bound 有限。"""
    import h5py
    with h5py.File(b2_feature_h5, "r") as f:
        n_cen = 0
        for k in sorted(f.keys()):
            g = f[k]
            if bool(int(g.attrs["event_observed"])):
                continue
            n_cen += 1
            r = np.asarray(g["rul"][:], float)
            lb = np.asarray(g["rul_lower_bound"][:], float)
            assert np.isnan(r).all(), f"{k} 删失轨迹的 rul 应全为 NaN"
            assert np.isfinite(lb).all(), f"{k} 下界必须有限 (它是唯一监督信号)"
        assert n_cen == 79


def test_eval_loader_keeps_censored_rul_as_nan(b2_data_mod, b2_config, b2_split):
    """评估侧必须保留 NaN: 删失点落不进 info 掩码, 于是 n 变小而不是被当 0 评分。"""
    d = b2_data_mod.prepare_b2(b2_config, b2_split, 72, with_test=True,
                               verbose=False)
    ds = d["lte"].dataset
    n_nan = 0
    n_tot = 0
    for i in range(0, len(ds), max(1, len(ds) // 400)):
        _x, _h, r = ds[i]
        v = r.numpy()
        n_tot += v.size
        n_nan += int(np.isnan(v).sum())
    assert n_nan > 0, "test 集必须含删失点的 NaN 真值 (它们不该被填成 0)"
    assert n_nan < n_tot, "test 集也必须含 event 轨迹的有限真值"


def test_train_loader_never_feeds_nan_but_marks_censored(
        b2_data_mod, b2_config, b2_split):
    """训练侧: 不能把 NaN 喂进梯度, 但删失样本必须被标记为 ev=False。

    这两件事必须同时成立 —— 只做前者 (用下界填充却标 ev=True) 就等于
    伪造了完整 RUL 监督, 正是 §4 禁止的。
    """
    d = b2_data_mod.prepare_b2(b2_config, b2_split, 72, verbose=False)
    ds = d["ltr"].dataset
    seen_cen = False
    for i in range(0, len(ds), max(1, len(ds) // 300)):
        item = ds[i]
        assert len(item) == 5, "训练集必须产出 5 元组 (含 ev/lb)"
        _x, _h, r, ev, lb = (v.numpy() for v in item)
        assert not np.isnan(r).any(), "训练标签不得含 NaN"
        assert not np.isnan(lb).any(), "下界不得含 NaN"
        if not ev.all():
            seen_cen = True
            # 删失窗的 rul 通道应等于其下界 (占位), 而非任何"猜出来的"真值
            assert np.allclose(r[ev < 0.5], lb[ev < 0.5])
    assert seen_cen, "训练集必须包含删失样本 (它们占 53%, 丢掉就是丢一半数据)"


def test_censored_loss_excludes_huber(b2_config):
    """删失样本绝不进 Huber: 只经 one-sided hinge。直接对损失函数验证。"""
    import torch

    from src.experiments.run_groups import _censored_rul_loss
    pred = torch.tensor([[0.5, 0.5]])
    true = torch.tensor([[0.9, 0.9]])
    lb = torch.tensor([[0.1, 0.1]])
    # 全删失且 pred >= lb -> 损失恰为 0 (真值 0.9 完全没被使用)
    ev0 = torch.tensor([[0.0, 0.0]])
    loss, L_O, L_C, n_obs, n_cen = _censored_rul_loss(
        pred, true, ev0, lb, 1.0, 1.0, 1.0, 1e-6, 1.0)
    assert n_obs == 0 and n_cen == 2
    assert float(loss) == 0.0, "pred 高于下界时删失项必须恰为 0"
    # 全观测 -> 损失由 Huber 主导, 必须 > 0
    ev1 = torch.tensor([[1.0, 1.0]])
    loss1, _, _, n_obs1, n_cen1 = _censored_rul_loss(
        pred, true, ev1, lb, 1.0, 1.0, 1.0, 1e-6, 1.0)
    assert n_obs1 == 2 and n_cen1 == 0
    assert float(loss1) > 0.0


def test_censor_report_never_turns_nan_into_zero(b2_metrics):
    """§9: censored 分层的 RUL 指标必须是 NaN + n=0, 绝不是 0。"""
    for row in b2_metrics["per_seed"]:
        for meth in ("target_only", "const_mean_info", "damage_extrapolation"):
            cr = row[meth]["censor_report"]
            cen = cr["censored_test"]
            v = cen["info_macro_rmse"]
            assert v is None or not np.isfinite(float(v)), \
                f"{meth} 删失分层 RMSE 应为 NaN, 实得 {v}"
            assert cen["n_evaluable_rul_points"] == 0
            assert cen["n_endpoints"] > 0, "删失分层本身必须非空"
            # event 分层必须有真数字, 否则整个报告没有意义
            assert np.isfinite(float(cr["event_observed_test"]
                                     ["info_macro_rmse"]))


def test_censored_tau_is_not_life_fraction(b2_metrics, b2_eval_mod):
    """删失轨迹的 tau 必须自报为观测进度而非寿命比例。"""
    tag = b2_eval_mod.CENSORED_TAU_SEMANTICS
    assert "not_life_fraction" in tag
    for row in b2_metrics["per_seed"]:
        cen = row["target_only"]["censor_report"]["censored_test"]
        assert cen["tau_semantics"] == tag


def test_axis_gives_censored_no_eol(b2_eval_mod, b2_config):
    """§4: b2_endpoint_axis 对删失轨迹必须给 eol_idx=None。"""
    axis = b2_eval_mod.b2_endpoint_axis(b2_config)
    assert len(axis) == 150
    n_cen = 0
    for a in axis.values():
        if a["event_observed"]:
            assert isinstance(a["eol_idx"], int) and a["eol_idx"] > 0
        else:
            n_cen += 1
            assert a["eol_idx"] is None, "删失轨迹不得有 EOL"
            assert a["tau_semantics"] == b2_eval_mod.CENSORED_TAU_SEMANTICS
    assert n_cen == 79
