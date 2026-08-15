"""tests/basilisk_b2/test_b2_input_contract.py —— §5 输入契约。

核心不变量: hi_damage_obs **不进普通输入**, 只作 HI 头的监督目标。
为什么这条必须自动化守: B1.9 已证明 HI_D_obs 是隐藏累积损伤的**重算**
(corr = 1.00000000), 把它放进输入等于直接把答案喂给模型 —— 得到的
任何"泛化能力"都是循环论证。
"""
from __future__ import annotations

import json

import numpy as np


def test_xt_cols_exclude_hi_and_truth(b2_feature_h5, b2_data_mod):
    """x_T 的 10 列里不含任何 HI / 真值 / 标签列。"""
    import h5py
    with h5py.File(b2_feature_h5, "r") as f:
        cols = list(json.loads(f.attrs["xt_cols"]))
    assert len(cols) == 10
    for bad in b2_data_mod.FORBIDDEN_PLAIN_INPUT_COLS:
        assert bad not in cols, f"{bad} 不得出现在 x_T"
    # b_hat 是遥测估计量, 属 B1.9 冻结核心输入 —— 必须仍在
    assert "b_hat" in cols
    assert "b_true" not in cols


def test_assert_no_hi_in_input_actually_fails(b2_data_mod, tmp_path):
    """哨兵: 校验函数必须真的能失败 (否则它是装饰品)。"""
    import h5py
    import pytest
    p = tmp_path / "bad.h5"
    with h5py.File(p, "w") as f:
        f.attrs["xt_cols"] = json.dumps(["I_m", "hi_damage_obs"])
    with pytest.raises(AssertionError):
        b2_data_mod.assert_no_hi_in_input(p)


def test_model_input_dim_is_core_only(b2_data_mod, b2_config, b2_split):
    """输入维度 = 10 (B1.9 冻结的 x_T 列数), 没有被偷偷加列。"""
    d = b2_data_mod.prepare_b2(b2_config, b2_split, 72, verbose=False)
    assert d["n_target"] == 10
    x, _h, _r, _e, _l = d["ltr"].dataset[0]
    assert x.shape[-1] == 10
    assert d["hi_key"] == "hi_damage_obs"


def test_mission_features_not_in_input_this_stage(b2_config, b2_data_mod,
                                                  b2_split):
    """§B3 的消融变量在 B2 必须固定关闭 —— 否则 B3 无从比较。"""
    assert b2_config["b2"]["mission_features_in_input"] is False
    d = b2_data_mod.prepare_b2(b2_config, b2_split, 72, verbose=False)
    assert d["n_target"] == 10, "B2 只用核心 10 列"


def test_hi_is_supervision_target_not_input(b2_data_mod, b2_config, b2_split):
    """HI 以监督通道 (第 2 个返回值) 出现, 而不是拼进 x。"""
    d = b2_data_mod.prepare_b2(b2_config, b2_split, 72, verbose=False)
    x, h, _r, _e, _l = d["ltr"].dataset[0]
    assert x.ndim == 3          # (K, L, n_feat)
    assert h.ndim == 1          # (K,) —— HI 是逐窗监督标量
    assert h.shape[0] == x.shape[0]
    hv = h.numpy()
    assert np.all((hv >= -1e-6) & (hv <= 1.0 + 1e-6)), "HI 应在 [0,1]"


def test_config_declares_hi_roles(b2_config):
    """config 必须显式声明 HI 的允许/禁止角色, 与 B1.9 下游契约一致。"""
    b2 = b2_config["b2"]
    assert b2["hi_in_plain_input"] is False
    assert b2["hi_in_auxiliary_loss"] is True
    assert b2["censor_aware"] is True
    assert b2["forbid_fake_rul_for_censored"] is True


def test_zscore_uses_train_rows_only(b2_data_mod, b2_config, b2_split):
    """标准化统计量只来自 train 行 —— 否则 val/test 分布信息泄漏进输入。

    做法: 独立重算 train 行的均值/方差, 断言 train 行标准化后被正确中心化。

    两处容差不是随手放宽的:
    * 均值用 float64 复算后残差 ~1e-13; 存储是 float32、train 有 231 万行,
      float32 累加残差可达 ~4e-2, 所以这里在 float64 下比对。
    * 标准差**不应**恰好为 1: 冻结栈的分母是 (sd + 1e-6), 对 raw std 极小的列
      (Tf_slope 4.1e-6 / b_hat 2.3e-5) 这个 eps 并不可忽略,
      std(z) = sd/(sd+1e-6) 解析上就是 0.802 / 0.959。
      因此按解析预期值比对, 而不是硬套 1.0 —— 后者会逼着人去改代码迁就测试。
    """
    d = b2_data_mod.prepare_b2(b2_config, b2_split, 72, verbose=False)
    h5 = b2_data_mod.ROOT / b2_config["transfer"]["target_feature_path"]
    xT, _hi, _r, _lb, _ev, tid, _n, tids = b2_data_mod.load_b2_target(
        h5, b2_config["transfer"]["target_hi_key"])
    name2i = {t: i for i, t in enumerate(tids)}
    tr = np.array([name2i[t] for t in b2_split["splits"]["train"]["tids"]])
    m = np.isin(tid, tr)
    X = xT[m].astype(np.float64)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    z = (X - mu) / (sd + 1e-6)
    assert np.abs(z.mean(axis=0)).max() < 1e-6
    expected_std = sd / (sd + 1e-6)
    assert np.abs(z.std(axis=0) - expected_std).max() < 1e-6
    assert d["n_target"] == xT.shape[1]


def test_zscore_would_detect_leakage(b2_data_mod, b2_config, b2_split):
    """哨兵: 若统计量混入 val/test 行, train 行就不再被中心化 —— 必须可察觉。

    没有这条, 上面那个测试无法区分"只用 train"与"用了全体"。
    """
    h5 = b2_data_mod.ROOT / b2_config["transfer"]["target_feature_path"]
    xT, _hi, _r, _lb, _ev, tid, _n, tids = b2_data_mod.load_b2_target(
        h5, b2_config["transfer"]["target_hi_key"])
    name2i = {t: i for i, t in enumerate(tids)}
    tr = np.array([name2i[t] for t in b2_split["splits"]["train"]["tids"]])
    m = np.isin(tid, tr)
    X = xT.astype(np.float64)
    mu_all, sd_all = X.mean(axis=0), X.std(axis=0)     # 全体统计量 (泄漏)
    z_tr = (X[m] - mu_all) / (sd_all + 1e-6)
    assert np.abs(z_tr.mean(axis=0)).max() > 1e-4, \
        "用全体统计量时 train 行仍被完美中心化 -> 该检测无鉴别力"
