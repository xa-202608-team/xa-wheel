"""tests/test_metrics.py — 锁定三档评估口径与宏平均的语义 (S2' 任务)

这些不变量是所有实验数字可比的前提: 一旦口径掩码或宏平均的定义漂移, 历史结果与
新结果就不再同口径, 但两者都会"看起来正常"。故用测试把语义钉死。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.physical_extrap import (        # noqa: E402
    CALIBERS, PRIMARY_CALIBER, PRIMARY_STAT, CAP_EPS,
    caliber_mask, pooled_metrics, macro_metrics, caliber_metrics,
    clip_to_label_domain, eval_point_indices, fill_nan_forward,
    phm_score, rmse, mae)


# ---------------------------------------------------------------- 口径掩码
def test_caliber_mask_definitions():
    """full ⊇ valid ⊇ info; valid 剔 RUL==0; info 再剔截顶饱和。"""
    t = np.array([0.0, 0.0, 0.3, 0.7, 1.0, 1.0])
    m_full = caliber_mask(t, "full")
    m_valid = caliber_mask(t, "valid")
    m_info = caliber_mask(t, "info")
    assert m_full.all()
    assert m_valid.tolist() == [False, False, True, True, True, True]
    assert m_info.tolist() == [False, False, True, True, False, False]
    # 嵌套关系
    assert (m_info <= m_valid).all() and (m_valid <= m_full).all()


def test_caliber_mask_cap_eps_boundary():
    """cap_eps 决定"多接近 1.0 算截顶"; 边界必须严格 (< 1-eps 才留)。"""
    eps = 1e-3
    t = np.array([1.0 - 2 * eps, 1.0 - eps, 1.0 - eps / 2, 1.0])
    m = caliber_mask(t, "info", cap_eps=eps)
    assert m.tolist() == [True, False, False, False]


def test_caliber_mask_rejects_unknown():
    with pytest.raises(ValueError):
        caliber_mask(np.array([0.5]), "everything")


def test_primary_caliber_is_info_pooled():
    """主结论口径不得静默改动 —— 改了它等于改了所有历史数字的含义。"""
    assert CALIBERS == ("full", "valid", "info")
    assert PRIMARY_CALIBER == "info"
    assert PRIMARY_STAT == "pooled"


# ---------------------------------------------------------------- 指标本体
def test_pooled_metrics_matches_closed_form():
    p = np.array([0.1, 0.5, 0.9])
    t = np.array([0.2, 0.4, 0.6])
    m = pooled_metrics(p, t)
    assert m["rmse"] == pytest.approx(np.sqrt(np.mean((p - t) ** 2)))
    assert m["mae"] == pytest.approx(np.mean(np.abs(p - t)))
    assert m["rmse"] == pytest.approx(rmse(p, t))
    assert m["mae"] == pytest.approx(mae(p, t))


def test_pooled_metrics_empty_is_nan():
    """空口径 (如某轨迹在 info 区无样本) 必须给 NaN 而不是 0 —— 0 会被误读为完美。"""
    m = pooled_metrics(np.array([]), np.array([]))
    assert all(np.isnan(m[k]) for k in ("rmse", "phm", "mae"))


def test_phm_penalises_late_more_than_early():
    """晚预测 (pred > true) 罚得比同幅度早预测更重 (late_scale < early_scale)。"""
    t = np.array([0.5])
    late = phm_score(t + 0.1, t)
    early = phm_score(t - 0.1, t)
    assert late > early > 0
    assert phm_score(t, t) == pytest.approx(0.0)


def test_phm_no_overflow_on_extreme_predictions():
    """未 clip 的物理外推曾产生 |d|~数十 → exp 溢出成 inf, 污染整张结果表。"""
    s = phm_score(np.array([1e4, 5.0]), np.array([0.1, 0.2]))
    assert np.isfinite(s) and s > 0


# ---------------------------------------------------------------- 宏平均
def test_macro_is_unweighted_mean_over_trajectories():
    """宏平均对轨迹等权: 长轨迹不因样本多而支配结果 (pooled 会)。"""
    # 轨迹 0: 90 个完美点; 轨迹 1: 10 个偏差 0.5 的点
    p = np.concatenate([np.full(90, 0.5), np.full(10, 1.0)])
    t = np.concatenate([np.full(90, 0.5), np.full(10, 0.5)])
    tids = np.concatenate([np.zeros(90, dtype=int), np.ones(10, dtype=int)])
    mac = macro_metrics(p, t, tids)
    pol = pooled_metrics(p, t)
    assert mac["n_traj"] == 2
    assert mac["rmse"] == pytest.approx((0.0 + 0.5) / 2)       # 等权
    assert pol["rmse"] == pytest.approx(np.sqrt(10 * 0.25 / 100))
    assert mac["rmse"] > pol["rmse"]                           # 小轨迹被放大


def test_macro_equals_pooled_when_single_trajectory():
    p, t = np.array([0.1, 0.4, 0.9]), np.array([0.2, 0.4, 0.7])
    tids = np.zeros(3, dtype=int)
    assert macro_metrics(p, t, tids)["rmse"] == pytest.approx(pooled_metrics(p, t)["rmse"])


# ---------------------------------------------------------------- 组合入口
def test_caliber_metrics_structure_and_top_level_keys():
    """顶层 rmse/phm/mae 必须等于主口径 (info/pooled) —— 下游 aggregate 依赖这一约定。"""
    t = np.array([0.0, 0.2, 0.5, 0.8, 1.0, 1.0])
    p = np.array([0.1, 0.3, 0.4, 0.9, 0.9, 1.0])
    tids = np.array([0, 0, 0, 1, 1, 1])
    m = caliber_metrics(p, t, tids)
    assert set(m["calibers"]) == set(CALIBERS)
    for c in CALIBERS:
        assert set(m["calibers"][c]) == {"pooled", "macro", "n", "share"}
    prim = m["calibers"][PRIMARY_CALIBER][PRIMARY_STAT]
    for k in ("rmse", "phm", "mae"):
        assert m[k] == pytest.approx(prim[k])
    assert m["has_tids"] is True
    assert m["n_total"] == len(t)
    assert m["calibers"]["full"]["share"] == pytest.approx(1.0)


def test_caliber_metrics_without_tids_degrades_to_single_group():
    t = np.array([0.2, 0.5, 0.8])
    p = np.array([0.3, 0.4, 0.9])
    m = caliber_metrics(p, t, None)
    assert m["has_tids"] is False
    assert m["calibers"]["info"]["macro"]["n_traj"] == 1
    assert m["calibers"]["info"]["macro"]["rmse"] == pytest.approx(
        m["calibers"]["info"]["pooled"]["rmse"])


def test_caliber_metrics_share_sums_consistently():
    t = np.concatenate([np.zeros(50), np.linspace(0.01, 0.99, 30), np.ones(20)])
    p = t.copy()
    m = caliber_metrics(p, t, np.zeros(len(t), dtype=int))
    assert m["calibers"]["full"]["n"] == 100
    assert m["calibers"]["valid"]["n"] == 50        # 30 info + 20 截顶
    assert m["calibers"]["info"]["n"] == 30


# ---------------------------------------------------------------- 基线辅助
def test_clip_to_label_domain():
    v = clip_to_label_domain(np.array([-3.0, 0.0, 0.5, 1.0, 42.0]))
    assert v.tolist() == [0.0, 0.0, 0.5, 1.0, 1.0]


def test_eval_point_indices_matches_window_ends():
    """基线取点必须与 TargetSeqDataset(K=1) 的窗末一致, 否则与模型不同口径。"""
    L, stride, T = 64, 50, 300
    pts = eval_point_indices(T, L, stride)
    starts = np.arange(0, T - L + 1, stride)
    assert pts.tolist() == (starts + L - 1).tolist()
    assert pts.max() < T
    assert len(eval_point_indices(10, L, stride)) == 0      # 短于窗长 → 无点


def test_fill_nan_forward_semantics():
    v, n = fill_nan_forward(np.array([np.nan, np.nan, 0.4, np.nan, 0.2]), fallback=1.0)
    assert n == 3
    assert v.tolist() == [1.0, 1.0, 0.4, 0.4, 0.2]
    v2, n2 = fill_nan_forward(np.array([0.5, 0.6]), fallback=1.0)
    assert n2 == 0 and v2.tolist() == [0.5, 0.6]


def test_default_cap_eps_is_tiny():
    """默认 cap_eps 只为吸收浮点误差, 不应大到吃掉真实 info 样本。"""
    assert 0 < CAP_EPS <= 1e-4
