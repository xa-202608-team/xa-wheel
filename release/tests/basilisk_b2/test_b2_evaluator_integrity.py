"""tests/basilisk_b2/test_b2_evaluator_integrity.py —— 自写 evaluator 的正确性。

B2 因为 `hi_b` 硬编码而不得不自带 loader / axis / 评估装配。新写的代码就是新的
出错面, 因此本文件的核心是: **证明 B2 的 evaluator 与冻结栈在同一份数据上给出
逐位相同的数字**, 只在"多带了删失轨迹"和"换了 HI 列名"两点上有意为之地不同。
"""
from __future__ import annotations

import numpy as np


def test_endpoint_axis_matches_loader_windows(b2_eval_mod, b2_data_mod,
                                              b2_config, b2_split):
    """axis 的 endpoint 数必须与评估 loader 的窗数逐轨迹一致。

    若两者错位, tau 会整体偏移, 所有 warning / PH 数字都会变成无声的错 ——
    所以这里不允许"截断凑数", 必须精确相等。
    """
    axis = b2_eval_mod.b2_endpoint_axis(b2_config)
    d = b2_data_mod.prepare_b2(b2_config, b2_split, 72, with_test=True,
                               verbose=False)
    for name, ids in (("val", d["va"]), ("test", d["te"])):
        loader = d["lva"] if name == "val" else d["lte"]
        n_axis = sum(len(axis[int(i)]["endpoint_idx"]) for i in ids)
        assert n_axis == len(loader.dataset), (name, n_axis,
                                              len(loader.dataset))


def test_attach_axis_rejects_misalignment(b2_eval_mod, b2_config):
    """哨兵: 预测点数超过 endpoint 数必须报错, 不得静默对齐。"""
    import pytest
    axis = b2_eval_mod.b2_endpoint_axis(b2_config)
    tid0 = 0
    n = len(axis[tid0]["endpoint_idx"])
    tids = np.full(n + 5, tid0, dtype=np.int64)     # 故意多 5 个点
    with pytest.raises(ValueError, match="拒绝对齐"):
        b2_eval_mod.attach_axis_b2(tids, axis)


def test_macro_rmse_empty_gives_nan_not_zero(b2_eval_mod):
    """空输入 -> (NaN, 0)。返回 0 会把"无法评估"伪装成"零误差"。"""
    v, n = b2_eval_mod._macro_rmse_over(np.array([]), np.array([]),
                                        np.array([], dtype=np.int64))
    assert n == 0
    assert not np.isfinite(v)


def test_macro_rmse_all_nan_gives_nan(b2_eval_mod):
    """全 NaN 真值 -> NaN + n=0 (删失分层就是这种情形)。"""
    p = np.array([0.5, 0.5])
    t = np.array([np.nan, np.nan])
    v, n = b2_eval_mod._macro_rmse_over(p, t, np.array([0, 0]))
    assert n == 0 and not np.isfinite(v)


def test_macro_rmse_known_value(b2_eval_mod):
    """已知答案校验: 两条轨迹各自 RMSE 0.1 / 0.3 -> 宏平均 0.2。"""
    p = np.array([0.1, 0.1, 0.3, 0.3])
    t = np.array([0.0, 0.2, 0.0, 0.6])
    v, n = b2_eval_mod._macro_rmse_over(p, t, np.array([0, 0, 1, 1]))
    assert n == 2
    assert abs(v - 0.2) < 1e-12


def test_const_mean_info_hits_uniform_anchor(b2_baselines_mod, b2_eval_mod,
                                             b2_data_mod, b2_config, b2_split):
    """const_mean_info 在 info 区应接近 0.5, 其 RMSE 应接近理论锚点 1/sqrt(12)。

    这同时验证了取点与归一化口径: 若 rul_scale 或 endpoint 取错, 常数值和
    RMSE 都不会落在这两个已知数上。
    """
    from src.baselines.trivial import UNIFORM_CONST_RMSE
    axis = b2_eval_mod.b2_endpoint_axis(b2_config)
    d = b2_data_mod.prepare_b2(b2_config, b2_split, 72, with_test=True,
                               verbose=False)
    c = b2_baselines_mod.const_mean_info(b2_config, axis, d)
    assert 0.4 < c < 0.6, c
    p, t, i = b2_baselines_mod.const_pred_on(axis, d["te"], c, b2_config, d)
    r = b2_eval_mod.evaluate_arrays_b2(p, t, i, b2_config, axis, d, tag="A")
    assert abs(r["info_macro_rmse"] - UNIFORM_CONST_RMSE) < 0.02, \
        (r["info_macro_rmse"], UNIFORM_CONST_RMSE)


def test_const_mean_info_uses_train_only(b2_baselines_mod, b2_eval_mod,
                                         b2_data_mod, b2_config, b2_split):
    """哨兵: 换掉 train 集必须改变常数值 —— 证明它真的只由 train 决定。"""
    import copy
    axis = b2_eval_mod.b2_endpoint_axis(b2_config)
    d = b2_data_mod.prepare_b2(b2_config, b2_split, 72, with_test=True,
                               verbose=False)
    base = b2_baselines_mod.const_mean_info(b2_config, axis, d)
    d2 = copy.copy(d)
    d2["tr"] = d["te"]              # 故意换成 test 轨迹
    other = b2_baselines_mod.const_mean_info(b2_config, axis, d2)
    assert abs(base - other) > 1e-9, "常数值与 train 集无关 -> 它没在用 train"


def test_damage_extrap_never_fills_zero(b2_baselines_mod, b2_eval_mod,
                                        b2_data_mod, b2_config, b2_split):
    """B 基线: 斜率不可用时输出 NaN, 不是 0。填 0 = 伪造"马上失效"。"""
    axis = b2_eval_mod.b2_endpoint_axis(b2_config)
    d = b2_data_mod.prepare_b2(b2_config, b2_split, 72, with_test=True,
                               verbose=False)
    ids = list(d["te"])[:2]
    p, t, i = b2_baselines_mod.damage_extrapolation(b2_config, axis, d, ids)
    assert len(p) == sum(len(axis[int(x)]["endpoint_idx"]) for x in ids)
    fin = np.isfinite(p)
    if not fin.all():
        assert not np.any(p[~fin] == 0.0)
    assert np.all(p[fin] >= 0.0), "外推 RUL 不得为负"


def test_evaluator_agrees_with_frozen_eval_test(b2_eval_mod, b2_data_mod,
                                                b2_config, b2_split):
    """关键一致性测试: B2 evaluator 与冻结 eval_test 在同一批预测上逐位相同。

    用一个**确定性的假模型** (常数输出) 而不是真训练, 使测试快且可复现。
    两侧都从同一个 loader 取点, 因此任何口径分叉都会立刻暴露。
    """
    import torch

    from src.experiments.run_groups import eval_test

    class ConstModel(torch.nn.Module):
        """固定输出的假模型 —— 只为比对两侧 evaluator 的口径。"""

        def forward(self, x):
            n = x.shape[0]
            hi = torch.full((n, 1), 0.5)
            rul = torch.full((n, 1), 0.42)
            return hi, rul, torch.zeros(n, 4)

    d = b2_data_mod.prepare_b2(b2_config, b2_split, 72, with_test=True,
                               verbose=False)
    d["device"] = "cpu"
    axis = b2_eval_mod.b2_endpoint_axis(b2_config)
    m = ConstModel()
    frozen = eval_test(m, d["lva"], "cpu", cap_eps=d["cap_eps"])
    mine = b2_eval_mod.evaluate_b2(m, d["lva"], b2_config, axis, d, tag="cmp")
    fc = frozen["calibers"]["info"]
    assert mine["info_n"] == fc["n"]
    assert abs(mine["info_macro_rmse"] - fc["macro"]["rmse"]) < 1e-12
    assert abs(mine["info_pooled_rmse"] - fc["pooled"]["rmse"]) < 1e-12


def test_warning_validity_requires_nonempty_strata(b2_gate_mod):
    """条件 6 的判定: 分母为空 / 指标为 NaN 时必须判无效, 不得靠 NaN 混过。

    同时确认它**不是**在要求"coverage 要高" —— 一个 coverage=0 但指标齐全、
    分母非空的结果仍算有效 (条件 6 问的是指标可用, 不是模型报得好)。
    把它写成"coverage 要高"就是给闸门放水。
    """
    good = {"warning_coverage": 0.0, "miss_rate": 1.0,
            "coverage_before_eol": 0.0, "miss_rate_before_eol": 1.0,
            "false_alarm_rate": 0.0, "n_observed": 36, "n_censored": 39}
    ok, det = b2_gate_mod.warning_valid(good)
    assert ok is True and det["valid"] is True
    assert det["complementary"] is True

    # 分母为空 -> 无效
    empty = {**good, "n_observed": 0}
    ok2, _ = b2_gate_mod.warning_valid(empty)
    assert ok2 is False
    empty_cen = {**good, "n_censored": 0}
    assert b2_gate_mod.warning_valid(empty_cen)[0] is False

    # 指标是 NaN -> 无效
    nan_row = {**good, "miss_rate_before_eol": float("nan")}
    assert b2_gate_mod.warning_valid(nan_row)[0] is False

    # coverage 与 miss 不互补 (统计口径自相矛盾) -> 无效
    bad = {**good, "coverage_before_eol": 0.5, "miss_rate_before_eol": 0.9}
    assert b2_gate_mod.warning_valid(bad)[0] is False


def test_strip_private_removes_arrays(b2_eval_mod):
    """落 JSON 前必须去掉 _raw / _warning_full 等大对象。"""
    d = {"a": 1, "_raw": {"x": [1, 2]}, "b": {"c": 2, "_p": 3}}
    out = b2_eval_mod.strip_private(d)
    assert out == {"a": 1, "b": {"c": 2}}


def _cen_report(b2_eval_mod, pred, lb, rul_scale):
    """构造一个纯删失分层, 只为验证下界违反率的单位口径。"""
    n = len(pred)
    tids = np.zeros(n, dtype=np.int64)
    true = np.full(n, np.nan)          # 删失: 无真实 RUL
    tau = np.linspace(0.0, 1.0, n)
    ev = {0: False}
    rep = b2_eval_mod.censor_strata_report(
        np.asarray(pred, float), true, tids, tau,
        np.asarray(lb, float), ev, 1e-6, rul_scale)
    return rep["censored_test"]


def test_lower_bound_violation_compares_in_normalized_units(b2_eval_mod):
    """lb 是原始步数, pred 是归一化 RUL —— 必须先除 rul_scale 再比较。

    回归保护: 早期实现直接拿原始 lb 与归一化 pred 比, 违反率被恒定锁死在
    1.0, 看上去像"所有方法都违反下界", 实际是量纲错误。
    """
    scale = 40000.0
    # 归一化后下界 = 0.25; pred 全在下界之上 -> 违反率必须是 0
    cen = _cen_report(b2_eval_mod, [0.4, 0.5, 0.6], [10000.0] * 3, scale)
    assert cen["n_lower_bound_evaluable"] == 3
    assert cen["lower_bound_violation_rate"] == 0.0
    # pred 全在下界之下 -> 1.0
    cen = _cen_report(b2_eval_mod, [0.1, 0.2, 0.24], [10000.0] * 3, scale)
    assert cen["lower_bound_violation_rate"] == 1.0
    # 一半违反 -> 0.5 (若单位错, 这里会退化成 1.0, 无法区分)
    cen = _cen_report(b2_eval_mod, [0.1, 0.9], [10000.0] * 2, scale)
    assert cen["lower_bound_violation_rate"] == 0.5


def test_censored_stratum_rul_metrics_stay_nan(b2_eval_mod):
    """删失分层的 RUL 指标恒为 NaN + n=0, 且携带原因串。"""
    cen = _cen_report(b2_eval_mod, [0.4, 0.5], [10000.0] * 2, 40000.0)
    assert not np.isfinite(cen["info_macro_rmse"])
    assert cen["n_evaluable_rul_points"] == 0
    assert "never_fabricated" in cen["rul_metrics_reason"]
    assert cen["tau_semantics"] == b2_eval_mod.CENSORED_TAU_SEMANTICS
