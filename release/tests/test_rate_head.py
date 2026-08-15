"""S5 速率预测 + 物理首达换算的不变量测试。

覆盖点 (对应 S5 改动 1/2/3/4):
  * first_passage_rul / first_passage_var 的解析正确性、eps 下界、非负余量截断
  * numpy 与 torch 两条实现必须逐位一致 (Wiener+PF 基线走 numpy, 训练走 torch,
    若两者不一致, "基线与模型复用同一模块"就是空话)
  * causal_lsq_slope 的**因果性** (未来点不得影响当前值) 与闭式解 vs polyfit 一致
  * build_trend_features 的自校准性 (不含任何仿真真值) 与无量纲性
  * RateHead 输出恒正、rate_head=False 时不建子模块 (direct_rul 与 S3 逐位可比)
  * rate_cfg 严格读取 (缺键必须抛)
  * RateWrap 窗末对齐 (交叉校验)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config                                          # noqa: E402
from src.models.rate_head import (RateHead, first_passage_rul,             # noqa: E402
                                  first_passage_var, gaussian_interval,
                                  gaussian_nll)
from src.sim.build_hi import (XT_COLS, causal_lsq_slope,                   # noqa: E402
                              build_trend_features)
from src.transfer.adapter import TransferModel                             # noqa: E402
from src.transfer.train_transfer import hi_local_slope                     # noqa: E402
from src.experiments.run_groups import rate_cfg                            # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config(str(ROOT / "configs" / "wheel.yaml"))


# ============================== 改动 1: 首达换算 ==============================

def test_first_passage_analytic():
    """RUL = (HI_fail - HI_t)/mu 的解析值。"""
    # 余量 0.5, 速率 0.25 -> RUL = 2.0
    assert first_passage_rul(0.5, 0.25, hi_fail=1.0, eps_mu=1e-6) == pytest.approx(2.0)
    # 余量 0.2, 速率 0.1 -> 2.0
    assert first_passage_rul(0.8, 0.1, hi_fail=1.0, eps_mu=1e-6) == pytest.approx(2.0)


def test_first_passage_margin_clamped_nonnegative():
    """HI 已越阈时余量必须截到 0 (RUL=0), 不能给负 RUL。"""
    r = first_passage_rul(np.array([1.0, 1.3, 2.0]), np.array([0.1, 0.1, 0.1]),
                          hi_fail=1.0, eps_mu=1e-6)
    assert np.all(np.asarray(r) == 0.0), f"越阈点 RUL 应为 0, 实际 {r}"


def test_first_passage_eps_mu_floor_prevents_blowup():
    """mu -> 0 时必须被 eps_mu 兜住, 不得产生 inf/nan。"""
    r = first_passage_rul(np.array([0.5]), np.array([0.0]), hi_fail=1.0, eps_mu=1e-2)
    assert np.isfinite(r).all()
    assert float(r[0]) == pytest.approx(0.5 / 1e-2)


def test_first_passage_rul_max_clip():
    """rul_max 给出时必须截顶 (归一 RUL 不应远超 1)。"""
    r = first_passage_rul(np.array([0.5]), np.array([1e-3]), eps_mu=1e-6, rul_max=2.0)
    assert float(r[0]) == pytest.approx(2.0)


def test_first_passage_var_formula():
    """Var = (HI_fail - HI_t) * sigma^2 / mu^3 (指令给定式)。"""
    hi, mu, sg = 0.6, 0.2, 0.05
    got = float(np.asarray(first_passage_var(hi, mu, sg, eps_mu=1e-6)))
    assert got == pytest.approx((1.0 - hi) * sg ** 2 / mu ** 3, rel=1e-9)


def test_numpy_torch_paths_agree():
    """同一模块的 numpy / torch 分支必须给同一个数 —— 否则"基线与模型同模块"不成立。"""
    hi = np.array([0.1, 0.4, 0.75, 0.99], dtype=np.float64)
    mu = np.array([0.05, 0.2, 0.5, 1.0], dtype=np.float64)
    sg = np.array([0.01, 0.02, 0.05, 0.1], dtype=np.float64)
    rn = np.asarray(first_passage_rul(hi, mu, eps_mu=1e-2), dtype=np.float64)
    rt = first_passage_rul(torch.from_numpy(hi), torch.from_numpy(mu),
                           eps_mu=1e-2).numpy()
    assert np.allclose(rn, rt, atol=1e-12), f"{rn} != {rt}"
    vn = np.asarray(first_passage_var(hi, mu, sg, eps_mu=1e-2), dtype=np.float64)
    vt = first_passage_var(torch.from_numpy(hi), torch.from_numpy(mu),
                           torch.from_numpy(sg), eps_mu=1e-2).numpy()
    assert np.allclose(vn, vt, atol=1e-12), f"{vn} != {vt}"


def test_gaussian_interval_and_nll_sane():
    """区间对称包住点估计; 完美预测的 NLL 小于偏差预测的 NLL。"""
    rul = np.array([0.5, 0.8])
    var = np.array([0.01, 0.04])
    lo, up = gaussian_interval(rul, var, z=1.645)
    assert np.all(lo <= rul) and np.all(up >= rul)
    assert np.allclose(up - rul, rul - lo), "区间应对称"
    good = float(np.asarray(gaussian_nll(rul, rul, var)).mean())
    bad = float(np.asarray(gaussian_nll(rul, rul + 0.5, var)).mean())
    assert bad > good, f"偏差预测 NLL {bad} 应大于完美预测 {good}"


def test_rate_head_outputs_strictly_positive_and_floored():
    """softplus + eps 下界: mu/sigma 恒 > 0 且 >= eps, 极端负输入也不塌到 0。"""
    torch.manual_seed(0)
    h = RateHead(latent_dim=16, eps_mu=1e-2, eps_sigma=1e-3)
    z = torch.randn(32, 16) * 50.0        # 制造极端 logits
    mu, sg = h(z)
    assert mu.shape == (32,) and sg.shape == (32,)
    assert torch.all(mu >= 1e-2 - 1e-12), f"mu 低于下界: {mu.min()}"
    assert torch.all(sg >= 1e-3 - 1e-12), f"sigma 低于下界: {sg.min()}"
    assert torch.isfinite(mu).all() and torch.isfinite(sg).all()


# ============================== 改动 4: 消融组逐位可比 ==============================

def test_rate_head_disabled_creates_no_submodule():
    """rate_head=False 时不得新建参数 —— direct_rul 的 state_dict 必须与 S3 逐位相同。"""
    kw = dict(encoder_type="tcn", n_features=12, n_target=10, latent_dim=16)
    m0 = TransferModel(**kw)                          # S3 形态
    m1 = TransferModel(**kw, rate_head=False)         # S5 消融组
    m2 = TransferModel(**kw, rate_head=True)          # S5 速率组
    assert m1.rate_head is None
    assert set(m0.state_dict()) == set(m1.state_dict()), "消融组 state_dict 键集变了"
    n0 = sum(p.numel() for p in m0.parameters())
    n1 = sum(p.numel() for p in m1.parameters())
    n2 = sum(p.numel() for p in m2.parameters())
    assert n0 == n1, f"消融组参数量变了 {n0} -> {n1}"
    assert n2 > n1, "速率组应比消融组多参数"


def test_forward_rate_raises_when_disabled():
    m = TransferModel(encoder_type="tcn", n_features=12, n_target=10, latent_dim=16)
    with pytest.raises(RuntimeError):
        m.forward_rate(torch.randn(2, 32, 10))


def test_forward_rate_shapes():
    m = TransferModel(encoder_type="tcn", n_features=12, n_target=10, latent_dim=16,
                      rate_head=True)
    hi, mu, sg, z = m.forward_rate(torch.randn(4, 32, 10))
    assert hi.shape == (4,) and mu.shape == (4,) and sg.shape == (4,)
    assert z.shape == (4, 16)
    assert torch.all(mu > 0) and torch.all(sg > 0)


# ============================== 改动 3: 长基线趋势特征 ==============================

def test_causal_lsq_slope_is_causal():
    """未来点不得影响当前值 —— 篡改 t 之后的样本, t 及之前的斜率必须逐位不变。"""
    rng = np.random.default_rng(0)
    v = np.cumsum(rng.random(500)) / 500.0
    s1 = causal_lsq_slope(v, 100)
    v2 = v.copy()
    v2[300:] += 10.0                       # 只改未来
    s2 = causal_lsq_slope(v2, 100)
    assert np.allclose(s1[:300], s2[:300], atol=1e-12), "斜率泄漏了未来信息"


def test_causal_lsq_slope_matches_polyfit():
    """闭式滚动 OLS 必须与逐窗 polyfit 一致 (闭式是为性能, 不是为近似)。"""
    rng = np.random.default_rng(1)
    v = np.cumsum(rng.random(300)) / 300.0
    W = 50
    got = causal_lsq_slope(v, W, min_periods=8)
    for i in (60, 120, 199, 299):
        lo = max(0, i - W + 1)
        seg = v[lo:i + 1]
        want = np.polyfit(np.arange(len(seg), dtype=float), seg, 1)[0]
        assert got[i] == pytest.approx(want, rel=1e-8, abs=1e-12), \
            f"i={i}: {got[i]} != polyfit {want}"


def test_causal_lsq_slope_exact_on_linear_ramp():
    """严格线性序列上斜率必须等于解析斜率。"""
    k = 3.5e-4
    v = k * np.arange(400, dtype=float)
    s = causal_lsq_slope(v, 100, min_periods=8)
    assert np.allclose(s[100:], k, rtol=1e-9), f"线性序列斜率偏离: {s[100:].min()}"


def test_causal_lsq_slope_no_nan_before_min_periods():
    """不足 min_periods 时返回 0.0 而非 NaN (NaN 会静默污染梯度)。"""
    s = causal_lsq_slope(np.arange(50, dtype=float), 20, min_periods=8)
    assert np.isfinite(s).all(), "斜率含 NaN/inf"
    assert np.all(s[:7] == 0.0), "不足 min_periods 应为 0.0"


def test_build_trend_features_dimensionless_and_self_calibrated(cfg):
    """Tf_ratio 无量纲: Tf_hat 整体乘常数 (等价于 b0 尺度差异) 后 Tf_ratio 不变。

    这是 S5 改动 3 的**核心动机** —— 逐轨迹 b0 有 10 倍跨度, 绝对水平因此不可辨识;
    相对自校准基线把这个尺度自由度直接除掉。
    """
    rng = np.random.default_rng(2)
    Tf = 0.01 + np.cumsum(rng.random(3000)) * 1e-5
    r1, s1 = build_trend_features(Tf, cfg["sim"])
    r2, s2 = build_trend_features(Tf * 7.3, cfg["sim"])       # 模拟 b0 大 7.3 倍
    assert np.allclose(r1, r2, rtol=1e-9), "Tf_ratio 不是无量纲的"
    assert np.isfinite(r1).all() and np.isfinite(s1).all()
    assert np.allclose(s2, s1 * 7.3, rtol=1e-9), "Tf_slope 应随尺度线性缩放"


def test_xt_cols_additive_and_config_consistent(cfg):
    """XT_COLS 为 10 维且与 config 的 n_features_target 一致; 前 8 列未动。"""
    assert list(XT_COLS[:8]) == ["I_m", "omega", "T", "T_cmd", "sigma_Im",
                                 "b_hat", "dT", "omega_err"]
    assert list(XT_COLS[8:]) == ["Tf_ratio", "Tf_slope"]
    assert len(XT_COLS) == int(cfg["model"]["n_features_target"]) == 10


# ============================== 改动 2: 速率标签 / config ==============================

def test_hi_local_slope_nonnegative_and_scaled(cfg):
    """速率标签恒非负; scale 线性作用。"""
    rc = rate_cfg(cfg)
    rc = dict(rc, slope_window=100, slope_min_periods=8)
    hi = np.clip(np.cumsum(np.abs(np.random.default_rng(3).normal(size=500))) / 5000, 0, 1)
    hi = np.maximum.accumulate(hi)
    m1 = hi_local_slope(hi, rc, 1.0)
    m2 = hi_local_slope(hi, rc, 1000.0)
    assert np.all(m1 >= 0.0), "速率标签出现负值"
    assert np.allclose(m2, m1 * 1000.0, rtol=1e-9)


def test_hi_local_slope_on_flat_hi_is_zero(cfg):
    """HI 恒定 (窗内 ΔHI=0, 即 S4 根因场景) 时速率标签为 0, 不是噪声。"""
    rc = dict(rate_cfg(cfg), slope_window=200, slope_min_periods=8)
    m = hi_local_slope(np.full(600, 0.37), rc, 1.0)
    assert np.allclose(m, 0.0, atol=1e-12)


def test_hi_local_slope_window_much_larger_than_L(cfg):
    """指令要求 slope_window >> L —— 短窗斜率信噪比 <1, 用它当标签就是在拟合噪声。"""
    rc = rate_cfg(cfg)
    L = int(cfg["model"]["input_len_L"])
    assert int(rc["slope_window"]) >= 10 * L, \
        f"slope_window={rc['slope_window']} 未显著大于 L={L}"


def test_rate_cfg_strict_on_missing_key(cfg):
    """rate_cfg 必须严格读取: 缺键抛错, 不得静默用默认值 (硬编码阈值禁令)。"""
    import copy
    bad = copy.deepcopy(cfg)
    del bad["rate"]["eps_mu"]
    with pytest.raises(KeyError):
        rate_cfg(bad)


def test_rate_cfg_capped_weight_is_zero(cfg):
    """指令: capped_weight 设 0 —— 截顶样本不进 L_huber (仍可进 L_rate)。"""
    rc = rate_cfg(cfg)
    assert float(rc["capped_weight"]) == 0.0, \
        f"rate.capped_weight 应为 0, 实际 {rc['capped_weight']}"


def test_rate_cfg_enabled_and_weights_positive(cfg):
    rc = rate_cfg(cfg)
    assert bool(rc["enabled"])
    assert float(rc["w_rate"]) > 0, "L_rate 是主项, 权重必须 > 0"
    assert float(rc["w_rul"]) > 0 and float(rc["w_nll"]) > 0
    assert float(rc["eps_mu"]) > 0 and float(rc["eps_sigma"]) > 0
    assert float(rc["interval_z"]) == pytest.approx(1.645, abs=1e-3), \
        "interval_z 应为 1.645 (双侧 90%)"
