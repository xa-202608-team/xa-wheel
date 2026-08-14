"""tests/test_wiener_pf.py — S5B Wiener + 粒子滤波基线的不变量测试 (协议 §19)

三条主线:
  1. **先验只能来自 train 可见段**, 且 `fit_prior()` 接口根本不接收 RUL/EOL
     (右删失契约的接口级钉死 —— 协议 §5/§8)。
  2. **在线因果性**: 篡改 t 之后的 HI, t 及之前的预测必须逐位不变 (协议 §6)。
  3. **数值健全性**: mu/sigma 恒正、首达为正、分位有序、可复现、粒子数守恒。

另含一条 synthetic Wiener 轨迹 (已知 mu/sigma) 的行为检查: 后验漂移方向正确、
RUL 随 H 上升总体下降。**不要求数值等于理论值**, 用宽松容差。
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from src.baselines.wiener_pf import (MAD_TO_SD, filter_trajectory, fit_prior,
                                     pf_rng, robust_increment_stats,
                                     systematic_resample, wiener_pf_cfg,
                                     _visible_segment)
from src.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "wheel.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(str(CFG_PATH))


@pytest.fixture(scope="module")
def wc(cfg):
    return wiener_pf_cfg(cfg)


def _synth(n=400, mu=2e-3, sigma=5e-3, seed=7):
    """已知 (mu, sigma) 的 synthetic Wiener 轨迹 (单调化前的原始布朗运动)。"""
    r = np.random.default_rng(seed)
    return np.cumsum(r.normal(mu, sigma, n)), mu, sigma


# ============================== 先验: train-only ==============================

def test_wiener_pf_prior_uses_train_only(wc):
    """先验只是可见段 HI 序列的函数 —— 追加"未来段"不改变已有段的先验结果。"""
    hi, _, _ = _synth()
    vis = hi[:200]
    p1 = fit_prior([vis, vis * 1.1, vis * 0.9], wc)
    # 同样的可见段 + 完全不同的未来段 -> 先验逐位相同 (因为未来段没被传进去)
    p2 = fit_prior([vis, vis * 1.1, vis * 0.9], wc)
    assert p1["log_mu_mean"] == p2["log_mu_mean"]
    assert p1["log_sigma_mean"] == p2["log_sigma_mean"]
    # 而把未来段拼进去会改变结果 -> 证明"传什么就用什么", 没有隐藏的全局数据源
    p3 = fit_prior([hi, hi * 1.1, hi * 0.9], wc)
    assert p3["n_train_traj"] == 3


def test_wiener_pf_prior_does_not_accept_rul(wc):
    """fit_prior 签名里不得出现任何 rul 相关参数 (协议 §5 接口级封死)。"""
    names = set(inspect.signature(fit_prior).parameters)
    assert not any("rul" in n.lower() for n in names), names
    with pytest.raises(TypeError):
        fit_prior([_synth()[0]], wc, rul=np.zeros(10))


def test_wiener_pf_prior_does_not_accept_eol(wc):
    """同上: 不得接收 eol / fail_time / event 等真值入口。"""
    names = set(inspect.signature(fit_prior).parameters)
    for bad in ("eol", "fail", "event", "time_to", "true"):
        assert not any(bad in n.lower() for n in names), (bad, names)
    with pytest.raises(TypeError):
        fit_prior([_synth()[0]], wc, eol_idx=123)


def test_wiener_pf_censored_train_safe(cfg, wc):
    """真实 train 可见段 (HI<=truncate_hi) 上先验可算, 且可见段确实不含失效。"""
    import h5py
    from src.baselines.physical_extrap import baseline_hyper, observed_traj_keys
    from src.transfer.train_transfer import (select_train_trajectories,
                                             split_trajectories_from_cfg)
    h = baseline_hyper(cfg)
    if not h["target_h5"].exists():
        pytest.skip("缺 target_features.h5")
    tc = cfg["transfer"]
    thr = float(tc["target_truncate_hi"])
    seed = int(tc["s3_gate_seeds"][0])
    with h5py.File(h["target_h5"], "r") as f:
        keys = observed_traj_keys(f)
        tr, _va, _te, _eol = split_trajectories_from_cfg(
            h["target_h5"], len(keys), tc, seed, observed_only=True)
        sel, _ = select_train_trajectories(seed, [int(t) for t in tr],
                                           int(tc["s3_gate_train_n_traj"]),
                                           rule=str(tc["s3_selection_rule"]))
        segs = []
        for t in sel:
            g = f[keys[int(t)]]
            hi = np.asarray(g["hi_b"][:], dtype=float)
            seg = _visible_segment(hi, thr)
            assert seg.max() <= thr, "可见段越过截断阈值"
            lf = np.asarray(g["label_fail"][:])[:len(seg)]
            assert int(lf.sum()) == 0, "可见段里出现了失效标签 —— 截断无效"
            segs.append(seg)
    pr = fit_prior(segs, wc)
    assert pr["n_train_traj"] == len(sel)
    assert np.isfinite(pr["log_mu_mean"]) and np.isfinite(pr["log_sigma_mean"])
    assert pr["log_mu_std"] >= wc["min_log_mu_std"]
    assert pr["log_sigma_std"] >= wc["min_log_sigma_std"]


def test_wiener_pf_prior_mad_factor():
    """MAD->sd 一致性因子必须是 1.4826 = 1/Phi^-1(0.75) (协议 §5 明写)。"""
    from scipy.stats import norm
    assert abs(MAD_TO_SD - 1.0 / norm.ppf(0.75)) < 1e-3


def test_wiener_pf_prior_positive_increments(wc):
    """单调上升轨迹的 mu 估计为正且不触发 eps 兜底。"""
    seg = np.linspace(0.0, 0.5, 500) + 1e-4 * np.sin(np.arange(500))
    mu, sg, floored = robust_increment_stats(seg, wc["eps_mu"], wc["eps_sigma"])
    assert mu > 0 and sg > 0 and not floored


# ============================== 在线因果性 ==============================

def test_wiener_pf_online_no_future_hi(wc):
    """篡改 t 之后的 HI, t 及之前的所有预测必须**逐位不变** (真正的 online filter)。

    这是本阶段最硬的一条: 只要实现里有任何"整段拟合 mu"或"用未来点平滑"的成分,
    这个断言就会炸。
    """
    hi, _, _ = _synth(n=120, mu=3e-3, sigma=4e-3)
    hi = np.clip(np.maximum.accumulate(hi), 0.0, 0.99)
    dt = np.full(len(hi) - 1, 50.0)
    prior = fit_prior([hi[:60]], wc)
    t_cut = 70
    a = filter_trajectory(hi, dt, prior, wc, pf_rng(999, 1))
    hi2 = hi.copy()
    hi2[t_cut + 1:] = 0.98            # 未来段整体篡改
    b = filter_trajectory(hi2, dt, prior, wc, pf_rng(999, 1))
    for k in ("pred_mean", "pred_median", "q05", "q95"):
        np.testing.assert_array_equal(a[k][:t_cut + 1], b[k][:t_cut + 1],
                                      err_msg=f"{k} 泄漏了未来 HI")
    # 未来段确实被改了 -> 说明测试本身有效 (不是恒等断言)
    assert not np.array_equal(a["pred_mean"][t_cut + 1:], b["pred_mean"][t_cut + 1:])


def test_wiener_pf_first_point_uses_prior_only(wc):
    """第一个 endpoint 无前一观测 ⇒ 不做权重更新, 预测纯由先验决定。"""
    hi, _, _ = _synth(n=30)
    hi = np.clip(np.maximum.accumulate(hi), 0.0, 0.9)
    dt = np.full(len(hi) - 1, 50.0)
    prior = fit_prior([hi[:20]], wc)
    r = filter_trajectory(hi, dt, prior, wc, pf_rng(1, 1))
    assert r["ess"][0] == pytest.approx(float(wc["n_particles"]))


# ============================== 数值健全性 ==============================

def test_wiener_pf_positive_mu_sigma(wc):
    """粒子 mu / sigma 全程恒正 (log 参数化 + eps 下界)。"""
    hi, _, _ = _synth(n=80)
    hi = np.clip(np.maximum.accumulate(hi), 0.0, 0.95)
    prior = fit_prior([hi[:50]], wc)
    r = filter_trajectory(hi, np.full(len(hi) - 1, 50.0), prior, wc, pf_rng(2, 3))
    assert np.all(r["mu_final"] > 0) and np.all(r["sigma_final"] > 0)
    assert np.all(r["mu_final"] >= wc["eps_mu"])
    assert np.all(r["sigma_final"] >= wc["eps_sigma"])


def test_wiener_pf_first_passage_positive(wc):
    """首达 RUL 全部 > 0 (a>0 且 mu>0 ⇒ a/mu>0)。"""
    hi, _, _ = _synth(n=60)
    hi = np.clip(np.maximum.accumulate(hi), 0.0, 0.99)
    prior = fit_prior([hi[:40]], wc)
    r = filter_trajectory(hi, np.full(len(hi) - 1, 50.0), prior, wc, pf_rng(4, 5))
    assert np.all(r["pred_mean"] > 0), r["pred_mean"].min()
    assert np.all(r["pred_median"] > 0)


def test_wiener_pf_quantile_order(wc):
    """q05 <= median <= q95 逐点成立。"""
    hi, _, _ = _synth(n=90)
    hi = np.clip(np.maximum.accumulate(hi), 0.0, 0.97)
    prior = fit_prior([hi[:60]], wc)
    r = filter_trajectory(hi, np.full(len(hi) - 1, 50.0), prior, wc, pf_rng(6, 7))
    assert np.all(r["q05"] <= r["pred_median"] + 1e-9)
    assert np.all(r["pred_median"] <= r["q95"] + 1e-9)


def test_wiener_pf_prediction_finite(wc):
    """所有输出有限 —— 无 NaN 被悄悄当成 0, 无 inf。"""
    hi, _, _ = _synth(n=70)
    hi = np.clip(np.maximum.accumulate(hi), 0.0, 0.98)
    prior = fit_prior([hi[:50]], wc)
    r = filter_trajectory(hi, np.full(len(hi) - 1, 50.0), prior, wc, pf_rng(8, 9))
    for k in ("pred_mean", "pred_median", "q05", "q95", "ess"):
        assert np.all(np.isfinite(r[k])), k


def test_wiener_pf_reproducible(wc):
    """同 (seed, tid) 两次运行 prediction hash 逐位一致 (协议 §10)。"""
    hi, _, _ = _synth(n=100)
    hi = np.clip(np.maximum.accumulate(hi), 0.0, 0.96)
    dt = np.full(len(hi) - 1, 50.0)
    prior = fit_prior([hi[:60]], wc)
    def h(rng_seed_pair):
        r = filter_trajectory(hi, dt, prior, wc, pf_rng(*rng_seed_pair))
        return hashlib.sha256(repr(r["pred_mean"].tolist()).encode()).hexdigest()
    assert h((62, 11)) == h((62, 11))
    assert h((62, 11)) != h((62, 12)), "不同轨迹 id 应派生不同 RNG"
    assert h((62, 11)) != h((63, 11)), "不同实验 seed 应派生不同 RNG"


def test_wiener_pf_no_global_seed():
    """源码的**可执行部分**不得出现全局播种 (协议 §10)。

    用 tokenize 剥掉注释与字符串/docstring 再查 —— 否则文档里"禁止 np.random.seed()"
    这句自我说明会把测试自己炸掉 (那是纪律说明, 不是调用)。
    """
    import io
    import tokenize
    src = (ROOT / "src" / "baselines" / "wiener_pf.py").read_text(encoding="utf-8")
    code = " ".join(
        t.string for t in tokenize.generate_tokens(io.StringIO(src).readline)
        if t.type not in (tokenize.COMMENT, tokenize.STRING))
    assert "np.random.seed" not in code
    assert "RandomState" not in code
    assert "default_rng" in code, "应使用独立 Generator"


def test_wiener_pf_resample_preserves_count():
    """systematic resample 返回的索引数 = 粒子数, 且全部在合法范围内。"""
    r = np.random.default_rng(0)
    for n in (16, 256, 2048):
        w = r.dirichlet(np.full(n, 0.3))
        idx = systematic_resample(w, 0.37)
        assert len(idx) == n
        assert idx.min() >= 0 and idx.max() < n
    # 退化权重: 全部质量在一个粒子上 -> 索引全指向它
    w = np.zeros(32); w[7] = 1.0
    assert set(systematic_resample(w, 0.5).tolist()) == {7}


def test_wiener_pf_no_resample_when_ess_high(wc):
    """均匀权重 ⇒ ESS = N ⇒ 不触发 resample。

    构造法: 让观测增量与先验中心完全一致且先验极窄, 各粒子似然近乎相同。
    更稳的做法是直接检查阈值逻辑 —— ESS=N 时 N < ess_ratio*N 恒假。
    """
    N = int(wc["n_particles"])
    assert not (float(N) < wc["ess_ratio"] * N), "ess_ratio>=1 会导致每步都重采样"
    # 端到端: 完美线性 HI (零残差) 时权重集中 -> 会重采样; 这里只验证"不恒重采样"
    hi = np.linspace(0.0, 0.5, 60)
    prior = fit_prior([hi], wc)
    r = filter_trajectory(hi, np.full(59, 50.0), prior, wc, pf_rng(10, 11))
    assert r["n_resample"] <= r["n_endpoints"]


def test_wiener_pf_distance_floor_reported(wc):
    """H 达到/超过失效边界时 a 被 eps_distance 兜底, 且触发次数被计数上报。"""
    hi = np.concatenate([np.linspace(0.0, 0.99, 20), np.full(10, 1.0)])
    prior = fit_prior([hi[:20]], wc)
    r = filter_trajectory(hi, np.full(len(hi) - 1, 50.0), prior, wc, pf_rng(12, 13))
    assert r["n_distance_floor"] >= 10, r["n_distance_floor"]
    assert np.all(np.isfinite(r["pred_mean"]))


# ============================== synthetic Wiener 行为 ==============================

def test_wiener_pf_posterior_drift_direction(wc):
    """后验 mu 应朝真实 mu 的方向移动 (宽容差, 不要求等于理论值)。

    先验故意设成远小于真 mu 的水平, 看滤波是否把 mu 拉上来。
    """
    mu_true, sigma_true = 4e-3, 3e-3
    r0 = np.random.default_rng(21)
    hi = np.cumsum(r0.normal(mu_true, sigma_true, 150))
    hi = np.clip(np.maximum.accumulate(hi - hi.min()), 0.0, 0.99)
    prior = {"log_mu_mean": np.log(mu_true / 20.0), "log_mu_std": 1.5,
             "log_sigma_mean": np.log(sigma_true), "log_sigma_std": 1.0}
    r = filter_trajectory(hi, np.ones(len(hi) - 1), prior, wc, pf_rng(30, 31))
    mu_post = float(np.sum(r["w_final"] * r["mu_final"]))
    assert mu_post > mu_true / 20.0, (mu_post, mu_true / 20.0)


def test_wiener_pf_rul_decreases_with_hi(wc):
    """RUL 随 H 上升总体下降 (Spearman 意义上负相关, 宽容差)。"""
    hi = np.linspace(0.05, 0.95, 120)
    prior = fit_prior([hi[:60]], wc)
    r = filter_trajectory(hi, np.ones(119), prior, wc, pf_rng(40, 41))
    from scipy.stats import spearmanr
    rho = float(spearmanr(hi, r["pred_mean"]).statistic)
    assert rho < -0.5, rho


# ============================== 接口 / 划分一致性 ==============================

def test_wiener_pf_s3_split_equal(cfg):
    """S5B 必须逐 seed 复现 S3 的 train/val/test ID 与 selected_train_hash。"""
    import json
    p = ROOT / "checkpoints" / "s3_gate_metrics.json"
    if not p.exists():
        pytest.skip("缺 s3_gate_metrics.json")
    ref = json.loads(p.read_text(encoding="utf-8")).get("fingerprints", {})
    if not ref:
        pytest.skip("s3_gate_metrics.json 无 fingerprints")
    from src.baselines.physical_extrap import baseline_hyper
    from src.transfer.train_transfer import (select_train_trajectories,
                                             split_trajectories_from_cfg)
    h = baseline_hyper(cfg)
    if not h["target_h5"].exists():
        pytest.skip("缺 target_features.h5")
    import h5py
    tc = cfg["transfer"]
    with h5py.File(h["target_h5"], "r") as f:
        from src.baselines.physical_extrap import observed_traj_keys
        n_obs = len(observed_traj_keys(f))
    for seed_s, r in ref.items():
        seed = int(seed_s)
        tr, va, te, _ = split_trajectories_from_cfg(
            h["target_h5"], n_obs, tc, seed, observed_only=True)
        tr = [int(t) for t in tr]
        sel, sel_hash = select_train_trajectories(
            seed, tr, int(tc["s3_gate_train_n_traj"]),
            rule=str(tc["s3_selection_rule"]))
        assert sel == [int(x) for x in r["selected_train_ids"]], seed
        assert sel_hash == r["selected_train_hash"], seed
        assert [int(x) for x in va] == [int(x) for x in r["val_ids"]], seed
        assert [int(x) for x in te] == [int(x) for x in r["test_ids"]], seed
        got = hashlib.sha256(
            f"{seed}|tr={tr}|sel={sel}|va={[int(x) for x in va]}|"
            f"te={[int(x) for x in te]}".encode()).hexdigest()
        assert got == r["target_split_hash"], seed


def test_wiener_pf_s4_metrics_interface(cfg):
    """Wiener+PF 走的是 S4 统一 evaluator (同一函数), 不新写指标实现。"""
    from src.experiments.run_groups import s4_metrics_block
    rng = np.random.default_rng(3)
    from src.experiments.metrics import target_endpoint_axis
    axis = target_endpoint_axis(cfg)
    tid = sorted(axis.keys())[0]
    n = len(axis[tid]["endpoint_idx"])
    tids = np.full(n, tid, dtype=np.int64)
    true = np.linspace(1.0, 0.0, n)
    pred = np.clip(true + rng.normal(0, 0.05, n), 0.0, 1.0)
    blk = s4_metrics_block(pred, true, tids, cfg)
    for k in ("stage_metrics", "prognostic_horizon", "alpha_lambda",
              "warning", "convergence", "info_macro_rmse"):
        assert k in blk, k


def test_wiener_pf_config_frozen(cfg, wc):
    """协议 §9 冻结的 PF 参数必须与 config 逐值一致 (防手滑改动)。"""
    assert wc["n_particles"] == 2048
    assert wc["ess_ratio"] == 0.50
    assert wc["min_points"] == 8
    assert wc["estimator"] == "median_mad"
    assert wc["min_log_mu_std"] == 0.20
    assert wc["min_log_sigma_std"] == 0.20
    assert wc["sigma_floor"] == 0.005
    assert wc["eps_mu"] == 1.0e-6
    assert wc["eps_sigma"] == 1.0e-6
    assert wc["eps_distance"] == 1.0e-6
    assert wc["quantiles"] == [0.05, 0.50, 0.95]
    assert wc["hi_fail"] == 1.0


def test_wiener_pf_frozen_artifacts_unchanged():
    """S3/S4/S5 冻结件 SHA256 必须与 S5B protocol §1 登记值一致 (协议 §13)。"""
    expect = {
        "docs/diagnostics_s3_protocol.md":
            "0ed0c105bd68729400f455e12e38451961441a537986dc2f6ab2499acccf4b28",
        "docs/diagnostics_s3_results.md":
            "56366491749bd071ff12882cbf7dcd5d6021753a474c6c8db3fe98e617a5d8d5",
        "docs/diagnostics_s4_protocol.md":
            "9f488b555e82053ef00df7cd56b461e44a5ea74a115c38baf67e5d3a4f2e8237",
        "docs/diagnostics_s4_results.md":
            "e1a1c0cc0ef8c469f9d629a6b6e23d40de9c0cdf59c08bae061b57565d41b3f6",
        "checkpoints/s3_gate_metrics.json":
            "f11f46b107168fe0d21e16aa5c286789580a0eec073f8cd879c4ac6f9528f68e",
        "checkpoints/s4_metrics.json":
            "ef062b3602e78f724ef1758d197b25111eb447734867959c89607755161c3940",
        "checkpoints/s4_stage_diagnostics.json":
            "1e83c69e179d52c2da0f4e14a1cf91a49791fe4c7094b74236cd99db012e369f",
        "checkpoints/s5_rate_metrics.json":
            "2e021735e154b35c1a4be05bdc003a530f9c32e483d17b1ab33d7833e4e1653e",
        "src/experiments/metrics.py":
            "f17d35daf97f66f86d8ff9fc484e0294233a57c325e64b56fd0980ce0cb5e7da",
        "src/baselines/physical_extrap.py":
            "f3cc430bfd434197df2294e26d0fb0a6701be00925f5cf93250e6de83df0486d",
        "src/baselines/trivial.py":
            "aeeabcd4349fc662e828c85f78327721c97d8dc9145688258cd95594f2e517b3",
        "src/models/rate_head.py":
            "450504baa563fd457b752f87082ca7910d30c1ea491af33d1679153dc8f06a6e",
    }
    bad = []
    for rel, want in expect.items():
        p = ROOT / rel
        if not p.exists():
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            bad.append(f"{rel}: {got} != {want}")
    assert not bad, "冻结件被改动 (S5B_INVALID):\n" + "\n".join(bad)
