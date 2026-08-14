"""baselines/wiener_pf.py — S5B: Wiener 过程 + 粒子滤波统计寿命预测基线

协议: docs/diagnostics_s5b_protocol.md (SHA256 于结果文档中记录)。
本模块**只做统计推断**, 不含任何神经网络训练, 不修改任何已冻结结果。

退化模型 (协议 §3):
    dH(t) = mu·dt + sigma·dW(t)        mu > 0, sigma > 0
    H_{k+1} ~ Normal(H_k + mu·Δt, sigma²·Δt + sigma_obs²)
    失效边界固定 H_fail = 1.0 (HI_B 已归一)

粒子状态 (协议 §4): (log_mu, log_sigma) —— **H 不入粒子**, 直接以观测 hi_b 为当前
状态。理由: hi_b 是自校准且经 cummax 单调化的低噪声量, 再引入 H_latent 会让
sigma 与 sigma_obs 的分解在单条轨迹上几乎无信息。代价: sigma 是"过程+观测"的
合成扩散强度 —— 结果文档明示, 不声称它是纯过程噪声。

先验 (协议 §5): 只用该 seed 的 5 条 target train 轨迹的**截断后可见段** (HI<=0.55)
的 HI 增量, median + 1.4826·MAD 稳健估计。`fit_prior()` 的签名**根本不接收**
RUL / EOL —— 接口层面封死偷看真值的可能。

在线滤波 (协议 §6): 严格因果, 逐 endpoint 先出预测再看观测, ESS 低于阈值做
systematic resample。禁止对整条 test 轨迹一次性拟合 mu。

首达 (协议 §7): 逆高斯 (Wald), mean=a/mu, shape=a²/sigma²; 方差 a·sigma²/mu³ 与
S5 rate-model 的 first_passage_var **同式**, 两阶段换算口径一致。

用法:
    python -m src.baselines.wiener_pf --config configs/wheel.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

MAD_TO_SD = 1.4826          # 1/Phi^-1(0.75): MAD -> 正态标准差一致性因子


# ============================================================
# 配置读取 (严格: 缺键即抛, 不静默用默认值)
# ============================================================

def wiener_pf_cfg(cfg: dict) -> dict:
    """严格读取 baselines.wiener_pf (协议 §9 冻结值)。

    与 run_groups.rate_cfg / prognostics_cfg 同纪律: **缺键抛 KeyError**, 绝不
    静默填默认值 —— 否则"参数走 config"这条纪律形同虚设 (改了代码里的默认值,
    protocol 里冻结的数字就悄悄失效了)。
    """
    w = cfg["baselines"]["wiener_pf"]
    p, o, n = w["prior"], w["observation"], w["numerical"]
    out = {
        "n_particles": int(w["n_particles"]),
        "ess_ratio": float(w["ess_ratio"]),
        "min_points": int(w["min_points"]),
        "estimator": str(p["estimator"]),
        "min_log_mu_std": float(p["min_log_mu_std"]),
        "min_log_sigma_std": float(p["min_log_sigma_std"]),
        "sigma_floor": float(o["sigma_floor"]),
        "eps_mu": float(n["eps_mu"]),
        "eps_sigma": float(n["eps_sigma"]),
        "eps_distance": float(n["eps_distance"]),
        "quantiles": [float(q) for q in w["quantiles"]],
        "hi_fail": float(cfg["baselines"]["hi_target"]),
    }
    if out["estimator"] != "median_mad":
        raise ValueError(f"未知 prior.estimator={out['estimator']!r} (协议 §5 只冻结 median_mad)")
    if not (0.0 < out["ess_ratio"] <= 1.0):
        raise ValueError(f"ess_ratio 应在 (0,1], 实际 {out['ess_ratio']}")
    if out["quantiles"] != [0.05, 0.50, 0.95]:
        raise ValueError(f"quantiles 应为 [0.05,0.5,0.95] (协议 §9), 实际 {out['quantiles']}")
    return out


def pf_rng(experiment_seed: int, trajectory_id: int) -> np.random.Generator:
    """由 (experiment_seed, trajectory_id) 派生确定性 PF RNG (协议 §10)。

    **禁止 np.random.seed() 全局播种** —— 全局状态会让"某条轨迹的采样"依赖于
    它前面跑过多少条轨迹, 换个循环顺序结果就变, 复现性直接失效。
    """
    h = hashlib.sha256(f"wiener_pf|{int(experiment_seed)}|{int(trajectory_id)}".encode())
    return np.random.default_rng(int.from_bytes(h.digest()[:8], "big") % (2 ** 63))


# ============================================================
# 先验估计 (train-only, 接口不接收 RUL/EOL)
# ============================================================

def robust_increment_stats(hi_visible: np.ndarray, eps_mu: float,
                           eps_sigma: float) -> tuple:
    """单条轨迹可见段 HI 增量的稳健 (mu, sigma) 估计。

    mu    = median(diff(hi))
    sigma = 1.4826 · MAD(diff(hi) - mu)

    只吃 HI 序列, 无从访问 RUL / EOL。mu <= 0 (可见段整体无上升趋势) 时用
    eps_mu 兜底并由调用方计数上报 —— 不静默丢弃该轨迹。
    """
    hi = np.asarray(hi_visible, dtype=float)
    d = np.diff(hi)
    if len(d) < 2:
        return float(eps_mu), float(eps_sigma), True
    mu = float(np.median(d))
    sd = MAD_TO_SD * float(np.median(np.abs(d - mu)))
    floored = mu <= 0.0
    return (max(mu, float(eps_mu)), max(sd, float(eps_sigma)), bool(floored))


def fit_prior(hi_segments, wc: dict) -> dict:
    """由 train 可见段 HI 序列估计 log_mu / log_sigma 的高斯先验 (协议 §5)。

    **签名只接收 (hi_segments, wc)** —— 没有 rul / eol / event / fail_time 入口。
    这是"train 轨迹也是右删失"这条最关键契约的接口级落点: 目标域 train 不可见
    真实 EOL, 任何以 time-to-EOL 为拟合目标的先验都是偷看真值。
    (tests/test_wiener_pf.py::test_wiener_pf_prior_does_not_accept_rul / _eol 钉死)

    hi_segments: list[np.ndarray], 每条 = 一条 train 轨迹**截断后可见**的 hi_b。
    返回 {log_mu_mean, log_mu_std, log_sigma_mean, log_sigma_std, per_traj, ...}
    """
    if not len(hi_segments):
        raise ValueError("fit_prior 收到空 train 段列表")
    mus, sgs, n_floored = [], [], 0
    per = []
    for seg in hi_segments:
        m, s, fl = robust_increment_stats(seg, wc["eps_mu"], wc["eps_sigma"])
        mus.append(m)
        sgs.append(s)
        n_floored += int(fl)
        per.append({"n": int(len(seg)), "mu": float(m), "sigma": float(s),
                    "mu_floored": bool(fl),
                    "hi_span": float(np.max(seg) - np.min(seg)) if len(seg) else 0.0})
    lmu = np.log(np.asarray(mus, dtype=float))
    lsg = np.log(np.asarray(sgs, dtype=float))

    def med_mad(v, floor):
        m = float(np.median(v))
        s = MAD_TO_SD * float(np.median(np.abs(v - m)))
        # 只有 5 条轨迹, MAD 可能退化为 0 -> 先验方差 0 会让粒子云塌成一点。
        # 下界是数值稳健性措施 (协议 §5, 出数前写定), 不是为改善 test RMSE。
        return m, max(s, float(floor)), bool(s < float(floor))

    lm, lms, lm_fl = med_mad(lmu, wc["min_log_mu_std"])
    ls, lss, ls_fl = med_mad(lsg, wc["min_log_sigma_std"])
    return {
        "log_mu_mean": lm, "log_mu_std": lms,
        "log_sigma_mean": ls, "log_sigma_std": lss,
        "mu_median": float(np.median(mus)), "sigma_median": float(np.median(sgs)),
        "n_train_traj": int(len(hi_segments)),
        "n_mu_floored": int(n_floored),
        "log_mu_std_floor_hit": lm_fl, "log_sigma_std_floor_hit": ls_fl,
        "per_traj": per,
    }


# ============================================================
# 在线粒子滤波
# ============================================================

def systematic_resample(w: np.ndarray, u: float) -> np.ndarray:
    """systematic resample 的索引 (粒子数守恒)。u ∈ [0,1) 为单个均匀随机数。

    比 multinomial 方差更低, 且只消耗 1 个随机数 —— 对可复现性友好。
    """
    n = len(w)
    pos = (np.arange(n, dtype=float) + float(u)) / n
    cum = np.cumsum(w)
    cum[-1] = 1.0                      # 防浮点累加使末项 < 1 导致 searchsorted 越界
    return np.searchsorted(cum, pos, side="left").clip(0, n - 1)


def _ig_quantiles_from_particles(a: float, mu: np.ndarray, sigma: np.ndarray,
                                 w: np.ndarray, qs, rng, n_draw: int = 4096):
    """加权粒子的逆高斯混合分布分位数 (协议 §7)。

    为什么不直接对 mean_i = a/mu_i 取分位: 那样丢掉 IG 自身的右偏尾巴, 区间会
    系统性偏窄 —— 覆盖率会被人为压低, 属于对自己不利但**不诚实**的口径。
    按权重给每个粒子分配采样数, 再从各自的 IG 抽样, 混合后取经验分位。
    """
    lam = a * a / np.maximum(sigma * sigma, 1e-300)      # 形状参数 a²/sigma²
    mean = a / mu
    cnt = rng.multinomial(int(n_draw), w)
    nz = cnt > 0
    if not nz.any():
        return np.full(len(qs), float(np.sum(w * mean)))
    draws = [rng.wald(mean[i], lam[i], size=int(cnt[i])) for i in np.nonzero(nz)[0]]
    s = np.concatenate(draws)
    return np.quantile(s, qs)


def filter_trajectory(hi_endpoints: np.ndarray, dt: np.ndarray, prior: dict,
                      wc: dict, rng: np.random.Generator) -> dict:
    """对单条 test 轨迹做在线粒子滤波 + 首达 RUL 预测 (协议 §6/§7)。

    hi_endpoints: 该轨迹在 evaluator endpoint 上的 hi_b (按时间升序)。
    dt          : 相邻 endpoint 的实际采样点间隔 (len = len(hi)-1, 通常恒为 stride)。

    **严格因果**: 第 k 个预测只用到 hi_endpoints[:k+1]。第 0 点无前一观测 ⇒
    不做权重更新, 用纯先验出预测。返回逐 endpoint 的 mean/median/q05/q95 及诊断。
    """
    hi = np.asarray(hi_endpoints, dtype=float)
    n = len(hi)
    N = wc["n_particles"]
    qs = wc["quantiles"]
    # 粒子初始化: 从先验抽 (log_mu, log_sigma)
    log_mu = rng.normal(prior["log_mu_mean"], prior["log_mu_std"], N)
    log_sg = rng.normal(prior["log_sigma_mean"], prior["log_sigma_std"], N)
    mu = np.maximum(np.exp(log_mu), wc["eps_mu"])
    sg = np.maximum(np.exp(log_sg), wc["eps_sigma"])
    w = np.full(N, 1.0 / N)

    pm, pmed, q05, q95 = (np.empty(n) for _ in range(4))
    ess_hist = np.empty(n)
    n_resample = 0
    n_dist_floor = 0
    n_prior_only = 0
    n_degenerate = 0                  # 似然全下溢 -> 保留上一步权重的次数

    for k in range(n):
        # ---- 1) 先出预测 (只用 <= k 的信息) ----
        if k > 0:
            # ---- 2/3) 读当前观测, 按 observation likelihood 更新权重 ----
            d_obs = hi[k] - hi[k - 1]
            dtk = float(dt[k - 1])
            var = sg * sg * dtk + wc["sigma_floor"] ** 2
            resid = d_obs - mu * dtk
            # log-likelihood 后 exp, 减最大值防下溢 (数值稳定的标准做法)
            ll = -0.5 * (np.log(var) + resid * resid / var)
            ll -= ll.max()
            wn = w * np.exp(ll)
            s = wn.sum()
            if s <= 0.0 or not np.isfinite(s):
                n_degenerate += 1     # 保留上一步权重, 不伪造均匀分布
            else:
                w = wn / s            # ---- 4) normalize ----
            # ---- 5/6) ESS + systematic resample ----
            ess = 1.0 / float(np.sum(w * w))
            if ess < wc["ess_ratio"] * N:
                idx = systematic_resample(w, float(rng.random()))
                mu, sg = mu[idx], sg[idx]
                w = np.full(N, 1.0 / N)
                n_resample += 1
        else:
            ess = float(N)
        ess_hist[k] = ess
        if k < wc["min_points"]:
            n_prior_only += 1

        # ---- 7/8) 首达 RUL 分布 (协议 §7) ----
        a_raw = wc["hi_fail"] - hi[k]
        a = max(a_raw, wc["eps_distance"])
        if a_raw < wc["eps_distance"]:
            n_dist_floor += 1
        pm[k] = float(np.sum(w * (a / mu)))
        ql = _ig_quantiles_from_particles(a, mu, sg, w, qs, rng)
        q05[k], pmed[k], q95[k] = float(ql[0]), float(ql[1]), float(ql[2])

    return {"pred_mean": pm, "pred_median": pmed, "q05": q05, "q95": q95,
            "ess": ess_hist, "n_resample": int(n_resample),
            "n_distance_floor": int(n_dist_floor),
            "n_prior_only": int(n_prior_only),
            "n_likelihood_degenerate": int(n_degenerate),
            "n_endpoints": int(n),
            # 末态粒子: 供测试检查正性 / 后验漂移方向, 不参与任何指标计算
            "mu_final": mu, "sigma_final": sg, "w_final": w}


# ============================================================
# seed 级评估 (复用 S3 数据契约 + S4 统一 evaluator)
# ============================================================

def _visible_segment(hi: np.ndarray, truncate_hi: float) -> np.ndarray:
    """train 轨迹按 truncate_hi 截断后的可见段 (与 S3 load_target 同规则)。

    S3 的截断规则: 首个 HI > truncate_hi 的位置记为 c, 可见段 = [0, c-1]。
    这里必须逐字复现, 否则先验用到了 S3 训练时看不见的数据 —— 那是另一种泄漏。
    """
    over = np.nonzero(np.asarray(hi, dtype=float) > float(truncate_hi))[0]
    return hi[:int(over[0])] if len(over) else hi


def evaluate_wiener_pf(cfg: dict, seed: int, verbose: bool = True) -> dict | None:
    """单 seed 的 Wiener+PF 评估。复用 S3 split/选取 + S4 统一 evaluator。

    返回 {calibers, stage_metrics, prognostic_horizon, alpha_lambda, warning,
          convergence, interval, prior, diagnostics, fingerprints, runtime}。
    """
    from src.baselines.physical_extrap import (baseline_hyper, caliber_mask,
                                               caliber_metrics,
                                               clip_to_label_domain,
                                               eval_point_indices,
                                               observed_traj_keys)
    from src.experiments.run_groups import s4_metrics_block
    from src.transfer.train_transfer import (describe_eol_split,
                                             select_train_trajectories,
                                             split_trajectories_from_cfg)

    wc = wiener_pf_cfg(cfg)
    h = baseline_hyper(cfg)
    tc = cfg["transfer"]
    truncate_hi = float(tc["target_truncate_hi"])
    n_train_sel = int(tc["s3_gate_train_n_traj"])
    rule = str(tc["s3_selection_rule"])
    h5p = h["target_h5"]
    if not h5p.exists():
        print(f"!! 缺 {h5p}; 先 python -m src.sim.build_hi --report")
        return None

    with h5py.File(h5p, "r") as f:
        keys = observed_traj_keys(f)
        n_full = int(len(f[sorted(f.keys())[0]]["hi_b"]))
        tr_all, va, te, eol = split_trajectories_from_cfg(
            h5p, len(keys), tc, seed, observed_only=True)
        tr_all = [int(t) for t in tr_all]
        va = [int(t) for t in va]
        te = [int(t) for t in te]
        sel, sel_hash = select_train_trajectories(seed, tr_all, n_train_sel, rule=rule)
        # rul_scale 与 S3/S5 完全同源 (协议 §2): cap_ratio × n_full, 不含 EOL 信息
        rul_scale = float(cfg["source"]["rul_cap_ratio"]) * float(n_full)

        if verbose:
            print(f">> 轨迹划分 train {len(tr_all)} / val {len(va)} / test {len(te)}"
                  f"   (S5B 先验只用 train {len(sel)} 条)")
            describe_eol_split(eol, tr_all, va, te)

        # ---- 先验: 只用被选中的 train 轨迹的截断后可见段 (协议 §5) ----
        t0 = time.perf_counter()
        segs = [_visible_segment(np.asarray(f[keys[int(t)]]["hi_b"][:], dtype=float),
                                 truncate_hi) for t in sel]
        prior = fit_prior(segs, wc)
        t_prior = time.perf_counter() - t0

        # ---- 在线滤波: 逐 test 轨迹 ----
        t1 = time.perf_counter()
        P, PMED, Q05, Q95, T, TID = [], [], [], [], [], []
        diag = {"n_resample": 0, "n_distance_floor": 0, "n_prior_only": 0,
                "n_likelihood_degenerate": 0, "n_endpoints": 0, "n_clip_upper": 0}
        for tid in te:
            g = f[keys[int(tid)]]
            hi = np.asarray(g["hi_b"][:], dtype=float)
            r_true = np.asarray(g["rul"][:], dtype=float) / rul_scale
            pts = eval_point_indices(len(hi), h["L"], h["stride"])
            if not len(pts):
                continue
            dt = np.diff(pts).astype(float)
            out = filter_trajectory(hi[pts], dt, prior, wc, pf_rng(seed, int(tid)))
            # 归一到项目统一 RUL 轴, 并与 physical_extrap / hi_extrap 同口径截顶
            pn = out["pred_mean"] / rul_scale
            diag["n_clip_upper"] += int(np.sum(pn > 1.0))
            P.append(clip_to_label_domain(pn))
            PMED.append(clip_to_label_domain(out["pred_median"] / rul_scale))
            Q05.append(clip_to_label_domain(out["q05"] / rul_scale))
            Q95.append(clip_to_label_domain(out["q95"] / rul_scale))
            T.append(r_true[pts])
            TID.append(np.full(len(pts), int(tid), dtype=np.int64))
            for k in ("n_resample", "n_distance_floor", "n_prior_only",
                      "n_likelihood_degenerate", "n_endpoints"):
                diag[k] += out[k]
        t_filter = time.perf_counter() - t1

    if not T:
        return None
    p = np.concatenate(P)
    pmed = np.concatenate(PMED)
    q05 = np.concatenate(Q05)
    q95 = np.concatenate(Q95)
    t = np.concatenate(T)
    tids = np.concatenate(TID)

    # ---- S4 统一 evaluator (协议 §11): 与所有其它方法同一实现 ----
    out = caliber_metrics(p, t, tids, cap_eps=h["cap_eps"])
    out.update(s4_metrics_block(p, t, tids, cfg))

    # ---- 区间 (协议 §8): 信息区口径, 与 S5 rate-model 的 iv90 同口径 ----
    m_info = caliber_mask(t, "info", h["cap_eps"])
    axis_tau = _endpoint_tau(cfg, tids)
    m_pre = m_info & (axis_tau <= 1.0)
    def cov(m):
        return (float(np.mean((t[m] >= q05[m]) & (t[m] <= q95[m])))
                if m.any() else float("nan"))
    out["interval"] = {
        "z_equivalent": "IG q05/q95", "nominal_coverage": 0.90,
        "coverage_info": cov(m_info),
        "coverage_info_before_eol": cov(m_pre),
        "mean_width_info": float(np.mean(q95[m_info] - q05[m_info]))
                           if m_info.any() else float("nan"),
        "n_info": int(m_info.sum()), "n_info_before_eol": int(m_pre.sum()),
    }
    out["median_info_macro_rmse"] = _macro_rmse(pmed, t, tids, m_info)
    out["prior"] = prior
    out["diagnostics"] = diag
    out["runtime"] = {"prior_fit_s": float(t_prior), "filter_s": float(t_filter),
                      "total_s": float(t_prior + t_filter),
                      "n_test_traj": int(len(te)), "n_points": int(len(t))}
    out["fingerprints"] = {
        "selected_train_hash": sel_hash,
        "target_split_hash": hashlib.sha256(
            f"{seed}|tr={tr_all}|sel={sel}|va={va}|te={te}".encode()).hexdigest(),
        "selected_train_ids": sel, "val_ids": va, "test_ids": te,
        "prediction_hash": hashlib.sha256(repr(p.tolist()).encode()).hexdigest(),
    }
    out["meta"] = {"seed": int(seed), "rul_scale": rul_scale,
                   "truncate_hi": truncate_hi, "n_full": n_full,
                   "cap_eps": h["cap_eps"]}
    out["_raw"] = {"pred": p, "true": t, "tids": tids,
                   "q05": q05, "q95": q95, "pred_median": pmed}
    return out


def _macro_rmse(p, t, tids, mask) -> float:
    """掩码内的 trajectory-macro RMSE (先逐轨迹后等权平均, 与 S4 同口径)。"""
    v = []
    for u in np.unique(tids[mask]):
        m = mask & (tids == u)
        if m.any():
            v.append(float(np.sqrt(np.mean((p[m] - t[m]) ** 2))))
    return float(np.mean(v)) if v else float("nan")


def _endpoint_tau(cfg: dict, tids: np.ndarray) -> np.ndarray:
    """逐预测点的 tau (归一寿命位置), 取自 S4 唯一时间轴 target_endpoint_axis。

    直接复用 attach_axis —— 它会在 endpoint 数不匹配时报错而非静默截断, 这正是
    我们想要的: tau 错位会让 pre-EOL coverage 变成无声的错数字。
    """
    from src.experiments.metrics import attach_axis, target_endpoint_axis
    tau, _hi, _ev = attach_axis(tids, target_endpoint_axis(cfg))
    return tau


# ============================================================
# CLI
# ============================================================

def main():
    from src.utils import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel.yaml")
    ap.add_argument("--seed", type=int, default=None,
                    help="默认取 transfer.s3_gate_seeds 的首个 seed")
    args = ap.parse_args()
    cfg = load_config(str(ROOT / args.config))
    seed = args.seed if args.seed is not None else int(cfg["transfer"]["s3_gate_seeds"][0])
    wc = wiener_pf_cfg(cfg)
    print("=" * 88)
    print(f"S5B Wiener + Particle Filter 基线 (seed {seed})")
    print("=" * 88)
    print(f">> PF: N={wc['n_particles']} ess_ratio={wc['ess_ratio']} "
          f"min_points={wc['min_points']} sigma_floor={wc['sigma_floor']}")
    print(f">> prior: {wc['estimator']} min_log_mu_std={wc['min_log_mu_std']} "
          f"min_log_sigma_std={wc['min_log_sigma_std']}")

    r = evaluate_wiener_pf(cfg, seed)
    if r is None:
        raise SystemExit("评估失败 (缺数据?)")
    pr, c = r["prior"], r["calibers"]
    print(f"\n-- Wiener 先验 (train-only, {pr['n_train_traj']} 条) --")
    print(f"   log_mu    ~ N({pr['log_mu_mean']:.4f}, {pr['log_mu_std']:.4f})"
          f"   -> mu_median    = {pr['mu_median']:.3e}")
    print(f"   log_sigma ~ N({pr['log_sigma_mean']:.4f}, {pr['log_sigma_std']:.4f})"
          f"   -> sigma_median = {pr['sigma_median']:.3e}")
    print(f"   mu 兜底条数 {pr['n_mu_floored']}  std 下界触发 "
          f"mu={pr['log_mu_std_floor_hit']} sigma={pr['log_sigma_std_floor_hit']}")

    ss = _shape(r["_raw"], r["meta"]["cap_eps"])
    print(f"\n-- 主指标 (info 口径) --")
    print(f"   info macro RMSE = {c['info']['macro']['rmse']:.4f}"
          f"   pooled = {c['info']['pooled']['rmse']:.4f}"
          f"   MAE = {c['info']['macro']['mae']:.4f}")
    print(f"   full macro RMSE = {c['full']['macro']['rmse']:.4f}")
    print(f"   macro corr = {ss['macro_corr']:+.4f}   pooled corr = {ss['pooled_corr']:+.4f}")
    print(f"   PSR = {ss['psr']:.4f}  (pred_std {ss['pred_std']:.4f} / "
          f"true_std {ss['true_std']:.4f})")
    print(f"\n-- 预后性 --")
    print(f"   PH (macro) = {r['prognostic_horizon']['macro_ph']:.4f}")
    for lam, v in r["alpha_lambda"]["by_lambda"].items():
        print(f"   alpha-lambda@{lam} = {v['accuracy']:.4f} (eligible {v['eligible_count']})")
    print(f"   convergence = {r['convergence']['macro_convergence']:.4f}")
    w = r["warning"]
    print(f"\n-- 预警 --")
    print(f"   coverage_before_eol = {w['coverage_before_eol']:.4f}"
          f"   miss_rate_before_eol = {w['miss_rate_before_eol']:.4f}")
    print(f"   false_alarm_rate = {w['false_alarm_rate']}"
          f"   conditional_lead_before_eol = {w['conditional_lead_before_eol']:.4f}"
          f" (n={w['conditional_lead_before_eol_n']})")
    iv = r["interval"]
    print(f"\n-- 不确定性区间 (标称 90%) --")
    print(f"   coverage_info = {iv['coverage_info']:.4f}"
          f"   before_eol = {iv['coverage_info_before_eol']:.4f}"
          f"   mean_width = {iv['mean_width_info']:.4f}")
    print(f"   (S5 rate-model iv90 = 0.7290, 同口径对照)")
    rt, dg = r["runtime"], r["diagnostics"]
    print(f"\n-- 效率 (statistical / online prognostics baseline) --")
    print(f"   prior fit {rt['prior_fit_s']:.3f}s   filtering {rt['filter_s']:.3f}s"
          f"   total {rt['total_s']:.3f}s")
    print(f"   test 轨迹 {rt['n_test_traj']}  预测点 {rt['n_points']}")
    print(f"\n-- 数值保护触发率 (协议 §7: 必须报出, 不做 silent clipping) --")
    n = max(dg["n_endpoints"], 1)
    print(f"   resample {dg['n_resample']}  distance_floor {dg['n_distance_floor']}"
          f" ({dg['n_distance_floor'] / n:.4f})  clip_upper {dg['n_clip_upper']}"
          f" ({dg['n_clip_upper'] / n:.4f})")
    print(f"   prior_only {dg['n_prior_only']} ({dg['n_prior_only'] / n:.4f})"
          f"  likelihood_degenerate {dg['n_likelihood_degenerate']}")
    fp = r["fingerprints"]
    print(f"\n-- 指纹 --")
    print(f"   split_hash      = {fp['target_split_hash']}")
    print(f"   selected_train  = {fp['selected_train_hash']}")
    print(f"   prediction_hash = {fp['prediction_hash']}")


def _shape(raw: dict, cap_eps: float) -> dict:
    """PSR / corr (info 掩码), 与 scripts/diag_generalization.shape_stats 同口径。"""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.diag_generalization import shape_stats
    return shape_stats(raw["pred"], raw["true"], raw["tids"], cap_eps)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
