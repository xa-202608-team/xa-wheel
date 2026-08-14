"""src/experiments/metrics.py — S4 分阶段 / 预后性 / 预警 评价指标

定位（协议 `docs/diagnostics_s4_protocol.md`）：**只增加评价能力，不改变任何训练行为**。
本模块被 `run_groups.eval_test`、`scripts/diag_s3_stages.py`、`scripts/run_s4_metrics.py`
以及 physical_extrap / trivial 基线共用同一份实现，保证"同划分、同取点、同口径、同指标"。

包含五组指标：

  1. `staged_metrics`        按 HI 健康阶段分箱的 RMSE/MAE/corr/PSR（§6）
  2. `prognostic_horizon`    首次进入并持续保持在 alpha 误差带内的归一提前量（§7）
  3. `alpha_lambda_accuracy` lambda 生命周期位置处是否落在 alpha 带内（§8）
  4. `warning_lead_time`     RUL 阈值 + persistence 的预警提前量 / 漏报 / 误报（§9）
  5. `convergence_metric`    归一寿命轴上 |误差| 曲线的面积（§10）

三条贯穿全模块的纪律：

* **macro 先按 trajectory 算再等权平均**，长轨迹绝不因点多而支配 macro（§6）。
* **同一 endpoint 只评价一次**：所有入口先经 `dedup_endpoints` 去重（§16-H）。
* **右删失轨迹不伪造 EOL**：`event_observed=False` 的轨迹一律返回 NaN + observable=False，
  绝不用"最后一个观测点"冒充 EOL（§7/§11）。空 bin 同样返回 NaN + n=0，不伪造 0。
"""
from __future__ import annotations

import numpy as np

# 空结果的统一占位: 绝不用 0.0 冒充"没有数据" (协议 §6/§20)
NAN = float("nan")

# 梯形积分: np.trapezoid 是 NumPy>=2.0 的新名字, 1.x 只有 np.trapz。
# 显式取别名而不是靠 try/except 包住调用点, 免得未来某次升级静默换了数值行为。
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ============================== 时间轴 (τ) 唯一来源 ==============================

def target_endpoint_axis(cfg: dict) -> dict:
    """逐轨迹 evaluator endpoint 的归一寿命位置 tau —— 全 S4 唯一的时间轴来源。

    模型侧 (TargetSeqDataset, K=1) 与基线侧 (eval_point_indices) 取的是同一批窗末
    时刻; 本函数按同一公式复算, 并给出

        tau(idx) = idx / eol_idx          (0 = 轨迹起点, 1 = EOL)

    tau 可 > 1: EOL 之后仍有采样点 (那段 RUL 恒为 0, info 口径会剔除)。

    event_observed 直接读 h5 attrs。**不允许**用"最后一个采样点"冒充 EOL ——
    删失轨迹返回 eol_idx=None / tau=None, 下游一律判 observable=False (协议 §7)。

    返回 {tid: {"endpoint_idx": np.ndarray, "tau": np.ndarray|None,
                "eol_idx": int|None, "event_observed": bool, "n_full": int}}
    tid 空间与 load_target(observed_only=True) 的 remap 严格同序 (都走 sorted(keys))。
    """
    import h5py

    from src.baselines.physical_extrap import eol_index, eval_point_indices

    tc = cfg["transfer"]
    L = int(cfg["model"]["input_len_L"])
    stride = int(tc.get("target_stride", 50))
    has_nodes = bool(tc.get("target_has_nodes", False))
    hi_key = "hi_array" if has_nodes else "hi_b"

    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    h5p = root / tc["target_feature_path"]

    axis, tid = {}, 0
    with h5py.File(h5p, "r") as f:
        for k in sorted(f.keys()):
            g = f[k]
            observed = bool(g.attrs.get("event_observed",
                                        np.asarray(g.get("label_fail", [])).any()))
            if not observed:
                continue                      # load_target(observed_only=True) 同样跳过
            n_full = int(len(g[hi_key]))
            idx = eval_point_indices(n_full, L, stride)
            e = eol_index(g)
            axis[tid] = {
                "key": str(k), "endpoint_idx": idx, "n_full": n_full,
                "eol_idx": int(e), "event_observed": True,
                "tau": (idx.astype(float) / float(e)) if e > 0 else None,
                "hi": np.asarray(g[hi_key][:], dtype=float)[idx],
            }
            tid += 1
    return axis


def attach_axis(tids, axis: dict):
    """把 `target_endpoint_axis` 的 (tau, hi) 按 loader 顺序贴到逐预测点上。

    TargetSeqDataset 以 groupby(tid) 构造样本, 同一 tid 内窗按 start 升序 —— 故对每个
    tid 按出现顺序依次取用其 endpoint 即可对齐。长度不符直接报错, **绝不截断凑数**:
    静默对齐错位会让 tau 整体偏移, 所有 PH / warning 数字变成无声的错。

    返回 (tau, hi, event_observed_dict)。
    """
    tids = np.asarray(tids, dtype=np.int64)
    ptr = {}
    tau = np.empty(len(tids), dtype=float)
    hi = np.empty(len(tids), dtype=float)
    for i, u in enumerate(tids):
        u = int(u)
        a = axis.get(u)
        if a is None:
            raise KeyError(f"tid={u} 不在 target_endpoint_axis 中 (划分口径不一致)")
        j = ptr.get(u, 0)
        if j >= len(a["endpoint_idx"]):
            raise ValueError(
                f"tid={u} 的预测点数 ({j + 1}) 超过 endpoint 数 "
                f"({len(a['endpoint_idx'])}) —— L/stride 口径不一致, 拒绝对齐")
        tau[i] = a["tau"][j] if a["tau"] is not None else NAN
        hi[i] = a["hi"][j]
        ptr[u] = j + 1
    for u, j in ptr.items():
        n = len(axis[int(u)]["endpoint_idx"])
        if j != n:
            raise ValueError(f"tid={u} 只用了 {j}/{n} 个 endpoint —— 对齐不完整")
    ev = {int(u): bool(axis[int(u)]["event_observed"]) for u in np.unique(tids)}
    return tau, hi, ev


# ============================== 通用工具 ==============================

def dedup_endpoints(tids, taus, *arrays):
    """同一 (trajectory, endpoint) 只保留首次出现 —— evaluator 端去重 (协议 §16-H)。

    为什么必须去重: 评估 loader 以 K=1 展开, 若上游取点重复 (stride 与 K 组合、或多次
    concat), 同一时刻会被计入两次, pooled 指标被静默加权。行为在此固定并被测试钉死。

    返回 (keep_mask, tids_kept, taus_kept, *arrays_kept)。保持原有相对顺序。
    """
    tids = np.asarray(tids, dtype=np.int64)
    taus = np.asarray(taus, dtype=float)
    seen = set()
    keep = np.zeros(len(tids), dtype=bool)
    for i in range(len(tids)):
        k = (int(tids[i]), float(taus[i]))
        if k not in seen:
            seen.add(k)
            keep[i] = True
    return (keep, tids[keep], taus[keep]) + tuple(np.asarray(a)[keep] for a in arrays)


def _corr(a, b) -> float:
    """Pearson 相关。任一侧方差为 0（常数输出）时返回 NaN，不返回 0。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return NAN
    sa, sb = a.std(), b.std()
    if sa <= 0 or sb <= 0:
        return NAN
    return float(np.corrcoef(a, b)[0, 1])


def _psr(pred, true) -> float:
    """pred_std / true_std —— 输出动态范围坍缩检测。true_std=0 时 NaN。"""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if len(true) < 2:
        return NAN
    ts = float(true.std())
    if ts <= 0:
        return NAN
    return float(pred.std() / ts)


def _macro(per_traj_values) -> float:
    """轨迹等权平均，忽略 NaN 轨迹；无有效轨迹返回 NaN。"""
    v = np.asarray([x for x in per_traj_values if np.isfinite(x)], dtype=float)
    return float(v.mean()) if len(v) else NAN


def tolerance_band(true_rul, alpha: float, absolute_floor: float):
    """alpha 误差带半宽 = max(alpha * true_rul, absolute_floor)（协议 §7）。

    为什么必须有 absolute_floor: true_rul → 0 时纯相对带 alpha*true 收缩到 0,
    任何有限精度的预测都必然"脱靶", PH 会被机械地判在 EOL 前一刻、alpha-lambda@0.7
    在临近失效处恒为 0 —— 那是度量的伪影而非模型缺陷。floor 走 config, 不硬编码。
    """
    t = np.abs(np.asarray(true_rul, dtype=float))
    return np.maximum(float(alpha) * t, float(absolute_floor))


def in_alpha_band(pred, true, alpha: float, absolute_floor: float):
    """|pred - true| <= max(alpha*true, absolute_floor) 的逐点布尔。"""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    return np.abs(pred - true) <= tolerance_band(true, alpha, absolute_floor)


# ============================== §6 分阶段指标 ==============================

def staged_metrics(pred, true, hi, trajectory_id, bins, tau=None) -> dict:
    """按 HI 健康阶段分箱的完整指标（协议 §6）。

    bins = [(lo, hi), ...]，半开区间 [lo, hi)。最后一个 bin 的上界建议取 1.000001
    以纳入 HI==1.0（config `evaluation.prognostics.hi_bins` 已如此冻结）。

    每个 bin 输出: n_points / n_trajectories / pooled_rmse / pooled_mae /
    macro_rmse / macro_mae / macro_corr / pred_std_true_std_ratio。

    macro_* 先按 trajectory 算再等权平均 —— 长轨迹绝不因点数多而支配 macro。
    空 bin 返回 NaN + n_points=0，**不伪造 0**。

    tau: 归一寿命位置，用作 endpoint 去重键（协议 §2 唯一 endpoint 规则）。
    缺省时退化为 (hi, true) 复合键 —— 在单调退化轨迹里 HI 与真实 RUL 都是时间的
    单调函数，故同一 (tid, hi, true) 必然是同一时刻。**不用行下标当键**：那样每行
    都天然唯一，去重形同虚设（本条曾被 test_duplicate_endpoint 抓出）。
    """
    pred_a = np.asarray(pred, dtype=float)
    true_a = np.asarray(true, dtype=float)
    hi_a = np.asarray(hi, dtype=float)
    if tau is None:
        # 复合键: 把 (hi, true) 打成一个可比标量对, 交给 dedup 的 (tid, key) 规则
        key = np.asarray([hash((round(float(h), 12), round(float(t), 12)))
                          for h, t in zip(hi_a, true_a)], dtype=float)
    else:
        key = np.asarray(tau, dtype=float)
    keep, tids, _k, pred, true, hi = dedup_endpoints(
        trajectory_id, key, pred_a, true_a, hi_a)
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    hi = np.asarray(hi, dtype=float)

    out = {"bins": [], "n_dropped_duplicate_endpoints": int((~keep).sum())}
    for lo, up in bins:
        m = (hi >= float(lo)) & (hi < float(up))
        rec = {"bin": [float(lo), float(up)], "n_points": int(m.sum())}
        if not m.any():
            rec.update({"n_trajectories": 0, "pooled_rmse": NAN, "pooled_mae": NAN,
                        "macro_rmse": NAN, "macro_mae": NAN, "macro_corr": NAN,
                        "pred_std_true_std_ratio": NAN,
                        "pred_mean": NAN, "pred_std": NAN,
                        "true_mean": NAN, "true_std": NAN})
            out["bins"].append(rec)
            continue
        p, t, g = pred[m], true[m], tids[m]
        uniq = np.unique(g)
        per_rmse, per_mae, per_corr = [], [], []
        for u in uniq:
            s = g == u
            per_rmse.append(float(np.sqrt(np.mean((p[s] - t[s]) ** 2))))
            per_mae.append(float(np.mean(np.abs(p[s] - t[s]))))
            per_corr.append(_corr(p[s], t[s]))
        rec.update({
            "n_trajectories": int(len(uniq)),
            "pooled_rmse": float(np.sqrt(np.mean((p - t) ** 2))),
            "pooled_mae": float(np.mean(np.abs(p - t))),
            "macro_rmse": _macro(per_rmse),
            "macro_mae": _macro(per_mae),
            "macro_corr": _macro(per_corr),
            "pred_std_true_std_ratio": _psr(p, t),
            "pred_mean": float(p.mean()), "pred_std": float(p.std()),
            "true_mean": float(t.mean()), "true_std": float(t.std()),
        })
        out["bins"].append(rec)
    return out


# ============================== §7 Prognostic Horizon ==============================

def prognostic_horizon(pred, true, tau, trajectory_id, event_observed,
                       alpha: float, absolute_floor: float) -> dict:
    """PH = 首次进入 alpha 带、且此后**始终**保持在带内的那个时刻到 EOL 的归一距离。

    tau ∈ [0,1] 为归一寿命位置（0 = 轨迹起点，1 = EOL）。逐轨迹按 tau 升序扫描，
    从后往前找最长的"末端连续命中"段，其起点 tau_hit 给出:

        PH_normalized = 1 - tau_hit

    即"还剩多少比例寿命时就已经稳定预测准了"，越大越好。

    右删失轨迹（event_observed=False）**没有真实 EOL**，tau 无从定义，一律
    PH=NaN / observable=False —— 绝不用最后一个观测点冒充 EOL（协议 §7/§11）。

    返回 {"per_trajectory": {tid: {...}}, "macro_ph": ..., "n_observable": ...,
          "n_censored": ...}。
    """
    keep, tids, taus, pred, true = dedup_endpoints(
        trajectory_id, tau, pred, true)
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    ev = {int(k): bool(v) for k, v in dict(event_observed).items()}

    per, n_obs, n_cen = {}, 0, 0
    for u in np.unique(tids):
        u = int(u)
        observable = ev.get(u, False)
        if not observable:
            n_cen += 1
            per[u] = {"ph_normalized": NAN, "ph_tau_hit": NAN,
                      "observable": False, "n_points": int((tids == u).sum()),
                      "reason": "right_censored_no_true_eol"}
            continue
        n_obs += 1
        s = tids == u
        order = np.argsort(taus[s], kind="stable")
        tt, pp, tr = taus[s][order], pred[s][order], true[s][order]
        hit = in_alpha_band(pp, tr, alpha, absolute_floor)
        # 从末尾往前找最长连续命中段 (要求"此后始终保持在带内")
        k = len(hit)
        while k > 0 and hit[k - 1]:
            k -= 1
        if k == len(hit):                       # 末点即脱靶 → 从未稳定进入
            per[u] = {"ph_normalized": 0.0, "ph_tau_hit": NAN, "observable": True,
                      "n_points": int(len(hit)), "reason": "never_stably_in_band"}
            continue
        tau_hit = float(tt[k])
        per[u] = {"ph_normalized": float(1.0 - tau_hit), "ph_tau_hit": tau_hit,
                  "observable": True, "n_points": int(len(hit)), "reason": "ok"}
    return {"per_trajectory": per,
            "macro_ph": _macro([v["ph_normalized"] for v in per.values()
                                if v["observable"]]),
            "n_observable": n_obs, "n_censored": n_cen,
            "alpha": float(alpha), "absolute_floor": float(absolute_floor)}


# ============================== §8 Alpha-Lambda ==============================

def alpha_lambda_accuracy(pred, true, tau, trajectory_id, event_observed,
                          alpha: float, absolute_floor: float, lambdas) -> dict:
    """在 lambda 生命周期位置处判断预测是否落入 alpha 误差带（协议 §8）。

    对每条 **event-observed** test 轨迹、每个 lambda，取 tau 最接近 lambda 的那个
    evaluator endpoint（每条轨迹只贡献 **1 个** 判定），检查是否命中 alpha 带。

    accuracy = success / eligible，其中 eligible = 有可用 endpoint 的观测轨迹数。
    **禁止用时间点数量当独立样本** —— 分母恒为轨迹数，长轨迹不会多投票（协议 §8）。

    删失轨迹不进入 eligible（无 EOL ⇒ 无 tau ⇒ 无法定位生命周期位置）。
    """
    keep, tids, taus, pred, true = dedup_endpoints(trajectory_id, tau, pred, true)
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    ev = {int(k): bool(v) for k, v in dict(event_observed).items()}

    out = {"alpha": float(alpha), "absolute_floor": float(absolute_floor),
           "by_lambda": {}, "per_trajectory": {}}
    for lam in lambdas:
        lam = float(lam)
        succ, elig, detail = 0, 0, {}
        for u in np.unique(tids):
            u = int(u)
            if not ev.get(u, False):
                detail[u] = {"eligible": False, "hit": None,
                             "reason": "right_censored"}
                continue
            s = tids == u
            tt, pp, tr = taus[s], pred[s], true[s]
            if not len(tt):
                detail[u] = {"eligible": False, "hit": None, "reason": "no_endpoint"}
                continue
            j = int(np.argmin(np.abs(tt - lam)))      # 每轨迹恰好一个判定点
            hit = bool(in_alpha_band(pp[j:j + 1], tr[j:j + 1],
                                     alpha, absolute_floor)[0])
            elig += 1
            succ += int(hit)
            detail[u] = {"eligible": True, "hit": hit, "tau": float(tt[j]),
                         "pred": float(pp[j]), "true": float(tr[j]),
                         "band": float(tolerance_band(tr[j:j + 1], alpha,
                                                      absolute_floor)[0])}
        out["by_lambda"][f"{lam:g}"] = {
            "lambda": lam, "success_count": int(succ), "eligible_count": int(elig),
            "accuracy": (float(succ) / elig) if elig else NAN}
        out["per_trajectory"][f"{lam:g}"] = detail
    return out


# ============================== §9 Warning Lead Time ==============================

def warning_lead_time(pred, tau, trajectory_id, event_observed,
                      rul_threshold: float, persistence: int) -> dict:
    """RUL 阈值 + persistence 的预警提前量、漏报率、误报率（协议 §9）。

    规则: 预测 RUL 连续 `persistence` 个 evaluator endpoint 满足
    `pred_rul <= rul_threshold`，则**首个**满足的 endpoint 时刻定为 warning time。

    对 event-observed 轨迹（EOL 在 tau=1）:
        normalized_lead = 1 - tau_warning      （> 0 = EOL 之前报出）
    delta 为归一寿命比例，不是物理秒 —— 需要绝对时间时由调用方乘以该轨迹时长。

    **同时报两类汇总（协议 §9）**:

      A. whole-risk（分母 = 全部 eligible 轨迹）
         warning_coverage / miss_rate / false_alarm_rate
         —— 永不报警的方法在这里必然 coverage=0、miss_rate=1，逃不掉。

      B. conditional / common-detection（分母 = 确实报警的轨迹）
         conditional_lead = 在报了警的轨迹上的平均提前量
         `per_trajectory` 里逐轨迹给出 normalized_lead，供上层做 **paired
         lead difference**（只在两方法都报警的公共轨迹上配对比较）。

    false alarm 定义: 在**删失轨迹**（已知未失效）上报警 = 误报。observed 轨迹上
    warning time 晚于 EOL 无法出现（tau<=1），故 observed 侧只有命中/漏报两态。
    """
    keep, tids, taus, pred = dedup_endpoints(trajectory_id, tau, pred)
    pred = np.asarray(pred, dtype=float)
    ev = {int(k): bool(v) for k, v in dict(event_observed).items()}
    P = max(1, int(persistence))
    thr = float(rul_threshold)

    per = {}
    for u in np.unique(tids):
        u = int(u)
        s = tids == u
        order = np.argsort(taus[s], kind="stable")
        tt, pp = taus[s][order], pred[s][order]
        below = pp <= thr
        idx = -1
        run = 0
        for i, b in enumerate(below):
            run = run + 1 if b else 0
            if run >= P:
                idx = i - P + 1                  # 首个满足的 endpoint
                break
        observed = ev.get(u, False)
        rec = {"warning_issued": bool(idx >= 0), "event_observed": bool(observed),
               "n_points": int(len(pp)),
               "tau_warning": float(tt[idx]) if idx >= 0 else NAN}
        if idx >= 0 and observed:
            rec.update({"normalized_lead": float(1.0 - tt[idx]),
                        "detected_before_eol": bool(tt[idx] < 1.0),
                        "false_warning": False})
        elif idx >= 0 and not observed:
            rec.update({"normalized_lead": NAN, "detected_before_eol": False,
                        "false_warning": True})        # 已知未失效却报警
        else:
            rec.update({"normalized_lead": NAN, "detected_before_eol": False,
                        "false_warning": False})
        per[u] = rec

    obs = [v for v in per.values() if v["event_observed"]]
    cen = [v for v in per.values() if not v["event_observed"]]
    n_obs, n_cen = len(obs), len(cen)
    hits = [v for v in obs if v["warning_issued"]]
    # **EOL 之前**报出的才有预警价值。tau 可以 > 1 (EOL 之后仍有采样点), 若只看
    # warning_issued, 一条"失效后才报警"的轨迹会被算进 coverage —— 那是把漏报
    # 伪装成命中。故 whole-risk 同时给出 before_eol 版本, 且它才是可部署性口径。
    hits_pre = [v for v in hits if v["detected_before_eol"]]
    leads = [v["normalized_lead"] for v in hits if np.isfinite(v["normalized_lead"])]
    leads_pre = [v["normalized_lead"] for v in hits_pre
                 if np.isfinite(v["normalized_lead"])]
    return {
        "rul_threshold": thr, "persistence": P,
        # A. whole-risk —— 分母是全部 observed / censored 轨迹, 漏报无处躲
        "warning_coverage": (float(len(hits)) / n_obs) if n_obs else NAN,
        "miss_rate": (1.0 - float(len(hits)) / n_obs) if n_obs else NAN,
        # A'. 只计 EOL 之前报出的 (**可部署性口径**, 严于上面两项)
        "coverage_before_eol": (float(len(hits_pre)) / n_obs) if n_obs else NAN,
        "miss_rate_before_eol": (1.0 - float(len(hits_pre)) / n_obs) if n_obs else NAN,
        "false_alarm_rate": (float(sum(v["false_warning"] for v in cen)) / n_cen)
                            if n_cen else NAN,
        "n_observed": n_obs, "n_censored": n_cen, "n_warned_observed": len(hits),
        "n_warned_before_eol": len(hits_pre),
        # B. conditional —— 只在报了警的轨迹上, 必须与 A 一起读
        "conditional_lead": _macro(leads),
        "conditional_lead_n": len(leads),
        "conditional_lead_before_eol": _macro(leads_pre),
        "conditional_lead_before_eol_n": len(leads_pre),
        "per_trajectory": per,
    }


def paired_lead_difference(w_a: dict, w_b: dict) -> dict:
    """common-detection: 只在**两方法都报警**的公共 observed 轨迹上配对比较提前量。

    协议 §9 的要求: 不允许只统计"成功检测"就把 miss_rate 藏起来。所以这里额外报出
    `n_common` 与两侧各自的 warned 集合大小，让"配对样本很少"这件事直接可见。
    """
    def warned(w):
        return {int(k) for k, v in w["per_trajectory"].items()
                if v["warning_issued"] and v["event_observed"]
                and np.isfinite(v["normalized_lead"])}
    ca, cb = warned(w_a), warned(w_b)
    common = sorted(ca & cb)
    d = [float(w_a["per_trajectory"][k]["normalized_lead"]
               - w_b["per_trajectory"][k]["normalized_lead"]) for k in common]
    return {"n_common": len(common), "n_warned_a": len(ca), "n_warned_b": len(cb),
            "mean_paired_lead_diff": float(np.mean(d)) if d else NAN,
            "median_paired_lead_diff": float(np.median(d)) if d else NAN,
            "per_trajectory_diff": {int(k): v for k, v in zip(common, d)}}


# ============================== §10 Convergence ==============================

def convergence_metric(pred, true, tau, trajectory_id, late_from: float = 0.5) -> dict:
    """归一寿命轴上 |pred - true| 曲线的归一化面积 —— **越小越好**（协议 §10）。

    逐轨迹按 tau 升序做梯形积分，再除以该轨迹的 tau 跨度，得到"平均绝对误差密度"；
    macro = 轨迹等权平均。同时给 `late_convergence` = 只在 tau >= late_from 区段上的
    同一量（临近失效阶段的收敛质量）。

    这不是换个名字的 RMSE: 它对**误差随寿命推进如何演化**敏感 —— 同一 RMSE 下
    "早期差晚期好"与"早期好晚期崩"给出的 late_convergence 截然不同（协议 §10）。
    单点轨迹无法积分 → 该轨迹返回 NaN，不退化为绝对误差。
    """
    keep, tids, taus, pred, true = dedup_endpoints(trajectory_id, tau, pred, true)
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    def area(tt, err):
        if len(tt) < 2:
            return NAN
        span = float(tt[-1] - tt[0])
        if span <= 0:
            return NAN
        return float(_trapz(err, tt) / span)

    per = {}
    for u in np.unique(tids):
        u = int(u)
        s = tids == u
        order = np.argsort(taus[s], kind="stable")
        tt = taus[s][order]
        err = np.abs(pred[s][order] - true[s][order])
        lm = tt >= float(late_from)
        per[u] = {"convergence": area(tt, err),
                  "late_convergence": area(tt[lm], err[lm]),
                  "n_points": int(len(tt))}
    return {"per_trajectory": per,
            "macro_convergence": _macro([v["convergence"] for v in per.values()]),
            "macro_late_convergence": _macro([v["late_convergence"]
                                              for v in per.values()]),
            "late_from": float(late_from)}


# ============================== 配置读取 ==============================

def prognostics_cfg(cfg: dict) -> dict:
    """读取 `evaluation.prognostics`（协议 §5）。缺键即报错，绝不静默用魔法默认值。"""
    pc = (cfg or {}).get("evaluation", {}).get("prognostics")
    if not pc:
        raise KeyError("configs 缺 evaluation.prognostics 段 (S4 协议 §5)")
    need = ("hi_bins", "alpha", "alpha_lambda", "warning", "convergence")
    miss = [k for k in need if k not in pc]
    if miss:
        raise KeyError(f"evaluation.prognostics 缺键 {miss} (S4 协议 §5)")
    return {
        "hi_bins": [tuple(float(x) for x in b) for b in pc["hi_bins"]],
        "alpha": float(pc["alpha"]),
        "absolute_floor": float(pc["absolute_floor"]),
        "lambdas": [float(x) for x in pc["alpha_lambda"]["lambdas"]],
        "rul_threshold": float(pc["warning"]["rul_threshold"]),
        "persistence": int(pc["warning"]["persistence"]),
        "normalize_time": bool(pc["convergence"]["normalize_time"]),
        "late_from": float(pc["convergence"]["late_from"]),
    }


# ============================== 统一入口 ==============================

def prognostic_metrics(pred, true, hi, tau, trajectory_id, event_observed,
                       pcfg: dict) -> dict:
    """一次算全五组指标 —— `eval_test` / 基线 / 诊断脚本共用的唯一入口（协议 §14）。

    pcfg = `prognostics_cfg(cfg)`。所有方法调用同一函数、同一参数 ⇒ 口径必然一致。
    """
    return {
        "stage_metrics": staged_metrics(pred, true, hi, trajectory_id,
                                        pcfg["hi_bins"], tau=tau),
        "prognostic_horizon": prognostic_horizon(
            pred, true, tau, trajectory_id, event_observed,
            pcfg["alpha"], pcfg["absolute_floor"]),
        "alpha_lambda": alpha_lambda_accuracy(
            pred, true, tau, trajectory_id, event_observed,
            pcfg["alpha"], pcfg["absolute_floor"], pcfg["lambdas"]),
        "warning": warning_lead_time(
            pred, tau, trajectory_id, event_observed,
            pcfg["rul_threshold"], pcfg["persistence"]),
        "convergence": convergence_metric(pred, true, tau, trajectory_id,
                                          pcfg["late_from"]),
    }
