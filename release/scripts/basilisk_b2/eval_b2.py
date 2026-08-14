#!/usr/bin/env python
"""scripts/basilisk_b2/eval_b2.py

BASILISK-B2 §8/§9 —— 评估与删失分层报告。

为什么 B2 需要自己的时间轴构造器 (而不是直接用 metrics.target_endpoint_axis):
  1. 那个函数把 HI 列名硬编码成 `hi_b` (metrics.py:61), B1.9 的 HI 叫 hi_damage_obs;
  2. 它 `if not observed: continue` —— 直接跳掉全部 79 条删失轨迹。
     §9 要求报告 censored test 分层, §6 的 false_alarm_rate 定义在删失轨迹上,
     跳掉它们就没法算。
  两个文件都在 B1.9 契约的 frozen_code 组里, 不可修改。

**指标计算本身全部 import 复用** (caliber_metrics / shape_stats / warning_lead_time /
prognostic_metrics), 本模块只负责: 换列名、把删失轨迹带进来、按删失分层汇总。

删失轨迹的时间轴 —— 这是本模块唯一需要小心的地方:
    event-observed:  tau = idx / eol_idx      (归一寿命位置, 1.0 = EOL)
    right-censored:  tau = idx / n_full       (**观测进度**, 不是寿命比例!)
                     命名为 observation_progress; event_observed=False 使
                     PH / alpha_lambda 结构上跳过它们, warning 只用它排序 endpoint
                     并判定"已知未失效却报警 = 误报"。
    绝不把 n_full 当成 EOL 使用 —— 那是伪造结局 (§4/§9 禁止)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diag_generalization import collect_pred, shape_stats  # noqa: E402
from src.baselines.physical_extrap import (  # noqa: E402
    caliber_mask,
    caliber_metrics,
    eval_point_indices,
)
from src.experiments.metrics import (  # noqa: E402
    paired_lead_difference,
    prognostics_cfg,
    warning_lead_time,
)

NAN = float("nan")

# 删失轨迹的 tau 语义标签。抽成模块级常量: 它必须在产出的 JSON 里显式出现,
# 让任何读数字的人立刻看到"这一列不是寿命比例"。
CENSORED_TAU_SEMANTICS = "observation_progress_not_life_fraction"


def b2_endpoint_axis(cfg: dict) -> dict:
    """全部 150 条轨迹的 evaluator endpoint 轴 (含删失)。

    tid 空间 = sorted(f.keys()) 的下标, 与 data_b2.load_b2_target 严格同序。
    """
    tc = cfg["transfer"]
    L = int(cfg["model"]["input_len_L"])
    stride = int(tc.get("target_stride", 50) or 50)
    hi_key = str(tc["target_hi_key"])
    h5p = ROOT / tc["target_feature_path"]

    axis: dict[int, dict] = {}
    with h5py.File(h5p, "r") as f:
        for tid, k in enumerate(sorted(f.keys())):
            g = f[k]
            ev = bool(int(g.attrs["event_observed"]))
            n_full = int(len(g[hi_key]))
            idx = eval_point_indices(n_full, L, stride)
            e = int(g.attrs["eol_idx"])
            if ev and e > 0:
                tau = idx.astype(float) / float(e)
                sem = "life_fraction_eol_at_1"
            else:
                # 观测进度轴 —— 只用于 endpoint 排序与误报判定, 不是寿命比例
                tau = idx.astype(float) / float(max(n_full - 1, 1))
                sem = CENSORED_TAU_SEMANTICS
            axis[tid] = {
                "key": str(k), "endpoint_idx": idx, "n_full": n_full,
                "eol_idx": (e if ev else None), "event_observed": ev,
                "tau": tau, "tau_semantics": sem,
                "hi": np.asarray(g[hi_key][:], dtype=float)[idx],
                "rul_lower_bound": np.asarray(
                    g["rul_lower_bound"][:], dtype=float)[idx],
            }
    return axis


def attach_axis_b2(tids, axis: dict):
    """按 loader 顺序把 (tau, hi, lb) 贴到逐预测点上; 长度不符直接报错。

    与 metrics.attach_axis 同一对齐逻辑 (groupby(tid) + 窗按 start 升序),
    只是多带一路 rul_lower_bound 并且不跳删失。绝不截断凑数。
    """
    tids = np.asarray(tids, dtype=np.int64)
    n = len(tids)
    tau = np.empty(n, float)
    hi = np.empty(n, float)
    lb = np.empty(n, float)
    ptr: dict[int, int] = {}
    for i, u in enumerate(tids):
        u = int(u)
        a = axis.get(u)
        if a is None:
            raise KeyError(f"tid={u} 不在 b2_endpoint_axis 中 (划分口径不一致)")
        j = ptr.get(u, 0)
        if j >= len(a["endpoint_idx"]):
            raise ValueError(
                f"tid={u} 预测点数 {j + 1} 超过 endpoint 数 "
                f"{len(a['endpoint_idx'])} —— L/stride 口径不一致, 拒绝对齐")
        tau[i], hi[i], lb[i] = a["tau"][j], a["hi"][j], a["rul_lower_bound"][j]
        ptr[u] = j + 1
    ev = {int(u): bool(axis[int(u)]["event_observed"]) for u in np.unique(tids)}
    return tau, hi, lb, ev


def _macro_rmse_over(pred, true, tids) -> tuple[float, int]:
    """逐轨迹 RMSE 再等权平均。无可评估轨迹 -> (NaN, 0), **绝不返回 0**。"""
    vals = []
    for u in np.unique(tids):
        m = tids == u
        p, t = pred[m], true[m]
        ok = np.isfinite(p) & np.isfinite(t)
        if not ok.any():
            continue
        vals.append(float(np.sqrt(np.mean((p[ok] - t[ok]) ** 2))))
    if not vals:
        return NAN, 0
    return float(np.mean(vals)), len(vals)


def censor_strata_report(pred, true, tids, tau, lb, ev, cap_eps,
                         rul_scale: float) -> dict:
    """§9 三个删失分层。空分层一律 NaN + n=0。

    event_observed_test : info 口径 RUL 指标 (唯一有真实 RUL 的分层)
    censored_test       : **没有真实 RUL, 因此 RUL 指标恒为 NaN + n=0**。
                          能算且合法的只有下界违反率:
                          frac(pred < rul_lower_bound) —— 用的是已知下界,
                          不是伪造的 EOL。
    common_evaluable_endpoints : 两类分层里 true 与 pred 同时有限的 endpoint 数,
                          即"配对比较真正站得住的样本量"。
    """
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    tids = np.asarray(tids, np.int64)
    is_ev = np.array([ev.get(int(t), False) for t in tids], bool)

    # --- event-observed test ---
    m_ev = is_ev & caliber_mask(true, "info", cap_eps)
    ev_rmse, ev_ntraj = _macro_rmse_over(pred[m_ev], true[m_ev], tids[m_ev])
    ev_block = {
        "n_endpoints": int(m_ev.sum()),
        "n_traj": ev_ntraj,
        "info_macro_rmse": ev_rmse,
        "note": "唯一有真实 RUL 的分层; 主指标来自这里",
    }

    # --- censored test ---
    m_cen = ~is_ev
    n_cen_pts = int(m_cen.sum())
    viol = NAN
    n_viol_ok = 0
    if n_cen_pts:
        # 单位必须对齐: pred 是被 rul_scale 归一化过的, 而 axis 里的
        # rul_lower_bound 是原始步数 (~1e4)。不除 rul_scale 的话
        # pred < lb 恒成立, 违反率会假性地锁死在 1.0。
        pc = pred[m_cen]
        lc = np.asarray(lb, float)[m_cen] / float(rul_scale)
        ok = np.isfinite(pc) & np.isfinite(lc)
        n_viol_ok = int(ok.sum())
        viol = float(np.mean(pc[ok] < lc[ok])) if ok.any() else NAN
    cen_block = {
        "n_endpoints": n_cen_pts,
        "n_traj": int(len(np.unique(tids[m_cen]))) if n_cen_pts else 0,
        # 明确写成 NaN, 不是 0 —— 没有真值就是不可评估
        "info_macro_rmse": NAN,
        "n_evaluable_rul_points": 0,
        "rul_metrics_reason": "right_censored_no_true_rul_never_fabricated",
        "lower_bound_violation_rate": viol,
        "n_lower_bound_evaluable": n_viol_ok,
        "lower_bound_violation_note": (
            "pred < rul_lower_bound 的 endpoint 占比; 两侧都在归一化 RUL "
            "单位下比较 (lb / rul_scale); 用已知下界判定, 不涉及任何伪造 EOL"),
        "tau_semantics": CENSORED_TAU_SEMANTICS,
    }

    # --- common evaluable ---
    fin = np.isfinite(pred) & np.isfinite(true)
    com_block = {
        "n_endpoints_pred_and_true_finite": int(fin.sum()),
        "n_endpoints_total": int(len(pred)),
        "n_endpoints_censored_excluded": n_cen_pts,
        "share_evaluable": (float(fin.mean()) if len(pred) else NAN),
        "rul_scale_steps": float(rul_scale),
    }
    return {"event_observed_test": ev_block,
            "censored_test": cen_block,
            "common_evaluable_endpoints": com_block}


def evaluate_b2(model, loader, cfg, axis, data, tag: str = "") -> dict:
    """一次评估: caliber 指标 + shape_stats + warning + §9 删失分层。"""
    pred, true, tids = collect_pred(model, loader, data["device"])
    return evaluate_arrays_b2(pred, true, tids, cfg, axis, data, tag=tag)


def evaluate_arrays_b2(pred, true, tids, cfg, axis, data, tag: str = "") -> dict:
    """与 evaluate_b2 同口径, 但接受现成的 (pred, true, tids) —— 供基线复用,
    保证模型与基线走**同一个 evaluator** (口径不可能分叉)。"""
    cap_eps = float(data["cap_eps"])
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    tids = np.asarray(tids, np.int64)
    tau, hi, lb, ev = attach_axis_b2(tids, axis)
    pc = prognostics_cfg(cfg)

    cal = caliber_metrics(pred, true, tids, cap_eps=cap_eps)
    ss = shape_stats(pred, true, tids, cap_eps)
    warn = warning_lead_time(pred, tau, tids, ev,
                             pc["rul_threshold"], pc["persistence"])
    strata = censor_strata_report(pred, true, tids, tau, lb, ev, cap_eps,
                                  data["rul_scale"])
    return {
        "tag": tag,
        "n_points": int(len(pred)),
        "info_macro_rmse": cal["calibers"]["info"]["macro"]["rmse"],
        "info_pooled_rmse": cal["calibers"]["info"]["pooled"]["rmse"],
        "info_n": cal["calibers"]["info"]["n"],
        "calibers": cal["calibers"],
        "shape": ss,
        # warning 的 per_trajectory 很大, 只在需要 paired lead 时单独留
        "warning": {k: v for k, v in warn.items() if k != "per_trajectory"},
        "_warning_full": warn,
        "censor_report": strata,
        "_raw": {"pred": pred, "true": true, "tids": tids, "tau": tau, "lb": lb},
    }


def paired_warning(a: dict, b: dict) -> dict:
    """两方法在**都报警**的公共 observed 轨迹上的配对提前量 (import 复用)。"""
    return paired_lead_difference(a["_warning_full"], b["_warning_full"])


def strip_private(d: dict) -> dict:
    """落 JSON 前去掉下划线开头的大对象 (数组 / per_trajectory)。"""
    return {k: (strip_private(v) if isinstance(v, dict) else v)
            for k, v in d.items() if not k.startswith("_")}
