"""baselines/physical_extrap.py

物理基线 (plan §四 Phase 6): 滑窗辨识 b̂ → 线性外推到 b_fail → RUL。
合理物理经验方法, 作为非黑箱对照。

**同时是全项目共享的评估指标模块** (S2' 任务 4)。放在这里而不是 run_groups.py 的原因:
run_groups.py 单向 import 本模块 (evaluate_physical / phm_score / mae), 若指标留在
run_groups 则 physical_extrap / trivial 反向 import 会造成循环依赖。

逐样本指标:
  RMSE       均方根误差
  PHM Score  晚预测重罚 (PHM2008: d=pred-true, d>0 用 exp(d/late), d<0 用 exp(-d/early))
  MAE        平均绝对误差

三档评估口径 (对所有方法对称应用, 归一化 RUL 口径):
  full   全部样本                      —— 含 65%~95.6% 的失效后零难度样本
  valid  RUL > 0                       —— 去掉失效后样本
  info   0 < RUL < 1-eps  (**主口径**)  —— 再去掉 rul_cap_ratio 造成的截顶饱和段,
                                          唯一有预测价值的区间
每档同时给 pooled (逐样本池化) 与 macro (先算逐轨迹指标再对轨迹等权平均)。

用法:
  python -m src.baselines.physical_extrap --config configs/wheel.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import load_config, set_seed                              # noqa: E402
from src.transfer.train_transfer import (                                # noqa: E402
    split_trajectories_from_cfg, describe_eol_split)


# ---------------------------------------------------------------- 逐样本指标
def rmse(pred, true) -> float:
    pred, true = np.asarray(pred), np.asarray(true)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred, true) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


_EXP_CLIP = 700.0       # float64 exp 溢出边界 (exp(709) 已接近 inf); 仅防 inf, 不改语义


def phm_score(pred, true, early_scale=0.13, late_scale=0.10) -> float:
    """归一化 RUL 的逐样本平均非对称分数 nPHM。

    指数参数 clip 到 ±_EXP_CLIP 防 float64 溢出成 inf —— 触发 clip 需 |d| > 70 倍
    归一 RUL 量程 (物理外推未约束到标签域时会出现), 正常预测 (|d| <= 数个量程) 不受影响。
    """
    d = np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)
    a = np.where(d < 0, np.clip(-d / early_scale, None, _EXP_CLIP),
                 np.clip(d / late_scale, None, _EXP_CLIP))
    return float(np.mean(np.exp(a) - 1.0))


# ---------------------------------------------------------------- 三档评估口径
CALIBERS = ("full", "valid", "info")
PRIMARY_CALIBER = "info"
PRIMARY_STAT = "pooled"
CAP_EPS = 1.0e-6            # RUL >= 1 - CAP_EPS 视为 rul_cap 截顶饱和 (归一 RUL 口径)


def caliber_eps(cfg: dict | None) -> float:
    """截顶判定 eps (config evaluation.capped_eps, 缺省 CAP_EPS)。"""
    if not cfg:
        return CAP_EPS
    return float(cfg.get("evaluation", {}).get("capped_eps", CAP_EPS))


def caliber_mask(t, caliber: str, cap_eps: float = CAP_EPS):
    """口径掩码 (t = 归一化真实 RUL, 1.0 = 截顶饱和值)。"""
    t = np.asarray(t, dtype=float)
    if caliber == "full":
        return np.ones_like(t, dtype=bool)
    if caliber == "valid":
        return t > 0
    if caliber == "info":
        return (t > 0) & (t < 1.0 - float(cap_eps))
    raise ValueError(f"未知评估口径 {caliber!r}; 应属于 {CALIBERS}")


def pooled_metrics(p, t) -> dict:
    if len(t) == 0:
        return {"rmse": float("nan"), "phm": float("nan"), "mae": float("nan")}
    p, t = np.asarray(p, dtype=float), np.asarray(t, dtype=float)
    return {"rmse": float(np.sqrt(np.mean((p - t) ** 2))),
            "phm": phm_score(p, t),
            "mae": float(np.mean(np.abs(p - t)))}


def macro_metrics(p, t, tids) -> dict:
    """宏平均: 先算每条轨迹的指标, 再对轨迹取平均 (等权, 不受轨迹长度影响)。"""
    if len(t) == 0:
        return {"rmse": float("nan"), "phm": float("nan"), "mae": float("nan"),
                "n_traj": 0}
    tids = np.asarray(tids, dtype=np.int64)
    per = [pooled_metrics(np.asarray(p)[tids == u], np.asarray(t)[tids == u])
           for u in np.unique(tids)]
    return {k: float(np.mean([m[k] for m in per])) for k in ("rmse", "phm", "mae")} | \
           {"n_traj": len(per)}


def caliber_metrics(p, t, tids=None, cap_eps: float = CAP_EPS) -> dict:
    """三档口径 × (pooled / macro) 完整评估 —— run_groups.eval_test 与所有基线共用。

    返回 dict:
      顶层 rmse/phm/mae  = 主口径 (info) 的 PRIMARY_STAT(pooled) 指标, 保持与
                           aggregate / write_results / 历史结果的键名兼容;
      "calibers"[c]      = {"pooled": {...}, "macro": {...}, "n": 样本数, "share": 占比}
      "has_tids"         = 是否拿到样本→轨迹映射 (False 时 macro 退化为单组)。
    """
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=float)
    if tids is None:
        tids, has_tids = np.zeros(len(t), dtype=np.int64), False
    else:
        tids, has_tids = np.asarray(tids, dtype=np.int64), True
    out = {"calibers": {}, "n_total": int(len(t)), "has_tids": bool(has_tids)}
    for c in CALIBERS:
        m = caliber_mask(t, c, cap_eps)
        out["calibers"][c] = {
            "pooled": pooled_metrics(p[m], t[m]),
            "macro": macro_metrics(p[m], t[m], tids[m]),
            "n": int(m.sum()),
            "share": float(m.sum()) / max(len(t), 1),
        }
    prim = out["calibers"][PRIMARY_CALIBER][PRIMARY_STAT]
    out.update({k: prim[k] for k in ("rmse", "phm", "mae")})   # 向后兼容的顶层键
    out["n_traj"] = int(out["calibers"][PRIMARY_CALIBER]["macro"]["n_traj"])
    out["primary"] = f"{PRIMARY_CALIBER}/{PRIMARY_STAT}"
    return out


def format_calibers(m: dict, indent="    ") -> str:
    """把 caliber_metrics 的三档结果排成可读表 (供实跑输出核对)。"""
    lines = [f"{indent}{'口径':<6} {'样本数':>7} {'占比':>7} "
             f"{'pooled RMSE':>12} {'pooled MAE':>11} {'pooled PHM':>11} "
             f"{'macro RMSE':>11} {'macro MAE':>10} {'轨迹数':>7}"]
    for c in CALIBERS:
        d = m["calibers"][c]
        pl, mc = d["pooled"], d["macro"]
        star = " *" if c == PRIMARY_CALIBER else "  "
        lines.append(f"{indent}{c:<4}{star} {d['n']:>7} {d['share']:>7.3f} "
                     f"{pl['rmse']:>12.4f} {pl['mae']:>11.4f} {pl['phm']:>11.2f} "
                     f"{mc['rmse']:>11.4f} {mc['mae']:>10.4f} {mc['n_traj']:>7}")
    lines.append(f"{indent}* = 主结论口径 ({PRIMARY_CALIBER}/{PRIMARY_STAT}); "
                 f"顶层 rmse/phm/mae 取此格")
    return "\n".join(lines)


# ---------------------------------------------------------------- 与模型侧对齐的取点
def eval_point_indices(T: int, L: int, stride: int) -> np.ndarray:
    """与 TargetSeqDataset(K=1) 完全一致的预测点索引 (窗末 = start + L - 1)。

    模型侧只在滑窗末尾出预测 (starts = range(0, T-L+1, stride)), 基线若逐点评估,
    样本集与模型不同 → 指标不可比。本函数让所有基线取到同一批时间点。
    """
    if T < int(L):
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, T - int(L) + 1, max(1, int(stride)), dtype=np.int64) + (int(L) - 1)


def observed_traj_keys(f: h5py.File) -> list[str]:
    """load_target(observed_only=True) 实际保留的 key 序列 (sorted + 跳过未观测)。

    与 train_transfer.load_target / target_eol_by_tid 严格同序, 故列表下标 = tid。
    """
    out = []
    for k in sorted(f.keys()):
        g = f[k]
        if bool(g.attrs.get("event_observed",
                            np.asarray(g.get("label_fail", [])).any())):
            out.append(k)
    return out


def eol_index(g) -> int:
    """轨迹 EOL index (attrs eol_idx 优先, 回退 argmax(label_fail))。"""
    if "eol_idx" in g.attrs:
        return int(g.attrs["eol_idx"])
    return int(np.argmax(g["label_fail"][:]))


def baseline_hyper(cfg: dict) -> dict:
    """基线共用超参 (全部走 YAML, 与模型侧 L/stride 同源)。"""
    tc = cfg["transfer"]
    bc = cfg.get("baselines", {})
    return {
        "L": int(cfg["model"]["input_len_L"]),
        "stride": int(tc.get("target_stride", 50)),
        "extrap_window": int(bc.get("extrap_window", 50)),
        "smooth_window": int(bc.get("smooth_window", 30)),
        "hi_target": float(bc.get("hi_target", 1.0)),
        "cap_eps": caliber_eps(cfg),
        "target_h5": ROOT / tc.get("target_feature_path",
                                   "data/features/wheel/schema_v1/target_features.h5"),
    }


# ---------------------------------------------------------------- 物理外推
def physical_extrap_rul(b_hat, t_idx, b_fail, window: int = 50) -> np.ndarray:
    """滑窗线性拟合 b̂ 趋势, 外推到 b_fail 估计 RUL。

    对每个 t, 用 [t-window, t] 的 b̂ 线性拟合 (slope, intercept),
    外推失效时刻 t_fail=(b_fail-intercept)/slope, RUL=t_fail-t。
    """
    n = len(b_hat)
    rul = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window)
        ti = t_idx[lo:i + 1]
        bi = b_hat[lo:i + 1]
        if len(ti) < 5:
            continue
        slope, intercept = np.polyfit(ti, bi, 1)
        if slope <= 1e-12:
            continue
        t_fail = (b_fail - intercept) / slope
        rul[i] = max(0.0, t_fail - t_idx[i])
    return rul


def clip_to_label_domain(v, cap: float = 1.0):
    """基线预测 clip 到归一 RUL 标签域 [0, cap]。

    必要性 (S2' 任务 4 实测): 未 clip 时物理外推 info RMSE=6.63 —— 因为 b̂ 斜率极小时
    外推出的 t_fail 可达真实 EOL 的数十倍, 落在 [0,1] 标签域外。模型侧 RUL 头的输出
    被训练天然约束在标签域内, 基线若不 clip 则是在比"是否知道 RUL 上界"而非比预测能力,
    会让基线人为看差、把前置判据放水。cap 取归一截顶值 1.0 (= rul_cap)。
    """
    return np.clip(np.asarray(v, dtype=float), 0.0, float(cap))


def fill_nan_forward(v: np.ndarray, fallback: float) -> tuple[np.ndarray, int]:
    """外推失败点 (slope<=0 / 窗内点不足 → NaN) 前向填充, 首段无值填 fallback。

    为什么必须填而不是丢: 丢 NaN 会让基线的评估样本集小于模型侧, 两者指标不可比
    (基线相当于"只在自己算得准的时刻答题")。前向填充 = 沿用上一次可用估计,
    是物理外推方法在线部署时的自然行为。
    """
    v = np.asarray(v, dtype=float).copy()
    nan_mask = np.isnan(v)
    n_filled = int(nan_mask.sum())
    if n_filled:
        idx = np.where(~nan_mask, np.arange(len(v)), -1)
        idx = np.maximum.accumulate(idx)
        out = np.where(idx >= 0, v[np.clip(idx, 0, None)], float(fallback))
        v = out
    return v, n_filled


def _fit_b_fail_and_scale(f, keys, tr_ids, smooth_window) -> tuple[float, float]:
    """train 侧估 b_fail 阈值 (EOL 处平滑 b̂ 的中位数) 与 rul_scale。

    rul_scale 与 run_groups 完全同口径: max(nanmax(rul over train 全样本), 1.0)。
    """
    thresholds, rul_max = [], []
    for i in tr_ids:
        g = f[keys[int(i)]]
        e = eol_index(g)
        thresholds.append(float(uniform_filter1d(g["b_hat"][:], smooth_window)[e]))
        rul_max.append(float(np.nanmax(g["rul"][:])))
    return float(np.median(thresholds)), max(float(np.max(rul_max)), 1.0)


def evaluate_physical(cfg: dict, seed: int, legacy: bool = False,
                      return_raw: bool = False) -> dict | None:
    """物理外推基线 (S2' 任务 4 起与 eval_test 完全同口径)。

    对齐要点 (对齐前 → 对齐后):
      1. 划分   : 模块内独立纯随机 permutation → split_trajectories_from_cfg (EOL 分层),
                  与 run_groups / diag_* 同一函数同一 seed → 同一批 test 轨迹;
      2. 取点   : 逐时间点 t < EOL → eval_point_indices (窗末, stride), 与模型同批时刻;
      3. 口径   : 单一 "非 NaN 且 t<EOL" → full / valid / info 三档 (主口径 info);
      4. 统计   : 仅逐轨迹平均 (宏) → 每档同时给 pooled 与 macro;
      5. NaN    : 直接丢弃 → 前向填充 (样本集与模型一致, 并报告填充占比);
      6. 值域   : 无约束 → clip 到 [0,1] 标签域 (见 clip_to_label_domain 说明)。

    legacy=True 时走对齐前的旧实现 (仅供打印差异对照)。
    return_raw=True 时额外返回 `_raw = {"pred","true","tids"}` (S4 §15: 让基线与三个
    迁移组走**同一个** s4_metrics_block, 口径不可能分叉)。
    """
    set_seed(seed, cfg["reproducibility"]["deterministic"])
    h = baseline_hyper(cfg)
    target_h5 = h["target_h5"]
    if not target_h5.exists():
        print(f"!! 缺 {target_h5}; 先 python -m src.sim.build_hi --report")
        return None
    with h5py.File(target_h5, "r") as f:
        keys = observed_traj_keys(f)
        if not keys:
            return None
        # 字段保护: 相控阵 target (x_global/hi_array) 无 b_hat, Arrhenius 基线待 PA6 后续
        if "b_hat" not in f[keys[0]]:
            print(f">> 物理外推基线: {target_h5} 无 b_hat 字段 (相控阵 target 用 x_global/hi_array), "
                  f"跳过 (Arrhenius 物理基线待 PA6 后续实现)")
            return None
        if legacy:
            return _evaluate_physical_legacy(f, keys, cfg, seed, h)

        tr, va, te, eol = split_trajectories_from_cfg(
            target_h5, len(keys), cfg["transfer"], seed, observed_only=True)
        b_fail, rul_scale = _fit_b_fail_and_scale(f, keys, tr, h["smooth_window"])

        P, T, TID, n_nan, n_pts = [], [], [], 0, 0
        for tid in te:
            g = f[keys[int(tid)]]
            b_hat = uniform_filter1d(g["b_hat"][:], h["smooth_window"])
            rul_true = g["rul"][:].astype(float) / rul_scale
            t_idx = np.arange(len(b_hat))
            pts = eval_point_indices(len(b_hat), h["L"], h["stride"])
            if not len(pts):
                continue
            est = physical_extrap_rul(b_hat, t_idx, b_fail,
                                      h["extrap_window"]) / rul_scale
            est, nf = fill_nan_forward(est[pts], fallback=1.0)
            est = clip_to_label_domain(est)
            n_nan += nf
            n_pts += len(pts)
            P.append(est)
            T.append(rul_true[pts])
            TID.append(np.full(len(pts), int(tid), dtype=np.int64))
    if not P:
        return None
    m = caliber_metrics(np.concatenate(P), np.concatenate(T),
                        np.concatenate(TID), cap_eps=h["cap_eps"])
    m.update({"b_fail": b_fail, "rul_scale": rul_scale,
              "n_test_traj": int(len(te)), "nan_fill_share": n_nan / max(n_pts, 1),
              "aligned": True, "seed": int(seed)})
    if return_raw:
        m["_raw"] = {"pred": np.concatenate(P), "true": np.concatenate(T),
                     "tids": np.concatenate(TID)}
    return m


def _evaluate_physical_legacy(f, keys, cfg, seed, h) -> dict | None:
    """对齐前的实现 (独立纯随机划分 + 逐点 + 丢 NaN + 仅宏平均), 只用于差异对照。"""
    ratios = cfg["transfer"]["split"]
    perm = np.random.default_rng(seed).permutation(len(keys))
    n_tr = int(round(ratios["train"] * len(keys)))
    n_va = int(round(ratios["val"] * len(keys)))
    tr_ids, te_ids = perm[:n_tr], perm[n_tr + n_va:]
    if not len(tr_ids) or not len(te_ids):
        return None
    b_fail, rul_max = _fit_b_fail_and_scale(f, keys, tr_ids, h["smooth_window"])
    rmses, scores, maes = [], [], []
    for i in te_ids:
        g = f[keys[int(i)]]
        b_hat = uniform_filter1d(g["b_hat"][:], h["smooth_window"])
        e = eol_index(g)
        rul_true_n = g["rul"][:].astype(float) / rul_max
        t_idx = np.arange(len(b_hat))
        rul_est = physical_extrap_rul(b_hat, t_idx, b_fail, h["extrap_window"]) / rul_max
        valid = ~np.isnan(rul_est) & (t_idx < e)
        if int(valid.sum()) < 5:
            continue
        re, rt = rul_est[valid], rul_true_n[valid]
        rmses.append(rmse(re, rt))
        scores.append(phm_score(re, rt))
        maes.append(mae(re, rt))
    if not rmses:
        return None
    return {"rmse": float(np.mean(rmses)), "phm": float(np.mean(scores)),
            "mae": float(np.mean(maes)), "n_traj": len(rmses),
            "b_fail": b_fail, "rul_scale": rul_max, "aligned": False,
            "seed": int(seed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="多种子, 逗号分隔 (报 mean±std)")
    ap.add_argument("--out", default="checkpoints/physical_metrics.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [args.seed if args.seed is not None else cfg["seed"]])

    rows, last = [], None
    for s in seeds:
        print(f"\n===== physical_extrap seed {s} =====")
        m_new = evaluate_physical(cfg, s)
        m_old = evaluate_physical(cfg, s, legacy=True)
        if m_new is None:
            return
        last = m_new
        print(f">> b_fail={m_new['b_fail']:.6g} rul_scale={m_new['rul_scale']:.1f} "
              f"test 轨迹 {m_new['n_test_traj']} 条, NaN 前向填充占比 "
              f"{m_new['nan_fill_share']:.3f}")
        print("  -- 对齐后 (统一三档口径 + EOL 分层划分 + 窗末取点) --")
        print(format_calibers(m_new))
        if m_old:
            print(f"  -- 对齐前 (旧实现, 独立随机划分 + 逐点 + 丢 NaN + 仅宏平均) --\n"
                  f"     RMSE={m_old['rmse']:.4f} PHM={m_old['phm']:.2f} "
                  f"MAE={m_old['mae']:.4f} ({m_old['n_traj']} traj) "
                  f"b_fail={m_old['b_fail']:.6g}")
        rows.append((s, m_new, m_old))

    if len(rows) > 1:
        print("\n===== 多种子汇总 (info/pooled) =====")
        v = np.array([r[1]["rmse"] for r in rows])
        print(f"  对齐后 info RMSE = {v.mean():.4f} ± {v.std(ddof=1):.4f}  "
              f"(seeds {[r[0] for r in rows]})")
        vo = np.array([r[2]["rmse"] for r in rows if r[2]])
        if len(vo) > 1:
            print(f"  对齐前 旧口径 RMSE = {vo.mean():.4f} ± {vo.std(ddof=1):.4f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(last, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n>> physical baseline (info/pooled): RMSE={last['rmse']:.4f} "
          f"PHM={last['phm']:.2f} MAE={last['mae']:.4f} -> {out}")


if __name__ == "__main__":
    main()
