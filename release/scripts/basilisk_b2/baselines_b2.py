#!/usr/bin/env python
"""scripts/basilisk_b2/baselines_b2.py

BASILISK-B2 §6 —— 非学习基线 A / B。

A. const_mean_info  常数预测器: 输出 **train 划分 info 区**真实 RUL 的均值。
   这是最强的**可部署**平凡基线, 也是 §8 闸门的比较对象。
   常数值只由 train 的 event-observed 轨迹算出 —— 绝不看 val/test,
   也绝不用删失轨迹的下界充当"真实 RUL 均值"。
   info 区 RUL 近似均匀时理论 RMSE = 1/sqrt(12) ≈ 0.2887 (锚点)。

B. damage_extrapolation  物理/损伤外推: 由当前 hi_damage_obs 与因果滑窗斜率
   线性外推到 hi_fail=1.0, 得剩余步数 → 归一 RUL。
   因果 = 斜率只用 [t-W, t] 的历史, 绝不含未来点。

§6 明确只比 A / B / C 三者, **不跑 source transfer** (那是 B4 的事)。
persistence 类读真值的方法一律不参与 —— 它们不可部署 (ORACLE_LIKE)。

斜率外推的实现直接 import src.baselines.trivial.hi_extrap_rul, 不重写。
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.physical_extrap import caliber_mask  # noqa: E402
from src.baselines.trivial import hi_extrap_rul  # noqa: E402

# 只读真值的乐观上界方法, 严禁进入本阶段任何排名 (与 trivial.ORACLE_LIKE 同精神)
EXCLUDED_ORACLE_METHODS = ("persistence", "true_rul_lookup")


def _endpoints_by_tid(axis: dict, ids) -> dict[int, np.ndarray]:
    return {int(t): axis[int(t)]["endpoint_idx"] for t in ids}


def const_mean_info(cfg: dict, axis: dict, data: dict) -> float:
    """A: train 划分 info 区真实 RUL 均值 (归一尺度)。只用 train + event-observed。"""
    tc = cfg["transfer"]
    h5p = ROOT / tc["target_feature_path"]
    scale = float(data["rul_scale"])
    cap_eps = float(data["cap_eps"])
    vals = []
    with h5py.File(h5p, "r") as f:
        keys = sorted(f.keys())
        for t in data["tr"]:
            a = axis[int(t)]
            if not a["event_observed"]:
                continue                      # 删失轨迹没有真实 RUL, 不进均值
            r = np.asarray(f[keys[int(t)]]["rul"][:], float)[a["endpoint_idx"]]
            r = r / scale
            vals.append(r[caliber_mask(r, "info", cap_eps)])
    if not vals or not len(np.concatenate(vals)):
        raise SystemExit("!! const_mean_info: train info 区为空, 无法定义常数")
    return float(np.mean(np.concatenate(vals)))


def predict_const(value: float, true: np.ndarray) -> np.ndarray:
    return np.full(len(true), float(value), dtype=float)


def damage_extrapolation(cfg: dict, axis: dict, data: dict, ids) -> tuple:
    """B: hi_damage_obs 因果滑窗斜率外推到 hi_fail。

    返回 (pred, true, tids) —— 与模型 collect_pred 完全相同的取点顺序
    (按 tid 升序, 每 tid 内 endpoint 升序), 因此可以直接进同一个 evaluator。
    """
    tc = cfg["transfer"]
    bc = cfg["b2"]["damage_extrapolation"]
    hi_key = str(bc.get("hi_key", tc["target_hi_key"]))
    hi_fail = float(bc["hi_fail"])
    W = int(bc["slope_window"])
    h5p = ROOT / tc["target_feature_path"]
    scale = float(data["rul_scale"])

    P, T, TI = [], [], []
    with h5py.File(h5p, "r") as f:
        keys = sorted(f.keys())
        for t in sorted(int(x) for x in ids):
            a = axis[t]
            g = f[keys[t]]
            hi = np.asarray(g[hi_key][:], float)
            # 全序列上算因果斜率外推, 再取 endpoint —— 保证每个 endpoint 的
            # 斜率窗口都是它自己的历史 [t-W, t], 不含未来
            rul_steps = hi_extrap_rul(hi, W, hi_target=hi_fail)
            idx = a["endpoint_idx"]
            p = rul_steps[idx] / scale
            # 斜率 <= 0 (HI 尚未上升) -> NaN。不填 0: 填 0 等于宣称"马上失效",
            # 是伪造的悲观预测。NaN 在 info 掩码里天然落选, n 会如实变小。
            r = np.asarray(g["rul"][:], float)[idx] / scale
            P.append(p)
            T.append(r)
            TI.append(np.full(len(idx), t, np.int64))
    return (np.concatenate(P), np.concatenate(T), np.concatenate(TI))


def const_pred_on(axis: dict, ids, value: float, cfg: dict, data: dict) -> tuple:
    """A 的预测在与模型相同取点上的展开 (pred, true, tids)。"""
    tc = cfg["transfer"]
    h5p = ROOT / tc["target_feature_path"]
    scale = float(data["rul_scale"])
    P, T, TI = [], [], []
    with h5py.File(h5p, "r") as f:
        keys = sorted(f.keys())
        for t in sorted(int(x) for x in ids):
            idx = axis[t]["endpoint_idx"]
            r = np.asarray(f[keys[t]]["rul"][:], float)[idx] / scale
            P.append(np.full(len(idx), float(value)))
            T.append(r)
            TI.append(np.full(len(idx), t, np.int64))
    return (np.concatenate(P), np.concatenate(T), np.concatenate(TI))
