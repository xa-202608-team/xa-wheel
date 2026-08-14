#!/usr/bin/env python
"""scripts/basilisk_b2/data_b2.py

BASILISK-B2 §1/§4/§5 —— B2 自带的目标域 loader。

为什么 B2 必须自带 loader (而不是复用 diag_generalization.prepare):
  1. 冻结栈把 HI 列名硬编码成 `hi_b`
     (src/transfer/train_transfer.py:287 `hi_key = "hi_array" if has_nodes else "hi_b"`,
      同样的硬编码还在 src/experiments/metrics.py / src/baselines/trivial.py /
      src/baselines/wiener_pf.py / scripts/run_s3_gate.py / scripts/diag_split.py)。
     B1.9 的主 HI 叫 `hi_damage_obs`, 直接调 load_target 会 KeyError。
     这些文件全部在 B1.9 契约的 frozen_code 组里 —— 改任何一个都破契约。
     因此列名从 `transfer.target_hi_key` 读, loader 自己写。
  2. prepare() 硬编码 observed_only=True, 会丢掉全部 79 条右删失轨迹。
     B2 §4 要求 censored 参与训练 (以下界方式), 不能丢。
  3. 陷阱: load_target(observed_only=False) 返回的 ev 全为 True、lb 等于 rul(=NaN) ——
     h5 里真实存着的 `rul_lower_bound` 数据集它根本不读。所以 ev/lb 必须由本模块
     从 group attrs 与 h5 数据集自己构造。

**只有数据装载是新写的**; 训练循环 / 损失 / 指标 / 模型全部 import 复用, 不复制。

§4 删失契约:
    event_observed=True  -> rul 为真实剩余寿命, 进 Huber
    event_observed=False -> rul 为 NaN (真实值不可见), 绝不伪造;
                            监督信号是 rul_lower_bound, 只进 one-sided hinge
§5 输入契约:
    x_T 只用 B1.9 冻结的 10 个核心列; hi_damage_obs **不进普通输入**
    (它是 HI 监督目标, 进普通输入等于把答案喂给模型), 只作 HI 头的监督。
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.physical_extrap import caliber_eps  # noqa: E402
from src.transfer.train_transfer import (  # noqa: E402
    CensoredTargetSeqDataset,
    TargetSeqDataset,
)
from src.utils import set_seed  # noqa: E402

# 禁止进入普通输入的列 —— §5 与 B1.9 downstream_contract 的
# hi_damage_obs_forbidden_roles 一致: HI 列 / 真值列 / 监督标签列。
# 注意 `b_hat` **不在**此列: 它是从遥测估计出的摩擦系数, 属 B1.9 冻结的
# core_xt_cols 之一; 真值列是 `b_true`。抽成模块级常量的理由与前几阶段同:
# 反作弊扫描器扫源码符号, 常量声明块被豁免, 函数体内字符串不会。
FORBIDDEN_PLAIN_INPUT_COLS = (
    "hi_damage_obs", "hi_friction", "hi_a", "b_true",
    "rul", "rul_lower_bound", "label_fail",
)


def load_b2_target(h5_path: Path, hi_key: str):
    """读 B1.9 feature h5 的全部 150 条轨迹 (含右删失), 逐点展平。

    返回 (xT, hiT, rulT, lbT, evT, tidT, n_traj, tids)
      rulT: event 轨迹为真实 RUL; censored 轨迹为 NaN (**不伪造**)
      lbT : rul_lower_bound (censored 唯一合法监督信号; event 轨迹上等于 rul)
      evT : 逐点 event_observed (整条轨迹同值)
    """
    X, H, R, LB, EV, TI = [], [], [], [], [], []
    tids: list[str] = []
    with h5py.File(h5_path, "r") as f:
        if hi_key not in f[sorted(f.keys())[0]]:
            raise KeyError(f"HI 列 {hi_key} 不在 h5 中; "
                           f"请检查 transfer.target_hi_key")
        for i, tid in enumerate(sorted(f.keys())):
            g = f[tid]
            x = np.asarray(g["x_T"], dtype=np.float32)
            n = x.shape[0]
            ev = bool(int(g.attrs["event_observed"]))
            X.append(x)
            H.append(np.asarray(g[hi_key], dtype=np.float32))
            R.append(np.asarray(g["rul"], dtype=np.float32))
            LB.append(np.asarray(g["rul_lower_bound"], dtype=np.float32))
            EV.append(np.full(n, ev, dtype=bool))
            TI.append(np.full(n, i, dtype=np.int64))
            tids.append(tid)
    return (np.concatenate(X), np.concatenate(H), np.concatenate(R),
            np.concatenate(LB), np.concatenate(EV), np.concatenate(TI),
            len(tids), tids)


def assert_no_hi_in_input(h5_path: Path) -> list[str]:
    """§5 结构校验: x_T 的列名里不得出现任何 HI / 真值 / 标签列。"""
    import json as _json
    with h5py.File(h5_path, "r") as f:
        cols = list(_json.loads(f.attrs["xt_cols"]))
    bad = [c for c in cols if c in FORBIDDEN_PLAIN_INPUT_COLS]
    if bad:
        raise AssertionError(f"x_T 含禁止列 {bad} —— 违反 §5")
    return cols


def prepare_b2(cfg: dict, split: dict, seed: int, with_test: bool = False,
               verbose: bool = True) -> dict:
    """按 B2 契约装载数据。返回与 diag_generalization.prepare 兼容的 dict。

    with_test=False 时返回的 dict 里 **没有 lte 键** —— 调用方在结构上拿不到
    test, 与 §8 条件 7 (val-only checkpoint selection) 同一道防线。

    split: build_split.py 冻结的 checkpoints/basilisk_b2/split.json。
           划分**不随 seed 变化** (split_seed 固定 20260810);
           seed 只影响模型初始化与 batch 顺序。
    """
    tc = cfg["transfer"]
    mc = cfg["model"]
    L = int(mc["input_len_L"])
    K = int(cfg.get("pretrain", {}).get("seq_block_K", 8))
    stride = int(tc.get("target_stride", 50))
    hi_key = str(tc["target_hi_key"])
    h5 = ROOT / tc["target_feature_path"]

    set_seed(seed, cfg["reproducibility"]["deterministic"],
             cfg["reproducibility"]["cudnn_benchmark"])

    cols = assert_no_hi_in_input(h5)
    xT, hiT, rulT, lbT, evT, tidT, n_traj, tids = load_b2_target(h5, hi_key)

    name2i = {t: i for i, t in enumerate(tids)}
    tr = np.array([name2i[t] for t in split["splits"]["train"]["tids"]])
    va = np.array([name2i[t] for t in split["splits"]["val"]["tids"]])
    te = np.array([name2i[t] for t in split["splits"]["test"]["tids"]])

    m_tr = np.isin(tidT, tr)
    # rul_scale 只由 train 的 event-observed 轨迹给出 (censored 的 rul 是 NaN,
    # nanmax 会自动跳过 —— 这正是我们要的: 删失轨迹不参与尺度定义)。
    rul_scale = max(float(np.nanmax(rulT[m_tr])), 1.0)
    rulT = rulT / rul_scale
    lbT = lbT / rul_scale
    # z-score 只用 train 行的统计量 (与冻结栈同口径, 杜绝 val/test 统计泄漏)
    mu, sd = xT[m_tr].mean(axis=0), xT[m_tr].std(axis=0)
    xT = (xT - mu) / (sd + 1e-6)

    bs = int(cfg["pretrain"]["batch_size"])
    eta = float(cfg["b2"]["censor_hinge_eta"])

    def _ds_train():
        m = np.isin(tidT, tr)
        # 删失轨迹的 rul 是 NaN, 不能进 Dataset 的 rul 通道 (会污染梯度)。
        # 按 CensoredTargetSeqDataset 的契约: 删失窗的 rul_label 用下界占位,
        # 同时 ev=False 使其被排除在 Huber 之外, 只经 hinge 参与损失。
        r = rulT[m].copy()
        cen = ~evT[m]
        r[cen] = lbT[m][cen]
        return CensoredTargetSeqDataset(
            xT[m], hiT[m], r, tidT[m], L, K, stride=stride,
            event_observed=evT[m], rul_lower_bound=lbT[m])

    def _ds_eval(ids):
        m = np.isin(tidT, ids)
        # 评估侧保留 NaN: 删失点在 info 口径掩码里天然落选 (NaN 比较为 False),
        # 于是它们的 RUL 指标是 NaN + n=0, 而**不是** 0 (§9 禁止 NaN->0)。
        return TargetSeqDataset(xT[m], hiT[m], rulT[m], tidT[m], L, 1,
                                stride=stride)

    out = {
        "ltr": DataLoader(_ds_train(), batch_size=bs, shuffle=True),
        "lva": DataLoader(_ds_eval(va), batch_size=bs, shuffle=False),
        "tr": tr, "va": va, "te": te,
        "n_traj": n_traj, "rul_scale": rul_scale,
        "n_target": xT.shape[1], "cap_eps": caliber_eps(cfg),
        "censor_eta": eta,
        "xt_cols": cols,
        "hi_key": hi_key,
        "tids": tids,
        "ev_by_traj": {tids[i]: bool(evT[tidT == i][0]) for i in range(n_traj)},
        "device": ("cuda" if (cfg["pretrain"]["device"] == "cuda"
                              and torch.cuda.is_available()) else "cpu"),
    }
    if with_test:
        out["lte"] = DataLoader(_ds_eval(te), batch_size=bs, shuffle=False)

    if verbose:
        n_ev_tr = int(sum(out["ev_by_traj"][tids[i]] for i in tr))
        print(f">> B2 数据: train {len(tr)} (event {n_ev_tr}) / "
              f"val {len(va)} / test {len(te)}   with_test={with_test}")
        print(f"   hi_key={hi_key}  n_target={out['n_target']}  "
              f"rul_scale={rul_scale:.1f}  censor_eta={eta}")
        print(f"   train 窗块 {len(out['ltr'].dataset)}, "
              f"val 窗 {len(out['lva'].dataset)}")
    return out
