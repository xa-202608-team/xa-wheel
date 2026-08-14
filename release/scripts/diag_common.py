"""scripts/diag_common.py — S0 诊断共享管线 (只读, 不改 src/)

三个诊断脚本 (diag_leak / diag_split / diag_stage) 共用的数据准备与训练封装。

设计原则:
  1. **严格复用 src/experiments/run_groups.py 的函数** (_build_model / eval_test /
     _train_with_early_stop / _train_epoch), 保证诊断结论与正式实验路径一致,
     不重新实现训练逻辑, 避免"诊断跑的不是正式那套"。
  2. 所有超参从 configs/wheel.yaml 读, 与 run_groups.run_one_group 同一 .get 口径
     (含默认值兜底: 飞轮 config 未显式写 epochs_s2 / target_stride 时沿用代码默认)。
  3. 唯一新增自由度 = zero_cols (置零指定 x_T 列), 供诊断 1 做特征泄漏消融。

S2' 任务 5: 原先由 diag_leak._inject_s2_pipeline 做的模块级 monkey-patch (分层划分 +
post-EOL 权重) 已直接落到本模块 —— 注入式做法让"诊断跑的到底是哪套"取决于调用顺序,
且 diag_split / diag_stage 拿不到同样的修复。现在 prepare_target 走
split_trajectories_from_cfg (EOL 分层), train_* 从 config 读 post_eol_weight /
capped_weight 并转发给 _train_with_early_stop, 三个诊断脚本自动同口径。

用法: 由 scripts/diag_*.py import, 不单独执行。
"""
from __future__ import annotations

import sys
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, set_seed                              # noqa: E402
from src.sim.build_hi import XT_COLS                                     # noqa: E402
from src.transfer.mmd import reset_global_memory_bank                    # noqa: E402
from src.transfer.train_transfer import (                                # noqa: E402
    TargetSeqDataset, SourceWindowDataset, split_trajectories,
    split_trajectories_from_cfg, describe_eol_split, load_target, load_source)
from src.baselines.physical_extrap import (                              # noqa: E402
    caliber_metrics, caliber_mask, format_calibers, caliber_eps,
    CALIBERS, PRIMARY_CALIBER, PRIMARY_STAT)
from src.experiments.run_groups import (                                 # noqa: E402
    _build_model, eval_test, _train_with_early_stop, CKPT_DIR)

CONFIG_DEFAULT = "configs/wheel.yaml"


def xt_col_index(name: str) -> int:
    """x_T 列名 → 列索引 (从 build_hi.XT_COLS 取, 不硬编码数字)。"""
    if name not in XT_COLS:
        raise KeyError(f"x_T 无列 {name}; 实际列 = {XT_COLS}")
    return XT_COLS.index(name)


def load_cfg(config: str = CONFIG_DEFAULT) -> dict:
    return load_config(str(ROOT / config) if not Path(config).is_absolute() else config)


def hyper(cfg: dict) -> dict:
    """抽出与 run_groups.run_one_group 完全一致的超参视图 (同 .get 默认口径)。"""
    tc = cfg["transfer"]
    mc = cfg["model"]
    _lc = cfg["loss"]
    return {
        "L": int(mc["input_len_L"]),
        "K": int(cfg.get("pretrain", {}).get("seq_block_K", 8)),
        "bins": [tuple(b) for b in tc["hi_bins"]],
        "lam": (float(_lc.get("lambda_hi", _lc.get("beta_hi", 1.0))),
                float(_lc.get("lambda_mono", _lc.get("mu_mono", 0.1))),
                float(_lc.get("lambda_smooth", _lc.get("nu_smooth", 0.1)))),
        "mmd_lambda": float(tc["mmd_lambda"]),
        "epochs": int(tc.get("epochs_s2", 20)),          # 飞轮 config 未写 → 代码默认 20
        "target_stride": int(tc.get("target_stride", 50)),
        "batch_size": int(cfg["pretrain"]["batch_size"]),
        "finetune_lr": float(tc["finetune_lr"]),
        "huber_delta": float(cfg["loss"]["huber_delta"]),
        # S2' 任务 5: 权重从 config 读 (原先由 diag_leak 注入), 三诊断脚本自动同口径
        "post_eol_weight": float(cfg.get("training", {}).get("post_eol_weight", 1.0)),
        "capped_weight": float(cfg.get("training", {}).get("capped_weight", 1.0)),
        "cap_eps": caliber_eps(cfg),
        "split_ratios": [tc["split"]["train"], tc["split"]["val"], tc["split"]["test"]],
        "stratify_by_eol": bool(tc["split"].get("stratify_by_eol", False)),
        "target_h5": ROOT / tc.get("target_feature_path",
                                   "data/features/wheel/schema_v1/target_features.h5"),
        "source_h5": ROOT / cfg["pretrain"].get(
            "source_feature_path", "data/features/wheel/schema_v1/source_features.h5"),
        "source_id_field": cfg.get("pretrain", {}).get("source_id_field", "bearing_id"),
        "val_bearing_ids": (cfg.get("source", {}).get("split", {}).get("val_device_ids") or
                            cfg.get("source", {}).get("split", {}).get("val_bearing_ids") or []),
        "has_nodes": bool(tc.get("target_has_nodes", False)),
        "encoder": mc["encoder"],
        "device": ("cuda" if (cfg["pretrain"]["device"] == "cuda"
                              and torch.cuda.is_available()) else "cpu"),
    }


def prepare_target(cfg: dict, seed: int, zero_cols=None, epochs=None, verbose=True):
    """复现 run_groups.run_one_group 的目标域数据准备。

    zero_cols: 需在**加载后、归一化与训练前**置零的 x_T 列索引 list (诊断 1 特征泄漏)。
               置零发生在 z-score 之前, 该列 mean=0/std=1e-6 → 归一后仍恒为 0
               (下方 assert 实测校验), 等价于把该维信息彻底从模型输入中移除。

    S2' 任务 5: 划分改用 split_trajectories_from_cfg (与 run_groups 同一函数, EOL 分层)。

    返回 dict: loaders (ltr/lva/lte) + 划分 id + rul_scale + 供分箱用的 hi 数组。
    """
    h = hyper(cfg)
    if epochs is not None:
        h["epochs"] = int(epochs)
    set_seed(seed, cfg["reproducibility"]["deterministic"],
             cfg["reproducibility"]["cudnn_benchmark"])

    xT, hiT, rulT, tidT, n_traj = load_target(
        h["target_h5"], h["has_nodes"], observed_only=True)

    # ---- 诊断 1: 指定列置零 (加载后、训练前) ----
    zeroed = []
    if zero_cols:
        xT = xT.copy()
        for c in zero_cols:
            xT[:, c] = 0.0
            zeroed.append(XT_COLS[c])

    tr, va, te, eol = split_trajectories_from_cfg(
        h["target_h5"], n_traj, cfg["transfer"], seed, observed_only=True)
    if verbose:
        print(f">> 轨迹划分 train {len(tr)} / val {len(va)} / test {len(te)}"
              f"  (stratify_by_eol={h['stratify_by_eol']})")
        describe_eol_split(eol, tr, va, te)
    rul_scale = max(float(np.nanmax(rulT[np.isin(tidT, tr)])), 1.0)
    rulT = rulT / rul_scale

    # 目标域 xT z-score 归一 (train 集) — 与 run_groups 完全一致
    _tr_mask = np.isin(tidT, tr)
    _fm = xT[_tr_mask].mean(axis=0)
    _fs = xT[_tr_mask].std(axis=0) + 1e-6
    xT = (xT - _fm) / _fs

    if zero_cols:                       # 实测校验置零列归一后确实恒为 0
        for c in zero_cols:
            mx = float(np.max(np.abs(xT[:, c])))
            assert mx == 0.0, f"置零列 {XT_COLS[c]} 归一后非零 (max|.|={mx:.3e})"

    def mask(ids):
        return np.isin(tidT, ids)

    def mkDS(ids, block_k):
        m = mask(ids)
        return TargetSeqDataset(xT[m], hiT[m], rulT[m], tidT[m],
                                h["L"], block_k, stride=h["target_stride"])

    bs = h["batch_size"]
    ds_tr, ds_va, ds_te = mkDS(tr, h["K"]), mkDS(va, 1), mkDS(te, 1)
    return {
        "h": h,
        "ltr": DataLoader(ds_tr, batch_size=bs, shuffle=True),
        "lva": DataLoader(ds_va, batch_size=bs, shuffle=False),
        "lte": DataLoader(ds_te, batch_size=bs, shuffle=False),
        "n_traj": n_traj, "tr": tr, "va": va, "te": te, "eol": eol,
        "rul_scale": rul_scale, "n_target": xT.shape[1],
        "sizes": (len(ds_tr), len(ds_va), len(ds_te)),
        "zeroed": zeroed,
    }


def train_kwargs(data: dict) -> dict:
    """转发给 _train_with_early_stop 的损失权重 kwargs (与 run_groups 同口径)。"""
    h = data["h"]
    return {"post_eol_weight": h["post_eol_weight"],
            "capped_weight": h["capped_weight"],
            "huber_delta": h["huber_delta"],
            "cap_eps": h["cap_eps"]}


def prepare_source(cfg: dict):
    """源域 loader (S3 MMD 的 z_S 用), 与 run_groups 同口径。"""
    h = hyper(cfg)
    featsS, hiS, bidS, tidxS = load_source(
        h["source_h5"], h["source_id_field"], h["val_bearing_ids"], train_only=True)
    dsS = SourceWindowDataset(featsS, hiS, bidS, tidxS, h["L"],
                              stride=max(1, len(featsS) // 2000))
    return featsS, DataLoader(dsS, batch_size=h["batch_size"], shuffle=True)


def train_target_only(cfg: dict, data: dict, n_features: int, seed: int, tag: str):
    """target_only 组: 随机初始化 encoder + 全参训练 + val early-stop。

    与 run_groups.run_one_group(mode='target_only') 逐行等价 (含 S2' 损失权重)。
    """
    h = data["h"]
    model = _build_model(cfg, n_features, data["n_target"], h["device"])
    model.freeze_encoder(False)
    opt = torch.optim.Adam(model.parameters(), lr=h["finetune_lr"])
    huber = nn.HuberLoss(delta=h["huber_delta"])
    mse = nn.MSELoss()
    _train_with_early_stop(model, data["ltr"], data["lva"], opt, h["device"],
                           huber, mse, h["lam"], h["epochs"], tag,
                           **train_kwargs(data))
    return model


def train_source_mmd(cfg: dict, data: dict, n_features: int, loader_S,
                     seed: int, tag: str, ckpt: Path):
    """source_mmd_finetune 组: S1 加载源 ckpt → S2 冻结 encoder → S3 解冻+MMD。

    与 run_groups.run_one_group(mode='source_mmd_finetune') 逐行等价。
    """
    h = data["h"]
    kw = train_kwargs(data)
    model = _build_model(cfg, n_features, data["n_target"], h["device"])
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"缺源域 checkpoint: {ckpt} (先跑 python -m src.train.pretrain)")
    model.load_pretrained(str(ckpt), h["device"])
    huber = nn.HuberLoss(delta=h["huber_delta"])
    mse = nn.MSELoss()

    model.freeze_encoder(True)                      # S2
    opt2 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=h["finetune_lr"])
    _train_with_early_stop(model, data["ltr"], data["lva"], opt2, h["device"],
                           huber, mse, h["lam"], h["epochs"], f"{tag} S2", **kw)

    reset_global_memory_bank()                      # S3
    model.freeze_encoder(False)
    opt3 = torch.optim.Adam(model.parameters(), lr=h["finetune_lr"])
    _train_with_early_stop(model, data["ltr"], data["lva"], opt3, h["device"],
                           huber, mse, h["lam"], h["epochs"], f"{tag} S3",
                           use_mmd=True, src_iter=cycle(loader_S),
                           bins=h["bins"], mmd_lambda=h["mmd_lambda"], **kw)
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    """在 loader 上收集 (pred_rul, true_rul, hi_end) — 供 HI 分箱误差分析。

    loader 的 TargetSeqDataset 以 K=1 构造, 故每样本 h/r 长度为 1 (窗末值)。
    """
    model.eval()
    P, T, H = [], [], []
    for x, hh, r in loader:
        x = x.to(device)
        B, Kk = x.size(0), x.size(1)
        _, rul_p, _ = model(x.reshape(B * Kk, x.size(2), x.size(3)))
        P.append(rul_p.cpu().numpy().reshape(-1))
        T.append(r.numpy().reshape(-1))
        H.append(hh.numpy().reshape(-1))
    return (np.concatenate(P) if P else np.zeros(0),
            np.concatenate(T) if T else np.zeros(0),
            np.concatenate(H) if H else np.zeros(0))


def observed_traj_keys(target_h5: Path) -> list[str]:
    """load_target(observed_only=True) 实际保留的 h5 key 序列。

    load_target 按 sorted(keys) 遍历、跳过未观测轨迹, 再把 tid remap 到 0..n-1,
    故新 tid j 恰对应"排序后第 j 个已观测轨迹"。诊断 2 靠此映射回溯原始 key。
    """
    import h5py
    out = []
    with h5py.File(target_h5, "r") as f:
        for k in sorted(f.keys()):
            g = f[k]
            observed = bool(g.attrs.get("event_observed",
                                        np.asarray(g.get("label_fail", [])).any()))
            if observed:
                out.append(k)
    return out
