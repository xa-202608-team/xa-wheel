#!/usr/bin/env python
"""scripts/run_feature_transfer.py

P1-3 特征空间迁移对照实验 (5 seeds × 2 conditions)。

**实验目的:**
    诚实回答一个对照问题: 如果把源域 (XJTU-SY 振动特征, 12 维) 上拟合的
    clip + z-score 归一化参数**冻结**后, 直接套到目标域 (Basilisk 遥测, 10 维)
    上, target_only 模型的 RMSE 会变好还是变差?

**设计动机:**
    项目正式迁移路线 (B5) 发生在 HI 动力学层 / 观测层 adapter, 不在原始特征层。
    本实验是 P1-3 spec 要求的特征层"强对照": 如果连最基础的归一化参数都不能跨域
    共享 (因为源=振动、目标=遥测, 物理量纲完全不同), 那就是HI层迁移必要性的
    直接证据。正/负结果都接受 —— 这是诚实的对照。

**严格配对 (spec P1-3):**
    * 固定 5 seeds (112-116, 与 B5 正式 seed 一致)
    * 同一 B2.1 冻结划分 (split_sha256 = 23e2b944...)
    * 同一模型配置 (n_features=12 / n_target=10, S2.5 冻结训练超参)
    * 同一 batch 顺序, 同一 optimizer, 同一 epoch 预算, 同一早停指标
    * 唯一差异: z-score 的 mean/std 来自源域 train 集还是目标域 train 集

**条件 A (native_scaler):** 目标域 train 行的 mean/std, B2/B5 的标准口径。
**条件 B (source_scaler):** 源域 (XJTU-SY) train 行的 mean/std, 冻结后套到目标域。

    源域 12 维 vs 目标域 10 维: 按 min(F_src, F_tgt)=10 逐列对齐 (源前 10 列
    的统计量套到目标 10 列上)。这是一种朴素的位置对齐, 不含任何物理对应 ——
    恰好刻画"把振动域统计量盲目套到遥测域"的特征层迁移, 对应论文里要否证的对象。

**输出:**
    JSON: reference/wheel/feature_transfer_paired.json
    {
      "experiment": "feature_space_transfer_5seeds",
      "seeds": [112, 113, 114, 115, 116],
      "conditions": {
        "native_scaler": {"per_seed_rmse": [...], "mean": ..., "std": ...},
        "source_scaler": {"per_seed_rmse": [...], "mean": ..., "std": ...}
      },
      "paired_delta": {"per_seed": [...], "mean": ..., "ci95": [...], "crosses_zero": ...}
    }

用法:
    python scripts/run_feature_transfer.py [--fast]

    --fast: 1 epoch, 用于 smoke; 正式报告必须用默认 (8 epochs, B5 冻结值)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.basilisk_b11.calibrate_degradation import (  # noqa: E402
    load_b11_config as load_cfg,
)
from scripts.basilisk_b2.data_b2 import (  # noqa: E402
    FORBIDDEN_PLAIN_INPUT_COLS,
    assert_no_hi_in_input,
    load_b2_target,
)
from scripts.basilisk_b2.eval_b2 import (  # noqa: E402
    b2_endpoint_axis,
    evaluate_b2,
    strip_private,
)
from scripts.basilisk_b2.run_gate import paired_bootstrap  # noqa: E402
from scripts.basilisk_b5.data_b5 import (  # noqa: E402
    TARGET_GEN_OFFSET,
    load_b21_split,
    reset_target_batch_order,
    resolve_input_schema_b5,
)
from scripts.diag_generalization import summarize_history  # noqa: E402
from src.baselines.physical_extrap import caliber_eps  # noqa: E402
from src.experiments.run_groups import (  # noqa: E402
    _build_model,
    _train_with_early_stop,
    training_hyper,
)
from src.transfer.train_transfer import (  # noqa: E402
    CensoredTargetSeqDataset,
    TargetSeqDataset,
)
from src.utils import set_seed  # noqa: E402

# ---- 常量 (spec P1-3 硬约束) ----
FORMAL_SEEDS = [112, 113, 114, 115, 116]   # 与 B5 正式 seed 一致
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 20260814                  # 与 B5 spec §10 bootstrap seed 一致

# 配置路径 (与 run_formal_transfer.py 完全相同)
CFG_PATH = "configs/wheel_basilisk_b5.yaml"

# 数据路径覆盖: 本地开发时 release/data/ 不存在, 数据在 04_数据/wheel/。
# Docker 内 release/data/ 是只读 volume mount (compose.yml: ../04_数据/wheel:/app/release/data:ro)
DATA_ROOT_LOCAL = ROOT.parent.parent.parent.parent / "04_数据" / "wheel"   # XA-202608_最终交付/04_数据/wheel
DATA_ROOT_DOCKER = ROOT / "data"

PURPOSE = (
    "P1-3 特征空间迁移对照实验: 固定 5 seeds × 同一 B2.1 冻结划分 × 同一模型配置, "
    "对比 target_only 在两种 z-score 归一化下的 RMSE。"
    "条件 A (native_scaler): 目标域 train 行 mean/std (B5 标准口径)。"
    "条件 B (source_scaler): 源域 XJTU-SY train 行 mean/std, 冻结后套到目标域。"
    "正/负结果都接受; 这是 HI 层迁移必要性的对照证据。"
)


# ============================================================
# 数据路径解析
# ============================================================

def _resolve_data_root() -> Path:
    """根据环境自动定位 data/ 目录。

    本地开发: XA-202608_最终交付/04_数据/wheel/
    Docker: /app/release/data/ (compose 只读 mount)
    """
    # Docker mount: release/data/features/...
    if (DATA_ROOT_DOCKER / "features" / "wheel").exists():
        return DATA_ROOT_DOCKER
    # 本地: 04_数据/wheel/features/...
    if (DATA_ROOT_LOCAL / "features" / "wheel").exists():
        return DATA_ROOT_LOCAL
    raise SystemExit(
        f"!! 找不到 wheel data 目录。检查:\n"
        f"   Docker: {DATA_ROOT_DOCKER}\n"
        f"   本地:  {DATA_ROOT_LOCAL}")


def _resolve_source_h5(data_root: Path) -> Path:
    """源域 XJTU-SY 特征 h5。"""
    p = data_root / "features" / "wheel" / "schema_v1" / "source_features.h5"
    if not p.exists():
        raise SystemExit(f"!! 缺源域特征 {p}")
    return p


def _resolve_target_h5(data_root: Path) -> Path:
    """目标域 Basilisk B19 特征 h5。"""
    p = data_root / "features" / "wheel" / "basilisk_b19" / "target_features.h5"
    if not p.exists():
        raise SystemExit(f"!! 缺目标域特征 {p}")
    return p


# ============================================================
# 源域归一化参数 (条件 B 的核心)
# ============================================================

def compute_source_normalization(source_h5: Path) -> dict:
    """从源域 (XJTU-SY) train 行计算 clip (1%/99%) + z-score (mean/std) 参数。

    这些参数**只从源域计算**, 然后冻结, 用于归一化目标域特征。
    返回 dict, 供 prepare_target_with_scaler() 使用。
    """
    with h5py.File(source_h5, "r") as f:
        splits = f["split"][:]
        feats = f["features"][:].astype(np.float32)
        feat_names = [s.decode() if isinstance(s, bytes) else str(s)
                      for s in f["feature_names"][:]]
        tr_mask = splits == b"train"
        if not tr_mask.any():
            raise SystemExit("!! 源域无 train split")

        # 训练集统计量 (与 pretrain 口径一致: 只看 train 集)
        train_feats = feats[tr_mask]
        src_mean = train_feats.mean(axis=0).astype(np.float32)
        src_std = train_feats.std(axis=0).astype(np.float32)
        # clip 百分位 (1% / 99%) — 源域 train 集计算
        src_p1 = np.percentile(train_feats, 1, axis=0).astype(np.float32)
        src_p99 = np.percentile(train_feats, 99, axis=0).astype(np.float32)

    return {
        "mean": src_mean,
        "std": src_std,
        "p1": src_p1,
        "p99": src_p99,
        "feature_names": feat_names,
        "n_train_rows": int(tr_mask.sum()),
    }


def compute_native_normalization(xT: np.ndarray, train_mask: np.ndarray) -> dict:
    """从目标域 train 行计算 z-score 参数 (B2/B5 标准口径)。"""
    return {
        "mean": xT[train_mask].mean(axis=0).astype(np.float32),
        "std": xT[train_mask].std(axis=0).astype(np.float32),
    }


def apply_source_scaler_to_target(xT_target: np.ndarray,
                                  src_norm: dict,
                                  target_n_features: int) -> np.ndarray:
    """把源域归一化参数套到目标域特征上。

    源域 12 维, 目标域 10 维。按 min(F_src, F_tgt)=10 逐列对齐:
      目标第 j 列的归一化 = (目标第 j 列 - 源第 j 列 mean) / (源第 j 列 std + 1e-6)

    clip 使用源域 1%/99% 百分位。

    返回归一化后的 xT (与输入同形状)。不修改输入。
    """
    n_src = len(src_norm["mean"])
    n_tgt = xT_target.shape[1]
    n_use = min(n_src, n_tgt)

    out = xT_target.copy()
    # 逐列: 先 clip, 再 z-score (与 pretrain 顺序一致)
    src_p1 = src_norm["p1"][:n_use]
    src_p99 = src_norm["p99"][:n_use]
    src_mu = src_norm["mean"][:n_use]
    src_sd = src_norm["std"][:n_use]

    # clip 到源域 [p1, p99]
    col_data = out[..., :n_use]
    col_clipped = np.clip(col_data, src_p1, src_p99)
    # z-score 用源域统计量
    col_norm = (col_clipped - src_mu) / (src_sd + 1e-6)
    out[..., :n_use] = col_norm.astype(np.float32)
    return out


def apply_native_scaler(xT_target: np.ndarray, native_norm: dict) -> np.ndarray:
    """标准目标域 z-score (B2 prepare_b2 的逐字复制)。"""
    mu = native_norm["mean"]
    sd = native_norm["std"]
    return ((xT_target - mu) / (sd + 1e-6)).astype(np.float32)


# ============================================================
# 带可配置 scaler 的 prepare_b2 替代实现
# ============================================================

def prepare_target_with_scaler(cfg: dict, split: dict, seed: int,
                               xT_normalized: np.ndarray,
                               verbose: bool = True) -> dict:
    """与 prepare_b2 完全同构, 但 xT 已预先用外部 scaler 归一化。

    区别: 这里不再在函数内计算 mean/std, 调用方负责传归一化后的 xT。
    其余 (rul_scale, 删失契约, K 窗块, DataLoader) 与 prepare_b2 逐字一致,
    以保证两种 scaler 条件下训练流程严格配对。
    """
    tc = cfg["transfer"]
    mc = cfg["model"]
    L = int(mc["input_len_L"])
    K = int(cfg.get("pretrain", {}).get("seq_block_K", 8))
    stride = int(tc.get("target_stride", 50))
    hi_key = str(tc["target_hi_key"])

    # 用已归一化的 xT 重新读取其他通道 (hi, rul, lb, ev, tid)
    target_h5_path_cfg = str(ROOT / tc["target_feature_path"])
    # 但本地环境可能不在 ROOT 下; 这里我们假设 caller 已传正确路径
    # 我们从 xT_normalized 的 shape 推断 n_target
    n_target = int(xT_normalized.shape[1])

    set_seed(seed, cfg["reproducibility"]["deterministic"],
             cfg["reproducibility"]["cudnn_benchmark"])

    # 重新加载 hi/rul/lb/ev/tid (与 prepare_b2 一致, 但 xT 用归一化版)
    target_h5 = _resolve_target_h5(_resolve_data_root())
    # hi/rul/lb/ev/tid
    _xt_raw, hiT, rulT, lbT, evT, tidT, n_traj, tids = load_b2_target(
        target_h5, hi_key)

    name2i = {t: i for i, t in enumerate(tids)}
    tr = np.array([name2i[t] for t in split["splits"]["train"]["tids"]])
    va = np.array([name2i[t] for t in split["splits"]["val"]["tids"]])
    te = np.array([name2i[t] for t in split["splits"]["test"]["tids"]])

    m_tr = np.isin(tidT, tr)
    # rul_scale 只由 train 的 event-observed 轨迹给出 (censored 的 rul 是 NaN)
    rul_scale = max(float(np.nanmax(rulT[m_tr])), 1.0)
    rulT_n = rulT / rul_scale
    lbT_n = lbT / rul_scale

    bs = int(cfg["pretrain"]["batch_size"])
    eta = float(cfg["b2"]["censor_hinge_eta"])

    def _ds_train():
        m = np.isin(tidT, tr)
        r = rulT_n[m].copy()
        cen = ~evT[m]
        r[cen] = lbT_n[m][cen]
        return CensoredTargetSeqDataset(
            xT_normalized[m], hiT[m], r, tidT[m], L, K, stride=stride,
            event_observed=evT[m], rul_lower_bound=lbT_n[m])

    def _ds_eval(ids):
        m = np.isin(tidT, ids)
        return TargetSeqDataset(xT_normalized[m], hiT[m], rulT_n[m], tidT[m],
                                L, 1, stride=stride)

    device = ("cuda" if (cfg["pretrain"]["device"] == "cuda"
                         and torch.cuda.is_available()) else "cpu")

    # 与 B5 prepare_b5 同款 batch generator (§6.1 三组共用 seed*1000+7)
    gtr = torch.Generator()
    gtr.manual_seed(int(seed) * 1000 + TARGET_GEN_OFFSET)

    out = {
        "ltr": DataLoader(_ds_train(), batch_size=bs, shuffle=True, generator=gtr),
        "lva": DataLoader(_ds_eval(va), batch_size=bs, shuffle=False),
        "lte": DataLoader(_ds_eval(te), batch_size=bs, shuffle=False),
        "tr": tr, "va": va, "te": te,
        "n_traj": n_traj, "rul_scale": rul_scale,
        "n_target": n_target,
        "cap_eps": caliber_eps(cfg),
        "censor_eta": eta,
        "hi_key": hi_key,
        "tids": tids,
        "ev_by_traj": {tids[i]: bool(evT[tidT == i][0]) for i in range(n_traj)},
        "device": device,
        "batch_order_generator_seed": int(seed) * 1000 + TARGET_GEN_OFFSET,
        "split_sha256": str(split["split_sha256"]),
    }
    if verbose:
        n_ev_tr = int(sum(out["ev_by_traj"][tids[i]] for i in tr))
        print(f">> [feature_transfer] train {len(tr)} (event {n_ev_tr}) / "
              f"val {len(va)} / test {len(te)}  seed {seed}")
        print(f"   n_target={n_target}  rul_scale={rul_scale:.1f}")
    return out


def reset_target_batch_order(d: dict) -> int:
    """与 data_b5.reset_target_batch_order 同 (重置 train loader generator)。"""
    s = int(d["batch_order_generator_seed"])
    d["ltr"].generator.manual_seed(s)
    return s


# ============================================================
# target_only 训练 (与 run_formal_transfer.run_group 同结构)
# ============================================================

def train_target_only(cfg: dict, data: dict, src_n_features: int,
                      seed: int, th: dict, axis: dict,
                      ckpt_dir: Path, tag_prefix: str) -> dict:
    """单条件单 seed 的 target_only 训练 + test 评估。

    与 run_formal_transfer.py 的 run_group(target_only) 分支严格同结构:
      - n_features = src_n_features (12, 满足 §6.3 架构公平性)
      - n_target = data['n_target']
      - Adam(lr=finetune_lr, weight_decay=th.weight_decay)
      - _train_with_early_stop (info_macro_rmse, max_epochs, patience)
    """
    t0 = time.time()
    reset_target_batch_order(data)

    torch.manual_seed(int(seed))
    model = _build_model(cfg, src_n_features, data["n_target"], data["device"])

    huber = nn.HuberLoss(delta=float(cfg["loss"]["huber_delta"]))
    mse = nn.MSELoss()
    lam = (float(cfg["loss"]["lambda_hi"]), float(cfg["loss"]["lambda_mono"]),
           float(cfg["loss"]["lambda_smooth"]))
    tcfg = cfg.get("training", {})
    e = int(th["max_epochs"])
    kw = dict(post_eol_weight=float(tcfg.get("post_eol_weight", 1.0)),
              huber_delta=float(cfg["loss"]["huber_delta"]),
              capped_weight=float(tcfg.get("capped_weight", 1.0)),
              cap_eps=float(data["cap_eps"]),
              early_stop_metric=th["early_stop_metric"],
              early_stop_patience=th["early_stop_patience"],
              censor_eta=float(data["censor_eta"]))
    tag = f"ft_{tag_prefix}_s{seed}"
    history: list = []
    budget: dict = {}

    model.freeze_encoder(False)
    opt = torch.optim.Adam(list(model.parameters()),
                           lr=float(cfg["transfer"]["finetune_lr"]),
                           weight_decay=th["weight_decay"])
    best = _train_with_early_stop(model, data["ltr"], data["lva"], opt,
                                  data["device"], huber, mse, lam, e, tag,
                                  history=history, budget=budget, **kw)

    m = evaluate_b2(model, data["lte"], cfg, axis, data, tag=f"{tag_prefix}_s{seed}")
    m["_meta"] = {
        "condition": tag_prefix,
        "seed": int(seed),
        "n_features_source_side": int(src_n_features),
        "n_target": int(data["n_target"]),
        "split_sha256": str(data["split_sha256"]),
        "batch_order_generator_seed": int(data["batch_order_generator_seed"]),
        "max_epochs": int(th["max_epochs"]),
        "early_stop_patience": int(th["early_stop_patience"]),
        "early_stop_metric": str(th["early_stop_metric"]),
        "optimizer": "Adam",
        "lr": float(cfg["transfer"]["finetune_lr"]),
        "weight_decay": float(th["weight_decay"]),
        "best_epoch": int(budget.get("best_epoch", -1)),
        "val_metric": {"name": str(th["early_stop_metric"]),
                       "value": float(best),
                       "selection_basis": "validation_only"},
        "secs": round(time.time() - t0, 1),
    }
    print(f"   [{tag_prefix} s{seed}] test info macro RMSE = "
          f"{m['info_macro_rmse']:.6f}  "
          f"(best_ep {m['_meta']['best_epoch']}, {m['_meta']['secs']}s)")
    return m


# ============================================================
# 主流程
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG_PATH)
    ap.add_argument("--fast", action="store_true",
                    help="冒烟: 只跑 1 epoch (结果不得入正式报告)")
    ap.add_argument("--seeds", type=str, default=None,
                    help="覆盖默认 seeds (逗号分隔; 默认 112,113,114,115,116)")
    ap.add_argument("--out", type=str, default=None,
                    help="输出 JSON 路径 (默认: ../../05_结果/reference/wheel/feature_transfer_paired.json)")
    a = ap.parse_args()

    # ---- 加载配置 (与 run_formal_transfer 同链) ----
    cfg = load_cfg(a.config)

    # ---- 数据路径 ----
    data_root = _resolve_data_root()
    source_h5 = _resolve_source_h5(data_root)
    target_h5 = _resolve_target_h5(data_root)

    print(f">> 数据根目录: {data_root}")
    print(f">> 源域 h5: {source_h5.name}")
    print(f">> 目标域 h5: {target_h5.name}")

    # ---- 冻结划分 (B2.1) ----
    split = load_b21_split(cfg)
    schema = resolve_input_schema_b5(cfg)
    th = training_hyper(cfg)

    seeds = FORMAL_SEEDS
    if a.seeds:
        seeds = [int(s) for s in a.seeds.split(",")]
    if a.fast:
        cfg["training"]["max_epochs"] = 1
        cfg["b2_frozen_training_expected"]["max_epochs"] = 1
        th = training_hyper(cfg)
        print("!! --fast: max_epochs=1, 结果只能用于冒烟, 不得入正式报告")

    print(f">> feature_space_transfer: seeds {seeds} × 2 conditions")
    print(f">> 训练超参 (S2.5 冻结): max_epochs={th['max_epochs']}, "
          f"patience={th['early_stop_patience']}, "
          f"early_stop={th['early_stop_metric']}")

    # ---- 计算源域归一化参数 (条件 B 的核心, 与 seed 无关) ----
    src_norm = compute_source_normalization(source_h5)
    print(f">> 源域归一化参数 (from XJTU-SY train, "
          f"{src_norm['n_train_rows']} rows × {len(src_norm['mean'])} features):")
    print(f"   mean[:5] = {src_norm['mean'][:5]}")
    print(f"   std[:5]  = {src_norm['std'][:5]}")
    print(f"   p1[:5]   = {src_norm['p1'][:5]}")
    print(f"   p99[:5]  = {src_norm['p99'][:5]}")

    # 源域 n_features (与 B5 §6.3 一致: 共用 source encoder 宽度)
    src_n_features = len(src_norm["mean"])   # 12
    print(f">> src_n_features = {src_n_features} (与 B5 §6.3 shared_n_features 一致)")

    # ---- 读取目标域原始 xT (归一化前的) ----
    hi_key = str(cfg["transfer"]["target_hi_key"])
    xT_raw, hiT, rulT, lbT, evT, tidT, n_traj, tids = load_b2_target(
        target_h5, hi_key)
    print(f">> 目标域 xT_raw shape: {xT_raw.shape}")

    # 校验 x_T 列无泄漏 (与 prepare_b2 一致)
    _ = assert_no_hi_in_input(target_h5)

    axis = b2_endpoint_axis(cfg)
    ckpt_dir = ROOT / "checkpoints" / "_feature_transfer"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- 逐 seed × 2 条件 训练 ----
    t_all = time.time()
    native_rmses: list[float] = []
    source_rmses: list[float] = []
    per_seed_meta: list[dict] = []

    for seed in seeds:
        print(f"\n{'-' * 66}\n[feature_transfer] seed {seed}\n{'-' * 66}")

        # 在每个 seed 下, 两种 scaler 都用同一份 split/seed/模型配置
        # 只差 xT 的归一化参数来源

        # --- 条件 A: native_scaler ---
        # 目标域 train 行 mean/std (B2 标准口径)
        # rul_scale 与删失契约在 prepare_target_with_scaler 内部计算
        # (rul_scale 来自 rulT, 不受 xT 归一化影响, 两条件下同值)
        name2i = {t: i for i, t in enumerate(tids)}
        tr_ids = np.array([name2i[t] for t in split["splits"]["train"]["tids"]])
        m_tr = np.isin(tidT, tr_ids)
        native_norm = compute_native_normalization(xT_raw, m_tr)
        xT_native = apply_native_scaler(xT_raw, native_norm)

        set_seed(seed, cfg["reproducibility"]["deterministic"],
                 cfg["reproducibility"]["cudnn_benchmark"])
        data_native = prepare_target_with_scaler(
            cfg, split, seed, xT_native, verbose=True)
        m_native = train_target_only(
            cfg, data_native, src_n_features, seed, th, axis,
            ckpt_dir, tag_prefix="native_scaler")

        # --- 条件 B: source_scaler ---
        # 用 XJTU-SY train 行 mean/std (冻结, 与 seed 无关), 套到目标域
        xT_source_scaled = apply_source_scaler_to_target(
            xT_raw, src_norm, xT_raw.shape[1])

        # 同一 seed 重置 RNG (与条件 A 同一个起点)
        set_seed(seed, cfg["reproducibility"]["deterministic"],
                 cfg["reproducibility"]["cudnn_benchmark"])
        data_source = prepare_target_with_scaler(
            cfg, split, seed, xT_source_scaled, verbose=True)
        m_source = train_target_only(
            cfg, data_source, src_n_features, seed, th, axis,
            ckpt_dir, tag_prefix="source_scaler")

        native_rmses.append(float(m_native["info_macro_rmse"]))
        source_rmses.append(float(m_source["info_macro_rmse"]))
        per_seed_meta.append({
            "seed": int(seed),
            "native_scaler": {
                "rmse": float(m_native["info_macro_rmse"]),
                "best_epoch": int(m_native["_meta"]["best_epoch"]),
                "secs": float(m_native["_meta"]["secs"]),
            },
            "source_scaler": {
                "rmse": float(m_source["info_macro_rmse"]),
                "best_epoch": int(m_source["_meta"]["best_epoch"]),
                "secs": float(m_source["_meta"]["secs"]),
            },
        })

    # ---- 配对统计 ----
    paired_delta = [n - s for n, s in zip(native_rmses, source_rmses)]
    # paired_delta[k] > 0: source_scaler 的 RMSE 更低 (源域归一化更好)
    # paired_delta[k] < 0: source_scaler 的 RMSE 更高 (源域归一化更差)
    bs = paired_bootstrap(paired_delta, N_BOOTSTRAP, BOOTSTRAP_SEED)

    ci_lo = float(bs["ci_lower"])
    ci_hi = float(bs["ci_upper"])
    crosses_zero = bool(ci_lo <= 0.0 <= ci_hi)
    mean_delta = float(np.mean(paired_delta))

    print(f"\n{'=' * 66}")
    print(f"[feature_transfer] 5-seed paired results")
    print(f"{'=' * 66}")
    print(f"  native_scaler RMSE per seed: "
          f"{[round(r, 6) for r in native_rmses]}  "
          f"mean={np.mean(native_rmses):.6f}  std={np.std(native_rmses):.6f}")
    print(f"  source_scaler RMSE per seed: "
          f"{[round(r, 6) for r in source_rmses]}  "
          f"mean={np.mean(source_rmses):.6f}  std={np.std(source_rmses):.6f}")
    print(f"  paired_delta (native - source) = "
          f"{[round(d, 6) for d in paired_delta]}")
    print(f"  mean_delta = {mean_delta:+.6f}")
    print(f"  bootstrap 95% CI = [{ci_lo:+.6f}, {ci_hi:+.6f}]")
    print(f"  crosses_zero = {crosses_zero}")
    print(f"  interpretation: "
          f"{'source_scaler worse' if mean_delta < 0 else 'source_scaler better'}"
          f" (CI {'crosses' if crosses_zero else 'does NOT cross'} zero)")
    print(f"  total time: {time.time() - t_all:.1f}s")

    # ---- 输出 JSON ----
    out_dir_default = (ROOT.parent.parent.parent.parent
                       / "05_结果" / "reference" / "wheel")
    out_path = (Path(a.out) if a.out
                else out_dir_default / "feature_transfer_paired.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "experiment": "feature_space_transfer_5seeds",
        "purpose": PURPOSE,
        "config": a.config,
        "config_sha256":
            hashlib.sha256(Path(a.config).read_bytes()).hexdigest()
            if Path(ROOT / a.config).exists() else None,
        "fast_mode": bool(a.fast),
        "seeds": [int(s) for s in seeds],
        "split_sha256": str(split["split_sha256"]),
        "split_reused_from": str(cfg["b5"]["split"]["reuse_from"]),
        "input_schema": schema,
        "training_hyper": dict(th),
        "src_n_features": int(src_n_features),
        "target_n_features": int(xT_raw.shape[1]),
        "source_normalization": {
            "source_dataset": "XJTU-SY",
            "source_feature_names": src_norm["feature_names"],
            "n_train_rows": int(src_norm["n_train_rows"]),
            "mean_first5": [float(x) for x in src_norm["mean"][:5]],
            "std_first5": [float(x) for x in src_norm["std"][:5]],
            "p1_first5": [float(x) for x in src_norm["p1"][:5]],
            "p99_first5": [float(x) for x in src_norm["p99"][:5]],
        },
        "feature_alignment_note": (
            "源域 12 维 (XJTU-SY 振动特征) vs 目标域 10 维 (Basilisk 遥测), "
            "按 min(F_src, F_tgt)=10 逐列位置对齐 (源前 10 列统计量套到目标 10 列)。"
            "这是朴素的跨域归一化迁移, 不含物理对应 —— 对应 HI 层迁移必要性的对照。"),
        "conditions": {
            "native_scaler": {
                "description": "目标域 train 行 mean/std (B2/B5 标准口径)",
                "per_seed_rmse": [float(r) for r in native_rmses],
                "mean": float(np.mean(native_rmses)),
                "std": float(np.std(native_rmses)),
                "median": float(np.median(native_rmses)),
            },
            "source_scaler": {
                "description": "源域 XJTU-SY train 行 mean/std, 冻结后套到目标域",
                "per_seed_rmse": [float(r) for r in source_rmses],
                "mean": float(np.mean(source_rmses)),
                "std": float(np.std(source_rmses)),
                "median": float(np.median(source_rmses)),
            },
        },
        "paired_delta": {
            "definition": "native_scaler_rmse - source_scaler_rmse; 正数 = source_scaler 更好",
            "per_seed": [float(d) for d in paired_delta],
            "mean": mean_delta,
            "std": float(np.std(paired_delta)),
            "ci95": [ci_lo, ci_hi],
            "ci95_unit": "one_paired_difference_per_seed",
            "ci95_method": f"paired_bootstrap n_rep={N_BOOTSTRAP} seed={BOOTSTRAP_SEED}",
            "crosses_zero": crosses_zero,
            "improve_count": int(sum(1 for d in paired_delta if d > 0)),
            "n_seeds": int(len(paired_delta)),
            "note": ("n=5 seed bootstrap 只描述训练随机性下差值方向是否稳定, "
                     "不是泛化误差的置信区间。"),
        },
        "per_seed_detail": per_seed_meta,
        "secs_total": round(time.time() - t_all, 1),
    }

    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8")
    print(f"\n>> 已写 {out_path}  ({result['secs_total']}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
