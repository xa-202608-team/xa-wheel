"""experiments/run_groups.py

对比实验编排 (plan §四 Phase 6 飞轮 / §8 相控阵八组), 每组 n 种子,
聚合 mean±std -> docs/results.md。

组件级参数化 (2026-07-21 PA6 阶段1): 从 config experiments.groups 读组名,
内部 mode 映射 + checkpoint 组件后缀, 飞轮 (4 组) / 相控阵 (8 组) 共用。

组 (相控阵 config §8):
  target_only_tcn / target_only_gru    仅目标域少样本 (TCN / GRU 基线)
  source_pretrain_finetune             源域预训练 + 微调
  source_mmd_physics                   源域 + 阶段 MMD + 物理一致性 (主迁移模型)
  timesfm_zeroshot / xreg / lora_xreg  TimesFM 辅助分支 (PA7; 未启用则 skip 占位)
  main_timesfm_fusion                  主模型 + TimesFM 概率融合 (PA7)
飞轮 config §P6: target_only / source_frozen / source_finetune / source_mmd_finetune / physical_extrap
(旧名保留兼容, 经 _GROUP_MAP 映射到内部 mode)

用法:
  python -m src.experiments.run_groups --config configs/wheel.yaml --seeds 5
  python -m src.experiments.run_groups --config configs/phased_array.yaml --smoke
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import load_config, set_seed                                    # noqa: E402
from src.transfer.adapter import TransferModel                                 # noqa: E402
from src.transfer.mmd import mmd_by_hi_bins, reset_global_memory_bank           # noqa: E402
from src.transfer.train_transfer import (                                      # noqa: E402
    TargetSeqDataset, SourceWindowDataset, split_trajectories,
    split_trajectories_from_cfg, describe_eol_split, load_target, load_source)
from src.baselines.physical_extrap import (                                    # noqa: E402
    evaluate_physical, phm_score, mae as mae_fn,
    CALIBERS, PRIMARY_CALIBER, PRIMARY_STAT, CAP_EPS,
    caliber_eps, caliber_mask, pooled_metrics, macro_metrics,
    caliber_metrics, format_calibers)

CKPT_DIR = ROOT / "checkpoints"

# ---- 评估口径 (S2' 改动 3; S2' 任务 4 起实现移到 src/baselines/physical_extrap.py) ----
# 三档对所有方法对称应用; RUL 已按 train 集 rul_scale 归一, 故 1.0 = 截顶值。
#   full 全部 / valid RUL>0 (剔零难度) / info 0<RUL<1-eps (剔零难度+截顶饱和) ← 主口径
# 指标实现集中在 physical_extrap 是为了让 trivial / physical_extrap / diag_* 与本模块
# 共用同一份口径代码 (run_groups 单向 import 它, 反向会循环)。下方别名保持旧符号可用。
_caliber_mask = caliber_mask
_pooled_metrics = pooled_metrics
_macro_metrics = macro_metrics

# ---- 早停可选指标 (S2.5 §6) ----
# 名字 → 从 eval_test/caliber_metrics 返回结构里取标量的函数。**只允许 val 指标**:
# _train_with_early_stop 的签名里根本没有 test loader, 结构上无法让 test 参与 checkpoint 选择
# (tests/test_generalization_gate.py::test_earlystop_never_reads_test 用 signature 钉死)。
# 名字里出现 "test" 一律拒绝, 防止后来者从这里开后门。
EARLY_STOP_METRICS = {
    "info_pooled_rmse": lambda m: m["calibers"]["info"]["pooled"]["rmse"],
    "info_macro_rmse":  lambda m: m["calibers"]["info"]["macro"]["rmse"],
    "full_pooled_rmse": lambda m: m["calibers"]["full"]["pooled"]["rmse"],
    "valid_pooled_rmse": lambda m: m["calibers"]["valid"]["pooled"]["rmse"],
}


def early_stop_value(metrics: dict, name: str) -> float:
    """按名字从 val 指标结构取早停标量; NaN → +inf (视为最差, 不会被选为 best)。"""
    if "test" in name.lower():
        raise ValueError(f"早停指标不得引用 test: {name}")
    if name not in EARLY_STOP_METRICS:
        raise KeyError(f"未知早停指标 {name}; 可选 {sorted(EARLY_STOP_METRICS)}")
    v = float(EARLY_STOP_METRICS[name](metrics))
    return float("inf") if not np.isfinite(v) else v


def training_hyper(cfg: dict) -> dict:
    """训练时长 / 早停 / 正则 的统一读取口径 (S2.5 唯一允许变动的四个自由度)。

    run_one_group 与 scripts/diag_generalization.py / scripts/run_s25_gate.py 共用,
    保证"调参跑的那套"与"正式实验跑的那套"是同一份参数解析。
    max_epochs 缺省回退到 transfer.epochs_s2 (S2'' 之前的隐式默认 20), 保持向后数值等价。
    """
    tcfg = cfg.get("training", {})
    tc = cfg.get("transfer", {})
    name = str(tcfg.get("early_stop_metric", "info_pooled_rmse"))
    if name not in EARLY_STOP_METRICS:
        raise KeyError(f"training.early_stop_metric={name} 未知; 可选 {sorted(EARLY_STOP_METRICS)}")
    return {
        "early_stop_metric": name,
        "early_stop_patience": int(tcfg.get("early_stop_patience", 0)),
        "max_epochs": int(tcfg.get("max_epochs", tc.get("epochs_s2", 20))),
        "weight_decay": float(tcfg.get("weight_decay", 0.0)),
    }


# 依赖 PA7 (TimesFM 分支未建): timesfm.enabled=False 时 skip + 占位
TIMESFM_GROUPS = {"timesfm_zeroshot", "timesfm_xreg", "timesfm_lora_xreg",
                  "main_timesfm_fusion"}

# config 组名 → (内部 mode, encoder_override)。None encoder = 用 config 默认
_GROUP_MAP = {
    # 相控阵 (config §8)
    "target_only_tcn":            ("target_only", "tcn"),
    "target_only_gru":            ("target_only", "gru"),
    "source_pretrain_finetune":   ("source_finetune", None),
    "source_mmd_physics":         ("source_mmd_finetune", None),
    # 飞轮旧名 (兼容)
    "target_only":                ("target_only", None),
    "source_frozen":              ("source_frozen", None),
    "source_finetune":            ("source_finetune", None),
    "source_mmd_finetune":        ("source_mmd_finetune", None),
}

# results.md 表格标签
LABELS = {
    "target_only":              "Target-only",
    "target_only_tcn":          "Target-only (TCN)",
    "target_only_gru":          "Target-only (GRU)",
    "source_frozen":            "Source+Frozen Encoder",
    "source_finetune":          "Source+Finetune",
    "source_pretrain_finetune": "Source+Finetune",
    "source_mmd_finetune":      "**Source+MMD+Finetune**",
    "source_mmd_physics":       "**Source+MMD+Physics**",
    "timesfm_zeroshot":         "TimesFM zero-shot *(PA7)*",
    "timesfm_xreg":             "TimesFM + XReg *(PA7)*",
    "timesfm_lora_xreg":        "TimesFM + LoRA + XReg *(PA7)*",
    "main_timesfm_fusion":      "主模型 + TimesFM 融合 *(PA7)*",
}


def _resolve_group(group_name, cfg):
    """config 组名 → (内部 mode, encoder_override, enabled, reason)。

    TimesFM 组依赖 PA7 (timesfm.enabled); 物理基线单独 evaluate; 其他查 _GROUP_MAP。
    """
    if group_name in TIMESFM_GROUPS:
        enabled = bool(cfg.get("timesfm", {}).get("enabled", False))
        reason = "TimesFM 已启用" if enabled else "TimesFM 未启用 (PA7 待推进)"
        return None, None, enabled, reason
    if group_name == "physical_extrap":
        return "__physical__", None, True, "物理基线 (单独 evaluate)"
    if group_name in _GROUP_MAP:
        mode, enc = _GROUP_MAP[group_name]
        return mode, enc, True, ""
    return None, None, False, "未知组名 (跳过)"


def _build_model(cfg, n_features, n_target, device, encoder_override=None,
                 rate_head=False):
    mc = cfg["model"]
    tc = cfg["transfer"]
    enc = encoder_override or mc["encoder"]
    rc = cfg.get("rate", {}) if rate_head else {}
    return TransferModel(
        encoder_type=enc, n_features=n_features, n_target=n_target,
        channels=mc["tcn"]["channels"], kernel_size=mc["tcn"]["kernel_size"],
        num_blocks=mc["tcn"]["num_blocks"], dropout=mc["tcn"]["dropout"],
        latent_dim=mc["latent_dim"], adapter_hidden=tc["adapter_hidden"],
        rate_head=bool(rate_head),
        rate_hidden=int(rc.get("rate_hidden", 0)),
        eps_mu=float(rc.get("eps_mu", 1e-2)),
        eps_sigma=float(rc.get("eps_sigma", 1e-3))).to(device)


# ============================================================
# S5 速率损失 + 首达换算 (改动 1/2)
# ============================================================

def rate_cfg(cfg: dict) -> dict:
    """读取 `rate` 段 (S5)。缺键即报错, 绝不静默用魔法默认值 (与 prognostics_cfg 同纪律)。"""
    rc = (cfg or {}).get("rate")
    if not rc:
        raise KeyError("configs 缺 rate 段 (S5 改动 1/2)")
    need = ("enabled", "eps_mu", "eps_sigma", "hi_fail", "rul_clip", "slope_window",
            "slope_estimator", "w_rate", "w_rul", "w_nll", "capped_weight",
            "post_eol_weight", "interval_z", "scale_by_rul_scale")
    miss = [k for k in need if k not in rc]
    if miss:
        raise KeyError(f"rate 段缺键 {miss} (S5)")
    return {
        "enabled": bool(rc["enabled"]),
        "eps_mu": float(rc["eps_mu"]), "eps_sigma": float(rc["eps_sigma"]),
        "rate_hidden": int(rc.get("rate_hidden", 0)),
        "hi_fail": float(rc["hi_fail"]), "rul_clip": float(rc["rul_clip"]),
        "slope_window": int(rc["slope_window"]),
        "slope_min_periods": int(rc.get("slope_min_periods", 8)),
        "slope_estimator": str(rc["slope_estimator"]),
        "theil_sen_subsample": int(rc.get("theil_sen_subsample", 64)),
        "w_rate": float(rc["w_rate"]), "w_rul": float(rc["w_rul"]),
        "w_nll": float(rc["w_nll"]),
        "capped_weight": float(rc["capped_weight"]),
        "post_eol_weight": float(rc["post_eol_weight"]),
        "interval_z": float(rc["interval_z"]),
        "scale_by_rul_scale": bool(rc["scale_by_rul_scale"]),
    }


def _rate_loss(mu_p, sigma_p, hi_obs, r, ev, lb, mu_t, rc, cap_eps, censor_eta):
    """S5 总损失 (改动 2):

        L = w_rate·L_rate + w_rul·L_huber(RUL_hat, 仅信息区) + w_nll·NLL + eta·L_C

    (1) **L_rate 主项**: mu_hat 对窗口内 HI 长基线局部斜率的监督 (Huber, delta 用
        config loss.huber_delta 的同一值语义, 此处固定 1.0 与 RUL 侧同式)。
        标签可靠、有 S4 证据支撑、且是网络唯一能从 32 小时窗口里真正看到的量。
        **全部样本参与** —— 包括截顶段与 post-EOL 段: 速率标签在那些位置依然有效
        (截顶只是 RUL 标签被夹到上限, 不是速率失真)。这正是改动 2 说的
        "capped_weight 设 0 的样本仍可参与 L_rate"。

    (2) **L_huber 副项**: 首达换算出的 RUL_hat 对真值。只在**信息区** 0<RUL<1-eps
        且 event_observed 的样本上生效 —— 截顶权重 rate.capped_weight=0 直接把
        RUL>=1-eps 的样本剔出该项 (仍留在 L_rate), post-EOL 按 post_eol_weight 降权。
        它的作用是给换算层一个尺度锚, 防 mu 整体偏移; 权重刻意小于 L_rate。

    (3) **NLL**: 用首达方差 Var=(hi_fail-HI)·sigma^2/mu^3 做高斯 NLL, 只在信息区
        且观测样本上算 —— 删失样本的"真值"只是下界, 拿它当高斯均值的目标会把区间
        往下拽。这一项让 sigma 有意义, 区间 90% 覆盖率才可校准 (完成判据附带项)。

    (4) **删失项**: 与 S3 完全同式 eta·mean[relu(lb - RUL_hat)]^2, 作用在换算后的
        RUL_hat 上。S4 已证 S3 的 L_O≡0 是坍缩机制之一; 此处 L_rate 提供了不依赖
        观测 RUL 的下拉力, 删失项只保留"不得低于已知下界"的约束语义。

    返回 (loss, 明细 dict)。
    """
    from src.models.rate_head import (first_passage_rul, first_passage_var,
                                      gaussian_nll)
    O = ev.bool()
    # ---- (1) L_rate: 全样本 ----
    L_rate = nn.functional.huber_loss(mu_p, mu_t, delta=1.0, reduction="mean")

    # ---- 首达换算 (唯一实现来自 src/models/rate_head.py) ----
    rul_p = first_passage_rul(hi_obs, mu_p, rc["hi_fail"], rc["eps_mu"],
                              rc["rul_clip"])

    # ---- (2) L_huber: 信息区 + 观测 ----
    info = O & (r > 0) & (r < 1.0 - float(cap_eps))
    post = O & (r <= 0)
    zero = mu_p.new_zeros(())
    if info.any() or post.any():
        per = nn.functional.huber_loss(rul_p, r, delta=1.0, reduction="none")
        w = torch.zeros_like(r)
        w = torch.where(info, torch.ones_like(r), w)
        w = torch.where(post, torch.full_like(r, rc["post_eol_weight"]), w)
        # 截顶段权重 = rate.capped_weight (默认 0 -> 完全不参与本项)
        capped = O & (r >= 1.0 - float(cap_eps))
        w = torch.where(capped, torch.full_like(r, rc["capped_weight"]), w)
        L_huber = (per * w).sum() / w.sum().clamp_min(1e-12)
    else:
        L_huber = zero

    # ---- (3) NLL: 信息区 + 观测 ----
    if info.any():
        var = first_passage_var(hi_obs[info], mu_p[info], sigma_p[info],
                                rc["hi_fail"], rc["eps_mu"])
        L_nll = gaussian_nll(r[info], rul_p[info], var).mean()
    else:
        L_nll = zero

    # ---- (4) 删失 hinge ----
    C = ~O
    if C.any() and censor_eta is not None:
        hinge = torch.relu(lb[C] - rul_p[C])
        L_C = (hinge * hinge).mean()
    else:
        L_C = zero

    loss = (rc["w_rate"] * L_rate + rc["w_rul"] * L_huber + rc["w_nll"] * L_nll
            + (float(censor_eta) if censor_eta is not None else 0.0) * L_C)
    return loss, {
        "L_rate": float(L_rate.detach()), "L_huber": float(L_huber.detach()),
        "L_nll": float(L_nll.detach()), "L_C": float(L_C.detach()),
        "n_info": int(info.sum()), "n_cen": int(C.sum()),
        "mu_mean": float(mu_p.detach().mean()),
        "mu_true_mean": float(mu_t.detach().mean()),
    }



def _weighted_huber(rul_p, r, delta, post_eol_weight, capped_weight=1.0,
                    cap_eps=CAP_EPS):
    """带 post-EOL / 截顶饱和 逐样本权重的 Huber (S2' 改动 4 + S2' 任务 2)。

    两段零信息量样本, 标签都是常数, 全权计入会让"输出常数"成为损失最优解:
      1. **失效后段** (归一 RUL == 0), 逐轨迹占 65%~95.6%      → post_eol_weight
      2. **截顶饱和段** (归一 RUL >= 1-cap_eps), 由 rul_cap_ratio=0.35 造成 → capped_weight

    只降权 (1) 而不降权 (2) 是 S2' 观察到的"阶跃函数"行为的怀疑根因: 损失被两端的
    常数平台主导 → 最优解退化为"低段输 0 / 高段输 1"的阶跃, 中间 info 区几乎不学。
    实测证据: full 口径 RMSE=0.1518 但 info 口径 0.2637 (常数预测器约 0.289) —— 全局
    看起来好、唯一有价值的区间里几乎不比常数强, 正是阶跃的指纹。

    **按权重降权而不硬删**: 硬删会破坏窗口连续性与轨迹级完整划分 (mono/smooth 约束
    依赖同轨迹连续 K 窗)。权重作用在 batch 内逐样本 (逐 (B,K) 元素) 的 elementwise
    Huber 上再做加权平均; 两权重都为 1.0 时与 nn.HuberLoss(reduction='mean') 数值等价。
    """
    per = nn.functional.huber_loss(rul_p, r, delta=delta, reduction="none")
    if post_eol_weight >= 1.0 and capped_weight >= 1.0:
        return per.mean()
    w = torch.ones_like(r)
    if post_eol_weight < 1.0:
        w = torch.where(r > 0, w, torch.full_like(r, float(post_eol_weight)))
    if capped_weight < 1.0:
        w = torch.where(r < 1.0 - float(cap_eps), w,
                        torch.full_like(r, float(capped_weight)))
    return (per * w).sum() / w.sum().clamp_min(1e-12)


def _censored_rul_loss(rul_p, r, ev, lb, delta, post_eol_weight, capped_weight,
                       cap_eps, eta):
    """S3 删失 RUL 损失 (协议 §8) = 观测项 Huber + 删失项 one-sided hinge。

    (A) 观测项: 只对 event_observed=True 的样本算 `_weighted_huber` —— 与 S2.5 完全同式
        (含 post_eol_weight / capped_weight 逐样本降权), 保证未截断轨迹的损失逐行不变。
    (B) 删失项: eta * mean[ max(0, rul_lower_bound - pred) ]^2, 只对 event_observed=False
        的样本。pred >= lb 时 relu 输出 0, 损失与梯度都恰为 0 —— 只惩罚"预测低于已知下界",
        允许模型给尚未失效的器件更大的 RUL。数学定义与 src/train/pretrain.py::_rul_loss
        的删失分支完全一致 (同一份语义, 不另发明).
    删失样本**绝不进入 Huber**: 它们的 rul 标签只是下界, 当精确值监督会把模型往下拽。

    返回 (loss, L_O, L_C, n_obs, n_cen) —— 后四项供审计打印。
    """
    O = ev.bool()
    C = ~O
    n_obs, n_cen = int(O.sum()), int(C.sum())
    if n_obs > 0:
        L_O = _weighted_huber(rul_p[O], r[O], delta, post_eol_weight,
                              capped_weight, cap_eps)
    else:
        L_O = rul_p.new_zeros(())
    if n_cen > 0:
        hinge = torch.relu(lb[C] - rul_p[C])
        L_C = (hinge * hinge).mean()
    else:
        L_C = rul_p.new_zeros(())
    return L_O + float(eta) * L_C, L_O, L_C, n_obs, n_cen


def _train_epoch(model, loader, opt, device, huber, mse, lam,
                 use_mmd=False, src_iter=None, bins=None, mmd_lambda=1.0,
                 post_eol_weight=1.0, huber_delta=1.0,
                 capped_weight=1.0, cap_eps=CAP_EPS,
                 censor_eta=None, budget=None, rc=None):
    """一个 epoch。censor_eta 非 None 且 loader 产出 5 元组时启用删失损失 (S3)。

    budget: 可选 dict, 累加 {"steps": 优化器 step 次数, "batches": batch 数,
            "n_obs": 观测样本数, "n_cen": 删失样本数} —— 供协议 §12 预算对齐审计。

    rc: 可选 rate_cfg(cfg) (S5)。给出且 loader 产出 **6 元组** 时走速率路径:
        model.forward_rate -> (mu, sigma) -> 首达换算 -> _rate_loss。
        rc=None 时本函数与 S3/S4 **逐行相同** (direct_rul 消融组与旧调用走此路)。
    """
    model.train()
    tl, n = 0.0, 0
    for batch in loader:
        ev = lb = mu_t = None
        if len(batch) == 6:                      # S5 速率 train loader
            x, h, r, ev, lb, mu_t = batch
            ev, lb, mu_t = ev.to(device), lb.to(device), mu_t.to(device)
        elif len(batch) == 5:                    # S3 删失 train loader
            x, h, r, ev, lb = batch
            ev, lb = ev.to(device), lb.to(device)
        else:
            x, h, r = batch
        if use_mmd:
            xs, hs = next(src_iter)
            xs, hs = xs.to(device), hs.to(device)
        x, h, r = x.to(device), h.to(device), r.to(device)
        B, Kk = x.size(0), x.size(1)
        xf = x.reshape(B * Kk, x.size(2), x.size(3))
        if mu_t is not None and rc is not None:
            # ---- S5 速率路径 ----
            hi_p, mu_p, sg_p, zT = model.forward_rate(xf)
            hi_p = hi_p.view(B, Kk)
            mu_p = mu_p.view(B, Kk)
            sg_p = sg_p.view(B, Kk)
            Lr, det = _rate_loss(mu_p, sg_p, h, r, ev, lb, mu_t, rc,
                                 cap_eps, censor_eta)
            n_obs, n_cen = int(ev.bool().sum()), det["n_cen"]
        else:
            hi_p, rul_p, zT = model(xf)
            hi_p = hi_p.view(B, Kk)
            rul_p = rul_p.view(B, Kk)
            if censor_eta is not None and ev is not None:
                Lr, _lo, _lc, n_obs, n_cen = _censored_rul_loss(
                    rul_p, r, ev, lb, huber_delta, post_eol_weight, capped_weight,
                    cap_eps, censor_eta)
            else:
                Lr = _weighted_huber(rul_p, r, huber_delta, post_eol_weight,
                                     capped_weight, cap_eps)
                n_obs, n_cen = int(r.numel()), 0
        Lh = mse(hi_p, h)
        d = hi_p[:, 1:] - hi_p[:, :-1]
        mono = torch.relu(-d).mean() if d.numel() else hi_p.new_zeros(())
        smooth = (d * d).mean() if d.numel() else hi_p.new_zeros(())
        loss = Lr + lam[0] * Lh + lam[1] * mono + lam[2] * smooth
        if use_mmd:
            zS = model.encoder(xs)
            loss = loss + mmd_lambda * mmd_by_hi_bins(zS, hs, zT, h.reshape(-1), bins)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += loss.item() * B
        n += B
        if budget is not None:
            budget["steps"] = budget.get("steps", 0) + 1
            budget["batches"] = budget.get("batches", 0) + 1
            budget["n_obs"] = budget.get("n_obs", 0) + int(n_obs)
            budget["n_cen"] = budget.get("n_cen", 0) + int(n_cen)
    return tl / max(n, 1)



@torch.no_grad()
def eval_test(model, loader, device, cap_eps=CAP_EPS, cfg=None, hi=None, rc=None):
    """三档口径 × (池化 / 宏平均) 评估 (S2' 改动 3), 对所有方法对称应用。

    指标计算委托给 physical_extrap.caliber_metrics —— 与 trivial / physical_extrap
    基线共用同一份口径实现, 保证"同划分、同口径、同指标"。返回结构见该函数 docstring。
    宏平均需要样本→轨迹映射: 从 loader.dataset 的 sample_tids 取 (TargetSeqDataset 提供);
    取不到时 macro 退化为单组 (与 pooled 相同), 并在 has_tids=False 里标明。

    ---- S4 扩展 (协议 docs/diagnostics_s4_protocol.md §14) ----
    `cfg` 给出时**追加** S4 预后性指标 key: `full_macro_rmse` / `info_macro_rmse` /
    `info_pooled_rmse` / `stage_metrics` / `prognostic_horizon` / `alpha_lambda` /
    `warning` / `convergence`。`cfg=None` 时行为与 S2.5/S3 **逐位相同** ——
    旧调用 (run_s25_gate / run_s3_gate / diag_*) 不传 cfg, 一个 key 都不会多, 不会崩。
    `hi` 可选覆盖 HI 数组 (测试注入用); 缺省从 cfg 的 target h5 复算 endpoint 轴。

    ---- S5 扩展 (改动 1) ----
    `rc` = rate_cfg(cfg) 给出时走速率路径: 模型输出 (mu, sigma), RUL 由
    `first_passage_rul(观测 HI, mu)` 换算 —— 与训练侧、与 Wiener+PF 基线**同一模块**。
    额外返回 `interval` (区间 90% 覆盖率 / 平均宽度) 与 `rate_stats`。
    `rc=None` 时与 S3/S4 逐位相同 (direct_rul 消融组走此路)。
    """
    model.eval()
    preds, labels, his, sigmas, mus = [], [], [], [], []
    from src.models.rate_head import (first_passage_rul, first_passage_var,
                                      gaussian_interval)
    for batch in loader:
        x, h, r = batch[0], batch[1], batch[2]
        x = x.to(device)
        B, Kk = x.size(0), x.size(1)
        xf = x.reshape(B * Kk, x.size(2), x.size(3))
        if rc is not None:
            _hp, mu_p, sg_p, _z = model.forward_rate(xf)
            hv = h.reshape(-1).numpy()
            mu_v = mu_p.cpu().numpy().reshape(-1)
            sg_v = sg_p.cpu().numpy().reshape(-1)
            # 换算用**观测 HI** (hi_b, 自校准), 不用模型预测的 hi_pred
            rul_v = first_passage_rul(hv, mu_v, rc["hi_fail"], rc["eps_mu"],
                                      rc["rul_clip"])
            preds.append(np.asarray(rul_v, dtype=float))
            his.append(hv)
            mus.append(mu_v)
            sigmas.append(sg_v)
        else:
            _, rul_p, _ = model(xf)
            preds.append(rul_p.cpu().numpy().reshape(-1))
        labels.append(r.reshape(-1).numpy())
    p = np.concatenate(preds) if preds else np.array([0.0])
    t = np.concatenate(labels) if labels else np.array([0.0])

    # 样本 → 轨迹 id (K=1 的评估 loader 下每样本展开为 1 个预测点)
    tids, has_tids = _loader_sample_tids(loader, len(t))
    out = caliber_metrics(p, t, tids if has_tids else None, cap_eps=cap_eps)
    if rc is not None and his:
        hv = np.concatenate(his)
        mu_v = np.concatenate(mus)
        sg_v = np.concatenate(sigmas)
        var = first_passage_var(hv, mu_v, sg_v, rc["hi_fail"], rc["eps_mu"])
        lo, up = gaussian_interval(p, var, rc["interval_z"])
        m = (t > 0) & (t < 1.0 - float(cap_eps))          # 信息区口径 (与主结论同)
        out["interval"] = {
            "z": rc["interval_z"],
            "nominal_coverage": float(1.0 - 2.0 * 0.05),  # z=1.645 -> 90%
            "coverage_info": float(np.mean((t[m] >= lo[m]) & (t[m] <= up[m])))
                             if m.any() else float("nan"),
            "mean_width_info": float(np.mean(up[m] - lo[m])) if m.any() else float("nan"),
            "n_info": int(m.sum()),
        }
        out["rate_stats"] = {
            "mu_mean": float(mu_v.mean()), "mu_std": float(mu_v.std()),
            "mu_min": float(mu_v.min()), "mu_max": float(mu_v.max()),
            "sigma_mean": float(sg_v.mean()),
            "frac_mu_at_floor": float(np.mean(mu_v <= rc["eps_mu"] * 1.001)),
        }
    if cfg is not None and has_tids:
        out.update(s4_metrics_block(p, t, tids, cfg, hi=hi))
    return out


def s4_metrics_block(pred, true, tids, cfg, hi=None) -> dict:
    """把 (pred, true, tids) 转成 S4 统一指标块 —— 模型侧与基线侧共用的唯一实现。

    这是 §15 "physical_extrap / const_mean / const_mean_info / hi_extrap 与三个迁移组
    使用完全相同的 evaluator" 的落点: 所有方法都从这里拿指标, 口径不可能分叉。

    两条口径**刻意分开**, 不混用 (协议 §2 "trajectory-macro 与 pooled 必须分开"同精神):

    * `stage_metrics` 走 **info 掩码** (0 < RUL < 1-eps) —— 与 S3 表 3 的 PSR /
      macro_corr 严格同口径, 故 S4 分阶段 PSR 可以直接和 S3 的 0.0239~0.2996 对读。
      若这里用全样本, EOL 之后那段 RUL 恒 0 会把 true_std 抬高、PSR 虚高, 造出
      "迁移组动态范围其实还不错"的假象。
    * PH / alpha-lambda / warning / convergence 走 **全部 endpoint** —— 这些指标依赖
      完整时间轴, 被 info 掩码打断会让"连续 persistence 个 endpoint"与
      "此后始终在带内"失去意义。
    """
    from src.baselines.physical_extrap import caliber_mask
    from src.experiments.metrics import (attach_axis, prognostic_metrics,
                                         prognostics_cfg, staged_metrics,
                                         target_endpoint_axis)
    pc = prognostics_cfg(cfg)
    cap_eps = float(cfg["evaluation"]["capped_eps"])
    axis = target_endpoint_axis(cfg)
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    tids = np.asarray(tids, dtype=np.int64)
    tau, hi_axis, ev = attach_axis(tids, axis)
    hi_use = np.asarray(hi, dtype=float) if hi is not None else hi_axis

    cal = caliber_metrics(pred, true, tids, cap_eps=cap_eps)
    blk = prognostic_metrics(pred, true, hi_use, tau, tids, ev, pc)
    m = caliber_mask(true, "info", cap_eps)          # 分阶段指标 = info 口径
    blk["stage_metrics"] = staged_metrics(pred[m], true[m], hi_use[m], tids[m],
                                          pc["hi_bins"], tau=tau[m])
    blk["stage_caliber"] = "info"
    blk.update({
        "full_macro_rmse": cal["calibers"]["full"]["macro"]["rmse"],
        "info_macro_rmse": cal["calibers"]["info"]["macro"]["rmse"],
        "info_pooled_rmse": cal["calibers"]["info"]["pooled"]["rmse"],
    })
    return blk


def _loader_sample_tids(loader, n_points):
    """从 loader.dataset 取样本→轨迹映射, 展平到逐预测点。取不到则返回全 0 + False。

    评估 loader 以 K=1 构造 (每样本 1 个预测点), 故 sample_tids 逐点重复 n_points/n_samples 次。
    Subset (smoke 模式) 通过 .indices 转发到底层 dataset。
    """
    ds = getattr(loader, "dataset", None)
    idx = None
    while ds is not None and not hasattr(ds, "sample_tids"):
        if hasattr(ds, "indices") and hasattr(ds, "dataset"):
            idx = ds.indices if idx is None else [ds.indices[i] for i in idx]
            ds = ds.dataset
        else:
            ds = None
    if ds is None:
        return np.zeros(n_points, dtype=np.int64), False
    st = np.asarray(ds.sample_tids, dtype=np.int64)
    if idx is not None:
        st = st[np.asarray(idx, dtype=np.int64)]
    if len(st) == 0:
        return np.zeros(n_points, dtype=np.int64), False
    rep = max(1, n_points // len(st))
    st = np.repeat(st, rep)
    if len(st) < n_points:              # 长度不整除时补齐 (不应发生, 兜底防越界)
        st = np.concatenate([st, np.full(n_points - len(st), st[-1], dtype=np.int64)])
    return st[:n_points], True


def _train_with_early_stop(model, ltr, lva, opt, device, huber, mse, lam, e, tag,
                           use_mmd=False, src_iter=None, bins=None, mmd_lambda=1.0,
                           post_eol_weight=1.0, huber_delta=1.0,
                           capped_weight=1.0, cap_eps=CAP_EPS,
                           trace=None, early_stop_metric=None,
                           early_stop_patience=0, history=None,
                           censor_eta=None, budget=None, rc=None):
    """训练至多 e epoch, 基于 **val** 指标 early-stop (恢复最佳模型)。

    防末段发散/过拟合 — 如 S3 MMD 某些 seed 末段单调上升→坍缩 (诊断 seed43: 0.17→0.34)。

    S2' 改动 3: 早停信号由 full 口径改为 info 口径 (0<RUL<1)。原因 (S1.5 实测):
    full 口径下 65%~95.6% 的样本 RUL 恒为 0, val RMSE 主要由这些零难度样本决定,
    与 test 上的真实排序反相关 (B: val 0.0928/test 0.1373, C: val 0.0853/test 0.1525),
    早停因此选到错误的 epoch。

    S2.5 §6: 早停指标改为可配 (training.early_stop_metric), 并支持 patience 提前终止。
    **本函数签名里没有 test loader** —— test 在结构上无法参与 checkpoint 选择。
      early_stop_metric: EARLY_STOP_METRICS 的键; None → 沿用 info/pooled (S2' 行为)
      early_stop_patience: 连续多少 epoch 未刷新最优则 break; 0 = 跑满 e (S2' 行为)
      history: 可选 list, 逐 epoch 追加 {epoch, train_loss, val_full_rmse,
               val_info_pooled_rmse, val_info_macro_rmse, es_value, is_best}
               —— 供 S2.5 过拟合诊断; 全部是 **val** 量, 不含 test。

    trace: 可选 callable(ep, val_metrics, model) —— 每 epoch 评完 val 后回调,
           供 scripts/diag_earlystop.py 逐 epoch 记录 test 指标 (S2' 任务 3)。
           训练本身不看它的返回值, 不影响早停决策。

    S3 (协议 §8/§12/§14):
      censor_eta: 非 None 时 train loader 需产出 5 元组, 启用删失损失 (观测 Huber +
                  eta·hinge)。val 评估完全不变 —— val 未截断, 一律走精确 RUL 口径。
      budget    : 可选 dict, 累加 target adaptation 的 steps/batches/best_epoch,
                  供三方法预算对齐审计。
      **本函数签名里仍然没有 test loader**, S3 早停路径结构上拿不到 test。

    S5 (改动 1/2):
      rc: rate_cfg(cfg)。给出时 train 走速率损失、val 走首达换算后的 RUL 评估 ——
          **早停指标名与口径完全不变** (仍是 val info macro RMSE), 故"rate 组"与
          "direct_rul 组"的模型选择规则严格同一条。rc=None 时逐行等于 S3 行为。
    """
    metric_name = early_stop_metric or f"{PRIMARY_CALIBER}_{PRIMARY_STAT}_rmse"
    best = (float("inf"), None)
    best_ep, bad = -1, 0
    for ep in range(e):
        tl = _train_epoch(model, ltr, opt, device, huber, mse, lam,
                          use_mmd=use_mmd, src_iter=src_iter, bins=bins,
                          mmd_lambda=mmd_lambda, post_eol_weight=post_eol_weight,
                          huber_delta=huber_delta, capped_weight=capped_weight,
                          cap_eps=cap_eps, censor_eta=censor_eta, budget=budget,
                          rc=rc)
        vm = eval_test(model, lva, device, cap_eps=cap_eps, rc=rc)
        if trace is not None:
            trace(ep, vm, model)
        cur = early_stop_value(vm, metric_name)
        improved = cur < best[0]
        if improved:
            best = (cur, {k: v.detach().clone() for k, v in model.state_dict().items()})
            best_ep, bad = ep, 0
        else:
            bad += 1
        if history is not None:
            _c = vm["calibers"]
            history.append({
                "epoch": ep, "train_loss": float(tl),
                "val_full_rmse": float(_c["full"]["pooled"]["rmse"]),
                "val_info_pooled_rmse": float(_c["info"]["pooled"]["rmse"]),
                "val_info_macro_rmse": float(_c["info"]["macro"]["rmse"]),
                "es_value": float(cur), "is_best": bool(improved)})
        if int(early_stop_patience) > 0 and bad >= int(early_stop_patience):
            print(f"    [{tag}] patience={early_stop_patience} 触发, ep{ep} 提前终止")
            break
    if best[1] is not None:
        model.load_state_dict(best[1])
    if budget is not None:
        budget["best_epoch"] = int(best_ep)
        budget["epochs_run"] = budget.get("epochs_run", 0) + (ep + 1 if e > 0 else 0)
    print(f"    [{tag}] best val_{metric_name}={best[0]:.4f} @ep{best_ep}")
    return best[0]



def run_one_group(mode, seed, cfg, smoke=False, encoder_override=None,
                  component="wheel", group_name=None):
    set_seed(seed, cfg["reproducibility"]["deterministic"], cfg["reproducibility"]["cudnn_benchmark"])
    L = int(cfg["model"]["input_len_L"])
    K = int(cfg.get("pretrain", {}).get("seq_block_K", 8))
    tc = cfg["transfer"]
    mc = cfg["model"]
    bins = [tuple(b) for b in tc["hi_bins"]]
    _lc = cfg["loss"]
    lam = (float(_lc.get("lambda_hi", _lc.get("beta_hi", 1.0))),       # 飞轮 lambda_hi / 相控阵 beta_hi
           float(_lc.get("lambda_mono", _lc.get("mu_mono", 0.1))),     # lambda_mono / mu_mono
           float(_lc.get("lambda_smooth", _lc.get("nu_smooth", 0.1)))) # lambda_smooth / nu_smooth
    mmd_lambda = float(tc["mmd_lambda"])
    enc = encoder_override or mc["encoder"]
    # checkpoint 组件后缀: 飞轮 source_tcn / 相控阵 source_phased_array_tcn
    suffix = "" if component == "wheel" else f"_{component}"

    has_nodes = bool(tc.get("target_has_nodes", False))
    xT, hiT, rulT, tidT, n_traj = load_target(
        ROOT / tc.get("target_feature_path", "data/features/wheel/schema_v1/target_features.h5"),
        has_nodes, observed_only=True)
    featsS, hiS, bidS, tidxS = load_source(
        ROOT / cfg["pretrain"].get("source_feature_path", "data/features/wheel/schema_v1/source_features.h5"),
        cfg.get("pretrain", {}).get("source_id_field", "bearing_id"),
        (cfg.get("source", {}).get("split", {}).get("val_device_ids") or
         cfg.get("source", {}).get("split", {}).get("val_bearing_ids") or []),
        train_only=True)
    tr, va, te, eol_by_tid = split_trajectories_from_cfg(
        ROOT / tc.get("target_feature_path", "data/features/wheel/schema_v1/target_features.h5"),
        n_traj, tc, seed, observed_only=True)
    print(f">> 轨迹划分 train {len(tr)} / val {len(va)} / test {len(te)}")
    describe_eol_split(eol_by_tid, tr, va, te)
    rul_scale = max(float(np.nanmax(rulT[np.isin(tidT, tr)])), 1.0)
    rulT = rulT / rul_scale

    # 目标域 xT z-score 归一 (train 集) — encoder 预训练权重期望归一输入 (与 pretrain/transfer 一致)
    _tr_mask = np.isin(tidT, tr)
    _fm = xT[_tr_mask].mean(axis=0)
    _fs = xT[_tr_mask].std(axis=0) + 1e-6
    xT = (xT - _fm) / _fs

    def mask(ids):
        return np.isin(tidT, ids)

    tstride = 1000 if smoke else int(tc.get("target_stride", 50))

    def mkDS(ids, block_k):
        ds = TargetSeqDataset(xT[mask(ids)], hiT[mask(ids)], rulT[mask(ids)],
                              tidT[mask(ids)], L, block_k, stride=tstride)
        if smoke:
            ds = Subset(ds, list(range(min(32, len(ds)))))
        return ds

    bs = 16 if smoke else int(cfg["pretrain"]["batch_size"])
    ltr = DataLoader(mkDS(tr, K), batch_size=bs, shuffle=True)
    lva = DataLoader(mkDS(va, 1), batch_size=bs, shuffle=False)
    lte = DataLoader(mkDS(te, 1), batch_size=bs, shuffle=False)
    dsS = SourceWindowDataset(featsS, hiS, bidS, tidxS, L, stride=max(1, len(featsS) // 2000))
    if smoke:
        dsS = Subset(dsS, list(range(min(64, len(dsS)))))
    lS = DataLoader(dsS, batch_size=bs, shuffle=True)
    device = "cuda" if (cfg["pretrain"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
    model = _build_model(cfg, featsS.shape[1], xT.shape[1], device, encoder_override=encoder_override)
    huber = nn.HuberLoss(delta=float(cfg["loss"]["huber_delta"]))
    mse = nn.MSELoss()
    e = 2 if smoke else int(tc.get("epochs_s2", 20))
    # S2' 改动 4 + 任务 2: 两段零信息量样本降权 (不硬删); 缺省 1.0 = 与改动前数值等价
    hd = float(cfg["loss"]["huber_delta"])
    _tcfg = cfg.get("training", {})
    pw = float(_tcfg.get("post_eol_weight", 1.0))
    cw = float(_tcfg.get("capped_weight", 1.0))
    ce = caliber_eps(cfg)
    # S2.5: 训练时长 / 早停口径 / patience / weight_decay 全部走 config
    th = training_hyper(cfg)
    if not smoke:
        e = th["max_epochs"]
    kw = dict(post_eol_weight=pw, huber_delta=hd, capped_weight=cw, cap_eps=ce,
              early_stop_metric=th["early_stop_metric"],
              early_stop_patience=th["early_stop_patience"])
    wd = th["weight_decay"]
    print(f">> 训练权重 post_eol_weight={pw} (RUL==0) / capped_weight={cw} "
          f"(RUL>=1-{ce:g}), huber_delta={hd}")
    print(f">> 训练时长/早停 max_epochs={e} early_stop_metric={th['early_stop_metric']} "
          f"patience={th['early_stop_patience']} weight_decay={wd}")

    # target_only 不加载 checkpoint (随机初始化, 支持 GRU 等 override encoder)
    if mode != "target_only":
        ckpt = str(CKPT_DIR / f"source{suffix}_{mc['encoder']}_pretrain.pt")
        if smoke and not Path(ckpt).exists():
            ckpt = str(CKPT_DIR / f"source{suffix}_{mc['encoder']}_smoke.pt")
        if Path(ckpt).exists():
            model.load_pretrained(ckpt, device)
        elif not smoke:
            raise FileNotFoundError(f"正式实验缺少 source checkpoint: {ckpt}")

    tag = f"{group_name or mode} seed{seed}"
    if mode == "source_frozen":
        model.freeze_encoder(True)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=float(tc["finetune_lr"]), weight_decay=wd)
        _train_with_early_stop(model, ltr, lva, opt, device, huber, mse, lam, e, tag, **kw)
    elif mode == "target_only":
        model.freeze_encoder(False)
        opt = torch.optim.Adam(model.parameters(), lr=float(tc["finetune_lr"]),
                               weight_decay=wd)
        _train_with_early_stop(model, ltr, lva, opt, device, huber, mse, lam, e, tag, **kw)
    elif mode == "source_finetune":
        model.freeze_encoder(False)
        opt = torch.optim.Adam(model.parameters(), lr=float(tc["finetune_lr"]),
                               weight_decay=wd)
        _train_with_early_stop(model, ltr, lva, opt, device, huber, mse, lam, e, tag, **kw)
    else:  # source_mmd_finetune (source_mmd_physics 内部 mode; 物理一致性损失 ρ·L_phys 待正式阶段)
        model.freeze_encoder(True)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=float(tc["finetune_lr"]), weight_decay=wd)
        _train_with_early_stop(model, ltr, lva, opt, device, huber, mse, lam, e, f"{tag} S2", **kw)
        reset_global_memory_bank()
        model.freeze_encoder(False)
        opt = torch.optim.Adam(model.parameters(), lr=float(tc["finetune_lr"]),
                               weight_decay=wd)
        _train_with_early_stop(model, ltr, lva, opt, device, huber, mse, lam, e, f"{tag} S3",
                               use_mmd=True, src_iter=cycle(lS), bins=bins,
                               mmd_lambda=mmd_lambda, **kw)

    m = eval_test(model, lte, device, cap_eps=ce)
    print(f"  [{tag}] test 三档口径:")
    print(format_calibers(m))
    m["mode"] = mode
    m["group"] = group_name or mode
    m["seed"] = seed
    m["encoder"] = enc
    m["rul_scale_train"] = rul_scale
    return m


def aggregate(by):
    agg = {}
    for group, ms in by.items():
        if not ms:
            continue
        agg[group] = {
            "rmse_mean": statistics.mean(m["rmse"] for m in ms),
            "rmse_std": statistics.pstdev(m["rmse"] for m in ms) if len(ms) > 1 else 0.0,
            "phm_mean": statistics.mean(m["phm"] for m in ms),
            "mae_mean": statistics.mean(m["mae"] for m in ms),
            "n": len(ms),
            "rmse_by_seed": {str(m["seed"]): m["rmse"] for m in ms},
        }
    return agg


def _paired_bootstrap_gain(target, transfer, n_boot=10000, seed=20260723):
    """同种子配对的 target RMSE - transfer RMSE 均值与 95% CI。"""
    common = sorted(set(target["rmse_by_seed"]) & set(transfer["rmse_by_seed"]))
    if len(common) < 5:
        return None
    d = np.asarray([target["rmse_by_seed"][s] - transfer["rmse_by_seed"][s]
                    for s in common], dtype=float)
    draws = np.random.default_rng(seed).choice(
        d, size=(int(n_boot), len(d)), replace=True).mean(axis=1)
    return float(d.mean()), [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def write_results(agg, physical, path, smoke, n_seeds, group_names, component):
    comp_label = "飞轮" if component == "wheel" else "相控阵天线"
    note = ("**注意: 基于 smoke/合成数据, 非最终数字; 正式迁移增益待真实数据预训练**\n"
            if smoke else "")
    lines = [
        f"# 对比实验结果 — {comp_label} (plan Phase 6 / §8)\n\n",
        note,
        f"每组 {n_seeds} 个随机种子, 报 mean±std (RMSE/nPHM/MAE 在归一化 RUL 空间)。\n\n",
        "| 实验组 | RMSE (mean±std) | nPHM | MAE |\n",
        "|--------|-----------------|-----------|-----|\n",
    ]
    for g in group_names:
        label = LABELS.get(g, g)
        if g in agg:
            a = agg[g]
            lines.append(f"| {label} | {a['rmse_mean']:.4f}±{a['rmse_std']:.4f} | "
                         f"{a['phm_mean']:.2f} | {a['mae_mean']:.4f} |\n")
        elif g in TIMESFM_GROUPS:
            lines.append(f"| {label} | — | — | — | *(PA7 待启用)* |\n")
        elif g == "physical_extrap":
            if physical:
                lines.append(f"| 物理外推基线 | {physical['rmse']:.4f} | {physical['phm']:.2f} | "
                             f"{physical['mae']:.4f} |\n")
            else:
                lines.append(f"| 物理外推基线 | — | — | — | *(待实现)* |\n")
    lines.append("\n**验收**:\n")
    # 验收: 最优迁移组 (source_* RMSE 最小) vs 同架构 target_only_tcn (迁移组均 TCN 编码器)
    src_in_agg = [g for g in group_names if g.startswith("source_") and g in agg]
    main_g = min(src_in_agg, key=lambda g: agg[g]["rmse_mean"]) if src_in_agg else None
    tgt_g = next((g for g in group_names if g == "target_only_tcn"), None) \
            or next((g for g in group_names if g.startswith("target_only")), None)
    if main_g and tgt_g in agg:
        stat = _paired_bootstrap_gain(agg[tgt_g], agg[main_g])
        if stat is None:
            lines.append("- 种子不足 5 个，不能下显著性结论。\n")
        else:
            gain, ci = stat
            relative = gain / max(agg[tgt_g]["rmse_mean"], 1e-12)
            passed = ci[0] > 0 and relative >= 0.05
            lines.append(f"- 配对增益 {gain:+.4f}, bootstrap 95% CI "
                         f"[{ci[0]:+.4f}, {ci[1]:+.4f}], 相对增益 {relative:.1%} → "
                         f"{'通过' if passed else '未通过'}\n")
    for g in src_in_agg:        # 各迁移组明细 (source_mmd_physics std>0.05 标不稳定)
        a = agg[g]
        note = " ⚠ std 大 (MMD 不稳定, 待调 mmd_lambda/数值稳定性)" if a["rmse_std"] > 0.05 else ""
        lines.append(f"  - {g}: {a['rmse_mean']:.4f}±{a['rmse_std']:.4f}{note}\n")
    if main_g and physical:
        better = agg[main_g]["rmse_mean"] < physical["rmse"]
        lines.append(f"- 主模型 vs 物理基线 RMSE: {'优于' if better else '待正式数据'} "
                     f"({agg[main_g]['rmse_mean']:.4f} vs {physical['rmse']:.4f})\n")
    Path(path).write_text("".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel.yaml")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    component = Path(args.config).stem
    n_seed = 1 if args.smoke else args.seeds
    group_names = cfg.get("experiments", {}).get("groups",
                   ["target_only", "source_frozen", "source_finetune", "source_mmd_finetune"])

    by = {}
    failures = []
    for gname in group_names:
        if gname == "physical_extrap":
            continue                              # 物理基线单独 evaluate (非学习组)
        mode, enc_ov, enabled, reason = _resolve_group(gname, cfg)
        by[gname] = []
        if not enabled:
            print(f">> {gname:28s}: skip ({reason})")
            continue
        for s in range(n_seed):
            seed = cfg["seed"] + s
            try:
                m = run_one_group(mode, seed, cfg, smoke=args.smoke,
                                  encoder_override=enc_ov, component=component,
                                  group_name=gname)
                by[gname].append(m)
                print(f">> {gname:28s} seed{seed} ({m['encoder']}): "
                      f"RMSE={m['rmse']:.4f} PHM={m['phm']:.2f} MAE={m['mae']:.4f}")
            except Exception as exc:    # noqa: BLE001
                print(f"!! {gname} seed{seed} 失败: {exc}")
                failures.append(f"{gname}/seed{seed}: {exc}")

    if failures:
        raise RuntimeError("对比实验存在失败，不写部分结果：" + "; ".join(failures))
    physical = evaluate_physical(cfg, cfg["seed"]) if "physical_extrap" in group_names else None
    agg = aggregate(by)
    write_results(agg, physical, ROOT / "docs" / "results.md", args.smoke, n_seed,
                  group_names, component)
    CKPT_DIR.mkdir(exist_ok=True)
    (CKPT_DIR / f"all_metrics_{component}.json").write_text(
        json.dumps({"agg": agg, "physical": physical, "groups": group_names},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(">> results -> docs/results.md")
    print(f">> all_metrics -> {CKPT_DIR / f'all_metrics_{component}.json'}")


if __name__ == "__main__":
    main()
