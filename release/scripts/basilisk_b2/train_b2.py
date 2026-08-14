#!/usr/bin/env python
"""scripts/basilisk_b2/train_b2.py

BASILISK-B2 §5/§7 —— target_only 训练 (censor-aware)。

与 diag_generalization.train_candidate 的唯一差别: train loader 产出 5 元组,
因此把 `censor_eta` 传给 _train_with_early_stop, 启用 §4 的删失损失
(观测项 Huber + eta·one-sided hinge)。其余逐行相同。

**训练循环 / 损失 / 早停 / 模型全部 import 复用**, 本模块只做装配。

§7 冻结超参 (S2.5 已验证, 本阶段不重新调参):
    early_stop_metric = info_macro_rmse   (val)
    max_epochs = 8, patience = 2, weight_decay = 1e-3
§8 条件 7: checkpoint 只按 val 选 —— _train_with_early_stop 的签名里没有
    test loader, 这是结构性保证, 不是口头承诺。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.run_groups import (  # noqa: E402
    _build_model,
    _train_with_early_stop,
    training_hyper,
)

# §6/§7: 本阶段唯一允许训练的模式。source 系一律不在此列 (那是 B4)。
ALLOWED_MODES = ("target_only",)


def train_target_only(cfg: dict, data: dict, seed: int, tag: str,
                      mode: str = "target_only"):
    """训一次 target_only, 返回 (model, history, hyper, budget)。"""
    if mode not in ALLOWED_MODES:
        raise ValueError(f"B2 §6 只允许 {ALLOWED_MODES}, 收到 {mode!r}")

    th = training_hyper(cfg)
    tcfg = cfg.get("training", {})
    # §7: 冻结超参必须与 config 里记录的期望值逐项一致, 否则立刻暴露
    exp = cfg["b2_frozen_training_expected"]
    for k, want in (("early_stop_metric", exp["early_stop_metric"]),
                    ("max_epochs", int(exp["max_epochs"])),
                    ("early_stop_patience", int(exp["early_stop_patience"])),
                    ("weight_decay", float(exp["weight_decay"]))):
        if th[k] != want:
            raise SystemExit(
                f"!! B2_CONFIG_INCOMPATIBLE: training.{k}={th[k]!r} "
                f"!= S2.5 冻结值 {want!r} —— §7 禁止在本阶段改超参")

    model = _build_model(cfg, data["n_target"], data["n_target"], data["device"])
    model.freeze_encoder(False)
    opt = torch.optim.Adam(model.parameters(),
                           lr=float(cfg["transfer"]["finetune_lr"]),
                           weight_decay=th["weight_decay"])
    huber = nn.HuberLoss(delta=float(cfg["loss"]["huber_delta"]))
    mse = nn.MSELoss()
    lam = (float(cfg["loss"]["lambda_hi"]),
           float(cfg["loss"]["lambda_mono"]),
           float(cfg["loss"]["lambda_smooth"]))
    history: list = []
    budget: dict = {}

    best_val = _train_with_early_stop(
        model, data["ltr"], data["lva"], opt, data["device"], huber, mse, lam,
        th["max_epochs"], tag,
        post_eol_weight=float(tcfg.get("post_eol_weight", 1.0)),
        huber_delta=float(cfg["loss"]["huber_delta"]),
        capped_weight=float(tcfg.get("capped_weight", 1.0)),
        cap_eps=data["cap_eps"],
        early_stop_metric=th["early_stop_metric"],
        early_stop_patience=th["early_stop_patience"],
        history=history,
        # §4: train loader 是 5 元组, 启用删失损失
        censor_eta=float(data["censor_eta"]),
        budget=budget)
    return model, history, th, {**budget, "best_val_es": float(best_val)}
