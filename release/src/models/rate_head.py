"""models/rate_head.py — S5 退化速率头 + 物理首达换算层

背景 (S4 实测根因, 不在此重新论证):
  L=64 × 1800s = 32 小时窗口, 而退化横跨 3 年 ⇒ 窗口内 ΔHI ≈ 0。模型无法从窗口
  看到趋势, 只能靠特征绝对水平反推 RUL, 而绝对水平被逐轨迹 b0 (10 倍跨度) /
  omega0 / Kt 打散 ⇒ 不可辨识 ⇒ 输出坍缩为条件均值 (PSR 0.08, stages_above=[])。

S5 改动 1: 把预测目标从 RUL 改成**退化速率**, RUL 由物理首达关系换算出来。

    (mu_hat, sigma_hat) = f(x_window)              网络只需回答"现在退化多快"
    RUL_hat   = (HI_fail - HI_t) / mu_hat          首达换算 (确定性, 无参数)
    Var[RUL]  = (HI_fail - HI_t) * sigma_hat^2 / mu_hat^3

为什么这能绕开根因: 速率是**窗口内局部可观测量** (HI 的局部斜率), 不依赖跨 3 年的
绝对水平; 而"还剩多少 HI 余量" (HI_fail - HI_t) 由自校准 HI 直接给出 —— 它是长基线
量, 但不需要网络去推断。网络负责的部分与它能看到的部分终于对齐了。

**本模块是 S5 的唯一首达换算实现**: Wiener + 粒子滤波基线必须复用 `first_passage_rul`
与 `first_passage_var`, 否则"模型 vs 物理基线"的比较里换算口径会分叉 (S5 §改动 1)。
函数刻意写成**纯函数 + 支持 numpy/torch 双入**, 让不带 autograd 的基线也能直接调。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 首达换算 (纯函数, 模型侧与物理基线侧共用)
# ============================================================

def first_passage_rul(hi_t, mu, hi_fail: float = 1.0, eps_mu: float = 1e-2,
                      rul_max: float | None = None):
    """物理首达换算: RUL = (HI_fail - HI_t) / mu。

    hi_t : 当前 (窗末) 健康指标, 与 mu 同 shape 或可广播。**必须是自校准 HI**
           (src/sim/build_hi.py 的 hi_b, 只用 Kt_hat, 不含仿真真值)。
    mu   : 退化速率, 单位 = ΔHI / 归一 RUL 单位 (即 dHI/dt · rul_scale)。
           调用方应已保正; 本函数再夹一次 eps_mu 下界作防御 (mu→0 时 RUL→∞)。
    hi_fail : 失效阈值, HI 定义已归一到 1.0 = 失效 (config baselines.hi_target)。
    rul_max : 可选上限 (归一 RUL 口径)。None = 不截顶。

    余量 (HI_fail - HI_t) 夹到 >= 0: HI 单调非减且可达 1.0, 超阈后余量为负会给出
    负 RUL —— 物理上该点已失效, RUL=0 才是正确语义。
    """
    is_t = isinstance(mu, torch.Tensor) or isinstance(hi_t, torch.Tensor)
    if is_t:
        hi_t = hi_t if isinstance(hi_t, torch.Tensor) else torch.as_tensor(hi_t)
        mu = mu if isinstance(mu, torch.Tensor) else torch.as_tensor(mu)
        rem = torch.clamp(float(hi_fail) - hi_t, min=0.0)
        r = rem / torch.clamp(mu, min=float(eps_mu))
        return r if rul_max is None else torch.clamp(r, max=float(rul_max))
    rem = np.clip(float(hi_fail) - np.asarray(hi_t, dtype=float), 0.0, None)
    r = rem / np.maximum(np.asarray(mu, dtype=float), float(eps_mu))
    return r if rul_max is None else np.minimum(r, float(rul_max))


def first_passage_var(hi_t, mu, sigma, hi_fail: float = 1.0,
                      eps_mu: float = 1e-2):
    """首达时间方差 (Wiener 首达一阶展开): Var[RUL] = (HI_fail-HI_t)·sigma^2 / mu^3。

    量纲自检: [余量] · [ΔHI/单位时间]^2 / [ΔHI/单位时间]^3 = [单位时间] · [余量]/[ΔHI]
    余量与 ΔHI 同量纲 ⇒ 结果是 [单位时间]^2 的一次幂 · 余量比 —— 与 Wiener 过程
    首达时间方差 σ²·d/μ³ 完全同式 (d = 漂移余量)。
    """
    is_t = any(isinstance(v, torch.Tensor) for v in (hi_t, mu, sigma))
    if is_t:
        hi_t = hi_t if isinstance(hi_t, torch.Tensor) else torch.as_tensor(hi_t)
        mu = mu if isinstance(mu, torch.Tensor) else torch.as_tensor(mu)
        sigma = sigma if isinstance(sigma, torch.Tensor) else torch.as_tensor(sigma)
        rem = torch.clamp(float(hi_fail) - hi_t, min=0.0)
        m = torch.clamp(mu, min=float(eps_mu))
        return rem * sigma * sigma / (m * m * m)
    rem = np.clip(float(hi_fail) - np.asarray(hi_t, dtype=float), 0.0, None)
    m = np.maximum(np.asarray(mu, dtype=float), float(eps_mu))
    s = np.asarray(sigma, dtype=float)
    return rem * s * s / (m ** 3)


def gaussian_interval(rul, var, z: float = 1.645):
    """由 (RUL, Var) 给出对称区间。z=1.645 ⇒ 90% 双侧覆盖 (标准正态 5%/95% 分位)。

    下界夹到 >= 0 (RUL 非负)。返回 (lo, hi)。
    """
    is_t = isinstance(rul, torch.Tensor)
    if is_t:
        s = torch.sqrt(torch.clamp(var, min=0.0))
        return torch.clamp(rul - float(z) * s, min=0.0), rul + float(z) * s
    s = np.sqrt(np.clip(var, 0.0, None))
    return np.clip(rul - float(z) * s, 0.0, None), rul + float(z) * s


def gaussian_nll(target, mean, var, var_floor: float = 1e-6):
    """高斯负对数似然 (常数项省略): 0.5·[log var + (y-mu)^2 / var]。

    var_floor 防 var→0 时 log 爆炸 / 梯度奇异。numpy / torch 双入 (与本模块其余
    函数同规格 —— 物理基线要算自己的 NLL 时不必再写一份)。返回逐元素值。
    """
    if any(isinstance(v, torch.Tensor) for v in (target, mean, var)):
        target, mean, var = (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
                             for v in (target, mean, var))
        v = torch.clamp(var, min=float(var_floor))
        return 0.5 * (torch.log(v) + (target - mean) ** 2 / v)
    v = np.maximum(np.asarray(var, dtype=float), float(var_floor))
    return 0.5 * (np.log(v) + (np.asarray(target, dtype=float)
                               - np.asarray(mean, dtype=float)) ** 2 / v)


# ============================================================
# 速率头
# ============================================================

class RateHead(nn.Module):
    """latent z → (mu_hat, sigma_hat), 两者都经 softplus 保正并设下界。

    mu 下界 eps_mu 的必要性: RUL = 余量/mu, mu→0 时 RUL→∞, 损失与梯度双双爆炸。
    softplus 只保证 > 0 而不保证离 0 有距离, 故必须显式加下界 (S5 §改动 1)。
    sigma 同理设 eps_sigma —— 它进 NLL 的 log, 贴 0 会让 NLL → -∞ (退化解:
    把 sigma 压到 0 换取无穷似然)。

    与源域 RULModel / TransferModel 的 hi_head / rul_head 并列, 命名独立
    (`rate_head.*`), 故加载源域 checkpoint 时它保持随机初始化, 不会静默错配。
    """

    def __init__(self, latent_dim: int = 64, hidden: int = 0,
                 eps_mu: float = 1e-2, eps_sigma: float = 1e-3):
        super().__init__()
        self.eps_mu = float(eps_mu)
        self.eps_sigma = float(eps_sigma)
        if int(hidden) > 0:
            self.net = nn.Sequential(
                nn.Linear(latent_dim, int(hidden)), nn.ReLU(),
                nn.Linear(int(hidden), 2))
        else:
            self.net = nn.Linear(latent_dim, 2)

    def forward(self, z: torch.Tensor):
        """z: (B, latent) → (mu (B,), sigma (B,)), 均 > 下界。"""
        o = self.net(z)
        mu = F.softplus(o[..., 0]) + self.eps_mu
        sigma = F.softplus(o[..., 1]) + self.eps_sigma
        return mu, sigma
