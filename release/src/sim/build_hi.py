"""sim/build_hi.py

飞轮 HI 构造 + 目标域特征 (plan §四 Phase 4)。

方案 A (工程版, 直观但低速/换向不稳):
  HI_A = |I_m| / Im_rated
方案 B (监督 HI, 写入 h5 的 `hi_b`): 由 config sim.failure.hi_source 选口径 ——
  "Tf" (S2' 起默认, **与失效判据同源**) / "b" (S1 旧口径, 仅供回退对比)。

  ---- hi_source="Tf": HI 挂总摩擦力矩 ----
  总摩擦力矩 T_f = T_c·sgn(ω) + b(t)·ω, 由力矩平衡 K_t·I_m = T_cmd + T_f 得
      T̂_f(t)  = K̂_t·mean_w(I_m) - mean_w(T_cmd)
      T_f,fail = K̂_t·I_m,rated  - mean_w(T_cmd)
      HI = cummax(clip((T̂_f - T̂_f,0) / (T_f,fail - T̂_f,0), 0, 1))
  只依赖 K̂_t (健康段自校准, 实测偏差 ~0.01%); **禁止 T̂_c / b̂_0 进入本路径** ——
  S1 已证 T_c 与 b_0·ω 的拆分天然病态, 引入即把可辨识性噪声灌进监督标签。

  为何这样能与失效判据同源: 分子分母的 mean_w(T_cmd) 同口径, K̂_t 为公因子, 故
      HI ≥ 1  ⟺  mean_w(I_m) ≥ I_m,rated
  而失效判据是 |I_m| > I_m,rated 连续 persistence_samples 点。取 Tf_window=1 +
  同一 persistence 判定, 二者**严格同源**, HI 首达 1.0 与 EOL 完全重合。
  实测 (100 轨迹, seed=42): 偏移恒为 0, HI 达 1.0 的轨迹数 == 失效轨迹数 60。
  对比: 若阈值改用标称 K_t,nom (Tf_fail_Kt_source="nom"), K̂_t 不再约掉,
  偏移最大达 49% EOL、达标轨迹数 63≠60 —— 故默认 "hat"。

  ---- hi_source="b": S1 旧口径 (黏性摩擦归一) ----
  滑窗 LSQ:  b̂ = Σ(y·ω) / Σ(ω²),  y = K_t·I_m - T_cmd - T_c·sgn(ω)
  HI = cummax(clip((b̂ - b̂_0)/(b_fail - b̂_0), 0, 1))
  S1.5 实测该口径与失效判据不同源 (差一个 T_c 项与 T_cmd 项): 一致组内 HI 首达 1.0
  相对 EOL 偏移 p5=-15540/p95=+15076 (截顶值的 82%), 9/100 条轨迹交叉表不一致。
  保留仅为回退对比, 不作默认。

注意: b̂ 本身**不因此下线** —— 仍在 x_T 第 6 维、仍写 h5、仍作 physical_extrap 输入,
接口不变。改的只是"监督 HI 挂在哪个物理量上"。

**自校准 (S1 起, 动机是在轨可行性)**:
  在轨没有"该飞轮的真实 K_t / T_c / b_0", 只有遥测。故本模块**禁止**从 h5 attrs
  读真值 Kt / Tc / b0 / omega0 进入特征 / HI / RUL 任何路径。改为:
    用每条轨迹前 healthy_frac 的健康段 (此时 b ≈ b_0 近似常值) 对
        K_t·I_m - T_c·sgn(ω) - b_0·ω = T_cmd
    做联合最小二乘, 一次估出 (K_t_hat, T_c_hat, b_0_hat), 之后全程只用估计值。
  HI 归一改用**全局固定阈值** (查器件手册即得, 不依赖单机真值):
        b_fail = (K_t_nom·I_m,rated - T_c_nom) / ω_ref
  b_true 仍写入 h5 供离线校核, 但不进入 x_T / HI / RUL。
  唯一读真值的地方是 main() 里 `--report` 的审计块 (仅用于打印估计偏差), 已隔离。

目标域特征 x_T (10维, 见 XT_COLS):
  [I_m, ω, T, T_cmd, σ(I_m), b̂, ΔT, |ω-ω_cmd|, Tf_ratio, Tf_slope]
  第 5 维 S1 起由 I_m/ω 改为 σ(I_m) (I_m 滑窗标准差): 原 I_m/ω 与 b̂ 信息近似等价
  (同属 b≈(K_t·I_m-T_cmd-T_c·sgn)/ω 这一条关系), 冗余; σ(I_m) 提供独立的
  波动度信息 (轴承磨损导致摩擦力矩抖动加剧)。
  第 9/10 维 S5 新增 (改动 3): 长基线趋势。前 8 维全是**窗口内瞬时量**, 而 L=64 窗
  只覆盖 32 小时、退化横跨 3 年 —— 窗口里根本没有趋势可看 (S4 实测 ΔHI≈0)。
  Tf_ratio = T̂_f/T̂_f,0 (逐轨迹自归一, 消掉 b0 的 10 倍量级跨度),
  Tf_slope = T̂_f 的长窗因果 LSQ 斜率 (窗长 sim.hi.trend_window >> L)。
  两列都只依赖 K̂_t + 遥测, 不含仿真真值。既有 8 维一列未删。

用法:
  python -m src.sim.build_hi --config configs/wheel.yaml --report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import load_config            # noqa: E402

XT_COLS = ["I_m", "omega", "T", "T_cmd", "sigma_Im", "b_hat", "dT", "omega_err",
           # ---- S5 改动 3: 长基线趋势特征 (第 9/10 维) ----
           # 根因是 L=64 窗口 (32 小时) 看不到 3 年尺度的退化趋势。窗口长度不能改
           # (禁止动 encoder/L), 故把"长基线趋势"预先算成特征喂进去 —— 窗口里每一
           # 时刻都带着它自己那一刻的长窗趋势读数。
           "Tf_ratio",     # T̂_f(t) / T̂_f,0  相对自校准健康基线, 无量纲 -> 跨轨迹可比
           "Tf_slope"]     # T̂_f 的长窗因果 LSQ 斜率 (窗长 >> L, 走 config)

# build_features 允许使用的遥测列 (不含任何仿真真值参数)
TELEMETRY_COLS = ["t", "omega", "omega_cmd", "I_m", "T", "T_cmd"]


def calibrate_healthy(Im, tcmd, omega_cmd, healthy_frac: float):
    """健康段联合最小二乘自校准 -> (Kt_hat, Tc_hat, b0_hat)。

    健康段内 b(t) ≈ b_0 (常值), 理论关系 K_t·I_m = T_cmd + T_c·sgn(ω) + b_0·ω。

    **回归方向**: 把带量测噪声的 I_m 放在**因变量**侧, 自变量只用星上干净指令量
    (T_cmd / sgn(ω_cmd) / ω_cmd, 均为解析指令, 在轨可知, 非仿真真值):
        I_m = a1·T_cmd + a2·sgn(ω_cmd) + a3·ω_cmd
        a1 = 1/K_t,  a2 = T_c/K_t,  a3 = b_0/K_t
    若反过来把带噪的 I_m / ω 当自变量, OLS 会因 errors-in-variables 系统性衰减系数
    (实测 K_t 偏 -75%, b_0 偏 -82%), 故必须保持此方向。

    三列量级差 6 个数量级, 先按列归一再 lstsq 以改善条件数。

    **可辨识性说明**: ω_cmd > 0 恒成立 -> sgn(ω_cmd) ≡ 1 是常列, 而 ω_cmd 本身是
    "常值 + 2% 正弦", 二者近似共线。故 T_c 与 b_0·ω 的**和**可精确辨识 (这正是 b̂ 所需),
    但二者的**拆分**天然病态 —— 报告里 Tc_hat / b0_hat 离散大是物理可辨识性下限,
    不是实现缺陷; K_t 由独立的 T_cmd 列 (纯正弦) 支撑, 故估计精度高 (实测偏差 ~0.01%)。
    """
    n = len(Im)
    m = int(np.clip(round(float(healthy_frac) * n), 8, n))
    A = np.stack([tcmd[:m], np.sign(omega_cmd[:m]), omega_cmd[:m]],
                 axis=1).astype(np.float64)
    y = np.asarray(Im[:m], dtype=np.float64)
    scale = np.maximum(np.linalg.norm(A, axis=0), 1e-30)
    coef, *_ = np.linalg.lstsq(A / scale, y, rcond=None)
    a1, a2, a3 = coef / scale
    Kt_hat = 1.0 / a1 if abs(a1) > 1e-30 else np.nan
    return float(Kt_hat), float(a2 * Kt_hat), float(a3 * Kt_hat)


def identify_b_hat(Im, tcmd, omega, Kt, Tc, win: int = 64) -> np.ndarray:
    """滑窗最小二乘辨识黏性摩擦 b̂ (抗噪, 非逐点除法)。

    理论关系 y = K_t·I_m - T_cmd - T_c·sgn(ω) = b·ω; 窗内 LSQ b̂ = Σ(yω)/Σ(ω²)。
    Kt / Tc 由 calibrate_healthy 估得 (自校准), 不得传入仿真真值。
    """
    y = Kt * Im - tcmd - Tc * np.sign(omega)
    yo = pd.Series(y * omega).rolling(win, min_periods=1).sum().values
    o2 = pd.Series(omega ** 2).rolling(win, min_periods=1).sum().values
    return yo / (o2 + 1e-20)


def rolling_std(v, win: int) -> np.ndarray:
    """因果滑窗标准差 (窗口未满时用已有样本, 首点置 0)。"""
    s = pd.Series(np.asarray(v, dtype=np.float64)).rolling(win, min_periods=2).std()
    return s.fillna(0.0).values


def failure_threshold(sim_cfg: dict) -> float:
    """全局固定失效阈值 b_fail = (Kt_nom·Im_rated - Tc_nom) / omega_ref。

    全部取自 config sim.failure 的**标称值**, 对所有轨迹一致, 不依赖单机真值。
    仅供 hi_source="b" 的旧口径使用 (S2' 起默认口径为 "Tf", 不走此函数)。
    """
    fc = sim_cfg["failure"]
    num = float(fc["Kt_nom"]) * float(fc["Im_rated_A"]) - float(fc["Tc_nom"])
    return num / max(abs(float(fc["omega_ref"])), 1e-12)


def _causal_fixed_scale(v, healthy_ref, failure_ref):
    den = np.maximum(np.asarray(failure_ref, dtype=np.float64)
                     - float(healthy_ref), 1e-12)
    return np.maximum.accumulate(
        np.clip((np.asarray(v) - float(healthy_ref)) / den, 0.0, 1.0))


def _causal_mean(v, win: int) -> np.ndarray:
    """因果滑窗均值 (win<=1 时恒等)。窗口未满时用已有样本。"""
    if int(win) <= 1:
        return np.asarray(v, dtype=np.float64)
    return pd.Series(np.asarray(v, dtype=np.float64)).rolling(
        int(win), min_periods=1).mean().values


def _apply_persistence(raw: np.ndarray, persistence: int,
                       suppress_margin: float) -> np.ndarray:
    """把"连续 persistence 点达 1.0 才算饱和"的判定注入 HI (与 EOL 判据同构)。

    wheel_sim 的失效判据是 |I_m| 连续 persistence_samples 点超 Im_rated, EOL 取该
    连续段的**起点**。HI 若只看单点达 1.0, 会被一次上行噪声尖峰提前触发, 与 EOL
    错位。故此处同构处理: 找到首个满足连续条件的段, 段起点之前的值压到 1.0 以下。

    压制上限 = 1.0 - suppress_margin, 必须显著大于下游"达到 1.0"的判定容差
    reach_tol, 否则压制无效 (实测 margin=1e-6 < tol=1e-3 时首达点提前 14.5% EOL)。
    """
    p = max(1, int(persistence))
    if p == 1:
        return raw
    hit = raw >= 1.0 - 1e-12
    out = raw.copy()
    run = np.convolve(hit.astype(np.int16), np.ones(p, dtype=np.int16), mode="valid")
    ok = np.zeros_like(hit)
    ok[p - 1:] = run >= p
    cut = (int(np.argmax(ok)) - p + 1) if ok.any() else len(out)
    out[:cut] = np.minimum(out[:cut], 1.0 - float(suppress_margin))
    return out


def build_hi_Tf(Im, tcmd, Kt_hat: float, sim_cfg: dict):
    """HI 挂总摩擦力矩 (S2' 改动 1, 默认口径)。返回 (HI, Tf_hat, Tf_fail_scalar)。

        T̂_f(t)  = K̂_t·mean_w(I_m) - mean_w(T_cmd)
        T_f,fail = K̂_t·I_m,rated  - mean_w(T_cmd)
        HI = cummax(clip((T̂_f - T̂_f,0)/(T_f,fail - T̂_f,0), 0, 1))

    健康基准 T̂_f,0 取前 healthy_frac 段的 T̂_f 均值 (与 calibrate_healthy 同一段)。
    **只用 K̂_t**: T̂_c / b̂_0 一律不得出现在本函数 (S1 已证其拆分不可辨识)。

    分子分母同用 mean_w(T_cmd) 且 K̂_t 为公因子 -> HI≥1 ⟺ mean_w(I_m) ≥ I_m,rated,
    与 wheel_sim 的失效判据同源; 再叠加同一 persistence 判定即完全对齐。
    """
    fc = sim_cfg["failure"]
    hc = sim_cfg["hi"]
    win = int(hc["Tf_window"])
    Im_rated = float(fc["Im_rated_A"])
    Kt = float(Kt_hat)

    tcmd_w = _causal_mean(tcmd, win)
    Tf_hat = Kt * _causal_mean(Im, win) - tcmd_w
    # 阈值的 K_t 来源: "hat" = 与分子同源(默认, K̂_t 约掉) / "nom" = config 标称值
    kt_src = str(fc.get("Tf_fail_Kt_source", "hat"))
    Kt_fail = Kt if kt_src == "hat" else float(fc["Kt_nom"])
    Tf_fail = Kt_fail * Im_rated - tcmd_w          # 与分子同口径的逐点阈值

    n = len(Tf_hat)
    m = int(np.clip(round(float(hc["healthy_frac"]) * n), 8, n))
    Tf_0 = float(np.mean(Tf_hat[:m]))
    raw = np.clip((Tf_hat - Tf_0) / np.maximum(Tf_fail - Tf_0, 1e-12), 0.0, 1.0)
    HI = np.maximum.accumulate(
        _apply_persistence(raw, int(fc.get("persistence_samples", 1)),
                           float(hc["suppress_margin"])))
    # 返回的标量阈值仅供审计打印 (逐点阈值的均值)
    return HI, Tf_hat, float(np.mean(Tf_fail))


def causal_lsq_slope(v, win: int, min_periods: int = 8) -> np.ndarray:
    """因果长窗最小二乘斜率 dv/d(样本下标) (S5 改动 3)。

    窗内对 v ~ a + b·t 做 OLS, 返回逐点 b̂。全部用 rolling 累加量闭式算, 不做逐窗
    循环 (窗长可达 2000+, 逐窗 polyfit 在 5 万点 × 100 轨迹上不可接受)。

        b̂ = [Σ(t·v) - Σt·Σv/n] / [Σt² - (Σt)²/n]

    **因果**: pandas rolling 只看当前及过去, 与 x_T 其余列 (_causal_mean /
    rolling_std / identify_b_hat) 同一纪律 —— 任何一列偷看未来都是标签泄漏。
    窗口未满时用已有样本 (min_periods), 不足 min_periods 或分母退化 -> 0.0
    (不是 NaN: x_T 会进 z-score 与网络, NaN 会污染整批梯度; 0 = "尚无趋势证据",
    语义正确且与"健康期斜率≈0"连续)。
    """
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    W = max(2, int(win))
    mp = max(2, int(min_periods))
    t = np.arange(n, dtype=np.float64)

    def _sum(a):
        return pd.Series(a).rolling(W, min_periods=mp).sum().values

    cnt = pd.Series(np.ones(n)).rolling(W, min_periods=mp).sum().values
    st, sv = _sum(t), _sum(v)
    stv, stt = _sum(t * v), _sum(t * t)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = stt - st * st / cnt
        num = stv - st * sv / cnt
        b = np.divide(num, den, out=np.zeros(n, dtype=np.float64),
                      where=np.isfinite(den) & (np.abs(den) > 1e-12))
    return np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)


def build_trend_features(Tf_hat, sim_cfg: dict):
    """T̂_f -> (Tf_ratio, Tf_slope) 两列长基线趋势特征 (S5 改动 3)。

    Tf_ratio = T̂_f(t) / T̂_f,0, T̂_f,0 = 前 healthy_frac 段均值 —— 与 build_hi_Tf
    的健康基准**同一段、同一算法**, 保证两者物理口径一致。
    做**比值而非差值**的理由: T̂_f 的绝对量级被逐轨迹 b0 (10 倍跨度) 打散, 差值仍
    带这个量级; 比值以每条轨迹自己的健康态为分母, 无量纲且跨轨迹可比 —— 这正是
    S4 定位到的"绝对水平不可辨识"的直接解药。

    Tf_slope 窗长取 sim.hi.trend_window (>> L=64), 由 config 唯一给出。

    自校准纪律: T̂_f 只由 K̂_t 与遥测算出 (见 build_hi_Tf), 故这两列同样不含任何
    仿真真值 (Kt / Tc / b0 / omega0)。
    """
    hc = sim_cfg["hi"]
    Tf = np.asarray(Tf_hat, dtype=np.float64)
    n = len(Tf)
    m = int(np.clip(round(float(hc["healthy_frac"]) * n), 8, n))
    Tf_0 = float(np.mean(Tf[:m]))
    den = Tf_0 if abs(Tf_0) > 1e-12 else (1e-12 if Tf_0 >= 0 else -1e-12)
    Tf_ratio = Tf / den
    Tf_slope = causal_lsq_slope(Tf, int(hc["trend_window"]),
                                int(hc.get("trend_min_periods", 8)))
    return Tf_ratio, Tf_slope


def build_features(df: pd.DataFrame, sim_cfg: dict):
    """遥测 -> (x_T(N,10), b_hat, HI_A, HI_B, calib)。

    **签名不含 params**: 本函数只吃遥测列 + config, 拿不到仿真真值 Kt/Tc/b0/omega0,
    从接口层面保证自校准闭环 (见 tests/test_hi.py::test_build_features_uses_no_truth)。
    calib = {"Kt_hat", "Tc_hat", "b0_hat", "b_fail", "hi_source", "Tf_fail"} 供审计。

    HI_B 口径由 config sim.failure.hi_source 决定:
      "Tf" (默认) 挂总摩擦力矩, 只用 Kt_hat, 与失效判据同源;
      "b"  挂黏性摩擦 b̂ (S1 旧口径, 回退对比用)。
    两种口径下 b_hat 都照常计算并进入 x_T / h5 (改的是监督标签, 不是特征)。
    """
    missing = [c for c in TELEMETRY_COLS if c not in df]
    if missing:
        raise KeyError(f"缺遥测列 {missing}; 需重跑 src.sim.wheel_sim 生成 omega_cmd 等列")
    Im = df["I_m"].values.astype(np.float64)
    omega = df["omega"].values.astype(np.float64)
    T = df["T"].values.astype(np.float64)
    tcmd = df["T_cmd"].values.astype(np.float64)      # 力矩指令列
    omega_cmd = df["omega_cmd"].values.astype(np.float64)

    hi_cfg = sim_cfg["hi"]
    Kt_hat, Tc_hat, b0_hat = calibrate_healthy(
        Im, tcmd, omega_cmd, float(hi_cfg["healthy_frac"]))
    b_hat = identify_b_hat(Im, tcmd, omega, Kt_hat, Tc_hat,
                           int(hi_cfg["lsq_window"]))

    HI_A = _causal_fixed_scale(
        np.abs(Im) / max(float(sim_cfg["failure"]["Im_rated_A"]), 1e-12), 0.0, 1.0)

    b_fail = failure_threshold(sim_cfg)
    hi_source = str(sim_cfg["failure"]["hi_source"])
    if hi_source not in ("Tf", "b"):
        raise ValueError(f"未知 sim.failure.hi_source={hi_source!r}; 应为 'Tf' 或 'b'")
    # T̂_f 两种口径下都要算: "Tf" 口径用它做 HI_B, "b" 口径只用它做趋势特征
    # (S5 改动 3 的 Tf_ratio/Tf_slope 不随监督 HI 口径改变, 两者可独立消融)。
    HI_Tf, Tf_hat, Tf_fail = build_hi_Tf(Im, tcmd, Kt_hat, sim_cfg)
    if hi_source == "Tf":
        # 只传 Kt_hat: Tc_hat / b0_hat 不进入 HI 路径 (S2' 硬约束)
        HI_B = HI_Tf
    else:
        # 健康基准用自校准 b0_hat; 若估计退化到 >= b_fail 则回退到 0, 保证分母正
        b_healthy = float(b0_hat) if b0_hat < b_fail else 0.0
        HI_B = _causal_fixed_scale(b_hat, b_healthy, b_fail)
        Tf_fail = float("nan")

    sigma_Im = rolling_std(Im, int(hi_cfg["sigma_window"]))
    dT = np.zeros_like(T)
    dT[1:] = np.diff(T)
    Tf_ratio, Tf_slope = build_trend_features(Tf_hat, sim_cfg)
    x_T = np.stack([
        Im, omega, T, tcmd, sigma_Im, b_hat, dT, np.abs(omega - omega_cmd),
        Tf_ratio, Tf_slope,
    ], axis=1)
    calib = dict(Kt_hat=Kt_hat, Tc_hat=Tc_hat, b0_hat=b0_hat, b_fail=b_fail,
                 hi_source=hi_source, Tf_fail=Tf_fail)
    return x_T, b_hat, HI_A, HI_B, calib


def _rel_dev(hat, true):
    return (float(hat) - float(true)) / (abs(float(true)) + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel.yaml")
    ap.add_argument("--in", dest="indir", default="data/simulated/wheel/sim_v1/seed_42")
    ap.add_argument("--out", default="data/features/wheel/schema_v1/target_features.h5")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sim_cfg = cfg["sim"]
    cap_ratio = float(cfg["source"]["rul_cap_ratio"])

    h5_in = ROOT / args.indir / "wheel_all.h5"
    if not h5_in.exists():
        print(f"!! 缺 {h5_in}; 先 python -m src.sim.wheel_sim --n_traj 100")
        sys.exit(1)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    b_errs, b_rel_errs, hi_maxs, hi_mins = [], [], [], []
    dev_Kt, dev_Tc, dev_b0 = [], [], []
    hi_offsets = []            # 已观测且 HI 达 1.0 的轨迹: HI 首达 1.0 - EOL
    hi_offsets_rel = []        # 同上, 除以 EOL (相对偏移, 供 2% 判据核对)
    n_traj, n_fail, n_hi_reach = 0, 0, 0
    HI_REACH_TOL = float(sim_cfg["hi"]["reach_tol"])   # config 唯一出处 (禁硬编码)
    with h5py.File(h5_in, "r") as fin, h5py.File(out, "w") as fout:
        for key in sorted(fin.keys()):
            g = fin[key]
            df = pd.DataFrame({c: g[c][:] for c in
                               TELEMETRY_COLS + ["b_true", "label_fail"]})
            # 特征/HI 只吃遥测 (b_true 与 attrs 真值不入此调用)
            xT, b_hat, HI_A, HI_B, calib = build_features(df, sim_cfg)
            b_true = df["b_true"].values
            lf = df["label_fail"].values
            observed = bool(lf.any())
            eol = int(np.argmax(lf)) if observed else len(df) - 1
            n = len(df)
            if observed:
                rul = np.minimum((eol - np.arange(n)).astype(float), cap_ratio * n)
                rul[eol:] = 0.0
                rul_lower_bound = rul.copy()
            else:
                rul = np.full(n, np.nan, dtype=float)
                rul_lower_bound = np.minimum(n - 1 - np.arange(n), cap_ratio * n)
            gg = fout.create_group(key)
            gg.create_dataset("x_T", data=xT.astype(np.float32))
            gg.create_dataset("hi_b", data=HI_B.astype(np.float32))
            gg.create_dataset("hi_a", data=HI_A.astype(np.float32))
            gg.create_dataset("b_hat", data=b_hat.astype(np.float32))
            gg.create_dataset("b_true", data=b_true.astype(np.float32))   # 仅离线校核
            gg.create_dataset("rul", data=rul.astype(np.float32))
            gg.create_dataset("rul_lower_bound", data=rul_lower_bound.astype(np.float32))
            gg.create_dataset("label_fail", data=lf.astype(np.int8))
            gg.attrs["eol_idx"] = eol
            gg.attrs["event_observed"] = int(observed)
            for k, v in calib.items():
                gg.attrs[k] = v if isinstance(v, str) else float(v)

            # 有噪条件下 b̂ 的真实误差 (逐点, 非去噪平滑后)
            b_rel_errs.append(float(np.mean(np.abs(b_hat - b_true))
                                    / (np.mean(np.abs(b_true)) + 1e-30)))
            b_errs.append(float(np.mean(np.abs(b_hat - b_true))))
            hi_mins.append(float(HI_B.min()))
            hi_maxs.append(float(HI_B.max()))
            reached = bool(HI_B.max() >= 1.0 - HI_REACH_TOL)
            n_hi_reach += int(reached)
            if observed and reached:
                first = int(np.argmax(HI_B >= 1.0 - HI_REACH_TOL))
                hi_offsets.append(first - eol)
                hi_offsets_rel.append((first - eol) / max(eol, 1))
            n_traj += 1
            n_fail += int(observed)

            if args.report:
                # ---- 审计专用块: 唯一读仿真真值处, 只为打印估计偏差 ----
                # 这些值不参与上面任何 dataset 的计算, 勿在此块外引用
                _audit_truth = {k: float(g.attrs[k]) for k in ["Kt", "Tc", "b0"]}
                dev_Kt.append(_rel_dev(calib["Kt_hat"], _audit_truth["Kt"]))
                dev_Tc.append(_rel_dev(calib["Tc_hat"], _audit_truth["Tc"]))
                dev_b0.append(_rel_dev(calib["b0_hat"], _audit_truth["b0"]))

    print(f">> 处理 {n_traj} 条轨迹 -> {out}")
    if args.report:
        def _stat(name, v):
            v = np.asarray(v)
            print(f"   {name:8s}: mean={v.mean():+.4f}  median={np.median(v):+.4f}  "
                  f"p5={np.quantile(v,0.05):+.4f}  p95={np.quantile(v,0.95):+.4f}  "
                  f"max|·|={np.max(np.abs(v)):.4f}")

        b_errs = np.array(b_errs)
        b_rel_errs = np.array(b_rel_errs)
        hi_source = str(sim_cfg["failure"]["hi_source"])
        print("\n===== P4 HI 构造报告 (S2' 一致性修复版) =====")
        print(f"x_T 维度 = {len(XT_COLS)}  (列: {XT_COLS})")
        print(f"失效轨迹数 = {n_fail}/{n_traj}")
        print(f"HI 口径 sim.failure.hi_source = {hi_source!r}")
        if hi_source == "Tf":
            fc = sim_cfg["failure"]
            print(f"  HI 挂总摩擦力矩: T̂_f = K̂_t·mean_w(I_m) - mean_w(T_cmd), "
                  f"w = hi.Tf_window = {int(sim_cfg['hi']['Tf_window'])}")
            print(f"  阈值 T_f,fail 的 K_t 来源 = "
                  f"{str(fc.get('Tf_fail_Kt_source', 'hat'))!r} "
                  f"('hat' 时 K̂_t 约掉 -> HI≥1 ⟺ mean_w(I_m) ≥ I_m,rated"
                  f"={float(fc['Im_rated_A'])}, 与失效判据同源)")
            print(f"  persistence 对齐 = {int(fc.get('persistence_samples', 1))} 点 "
                  f"(与 wheel_sim 失效判据同一值)")
            print(f"  Tc_hat / b0_hat 未进入 HI 路径 (S2' 硬约束)")
        else:
            print(f"  HI 挂黏性摩擦 b̂ (S1 旧口径, 回退对比用); "
                  f"b_fail = {failure_threshold(sim_cfg):.4e}")
        print("\n-- 自校准估计相对真值偏差 (审计块, 真值不入特征路径) --")
        _stat("Kt_hat", dev_Kt)
        _stat("Tc_hat", dev_Tc)
        _stat("b0_hat", dev_b0)
        print("\n-- 有噪条件下 b̂ 真实误差 (逐点, 未做去噪平滑) --")
        print(f"   MAE       : mean={b_errs.mean():.4e}  max={b_errs.max():.4e}")
        print(f"   相对 MAE  : mean={b_rel_errs.mean():.4f}  "
              f"median={np.median(b_rel_errs):.4f}  max={b_rel_errs.max():.4f}")
        print(f"   (b̂ 仍在 x_T 第 {XT_COLS.index('b_hat') + 1} 维 / h5 / "
              f"physical_extrap 输入, 未因 HI 换口径而下线)")

        # ---- HI 范围: 两个口径都打印 (S1.5 澄清过的歧义) ----
        hi_maxs = np.array(hi_maxs)
        hi_mins = np.array(hi_mins)
        print(f"\n-- HI_B 范围 (口径 = {hi_source!r}) --")
        print(f"   口径 1 各轨迹 max 的均值 mean(max_t HI) = {hi_maxs.mean():.4f}")
        print(f"   口径 2 全局 max                        = {hi_maxs.max():.4f}")
        print(f"          各轨迹 max 的最小值             = {hi_maxs.min():.4f}")
        print(f"          各轨迹 min 的均值               = {hi_mins.mean():.4f}")
        print(f"   达到 1.0 的轨迹数 = {n_hi_reach}/{n_traj}  "
              f"(失效轨迹数 = {n_fail}; 两者相等 = {n_hi_reach == n_fail})")

        # ---- HI 首达 1.0 与 EOL 的偏移 (S2' 改动 1 的成败判据) ----
        print(f"\n-- HI 首达 1.0 相对 EOL 的偏移 (已观测且达 1.0 的 "
              f"{len(hi_offsets)} 条; 负=HI 先饱和) --")
        if hi_offsets:
            o = np.array(hi_offsets, dtype=float)
            r = np.abs(np.array(hi_offsets_rel, dtype=float))
            print(f"   绝对偏移 (样本数): mean={o.mean():+.1f}  "
                  f"p5={np.quantile(o, 0.05):+.1f}  p50={np.median(o):+.1f}  "
                  f"p95={np.quantile(o, 0.95):+.1f}")
            print(f"   相对偏移 |Δ|/EOL : mean={r.mean():.4f}  "
                  f"p5={np.quantile(r, 0.05):.4f}  p50={np.median(r):.4f}  "
                  f"p95={np.quantile(r, 0.95):.4f}  max={r.max():.4f}")
            print(f"   超出 2% 判据的轨迹数 = {int((r > 0.02).sum())}/{len(r)}")
        else:
            print("   无样本 (无轨迹同时满足 已观测 & HI 达 1.0)")
        print("b̂ 用健康段自校准 Kt_hat/Tc_hat + 滑窗最小二乘; 未使用仿真真值")
        print("==========================\n")


if __name__ == "__main__":
    main()
