# Wheel 输入遥测约定

> 本文件定义反作用飞轮组件的输入遥测字段规范。校验器 `scripts/validate_input_schema.py`
> 按此约定对 CSV 输入做字段存在性、单位一致性与数据类型检查。
>
> 数据来源：
> - 源域：XJTU-SY 轴承故障数据集（25.6 kHz 振动信号）
> - 目标域：Basilisk 仿真飞轮工况（1 s 分辨率 → 30 min 退化窗聚合）
>
> 参见 `release/configs/wheel.yaml`、`release/src/sim/build_hi.py`（`XT_COLS` / `TELEMETRY_COLS`）。

## 目标域 — 原始遥测 (TELEMETRY_COLS, 30 min 窗)

飞轮退化仿真产出的原始遥测列（`build_hi.py::TELEMETRY_COLS`）：

| 字段 | 单位 | 采样率 | 必需 | 说明 |
|------|------|--------|------|------|
| `t` | s | 30 min (1800 s) | 是 | 时间戳，从轨迹起点计 |
| `omega` | rad/s | 30 min | 是 | 飞轮角速度（实测，含量测噪声） |
| `omega_cmd` | rad/s | 30 min | 是 | 指令角速度（解析指令，星上可知，干净无噪） |
| `I_m` | A | 30 min | 是 | 电机电流（实测，含量测噪声） |
| `T` | °C | 30 min | 是 | 飞轮外壳温度（实测遥测，OBSERVABLE_DIRECT） |
| `T_cmd` | N·m | 30 min | 是 | 指令力矩（解析指令，星上可知，干净无噪） |

## 目标域 — 特征 x_T (XT_COLS, 10 维)

由遥测派生的模型输入特征（`build_hi.py::XT_COLS`），不含量仿真真值：

| 字段 | 单位 | 采样率 | 必需 | 说明 |
|------|------|--------|------|------|
| `I_m` | A | 30 min | 是 | 电机电流（原始遥测列） |
| `omega` | rad/s | 30 min | 是 | 飞轮角速度（原始遥测列） |
| `T` | °C | 30 min | 是 | 飞轮温度（原始遥测列） |
| `T_cmd` | N·m | 30 min | 是 | 指令力矩（原始遥测列） |
| `sigma_Im` | A | 30 min | 是 | I_m 滑窗标准差（窗长 `hi.sigma_window` = 64 样本），轴承磨损致摩擦抖动加剧的可观测 |
| `b_hat` | N·m·s/rad | 30 min | 是 | 滑窗 LSQ 辨识黏性摩擦系数（健康段自校准 Kt_hat/Tc_hat 估计值，非仿真真值） |
| `dT` | °C | 30 min | 是 | 温度变化率，热-退化耦合的可观测量 |
| `omega_err` | rad/s | 30 min | 是 | 角速度跟踪误差 = |omega - omega_cmd| |
| `Tf_ratio` | — (无量纲) | 30 min | 是 | 总摩擦力矩比值 = T_f(t)/T_f,0（逐轨迹自归一，消 b0 量级跨度） |
| `Tf_slope` | — (无量纲) | 30 min | 是 | T_f 的长窗因果 LSQ 斜率（窗长 `hi.trend_window` = 2000 >> L=64） |

## 目标域 — 任务工况统计 (g_duty_inputs, OBSERVABLE_DERIVED)

B19 可观测性审计的 4 个占空比驱动输入（来自 mission_features 组，**窗内 1 Hz 聚合**）：

| 字段 | 单位 | 采样率 | 必需 | 说明 |
|------|------|--------|------|------|
| `speed_util` | — (无量纲) | 30 min | 否 | 轮速利用率 = omega_p95 / omega_rated |
| `torque_util` | — (无量纲) | 30 min | 否 | 力矩利用率 = torque_p95 / torque_rated |
| `zero_crossing_rate` | — (1/窗) | 30 min | 否 | 零穿越率 = zero_crossing_count / n_per_window |
| `maneuver_fraction` | — (无量纲) | 30 min | 否 | 机动占比（|u| > 0.02·u_max 的窗内比例） |

> **部署前提**：以上 4 个统计量需**星上按窗做 1 Hz 聚合后下传统计量**，
> 不能由已存储的低速遥测（30 min 采样点）反推（实测 corr(|omega_tel|, omega_p95) < 0.79）。

## 源域 — XJTU-SY 轴承振动 (12 维退化特征)

源域预训练使用 25.6 kHz 振动信号按 60 s 窗聚合的 12 维退化特征
（`source.features.target_dim: 12`）：

| 特征类别 | 典型字段 | 说明 |
|----------|----------|------|
| 时域统计 | RMS, 峰值, 峭度, 偏度, 波形因子 | 振动信号窗内统计量 |
| 频域特征 | 主频漂移, 谱重心, 谱峭度 | FFT 频带能量分布 |
| 包络谱 | BPFO/BPFI/BSF 幅值 | 轴承特征频率包络幅值 |

> 源域采样率 25.6 kHz（`source.sample_rate_hz: 25600`），窗长 60 s（`window_sec: 60`），
> 与目标域 30 min 退化窗不同尺度。迁移发生在 HI 动力学层（退化趋势映射），
> 非原始振动波形层。

## 仿真参数（configs/wheel.yaml 关键值）

| 参数 | 值 | 说明 |
|------|------|------|
| 目标域采样周期 | 1800 s (30 min) | `sim.sample_period_s` |
| LEO 轨道周期 | 5700 s | `sim.leo_period_s`（姿态指令周期） |
| 仿真年限 | 3.0 年 | `sim.duration_years` |
| 失效判据 | I_m > 1.5 A 持续 4 点 | `sim.failure.Im_rated_A` / `persistence_samples` |
| 源域采样率 | 25600 Hz | `source.sample_rate_hz`（XJTU-SY 25.6 kHz） |
| 源域窗长 | 60 s | `source.window_sec` |

## 禁止列（forbidden_xt_cols）

以下仿真真值列禁止出现在 x_T 中（`configs/wheel_basilisk_b19.yaml`）：

`D`, `damage`, `hi_damage_obs`, `hi_d_obs`, `eol_idx`, `rul`, `rul_lower_bound`,
`b_true`, `Delta`, `tau_years`, `b0`, `omega0`, `Kt`, `Tc`

## 数值约定

- `I_m` 失效阈值 = 1.5 A（`Im_rated_A`），连续 4 个采样点超限即判失效
- `omega` 典型范围 1000–2000 rad/s（初始转速偏置 + 散布）
- `T` Arrhenius 参考温度 = 293.15 K（20 °C），激活能 Ea = 0.3 eV
- `Tf_ratio` 无量纲，=1.0 对应健康基线，随退化单调增长
- `b_hat` 通过健康段自校准估计（不用仿真真值 Kt/Tc/b0）
- x_T 中禁止出现未来信息（chronological-only：禁止全轨迹归一 / EOL 归一 / 末值归一）
