# BASILISK-B1.9 §2/§4/§5 可观测损伤代理推导

- protocol sha256: `e95bf867e083aa0d…`
- 轨迹数: 150 (event 71 / censored 79)

## 1. 方程 (冻结)

```
HI_D_obs(t) = clip(cumsum(g_duty_obs(t) * a_T(T_obs(t))) * dt / SEC_PER_YEAR / L_ref, 0, 1)
```

| 项 | 值 | 实现来源 (import, 非复制) |
|---|---|---|
| `L_ref` | 3.0 years | B1.8 NOMINAL |
| `g_duty` 权重 | speed 0.4 / torque 0.4 / zc 0.1 / man 0.1 | `basilisk_b1/calibrate_degradation.py::g_duty` |
| `Ea` / `T_ref` | 0.3 eV / 293.15 K | `damage_model.py::arrhenius_accel` |
| `dt` | 1800.0 s | `sim.sample_period_s` |
| 累加 | 右端点 cumsum, 仅时间正序 | `damage_model.py::damage_series` |

### reference duty (冻结 profile 库决定, 无可调参数)

| 驱动量 | reference 值 |
|---|---|
| `speed_util` | 0.404189828 |
| `torque_util` | 0.284426851 |
| `zero_crossing_rate` | 0.000185185 |
| `maneuver_fraction` | 0.463407407 |

## 2. HI_D_obs 数值特征

| 组 | median | p10/p90 | min | max |
|---|---|---|---|---|
| event-observed @EOL | 1.000000 | p10 1.000000 | 1.000000 | 1.000000 |
| censored @末点 | 0.600160 | p90 0.903155 | 0.274118 | 0.984823 |

- censored 中 `HI >= 1` 的条数: **0** (§8 Gate 9 要求不得被强制为 1)
- 全部有限: True; 全部落在 [0,1]: True; 全部单调非降: True

## 3. §4 前缀不变性 (截断未来 50% 后重算)

- 抽查 10 条; max_abs_diff = **0.000e+00**; 逐位相同 = **True**

容差为 0 —— cumsum 的前缀在数学上就该逐位相同, 任何非零差异都说明构造式里混入了未来信息。

## 4. §5 IMPLEMENTATION_AUDIT (离线, 不进特征)

| 量 | 值 |
|---|---|
| corr 最小值 | 1.00000000 |
| corr 中位数 | 1.00000000 |
| RMSE 中位数 | 3.9538e-10 |
| RMSE 最大值 | 2.7368e-09 |
| max_abs_error 最大值 | 4.7996e-09 |

## 5. 必须诚实声明的一点

必须诚实声明: 这是 recomputation 而非 estimation。B1.8 的内部累积损伤本身就是由同一组已存储窗统计量经同一组方程算出的, 因此 HI_D_obs 与它的一致性接近机器精度, 属于**必然**, 不是'模型学得好'的证据。直接后果: §8 Gate 10 (corr > 0.95) 近乎恒真, 只能证明实现没写错, 不能被引用为 HI 具有预测能力的证据。真正有约束力的是 §4 前缀不变性与 Gate 3/4/5/6/9/11/13。

## 6. 本脚本对 truth 的处置

HI_D_obs 只从 mission_features 窗统计 + 遥测 T + 已冻结方程重新独立计算; 它不读取 simulator 内部累积损伤, 不读取 b_true, 不读取 trajectory attrs 里的采样参数, 不读取 EOL/RUL。truth/D 仅在 feature 构建完成后被读取一次, 用途标签固定为 IMPLEMENTATION_AUDIT: 只进 RMSE/corr/max_abs 三个报告数字, 不进构造公式、不进 x_T、不参与任何归一化、不作 train-time input。

## 7. 禁止的归一化方式 (全部未使用)

- `full_trajectory_normalization`
- `future_max`
- `final_value_normalization`
- `eol_normalization`
- `trajectory_level_min_max_using_future_rows`
