# BASILISK-B1.9 冻结的主健康指标定义

**判定: `B19_FEATURE_READY`** (build 14/14, 独立复核 14/14, 两侧一致 = True)

- protocol sha256: `e95bf867e083aa0d3095f31049c7587ab3acc21a463463eac640a427008e8a3e`
- config sha256: `95cdb74ad61fc9f7fc53031ec6f662d362aaa07cc977b374c762dc2fec7ae1bf`
- 特征文件: `data/features/wheel/basilisk_b19/target_features.h5`
- content sha256: `510dd14e5f3f6ae214c2b2ed85f3eb0227504f4ccc0529ace233d5a26f378cba`

## 1. 主监督 HI

- dataset 名: **`hi_damage_obs`** (符号 `HI_D_obs`)

```
HI_D_obs(t) = clip(cumsum(g_duty_obs(t) * a_T(T_obs(t))) * dt / SEC_PER_YEAR / L_ref, 0, 1)
```

| 项 | 值 |
|---|---|
| `L_ref` | 3.0 years (= B1.8 NOMINAL, 不得改) |
| clip | [0.0, 1.0] |
| 累加方向 | `chronological_only` |
| 归一化分母 | L_ref (constant) |
| 失效定义 | `D >= 1` (B1.8 冻结, B1.9 未改) |

### 方程实现来源 (import, 非复制)

| 部件 | 来源 |
|---|---|
| `g_duty` | `scripts/basilisk_b1/calibrate_degradation.py::g_duty` |
| `a_T` | `src/sim/damage_model.py::arrhenius_accel` |
| `stress` | `src/sim/damage_model.py::stress_series` |
| `cumulative` | `src/sim/damage_model.py::damage_series` |
| `drives` | `scripts/basilisk_b11/calibrate_degradation.py::utilization_from_duty` |
| `reference` | `scripts/basilisk_b11/calibrate_degradation.py::reference_duty` |

### 冻结参数

| 参数 | 值 |
|---|---|
| `wear_drive.speed_util_weight` | 0.4 |
| `wear_drive.torque_util_weight` | 0.4 |
| `wear_drive.zero_crossing_weight` | 0.1 |
| `wear_drive.maneuver_weight` | 0.1 |
| `wear_drive.speed_util_exponent` | 2.0 |
| `wear_drive.torque_util_exponent` | 1.0 |
| `wear_drive.ref_floor` | 0.0001 |
| `wear_drive.g_duty_max` | 20.0 |
| `arrhenius.Ea_eV` | 0.3 |
| `arrhenius.T_ref_K` | 293.15 |
| `arrhenius.k_B_eV_per_K` | 8.617e-05 |
| `reference_duty.speed_util` | 0.404189828 |
| `reference_duty.torque_util` | 0.284426851 |
| `reference_duty.zero_crossing_rate` | 0.000185185 |
| `reference_duty.maneuver_fraction` | 0.463407407 |
| `dt_s` | 1800.0 |
| `n_per_window` | 1800 |

## 2. 输入可观测性

- §3 判定: `B19_OBSERVABILITY_OK`

| 输入 | provenance |
|---|---|
| `speed_util` | OBSERVABLE_DERIVED |
| `torque_util` | OBSERVABLE_DERIVED |
| `zero_crossing_rate` | OBSERVABLE_DERIVED |
| `maneuver_fraction` | OBSERVABLE_DERIVED |
| 温度 `T` | OBSERVABLE_DIRECT |

**部署前提**: omega_p95 / torque_p95 / zero_crossing_count / maneuver_fraction 是窗内1 Hz 序列的聚合统计 (每窗 1800 点), 而本数据集的下传遥测只保存每窗 1 个1800 s 采样点。因此它们是 OBSERVABLE_DERIVED 而非 OBSERVABLE_DIRECT, 且其部署前提是**星上按窗做 1 Hz 聚合后下传统计量**, 不能由已存储的低速遥测反推 —— 实测 corr(|omega_telemetry|, omega_p95) 远小于 1 即是证据。这是一条对可部署性的真实约束, 必须写进 limitations, 不得含糊成'都是遥测所以可观测'。

## 3. `x_T` schema (与 B1.8 逐位相同)

```
I_m, omega, T, T_cmd, sigma_Im, b_hat, dT, omega_err, Tf_ratio, Tf_slope
```

- 列数 10; 与 B1.8 逐位相同 = **True**
- `damage_proxy_in_xT = False`, `mission_features_in_xT = False`

### HDF5 dataset 清单

- `x_T`
- `hi_damage_obs`
- `hi_friction`
- `hi_a`
- `b_hat`
- `b_true`
- `rul`
- `rul_lower_bound`
- `label_fail`
- `mission_features(group)`

## 4. 辅助 HI

- `hi_friction` — mechanism_comparison_and_report_ablation
- 旧 friction 口径, 锚在已废止的 Tf_fail 上, 不得作主监督

## 5. HI 分布

| 组 | n | median | p10/p90 | min | max |
|---|---|---|---|---|---|
| event @EOL | 71 | 1.000000 | p10 1.000000 | 1.000000 | 1.000000 |
| censored @末点 | 79 | 0.600160 | p90 0.903155 | 0.274118 | 0.984824 |

- censored 被强制 HI=1 的条数: **0**

## 6. IMPLEMENTATION_AUDIT (不是能力证据)

- corr min 1.00000000; RMSE max 2.737e-09; max_abs max 4.800e-09

> Gate 10 近乎恒真: HI_D_obs 是对同一组已存储窗统计量按同一组方程的 recomputation, 不是 estimation。它只能证明实现没写错, 不能作为 HI 具有预测能力的证据。真正有约束力的是 Gate 3/4/5/6/9/11/13。

## 7. 关于'选择'这个词

本阶段没有做 HI 候选筛选 —— 主 HI 在 §2 protocol 里就已单点冻结, 在任何 HI 数值被算出之前。协议明令不得用 RUL RMSE 选 HI, 因此这里不存在'从若干候选中挑一个'的动作, selected 只是把已冻结的那一个记录下来。

## 8. 下游契约 (B2 起必须遵守)

- 必须使用: `data/features/wheel/basilisk_b19/target_features.h5`
- 不得更改: `L_ref`, `failure_definition`, `hi_equation`, `normalization_denominator`, `xt_cols`, `b18_frozen_dataset`
- `hi_damage_obs` 允许的角色: `supervision_target`, `health_auxiliary_loss`
- `hi_damage_obs` 禁止的角色: `plain_input_column_in_xT`

## 冻结件用途

冻结件是 B2 及之后阶段唯一可引用的 HI 定义来源。它记录方程、L_ref、g_duty 权重、Arrhenius 参数、reference duty、特征文件哈希与 14 条 Gate 结果, 使'后续阶段悄悄换刻度'在契约层可被发现。

- `trained_any_model = False`
