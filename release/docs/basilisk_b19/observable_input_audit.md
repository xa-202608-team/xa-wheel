# BASILISK-B1.9 §3 可观测输入审计

- protocol sha256: `e95bf867e083aa0d…`
- 数据集: `data/sim/wheel_basilisk_b18/final/wheel_all.h5` (150 条 trajectory)
- 判定: **B19_OBSERVABILITY_OK** (8/8)

## 1. provenance 逐项

| role | 量 | 声明来源 | 派生式 | h5 容器 | provenance |
|---|---|---|---|---|---|
| g_duty_input | `omega_p95` | `mission_features/omega_p95` | `abs(.)/omega_rated` | mission_features_group | **OBSERVABLE_DERIVED** |
| g_duty_input | `torque_p95` | `mission_features/torque_p95` | `abs(.)/torque_rated` | mission_features_group | **OBSERVABLE_DERIVED** |
| g_duty_input | `zero_crossing_count` | `mission_features/zero_crossing_count` | `./n_per_window` | mission_features_group | **OBSERVABLE_DERIVED** |
| g_duty_input | `maneuver_fraction` | `mission_features/maneuver_fraction` | `identity` | mission_features_group | **OBSERVABLE_DERIVED** |
| temperature | `T` | `telemetry/T` | `identity` | trajectory_dataset | **OBSERVABLE_DIRECT** |
| hidden_reference | `D` | `-` | `-` | truth_group | **HIDDEN_TRUTH** |
| hidden_reference | `b_true` | `-` | `-` | trajectory_dataset | **HIDDEN_TRUTH** |
| hidden_reference | `Delta` | `-` | `-` | trajectory_attrs | **HIDDEN_TRUTH** |
| hidden_reference | `tau_years` | `-` | `-` | trajectory_attrs | **HIDDEN_TRUTH** |
| hidden_reference | `b0` | `-` | `-` | trajectory_attrs | **HIDDEN_TRUTH** |
| hidden_reference | `omega0` | `-` | `-` | trajectory_attrs | **HIDDEN_TRUTH** |
| hidden_reference | `Kt` | `-` | `-` | trajectory_attrs | **HIDDEN_TRUTH** |
| hidden_reference | `Tc` | `-` | `-` | trajectory_attrs | **HIDDEN_TRUTH** |
| hidden_reference | `label_fail` | `-` | `-` | trajectory_dataset | **HIDDEN_TRUTH** |
| hidden_reference | `T_base` | `-` | `-` | trajectory_attrs | **HIDDEN_TRUTH** |
| hidden_reference | `T_amp` | `-` | `-` | trajectory_attrs | **HIDDEN_TRUTH** |
| hidden_reference | `seed_traj` | `-` | `-` | trajectory_attrs | **HIDDEN_TRUTH** |
| hidden_reference | `D` | `-` | `-` | truth_group | **HIDDEN_TRUTH** |

## 2. 派生链核验 (声明式 vs B11 官方函数)

`n_per_window = 1800`

| trajectory | 驱动量 | max_abs_diff | 逐位相同 |
|---|---|---|---|
| traj_000 | `speed_util` | 0.000e+00 | 是 |
| traj_000 | `torque_util` | 0.000e+00 | 是 |
| traj_000 | `zero_crossing_rate` | 0.000e+00 | 是 |
| traj_000 | `maneuver_fraction` | 0.000e+00 | 是 |
| traj_001 | `speed_util` | 0.000e+00 | 是 |
| traj_001 | `torque_util` | 0.000e+00 | 是 |
| traj_001 | `zero_crossing_rate` | 0.000e+00 | 是 |
| traj_001 | `maneuver_fraction` | 0.000e+00 | 是 |
| traj_002 | `speed_util` | 0.000e+00 | 是 |
| traj_002 | `torque_util` | 0.000e+00 | 是 |
| traj_002 | `zero_crossing_rate` | 0.000e+00 | 是 |
| traj_002 | `maneuver_fraction` | 0.000e+00 | 是 |
| traj_003 | `speed_util` | 0.000e+00 | 是 |
| traj_003 | `torque_util` | 0.000e+00 | 是 |
| traj_003 | `zero_crossing_rate` | 0.000e+00 | 是 |
| traj_003 | `maneuver_fraction` | 0.000e+00 | 是 |
| traj_004 | `speed_util` | 0.000e+00 | 是 |
| traj_004 | `torque_util` | 0.000e+00 | 是 |
| traj_004 | `zero_crossing_rate` | 0.000e+00 | 是 |
| traj_004 | `maneuver_fraction` | 0.000e+00 | 是 |

## 3. 部署前提 (必须如实记录, 不得含糊)

omega_p95 / torque_p95 / zero_crossing_count / maneuver_fraction 是窗内1 Hz 序列的聚合统计 (每窗 1800 点), 而本数据集的下传遥测只保存每窗 1 个1800 s 采样点。因此它们是 OBSERVABLE_DERIVED 而非 OBSERVABLE_DIRECT, 且其部署前提是**星上按窗做 1 Hz 聚合后下传统计量**, 不能由已存储的低速遥测反推 —— 实测 corr(|omega_telemetry|, omega_p95) 远小于 1 即是证据。这是一条对可部署性的真实约束, 必须写进 limitations, 不得含糊成'都是遥测所以可观测'。

| trajectory | corr(\|omega_tel\|, omega_p95) | 可由已存遥测反推 |
|---|---|---|
| traj_000 | 0.7868 | **否** |
| traj_001 | 0.7228 | **否** |
| traj_002 | 0.7602 | **否** |
| traj_003 | 0.7262 | **否** |
| traj_004 | 0.7546 | **否** |

## 4. 检查项

| # | 检查 | 结果 | 说明 |
|---|---|---|---|
| 1 | §3 四个 g_duty 输入均非 HIDDEN_TRUTH | PASS | 违规 = []; provenance = speed_util=OBSERVABLE_DERIVED, torque_util=OBSERVABLE_DERIVED, zero_crossing_rate=OBSERVABLE_DERIVED, maneuver_fraction=OBSERVABLE_DERIVED |
| 2 | §3 g_duty 输入齐备 (4 项) | PASS | 实得 4 项 |
| 3 | §3 温度来自遥测 dataset | PASS | T 容器 = trajectory_dataset, provenance = OBSERVABLE_DIRECT |
| 4 | §3 黑名单量全部落在 truth 侧 (证明与输入集不相交) | PASS | 13 项, 非 HIDDEN_TRUTH 的 = [] |
| 5 | §3 输入集 ∩ 黑名单集 = ∅ | PASS | 交集 = [] |
| 6 | §3 声明派生式与 B11 官方派生函数逐位一致 | PASS | 抽查 5 条; max_abs_diff = 0.000e+00 |
| 7 | §3 全轨迹结构一致 (每条都有全部可观测输入且不在 truth 侧) | PASS | 违规 0 项: [] |
| 8 | §3 已记录: 窗统计量不可由低速遥测反推 (部署前提) | PASS | max corr(|omega_tel|, omega_p95) = 0.7868 < 0.999, 故需星上 1 Hz 窗聚合 |

## 5. 本脚本读取 truth 容器的目的

本脚本只判定 provenance, 不构造 HI, 不训练, 不产出任何 RUL 数字。它读取 truth 容器 (truth/D, b_true, trajectory attrs) 的**唯一目的**是证明这些量与 HI_D_obs 的输入集**不相交** —— 即把它们定位出来并标为HIDDEN_TRUTH, 从而使 g_duty 输入不含其中任一项这件事可被机器核验。定位不等于使用: 本脚本不把 truth 数值写入任何下游可训练文件。

## 判定

**B19_OBSERVABILITY_OK**
