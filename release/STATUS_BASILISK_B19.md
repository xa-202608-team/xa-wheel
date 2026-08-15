# STATUS — BASILISK-B1.9

## 判定

```
B19_OBSERVABILITY_OK        (§3, 8/8)
B19_DAMAGE_PROXY_DERIVED    (§2/§4)
B19_FEATURE_READY           (§8, 14/14, 构建侧与独立审计侧一致)
```

**结论：`B19_FEATURE_READY`** —— 主健康指标 `hi_damage_obs` 已冻结，
下游 BASILISK-B2 的前置条件满足。

## 本阶段回答的唯一问题

> 能否构建一个不读取仿真内部隐藏损伤、只使用可部署遥测 / 任务工况、
> 同时与 `D >= 1` 失效定义同语义的健康指标？

能。B1.8 的 Gate 9（`event_observed_hi_p95_above_min`，阈值 0.8，
实测 `p95(HI_B)` 最小值 0.1257）之所以 FAIL，唯一原因是旧 `HI_B`/`Tf`
刻度仍锚在**已废止的 friction threshold** 上，与新失效定义语义脱钩。
B1.9 把监督刻度重锚到与 `D >= 1` 同源的 `L_ref` 归一化累积损伤上，
**未改动失效定义本身**。

## 冻结的主 HI

```
HI_D_obs[k] = clip( cumsum(stress_obs)[k] · dt / SEC_PER_YEAR / L_ref , 0, 1 )
stress_obs[k] = g_duty(drives_obs)[k] · a_T(T_obs[k])
```

`L_ref = 3.0 years`（= B1.8 NOMINAL，未改）；分母是**常量 `L_ref`**，
不是 EOL、不是末值、不是全轨迹最大值。

## 关键数字

| 项 | 值 |
|---|---|
| 轨迹数 | 150（event 71 / censored 79） |
| Gate | 14/14 PASS |
| event EOL 处 HI | median 1.000000 / p10 1.000000 / min 1.000000 |
| censored 末点 HI | median 0.600160 / p90 0.903155 / max 0.984823 / **n(HI≥1) = 0** |
| 前缀不变性 | 逐位相同，`max_abs_diff = 0.000e+00`，容差 **精确 0.0** |
| `x_T` | 10 列，与 B1.8 逐位相同（`max_abs_diff = 0.000e+00`） |
| feature content sha256 | `510dd14e…`（构建侧与审计侧相同） |
| 不一致 Gate | `[]` |
| 训练过任何模型 | 否（`trained_any_model = false`） |

## 必须同时读的诚实声明

`hi_damage_obs` 是 **recomputation，不是 estimation**：它与 B1.8 内部
累积损伤的一致性达机器精度（`corr_min = 1.00000000`，`max_abs_error = 4.8e-09`），
因为后者本来就是由同一组已存储观测量经同一组方程算出的。因此

- **Gate 10（`corr > 0.95`）近乎恒真**，仅标记为 `IMPLEMENTATION_AUDIT`，
  **不得**被引用为"HI 有预测能力"的证据；
- **Gate 7/8 基本由构造保证**（EOL 定义即 `D` 首达 1，而 HI = clip(D,0,1)）；
- 真正有约束力的是 Gate 3/4/5/6/9/11/13（时序单向、不读隐藏真值、
  不读 `b_true`、分母为常量、删失不被强制为 1、前缀不变、`x_T` 无泄漏）。

另有部署前提：四个窗统计量是 `OBSERVABLE_DERIVED` 而非 `OBSERVABLE_DIRECT`
（`corr(|omega_tel|, omega_p95) = 0.7868`，无法由已存储的低速遥测反推），
在轨使用需**星上按窗做 1 Hz 聚合后再下传**。

细节见 [limitations.md](docs/basilisk_b19/limitations.md)。

## 本阶段未做

未训练任何 RUL 模型，未跑 `target_only` / transfer，未产出任何 RUL 精度数字；
未改 `L_ref`、未改失效定义、未重新生成 lifetime dataset、未覆盖 B1.8
feature 文件。旧 friction HI 以 `hi_friction` 原样保留（辅助信号）。

## 产物

| 文件 | 内容 |
|---|---|
| `docs/basilisk_b19/protocol.md` | §2 协议（在任何 HI 数值产生**之前**冻结） |
| `docs/basilisk_b19/baseline_contract.json` | 290 文件 + 378 数值项 = 668 项 |
| `docs/basilisk_b19/observable_input_audit.md` | §3 provenance 审计 |
| `docs/basilisk_b19/damage_proxy_derivation.md` | §2/§4 构造推导 |
| `docs/basilisk_b19/feature_build.md` | 构建侧自查 |
| `docs/basilisk_b19/feature_report.md` | **独立审计侧**报告 |
| `docs/basilisk_b19/selected_hi_definition.md` | 冻结的 HI 定义 |
| `docs/basilisk_b19/limitations.md` | 局限与诚实声明 |
| `docs/basilisk_b19/REPRODUCE.md` | 八步复现序列 |
| `data/features/wheel/basilisk_b19/target_features.h5` | 主产物 |
| `checkpoints/basilisk_b19/frozen_feature_definition.json` | 下游契约 |

## 下游契约（B2 必须遵守）

- 必须使用 `data/features/wheel/basilisk_b19/target_features.h5`；
- 不得改动 `L_ref`、失效定义、HI 方程、归一化分母、`x_T` 列、B1.8 冻结数据；
- `hi_damage_obs` 允许的角色：监督目标 / 健康辅助损失；
- `hi_damage_obs` **禁止**的角色：`x_T` 里的普通输入列（否则构成评价循环）。
