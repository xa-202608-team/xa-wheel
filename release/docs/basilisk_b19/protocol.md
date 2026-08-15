# BASILISK-B1.9 协议（在生成任何 feature 之前冻结）

**本文件在 `hi_damage_obs` 被计算出来之前冻结。** 冻结后 `protocol_hash.json`
记录其 sha256，所有下游脚本每次运行都核验该哈希；不一致即 `B19_PROTOCOL_DRIFT`
并停止。

---

## §0 唯一研究问题

> 能否构造一个**不读取 hidden `D`**、只使用可部署遥测 / 任务工况、
> 同时与 D-based EOL **同语义**的健康指标？

本阶段**不回答**："这个 HI 能不能提高 RUL 精度？" —— 那是 B2 之后的事。
本阶段也**不重新**回答失效定义的 provenance 问题：`D = 1` 仍然是
B1.8 冻结的 `engineering-assumption failure definition`，不是厂家规格。

## §1 B1.9 存在的唯一理由

B1.8 十条特征 Gate 中 9/10 PASS，唯一 FAIL 是 Gate 9
`event_observed_hi_p95_above_min`：旧监督 HI（`hi_source = "Tf"`）的归一化分母
`Tf_fail = K̂_t·I_m,rated − mean(T_cmd)`（等价 `b_fail = 2.4167e-05`）
仍锚在 B1.8 §4 **已废止**的 friction threshold 上，而 EOL 已改为 `D >= 1`。
两者脱钩，导致 18/71 条 event 轨迹在 `D = 1` 时 `b_true < b_fail`，
`p95(HI_B)` 最小仅 **0.1257**（门限 > 0.8）。

B1.9 的做法：**换刻度锚点**，不换失效定义。

## §2 主 HI 定义（冻结）

```
HI_D_obs(t) = clip(
                cumulative_integral( g_duty_obs(t) · a_T(T_obs(t)) ) / L_ref,
                0, 1 )
```

离散实现（与 B1.8 `damage_series` 同一种右端点累加，不换 trapezoid）：

```
HI_D_obs[k] = clip( cumsum(stress_obs)[k] · dt / SEC_PER_YEAR / L_ref , 0, 1 )
stress_obs[k] = g_duty(drives_obs)[k] · a_T(T_obs[k])
```

### 冻结参数（全部沿用 B1.8，不新建第二套）

| 项 | 值 / 来源 |
|---|---|
| `L_ref` | **3.0 years**（= B1.8 NOMINAL `L_ref`） |
| `g_duty` 权重 | speed 0.40 / torque 0.40 / zero-crossing 0.10 / maneuver 0.10（权重和 = 1） |
| `g_duty` 指数 | speed 2.0 / torque 1.0 |
| `g_duty` clip | `[G_DUTY_MIN = 0.05, g_duty_max = 20.0]` |
| Arrhenius | `a_T(T) = exp(-Ea/k_B · (1/T − 1/T_ref))` |
| `Ea` / `T_ref` | 0.3 eV / 293.15 K |
| reference duty | 冻结 `profiles.h5` 决定，无可调参数 |
| 累加方向 | **仅按时间正序（chronological cumulative only）** |

`g_duty` 由 **import** B1 的已验证实现取得（`scripts/basilisk_b1/calibrate_degradation.py::g_duty`），
Arrhenius 由 **import** `src/sim/damage_model.py::arrhenius_accel` 取得。
两者都不在 B1.9 侧复制或重写 —— 这是强制手段，使得"悄悄改权重"在结构上不可能。

## §3 关键区分（本阶段最容易被误读的一点）

> `HI_D_obs` **不是**读取 simulator 的 hidden `D`。

它必须从

- mission-profile observable statistics（`omega_p95` / `torque_p95` /
  `zero_crossing_count` / `maneuver_fraction`）
- measured temperature（遥测 `T`）
- frozen project equations（`g_duty` + Arrhenius + `L_ref`）

**重新独立计算**。§3 的可观测性审计逐项核验四个 `g_duty` 输入与温度的 provenance，
分类为 `OBSERVABLE_DIRECT` / `OBSERVABLE_DERIVED` / `HIDDEN_TRUTH`。
任一输入落在 `HIDDEN_TRUTH` ⇒ `B19_OBSERVABILITY_FAIL` 并停止。

### 必须诚实声明的一点：这是 recomputation，不是 estimation

B1.8 的 `D_true` 本身就是由**同一组已存储的观测量**经**同一组方程**算出的。
因此 `HI_D_obs` 与 `D_true` 的一致性接近机器精度，属于**必然**，
而不是"模型学得好"的证据。

这带来一个诚实的后果：**§8 Gate 10（`corr > 0.95`）几乎是恒真的**。
它只能验证"实现没写错"（`IMPLEMENTATION_AUDIT`），
**不能**被引用为"HI 具有预测能力"的证据。这一点必须写入
`limitations.md` 与 `selected_hi_definition.md`，不得在报告中含糊。

真正有约束力的是 §4（chronological-only + prefix invariance）与
§8 的 Gate 3/4/5/6/9/11/13 —— 它们保证这条 HI 在**部署时可算**且**不泄漏未来**。

## §4 chronological-only 强约束

`HI_D_obs` 在时刻 `t` 只能使用 `<= t` 的数据。

**禁止**：

- full-trajectory normalization
- future max
- final value normalization
- EOL normalization
- trajectory-level min-max using future rows

强制测试（§10 `test_damage_proxy_prefix_invariant`）：
**截断同一轨迹未来 50% 后，前 50% 的 `HI_D_obs` 必须逐位相同**（`atol = 0.0`，
不给容差 —— 因为 cumsum 前缀在数学上就该逐位相同，任何非零差异都说明有未来泄漏）。

## §5 与 hidden `D` 的比较：只允许作为 audit

允许在 feature build **完成之后**、**offline only** 比较
`HI_D_obs` vs `D_true`，报告 `RMSE` / `correlation` / `max_abs_error`。

但 `D_true`：

- 不得进入构造公式；
- 不得进入 `x_T`；
- 不得参与归一化；
- 不得参与 train-time input。

该比较的用途标签固定为 **`IMPLEMENTATION_AUDIT`**，不是训练特征来源。

## §6 保留 friction HI 作为辅助

旧 `HI_B` / `HI_Tf` **不删除**，以 `hi_friction` 保存，用途：

- mechanism comparison
- report ablation

主监督 HI 改为 `hi_damage_obs`。

**禁止覆盖 B1.8 的 feature 文件** —— B1.9 写入独立路径
`data/features/wheel/basilisk_b19/target_features.h5`。

## §7 `x_T` 设计

核心 10 列继续保持，语义与位置不变：

```
I_m, omega, T, T_cmd, sigma_Im, b_hat, dT, omega_err, Tf_ratio, Tf_slope
```

**本阶段不把 `HI_D_obs` 作为普通输入追加。**

理由（必须写明，否则后续容易被"顺手加进去"）：它是
**target / auxiliary health signal**。若把它当普通输入喂进去，
Target-only 模型很可能直接对这条 HI 做积分外推就推出 RUL，
形成**评价循环** —— 模型看似学会了预测，实际只是在复述监督信号的构造式。

HDF5 中分开保存：

```
x_T                 (N, 10)  训练输入
hi_damage_obs       (N,)     主监督 HI
hi_friction         (N,)     辅助 HI（机制对比）
rul                 (N,)     event-observed 才有数值，censored 全 NaN
rul_lower_bound     (N,)     censored 用
event_observed      attr     0/1
mission_features    group    辅助组，不进 x_T
```

## §8 Feature Gate（14 条，全部满足才 READY）

| # | Gate |
|---|---|
| 1 | `hi_damage_obs` finite |
| 2 | `0 <= HI <= 1` |
| 3 | chronological-only |
| 4 | no hidden `D` input |
| 5 | no `b_true` input |
| 6 | no EOL/RUL normalization |
| 7 | event-observed EOL 处 **median HI >= 0.90** |
| 8 | event-observed EOL 处 **p10 HI >= 0.75** |
| 9 | censored trajectory **不得**被强制 `HI = 1` |
| 10 | `correlation(HI_D_obs, D_true) > 0.95`（仅 `IMPLEMENTATION_AUDIT`） |
| 11 | same-prefix invariance PASS |
| 12 | feature content hash reproducible |
| 13 | `x_T` truth leakage = 0 |
| 14 | `mission_features_in_xT = false` |

全部 PASS ⇒ **`B19_FEATURE_READY`**；否则 **`B19_FEATURE_NOT_READY`**。

**禁止调低阈值来通过。** 阈值（0.90 / 0.75 / 0.95）在本文件冻结时即已确定，
在任何 HI 数值被算出来之前。

## §9 输入 / 输出

- 输入：B1.8 冻结的 150 条 trajectory dataset（`data/sim/wheel_basilisk_b18/final`）
- 输出：`data/features/wheel/basilisk_b19/target_features.h5`
- **不得重新生成 lifetime dataset**

## §10 边界

本阶段**不训练 RUL**：不跑 target_only、不跑 transfer、不产出任何 RUL 精度数字。
不改 `L_ref`、不改 failure definition、不改 B1.8 正式数据、不覆盖 B1.8 feature。
