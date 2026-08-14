# STATUS — BASILISK-B1.8

**判决: `B18_FEATURE_NOT_READY`**

阶段性质: assumption-transparent cumulative damage model。
**已生成 150 条寿命数据 + 特征; 未训练任何 RUL 模型 (§21)。**

失效定义性质: **`D = 1` 是项目定义的仿真失效状态**, 不是 Honeywell、BCT、
NanoAvionics 或 Basilisk 官方给出的硬件失效阈值。详见
[docs/basilisk_b18/limitations.md](docs/basilisk_b18/limitations.md)。

---

## 1. 本阶段回答的问题

> 在 Basilisk 生成的真实姿态控制/轮载荷工况下, 能否构建一个透明、可复现、
> 物理方向正确的累计损伤仿真场景, 用于后续 RUL 算法研究?

**答: 损伤模型部分成立, 监督 HI 刻度不成立。**

- ✅ 累计损伤模型有效: §14 **14/14 Gate PASS**, §15 ordering PASS,
  `B18_SENSITIVITY_OK`, primary NOMINAL 冻结于结果之前;
- ✅ 正式数据集 150 条通过 §18 审计: event 71 / censored 79, NaN/Inf = 0;
- ❌ 特征层 §20 Gate 9 FAIL: 监督 `HI_B` 的归一化刻度仍锚在**已废止的摩擦阈值**上,
  与新的 `D >= 1` 失效定义不同源。

本阶段**不回答**"这个阈值是不是厂家真实失效阈值" —— 答案明确是
`NO → ASSUMED SIMULATION FAILURE DEFINITION`。

## 2. 判决依据

| Gate 组 | 结果 | 依据 |
|---|---|---|
| §2 baseline contract (before/after) | ✅ PASS | 575 items 全 unchanged |
| §4 派生验证 (`derive_damage_model`) | ✅ PASS | weights_sum = 1 精确, 全部 reference 偏差 `0.000e+00`, `derivation_verified = True` |
| §14 模型有效性 14 条 | ✅ **14/14 PASS** | `B18_SENSITIVITY_OK` |
| §15 ordering sanity | ✅ PASS | FAST ≥ NOMINAL ≥ SLOW (event frac), median EOL 反序成立 |
| §16 primary 冻结 | ✅ `B18_SCENARIO_READY` | NOMINAL, `L_ref = 3.0 y`, `selected_before_results = True` |
| §18 数据审计 | ✅ 报告完成 | 未因结果调模型 |
| §20 特征 10 条 | ❌ **9/10** | Gate 9 `event_observed_hi_p95_above_min` FAIL |

Gate 9 是唯一 FAIL, 直接决定 verdict。

## 3. Gate 9 根因: 语义不一致, 不是可调参数

`HI_B` 走 `hi_source = "Tf"` 口径, 分母
`Tf_fail = K̂_t · I_m,rated − mean(T_cmd)`（等价 `b_fail = 2.4167e-05`）
锚在 §4 已废止的**摩擦失效阈值**上; 而 EOL 已改为 `D >= 1`。两者脱钩:

| 量 | 统计 |
|---|---|
| `b_true(EOL) / b_fail` | min 0.272, median 1.874, max 3.922 |
| `b_true(EOL) < b_fail` | **18 / 71** 条 |
| corr(`b_E/b_fail`, `p95(HI_B)`) | **0.6619** |
| event-observed `p95(HI_B)` | min **0.1257** (门限 > 0.8), median 0.8554 |

`b0` 抽小的轨迹即使损伤走满 `D = 1`, 摩擦力矩仍远未触及旧 `Tf_fail`,
HI 只爬到 0.13 即被截断。

**三条"能让它变绿"的路全部被协议禁止**:

- 改 HI 公式 → 违反 §19 (禁止改 HI 公式);
- 降 `hi_p95_min` 门限 → outcome tuning, 违反 §8/§13/§20;
- 调 `L_ref` / 重抽 `b0` → 违反 §7/§17, 且 §14 明令 Gate 失败禁止回头改 `L_ref`。

故如实上报 NOT_READY, 留给下一阶段以新 provenance 决策解决。

## 4. Provenance 三级分类 (18 项)

| 级别 | 数量 | 内容 |
|---|---|---|
| **A. DIRECT** | 5 | Basilisk profile wheel speed、commanded torque、HR16 `Omega_max`、`u_max`、mission mode |
| **B. PROJECT_DOCUMENTED_ASSUMPTION** | 9 | `b0` range、`Delta` range、Arrhenius 模型、noise model、`Kt`/`Tc` 遥测假设 |
| **C. NEW_B18_ASSUMPTION** | 4 | **累计损伤失效定义 `D>=1`**、life-scale multiplier `L_ref` 等 |

**B/C 级不得写成 DIRECT。** 逐条登记见
[docs/basilisk_b18/assumption_registry.md](docs/basilisk_b18/assumption_registry.md)。

## 5. 损伤模型

```
dD/dt     = stress(t) / L_ref
stress(t) = g_duty(t) · a_T(T(t))          stress >= 0, D(t) 单调递增
b(t)      = b0 · (1 + Delta · D(t))        (D <= 1 区域)
EOL       ⇔ D(t) >= 1                       (唯一判据)
```

reference 恒等式: `g_duty(reference) = 1.0` 且 `T = T_ref` ⇒ `stress = 1` ⇒
`D(t) = t / L_ref`。三场景实测与解析解 `max_abs_err = 0.000e+00`。

温度角色: **environmental / telemetry condition, NOT a physics-derived wheel
thermal state** —— 不得称为 thermal model (§11)。

润滑突变 `U(1.3, 1.8)` **只影响 `b(t)`/telemetry, 不跳变 `D`** (§10)。

## 6. `L_ref` registry (§7, 不允许按 failure fraction 调)

`H = 3.0 years`

| 场景 | 倍数 | `L_ref` |
|---|---|---|
| SLOW | 1.5 H | 4.50 y |
| NOMINAL | 1.0 H | 3.00 y |
| FAST | 0.75 H | 2.25 y |

`primary_scenario = NOMINAL`, 理由逐字: **"reference duty 下设计寿命等于任务仿真
horizon"**。冻结记录 `selected_before_results = True`。

## 7. 数据与哈希

| 项 | 值 |
|---|---|
| protocol sha256 | `e25d6bc84a82ece5fdf64b44dac0fa74e2165a61f9cfca9f5a42b54c8f40cf49` |
| config sha256 | `45beb9d52dd555fe66178c8092f8089b486b2efb38d106545b272bf4fafbc3f0` |
| paired_trajectory_hash | `ff1de559b2746dcb394c4de6d70bda4d96ca594d1b953f58a4fa390366f41cf6` |
| dataset content sha256 | `61678e582f82bd136ffe509de7ec1274c07c39fe791a2a762fcf562a3d36a0f7` |
| feature content sha256 | `4f7b7e3f68685cc933aadcceb63bc6bdb5cc320eaee4a91616dc7174d23711a8` |
| n_traj | 150 (event 71 / censored 79) |
| EOL p5/p50/p95 | 1.272 / 1.782 / 2.764 y (event-observed) |
| early / late | 0.0000 / 0.0704 |

## 8. 测试

`tests/basilisk_b18/` **60 passed, 13 skipped**。§22 要求的 14 个函数名逐字存在
且各一次。

## 9. 历史判决 (全部保持终局, 不得修改或重新解释)

| 阶段 | 判决 |
|---|---|
| BASILISK V1 | `BASILISK_V1_READY` |
| B1 | `B1_CALIBRATION_FAIL` |
| B1.1 | `B11_CALIBRATION_FAIL` |
| B1.2 | `B12_FAILURE_DEFINITION_FAIL` |
| B1.3 | `B13_CALIBRATION_FAIL` |
| B1.4 | `B14_CALIBRATION_FAIL` |
| B1.5 | `B15_FAILURE_SEMANTICS_MISMATCH` |
| B1.6 | `B16_NO_DOCUMENTED_WHEEL` |
| B1.7 | `B17_THERMAL_PROVENANCE_INSUFFICIENT` |
| **B1.8** | **`B18_FEATURE_NOT_READY`** |

## 10. 下一阶段的唯一待决问题

监督 HI 的刻度锚点应该是什么? 三条候选路径, 各自需要独立的 provenance 决策:

1. 让 HI 与 `D` 同源 (HI 直接定义为 `D` 的可观测代理) —— 需新阶段明确
   "HI 定义变更" 的 provenance, 不能在 B1.8 内偷改;
2. 为 `b_fail` 建立与 `D=1` 一致的**假设性**映射, 并标注为 NEW ASSUMPTION;
3. 承认 `Tf` 口径 HI 只适用于 friction-threshold EOL, 在 D-based 场景下改用
   censoring-aware 的 RUL 直接监督, 不经 HI 中介。

在此之前, `data/features/wheel/basilisk_b18/target_features.h5` **不应作为正式
benchmark 使用**。
