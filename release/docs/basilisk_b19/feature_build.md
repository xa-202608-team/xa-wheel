# BASILISK-B1.9 §6/§7/§9 特征构建报告

| 字段 | 值 |
|---|---|
| 输入 (B1.8 冻结) | `data/sim/wheel_basilisk_b18/final` |
| 输出 | `data/features/wheel/basilisk_b19/target_features.h5` |
| n_traj | 150 (event 71 / censored 79) |
| 主监督 HI | `hi_damage_obs` |
| 辅助 HI | `hi_friction` (旧 friction 口径, 原样保留) |
| x_T 列 | `['I_m', 'omega', 'T', 'T_cmd', 'sigma_Im', 'b_hat', 'dT', 'omega_err', 'Tf_ratio', 'Tf_slope']` |
| x_T 与 B1.8 逐位相同 | **True** |
| feature content_sha256 | `510dd14e5f3f6ae214c2b2ed85f3eb0227504f4ccc0529ace233d5a26f378cba` |
| verdict | `B19_FEATURE_READY` |

## 1. 为什么换监督 HI

B1.8 Gate 9 FAIL 的根因: 旧 `hi_b` 的归一化分母 `Tf_fail` 仍锚在**已废止**
的 friction threshold 上, 而失效定义已改为 `D >= 1`。两者不同源, 导致
18/71 条 event 轨迹在失效时
摩擦仍远低于旧阈值, `p95(hi_b)` 最小仅 0.1257。

B1.9 换刻度锚点 (`L_ref` 归一的累积损伤), **不换失效定义**。

## 2. §7 x_T 设计

- 核心 10 列语义与位置不变: `['I_m', 'omega', 'T', 'T_cmd', 'sigma_Im', 'b_hat', 'dT', 'omega_err', 'Tf_ratio', 'Tf_slope']`;
- `build_features` 签名 = `['df', 'sim_cfg']` —— 不含 `params`, 结构上拿不到真值;
- `damage_proxy_in_xT = False` —— 主监督 HI **不进输入**, 理由见 §7 (评价循环);
- `mission_features_in_xT = False` (150/150 条搬入辅助组)。

## 3. HI 数值分布

| 组 | median | p10/p90 | min | max |
|---|---|---|---|---|
| `hi_damage_obs` @EOL (event) | 1.000000 | p10 1.000000 | 1.000000 | 1.000000 |
| `hi_damage_obs` @末点 (censored) | 0.600160 | p90 0.903155 | 0.274118 | 0.984823 |
| `hi_friction` (全体) | - | - | 0.000000 | 1.000000 |

- censored 中被强制 `HI = 1` 的条数: **0** (Gate 9 要求 = 0)

## 4. §5 IMPLEMENTATION_AUDIT (离线, 不进特征)

- `corr` 最小 1.00000000 / 中位 1.00000000 (门限 > 0.95)
- `RMSE` 最大 2.7368e-09; `max_abs_error` 最大 4.7996e-09

> **这是 recomputation 而非 estimation。** 详见 `limitations.md` —— 该
> 相关性接近 1 属必然, 只能证明实现没写错, 不能被引用为 HI 有预测能力。

## 5. §4 前缀不变性

- 抽查 10 条, max_abs_diff = **0.000e+00**, 逐位相同 = **True** (容差 0.0)

## 6. §8 Feature Gate (14 条)

| # | gate | 结果 | 详情 |
|---|---|---|---|
| 1 | `hi_damage_obs_finite` | PASS | HI 全部有限 = True; x_T NaN 0 / Inf 0 (删失 rul 的 NaN 单独统计, 不计入此处) |
| 2 | `hi_in_unit_interval` | PASS | 全部落在 [0.0, 1.0]; 单调非降 = True |
| 3 | `chronological_only` | PASS | 截断未来 50% 后前缀逐位相同 (10 条, max_abs_diff 0.000e+00); reference 与 §2 冻结值偏差 0.000e+00 |
| 4 | `no_hidden_damage_input` | PASS | x_T 无 damage 类列; 构造函数源码不含 truth/b_true/attrs 访问 = True; 输出无 truth 组 = True |
| 5 | `no_b_true_input` | PASS | b_true 仅作独立 dataset 供离线校核, 不在 x_T 内; 构造式不读它 |
| 6 | `no_eol_or_rul_normalization` | PASS | 归一化分母 = L_ref (3.0 y, 常量), 非 EOL/RUL/末值; x_T 无 eol/rul 列; 禁用归一化 = ['full_trajectory_normalization', 'future_max', 'final_value_normalization', 'eol_normalization', 'trajectory_level_min_max_using_future_rows'] |
| 7 | `event_eol_median_hi_above_min` | PASS | event n=71, median HI@EOL = 1.000000 (>= 0.9) |
| 8 | `event_eol_p10_hi_above_min` | PASS | p10 HI@EOL = 1.000000 (>= 0.75); min = 1.000000 |
| 9 | `censored_not_forced_to_one` | PASS | censored 79 条中 HI>=1 的 0 条; 末点 HI max = 0.984823, median 0.600160 |
| 10 | `implementation_audit_corr_above_min` | PASS | corr min 1.00000000 (> 0.95); **仅 IMPLEMENTATION_AUDIT** —— 因 HI 是 recomputation, 该项近乎恒真, 不得引作预测能力证据 |
| 11 | `same_prefix_invariance` | PASS | 10/10 条逐位相同, 容差 0.0 |
| 12 | `feature_content_hash_reproducible` | PASS | 同文件两次重算一致: 510dd14e5f3f6ae2… |
| 13 | `xt_truth_leakage_zero` | PASS | 非 allowlist dataset []; 真值 attrs []; 黑名单命中 x_T 列 []; build_features 签名 ['df', 'sim_cfg'] (无 params) |
| 14 | `mission_features_not_in_xt` | PASS | mission_features_in_xT = False, damage_proxy_in_xT = False; 辅助组 150/150 条 |

**14/14 PASS** — verdict: `B19_FEATURE_READY`

## 7. 无泄漏证据

- 非 allowlist dataset: `[]`
- 真值 attrs 残留: `[]`
- 黑名单命中 x_T 列: `[]`
- truth 组存在于输出: []
- `x_T` 与 `hi_damage_obs` 的最大逐列相关: 0.999872 (仅描述性; 输入本就与健康状态相关, 这不是泄漏判据 —— 判据是 hi 不在 x_T 内且构造不读 truth)

## 8. 右删失处理

- censored 79 条, `rul` 全 NaN = True —— 不伪造 EOL;
- 仅给 `rul_lower_bound`; Gate 的 NaN 计数只针对 `x_T`。

## 9. 边界

- 主监督 HI 换成与 D>=1 同源的可观测损伤代理; x_T 十列逐位不变; hi_damage_obs 不进 x_T (它是 target / auxiliary health signal, 进输入会造成评价循环); 旧 friction HI 原样保留为 hi_friction; truth 侧的累积损伤与 b_true 只用于label 构建与 IMPLEMENTATION_AUDIT, 绝不进入 x_T; 14 条 Gate 不满足即报 NOT_READY 并停止, 不调阈值不调参。
- 本阶段**不训练**任何 RUL 模型 (§12): 无 target_only / transfer。
