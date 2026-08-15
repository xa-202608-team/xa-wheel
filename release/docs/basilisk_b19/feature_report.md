# BASILISK-B1.9 §8 特征 Gate 复核报告 (独立重算)

- protocol sha256: `e95bf867e083aa0d…`
- 特征文件: `data/features/wheel/basilisk_b19/target_features.h5`
- content sha256: `510dd14e5f3f6ae214c2b2ed85f3eb0227504f4ccc0529ace233d5a26f378cba`
- 轨迹: 150 (event 71 / censored 79), x_T [10] 列
- 主监督 HI: `hi_damage_obs`; 辅助 HI: `hi_friction`
- **判定: B19_FEATURE_READY** (14/14)

## 1. 为什么要独立复核

`build_features.py` 的 Gate 是**构建期自证** —— 它手上有内存里的中间量。
本脚本只从落盘 h5 + 源数据 + 源码重新取证, 独立重算全部 14 条, 再与
构建期结论逐条比对。两侧不一致即报 `B19_AUDIT_MISMATCH`, 因为那比任一
单侧 FAIL 更严重。

- 构建期 verdict: `B19_FEATURE_READY`
- 复核 verdict: `B19_FEATURE_READY`
- 结论不一致的 Gate: `[]`
- content hash 两侧相同: True

## 2. 14 条 Gate

| # | gate | 结果 | 详情 |
|---|---|---|---|
| 1 | `hi_damage_obs_finite` | PASS | 落盘 HI 全部有限 = True; x_T NaN 0 / Inf 0; 长度一致性异常 [] |
| 2 | `hi_in_unit_interval` | PASS | [0.0, 1.0] 内; 单调非降 = True |
| 3 | `chronological_only` | PASS | 从源数据独立重算前缀, 10 条逐位相同; max_abs_diff(float32) 0.000e+00 |
| 4 | `no_hidden_damage_input` | PASS | x_T 无 damage 列; 构造函数源码 truth 相关符号命中 = []; 落盘无 truth 组 = True |
| 5 | `no_b_true_input` | PASS | b_true 不在 x_T, 亦不出现在构造函数源码中 |
| 6 | `no_eol_or_rul_normalization` | PASS | 构造函数无 EOL/末值归一化符号 (命中 []); 分母 = L_ref 3.0 (常量, 与 protocol 一致) |
| 7 | `event_eol_median_hi_above_min` | PASS | event n=71, median 1.000000 (>= 0.9) |
| 8 | `event_eol_p10_hi_above_min` | PASS | p10 1.000000 (>= 0.75), min 1.000000 |
| 9 | `censored_not_forced_to_one` | PASS | censored 79 条, HI>=1 的 0 条; 末点 max 0.984824 median 0.600160; rul 全 NaN = True (不伪造 EOL) |
| 10 | `implementation_audit_corr_above_min` | PASS | corr min 1.00000000 (> 0.95); RMSE max 2.737e-09; **IMPLEMENTATION_AUDIT only** |
| 11 | `same_prefix_invariance` | PASS | 容差 0.0; 10 条全部逐位相同 |
| 12 | `feature_content_hash_reproducible` | PASS | 本脚本独立口径两次一致 510dd14e5f3f6ae2…; 与 build 期声明 510dd14e5f3f6ae2… 相同 = True |
| 13 | `xt_truth_leakage_zero` | PASS | 非 allowlist dataset []; 真值 attrs []; 黑名单命中 x_T 列 []; build_features 签名 ['df', 'sim_cfg'] |
| 14 | `mission_features_not_in_xt` | PASS | mission_features_in_xT = False; damage_proxy_in_xT = False; 辅助组 150/150; x_T 列数 10 |

## 3. HI 分布

| 组 | n | median | p10/p90 | min | max |
|---|---|---|---|---|---|
| `hi_damage_obs` @EOL (event) | 71 | 1.000000 | p10 1.000000 | 1.000000 | 1.000000 |
| `hi_damage_obs` @末点 (censored) | 79 | 0.600160 | p90 0.903155 | 0.274118 | 0.984824 |
| `hi_friction` (全体范围) | 150 | - | - | 0.000000 | 1.000000 |

- censored 被强制 `HI = 1` 的条数: **0** (要求 = 0)
- censored `rul` 全 NaN: True —— **不伪造 EOL**; event 轨迹 `rul` 全有限: True

## 4. §5 IMPLEMENTATION_AUDIT

| 量 | 值 |
|---|---|
| corr min | 1.00000000 |
| corr median | 1.00000000 |
| RMSE median | 3.9538e-10 |
| RMSE max | 2.7368e-09 |
| max_abs_error max | 4.7996e-09 |

> Gate 10 近乎恒真: HI_D_obs 是对同一组已存储窗统计量按同一组方程的 recomputation, 不是 estimation。它只能证明实现没写错, 不能作为 HI 具有预测能力的证据。真正有约束力的是 Gate 3/4/5/6/9/11/13。

## 5. 无泄漏证据 (源码级 + 落盘级)

- 构造函数中 truth 相关符号命中: `[]` (要求空)
- 构造函数中 EOL/末值归一化符号命中: `[]` (要求空)
- 落盘非 allowlist dataset: `[]`
- 落盘 truth 组: `[]`
- 落盘真值 attrs: `[]`
- 黑名单命中 x_T 列: `[]`
- `build_features` 签名: `['df', 'sim_cfg']` (无 `params`)

## 6. 边界

- 本脚本只读落盘特征文件与源码, 独立重算 14 条 Gate。它打开 B1.8 源数据的 truth 容器仅为重做 §5 IMPLEMENTATION_AUDIT 的三个报告数字 (corr/RMSE/max_abs), 该数值不写入任何特征文件、不参与归一化、不进 x_T。
- `trained_any_model = False`

## 判定

**B19_FEATURE_READY**
