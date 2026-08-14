# B7 frozen result index（§4）

`frozen_result_index_sha256 = 4ee10b5978f9ef93aab9b537aa507d408ce3dee893b4f886f2ab8c5db8738919`

本文件是 B7 全部图表与报告数字的**唯一**来源索引。B7 只读不算：

- `read_only = true`
- `recomputed_any_metric = false`
- `called_trainer_or_evaluator = false`
- `produced_new_metric_definition = false`

允许的计算仅限纯确定性格式转换（per_seed → std、两个已冻结均值相减）。

## 冻结来源与哈希

| key | path | sha256 |
|---|---|---|
| `b18_dataset_audit` | `checkpoints/basilisk_b18/dataset_audit.json` | `d39453f403509a66…` |
| `b18_primary_scenario` | `checkpoints/basilisk_b18/frozen_primary_scenario.json` | `a62e2afe463bee66…` |
| `b19_feature_definition` | `checkpoints/basilisk_b19/frozen_feature_definition.json` | `f8c3deade5ebd715…` |
| `b21_split_manifest` | `docs/basilisk_b21/split_manifest.json` | `d8700e8a7bda7c45…` |
| `b21_summary` | `checkpoints/basilisk_b21/summary.json` | `88008eee21bb6716…` |
| `b5_summary` | `checkpoints/basilisk_b5/summary.json` | `09b21668958bef67…` |
| `b5_formal_metrics` | `checkpoints/basilisk_b5/formal_metrics.json` | `6233e17139c1e606…` |
| `b5_paired_gain` | `checkpoints/basilisk_b5/paired_gain.json` | `d74d1cf943b792dc…` |
| `b5_warning_metrics` | `checkpoints/basilisk_b5/warning_metrics.json` | `382181208850391e…` |
| `b5_lifetime_bins` | `checkpoints/basilisk_b5/lifetime_bins.json` | `508756a7debea385…` |
| `b6_summary` | `checkpoints/basilisk_b6/summary.json` | `3778165f62df7bb3…` |
| `b6_final_verdict` | `checkpoints/basilisk_b6/final_verdict.json` | `5e4c2078cd922e2d…` |
| `b6_paired_statistics` | `checkpoints/basilisk_b6/paired_statistics.json` | `5b4bb315bfefce45…` |
| `b6_lifetime_bins` | `checkpoints/basilisk_b6/lifetime_bins.json` | `770ca1addc53910b…` |
| `b6_warning_metrics` | `checkpoints/basilisk_b6/warning_metrics.json` | `0cdaa27952201068…` |
| `b6_all_metrics` | `checkpoints/basilisk_b6/all_metrics.json` | `3972059675cf911b…` |
| `b6_label_subset_manifest` | `checkpoints/basilisk_b6/label_subset_manifest.json` | `dd729223c6027a25…` |
| `b6_raw_predictions` | `checkpoints/basilisk_b6/all_raw.npz` | `13fb4e7a14455a50…` |

## 七类事实

| 类 | 主题 | 关键结论 |
|---|---|---|
| A | B2.1 generalization / coverage-aware trajectory split | B21_GENERALIZATION_PASS |
| B | B5 full-label confirmatory transfer comparison | B5_NO_POSITIVE_TRANSFER |
| C | B6 failure-label scarcity formal matrix | B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER |
| D | B6 prognostic warning metrics | 学习方法 PRIMARY 档 miss 0.51–0.61，不满足可部署告警要求 |
| E | B6 lifetime-bin resolved RMSE | bin 边界 [26846.0, 37395.0]，`recomputed_from_b6_test = false` |
| F | final engineering recommendation | damage_extrapolation |
| G | final flywheel transfer conclusion | NO_POSITIVE_TRANSFER_SUPPORTED |

## B5 全标签确认性对比（Table A 来源）

| method | info macro RMSE | std | corr | coverage | miss |
|---|---:|---:|---:|---:|---:|
| `damage_extrapolation` | 0.054312 | 0.000000 | 0.9858 | 1.0000 | 0.0000 |
| `target_only` | 0.241024 | 0.005696 | 0.8880 | 0.4800 | 0.5200 |
| `source_mmd_finetune` | 0.241607 | 0.004066 | 0.9114 | 0.4514 | 0.5486 |
| `source_finetune` | 0.242659 | 0.005291 | 0.8980 | 0.3886 | 0.6114 |

## B6 标签稀缺矩阵（Table B 来源，info macro RMSE）

| n_event | role | target_only | source_ft | source_mmd | const | damage |
|---:|---|---:|---:|---:|---:|---:|
| 3 | SECONDARY | 0.335007 | 0.357837 | 0.317366 | 0.288629 | 0.054312 |
| 5 | PRIMARY | 0.339625 | 0.354721 | 0.310147 | 0.288629 | 0.054312 |
| 10 | SECONDARY | 0.298031 | 0.327030 | 0.292611 | 0.288629 | 0.054312 |
| 21 | SECONDARY | 0.238093 | 0.241492 | 0.244525 | 0.288629 | 0.054312 |

## B6 配对增益（Table C 来源，gain = target_only − transfer）

| n_event | method | mean | median | improve | CI95 |
|---:|---|---:|---:|---:|---|
| 3 | `source_finetune` | -0.022830 | -0.021927 | 2/5 | [-0.056178, +0.010519] |
| 3 | `source_mmd_finetune` | +0.017641 | +0.012986 | 3/5 | [-0.017116, +0.054113] |
| 5 **(PRIMARY)** | `source_finetune` | -0.015096 | -0.010302 | 2/5 | [-0.029398, -0.001197] |
| 5 **(PRIMARY)** | `source_mmd_finetune` | +0.029478 | +0.043665 | 3/5 | [+0.001531, +0.057866] |
| 10 | `source_finetune` | -0.028999 | -0.021720 | 0/5 | [-0.046837, -0.015607] |
| 10 | `source_mmd_finetune` | +0.005420 | +0.010316 | 4/5 | [-0.033723, +0.037254] |
| 21 | `source_finetune` | -0.003399 | -0.003983 | 1/5 | [-0.005659, -0.000353] |
| 21 | `source_mmd_finetune` | -0.006432 | -0.001376 | 1/5 | [-0.017535, +0.001088] |

## PRIMARY 档（n_event_labeled = 5）主指标

| method | info macro RMSE |
|---|---:|
| `target_only` | 0.339625 |
| `source_finetune` | 0.354721 |
| `source_mmd_finetune` | 0.310147 |
| `const_mean_info` | 0.288629 |
| `damage_extrapolation` | 0.054312 |

## 边界措辞（不可改写）

- 结论是**未能证明正向迁移**，而不是证明迁移无效。
- 英文：No positive transfer was supported by the frozen confirmatory protocol.
- 中文：在本研究冻结的确认性协议下, 未获得足以支持正向迁移的证据。
- 条件性信号：在低失效标签 n=5 条件下, MMD 出现方向一致的条件性信号 (mean +0.029478, median +0.043665, CI95 lower +0.001531, 三个 lifetime bin gain 均非负), 但未通过全部预登记门槛 (improve 3/5, corr, catastrophic, beats-const 均未过), 因此不足以支持正向迁移结论。
- 禁止表述：“迁移显著提升”、“迁移明显优于”、“MMD 显著有效”、“证明迁移有效”、“达到可上线部署要求”

## D>=1 的语义

- EN: D>=1 is a project-defined simulated failure state, not a manufacturer hardware failure specification.
- ZH: D>=1 是项目定义的仿真失效状态, 不是制造商硬件失效规格。
- Basilisk 负责 attitude dynamics / controller / wheel speed / command torque / mission mode / duty; 退化模型、D(t)、EOL 与 RUL 标签由本项目自建模型定义。

## 工程推荐

- `ENGINEERING_RECOMMENDATION = damage_extrapolation`
- 理由：damage_extrapolation 在 PRIMARY 档 RMSE 0.054312, 显著优于全部学习方法。它不是 '学习算法失败后被拿出来救场的基线', 而是在本仿真域中结构先验最强的方法 —— 因为仿真的累计损伤方程本身已知, physics baseline 直接持有该结构。因此它被明确列为工程推荐。
- 局限：局限: 现实飞轮的真 damage law 并未知。该基线代表**仿真内的** model-informed upper-quality reference, 不能声称在真实在轨场景中也能达到同样精度。
