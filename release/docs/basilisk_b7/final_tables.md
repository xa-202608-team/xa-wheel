# B7 四张主表（§5）

所有数字来自 `checkpoints/basilisk_b7/frozen_result_index.json`（`sha256 = 4ee10b5978f9ef93…`），B7 未重算任何指标。

- `FINAL_TRANSFER_CONCLUSION = NO_POSITIVE_TRANSFER_SUPPORTED`
- `ENGINEERING_RECOMMENDATION = damage_extrapolation`

### Table A — B5 full-label confirmatory comparison (target train labels: 21 event + 24 censored)

判定：**`B5_NO_POSITIVE_TRANSFER`**。主指标 = `test_info_trajectory_macro_rmse`，5 个确认性 seed [112, 113, 114, 115, 116]。

| method | info macro RMSE | std | macro corr | warning coverage | miss rate | catastrophic rate |
|---|---:|---:|---:|---:|---:|---:|
| `damage_extrapolation (physics)` | 0.054312 | 0.000000 | 0.9858 | 1.0000 | 0.0000 | 0.0000 |
| `target_only` | 0.241024 | 0.005696 | 0.8880 | 0.4800 | 0.5200 | 0.0000 |
| `source_mmd_finetune` | 0.241607 | 0.004066 | 0.9114 | 0.4514 | 0.5486 | 0.0000 |
| `source_finetune` | 0.242659 | 0.005291 | 0.8980 | 0.3886 | 0.6114 | 0.0057 |

预登记门槛通过数：`source_finetune` 1/8，`source_mmd_finetune` 3/8。

> const_mean_info 作为参考下界列出, 不参与工程推荐候选集; coverage 与 miss 必须成对阅读。

### Table B — B6 failure-label scarcity matrix (primary metric: test info-trajectory macro RMSE)

PRIMARY 判定：**`B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER`**（PRIMARY = n_event_labeled 5）。

| n_event | role | target_only | source_finetune | source_mmd_finetune | const_mean_info | damage_extrapolation |
|---:|---|---:|---:|---:|---:|---:|
| 3 | SECONDARY | 0.335007 | 0.357837 | 0.317366 | 0.288629 | 0.054312 |
| **5** | **PRIMARY** | 0.339625 | 0.354721 | 0.310147 | 0.288629 | 0.054312 |
| 10 | SECONDARY | 0.298031 | 0.327030 | 0.292611 | 0.288629 | 0.054312 |
| 21 | SECONDARY | 0.238093 | 0.241492 | 0.244525 | 0.288629 | 0.054312 |

稀缺轴 = `event_observed_failure_labelled_trajectories`，**不是** `total_target_trajectories`；每一档都保留全部 24 条 right-censored 训练轨迹。

> 稀缺轴是 event-observed 失效标签轨迹数, 不是目标域总轨迹数; 24 条 right-censored 训练轨迹在每一档都全部保留。damage_extrapolation 在四档全部领先所有学习方法。

### Table C — paired transfer gain vs failure-label budget (gain = target_only RMSE − transfer RMSE; positive value favors transfer)

`gain = RMSE_target_only − RMSE_source_method`，**正数 = 迁移方法更好**。符号在 protocol 冻结时已固定，不得反转；配对只在**同一标签档、同一 seed** 内进行。误差区间为 B6 已冻结的 95% 配对 bootstrap（one paired difference per seed (never per time point)），B7 未重新 bootstrap。

**source_finetune**

| n_event | role | mean gain | median gain | improve seeds | CI95 |
|---:|---|---:|---:|---:|---|
| 3 | SECONDARY | -0.022830 | -0.021927 | 2/5 | [-0.056178, +0.010519] |
| **5** | ◀ **PRIMARY** | -0.015096 | -0.010302 | 2/5 | [-0.029398, -0.001197] |
| 10 | SECONDARY | -0.028999 | -0.021720 | 0/5 | [-0.046837, -0.015607] |
| 21 | SECONDARY | -0.003399 | -0.003983 | 1/5 | [-0.005659, -0.000353] |

**source_mmd_finetune**

| n_event | role | mean gain | median gain | improve seeds | CI95 |
|---:|---|---:|---:|---:|---|
| 3 | SECONDARY | +0.017641 | +0.012986 | 3/5 | [-0.017116, +0.054113] |
| **5** | ◀ **PRIMARY** | +0.029478 | +0.043665 | 3/5 | [+0.001531, +0.057866] |
| 10 | SECONDARY | +0.005420 | +0.010316 | 4/5 | [-0.033723, +0.037254] |
| 21 | SECONDARY | -0.006432 | -0.001376 | 1/5 | [-0.017535, +0.001088] |

**PRIMARY 档 MMD 的条件性信号（必须带限定词阅读）**

在 n_event_labeled = 5 这一 PRIMARY 档，`source_mmd_finetune` 的 mean gain = +0.029478、median gain = +0.043665、CI95 lower = +0.001531 均为正；**但** 仅 3/5 个 seed 改善，预登记门槛通过 6/10（未通过条件编号 [3, 6, 7, 10]）。该结果只能表述为条件性低标签信号，**不得**表述为显著正迁移。

> PRIMARY = n_event_labeled 5, 在看到任何 B6 结果之前固定; 其余三档为 SECONDARY, 只用于趋势描述, 不能覆盖 PRIMARY 判定。

### Table D — PRIMARY level (n_event_labeled = 5): lifetime-bin RMSE and prognostic warning metrics

寿命分箱边界 `[26846.0, 37395.0]` 来自 `docs/basilisk_b21/split_manifest.json::lifetime_bins.event`，`recomputed_from_b6_test = false`；test 各箱轨迹数 {'short': 12, 'medium': 11, 'long': 12}。

| method | RMSE short | RMSE medium | RMSE long | warning coverage | miss rate | false alarm | PH |
|---|---:|---:|---:|---:|---:|---:|---:|
| `damage_extrapolation (physics)` | 0.048726 | 0.051068 | 0.062872 | 1.0000 | 0.0000 | 0.0256 | 0.0000 |
| `target_only` | 0.324676 | 0.334724 | 0.359067 | 0.4286 | 0.5714 | 0.0154 | -0.0377 |
| `source_mmd_finetune` | 0.315279 | 0.304321 | 0.310355 | 0.3943 | 0.6057 | 0.0154 | -0.0176 |
| `source_finetune` | 0.341569 | 0.350776 | 0.371489 | 0.4914 | 0.5086 | 0.0205 | -0.0651 |

分箱增益非负的箱数：`source_finetune` 0/3，`source_mmd_finetune` 3/3。

> 学习方法在 PRIMARY 档 miss rate 约 0.51-0.61, 即超过半数失效轨迹未能在 EOL 前发出有效告警, 当前不满足可部署告警要求。不得因 RMSE 数值较小就声称可直接投入部署。
>
> const_mean_info 不参与本表 —— 它没有随时间变化的告警行为; PH 只在 event-observed 轨迹上计算, right-censored 轨迹排除, 不伪造 EOL。damage_extrapolation 必须留在正表内。
