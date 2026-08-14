# BASILISK-B5 寿命分 bin 结果（§11 / §12）

**上级文档**：`results.md` ｜ **判定不在本页产生**（本页只出数字，判定见 `results.md` §4）

- bin 来源：`docs/basilisk_b21/split_manifest.json` → `lifetime_bins.event`
- bin 边界：**`[26846.0, 37395.0]`**（B2.1 冻结）
- `quantile_basis` = `dataset_level_event_eol_before_training`（在任何训练之前确定）
- `recomputed_from_b5_test` = **False**
- `split_sha256` = `23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932`

> **不得根据 B5 test 分布重算 bin。** 按结果重新分组等于挑一个好看的切法。本页所有 bin
> 归属在 B2.1 划分时就已固定，B5 只是把 74 条 test 轨迹按那套边界落进去。

---

## 1. test 侧 bin 计数

| bin | 边界（EOL） | test event 轨迹数 |
|-----|------------|-----------------:|
| `short` | ≤ 26846.0 | 12 |
| `medium` | (26846.0, 37395.0] | 11 |
| `long` | > 37395.0 | 12 |
| **合计** | | **35** |

74 条 test 轨迹 = 35 条 event + 39 条右删失。分 bin 只在 35 条 event 上进行 ——
**右删失轨迹没有 EOL，不参与 EOL 分 bin，也不为它们编造 EOL。**

---

## 2. 逐 bin 主指标（5 seed 均值）

### 2.1 RMSE（info-zone trajectory-macro）

| bin | `target_only` | `source_finetune` | `source_mmd_finetune` | `const_mean_info` | `damage_extrapolation` |
|-----|-------------:|-----------------:|---------------------:|-----------------:|----------------------:|
| short | 0.223777 | 0.228210 | 0.224698 | 0.288670 | **0.048726** |
| medium | 0.259158 | 0.257123 | 0.252076 | 0.288611 | **0.051068** |
| long | 0.241649 | 0.243850 | 0.248921 | 0.288605 | **0.062872** |

### 2.2 MAE

| bin | `target_only` | `source_ft` | `source_mmd` | `const` | `damage` |
|-----|-------------:|-----------:|------------:|-------:|--------:|
| short | 0.190753 | 0.195414 | 0.193096 | 0.249996 | 0.034949 |
| medium | 0.217655 | 0.215751 | 0.212969 | 0.249945 | 0.036759 |
| long | 0.201086 | 0.201970 | 0.208274 | 0.249940 | 0.043533 |

### 2.3 corr

| bin | `target_only` | `source_ft` | `source_mmd` | `const` | `damage` |
|-----|-------------:|-----------:|------------:|-------:|--------:|
| short | 0.9166 | 0.9189 | 0.9332 | −0.0000 | 0.9877 |
| medium | 0.8887 | 0.8924 | 0.9135 | −0.0000 | 0.9868 |
| long | 0.8587 | 0.8823 | 0.8878 | 0.0000 | 0.9830 |

`const_mean_info` 的 bin 内 corr 为 ±0.0000 而非 NaN：bin 内该组仍是常数预测，
数值上落在 0 附近。全局口径下它是 NaN（见 `results.md` 主表）。

---

## 3. 逐 bin 配对增益（§9 条件 8 的输入）

`gain = RMSE_target_only − RMSE_transfer`，正数 = 迁移在该 bin 改进。

| bin | `gain_ft` | `gain_ft > 0` | `gain_mmd` | `gain_mmd > 0` |
|-----|---------:|:-------------:|----------:|:--------------:|
| short | −0.004433 | 否 | −0.000921 | 否 |
| medium | **+0.002035** | **是** | **+0.007082** | **是** |
| long | −0.002201 | 否 | −0.007272 | 否 |
| **正 bin 数** | **1 / 3** | | **1 / 3** | |

§9 条件 8 要求 **≥ 2/3**。两个方法都只有 1/3 → **条件 8 未通过**。

逐 bin 与物理基线的差距（`transfer_vs_damage = RMSE_damage − RMSE_transfer`）：

| bin | ft | mmd |
|-----|---:|----:|
| short | −0.179484 | −0.175972 |
| medium | −0.206054 | −0.201008 |
| long | −0.180978 | −0.186049 |

三个 bin 内两个迁移方法都远不如物理外推，没有任何一个寿命段例外。

---

## 4. §12 短寿命鲁棒性（short bin 五项必录）

| 组 | mean RMSE | 逐 seed RMSE (112/113/114/115/116) | catastrophic | pred_std | true_std | PSR |
|----|---------:|------------------------------------|-------------:|---------:|---------:|----:|
| `target_only` | 0.223777 | 0.219218 / 0.229730 / 0.218831 / 0.235503 / 0.215605 | 0 | 0.2628 | 0.2887 | 0.9104 |
| `source_finetune` | 0.228210 | 0.227641 / 0.245037 / 0.215425 / 0.238856 / 0.214092 | 0 | 0.2653 | 0.2887 | 0.9192 |
| `source_mmd_finetune` | 0.224698 | 0.227090 / 0.220253 / 0.224974 / 0.228795 / 0.222378 | 0 | 0.2637 | 0.2887 | 0.9136 |
| `const_mean_info` | 0.288670 | 0.288670 ×5 | 0 | 0.0000 | 0.2887 | 0.0000 |
| `damage_extrapolation` | 0.048726 | 0.048726 ×5 | 0 | 0.2933 | 0.2887 | 1.0161 |

读法：

- short bin 内 **三组的 catastrophic 计数都是 0**，短寿命端没有出现灾难性外推。
- 逐 seed 离散度：`source_mmd_finetune` 最窄（0.2203–0.2288，跨度 0.0085），
  `target_only` 中等（0.2156–0.2355，跨度 0.0199），`source_finetune` 最宽
  （0.2141–0.2450，跨度 0.0309）。**MMD 在短寿命端更稳定，但均值仍不优于 Target-only。**
  稳定性不是增益，不能替代 §9 的门槛。
- 三个学习组 PSR 都在 0.91–0.92，略欠预测方差（预测比真值平缓），但都远低于
  `HIGH_VARIANCE_WARNING` 阈值 3。

---

## 5. 关于 `localized transfer benefit` 的判读（§12）

`localized_transfer_benefit`：**ft = False，mmd = False**

§12 说的"局部收益"指增益**只在 short bin 为正**。本阶段的实测恰好相反：

- 唯一为正的 bin 是 **medium**；
- **short bin 上两个迁移方法都略差于 Target-only**（ft −0.004433，mmd −0.000921）。

因此：

> 本阶段**不能**声称"迁移带来全周期提升"，也**不能**声称"迁移带来短寿命局部增益"。
> 唯一诚实的说法是：中寿命段出现了一个孤立的正增益，短寿命与长寿命段为负，
> 三 bin 未构成一致方向。

medium 段的 mmd 增益（+0.007082）在数值上是三 bin 里最大的，但它同时被 long 段
几乎等量的负增益（−0.007272）抵消 —— 这更像是 bin 间噪声再分配，而不是某个寿命段
的系统性收益。

---

## 6. 右删失侧 bin 退化（如实报告）

| 项 | 值 |
|----|----|
| `key` | `observed_duration` |
| `quantile_basis` | `dataset_level_observed_duration_before_training` |
| `edges` | `[52596.0, 52596.0]` |
| `counts` | `short_obs: 0`, `medium_obs: 0`, `long_obs: 79` |
| `degenerate` | **true** |
| `n_effective_bins` | **1** |
| `forbid_fake_eol` | true |

全部 79 条右删失轨迹的观测时长相同（仿真在同一 horizon 截断），三分位边界重合，
三分位分桶退化成单一桶。

**此处如实报告 `degenerate = true` 与实际 bin 数 1，绝不为了凑三个 bin 而编造边界，
也绝不改用任何寿命型代理量 —— 那等于发明 EOL。** 因此右删失侧不产出分 bin RMSE 对比，
本页第 2–5 节的所有分 bin 数字都只基于 35 条 event 轨迹。

---

## 7. 空 bin 约定

`empty_bin_note`：空 bin 返回 **NaN + n = 0**，绝不写 0。没有可评估轨迹与 RMSE
恰好为 0 是两件完全不同的事。本阶段三个 event bin 的 `n_traj_in_bin` 分别为
12 / 11 / 12，均非空，未触发该分支；约定仍写在产物里以便复核。

---

数据来源：`checkpoints/basilisk_b5/lifetime_bins.json`
（`by_group` / `gain_by_bin` / `short_life` / `censored_bins`）
