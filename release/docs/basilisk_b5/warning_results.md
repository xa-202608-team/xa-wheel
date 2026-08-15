# BASILISK-B5 报警 / 预后指标（§13）

**上级文档**：`results.md` ｜ **判定不在本页产生**（本页只出数字，§9 条件 7 的输入见第 6 节）

- evaluator：**冻结的 S4/Basilisk** —— `warning_lead_time` / `prognostic_horizon` /
  `alpha_lambda_accuracy` / `convergence_metric`，本阶段未改一行
- `split_sha256` = `23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932`
- 正式 seed：112 / 113 / 114 / 115 / 116

预后参数（走 config，未硬编码）：

| 参数 | 值 |
|------|----|
| `rul_threshold` | 0.2 |
| `persistence` | 3 |
| `alpha` | 0.2 |
| `lambdas` | 0.3 / 0.5 / 0.7 |
| `absolute_floor` | 0.02 |
| `late_from` | 0.5 |

---

## 0. 三条纪律（先读这个再读表）

**NaN 纪律**：NaN 一律保留 NaN，并同时给出 `n_evaluable` / `eligible_count`。
绝不把 NaN 转成 0 —— 不可评估与指标恰好为 0 是两件完全不同的事，把前者写成后者会让
读表的人以为 PH = 0（预警毫无提前量），而真相是这条轨迹右删失、根本没有真 EOL 可比。

**miss 纪律**：coverage 与 miss rate **必须成对出现**。不得只报成功检测数而隐藏
miss rate —— 那会把 warning 的可用性夸大。`warning_valid_count = 5/5` 只说明这组指标
口径自洽（coverage + miss = 1 等），**不等于 warning 可用**。

**删失纪律**：右删失 test 轨迹不得伪造 EOL 指标。PH / α-λ 只在 event-observed
轨迹上计算，删失轨迹计入 `n_censored_excluded`，**不进分子也不进分母**。

test 侧构成：74 条轨迹 = **35 条 event** + **39 条右删失**。
每个 seed 每个组的 `n_evaluable_traj = 35`、`n_censored_excluded = 39`，五个 seed 全一致。

---

## 1. 报警覆盖与漏报（成对）

| 组 | coverage | miss rate | coverage (before EOL) | miss rate (before EOL) | false alarm |
|----|--------:|---------:|---------------------:|----------------------:|-----------:|
| `target_only` | 0.4800 | **0.5200** | 0.3657 | 0.6343 | 0.0256 |
| `source_finetune` | 0.3886 | **0.6114** | 0.3486 | 0.6514 | 0.0256 |
| `source_mmd_finetune` | 0.4514 | **0.5486** | 0.3657 | 0.6343 | 0.0462 |
| `const_mean_info` | 0.0000 | **1.0000** | 0.0000 | 1.0000 | 0.0000 |
| `damage_extrapolation` | 1.0000 | **0.0000** | 1.0000 | 0.0000 | 0.0256 |

逐 seed（112 / 113 / 114 / 115 / 116）：

| 组 | coverage 逐 seed | miss rate 逐 seed |
|----|------------------|-------------------|
| `target_only` | 0.4571 / 0.5429 / 0.3429 / 0.5143 / 0.5429 | 0.5429 / 0.4571 / 0.6571 / 0.4857 / 0.4571 |
| `source_finetune` | 0.3714 / 0.3714 / 0.3143 / 0.3429 / 0.5429 | 0.6286 / 0.6286 / 0.6857 / 0.6571 / 0.4571 |
| `source_mmd_finetune` | 0.6286 / 0.2286 / 0.4286 / 0.3714 / 0.6000 | 0.3714 / 0.7714 / 0.5714 / 0.6286 / 0.4000 |

**必须与 coverage 一起读的话**：三个学习组的 miss rate 都在 **0.52–0.61**，即一半以上
的失效没有被提前报出来。这不是一个可部署的报警器。`source_finetune` 最差（0.6114），
`source_mmd_finetune` 居中（0.5486），`target_only` 最好（0.5200）—— **两个迁移方法的
报警能力都没有超过不迁移的基线。**

`source_mmd_finetune` 的 coverage 逐 seed 跨度极大（0.2286–0.6286），说明它的报警行为
高度依赖训练随机性；这一项的均值 0.4514 不该被当成稳定能力读。

`before EOL` 口径（只算 EOL 之前触发的报警）比全口径更严，三个学习组的 miss rate
进一步升到 0.634–0.651。真正有工程价值的是这一列。

---

## 2. 预后前置期 PH（Prognostic Horizon）

只在 35 条 event 轨迹上计算，39 条删失轨迹排除。

| 组 | PH mean | 逐 seed PH | n_evaluable | n_censored_excluded |
|----|-------:|------------|:-----------:|:-------------------:|
| `target_only` | **−0.026780** | −0.038395 / −0.038332 / 0.000000 / −0.028585 / −0.028585 | 35 ×5 | 39 ×5 |
| `source_finetune` | **−0.017413** | 0.000000 / 0.000000 / −0.028585 / 0.000000 / −0.058482 | 35 ×5 | 39 ×5 |
| `source_mmd_finetune` | **−0.037056** | −0.031360 / −0.038332 / −0.059158 / −0.028585 / −0.027844 | 35 ×5 | 39 ×5 |
| `const_mean_info` | 0.000000 | 0.0 ×5 | 35 ×5 | 39 ×5 |
| `damage_extrapolation` | 0.000000 | 0.0 ×5 | 35 ×5 | 39 ×5 |

**PH 为负 = 报警晚于 RUL 阈值穿越点。** 三个学习组全为负，没有一个能提前预警。

`source_finetune` 的 PH 均值（−0.0174）在三组里"最好"，但它的构成方式必须说清：
五个 seed 里有三个 PH 恰为 **0.000000**，而它同时是 miss rate 最高的一组（0.6114）——
根本没触发报警的轨迹不贡献负前置期。**因此不能把 FT 的 PH 优势解读为预警更早，
它更可能是"报得更少"的副产物。** 这正是 §13 要求 coverage 与 miss 成对读的原因，
也是不能把 PH 单独拿出来排名的原因。

`const_mean_info` 的 PH = 0 是因为它 coverage = 0，从未触发报警。
`damage_extrapolation` 的 PH = 0 则相反：它的 HI 轨迹与真值几乎重合（corr 0.9858），
报警触发点与阈值穿越点同时到达，前置期精确为 0 而非负。**两个 0 的含义完全不同。**

---

## 3. α-λ 精度（α = 0.2）

只在 35 条 event 轨迹上计算，`eligible_count = 35` ×5。

| 组 | λ=0.3 | λ=0.5 | λ=0.7 |
|----|-----:|-----:|-----:|
| `target_only` | 0.7257 | 0.4629 | 0.2400 |
| `source_finetune` | **0.7829** | 0.4743 | **0.2800** |
| `source_mmd_finetune` | 0.7143 | **0.4800** | 0.2629 |
| `const_mean_info` | 0.0000 | 0.0857 | 0.4571 |
| `damage_extrapolation` | 0.5714 | **0.7429** | **0.9714** |

逐 seed 成功计数（`success_count` / 35）：

| 组 | λ=0.3 | λ=0.5 | λ=0.7 |
|----|-------|-------|-------|
| `target_only` | 26/27/25/23/26 | 15/16/18/15/17 | 11/8/9/6/8 |
| `source_finetune` | 26/29/27/29/26 | 17/14/20/16/16 | 8/12/11/11/7 |
| `source_mmd_finetune` | 26/24/25/23/27 | 19/16/15/16/18 | 9/11/7/9/10 |
| `damage_extrapolation` | 20 ×5 | 26 ×5 | 34 ×5 |

三个学习组在 λ=0.7（临近失效）掉到 0.24–0.28，而 `damage_extrapolation` 达到
**0.9714**。α-λ 的意义就在于"越靠近失效越要准"，学习模型恰好在这一段最差。
`const_mean_info` 在 λ=0.7 反而有 0.4571 —— 常数在寿命末段偶然落进容差带，
这是 α-λ 指标本身的退化行为，不是常数预测有用，**不得作为方法优劣依据**。

α-λ 上迁移方法相对 Target-only 的小幅优势（λ=0.3 上 +0.057，λ=0.7 上 +0.040）
不属于 §9 的八条门槛，本阶段**不用它论证正转移**。

---

## 4. 收敛性指标（convergence）

| 组 | macro mean | n_seeds_evaluable | late mean（λ ≥ 0.5 段） | late n_seeds |
|----|----------:|:-----------------:|----------------------:|:------------:|
| `target_only` | 0.210781 | 5 | 0.247884 | 5 |
| `source_finetune` | 0.216385 | 5 | 0.259888 | 5 |
| `source_mmd_finetune` | 0.213609 | 5 | 0.250795 | 5 |
| `const_mean_info` | 0.412346 | 5 | 0.382774 | 5 |
| `damage_extrapolation` | **NaN** | **0** | **NaN** | **0** |

`damage_extrapolation` 的 convergence 为 **NaN**，`n_seeds_evaluable = 0`：其误差曲线
不产生可评估的收敛区间。**保留 NaN，未转成 0** —— 若写成 0 会被误读成"收敛性最优"，
而真相是这一项对它不可评估。

---

## 5. `warning_valid_count`

五个组均为 **5/5**。再次强调：这只表示口径自洽（coverage + miss = 1 等），
**不表示 warning 可用**。真正的可用性判据是第 1 节的 miss rate（0.52–0.61）
与第 2 节的 PH（全为负）。

---

## 6. 给 §9 条件 7 的输入

| 项 | 值 |
|----|----|
| `warning_miss_tol` | 0.05 |
| `target_only_miss_mean` | 0.520000 |
| `ft_miss_mean` | 0.611429 |
| `mmd_miss_mean` | 0.548571 |
| 容许上限（target + tol） | 0.570000 |
| `ft_within_tol` | **False**（0.611429 > 0.570000） |
| `mmd_within_tol` | **True**（0.548571 ≤ 0.570000） |

→ `source_finetune` 条件 7 **未通过**；`source_mmd_finetune` 条件 7 **通过**。
最终判定见 `results.md` §4。

---

数据来源：`checkpoints/basilisk_b5/warning_metrics.json`
（`by_group` / `per_seed` / `gate_inputs` / 三条 discipline 字段）
