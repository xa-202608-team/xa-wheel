# 飞轮线迁移学习最终结论（BASILISK-B6 §21）

```
FINAL_TRANSFER_CONCLUSION = NO_POSITIVE_TRANSFER_SUPPORTED
```

（允许取值只有两个：`NO_POSITIVE_TRANSFER_SUPPORTED` / `CONDITIONAL_LOW_LABEL_TRANSFER_ONLY`；本阶段取前者，后者未成立。）

| 项 | 值 |
|---|---|
| B6 PRIMARY 判定 | `B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER` |
| B5/B6 关系 | 情况 **A**（B5 negative + B6 primary negative） |
| 合并表述 | `NO_POSITIVE_TRANSFER_SUPPORTED` |
| ENGINEERING_RECOMMENDATION | `damage_extrapolation` |
| PRIMARY 档 | `n_event_labeled = 5`（event 5 + censored 24 = 29 条 train） |
| 正式 seeds | `[122, 123, 124, 125, 126]` |

下面按 §21 要求，**分别**解释五件事。

---

## 1. B5 全量数据证据（b5_full_data_evidence）

B5 在**完整** B2.1 train（45 条：event 21 + censored 24）上做正式迁移检验，
seeds `[112..116]`，输入 `CORE_ONLY` / 12 维：

| 方法 | 平均 test info macro RMSE |
|---|---|
| `target_only` | 0.241024 |
| `source_finetune` | 0.242659（gain −0.001635，improve 1/5，CI95 [−0.003828, +0.000704]） |
| `source_mmd_finetune` | 0.241607（gain −0.000583，improve 2/5，CI95 [−0.006589, +0.005423]） |
| `damage_extrapolation` | 0.054312 |
| `const_mean_info` | 0.288629 |

结论 `B5_NO_POSITIVE_TRANSFER`（FT 1/8 条通过，MMD 3/8 条通过），
工程推荐 `damage_extrapolation`。

**该结论永久有效，不因 B6 被覆盖。** 即"在目标域失效标签充足时，
源域预训练相对 Target-only 没有正向收益"。

## 2. B6 低标签证据（b6_low_label_evidence）

B6 把稀缺压在 **event-observed 失效标签数**上（`n_event_labeled ∈ {3,5,10,21}`），
每档保留**全部 24 条** censored，因此 train 为 27 / 29 / 34 / 45。
4 档 × 3 方法 × 5 seed = **60 次训练**，无自动重试，全部 cell 公平性 PASS。

### PRIMARY（n=5）十条门槛

`source_finetune` **3/10 通过** → FAIL：

| # | 条件 | 结果 |
|---|---|---|
| 1 | mean gain > 0 | FAIL −0.015096 |
| 2 | median gain > 0 | FAIL −0.010302 |
| 3 | improve ≥ 4/5 | FAIL 2/5 |
| 4 | bootstrap CI95 下界 > 0 | FAIL −0.029398 |
| 5 | ≥ 2/3 寿命 bin gain ≥ 0 | FAIL 0/3 |
| 6 | macro corr 不低于 target − 0.02 | PASS 0.8641 vs 0.8576 |
| 7 | catastrophic 不更差 | FAIL 0.005714 vs 0.000000 |
| 8 | warning miss 不超 target + 0.05 | PASS 0.5086 vs 0.5714 |
| 9 | 同数据同预算公平性 | PASS |
| 10 | 打赢 `const_mean_info` | FAIL +0.066091（RMSE 更高） |

`source_mmd_finetune` **6/10 通过** → FAIL：

| # | 条件 | 结果 |
|---|---|---|
| 1 | mean gain > 0 | **PASS +0.029478** |
| 2 | median gain > 0 | **PASS +0.043665** |
| 3 | improve ≥ 4/5 | FAIL 3/5 |
| 4 | bootstrap CI95 下界 > 0 | **PASS +0.001531** |
| 5 | ≥ 2/3 寿命 bin gain ≥ 0 | **PASS 3/3** |
| 6 | macro corr 不低于 target − 0.02 | FAIL 0.8161 vs 0.8576 − 0.02 |
| 7 | catastrophic 不更差 | FAIL 0.005714 vs 0.000000 |
| 8 | warning miss 不超 target + 0.05 | PASS 0.6057 vs 0.6214 |
| 9 | 同数据同预算公平性 | PASS |
| 10 | 打赢 `const_mean_info` | FAIL +0.021517（RMSE 更高） |

**MMD 臂是本项目迄今最接近正向的迁移证据**：均值、中位数、bootstrap CI 下界
三项方向性指标全部为正，且三个寿命 bin 全部非负。但它在四条上失败——
逐 seed 一致性只有 3/5、形状相关性下降超过容差、灾难性误差率高于
Target-only、且 RMSE 仍高于常数标尺。十条必须全过，故判 FAIL。

> **第 10 条是最关键的否决理由**：n=5 档三个学习方法的 RMSE
> （0.3396 / 0.3547 / 0.3101）**全部高于** `const_mean_info` 的 0.288629。
> 在只有 5 条失效标签时，三个学习臂都还没有稳定超过"输出常数"这一最低标尺，
> 因此讨论"迁移是否带来增益"在工程意义上是次要问题。

### SECONDARY（趋势，不推翻 primary）

| n_event | FT mean gain (improve) | MMD mean gain (improve) |
|---|---|---|
| 3 | −0.022830 (2/5) | +0.017641 (3/5) |
| **5 (PRIMARY)** | **−0.015096 (2/5)** | **+0.029478 (3/5)** |
| 10 | −0.028999 (0/5) | +0.005420 (4/5) |
| 21 | −0.003399 (1/5) | −0.006432 (1/5) |

MMD 在 3 / 5 / 10 三档均值为正、n=21 转负，最大均值增益出现在 **n=5**。
仅 n=10 达到 improve 4/5，因此记录
`SECONDARY_LOW_LABEL_SIGNAL_AT_N10 [source_mmd_finetune]`，
明确标记 `NOT_PRIMARY_CONFIRMATORY_EVIDENCE`。

**§16 纪律：secondary 不推翻 primary。** 即使 n=10 的 MMD 达到 4/5，
primary n=5 未通过，正式判定仍然是
`B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER`。

"标签越少增益越大"的迹象判定为 **False**（最大增益在 n=5 而非最小档 n=3）。
只有 4 个采样点，**未拟合任何趋势线**，以上仅为描述性观察。

## 3. 寿命分箱行为（lifetime_bin_behavior）

PRIMARY 档（bin 边界 `[26846, 37395]`，取自 B2.1 冻结件，未从 B6 test 重算）：

| bin | n | target_only | source_ft | source_mmd | gain_ft | gain_mmd |
|---|---|---|---|---|---|---|
| short | 12 | 0.324676 | 0.341569 | 0.315279 | −0.016893 | **+0.009397** |
| medium | 11 | 0.334724 | 0.350776 | 0.304321 | −0.016052 | **+0.030403** |
| long | 12 | 0.359067 | 0.371489 | 0.310355 | −0.012422 | **+0.048712** |

两点值得记录：

1. **FT 在三个 bin 全部为负**，不存在"只在某一寿命段有效"的局部收益
   （`localized_transfer_benefit.ft = false`）。
2. **MMD 在三个 bin 全部为正，且增益随寿命增长而增大**（+0.009 → +0.030 → +0.049）。
   这不是局部收益（`localized_transfer_benefit.mmd = false`，因为它并非"仅 short 为正"），
   而是一个全段一致、在长寿命段更明显的方向性改善。这是 MMD 臂通过第 5 条的原因，
   也是本阶段唯一具有物理可解释性的正向迹象——但它没能同时满足一致性与
   形状保真度要求。

## 4. 告警行为（warning_behavior）

PRIMARY 档告警与预测视界（漏报率与覆盖率成对报出，不隐藏 miss）：

| 方法 | coverage | **miss rate** | false alarm | PH |
|---|---|---|---|---|
| `target_only` | 0.4286 | **0.5714** | 0.0154 | −0.0377 |
| `source_finetune` | 0.4914 | **0.5086** | 0.0205 | −0.0651 |
| `source_mmd_finetune` | 0.3943 | **0.6057** | 0.0154 | −0.0176 |
| `damage_extrapolation` | 1.0000 | **0.0000** | 0.0256 | 0.0000 |

- 三个学习方法的漏报率都在 **0.51–0.61** 区间，即一半以上的失效未能提前告警。
  在只有 5 条失效标签的条件下，这三个模型都不具备可部署的告警能力。
- 两个 source 臂都通过了第 8 条（miss 未超 target + 0.05），但这是因为
  **Target-only 自身的漏报率已经很高**，"不更差"是一个很低的门槛，
  不构成正面证据。
- PH 均为负值（学习方法在真实 EOL 之后才越过告警阈值）；
  `damage_extrapolation` 的 PH 为 0、漏报为 0。
- 所有依赖真 EOL 的指标对右删失轨迹保持 `NaN`，未转 0；
  `n_censored_excluded` 已逐项记录。

## 5. damage 基线比较（damage_baseline_comparison）

damage 基线留在**主表内**逐档报出，未降级为脚注。
（差值定义：正数 = 该学习方法 RMSE 更高，即不如物理外推。）

| n_event | target − damage | ft − damage | mmd − damage | damage RMSE |
|---|---|---|---|---|
| 3 | +0.280695 | +0.303525 | +0.263054 | 0.054312 |
| **5** | **+0.285313** | **+0.300409** | **+0.255835** | 0.054312 |
| 10 | +0.243719 | +0.272718 | +0.238299 | 0.054312 |
| 21 | +0.183781 | +0.187180 | +0.190213 | 0.054312 |

**每一档、每一个学习方法都显著落后于物理外推基线**（差距 0.18–0.30，
而 damage 自身 RMSE 仅 0.054312）。§18 要求原样写出的定论：

> **学习模型未超过基于已知累计损伤结构的物理外推基线。**

这一事实不因赛题主题是迁移学习而被隐藏或弱化。需要同时说明的是：
`damage_extrapolation` 使用了**已知的累计损伤结构**这一机理先验，
学习臂没有该先验；所以它领先说明的是"在损伤结构已知的场景中物理外推是更强的
可部署选择"，而非"学习方法本质更差"。详见 `limitations.md` §8。

---

## 工程推荐（§19）

```
ENGINEERING_RECOMMENDATION = damage_extrapolation
```

优先级冻结为 (1) primary 档 test info macro RMSE →(2) warning miss →
(3) catastrophic rate →(4) 简洁性/可部署性。`damage_extrapolation` 在
primary 档以 RMSE 0.054312、漏报 0.0 同时占据前两项，排名首位。
`const_mean_info` / persistence / true-RUL 查表作为 oracle 或非可部署标尺已排除。

与 B5 的工程推荐**一致**——两个阶段独立得到同一结论。

## §20 合并解释

- B5（全量失效标签）：无正向迁移。
- B6（失效标签稀缺，primary n=5）：无正向低标签迁移。
- 二者同为 negative → **情况 A** → `NO_POSITIVE_TRANSFER_SUPPORTED`。

因为落在情况 A，**不存在**"条件性低标签收益"的主张。
（若曾落在情况 B，也只能表述为
`NO_GENERAL_POSITIVE_TRANSFER, CONDITIONAL_BENEFIT_UNDER_FAILURE_LABEL_SCARCITY`；
**绝对禁止**表述为"迁移学习总体显著优于 Target-only"。）

## 诚实边界

本结论是**"未能证明正向迁移"**，不是"已证明迁移无效"。
seed 数为 5、标签档只有 4 个、方法集刻意收口、训练预算固定为 8 epoch，
统计功效不足与效应不存在在此设定下无法区分。
完整的不可外推边界见 `docs/basilisk_b6/limitations.md`。

## 到此停止

B6 是飞轮线最后一个允许产生新核心实验数字的阶段。
之后只允许 **B7（图表 + 报告）** 与 **B8（打包 + Docker）**。
不新增算法，不调 MMD，不改 split，不改 HI/RUL。
