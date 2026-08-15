# BASILISK-B5 正式迁移评估结果

**阶段**：`BASILISK_B5` ｜ **标签**：`FORMAL_TRANSFER_EVALUATION`
**最终判定**：`B5_NO_POSITIVE_TRANSFER`（§19 组合 **D**）
**工程推荐**：`ENGINEERING_RECOMMENDATION = damage_extrapolation`

- `protocol_sha256` = `592aeab38458bcec0673d54b7b8c1c1fb3f6cd1a58207cb73db49c2eb3786a57`
- `config_sha256` = `09f02f4916c1eb115611916eb1e3235e255155523020abb48986f8eff018ee6f`
- `split_sha256` = `23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932`（B2.1 冻结划分，未重划）
- 输入 schema：`CORE_ONLY`（由 `B3X_NO_STABILIZING_SIGNAL` 决定，无 mission 特征）
- 正式 seed：**112 / 113 / 114 / 115 / 116**（全新 confirmatory seed，与 B2.1 的 102–106、B4X 的 92–94、replay 的 72/74/76 无交集）
- 全部门槛在出任何 B5 数字之前冻结于 `checkpoints/basilisk_b5/protocol_hash.json`

这是第一次允许产生"正式迁移结论"的阶段。本页的结论只依据本阶段的正式 test 表。

---

## 0. 本阶段回答的三个问题

| 问题 | 答案 |
|------|------|
| Q1 消除寿命覆盖不匹配后，`source_finetune` 是否稳定优于 Target-only？ | **否**。8 条门槛通过 1 条，`gain_ft` 均值 −0.001635，5 个 seed 里仅 1 个为正 |
| Q2 `source_mmd_finetune` 是否稳定优于 Target-only？ | **否**。8 条门槛通过 3 条，`gain_mmd` 均值 −0.000583，5 个 seed 里 2 个为正 |
| Q3 即使存在正转移，迁移模型是否仍不如 `damage_extrapolation`？ | **仍然不如，且差距很大**。`transfer_vs_damage` 为 −0.188（FT）/ −0.187（MMD） |

---

## 1. 主结果表（§18）

RMSE 为 test information-zone trajectory-macro RMSE 的 5 seed 均值。`damage_extrapolation`
是一条**强基线**，在本表中与其它方法同级并列，不是脚注。

| 方法 | RMSE info macro | RMSE pooled | MAE | corr | PSR | warning coverage | miss rate | PH | α-λ (0.3/0.5/0.7) | catastrophic rate |
|------|----------------:|------------:|----:|-----:|----:|-----------------:|----------:|---:|:-----------------:|------------------:|
| `target_only` | 0.241024 | 0.254462 | 0.203165 | 0.8880 | 0.818 | 0.4800 | 0.5200 | −0.0268 | 0.726 / 0.463 / 0.240 | 0.000000 |
| `source_finetune` | 0.242659 | 0.255110 | 0.204378 | 0.8980 | 0.827 | 0.3886 | 0.6114 | −0.0174 | 0.783 / 0.474 / 0.280 | 0.005714 |
| `source_mmd_finetune` | 0.241607 | 0.254740 | 0.204780 | 0.9114 | 0.831 | 0.4514 | 0.5486 | −0.0371 | 0.714 / 0.480 / 0.263 | 0.000000 |
| `const_mean_info` | 0.288629 | 0.288630 | 0.249960 | NaN | 0.000 | 0.0000 | 1.0000 | 0.0000 | 0.000 / 0.086 / 0.457 | 0.000000 |
| **`damage_extrapolation`** | **0.054312** | **0.056606** | **0.038413** | **0.9858** | 1.035 | **1.0000** | **0.0000** | 0.0000 | 0.571 / 0.743 / 0.971 | 0.000000 |

- `const_mean_info` 的 corr 为 **NaN**：常数预测无方差，相关系数无定义。此处保留 NaN，未转成 0。
- PSR = `pred_std / true_std`；三个学习组均在 0.82–0.83，无 `HIGH_VARIANCE_WARNING`（阈值 3）。PSR 是 §17 的**诊断项**，不进入正式 Gate。
- `damage_extrapolation` 的 PH 为 0：其 HI 轨迹与真值几乎重合，报警触发点与 EOL 判据同时到达，前置期为 0 而非负。

逐 seed 明细见 `checkpoints/basilisk_b5/formal_metrics.json`。

### 1.1 关于 `target_only` 数值的重要说明（§6.3）

为满足 §6 的架构公平性，本阶段三个学习组统一建 `n_features = 12`（源域 encoder
第一层宽度）/ `n_target = 10`。B2/B2.1 的 `train_target_only` 建的是 `(10, 10)`。
因此：

> **B5 的 `target_only` = 0.241024 是本阶段的一次重跑，它不等于 B2.1 的 0.2441。**
> 跨阶段相减是不同尺子。所有 gain 只在 B5 自己的三组之间计算。

---

## 2. 逐 seed 主指标（§23 items 4–6）

| seed | `target_only` | `source_finetune` | `source_mmd_finetune` |
|-----:|-------------:|-----------------:|---------------------:|
| 112 | 0.240378 | 0.241751 | 0.247992 |
| 113 | 0.243926 | 0.249137 | 0.237555 |
| 114 | 0.233967 | 0.235764 | 0.242433 |
| 115 | 0.248886 | 0.246544 | 0.241306 |
| 116 | 0.237965 | 0.240099 | 0.238751 |
| **均值** | **0.241024** | **0.242659** | **0.241607** |
| 标准差 | 0.005696 | 0.005291 | 0.004066 |

选点与配对证据（§7 三项必录，均为 validation-only）：

| seed | 组 | best_epoch | val info_macro_rmse | checkpoint_sha256 (前 10) | batch_order (前 10) | init_weights (前 10) |
|-----:|----|-----------:|--------------------:|---------------------------|---------------------|----------------------|
| 112 | target_only | 1 | 0.288930 | `06006bcddc` | `fa1fb7a7ad` | `8084dd553b` |
| 112 | source_finetune | 1 | 0.289480 | `75e9736018` | `fa1fb7a7ad` | `8084dd553b` |
| 112 | source_mmd_finetune | 4 | 0.281748 | `4c7996fa96` | `fa1fb7a7ad` | `8084dd553b` |
| 113 | target_only | 1 | 0.287297 | `e4b84f397c` | `9586577f9f` | `ba586094b7` |
| 113 | source_finetune | 0 | 0.284560 | `a50f1585ce` | `9586577f9f` | `ba586094b7` |
| 113 | source_mmd_finetune | 1 | 0.278190 | `25336a2506` | `9586577f9f` | `ba586094b7` |
| 114 | target_only | 0 | 0.285360 | `39c8993a3b` | `bb3ffe7de6` | `ba67d28330` |
| 114 | source_finetune | 0 | 0.290028 | `5ad6eb510e` | `bb3ffe7de6` | `ba67d28330` |
| 114 | source_mmd_finetune | 0 | 0.277546 | `9a266ceba4` | `bb3ffe7de6` | `ba67d28330` |
| 115 | target_only | 3 | 0.281785 | `04f4f86c40` | `1d4830141b` | `6a18344a9f` |
| 115 | source_finetune | 0 | 0.300039 | `43d805ee70` | `1d4830141b` | `6a18344a9f` |
| 115 | source_mmd_finetune | 0 | 0.263635 | `2900547875` | `1d4830141b` | `6a18344a9f` |
| 116 | target_only | 1 | 0.287258 | `7ad2bef716` | `f125ef99c2` | `77a15c37ba` |
| 116 | source_finetune | 1 | 0.279085 | `c9f6e3325b` | `f125ef99c2` | `77a15c37ba` |
| 116 | source_mmd_finetune | 0 | 0.294690 | `748c6d76bc` | `f125ef99c2` | `77a15c37ba` |

同一 seed 内三组的 `batch_order` 与 `init_weights` 指纹逐位相同 —— 这是 §6 严格配对的
机器可验证证据。三组的**唯一**差异是初始化（随机 / 源域预训练）与 MMD 开关。

---

## 3. 配对增益与 seed 级 bootstrap（§8/§10）

`gain_ft(seed) = RMSE_target(seed) − RMSE_source_ft(seed)`，正数 = 迁移改进。

| seed | `gain_ft` | `gain_mmd` |
|-----:|---------:|----------:|
| 112 | −0.001373 | −0.007614 |
| 113 | −0.005211 | +0.006370 |
| 114 | −0.001797 | −0.008466 |
| 115 | **+0.002342** | **+0.007580** |
| 116 | −0.002134 | −0.000786 |

| 统计量 | `gain_ft` | `gain_mmd` |
|--------|---------:|----------:|
| mean | −0.001635 | −0.000583 |
| median | −0.001797 | −0.000786 |
| std | 0.002690 | 0.007527 |
| improve_count | **1 / 5** | **2 / 5** |
| bootstrap CI95 | [−0.003828, +0.000704] | [−0.006589, +0.005423] |
| CI 下界 > 0 | 否 | 否 |

bootstrap 配置：`unit = one_paired_difference_per_seed`，**n = 5**，5000 次重抽，
固定种子 20260814。重抽单位是 seed 级配对差，未把时间点当独立样本，未用轨迹数量
抬高显著性。**n = 5 的区间只描述训练随机性下差值方向是否稳定，不是泛化误差的
置信区间** —— 见 `limitations.md`。

两个方法的 CI 都跨过 0：本阶段的证据既不支持正转移，也不足以断言"迁移有害"。
方向上两者的均值都为负，但幅度（约 0.0006–0.0016）远小于 seed 间波动（std 0.0027–0.0075）。

---

## 4. 正式 Gate 判定（§9，八条全满足才算正转移）

### 4.1 `source_finetune` → `B5_FT_NO_POSITIVE_TRANSFER`（1/8）

| # | 条件 | 实测 | 结论 |
|--:|------|------|:----:|
| 1 | ≥ 4/5 seed gain > 0 | 1/5 | 未通过 |
| 2 | mean(gain) > 0 | −0.001635 | 未通过 |
| 3 | median(gain) > 0 | −0.001797 | 未通过 |
| 4 | bootstrap CI95 下界 > 0 | −0.003828 | 未通过 |
| 5 | corr 无系统性退化（≥ target − 0.02） | 0.8980 ≥ 0.8680 | **通过** |
| 6 | catastrophic rate 不高于 target_only | 0.005714 > 0.000000 | 未通过 |
| 7 | warning miss rate ≤ target + 0.05 | 0.6114 > 0.5700 | 未通过 |
| 8 | short/medium/long 至少 2/3 个 bin gain > 0 | 1/3 | 未通过 |

### 4.2 `source_mmd_finetune` → `B5_MMD_NO_POSITIVE_TRANSFER`（3/8）

| # | 条件 | 实测 | 结论 |
|--:|------|------|:----:|
| 1 | ≥ 4/5 seed gain > 0 | 2/5 | 未通过 |
| 2 | mean(gain) > 0 | −0.000583 | 未通过 |
| 3 | median(gain) > 0 | −0.000786 | 未通过 |
| 4 | bootstrap CI95 下界 > 0 | −0.006589 | 未通过 |
| 5 | corr 无系统性退化 | 0.9114 ≥ 0.8680 | **通过** |
| 6 | catastrophic rate 不高于 target_only | 0.000000 ≤ 0.000000 | **通过** |
| 7 | warning miss rate ≤ target + 0.05 | 0.5486 ≤ 0.5700 | **通过** |
| 8 | 至少 2/3 个 bin gain > 0 | 1/3 | 未通过 |

MMD 的 `mmd_lambda` 保持冻结值 **1.0**，未做网格搜索、未逐 seed 调、未在 B5 test 上调、
未重新挑选源域层。MMD 表现优于 FT（3/8 vs 1/8，且 corr 最高 0.9114）但仍不满足门槛，
按 §16 直接判 `NO_POSITIVE_TRANSFER`，不因"接近"而放宽。

**门槛未被降低**：`gate_thresholds` 与 `protocol_hash.json` 中冻结的那份逐项一致，
`summarize_b5.py` 会在两者不符时直接抛 `B5_INVALID`。

---

## 5. 与物理基线的比较（§14）

`transfer_vs_damage = RMSE_damage − RMSE_transfer`，负数 = 迁移不如物理基线。

| 方法 | RMSE | `damage_extrapolation` | `transfer_vs_damage` | 判读 |
|------|-----:|----------------------:|--------------------:|------|
| `source_finetune` | 0.242659 | 0.054312 | **−0.188347** | 不如物理基线 |
| `source_mmd_finetune` | 0.241607 | 0.054312 | **−0.187295** | 不如物理基线 |

按 §14 的要求原样写出：

> **迁移模型可能相对 Target-only 有增益，但未达到物理 damage extrapolation 基线。**

在本阶段，前半句甚至不成立 —— 两个迁移方法相对 Target-only 的均值增益都是负的。
差距量级值得强调：物理外推的 RMSE 约为三个学习组的 **1/4.5**，且 miss rate 为 0、
corr 0.9858。这不是"略优"，是量级差异。

---

## 6. 寿命分 bin（§11）与短寿命鲁棒性（§12）

bin 边界沿用 B2.1 冻结的 `[26846.0, 37395.0]`，**未按 B5 test 分布重算**。
test 侧 event 轨迹计数：short 12 / medium 11 / long 12。

| bin | `target_only` | `source_ft` | `source_mmd` | `damage` | `gain_ft` | `gain_mmd` |
|-----|-------------:|-----------:|------------:|--------:|---------:|----------:|
| short | 0.223777 | 0.228210 | 0.224698 | 0.048726 | −0.004433 | −0.000921 |
| medium | 0.259158 | 0.257123 | 0.252076 | 0.051068 | **+0.002035** | **+0.007082** |
| long | 0.241649 | 0.243850 | 0.248921 | 0.062872 | −0.002201 | −0.007272 |

`gain > 0` 的 bin 数：**ft 1/3，mmd 1/3**（§9 条件 8 要求 ≥ 2）。

`localized_transfer_benefit`：**ft = False，mmd = False**。唯一为正的 bin 是 **medium**，
不是 short —— 因此本阶段既不能声称全周期提升，也不能声称"短寿命局部增益"。
短寿命端两个迁移方法都略差于 Target-only。

短寿命细表（§12 五项）：

| 组 | short mean RMSE | 逐 seed RMSE | catastrophic | pred_std | true_std | PSR |
|----|---------------:|--------------|-------------:|---------:|---------:|----:|
| `target_only` | 0.223777 | 0.219218 / 0.229730 / 0.218831 / 0.235503 / 0.215605 | 0 | 0.2628 | 0.2887 | 0.9104 |
| `source_finetune` | 0.228210 | 0.227641 / 0.245037 / 0.215425 / 0.238856 / 0.214092 | 0 | 0.2653 | 0.2887 | 0.9192 |
| `source_mmd_finetune` | 0.224698 | 0.227090 / 0.220253 / 0.224974 / 0.228795 / 0.222378 | 0 | 0.2637 | 0.2887 | 0.9136 |

short bin 内三组的 catastrophic 计数都是 0。MMD 的逐 seed 离散度最小
（0.2203–0.2288），FT 最大（0.2141–0.2450）—— 这是稳定性上的差别，但不构成增益。

删失侧 bin：`degenerate = true`，`n_effective_bins = 1`。全部 79 条右删失轨迹的
观测时长相同（仿真在同一 horizon 截断），三分位退化成单一 bin。此处如实报告，
未为凑三个 bin 编造边界，也未改用任何寿命型代理量。

---

## 7. 报警与预后指标（§13）

复用冻结的 S4/Basilisk evaluator。74 条 test 轨迹中 **35 条 event / 39 条右删失**；
PH、α-λ 只在 35 条 event 上计算，39 条删失轨迹被显式排除，**未伪造 EOL**。

| 组 | coverage | miss rate | false alarm | PH (macro) | convergence | late convergence | warning valid |
|----|--------:|---------:|-----------:|----------:|-----------:|-----------------:|:-------------:|
| `target_only` | 0.4800 | 0.5200 | 0.0256 | −0.026780 | 0.2108 | 0.2479 | 5/5 |
| `source_finetune` | 0.3886 | 0.6114 | 0.0256 | −0.017413 | 0.2164 | 0.2599 | 5/5 |
| `source_mmd_finetune` | 0.4514 | 0.5486 | 0.0462 | −0.037056 | 0.2136 | 0.2508 | 5/5 |
| `const_mean_info` | 0.0000 | 1.0000 | 0.0000 | 0.000000 | 0.4123 | 0.3828 | 5/5 |
| `damage_extrapolation` | 1.0000 | 0.0000 | 0.0256 | 0.000000 | NaN | NaN | 5/5 |

- `warning valid = 5/5` 只说明这组指标口径自洽，**不等于 warning 可用**：三个学习组的
  miss rate 在 0.52–0.61，即一半以上的失效没被提前报出来。这一点必须与 coverage 成对读。
- PH 为负意味着报警**晚于** RUL 阈值穿越点。三个学习组都是负值。
- `damage_extrapolation` 的 convergence 为 **NaN**（`n_evaluable = 0`）：保留 NaN，未转成 0。
- §9 条件 7：`ft_within_tol = False`（0.6114 > 0.5700），`mmd_within_tol = True`（0.5486 ≤ 0.5700）。

---

## 8. 最终判定（§19）与工程推荐（§15）

**迁移判定**：

- `B5_FT_NO_POSITIVE_TRANSFER`
- `B5_MMD_NO_POSITIVE_TRANSFER`
- 组合 **D** → **`B5_NO_POSITIVE_TRANSFER`**

**工程推荐**（与迁移判定分开，依据只能是正式 test 表主指标）：

| 排名 | 方法 | RMSE info macro |
|-----:|------|---------------:|
| 1 | **`damage_extrapolation`** | **0.054312** |
| 2 | `target_only` | 0.241024 |
| 3 | `source_mmd_finetune` | 0.241607 |
| 4 | `source_finetune` | 0.242659 |

**`ENGINEERING_RECOMMENDATION = damage_extrapolation`**

`persistence` 与 `true_rul_lookup` 是 oracle，已排除在候选之外，不进入可部署方法排名。

这两个判定回答的是不同问题：前者问"源域预训练是否帮到了这个模型"，后者问
"现在该部署什么"。本阶段两者恰好都指向"不是迁移模型"，但即使某个迁移方法通过了
Gate，推荐仍可能是 `damage_extrapolation`。

---

## 9. 这个结论意味着什么、不意味着什么

**意味着**：在消除"寿命覆盖不匹配"这一已知缺陷之后（B2.1 划分，train/val/test
三侧短寿命都有覆盖），用全新的五个 confirmatory seed、在预登记的八条门槛下，
`source_finetune` 与 `source_mmd_finetune` **都没能证明相对 Target-only 的稳定增益**。
这是一个正当的负面结论，不是实验失败。

**不意味着**：

- 不意味着"迁移一定有害"。两个 CI 都跨 0，方向证据不足以支撑相反的强断言。
- 不意味着结论可外推到其它划分、其它输入 schema、低数据或截断场景。
- 不意味着 B4X 的 `B4X_TRANSFER_STABILIZATION_SIGNAL` 被否证 —— 那是探索性信号，
  在不同 seed、不同划分上得到，本阶段既未继承它也未推翻它。
- 不改写 `B2_GENERALIZATION_FAIL`：它在自己的划分上保持终局。

**不得做的事**：不得通过降低门槛、换 seed、改超参、事后挑子集来翻转这个结论。

---

## 10. 下一阶段建议（§20，不自动执行）
因未成立正式正转移，建议的下一阶段是：

> **B6: baseline formal matrix + negative-transfer conclusion**

**本阶段到此停止。低数据矩阵与截断矩阵都不自动跑，B6 不自动执行。**

相关文档：`protocol.md`（冻结协议）｜`lifetime_bin_results.md`｜`warning_results.md`
｜`limitations.md`｜`REPRODUCE.md`｜`../../STATUS_BASILISK_B5.md`
