# BASILISK-B6 局限与不可外推的边界

本文件记录 B6 结论**不能**被延伸到哪里。写在结论之前，避免读者把
"低标签压力测试"读成"迁移学习总体评价"。

---

## 1. 稀缺轴只压了一个维度

B6 只减少 **event-observed 失效标签轨迹数**（`n_event_labeled ∈ {3,5,10,21}`），
每一档都保留 train 中**全部 24 条** censored 轨迹。

因此 B6 回答的是"失效标签稀缺"，**不是**：

- target 轨迹总量稀缺（那会重新制造 B2 的 coverage mismatch，已被明确禁止）；
- 遥测通道缺失；
- 工况分布偏移；
- 源域数据量不足。

把 B6 结论读成"小样本迁移的一般结论"是过度外推。

## 2. 只有 4 个标签档、5 个 seed

- 敏感性分析只有 4 个采样点（n = 3, 5, 10, 21），**不得拟合趋势线**，
  只能做描述性判断（是否出现"标签越少增益越大"的迹象）。
- 显著性单位是 **seed 级配对差，n = 5**。5 个配对差的 bootstrap CI 很宽，
  这是设计上的诚实代价，不是可以靠增加端点数量"变窄"的东西——把时间点或
  轨迹条数当独立样本会严重高估显著性，本阶段明确禁止。
- 因此"未通过十条门槛"应读作**未能证明正向迁移**，而不是"已证明迁移无效"。
  统计功效不足与效应不存在，在 n=5 下无法区分。

## 3. 各档是嵌套子集

n=3 ⊂ n=5 ⊂ n=10 ⊂ n=21。这是刻意的：否则趋势会被"换了一组轨迹"混淆。
代价是各档之间**不独立**，四个档的结果不能当作四次独立重复实验来累加证据。

## 4. PRIMARY 档是唯一正式判定口径

只有 **n = 5** 允许产出正式低标签判定，且这是在看到任何 B6 结果**之前**固定的
（选 5 是因为它位于 3 与 10 之间）。

- n=3 / n=10 / n=21 仅作趋势，**不得单独推翻 primary**。
- 即使某个 secondary 档出现 5/5 全胜的大幅增益，只要 primary 未通过，
  正式判定仍然是 `B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER`，
  相应信号只能标记为 `NOT_PRIMARY_CONFIRMATORY_EVIDENCE`。

## 5. 方法集是收口的，不是穷举的

B6 只有三个训练臂（target_only / source_finetune / source_mmd_finetune）
加两个评估基线（damage_extrapolation / const_mean_info）。本阶段**禁止**
新增迁移方法、Wiener 过程粒子滤波、速率模型、新的 MMD lambda、
mission-feature 臂。

因此"源域预训练在此设定下无正向收益"**不等于**"任何迁移方法都无收益"。
它是对**这一个**迁移配置（HI/退化动力学层的 finetune 与 MMD 对齐、
lambda = 1.0、8 epoch 预算、CORE_ONLY 12 维输入）的判定。

## 6. 右删失轨迹限制了可用指标

test 集含右删失轨迹，它们没有真实 EOL。所有依赖真 EOL 的指标
（PH、α-λ、bin RMSE 的 full caliber）对这些轨迹保持 `NaN`，
**绝不转 0**；空 bin 返回 `NaN` + `n = 0`。

后果：

- `full` caliber 的 macro RMSE 在实践中恒为 NaN（该口径包含删失点），
  主指标因此固定为 `info` caliber 的 trajectory macro RMSE。
- 告警与 PH 的可评估轨迹数少于 test 总数，`n_censored_excluded` 已逐项记录。
- 不得因为"只统计成功检测"而让漏报率看起来更好——coverage 与 miss rate
  必须成对出现。

## 7. 训练预算很小且刻意不调

`max_epochs = 8`、`patience = 2`、`mmd_lambda = 1.0`，全部继承 B5 并冻结。
这是为了让 B5 与 B6 可比，代价是：

- 三阶段的 MMD 臂每阶段各自受同一 epoch 上限约束，累计 epoch 天然多于单阶段臂；
  公平性比的是**预算参数**（上限、patience、优化器、lr），不是累计 epoch 总数。
- 训练崩溃保留为正式结果，不得换 seed 重跑；只有明确代码错误 / NaN 才标
  `INVALID`，且仍不得换 seed。
- 若某方法需要更长训练才能显出优势，B6 无法观察到——但放宽预算就破坏了
  与 B5 的可比性以及同 cell 的公平性，所以这是有意接受的盲区。

## 8. damage 基线的对比是"带信息"的

`damage_extrapolation` 使用**已知的累计损伤结构**做物理外推。它在这个仿真设定下
拥有学习模型没有的机理先验，因此它领先不代表"学习方法本质更差"，而代表
**在损伤结构已知的场景里，物理外推是更强的可部署选择**。

反过来也必须诚实：不能因为赛题主题叫迁移学习就把这个事实藏成脚注。
它必须留在主表内逐档报出。

`const_mean_info` 只作常数标尺，**不是可部署方法**，
不进入工程推荐候选（与 persistence、true-RUL 查表同列排除）。

## 9. B5 与 B6 的关系是并列证据，不是覆盖

- B5（全量 target 数据，seeds 112–116）结论 `B5_NO_POSITIVE_TRANSFER` **永久有效**，
  不因 B6 结果被覆盖。
- 若 B6 primary 出现正向（情况 B），也只能表述为
  `NO_GENERAL_POSITIVE_TRANSFER, CONDITIONAL_BENEFIT_UNDER_FAILURE_LABEL_SCARCITY`。
- **绝对禁止**表述为"迁移学习总体显著优于 Target-only"。

## 10. 上游冻结事实继续限制解释范围

| 冻结结论 | 对 B6 解释的限制 |
|---|---|
| `B2_GENERALIZATION_FAIL` | B2 原始划分的泛化失败仍然成立，B2.1 只修了 lifetime-support mismatch |
| `B3X_NO_STABILIZING_SIGNAL` | mission feature 不得回到输入 schema，即使某些 seed"看起来更好" |
| `B4X_TRANSFER_STABILIZATION_SIGNAL` | 状态为 `EXPLORATORY_ONLY`，不得当作正向迁移证据引用 |
| `B21_GENERALIZATION_PASS` | 仅指 B2.1 划分下的泛化可用性，不含任何迁移主张 |
| S5B 阶段已关闭 | 不得重开或重新解释 |

## 11. 平台与可复现性边界

- 划分、标签子集、协议哈希：**零容差**，必须逐字符相同。
- CPU 训练与评估：同平台完全一致。
- GPU 训练：允许 ±5%（CUDA 非确定性）。若 ±5% 内的抖动足以翻转某条门槛判定，
  该判定就应视为不稳健——十条门槛全过的设计正是为了降低这种敏感性。

## 12. 本阶段之后

B6 是飞轮线**最后一个**允许产生新核心实验数字的阶段。
之后只允许 B7（图表 + 报告）与 B8（打包 + Docker）。
任何"再跑一组看看"都属于越界。
