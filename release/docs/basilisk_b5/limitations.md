# BASILISK-B5 局限性

本页列出会限制 `B5_NO_POSITIVE_TRANSFER` 这个结论适用范围的因素。**它不削弱结论本身**——
八条预登记门槛在出数字之前就已冻结，两个迁移方法分别只通过 1/8 与 3/8，这个判定是
按规则得出的。本页要防的是另一件事：把这个结论用到它管不到的地方。

---

## 1. n = 5 的 bootstrap 不是泛化误差置信区间

配对 bootstrap 的重抽单位是 **seed 级配对差，n = 5**，5000 次重抽，固定种子 20260814。

| 项 | `gain_ft` | `gain_mmd` |
|----|---------:|----------:|
| CI95 | [−0.003828, +0.000704] | [−0.006589, +0.005423] |
| 区间宽度 | 0.004532 | 0.012012 |
| mean | −0.001635 | −0.000583 |

必须明确：

> **这个区间只描述"在固定划分与固定协议下，重复训练的随机性会让配对差落在哪里"，
> 不是泛化误差的置信区间。** 它对"换一个划分会怎样""换一批轨迹会怎样"一无所知。

- n = 5 的经验分布最多只有 5 个不同取值，重抽 5000 次并不增加信息量，只是让分位数
  读数稳定。区间宽度受这 5 个点支配。
- 因此不能反向使用：CI 跨 0 **不足以断言"迁移有害"**，只能说"没有证据支持迁移有益"。
  两个方法的均值方向都为负，但幅度（0.0006–0.0016）远小于 seed 间标准差
  （0.0027–0.0075），方向本身不稳。
- 也不能用轨迹数（74）或时间点数量去替换这个 n。禁止把时间点当独立样本抬高显著性
  是贯穿全项目的纪律，本阶段同样遵守（`forbid_endpoint_bootstrap = True`,
  `forbid_trajectory_count_significance = True`）。

**若要收紧这个区间，唯一正当做法是增加正式 seed 数量并重新预登记，不是换统计口径。**

---

## 2. `damage_extrapolation` 以约 0.187–0.188 RMSE 全面压制学习模型

| 方法 | RMSE info macro | 与 damage 之差 |
|------|---------------:|--------------:|
| `damage_extrapolation` | **0.054312** | — |
| `target_only` | 0.241024 | −0.186712 |
| `source_mmd_finetune` | 0.241607 | **−0.187295** |
| `source_finetune` | 0.242659 | **−0.188347** |

这不是"略优"，是量级差异（约 4.5 倍）。物理外推同时在 corr（0.9858）、miss rate（0）、
α-λ λ=0.7（0.9714）三项上都大幅领先，且它在 short/medium/long 三个 bin 内**无一例外**
地领先。

诚实的表述（§14 强制原句）：

> **迁移模型可能相对 Target-only 有增益，但未达到物理 damage extrapolation 基线。**

在本阶段，连这句话的前半段都不成立 —— 两个迁移方法的均值增益都是负的。因此
**绝不能把"正转移"写成"最佳方法"，本阶段甚至没有正转移可写。**

需要补充的解释性说明（不改变上面的排名）：`damage_extrapolation` 直接沿仿真的损伤
积累律外推，与生成 HI 的机理同源，在这个仿真目标域上占据结构性优势。这解释了差距
为何如此之大，但**不构成为学习模型开脱的理由**：正式 test 表就是排名依据，
`ENGINEERING_RECOMMENDATION = damage_extrapolation` 由此得出。若要论证学习模型在真实
星上数据（机理不完全已知、遥测含真实噪声）上的价值，那是另一个阶段的另一套证据，
不能靠本阶段的数字外推。

---

## 3. 三个学习组的报警器都不可部署

| 组 | miss rate | miss rate (before EOL) | PH |
|----|---------:|----------------------:|---:|
| `target_only` | 0.5200 | 0.6343 | −0.026780 |
| `source_finetune` | 0.6114 | 0.6514 | −0.017413 |
| `source_mmd_finetune` | 0.5486 | 0.6343 | −0.037056 |

三个组都漏掉一半以上的失效，且 PH 全为负（报警晚于阈值穿越点）。`warning_valid_count
= 5/5` 只是口径自洽，不是可用性。

特别提醒一个容易被误读的数字：`source_finetune` 的 PH 均值（−0.0174）在三组里最"好"，
但它五个 seed 里有三个 PH 恰为 0，同时它的 miss rate 最高（0.6114）——
**没触发报警的轨迹不贡献负前置期，这个 PH 优势是"报得更少"的副产物，不是预警更早。**
不得单独引用它。

`source_mmd_finetune` 的 coverage 逐 seed 在 0.2286–0.6286 之间跳动，说明其报警行为
高度依赖训练随机性，均值 0.4514 不该被当作稳定能力。

---

## 4. medium-bin 的正增益不是"短寿命局部收益"

| bin | `gain_ft` | `gain_mmd` |
|-----|---------:|----------:|
| short | −0.004433 | −0.000921 |
| medium | **+0.002035** | **+0.007082** |
| long | −0.002201 | −0.007272 |

`localized_transfer_benefit`：**ft = False，mmd = False**。

§12 定义的"局部收益"指增益只在 **short** bin 为正。本阶段唯一为正的是 **medium**，
而 short 段两个迁移方法都略差。因此：

- 不能声称"迁移带来全周期提升"（3 个 bin 只有 1 个为正）；
- **也不能声称"迁移带来短寿命局部增益"**（short 段为负）。

进一步的警示：mmd 在 medium 的 +0.007082 与在 long 的 −0.007272 几乎等量反号。
这更像 bin 间噪声再分配，而非某个寿命段的系统性收益。**不得只摘 medium 那一格
写进结论。**

---

## 5. 右删失侧 bin 退化，删失轨迹上无 RUL 结论

| 项 | 值 |
|----|----|
| `edges` | `[52596.0, 52596.0]` |
| `counts` | `short_obs: 0`, `medium_obs: 0`, `long_obs: 79` |
| `degenerate` | **true** |
| `n_effective_bins` | **1** |

全部 79 条右删失轨迹的观测时长相同（仿真在同一 horizon 截断），三分位边界重合。
后果：

- 分 bin 分析（本阶段全部分 bin 数字）只覆盖 **35 条 event 轨迹**，即 test 集的
  **47%**（35/74）。另外 39 条删失轨迹不参与 EOL 分 bin、不参与 PH / α-λ。
- 因此本阶段对"长寿命/尚未失效个体"的预测质量**基本没有测量**。
  这是覆盖面上的真实空缺，不是可以用别的指标补上的。
- 绝不为了凑三个 bin 而编造边界，也绝不改用寿命型代理量 —— 那等于发明 EOL。

---

## 6. 结论的适用边界

`B5_NO_POSITIVE_TRANSFER` 只在以下条件下成立，换任一条都需重新评估：

| 维度 | 本阶段固定值 |
|------|-------------|
| 划分 | B2.1 冻结划分，`split_sha256 = 23e2b944…`，45/31/74 |
| 输入 schema | `CORE_ONLY`（由 `B3X_NO_STABILIZING_SIGNAL` 决定，无 mission 特征） |
| 迁移方法 | 只有 `source_finetune` 与 `source_mmd_finetune`（未加第四种） |
| MMD | 冻结 `mmd_lambda`，未做任何搜索 |
| 架构 | 三组统一 `n_features = 12` / `n_target = 10` |
| seed | 112–116（5 个） |
| 数据量 | 全量目标域，未做低数据 / 截断矩阵 |

尤其是：

- **未测低数据场景。** 迁移最可能显效的地方（目标域样本很少）本阶段完全没碰。
  `low_data_matrix_auto_run = False` —— 这是 §20 明令不自动跑的，属于 B6 的范围。
- **未测其它输入 schema。** `CORE_ONLY` 是被 `B3X_NO_STABILIZING_SIGNAL` 逼出来的选择，
  不是最优 schema 的搜索结果。
- **未测第三种迁移方法。** §4 禁止加，因此"是否存在能成立的迁移方案"这个问题
  本阶段无法回答，只能回答"这两个方案不成立"。

---

## 7. 与既有阶段结论的关系

- **不改写 `B2_GENERALIZATION_FAIL`**：它在自己的划分上保持终局，本阶段不追溯修改。
- **不继承也不否证 `B4X_TRANSFER_STABILIZATION_SIGNAL`**：那是探索性信号，在不同 seed、
  不同划分上得到。B4X 未被提升为正式 positive-transfer 结论
  （`b4x_must_not_be_promoted`），本阶段的负面结论也不构成对它的反驳。
  两者是不同条件下的不同观测。
- **`B3X_NO_STABILIZING_SIGNAL` 是本阶段 schema 选择的依据**，未被重新解释。
- **S5B 阶段已关闭**，本阶段未重开、未重新解释。

---

## 8. `target_only` 数字的跨阶段不可比性

为满足 §6 架构公平性，本阶段三组统一 `n_features = 12`（源域 encoder 第一层宽度）；
B2/B2.1 的 `train_target_only` 建的是 `(10, 10)`。

> **B5 的 `target_only` = 0.241024 是本阶段的重跑，不等于 B2.1 的 0.2441。**
> 跨阶段相减是不同尺子。所有 gain 只在 B5 自己的三组之间计算。

同理，本阶段的 `damage_extrapolation` = 0.054312 与 B2.1 记录的 0.0543 数值接近，
但也只在本阶段口径下被引用。

---

## 9. 一个已知的、协议预告过的测试失败

`tests/basilisk_b21/test_b21_discipline.py::test_b21_b5_absent_and_never_auto_run`
在 B5 完成后**必然失败**，且这是预期行为，不是回归。

该测试断言 `docs/basilisk_b5/results.md`、`checkpoints/basilisk_b5/metrics.json`、
`STATUS_BASILISK_B5.md` **必须缺席** —— 它是为"B5 尚未运行"这一状态写的守卫。
B5 正当地取代了那个状态，因此其中的 `docs/basilisk_b5/results.md` 与
`STATUS_BASILISK_B5.md` 两项前提已到期。

`protocol.md` §1.1 在 B5 开跑之前就写明了这一点：

> B2.1 的 `verify_baseline.py --tag after` 从 B5 开始不再适用于 B5 命名空间那三项，
> 这一点在此明示，而不是靠悄悄改文件掩盖。

处理方式（严格按 §1.1）：

- **不修改 `tests/basilisk_b21/test_b21_discipline.py`** —— 它是 `b21_tests` 契约组的
  冻结项（`fb84202d7b3118a4…`），改它就等于改冻结产物，会直接触发 `B5_INVALID`。
- 这条防线由 B5 自己的 `b5_must_stay_absent` 组接管，内容换成 **B6 的产物**：
  `checkpoints/basilisk_b6/metrics.json`、`docs/basilisk_b6/results.md`、
  `STATUS_BASILISK_B6.md`、`checkpoints/basilisk_b5/low_data_matrix.json`、
  `checkpoints/basilisk_b5/truncation_matrix.json` —— 这五项目前全部确认缺席，
  由 `tests/basilisk_b5/test_b5_baseline_contract.py::test_b5_b6_artifacts_must_stay_absent`
  断言。
- 契约本身完好：`verify_baseline.py --tag after` 报 **1067/1067 不变**，
  `B5_BASELINE_CONTRACT_OK`。

即：这不是"有一个测试挂了没管"，而是一个到期的阶段守卫按预案完成了职责交接。
全量结果为 **964 passed / 1 failed / 28 skipped / 1 xfailed**，唯一的 failed 就是它。

---

## 10. 其它口径说明

- **PSR 不在正式 Gate**（§17）。三个学习组 PSR 在 0.818–0.831（略欠预测方差），
  均未触发 `HIGH_VARIANCE_WARNING`（阈值 3）。这是诊断项，不用于判定。
- **nPHM 不是模型选择指标**，本阶段未用于任何选择。
- **`const_mean_info` 的 corr = NaN**（常数无方差，相关系数无定义），
  `damage_extrapolation` 的 convergence = NaN（`n_evaluable = 0`）。
  **两处都保留 NaN，未转成 0。**
- **`const_mean_info` 在 α-λ λ=0.7 上有 0.4571**：常数在寿命末段偶然落进容差带，
  是该指标本身的退化行为，不是常数预测有用，不得作方法优劣依据。
- **`persistence` 与 `true_rul_lookup` 是 oracle**，已排除在可部署方法排名之外。
- **checkpoint 选择只用 validation**，无任何 test 参与选点或早停。
- **LLM 不参与本阶段任何数值路径。**

---

相关文档：`results.md`（主表与判定）｜`lifetime_bin_results.md`｜`warning_results.md`
｜`protocol.md`（冻结协议）｜`REPRODUCE.md`
