# BASILISK-B2.1 局限与不可主张事项

> **`SPLIT_COVERAGE_ROBUSTNESS`**
>
> **`B2_GENERALIZATION_FAIL` 保持终局** · **`B3X_NO_STABILIZING_SIGNAL`** ·
> **`B4X_TRANSFER_STABILIZATION_SIGNAL` 仅 `EXPLORATORY_ONLY`** ·
> **不存在任何正式的 positive-transfer 结论。**

`B21_GENERALIZATION_PASS` 是一个**范围极窄**的结论。本文件记录它不包含什么。

---

## 1. PASS 不推翻 B2, 也不追溯修改 B2

`B2_GENERALIZATION_FAIL` 依旧是**那个划分上的**终局判定, 不因本阶段而改写、
撤回或重新解释。B2.1 的含义只有一句:

> 在一个覆盖 short/medium/long 三段寿命的划分上, Target-only 能稳定超过
> `const_mean_info`。

它**不**说明: B2 判错了 / 模型本身变强了 / 模型可部署 / 之前的
short-life 崩溃是"划分的锅所以不算问题"。B2 与 B2.1 是**两个不可比的
test 集**, 主指标数值不得跨阶段直接相减当作"改进量"。

## 2. 划分变了, 所以 B2 → B2.1 的数值不是同一把尺子

B2 test 与 B2.1 test 的轨迹集合、比例 (B2 与 B2.1 的 30/20/50 分层方式不同)、
寿命构成都不同。B2 target_only 1.0142 → B2.1 0.2435 **不是模型进步**,
是"被评测的对象换了"。文中一切跨阶段对照仅用于说明**方差来源**, 不作为增益。

## 3. 删失三分位轴退化 (degenerate)

79 条删失轨迹的 `observed_duration` **全等于 52596** —— 仿真在单一 horizon
截断。三分位边界塌缩为 `[52596.0, 52596.0]`, 有效箱数 = **1** (`long_obs: 79`)。

因此**"删失轨迹按观测时长分层"这一条在本数据集上实际不起作用**。
manifest 里 `degenerate: true` + `degenerate_note` 如实记录。

选择如实报告而非补救, 理由: 为了凑出三个非空箱, 唯一做法是编造边界或换用
某个"寿命型代理量" —— 后者等价于**给右删失轨迹发明 EOL**, 是被明令禁止的。
若未来仿真提供多 horizon, 这条轴才会真正生效。

## 4. `damage_extrapolation` 全面优于 target_only

| | short | medium | long | overall (mean) |
|---|---|---|---|---|
| target_only | 0.2340 | 0.2545 | 0.2448 | 0.2441 |
| damage_extrapolation | 0.0487 | 0.0511 | 0.0629 | 0.0543 |

物理外推基线在**每个 seed、每个寿命段**上都比学习模型好约 4.5 倍。
§12 的 gate 只以 `const_mean_info` 为判据, 所以 PASS 在协议上成立;
但这必须与判定同时呈现:

> **本阶段没有证明学习模型优于物理方法。** 它只证明学习模型不再退化到
> 常数预测以下。在"该不该上学习模型"这个问题上, 当前证据仍指向物理外推。

这与 B2 的观察一致, 不是新问题, 也不因划分修复而消失 —— 说明它是**建模层
问题**, 而非划分层问题。

## 5. warning 指标"有效"不等于"够用"

条件 6 检查的是 coverage/miss 互补、分母正确、未隐藏 miss, **不是**报警质量。
实得 coverage 仅 **0.314–0.486**, 即 **miss 0.514–0.686**: 一半以上的失效
轨迹没有在 EOL 前给出有效预警。这个系统按当前指标**不可用于在轨告警**。
(B2 为 miss 0.694–0.944, 只是更差。)

## 6. 删失下界违反仍存在

违反率 0.060–0.175: 仍有 6%–17.5% 的删失取点被预测出比"至少还能活多久"
更短的剩余寿命。删失轨迹的一切 RUL 类指标恒为 **NaN + n_evaluable = 0**,
不参与任何均值 —— 但违反率本身是真实缺陷, 不因指标为 NaN 而被抵消。

## 7. anchor 规则的性质

§5.1 的 `short_bin_anchor_order = [train, val, test]` 使
`train min EOL ≤ test min EOL` 与 `val min EOL ≤ test min EOL`
**按构造成立**, 而不是靠试 seed 试出来。它:

* 在任何 B2.1 数值产生**之前**写入 protocol.md 并哈希固定
  (`frozen_before_any_b21_number: true`);
* 只读 EOL 的**序**, 不读任何 test 性能。

代价必须讲明: test 的最小 event EOL 因此**被抬高**到 22126, 即
**test 不再包含数据集中最短命的那三条轨迹**。也就是说, B2.1 修掉了
"test 比 train 更极端"的缺陷, 手段是**削弱 test 的极端性**。
"模型在全数据集最短寿命轨迹上表现如何"这个问题, B2.1 **没有回答**。

## 8. 五个 seed 只是开发种子

`[102, 103, 104, 105, 106]` 是开发种子, 不是独立测试重复。CI
`[+0.041375, +0.047920]` 的抽样单位是 **一个 seed 一个配对差值**, n = 5,
它刻画的是**训练随机性**下的稳定度, **不是**泛化误差的置信区间。
禁止把 endpoint 或时间点当独立样本 —— 这是 CI 唯一的合法解读。

## 9. 本阶段不含迁移

`forbid_transfer: true`。没有跑源域→目标域迁移, 没有任何迁移增益数字,
不得由 PASS 推出"迁移会有效"。B4X 的稳定化信号维持 `EXPLORATORY_ONLY`。

## 10. 与 S5B 无关

S5B 阶段已关闭, 本阶段未重开、未引用、未重新解释其任何结论。

---

## 下一阶段 (只是计划)

**`B5 formal transfer on the frozen B2.1 split`** —— 见
[b5_plan.md](b5_plan.md)。**B5 未运行, 且不得自动运行。**
