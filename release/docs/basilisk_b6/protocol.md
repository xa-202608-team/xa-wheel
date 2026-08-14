# BASILISK-B6 Protocol —— Failure-Label Scarcity Formal Matrix

**阶段标签**: `FAILURE_LABEL_SCARCITY_FORMAL_MATRIX`
**冻结时点**: 本文件在**任何 B6 数字产生之前**冻结, 哈希记入
`checkpoints/basilisk_b6/protocol_hash.json`。之后所有判定脚本只能按这份哈希里
记录的阈值判定; 出数字之后改本文件 = 事后调门槛 = `B6_INVALID`。

这是飞轮线**最后一个允许产生新的核心实验数字**的阶段。

---

## §0 前置冻结事实 (只读, 不得覆盖、不得重新解释)

| 事实 | 值 | 地位 |
|------|-----|------|
| B1.8 | `B18_SCENARIO_READY` | 冻结 |
| B1.9 | `B19_FEATURE_READY` | 冻结 |
| B2 | `B2_GENERALIZATION_FAIL` | 在其自身划分上**终局** |
| B3X | `B3X_NO_STABILIZING_SIGNAL` | 终局 → 正式输入 schema 固定 `CORE_ONLY` |
| B4X | `B4X_TRANSFER_STABILIZATION_SIGNAL` | 仍然只是 `EXPLORATORY_ONLY` |
| B2.1 | `B21_GENERALIZATION_PASS` | 划分地基 |
| B2.1 split | `23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932` | 冻结 |

### B5 正式结果 (全量目标域 confirmatory)

正式 seeds `[112, 113, 114, 115, 116]`，主指标 = test information-zone
trajectory-macro RMSE:

| 方法 | mean info macro RMSE |
|------|---------------------|
| `target_only` | **0.241024** |
| `source_finetune` | **0.242659** |
| `source_mmd_finetune` | **0.241607** |
| `damage_extrapolation` | **0.054312** |

| gain | mean | improve | CI95 |
|------|------|---------|------|
| `gain_ft` | **−0.001635** | **1/5** | **[−0.003828, +0.000704]** |
| `gain_mmd` | **−0.000583** | **2/5** | **[−0.006589, +0.005423]** |

* 正式判定: **`B5_NO_POSITIVE_TRANSFER`**
* `ENGINEERING_RECOMMENDATION = damage_extrapolation`

**B5 的结论不得因 B6 被覆盖。** 即使某个低标签档出现正增益, 那也只能构成
**conditional low-label transfer benefit**, **绝不是 overall positive transfer**。

---

## §1 唯一的科学问题

> 当目标域只有**少量完整失效轨迹**可用于训练时, source 预训练是否比 Target-only
> 更有价值?

被削减的是 **`event-observed failure-labelled trajectories`**, **不是**目标域轨迹
总量。理由: 真实航天约束是**失效标签稀缺**, 而退化 / 未失效运行数据仍可大量存在。

若把稀缺定义成 `n_train` 总量下降, 就会重新制造 B2 的 coverage mismatch —— 那是在
测另一个 (已知会失败的) 问题, 不是本阶段的问题。

---

## §2 禁改面

禁止修改: B1.8 / B1.9 / B2 / B2.1 / B3X / B4X / **B5** / analytic S2.5–S5B 的产物,
source checkpoints, network architecture, optimizer, loss, MMD lambda, HI, RUL,
mission profile, split。

---

## §3 baseline 契约 (至少冻结)

B1.8 dataset hash ｜ B1.9 feature hash ｜ B2.1 split hash ｜ B2.1 train·val·test
IDs ｜ **B5 protocol hash / B5 config hash / B5 results hash** ｜ source
checkpoint hash ｜ encoder architecture ｜ MMD lambda ｜ metrics implementation ｜
damage baseline implementation ｜ target-only architecture。

`--tag before` 与 `--tag after` 必须**逐项一致**。任何一项改变即 `B6_INVALID`。

---

## §4 输入 schema

沿用 B5: **`CORE_ONLY`, `n_features = 12`**。

**不得恢复 mission-feature arm。** 不得因为 B3X 在某些 seed 上"看起来更好"就把
mission features 偷偷带回 B6 —— B3X 的正式判定是 `NO_STABILIZING_SIGNAL`, 终局。

`x_T` 中禁止出现: `hi_damage_obs`, `hi_friction`, `hi_a`, `b_true`, `rul`,
`rul_lower_bound`, `label_fail`, `eol_idx`, `D_true`。

---

## §5 划分

完全复用 B2.1, `split_sha256 = 23e2b944...5c932`。

| split | n | event | censored |
|-------|---|-------|----------|
| train | 45 | 21 | 24 |
| val | 31 | 15 | 16 |
| test | 74 | 35 | 39 |

**val / test 永远不动。**

---

## §6 失效标签稀缺的正式定义

**不是** `n_train = 3/5/10`。正式定义:

```
n_event_labeled ∈ {3, 5, 10, 21}
始终保留 train 中全部 24 条 censored trajectories
```

| 档 | event | censored | train 总数 |
|----|-------|----------|-----------|
| n=3 | 3 | 24 | 27 |
| n=5 | 5 | 24 | 29 |
| n=10 | 10 | 24 | 34 |
| n=21 | 21 | 24 | 45 (= B2.1 全量 train) |

未被选中的 event-observed 轨迹**完全从该档 train 移除**。

**禁止**:
1. 把已知真 EOL 的轨迹改标成 censored —— 那是伪造删失;
2. 使用它们的任何未来 / EOL 信息;
3. 以 unlabelled input 的形式偷偷塞回。

---

## §7 子集选择 (coverage-aware, 训练前定死)

bin 轴取 B2.1 的 event lifetime bins `short / medium / long`, edges 继续使用冻结的
**`[26846.0, 37395.0]`**, **不得从 B6 test 重算**。

| 档 | short | medium | long |
|----|-------|--------|------|
| n=3 | 1 | 1 | 1 |
| n=5 | 2 | 1 | 2 |
| n=10 | 3 | 3 | 4 |
| n=21 | 7 | 7 | 7 (= 全部) |

具体轨迹 ID **必须在任何训练之前生成**, 方式 = 从固定 `subset_seed` 出发在每个 bin
内做确定性 shuffle 后取前 k 条。

* 冻结 **`subset_seed = 20260814`**
* 写入 `checkpoints/basilisk_b6/label_subset_manifest.json`, 计算
  `subset_manifest_sha256`
* **一旦写出, 禁止更改。**

---

## §8 primary vs sensitivity

* 正式 **PRIMARY = `n_event_labeled = 5`** —— 唯一允许产生正式"低标签迁移"判定的
  档。选它的理由: 位于 3 与 10 之间, 且在看到任何 B6 结果之前就已固定, 避免"挑一个
  最好看的标签档"。
* SECONDARY (敏感性): n=3, n=10, n=21 —— **只**能用于 gain-vs-label-amount 趋势,
  **不得单独推翻 primary verdict**。

---

## §9 全新正式 seeds

`[122, 123, 124, 125, 126]`

禁用 (行为已被观察过): `72–76`, `92–94`, `102–106`, `112–116`。

---

## §10 方法矩阵 (每档)

训练: A `target_only` ｜ B `source_finetune` ｜ C `source_mmd_finetune`
只评估不训练: D `damage_extrapolation` ｜ E `const_mean_info`

**不得加入**: Wiener PF, rate model, 新迁移方法, 新 MMD lambda, mission feature
arm。**B6 是收口, 不是扩方法。**

---

## §11 公平性

同一 `n_event` 与同一 `seed` 下, target / ft / mmd 必须保持完全一致:
exact train IDs, exact val IDs, exact test IDs, batch order signature,
init weight signature, optimizer, training budget, early stopping,
checkpoint rule, censoring loss, evaluator。

**唯一允许的差别**: source initialization; MMD term (仅 MMD arm)。

> 实现要点: `DataLoader` 的 `generator` 是**有状态**的, 每个 epoch 都会推进。三个
> 方法顺序训练时若共用同一 loader 对象而不重置, 第二、三组拿到的是"接着走"的 batch
> 顺序。因此每组开训前必须 `reset_target_batch_order`, 并在 reset 之后、训练之前量
> `batch_order_signature`, 量完再 reset 一次 (遍历会推进 generator)。

---

## §12 checkpoint 选择

**全部 validation-only。**

禁止: test early stopping ｜ test checkpoint selection ｜ 看到某 seed 的 test 之后
重训 ｜ 重跑一个"看起来失败"的 seed ｜ 自动 retry。

训练崩溃也必须作为正式结果保留, 除非存在确定的代码错误 / NaN —— 那时整个 cell 标
`INVALID`, **不得换 seed**。

---

## §13 主指标

`test_info_trajectory_macro_rmse`

```
gain_ft  = target_only − source_finetune
gain_mmd = target_only − source_mmd_finetune
正数 = source 方法更好
```

---

## §14 每档必报

info macro RMSE ｜ info pooled RMSE ｜ full macro RMSE ｜ MAE ｜ macro corr ｜
PSR ｜ catastrophic rate ｜ warning coverage ｜ miss rate ｜ false alarm ｜ PH ｜
alpha-lambda ｜ convergence; 外加 short / medium / long event-lifetime-bin RMSE。

删失轨迹: 无真 EOL 的指标一律保持 **NaN**, **不得转成 0**, 并同时给出
`n_evaluable`。

---

## §15 PRIMARY (n=5) 正式统计判定 —— 十条

FT 与 MMD **各自独立判定**。正式标签
`B6_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER` 只在某个迁移方法**十条全满足**时给出:

| # | 条件 |
|---|------|
| 1 | mean paired gain > 0 |
| 2 | median paired gain > 0 |
| 3 | improve_count ≥ 4/5 |
| 4 | seed 级 paired bootstrap 95% CI 下界 > 0 |
| 5 | short/medium/long 中至少 2/3 段 gain ≥ 0 |
| 6 | macro corr 不低于 target 超过 0.02 |
| 7 | catastrophic rate 不高于 target |
| 8 | warning miss rate 不高于 target + 0.05 |
| 9 | same-data-same-budget fairness PASS |
| 10 | 该 source 方法至少胜过 `const_mean_info` |

FT 与 MMD 都失败 → **`B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER`**。

**禁止降低门槛。** NaN 一律判不通过, 不当作通过。

---

## §16 secondary 不得推翻 primary

即使 n=3 的 MMD 以很大幅度 5/5 全胜, 只要 primary n=5 未通过, 最终正式判定仍然必须
是 `B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER`。

可以额外写 `SECONDARY_LOW_LABEL_SIGNAL_AT_N3`, 但必须标注
**`NOT_PRIMARY_CONFIRMATORY_EVIDENCE`**。

---

## §17 敏感性趋势

x = `n_event_labeled` (3, 5, 10, 21), y = paired gain, FT 与 MMD 分开; 输出 mean /
median / CI / improve_count。分析是否出现"标签越少增益越大", **但不要强行拟合趋势
线** —— 只有 4 个采样点, 只做描述性分析。

---

## §18 damage_extrapolation 必须在主表

**不能变成脚注。** 每档报 `target − damage`, `ft − damage`, `mmd − damage`。

若 damage 仍明显领先, 必须原样写出:

> **学习模型未超过基于已知累计损伤结构的物理外推基线。**

不要因为比赛主题叫迁移学习而隐藏这个事实。

---

## §19 ENGINEERING_RECOMMENDATION

单一推荐。候选只有 `target_only`, `source_finetune`, `source_mmd_finetune`,
`damage_extrapolation`。禁止 const / oracle / true-rul lookup。

优先级冻结: (1) primary test info macro RMSE → (2) warning miss →
(3) catastrophic rate → (4) simplicity / deployability。

若 damage 仍明显第一, 继续推荐 `damage_extrapolation`。

---

## §20 B5 / B6 关系

* B5 = 全量目标域 confirmatory transfer test, 正式结论**永久**保持
  `B5_NO_POSITIVE_TRANSFER`。
* B6 = 失效标签稀缺 stress test。

| 情况 | 条件 | 最终结论 |
|------|------|---------|
| A | B5 负 + B6 primary 负 | `NO_POSITIVE_TRANSFER_SUPPORTED` |
| B | B5 负 + B6 n=5 正 | `NO_GENERAL_POSITIVE_TRANSFER, CONDITIONAL_BENEFIT_UNDER_FAILURE_LABEL_SCARCITY` |

**绝对禁止把情况 B 写成"迁移学习总体显著优于 Target-only"。**

---

## §21 飞轮线最终迁移结论

写在 `docs/basilisk_b6/final_conclusion.md`, 只允许两种主结论之一:

* A `FINAL_TRANSFER_CONCLUSION = NO_POSITIVE_TRANSFER_SUPPORTED`
* B `FINAL_TRANSFER_CONCLUSION = CONDITIONAL_LOW_LABEL_TRANSFER_ONLY`

必须分别说明: B5 全量证据 ｜ B6 低标签证据 ｜ lifetime-bin 行为 ｜ warning 行为 ｜
damage 基线比较。

---

## §22 陈旧 B2.1 lifecycle guard

`tests/basilisk_b21/test_b21_discipline.py::test_b21_b5_absent_and_never_auto_run`
失败, 因为 B5 已被人工授权运行。

**本阶段不得悄悄删除该测试, 也不得为了 pytest 全绿修改冻结的 b21 test。**

必须验证: 其唯一失败原因确定是"B5 原本要求缺席, 现在因人工授权而存在"; 不是算法
错误; 不是哈希变化。记录写入 `docs/basilisk_b6/stale_guard_retirement.md`
(`guard_purpose` / `authorization_transition` / `why_obsolete` /
`replacement_guard`)。

B6 的新 guard (`b6_must_stay_absent`) 必须检查: B7/S7 数字修改产物不存在;
B8/S8 packaging 尚未运行。

最终测试结果可以报告**恰好 1 个已知 stale lifecycle guard 失败**, 但必须证明**所有
算法性测试通过**。该 guard 在 B7/S8 code-freeze-governance 阶段正式退休。

---

## §23 必须新增的测试 (至少)

`test_b6_old_artifacts_unchanged` ｜ `test_b6_split_exactly_b21` ｜
`test_b6_label_subset_manifest_frozen` ｜ `test_b6_n3_bin_coverage` ｜
`test_b6_n5_bin_coverage` ｜ `test_b6_n10_bin_coverage` ｜
`test_b6_all_censored_train_retained` ｜
`test_b6_unused_events_not_relabelled_censored` ｜ `test_b6_new_formal_seeds` ｜
`test_b6_same_batch_order_across_methods` ｜ `test_b6_same_init_signature` ｜
`test_b6_validation_only_checkpoint` ｜ `test_b6_seed_level_bootstrap` ｜
`test_b6_primary_is_n5` ｜ `test_b6_secondary_cannot_override_primary` ｜
`test_b6_damage_in_main_table` ｜ `test_b6_no_mmd_search` ｜
`test_b6_final_conclusion_consistent`

只允许已知的 stale B21 guard 失败; **任何第二个 failure → `B6_INVALID`, 停止。**

---

## §24 强制执行顺序

1. `verify_baseline.py --tag before`
2. `freeze_protocol.py`
3. `build_label_subsets.py`
4. `audit_label_subsets.py` —— **必须在此打印并人工确认 n=3/5/10/21 IDs 与
   lifetime-bin counts**
5. `run_formal_matrix.py` —— 矩阵 = 4 档 × 3 训练方法 × 5 seeds = **60 次训练**;
   基线单独评估; **禁止自动 retry**
6. `paired_statistics.py`
7. `summarize_matrix.py`
8. `final_transfer_verdict.py`
9. `pytest tests/basilisk_b6 -q`
10. `pytest tests/ -q`
11. `verify_baseline.py --tag after`

---

## §25 输出产物

`checkpoints/basilisk_b6/{protocol_hash, label_subset_manifest, all_metrics,
paired_statistics, final_verdict}.json`;
`docs/basilisk_b6/{baseline_contract.json, protocol.md, label_subset_manifest.md,
results.md, transfer_gain_vs_labels.md, lifetime_bin_results.md,
warning_results.md, final_conclusion.md, limitations.md, REPRODUCE.md,
stale_guard_retirement.md}`; `STATUS_BASILISK_B6.md`。

---

## §26 出口

B6 完成即停。**不自动运行 B7。不自动运行 B8。不新增算法。不调 MMD。不改 split。
不改 HI/RUL。** 下一步只允许 B7 出图/报告与 B8 打包/Docker。
