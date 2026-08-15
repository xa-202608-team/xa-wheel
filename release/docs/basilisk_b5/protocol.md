# BASILISK-B5 Protocol —— Formal Transfer Evaluation on Frozen B2.1 Split

**阶段**: `BASILISK_B5` ｜ **label**: `FORMAL_TRANSFER_EVALUATION`
**状态**: `PROTOCOL_FROZEN_BEFORE_ANY_B5_NUMBER`
**配置**: `configs/wheel_basilisk_b5.yaml`

> 本文件必须在**任何 B5 数字产生之前**写完并哈希 (`freeze_protocol.py`)。
> 之后所有判定只能按本文件执行, 不得因为看到结果而修改门槛、增加条件、
> 或把诊断项事后升格为 Gate 项。

---

## §0 本阶段定位

**这是第一次允许产生"正式迁移结论"的阶段。**

前置事实 (全部冻结, 不得重新解释):

| 事实 | 值 | 本阶段中的地位 |
|---|---|---|
| B2.1 判定 | `B21_GENERALIZATION_PASS` (7/7) | 地基。不成立则 `B5_INVALID` |
| B2.1 划分 | `split_sha256 = 23e2b944...5c932` | 直接复用, 禁止重新分层 |
| 寿命覆盖 | train/val/test 三段 event 寿命覆盖完整 | 本阶段得以成立的前提 |
| B2.1 Target-only | mean info macro RMSE ≈ **0.2441**, 5/5 稳定 | 见 §6.3 的重要限定 |
| B2.1 paired gain vs const | **+0.044495**, bootstrap CI 下界 > 0 | 说明学习模型确实优于常数锚点 |
| B2.1 catastrophic | 0/5 | — |
| `damage_extrapolation` | ≈ **0.0543** | **强基线**, 见 §14 |
| B3X 判定 | `B3X_NO_STABILIZING_SIGNAL` | 决定输入 schema = `CORE_ONLY` (§3) |
| B4X 判定 | `B4X_TRANSFER_STABILIZATION_SIGNAL` | **探索性**, 不得继承为正式结论 |

**B4X 的信号不是本阶段的结论。** B4X 是 `formal_stage: false` 的诊断阶段, 用的是
开发 seed `[92, 93, 94]`、B2 的旧划分、以及"寿命覆盖不完整"的 test 集。B5 必须
在冻结的 B2.1 划分上、用全新的正式 seed、独立重新证明或否证正转移。

**本阶段禁止重新调任何模型或迁移超参。**

### 三个正式问题

- **Q1**: 消除寿命覆盖不匹配之后, `source_finetune` 是否**稳定**优于 Target-only?
- **Q2**: `source_mmd_finetune` 是否**稳定**优于 Target-only?
- **Q3**: 即使存在正转移, 迁移模型是否**仍然输给** `damage_extrapolation`?

Q3 与 Q1/Q2 是**两个独立问题**。Q1/Q2 问"迁移是否有增益", Q3 问"这个方法是否
值得部署"。两者答案可以同时是"是" —— 见 §15。

---

## §1 禁止修改清单

以下一律不得修改 (由 `verify_baseline.py` 的契约逐项钉死, 任何一项改变 → 立即
`B5_INVALID`):

B1.8 ｜ B1.9 ｜ B2 ｜ B2.1 ｜ B3X ｜ B4X ｜ analytic S2.5–S5B ｜ source
checkpoints ｜ model architecture ｜ MMD lambda ｜ optimizer ｜ loss weights ｜
early stopping ｜ HI ｜ RUL ｜ mission profile ｜ split。

B5 的产物全部落在独立命名空间: `configs/wheel_basilisk_b5.yaml`、
`scripts/basilisk_b5/`、`tests/basilisk_b5/`、`checkpoints/basilisk_b5/`、
`docs/basilisk_b5/`、`STATUS_BASILISK_B5.md`。

### §1.1 关于 B2.1 契约的缺席断言

B2.1 的契约把 `checkpoints/basilisk_b5/metrics.json`、`docs/basilisk_b5/results.md`、
`STATUS_BASILISK_B5.md` 钉为"必须缺席" —— 那是为"B5 尚未运行"的状态写的。B5 正当
地取代该状态。处理方式:

- B5 的契约继承 B2.1 全部 974 项, 但把 `b21_must_stay_absent` 组中指向 B5 自身
  命名空间的条目**显式剔除**, 并在此处记录这一转移;
- 由 B5 自己的 `b5_must_stay_absent` 组接管这条防线, 内容改为 **B6 的产物**
  (§20: 本阶段完成即停);
- **不修改 B2.1 的契约文件本身** —— 它已冻结。B2.1 的 `verify_baseline.py --tag
  after` 从 B5 开始不再适用于 B5 命名空间那三项, 这一点在此明示, 而不是靠悄悄
  改文件掩盖。

---

## §2 Baseline 契约

`scripts/basilisk_b5/verify_baseline.py` 冻结并在结束时重新校验:

B1.8 dataset hash ｜ B1.9 feature hash ｜ B2.1 split hash ｜ **B2.1 train / val /
test IDs (三个集合各自独立求 sha256)** ｜ B2.1 protocol hash ｜ B2.1 metrics hash
｜ source checkpoint hash ｜ encoder architecture hash ｜ MMD lambda ｜ train
config ｜ evaluator / metrics hash。

ID 级哈希不靠"split.json 整文件哈希"间接覆盖: 整文件哈希只能告诉你"有人改了这个
文件", 分不清改的是注释还是把一条轨迹从 test 挪进 train。三个 ID 集合排序后各自
哈希, 加上逐 split 的条数与 min event EOL, 任何一条轨迹换组都会立刻暴露。

**任何冻结项改变 → 立即 `B5_INVALID`, 停止, 不出结论。**

---

## §3 正式输入 schema: `CORE_ONLY`

B3X 判定为 `B3X_NO_STABILIZING_SIGNAL` → 按 B3X 预登记的规则表:

| B3X 判定 | 输入 schema |
|---|---|
| `B3X_STABILIZING_SIGNAL` | `core_plus_mission` |
| `B3X_NO_STABILIZING_SIGNAL` | **`core_only`** ← 本阶段 |

**不加入 mission features。** 使用 B1.9 / B2.1 已冻结的 `x_T` schema。

绝不作为输入喂入: `hi_damage_obs`、`hi_friction`、`hi_a`、`b_true`、`D_true`、
`rul`、`rul_lower_bound`、`label_fail`、`eol_idx`、任何 mission 未来统计量、任何
EOL 信息。由 `assert_no_hi_in_input` 在数据装载时硬断言。

三组共享同一份输入 schema (§6)。

---

## §4 正式比较组

**训练** (三组, 不多不少):

1. `target_only`
2. `source_finetune`
3. `source_mmd_finetune`

**只评估不训练** (两组):

4. `const_mean_info`
5. `damage_extrapolation`

**不得新增第四种迁移方法。** 想试别的方法 → 另开阶段, 不在本阶段的正式表里。

---

## §5 正式 seed

**`[112, 113, 114, 115, 116]`** —— 全新。

禁止复用: B2.1 的 `[102..106]`、B4X 的 `[92, 93, 94]`、replay 的 `72/74/76`。

理由: 本阶段是正式 confirmatory evaluation, 不能继续消费开发 seed。开发 seed 已经
被用来看过结果、调过流程, 在它们上面再出"正式结论"等于用训练集报测试精度。
`run_formal_transfer.py` 硬断言正式 seed 与禁用列表**无交集**。

---

## §6 三组严格配对

同一 seed 下, 三组共享 (逐项由测试钉死):

- 相同 train / val / test IDs
- 相同 target batch 顺序
- 相同 target 预处理 (z-score 统计量只从 train 行估计)
- 相同 optimizer (Adam, `lr = 1e-4`, `weight_decay = 1e-3`)
- 相同 epoch 预算 (`max_epochs = 8`)
- 相同 early-stop 指标 (`info_macro_rmse`, patience 2)
- 相同 checkpoint 选择规则 (§7)
- 相同 evaluator

**唯一允许的差异**:

| 组 | 差异 |
|---|---|
| `target_only` | 随机 / target 初始化 |
| `source_finetune` | 冻结的 source 预训练初始化 |
| `source_mmd_finetune` | source 预训练初始化 + **已冻结的** MMD |

**不得因为某组失败而单独加 epoch。**

### §6.1 batch 顺序如何做到真的相同

`prepare_b2` 建的 `DataLoader(shuffle=True)` 不带 generator, 采样顺序取自全局
torch RNG; 而 `source_mmd_finetune` 还会对 shuffle 的 source loader 调
`next(src_iter)`, 同样消耗全局 RNG。结果是 MMD 组的 target batch 顺序会和另外两组
分叉 —— 那就不是配对实验了。

B5 沿用 B4X 的修法: target loader 与 source loader **各自持有显式 generator**,
种子分别为 `seed * 1000 + 7` 与 `seed * 1000 + 11`。这样 MMD 组消耗 source 随机性
不会污染 target 采样顺序, 三组的 target batch 序列逐 batch 一致。

### §6.2 初始权重按位相同

三组模型形状完全一致 (§6.3), 因此在 `_build_model` 之前调
`torch.manual_seed(seed)` 可使三组的初始权重**按位相同**。唯一区别只剩
`load_pretrained` 与 `use_mmd` 两个开关。

### §6.3 关键限定: B5 的 `target_only` 不是 B2.1 的 0.2441

source checkpoint 的 encoder 第一层期望 **12 通道**输入; B2 / B2.1 的
`train_target_only` 建的是 `(10, 10)`。要让三组共享同一架构 (§6 的公平性要求),
B5 必须像 B4X 一样把三组统一建成 `n_features = 12`。

因此:

- **B5 的 `target_only` 是本阶段的一次重跑, 其数值不等于 B2.1 的 0.2441。**
- 所有 gain 只在 **B5 自己的三组之间**计算, 绝不拿 B5 的迁移组去减 B2.1 的
  0.2441。跨阶段相减是不同尺子。
- B2.1 的 0.2441 在本阶段只作为"上一阶段的历史锚点"出现在文档里, 不进 gain 公式。

配置中以 `fairness.shared_n_features: 12` 与
`fairness.target_only_is_rerun_not_b21_number: true` 记录, 并由
`tests/basilisk_b5/test_training_fairness.py` 钉死。

---

## §7 Checkpoint 选择: 仅用 validation

- 选择依据: **validation 指标, 仅此一项**。
- 禁止: 按 test 选 checkpoint ｜ 按 test 早停 ｜ 跨 seed 挑最好的 checkpoint ｜
  跨 seed 换组。
- 逐组记录: `best_epoch`、`val_metric`、`checkpoint_sha256`。

结构性保证: `_train_with_early_stop` 的签名里**没有 test loader**, 且
`early_stop_value` 对任何含 `"test"` 的指标名直接抛错。

---

## §8 主指标与 gain 定义

**主指标** = test **information-zone trajectory-macro RMSE**
(`info_macro_rmse`, 逐轨迹算 RMSE 再取宏平均, 不可评估的轨迹计 NaN 而非 0)。

```
gain_ft(seed)  = RMSE_target_only(seed) - RMSE_source_finetune(seed)
gain_mmd(seed) = RMSE_target_only(seed) - RMSE_source_mmd_finetune(seed)
```

**正数 = 迁移带来改进。** 符号定义在此冻结, 由
`test_paired_statistics.py::test_b5_gain_sign_definition` 钉死。

---

## §9 正式 positive-transfer Gate

**FT 与 MMD 各自独立判定。八条全部满足才算正转移。**

| # | 条件 | 阈值 |
|---|---|---|
| 1 | gain > 0 的 seed 数 | **≥ 4/5** |
| 2 | `mean(gain)` | **> 0** |
| 3 | `median(gain)` | **> 0** |
| 4 | paired seed bootstrap 95% CI 下界 | **> 0** |
| 5 | 无系统性 corr 退化 | `corr_source_mean >= corr_target_mean - 0.02` |
| 6 | catastrophic error rate | 不高于 `target_only` |
| 7 | warning miss rate | 不高于 `target_only + 0.05` |
| 8 | short / medium / long 三个 bin 中 gain > 0 的 bin 数 | **≥ 2/3** |

判定标签:

- 八条全过 → `B5_FT_POSITIVE_TRANSFER` / `B5_MMD_POSITIVE_TRANSFER`
- 任一条不满足 → `B5_FT_NO_POSITIVE_TRANSFER` / `B5_MMD_NO_POSITIVE_TRANSFER`

**禁止降低门槛。** 阈值在本文件哈希时已冻结; 出数字后调阈值 = 事后编故事。

---

## §10 Paired bootstrap

- **单位 = seed 级 paired gain, n = 5**
- 重采样次数 **≥ 5000** (本阶段 = 5000), bootstrap seed 固定 `20260814`
- 使用**局部** `np.random.default_rng(seed)`, 绝不用全局种子
- 报告: `mean`、`median`、`std`、`ci95`、`improve_count`

禁止:

- 用 endpoint bootstrap 替代 seed bootstrap
- 把时间点当独立样本
- 用轨迹数量抬高显著性

n = 5 的 bootstrap 只描述"训练随机性下这个差值的方向是否稳定", 不是泛化误差的
置信区间。这一点在 `limitations.md` 中如实写明。

---

## §11 寿命分 bin 分析

**bin 边界直接取 B2.1 冻结的 `short` / `medium` / `long`**:
`edges = [26846.0, 37395.0]` (`quantile_basis =
dataset_level_event_eol_before_training`)。

**不得根据 B5 test 分布重算 bin。** 用 `np.digitize` 按预登记边界重建成员关系。

逐组逐 bin 报告: RMSE ｜ MAE ｜ corr ｜ warning coverage ｜ catastrophic count。
逐 bin 计算 `target - FT` 与 `target - MMD`。

空 bin → NaN + `n = 0`, 不伪造 0。

右删失轨迹的 bin 轴在 B2.1 已确认退化 (79 条 `observed_duration` 全部相同,
`n_effective_bins = 1`)。本阶段照旧如实报告, 不为了凑三个 bin 而发明 EOL。

---

## §12 短寿命鲁棒性 (本阶段起可正式使用)

在 `short` bin 内, 逐组报告: mean RMSE ｜ per-seed RMSE ｜ catastrophic count ｜
`pred_std` ｜ `true_std`。

**若迁移优势只出现在 short bin**, 判 **`localized_transfer_benefit`**, 并明确写成
"局部收益", **不要泛化成全周期提升**。

---

## §13 Warning / prognostics

复用已冻结的 S4 / Basilisk evaluator。报告: warning coverage ｜ miss rate ｜
coverage_before_eol ｜ miss_rate_before_eol ｜ false alarm rate ｜ PH ｜
alpha-lambda ｜ convergence。

- **NaN 必须保留 NaN, 并同时给出 `n_evaluable`。绝不转成 0。**
- 不得只统计成功检测而隐藏 miss rate。
- 右删失轨迹没有真 EOL → PH / alpha-lambda 保持 NaN, 不伪造 EOL 指标。

---

## §14 `damage_extrapolation` 是强基线

`damage_extrapolation` (B2.1 上 ≈ 0.0543) **必须留在主结果表里**, 不得降级成脚注。

```
transfer_vs_damage = RMSE_damage - RMSE_transfer
```

若为负 (迁移更差), 必须原样写出这句话:

> **迁移模型可能相对 Target-only 有增益, 但未达到物理 damage extrapolation 基线。**

**不能把"存在正转移"写成"最佳方法"。** 这是两个不同命题。

---

## §15 推荐结论与迁移结论分离

`ENGINEERING_RECOMMENDATION` ∈ {`target_only`, `source_finetune`,
`source_mmd_finetune`, `damage_extrapolation`}, **单独判定**。

允许出现 `B5_FT_POSITIVE_TRANSFER = true` 同时
`ENGINEERING_RECOMMENDATION = damage_extrapolation` —— **因为两者回答的问题不同**:
前者问"预训练是否帮到了这个模型", 后者问"现在该部署什么"。

推荐依据必须是**正式 test 表的主指标**, 不得依据任何机理性 / 叙事性解释。
`persistence` 与 `true_rul_lookup` 是 oracle, 排除在候选之外。

---

## §16 MMD 纪律

- 保持已冻结的 `mmd_lambda = 1.0`, 以及冻结的 `hi_bins`。
- **绝对禁止**: grid search ｜ seed 特异 lambda ｜ 按 B5 test 反调 ｜ 重新选择
  source layers。
- MMD 失败就如实判 `B5_MMD_NO_POSITIVE_TRANSFER`。不试图救它。

---

## §17 输出稳定性 (诊断项, 不进 Gate)

报告 `PSR = pred_std / true_std`, 且 **`pred_std`、`true_std`、`PSR` 三者同时给出**
—— 只给一个上界会掩盖"预测被压平"和"预测发散"是两个相反的病。

`PSR > 3` → 明确标注 **`HIGH_VARIANCE_WARNING`**。

**这是诊断项, 不新增到正式 Gate, 因为本阶段 protocol 必须在出数字前冻结。**
出数字后往 Gate 里加条件, 无论加的条件多合理, 都是事后调门槛。

---

## §18 主结果表

行: `target_only` ｜ `source_finetune` ｜ `source_mmd_finetune` ｜
`const_mean_info` ｜ `damage_extrapolation`

列: RMSE info macro ｜ RMSE pooled ｜ MAE ｜ corr ｜ PSR ｜ warning coverage ｜
miss rate ｜ PH ｜ alpha-lambda ｜ catastrophic rate

per-seed 明细另存为独立表, 不与主表混排。

---

## §19 最终判定

必须是以下四种组合之一:

| 组合 | 含义 |
|---|---|
| **A** | FT positive + MMD positive |
| **B** | FT positive + MMD negative |
| **C** | FT negative + MMD positive |
| **D** | `B5_NO_POSITIVE_TRANSFER` (两者皆否) |

外加一个**独立**的 `ENGINEERING_RECOMMENDATION` (§15)。

---

## §20 退出纪律

- **不自动跑低数据矩阵。**
- **B5 完成后停止。**
- 若成立正式正转移 → 下一阶段 `B6: low-data / truncation transfer matrix`
- 若不成立 → 下一阶段 `B6: baseline formal matrix + negative-transfer conclusion`
- **两者都不要自动执行**, 只写入建议。

---

## §21 执行顺序

```
1. python scripts/basilisk_b5/verify_baseline.py --tag before
2. python scripts/basilisk_b5/freeze_protocol.py --config configs/wheel_basilisk_b5.yaml
3. python scripts/basilisk_b5/run_formal_transfer.py --config configs/wheel_basilisk_b5.yaml
4. python scripts/basilisk_b5/analyze_paired_gain.py
5. python scripts/basilisk_b5/analyze_lifetime_bins.py
6. python scripts/basilisk_b5/analyze_warning_metrics.py
7. python scripts/basilisk_b5/summarize_b5.py
8. python -m pytest tests/ -q
9. python scripts/basilisk_b5/verify_baseline.py --tag after
```

---

## §22 沿用的既有纪律

- 右删失 test 轨迹不得伪造 EOL 指标。
- 禁止把时间点数量当独立样本。
- 不得只报成功检测而隐藏 miss rate。
- `persistence` 不进正式可部署方法排名。
- nPHM 不得成为模型选择指标。
- 无 NaN 被悄悄转成 0; 空 bin 返回 NaN + `n = 0`。
- 不硬编码阈值 —— 全部走 config。
- 迁移发生在 HI / 退化动力学层, 不在原始波形层。
- LLM 不参与数值寿命预测。
- 禁止全局 `np.random.seed()`。
- S5B 阶段已关闭, 不得重开或重新解释。
- 按"机台 / 电芯 / 器件"个体划分 train/val。
- Basilisk 不得写入 `requirements.txt` / `Dockerfile` / `docker-compose.yml`。
