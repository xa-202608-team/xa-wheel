# BASILISK-B2 结果 —— Target-only Generalization Gate

**判定：`B2_GENERALIZATION_FAIL`（7 项判据通过 4 项）**

判据在 `protocol.md` §8 中于任何 test 数字产生之前写定，本文件只填数字、不改判据。
按 §7，本次 FAIL **不是**重新搜索超参的触发条件；按 §8，它必须作为 FAIL 上报，
不得改写成"部分成功"。

---

## §1 冻结输入

| 项 | 值 |
|---|---|
| feature h5 | `data/features/wheel/basilisk_b19/target_features.h5` |
| HI（主） | `hi_damage_obs` |
| split sha256 | `079296e5fba206da6ba9dab1078a8eaa6957e37ea97d0782b7a3fa01164113ae` |
| split_seed | `20260810` |
| gate seeds | `[72, 73, 74, 75, 76]` |
| fast_smoke | `false`（判定用完整预算） |

冻结训练超参（沿用 S2.5 已选 candidate D，**未重新调参**）：

| 项 | 值 |
|---|---|
| early_stop_metric | `info_macro_rmse`（val） |
| max_epochs | 8 |
| early_stop_patience | 2 |
| weight_decay | 1e-3 |

## §2 划分

trajectory-level，150 条（event-observed 71 / 右删失 79），首要分层键为
`event_observed / censored`，event 组内按 `eol_idx` 分位做 4 段二级分层，
删失组不分层。

| 划分 | n | event | censored | event_frac |
|---|---|---|---|---|
| train | 44 | 20 | 24 | 0.4545 |
| val | 31 | 15 | 16 | 0.4839 |
| test | 75 | 36 | 39 | 0.4800 |

`disjoint = true`，`covers_all = true`。

## §8 主结果（主指标 = test info trajectory-macro RMSE，越小越好）

| seed | target_only | const_mean_info | damage_extrap | gain vs const | PSR | macro_corr | best_ep |
|---|---|---|---|---|---|---|---|
| 72 | 1.0142 | 0.2886 | 0.0544 | **−0.7256** | 9.806 | 0.712 | 0 |
| 73 | **0.2563** | 0.2886 | 0.0544 | **+0.0323** | 0.742 | 0.825 | 0 |
| 74 | 0.7887 | 0.2886 | 0.0544 | **−0.5001** | 6.898 | 0.685 | 0 |
| 75 | **0.2646** | 0.2886 | 0.0544 | **+0.0240** | 0.820 | 0.836 | 0 |
| 76 | 0.7073 | 0.2886 | 0.0544 | **−0.4187** | 5.883 | 0.712 | 1 |

`const_mean_info` 常数值 = 0.50002（只由 train 的 event-observed 轨迹算出，
删失轨迹不参与该均值）。`damage_extrapolation` 不含随机性、与 seed 无关，
5 个 seed 复用同一份预测数组，因此其数字必然一致。

### 判据逐条

| # | 判据 | 需要 | 实得 | 结果 |
|---|---|---|---|---|
| 1 | target_only 优于 const 的 seed 数 | ≥ 4/5 | 2/5 | **FAIL** |
| 2 | mean paired gain > 0 | > 0 | −0.317628 | **FAIL** |
| 3 | bootstrap 95% CI 下界 > 0 | > 0 | −0.574034 | **FAIL** |
| 4 | macro corr > 0 的 seed 数 | ≥ 4/5 | 5/5 | PASS |
| 5 | PSR ≥ 0.30 的 seed 数 | ≥ 4/5 | 5/5 | PASS |
| 6 | warning / pre-EOL 指标有效 | 5/5 | 5/5 | PASS |
| 7 | checkpoint 只按 val 选 | 5/5 | 5/5 | PASS |

mean paired gain = **−0.317628**，95% CI **[−0.574034, −0.062896]**
（n = 5 个 **seed 级**配对差，2000 次重采样；重采样单位是 seed，
不是时间点 —— 时间点不是独立样本）。

→ **`B2_GENERALIZATION_FAIL`（4/7）**

### 判据 4/5 通过而 1/2/3 失败意味着什么

这不是"接近通过"。两组判据测的是不同东西：

- 4/5 测**形状**：5/5 个 seed 的 macro_corr 在 0.685–0.836，模型确实学到了
  RUL 随退化单调下降的趋势，不是常数输出。
- 1/2/3 测**尺度与稳定性**：这是失败的地方。

判据 5 用的是"PSR ≥ 0.30"这个**下界**门槛，它的设计意图是排除"预测几乎不动"
的退化解。崩塌 seed 的 PSR 是 9.806 / 6.898 / 5.883 —— 远**高于**下界，
所以形式上通过，但方向恰恰相反：预测标准差是真值的 6–10 倍。
这里必须明说：**判据 5 在本次运行中没有起到筛选作用**，
它通过不代表尺度健康。按 §8 判据不得事后增删，故此处只作记录，不改判据。

## 失败模式诊断

### 双峰，而非普遍变差

5 个 seed 明确分成两组：

| 组 | seeds | test macro | test median | RMSE>1.0 的轨迹数 | pred 全局 max |
|---|---|---|---|---|---|
| 正常 | 73, 75 | 0.2563 / 0.2646 | 0.2354 / 0.2410 | 0 / 0 | 2.72 / 1.76 |
| 崩塌 | 72, 74, 76 | 1.0142 / 0.7887 / 0.7073 | 0.2517 / 0.2404 / 0.2548 | 3 / 3 / 3 | 30.02 / 17.50 / 16.81 |

**中位数轨迹 RMSE 在全部 5 个 seed 都是 0.24–0.25，都优于 const 的 0.2886。**
崩塌 seed 里也有 21–23 / 36 条轨迹优于 const。macro 平均被少数轨迹拖掉：

| seed | 最差 3 条（tid: RMSE, pred_max） |
|---|---|
| 72 | 19: 14.260 (27.90) ｜ 109: 9.857 (16.07) ｜ 111: 3.396 (5.90) |
| 74 | 19: 10.026 (17.06) ｜ 109: 7.052 (9.66) ｜ 111: 2.388 (3.80) |
| 76 | 19: 8.564 (15.54) ｜ 109: 5.930 (9.05) ｜ 111: 2.064 (3.46) |

三个崩塌 seed 炸的是**同一组轨迹（19 / 109 / 111）**，这排除了"随机不稳定"，
指向一个可复现的结构原因。

### 结构原因：短寿命轨迹落在 train/val 覆盖之外

崩塌三条都是 test 中 EOL 最短的一批：

| 划分 | n_event | EOL min | med | max | EOL ≤ 22300 的条数 |
|---|---|---|---|---|---|
| train | 20 | 22336 | 34096 | 51102 | **0** |
| val | 15 | 22426 | 29928 | 49105 | **0** |
| test | 36 | 21658 | 31669 | 51576 | **4** |

崩塌轨迹 EOL = 21848 / 22281 / 27708，其中 19 与 109 比 **train 最短寿命
（22336）和 val 最短寿命（22426）都更短**。也就是说：test 含有 4 条比训练集
和验证集里任何一条都短命的轨迹，模型在这个区段没有训练信号，
验证集也没有能力发现它做错了。

这是 150 条 / 71 event 的样本量下，trajectory-level 分层划分的固有后果，
不是划分实现的 bug（`stratify_by = event_censored`，event 组内已按 EOL 分位
分 4 段；但 44/31/75 的规模下最短那段每层只有 1–2 条，无法同时覆盖三个划分）。

### 误差是全局尺度失控，不是末期发散

按 τ 分箱（info 口径，τ ≤ 1）看 seed 72 的全部 test event 取点：

| τ 区间 | n | RMSE | pred max |
|---|---|---|---|
| [0, 0.5) | 2169 | 3.126 | 15.63 |
| [0.5, 0.9) | 8705 | 2.832 | 27.90 |
| [0.9, 1.0) | 2377 | 3.236 | 27.49 |

三段 RMSE 相当（≈3），即误差**不集中在寿命末期**，而是整条轨迹的预测尺度
被整体放大。对照 seed 73 同样分箱为 0.299 / 0.239 / 0.338。

机理上与两点吻合（记录为观察，未做消融验证，B2 不允许改配置重跑）：
`rul_head` 是 `F.softplus`，上方无界（`src/models/tcn_encoder.py:114`）；
而 §4 的删失监督是**单侧**铰链 `relu(lb − pred)²`，只向上推、从不向下拉
（这是正确的 —— 右删失只提供下界，构造上界就是伪造结局）。
train 侧 65% 的删失取点归一化后 `lb = 1.0`（`rul` 与 `rul_lower_bound`
在 B1.9 中同被截断于 18408.6 步，`rul_scale = 18408.6`），
即在真值上限处存在持续向上的压力，配合无界头部，
在训练信号未覆盖的短寿命轨迹上可以跑飞。

### val 为什么选不出好模型（判据 7 通过，但选择信号无效）

| seed | best val info_macro | test info_macro |
|---|---|---|
| 72 | 0.23650 | 1.0142 |
| 73 | 0.23614 | 0.2563 |
| 74 | 0.24173 | 0.7887 |
| 75 | 0.24347 | 0.2646 |
| 76 | 0.23612 | 0.7073 |

5 个 seed 的 best val 全部落在 **0.2361–0.2435** 这个 0.007 宽的带内，
而 test 相差 4 倍。val 上崩塌 seed 的 `pred_max(info) = 1.05`，完全正常；
同一模型在 test 上 `pred_max(info) = 27.90`。
原因就是上一节：**触发崩塌的短寿命轨迹在 val 中不存在**，
所以 val 在原理上无法区分这两组 seed。

判据 7 检查的是"checkpoint 只按 val 选、没有偷看 test"，这一条确实成立
（`_train_with_early_stop` 结构上不接收 test loader，由
`tests/test_generalization_gate.py::test_earlystop_never_reads_test` 钉住）。
但"没有偷看 test"与"val 是有效的选择信号"是两件事，本次是前者成立、后者不成立。

## §9 删失分层指标

`n_endpoints`：event-observed test 13251（36 条）｜censored test 40989（39 条）。
`share_evaluable = 0.4800`，`rul_scale = 18408.6` 步。

| 分层 | target_only（各 seed） | const_mean_info | damage_extrap |
|---|---|---|---|
| event-observed test `info_macro_rmse` | 1.0142 / 0.2563 / 0.7887 / 0.2646 / 0.7073 | 0.2886 | 0.0544 |
| censored test `info_macro_rmse` | **NaN**（5/5） | **NaN** | **NaN** |
| censored `n_evaluable_rul_points` | **0** | 0 | 0 |
| censored 下界违反率 | 0.1226 / 0.1682 / 0.1286 / 0.1552 / 0.1482 | **0.8249** | 1.0000 |

删失分层的 RUL 指标恒为 `NaN` + `n = 0`，附
`rul_metrics_reason = "right_censored_no_true_rul_never_fabricated"`。
**没有把任何 NaN 转成 0**：右删失轨迹没有真实 EOL，它的 RUL 误差不是 0，
而是不可评估。删失轨迹的 τ 轴带
`tau_semantics = "observation_progress_not_life_fraction"` 标签，
它是观测进度，不是寿命比例。

删失轨迹上唯一合法可算的量是**下界违反率** `frac(pred < rul_lower_bound)`
（两侧都在归一化 RUL 单位下比较）。这一列给出一个 §8 主指标看不到的信号：
`target_only` 违反率 12–17%，而 `const_mean_info` 高达 **82.5%**、
`damage_extrapolation` 100%。即在 39 条删失轨迹上，
两个基线系统性地预测了比已知下界更短的寿命（提前报死），
而学习模型基本尊重下界。这不改变 §8 的判定（判据 1/2/3 只用 event-observed
的 RUL 误差），但它说明 const 基线的 0.2886 并非全面更优。

> 修正记录：本表的下界违反率在首次实现中因量纲错误（原始步数 `lb` 与归一化
> `pred` 直接比较）被恒定锁死为 1.0，已修复为先除 `rul_scale` 再比较，
> 并加回归测试 `test_lower_bound_violation_compares_in_normalized_units` 钉住。
> 该量不参与任何 §8 判据，修复前后 §8 判定与全部数字逐位不变。

## §6 基线说明

| 基线 | 角色 | test info_macro_rmse |
|---|---|---|
| A `const_mean_info` | §8 判据 1–3 的对比基线（最强平凡可部署基线） | 0.2886 |
| B `damage_extrapolation` | 物理/损伤外推参考 | 0.0544 |
| C `target_only` | 被考核的模型 | 见上表 |

`persistence` 与 `true_rul_lookup` 列入 `excluded_oracle_methods`，
不参与任何正式可部署方法排名。**未跑 source transfer**
（`source_transfer_run = false`），**未做 S3 truncation**
（`truncation_applied = false`）。

关于 B 基线远优于 A 与 C：`damage_extrapolation` 使用 `hi_damage_obs` 的因果
窗口斜率做外推，而本阶段的 EOL 定义正是 `D >= 1`，即该基线与失效判据同源，
其 0.0544 是"用退化量本身外推退化量"的结果，不构成对学习模型的公平上界。
B2 §6 只要求它作为参考量存在，不据此判定。

## §6 warning / pre-EOL 指标（判据 6）

5/5 个 seed 的 warning 指标结构有效：`n_observed = 36`、`n_censored = 39` 均非空，
`coverage_before_eol` 与 `miss_rate_before_eol` 严格互补，指标非 NaN。

实际数值必须一并公开，不能只报"有效"：

| seed | coverage_before_eol | miss_rate_before_eol | false_alarm_rate |
|---|---|---|---|
| 72 | 0.056 | **0.944** | 0.000 |
| 73 | 0.194 | **0.806** | 0.000 |
| 74 | 0.083 | **0.917** | 0.000 |
| 75 | 0.194 | **0.806** | 0.000 |
| 76 | 0.083 | **0.917** | 0.000 |

**漏报率 81–94%**，即在 `rul_threshold = 0.2`、`persistence = 3` 的设定下，
模型在 EOL 前几乎不发出预警。判据 6 只检验指标**有效性**（非空、互补、非 NaN），
不检验覆盖率高低 —— 把它写成"coverage 要高"会变成给闸门放水，
故设计上不那样写。但漏报率本身必须显式报出，不能因为只统计成功检测而隐藏。
`paired_lead_vs_const` 的 `n_common = 0`（const 恒定输出 0.5 > 阈值，从不报警），
故配对提前量为 NaN，不是 0。

## 结论与后续

B2 回答的问题是："在 B1.9 冻结的 Basilisk 数据 + D-based EOL + `hi_damage_obs`
上，Target-only 模型能否**稳定**优于最强平凡基线？"

答案是**否**。模型学到了正确的退化形状（corr 0.685–0.836，中位轨迹 RMSE
优于 const），但在训练/验证集未覆盖的短寿命轨迹上尺度失控，
且验证集在原理上无法发现这一失控，因此 5 个 seed 里 3 个崩塌。

**因此不进入 B3、不进入 B4。** 两者的前置条件均为 `B2_GENERALIZATION_PASS`，
该条件未取得。在 Target-only 基线尚未稳定成立之前做迁移增益对比，
得到的任何"迁移有效"结论都无法归因 —— 基线本身有 3/5 概率崩塌。

按 §7，本次 FAIL 不触发超参搜索；按 §10，B2 到此停止。
若后续要重新开闸，需要新的、独立立项的阶段，且其协议须在看到 test 之前写定。
可记录的候选方向（**本阶段不执行、不构成结论**）：
扩大 event-observed 轨迹数使短寿命段能同时覆盖 train/val/test；
或让 val 的分层显式包含最短寿命段，使选择信号具备发现该失效模式的能力。
