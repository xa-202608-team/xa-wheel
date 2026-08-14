# BASILISK-B2.1 协议 —— Split-Coverage Robustness Gate

> **`SPLIT_COVERAGE_ROBUSTNESS`**
>
> **前置事实 (本阶段不覆盖、不重新解释):**
> **`B2_GENERALIZATION_FAIL` 保持终局** ·
> **`B3X_NO_STABILIZING_SIGNAL`** ·
> **`B4X_TRANSFER_STABILIZATION_SIGNAL` 仅为 `EXPLORATORY_ONLY`** ·
> **不存在任何正式的 positive-transfer 结论。**

本文件在**任何 B2.1 划分与任何 B2.1 数字存在之前**冻结, 其 sha256 记入
`checkpoints/basilisk_b21/protocol_hash.json`。

---

## 1. 本阶段唯一要解决的结构缺陷

**lifetime-support mismatch。**

B2 的划分里, test 含有比 train 与 val **全部** event 轨迹都更短命的 event
轨迹。B2/B3X/B4X 三个阶段共同量化的事实:

| split | min event EOL |
|---|---|
| train | 22336 |
| val | 22426 |
| test | **21658** |

test 有 4 条 event 轨迹的 EOL 低于 train 的最短寿命。也就是说, **失效区既不在
训练信号里, 也不在模型选择信号里**。B2 的泛化失败、B3X 的无稳定信号、B4X 的
"source 只是更保守"三者都指向同一区域。

本阶段做的唯一一件事: **换一个覆盖 short/medium/long 三档寿命的划分协议**,
在完全不动模型的前提下, 重问一次 B2 的泛化问题。

**这不是给 B2 翻案。** 若本阶段仍然 FAIL, `B2_GENERALIZATION_FAIL` 依旧成立,
只是排除了"划分覆盖不足"这一个可能解释。若本阶段 PASS, 得到的也只是
"在覆盖完整的划分上 target_only 能稳定学到任务"这一条独立结论,
它**不追溯修改** B2 的判定。

---

## 2. 冻结面 (§0)

由 `scripts/basilisk_b21/verify_baseline.py` 契约核验 (974 项)。禁止修改:

* model architecture / optimizer / loss
* HI (`hi_damage_obs`) / RUL / 删失契约
* source checkpoints / MMD lambda
* mission features
* B2 / B3X / B4X 的**任何**旧产物 (config / scripts / metrics / docs / status)

**本阶段唯一允许改变的是划分协议本身。** 训练配置逐项复用 B2 冻结值
(§8), 由 `train_b2.train_target_only` 内部对 `b2_frozen_training_expected`
的硬校验保证。

---

## 3. EOL 的使用边界 (§2)

event-observed 轨迹的 `eol_idx` 在本阶段**只用于划分分层**
(`eol_use: split_stratification_only`)。它不进入输入、不进入损失、
不进入任何评估指标的定义。

**右删失轨迹绝不被赋予 EOL。** 它们只有观测截止长度
(`observed_duration = len(x_T)`), 该长度**不是**寿命, 不得当作 EOL 使用
(与 B2 §4/§9 一致)。

---

## 4. 预登记的寿命分箱 (§2/§3)

分箱定义在跑数之前写定, 分位点由**数据集级**统计给出, 在**任何模型训练之前**
计算完成。

### 4.1 event-observed 轨迹 —— EOL 三分位

```
q1 = quantile(all_event_eol, 1/3)
q2 = quantile(all_event_eol, 2/3)
short  = eol <  q1        (lower tercile)
medium = q1 <= eol < q2   (middle tercile)
long   = eol >= q2        (upper tercile)
```

分位基准 = `dataset_level_event_eol_before_training`, 即全部 71 条
event 轨迹, 与划分结果无关、与任何模型输出无关。

### 4.2 右删失轨迹 —— 观测时长三分位, **单独一条轴**

```
c1 = quantile(all_censored_observed_duration, 1/3)
c2 = quantile(all_censored_observed_duration, 2/3)
short_obs / medium_obs / long_obs 同上规则
```

**退化情形必须如实报告。** 若全部删失轨迹的观测时长相同 (仿真在同一 horizon
截断就会如此), 三分位会退化成单一 bin。此时 manifest 中
`censored.degenerate = true`, 并记录实际 bin 数。**绝不为了凑三个 bin 而
编造边界**, 也绝不因此改用任何寿命型代理量 —— 那等于发明 EOL。

---

## 5. 联合分层与确定性划分 (§4)

* 层级: **trajectory**。同一条退化轨迹绝不跨 split。
* 比例: train 30% / val 20% / test 50%。
* 联合分层键: `(event_censored, lifetime_bin)`
  —— event 侧 3 个 bin + censored 侧 ≤3 个 bin, 每个层格独立按比例配额切分。
* `split_seed = 20260813`, 局部 `np.random.default_rng`, 不使用全局
  `np.random.seed()`。

### 5.1 short-life 锚定规则 (预登记)

§5 要求 `train min event EOL ≤ test min event EOL` 且
`val min event EOL ≤ test min event EOL`。随机配额切分不保证这一点, 因此
在 **short event bin 内部**使用一条确定性锚定规则:

> 将 short event bin 按 EOL **升序**排序, 前三条依次强制分配到
> `train / val / test`; 其余按配额随机分配。

`short_bin_anchor_order = [train, val, test]`。

这条规则是**覆盖驱动**的: 它只看 EOL 排序, 不看任何模型输出、不看任何 test
表现。它直接由 §5 的两条不等式反推而来 —— 全局最短的 event 轨迹进 train,
次短进 val, 第三短进 test, 于是两条不等式在构造上成立。

**它不是"挑一个好看的划分"**: 规则在数字存在之前写定, 且与性能指标无关。

---

## 6. 硬性覆盖要求 (§5)

划分产出后逐条硬校验, 任何一条不满足 -> **`B21_SPLIT_INVALID`**:

| # | 要求 |
|---|---|
| 1 | 三个 split 无轨迹重叠, 且并集 = 全部 150 条 |
| 2 | 每个 event lifetime bin 在 train / val / test 都有代表 |
| 3 | `train min event EOL ≤ test min event EOL` |
| 4 | `val min event EOL ≤ test min event EOL` |
| 5 | train 与 val 都含 short-life event 轨迹 |
| 6 | 删失比例在三个 split 都有体现 (每个 split 的 censored 数 > 0) |

## 7. 禁止用 test 表现挑划分 (§7)

**只跑一次确定性划分。** 若覆盖校验不通过, 输出 `B21_SPLIT_INVALID` 并停止,
**不得换 split_seed 反复尝试直到某次看起来好**。

`forbid_seed_search: true` 写进 config; `build_b21_split.py` 结构上不接受
"多 seed 择优"参数; `tests/basilisk_b21/` 有对应的纪律测试。

划分脚本不读取任何 metrics 文件, 不 import 任何模型或评估模块 —— 结构上
它拿不到 test 表现。

---

## 8. 训练配置 (§8)

逐项复用 B2 冻结的 Target-only 配置, 不重新调参:

| 项 | 冻结值 |
|---|---|
| `early_stop_metric` | `info_macro_rmse` (val) |
| `max_epochs` | 8 |
| `early_stop_patience` | 2 |
| `weight_decay` | 1e-3 |
| `finetune_lr` | 1e-4 |
| `batch_size` | 128 |
| `post_eol_weight` | 0.1 |
| `capped_weight` | 0.1 |
| `target_hi_key` | `hi_damage_obs` |
| censor-aware | `censor_hinge_eta = 1.0`, one-sided hinge |

checkpoint 选择**只用 validation**: `train_target_only` 的调用链里没有 test
loader (`prepare_b2(..., with_test=True)` 只把 test 交给 evaluator, 不交给
训练循环) —— 这是结构性保证, 不是口头承诺。

---

## 9. 五个新 seed 与三个方法 (§9)

* seeds = `[102, 103, 104, 105, 106]` (全新, 与 B2/B3X/B4X 的 seed 不重叠)
* 只比三个方法: `target_only` / `const_mean_info` / `damage_extrapolation`
* **不跑任何迁移方法** (`forbid_transfer: true`)。source checkpoint 在本阶段
  只是被契约钉死的只读文件。

`damage_extrapolation` 不含随机性且与 seed 无关 (划分固定、无模型),
逐 seed 缓存同一份预测数组, 各 seed 看到的该基线数字必然完全一致。

---

## 10. 主指标 (§10)

> **test info trajectory-macro RMSE**

`info` 口径 = 有真实 RUL 且未被 caliber 掩掉的 endpoint; 逐轨迹算 RMSE 再
等权平均。**绝不把时间点当独立样本**。

---

## 11. 同时报告的指标 (§11)

| 指标 | 口径 |
|---|---|
| macro correlation | 逐轨迹 pred/true 相关再平均 |
| PSR | pred/true 标准差比 |
| warning coverage / miss | `warning_lead_time`; miss rate 必须显式报告, 不得只报成功检测 |
| short-life subset RMSE | test 中 event 且 lifetime_bin = short 的轨迹 |
| medium-life RMSE | 同上, medium |
| long-life RMSE | 同上, long |
| censored lower-bound violation rate | `frac(pred < rul_lower_bound)`, 两侧同在归一化 RUL 单位下比较 |

**空子集一律 NaN + n = 0, 绝不写 0。** 右删失轨迹没有真实 RUL, 其 RUL 类
指标恒为 NaN + n = 0 —— 不伪造 EOL, 不把 NaN 悄悄转成 0。

`nPHM` / PH 只作描述性报告, **不得成为模型选择指标**。
`persistence` 不进入任何可部署方法排名。

### 11.1 catastrophic 阈值

```
threshold = 2.0 x median(target_only validation trajectory RMSE)
```

**只允许来自 validation**, 禁止用 test 分布选阈值
(`forbid_test_derived_threshold: true`)。不可评估就是不可评估:
rate = NaN, count = None, 绝不写 0。

"catastrophic short-life failure" 的判定 = 该 seed 的 short-life test 轨迹中
catastrophic 条数 > 0。

---

## 12. `B21_GENERALIZATION_PASS` 的七个条件 (§12)

全部满足才 PASS, 任一不满足 -> `B21_GENERALIZATION_FAIL`。

| # | 条件 | 阈值 |
|---|---|---|
| 1 | target_only 优于 `const_mean_info` 的 seed 数 | ≥ 4/5 |
| 2 | mean paired gain > 0 | > 0 |
| 3 | seed 级配对 bootstrap 95% CI lower > 0 | > 0 |
| 4 | macro corr > 0 的 seed 数 | ≥ 4/5 |
| 5 | 无 catastrophic short-life failure 的 seed 数 | ≥ 4/5 |
| 6 | warning 指标有效 | 5/5 |
| 7 | checkpoint 只按 validation 选 | 5/5 |

**bootstrap 单位 = 一个 seed 一个配对差值** (5 个数, `ci_unit:
one_paired_difference_per_seed`), 有放回重采样 2000 次,
`bootstrap_seed = 20260813`。绝不把 endpoint 当独立样本 —— 那会把 CI 缩小
两个数量级, 造出虚假显著性。

条件 6 的"有效"= 分母非空且各项算得出来 + `coverage_before_eol` 与
`miss_rate_before_eol` 互补。**不要求 coverage 高** —— 条件 6 问的是指标可用,
不是模型报警报得好; 写成"coverage 要高"就是给闸门放水。

---

## 13. 出口规则 (§13/§14)

* **`B21_GENERALIZATION_FAIL` -> 停止。不跑迁移。**
* **`B21_GENERALIZATION_PASS` -> 停止**, 并写下下一阶段计划:
  **`B5 formal transfer on the frozen B2.1 split`**。

**两种情形都不得自动运行 B5** (`forbid_auto_run_b5: true`)。
`checkpoints/basilisk_b5/` 与 `STATUS_BASILISK_B5.md` 由契约钉为必须缺席。

---

## 14. 表述纪律

* 本阶段不产生任何迁移结论。B4X 的 `B4X_TRANSFER_STABILIZATION_SIGNAL` 依旧是
  `EXPLORATORY_ONLY`, 本阶段不引用它作为正面证据。
* PASS 不等于"B2 判错了"。它只说明: 在覆盖完整的划分上, 该结论不再成立,
  于是"划分覆盖不足"作为 B2 失败的一个候选解释被确认为实质因素。
* FAIL 意味着: 修好 lifetime-support mismatch **之后**问题依旧,
  失败原因在别处 (已知的无界 softplus RUL 头 + 单边删失 hinge 外推失控)。
