# BASILISK-B2.1 结果 —— Split-Coverage Robustness Gate

> **`SPLIT_COVERAGE_ROBUSTNESS`**
>
> **前置事实 (本阶段不覆盖、不重新解释):**
> **`B2_GENERALIZATION_FAIL` 保持终局** ·
> **`B3X_NO_STABILIZING_SIGNAL`** ·
> **`B4X_TRANSFER_STABILIZATION_SIGNAL` 仅为 `EXPLORATORY_ONLY`** ·
> **不存在任何正式的 positive-transfer 结论。**
>
> 本阶段**不跑迁移**, 不产生任何迁移结论。

判定: **`B21_GENERALIZATION_PASS`** (7/7)

---

## 1. 旧产物哈希核验

`scripts/basilisk_b21/verify_baseline.py` 契约 **974 项** (396 文件 + 578 数值),
`--tag before` 与 `--tag after` 两次均 `B21_BASELINE_CONTRACT_OK`。
链条: `... BASILISK_B2 → B2_GENERALIZATION_FAIL`,
`BASILISK_B3X → B3X_NO_STABILIZING_SIGNAL`,
`BASILISK_B4X → B4X_TRANSFER_STABILIZATION_SIGNAL`。

契约内新增钉死的 lifetime-support mismatch 证据 (使该缺陷不能被事后否认):

| 键 | 值 |
|---|---|
| `b2_train_min_event_eol` | 22336 |
| `b2_val_min_event_eol` | 22426 |
| `b2_test_min_event_eol` | **21658** |
| `b2_n_test_event_below_train_min` | **4** |
| `b2_lifetime_support_mismatch` | `True` |

---

## 2. 新划分 (`docs/basilisk_b21/split_manifest.json`)

`split_sha256 = 23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932`
`split_seed = 20260813`, level = trajectory, 150 条 (event 71 / censored 79),
`verdict = B21_SPLIT_VALID` (6/6 覆盖校验通过)。

预登记分箱 (数据集级三分位, 训练之前算定):

* event EOL edges = **[26846, 37395]** → short 24 / medium 23 / long 24
* censored 观测时长 edges = [52596, 52596] → **degenerate = true**,
  `n_effective_bins = 1` (79 条删失轨迹全部在同一仿真 horizon 截断)。
  如实报告, **未编造边界**, 也未改用任何寿命型代理量。

### 2.1 lifetime-bin 覆盖表

| split | n | event | censored | event EOL 范围 | short | medium | long |
|---|---|---|---|---|---|---|---|
| train | 45 | 21 | 24 | [21658, 50139] | ✓ | ✓ | ✓ |
| val | 31 | 15 | 16 | [21848, 51576] | ✓ | ✓ | ✓ |
| test | 74 | 35 | 39 | [22126, 47800] | ✓ (12) | ✓ (11) | ✓ (12) |

### 2.2 §5 六条硬性覆盖校验

| # | 要求 | 实得 |
|---|---|---|
| 1 | 无重叠 + 并集 = 150 | overlap=False, union_ok=True |
| 2 | 每个 event bin 在三个 split 都有 | short/medium/long × 3 |
| 3 | `train min EOL ≤ test min EOL` | **21658 ≤ 22126** |
| 4 | `val min EOL ≤ test min EOL` | **21848 ≤ 22126** |
| 5 | train 与 val 都含 short-life | train=True, val=True |
| 6 | 删失在三个 split 都有 | 24 / 16 / 39 |

对照 B2: `test 21658 < train 22336` 与 `< val 22426` —— **缺陷已消除**。
short bin 锚定 (§5.1 预登记规则, 按 EOL 升序):
train ← 21658, val ← 21848, test ← 22126。

---

## 3. 五 seed Target-only 结果 (seeds 102–106)

主指标 = test info trajectory-macro RMSE。基线 `const_mean_info` 与
`damage_extrapolation` 与模型走**同一个 evaluator**、**同一批 test 取点**。

| seed | target_only | const_mean_info | damage_extrap | gain vs const | macro corr | PSR |
|---|---|---|---|---|---|---|
| 102 | 0.2435 | 0.2886 | 0.0543 | **+0.0452** | 0.871 | 0.826 |
| 103 | 0.2379 | 0.2886 | 0.0543 | **+0.0507** | 0.912 | 0.855 |
| 104 | 0.2488 | 0.2886 | 0.0543 | **+0.0398** | 0.879 | 0.903 |
| 105 | 0.2441 | 0.2886 | 0.0543 | **+0.0445** | 0.887 | 0.852 |
| 106 | 0.2463 | 0.2886 | 0.0543 | **+0.0423** | 0.932 | 0.824 |

* **5/5** seed 优于 `const_mean_info`
* mean paired gain = **+0.044495**
* seed 级配对 bootstrap 95% CI = **[+0.041375, +0.047920]**, lower > 0
  (n = 5 个配对差值, 2000 次重采样, **绝不把 endpoint 当独立样本**)

对照 B2 的同一模型同一超参 (仅划分不同):
B2 五 seed target_only = 1.0142 / 0.2563 / 0.7887 / 0.2646 / 0.7073,
3/5 崩溃到 0.7–1.0; B2.1 五 seed 全部落在 **[0.2379, 0.2488]**。
**方差塌缩**是本阶段最显著的变化。

---

## 4. 分寿命段 RMSE (§11)

只由 **event-observed** 的 test 轨迹构成; bin 归属来自 manifest 预登记分箱。

| seed | short-life | medium-life | long-life |
|---|---|---|---|
| 102 | 0.2274 | 0.2615 | 0.2430 |
| 103 | 0.2196 | 0.2554 | 0.2402 |
| 104 | 0.2405 | 0.2559 | 0.2507 |
| 105 | 0.2351 | 0.2500 | 0.2478 |
| 106 | 0.2475 | 0.2495 | 0.2421 |
| **mean** | **0.2340** | **0.2545** | **0.2448** |

三方法在三段上的均值对照:

| 方法 | short | medium | long |
|---|---|---|---|
| target_only | 0.2340 | 0.2545 | 0.2448 |
| const_mean_info | 0.2887 | 0.2886 | 0.2886 |
| damage_extrapolation | **0.0487** | **0.0511** | **0.0629** |

**short-life 不再是崩溃区。** 这是与 B2/B3X/B4X 最本质的差别: 那三个阶段里
short-EOL 子集 RMSE 高达 0.86–23.65 (tid 19 一度到 36.997), 而真值上界为 0.999;
现在 short-life 反而是三段中最低的。

**必须同时说明**: `damage_extrapolation` (物理外推, 非学习基线) 在三段上都
显著优于 target_only —— 与 B2 一致。§12 的 gate 只以 `const_mean_info`
为判据 (它是最强的**可部署平凡**基线), 因此本阶段 PASS **不等于**
target_only 是最好方法。这一点不隐藏。

---

## 5. catastrophic 计数 (§11.1)

阈值 = 2.0 × median(target_only **validation** 逐轨迹 RMSE)。
**禁止 test 派生**; 逐 seed 的 `split = validation`, `derived_from_test = False`。

| seed | 阈值 | overall catastrophic | short-life catastrophic |
|---|---|---|---|
| 102 | 0.4780 | 0 | **0** |
| 103 | 0.6229 | 0 | **0** |
| 104 | 0.5390 | 0 | **0** |
| 105 | 0.5788 | 0 | **0** |
| 106 | 0.5782 | 0 | **0** |

5/5 seed 无 catastrophic short-life failure。对照 B4X: tid 19 在全部 18 个
cell 中都是 catastrophic。

---

## 6. warning 指标 (§11)

| seed | coverage | **miss** | coverage_before_eol | **miss_before_eol** | valid |
|---|---|---|---|---|---|
| 102 | 0.429 | **0.571** | 0.429 | 0.571 | ✓ |
| 103 | 0.486 | **0.514** | 0.486 | 0.514 | ✓ |
| 104 | 0.486 | **0.514** | 0.486 | 0.514 | ✓ |
| 105 | 0.371 | **0.629** | 0.371 | 0.629 | ✓ |
| 106 | 0.314 | **0.686** | 0.314 | 0.686 | ✓ |

**miss rate 显式报告, 未只统计成功检测。** coverage 与 miss 严格互补
(|c + m − 1| < 1e-9)。条件 6 问的是"指标可用", 不是"报警报得好" ——
0.31–0.49 的 coverage 依然很差, 只是比 B2 的 0.06–0.31 (miss 0.694–0.944) 好。

### 6.1 删失下界违反率

| seed | 102 | 103 | 104 | 105 | 106 |
|---|---|---|---|---|---|
| violation rate | 0.110 | 0.114 | 0.060 | 0.175 | 0.102 |

`frac(pred < rul_lower_bound)`, 两侧同在归一化 RUL 单位下比较。
右删失轨迹的 RUL 类指标恒为 **NaN + n = 0** —— 不伪造 EOL, 不把 NaN 转成 0。

---

## 7. §12 七条件判定

| # | 条件 | 需 | 实得 | |
|---|---|---|---|---|
| 1 | 优于 const_mean_info 的 seed 数 | ≥ 4/5 | **5/5** | PASS |
| 2 | mean paired gain > 0 | > 0 | **+0.044495** | PASS |
| 3 | bootstrap 95% CI lower > 0 | > 0 | **+0.041375** | PASS |
| 4 | macro corr > 0 的 seed 数 | ≥ 4/5 | **5/5** | PASS |
| 5 | 无 catastrophic short-life 的 seed 数 | ≥ 4/5 | **5/5** | PASS |
| 6 | warning 指标有效 | 5/5 | **5/5** | PASS |
| 7 | checkpoint 只按 validation 选 | 5/5 | **5/5** | PASS |

→ **`B21_GENERALIZATION_PASS` (7/7)**

判定由 `summarize_b21.py` 从 `conditions` 重算, 与 gate 写入值逐字一致,
不信任已落盘的 verdict 字段。

---

## 8. 出口 (§13/§14)

PASS → **停止**, 只写下下一阶段计划:

> **`B5 formal transfer on the frozen B2.1 split`**

**B5 未运行, 也不会被自动运行。** `b5_auto_run = false`;
`summarize_b21.py` 不 import 任何迁移训练入口、不写任何 `basilisk_b5` 路径;
契约把 `checkpoints/basilisk_b5/metrics.json` / `docs/basilisk_b5/results.md` /
`STATUS_BASILISK_B5.md` 钉为必须缺席 (两次核验均 MISSING)。
