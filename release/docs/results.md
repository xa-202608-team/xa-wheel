# 反作用飞轮组件 — 正式实验结果（冻结版）

**状态：`B7_REPORT_READY` 候选 — 本文档为最终结果表达，所有数字已冻结。**

本文档只呈现两个确认性阶段的结果：

| 来源 | 内容 | 判定 |
|---|---|---|
| **B5** | 全标签确认性对比（target 训练集 21 event + 24 censored） | `B5_NO_POSITIVE_TRANSFER` |
| **B6** | 失效标签稀缺性矩阵（n_event = 3 / **5 = PRIMARY** / 10 / 21） | `B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER` |

**最终结论**

- `FINAL_TRANSFER_CONCLUSION = NO_POSITIVE_TRANSFER_SUPPORTED`
- `ENGINEERING_RECOMMENDATION = damage_extrapolation`

> 措辞边界（全文强制）：本研究的结论是 **"在冻结的确认性协议下未能证明正向迁移"**，而**不是** "证明迁移无效"。两者的证据强度不同，不可互换。

**数据来源与不可变性**

本文档所有数字读取自 `checkpoints/basilisk_b7/frozen_result_index.json`
（`sha256 = 4ee10b5978f9ef93aab9b537aa507d408ce3dee893b4f886f2ab8c5db8738919`），
该索引由 `scripts/basilisk_b7/collect_frozen_results.py` 从 B1.8 / B1.9 / B2.1 / B5 / B6 的冻结产物只读汇总而成。

B7 阶段**未**训练任何模型、**未**重新评估任何 seed、**未**重新 bootstrap 任何置信区间、**未**引入任何新的评价定义。数值一致性由 `scripts/basilisk_b7/audit_report_numbers.py` 自动核对，措辞边界由 `scripts/basilisk_b7/audit_claims.py` 自动核对。

**主指标定义**

- 主指标 = `test_info_trajectory_macro_rmse`：先在每条 test 轨迹内部算 RMSE，再对轨迹取宏平均（不以时间点数量作为独立样本，避免长寿命轨迹主导）。
- 右删失（right-censored）轨迹没有真实 EOL，**不参与** RMSE / PH 等需要真值 EOL 的指标，也不伪造 EOL 补齐。
- 单位：RUL 已按 `rul_scale = 18408.599609375`（采样点）归一化，故 RMSE 为无量纲。

---

## 1. 方法集合

| 方法 | 类型 | 说明 |
|---|---|---|
| `damage_extrapolation` | 物理外推基线 | 由可观测退化 HI 反演损伤速率并线性外推至 `D = 1`。**工程推荐方法。** |
| `target_only` | 学习方法 | 仅用目标域（Basilisk 任务工况 + 项目自建退化模型）标签训练，无源域预训练。 |
| `source_finetune` | 学习方法（迁移） | XJTU-SY 源域编码器预训练 → 目标域微调。 |
| `source_mmd_finetune` | 学习方法（迁移） | 源域预训练 → 目标域微调 + MMD 域对齐正则。 |
| `const_mean_info` | 参考下界 | 常数预测（训练集均值）。**仅作为退化的参照线，不参与工程推荐候选集。** |

迁移作用在**健康指标 / 退化动力学层**，不在原始波形层：源域学退化趋势动力学，目标域使用遥测派生的可观测 HI。

---

## 2. Table A — B5 全标签确认性对比

判定：**`B5_NO_POSITIVE_TRANSFER`**。5 个确认性 seed `[112, 113, 114, 115, 116]`，目标域训练集 21 event + 24 censored。

| method | info macro RMSE | std | macro corr | warning coverage | miss rate | catastrophic rate |
|---|---:|---:|---:|---:|---:|---:|
| `damage_extrapolation` (physics) | **0.054312** | 0.000000 | 0.9858 | 1.0000 | 0.0000 | 0.0000 |
| `target_only` | 0.241024 | 0.005696 | 0.8880 | 0.4800 | 0.5200 | 0.0000 |
| `source_mmd_finetune` | 0.241607 | 0.004066 | 0.9114 | 0.4514 | 0.5486 | 0.0000 |
| `source_finetune` | 0.242659 | 0.005291 | 0.8980 | 0.3886 | 0.6114 | 0.0057 |

预登记门槛通过数：`source_finetune` 1/8，`source_mmd_finetune` 3/8。

### 2.1 B5 全标签配对增益（与 Table C 同一符号约定）

`gain = RMSE_target_only − RMSE_source_method`，**正数 = 迁移方法更好**。配对单位是 seed（5 个正式 seed 各贡献一个配对差值，**绝不按时间点配对**）。CI95 为 B5 已冻结的配对 bootstrap 区间，B7 未重新 bootstrap。

| 迁移方法 | mean gain | median gain | improve seeds | CI95 |
|---|---:|---:|---:|---|
| `source_finetune` | −0.001635 | −0.001797 | 1/5 | [−0.003828, +0.000704] |
| `source_mmd_finetune` | −0.000583 | −0.000786 | 2/5 | [−0.006589, +0.005423] |

两个方法的 mean gain 均为**负**（迁移更差），且两条 CI95 均**跨零** → 全标签条件下无任何一侧达到统计可分辨的增益。这是 `B5_NO_POSITIVE_TRANSFER` 的直接数值依据。

**读法**

1. 在全标签条件下，两个迁移方法相对 `target_only` 的配对增益（−0.000583 / −0.001635）其绝对值均**小于**各自 seed 间标准差（≈0.004–0.006），即差异被训练随机性完全淹没 → 无正迁移证据。
2. `source_mmd_finetune` 的 macro corr（0.9114）确实高于 `target_only`（0.8880），但 corr 单项优势未能带动主指标，也未通过预登记门槛集 → 只能作为方向性观察，不构成正迁移结论。
3. `coverage` 与 `miss` 必须成对阅读：三个学习方法的 miss rate 均在 0.52–0.61，即**超过半数失效轨迹未能在 EOL 前发出有效告警**。
4. `damage_extrapolation` 在 RMSE 上领先约 4.4 倍，且 miss rate = 0、coverage = 1.0。此处不作截断展示，差距如实呈现（见 Figure 3）。

---

## 3. Table B — B6 失效标签稀缺性矩阵

PRIMARY 判定：**`B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER`**。PRIMARY 档 = `n_event_labeled = 5`，在看到任何 B6 结果之前固定。5 个 seed `[122, 123, 124, 125, 126]`。

| n_event | role | target_only | source_finetune | source_mmd_finetune | const_mean_info | damage_extrapolation |
|---:|---|---:|---:|---:|---:|---:|
| 3 | SECONDARY | 0.335007 | 0.357837 | 0.317366 | 0.288629 | 0.054312 |
| **5** | **PRIMARY** ◀ | **0.339625** | **0.354721** | **0.310147** | **0.288629** | **0.054312** |
| 10 | SECONDARY | 0.298031 | 0.327030 | 0.292611 | 0.288629 | 0.054312 |
| 21 | SECONDARY | 0.238093 | 0.241492 | 0.244525 | 0.288629 | 0.054312 |

**读法**

1. 稀缺轴 = `event_observed_failure_labelled_trajectories`（event-observed 失效标签轨迹数），**不是** `total_target_trajectories`。每一档都保留全部 24 条 right-censored 训练轨迹 —— 稀缺的是"看到过失效"的标签，不是数据量。
2. 在 n_event ≤ 10 的三档，**所有学习方法都不如常数基线 `const_mean_info`（0.288629）**。这是本研究最重要的负面事实之一：低失效标签条件下，学习模型尚未稳定超过"不学习"。
3. 只有 n_event = 21（全标签）时，`target_only`（0.238093）与 `source_finetune`（0.241492）才优于常数基线。
4. `damage_extrapolation`（0.054312）在四档**全部领先所有学习方法**，且不随标签预算变化 —— 它不消耗失效标签。

---

## 4. Table C — 配对迁移增益 vs 失效标签预算

`gain = RMSE_target_only − RMSE_source_method`，**正数 = 迁移方法更好**。符号在 protocol 冻结时固定，不得反转。配对只在**同一标签档、同一 seed** 内进行；每个 seed 贡献一个配对差值（**绝不按时间点配对**）。误差区间为 B6 已冻结的 95% 配对 bootstrap CI，B7 未重新 bootstrap。

### 4.1 `source_finetune`

| n_event | role | mean gain | median gain | improve seeds | CI95 |
|---:|---|---:|---:|---:|---|
| 3 | SECONDARY | −0.022830 | −0.021927 | 2/5 | [−0.056178, +0.010519] |
| **5** | **PRIMARY** ◀ | **−0.015096** | **−0.010302** | **2/5** | **[−0.029398, −0.001197]** |
| 10 | SECONDARY | −0.028999 | −0.021720 | 0/5 | [−0.046837, −0.015607] |
| 21 | SECONDARY | −0.003399 | −0.003983 | 1/5 | [−0.005659, −0.000353] |

`source_finetune` 在四档全部为负增益，n=5 / 10 / 21 三档的 CI95 上界均 < 0，即**朴素微调在本设置下稳定有害**。

### 4.2 `source_mmd_finetune`

| n_event | role | mean gain | median gain | improve seeds | CI95 |
|---:|---|---:|---:|---:|---|
| 3 | SECONDARY | +0.017641 | +0.012986 | 3/5 | [−0.017116, +0.054113] |
| **5** | **PRIMARY** ◀ | **+0.029478** | **+0.043665** | **3/5** | **[+0.001531, +0.057866]** |
| 10 | SECONDARY | +0.005420 | +0.010316 | 4/5 | [−0.033723, +0.037254] |
| 21 | SECONDARY | −0.006432 | −0.001376 | 1/5 | [−0.017535, +0.001088] |

### 4.3 PRIMARY 档 MMD 的条件性信号 —— 必须带限定词阅读

在 `n_event_labeled = 5`（PRIMARY）档，`source_mmd_finetune` 出现方向一致的正信号：

- mean gain = **+0.029478**（正）
- median gain = **+0.043665**（正）
- CI95 lower = **+0.001531**（正）
- 寿命分箱增益非负箱数 = **3/3**（short / medium / long 全部非负，见 Table D）

**但同时未通过全部预登记门槛：**

| 未通过项 | 事实 |
|---|---|
| seed 一致性 | 仅 **3/5** seed 改善（2 个 seed 变差） |
| corr 门槛 | 未通过 |
| catastrophic 门槛 | 未通过 |
| const 对比 | 0.310147 **仍差于**常数基线 0.288629 |
| 门槛总计 | 通过 **6/10**，未通过条件编号 `[3, 6, 7, 10]` |

**因此该结果只能表述为"条件性低标签信号 / conditional low-label signal"，不得表述为"显著正迁移"、"迁移明显优于"或"证明 MMD 有效"。**

一个方法在主指标上有正的配对均值、却仍然差于常数预测，说明它改善的是"相对另一个同样不够好的学习方法"的排名，而不是达到了可用精度。

其余三档（3 / 10 / 21）为 SECONDARY，仅用于趋势描述，**不能覆盖 PRIMARY 判定**。趋势本身也不支持正迁移：MMD 增益在 n=21 转为负值，即标签充足时域对齐正则反而成为约束。

---

## 5. Table D — PRIMARY 档（n_event = 5）寿命分箱 RMSE 与预测性告警指标

寿命分箱边界 `[26846.0, 37395.0]`（采样点）来自 `docs/basilisk_b21/split_manifest.json::lifetime_bins.event`，`recomputed_from_b6_test = false`（未从 B6 test 重算边界，避免用测试集定义评价）。test 各箱轨迹数 `{short: 12, medium: 11, long: 12}`。

| method | RMSE short | RMSE medium | RMSE long | warning coverage | miss rate | false alarm | PH |
|---|---:|---:|---:|---:|---:|---:|---:|
| `damage_extrapolation` (physics) | **0.048726** | **0.051068** | **0.062872** | 1.0000 | 0.0000 | 0.0256 | 0.0000 |
| `target_only` | 0.324676 | 0.334724 | 0.359067 | 0.4286 | 0.5714 | 0.0154 | −0.0377 |
| `source_mmd_finetune` | 0.315279 | 0.304321 | 0.310355 | 0.3943 | 0.6057 | 0.0154 | −0.0176 |
| `source_finetune` | 0.341569 | 0.350776 | 0.371489 | 0.4914 | 0.5086 | 0.0205 | −0.0651 |

分箱增益非负箱数：`source_finetune` **0/3**，`source_mmd_finetune` **3/3**。

**读法**

1. `source_mmd_finetune` 在三个寿命箱**全部**取得非负增益，且增益随寿命增长（short +0.009397 / medium +0.030403 / long +0.048712）—— 这是 §4.3 条件性信号的分箱侧证据，但同样受 3/5 seed 限制。
2. **所有学习方法的 miss rate 在 0.51–0.61**，即超过半数失效轨迹未能在 EOL 前发出有效告警。三者的 PH（prognostic horizon）均为**负值**，意味着告警平均发生在 EOL 之后。**当前学习模型不满足可部署告警要求。** 不得因 RMSE 数值"看起来不大"就声称可直接投入部署。
3. `damage_extrapolation`：coverage = 1.0、miss = 0、PH = 0.0000、false alarm = 0.0256。它是本表中唯一具备告警可用性的方法，且**留在正表内，不放脚注**。
4. `const_mean_info` 不进入本表：常数预测没有随时间变化的告警行为，其 coverage / PH 无定义。
5. PH 只在 event-observed 轨迹上计算，right-censored 轨迹排除，不伪造 EOL。空箱按规则返回 NaN + n=0，不伪造 0。

---

## 6. 工程推荐与其边界

**推荐方法：`damage_extrapolation`（物理损伤外推）。**

理由：在 B5 全标签、B6 四个标签预算档、以及 PRIMARY 档全部三个寿命箱上，它的 RMSE 一致领先所有学习方法约 5–6 倍，且是唯一 miss rate = 0 / coverage = 1.0 / PH ≥ 0 的方法；它不消耗失效标签，因而在真实在轨失效样本极度稀缺的场景下具备结构优势。

**必须同时声明的局限（不可省略）：**

1. 本项目的目标域退化由**已知的**累计损伤方程生成（`dD/dt = g_duty · a_T(T) / L_ref`，`EOL iff D ≥ 1`）。物理外推基线因此拥有与生成机理同构的**强结构先验**。
2. 换言之，它在本仿真内代表的是 **model-informed upper-quality reference（模型知情的高质量参照）**，**不能声称在现实飞轮上也能达到同样精度** —— 现实飞轮的真实损伤律并未知。
3. 它被列为工程推荐，是因为它在本研究的所有冻结协议下都是最优且唯一告警可用的方法；**不是**因为"学习方法失败了所以拿它来遮掩"。这一基线自 B5 起就在正表内，全程未被隐藏。
4. `D ≥ 1` 是**项目定义的仿真失效状态，不是厂商硬件失效阈值规格**。Basilisk 提供任务工况（姿态动力学、控制器、轮速、指令力矩、任务模式、占空比）；退化模型、`D(t)`、EOL 与全部 RUL 标签均由本项目定义，**不是 Basilisk 生成寿命标签**。

---

## 7. 最终迁移结论

> **No positive transfer was supported by the frozen confirmatory protocol.**
>
> **在本研究冻结的确认性协议下，未获得足以支持正向迁移的证据。**

补充限定：

> 在低失效标签 `n_event = 5` 条件下，MMD 出现方向一致的条件性信号（mean / median / CI95 lower 均为正，三个寿命箱增益全部非负），但未通过全部预登记门槛（仅 3/5 seed 改善，corr 与 catastrophic 门槛未过，且仍差于常数基线），**不足以支持正向迁移主张**。

结论字段：

```
FINAL_TRANSFER_CONCLUSION = NO_POSITIVE_TRANSFER_SUPPORTED
ENGINEERING_RECOMMENDATION = damage_extrapolation
```

未来若要推翻此结论，需要的不是重新解释现有数字，而是新的确认性证据：更多 seed 以检验 3/5 的一致性、以及先让学习方法稳定超过 `const_mean_info` 这一更基本的门槛。

---

## 8. 配图索引

| 图 | 文件 | 内容 |
|---|---|---|
| Figure 1 | `docs/figures/basilisk_final/fig1_basilisk_health_trajectory.{png,svg}` | 单条冻结 test 轨迹：可观测 HI / 损伤代理 + EOL 标记；真实 RUL 与各方法预测 |
| Figure 2 | `docs/figures/basilisk_final/fig2_transfer_gain_vs_labels.{png,svg}` | 配对增益 vs 失效标签预算，误差棒为 B6 冻结 CI95 |
| Figure 3 | `docs/figures/basilisk_final/fig3_primary_method_comparison.{png,svg}` | PRIMARY 档四方法 × 三寿命箱 RMSE（主图保留完整比例） |
| Figure 4 | `docs/figures/basilisk_final/fig4_health_management_flow.{png,svg}` | 健康管理流程：任务工况 → 遥测 → 可观测特征 → HI → 删失感知 RUL → 告警 → 维护决策 |
| Figure 5 | `docs/figures/basilisk_final/fig5_split_coverage_diagnostic.{png,svg}` | B2 划分覆盖缺陷 → B2.1 覆盖感知划分（诊断用，**两图 test split 不同，其 RMSE 不构成算法公平比较**） |

图表生成脚本：`scripts/basilisk_b7/make_figures.py`（只读冻结产物）；轨迹与 seed 选择规则完全确定性（取中位数而非最优），记录于 `docs/basilisk_b7/figure_manifest.md`。

---

## 附录 A — 开发历程与被否决的中间结论（historical / rejected）

以下内容**仅作为方法论诊断记录保留，不构成本项目的任何最终结论**。主结论只使用 B5 与 B6。

| 阶段 | 判定 | 状态 | 说明 |
|---|---|---|---|
| B2 | `B2_GENERALIZATION_FAIL` | **historical — 保留为划分覆盖缺陷的诊断证据** | test split 未覆盖 short-life 尾部，导致泛化评估不稳定。此失败被公开保留而非隐藏。 |
| B2.1 | `B21_GENERALIZATION_PASS` | 现行 | 覆盖感知分层划分，short / medium / long 三箱在 train / val / test 均有覆盖。`split_sha256 = 23e2b944…`，`split_seed = 20260813`。 |
| B3X | `B3X_NO_STABILIZING_SIGNAL` | **historical — 已否决** | mission features 未提供稳定化信号，**不进入**最终正式输入（`mission_features_in_xT = false`）。 |
| B4X | `B4X_TRANSFER_STABILIZATION_SIGNAL` | **`EXPLORATORY_ONLY` — 不得作为正迁移证据** | 探索性稳定化信号。**不得写成 positive transfer**，其数字**不得混入**上文任何正式主表。 |

**B2 与 B2.1 的 RMSE 不可作算法公平比较** —— 两者 test split 不同（`079296e5…` vs `23e2b944…`）。Figure 5 因此只展示覆盖分布，不展示跨 split 的 RMSE 对比。

早期基于解析退化路线（analytic route）的数字与本 Basilisk 正式路线不可混用；smoke 级别运行、临时门槛表、以及一切"5% RMSE 改善"式的早期验收草案均已作废，不作为结论依据。

---

## 附录 B — 冻结产物指纹

| 产物 | 指纹 / 关键参数 |
|---|---|
| B1.8 数据集 | `content_sha256 = 61678e582f82bd136ffe509de7ec1274c07c39fe791a2a762fcf562a3d36a0f7`；150 轨迹；71 event / 79 censored；`event_fraction = 0.473333…`；`L_ref_years = 3.0`；`eol_threshold_D = 1.0`；`is_manufacturer_failure_specification = false` |
| B1.9 特征 | HI = `hi_damage_obs`（`HI_D_obs`）；`dt_s = 1800.0`（30 min 聚合）；`n_per_window = 1800`；`xT` 10 列 / core 8 列；`damage_proxy_in_xT = false`；`mission_features_in_xT = false` |
| B2.1 划分 | `split_sha256 = 23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932`；train 45 (21e/24c) / val 31 (15e/16c) / test 74 (35e/39c) |
| B6 协议 | `protocol_sha256 = 433ccaf2515669c1f5afbc76029de068329afc7f67f0c10ae3372d14409b8412`；`subset_manifest_sha256 = f7fe1ababe7b23db5d0ba41a7c644505267f684400a22aebb5144bf6ce041b1d` |
| B7 冻结索引 | `sha256 = 4ee10b5978f9ef93aab9b537aa507d408ce3dee893b4f886f2ab8c5db8738919` |

完整契约见 `docs/basilisk_b7/baseline_contract.json`；索引明细见 `docs/basilisk_b7/frozen_result_index.md`；复现说明见 `docs/basilisk_b7/REPRODUCE.md`。

**部署依赖规则：Basilisk 不进入 `requirements.txt` / `Dockerfile` / `docker-compose.yml`。** 最终复现链使用已提交的 mission profile 库；Basilisk 重新生成仅为可选的溯源路径（optional provenance path）。
