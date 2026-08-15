# STATUS — BASILISK-B6

**阶段**：`BASILISK_B6` — Failure-Label Scarcity Formal Matrix + Final Transfer Conclusion
**状态**：**已完成**
**PRIMARY 判定**：**`B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER`**
**最终结论**：**`FINAL_TRANSFER_CONCLUSION = NO_POSITIVE_TRANSFER_SUPPORTED`**（§20 情况 **A**）
**工程推荐**：**`ENGINEERING_RECOMMENDATION = damage_extrapolation`**

这是**飞轮线最后一个允许产生新的核心实验数字的阶段**。之后只允许
B7（图表 + 报告）与 B8（打包 + Docker）。

---

## 本阶段回答的唯一科学问题

> 当目标域只有少量**完整失效轨迹**可用于训练时，源域预训练是否比 Target-only 更有价值？

被削减的是 `event-observed failure-labelled trajectories`，**不是**目标域轨迹总量——
真实航天约束是失效标签稀缺，退化/未失效运行数据仍可大量存在。因此每一档都
**完整保留 train 中全部 24 条 censored 轨迹**，避免重演 B2 的寿命覆盖不匹配。

---

## 冻结件

| 项 | 值 |
|---|---|
| `protocol_sha256` | `433ccaf2515669c1f5afbc76029de068329afc7f67f0c10ae3372d14409b8412` |
| `subset_manifest_sha256` | `f7fe1ababe7b23db5d0ba41a7c644505267f684400a22aebb5144bf6ce041b1d` |
| `split_sha256`（完全复用 B2.1） | `23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932` |
| 输入 schema | `CORE_ONLY`，`n_features = 12`（不恢复 mission-feature arm） |
| 划分 | train 45（event 21 / censored 24）、val 31、test 74；**val / test 全程未动** |
| bin 边界 | `[26846, 37395]`（取自 B2.1 冻结件，未从 B6 test 重算） |
| `subset_seed` | `20260814` |
| 正式 seeds | `[122, 123, 124, 125, 126]`（禁用 72-76 / 92-94 / 102-106 / 112-116） |

## 标签稀缺档与 train 组成（§6/§7）

| n_event_labeled | event | censored | train 总数 | bin 配额 short/medium/long |
|---|---|---|---|---|
| 3 | 3 | 24 | 27 | 1 / 1 / 1 |
| **5（PRIMARY）** | 5 | 24 | **29** | 2 / 1 / 2 |
| 10 | 10 | 24 | 34 | 3 / 3 / 4 |
| 21 | 21 | 24 | 45 | 全部 |

各档 ID 在**任何训练之前**由每 bin 的确定性 shuffle 生成，嵌套包含
（n=3 ⊂ n=5 ⊂ n=10 ⊂ n=21）。未被选中的 event 轨迹**完全从该档 train 移除**——
未改标成 censored、未以无标签输入偷偷加回、未使用其任何 EOL 信息。

## 方法矩阵（§10，收口不扩方法）

训练：`target_only` / `source_finetune` / `source_mmd_finetune`
仅评估：`damage_extrapolation` / `const_mean_info`

**4 档 × 3 方法 × 5 seed = 60 次训练**，`fast: false`，耗时 5905 s，
100 组预测。零公平性失败，**无任何自动重试**，无换 seed，无 test 早停/选点。

---

## 主表（test information-zone trajectory-macro RMSE，5 seed 均值）

| n_event | `target_only` | `source_finetune` | `source_mmd_finetune` | `const_mean_info` | **`damage_extrapolation`** |
|---|---:|---:|---:|---:|---:|
| 3 | 0.335007 | 0.357837 | 0.317366 | 0.288629 | **0.054312** |
| **5（PRIMARY）** | **0.339625** | **0.354721** | **0.310147** | 0.288629 | **0.054312** |
| 10 | 0.298031 | 0.327030 | 0.292611 | 0.288629 | **0.054312** |
| 21 | 0.238093 | 0.241492 | 0.244525 | 0.288629 | **0.054312** |

## 配对增益（seed 级 bootstrap；正 = source 方法更好）

`source_finetune`：

| n_event | mean | median | improve | CI95 |
|---|---:|---:|:-:|---|
| 3 | −0.022830 | −0.021927 | 2/5 | [−0.056178, +0.010519] |
| **5** | **−0.015096** | −0.010302 | 2/5 | [−0.029398, −0.001197] |
| 10 | −0.028999 | −0.021720 | 0/5 | [−0.046837, −0.015607] |
| 21 | −0.003399 | −0.003983 | 1/5 | [−0.005659, −0.000353] |

`source_mmd_finetune`：

| n_event | mean | median | improve | CI95 |
|---|---:|---:|:-:|---|
| 3 | +0.017641 | +0.012986 | 3/5 | [−0.017116, +0.054113] |
| **5** | **+0.029478** | +0.043665 | 3/5 | [+0.001531, +0.057866] |
| 10 | +0.005420 | +0.010316 | 4/5 | [−0.033723, +0.037254] |
| 21 | −0.006432 | −0.001376 | 1/5 | [−0.017535, +0.001088] |

---

## PRIMARY（n=5）十条门槛判定（§15）

| # | 条件 | `source_finetune` | `source_mmd_finetune` |
|---|---|:-:|:-:|
| 1 | mean paired gain > 0 | FAIL −0.015096 | **PASS +0.029478** |
| 2 | median > 0 | FAIL −0.010302 | **PASS +0.043665** |
| 3 | improve ≥ 4/5 | FAIL 2/5 | FAIL 3/5 |
| 4 | bootstrap CI95 下界 > 0 | FAIL −0.029398 | **PASS +0.001531** |
| 5 | ≥ 2/3 寿命 bin gain ≥ 0 | FAIL 0/3 | **PASS 3/3** |
| 6 | macro corr ≥ target − 0.02 | PASS 0.8641 | FAIL 0.8161 vs 0.8576 |
| 7 | catastrophic 不更高 | FAIL 0.005714 vs 0.000000 | FAIL 0.005714 vs 0.000000 |
| 8 | warning miss ≤ target + 0.05 | PASS 0.5086 vs 0.5714 | PASS 0.6057 vs 0.5714 |
| 9 | 同数据同预算公平性 | PASS | PASS |
| 10 | 打赢 `const_mean_info` | FAIL +0.066091 | FAIL +0.021517 |
| | **合计** | **3 / 10 → FAIL** | **6 / 10 → FAIL** |

十条必须全过。两者都未通过 →
**`B6_PRIMARY_VERDICT = B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER`**。

MMD 臂是本项目迄今最接近正向的迁移证据（均值、中位数、CI 下界、三个 bin 全部为正），
但逐 seed 一致性只 3/5、形状相关性下降超容差、灾难性误差率更高，且 RMSE 仍高于
常数标尺，因此不构成正式结论。

## 寿命分箱（PRIMARY 档）

| bin | n | target | ft | mmd | gain_ft | gain_mmd |
|---|:-:|---:|---:|---:|---:|---:|
| short | 12 | 0.324676 | 0.341569 | 0.315279 | −0.016893 | +0.009397 |
| medium | 11 | 0.334724 | 0.350776 | 0.304321 | −0.016052 | +0.030403 |
| long | 12 | 0.359067 | 0.371489 | 0.310355 | −0.012422 | +0.048712 |

`localized_transfer_benefit`：ft `false`（三 bin 全负），mmd `false`（三 bin 全正，
非局部收益）。空 bin 返回 NaN + n=0，未伪造 0。

## 告警 / PH（PRIMARY 档；漏报率与覆盖率成对报出）

| 方法 | coverage | **miss rate** | false alarm | PH |
|---|---:|---:|---:|---:|
| `target_only` | 0.4286 | 0.5714 | 0.0154 | −0.0377 |
| `source_finetune` | 0.4914 | 0.5086 | 0.0205 | −0.0651 |
| `source_mmd_finetune` | 0.3943 | 0.6057 | 0.0154 | −0.0176 |
| `damage_extrapolation` | 1.0000 | 0.0000 | 0.0256 | 0.0000 |

三个学习方法漏报率均在 0.51–0.61，均不具备可部署告警能力。右删失轨迹上
依赖真 EOL 的指标保持 `NaN`，未转 0。

## vs damage 基线（§18，留在主表内）

| n_event | target − damage | ft − damage | mmd − damage |
|---|---:|---:|---:|
| 3 | +0.280695 | +0.303525 | +0.263054 |
| **5** | +0.285313 | +0.300409 | +0.255835 |
| 10 | +0.243719 | +0.272718 | +0.238299 |
| 21 | +0.183781 | +0.187180 | +0.190213 |

每档每方法都显著落后。§18 定论原样写出：

> **学习模型未超过基于已知累计损伤结构的物理外推基线。**

## §17 趋势（仅描述，未拟合趋势线；只有 4 个点）

- FT：(3, −0.022830, 2/5) (5, −0.015096, 2/5) (10, −0.028999, 0/5) (21, −0.003399, 1/5)
  —— 0/4 档均值为正，最大均值在 n=21，`fewer_labels_larger_gain = False`。
- MMD：(3, +0.017641, 3/5) (5, +0.029478, 3/5) (10, +0.005420, 4/5) (21, −0.006432, 1/5)
  —— 3/4 档均值为正，最大均值在 n=5，`fewer_labels_larger_gain = False`。

## SECONDARY（§16，不推翻 primary）

`SECONDARY_LOW_LABEL_SIGNAL_AT_N10` [`source_mmd_finetune`]，mean +0.005420，improve 4/5，
标记 **`NOT_PRIMARY_CONFIRMATORY_EVIDENCE`**。primary 未通过，正式判定不变。

---

## B5 与 B6 的关系（§20）

| 阶段 | 条件 | 结论 |
|---|---|---|
| B5 | 全部 21 条失效标签 | `B5_NO_POSITIVE_TRANSFER` |
| B6 | 失效标签稀缺，PRIMARY n=5 | `B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER` |

两者同为 negative → **情况 A** → `NO_POSITIVE_TRANSFER_SUPPORTED`。
**B5 的结论未被 B6 覆盖**（`b5_overwritten_by_b6 = false`）；
两阶段是平行证据，且独立得出同一工程推荐。

不存在"条件性低标签收益"的主张。**绝对禁止**表述为
"迁移学习总体显著优于 Target-only"。

## 诚实边界

本结论是"**未能证明正向迁移**"，不是"已证明迁移无效"。5 seed / 4 档 /
收口方法集 / 固定小预算下，统计功效不足与效应不存在无法区分。
完整边界见 `docs/basilisk_b6/limitations.md`。

---

## 产物

`checkpoints/basilisk_b6/`：`protocol_hash.json`、`label_subset_manifest.json`、
`all_metrics.json`、`all_raw.npz`、`paired_statistics.json`、`lifetime_bins.json`、
`warning_metrics.json`、`summary.json`、`final_verdict.json`

`docs/basilisk_b6/`：`baseline_contract.json`、`protocol.md`、
`label_subset_manifest.md`、`results.md`、`transfer_gain_vs_labels.md`、
`lifetime_bin_results.md`、`warning_results.md`、`final_conclusion.md`、
`limitations.md`、`REPRODUCE.md`、`stale_guard_retirement.md`

## 已知的一个 stale 生命周期 guard（§22）

`tests/basilisk_b21/test_b21_discipline.py::test_b21_b5_absent_and_never_auto_run`
失败，唯一原因是 **B5 已获人工正式授权运行**，因此
`docs/basilisk_b5/results.md` 与 `STATUS_BASILISK_B5.md` 不再缺席。
不是算法错误，不是 hash 变化（契约 hash 行本身 PASS）。
**本阶段未删除、未修改该测试**，在 B7/S8 退役。详见 `stale_guard_retirement.md`。
B6 的接力 guard 检查 B7/S7 数字产物缺席、B8/S8 打包未运行。

## 下一步（唯一允许）

- **B7**：图表 + 报告（不产生新核心实验数字）
- **B8**：打包 + Docker

不新增算法，不调 MMD，不改 split，不改 HI/RUL。**到此停止。**
