# FINAL_RESULTS — 正式结果汇总

**来源**：`checkpoints/basilisk_b5/summary.json`、`checkpoints/basilisk_b6/paired_statistics.json`、
`checkpoints/basilisk_b6/final_verdict.json`（均为冻结产物，本文件只做转录，不做重算）

---

## 1. 最终判定

| 字段 | 值 |
|---|---|
| `FINAL_TRANSFER_CONCLUSION` | **`NO_POSITIVE_TRANSFER_SUPPORTED`** |
| `ENGINEERING_RECOMMENDATION` | **`damage_extrapolation`** |
| 冻结证据 | `B5_NO_POSITIVE_TRANSFER`、`B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER` |
| `protocol_sha256` (B6) | `433ccaf2515669c1...` |
| `split_sha256` | `23e2b94445e698ea...` |
| `input_schema` / `n_features` | `core_only` / 12 |

---

## 2. B5 全标签正式表

主指标：test information-zone trajectory-macro RMSE（越小越好），5 formal seeds `112–116`。

| 方法 | info macro RMSE | std | info pooled RMSE | MAE |
|---|---|---|---|---|
| `target_only` | 0.241024 | 0.005696 | 0.254462 | 0.203165 |
| `source_finetune` | 0.242659 | 0.005291 | 0.255110 | 0.204378 |
| `source_mmd_finetune` | 0.241607 | 0.004066 | 0.254740 | 0.204780 |
| `const_mean_info` *(基线尺)* | 0.288629 | 0.0 | 0.288630 | 0.249960 |
| `damage_extrapolation` *(物理外推)* | **0.054312** | ~0 | 0.056606 | 0.038413 |

**读法**：两个迁移组都**没有**优于 `target_only`（0.2427 / 0.2416 vs 0.2410），
即全标签条件下源域预训练未带来正增益。三组均显著优于 `const_mean_info`，说明模型确实学到了东西，
只是**迁移这一步**没有贡献。

---

## 3. B6 低标签矩阵（主结果）

**问题**：当目标域只有极少量失效轨迹（event-observed failure labels）可供训练时，
source 预训练是否比 target-only 有价值？

| 项 | 值 |
|---|---|
| PRIMARY 标签量 | `n_event_labeled = 5`（event 5 + censored 24，共 29） |
| 次级标签量 | 3 / 10 / 21 |
| formal seeds | 122–126 |
| 门禁条件数 | 10 项（全部预先冻结于 `protocol_hash.json`） |
| 判定 | **`B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER`** |

PRIMARY (level 5) 方法排序（依据 `engineering_recommendation.ranking`）：

| 排名 | 方法 | info macro RMSE | warning miss rate | catastrophic rate |
|---|---|---|---|---|
| 1 | `damage_extrapolation` | **0.054312** | **0.000** | **0.000** |
| 2 | `source_mmd_finetune` | 0.310147 | 0.606 | 0.0057 |
| 3 | `target_only` | 0.339625 | 0.571 | 0.000 |
| 4 | `source_finetune` | 0.354721 | 0.509 | 0.0057 |

**读法**：即使把失效标签压到 5 条（迁移学习最该发挥作用的场景），
source 预训练在预先登记的十项门槛下**仍未**证明稳定优于 target-only。
`const_mean_info` / `persistence` / `true_rul_lookup` 属基线尺或 oracle，已排除在推荐候选之外。

---

## 4. 两个结论必须分开读

`final_verdict.json` 明确要求分别陈述，不可混为一谈：

1. **迁移判定为负**：源域预训练是否达到预登记门槛 —— 答案是**没有**。
2. **工程推荐 `damage_extrapolation`**：在该部件、该 HI 定义下，物理损伤外推以
   RMSE 0.0543、miss rate 0、catastrophic rate 0 全面领先，且更简单、更易部署。

第 2 条不是第 1 条的推论。即便迁移为正，`damage_extrapolation` 仍是首选
（`damage_still_first = true`）。

---

## 5. 结论的适用边界

该负结论**仅**在以下条件下成立，不可外推：

- B2.1 冻结划分、`core_only` 输入 schema、`n_features = 12`
- 本项目的仿真目标域与 `hi_damage_obs` 这一 HI 定义
- 失效定义 `D >= 1`（**项目自定义的仿真失效态**，非厂家硬件规格）
- 所列的正式 seed 集与预登记门槛

换 HI 定义、换输入 schema、换真实在轨数据，都需要重新评估。
局限性详见 `docs/basilisk_b5/limitations.md`、`docs/basilisk_b6/limitations.md`。

---

## 6. 图表

`docs/figures/basilisk_final/`（5 PNG + 5 SVG），对应 B7 出图阶段产物。

---

## 7. 复现

```bash
python scripts/run_all.py --verify   # 校验本文件所有数字的来源文件完整且哈希一致
python scripts/run_all.py --fast     # 真实训练链路冒烟（数值不入本文件）
```

重跑正式数字需 `--full --i-understand-this-overwrites`（会覆写冻结证据，数小时）。
