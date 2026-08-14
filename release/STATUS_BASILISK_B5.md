# STATUS — BASILISK-B5

**阶段**：`BASILISK_B5` — Formal Transfer Evaluation on Frozen B2.1 Split
**状态**：**已完成**
**最终判定**：**`B5_NO_POSITIVE_TRANSFER`**（§19 组合 **D**）
**工程推荐**：**`ENGINEERING_RECOMMENDATION = damage_extrapolation`**

这是项目里**第一个被允许产生"正式迁移结论"的阶段**。此前所有迁移相关信号
（S3、B3X、B4X）都是探索性的，不构成结论。

---

## 判定摘要

| 方法 | 门槛通过 | 判定 |
|------|:-------:|------|
| `source_finetune` | **1 / 8** | `B5_FT_NO_POSITIVE_TRANSFER` |
| `source_mmd_finetune` | **3 / 8** | `B5_MMD_NO_POSITIVE_TRANSFER` |
| 组合 | D | **`B5_NO_POSITIVE_TRANSFER`** |

在消除寿命覆盖不匹配（B2.1 划分）之后，用五个全新 confirmatory seed 与八条预登记
门槛检验，**两个迁移方法都没能证明相对 Target-only 的稳定增益**。这是一个正当的
负面结论，不是实验失败。

`source_finetune` 唯一通过的是 corr 条件；`source_mmd_finetune` 额外通过了
catastrophic 与 warning miss 两条，但四条核心统计条件（≥4/5 seed、mean、median、
bootstrap CI 下界）与分 bin 条件全部未通过。

---

## 三个正式问题的答案

| 问题 | 答案 |
|------|------|
| Q1 `source_finetune` 是否稳定优于 Target-only？ | **否**（gain 均值 −0.001635，1/5 seed 为正） |
| Q2 `source_mmd_finetune` 是否稳定优于 Target-only？ | **否**（gain 均值 −0.000583，2/5 seed 为正） |
| Q3 迁移模型是否仍不如 `damage_extrapolation`？ | **仍然不如，且差距约 0.187–0.188 RMSE（约 4.5 倍）** |

---

## 关键数字

主指标 = test information-zone trajectory-macro RMSE，5 seed 均值：

| 组 | RMSE | corr | miss rate | catastrophic |
|----|-----:|-----:|---------:|-------------:|
| `target_only` | 0.241024 | 0.8880 | 0.5200 | 0.000000 |
| `source_finetune` | 0.242659 | 0.8980 | 0.6114 | 0.005714 |
| `source_mmd_finetune` | 0.241607 | 0.9114 | 0.5486 | 0.000000 |
| `const_mean_info` | 0.288629 | NaN | 1.0000 | 0.000000 |
| **`damage_extrapolation`** | **0.054312** | **0.9858** | **0.0000** | 0.000000 |

配对增益（seed 级，n=5，5000 次 bootstrap，种子 20260814）：

| 项 | `gain_ft` | `gain_mmd` |
|----|---------:|----------:|
| mean | −0.001635 | −0.000583 |
| median | −0.001797 | −0.000786 |
| improve | 1/5 | 2/5 |
| CI95 | [−0.003828, +0.000704] | [−0.006589, +0.005423] |

两个 CI 都跨过 0：证据既不支持正转移，**也不足以断言迁移有害**。

分 bin 正增益数：ft **1/3**、mmd **1/3**（唯一为正的是 medium，不是 short）。
`localized_transfer_benefit` 两者均为 **False** —— 既不能说全周期提升，
也不能说短寿命局部收益。

---

## 冻结链

| 阶段 | 判定 |
|------|------|
| S2.5 | `S25_PASS` |
| S3 | `S3_NO_TRANSFER_SIGNAL` |
| S4 | frozen metrics |
| S5 | rate model done |
| S5B | `S5B_BASELINE_WEAK`（已关闭，不重开） |
| BASILISK_V1 | `BASILISK_V1_READY` |
| B1–B1.7 | 依次 CALIBRATION / SEMANTICS / PROVENANCE FAIL |
| B1.8 | `B18_SCENARIO_READY` / `B18_FEATURE_NOT_READY` |
| B1.9 | `B19_FEATURE_READY` |
| B2 | `B2_GENERALIZATION_FAIL`（终局，未追溯修改） |
| B3X | `B3X_NO_STABILIZING_SIGNAL` |
| B4X | `B4X_TRANSFER_STABILIZATION_SIGNAL`（探索性，**未被提升为正式结论**） |
| B2.1 | `B21_GENERALIZATION_PASS` |
| **B5** | **`B5_NO_POSITIVE_TRANSFER`** |

B5 的负面结论**不否证 B4X**，也**不改写 B2** —— 那是不同条件下的不同观测。

---

## 冻结哈希

| 项 | 值 |
|----|----|
| `split_sha256`（B2.1，未重划） | `23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932` |
| `protocol_sha256` | `592aeab38458bcec0673d54b7b8c1c1fb3f6cd1a58207cb73db49c2eb3786a57` |
| `config_sha256` | `09f02f4916c1eb115611916eb1e3235e255155523020abb48986f8eff018ee6f` |
| baseline contract | 415 个文件逐一哈希，前后两次核对均 `B5_BASELINE_CONTRACT_OK` |

正式 seed **112–116**（全新 confirmatory，与 102–106 / 92–94 / 72–76 无交集）。
输入 schema **`CORE_ONLY`**，无 mission 特征。Gate 八条门槛在出任何 B5 数字之前冻结。

---

## 产物

代码：`configs/wheel_basilisk_b5.yaml`、`scripts/basilisk_b5/`（7 个脚本 + `data_b5.py`）、
`tests/basilisk_b5/`（8 个测试文件）

数据：`checkpoints/basilisk_b5/`（`protocol_hash` / `formal_metrics` / `formal_raw.npz` /
`paired_gain` / `lifetime_bins` / `warning_metrics` / `summary` + 15 个 checkpoint）

文档：

| 文档 | 内容 |
|------|------|
| `docs/basilisk_b5/protocol.md` | 冻结协议（哈希后不可改） |
| `docs/basilisk_b5/baseline_contract.json` | 415 文件基线契约 |
| `docs/basilisk_b5/results.md` | **主结果表 + 八条门槛逐条判定 + 最终判定** |
| `docs/basilisk_b5/lifetime_bin_results.md` | 分 bin + 短寿命鲁棒性 + 删失退化 |
| `docs/basilisk_b5/warning_results.md` | coverage/miss/PH/α-λ/convergence |
| `docs/basilisk_b5/limitations.md` | **结论适用边界（必读）** |
| `docs/basilisk_b5/REPRODUCE.md` | 复现步骤与预期数值 |

B5 **未写入** `requirements.txt` / `Dockerfile` / `docker-compose.yml`。

---

## 测试状态

全量 `python -m pytest tests/ -q`：**964 passed / 1 failed / 28 skipped / 1 xfailed**
（`tests/basilisk_b5/` 单独 75 passed / 15 skipped）。

唯一的 failed 是
`tests/basilisk_b21/test_b21_discipline.py::test_b21_b5_absent_and_never_auto_run`，
**这是 `protocol.md` §1.1 在 B5 开跑前就预告过的到期守卫**：它断言 B5 产物必须缺席，
而 B5 已正当运行。该文件是 `b21_tests` 契约组的冻结项，**不修改**；防线由 B5 自己的
`b5_must_stay_absent` 组接管（内容换成 B6 产物，目前全部缺席）。
详见 `docs/basilisk_b5/limitations.md` §9。

基线契约前后两次核对：**1067 项全部不变**，`B5_BASELINE_CONTRACT_OK`。

---

## 必读的限制（详见 `limitations.md`）

1. **n=5 的 bootstrap CI 不是泛化误差置信区间**，只描述训练随机性下的方向稳定性。
2. **`damage_extrapolation` 以约 0.187–0.188 RMSE 全面压制**三个学习组，三个 bin 无一例外。
3. **三个学习组的报警器都不可部署**：miss rate 0.52–0.61，PH 全为负。
   `source_finetune` 的 PH "最好"是"报得更少"的副产物，不得单独引用。
4. **medium-bin 的正增益不是短寿命局部收益**，且与 long-bin 的负增益几乎等量反号。
5. **右删失侧 bin 退化**（`n_effective_bins = 1`，79 条观测时长相同），
   分 bin 只覆盖 35/74 条轨迹 —— 长寿命/未失效个体基本未被测量。
6. **未测低数据场景、未测其它 schema、未测第三种迁移方法** ——
   本阶段只能回答"这两个方案不成立"，不能回答"是否存在能成立的方案"。
7. **B5 的 `target_only` = 0.241024 ≠ B2.1 的 0.2441**（架构宽度 12 vs 10，是重跑）。

---

## 下一阶段建议（不自动执行）

> **B6: baseline formal matrix + negative-transfer conclusion**

`low_data_matrix_auto_run = False`，`forbid_auto_run_b6 = True`。

**低数据矩阵与截断矩阵都不自动跑，B6 不自动执行。B5 到此停止。**
