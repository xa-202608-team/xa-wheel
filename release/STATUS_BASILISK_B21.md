# STATUS — BASILISK-B2.1

**阶段**: `SPLIT_COVERAGE_ROBUSTNESS` (Split-Coverage Robustness Gate)
**判定**: **`B21_GENERALIZATION_PASS`** (7/7)
**出口**: **停止**。下一阶段只写计划, 不执行。

---

## 前置事实 (本阶段不覆盖、不重新解释)

| 阶段 | 判定 | 状态 |
|---|---|---|
| BASILISK-B2 | `B2_GENERALIZATION_FAIL` | **保持终局** |
| BASILISK-B3X | `B3X_NO_STABILIZING_SIGNAL` | 不变 |
| BASILISK-B4X | `B4X_TRANSFER_STABILIZATION_SIGNAL` | **`EXPLORATORY_ONLY`** |
| 正式 positive transfer | —— | **不存在此结论** |

本阶段 `forbid_transfer: true`, 未跑迁移。

---

## 处理的结构缺陷

**lifetime-support mismatch** —— B2 的 test 含有比全部 train/val 更短命的
event 轨迹:

| | train | val | test |
|---|---|---|---|
| B2 min event EOL | 22336 | 22426 | **21658** ← 4 条低于 train 最小值 |
| B2.1 min event EOL | **21658** | **21848** | 22126 |

B2.1 使 `train min ≤ test min` 与 `val min ≤ test min` **按构造成立**
(§5.1 预登记 anchor 规则, 只看 EOL 的序, 不看 test 性能)。

---

## 结果摘要

* 划分: `B21_SPLIT_VALID` 6/6, `split_sha256 = 23e2b944...5c932`,
  train 45 / val 31 / test 74, 三个 event bin 在三个 split 全部非空。
* 五 seed `[102..106]` test info trajectory-macro RMSE:
  **0.2435 / 0.2379 / 0.2488 / 0.2441 / 0.2463** ——
  对照 B2 的 1.0142 / 0.2563 / 0.7887 / 0.2646 / 0.7073 (3/5 崩溃),
  **方差塌缩**。
* vs `const_mean_info`: **5/5 优**, mean gain **+0.044495**,
  bootstrap 95% CI **[+0.041375, +0.047920]**。
* 分寿命段 mean RMSE: short **0.2340** / medium **0.2545** / long **0.2448**
  —— short-life 不再崩溃, 且为三段最低。
* catastrophic short-life failure: **0 / 5 seed 全部为 0** (阈值只由 validation 派生)。
* macro corr 0.871–0.932 (5/5 > 0); PSR 0.824–0.903。
* warning: coverage 0.314–0.486, **miss 0.514–0.686** (显式报告, 未隐藏)。
* 删失下界违反率 0.060–0.175; 删失轨迹 RUL 指标恒 NaN + n=0。

## 必须并列陈述的事实

**`damage_extrapolation` (0.0543) 在每个 seed、每个寿命段都显著优于
`target_only` (0.2441)。** §12 gate 只以 `const_mean_info` 为判据, 故 PASS
在协议上成立, 但本阶段**没有证明学习模型优于物理外推**。

---

## 核验

| 项 | 结果 |
|---|---|
| B2.1 契约 (`verify_baseline.py --tag before/after`) | **974 项不变**, `B21_BASELINE_CONTRACT_OK` |
| B4X / B3X / B2 / B1.9 / B1.8 契约 `--tag after` | **914 / 838 / 743 / 668 / 575 全部不变 OK** |
| B2.1 专项测试 | `tests/basilisk_b21/` **51 passed** |
| 全量测试 | **890 passed, 13 skipped, 1 xfailed** |
| protocol 冻结 | `frozen_before_any_b21_number: true`, 哈希已记录 |
| B5 产物 | `checkpoints/basilisk_b5/`, `docs/basilisk_b5/`, `STATUS_BASILISK_B5.md`, `formal_transfer_verdict.json`, `positive_transfer_proven.json` —— **全部 MISSING (契约钉死)** |

未修改: 模型结构 / optimizer / loss / HI / RUL / 源域 checkpoint /
MMD lambda / mission features / B2·B3X·B4X 旧产物。

---

## 产物

| 文件 | 内容 |
|---|---|
| `configs/wheel_basilisk_b21.yaml` | b21 配置 (base = b2) |
| `docs/basilisk_b21/protocol.md` | 划分协议 (任何数字之前冻结) |
| `docs/basilisk_b21/split_manifest.json` | ID / strata / counts / EOL 范围 / split hash |
| `docs/basilisk_b21/results.md` | 完整结果 |
| `docs/basilisk_b21/limitations.md` | 十条局限与不可主张事项 |
| `docs/basilisk_b21/b5_plan.md` | 下一阶段计划 (`PLANNED_NOT_RUN`) |
| `scripts/basilisk_b21/{verify_baseline,build_b21_split,run_b21_gate,summarize_b21}.py` | 流程 |
| `checkpoints/basilisk_b21/{gate_metrics,summary,protocol_hash}.json` | 指标 |
| `tests/basilisk_b21/` | 51 项纪律测试 |

---

## 下一阶段 (仅计划)

**`B5 formal transfer on the frozen B2.1 split`** ——
详见 [docs/basilisk_b21/b5_plan.md](docs/basilisk_b21/b5_plan.md)。

**B5 未运行, 不得自动运行**, 需显式新指令。
