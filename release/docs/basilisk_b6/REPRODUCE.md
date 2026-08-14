# BASILISK-B6 复现说明

**阶段**：`FAILURE_LABEL_SCARCITY_FORMAL_MATRIX`
**科学问题**：当目标域只有少量**完整失效轨迹**可用于训练时，源域预训练是否比
Target-only 更有价值？

> 稀缺轴是 `n_event_labeled`（event-observed 失效标签轨迹数），
> **不是** target 轨迹总数。真实航天约束是失效标签稀缺，而退化 / 未失效运行数据
> 仍可大量存在——所以每一档都保留 train 中**全部 24 条** censored 轨迹。

B6 是飞轮线**最后一个**允许产生新核心实验数字的阶段。

---

## 0. 环境

```bash
# Win10 + conda（本阶段实际运行环境）
conda activate DP_env
python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
```

全流程 CPU 可跑。仿真与划分跨 OS 完全一致；GPU 训练允许 ±5% 容差
（CUDA 非确定性）。所有命令均以仓库根目录为工作目录。

## 1. 前置冻结件（B6 不生成、只读取）

| 冻结件 | 用途 |
|---|---|
| `docs/basilisk_b21/split_manifest.json` | 划分（train 45 / val 31 / test 74）与寿命 bin 边界 |
| `checkpoints/basilisk_b5/protocol_hash.json` | B5 协议哈希（写入 B6 基线契约） |
| `checkpoints/basilisk_b5/summary.json` | B5 正式结论 `B5_NO_POSITIVE_TRANSFER` |
| 源域 checkpoint | 迁移臂的初始权重（**禁止改动**） |

关键哈希（必须逐字符相同，否则 `B6_INVALID`）：

```
split_sha256           = 23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932
protocol_sha256        = 433ccaf2515669c1f5afbc76029de068329afc7f67f0c10ae3372d14409b8412
subset_manifest_sha256 = f7fe1ababe7b23db5d0ba41a7c644505267f684400a22aebb5144bf6ce041b1d
```

## 2. 参数

唯一配置入口 `configs/wheel_basilisk_b6.yaml`（继承 b5 → b21 → …）。
**没有任何阈值硬编码在脚本里**；判定阈值在跑出任何 B6 数字之前就冻结进
`checkpoints/basilisk_b6/protocol_hash.json`，判定脚本从冻结件读取并与实时
config 交叉核对，不一致即 `B6_INVALID`。

| 项 | 值 |
|---|---|
| 输入 schema | `CORE_ONLY`，`n_features = 12`（**不得恢复 mission-feature 臂**） |
| 标签档 | `n_event_labeled ∈ {3, 5, 10, 21}` → train 27 / 29 / 34 / 45 |
| PRIMARY | **n = 5**（唯一允许产出正式低标签判定的档） |
| SECONDARY | n = 3, 10, 21（仅趋势，不得单独推翻 primary） |
| 正式 seeds | `[122, 123, 124, 125, 126]` |
| 禁用 seeds | 72–76, 92–94, 102–106, 112–116 |
| 子集选择种子 | `subset_seed = 20260814` |
| 寿命 bin 边界 | `[26846, 37395]`（取自 B2.1 冻结件，**不得从 B6 test 重算**） |
| MMD lambda | `1.0`（全程冻结，无搜索） |
| max_epochs / patience | 8 / 2 |
| bootstrap | 单位 = seed 级配对增益，n=5，5000 次重采样，种子 20260814 |

## 3. 完整复现命令（必须按此顺序）

```bash
# 1) 冻结前基线核验
python scripts/basilisk_b6/verify_baseline.py --tag before

# 2) 冻结协议与判定阈值（在任何 B6 数字之前）
python scripts/basilisk_b6/freeze_protocol.py

# 3) 生成标签子集清单（在任何训练之前）
python scripts/basilisk_b6/build_label_subsets.py

# 4) 人工确认各档轨迹 ID 与 bin 计数
python scripts/basilisk_b6/audit_label_subsets.py

# 5) 正式矩阵: 4 档 × 3 方法 × 5 seed = 60 次训练（基线另行评估, 禁止自动重试）
python scripts/basilisk_b6/run_formal_matrix.py

# 6) 配对统计 + 寿命分箱 + 告警指标
python scripts/basilisk_b6/paired_statistics.py

# 7) 主表与趋势汇总
python scripts/basilisk_b6/summarize_matrix.py

# 8) 十条门槛判定 + 最终迁移结论
python scripts/basilisk_b6/final_transfer_verdict.py

# 9) B6 自身测试
python -m pytest tests/basilisk_b6 -q

# 10) 全量测试
python -m pytest tests/ -q

# 11) 冻结后基线核验（before / after 必须逐项一致）
python scripts/basilisk_b6/verify_baseline.py --tag after
```

冒烟检查（**结果不得进入任何正式报告**）：

```bash
python scripts/basilisk_b6/run_formal_matrix.py --fast --levels 5
```

`--fast` 会把 `max_epochs` 压到 1，并在 `all_metrics.json` 里写下
`fast: true`；步骤 6/7/8 三个脚本都会拒绝消费 `fast: true` 的指标。

## 4. 产物

| 路径 | 内容 |
|---|---|
| `checkpoints/basilisk_b6/protocol_hash.json` | 冻结协议 + 十条判定阈值 |
| `checkpoints/basilisk_b6/label_subset_manifest.json` | 各档 train 轨迹清单（嵌套） |
| `checkpoints/basilisk_b6/all_metrics.json` | 60 个训练 cell + 2 个评估基线的全部指标 |
| `checkpoints/basilisk_b6/all_raw.npz` | 逐 cell 预测原始数组 |
| `checkpoints/basilisk_b6/paired_statistics.json` | seed 级配对增益 + bootstrap CI + 敏感性 |
| `checkpoints/basilisk_b6/lifetime_bins.json` | short / medium / long 分箱结果 |
| `checkpoints/basilisk_b6/warning_metrics.json` | 告警覆盖 / 漏报 / 误报 / PH / α-λ / 收敛 |
| `checkpoints/basilisk_b6/summary.json` | 四档主表 + damage 比较 + 趋势 |
| `checkpoints/basilisk_b6/final_verdict.json` | primary 十条门槛 + 工程推荐 + 最终结论 |
| `docs/basilisk_b6/*.md` | 协议、子集清单、结果、趋势、分箱、告警、结论、局限 |
| `STATUS_BASILISK_B6.md` | 阶段状态 |

## 5. 判定口径（复现时必须核对同一口径）

- **主指标**：`test_info_trajectory_macro_rmse`。
- **增益定义**：`gain_ft = RMSE(target_only) − RMSE(source_finetune)`，
  `gain_mmd` 同理。**正数 = source 方法更好。**
- **PRIMARY 十条门槛必须全过**才判
  `B6_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER`；任一条件取值为 NaN 一律计 FAIL。
- **右删失轨迹**无真 EOL 的指标保持 `NaN`，**绝不转 0**；空 bin 返回
  `NaN` + `n = 0`。
- **checkpoint 选择全部 validation-only**；禁止 test 早停 / test 选点 /
  看到 test 后重训 / 换 seed 重跑。训练崩溃保留为正式结果。
- **配对公平性**：同一 `n_event` + 同一 seed 下三组共用完全相同的
  train/val/test ID、batch 顺序签名、初始权重签名、优化器、训练预算、
  早停、checkpoint 规则、删失损失、评估器。唯一允许的差别是源域初始化与
  （仅 MMD 臂的）MMD 项。
- **显著性单位是 seed 级配对差**，禁止把端点数或轨迹条数当独立样本。

## 6. 跨平台容差

| 环节 | 容差 |
|---|---|
| 划分 / 标签子集 / 哈希 | **零容差**，必须逐字符相同 |
| CPU 训练与评估 | 同平台完全一致 |
| GPU 训练 | ±5%（CUDA 非确定性），判定结论不得因此翻转 |

若哈希不一致，**不要**继续跑后续步骤——那说明冻结链已断，应判 `B6_INVALID`。

## 7. 已知测试状态

全量测试允许且仅允许出现 **1 个**已知陈旧生命周期守卫失败：

```
tests/basilisk_b21/test_b21_discipline.py::test_b21_b5_absent_and_never_auto_run
```

原因、核实过程与接替守卫见 `docs/basilisk_b6/stale_guard_retirement.md`。
出现**第二个** failure 即判 `B6_INVALID` 并停止。本阶段**不得**为了让 pytest
全绿而修改这个冻结的 b21 测试。

## 8. 到此停止

B6 之后**只允许** B7（图表 + 报告）与 B8（打包 + Docker）。
不得自动运行 B7 / B8，不得新增算法，不得调 MMD，不得改 split，不得改 HI / RUL。
