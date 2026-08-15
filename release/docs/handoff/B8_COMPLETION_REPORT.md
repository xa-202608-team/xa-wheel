# B8-PREP 完成报告 — Project Cleanup + Docker Handoff Packaging

**阶段**：`BASILISK-B8-PREP`
**结论**：`B8_HANDOFF_READY`
**日期**：2026-08-10

> 本文件刻意**不**放在 `STATUS_BASILISK_B8.md` / `docs/basilisk_b8/packaging.md` /
> `checkpoints/basilisk_b8/package_manifest.json`。这三个路径被 B6 基线契约
> (`groups.b6_must_stay_absent`) 钉为 `MISSING`，创建它们会直接触发守卫失败。
> 故 B8 产物统一落在 `docs/handoff/` 与 `release/`。

---

## 1. 数据保护结果（最高优先级约束）

`DATA_TREE_PROTECTED = true` —— 全程零违规。

| 核查 | 结果 |
|---|---|
| `data/` 内被删除的文件 | **0**（`deletion_log.md` 中 `data/` 出现次数 = 0） |
| `data/` 内被改写的文件 | **0** |
| `data/` 内被移动/重命名的文件 | **0** |
| 契约登记的算法产物哈希核对（`checkpoints/` + `data/`） | **111 项全部一致** |
| 契约冻结文件消失 | **0** |
| `data/` 文件数 | 70,780 → 70,781 |

那 +1 是本阶段**新增**的 `data/MANIFEST_DOCKER_HANDOFF.md`（数据说明交付项），
属于新建文件，未触碰任何既有数据。

原仓库 `data/MANIFEST_DOCKER_HANDOFF.md` 中的路径后来发现有误（指向旧 analytic 产物），
但因 `data/` 为只读保护区，**未在原处修正**；修订版写在
`release/LEO_Bearings_Docker_Handoff/data/MANIFEST_DOCKER_HANDOFF.md`，
并在文首显式说明原文件已过时。

---

## 2. 清理结果

| 项 | 值 |
|---|---|
| 删除文件数 | 220 |
| 回收空间 | 5.37 MB |
| 删除范围 | **仅** `__pycache__` / `.pytest_cache` |
| 历史实验证据 | **全部保留**（原仓库留证，release 用白名单构建） |

明细见 `deletion_log.md`。SAFE_DELETE 前置了 9 层嵌套 guard（`data/`、`checkpoints/`、
`*.h5`、`*.pt`、`src/*.py`、`configs/*.yaml` 等），命中即 fail fast。

---

## 3. 交付包

| 项 | 值 |
|---|---|
| 路径 | `release/LEO_Bearings_Docker_Handoff/` |
| 文件数 | 437 |
| 体积 | 1886.39 MB |
| SHA256 全量校验 | **437/437 一致，0 坏，0 缺** |
| `verify_release.py` | **25/25 通过** → `RELEASE_VERIFY_PASS` |
| `run_all.py --verify` | 通过 |
| `run_all.py --fast` | **通过（真实训练，非桩）** |

---

## 4. 本阶段修正的两个真实缺陷

首版打包"验证通过"是假通过。两个缺陷互相掩盖：

### 缺陷 1 — 排除规则子串匹配，静默剔除依赖闭包

`build_release.py` 的 `EXCLUDE_PATTERNS` 用子串判断历史阶段脚本：

- 模式 `basilisk_b1` 连坐命中 `basilisk_b11` / `b18` / `b19`
- 模式 `basilisk_b2` 连坐命中 `b21`

后果：B5/B6 运行时真正需要的模块、B2.1 划分文件、B1.9 特征文件全部被剔除。
首版 release 缺失全部 8 项只读输入（含源域预训练权重 `source_tcn_pretrain.pt`），
**根本跑不了训练**。

**修复**：改为按路径段精确比对；新增 `REQUIRED_PHASE_SCRIPTS` 依赖闭包白名单，
优先级高于阶段排除。其中 `scripts/basilisk_b1/calibrate_degradation.py` 必须显式列出——
它是被 `scripts/basilisk_b11/calibrate_degradation.py` 用 `importlib` 按**文件路径**
动态加载的，不出现在任何 import 语句里，靠扫 import 图找不到它。

### 缺陷 2 — `--fast` 是桩，掩盖了缺陷 1

原 `--fast` 只打印 `"Full inference logic will be added here"` 然后报成功。
于是一个跑不了训练的包，照样输出 `✅ FAST PASSED`。

**修复**：`--fast` 改为调用**正式 B5 脚本本身**
（`scripts/basilisk_b5/run_formal_transfer.py`），差异只有 seeds `[112]`、
`max_epochs=1`、输出重定向到 `checkpoints/_smoke_b5`，并断言冻结证据未被污染。

### 附带修正的数据路径错误

首版白名单复制的两个"关键数据文件"其实都是旧 analytic 遗留产物
（`data/simulated/wheel/sim_v1/**`、`data/features/wheel/schema_v1/target_features.h5`），
在 `scripts/basilisk/generate_wheel_dataset.py` 里被显式标为 `FORBIDDEN_OUT`。
B5/B6 实际读的是 `data/features/wheel/basilisk_b19/target_features.h5`。已更正。

### 防复发

`verify_release.py` 新增第 11 项「B5/B6 dependency closure」，钉死 28 项依赖
（只读输入 + protocol.md + 运行时动态加载模块 + 冒烟 config）。缺任意一项即
`RELEASE_VERIFY_FAILED`。

---

## 5. 跑通证据

`--fast` 在**原仓库**与**release 包内**分别执行，test info macro RMSE 逐位一致：

| 组 | 原仓库 | release 包内 |
|---|---|---|
| `target_only` | 0.237267 | 0.237267 |
| `source_finetune` | 0.241459 | 0.241459 |
| `source_mmd_finetune` | 0.239710 | 0.239710 |

日志含 `已加载 source encoder (8 missing / 0 unexpected)` ×2，证明源域预训练权重
真的被加载而非静默跳过。冒烟耗时约 4 分钟（CPU）。

> 这些是 1 seed × 1 epoch 的冒烟值，**不得进入正式报告**。正式数字见
> `release/LEO_Bearings_Docker_Handoff/FINAL_RESULTS.md`。

---

## 6. 交付项覆盖（用户 §5 要求逐条）

| 要求 | 落地 | 验证方式 |
|---|---|---|
| 数据说明 | `data/MANIFEST_DOCKER_HANDOFF.md`（release 修订版） | 路径经 `--fast` 实跑验证 |
| 仿真数据 | `data/sim/wheel_basilisk_b18/final/wheel_all.h5` | `verify_release` 第 4 项哈希 |
| PyTorch 训练 | `src/models/` + `src/transfer/` | `--fast` 实跑 |
| 源域预训练/加载 | `checkpoints/source_tcn_pretrain.pt` + `source_features.h5` | `--fast` 日志 `已加载 source encoder` |
| 目标域微调 | `scripts/basilisk_b5/run_formal_transfer.py` | `--fast` 三组训练日志 |
| 指标 | `checkpoints/basilisk_b5/summary.json`、`basilisk_b6/paired_statistics.json` | `--verify` 第 3 项 |
| 图表 | `docs/figures/basilisk_final/` (5 PNG + 5 SVG) | `--verify` 第 5 项 |
| 最终结果 | `FINAL_RESULTS.md` | 与 `final_verdict.json` 一致 |
| 复现说明 | `README_DOCKER_HANDOFF.md` + `DOCKER_HANDOFF_CHECKLIST.md` | 30+ 项签收清单 |

---

## 7. 全量测试

`python -m pytest tests/ -q` → **2 failed, 1092 passed, 29 skipped, 1 xfailed**。

两个失败均为 `docs/basilisk_b6/stale_guard_retirement.md` 已定性归档的
**已知陈旧生命周期守卫**（`allowed_failure_kind = known_stale_lifecycle_guard`）：

| 测试 | 断言 | 失败原因 |
|---|---|---|
| `test_b6_old_artifacts_unchanged` | `STATUS_BASILISK_B7.md` 本应缺席 | B7 已获人工授权运行 |
| `test_b6_b7_b8_artifacts_must_stay_absent` | 同上 | 同上 |

两者是**同一原因**：B6 阶段的契约把 B7 产物钉为 `MISSING`，而 B7 现已合法完成。

**取证**（三条 §22 必核事项，本阶段实测）：

```
契约冻结文件已消失 (=删错东西):   []   ← 零
算法产物哈希不一致:               []   ← 零 (111 项全部一致)
本应缺席却存在:                   ['STATUS_BASILISK_B7.md']   ← 仅此一项
```

`STATUS_BASILISK_B7.md` 的时间戳为 19:13:51，早于 B8 工作起点（19:40+）——
与本阶段无关。按纪律**未删除、未修改**这两个测试；它们在
`B7_S8_CODE_FREEZE_GOVERNANCE` 阶段统一退役。

本阶段**未引入任何新的非该类别失败**。

---

## 8. 遗留事项（交给 Docker 负责人）

1. 大数据文件约 1.65 GB（`basilisk_b19` 特征 837 MB + `b18` 仿真 807 MB），
   建议 volume 挂载而非烤进镜像层，且 `data/` 挂载为只读 `:ro`。
2. `--full` **不要**纳入封装验收流程——它会覆写冻结证据，且需数小时。
   验收用 `--verify` + `--fast`。
3. `DATA_CLEANUP_CANDIDATES` 未生成删除动作：本阶段发现的旧 analytic 产物
   （`data/simulated/wheel/sim_v1/**`、`data/features/wheel/schema_v1/target_features.h5`）
   按约束**仅记录不删除**，见本文件 §4。是否清理由人类决定。

---

`B8_HANDOFF_READY`
