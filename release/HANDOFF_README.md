# Reaction Wheel RUL Docker Handoff

## 1. 当前状态

**`B8_HANDOFF_READY`**

final scientific conclusion:

```
FINAL_TRANSFER_CONCLUSION = NO_POSITIVE_TRANSFER_SUPPORTED
```

engineering recommendation:

```
damage_extrapolation
```

说明：

> 这不是「迁移无效」的证明，而是**在冻结协议与预算下未得到稳定正迁移证据**。
> 两个冻结结论各自独立、不得互相覆盖：
> - `B5_NO_POSITIVE_TRANSFER`（整体正式评估，5 seed）
> - `B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER`（失效标签稀缺矩阵）
>
> 即使某个低标签档出现正增益，那也只构成 conditional low-label transfer
> benefit，**绝不是 overall positive transfer**。

## 2. 师兄只需要做什么

只做：

- `Dockerfile`
- `docker-compose.yml`（如需要）
- container entrypoint
- clean environment reproduction

**不要修改**：

- algorithm
- config affecting numbers
- checkpoint
- dataset
- split

> 本仓库的算法与实验线**已冻结**。任何改动都会打破 19 项逐字节哈希核验。

## 3. 数据

data 大约：**~1.65 GB**（release 内 `data/` 8 个文件，均为 B5/B6 依赖闭包）

推荐：**read-only volume**

示例：

```yaml
volumes:
  - ./data:/workspace/data:ro
```

Basilisk：

> **不进入 Docker dependency。**
> Basilisk 仅作为数据生成器在 B1.x 阶段离线使用，运行/复现直接读取已生成的
> mission profile（`data/mission_profile/basilisk_v1/`）。容器内无需安装。

## 4. 快速验收

跑这条命令：

```bash
python scripts/run_all.py --fast
```

期望（≈4 分钟，CPU 可跑）：

```
target_only       0.237267
source_finetune   0.241459
source_mmd_finetune   0.239710
```

tolerance：

```
absolute <= 1e-6
```

> **禁止通过修改 expected value 解决数值不一致。** 若超出容差 →
> `B8_HANDOFF_INVALID`，请回报而不要调参。

同时必须在日志里看到（出现 2 次，finetune 与 mmd 各一次）：

```
已加载 source encoder (8 missing / 0 unexpected —— heads/adapter 按设计不迁移)
```

`8 missing / 0 unexpected` 是**源域权重真实加载**的证据：encoder 全部命中，
只有 heads/adapter 按设计不迁移。若变成 `0 missing` 或出现 `unexpected`，
说明加载路径不对。

## 5. Tests

跑：

```bash
python scripts/handoff/run_release_tests.py
```

期望：**0 failed**（488 passed / 185 skipped）

### full_test_suite_present = true

`tests/` 122 个测试文件与原仓库**逐字节一致**，一个都没删、没加 skip、
没加 xfail。

### full_artifact_history_present = false ← 这是为什么要用上面那条命令

release 是**依赖闭包子集**：只带跑通 B5/B6 复现链所必需的产物，按 §11
刻意不带 B1.x 各阶段的历史中间产物（候选 HDF5 / 早期 analytic checkpoints /
历史脚本，合计数 GB）。

因此若直接跑 `python -m pytest tests/ -q`，会看到 **114 failed / 45 个文件**。
这些失败**全部**是同一类：「被审计的历史文件不在 release 里」
（`已消失` / `缺失` / `FileNotFoundError` / `ImportError: 缺 ...`），
不是代码缺陷 —— 同样这批测试在原仓库内全部 PASS（**1098 passed / 0 failed**）。

被排除的 45 个文件逐条列在 [`RELEASE_TEST_SCOPE.txt`](RELEASE_TEST_SCOPE.txt)。

## 6. 禁止运行

不要在普通验收跑：

```bash
python scripts/run_all.py --full
```

因为**会重训并可能覆盖冻结实验产物**。

脚本已加人工确认闸门，只有显式加上才允许：

```bash
--i-understand-this-overwrites
```

## 7. Docker 验收目标

师兄完成 Docker 后必须证明：

| | 项 |
|---|---|
| A | clean image builds |
| B | data read-only mount works |
| C | `--fast` passes |
| D | 三个 smoke 数字一致（1e-6） |
| E | source encoder loaded（`8 missing / 0 unexpected`） |
| F | pytest / release verification pass |
| G | 容器内不需要 Basilisk |

逐项勾选清单：[`docs/handoff/DOCKER_CHECKLIST.md`](docs/handoff/DOCKER_CHECKLIST.md)

## 8. 关键路径

| 用途 | 路径 |
|------|------|
| target feature（B1.9 正式输入，12 维 CORE_ONLY） | `data/features/wheel/basilisk_b19/target_features.h5` |
| source checkpoint（迁移起点） | `checkpoints/source_tcn_pretrain.pt` |
| B5 results（正式迁移评估，5 seed） | `checkpoints/basilisk_b5/formal_metrics.json`<br>`checkpoints/basilisk_b5/summary.json` |
| B6 results（失效标签稀缺矩阵） | `checkpoints/basilisk_b6/all_metrics.json`<br>`checkpoints/basilisk_b6/final_verdict.json` |
| docs/results.md（最终结果表达面） | `docs/results.md` |
| technical report | `docs/技术方案报告/` |
| figures | `docs/figures/basilisk_final/`（fig1–fig5，png+svg） |
| REPRODUCE | `docs/REPRODUCE.md` |
| MANIFEST | `RELEASE_MANIFEST.sha256`（+ `RELEASE_MANIFEST.json`） |

### 附：验证与审计入口

```bash
python scripts/handoff/verify_release.py            # release 完整性 25/25
python scripts/handoff/verify_manifest.py           # manifest ↔ 实际文件
python scripts/handoff/audit_frozen_evidence.py     # 19 项冻结证据逐字节
python scripts/basilisk_b7/audit_report_numbers.py  # 报告数字语义审计
```

### 数据纪律

`data/` 为 **READ ONLY**。不得删除原始公开数据 / processed source data /
Basilisk profile / target dataset / target feature / manifest-provenance。
可清理项只登记在 `docs/handoff/DATA_CLEANUP_CANDIDATES.md`，不执行删除。
