# DOCKER_CHECKLIST.md —— 师兄 Docker 封装验收清单

**阶段**: `B8-HANDOFF-FINAL` §13
**前置状态**: `B8_HANDOFF_READY`
**冻结结论**: `FINAL_TRANSFER_CONCLUSION = NO_POSITIVE_TRANSFER_SUPPORTED`
**工程建议**: `damage_extrapolation`

> 只做 Docker / 环境 / 提交外裹层。**不得修改影响实验数字的代码。**

---

## 清单

```
[ ] Python/PyTorch environment builds
[ ] CUDA optional path works or CPU fallback documented
[ ] Basilisk NOT installed
[ ] data mounted read-only
[ ] source checkpoint found
[ ] target feature found
[ ] --fast executes real B5 code
[ ] target smoke RMSE matches
[ ] FT smoke RMSE matches
[ ] MMD smoke RMSE matches
[ ] source encoder load message present
[ ] verify_release passes
[ ] pytest has 0 failed
[ ] docs/results numeric audit passes
[ ] no frozen checkpoint modified
[ ] final image/release size recorded
```

---

## 逐项验证方法

### `[ ] Python/PyTorch environment builds`
`requirements.txt` 在 release 根目录。参考环境：Python 3.12 + PyTorch（CPU 即可）。

### `[ ] CUDA optional path works or CPU fallback documented`
`--fast` 全流程 CPU 可跑（实测约 4 分钟）。CPU 仿真跨 OS 完全一致；
GPU 训练允许 ±5% 容差（CUDA 非确定性）。**验收请用 CPU** —— 1e-6 容差
只在 CPU 下成立。

### `[ ] Basilisk NOT installed`
容器内**不要**装 Basilisk。它只在 B1.x 阶段离线生成 mission profile；
运行时直接读 `data/mission_profile/basilisk_v1/`。
自查：`pip list | grep -i basilisk` 应为空，且 `--fast` 仍然通过。

### `[ ] data mounted read-only`
```yaml
volumes:
  - ./data:/workspace/data:ro
```
**不建议** `COPY` 整个 data 进 image（~1.65 GB），除非师兄明确决定这么做。
自查：容器内 `touch /workspace/data/x` 应失败。

### `[ ] source checkpoint found`
```
checkpoints/source_tcn_pretrain.pt
sha256 = d528c2d1b12b83e305bdc0809293e9b7ca8e84e07952dfb2c96bc46a5a8f7b66
```

### `[ ] target feature found`
```
data/features/wheel/basilisk_b19/target_features.h5
sha256 = 575d708ae9330dfa37d407cf639ebbbd098c473ce1fc137ff1eef8feaab9c293
```

### `[ ] --fast executes real B5 code`
```bash
python scripts/run_all.py --fast
```
它**真的**调用官方 `scripts/basilisk_b5/run_formal_transfer.py`
（不是 stub）。日志应出现 `[B5 正式] seed 112` 三组训练段落，
并写出 `checkpoints/_smoke_b5/formal_metrics.json`。
> 该冒烟输出走独立命名空间 `_smoke_b5`，**不污染**
> `checkpoints/basilisk_b5/`（脚本内置断言：冻结证据须保持 5 seed）。

### `[ ] target smoke RMSE matches`
```
target_only            0.237267    (|Δ| <= 1e-6)
```

### `[ ] FT smoke RMSE matches`
```
source_finetune        0.241459    (|Δ| <= 1e-6)
```

### `[ ] MMD smoke RMSE matches`
```
source_mmd_finetune    0.239710    (|Δ| <= 1e-6)
```
> 三个数字任一超出容差 → `B8_HANDOFF_INVALID`。
> **禁止通过修改 expected value 解决。**

### `[ ] source encoder load message present`
日志须出现 **2 次**（finetune + mmd 各一次）：
```
已加载 source encoder (8 missing / 0 unexpected —— heads/adapter 按设计不迁移)
```
`8 missing / 0 unexpected` = encoder 权重全部命中、heads/adapter 按设计不迁移。
若变成 `0 missing` 或出现 `unexpected`，说明加载路径错了。

### `[ ] verify_release passes`
```bash
python scripts/handoff/verify_release.py
```
期望 **25/25 PASS**（含 11 项检查，其中第 11 项钉死 28 条依赖闭包）。

### `[ ] pytest has 0 failed`
```bash
python scripts/handoff/run_release_tests.py
```
期望 **488 passed / 185 skipped / 0 failed**。

> ⚠️ 直接跑 `python -m pytest tests/ -q` 会看到 **114 failed** ——
> 这些**不是缺陷**。release 是依赖闭包子集
> （`full_test_suite_present = true` 但 `full_artifact_history_present = false`），
> 那批测试审计的是刻意未打包的 B1.x 历史中间产物。
> 详见 `RELEASE_TEST_SCOPE.txt` 与 `HANDOFF_README.md` §5。

### `[ ] docs/results numeric audit passes`
```bash
python scripts/basilisk_b7/audit_report_numbers.py
```
期望末行 `REPORT_NUMERIC_AUDIT_PASS`（92 审计项 / 86 数值项 / 0 未命中，
5 项语义冻结全 PASS）。

### `[ ] no frozen checkpoint modified`
```bash
python scripts/handoff/audit_frozen_evidence.py
```
期望末行 `FROZEN_EVIDENCE_AUDIT_PASS`（19 项逐字节 unchanged / 0 drift）。
覆盖：source checkpoint、B1.8 dataset、B1.9 feature、B2.1 split、
B5 protocol+results、B6 protocol+subset manifest+results、
docs/results.md 语义冻结。

### `[ ] final image/release size recorded`
封装完成后请记录并回报：

| 项 | 值 |
|----|-----|
| release 目录文件数 | 464（清理后） |
| release 目录大小 | ~1.9 GB |
| docker image 大小 | ____（待填） |
| image 内是否含 data | ____（建议：否，走 ro volume） |
| 构建耗时 | ____ |
| `--fast` 容器内耗时 | ____ |

---

## 完成后请回报

1. 上表 4 个待填项
2. A–G 七项验收（见 `HANDOFF_README.md` §7）是否全部通过
3. 任何一项数字不一致 —— **请直接回报，不要调参**
