# XA-202608 Wheel Component

本私有仓库只承载飞轮组件。当前契约版本：`component-contract-v1.0.0`。
正式组件版本只能由 `@xa-202608-team/integrators` 签发。

> 飞轮反作用轮 RUL 预测组件，容器化复现包。
> 算法与实验线冻结于 B8_HANDOFF_READY 交付，本仓库只保留 `release/` 代码基线与容器封装，不改任何算法/数据/checkpoint/指标。

## 环境基准

与相控阵组件**完全相同**（统一环境）：

| 包 | 版本 |
|----|------|
| Python | 3.12.x |
| torch | 2.11.0 (cu130 源) |
| numpy | 2.4.3 |
| pandas | 3.0.3 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.0 |
| PyYAML | 6.0.3 |
| h5py | 3.16.0 |
| matplotlib | 3.10.9 |
| pytest | 9.0.3 |
| tqdm | 4.67.3 |

## 仓库布局（代码 / 线下工件分离）

```
xa-wheel/
├── Dockerfile                 ← 基于 python:3.12-slim, 与相控阵同源依赖
├── requirements.unified.lock  ← 统一版本锁 (与相控阵逐项相等)
├── entrypoint.sh              ← verify / reproduce-smoke / reproduce-full 入口
├── .dockerignore
├── README.md                  ← 本文件
├── PROGRESS.md                ← 装配与验证进度日志
├── release/                   ← 飞轮原交付包（代码基线，层级保持原样）
│   ├── CODE_MANIFEST.sha256      ← Git 代码清单（只列受跟踪代码/配置/测试/文档）
│   ├── src/  scripts/  configs/  tests/  docs/
│   └── (data/ checkpoints/ results/ 不入库，线下装配)
├── handoff/
│   ├── artifact-map.yaml      ← 线下工件槽位映射（data/checkpoints/results → 交付 payload）
│   └── HANDOFF.md             ← 线下装配与两阶段校验手册
├── data/  checkpoints/  results/   ← 本地工件槽位（.gitignore 整目录忽略，不入库）
└── tests/test_repository_boundary.py  ← 层级与工件边界回归测试
```

数据在交付包 `04_数据/wheel/`（1.65 GB），容器内以 `/app/release/data:ro` 只读挂载；
槽位映射见 `handoff/artifact-map.yaml`。

## 用法

```bash
# 完整性验证 (< 5 分钟)
docker compose run --rm wheel verify

# 真实 B5 冒烟复现 (~10 分钟, CPU)
docker compose run --rm wheel reproduce-smoke

# 正式 B5+B6 全协议复现 (小时级, 会覆写正式命名空间, 可选)
docker compose run --rm wheel reproduce-full
```

### verify 做什么

依次执行 5 步，任一失败即整体失败：

| 步骤 | 脚本 | 期望 |
|------|------|------|
| 1 | `verify_manifest.py` | 代码基线 CODE_MANIFEST 校验通过（线下工件另见 `--artifact-root` 两阶段校验） |
| 2 | `verify_release.py` | 25/25 RELEASE_VERIFY_PASS |
| 3 | `audit_frozen_evidence.py` | 19 unchanged FROZEN_EVIDENCE_AUDIT_PASS |
| 4 | `run_all.py --verify` | 5 步完整性 VERIFY PASSED |
| 5 | 运行时 pytest | 472 passed / 186 skipped / 0 failed |

步骤 5 排除两类测试（不删、不加 skip/xfail，仅 `--ignore`）：
- **阶段考古类**：`RELEASE_TEST_SCOPE.txt` 列出的 ~44 文件（审计 B1.x 历史中间产物，release 是依赖闭包子集故对应产物按 §11 未放入，缺失属设计而非缺陷；原仓库内全 PASS）
- **test_p95_reference.py**：经 `calibrate_degradation.py` L246/L290 → `from src.sim import basilisk_bridge` 间接需要 Basilisk runtime；Basilisk 是可选证据生成依赖，不装入运行时镜像

### reproduce-smoke 期望指标

```
target_only         RMSE = 0.237267
source_finetune     RMSE = 0.241459
source_mmd_finetune RMSE = 0.239710
```

绝对误差不超过 1e-6（CPU、单线程、固定种子确定性保证）。

## 完整性校验（两阶段）

```bash
# 阶段一（无 --artifact-root）：只校验 Git 代码基线
python release/scripts/handoff/verify_manifest.py

# 阶段二（有 --artifact-root）：先校验代码基线，
#        再校验 $ARTIFACT_ROOT/HANDOFF_MANIFEST.json 及其登记文件哈希
python release/scripts/handoff/verify_manifest.py --artifact-root /path/to/offline/artifacts
```

原交付 `RELEASE_MANIFEST.sha256`（470 条，含 data/checkpoints 等线下工件）不进入 Git，
仅作线下装配 HANDOFF_MANIFEST.json 的参考输入；Git 内代码真值以 `release/CODE_MANIFEST.sha256` 为准。

## 科学结论（冻结，不可覆盖）

```
FINAL_TRANSFER_CONCLUSION = NO_POSITIVE_TRANSFER_SUPPORTED
ENGINEERING_RECOMMENDATION = damage_extrapolation
```

这是在冻结协议与预算下未得到稳定正迁移证据的独立结论，不代表"迁移无效的证明"。详见 `release/FINAL_RESULTS.md`。
