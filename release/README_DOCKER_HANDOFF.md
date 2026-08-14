# README — Docker Handoff Package

**项目**：2026 挑战杯 — LEO 通信卫星组件寿命预测与在轨健康管理（反作用飞轮组件）
**交付状态**：`B8_HANDOFF_READY`
**收包时的正式结论**：`FINAL_TRANSFER_CONCLUSION = NO_POSITIVE_TRANSFER_SUPPORTED`
**工程推荐**：`ENGINEERING_RECOMMENDATION = damage_extrapolation`

---

## 0. 三十秒上手

```bash
cd LEO_Bearings_Docker_Handoff
pip install -r requirements.txt

python scripts/run_all.py --verify   # < 1 min，完整性 + 哈希 + 冻结指标
python scripts/run_all.py --fast     # ~5 min，真实训练链路冒烟（CPU 即可）
```

两条都通过，就说明这个包在你的环境里是自洽可跑的。

---

## 1. 三种运行模式

| 模式 | 耗时 | 干什么 | 会不会写正式结果 |
|---|---|---|---|
| `--verify` | < 1 min | 校验文件存在性、SHA256、导入、冻结指标可解析、最终判定一致、图表齐全 | 不写任何文件 |
| `--fast` | ~5 min (CPU) | **跑正式 B5 脚本本身**：源域 TCN 权重加载 → 三组 PyTorch 训练 → 目标域微调 → MMD 对齐 → test 指标 + 配对增益 | 只写 `checkpoints/_smoke_b5/`，**不碰**冻结证据 |
| `--full` | 数小时 | 重跑正式 B5 + B6 全协议矩阵 | **会覆写** `checkpoints/basilisk_b5|b6`，需显式加 `--i-understand-this-overwrites` |

`--fast` 与正式 B5 的差异只有三处（见 `configs/wheel_basilisk_b5_smoke.yaml`）：
seeds 收窄为 `[112]`、`max_epochs=1`、输出重定向。**其余逐字相同**——它跑的是同一份
`scripts/basilisk_b5/run_formal_transfer.py`，不是替身桩。

> 冒烟数值（约 `target_only 0.2373 / source_finetune 0.2415 / source_mmd 0.2397`）
> 只用于证明链路通，**不得进入正式报告**。正式数字见 `FINAL_RESULTS.md`。

---

## 2. 交付项对照（赛题要求逐条落地）

| 要求 | 位置 | 怎么验 |
|---|---|---|
| 数据说明 | `data/MANIFEST_DOCKER_HANDOFF.md` | 直接读；路径均经 `--fast` 实跑验证 |
| 仿真数据 | `data/sim/wheel_basilisk_b18/final/wheel_all.h5` | `--verify` 第 4 项哈希校验 |
| 源域预训练 / 加载 | `checkpoints/source_tcn_pretrain.pt` + `data/features/wheel/schema_v1/source_features.h5` | `--fast` 日志出现 `已加载 source encoder` |
| 目标域微调 | `scripts/basilisk_b5/run_formal_transfer.py` | `--fast` 三组训练日志 |
| PyTorch 训练 | `src/models/` + `src/transfer/` | `--fast` 实跑 |
| 指标 | `checkpoints/basilisk_b5/summary.json`、`checkpoints/basilisk_b6/paired_statistics.json` | `--verify` 第 3 项 |
| 图表 | `docs/figures/basilisk_final/` (5 PNG + 5 SVG) | `--verify` 第 5 项 |
| 最终结果 | `FINAL_RESULTS.md` | 与 `checkpoints/basilisk_b6/final_verdict.json` 一致 |
| 复现说明 | 本文件 + `docs/basilisk_b5/REPRODUCE.md`、`docs/basilisk_b6/REPRODUCE.md` | — |

---

## 3. 运行链路

```
data/features/wheel/schema_v1/source_features.h5   (源域 XJTU-SY 派生特征)
        │  src/train/pretrain.py（已完成，权重随包分发）
        ▼
checkpoints/source_tcn_pretrain.pt                 (源域 TCN 编码器)
        │
        │        data/features/wheel/basilisk_b19/target_features.h5  (目标域 B1.9 特征)
        │                    │  ← 上游 data/sim/wheel_basilisk_b18/final/wheel_all.h5
        ▼                    ▼
   scripts/basilisk_b5/run_formal_transfer.py
   三组严格配对: target_only / source_finetune / source_mmd_finetune
   + eval-only: const_mean_info / damage_extrapolation
        ▼
checkpoints/basilisk_b5/  →  scripts/basilisk_b6/run_formal_matrix.py (低标签矩阵)
        ▼
checkpoints/basilisk_b6/final_verdict.json  →  docs/figures/basilisk_final/
```

**迁移层原则**：迁移发生在健康指标 / 退化动力学层，不在原始波形层。
源域随包分发的是派生特征而非振动波形，正是这条原则的体现。

---

## 4. Basilisk 不是运行时依赖

Basilisk 只用来**生成任务工况剖面**，产物已冻结为 `data/mission_profile/basilisk_v1/profiles.h5`。

- `src/` 内**没有任何** Basilisk import——由 `verify_release.py` 第 8 项自动校验
- Docker 镜像**不需要**安装 Basilisk
- 生成剖面的代码**不在**本包内（不需要）

---

## 5. 环境

见 `requirements.txt`。关键点：

- Python 3.11（开发环境为 conda `DP_env`，Python 3.11）
- 核心预测器为 PyTorch，CPU 可跑完 `--verify` 与 `--fast`
- `--full` 建议 GPU；CUDA 非确定性下允许 ±5% 容差，CPU 侧仿真跨 OS 完全一致

---

## 6. 请不要做这些

| 禁止 | 为什么 |
|---|---|
| 重新生成 / 覆写 `data/**` | 数据为 `READ_ONLY_CRITICAL`，且划分哈希被钉死，改动即 `B5_INVALID` |
| 修改 `configs/wheel_basilisk_b5.yaml` / `_b6.yaml` | 门槛与超参已冻结；改动等于事后调门槛 |
| 修改 `docs/basilisk_b*/protocol.md` | 启动时校验 sha256，不一致直接中止 |
| 重新划分 train/val/test | 按轨迹个体划分，重划会引入泄漏 |
| 跑 `--full` 来"验收" | 会覆写冻结证据。验收用 `--verify` + `--fast` |
| 拿 `--fast` 的数字写报告 | 那是 1 seed × 1 epoch 的冒烟值 |

---

## 7. 出问题时

| 症状 | 原因 | 处理 |
|---|---|---|
| `No module named 'src'` | 未在包根目录执行 | `cd` 到包根，或设 `PYTHONPATH=.` |
| `B5_INVALID` / 划分哈希不符 | `data/` 或 split 被改动 | 从原始交付包恢复该文件 |
| `缺 protocol_hash.json` | 协议文档缺失 | 确认 `docs/basilisk_b5/`、`docs/basilisk_b6/` 完整 |
| 依赖闭包报缺项 | 打包不完整 | 跑 `python scripts/handoff/verify_release.py` 看缺哪些 |

---

## 8. 完整性自检

```bash
python scripts/handoff/verify_release.py
```

期望输出 `RESULTS: 25/25 checks passed` 与 `✅ RELEASE_VERIFY_PASS`。
第 11 项是 B5/B6 依赖闭包检查（28 项），它存在的原因见 `DOCKER_HANDOFF_CHECKLIST.md` 附注。

逐项签收清单见 `DOCKER_HANDOFF_CHECKLIST.md`。
