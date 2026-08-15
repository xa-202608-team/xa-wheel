# 复现说明 (REPRODUCE)

> 对应 `docs/flywheel_dev_plan.md` Phase 7。赛题工程实现与可复现性 50 分。

## 0. 赛题要求对照 (第六节·评选标准)

工程实现与可复现性 50 分 = 15(代码+数据可跑通复现) + 10(PyTorch 流程可复跑) + 9(代码结构) + **8(Docker 封装)** + 8(文档)。
- **一票否决前置**: "代码是否可完整跑通并复现主要实验结果"。本仓库 `bash scripts/run_all.sh --fast` 已端到端跑通。
- **Docker 是 8 分建议项** (赛题原文:"建议通过Docker容器方式进行整体封装"), 非强制; 但能显著提升复现效率。

## 1. 环境

- **Python 3.12**, PyTorch 2.x (CUDA 可选, 有 GPU 加速)
- 依赖: `pip install -r requirements.txt`
- (可选, 推荐) **Docker**: `docker compose build`

```bash
# 本地 conda (开发/训练, Win10 或 WSL2 均可)
pip install -r requirements.txt
# 或 Docker (复现交付)
docker compose build
docker compose run --rm pipeline bash scripts/run_all.sh --fast
```

## 2. 数据准备

### 源域 — XJTU-SY 轴承退化 (主)
```bash
bash scripts/data/download_source.sh               # github mirror (需 github 可达)
SOURCE=local bash scripts/data/download_source.sh  # 指向已手动放置的数据
bash scripts/data/download_source.sh --check       # 校验 3 工况 × 5 轴承 = 15
```
> github 不可达时, 用 `--synthetic` 生成合成源域验证管线:
> `python -m src.data.preprocess.wheel_features --synthetic --report`

### 目标域 — 飞轮仿真 (自生成, 无外部依赖)
```bash
python -m src.sim.wheel_sim --config configs/wheel.yaml --n_traj 100   # data/simulated/wheel/sim_v1/seed_42/
python -m src.sim.build_hi    --config configs/wheel.yaml --report      # HI + x_T → data/features/wheel/schema_v1/
```

## 3. 全流程

```bash
bash scripts/run_all.sh           # 正式 (需真实 XJTU-SY)
bash scripts/run_all.sh --fast    # smoke (合成数据, 验证整条管线)
bash scripts/run_experiments.sh --seeds 5   # 仅对比实验
```

## 4. 关键参数 (唯一差异点)

全部超参 / 仿真参数 / 种子集中在 **`configs/wheel.yaml`**:
- `seed: 42` (全局种子)
- 飞轮物理区间 (`sim.physics`): J / Kt / Tc / b0 / Δ / τ
- 模型 (`model`): TCN 通道/核/块数, latent_dim
- 损失权重 (`loss`): λ_hi / λ_mono / λ_smooth
- 迁移 (`transfer`): HI 分箱 / MMD λ / 轨迹划分比

## 5. 各阶段命令与产出

| 阶段 | 命令 | 产出 |
|------|------|------|
| P1 特征工程 | `python -m src.data.preprocess.wheel_features --report` | `data/features/wheel/schema_v1/source_features.h5` |
| P2 预训练 | `python -m src.train.pretrain` | `checkpoints/source_tcn_pretrain.pt` |
| P3 仿真 | `python -m src.sim.wheel_sim --n_traj 100` | `data/simulated/wheel/sim_v1/seed_42/wheel_traj_*.csv` + `wheel_all.h5` |
| P4 HI | `python -m src.sim.build_hi --report` | `data/features/wheel/schema_v1/target_features.h5` |
| P5 迁移 | `python -m src.transfer.train_transfer` | `checkpoints/transfer_tcn_transfer.pt` |
| P6 实验 | `python -m src.experiments.run_groups --seeds 5` | `docs/results.md` |

## 6. 可复现性验证

- **固定种子**: `utils/seed.py` 固定 python/numpy/torch/cuda; `tests/test_repro.py` 断言同种子两次运行哈希一致。
- **仿真可复现**: `python -m src.sim.wheel_sim --n_traj 2 --seed 7 --hash` 两次运行哈希一致 (`838288e3240f2c9c`)。
- **测试**: `python -m pytest tests/ -q` (28 项, 覆盖 P0-P6 不变量)。

## 7. 预期结果

- **迁移增益**: Source+MMD+Finetune 的 RMSE 显著低于 Target-only (核心数字, 见 `docs/results.md`)。
- 主模型在 RMSE / PHM Score 上优于物理外推基线 (真实源域预训练后)。

## 8. 目录结构

见 `docs/flywheel_dev_plan.md` §2.1。大数据 (`data/raw`, `data/extracted`, `data/simulated`, `data/features`) 与
训练产物 (`checkpoints/`) 不入库 (见 `.gitignore`), 由流程重新生成。

## 9. 跨平台复现容差说明 (Win10 训练 → Docker 交付)

开发可在 Win10 (conda) 或 WSL2/Linux 进行, 最终交付 Docker(Linux) 镜像。赛题只考察**最终交付物能否复现**, 不考察开发 OS。

- **CPU 仿真** (`wheel_sim.py`, 纯 numpy 固定种子): 跨 OS **完全一致** (已验证 hash `838288e3240f2c9c`)。
- **GPU 训练** (PyTorch+CUDA): 不同 OS/驱动/cudnn 下, 固定种子结果可能有**小数点后几位的微小差异**
  (CUDA 非确定性算子所致), 属正常现象, 不影响"主要实验结果复现" (趋势/数量级一致)。
- **复现判据**: 迁移增益方向一致 (Source+MMD+Finetune RMSE < Target-only), 数值在 ±5% 容差内即视为复现。
- **行尾符**: `.gitattributes` 强制 `.sh`/`.py` 用 LF, 避免 Windows CRLF 进 Linux 容器报错。
- **工作流**: Win10 训练产出参考 `results.md` + checkpoint → 最后 WSL2 内 Docker 封装并复跑验证
  (Docker 内复跑的数字才是交付证据, 非开发机数字)。
