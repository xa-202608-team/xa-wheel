#!/usr/bin/env bash
# =====================================================================
# 反作用飞轮全流程一键复现 (plan §四 Phase 7)
#   download -> features -> pretrain -> sim -> HI -> transfer -> experiments
#
# 用法:
#   bash scripts/run_all.sh           # 正式全流程 (需真实 XJTU-SY)
#   bash scripts/run_all.sh --fast    # smoke (合成数据, 验证管线)
# =====================================================================
set -euo pipefail
FAST=0
[[ "${1:-}" == "--fast" ]] && FAST=1
cd "$(dirname "$0")/.."

SM=""
[[ $FAST -eq 1 ]] && SM="--smoke"
step() { echo ""; echo "========== $1 =========="; }

# 1. 源域数据获取与特征工程
step "P1 源域特征工程"
if [[ $FAST -eq 1 ]]; then
  python -m src.data.preprocess.wheel_features --config configs/wheel.yaml --synthetic --report
else
  bash scripts/data/download_source.sh
  python -m src.data.preprocess.wheel_features --config configs/wheel.yaml --report
fi

# 2. 源域预训练
step "P2 源域预训练 (TCN 双头)"
python -m src.train.pretrain --config configs/wheel.yaml $SM

# 3. 飞轮退化仿真
step "P3 飞轮仿真"
N_TRAJ=100
[[ $FAST -eq 1 ]] && N_TRAJ=30
python -m src.sim.wheel_sim --config configs/wheel.yaml --n_traj "$N_TRAJ"

# 4. 飞轮 HI 构造
step "P4 HI 构造 + 目标域特征"
python -m src.sim.build_hi --config configs/wheel.yaml --report

# 5. 三阶段迁移训练
step "P5 三阶段迁移训练"
python -m src.transfer.train_transfer --config configs/wheel.yaml $SM

# 6. 对比实验
step "P6 基线与对比实验"
python -m src.experiments.run_groups --config configs/wheel.yaml $SM

step "完成"
echo ">> 全流程结束。结果见 docs/results.md, 复现说明见 docs/REPRODUCE.md"
