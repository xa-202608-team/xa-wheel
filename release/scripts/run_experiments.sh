#!/usr/bin/env bash
# =====================================================================
# 5 组 × n 种子 对比实验 (plan §四 Phase 6)
#   target_only / source_frozen / source_finetune / source_mmd_finetune + 物理基线
# 产出 docs/results.md (mean±std)
#
# 用法: bash scripts/run_experiments.sh --seeds 5
# =====================================================================
set -euo pipefail
SEEDS=5
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --smoke)  SMOKE=1; shift ;;
    *) shift ;;
  esac
done
cd "$(dirname "$0")/.."
if [[ -n "${SMOKE:-}" ]]; then
  python -m src.experiments.run_groups --config configs/wheel.yaml --smoke
else
  python -m src.experiments.run_groups --config configs/wheel.yaml --seeds "$SEEDS"
fi
