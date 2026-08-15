#!/usr/bin/env bash
# =====================================================================
# 飞轮组件容器入口
#
#   子命令:
#     verify           完整性验证 (manifest + 接收检查 + 冻结证据 + 完整性 + pytest)
#     reproduce-smoke  真实 B5 冒烟 (源域加载 -> 三组训练 -> 指标, ~10 分钟)
#     reproduce-full   正式 B5+B6 协议复现 (小时级, 会覆写正式命名空间)
#
#   产物持久化: smoke/full 完成后, 指标/日志/环境指纹写入 /results/reproduced/wheel/
#   任何步骤失败 (非零退出码) 即整体失败, 不吞错误。
# =====================================================================
set -euo pipefail

# 确定性默认 (docker-compose.yml 已设; 此处为 docker run 独立可用)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONPATH="${PYTHONPATH:-/app/release}"

cd /app/release

step() { echo ""; echo "========================================== $1 =========================================="; }

PERSIST_DIR="/results/reproduced/wheel"

generate_env_fingerprint() {
    # 生成环境指纹 JSON 写入 $PERSIST_DIR/environment_fingerprint.json
    mkdir -p "$PERSIST_DIR"
    python -c "
import sys, json, torch, numpy, pandas, scipy, sklearn, yaml, h5py, matplotlib, pytest, tqdm
fp = {
    'python': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
    'torch': torch.__version__,
    'numpy': numpy.__version__,
    'pandas': pandas.__version__,
    'scipy': scipy.__version__,
    'scikit-learn': sklearn.__version__,
    'PyYAML': yaml.__version__,
    'h5py': h5py.__version__,
    'matplotlib': matplotlib.__version__,
    'pytest': pytest.__version__,
    'tqdm': tqdm.__version__,
    'torch_num_threads': torch.get_num_threads(),
    'cuda_available': torch.cuda.is_available(),
}
print(json.dumps(fp, indent=2))
" > "$PERSIST_DIR/environment_fingerprint.json"
    echo "环境指纹已写入 $PERSIST_DIR/environment_fingerprint.json"
}

CMD="${1:-verify}"
case "$CMD" in

  verify)
    step "1/5 verify_manifest.py - manifest 470/470"
    python scripts/handoff/verify_manifest.py

    step "2/5 verify_release.py - 接收检查 25/25"
    python scripts/handoff/verify_release.py

    step "3/5 audit_frozen_evidence.py - 冻结证据 19/19"
    python scripts/handoff/audit_frozen_evidence.py

    step "4/5 run_all.py --verify - 5 步完整性检查"
    python scripts/run_all.py --verify

    step "5/5 运行时 pytest - 472 passed / 186 skipped / 0 failed"
    # 阶段考古类由 RELEASE_TEST_SCOPE.txt 排除 (release 是依赖闭包子集,
    # 被审计的历史中间产物按 §11 刻意未放入; 原仓库内这些测试全 PASS)
    IGNORES=""
    while IFS= read -r line; do
      line="${line%%#*}"
      line="$(echo "$line" | tr -d '[:space:]')"
      [ -z "$line" ] && continue
      IGNORES="$IGNORES --ignore=$line"
    done < RELEASE_TEST_SCOPE.txt
    # Basilisk 可选证据生成依赖, 不装入运行时镜像:
    # test_p95_reference.py 经 calibrate_degradation.py L246/L290
    # -> from src.sim import basilisk_bridge 间接需要 Basilisk runtime
    IGNORES="$IGNORES --ignore=tests/basilisk_b11/test_p95_reference.py"
    echo "排除 $(echo "$IGNORES" | grep -o '\-\-ignore=' | wc -l) 个测试文件"
    python -m pytest tests/ -q --no-header -p no:cacheprovider $IGNORES

    step "✅ VERIFY PASSED"
    echo "manifest 470/470 | 接收检查 25/25 | 冻结证据 19/19 | 完整性 5/5 | 运行时 0 failed"
    ;;

  reproduce-smoke)
    mkdir -p "$PERSIST_DIR"
    step "run_all.py --fast - 真实 B5 冒烟 (1 seed × 1 epoch)"
    echo "流程: 源域 TCN 权重加载 -> target_only / source_finetune / source_mmd_finetune"
    echo "      -> 目标域微调 -> MMD 对齐 -> test 指标"
    echo "原交付基线 (torch 2.5.1): target_only=0.237267  source_finetune=0.241459  source_mmd_finetune=0.239710"
    echo ""
    python scripts/run_all.py --fast 2>&1 | tee "$PERSIST_DIR/smoke_run.log"

    # 持久化冒烟指标
    if [ -f checkpoints/_smoke_b5/formal_metrics.json ]; then
        cp checkpoints/_smoke_b5/formal_metrics.json "$PERSIST_DIR/smoke_formal_metrics.json"
        echo "冒烟指标已持久化: $PERSIST_DIR/smoke_formal_metrics.json"
    fi

    # 环境指纹
    generate_env_fingerprint

    step "✅ reproduce-smoke 完成 - 冒烟数值不入正式报告"
    echo "产物目录: $PERSIST_DIR"
    ;;

  reproduce-full)
    mkdir -p "$PERSIST_DIR"
    step "run_all.py --full - 正式 B5+B6 协议复现"
    echo "⚠️  耗时数小时, 会覆写 checkpoints/basilisk_b5 | basilisk_b6 正式命名空间"
    echo ""
    python scripts/run_all.py --full --i-understand-this-overwrites 2>&1 | tee "$PERSIST_DIR/full_run.log"

    # 持久化正式结果 (JSON 指标文件)
    for ns in basilisk_b5 basilisk_b6; do
        if [ -d "checkpoints/$ns" ]; then
            mkdir -p "$PERSIST_DIR/$ns"
            cp -f checkpoints/$ns/*.json "$PERSIST_DIR/$ns/" 2>/dev/null || true
        fi
    done
    echo "正式结果已持久化: $PERSIST_DIR/"

    # 环境指纹
    generate_env_fingerprint

    step "✅ reproduce-full 完成 - 正式结果已重跑"
    ;;

  *)
    echo "未知命令: $CMD"
    echo "用法: docker compose run --rm wheel [verify | reproduce-smoke | reproduce-full]"
    exit 1
    ;;

esac
