#!/usr/bin/env bash
# =====================================================================
# 源域轴承退化数据下载与完整性校验
#   XJTU-SY (主) + IMS / FEMTO (辅, 跨数据集验证)
#
# 用法:
#   bash data/download_source.sh            # 下载 + 校验
#   bash data/download_source.sh --check    # 仅校验已下载数据
#
# 数据源选择 (环境变量 SOURCE):
#   SOURCE=github    从 GitHub CSV mirror clone (需 github 可达)
#   SOURCE=official  提示官方注册下载 (biaowang.tech)
#   SOURCE=local     假定数据已手动放置到 data/source/XJTU-SY (默认)
#
# XJTU-SY 目录结构 (15 轴承 = 3 工况 × 5):
#   XJTU-SY/
#     35Hz12kN/   Bearing1_1 ... Bearing1_5
#     37.5Hz11kN/ Bearing2_1 ... Bearing2_5
#     40Hz10kN/   Bearing3_1 ... Bearing3_5
#   每个 csv = 1.28s @ 25.6 kHz = 32768 点, 列 [Horizontal, Vertical]
# =====================================================================
set -euo pipefail

SOURCE="${SOURCE:-local}"
DATA_DIR="${DATA_DIR:-data/source}"
XJTU_DIR="${DATA_DIR}/XJTU-SY"
IMS_DIR="${DATA_DIR}/IMS"
FEMTO_DIR="${DATA_DIR}/FEMTO"

# 工况 -> (目录名, 轴承前缀)
CONDITIONS=("35Hz12kN:1" "37.5Hz11kN:2" "40Hz10kN:3")

download_xjtu() {
  echo ">> [XJTU-SY] SOURCE=$SOURCE"
  case "$SOURCE" in
    github)
      # GitHub CSV mirror (数据量大, 可能需 Git LFS; github 当前不可达时会失败)
      if [[ -z "${XJTU_GH_REPO:-}" ]]; then
        echo "!! 未设置 XJTU_GH_REPO；官方 GitHub 不含数据文件。"
        return 1
      fi
      local repo="$XJTU_GH_REPO"
      echo ">> git clone --depth 1 $repo -> $XJTU_DIR"
      git clone --depth 1 "$repo" "$XJTU_DIR" || {
        echo "!! GitHub 下载失败 (github 不可达或仓库不存在)"
        echo "!! 备选: SOURCE=official 手动注册下载, 或 SOURCE=local 指向已有数据"
        return 1
      }
      ;;
    official)
      echo ">> 官方数据需注册下载:"
      echo "     https://biaowang.tech/xjtu-sy-bearing-datasets/"
      echo ">> 手动下载后解压到 $XJTU_DIR, 再运行: bash data/download_source.sh --check"
      return 1
      ;;
    local)
      echo ">> local 模式: 假定数据已在 $XJTU_DIR"
      ;;
    *)
      echo "!! 未知 SOURCE=$SOURCE (可选: github|official|local)"; exit 1
      ;;
  esac
}

check_xjtu() {
  echo ">> [XJTU-SY] 完整性校验 (期望 3 工况 × 5 轴承 = 15)"
  local count=0 total_files=0
  for entry in "${CONDITIONS[@]}"; do
    cond="${entry%%:*}"; idx="${entry##*:}"
    for n in 1 2 3 4 5; do
      bearing="Bearing${idx}_${n}"
      dir="$XJTU_DIR/$cond/$bearing"
      if [[ -d "$dir" ]]; then
        files=$(find "$dir" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l)
        printf "   OK   %-10s/%-12s (%s csv)\n" "$cond" "$bearing" "$files"
        [[ "$files" -gt 0 ]] && { count=$((count+1)); total_files=$((total_files+files)); }
      else
        printf "   MISS %-10s/%-12s\n" "$cond" "$bearing"
      fi
    done
  done
  echo ">> 轴承 $count / 15  |  csv 文件总数 $total_files"
  if [[ "$count" -ne 15 ]]; then
    echo "!! XJTU-SY 不完整 ($count/15)"; exit 1
  fi
  echo ">> [XJTU-SY] 完整性 OK"
}

download_ims() {
  echo ">> [IMS] NASA 辅源 (可选, 跨数据集验证)"
  if [[ -d "$IMS_DIR" && -n "$(ls -A "$IMS_DIR" 2>/dev/null)" ]]; then
    echo "   IMS 已存在"; return 0
  fi
  echo "   IMS 原始: https://www.nasa.gov/intelligent-systems-division/  (需 NASA data repository)"
  echo "   跳过自动下载 (辅源, 不影响主流程)"
}

download_femto() {
  echo ">> [FEMTO] IEEE PHM 2012 辅源 (可选)"
  if [[ -d "$FEMTO_DIR" && -n "$(ls -A "$FEMTO_DIR" 2>/dev/null)" ]]; then
    echo "   FEMTO 已存在"; return 0
  fi
  echo "   FEMTO 原始: https://www.femto-st.fr/  (IEEE PHM 2012 challenge)"
  echo "   跳过自动下载 (辅源, 不影响主流程)"
}

main() {
  case "${1:-}" in
    --check)
      check_xjtu
      ;;
    "")
      download_xjtu
      check_xjtu
      download_ims
      download_femto
      ;;
    *)
      echo "usage: bash data/download_source.sh [--check]"
      echo "env:   SOURCE=github|official|local  DATA_DIR=data/source"
      exit 1
      ;;
  esac
}

main "$@"
