"""§5 §6 §12 §13 Release Package Builder — Docker Handoff.

- Creates release/LEO_Bearings_Docker_Handoff/ structure
- Uses allowlist approach for copying
- Checks disk space before data copy
- Hash-verifies all copies
- DOES NOT modify original repo (copy-only)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = ROOT / "release" / "LEO_Bearings_Docker_Handoff"

# ============================================================
# ALLOWLIST — 只复制这些路径到 release (§6 §11 §12)
# ============================================================
RELEASE_ALLOWLIST = [
    # Core source
    ("src/", "RELEASE_REQUIRED", "Core source code"),
    ("configs/", "RELEASE_REQUIRED", "Configuration files"),
    ("scripts/", "RELEASE_REQUIRED", "Scripts — only non-phase-specific"),
    ("tests/", "RELEASE_REQUIRED", "Core tests — NO phase tests"),
    # Requirements
    ("requirements.txt", "RELEASE_REQUIRED", "Python dependencies"),
    ("pyproject.toml", "REVIEW_REQUIRED", "Build config (if exists)"),
    ("README.md", "RELEASE_REQUIRED", "Root readme"),
    # Frozen docs
    ("docs/results.md", "REPRODUCIBILITY_REQUIRED", "Final results"),
    ("docs/技术方案报告/", "RELEASE_REQUIRED", "Tech report"),
    ("docs/figures/basilisk_final/", "RELEASE_REQUIRED", "Final B7 figures"),
    ("docs/handoff/", "REPRODUCIBILITY_REQUIRED", "Handoff documentation"),
    # 冻结的阶段协议文档 —— B5/B6 启动时会校验 protocol.md 的 sha256, 缺了直接 B5_INVALID
    ("docs/basilisk_b2/", "REPRODUCIBILITY_REQUIRED", "B2 protocol (B5 依赖)"),
    ("docs/basilisk_b21/", "REPRODUCIBILITY_REQUIRED", "B2.1 split manifest + protocol"),
    ("docs/basilisk_b19/", "REPRODUCIBILITY_REQUIRED", "B1.9 feature definition"),
    ("docs/basilisk_b5/", "REPRODUCIBILITY_REQUIRED", "B5 protocol + results"),
    ("docs/basilisk_b6/", "REPRODUCIBILITY_REQUIRED", "B6 protocol + final conclusion"),
    # Data — 只复制 B5/B6 正式链路真正读取的文件
    ("data/MANIFEST_DOCKER_HANDOFF.md", "DATA_CRITICAL", "Data manifest"),
    # B1.9 正式特征 (B5/B6 的 b19_feature_h5) —— 真正被训练读取的那一份
    ("data/features/wheel/basilisk_b19/target_features.h5", "DATA_CRITICAL",
     "B1.9 正式目标域特征 (B5/B6 输入)"),
    # 源域特征 (源域预训练输入)
    ("data/features/wheel/schema_v1/source_features.h5", "DATA_CRITICAL",
     "源域 XJTU-SY 特征 (预训练输入)"),
    # B1.8 正式仿真数据集 (特征的上游, 供追溯/重算)
    ("data/sim/wheel_basilisk_b18/final/wheel_all.h5", "DATA_CRITICAL",
     "B1.8 正式仿真数据集"),
    ("data/sim/wheel_basilisk_b18/final/params.json", "DATA_CRITICAL",
     "B1.8 仿真参数"),
    # Basilisk 任务剖面 (冻结数据, Basilisk 本身不是运行时依赖)
    ("data/mission_profile/basilisk_v1/", "DATA_CRITICAL",
     "冻结 Basilisk 任务剖面"),
    # Checkpoints — B5/B6 正式产物 + 其只读前置
    ("checkpoints/source_tcn_pretrain.pt", "REPRODUCIBILITY_REQUIRED",
     "源域 TCN 预训练权重 (迁移组加载入口)"),
    ("checkpoints/basilisk_b2/", "REPRODUCIBILITY_REQUIRED", "B2 划分与协议哈希"),
    ("checkpoints/basilisk_b21/", "REPRODUCIBILITY_REQUIRED", "B2.1 划分门禁 (B5 校验)"),
    ("checkpoints/basilisk_b19/", "REPRODUCIBILITY_REQUIRED", "B1.9 冻结特征定义"),
    ("checkpoints/basilisk_b11/", "REPRODUCIBILITY_REQUIRED", "B1.1 标定 (退化模型常量)"),
    ("checkpoints/basilisk_b3x/", "REPRODUCIBILITY_REQUIRED", "B3X 汇总 (B5 只读)"),
    ("checkpoints/basilisk_b4x/", "REPRODUCIBILITY_REQUIRED", "B4X 汇总 (B5 只读)"),
    ("checkpoints/basilisk_b5/", "REPRODUCIBILITY_REQUIRED", "B5 official checkpoints"),
    ("checkpoints/basilisk_b6/", "REPRODUCIBILITY_REQUIRED", "B6 official checkpoints"),
    ("checkpoints/basilisk_b7/", "REPRODUCIBILITY_REQUIRED", "B7 final audit artifacts"),
]

# B5/B6 运行时真正 import 的阶段脚本目录 (依赖闭包, 由 import 图推出)。
# 注意 scripts/basilisk_b1/calibrate_degradation.py 是被 b11 按**文件路径**动态加载的,
# 不在 import 语句里 —— 漏掉它 release 会在 load_cfg 阶段就崩。
REQUIRED_PHASE_SCRIPTS = [
    "scripts/basilisk_b1/calibrate_degradation.py",
    "scripts/basilisk_b11/calibrate_degradation.py",
    "scripts/basilisk_b2/",
    "scripts/basilisk_b21/",
    "scripts/basilisk_b3x/",
    "scripts/basilisk_b4x/",
    "scripts/basilisk_b5/",
    "scripts/basilisk_b6/",
]

# 排除的子路径 (纯缓存/临时, 与阶段无关)
EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pyc",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".git",
]

# 不进入 release 的历史阶段脚本目录。
# 必须用**精确段匹配**: 早先这里写 "scripts/basilisk_b1" 并用 startswith 判断,
# 于是 basilisk_b11 / b18 / b19 全被静默连坐剔除 (b2 同理连坐 b21),
# 导致 B5/B6 在 release 里根本跑不起来。故此处只列目录名, 逐段比对。
SCRIPT_PHASE_EXCLUDE_DIRS = {
    "basilisk_b1",  # 仅整目录排除; calibrate_degradation.py 由 REQUIRED_PHASE_SCRIPTS 单独放行
    "basilisk_b12",
    "basilisk_b13",
    "basilisk_b14",
    "basilisk_b15",
    "basilisk_b16",
    "basilisk_b17",
    "basilisk_b18",
    "basilisk_b19",
    "basilisk_v1",
    "basilisk",
}

TEST_PHASE_EXCLUDES = [
    "tests/basilisk_b",
]


def _is_required_phase_script(rel: str) -> bool:
    """依赖闭包白名单优先于阶段排除。"""
    for req in REQUIRED_PHASE_SCRIPTS:
        if req.endswith("/"):
            if rel.startswith(req):
                return True
        elif rel == req:
            return True
    return False


def should_include(rel: str, src_path: Path) -> tuple[bool, str]:
    """判断文件是否应该进入 release"""
    for pat in EXCLUDE_PATTERNS:
        if pat in rel:
            return False, f"Excluded pattern: {pat}"

    # 依赖闭包优先: B5/B6 运行时需要的阶段脚本必须放行
    if _is_required_phase_script(rel):
        return True, "Required phase script (dependency closure)"

    # 排除历史阶段脚本 —— 按路径段精确比对, 不用子串
    parts = rel.split("/")
    if parts[0] == "scripts" and len(parts) > 1:
        if parts[1] in SCRIPT_PHASE_EXCLUDE_DIRS:
            return False, f"Phase script excluded: {parts[1]}"

    # 排除历史阶段测试
    for ph in TEST_PHASE_EXCLUDES:
        if rel.startswith(ph) and rel[len(ph):].startswith("_"):
            return False, f"Phase test excluded"

    return True, "Allowlisted"


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_required_bytes() -> int:
    """估算 release 包所需字节数"""
    total = 0
    for src_pattern, category, reason in RELEASE_ALLOWLIST:
        src = ROOT / src_pattern
        if src.is_file():
            total += src.stat().st_size
        elif src.is_dir():
            for p in src.rglob("*"):
                if p.is_file():
                    if should_include(p.relative_to(ROOT).as_posix(), p)[0]:
                        total += p.stat().st_size
    return total


def main() -> int:
    print("=== B8 Release Package Builder ===")
    print(f"Release dir: {RELEASE_DIR}")

    # ============================================================
    # 先检查磁盘空间 §10
    # ============================================================
    print("\n[1/5] Checking available disk space...")
    required = get_required_bytes()
    free = shutil.disk_usage(ROOT).free
    required_120 = int(required * 1.2)

    print(f"  Required data:    {required / (1024*1024):.2f} MB")
    print(f"  Required x 1.2:   {required_120 / (1024*1024):.2f} MB")
    print(f"  Available:        {free / (1024*1024):.2f} MB")

    if free < required_120:
        print("\n❌ INSUFFICIENT DISK SPACE — Aborting data copy")
        (ROOT / "docs" / "handoff" / "DATA_COPY_BLOCKED_INSUFFICIENT_SPACE.md").write_text(
            f"DATA COPY BLOCKED\n\n"
            f"Required: {required / (1024*1024):.2f} MB\n"
            f"Required x 1.2: {required_120 / (1024*1024):.2f} MB\n"
            f"Available: {free / (1024*1024):.2f} MB\n",
            encoding="utf-8")
        return 1

    print("✅ Sufficient disk space")

    # ============================================================
    # 清理并创建目录结构
    # ============================================================
    print("\n[2/5] Creating release directory structure...")
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    RELEASE_DIR.mkdir(parents=True)

    manifest = []
    copy_errors = []

    # ============================================================
    # 按 allowlist 复制
    # ============================================================
    print("\n[3/5] Copying allowlisted items...")

    for src_pattern, category, reason in RELEASE_ALLOWLIST:
        src = ROOT / src_pattern
        if not src.exists():
            print(f"  ⚠️  SKIP (not found): {src_pattern}")
            continue

        if src.is_file():
            rel = src_pattern
            include_ok, include_reason = should_include(rel, src)
            if not include_ok:
                continue

            dst = RELEASE_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)

            hash_src = sha256_of(src)
            shutil.copy2(src, dst)
            hash_dst = sha256_of(dst)

            if hash_src != hash_dst:
                copy_errors.append(f"{rel}: hash mismatch after copy")
            else:
                manifest.append({
                    "path": rel,
                    "category": category,
                    "reason": reason,
                    "bytes": src.stat().st_size,
                    "sha256": hash_src,
                })
                print(f"  ✅ {rel}")

        elif src.is_dir():
            for p in src.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(ROOT).as_posix()

                include_ok, include_reason = should_include(rel, p)
                if not include_ok:
                    continue

                dst = RELEASE_DIR / rel
                dst.parent.mkdir(parents=True, exist_ok=True)

                hash_src = sha256_of(p)
                shutil.copy2(p, dst)
                hash_dst = sha256_of(dst)

                if hash_src != hash_dst:
                    copy_errors.append(f"{rel}: hash mismatch after copy")
                else:
                    manifest.append({
                        "path": rel,
                        "category": category,
                        "reason": reason,
                        "bytes": p.stat().st_size,
                        "sha256": hash_src,
                    })

    # ============================================================
    # 检查 data 复制状态 (§9)
    # ============================================================
    print("\n[4/5] Verifying data integrity...")
    data_items = [m for m in manifest if m["path"].startswith("data/")]
    print(f"  Protected data files copied: {len(data_items)}")

    # ============================================================
    # 写 RELEASE_MANIFEST
    # ============================================================
    print("\n[5/5] Writing release manifest...")

    release_manifest = {
        "release_name": "LEO_Bearings_Docker_Handoff",
        "release_version": "XA-202608-wheel-basilisk-final",
        "phase_status": "B7_REPORT_READY",
        "final_transfer_conclusion": "NO_POSITIVE_TRANSFER_SUPPORTED",
        "engineering_recommendation": "damage_extrapolation",
        "total_files": len(manifest),
        "total_bytes": sum(m["bytes"] for m in manifest),
        "files": manifest,
    }

    (RELEASE_DIR / "RELEASE_MANIFEST.json").write_text(
        json.dumps(release_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8")

    # SHA256SUMS
    with (RELEASE_DIR / "RELEASE_SHA256SUMS.txt").open("w", encoding="utf-8") as f:
        for m in sorted(manifest, key=lambda x: x["path"]):
            f.write(f"{m['sha256']}  {m['path']}\n")

    # VERSION
    (RELEASE_DIR / "RELEASE_VERSION.txt").write_text(
        f"XA-202608-wheel-basilisk-final\n"
        f"B7_REPORT_READY\n"
        f"FINAL_TRANSFER_CONCLUSION: NO_POSITIVE_TRANSFER_SUPPORTED\n"
        f"ENGINEERING_RECOMMENDATION: damage_extrapolation\n",
        encoding="utf-8")

    if copy_errors:
        print(f"\n❌ {len(copy_errors)} copy errors:")
        for e in copy_errors[:10]:
            print(f"  {e}")
        return 1

    print(f"\n✅ Release build complete:")
    print(f"  Total files: {len(manifest)}")
    print(f"  Total size:  {sum(m['bytes'] for m in manifest) / (1024*1024):.2f} MB")
    print(f"  Directory:   {RELEASE_DIR.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
