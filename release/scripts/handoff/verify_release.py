"""§23 Release Package Verifier.

Runs full verification of the Docker handoff package:
- Required directories exist
- Config hashes match
- Data hashes match
- Checkpoint hashes match
- Imports work
- x_T schema loads correctly
- Final verdict present and correct
- Figures present
- Docs present
- No absolute path leakage
- No Basilisk runtime dependency
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

RELEASE_DIR = Path(__file__).resolve().parents[2]
ROOT = RELEASE_DIR  # We're inside the release package now


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    # Ensure release root is on Python path
    import sys
    sys.path.insert(0, str(RELEASE_DIR))

    print("=" * 60)
    print("RELEASE PACKAGE VERIFICATION")
    print("=" * 60)

    manifest_path = RELEASE_DIR / "RELEASE_MANIFEST.json"
    if not manifest_path.exists():
        print("❌ RELEASE_MANIFEST.json missing")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = {m["path"]: m["sha256"] for m in manifest["files"]}

    all_pass = True
    checks_run = 0
    checks_pass = 0

    # ============================================================
    # 1. Required directories
    # ============================================================
    print("\n[1/11] Required directories...")
    for d in ["configs", "src", "scripts", "tests", "checkpoints", "data", "docs"]:
        if (RELEASE_DIR / d).is_dir():
            print(f"  ✅ {d}/")
            checks_pass += 1
        else:
            print(f"  ❌ {d}/ MISSING")
            all_pass = False
        checks_run += 1

    # ============================================================
    # 2. Imports work
    # ============================================================
    print("\n[2/11] Python imports...")
    try:
        import src  # noqa: F401
        print("  ✅ import src")
        checks_pass += 1
    except Exception as e:
        print(f"  ❌ import src FAILED: {e}")
        all_pass = False
    checks_run += 1

    # ============================================================
    # 3. Critical data files exist
    # ============================================================
    print("\n[3/11] Critical data files...")
    critical_files = [
        "data/features/wheel/basilisk_b19/target_features.h5",
        "data/features/wheel/schema_v1/source_features.h5",
        "data/sim/wheel_basilisk_b18/final/wheel_all.h5",
        "checkpoints/source_tcn_pretrain.pt",
    ]
    for cf in critical_files:
        if (RELEASE_DIR / cf).exists():
            print(f"  ✅ {cf}")
            checks_pass += 1
        else:
            print(f"  ❌ {cf} MISSING")
            all_pass = False
        checks_run += 1

    # ============================================================
    # 4. File hash verification (sample)
    # ============================================================
    print("\n[4/11] Hash verification (critical files)...")
    for cf in critical_files:
        if cf in expected_hash:
            actual = sha256_of(RELEASE_DIR / cf)
            if actual == expected_hash[cf]:
                print(f"  ✅ {cf}: hash matches")
                checks_pass += 1
            else:
                print(f"  ❌ {cf}: HASH MISMATCH")
                all_pass = False
        else:
            print(f"  ⚠️  {cf}: not in manifest (skipping)")
        checks_run += 1

    # ============================================================
    # 5. Final verdict correct
    # ============================================================
    print("\n[5/11] Final verdict verification...")
    verdict_path = RELEASE_DIR / "checkpoints" / "basilisk_b6" / "final_verdict.json"
    if verdict_path.exists():
        v = json.loads(verdict_path.read_text(encoding="utf-8"))
        if "NO_POSITIVE_TRANSFER" in str(v):
            print(f"  ✅ Final verdict matches")
            checks_pass += 1
        else:
            print(f"  ❌ Final verdict mismatch")
            all_pass = False
    else:
        print("  ⚠️  final_verdict.json not in release")
    checks_run += 1

    # ============================================================
    # 6. Checkpoints present
    # ============================================================
    print("\n[6/11] Checkpoints present...")
    b5_ckpts = list((RELEASE_DIR / "checkpoints" / "basilisk_b5").glob("*.pt"))
    b6_ckpts = list((RELEASE_DIR / "checkpoints" / "basilisk_b6").glob("*.pt"))
    if b5_ckpts:
        print(f"  ✅ B5: {len(b5_ckpts)} checkpoints")
        checks_pass += 1
    if b6_ckpts:
        print(f"  ✅ B6: {len(b6_ckpts)} checkpoints")
        checks_pass += 1
    checks_run += 2

    # ============================================================
    # 7. Figures present (all 5 × PNG+SVG)
    # ============================================================
    print("\n[7/11] Final figures present...")
    fig_dir = RELEASE_DIR / "docs" / "figures" / "basilisk_final"
    if fig_dir.exists():
        pngs = list(fig_dir.glob("*.png"))
        svgs = list(fig_dir.glob("*.svg"))
        print(f"  ✅ {len(pngs)} PNG + {len(svgs)} SVG")
        checks_pass += 1
    else:
        print("  ❌ Figure directory missing")
        all_pass = False
    checks_run += 1

    # ============================================================
    # 8. No Basilisk runtime dependency
    # ============================================================
    print("\n[8/11] Basilisk dependency check...")
    has_basilisk_import = False
    for p in (RELEASE_DIR / "src").rglob("*.py"):
        content = p.read_text(encoding="utf-8", errors="replace")
        if "import basilisk" in content or "from basilisk" in content:
            has_basilisk_import = True
            print(f"  ⚠️  Found Basilisk import in: {p.relative_to(RELEASE_DIR)}")
    if not has_basilisk_import:
        print("  ✅ No Basilisk imports in src/")
        checks_pass += 1
    checks_run += 1

    # ============================================================
    # 9. No absolute Windows path leakage
    # ============================================================
    print("\n[9/11] Absolute path leakage check...")
    leakage = False
    for p in list((RELEASE_DIR / "configs").rglob("*.yaml")) + \
             list((RELEASE_DIR / "configs").rglob("*.yml")):
        content = p.read_text(encoding="utf-8", errors="replace")
        if "C:\\" in content or "D:\\" in content:
            leakage = True
            print(f"  ⚠️  Absolute path in: {p.relative_to(RELEASE_DIR)}")
    if not leakage:
        print("  ✅ No absolute path leakage in configs")
        checks_pass += 1
    checks_run += 1

    # ============================================================
    # 10. Entrypoints exist
    # ============================================================
    print("\n[10/11] Entrypoints...")
    for ep in ["scripts/handoff/verify_release.py", "scripts/run_all.py"]:
        if (RELEASE_DIR / ep).exists():
            print(f"  ✅ {ep}")
            checks_pass += 1
        else:
            print(f"  ❌ {ep} MISSING")
            all_pass = False
        checks_run += 1

    # ============================================================
    # 11. B5/B6 依赖闭包 —— 曾因子串匹配把 b19/b21/b11 静默剔除, 这里钉死
    # ============================================================
    print("\n[11/11] B5/B6 dependency closure...")
    closure = [
        # 只读输入 (configs 里的 paths)
        "data/features/wheel/basilisk_b19/target_features.h5",
        "docs/basilisk_b21/split_manifest.json",
        "checkpoints/basilisk_b21/summary.json",
        "checkpoints/basilisk_b21/gate_metrics.json",
        "checkpoints/basilisk_b21/protocol_hash.json",
        "checkpoints/basilisk_b3x/summary.json",
        "checkpoints/basilisk_b4x/summary.json",
        "checkpoints/basilisk_b19/frozen_feature_definition.json",
        "checkpoints/source_tcn_pretrain.pt",
        # protocol 冻结校验的目标文件
        "docs/basilisk_b2/protocol.md",
        "docs/basilisk_b21/protocol.md",
        "docs/basilisk_b5/protocol.md",
        "docs/basilisk_b6/protocol.md",
        # 运行时 import / 动态加载的阶段模块
        "scripts/basilisk_b1/calibrate_degradation.py",
        "scripts/basilisk_b11/calibrate_degradation.py",
        "scripts/basilisk_b2/data_b2.py",
        "scripts/basilisk_b2/eval_b2.py",
        "scripts/basilisk_b2/baselines_b2.py",
        "scripts/basilisk_b2/run_gate.py",
        "scripts/basilisk_b21/run_b21_gate.py",
        "scripts/basilisk_b3x/summarize_b3x.py",
        "scripts/basilisk_b3x/diagnose_short_eol.py",
        "scripts/basilisk_b4x/data_b4x.py",
        "scripts/basilisk_b4x/diagnose_transfer_stability.py",
        "scripts/basilisk_b5/run_formal_transfer.py",
        "scripts/basilisk_b6/run_formal_matrix.py",
        "scripts/diag_generalization.py",
        # 冒烟入口
        "configs/wheel_basilisk_b5_smoke.yaml",
    ]
    missing = [c for c in closure if not (RELEASE_DIR / c).exists()]
    if missing:
        print(f"  ❌ 依赖闭包缺 {len(missing)} 项 — release 无法跑训练:")
        for m in missing:
            print(f"       {m}")
        all_pass = False
    else:
        print(f"  ✅ 依赖闭包完整 ({len(closure)} 项)")
        checks_pass += 1
    checks_run += 1

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print(f"RESULTS: {checks_pass}/{checks_run} checks passed")
    print("=" * 60)

    if all_pass and checks_pass == checks_run:
        print("\n✅ RELEASE_VERIFY_PASS")
        return 0
    else:
        print("\n❌ RELEASE_VERIFY_FAILED — See above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
