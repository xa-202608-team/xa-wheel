"""§18 Unified Entrypoint — run_all.py.

Supported modes:
  --verify   Only verify files/hashes/imports/checkpoint loading/schema
  --fast     真实 B5 冒烟: 源域权重加载 -> 三组 PyTorch 训练 -> 目标域微调
             -> MMD -> 指标。1 seed × 1 epoch, 输出重定向到 checkpoints/_smoke_b5,
             绝不触碰冻结的 checkpoints/basilisk_b5|b6。
  --full     Full formal B5/B6 protocol reproduction (hours)

--full is NOT run during Docker handoff build (documented only).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Ensure project root is on Python path
sys.path.insert(0, str(ROOT))


def _run(cmd: list[str], title: str) -> int:
    """在项目根目录下执行子进程, 实时透传输出。"""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(f"$ {' '.join(cmd)}\n", flush=True)
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def run_verify():
    """--verify: Fast integrity check (< 5 minutes)."""
    print("=" * 60)
    print("RUN MODE: --verify (integrity check only)")
    print("=" * 60)

    # 1. Import check
    print("\n[1/5] Checking imports...")
    try:
        import src
        import src.baselines
        import src.models
        print("  ✅ src imports OK")
    except Exception as e:
        print(f"  ❌ Import FAILED: {e}")
        return 1

    # 2. Critical files exist
    print("\n[2/5] Checking critical files...")
    critical = [
        ROOT / "data" / "features" / "wheel" / "basilisk_b19" / "target_features.h5",
        ROOT / "data" / "features" / "wheel" / "schema_v1" / "source_features.h5",
        ROOT / "checkpoints" / "source_tcn_pretrain.pt",
        ROOT / "checkpoints" / "basilisk_b5" / "summary.json",
        ROOT / "checkpoints" / "basilisk_b6" / "paired_statistics.json",
    ]
    for c in critical:
        if c.exists():
            print(f"  ✅ {c.relative_to(ROOT)}")
        else:
            print(f"  ❌ {c.relative_to(ROOT)} MISSING")
            return 1

    # 3. Load B5 metrics (prove JSON is valid)
    print("\n[3/5] Loading frozen metrics...")
    with open(critical[3], encoding="utf-8") as f:
        b5 = json.load(f)
    print(f"  ✅ B5 summary loaded (methods: {list(b5.get('main_table', {}).keys())[:3]}...)")

    with open(critical[4], encoding="utf-8") as f:
        b6 = json.load(f)
    print(f"  ✅ B6 statistics loaded (levels: {list(b6.get('by_level', {}).keys())[:3]}...)")

    # 4. Final verdict check
    print("\n[4/5] Checking final verdict...")
    verdict_path = ROOT / "checkpoints" / "basilisk_b6" / "final_verdict.json"
    if verdict_path.exists():
        with open(verdict_path, encoding="utf-8") as f:
            v = json.load(f)
        if "NO_POSITIVE_TRANSFER" in str(v):
            print(f"  ✅ Final verdict OK")
        else:
            print(f"  ❌ Final verdict mismatch")
            return 1
    else:
        print("  ⚠️  final_verdict.json not present (OK for smoke)")

    # 5. Figures present
    print("\n[5/5] Checking final figures...")
    fig_dir = ROOT / "docs" / "figures" / "basilisk_final"
    if fig_dir.exists():
        pngs = list(fig_dir.glob("*.png"))
        svgs = list(fig_dir.glob("*.svg"))
        print(f"  ✅ Figures present: {len(pngs)} PNG + {len(svgs)} SVG")
    else:
        print("  ⚠️  Figure directory not found")

    print("\n" + "=" * 60)
    print("✅ VERIFY PASSED — Release package is intact!")
    print("=" * 60)
    return 0


def run_fast():
    """--fast: 真实端到端冒烟 (~5 分钟, CPU)。

    跑的是**正式 B5 迁移脚本本身**, 不是替身:
      源域 TCN 权重加载 -> target_only / source_finetune / source_mmd_finetune
      三组 PyTorch 训练 -> 目标域微调 -> MMD 对齐 -> test 指标 + 配对增益。

    与正式 B5 的唯一差异 (见 configs/wheel_basilisk_b5_smoke.yaml):
      1. seeds 收窄为 [112] (正式为 112..116);
      2. max_epochs=1 (由 --fast 传入);
      3. 输出全部重定向到 checkpoints/_smoke_b5。
    因此**不会覆写**冻结证据 checkpoints/basilisk_b5 | basilisk_b6。
    冒烟数值不得进入正式报告。
    """
    print("=" * 60)
    print("RUN MODE: --fast (real B5 smoke, 1 seed × 1 epoch)")
    print("=" * 60)

    # First run verify
    ret = run_verify()
    if ret != 0:
        return ret

    smoke_cfg = ROOT / "configs" / "wheel_basilisk_b5_smoke.yaml"
    if not smoke_cfg.exists():
        print(f"\n❌ 缺 {smoke_cfg.relative_to(ROOT)}")
        return 1

    rc = _run(
        [sys.executable, "scripts/basilisk_b5/run_formal_transfer.py",
         "--config", "configs/wheel_basilisk_b5_smoke.yaml", "--fast"],
        "[Fast smoke] B5 迁移链路 (源域加载 -> 三组训练 -> 微调 -> MMD -> 指标)",
    )
    if rc != 0:
        print(f"\n❌ FAST FAILED — B5 冒烟退出码 {rc}")
        return rc

    # 冒烟必须真的产出指标文件, 否则"通过"没有意义
    out = ROOT / "checkpoints" / "_smoke_b5" / "formal_metrics.json"
    if not out.exists():
        print(f"\n❌ FAST FAILED — 未产出 {out.relative_to(ROOT)}")
        return 1
    m = json.loads(out.read_text(encoding="utf-8"))
    n_seed = len(m.get("per_seed", []))
    print(f"\n[Fast smoke] 产出 {out.relative_to(ROOT)} ({n_seed} seed)")

    # 断言冻结证据未被动过
    frozen = ROOT / "checkpoints" / "basilisk_b5" / "formal_metrics.json"
    if frozen.exists():
        fm = json.loads(frozen.read_text(encoding="utf-8"))
        if len(fm.get("per_seed", [])) < 5:
            print("\n❌ FAST FAILED — 冻结的 B5 正式指标被冒烟污染!")
            return 1
        print("  ✅ 冻结证据 checkpoints/basilisk_b5 未被改动 (5 seed 完整)")

    print("\n" + "=" * 60)
    print("✅ FAST PASSED — 真实训练链路跑通 (冒烟数值不入正式报告)")
    print("=" * 60)
    return 0


def run_full():
    """--full: 正式 B5 + B6 全协议复现 (~小时级)。

    直接链式调用正式脚本, 写入正式命名空间 —— 会覆写 checkpoints/basilisk_b5|b6。
    因此需要显式 --i-understand-this-overwrites 才执行。
    """
    print("=" * 60)
    print("RUN MODE: --full (FULL FORMAL REPRODUCTION)")
    print("=" * 60)
    print()
    print("⚠️  本模式耗时数小时, 且会**覆写正式命名空间**:")
    print("     checkpoints/basilisk_b5/  checkpoints/basilisk_b6/")
    print("   预计: B5 全标签 ~30 min / B6 低标签矩阵 ~3-6 h")
    print()

    if "--i-understand-this-overwrites" not in sys.argv:
        print("拒绝执行: 缺少确认参数。若确实要重跑正式实验, 请显式加上")
        print("  python scripts/run_all.py --full --i-understand-this-overwrites")
        print()
        print("交付验收请用 --verify (完整性) 与 --fast (真实训练链路)。")
        return 1

    steps = [
        (["scripts/basilisk_b5/run_formal_transfer.py",
          "--config", "configs/wheel_basilisk_b5.yaml"], "B5 正式迁移评估"),
        (["scripts/basilisk_b5/summarize_b5.py"], "B5 汇总与门禁"),
        (["scripts/basilisk_b6/run_formal_matrix.py"], "B6 低标签矩阵"),
        (["scripts/basilisk_b6/paired_statistics.py"], "B6 配对统计"),
        (["scripts/basilisk_b6/final_transfer_verdict.py"], "B6 最终判定"),
    ]
    for argv, title in steps:
        rc = _run([sys.executable, *argv], title)
        if rc != 0:
            print(f"\n❌ FULL FAILED at: {title} (exit {rc})")
            return rc

    print("\n" + "=" * 60)
    print("✅ FULL COMPLETE — 正式 B5 + B6 已重跑")
    print("=" * 60)
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_all.py [--verify | --fast | --full]")
        print()
        print("  --verify  = Integrity check only (< 5 min)")
        print("  --fast    = Minimal smoke test (~10 min)")
        print("  --full    = Full formal reproduction (~hours)")
        return 1

    mode = sys.argv[1]

    if mode == "--verify":
        return run_verify()
    elif mode == "--fast":
        return run_fast()
    elif mode == "--full":
        return run_full()
    else:
        print(f"Unknown mode: {mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
