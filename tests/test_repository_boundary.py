"""仓库边界与 release 层级回归测试。

层级约束（Task 7 简报 Step 4）：
- 飞轮代码基线保持 release/ 前缀层级（release/src、release/scripts），
  不得出现根 src/ 平铺；
- release/data|checkpoints|results 与根 data/|results/|checkpoints/ 槽位
  不得有任何受跟踪工件文件（线下工件经 handoff/artifact-map.yaml 装配，
  不入 Git）。
"""
from pathlib import Path
import subprocess


def _tracked_files(root: Path) -> list[str]:
    """返回 git 索引受跟踪文件（POSIX 相对路径）；非 git 环境跳过断言。"""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    return [p.decode("utf-8", errors="replace") for p in proc.stdout.split(b"\x00") if p]


def test_wheel_release_layout_is_preserved() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "release" / "src" / "sim" / "wheel_sim.py").is_file()
    assert (root / "release" / "scripts" / "run_all.py").is_file()
    assert not (root / "src").exists()


def test_release_artifact_slots_have_no_tracked_files() -> None:
    """release/data|checkpoints|results 不得有受跟踪工件文件。"""
    root = Path(__file__).resolve().parents[1]
    tracked = _tracked_files(root)
    offenders = [
        rel for rel in tracked
        if rel.startswith(("release/data/", "release/checkpoints/", "release/results/"))
    ]
    assert offenders == [], f"release/ 工件槽位存在受跟踪文件: {offenders}"


def test_root_artifact_slots_have_no_tracked_files() -> None:
    """根 data/|results/|checkpoints/ 槽位不得有受跟踪文件（含描述文件）。"""
    root = Path(__file__).resolve().parents[1]
    tracked = _tracked_files(root)
    offenders = [
        rel for rel in tracked
        if rel.startswith(("data/", "results/", "checkpoints/"))
    ]
    assert offenders == [], f"根工件槽位存在受跟踪文件: {offenders}"


def test_no_tracked_binary_artifacts() -> None:
    """权重/数组/日志类工件后缀不得受跟踪。"""
    root = Path(__file__).resolve().parents[1]
    tracked = _tracked_files(root)
    offenders = [
        rel for rel in tracked
        if rel.endswith((".pt", ".pth", ".ckpt", ".npz", ".log", ".h5", ".hdf5"))
    ]
    assert offenders == [], f"存在受跟踪工件后缀文件: {offenders}"


def test_offline_release_manifest_not_tracked() -> None:
    """原 RELEASE_MANIFEST.sha256（含线下工件条目）不得作为 Git 代码真值受跟踪。"""
    root = Path(__file__).resolve().parents[1]
    tracked = _tracked_files(root)
    assert "release/RELEASE_MANIFEST.sha256" not in tracked
    assert (root / "release" / "CODE_MANIFEST.sha256").is_file()
