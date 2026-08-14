"""§21 SAFE_DELETE Execution — Only cache/temp files.

GUARDS:
  - No path startswith data/
  - No path startswith checkpoints/
  - No path startswith docs/basilisk_
  - No *.h5
  - No *.pt/*.pth/*.ckpt
  - No *.json metrics
  - No config YAML
  - No test source
  - No src source
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SAFE_DELETE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
    "htmlcov",
    ".ipynb_checkpoints",
    "*.swp",
    "*.swo",
    "*~",
]


def is_safe_to_delete(p: Path) -> tuple[bool, str]:
    rel = p.relative_to(ROOT).as_posix()

    # 🔒 CRITICAL GUARDS — NEVER delete these
    if rel.startswith("data/"):
        return False, "VIOLATION: data/ directory"
    if rel.startswith("checkpoints/"):
        return False, "VIOLATION: checkpoints/ directory"
    if rel.startswith("docs/basilisk_"):
        return False, "VIOLATION: docs/basilisk_ directory"

    # 🔒 FILE EXTENSION GUARDS
    if p.suffix.lower() in {".h5", ".hdf5"}:
        return False, "VIOLATION: *.h5 file"
    if p.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        return False, "VIOLATION: model checkpoint"
    # Allow .json only if it's clearly a cache file
    if p.suffix.lower() == ".json" and not any(c in rel for c in ["cache", "__pycache__"]):
        # Check if it's in metrics directory — don't delete
        if "metrics" in rel or "result" in rel or "summary" in rel:
            return False, "VIOLATION: JSON metric/result"

    # 🔒 SOURCE/TEST GUARDS
    if rel.startswith("src/") and p.suffix.lower() == ".py":
        return False, "VIOLATION: src/ Python source"
    if rel.startswith("tests/") and p.suffix.lower() == ".py":
        return False, "VIOLATION: tests/ Python source"
    if rel.startswith("configs/") and p.suffix.lower() in {".yaml", ".yml"}:
        return False, "VIOLATION: config YAML"

    # ✅ Pattern-based safe deletion
    name = p.name
    for pat in SAFE_DELETE_PATTERNS:
        if pat.startswith("*."):
            if name.endswith(pat[1:]):
                return True, f"Matched cache pattern: {pat}"
        elif pat in name or pat in rel:
            return True, f"Matched cache directory/file: {pat}"

    return False, "Not in safe delete list"


def main() -> int:
    print("=" * 60)
    print("SAFE_DELETE EXECUTION — Cache/Temp Files Only")
    print("=" * 60)

    to_delete = []

    # Scan
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()

        # Skip release dir — don't delete what we just built
        if rel.startswith("release/"):
            continue
        if rel.startswith(".git/"):
            continue

        is_safe, reason = is_safe_to_delete(p)
        if is_safe:
            to_delete.append((p, reason))

    print(f"\nFound {len(to_delete)} cache/temp files eligible for deletion")
    total_bytes = sum(p.stat().st_size for p, _ in to_delete)
    print(f"Total reclaimable: {total_bytes / (1024*1024):.2f} MB")

    # Double guard: Verify no protected files are in the delete list
    violations = []
    for p, reason in to_delete:
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith("data/") or rel.startswith("checkpoints/") or rel.startswith("docs/basilisk_"):
            violations.append((rel, "Protected directory in delete list"))
        if p.suffix.lower() in {".h5", ".pt", ".pth", ".ckpt"}:
            violations.append((rel, "Protected file type in delete list"))

    if violations:
        print(f"\n❌ {len(violations)} VIOLATIONS IN DELETE LIST — ABORTING:")
        for rel, reason in violations:
            print(f"  {rel}: {reason}")
        return 1

    # Verify no violations in our deletion list
    print("\nVerifying all deletions are safe...")
    for p, reason in to_delete:
        assert "__pycache__" in str(p) or ".pyc" in str(p) or "cache" in str(p).lower(), \
            f"UNSAFE FILE IN DELETE LIST: {p}"
    print("✅ All deletions verified safe")

    # Execute
    print("\nExecuting deletion...")
    deleted = 0
    for p, reason in to_delete:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            print(f"  ⚠️  Failed to delete {p}: {e}")

    # Write deletion log
    log_path = ROOT / "docs" / "handoff" / "deletion_log.md"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("# Deletion Log — SAFE_DELETE Execution\n\n")
        f.write(f"Files deleted: {deleted}\n")
        f.write(f"Bytes reclaimed: {total_bytes} ({total_bytes/(1024*1024):.2f} MB)\n\n")
        f.write("## Deleted Files\n\n")
        for p, reason in to_delete:
            f.write(f"- {p.relative_to(ROOT)} ({reason})\n")

    print(f"\n✅ DONE: {deleted} files deleted, {total_bytes/(1024*1024):.2f} MB reclaimed")
    print(f"Log written to: docs/handoff/deletion_log.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
