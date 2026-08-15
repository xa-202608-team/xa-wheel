"""生成 RELEASE_MANIFEST.sha256 —— B8-HANDOFF-FINAL §14。

覆盖 release 中**所有**文件，每行: relative_path | size | sha256
排除自身 (RELEASE_MANIFEST.sha256) —— 自引用无法自洽。

生成后请用独立读取程序验证一次: scripts/handoff/verify_manifest.py
要求 manifest entries == actual files。
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REL = Path(__file__).resolve().parents[2]
OUT = REL / "RELEASE_MANIFEST.sha256"

# 自引用文件不入 manifest
SELF_EXCLUDE = {"RELEASE_MANIFEST.sha256"}


def is_runtime_byproduct(rel: str) -> bool:
    """Python 解释器再生的字节码不是交付内容, 否则跑过一次脚本 manifest 就漂移。"""
    return "__pycache__" in rel.split("/") or rel.endswith(".pyc")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    rows: list[tuple[str, int, str]] = []
    for p in sorted(REL.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(REL).as_posix()
        if rel in SELF_EXCLUDE:
            continue
        if is_runtime_byproduct(rel):
            continue
        rows.append((rel, p.stat().st_size, sha256(p)))

    total = sum(r[1] for r in rows)
    lines = [
        "# RELEASE_MANIFEST.sha256 —— B8-HANDOFF-FINAL §14",
        "#",
        "# 格式: <relative_path>\\t<size_bytes>\\t<sha256>",
        "# 排除: RELEASE_MANIFEST.sha256 自身 (自引用) + __pycache__/*.pyc (运行时再生)",
        f"# entries: {len(rows)}",
        f"# total_bytes: {total}",
        "#",
        "# 验证: python scripts/handoff/verify_manifest.py",
        "#",
    ]
    for rel, sz, h in rows:
        lines.append(f"{rel}\t{sz}\t{h}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"entries      : {len(rows)}")
    print(f"total_bytes  : {total:,} ({total / 1e9:.3f} GB)")
    print(f"已写 {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
