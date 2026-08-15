"""刷新 release manifest —— 覆盖打包后新增/改写的文件。

为什么需要它: README_DOCKER_HANDOFF.md / FINAL_RESULTS.md / DOCKER_HANDOFF_CHECKLIST.md
以及 release 内修订版的 data/MANIFEST_DOCKER_HANDOFF.md 是在 build_release.py 跑完之后
才写入的。若不刷新, RELEASE_MANIFEST.json 里就会缺条目或留旧哈希 —— 那样"哈希校验通过"
是假话。

本脚本只重算 release/ 内部的哈希, 不读也不写原仓库 data/。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = ROOT / "release" / "LEO_Bearings_Docker_Handoff"

# 这三个文件本身是 manifest 产物, 不进 manifest (否则自指)
SELF = {"RELEASE_MANIFEST.json", "RELEASE_SHA256SUMS.txt", "RELEASE_VERSION.txt"}

EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    mf_path = RELEASE_DIR / "RELEASE_MANIFEST.json"
    if not mf_path.exists():
        print(f"!! 缺 {mf_path} —— 先跑 build_release.py")
        return 1

    old = json.loads(mf_path.read_text(encoding="utf-8"))
    old_by_path = {m["path"]: m for m in old.get("files", [])}

    manifest: list[dict] = []
    added, changed, unchanged = [], [], 0

    for p in sorted(RELEASE_DIR.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(RELEASE_DIR).as_posix()
        if rel in SELF:
            continue
        if any(part in EXCLUDE_PARTS for part in p.parts):
            continue
        # 冒烟产物不进 manifest
        if rel.startswith("checkpoints/_smoke"):
            continue

        digest = sha256_of(p)
        entry = {
            "path": rel,
            "bytes": p.stat().st_size,
            "sha256": digest,
        }
        prev = old_by_path.get(rel)
        if prev is None:
            added.append(rel)
        elif prev.get("sha256") != digest:
            changed.append(rel)
        else:
            unchanged += 1
            # 保留原有分类信息
            for k in ("category", "reason"):
                if k in prev:
                    entry[k] = prev[k]
        if prev is not None:
            for k in ("category", "reason"):
                if k in prev and k not in entry:
                    entry[k] = prev[k]
        manifest.append(entry)

    removed = [r for r in old_by_path if r not in {m["path"] for m in manifest}]

    out = dict(old)
    out["total_files"] = len(manifest)
    out["total_bytes"] = sum(m["bytes"] for m in manifest)
    out["files"] = manifest
    mf_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                       encoding="utf-8")

    with (RELEASE_DIR / "RELEASE_SHA256SUMS.txt").open("w", encoding="utf-8") as f:
        for m in sorted(manifest, key=lambda x: x["path"]):
            f.write(f"{m['sha256']}  {m['path']}\n")

    print(f">> manifest 刷新完成: {len(manifest)} 文件, "
          f"{out['total_bytes'] / (1 << 20):.2f} MB")
    print(f"   新增 {len(added)} / 变更 {len(changed)} / 未变 {unchanged} / "
          f"移除 {len(removed)}")
    for r in added:
        print(f"     + {r}")
    for r in changed:
        print(f"     ~ {r}")
    for r in removed:
        print(f"     - {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
