"""独立两阶段完整性验证 —— 代码基线与线下工件分离（B8-HANDOFF-FINAL §14 演进）。

这是一个**独立读取程序**：不 import 生成器，自己重新遍历、重算哈希，
再与清单逐条比对。调用方式：

    无 --artifact-root:
      只校验 release/CODE_MANIFEST.sha256 —— Git 受跟踪代码基线
      (manifest entries == release/ 下实际代码/配置/测试/文档文件,
       逐条 sha256 重算比对, 集合与内容双向核对)

    有 --artifact-root $ARTIFACT_ROOT:
      先校验 release/CODE_MANIFEST.sha256 (同上, 代码存在性),
      再校验 $ARTIFACT_ROOT/HANDOFF_MANIFEST.json 及其登记的线下工件
      (逐条存在性 + size + sha256 重算, 线下工件存在性;
       另做"磁盘->manifest"方向: 枚举 ARTIFACT_ROOT 下普通文件,
       未在 HANDOFF_MANIFEST.json 登记的多余文件报 FAIL)

任何不一致 -> 退出码 1。

与原 RELEASE_MANIFEST.sha256 校验的对应关系（严格性未降低，只拆两阶段）:
- 原三向核对 entries==files / size / sha256 中, 代码阶段的集合与 sha256
  两向由 CODE_MANIFEST.sha256 (sha256sum 标准格式, 无 size 字段) 承担;
- size 一向随线下工件清单迁入 HANDOFF_MANIFEST.json (契约
  schemas/handoff-manifest.schema.json: files[].path/role/size/sha256),
  冻结科学结论对应的 checkpoint/结果哈希仍在工件阶段逐条核对;
- 原 RELEASE_MANIFEST.sha256 不再作为 Git 代码校验真值, 仅作线下装配
  HANDOFF_MANIFEST.json 的参考输入。

代码阶段"实际文件"集合的取法:
- 在 git 仓库内 -> `git ls-files` 过滤 release/ 前缀 (精确受跟踪语义);
- 非 git 环境 (如 Docker 镜像内 COPY 出的 release/) -> 目录遍历,
  排除运行产物 (__pycache__/*.pyc/.pytest_cache)、线下工件槽位
  (data/checkpoints/results) 与线下参考清单 RELEASE_MANIFEST.sha256。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REL = Path(__file__).resolve().parents[2]
CODE_MAN = REL / "CODE_MANIFEST.sha256"
SELF_EXCLUDE = {"CODE_MANIFEST.sha256"}
# 线下参考清单: 工作树可能存在(未跟踪), 不属于代码基线
OFFLINE_REFERENCE_EXCLUDE = {"RELEASE_MANIFEST.sha256"}
# 线下工件槽位子树: 装配时才落盘, 不属于代码基线
ARTIFACT_SUBDIRS = ("data", "checkpoints", "results")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_runtime_artifact(rel_posix: str) -> bool:
    parts = rel_posix.split("/")
    if "__pycache__" in parts or ".pytest_cache" in parts:
        return True
    if rel_posix.endswith(".pyc"):
        return True
    return parts[0] in ARTIFACT_SUBDIRS


def git_tracked_release_files() -> "set[str] | None":
    """返回 git 索引中 release/ 下文件的 release/ 相对 POSIX 路径。

    非 git 仓库 (或 git 不可用) 时返回 None, 由调用方退回目录遍历。
    """
    try:
        # cwd = release/, pathspec "." 使输出路径相对 release/ (与 manifest 一致);
        # -z + quotepath=false 保证中文等非 ASCII 路径不做八进制转义
        proc = subprocess.run(
            ["git", "-C", str(REL), "-c", "core.quotepath=false",
             "ls-files", "-z", "--", "."],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    files = set()
    for raw in proc.stdout.split(b"\x00"):
        if not raw:
            continue
        rel = raw.decode("utf-8", errors="replace").strip()
        if not rel or rel in SELF_EXCLUDE or rel in OFFLINE_REFERENCE_EXCLUDE:
            continue
        if not _is_runtime_artifact(rel):
            files.add(rel)
    return files or None


def walk_release_files() -> "set[str]":
    """非 git 环境的退回枚举: release/ 下除运行产物/工件槽位/线下清单外的文件。"""
    actual = set()
    for p in REL.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(REL).as_posix()
        if rel in SELF_EXCLUDE or rel in OFFLINE_REFERENCE_EXCLUDE:
            continue
        if _is_runtime_artifact(rel):
            continue
        actual.add(rel)
    return actual


def verify_code_manifest() -> bool:
    """阶段一: 校验 CODE_MANIFEST.sha256 (sha256sum 格式: 哈希 + 两空格 + 路径)。"""
    print("=" * 70)
    print("阶段一: CODE_MANIFEST.sha256 代码基线独立验证")
    print("=" * 70)
    if not CODE_MAN.exists():
        print("缺 CODE_MANIFEST.sha256")
        print("CODE_MANIFEST_VERIFY_FAIL")
        return False

    declared: "dict[str, str]" = {}
    for ln in CODE_MAN.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        # sha256sum 标准格式: <sha256>  <path> (两个空格分隔)
        if "  " not in ln:
            print(f"格式错误: {ln[:80]}")
            print("CODE_MANIFEST_VERIFY_FAIL")
            return False
        digest, path = ln.split("  ", 1)
        declared[path.strip()] = digest

    tracked = git_tracked_release_files()
    source = "git ls-files (release/)" if tracked is not None else "目录遍历 (非 git 环境)"
    actual = tracked if tracked is not None else walk_release_files()

    missing = sorted(set(declared) - actual)      # manifest 有, 磁盘没有
    extra = sorted(actual - set(declared))        # 磁盘有, manifest 没有
    hash_bad: "list[str]" = []

    for rel in sorted(set(declared) & actual):
        real_h = sha256(REL / rel)
        if real_h != declared[rel]:
            hash_bad.append(f"{rel}: {declared[rel][:16]} -> {real_h[:16]}")

    print(f"实际文件来源     : {source}")
    print(f"manifest entries : {len(declared)}")
    print(f"actual files     : {len(actual)}")
    print(f"hash mismatch    : {len(hash_bad)}")
    print(f"in manifest only : {len(missing)}")
    print(f"on disk only     : {len(extra)}")

    for label, items in (("MISSING", missing), ("EXTRA", extra), ("HASH", hash_bad)):
        for it in items[:15]:
            print(f"  {label}: {it}")

    ok = (len(declared) == len(actual) and not missing and not extra and not hash_bad)
    if ok:
        print(f"CODE_MANIFEST_VERIFY_PASS —— {len(declared)}/{len(declared)} match, "
              f"entries == actual files")
    else:
        print("CODE_MANIFEST_VERIFY_FAIL")
    print("=" * 70)
    return ok


def verify_artifact_root(artifact_root: Path) -> bool:
    """阶段二: 校验 $ARTIFACT_ROOT/HANDOFF_MANIFEST.json 及其登记文件哈希。

    HANDOFF_MANIFEST.json 采用契约 schemas/handoff-manifest.schema.json
    的 files[] 结构 (path/role/size/sha256/...), path 相对 ARTIFACT_ROOT。

    双向核对:
    - manifest -> 磁盘: 逐条存在性 + size + sha256 重算;
    - 磁盘 -> manifest: 枚举 ARTIFACT_ROOT 下普通文件 (HANDOFF_MANIFEST.json
      自身除外), 未登记的多余文件报 FAIL (EXTRA), 防止装配目录夹带未审计内容。
    """
    print("=" * 70)
    print("阶段二: HANDOFF_MANIFEST.json 线下工件独立验证")
    print("=" * 70)
    man = artifact_root / "HANDOFF_MANIFEST.json"
    if not man.is_file():
        print(f"缺 {man}")
        print("HANDOFF_MANIFEST_VERIFY_FAIL")
        print("=" * 70)
        return False

    try:
        doc = json.loads(man.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"HANDOFF_MANIFEST.json 解析失败: {exc}")
        print("HANDOFF_MANIFEST_VERIFY_FAIL")
        print("=" * 70)
        return False

    entries = doc.get("files")
    if not isinstance(entries, list) or not entries:
        print("HANDOFF_MANIFEST.json 缺少非空 files[] (契约要求逐文件 path/size/sha256)")
        print("HANDOFF_MANIFEST_VERIFY_FAIL")
        print("=" * 70)
        return False

    missing: "list[str]" = []
    size_bad: "list[str]" = []
    hash_bad: "list[str]" = []
    extra: "list[str]" = []

    declared_paths: "set[str]" = set()
    root_resolved = artifact_root.resolve()
    for ent in entries:
        rel = str(ent.get("path", ""))
        declared_paths.add(rel)
        declared_size = ent.get("size")
        declared_hash = str(ent.get("sha256", ""))
        p = (artifact_root / rel).resolve()
        try:
            p.relative_to(root_resolved)
        except ValueError:
            print(f"路径逃逸 ARTIFACT_ROOT: {rel}")
            print("HANDOFF_MANIFEST_VERIFY_FAIL")
            print("=" * 70)
            return False
        if not p.is_file():
            missing.append(rel)
            continue
        real_sz = p.stat().st_size
        if isinstance(declared_size, int) and real_sz != declared_size:
            size_bad.append(f"{rel}: {declared_size} -> {real_sz}")
            continue
        if declared_hash:
            real_h = sha256(p)
            if real_h != declared_hash:
                hash_bad.append(f"{rel}: {declared_hash[:16]} -> {real_h[:16]}")

    # 磁盘 -> manifest 方向: ARTIFACT_ROOT 下未登记的普通文件一律 EXTRA
    man_resolved = man.resolve()
    for p in sorted(root_resolved.rglob("*")):
        if not p.is_file():
            continue
        if p == man_resolved:
            continue
        try:
            rel = p.relative_to(root_resolved).as_posix()
        except ValueError:
            continue
        if rel not in declared_paths:
            extra.append(rel)

    print(f"manifest entries : {len(entries)}")
    print(f"missing          : {len(missing)}")
    print(f"size mismatch    : {len(size_bad)}")
    print(f"hash mismatch    : {len(hash_bad)}")
    print(f"extra undeclared : {len(extra)}")

    for label, items in (("MISSING", missing), ("SIZE", size_bad), ("HASH", hash_bad),
                         ("EXTRA", extra)):
        for it in items[:15]:
            print(f"  {label}: {it}")

    ok = not missing and not size_bad and not hash_bad and not extra
    if ok:
        print(f"HANDOFF_MANIFEST_VERIFY_PASS —— {len(entries)}/{len(entries)} match, "
              f"size + sha256 + 双向集合核对一致")
    else:
        print("HANDOFF_MANIFEST_VERIFY_FAIL")
    print("=" * 70)
    return ok


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="两阶段完整性验证: 代码基线 (CODE_MANIFEST.sha256) "
                    "与可选线下工件 (HANDOFF_MANIFEST.json)。"
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="线下工件根目录: 提供时先校验代码基线, 再校验"
             " $ARTIFACT_ROOT/HANDOFF_MANIFEST.json 及其登记文件哈希",
    )
    args = parser.parse_args(argv)

    code_ok = verify_code_manifest()
    if args.artifact_root is not None:
        artifact_ok = verify_artifact_root(Path(args.artifact_root))
        ok = code_ok and artifact_ok
    else:
        ok = code_ok

    if ok:
        print("MANIFEST_VERIFY_PASS")
        return 0
    print("MANIFEST_VERIFY_FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
