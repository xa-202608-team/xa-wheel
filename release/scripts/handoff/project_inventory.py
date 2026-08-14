"""§1 Project Inventory — 全项目扫描与分类。

严格遵循: data/** = READ_ONLY_CRITICAL
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# SAFE_DELETE 严格定义 (§2)
# ============================================================
SAFE_DELETE_PATTERNS = [
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".coverage",
    "htmlcov/",
    ".ipynb_checkpoints/",
    ".DS_Store",
    "Thumbs.db",
    "*.swp",
    "*.swo",
    "*~",
]

SAFE_DELETE_DIRNAMES = {"tmp", "temp", "cache", "__pycache__",
                        ".pytest_cache", ".mypy_cache", ".ruff_cache",
                        "htmlcov", ".ipynb_checkpoints"}

# ============================================================
# Category 定义 (§1)
# ============================================================
def categorize(p: Path, rel: str) -> tuple[str, str]:
    """返回 (category, reason)"""
    name = p.name
    suffix = p.suffix.lower()

    # ------------------------------
    # D. DATA_CRITICAL
    # ------------------------------
    if rel.startswith("data/"):
        if suffix in {".h5", ".hdf5", ".npz", ".json", ".csv", ".parquet", ".pt", ".pth", ".ckpt"}:
            return "DATA_CRITICAL", "Protected data artifact"
        return "DATA_CRITICAL", "data/ directory content"

    # ------------------------------
    # E. SAFE_DELETE candidates (§2)
    # ------------------------------
    for pat in SAFE_DELETE_PATTERNS:
        if pat.endswith("/") and pat.rstrip("/") in rel:
            return "SAFE_DELETE", f"Matched cache dir pattern: {pat}"
        if pat.startswith("*"):
            if name.endswith(pat[1:]):
                return "SAFE_DELETE", f"Matched cache file pattern: {pat}"
    if p.parent.name in SAFE_DELETE_DIRNAMES:
        return "SAFE_DELETE", f"Parent dir is cache: {p.parent.name}"
    if p.is_dir() and p.name in SAFE_DELETE_DIRNAMES:
        return "SAFE_DELETE", f"Cache directory: {p.name}"

    # ------------------------------
    # A. RELEASE_REQUIRED (最终交付必需)
    # ------------------------------
    if rel.startswith("src/"):
        return "RELEASE_REQUIRED", "Core source code"
    if rel.startswith("configs/"):
        return "RELEASE_REQUIRED", "Configuration files"
    if rel.startswith("tests/") and not rel.startswith("tests/basilisk_b"):
        return "RELEASE_REQUIRED", "Core test infrastructure"
    if rel in {"requirements.txt", "pyproject.toml", "setup.py", "README.md"}:
        return "RELEASE_REQUIRED", "Project infrastructure"

    # ------------------------------
    # B. REPRODUCIBILITY_REQUIRED (可复现性必需)
    # ------------------------------
    if "basilisk_b" in rel and "/test_" in rel:
        return "REPRODUCIBILITY_REQUIRED", "Baseline guard test"
    if rel.startswith("checkpoints/"):
        if "basilisk_b5" in rel or "basilisk_b6" in rel or "basilisk_b7" in rel:
            return "REPRODUCIBILITY_REQUIRED", "B5/B6/B7 frozen metric/checkpoint"
    if rel.startswith("scripts/basilisk_b5") or rel.startswith("scripts/basilisk_b6"):
        return "REPRODUCIBILITY_REQUIRED", "B5/B6 official protocol scripts"

    # ------------------------------
    # C. HISTORICAL_EVIDENCE (历史证据)
    # ------------------------------
    if "basilisk_b" in rel:
        return "HISTORICAL_EVIDENCE", "Phase artifact: evidence chain"
    if "STATUS_BASILISK" in rel:
        return "HISTORICAL_EVIDENCE", "Phase status: frozen evidence"
    if "baseline_contract" in rel:
        return "HISTORICAL_EVIDENCE", "Baseline contract: integrity"
    if rel.startswith("docs/basilisk_"):
        return "HISTORICAL_EVIDENCE", "Phase documentation"

    # ------------------------------
    # docs/figures - 最終图 = RELEASE_REQUIRED
    # ------------------------------
    if rel.startswith("docs/figures/basilisk_final/"):
        return "RELEASE_REQUIRED", "Final B7 figures"
    if rel.startswith("docs/figures/"):
        return "HISTORICAL_EVIDENCE", "Historical figures"
    if rel.startswith("docs/") and rel != "docs/results.md":
        return "HISTORICAL_EVIDENCE", "Historical documentation"
    if rel == "docs/results.md":
        return "REPRODUCIBILITY_REQUIRED", "Final results document"

    # ------------------------------
    # scripts/ - 主脚本 = RELEASE_REQUIRED
    # ------------------------------
    if rel.startswith("scripts/") and not "basilisk_b" in rel:
        return "RELEASE_REQUIRED", "Core script infrastructure"

    # ------------------------------
    # F. REVIEW_REQUIRED (需人工审核)
    # ------------------------------
    if rel.startswith("."):
        return "REVIEW_REQUIRED", "Dotfile: review before release"
    if rel.endswith(".md") or rel.endswith(".txt"):
        return "REVIEW_REQUIRED", "Document: review content"

    return "REVIEW_REQUIRED", "Uncategorized — manual review"


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print(f"Scanning project root: {ROOT}")
    print(f"DATA_PROTECTION: data/** = READ_ONLY_CRITICAL")

    inventory = []
    category_stats = {}

    # 只扫描文件，不修改，不删除
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()

        # Skip git dir
        if rel.startswith(".git/"):
            continue
        # Skip release dir we just created
        if rel.startswith("release/"):
            continue

        category, reason = categorize(p, rel)

        stat = p.stat()
        size = stat.st_size

        # 大文件 (> 100MB) 只做 content hash 摘要，小文件全量
        file_hash = sha256_of(p) if size < 100 * 1024 * 1024 else f"partial_{size}"

        inventory.append({
            "path": rel,
            "size_bytes": size,
            "suffix": p.suffix.lower(),
            "sha256": file_hash,
            "category": category,
            "reason": reason,
        })

        if category not in category_stats:
            category_stats[category] = {"count": 0, "total_bytes": 0}
        category_stats[category]["count"] += 1
        category_stats[category]["total_bytes"] += size

    # ============================================================
    # 按目录汇总
    # ============================================================
    dir_summary = {}
    for item in inventory:
        parts = item["path"].split("/")
        top = parts[0] if parts else "root"
        if top not in dir_summary:
            dir_summary[top] = {"count": 0, "total_bytes": 0}
        dir_summary[top]["count"] += 1
        dir_summary[top]["total_bytes"] += item["size_bytes"]

    # ============================================================
    # 输出 CSV + MD
    # ============================================================
    csv_path = ROOT / "docs" / "handoff" / "project_inventory.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("path,size_bytes,suffix,sha256,category,reason\n")
        for item in inventory:
            f.write(f"{item['path']},{item['size_bytes']},{item['suffix']},{item['sha256']},{item['category']},{item['reason']}\n")

    md_path = ROOT / "docs" / "handoff" / "project_inventory.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Project Inventory — B8-PREP\n\n")
        f.write(f"Total files scanned: {len(inventory)}\n\n")

        f.write("## Category Summary\n\n")
        f.write("| Category | Files | Total Size |\n")
        f.write("|---|---:|---:|\n")
        for cat in ["A. RELEASE_REQUIRED", "B. REPRODUCIBILITY_REQUIRED",
                     "C. HISTORICAL_EVIDENCE", "D. DATA_CRITICAL",
                     "E. SAFE_DELETE", "F. REVIEW_REQUIRED"]:
            key = cat.split(". ")[1] if ". " in cat else cat
            s = category_stats.get(key, {"count": 0, "total_bytes": 0})
            f.write(f"| {cat} | {s['count']} | {s['total_bytes'] / (1024*1024):.2f} MB |\n")

        f.write("\n## Directory Summary\n\n")
        f.write("| Directory | Files | Total Size |\n")
        f.write("|---|---:|---:|\n")
        for d, s in sorted(dir_summary.items(), key=lambda x: -x[1]["total_bytes"]):
            f.write(f"| {d}/ | {s['count']} | {s['total_bytes'] / (1024*1024):.2f} MB |\n")

        f.write("\n## SAFE_DELETE Candidates\n\n")
        safe = [i for i in inventory if i["category"] == "SAFE_DELETE"]
        f.write(f"Total: {len(safe)} files, {sum(i['size_bytes'] for i in safe) / (1024*1024):.2f} MB\n\n")
        f.write("| Path | Size (KB) | Reason |\n")
        f.write("|---|---:|---|\n")
        for i in sorted(safe, key=lambda x: -x["size_bytes"]):
            f.write(f"| {i['path']} | {i['size_bytes'] / 1024:.1f} | {i['reason']} |\n")

        f.write("\n## DATA_CRITICAL Protection Summary\n\n")
        data_items = [i for i in inventory if i["category"] == "DATA_CRITICAL"]
        f.write(f"Total protected data files: {len(data_items)}\n")
        f.write(f"Total protected data size: {sum(i['size_bytes'] for i in data_items) / (1024*1024):.2f} MB\n\n")
        f.write("`DATA_TREE_PROTECTED = true` — 所有 data/ 文件已列入只读保护\n")

    print(f"\nInventory written to:")
    print(f"  {csv_path.relative_to(ROOT)}")
    print(f"  {md_path.relative_to(ROOT)}")

    print(f"\nCategory summary:")
    for cat, s in sorted(category_stats.items()):
        print(f"  {cat:30s}: {s['count']:4d} files, {s['total_bytes'] / (1024*1024):8.2f} MB")

    print(f"\nData protection: data/ = READ_ONLY_CRITICAL ({len([i for i in inventory if i['category'] == 'DATA_CRITICAL'])} files)")

    # 写 JSON 供后续步骤使用
    json_path = ROOT / "docs" / "handoff" / "project_inventory.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "inventory": inventory,
            "category_stats": category_stats,
            "dir_summary": dir_summary,
            "data_tree_protected": True,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSAFE_DELETE candidates: {len([i for i in inventory if i['category'] == 'SAFE_DELETE'])} files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
