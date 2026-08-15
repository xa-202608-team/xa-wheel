"""§4 Local Path & Secrets Audit — 扫描敏感文件与绝对路径。

- 扫描密钥/token/凭证
- 扫描源代码/文档中的本机绝对路径
- 输出 audit report 到 docs/handoff/local_path_audit.md
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SECRET_PATTERNS = [
    (".env", "Env file with secrets"),
    ("*.key", "Private key"),
    ("*.pem", "PEM certificate/key"),
    ("credentials*", "Credentials file"),
    ("token*", "Token file"),
    ("secret*", "Secret file"),
    ("local_settings*", "Local settings"),
]

SECRET_DIRS = [
    ".vscode/",
    ".idea/",
    ".claude/",
    "__pycache__/",
]

ABSOLUTE_PATH_PATTERNS = [
    (r"C:\\", "Windows C: drive"),
    (r"D:\\", "Windows D: drive"),
    (r"/home/", "Unix home"),
    (r"/Users/", "macOS home"),
    (r"Anaconda", "Anaconda absolute path"),
    (r"conda\\", "Conda absolute path"),
    (r"miniconda", "Miniconda absolute path"),
]


def matches_pattern(name: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if pat.startswith("*"):
            if name.endswith(pat[1:]):
                return True
        elif pat.endswith("/"):
            if pat.rstrip("/") in name:
                return True
        elif name == pat:
            return True
    return False


def main() -> int:
    print("=== §4 Local Path & Secrets Audit ===")

    secret_hits = []
    absolute_path_hits = []

    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()

        if rel.startswith(".git/") or rel.startswith("release/"):
            continue

        # ------------------------------
        # 敏感文件扫描
        # ------------------------------
        if matches_pattern(p.name, [sp[0] for sp in SECRET_PATTERNS]):
            for pat_name, reason in SECRET_PATTERNS:
                if matches_pattern(p.name, [pat_name]):
                    secret_hits.append({"path": rel, "reason": reason})
                    break

        # ------------------------------
        # 源代码/文档中的绝对路径扫描
        # ------------------------------
        if p.suffix.lower() in {".py", ".md", ".txt", ".yaml", ".yml",
                                 ".json", ".sh", ".ps1", ".bat"}:
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                for pat_re, reason in ABSOLUTE_PATH_PATTERNS:
                    if re.search(pat_re, content):
                        # Count occurrences
                        count = len(re.findall(pat_re, content))
                        absolute_path_hits.append({
                            "path": rel,
                            "pattern": pat_re,
                            "reason": reason,
                            "count": count,
                        })
                        break
            except Exception:
                pass  # Skip binary files

    # ============================================================
    # 写报告
    # ============================================================
    md_path = ROOT / "docs" / "handoff" / "local_path_audit.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Local Path & Secrets Audit — B8-PREP\n\n")

        f.write("## Secret / Private File Candidates\n\n")
        if secret_hits:
            f.write(f"Found {len(secret_hits)} potential secret files:\n\n")
            f.write("| Path | Reason |\n")
            f.write("|---|---|\n")
            for h in secret_hits:
                f.write(f"| {h['path']} | {h['reason']} |\n")
        else:
            f.write("✅ No secret/private key files found.\n")

        f.write("\n## Absolute Path References in Source/Docs\n\n")
        if absolute_path_hits:
            f.write(f"Found {len(absolute_path_hits)} files with absolute path references:\n\n")
            f.write("| Path | Pattern | Reason | Count |\n")
            f.write("|---|---|---|---:|\n")
            for h in sorted(absolute_path_hits, key=lambda x: -x["count"]):
                f.write(f"| {h['path']} | `{h['pattern']}` | {h['reason']} | {h['count']} |\n")
        else:
            f.write("✅ No absolute path references found.\n")

        f.write("\n## Recommendations\n\n")
        f.write("- Secret/private key files: DO NOT include in Docker handoff package.\n")
        f.write("- Absolute path references: Convert to project-relative paths where possible.\n")
        f.write("- Historical frozen documents containing absolute paths: DO NOT modify original;\n")
        f.write("  ensure runtime code does not depend on them.\n")

    print(f"\nAudit written to: {md_path.relative_to(ROOT)}")
    print(f"  Secret candidates: {len(secret_hits)}")
    print(f"  Absolute path hits: {len(absolute_path_hits)}")

    # 写 JSON
    json_path = ROOT / "docs" / "handoff" / "local_path_audit.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "secret_hits": secret_hits,
            "absolute_path_hits": absolute_path_hits,
            "audit_passed": len(secret_hits) == 0,
        }, f, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
