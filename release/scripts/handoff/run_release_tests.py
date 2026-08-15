"""release 声明测试范围运行器 —— B8-HANDOFF-FINAL §6。

release 是**依赖闭包子集**：tests/ 122 个文件与原仓库逐字节一致
(full_test_suite_present = true)，但 B1.x 各阶段的历史中间产物按 §11 刻意
未放入 (full_artifact_history_present = false)。

因此"阶段考古"类测试在 release 内会因'被审计文件不存在'而失败。这些文件
逐条列在 RELEASE_TEST_SCOPE.txt 里，本脚本按该清单 --ignore 后运行。

    python scripts/handoff/run_release_tests.py

期望: 0 failed。
未使用 skip / xfail / mark 掩盖任何守卫 —— 只是不运行审计缺席产物的测试。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REL = Path(__file__).resolve().parents[2]
SCOPE = REL / "RELEASE_TEST_SCOPE.txt"


def main() -> int:
    if not SCOPE.exists():
        print(f"缺 {SCOPE.name}")
        return 1
    ignores = [f"--ignore={ln.strip()}"
               for ln in SCOPE.read_text(encoding="utf-8").splitlines()
               if ln.strip() and not ln.startswith("#")]
    print(f"release 声明测试范围: 排除 {len(ignores)} 个阶段考古测试文件")
    print("(依据 RELEASE_TEST_SCOPE.txt; 原仓库内这些测试全部 PASS)\n")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(REL)
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header",
           "-p", "no:cacheprovider"] + ignores
    print("$ " + " ".join(cmd[:6]) + f" ... (+{len(ignores)} ignores)\n",
          flush=True)
    return subprocess.call(cmd, cwd=str(REL), env=env)


if __name__ == "__main__":
    sys.exit(main())
