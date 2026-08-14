"""tests/lifecycle_registry.py —— 全库唯一的"生命周期治理"注册表。

## 为什么需要这个模块

B1 → B1.9 → B2 → B2.1 → B3X/B4X → B5 → B6 各阶段都在开工时写一份
`docs/basilisk_*/baseline_contract.json`, 把"当时磁盘上每个既有文件的 sha256"
钉住, 并由各自的 `test_*_old_artifacts_unchanged` 逐字节复核。这个机制对
**算法产物**是完全正确的, 必须永久保留 —— 数据集 / 特征 / split / 权重 /
metrics JSON / verdict 一旦漂移就是造假。

但同一份契约里也顺手钉住了 `docs/results.md` —— 它在 B1..B6 期间的角色是
"前阶段的只读引用文档", 所以"不得改动"当时成立。到了 B7 (结果表达阶段),
项目授权明确要求**改写** `docs/results.md` 作为最终交付表达面。此时那 12 条
断言变成了 **stale lifecycle guard**: 它们保护的不是数值正确性, 而是一个
"未来阶段不得推进"的生命周期假设, 而该假设已被显式人工授权取代。

## 治理原则 (一次性写在这里, 不做 12 份白名单)

    algorithm artifacts        -> immutable  (永久逐字节冻结, 任何阶段都不得改)
    final presentation surface -> B7 mutable (哈希冻结退役, 改为语义冻结)

"语义冻结"强于"哈希冻结": 哈希只能证明文件没被动过, 而语义审计能证明文件
里**每一个数字都与冻结的 B5/B6 machine-readable artifacts 对得上**
(见 `scripts/basilisk_b7/audit_report_numbers.py`)。

## 反滥用设计

* `PRESENTATION_MUTABLE` 是**封闭白名单**, 且 `assert_registry_is_not_widened()`
  会拒绝任何把算法产物混进来的改动 (路径前缀 + 后缀双重校验)。
* `hash_exempt()` 只对白名单命中项返回 True; 调用方必须同时断言
  "实检项数量下限" 与 "豁免项数量上限", 防止豁免逻辑被用来掏空整份契约。
* 退役理由逐条记录在 `docs/basilisk_b7/stale_guard_retirement.md`。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# 1. B7 授权可变的"最终表达面" —— 封闭白名单
# --------------------------------------------------------------------------
# 授权来源: BASILISK-B7 §2「允许修改」+ 本阶段追加的治理授权。
# 判据: 这些文件不参与任何数值计算, 不被任何训练/评估脚本读取, 删掉或改写
#       都不会使任何实验数字发生一比特变化。
PRESENTATION_MUTABLE: tuple[str, ...] = (
    "docs/results.md",
)

# 目录级授权 (前缀匹配)。同样只含表达面。
PRESENTATION_MUTABLE_PREFIXES: tuple[str, ...] = (
    "docs/figures/",
    "docs/技术方案报告/",
    "docs/basilisk_b7/",
    "docs/数据集下载/",
)

# --------------------------------------------------------------------------
# 2. 生命周期治理豁免 —— 被改造的 guard 文件本身
# --------------------------------------------------------------------------
# B5/B6 契约把 B2.1 与 B5 的测试文件也钉了哈希。B7 §14 授权把其中的
# stale lifecycle guard 改造为 lifecycle transition guard (保留守卫、不删测试),
# 因此这两个文件的哈希必然变化。它们是**测试代码**, 不是实验产物。
LIFECYCLE_GUARD_RETIRED: tuple[str, ...] = (
    "tests/basilisk_b21/test_b21_discipline.py",
    "tests/basilisk_b5/test_b5_baseline_contract.py",
    # B8-HANDOFF §2 授权: B6 契约测试里"B7/B8 产物永久缺席"的两条断言退役,
    # 改造为 lifecycle transition guard + 上游未污染逐字节核查。
    # 该文件被 B7 契约钉了哈希, 改造后哈希必然变化。它是**测试治理文件**,
    # 不是数据/结果/权重/算法产物 —— 改它动不了任何实验数字。
    # 逐条依据: docs/handoff/stale_guard_retirement_final.md
    "tests/basilisk_b6/test_b6_baseline_contract.py",
)

# --------------------------------------------------------------------------
# 3. 永久不可变 —— 任何阶段、任何授权都不得放进上面的白名单
# --------------------------------------------------------------------------
# 这是"反向护栏": 若有人以后试图把算法产物塞进 mutable 白名单,
# assert_registry_is_not_widened() 会在测试期直接失败。
IMMUTABLE_PREFIXES: tuple[str, ...] = (
    "checkpoints/",
    "data/",
    "src/",
    "configs/",
    "scripts/",
)
IMMUTABLE_SUFFIXES: tuple[str, ...] = (
    ".h5", ".npz", ".pt", ".pth", ".ckpt", ".yaml", ".yml", ".py",
)
IMMUTABLE_BASENAMES: tuple[str, ...] = (
    "split_manifest.json", "protocol.json", "protocol.md",
    "protocol_hash.json", "baseline_contract.json",
    "target_features.h5", "wheel_all.h5",
    "formal_metrics.json", "all_metrics.json", "final_verdict.json",
    "summary.json", "paired_gain.json", "paired_statistics.json",
    "warning_metrics.json", "lifetime_bins.json",
    "label_subset_manifest.json",
)


def is_presentation_mutable(rel: str) -> bool:
    """该相对路径是否属于 B7 授权可变的最终表达面。"""
    rel = rel.replace("\\", "/")
    if rel in PRESENTATION_MUTABLE:
        return True
    return any(rel.startswith(p) for p in PRESENTATION_MUTABLE_PREFIXES)


def is_retired_lifecycle_guard(rel: str) -> bool:
    """该路径是否是被 B7 §14 授权改造的 lifecycle guard 测试文件。"""
    return rel.replace("\\", "/") in LIFECYCLE_GUARD_RETIRED


def hash_exempt(rel: str) -> bool:
    """契约逐字节比对时是否应跳过该项。

    只有两类命中: B7 授权的表达面, 以及被授权改造的 lifecycle guard 测试。
    其余一律返回 False —— 包括所有算法产物。
    """
    return is_presentation_mutable(rel) or is_retired_lifecycle_guard(rel)


def assert_registry_is_not_widened() -> None:
    """护栏: 白名单里绝不允许出现算法产物。

    任何未来的"顺手加一项"都会在这里被挡下。校验三层: 目录前缀、文件后缀、
    已知的关键产物文件名。
    """
    bad: list[str] = []
    for rel in PRESENTATION_MUTABLE:
        r = rel.replace("\\", "/")
        if any(r.startswith(p) for p in IMMUTABLE_PREFIXES):
            bad.append(f"{rel}: 命中不可变目录前缀")
        if r.endswith(IMMUTABLE_SUFFIXES):
            bad.append(f"{rel}: 命中不可变后缀 (代码/权重/数据/配置)")
        if Path(r).name in IMMUTABLE_BASENAMES:
            bad.append(f"{rel}: 命中关键产物文件名")
    for pre in PRESENTATION_MUTABLE_PREFIXES:
        p = pre.replace("\\", "/")
        if any(p.startswith(q) for q in IMMUTABLE_PREFIXES):
            bad.append(f"{pre}: 目录前缀越界到不可变命名空间")
    # lifecycle guard 豁免只允许指向 tests/ 下的测试文件
    for rel in LIFECYCLE_GUARD_RETIRED:
        r = rel.replace("\\", "/")
        if not r.startswith("tests/"):
            bad.append(f"{rel}: lifecycle guard 豁免只能是 tests/ 下的测试文件")
    assert not bad, "LIFECYCLE_REGISTRY_WIDENED (禁止):\n  " + "\n  ".join(bad)


# --------------------------------------------------------------------------
# 4. 语义冻结 —— 取代 docs/results.md 的哈希冻结
# --------------------------------------------------------------------------
# 旧约束: hash(docs/results.md) == 冻结值
# 新约束: 下列五项全部为 True (由 audit_report_numbers.py 机器核对)
SEMANTIC_FREEZE_KEYS: tuple[str, ...] = (
    "report_numbers_match_frozen_metrics",
    "report_contains_final_transfer_conclusion",
    "report_contains_engineering_recommendation",
    "report_contains_negative_transfer_disclosure",
    "report_contains_damage_baseline",
)

REPORT_NUMBER_AUDIT = ROOT / "checkpoints/basilisk_b7/report_number_audit.json"
