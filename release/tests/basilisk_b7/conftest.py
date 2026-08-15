"""B7 conftest — 共享的 frozen artifact 路径与 hash 验证工具。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

FROZEN_ARTIFACTS = {
    # B5 冻结产物 (算法结果, B7 只读不改)
    "b5_summary": ROOT / "checkpoints" / "basilisk_b5" / "summary.json",
    "b5_paired_gain": ROOT / "checkpoints" / "basilisk_b5" / "paired_gain.json",
    "b5_verdict": ROOT / "checkpoints" / "basilisk_b5" / "final_verdict.json",
    # B6 冻结产物 (算法结果, B7 只读不改)
    "b6_statistics": ROOT / "checkpoints" / "basilisk_b6" / "paired_statistics.json",
    "b6_warning": ROOT / "checkpoints" / "basilisk_b6" / "warning_metrics.json",
    "b6_lifetime": ROOT / "checkpoints" / "basilisk_b6" / "lifetime_bins.json",
    "b6_verdict": ROOT / "checkpoints" / "basilisk_b6" / "final_verdict.json",
    # B2.1 冻结产物
    "b21_split": ROOT / "checkpoints" / "basilisk_b21" / "split_manifest.json",
    # B7 产物 (本阶段生成)
    "b7_final_tables": ROOT / "checkpoints" / "basilisk_b7" / "final_tables.json",
    "b7_frozen_index": ROOT / "checkpoints" / "basilisk_b7" / "frozen_result_index.json",
    "b7_number_audit": ROOT / "checkpoints" / "basilisk_b7" / "report_number_audit.json",
    "b7_claim_audit": ROOT / "checkpoints" / "basilisk_b7" / "claim_audit.json",
}

B7_DOCS = {
    "final_tables_md": ROOT / "docs" / "basilisk_b7" / "final_tables.md",
    "frozen_index_md": ROOT / "docs" / "basilisk_b7" / "frozen_result_index.md",
    "figure_manifest": ROOT / "docs" / "basilisk_b7" / "figure_manifest.md",
    "limitations": ROOT / "docs" / "basilisk_b7" / "limitations.md",
    "retirement": ROOT / "docs" / "basilisk_b7" / "stale_guard_retirement.md",
    "baseline_contract_json": ROOT / "docs" / "basilisk_b7" / "baseline_contract.json",
}

FIGURES = [
    ROOT / "docs" / "figures" / "basilisk_final" / f"fig{i}_{name}.{ext}"
    for i, name in [(1, "basilisk_health_trajectory"), (2, "transfer_gain_vs_labels"),
                    (3, "primary_method_comparison"), (4, "health_management_flow"),
                    (5, "split_coverage_diagnostic")]
    for ext in ("png", "svg")
]


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="session")
def frozen_artifacts() -> dict[str, Path]:
    return FROZEN_ARTIFACTS


@pytest.fixture(scope="session")
def b7_docs() -> dict[str, Path]:
    return B7_DOCS


@pytest.fixture(scope="session")
def figures() -> list[Path]:
    return FIGURES
