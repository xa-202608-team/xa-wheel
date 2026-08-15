"""tests/basilisk_b16/test_b16_old_artifacts_unchanged.py —— §15: 旧阶段产物零改动。

B1.6 的授权极窄: 只新增 `*_b16` 命名空间。任何对 analytic S2.5–S5B、
basilisk_v1、b1…b15、`configs/wheel.yaml`、`src/sim/wheel_sim.py` 的改动都是
越权 —— 契约会抓到哈希变化, 本文件额外从"结论层"再抓一次。
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# 上游七个失败结论 —— B1.6 不得重新解释其中任何一个
FROZEN_VERDICTS = {
    "S5B": "S5B_BASELINE_WEAK",
    "BASILISK_V1": "BASILISK_V1_READY",
    "BASILISK_B1": "B1_CALIBRATION_FAIL",
    "BASILISK_B1.1": "B11_CALIBRATION_FAIL",
    "BASILISK_B1.2": "B12_FAILURE_DEFINITION_FAIL",
    "BASILISK_B1.3": "B13_CALIBRATION_FAIL",
    "BASILISK_B1.4": "B14_CALIBRATION_FAIL",
    "BASILISK_B1.5": "B15_FAILURE_SEMANTICS_MISMATCH",
}


def test_old_b15_unchanged(b16_contract):
    """B1.5 的结论与关键数值必须逐字保留在契约里且当前仍然成立。"""
    exp = b16_contract["b15_expected"]
    assert exp["b15_threshold_verdict"] == "B15_FAILURE_SEMANTICS_MISMATCH"
    now = b16_contract["b15_results"]
    for k, v in exp.items():
        assert now.get(k) == v, f"B1.5 结果 {k} 变了: {v} -> {now.get(k)}"


def test_old_artifacts_unchanged(b16_contract):
    """七个上游结论在 frozen_chain 里原样存在。"""
    chain = b16_contract["frozen_chain"]
    for k, v in FROZEN_VERDICTS.items():
        assert chain.get(k) == v, f"{k} 结论被改写: {chain.get(k)}"


@pytest.mark.parametrize("rel", [
    "configs/wheel.yaml",
    "src/sim/wheel_sim.py",
    "docs/results.md",
])
def test_protected_files_in_contract(b16_contract, rel):
    """§0 明令不得修改的文件必须被契约覆盖 (否则改了也抓不到)。"""
    assert rel in b16_contract["flat_sha256"], f"{rel} 未纳入契约"
    assert b16_contract["flat_sha256"][rel] != "MISSING"


def test_b16_namespace_only():
    """B1.6 新增的脚本/测试/文档必须全部落在 b16 命名空间内。"""
    for d, pat in (("scripts", "basilisk_b16"), ("tests", "basilisk_b16"),
                   ("docs", "basilisk_b16")):
        p = ROOT / d / pat
        assert p.is_dir(), f"缺 {d}/{pat}"
    # 不得在 requirements / Dockerfile 里写入 Basilisk (跨阶段长期约束)
    for f in ("requirements.txt", "Dockerfile", "docker-compose.yml"):
        p = ROOT / f
        if not p.exists():
            continue
        txt = io.open(p, encoding="utf-8", errors="replace").read().lower()
        assert "basilisk" not in txt, f"{f} 里出现 Basilisk —— 长期禁止"


def test_b16_config_has_no_tunable_simulation_params():
    """§13: b16 config 不得含任何可调退化/仿真参数。

    只扫**键名**。禁止项清单本身写在注释与 `*_forbidden` 值里, 那是声明而非
    参数, 不能自触发 —— B1.5 曾在此误报一次。
    """
    import yaml
    cfg = yaml.safe_load(
        io.open(ROOT / "configs" / "wheel_basilisk_b16.yaml",
                encoding="utf-8").read())
    banned = ("q0", "b0", "delta", "tau", "horizon", "im_rated", "wear",
              "threshold_value", "n_traj", "seed")
    seen = []

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                seen.append(k)
                assert str(k).lower() not in banned, \
                    f"b16 config 出现可调参数键 {path}.{k}"
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(cfg)
    assert len(seen) > 30, f"扫描到的键太少 ({len(seen)}), 检查逻辑可能失效"
