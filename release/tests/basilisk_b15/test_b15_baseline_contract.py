"""tests/basilisk_b15/test_b15_baseline_contract.py

Basilisk-B1.5 §2 —— baseline 契约测试。

键名照 `docs/basilisk_b15/baseline_contract.json` 实际结构读: 扁平表叫
`flat_sha256`, 分组表叫 `groups`, 数值结论按阶段分块 (`v1_content_hashes`,
`b1_results`/`b1_expected`, … `b14_results`/`b14_expected`)。猜键名会让断言
永远命中 KeyError 分支, 保护形同虚设 —— 本项目已犯过两次, 此处显式钉死。
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# 共享生命周期治理注册表 —— 全库唯一来源。
# B7 授权改写 docs/results.md 作为最终交付表达面, 因此该文件的"哈希永久不变"
# 约束退役, 改由语义审计 (scripts/basilisk_b7/audit_report_numbers.py) 看守。
# 算法产物 (数据/特征/split/权重/metrics/verdict) 仍然逐字节冻结, 不受影响。
# 退役理由逐条记录: docs/basilisk_b7/stale_guard_retirement.md
# --------------------------------------------------------------------------
import importlib.util as _ilu   # noqa: E402

_lr_spec = _ilu.spec_from_file_location(
    "lifecycle_registry", ROOT / "tests" / "lifecycle_registry.py")
lifecycle = _ilu.module_from_spec(_lr_spec)
_lr_spec.loader.exec_module(lifecycle)
CONTRACT = ROOT / "docs" / "basilisk_b15" / "baseline_contract.json"

NUMERIC_BLOCKS = ("v1", "b1", "b11", "b12", "b13", "b14")

# §2 明列必须冻结的对象
MUST_BE_FROZEN = (
    "src/sim/wheel_sim.py",
    "configs/wheel.yaml",
)

B13_WHEEL_SIM_SHA = (
    "6a768488fadbe7ea79b1a9416bc42bdd99bd4ea4e9bdc2bdae2e372c31ccc4c3")


def _load() -> dict:
    assert CONTRACT.exists(), f"缺少 {CONTRACT}"
    return json.loads(io.open(CONTRACT, encoding="utf-8").read())


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _numeric_pairs(contract: dict):
    for prefix in NUMERIC_BLOCKS:
        for suffix in ("_results", "_expected"):
            name = prefix + suffix
            if prefix == "v1" and suffix == "_results":
                name = "v1_content_hashes"   # v1 块的现场读叫这个名
            if name in contract:
                yield name, contract[name]


def test_contract_shape():
    c = _load()
    assert c["stage"] == "BASILISK_B1.5"
    assert "flat_sha256" in c and "groups" in c
    assert c["n_files"] == len(c["flat_sha256"])
    assert c["n_files"] >= 160
    # B1.5 是纯审计阶段, 契约里不得有 b15 的 mutable 组
    assert not any("b15" in g for g in c["groups"]), \
        "B1.5 无 mutable 组 —— 不应把自己的产物放进冻结契约"


def test_groups_and_flat_consistent():
    c = _load()
    merged = {}
    for g, files in c["groups"].items():
        merged.update(files)
    assert merged == c["flat_sha256"]


def test_frozen_chain_records_five_failures():
    """B1 / B1.1 / B1.2 / B1.3 / B1.4 的失败结论必须原样在链上。"""
    c = _load()
    fc = c["frozen_chain"]
    assert fc["BASILISK_B1"] == "B1_CALIBRATION_FAIL"
    assert fc["BASILISK_B1.1"] == "B11_CALIBRATION_FAIL"
    assert fc["BASILISK_B1.2"] == "B12_FAILURE_DEFINITION_FAIL"
    assert fc["BASILISK_B1.3"] == "B13_CALIBRATION_FAIL"
    assert fc["BASILISK_B1.4"] == "B14_CALIBRATION_FAIL"


def test_section2_mandated_objects_are_in_contract():
    """§2 逐项要求: wheel_sim.py / configs/wheel.yaml 必须被冻结。"""
    c = _load()
    flat = c["flat_sha256"]
    for rel in MUST_BE_FROZEN:
        assert rel in flat, f"§2 要求冻结 {rel}, 但契约里没有"
        assert flat[rel] != "MISSING"
    # analytic S2.5-S5B 与 basilisk profile 必须有对应分组
    gnames = " ".join(c["groups"].keys()).lower()
    assert "analytic" in gnames or any(
        "analytic" in r or "s2" in r or "s5" in r for r in flat)


def test_f2_threshold_and_ranges_pinned():
    """§2: F2 threshold / documented b0 range / Delta / tau 必须钉死。"""
    c = _load()
    b13 = c["b13_expected"]
    assert b13["b13_f2_threshold_Nm"] == "0.14311462970213382"
    assert b13["b13_Delta_min"] == "3.0"
    assert b13["b13_Delta_max"] == "10.0"
    assert b13["b13_tau_min"] == "0.5"
    assert b13["b13_tau_max"] == "2.0"
    assert b13["b13_wheel_sim_sha256"] == B13_WHEEL_SIM_SHA
    b14 = c["b14_expected"]
    assert b14["b14_b0_documented_min"] == "1e-06"
    assert b14["b14_b0_documented_max"] == "1e-05"
    # F2 threshold 的两个构成量 —— B1.5 的审计对象
    assert b14["b12_f2_eta_margin"] == "0.715573148510669"
    assert b14["b12_f2_torque_util_ref_p95"] == "0.284426851489331"
    assert b14["b12_f2_u_max_Nm"] == "0.2"
    assert b14["b12_f2_threshold_confidence"] == "DERIVED"


def test_b14_conclusions_pinned():
    c = _load()
    e = c["b14_expected"]
    assert e["b14_verdict"] == "B14_CALIBRATION_FAIL"
    assert e["b14_n_pass"] == "8"
    assert e["b14_n_gate"] == "14"
    assert e["b14_failure_fraction"] == "0.0"
    assert e["b14_reachability_verdict"] == "STRUCTURALLY_UNREACHABLE"
    assert e["b14_prediction_matched"] == "True"
    assert e["b14_frozen_calibration_written"] == "False"


def test_numeric_blocks_results_match_expected():
    """每个阶段的现场读值必须等于该阶段的期望值。

    应核对数从契约自身推导 (= 全部 `*_expected` 项之和), 不写死魔数 ——
    契约总数值项 = results + expected 两份, 故 results 侧约为其一半。
    """
    c = _load()
    n_expected_total = sum(
        len(block) for name, block in _numeric_pairs(c)
        if name.endswith("_expected"))
    n = 0
    for name, block in _numeric_pairs(c):
        prefix = name.split("_")[0]
        exp_name = f"{prefix}_expected"
        if not name.endswith("_expected") and exp_name in c:
            for k, v in c[exp_name].items():
                assert block.get(k) == v, \
                    f"{name}[{k}] = {block.get(k)!r} != 期望 {v!r}"
                n += 1
    assert n == n_expected_total, \
        f"核对了 {n} 项, 但契约声明了 {n_expected_total} 项期望值 —— 有块被漏读"
    assert n >= 80, f"只核对了 {n} 个数值项, 太少"


def test_old_artifacts_unchanged():
    """重算全部文件 hash; MISSING 项必须**仍然**缺失。"""
    c = _load()
    n_checked = n_missing = 0
    _n_hashed = _n_exempt = 0
    bad = []
    for rel, h in c["flat_sha256"].items():
        p = ROOT / rel
        if h == "MISSING":
            n_missing += 1
            assert not p.exists(), \
                f"{rel} 原本缺失却出现了 —— B1.5 越界生成了前阶段产物"
            continue
        assert p.exists(), f"{rel} 被删除了"
        if lifecycle.hash_exempt(rel):
            _n_exempt += 1
            n_checked += 1   # 计入总账, 只跳过逐字节比对
            continue
        now = _sha256(p)
        _n_hashed += 1
        if now != h:
            bad.append((rel, h, now))
        n_checked += 1
    assert not bad, f"以下文件被修改: {bad}"
    assert n_checked >= 140
    assert n_missing > 0, "MISSING 清单为空, 说明契约没在钉死缺失状态"
    # 防掏空 (B7 生命周期治理): 豁免只允许命中封闭白名单, 且实检项须占绝大多数。
    lifecycle.assert_registry_is_not_widened()
    assert _n_exempt <= 3, f"哈希豁免项过多 ({_n_exempt}) —— 疑似滥用"
    assert _n_hashed >= 20, f"逐字节实检项过少 ({_n_hashed}) —— 契约可能被掏空"



def test_b14_frozen_calibration_must_stay_absent():
    """§8 的连带要求: 不得回头冻结 B1.4 的标定。"""
    c = _load()
    rel = "checkpoints/basilisk_b14/frozen_calibration.json"
    assert c["flat_sha256"][rel] == "MISSING"
    assert not (ROOT / rel).exists()
