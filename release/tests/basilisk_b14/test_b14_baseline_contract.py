"""tests/basilisk_b14/test_b14_baseline_contract.py

§21: `test_b14_old_artifacts_unchanged` 及 baseline 契约完整性。

契约由 `scripts/basilisk_b14/verify_baseline.py --tag before/after` 写出/校验,
覆盖 analytic lineage + basilisk_v1 / b1 / b11 / b12 / b13 全部产物的文件 hash
与数值结论。

键名照 docs/basilisk_b14/baseline_contract.json 实际结构读:
  * 文件 hash 扁平表叫 `flat_sha256` (另有分组表 `groups`), 不是 `files`;
  * 数值结论按阶段分块 `v1_*` / `b1_*` / `b11_*` / `b12_*` / `b13_*`,
    每块都有 `_results` (现场读) 与 `_expected` (钉死值) 两份, 没有统一的
    `numeric` 键。猜键名会让断言永远命中 KeyError 分支, 保护形同虚设。
  * before/after 共用同一个契约文件: `--tag after` 是**校验**而非另写一份,
    所以一致性由 verify 的退出码保证, 不靠比较两个 JSON。
"""
from __future__ import annotations

import hashlib

import pytest

from conftest import ROOT, read_json

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

NUMERIC_BLOCKS = ("v1", "b1", "b11", "b12", "b13")


def _numeric_pairs(contract: dict):
    """把所有 (块名, 键, 值) 摊平, 覆盖 _results 与 _expected 两份。"""
    for prefix in NUMERIC_BLOCKS:
        for suffix in ("_results", "_expected"):
            name = prefix + suffix
            if prefix == "v1" and suffix == "_results":
                name = "v1_content_hashes"          # v1 块的现场读叫这个名
            block = contract.get(name)
            if block is None:
                continue
            for k, v in block.items():
                yield name, k, v


def test_contract_exists(contract_rec):
    assert contract_rec is not None, \
        "缺 docs/basilisk_b14/baseline_contract.json —— 必须先跑 verify_baseline.py --tag before"
    assert contract_rec["stage"] == "BASILISK_B1.4"
    # 冻结链上的五条前阶段结论必须原样在册
    fc = contract_rec["frozen_chain"]
    assert fc["BASILISK_V1"] == "BASILISK_V1_READY"
    assert fc["BASILISK_B1"] == "B1_CALIBRATION_FAIL"
    assert fc["BASILISK_B1.1"] == "B11_CALIBRATION_FAIL"
    assert fc["BASILISK_B1.2"] == "B12_FAILURE_DEFINITION_FAIL"
    assert fc["BASILISK_B1.3"] == "B13_CALIBRATION_FAIL"


def test_b14_old_artifacts_unchanged(contract_rec):
    """§21 命名测试: 既有 basilisk_v1/b1/b11/b12/b13 产物一个都不许变。

    契约里每个条目的 sha256 现场重算并逐位比对; MISSING 条目 (FAIL 分支本就
    不存在的产物) 必须**仍然**不存在 —— 若突然出现, 说明 B1.4 越界生成了
    前阶段本不该有的东西。
    """
    files = contract_rec["flat_sha256"]
    assert len(files) == contract_rec["n_files"]
    n_checked = n_missing = 0
    _n_hashed = _n_exempt = 0
    drift = []
    for rel, expect in files.items():
        p = ROOT / rel
        if expect == "MISSING":
            n_missing += 1
            assert not p.exists(), \
                f"{rel} 在契约中记为 MISSING 但现在存在 —— B1.4 越界生成了前阶段产物"
            continue
        assert p.exists(), f"{rel} 契约存在但文件已消失"
        if lifecycle.hash_exempt(rel):
            _n_exempt += 1
            n_checked += 1   # 计入总账, 只跳过逐字节比对
            continue
        now = hashlib.sha256(p.read_bytes()).hexdigest()
        _n_hashed += 1
        if now != expect:
            drift.append((rel, expect[:12], now[:12]))
        n_checked += 1
    assert not drift, f"既有产物被改动: {drift}"
    assert n_checked >= 100, f"契约覆盖文件过少 ({n_checked})"
    assert n_missing > 0, "契约未钉死任何 must_stay_absent 条目"
    # 防掏空 (B7 生命周期治理): 豁免只允许命中封闭白名单, 且实检项须占绝大多数。
    lifecycle.assert_registry_is_not_widened()
    assert _n_exempt <= 3, f"哈希豁免项过多 ({_n_exempt}) —— 疑似滥用"
    assert _n_hashed >= 20, f"逐字节实检项过少 ({_n_hashed}) —— 契约可能被掏空"



def test_groups_and_flat_consistent(contract_rec):
    """分组表与扁平表必须一致 —— 防止某组被悄悄摘掉后 flat 仍旧好看。"""
    flat = contract_rec["flat_sha256"]
    merged = {}
    for g, files in contract_rec["groups"].items():
        for rel, h in files.items():
            merged[rel] = h
    assert merged == flat
    # B1.4 不得给自己开 mutable 组: 所有组名都指向前阶段或只读源码
    assert "b14" not in " ".join(contract_rec["groups"].keys()).lower()
    assert "must_stay_absent" in contract_rec["groups"]


def test_contract_numeric_conclusions_unchanged(contract_rec):
    """数值结论 (B1/B1.1/B1.2/B1.3 的判定与关键常数) 逐值不变。"""
    pairs = list(_numeric_pairs(contract_rec))
    assert len(pairs) >= 100, f"数值项过少 ({len(pairs)})"
    exp = contract_rec["b13_expected"]
    res = contract_rec["b13_results"]
    # B1.3 的判定必须仍是 FAIL 13/14 —— 不许被"修好"
    assert exp["b13_verdict"] == "B13_CALIBRATION_FAIL"
    assert exp["b13_n_pass"] == "13"
    assert exp["b13_n_gate"] == "14"
    assert exp["b13_failed_gate"] == "early_eol_fraction"
    assert exp["b13_early_eol_fraction"] == "0.3"
    assert exp["b13_frozen_calibration_written"] == "False"
    # 现场读与钉死值必须一致 (契约写入时若已漂移, 这里立刻暴露)
    for k, v in exp.items():
        assert res.get(k) == v, f"b13 现场值与钉死值不符: {k} {res.get(k)} != {v}"
    # 前阶段的失败结论同样不许变
    assert contract_rec["b12_expected"]["b12_electrical_verdict"] == \
        "PROVENANCE_INSUFFICIENT"


def test_before_after_contract_verifies_clean(contract_rec):
    """§22 step 1 / step 12: before/after 共用同一契约, after 必须校验通过。

    这里做的是与 verify() 等价的独立复核: 逐项重算并比对, 不调用被测脚本本身。
    """
    if contract_rec is None:
        pytest.skip("契约未生成")
    bad = []
    n_ex = 0
    for rel, h_ref in contract_rec["flat_sha256"].items():
        if lifecycle.hash_exempt(rel):
            n_ex += 1
            continue
        p = ROOT / rel
        now = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"
        if now != h_ref:
            bad.append(rel)
    assert not bad, f"契约项漂移: {bad}"
    lifecycle.assert_registry_is_not_widened()
    assert n_ex <= 3, f"哈希豁免项过多 ({n_ex}) —— 疑似滥用"


def test_b13_protocol_and_probe_hashes_pinned(contract_rec):
    """B1.3 的 protocol / probe hash 必须以完整 sha256 钉住。"""
    exp = contract_rec["b13_expected"]
    assert exp["b13_protocol_sha256"] == \
        "53efee3d85de9bcc4f69cc8f629cce7a2a164b32242c7d65be43c8f6f5db1667"
    assert exp["b13_probe_content_sha256"] == \
        "f029a91d34178110a203200458a3b7b3885707e5d750d4500269a99f1bb9f7d8"
    assert exp["b13_paired_trajectory_hash"] == \
        "ff1de559b2746dcb394c4de6d70bda4d96ca594d1b953f58a4fa390366f41cf6"
    for k in ("b13_protocol_sha256", "b13_formula_sha256",
              "b13_probe_content_sha256", "b13_paired_trajectory_hash"):
        assert len(exp[k]) == 64


def test_b13_frozen_constants_pinned(contract_rec):
    """B1.4 只读的那批常数必须与 B1.3 契约逐字符一致。"""
    exp = contract_rec["b13_expected"]
    assert exp["b13_f2_threshold_Nm"] == "0.14311462970213382"
    assert exp["b13_b_fail_f2_Nms_per_rad"] == "0.0005586102246421416"
    assert exp["b13_b_fail_old_Nms_per_rad"] == "2.416666666666667e-05"
    assert exp["b13_b0_scale"] == "23.114905847261028"
    assert exp["b13_omega_ref_rad_s"] == "253.95995891950514"
    assert exp["b13_q0_old_min"] == "0.04137931034482758"
    assert exp["b13_q0_old_max"] == "0.41379310344827586"
    assert exp["b13_Delta_min"] == "3.0" and exp["b13_Delta_max"] == "10.0"
    assert exp["b13_tau_min"] == "0.5" and exp["b13_tau_max"] == "2.0"
    assert exp["b13_resolved_branch"] == "MULTIPLICATIVE_b0_PREFACTOR"


def test_wheel_sim_unchanged(contract_rec):
    """仿真核心 src/sim/wheel_sim.py 不得改动 (退化方程是全链条基准)。"""
    rel = "src/sim/wheel_sim.py"
    files = contract_rec["flat_sha256"]
    assert rel in files, "契约未覆盖 wheel_sim.py"
    now = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    assert now == files[rel], "wheel_sim.py 被改动 —— 退化方程基准失效"
    assert now == "6a768488fadbe7ea79b1a9416bc42bdd99bd4ea4e9bdc2bdae2e372c31ccc4c3", \
        "wheel_sim.py hash 与 B1.3 报告公开值不符"
    assert contract_rec["b13_expected"]["b13_wheel_sim_sha256"] == now


def test_basilisk_not_written_into_deploy_files():
    """既定约束: Basilisk 不得写入 requirements.txt / Dockerfile / docker-compose.yml。"""
    for rel in ("requirements.txt", "Dockerfile", "docker-compose.yml"):
        p = ROOT / rel
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="replace").lower()
        assert "basilisk" not in txt, f"{rel} 出现 basilisk —— 违反既定约束"
