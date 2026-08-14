"""tests/basilisk_b1/test_old_basilisk_v1_unchanged.py

B1 §1 —— baseline contract 自动化守卫。

B1 全程只允许**新增** B1 命名空间下的产物; 旧 analytic lineage (S2.5/S3/S4/S5/S5B)
与已冻结的 BASILISK-V1 产物必须逐字节不变。本文件把 §1 的"任务结束必须逐项
unchanged"从人工核对变成可自动执行的测试。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

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

CONTRACT = ROOT / "docs/basilisk_b1/baseline_contract.json"
# BASILISK-V1 冻结时公布的三个 hash (硬编码, 防契约文件本身被整体替换后仍"自洽")。
# 注意口径差异 —— 这三个不是同一种 hash:
#   profile  : provenance.json 记录的 profiles **content** hash (数值内容)
#   dataset  : v1 wheel_all.h5 的 **file** hash
#   feature  : v1 target_features.h5 的 **file** hash
# v1 阶段用的是 file hash; B1 改用 content hash 是因为 HDF5 容器带创建时间戳,
# 但**不得**因此把 v1 的冻结值换算成另一种口径去比 —— 那等于换了参照物。
V1_EXPECTED = {
    "v1_profile_content_sha256":
        "21aab0036e651239a31de724ab24d119d82ed8cd2786620506900068e8b0ebaf",
    "v1_dataset_file_sha256":
        "138b117a0b826aadd779ee5d64b872b413647f98322a7d8e7ef84d73c7b379c6",
    "v1_feature_file_sha256":
        "c59983d13187b1b0734be4321d20fca78566888f3e434dba61d6debf262a222a",
}
V1_FEATURE_H5 = ROOT / "data/features/wheel/basilisk_v1/target_features.h5"
V1_DATASET_H5 = (ROOT
                 / "data/sim/wheel_basilisk_v1/seed_20260808_basilisk/wheel_all.h5")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _contract() -> dict:
    if not CONTRACT.exists():
        pytest.skip("B1 baseline_contract.json 未生成 (先跑 verify_baseline.py)")
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_old_basilisk_v1_unchanged():
    """契约中全部文件的 sha256 必须与 B1 开工前冻结值一致。

    B1 没有 mutable group —— 连 `src/sim/wheel_sim.py` 都在不可变组里 (§4 要求
    保留现有退化结构, 标定只发生在调用侧), 所以这里是全量逐项比对。
    MISSING 也是契约的一部分: 冻结时不存在的文件后来出现同样是违约。
    """
    c = _contract()
    violations = []
    _n_hashed = _n_exempt = 0
    for group, files in c["groups"].items():
        for rel, expect in files.items():
            p = ROOT / rel
            if expect == "MISSING":
                if p.exists():
                    violations.append(f"{rel}: 冻结时不存在, 现已出现")
                continue
            if lifecycle.hash_exempt(rel):
                _n_exempt += 1
                continue
            if not p.exists():
                violations.append(f"{rel}: 冻结时存在, 现已丢失")
                continue
            got = _sha256(p)
            _n_hashed += 1
            if got != expect:
                violations.append(f"{rel}: {expect[:12]} -> {got[:12]}")
    assert not violations, \
        "B1_BASELINE_CONTRACT_VIOLATION:\n" + "\n".join(violations)
    # 防掏空 (B7 生命周期治理): 豁免只允许命中封闭白名单, 且实检项须占绝大多数。
    lifecycle.assert_registry_is_not_widened()
    assert _n_exempt <= 3, f"哈希豁免项过多 ({_n_exempt}) —— 疑似滥用"
    assert _n_hashed >= 20, f"逐字节实检项过少 ({_n_hashed}) —— 契约可能被掏空"



def test_contract_has_no_mutable_group():
    """B1 契约不得设置任何"允许改动"的豁免组 (与 v1 的区别)。"""
    c = _contract()
    assert "mutable_group" not in c or c.get("mutable_group") in (None, ""), \
        "B1 契约出现 mutable group —— B1 不允许改动任何既有文件"
    assert "allowed_minimal_edit" not in c["groups"], \
        "B1 契约含 allowed_minimal_edit 组 (应为空)"


def test_wheel_sim_is_in_immutable_group():
    """§4: wheel_sim.py 必须在不可变组里 —— 退化模型不得被 B1 改动。"""
    c = _contract()
    found = None
    for group, files in c["groups"].items():
        if "src/sim/wheel_sim.py" in files:
            found = group
    assert found is not None, "契约未覆盖 src/sim/wheel_sim.py"
    assert found == "frozen_code", \
        f"wheel_sim.py 被放进 {found} 组, 应在 frozen_code (不可变)"


def test_contract_covers_all_frozen_lineages():
    """契约必须覆盖 S2.5/S3/S4/S5/S5B + BASILISK-V1 全部关键产物。

    防止契约被悄悄缩小 —— 一个不检查旧结果的契约等于没有契约。
    """
    c = _contract()
    allrel = {r for files in c["groups"].values() for r in files}
    must = ["configs/wheel.yaml",
            "configs/wheel_basilisk.yaml",
            "data/features/wheel/schema_v1/target_features.h5",
            "checkpoints/s3_gate_metrics.json",
            "checkpoints/s4_metrics.json",
            "checkpoints/s5_rate_metrics.json",
            "checkpoints/s5b_wiener_pf_metrics.json",
            "docs/results.md",
            "src/sim/build_hi.py",
            "src/sim/wheel_sim.py",
            "src/sim/basilisk_bridge.py",
            "src/sim/basilisk_profile.py",
            "data/mission_profile/basilisk_v1/profiles.h5",
            "data/features/wheel/basilisk_v1/target_features.h5"]
    missing = [m for m in must if m not in allrel]
    assert not missing, f"契约漏掉冻结产物: {missing}"
    assert c["n_files"] >= 38, f"契约文件数 {c['n_files']} < 38 (被缩小)"


def test_v1_content_hashes_match_hardcoded():
    """BASILISK-V1 的 content hash 必须同时匹配契约记录与硬编码期望值。

    双源比对的意义: 若有人为了让契约通过而同时改了产物和契约文件, 硬编码值
    仍会暴露 —— 这两个 hash 是 BASILISK-V1 冻结声明的一部分。
    """
    c = _contract()
    rec = c.get("v1_content_hashes") or {}
    exp = c.get("v1_expected") or V1_EXPECTED
    for k, v in V1_EXPECTED.items():
        assert exp.get(k) == v, f"契约记录的 v1_expected[{k}] 被改动"
        if rec.get(k) in (None, "ABSENT"):
            pytest.skip(f"{k} 未在契约中实测记录")
        assert rec[k] == v, f"BASILISK-V1 {k} 变了: {v[:12]} -> {rec[k][:12]}"


def test_v1_feature_file_still_readable_and_intact():
    """直接重算 v1 两个 h5 的 **file** hash, 与冻结值比对 (不依赖契约文件)。

    这里必须用 file hash —— v1 冻结声明公布的就是 file hash。用 B1 的 content
    hash 口径去比会得到不同数字, 那不是"v1 变了", 而是换了参照物。
    """
    checked = 0
    for p, key in ((V1_FEATURE_H5, "v1_feature_file_sha256"),
                   (V1_DATASET_H5, "v1_dataset_file_sha256")):
        if not p.exists():
            continue
        checked += 1
        got = _sha256(p)
        assert got == V1_EXPECTED[key], \
            f"BASILISK-V1 {p.name} file hash 变了: {V1_EXPECTED[key][:12]} -> {got[:12]}"
    if checked == 0:
        pytest.skip("v1 h5 产物不存在")


def test_v1_feature_content_hash_stable_within_session(b1_build_features):
    """v1 特征文件的数值内容也必须稳定 —— 用 B1 的 content 口径自比一致性。

    这不与上一个测试重复: file hash 保证"字节没动", content hash 保证"数值可
    被稳定重算"。后者没有冻结期望值可比 (v1 阶段未公布 content hash), 所以只
    断言同一文件连算两次一致, 并把实测值留在失败信息里供人工登记。
    """
    bf = b1_build_features
    if not V1_FEATURE_H5.exists():
        pytest.skip("v1 target_features.h5 不存在")
    h1 = bf.feature_content_hash(V1_FEATURE_H5)
    h2 = bf.feature_content_hash(V1_FEATURE_H5)
    assert h1 == h2 and len(h1) == 64, f"content hash 不稳定: {h1[:16]} vs {h2[:16]}"


def test_b1_artifacts_are_all_new_paths():
    """B1 产物路径必须全部是新增的, 不与契约覆盖的任何文件重叠。"""
    c = _contract()
    frozen = {r for files in c["groups"].values() for r in files}
    for rel in ("data/sim/wheel_basilisk_b1", "data/features/wheel/basilisk_b1",
                "docs/basilisk_b1", "checkpoints/basilisk_b1"):
        d = ROOT / rel
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file():
                rp = p.relative_to(ROOT).as_posix()
                assert rp not in frozen, f"B1 产物 {rp} 与冻结契约文件冲突"


def test_v1_loader_untouched_by_b1():
    """§B1 设计决策: 共享的 v1 config loader 不得被 B1 改动。

    B1 需要递归解析两级 base_config, 但 src/utils/basilisk_config.load_basilisk_config
    被 v1 脚本共用且在冻结契约的影响半径内, 因此 B1 自带 load_b1_config, 而不是
    去改共享 loader。本测试钉死这个决策。
    """
    src = (ROOT / "src/utils/basilisk_config.py").read_text(encoding="utf-8")
    assert "b1" not in src.lower(), "共享 loader 被 B1 污染"
    cal_src = (ROOT / "scripts/basilisk_b1/calibrate_degradation.py").read_text(
        encoding="utf-8")
    assert "def load_b1_config" in cal_src, "B1 未自带递归 config loader"
