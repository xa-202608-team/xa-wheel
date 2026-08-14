"""tests/basilisk_b11/test_b11_baseline_and_protocol.py

B1.1 §18 —— 前两组硬测试:
  * `test_b11_old_b1_unchanged`     : §1 baseline contract 自动化守卫 (含 B1 产物)
  * `test_b11_protocol_frozen`      : §2 protocol.md 在生成任何 candidate **之前**
                                      冻结, 且此后 hash 不变

契约里的 MISSING 也是契约的一部分: 冻结时不存在的文件后来出现同样是违约。
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

CONTRACT = ROOT / "docs/basilisk_b11/baseline_contract.json"
PROTOCOL = ROOT / "docs/basilisk_b11/protocol.md"
PROTOCOL_REC = ROOT / "checkpoints/basilisk_b11/protocol_hash.json"

# BASILISK-V1 冻结时公布的三个 hash (硬编码, 防契约文件本身被整体替换后仍"自洽")。
# 口径差异: profile 是 content hash, dataset/feature 是 **file** hash —— v1 阶段
# 公布的就是 file hash, 不得换算成 content 口径去比 (那等于换了参照物)。
V1_EXPECTED = {
    "v1_profile_content_sha256":
        "21aab0036e651239a31de724ab24d119d82ed8cd2786620506900068e8b0ebaf",
    "v1_dataset_file_sha256":
        "138b117a0b826aadd779ee5d64b872b413647f98322a7d8e7ef84d73c7b379c6",
    "v1_feature_file_sha256":
        "c59983d13187b1b0734be4321d20fca78566888f3e434dba61d6debf262a222a",
}
# B1 冻结结论 (B1.1 开工前的既有事实, 不得被"修好")
B1_EXPECTED = {
    "b1_candidate_A_content_sha256":
        "9e84de4084339cfa057a959617e73dd836d78b03f98e0117f7b60b0564250fda",
    "b1_candidate_B_content_sha256":
        "f1cc45014158f0497632c048da51db6d185e3d98a85c3774aeea154b9b046025",
    "b1_candidate_C_content_sha256":
        "97394e6501a5dd07206f115dca91345358c467dcaa1b17fd98e57bc58de8c2f0",
    "b1_verdict": "B1_CALIBRATION_FAIL",
    "b1_selected_candidate": "None",
    "b1_b0_scale": "7.225772358824458",
    "b1_speed_util_ref_rms": "0.33039016838980756",
}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _contract() -> dict:
    if not CONTRACT.exists():
        pytest.skip("B1.1 baseline_contract.json 未生成 (先跑 verify_baseline.py)")
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_b11_old_b1_unchanged():
    """契约中全部文件 (含 B1 全部产物) 的 sha256 必须与 B1.1 开工前一致。

    B1.1 没有 mutable group —— 连 `src/sim/wheel_sim.py`、
    `configs/wheel_basilisk_b1.yaml` 都在不可变组里 (§3 只改标定口径,
    不改退化结构)。
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
        "B11_BASELINE_CONTRACT_VIOLATION:\n" + "\n".join(violations)
    # 防掏空 (B7 生命周期治理): 豁免只允许命中封闭白名单, 且实检项须占绝大多数。
    lifecycle.assert_registry_is_not_widened()
    assert _n_exempt <= 3, f"哈希豁免项过多 ({_n_exempt}) —— 疑似滥用"
    assert _n_hashed >= 20, f"逐字节实检项过少 ({_n_hashed}) —— 契约可能被掏空"



def test_b11_contract_covers_b1_and_v1():
    """契约必须同时覆盖 analytic / BASILISK-V1 / B1 三个 lineage 的关键产物。

    一个不检查 B1 旧结论的契约等于没有契约 —— B1.1 的整个正当性建立在
    "B1 的失败结论保持原样, 只新增 B1.1 命名空间"之上。
    """
    c = _contract()
    allrel = {r for files in c["groups"].values() for r in files}
    must = ["configs/wheel.yaml", "configs/wheel_basilisk.yaml",
            "configs/wheel_basilisk_b1.yaml",
            "data/features/wheel/schema_v1/target_features.h5",
            "checkpoints/s3_gate_metrics.json",
            "checkpoints/s4_metrics.json",
            "checkpoints/s5_rate_metrics.json",
            "checkpoints/s5b_wiener_pf_metrics.json",
            "docs/results.md",
            "src/sim/build_hi.py", "src/sim/wheel_sim.py",
            "src/sim/basilisk_bridge.py",
            "data/mission_profile/basilisk_v1/profiles.h5",
            "data/features/wheel/basilisk_v1/target_features.h5",
            "docs/basilisk_b1/candidate_report.md",
            "checkpoints/basilisk_b1/candidates.json",
            "checkpoints/basilisk_b1/candidate_audit.json",
            "scripts/basilisk_b1/calibrate_degradation.py"]
    missing = [m for m in must if m not in allrel]
    assert not missing, f"契约漏掉冻结产物: {missing}"


def test_b11_contract_has_no_mutable_group():
    """B1.1 契约不得设置任何"允许改动"的豁免组。"""
    c = _contract()
    assert c.get("mutable_group") in (None, ""), \
        "B1.1 契约出现 mutable group —— 不允许改动任何既有文件"
    assert "allowed_minimal_edit" not in c["groups"]


def test_b11_wheel_sim_in_immutable_group():
    """§2: wheel_sim.py 必须在不可变组 —— 退化模型结构不得被 B1.1 改动。"""
    c = _contract()
    found = [g for g, files in c["groups"].items()
             if "src/sim/wheel_sim.py" in files]
    assert found == ["frozen_code"], f"wheel_sim.py 在 {found}, 应仅在 frozen_code"


def test_b11_records_b1_verdict_and_hashes():
    """契约必须记录 B1 的三个 candidate hash 与 B1_CALIBRATION_FAIL 结论。"""
    c = _contract()
    num = c.get("b1_results") or {}
    exp = c.get("b1_expected") or B1_EXPECTED
    for k, v in B1_EXPECTED.items():
        assert exp.get(k) == v, f"契约记录的 b1_expected[{k}] 被改动: {exp.get(k)}"
        if num.get(k) in (None, "ABSENT"):
            pytest.skip(f"{k} 未在契约中实测记录")
        assert num[k] == v, f"B1 {k} 变了: {v} -> {num[k]}"


def test_b11_records_v1_hashes():
    """契约必须同时记录并匹配 BASILISK-V1 的三个冻结 hash。"""
    c = _contract()
    num = c.get("v1_content_hashes") or {}
    exp = c.get("v1_expected") or V1_EXPECTED
    for k, v in V1_EXPECTED.items():
        assert exp.get(k) == v, f"契约记录的 v1_expected[{k}] 被改动"
        if num.get(k) in (None, "ABSENT"):
            pytest.skip(f"{k} 未实测记录")
        assert num[k] == v, f"v1 {k} 变了: {v[:12]} -> {num[k][:12]}"


def test_b11_artifacts_are_all_new_paths():
    """B1.1 产物路径必须全部新增, 不与契约覆盖的任何文件重叠。"""
    c = _contract()
    frozen = {r for files in c["groups"].values() for r in files}
    for rel in ("data/sim/wheel_basilisk_b11", "data/features/wheel/basilisk_b11",
                "docs/basilisk_b11", "checkpoints/basilisk_b11",
                "scripts/basilisk_b11", "tests/basilisk_b11"):
        d = ROOT / rel
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file():
                rp = p.relative_to(ROOT).as_posix()
                assert rp not in frozen, f"B1.1 产物 {rp} 与冻结契约文件冲突"


def test_b11_does_not_write_into_b1_or_v1_dirs():
    """B1.1 的输出路径配置必须全部落在 b11 命名空间。"""
    import yaml
    cfg = yaml.safe_load(
        (ROOT / "configs/wheel_basilisk_b11.yaml").read_text(encoding="utf-8"))
    for key in ("sim_dir", "candidate_dir", "feature_h5", "ckpt_dir", "doc_dir"):
        v = str(cfg["paths"][key])
        assert "b11" in v, f"paths.{key} = {v} 不在 b11 命名空间"
        for bad in ("basilisk_v1", "wheel_basilisk_b1/", "basilisk_b1/"):
            assert bad not in v, f"paths.{key} = {v} 指向被冻结的 {bad}"


# --------------------------------------------------------------------------
# §2 protocol frozen
# --------------------------------------------------------------------------

def test_b11_protocol_frozen():
    """protocol.md 的 SHA256 必须与 protocol_hash.json 记录一致。"""
    if not PROTOCOL.exists():
        pytest.skip("protocol.md 未生成")
    if not PROTOCOL_REC.exists():
        pytest.skip("protocol_hash.json 未生成 (先跑 audit_peak_reference.py)")
    rec = json.loads(PROTOCOL_REC.read_text(encoding="utf-8"))
    got = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert got == rec["protocol_sha256"], \
        (f"protocol.md 在冻结后被修改: {rec['protocol_sha256'][:16]} -> {got[:16]}"
         " —— §2 明令冻结后不得修改协议")


def test_b11_protocol_frozen_before_candidates():
    """§2 时序: protocol.md 的 mtime 必须早于任何 candidate 产物。

    "先冻结协议, 再生成 candidate"是本阶段最重要的防事后合理化机制;
    只比对 hash 不足以证明时序。
    """
    if not PROTOCOL.exists():
        pytest.skip("protocol.md 未生成")
    cand_json = ROOT / "checkpoints/basilisk_b11/candidates.json"
    if not cand_json.exists():
        pytest.skip("candidates.json 未生成")
    assert PROTOCOL.stat().st_mtime <= cand_json.stat().st_mtime, \
        "protocol.md 比 candidates.json 更新 —— 协议疑似在看到结果后被改写"


def test_b11_protocol_declares_forbidden_changes():
    """protocol.md 必须明文登记本阶段的禁止项 (可被人工与自动双重核对)。"""
    if not PROTOCOL.exists():
        pytest.skip("protocol.md 未生成")
    txt = PROTOCOL.read_text(encoding="utf-8")
    for token in ("g_duty", "Im_rated", "failure criterion", "candidate",
                  "RUL", "5.1"):
        assert token in txt, f"protocol.md 未登记 {token} 相关约定"
    assert "3" in txt, "protocol.md 未登记 candidate 上限"


def test_b11_candidate_registry_recorded_in_protocol():
    """三个 candidate 名必须在协议里预先登记 (而非事后补写)。"""
    if not PROTOCOL.exists():
        pytest.skip("protocol.md 未生成")
    txt = PROTOCOL.read_text(encoding="utf-8")
    for name in ("P95_UTILIZATION_ONLY", "P95_UTILIZATION_DURATION_1P5",
                 "P95_UTILIZATION_DURATION_2P0"):
        assert name in txt, f"protocol.md 未预先登记 candidate {name}"
