"""tests/basilisk_b13/test_b13_baseline_contract.py

§1/§19 —— 前四阶段产物在 B1.3 期间**逐字未变**。

包含 §19 命名测试:
  * test_b13_old_artifacts_unchanged
  * test_final_dataset_isolated (数据/特征只落在 b13 命名空间)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import CKPT, DOCS, ROOT, read_json  # noqa: F401

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

# B1.3 **唯一**允许写入的路径前缀 (§0 隔离命名空间)
ALLOWED_WRITE_PREFIXES = (
    "configs/wheel_basilisk_b13.yaml",
    "scripts/basilisk_b13/",
    "tests/basilisk_b13/",
    "checkpoints/basilisk_b13/",
    "data/sim/wheel_basilisk_b13/",
    "data/features/wheel/basilisk_b13/",
    "docs/basilisk_b13/",
    "STATUS_BASILISK_B13.md",
)

# §0 明令不得修改的既有资产
MUST_NOT_TOUCH = (
    "configs/wheel.yaml",
    "configs/wheel_basilisk.yaml",
    "configs/wheel_basilisk_b11.yaml",
    "configs/wheel_basilisk_b12.yaml",
    "src/sim/wheel_sim.py",
)


def test_b13_old_artifacts_unchanged(contract_rec, b13_verify):
    """§1: baseline contract 登记的每个文件 hash 与每个数值项**当下仍然一致**。

    注意: `baseline_contract.json` 里**没有** `n_ok` / `n_fail` 字段 (它只存
    `flat_sha256` + 各阶段数值块)。所以这里必须真的重新算一遍 hash 去比,
    而不是读一个"通过计数" —— 读计数只能证明写契约那一刻是好的。
    """
    if contract_rec is None:
        pytest.skip("baseline_contract.json 未生成 (需先跑 verify_baseline.py --tag before)")
    _n_hashed = _n_exempt = 0

    bad = []
    for rel, h_ref in contract_rec["flat_sha256"].items():
        if lifecycle.hash_exempt(rel):
            _n_exempt += 1
            continue
        h_now = b13_verify.sha256_of(rel)
        _n_hashed += 1
        if h_now != h_ref:
            bad.append((rel, h_ref[:16], str(h_now)[:16]))
    assert not bad, f"冻结文件被改动: {bad[:10]}"

    # 防掏空 (B7 生命周期治理): 豁免只允许命中封闭白名单, 且实检项须占绝大多数。
    lifecycle.assert_registry_is_not_widened()
    assert _n_exempt <= 3, f"哈希豁免项过多 ({_n_exempt}) —— 疑似滥用"
    assert _n_hashed >= 20, f"逐字节实检项过少 ({_n_hashed}) —— 契约可能被掏空"

    n_numeric = 0
    for name, reader, expected, prefix in b13_verify.NUMERIC_BLOCKS:
        now = reader()
        for k, h_ref in contract_rec[name].items():
            n_numeric += 1
            assert now.get(k) == h_ref, \
                f"[{prefix}_result] {k}: 契约 {h_ref} -> 现在 {now.get(k, 'ABSENT')}"
        for k, h_exp in expected.items():
            n_numeric += 1
            assert now.get(k) == h_exp, \
                f"[{prefix}_expected] {k}: 期望 {h_exp} -> 现在 {now.get(k, 'ABSENT')}"

    n_total = contract_rec["n_files"] + n_numeric
    assert contract_rec["n_files"] >= 116, \
        f"契约文件数退化到 {contract_rec['n_files']} (应 >= 116) —— 是否有 group 被删?"
    assert n_total >= 190, f"契约总项数 {n_total} < 190"


def test_b13_contract_covers_all_prior_stages(b13_verify):
    """契约必须同时覆盖 v1 / B1 / B1.1 / B1.2 四个阶段, 不能只挑一个。"""
    groups = b13_verify.FROZEN_GROUPS
    for g in ("basilisk_v1_artifacts", "basilisk_b1_artifacts",
              "basilisk_b11_artifacts", "basilisk_b12_artifacts",
              "must_stay_absent", "frozen_code"):
        assert g in groups, f"契约缺 group {g}"
    names = [b for _, _, _, b in b13_verify.NUMERIC_BLOCKS]
    assert set(names) == {"v1", "b1", "b11", "b12"}, \
        f"数值复核块不全: {names}"


def test_b13_must_not_touch_files_are_in_contract(b13_verify):
    """§0 的 must-not-touch 清单必须真的在契约里被 hash, 否则"未修改"是空口承诺。"""
    flat = {f for v in b13_verify.FROZEN_GROUPS.values() for f in v}
    for rel in MUST_NOT_TOUCH:
        assert rel in flat, f"{rel} 不在 baseline 契约里 —— 无法证明它未被改动"


def test_b13_prior_stage_verdicts_frozen(b13_verify):
    """前阶段的**结论**也必须被钉住 —— 不能靠"重解释"旧结果给本阶段开路。

    键名带 `b12_` 前缀 (实测自 B12_EXPECTED, 勿凭印象去猜裸 `verdict`)。
    """
    e = b13_verify.B12_EXPECTED
    assert e["b12_verdict"] == "B12_FAILURE_DEFINITION_FAIL"
    assert e["b12_selected_candidate"] == "None", \
        "B1.2 没有选中任何 candidate —— 这条结论不得被改写"
    assert e["b12_electrical_verdict"] == "PROVENANCE_INSUFFICIENT"
    assert e["b12_f1_eligible"] == "False", \
        "F1 (电流判据) 已被 provenance 否决, 不得在 B1.3 复活"


def test_final_dataset_isolated(probe_rec, dataset_rec, feature_rec, b13_cfg):
    """§0/§14/§16: 所有 B1.3 产物路径都落在 b13 命名空间内。"""
    p = b13_cfg["paths"]
    for k in ("sim_dir", "probe_dir", "final_dir", "feature_h5", "ckpt_dir",
              "doc_dir"):
        v = str(p[k]).replace("\\", "/")
        assert "b13" in v, f"paths.{k} = {v} 不在 b13 命名空间"
        # 前缀表里的目录项带尾斜杠, 而 config 里的目录值不带 (如
        # `data/sim/wheel_basilisk_b13`)。补一个尾斜杠再比, 否则目录项永远不匹配。
        assert (v + "/").startswith(ALLOWED_WRITE_PREFIXES), \
            f"paths.{k} = {v} 越出 §0 允许的写入前缀"
    # 不得写进前阶段目录
    for rec, name in ((probe_rec, "probe"), (dataset_rec, "dataset"),
                      (feature_rec, "feature")):
        if rec is None:
            continue
        blob = json.dumps(rec, ensure_ascii=False)
        for bad in ("basilisk_b12/", "basilisk_b11/", "wheel_basilisk_b12",
                    "wheel_basilisk_b11"):
            # 允许**读取**前阶段 (provenance 字符串), 但不允许输出路径指向它们
            for key in ("out_h5", "feature_h5", "sim_dir", "final_dir",
                        "probe_dir"):
                v = str(rec.get(key, ""))
                assert bad not in v, f"{name}.{key} = {v} 指向前阶段目录"
        assert blob  # 保持显式


def test_b13_no_stray_files_outside_namespace():
    """B1.3 不得在前阶段目录里新建文件 (抽查 must_stay_absent 之外的常见误落点)。"""
    strays = []
    for d in ("checkpoints/basilisk_b12", "checkpoints/basilisk_b11",
              "checkpoints/basilisk_b1", "docs/basilisk_b12",
              "docs/basilisk_b11"):
        p = ROOT / d
        if not p.exists():
            continue
        for f in p.iterdir():
            if "b13" in f.name.lower():
                strays.append(str(f.relative_to(ROOT)).replace("\\", "/"))
    assert not strays, f"B1.3 把文件写进了前阶段目录: {strays}"
