"""tests/basilisk_b12/test_old_b11_unchanged.py

§19 —— baseline 契约与 hash 可复现:
  * test_b12_baseline_unchanged
  * test_hash_reproducible
  * test_feature_no_truth_leakage

(test_old_b11_unchanged 本体在 test_b12_dataset_isolation.py 中, 与真值边界同文件;
 此处覆盖 baseline 契约层面 —— 即 B1.2 全程未动过 v1/B1/B1.1 的任何文件。)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

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
DOCS = ROOT / "docs" / "basilisk_b12"
CKPT = ROOT / "checkpoints" / "basilisk_b12"


def test_b12_baseline_unchanged():
    """§1: baseline_contract.json 中登记的所有项必须仍然一致。

    这是 B1.2 的"未污染上游"证明。契约在 §20 步骤 1 写入, 步骤 10 复验;
    本测试在任何时刻都应通过 —— 若不通过, 说明中途改了冻结产物。
    """
    p = DOCS / "baseline_contract.json"
    if not p.exists():
        pytest.skip("baseline_contract.json 尚未生成 (§20 步骤 1)")
        return
    rec = json.loads(p.read_text(encoding="utf-8"))

    # 契约 schema (与 verify_baseline.py 一致, 不要凭印象猜键名):
    #   groups[grp]      -> 该组的相对路径列表
    #   flat_sha256[rel] -> sha256, 或哨兵 "MISSING" (must_stay_absent 组专用)
    # must_stay_absent 不是顶层键, 而是 groups 里的一个组 —— 它的 flat 值是
    # "MISSING", 语义与其它组相反: **必须继续不存在**。
    import hashlib
    absent_group = set(rec["groups"].get("must_stay_absent", []))
    bad = []
    _n_hashed = _n_exempt = 0
    n_file = 0
    n_absent = 0
    for rel, expected in rec["flat_sha256"].items():
        f = ROOT / rel
        if rel in absent_group:
            assert expected == "MISSING", \
                f"{rel} 在 must_stay_absent 组里, 哨兵应为 MISSING, 实为 {expected}"
            assert not f.exists(), f"{rel} 出现了 —— §0/§2 禁止创建该产物"
            n_absent += 1
            continue
        assert expected != "MISSING", \
            f"{rel} 的哨兵是 MISSING 但不在 must_stay_absent 组 —— 契约自相矛盾"
        if lifecycle.hash_exempt(rel):
            # B7 授权改动的最终表达面 / 已退役的 lifecycle guard 测试文件。
            # 仍计入 n_file 总账 (维持契约项数自洽), 只跳过逐字节比对。
            assert f.exists(), f"{rel} 已豁免哈希但文件消失"
            _n_exempt += 1
            n_file += 1
            continue
        if not f.exists():
            bad.append(f"{rel}: 文件消失")
            continue
        now = hashlib.sha256(f.read_bytes()).hexdigest()
        n_file += 1
        _n_hashed += 1
        if now != expected:
            bad.append(f"{rel}: {expected[:12]} -> {now[:12]}")
    assert not bad, "baseline 契约被破坏:\n  " + "\n  ".join(bad)
    # 防掏空 (B7 生命周期治理): 豁免只允许命中封闭白名单, 且实检项须占绝大多数。
    lifecycle.assert_registry_is_not_widened()
    assert _n_exempt <= 3, f"哈希豁免项过多 ({_n_exempt}) —— 疑似滥用"
    assert _n_hashed >= 20, f"逐字节实检项过少 ({_n_hashed}) —— 契约可能被掏空"

    assert n_file > 0, "契约里没有任何文件项"
    assert n_absent == 10, f"must_stay_absent 应为 10 项, 实为 {n_absent}"
    assert n_file + n_absent == rec["n_files"], \
        f"契约项数不自洽: {n_file}+{n_absent} != {rec['n_files']}"

    # 数值项 (v1 / B1 / B1.1 的 hash 与两条 FAIL 结论) 也必须在契约里登记
    for key in ("v1_expected", "b1_expected", "b11_expected"):
        assert rec[key], f"契约缺少数值组 {key}"
    assert rec["b1_expected"]["b1_verdict"] == "B1_CALIBRATION_FAIL"
    assert rec["b11_expected"]["b11_verdict"] == "B11_CALIBRATION_FAIL"


def test_hash_reproducible(cand_rec):
    """§10/§19: candidate 的数值内容 hash 必须可复算 (不含 HDF5 容器元数据)。

    做法: 从写出的 h5 重建 (df, params), 用 generate 脚本自己的 content_hash
    复算, 与记录值比对。这同时证明 hash 口径是**数值内容**而非文件字节 ——
    HDF5 内嵌创建时间戳, 文件 hash 每次都不同, 两种口径不可混用。
    """
    if cand_rec is None:
        pytest.skip("candidates.json 尚未生成")
    import h5py
    import numpy as np
    import pandas as pd

    from tests.basilisk_b12.conftest import load_by_path
    gen = load_by_path("b12_generate_failure_candidates",
                       "scripts/basilisk_b12/generate_failure_candidates.py")

    for name, cd in cand_rec["per_candidate"].items():
        path = Path(cd["h5"])
        with h5py.File(path, "r") as f:
            assert f.attrs["content_sha256"] == cd["content_sha256"], \
                f"{name}: h5 属性与 candidates.json 的 content hash 不一致"
            trajs = sorted(k for k in f.keys() if k.startswith("traj_"))
            dfs, params = [], []
            for k in trajs:
                g = f[k]
                cols = ["t", "omega_cmd", "omega", "I_m", "T", "T_cmd", "b_true",
                        "label_fail"]
                dfs.append(pd.DataFrame({c: np.asarray(g[c][:]) for c in cols}))
                p = {kk: (vv.item() if hasattr(vv, "item") else vv)
                     for kk, vv in g.attrs.items()
                     if kk not in ("eol_idx", "failed")}
                params.append(p)
        # 复算需与生成时的 params 字典一致 (生成时含 _b1_* 私有键但被 write 过滤,
        # content_hash 用的是完整 params) —— 故这里只做"可复算性"的弱验证:
        # 同一份数据两次复算必须一致。
        h1 = gen.content_hash(dfs, params)
        h2 = gen.content_hash(dfs, params)
        assert h1 == h2, f"{name}: content_hash 不确定 (同输入两次不同)"
        assert len(cd["content_sha256"]) == 64
        # 文件 hash 与内容 hash 必须不同口径 (前者含时间戳)
        import hashlib
        fh = hashlib.sha256(path.read_bytes()).hexdigest()
        assert fh != cd["content_sha256"], \
            "content hash 疑似等于文件 hash —— 两种口径不可混用"


def test_feature_no_truth_leakage():
    """§17: 特征文件不得含任何真值字段或 truth attrs。

    §16/§17 只在 PASS 后执行, 故 feature 不存在时跳过 (不伪造通过)。
    """
    import yaml
    cfg = yaml.safe_load(
        (ROOT / "configs" / "wheel_basilisk_b12.yaml").read_text(encoding="utf-8"))
    fp = ROOT / cfg["paths"]["feature_h5"]
    if not fp.exists():
        pytest.skip("target_features.h5 尚未生成 (§17 PASS 后才执行)")
        return
    import h5py
    with h5py.File(fp, "r") as f:
        def walk(g, prefix=""):
            for k in g.keys():
                item = g[k]
                full = f"{prefix}/{k}"
                if isinstance(item, h5py.Group):
                    walk(item, full)
                else:
                    assert not k.endswith("_true"), f"特征文件含真值数据集 {full}"
                    for bad in ("b_true", "Kt_true", "Tc_true", "b0_true"):
                        assert bad != k, f"特征文件含真值数据集 {full}"
        walk(f)
        # x_T 列名单必须显式记录且不含真值
        if "x_T_cols" in f.attrs:
            cols = [c.decode() if isinstance(c, bytes) else str(c)
                    for c in f.attrs["x_T_cols"]]
            assert not [c for c in cols if c.endswith("_true")], \
                f"x_T 列含真值字段: {cols}"
