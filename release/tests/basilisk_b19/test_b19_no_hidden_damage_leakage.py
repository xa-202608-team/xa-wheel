"""B1.9 §5/§7: hidden 真值不得进入特征或输入。

含 §10 要求的以下函数名 (字节一致):
  test_xt_does_not_include_damage_proxy
  test_truth_never_enters_xt
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from conftest import CK, DOCS, ROOT, SCRIPTS

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALLOWED_DATASETS = {"x_T", "hi_damage_obs", "hi_friction", "hi_a", "b_hat",
                    "b_true", "rul", "rul_lower_bound", "label_fail",
                    "mission_features"}


def test_xt_does_not_include_damage_proxy(b19_feature_h5, b19_feature_stats):
    """§7: hi_damage_obs 是 target/auxiliary, 不得作为普通输入列进 x_T。

    若它进了 x_T, Target-only 可以直接对这条 HI 积分外推出 RUL, 形成评价循环 ——
    模型看似学会预测, 实际只是在复述监督信号的构造式。
    """
    with h5py.File(b19_feature_h5, "r") as f:
        cols = json.loads(str(f.attrs["xt_cols"]))
        assert bool(f.attrs["damage_proxy_in_xT"]) is False
        assert bool(f.attrs["mission_features_in_xT"]) is False
        primary = str(f.attrs["primary_hi"])
        assert primary == "hi_damage_obs"
        assert primary not in cols
        for c in cols:
            low = c.lower()
            assert "damage" not in low and low != "d" and "hi" not in low, c
        # 形状证据: HI 是独立 (N,) dataset, 不是 x_T 的第 11 列
        key = next(k for k in f.keys() if k.startswith("traj_"))
        assert f[key]["x_T"].shape[1] == 10
        assert f[key][primary].ndim == 1
    assert b19_feature_stats["damage_proxy_in_xT"] is False


def test_truth_never_enters_xt(b19_feature_h5, b19_config):
    """§7/§8 Gate 13: x_T 不含任何真值列; 输出无 truth 组与真值 attrs。"""
    forb_cols = tuple(b19_config["feature_gate"]["forbidden_xt_cols"])
    forb_attr = tuple(b19_config["feature_gate"]["forbidden_truth_attrs"])
    with h5py.File(b19_feature_h5, "r") as f:
        cols = json.loads(str(f.attrs["xt_cols"]))
        assert not [c for c in cols if c in forb_cols]
        low = [c.lower() for c in cols]
        for bad in ("b_true", "d", "eol", "rul", "kt", "tc", "b0", "omega0",
                    "delta", "tau_years", "seed_traj"):
            assert bad not in low, f"x_T 出现真值列 {bad}"
        for key in (k for k in f.keys() if k.startswith("traj_")):
            assert "truth" not in f[key], f"{key} 残留 truth 组"
            for name in f[key].keys():
                assert name in ALLOWED_DATASETS, f"{key}/{name} 非 allowlist"
            for at in f[key].attrs:
                assert at not in forb_attr, f"{key}.{at} 是真值 attr"

    # 结构性保证: build_features 签名不含 params
    from src.sim.build_hi import build_features
    assert "params" not in list(inspect.signature(build_features).parameters)


def test_build_features_does_not_read_truth_group():
    """构建脚本读源 h5 时只取遥测列 + b_true/label_fail, 不读 truth 组。"""
    src = (SCRIPTS / "build_features.py").read_text(encoding="utf-8")
    # 允许在 IMPLEMENTATION_AUDIT 段落读 truth/D, 但必须与 pd.DataFrame 构造分离
    i = src.index("pd.DataFrame(")
    j = src.index("gg.create_dataset(\"x_T\"")
    window = src[i:j]
    assert 'g["truth"]' not in window, "特征计算窗口内出现 truth 组访问"
    assert "TELEMETRY_COLS" in window


def test_censored_not_forced_to_one(b19_feature_h5, b19_feature_audit):
    """§8 Gate 9: 右删失轨迹不得被强制 HI = 1, rul 必须为 NaN 不伪造 EOL。"""
    n_at_one = 0
    n_cens = 0
    with h5py.File(b19_feature_h5, "r") as f:
        for key in (k for k in f.keys() if k.startswith("traj_")):
            g = f[key]
            if int(g.attrs["event_observed"]):
                continue
            n_cens += 1
            hi = np.asarray(g["hi_damage_obs"][:], dtype=np.float64)
            rul = np.asarray(g["rul"][:], dtype=np.float64)
            n_at_one += int(hi[-1] >= 1.0)
            assert np.all(np.isnan(rul)), f"{key}: 删失轨迹的 rul 被伪造"
            lb = np.asarray(g["rul_lower_bound"][:], dtype=np.float64)
            assert np.all(np.isfinite(lb)), f"{key}: 缺 rul_lower_bound"
    assert n_cens > 0
    assert n_at_one == 0, f"{n_at_one}/{n_cens} 条删失轨迹被强制 HI=1"
    assert b19_feature_audit["hi_censored"]["n_at_one"] == 0
    assert b19_feature_audit["censored_rul_all_nan"] is True


def test_implementation_audit_is_labelled_audit_only(b19_feature_audit,
                                                    b19_damage_proxy):
    """§5: 与 hidden 真值的比较必须挂 IMPLEMENTATION_AUDIT 标签, 不得混作特征来源。"""
    assert b19_feature_audit["implementation_audit"]["purpose"] == "IMPLEMENTATION_AUDIT"
    assert b19_damage_proxy["stats"]["implementation_audit"]["purpose"] == \
        "IMPLEMENTATION_AUDIT"
    # 该比较结果不得出现在特征文件的 dataset 名里
    for bad in ("corr", "rmse", "audit", "d_true"):
        assert bad not in ALLOWED_DATASETS


def test_recomputation_disclosed_not_celebrated(b19_feature_audit, b19_frozen):
    """诚实性: corr≈1 必须被声明为 recomputation 的必然结果, 不得当能力证据。"""
    assert b19_feature_audit["is_recomputation_not_estimation"] is True
    assert b19_frozen["selected_primary_hi"]["is_recomputation_not_estimation"] is True
    disc = b19_feature_audit["recomputation_disclosure"]
    assert "recomputation" in disc and "estimation" in disc
    for doc in ("limitations.md", "feature_report.md", "selected_hi_definition.md",
                "damage_proxy_derivation.md", "protocol.md"):
        txt = (DOCS / doc).read_text(encoding="utf-8")
        assert "recomputation" in txt, f"{doc} 未声明 recomputation"


def test_b18_artifacts_untouched_by_feature_build(b19_feature_stats):
    """§6: 构建过程不得改动 B1.8 特征文件。"""
    assert b19_feature_stats["b18_feature_sha256_before"] == \
        b19_feature_stats["b18_feature_sha256_after"]
    assert b19_feature_stats["feature_h5"] != \
        "data/features/wheel/basilisk_b18/target_features.h5"
