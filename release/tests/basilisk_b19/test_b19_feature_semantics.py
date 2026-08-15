"""B1.9 §6/§7: 特征语义 —— 主/辅 HI 分工、x_T 不变、监督标签结构。

含 §10 要求的以下函数名 (字节一致):
  test_feature_hash_reproducible
"""
from __future__ import annotations

import hashlib
import json

import h5py
import numpy as np
import pytest

from conftest import CK, DOCS, ROOT


def _content_hash(path):
    """与脚本侧同口径的内容 hash, 测试侧独立实现 (不 import 脚本函数)。"""
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        for key in sorted(k for k in f.keys() if k.startswith("traj_")):
            h.update(key.encode())
            g = f[key]
            for name in sorted(g.keys()):
                if name == "mission_features":
                    continue
                h.update(name.encode())
                a = np.ascontiguousarray(g[name][:], dtype=np.float64)
                h.update(str(a.shape).encode())
                h.update(np.nan_to_num(a, nan=-1.0e300).tobytes())
    return h.hexdigest()


def test_feature_hash_reproducible(b19_feature_h5, b19_feature_stats,
                                   b19_feature_audit):
    """§8 Gate 12: 内容哈希可复现, 且三方 (build/audit/测试) 口径一致。"""
    h1 = _content_hash(b19_feature_h5)
    h2 = _content_hash(b19_feature_h5)
    assert h1 == h2
    assert h1 == b19_feature_stats["feature_content_sha256"], "与 build 期不一致"
    assert h1 == b19_feature_audit["feature_content_sha256"], "与复核不一致"


def test_primary_and_auxiliary_hi_both_present(b19_feature_h5):
    """§6: 主监督 HI 换成损伤代理, 旧 friction HI 保留为辅助, 二者都在。"""
    with h5py.File(b19_feature_h5, "r") as f:
        assert str(f.attrs["primary_hi"]) == "hi_damage_obs"
        assert str(f.attrs["auxiliary_hi_friction"]) == "hi_friction"
        for key in list(k for k in f.keys() if k.startswith("traj_"))[:5]:
            g = f[key]
            assert "hi_damage_obs" in g and "hi_friction" in g
            n = g["x_T"].shape[0]
            assert g["hi_damage_obs"].shape == (n,)
            assert g["hi_friction"].shape == (n,)


def test_primary_and_auxiliary_hi_are_different_signals(b19_feature_h5):
    """辅助 HI 不能只是主 HI 的副本 —— 否则 §6 的机制对比是空的。"""
    with h5py.File(b19_feature_h5, "r") as f:
        key = next(k for k in f.keys() if k.startswith("traj_"))
        a = np.asarray(f[key]["hi_damage_obs"][:], dtype=np.float64)
        b = np.asarray(f[key]["hi_friction"][:], dtype=np.float64)
    assert np.abs(a - b).max() > 1e-6, "主/辅 HI 数值相同 —— 机制对比无意义"


def test_xt_schema_unchanged_from_b18(b19_feature_h5, b19_feature_stats):
    """§7: 核心 10 列语义与位置不变, 且与 B1.8 逐位相同 (只换监督信号)。"""
    expect = ["I_m", "omega", "T", "T_cmd", "sigma_Im", "b_hat", "dT",
              "omega_err", "Tf_ratio", "Tf_slope"]
    with h5py.File(b19_feature_h5, "r") as f:
        cols = json.loads(str(f.attrs["xt_cols"]))
    assert cols == expect, cols
    assert b19_feature_stats["xt_identical_to_b18"] is True
    assert b19_feature_stats["xt_max_abs_diff_vs_b18"] == 0.0


def test_hdf5_layout_matches_protocol(b19_feature_h5):
    """§7: 规定的各 dataset 分开保存, 不合并成一个大矩阵。"""
    need = ("x_T", "hi_damage_obs", "hi_friction", "rul", "rul_lower_bound")
    with h5py.File(b19_feature_h5, "r") as f:
        for key in list(k for k in f.keys() if k.startswith("traj_"))[:5]:
            g = f[key]
            for d in need:
                assert d in g, f"{key} 缺 {d}"
            assert "event_observed" in g.attrs
            assert "mission_features" in g


def test_event_hi_reaches_failure_scale(b19_feature_audit, b19_config):
    """§8 Gate 7/8: event-observed 在 EOL 处 HI 必须接近 1 (与 D>=1 同源)。

    门限在 protocol 冻结时即已确定 (0.90 / 0.75), 不得事后调低。
    """
    fg = b19_config["feature_gate"]
    assert float(fg["event_eol_median_hi_min"]) == 0.90
    assert float(fg["event_eol_p10_hi_min"]) == 0.75
    he = b19_feature_audit["hi_event"]
    assert he["median"] >= 0.90
    assert he["p10"] >= 0.75


def test_censored_hi_spread_is_informative(b19_feature_audit):
    """删失组 HI 必须有分布 (不是常数) —— 否则监督信号对删失无信息。"""
    hc = b19_feature_audit["hi_censored"]
    assert hc["n"] > 0
    assert hc["max"] > hc["min"] + 0.05, f"删失组 HI 近乎常数: {hc}"
    assert hc["max"] < 1.0, "删失组出现 HI=1, 与 Gate 9 矛盾"


def test_l_ref_is_normalization_denominator(b19_feature_h5, b19_frozen):
    """§8 Gate 6: 归一化分母是常量 L_ref, 不是 EOL/末值/未来 max。"""
    with h5py.File(b19_feature_h5, "r") as f:
        assert float(f.attrs["L_ref_years"]) == 3.0
    s = b19_frozen["selected_primary_hi"]
    assert s["L_ref_years"] == 3.0
    assert s["normalization_denominator"] == "L_ref (constant)"
    assert s["cumulative_direction"] == "chronological_only"


def test_failure_definition_unchanged(b19_feature_h5):
    """§10: 不改 failure definition —— 仍是 D >= 1 的工程假设, 非厂家规格。"""
    with h5py.File(b19_feature_h5, "r") as f:
        assert str(f.attrs["eol_rule"]) == "D >= 1"
        assert str(f.attrs["failure_definition_semantics"]) == \
            "engineering_assumption_failure_definition"
        assert bool(f.attrs["is_manufacturer_failure_specification"]) is False
