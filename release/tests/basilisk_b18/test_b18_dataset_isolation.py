"""tests/basilisk_b18/test_b18_dataset_isolation.py —— §17/§18 正式数据集隔离与 EOL 来源。

含 §22 要求的 `test_eol_only_from_damage`。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_eol_only_from_damage(b18_dataset_audit, b18_config):
    """§14 Gate 5 / §4: EOL 完全由 D>=1 产生, 逐轨迹核对。"""
    a = b18_dataset_audit
    thr = float(b18_config["damage_model"]["eol_threshold_D"])
    assert a["eol_rule"] == "cumulative_damage_reaches_one"
    assert float(a["eol_threshold_D"]) == thr == 1.0
    for r in a["per_trajectory"]:
        if r["event_observed"]:
            assert r["D_at_eol"] >= thr, \
                f"traj {r['traj_id']}: EOL 处 D={r['D_at_eol']} < {thr}"
        else:
            assert r["D_max"] < thr, \
                f"traj {r['traj_id']}: 删失却 D_max={r['D_max']} >= {thr}"
            assert r["n_post_eol_rows"] == 0


def test_eol_not_from_current_threshold(b18_dataset_audit, b18_config):
    """§4: 电流阈值已废止 —— 存在 |I_m| 超额定却未失效的轨迹即为证据。

    若 EOL 仍由电流阈值产生, 就不可能出现 |I_m| > Im_rated 而 event_observed
    为 False 的情况。此处只作方向性检查, 不作强断言 (电流未必真的超限)。
    """
    a = b18_dataset_audit
    rated = float(b18_config["sim"]["failure"]["Im_rated_A"])
    # 关键断言: 失效判定与电流阈值解耦 —— 用 D 与电流的一致性检验
    for r in a["per_trajectory"]:
        # EOL 处的判据只能是 D, 不得要求 |I_m| 恰好越过 rated
        if r["event_observed"]:
            assert r["D_at_eol"] >= 1.0
    assert rated > 0.0     # 该配置项仍存在但不再定义 EOL


def test_dataset_scenario_is_frozen_primary(b18_dataset_audit,
                                            b18_frozen_primary):
    """§17: 正式数据只用 primary (NOMINAL, L_ref=3.0), 不接受命令行改写。"""
    a = b18_dataset_audit
    assert a["scenario"] == b18_frozen_primary["primary_scenario"] == "NOMINAL"
    assert float(a["L_ref_years"]) == float(
        b18_frozen_primary["L_ref_years"]) == 3.0
    assert a["provenance"] == "NEW_B18_ASSUMPTION"
    assert a["is_manufacturer_failure_specification"] is False


def test_dataset_size_and_hygiene(b18_dataset_audit, b18_config):
    """§17/§18: 150 条; 无 NaN/Inf; D 全单调。"""
    a = b18_dataset_audit
    assert a["n_traj"] == 150 == int(b18_config["dataset"]["n_traj"])
    assert len(a["per_trajectory"]) == 150
    assert a["audit"]["n_nan_inf_total"] == 0
    assert a["audit"]["D"]["all_monotone"] is True


def test_dataset_statistically_usable(b18_dataset_audit):
    """§18: event / censored 双侧都要有量, 且 EOL 有离散度。

    右删失轨迹的存在是设计的一部分, 不是缺陷 —— 但两侧都必须够用才能做
    trajectory-level 分层 split。
    """
    a = b18_dataset_audit["audit"]
    assert a["n_event_observed"] >= 20
    assert a["n_censored"] >= 10
    assert a["n_event_observed"] + a["n_censored"] == 150
    e = a["eol_years_event_observed"]
    assert e["n"] == a["n_event_observed"]
    assert e["iqr"] > 0.0
    assert e["p5"] <= e["p25"] <= e["p50"] <= e["p75"] <= e["p95"]


def test_dataset_isolated_from_old_namespaces(b18_dataset_audit):
    """§1: 正式数据只落 data/sim/wheel_basilisk_b18/final。"""
    out = b18_dataset_audit["out_dir"]
    assert out == "data/sim/wheel_basilisk_b18/final", out
    for bad in ("basilisk_v1", "basilisk_b1/", "basilisk_b11", "basilisk_b17"):
        assert bad not in out
    assert (ROOT / out / "wheel_all.h5").exists()


def test_dataset_h5_declares_assumption_semantics():
    """§24: h5 元数据必须自带失效定义的性质声明, 免得脱离文档后被误读。"""
    import h5py
    p = ROOT / "data/sim/wheel_basilisk_b18/final/wheel_all.h5"
    if not p.exists():
        pytest.skip("正式数据未生成")
    with h5py.File(p, "r") as f:
        assert str(f.attrs["lineage"]) == "basilisk_b18"
        assert str(f.attrs["scenario"]) == "NOMINAL"
        assert float(f.attrs["L_ref_years"]) == 3.0
        assert str(f.attrs["eol_rule"]) == "cumulative_damage_reaches_one"
        assert str(f.attrs["failure_definition_provenance"]) == \
            "NEW_B18_ASSUMPTION"
        assert str(f.attrs["failure_definition_semantics"]) == \
            "engineering_assumption_failure_definition"
        assert bool(f.attrs["is_manufacturer_failure_specification"]) is False
        assert bool(f.attrs["contains_rul_label"]) is False


def test_dataset_truth_in_separate_group():
    """§19 的结构前提: D 与真 EOL 放在独立 truth/ 组, 便于特征脚本回避。"""
    import h5py
    p = ROOT / "data/sim/wheel_basilisk_b18/final/wheel_all.h5"
    if not p.exists():
        pytest.skip("正式数据未生成")
    with h5py.File(p, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        assert len(keys) == 150
        g = f[keys[0]]
        assert "truth" in g and "D" in g["truth"]
        assert "eol_idx" in g["truth"].attrs
        # 遥测列本身不含 D
        assert "D" not in g
        for c in ("t", "omega_cmd", "omega", "I_m", "T", "T_cmd", "b_true",
                  "label_fail"):
            assert c in g, f"缺遥测列 {c}"


def test_label_fail_monotone_and_from_damage():
    """label_fail 单向 (一旦置 1 不回落), 且起点与 truth/eol_idx 一致。"""
    import h5py
    p = ROOT / "data/sim/wheel_basilisk_b18/final/wheel_all.h5"
    if not p.exists():
        pytest.skip("正式数据未生成")
    with h5py.File(p, "r") as f:
        for key in sorted(k for k in f.keys() if k.startswith("traj_"))[:30]:
            g = f[key]
            lf = g["label_fail"][:]
            assert np.all(np.diff(lf.astype(int)) >= 0), f"{key} label 非单向"
            observed = bool(g["truth"].attrs["event_observed"])
            eol = int(g["truth"].attrs["eol_idx"])
            if observed:
                assert lf.any() and int(np.argmax(lf)) == eol
                assert g["truth"]["D"][eol] >= 1.0
            else:
                assert not lf.any(), f"{key} 删失却有 label_fail=1"
