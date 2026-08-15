"""tests/basilisk/test_profile_schema.py

§18 —— profiles.h5 schema / 模式差异 / eclipse 诚实性。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.basilisk_config import load_basilisk_config    # noqa: E402
from src.sim.basilisk_profile import MODES                    # noqa: E402

CFG = load_basilisk_config("configs/wheel_basilisk.yaml")
PROFILE_H5 = ROOT / CFG["paths"]["profile_h5"]
DUTY_JSON = ROOT / CFG["paths"]["duty_stats"]
PROV_JSON = ROOT / CFG["paths"]["provenance"]

REQUIRED_DATASETS = ("time_s", "sigma_BN", "omega_BN_B", "wheel_speed_rad_s",
                     "motor_torque_Nm")
REQUIRED_ATTRS = ("mode", "seed", "dt_s", "generator", "basilisk_version",
                  "script_sha256")

pytestmark = pytest.mark.skipif(
    not PROFILE_H5.exists(),
    reason="profiles.h5 未生成 (先跑 scripts/basilisk/generate_profiles.py)")


def _h5():
    import h5py
    return h5py.File(PROFILE_H5, "r")


def test_basilisk_profile_schema():
    """§6: 每 mode × realization 必须有全部必需 dataset 与 attrs。"""
    n_runs = int(CFG["sim"]["profile"]["n_runs_per_mode"])
    with _h5() as f:
        for mode in MODES:
            assert mode in f, f"缺 mode 组 {mode}"
            runs = sorted(f[mode].keys())
            assert len(runs) >= 3, f"{mode} 只有 {len(runs)} 个 realization (§4 要求 >=3)"
            assert len(runs) == n_runs
            for rk in runs:
                g = f[f"{mode}/{rk}"]
                for d in REQUIRED_DATASETS:
                    assert d in g, f"{mode}/{rk} 缺 dataset {d}"
                for a in REQUIRED_ATTRS:
                    assert a in g.attrs, f"{mode}/{rk} 缺 attr {a}"
                assert g.attrs["generator"] == "basilisk"
                assert g.attrs["mode"] == mode
                assert float(g.attrs["dt_s"]) == 1.0
                n = g["time_s"].shape[0]
                assert g["sigma_BN"].shape == (n, 3)
                assert g["omega_BN_B"].shape == (n, 3)
                assert g["wheel_speed_rad_s"].shape[0] == n
                assert g["motor_torque_Nm"].shape == g["wheel_speed_rad_s"].shape


def test_basilisk_profile_dt_is_one_second():
    """§11: Basilisk 原始分辨率必须是 1 s (不是 30 min 瞬时采样)。"""
    with _h5() as f:
        assert float(f.attrs["dt_s"]) == 1.0
        for mode in MODES:
            for rk in sorted(f[mode].keys()):
                t = np.asarray(f[f"{mode}/{rk}/time_s"][:], dtype=float)
                d = np.diff(t)
                assert np.allclose(d, 1.0, atol=1e-6), \
                    f"{mode}/{rk} 采样间隔非 1 s: {np.unique(np.round(d,6))[:5]}"


def test_basilisk_profile_all_finite():
    with _h5() as f:
        for mode in MODES:
            for rk in sorted(f[mode].keys()):
                g = f[f"{mode}/{rk}"]
                for d in REQUIRED_DATASETS:
                    v = np.asarray(g[d][:], dtype=float)
                    assert np.isfinite(v).all(), f"{mode}/{rk}/{d} 含 NaN/Inf"


def test_basilisk_modes_not_identical():
    """§8: 五模式必须真正可区分, 禁止只是不同幅值的同一正弦。

    判据与 audit_profiles.py 的 H4 同源: 至少 min_distinct_metrics 项指标满足
    组间极差 / 组内散布 >= separation_ratio。
    """
    duty = json.loads(DUTY_JSON.read_text(encoding="utf-8"))["modes"]
    acfg = CFG["audit"]
    getters = {
        "speed_rms": lambda s: s["speed"]["rms"],
        "speed_p95": lambda s: s["speed"]["p95_abs"],
        "torque_rms": lambda s: s["torque"]["rms"],
        "torque_p95": lambda s: s["torque"]["p95_abs"],
        "zero_crossings": lambda s: float(s["zero_crossing_count"]),
        "maneuver_fraction": lambda s: s["maneuver_fraction"],
    }
    n_ok = 0
    for name in acfg["distinct_metrics"]:
        get = getters[name]
        per = {m: [get(r) for r in duty[m]["per_run"]] for m in MODES}
        means = np.array([np.mean(per[m]) for m in MODES])
        within = float(np.median([np.std(per[m]) for m in MODES]))
        between = float(means.max() - means.min())
        n_ok += bool(between > 0.0 if within <= 0.0
                     else between / within >= float(acfg["separation_ratio"]))
    assert n_ok >= int(acfg["min_distinct_metrics"]), \
        f"仅 {n_ok} 项指标可区分, 五模式高度同质 -> BASILISK_PROFILE_FAIL"


def test_imaging_torque_above_cruise():
    """§8 H1: imaging 的 torque_rms 必须高于 cruise。"""
    duty = json.loads(DUTY_JSON.read_text(encoding="utf-8"))["modes"]
    ti = np.mean([r["torque"]["rms"] for r in duty["imaging"]["per_run"]])
    tc = np.mean([r["torque"]["rms"] for r in duty["cruise"]["per_run"]])
    assert ti > tc, f"imaging torque_rms {ti:.4e} 未超过 cruise {tc:.4e}"


def test_desat_has_momentum_release():
    """§8 H2: desat 必须有轮动量卸载形态, 且 cruise 对照没有。"""
    sys.path.insert(0, str(ROOT / "scripts" / "basilisk"))
    from audit_profiles import wheel_momentum_series, count_release_events
    acfg = CFG["audit"]
    need = int(acfg["desat_min_release_events"])
    drop = float(acfg["desat_release_drop_frac"])
    with _h5() as f:
        for rk in sorted(f["desat"].keys()):
            hw = wheel_momentum_series(f, f"desat/{rk}")
            n_ev, _ = count_release_events(hw, drop)
            assert n_ev >= need, f"desat/{rk} 卸载事件 {n_ev} < {need}"
        # 对照: cruise 不应有卸载形态 (否则"卸载"不是 desat 的特征)
        for rk in sorted(f["cruise"].keys()):
            hw = wheel_momentum_series(f, f"cruise/{rk}")
            n_ev, _ = count_release_events(hw, drop)
            assert n_ev < need, f"cruise/{rk} 也有 {n_ev} 次卸载, desat 形态不特异"


def test_eclipse_fraction_not_fabricated():
    """§7: 未建遮蔽模型 -> eclipse_fraction 必须是 null, 禁止伪造数字。"""
    duty = json.loads(DUTY_JSON.read_text(encoding="utf-8"))["modes"]
    for m in MODES:
        assert duty[m]["aggregate"]["eclipse_fraction"] is None, \
            f"{m} 的 eclipse_fraction 被伪造成了具体数值"
        for r in duty[m]["per_run"]:
            assert r["eclipse_fraction"] is None


def test_profile_provenance_complete():
    """§9: provenance.json 必须含全部溯源字段。"""
    prov = json.loads(PROV_JSON.read_text(encoding="utf-8"))
    for k in ("generator", "basilisk_version", "python_version",
              "generation_timestamp", "config_sha256", "script_sha256",
              "seed", "profiles_sha256"):
        assert k in prov and prov[k] not in (None, ""), f"provenance 缺 {k}"
    assert prov["generator"] == "basilisk"
    assert len(prov["profiles_sha256"]) == 64


def test_profiles_declare_no_labels():
    """§0 边界: profiles.h5 必须机器可读地声明不含 HI/EOL/RUL 标签。"""
    with _h5() as f:
        for flag in ("contains_rul_label", "contains_hi", "contains_eol"):
            assert flag in f.attrs, f"缺声明 attr {flag}"
            assert not bool(f.attrs[flag]), f"{flag} 为 True, 违反 §0"
