"""B1.9 §2/§3/§4: 可观测损伤代理的构造性质 (真跑, 不读 JSON 结论)。

含 §10 要求的以下函数名 (字节一致):
  test_damage_proxy_uses_only_observables
  test_damage_proxy_prefix_invariant
  test_damage_proxy_monotone
  test_damage_proxy_bounded
"""
from __future__ import annotations

import inspect

import h5py
import numpy as np
import pytest

from conftest import CK, DOCS, ROOT, SCRIPTS, load_script

# 抽查条数 —— 真跑积分, 全 150 条会让整套测试慢一个数量级
N_PROBE = 4


def _probe_keys(path, n=N_PROBE):
    with h5py.File(path, "r") as f:
        return sorted(k for k in f.keys() if k.startswith("traj_"))[:n]


def test_damage_proxy_uses_only_observables(b19_config, b19_source_h5,
                                            b19_proxy_module, b19_reference):
    """构造只碰 mission_features + 遥测 T。

    做法: 把源 h5 的一条轨迹复制成"只含可观测量"的内存 h5 (删掉 truth 组、
    b_true、label_fail 与全部 trajectory attrs), 再跑构造函数。若构造式偷偷
    依赖任何真值, 这里必然 KeyError/AttributeError 而不是安静通过。
    """
    key = _probe_keys(b19_source_h5, 1)[0]
    with h5py.File(b19_source_h5, "r") as fs:
        g = fs[key]
        full = b19_proxy_module.hi_damage_obs(g, b19_config, b19_reference)["hi"]
        mf = {k: g["mission_features"][k][:] for k in g["mission_features"].keys()}
        T = g["T"][:]

    # 只含可观测量的替身
    with h5py.File("b19_obs_only.h5", "w", driver="core", backing_store=False) as fm:
        gg = fm.create_group("t")
        gg.create_dataset("T", data=T)
        mg = gg.create_group("mission_features")
        for k, v in mf.items():
            mg.create_dataset(k, data=v)
        stripped = b19_proxy_module.hi_damage_obs(gg, b19_config, b19_reference)["hi"]

    assert stripped.shape == full.shape
    d = np.abs(stripped - full)
    assert d.max() == 0.0, (
        f"剥离 truth 后结果改变 (max_abs_diff {d.max():.3e}) —— 说明构造依赖真值")


def test_damage_proxy_never_reads_D(b19_proxy_module):
    """§5: 构造函数源码不得出现 truth 组 / 累积损伤真值的访问。"""
    src = (inspect.getsource(b19_proxy_module.hi_damage_obs)
           + inspect.getsource(b19_proxy_module.drives_from_mission_features))
    for tok in ('"truth"', "'truth'", "D_true", "truth/"):
        assert tok not in src, f"构造函数出现 {tok!r}"


def test_damage_proxy_never_reads_Eol(b19_proxy_module):
    """§4: 构造函数不得出现 EOL / RUL / label / attrs 真值访问。"""
    src = (inspect.getsource(b19_proxy_module.hi_damage_obs)
           + inspect.getsource(b19_proxy_module.drives_from_mission_features))
    for tok in ("eol", "rul", "label_fail", "b_true", ".attrs["):
        assert tok not in src, f"构造函数出现 {tok!r}"


def test_damage_proxy_prefix_invariant(b19_config, b19_source_h5,
                                       b19_proxy_module, b19_reference):
    """§4 核心测试: 截断未来 50% 后, 前 50% 必须**逐位**相同 (atol = 0)。

    容差为 0 是刻意的 —— cumsum 的前缀在数学上就该逐位相同; 任何非零差异都说明
    构造式里混入了未来信息 (full-trajectory 归一化 / future max / 末值归一)。
    """
    frac = float(b19_config["chronological"]["prefix_truncation_test_frac"])
    atol = float(b19_config["feature_gate"]["prefix_invariance_atol"])
    assert atol == 0.0, "前缀不变性容差被放宽 —— 这是 outcome tuning"
    with h5py.File(b19_source_h5, "r") as fs:
        for key in sorted(k for k in fs.keys() if k.startswith("traj_"))[:N_PROBE]:
            g = fs[key]
            full = b19_proxy_module.hi_damage_obs(g, b19_config, b19_reference)["hi"]
            n_cut = max(2, int(full.size * frac))
            cut = b19_proxy_module.hi_damage_obs(g, b19_config, b19_reference,
                                                 n_end=n_cut)["hi"]
            assert cut.size == n_cut
            d = np.abs(cut - full[:n_cut])
            assert d.max() <= atol, f"{key}: 前缀不一致 max_abs_diff {d.max():.3e}"


def test_damage_proxy_monotone(b19_config, b19_source_h5, b19_proxy_module,
                               b19_reference):
    """stress >= 0 => 累积损伤单调非降 => HI 单调非降。"""
    with h5py.File(b19_source_h5, "r") as fs:
        for key in sorted(k for k in fs.keys() if k.startswith("traj_"))[:N_PROBE]:
            r = b19_proxy_module.hi_damage_obs(fs[key], b19_config, b19_reference)
            assert np.all(r["stress"] >= 0.0), f"{key}: stress 出现负值"
            assert np.all(np.diff(r["D_obs"]) >= 0.0), f"{key}: 累积量非单调"
            assert np.all(np.diff(r["hi"]) >= 0.0), f"{key}: HI 非单调"


def test_damage_proxy_bounded(b19_config, b19_source_h5, b19_proxy_module,
                              b19_reference):
    """clip 到 [0, 1] 且有限。"""
    lo = float(b19_config["damage_proxy"]["clip_lo"])
    hi_ = float(b19_config["damage_proxy"]["clip_hi"])
    assert (lo, hi_) == (0.0, 1.0)
    with h5py.File(b19_source_h5, "r") as fs:
        for key in sorted(k for k in fs.keys() if k.startswith("traj_"))[:N_PROBE]:
            hi = b19_proxy_module.hi_damage_obs(fs[key], b19_config,
                                                b19_reference)["hi"]
            assert np.all(np.isfinite(hi))
            assert hi.min() >= lo and hi.max() <= hi_


def test_proxy_reuses_frozen_equations(b19_damage_proxy):
    """§2: 方程必须来自已冻结实现, 不得在 B1.9 侧新写一份。"""
    srcs = b19_damage_proxy["implementation_sources"]
    assert srcs["g_duty"] == "scripts/basilisk_b1/calibrate_degradation.py::g_duty"
    assert srcs["a_T"] == "src/sim/damage_model.py::arrhenius_accel"
    assert srcs["cumulative"] == "src/sim/damage_model.py::damage_series"
    ws = b19_damage_proxy["wear_drive"]
    total = (ws["speed_util_weight"] + ws["torque_util_weight"]
             + ws["zero_crossing_weight"] + ws["maneuver_weight"])
    assert abs(total - 1.0) < 1e-12, f"g_duty 权重和 = {total}"
    ar = b19_damage_proxy["arrhenius"]
    assert ar["Ea_eV"] == 0.3 and ar["T_ref_K"] == 293.15
    assert b19_damage_proxy["L_ref_years"] == 3.0


def test_observability_verdict_ok(b19_observable_audit):
    """§3: 四个 g_duty 输入与温度均不得为 HIDDEN_TRUTH。"""
    a = b19_observable_audit
    assert a["verdict"] == "B19_OBSERVABILITY_OK"
    gd = [r for r in a["rows"] if r["role"] == "g_duty_input"]
    assert len(gd) == 4
    for r in gd:
        assert r["provenance"] in a["required_provenance"], r
        assert r["provenance"] not in a["forbidden_provenance"], r
    t = next(r for r in a["rows"] if r["role"] == "temperature")
    assert t["provenance"] == "OBSERVABLE_DIRECT"
    assert t["container"] == "trajectory_dataset"


def test_observability_records_deployment_caveat(b19_observable_audit):
    """部署前提必须如实记录: 窗统计量不可由已存低速遥测反推。"""
    a = b19_observable_audit
    rows = a["telemetry_recoverability"]
    assert rows, "缺可恢复性证据"
    assert all(not r["recoverable_from_stored_telemetry"] for r in rows)
    assert all(r["corr_abs_omega_telemetry_vs_omega_p95"] < 0.999 for r in rows)
    md = (DOCS / "observable_input_audit.md").read_text(encoding="utf-8")
    assert "OBSERVABLE_DERIVED" in md and "1 Hz" in md


def test_observability_hidden_names_disjoint_from_inputs(b19_observable_audit):
    """输入集与黑名单集必须无交集 —— 这是"不相交"的机器化证明。"""
    a = b19_observable_audit
    inputs = {r["source_name"] for r in a["rows"]
              if r["role"] in ("g_duty_input", "temperature")}
    hidden = {r["source_name"] for r in a["rows"]
              if r["role"] == "hidden_reference"}
    assert hidden, "黑名单量一个都没定位到 —— 审计等于空转"
    assert not (inputs & hidden), f"交集 {sorted(inputs & hidden)}"
    for r in a["rows"]:
        if r["role"] == "hidden_reference":
            assert r["provenance"] == "HIDDEN_TRUTH", r
