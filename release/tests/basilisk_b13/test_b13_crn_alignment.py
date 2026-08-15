"""tests/basilisk_b13/test_b13_crn_alignment.py

§9/§11/§12/§19 —— CRN 延续与 probe Gate。

§19 命名测试:
  * test_crn_hash_preserved
  * test_probe_gate
  * test_healthy_f2_no_trigger
"""
from __future__ import annotations

import numpy as np
import pytest

PAIRED_HASH = "ff1de559b2746dcb394c4de6d70bda4d96ca594d1b953f58a4fa390366f41cf6"


def test_crn_hash_preserved(probe_rec, b13_cfg):
    """§9: paired_trajectory_hash 必须与 B1.1/B1.2 逐字相同。

    这是"只有退化尺度变了"的硬证据: 该 hash 覆盖与 horizon 无关的共享输入
    (原始物理参数抽样 + base-horizon duty 前缀)。b0_scale 作用在 apply_calibration
    之后, 不进入 hash —— 所以 hash 变了就说明动了 RNG / duty / profile。
    """
    if probe_rec is None:
        pytest.skip("需先跑 generate_probe.py")
    assert probe_rec["paired_trajectory_hash"] == PAIRED_HASH, (
        f"CRN 被破坏: {probe_rec['paired_trajectory_hash']} != {PAIRED_HASH}")
    assert probe_rec["crn_preserved"] is True
    assert probe_rec["paired_trajectory_hash_expected"] == PAIRED_HASH
    assert str(b13_cfg["probe"]["expected_paired_trajectory_hash"]) == PAIRED_HASH


def test_crn_seed_unchanged(probe_rec, b13_cfg):
    """seed 必须仍是 20260809 (换 seed 等于换一批轨迹, CRN 配对失效)。"""
    assert int(b13_cfg["seed"]) == 20260809
    if probe_rec is not None:
        assert int(probe_rec["seed"]) == 20260809


def test_probe_gate(probe_audit_rec):
    """§11/§12: probe 审计必须给出 14 条 Gate, 且口径正确。"""
    if probe_audit_rec is None:
        pytest.skip("需先跑 audit_probe.py")
    assert int(probe_audit_rec["n_gate"]) == 14, \
        f"Gate 数 {probe_audit_rec['n_gate']} != 14"
    names = [g["name"] for g in probe_audit_rec["gates"]]
    assert len(set(names)) == 14, f"Gate 名重复: {names}"
    idx = [g["gate"] for g in probe_audit_rec["gates"]]
    assert idx == list(range(1, 15)), f"Gate 编号不连续: {idx}"
    # Gate 13 / 14 必须是 B1.3 新增的那两条
    assert names[12] == "q0_distribution_match"
    assert names[13] == "normalized_q_trajectory_identity"


def test_gate12_uses_f2_not_current_margin(probe_audit_rec, b13_cfg):
    """§11 明令: Gate 12 针对 **F2 摩擦力矩阈值**, 绝不能是 current margin。"""
    if probe_audit_rec is None:
        pytest.skip("需先跑 audit_probe.py")
    g12 = next(g for g in probe_audit_rec["gates"] if g["gate"] == 12)
    cal = g12["caliber"]
    assert "摩擦力矩" in cal or "friction" in cal.lower(), \
        f"Gate 12 口径描述未提摩擦力矩: {cal}"
    assert "current margin" in cal or "不是" in cal, \
        "Gate 12 未显式声明它不是 current margin 判据"
    assert str(b13_cfg["gate"]["healthy_trigger_statistic"]) == "friction_torque_q95"
    # 阈值必须是 F2 的力矩值, 不是电流预算
    assert "0.14311462970213382" in cal, "Gate 12 未绑定 F2 力矩阈值"


def test_gate14_tolerance_frozen_before_probe(b13_cfg, scaling_rec,
                                              probe_audit_rec):
    """§12: Gate 14 容差必须**先冻结在 protocol 里**, 不是看到结果后选的。"""
    g = b13_cfg["gate"]
    assert str(g["q_trajectory_tolerance_mode"]) == "ulp"
    assert float(g["q_trajectory_tolerance_ulp"]) == 8.0
    assert float(g["q_trajectory_max_abs_diff"]) == 1.0e-12
    # protocol 在任何 probe 数据之前冻结
    if scaling_rec is not None:
        assert scaling_rec["frozen_before_any_probe_data"] is True
    if probe_audit_rec is None:
        pytest.skip("需先跑 audit_probe.py")
    g14 = next(x for x in probe_audit_rec["gates"] if x["gate"] == 14)
    v = g14["value"]
    # 有效容差必须由 ULP 口径导出, 且实测值确实在容差内
    assert v["tolerance_mode"] == "ulp"
    assert v["max_abs_diff"] <= v["tolerance_effective"]
    assert v["max_abs_diff"] <= v["absolute_backstop"]


def test_healthy_f2_no_trigger(probe_audit_rec, probe_rec):
    """§11 Gate 12: 健康段 (前 5%) 的 F2 窗级统计量不得系统性误触发。

    口径必须是 B1.2 冻结的**按健康窗数加权的总占比**, 不是"最差单条轨迹"。
    同时把逐轨迹最大值一并检查存在性 —— 不允许只报总占比而把 miss 隐藏掉。
    """
    if probe_audit_rec is None:
        pytest.skip("需先跑 audit_probe.py")
    g12 = next(g for g in probe_audit_rec["gates"] if g["gate"] == 12)
    v = g12["value"]
    assert "weighted_fraction" in v and "per_traj_max" in v, \
        "Gate 12 必须同时披露加权总占比与逐轨迹最大值 (不得隐藏最差情形)"
    assert np.isfinite(v["weighted_fraction"]), "加权占比是 NaN —— 无有效健康窗"
    assert v["n_healthy_windows"] > 0
    assert g12["pass"] is True, (
        f"健康段误触发率 {v['weighted_fraction']:.4f} 超限; "
        f"逐轨迹最大 {v['per_traj_max']:.4f}")


def test_censored_trajectories_not_counted_as_events(probe_audit_rec, probe_rec):
    """删失轨迹的 eol_idx = n-1 是约定值, 不得当成观测到的 EOL。"""
    if probe_audit_rec is None or probe_rec is None:
        pytest.skip("需先跑 audit_probe.py")
    recs = probe_rec["per_trajectory"]
    n_ev = sum(1 for r in recs if r["failed"])
    n_cs = len(recs) - n_ev
    assert probe_audit_rec["n_event_observed"] == n_ev
    assert probe_audit_rec["n_censored"] == n_cs
    assert n_cs > 0, "无删失轨迹 —— Gate 2 应已拦截"
    # 删失轨迹的 eol_idx 必须都等于末样本索引 (约定值)
    for r in recs:
        if not r["failed"]:
            assert r["eol_idx"] == r["n_samples"] - 1, \
                f"删失轨迹 {r['traj_id']} 的 eol_idx 不是约定末索引"
    # Gate 5/6/11 的口径声明必须写明"仅 event-observed"
    for gi in (5, 6, 11):
        g = next(x for x in probe_audit_rec["gates"] if x["gate"] == gi)
        assert "event-observed" in g["caliber"], \
            f"Gate {gi} 未声明只统计 event-observed"


def test_no_empty_bin_faked_as_zero(probe_audit_rec):
    """项目纪律: 空子集返回 NaN + n=0, 不伪造 0。"""
    if probe_audit_rec is None:
        pytest.skip("需先跑 audit_probe.py")
    st = probe_audit_rec["stats"]
    # 有事件观测时这些量必须是有限值; 若无事件必须是 NaN 而不是 0
    n_ev = probe_audit_rec["n_event_observed"]
    for k in ("eol_iqr_event_samples", "degradation_fraction_at_eol_event_p50"):
        v = st[k]
        if n_ev == 0:
            assert v is None or (isinstance(v, float) and np.isnan(v)), \
                f"{k} 在无事件时应为 NaN, 实际 {v}"
        else:
            assert np.isfinite(v), f"{k} 应为有限值, 实际 {v}"
