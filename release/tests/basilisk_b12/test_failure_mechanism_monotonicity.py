"""tests/basilisk_b12/test_failure_mechanism_monotonicity.py

§19 —— 退化机理与配对生成的测试:
  * test_f3_truth_only_for_label
  * test_candidates_share_same_wear_trajectory
  * test_healthy_window_not_triggered
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

F1, F2, F3 = "CURRENT_WINDOW_P95", "FRICTION_TORQUE_P95", "VISCOUS_FRICTION_STATE"
TRUTH_FIELDS = ("b_true", "Kt", "Tc", "b0")


def test_f3_truth_only_for_label(b12_gen, b12_cfg, cand_rec):
    """§7/§8: F3 用隐藏真值 b_true, 但只能作 simulator 与 EOL truth。

    这里验证两件事:
      1. F3 统计量确实就是 b_true (不是任何可观测量的代理);
      2. b_true 单调不减 -> F3 的 EOL 判定具有单调性 (越限后不会自愈)。
    """
    import pandas as pd
    b = np.linspace(1e-5, 1e-3, 400)      # 单调递增的退化状态
    df = pd.DataFrame({"b_true": b, "omega_cmd": np.full(400, 100.0),
                       "I_m": np.zeros(400), "T_cmd": np.zeros(400)})
    stat = b12_gen.candidate_statistic(df, F3, {"Tc": 1.25e-3, "Kt": 0.025}, b12_cfg)
    assert np.allclose(stat, b), "F3 统计量必须就是 b_true"

    spec = b12_cfg["failure_candidates"]["spec"][F3]
    assert spec["statistic"] == "viscous_state_b_true"

    # 单调退化 -> 一旦连续越限就不再回落, label 单调
    win_n, pers = 8, 4
    dec = b12_gen.redecide_eol(stat, float(np.percentile(b, 60)), win_n, 0.95,
                              pers, len(b))
    assert dec["failed"] is True
    assert np.all(np.diff(dec["label"]) >= 0), "F3 的 label 必须单调"
    wq = dec["window_stat"]
    assert np.all(np.diff(wq) >= -1e-18), \
        "b_true 单调递增时窗级 q95 也应单调不减"

    if cand_rec is None:
        pytest.skip("candidates.json 尚未生成")
    d = cand_rec["per_candidate"][F3]["threshold_derivation"]
    # b_fail 必须由物理能力约束推导, 不得由 RUL 结果选择
    avail = float(d["available_viscous_torque_Nm"])
    om_ref = float(d["omega_reference_rad_s"])
    assert abs(avail / om_ref - float(cand_rec["per_candidate"][F3]["threshold"])) \
        < 1e-18, "b_fail 无法由 available_viscous_torque / omega_reference 复算"
    assert abs(om_ref - float(d["speed_util_ref_p95"])
               * float(d["omega_rated_rad_s"])) < 1e-12, \
        "omega_reference 不是 speed_util_ref_p95 * Omega_rated"


def test_candidates_share_same_wear_trajectory(cand_rec):
    """§10/§11: 三变体必须共享完全相同的退化过程, 唯一差异是 EOL definition。

    验证强度依次递增:
      1. paired_trajectory_hash 相同 (共享输入);
      2. 与 B1/B1.1 的 paired hash 相同 (CRN 沿用);
      3. **逐列逐值**比对三份 h5 的 b_true / omega / I_m / T / T_cmd / omega_cmd
         —— 这是 §11 的实质要求, 只比 hash 不够。
    """
    if cand_rec is None:
        pytest.skip("candidates.json 尚未生成")
    import h5py

    ph = cand_rec["paired_trajectory_hash"]
    assert ph == "ff1de559b2746dcb394c4de6d70bda4d96ca594d1b953f58a4fa390366f41cf6", \
        f"paired_trajectory_hash 与 B1/B1.1 不一致 ({ph[:16]}…) —— CRN 对齐被破坏"

    files = {n: Path(cd["h5"]) for n, cd in cand_rec["per_candidate"].items()}
    assert len(files) == 3
    shared_cols = ("b_true", "omega", "omega_cmd", "I_m", "T", "T_cmd", "t")
    handles = {n: h5py.File(p, "r") for n, p in files.items()}
    try:
        for n, f in handles.items():
            assert f.attrs["paired_trajectory_hash"] == ph, f"{n} paired hash 不符"
        ref_name = F2
        ref = handles[ref_name]
        trajs = sorted(k for k in ref.keys() if k.startswith("traj_"))
        assert len(trajs) == cand_rec["n_traj"]
        # 全部轨迹逐列比对 (数据量可接受: 3 × 60 × 7 列)
        for k in trajs:
            for c in shared_cols:
                a = np.asarray(ref[k][c][:])
                for other in (F3, F1):
                    b = np.asarray(handles[other][k][c][:])
                    assert a.shape == b.shape, f"{k}/{c} 形状不同 ({other})"
                    assert np.array_equal(a, b), \
                        f"{k}/{c} 在 {ref_name} 与 {other} 之间不同 —— " \
                        "§11 要求三变体只重判 EOL, 不得重新生成退化过程"
            # 物理参数也必须逐个相同
            for att in ("b0", "Kt", "Tc", "J", "tau_years", "omega0", "seed_traj"):
                v = ref[k].attrs[att]
                for other in (F3, F1):
                    assert handles[other][k].attrs[att] == v, \
                        f"{k} 参数 {att} 在 {other} 中不同"
        # 唯一允许不同的是 label_fail / eol / window_stat
        diff = 0
        for k in trajs:
            if not np.array_equal(np.asarray(ref[k]["label_fail"][:]),
                                  np.asarray(handles[F3][k]["label_fail"][:])):
                diff += 1
        assert diff > 0, \
            "F2 与 F3 的 label_fail 完全相同 —— 两个判据退化成了同一个, 需检查阈值"
    finally:
        for f in handles.values():
            f.close()


def test_healthy_window_not_triggered(audit_rec, b12_cfg):
    """§12 Gate 12: 健康段窗级统计量不得触发判据 (窗级口径, 非逐点)。

    本测试不重判 PASS/FAIL (那是 Gate 的职责), 而是校验:
      1. Gate 12 确实存在且用窗级口径;
      2. 健康段定义与 sim.hi.healthy_frac 一致 (口径不得漂移);
      3. 记录的越限占比是 [0,1] 内的有效数, 不是被悄悄填的 0。
    """
    hf = float(b12_cfg["gate"]["healthy_frac"])
    assert hf == float(b12_cfg["sim"]["hi"]["healthy_frac"]), \
        "gate.healthy_frac 必须等于 sim.hi.healthy_frac (口径不得漂移)"

    if audit_rec is None:
        pytest.skip("candidate_audit.json 尚未生成")
    hmax = float(b12_cfg["gate"]["healthy_trigger_frac_max"])
    for name, r in audit_rec["per_candidate"].items():
        g12 = [x for x in r["gates"] if x["name"].startswith("12 ")]
        assert len(g12) == 1, f"{name} 缺 Gate 12"
        assert "窗级" in g12[0]["detail"], \
            f"{name} Gate 12 未声明窗级口径 (逐点口径会低估, 不可混用)"
        v = r["healthy_trigger_fraction_window"]
        assert np.isfinite(v), f"{name} 健康触发占比为 NaN 却未标 n=0"
        assert 0.0 <= v <= 1.0, f"{name} 健康触发占比越界: {v}"
        assert g12[0]["pass"] == bool(v < hmax), \
            f"{name} Gate 12 判定与记录值不一致"
    assert "不可互引" in audit_rec["gate_caliber_note"], \
        "审计记录未声明口径不可互引 (三种口径混用会产生伪结论)"
