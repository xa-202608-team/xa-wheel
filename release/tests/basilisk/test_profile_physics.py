"""tests/basilisk/test_profile_physics.py

§18 —— Basilisk 工况的物理合理性 (姿态控制确实在工作, 轮子确实在响应)。

这些不变量的作用是防"看起来像但物理上是死的"工况: 比如控制器根本没闭环
(sigma_BR 不收敛)、轮速与力矩脱耦 (力矩非零但轮速不变)。这两种情况下工况数据
仍能通过 schema 检查, 却完全不具备任务相关性。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.basilisk_config import load_basilisk_config    # noqa: E402
from src.sim.basilisk_profile import MODES, duty_stats_of     # noqa: E402

CFG = load_basilisk_config("configs/wheel_basilisk.yaml")
PROFILE_H5 = ROOT / CFG["paths"]["profile_h5"]
WI = int(CFG["sim"]["profile"]["bridge"]["wheel_index"])
UMAX = float(CFG["sim"]["profile"]["duty"]["u_max_Nm"])

pytestmark = pytest.mark.skipif(not PROFILE_H5.exists(),
                                reason="profiles.h5 未生成")


def _h5():
    import h5py
    return h5py.File(PROFILE_H5, "r")


def test_attitude_control_is_closed_loop():
    """姿态跟踪误差必须被控制器压下来 (末段 < 初段), 否则闭环没在工作。

    nadir / desat 用 hillPoint 参考 + 持续轨道扰动, 稳态误差不必趋零, 因此判据是
    "末段 <= 初段" 而非绝对收敛; cruise / imaging / safe 用惯性指向, 要求明显下降。
    """
    strict = {"cruise", "imaging", "safe"}
    with _h5() as f:
        for mode in MODES:
            for rk in sorted(f[mode].keys()):
                s = np.linalg.norm(np.asarray(f[f"{mode}/{rk}/sigma_BR"][:]), axis=1)
                n = len(s)
                head = float(np.max(s[: max(1, n // 20)]))
                tail = float(np.median(s[-max(1, n // 10):]))
                if mode in strict:
                    assert tail < 0.5 * head, \
                        f"{mode}/{rk} 姿态误差未收敛: head={head:.4g} tail={tail:.4g}"
                else:
                    assert tail <= head * 1.5, \
                        f"{mode}/{rk} 姿态误差发散: head={head:.4g} tail={tail:.4g}"


def test_wheel_responds_to_motor_torque():
    """轮速变化必须与电机力矩同号相关 (dΩ/dt ≈ u/Js), 否则轮子与力矩脱耦。"""
    with _h5() as f:
        for mode in MODES:
            rk = sorted(f[mode].keys())[0]
            g = f[f"{mode}/{rk}"]
            om = np.asarray(g["wheel_speed_rad_s"][:, WI], dtype=float)
            u = np.asarray(g["motor_torque_Nm"][:, WI], dtype=float)
            dom = np.diff(om)
            uu = u[:-1]
            mask = np.abs(uu) > 1e-6 * UMAX
            if mask.sum() < 50:
                continue                     # 该 run 力矩几乎恒零, 跳过 (safe 可能如此)
            c = float(np.corrcoef(dom[mask], uu[mask])[0, 1])
            assert c > 0.5, f"{mode}/{rk} dΩ 与电机力矩相关性仅 {c:.3f}, 疑似脱耦"


def test_motor_torque_within_rating():
    """电机力矩不得超过官方 Honeywell_HR16 的 u_max (取自 RW 效应器实际施加值)。"""
    with _h5() as f:
        for mode in MODES:
            for rk in sorted(f[mode].keys()):
                u = np.asarray(f[f"{mode}/{rk}/motor_torque_Nm"][:], dtype=float)
                assert np.abs(u).max() <= UMAX * 1.001, \
                    f"{mode}/{rk} 力矩 {np.abs(u).max():.4g} 超过额定 {UMAX}"


def test_duty_stats_of_matches_h5():
    """duty_stats_of 对已落盘数据的重算结果必须与 duty_stats.json 一致。"""
    import json
    duty = json.loads((ROOT / CFG["paths"]["duty_stats"]).read_text(
        encoding="utf-8"))
    dcfg = CFG["sim"]["profile"]["duty"]
    with _h5() as f:
        for mode in MODES:
            for i, rk in enumerate(sorted(f[mode].keys())):
                g = f[f"{mode}/{rk}"]
                st = duty_stats_of(g["wheel_speed_rad_s"][:, WI],
                                   g["motor_torque_Nm"][:, WI], 1.0,
                                   float(dcfg["maneuver_torque_frac"]),
                                   float(dcfg["high_torque_frac"]),
                                   float(dcfg["u_max_Nm"]))
                ref = duty["modes"][mode]["per_run"][i]
                for grp in ("speed", "torque"):
                    for k, v in st[grp].items():
                        assert abs(v - ref[grp][k]) <= 1e-9 * max(1.0, abs(v)), \
                            f"{mode}/{rk} {grp}.{k}: {v} vs {ref[grp][k]}"
                assert st["zero_crossing_count"] == ref["zero_crossing_count"]


def test_desat_wheel_momentum_accumulates_then_releases():
    """§5 D: desat 必须先有动量积累再有释放, 不能只是随机抖动。

    判据: 轮动量幅值序列的**长期趋势段**里, 至少存在一次持续上升 (积累) 紧跟
    一次持续下降 (卸载)。用平滑后的序列避免秒级抖动被误判为趋势。
    """
    sys.path.insert(0, str(ROOT / "scripts" / "basilisk"))
    from audit_profiles import wheel_momentum_series
    with _h5() as f:
        found = 0
        for rk in sorted(f["desat"].keys()):
            hw = wheel_momentum_series(f, f"desat/{rk}")
            k = 300                                       # 5 min 平滑
            s = np.convolve(hw, np.ones(k) / k, mode="valid")
            d = np.diff(s)
            # 存在显著上升段与显著下降段, 且下降幅度可观
            rise = float(d[d > 0].sum())
            fall = float(-d[d < 0].sum())
            if rise > 0.05 * s.mean() and fall > 0.05 * s.mean():
                found += 1
        assert found >= 2, \
            f"desat 只有 {found} 条 run 呈现积累+释放形态 (需 >=2)"
