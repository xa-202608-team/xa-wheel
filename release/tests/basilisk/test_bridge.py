"""tests/basilisk/test_bridge.py

§18 —— 桥接层边界测试。这一组是整个任务最重要的测试:
  它们钉死 "Basilisk 只提供工况, 绝不产生 RUL/HI/EOL 标签" 这条边界。
"""
from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.basilisk_config import load_basilisk_config      # noqa: E402
from src.sim import basilisk_bridge as bb                       # noqa: E402
from src.sim import basilisk_profile as bp                      # noqa: E402

CFG = load_basilisk_config("configs/wheel_basilisk.yaml")
PROFILE_H5 = ROOT / CFG["paths"]["profile_h5"]


def _lib():
    if not PROFILE_H5.exists():
        pytest.skip("profiles.h5 未生成")
    return bb.load_profile_library(
        PROFILE_H5, int(CFG["sim"]["profile"]["bridge"]["wheel_index"]))


# --------------------------------------------------------------------------
# §0 分工边界: Basilisk 绝不产生标签
# --------------------------------------------------------------------------
def test_basilisk_never_generates_rul():
    """basilisk_profile / basilisk_bridge 都不得触及标签相关符号或模块。

    做三件事:
      1) 静态检查 import: 不得 import build_hi / models / transfer / baselines
      2) 静态检查标识符: 不得出现 rul / eol / hi_b / label_fail / failure_threshold
         等标签符号 (排除注释与字符串, 因为两个模块的 docstring 明确写着这些禁令词)
      3) 检查两个模块的公开 API 返回值里没有任何标签字段
    """
    forbidden_mods = ("build_hi", "src.models", "src.transfer", "src.baselines",
                      "src.experiments", "rate_head")
    forbidden_names = ("rul", "eol", "hi_b", "hi_a", "label_fail",
                       "failure_threshold", "build_hi_tf", "rul_lower_bound")
    for mod in (bp, bb):
        src = Path(inspect.getfile(mod)).read_text(encoding="utf-8")
        tree = ast.parse(src)
        # 1) imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for al in node.names:
                    assert not any(f in al.name for f in forbidden_mods), \
                        f"{mod.__name__} import 了 {al.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not any(f in node.module for f in forbidden_mods), \
                    f"{mod.__name__} from-import 了 {node.module}"
        # 2) 标识符 (只看 Name / Attribute / 函数与参数名, 不看注释与字符串)
        ids = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                ids.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                ids.add(node.attr.lower())
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ids.add(node.name.lower())
                for a in node.args.args + node.args.kwonlyargs:
                    ids.add(a.arg.lower())
        for f in forbidden_names:
            assert f not in ids, f"{mod.__name__} 使用了标签标识符 {f!r}"

    # 3) 返回值字段
    lib = _lib()
    duty = bb.build_duty_series(lib, CFG, traj_id=0, seed=7, n_windows=6)
    keys = set(duty.keys()) | set(duty["stats"].keys())
    for k in keys:
        kl = k.lower()
        assert "rul" not in kl and "eol" not in kl and "fail" not in kl \
            and kl != "hi", f"build_duty_series 返回了标签字段 {k}"


def test_bridge_uses_one_second_data():
    """§11: 桥接必须消耗 1 s 数据, 每窗 n_per_window = 1800 点。"""
    lib = _lib()
    n_win = 5
    duty = bb.build_duty_series(lib, CFG, traj_id=0, seed=7, n_windows=n_win)
    dt_deg = float(CFG["sim"]["sample_period_s"])
    dt_bsk = float(CFG["sim"]["profile"]["dt_s"])
    expect = int(round(dt_deg / dt_bsk))
    assert expect == 1800, f"n_per_window 应为 1800, 实为 {expect}"
    assert duty["n_per_window"] == expect
    assert duty["n_basilisk_samples"] == n_win * expect, \
        "消耗的 1 s 点数不等于 窗数 x 每窗点数 -> 疑似瞬时采样"


def test_bridge_rejects_point_sampling():
    """§11: 若把退化周期配成等于 Basilisk dt (即瞬时采样), 桥接必须报错。"""
    cfg = json.loads(json.dumps(CFG))
    cfg["sim"]["sample_period_s"] = 1.0          # n_per_window = 1
    with pytest.raises(ValueError, match="§11"):
        bb.build_duty_series(_lib(), cfg, traj_id=0, seed=7, n_windows=3)


def test_bridge_aggregates_to_window():
    """§10: 窗内聚合必须真的用到全部 1800 点, 不是取首点/末点。

    构造法: 手工造一个已知窗 (前半 +100, 后半 -100), 只有真聚合才能得到
    mean=0 / rms=100 / 穿零=1; 取瞬时点会得到 ±100 与 0 次穿零。
    """
    n = 1800
    sp = np.concatenate([np.full(n // 2, 100.0), np.full(n // 2, -100.0)])
    tq = np.concatenate([np.zeros(n // 2), np.full(n // 2, 0.1)])
    agg = bb.aggregate_window(sp, tq, maneuver_torque_frac=0.02, u_max_Nm=0.2)
    assert abs(agg["omega_mean"]) < 1e-12
    assert abs(agg["omega_rms"] - 100.0) < 1e-9
    assert abs(agg["omega_std"] - 100.0) < 1e-9
    assert agg["zero_crossing_count"] == 1.0
    assert abs(agg["torque_mean_abs"] - 0.05) < 1e-12
    assert abs(agg["maneuver_fraction"] - 0.5) < 1e-12
    # 全部 10 项工况统计都必须存在 (§10 清单)
    assert set(agg.keys()) == set(bb.DUTY_KEYS)


def test_bridge_window_stats_match_manual_recompute():
    """从真实 profile 抽一窗, 手工重算与 aggregate_window 必须一致。"""
    lib = _lib()
    r = lib["imaging"][0]
    sp, tq = r["speed"][:1800], r["torque"][:1800]
    agg = bb.aggregate_window(sp, tq, 0.02, 0.2)
    assert abs(agg["omega_rms"] - float(np.sqrt((sp ** 2).mean()))) < 1e-12
    assert abs(agg["torque_p95"] - float(np.percentile(np.abs(tq), 95))) < 1e-12
    assert abs(agg["omega_p95"] - float(np.percentile(np.abs(sp), 95))) < 1e-12


def test_mode_weights_sum_one():
    """§12: mode weights 采样后必须归一化到 1, 且落在冻结区间的相对序内。"""
    wr = CFG["sim"]["profile"]["mode_weights_range"]
    for tid in range(30):
        rng = bb.bridge_rng(20260808, tid)
        w = bb.sample_mode_weights(rng, wr)
        assert abs(sum(w.values()) - 1.0) < 1e-12, f"traj {tid} 权重和 {sum(w.values())}"
        assert set(w.keys()) == set(bb.MODES)
        assert all(v > 0 for v in w.values())


def test_mode_weights_range_frozen():
    """§12: 冻结的 mode_weights_range 数值不得漂移。"""
    wr = CFG["sim"]["profile"]["mode_weights_range"]
    expect = {"cruise": [0.20, 0.50], "nadir": [0.15, 0.40],
              "imaging": [0.05, 0.25], "desat": [0.02, 0.12],
              "safe": [0.02, 0.15]}
    for m, (lo, hi) in expect.items():
        assert abs(wr[m][0] - lo) < 1e-12 and abs(wr[m][1] - hi) < 1e-12, \
            f"{m} 权重区间被改动: {wr[m]} != {[lo, hi]}"


def test_mode_schedule_is_segmented_and_multimode():
    """§13: 任务序列必须成段 (不是逐窗白噪声), 且至少用到 2 种 mode。"""
    wr = CFG["sim"]["profile"]["mode_weights_range"]
    seg_range = tuple(CFG["sim"]["profile"]["bridge"]["segment_windows"])
    for tid in range(10):
        rng = bb.bridge_rng(42, tid)
        w = bb.sample_mode_weights(rng, wr)
        sched = bb.build_mode_schedule(rng, w, 400, seg_range)
        assert len(sched) == 400
        assert len(np.unique(sched)) >= 2, f"traj {tid} 只用了 1 种 mode"
        # 成段性: 相邻窗同 mode 的比例应远高于随机抽样的 1/5
        same = float(np.mean(sched[1:] == sched[:-1]))
        assert same > 0.7, f"traj {tid} 相邻同 mode 比例仅 {same:.3f}, 未成段"


def test_bridge_deterministic():
    """同 (seed, traj_id) 两次桥接结果逐位一致。"""
    lib = _lib()
    d1 = bb.build_duty_series(lib, CFG, traj_id=3, seed=7, n_windows=20)
    d2 = bb.build_duty_series(lib, CFG, traj_id=3, seed=7, n_windows=20)
    assert np.array_equal(d1["mode_id"], d2["mode_id"])
    assert d1["mode_weights"] == d2["mode_weights"]
    for k in bb.DUTY_KEYS:
        assert np.array_equal(d1["stats"][k], d2["stats"][k]), f"{k} 不一致"
    # 不同 traj_id 必须得到不同轨迹。注意 mode_id 要用真实量级的窗数来比:
    # n_windows=20 < 最小段长 48 时, "至少 2 种 mode" 兜底 (build_mode_schedule
    # 末尾) 会把序列压成 [主 mode]*10 + [次 mode]*10 的同一形态, 掩盖 RNG 差异。
    # 真实轨迹约 5.3e4 窗, 兜底几乎不触发。这里同时校验短序列下 stats 仍不同
    # (随机相位起点不同), 以证明差异确实来自 traj_id 派生的 RNG。
    d3 = bb.build_duty_series(lib, CFG, traj_id=4, seed=7, n_windows=20)
    assert not np.array_equal(d1["stats"]["omega_rms"], d3["stats"]["omega_rms"]), \
        "不同 traj_id 得到同一工况统计"
    long1 = bb.build_duty_series(lib, CFG, traj_id=3, seed=7, n_windows=600)
    long3 = bb.build_duty_series(lib, CFG, traj_id=4, seed=7, n_windows=600)
    assert not np.array_equal(long1["mode_id"], long3["mode_id"]), \
        "不同 traj_id 得到同一任务序列"


def test_duty_to_sim_inputs_contract():
    """三序列长度一致、有限、omega_cmd 幅值 = omega_rms。"""
    lib = _lib()
    duty = bb.build_duty_series(lib, CFG, traj_id=0, seed=7, n_windows=12)
    params = {"J": 0.03}
    si = bb.duty_to_sim_inputs(duty, params, 0.07957747154594767)
    assert set(si.keys()) == {"omega_cmd", "domega_dt", "T_cmd"}
    n = len(duty["mode_id"])
    for k, v in si.items():
        assert len(v) == n, f"{k} 长度 {len(v)} != {n}"
        assert np.isfinite(v).all(), f"{k} 含 NaN/Inf"
    assert np.allclose(np.abs(si["omega_cmd"]), duty["stats"]["omega_rms"])
    assert np.allclose(si["T_cmd"], params["J"] * si["domega_dt"])


def test_load_profile_library_rejects_labeled_file(tmp_path):
    """§0: 若 profiles.h5 声明含标签, load_profile_library 必须拒绝加载。"""
    import h5py
    p = tmp_path / "bad.h5"
    with h5py.File(p, "w") as f:
        f.attrs["dt_s"] = 1.0
        f.attrs["contains_rul_label"] = True
    with pytest.raises(ValueError, match="§0"):
        bb.load_profile_library(p, 0)
