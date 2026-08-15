"""飞轮仿真物理合理性测试 (plan P3 验收第 2 条)。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config, set_seed                       # noqa: E402
from src.sim.wheel_sim import sample_params, simulate             # noqa: E402


def _make_traj(seed: int):
    cfg = load_config()
    set_seed(seed, cfg["reproducibility"]["deterministic"])
    rng = np.random.default_rng(seed)
    params = sample_params(rng, cfg["sim"])
    traj_rng = np.random.default_rng(params["seed_traj"])
    df, eol_idx, failed = simulate(params, cfg["sim"], traj_rng)
    return df, params, failed


def test_b_monotone_nondecreasing():
    """b(t) 应总体非减 (退化加深, 允许润滑突变上跳)。"""
    df, _, _ = _make_traj(1)
    diffs = np.diff(df["b_true"].values)
    # 99 分位点应 >= 0 (允许极个别数值噪声)
    assert np.quantile(diffs, 0.99) >= 0, "b(t) 应非减"


def test_b_increase_implies_Im_increase():
    """b 与 |I_m| 应强正相关 (去噪后)。"""
    df, _, _ = _make_traj(2)
    b = df["b_true"].values
    im = np.abs(df["I_m"].values)
    w = 50
    bs = pd.Series(b).rolling(w, 1).mean().values
    ims = pd.Series(im).rolling(w, 1).mean().values
    corr = np.corrcoef(bs, ims)[0, 1]
    assert corr > 0.9, f"b 与 |I_m| 相关性 {corr:.3f} 应 > 0.9"


def test_omega_tracks_command_and_b_grows():
    """omega 与基准转速同量级 (跟踪); b 末值 > 初值 (退化)。"""
    df, params, _ = _make_traj(3)
    omega = df["omega"].values
    b = df["b_true"].values
    assert b[-1] > b[0], "b 末值应大于初值"
    # omega 均值应在基准转速 ±50% 内
    assert 0.5 * params["omega0"] < np.mean(np.abs(omega)) < 1.5 * params["omega0"], \
        f"omega 均值 {np.mean(np.abs(omega)):.1f} 偏离基准 {params['omega0']}"


def test_failure_triggered_in_range():
    """多种子下应有失效轨迹 (Δ, τ 合理范围触发失效判据)。"""
    failed_any = False
    for s in range(12):
        _, _, failed = _make_traj(s)
        failed_any = failed_any or failed
    assert failed_any, "12 条种子内应至少一条失效"


def test_output_columns():
    """输出列与遥测量同构。"""
    df, _, _ = _make_traj(4)
    required = {"t", "omega", "I_m", "T", "T_cmd", "b_true", "label_fail"}
    assert required.issubset(set(df.columns)), f"缺列: {required - set(df.columns)}"


def test_seed_reproducibility():
    """同一 seed 两次仿真输出哈希一致。"""
    df1, _, _ = _make_traj(7)
    df2, _, _ = _make_traj(7)
    h1 = pd.util.hash_pandas_object(df1, index=True).values.tobytes()
    h2 = pd.util.hash_pandas_object(df2, index=True).values.tobytes()
    assert h1 == h2, "同 seed 两次仿真不一致"


# ---------- S1: 量测噪声必须逐点采样 ----------

def _meas_residual(df, params, cfg_sim):
    """重算无噪理论量, 得到 omega / I_m 的量测残差 (噪声 + 每轨迹固定偏置)。"""
    t_s = df["t"].values
    omega_cmd = df["omega_cmd"].values
    # b_true 已写入 h5/df, 用它重建无噪 omega 与 I_m (与 simulate 内部同式)
    b = df["b_true"].values
    leo = float(cfg_sim["leo_period_s"])
    domega_dt = (params["omega0"] * 0.02 * 2 * np.pi / leo
                 * np.cos(2 * np.pi * t_s / leo))
    Tf = params["Tc"] * np.sign(omega_cmd) + b * omega_cmd
    T_cmd = params["J"] * domega_dt
    Im_clean = (T_cmd + Tf) / params["Kt"]
    err = 0.001 * params["omega0"] * (b / params["b0"] - 1.0)
    omega_clean = omega_cmd - err
    return df["omega"].values - omega_clean, df["I_m"].values - Im_clean


def test_noise_is_pointwise_not_scalar():
    """量测噪声必须逐点采样 (size=N), 不能是标量常偏。

    S1 修复前 traj_rng.normal(0, sigma) 未传 size, 返回标量 -> 整条轨迹被同一常数
    平移, 残差的逐点差分恒为 0。这条断言锁住该缺陷。
    """
    df, params, _ = _make_traj(1)
    r_om, r_im = _meas_residual(df, params, load_config()["sim"])
    for name, r in [("omega", r_om), ("I_m", r_im)]:
        d = np.diff(r)
        # 相邻样本量测残差不应相等 (标量噪声下 d 恒为 0)
        assert np.all(d != 0), f"{name} 量测残差存在相邻相等 -> 噪声非逐点"
        assert r.std() > 0, f"{name} 量测残差标准差应 > 0"


def test_noise_std_matches_config_ratio():
    """逐点噪声标准差应与 config noise_ratio × 满量程 量级一致。"""
    cfg = load_config()
    dist = cfg["sim"]["disturbance"]
    nr, br = float(dist["noise_ratio"]), float(dist["meas_bias_ratio"])
    df, params, _ = _make_traj(2)
    r_om, r_im = _meas_residual(df, params, cfg["sim"])
    for name, r, fs in [("omega", r_om, np.max(np.abs(df["omega"].values))),
                        ("I_m", r_im, np.max(np.abs(df["I_m"].values)))]:
        # 去掉每轨迹固定偏置后, 残差 std 应 ≈ nr × 满量程 (±50% 容差, 满量程用有噪值估)
        s = float(np.std(r - np.mean(r)))
        lo, hi = 0.5 * nr * fs, 1.5 * nr * fs
        assert lo < s < hi, f"{name} 逐点噪声 std={s:.4g} 应落在 [{lo:.4g},{hi:.4g}]"
        # 固定偏置量级应与 meas_bias_ratio 相符 (3σ 上限)
        assert abs(float(np.mean(r))) < 4.0 * br * fs, \
            f"{name} 固定偏置 {np.mean(r):.4g} 超出 4σ={4.0*br*fs:.4g}"


def test_omega_and_Im_noise_independent():
    """omega 与 I_m 两路噪声应独立采样 (相关系数接近 0)。"""
    cfg = load_config()
    df, params, _ = _make_traj(3)
    r_om, r_im = _meas_residual(df, params, cfg["sim"])
    c = float(np.corrcoef(r_om - r_om.mean(), r_im - r_im.mean())[0, 1])
    assert abs(c) < 0.1, f"两路量测噪声相关系数 {c:.4f} 应接近 0 (独立采样)"


def test_meas_bias_varies_across_trajectories():
    """每轨迹固定量测偏置应在轨迹间变化 (是 nuisance 变量, 不是常量)。"""
    cfg = load_config()
    biases = []
    for s in range(6):
        df, params, _ = _make_traj(s)
        r_om, _ = _meas_residual(df, params, cfg["sim"])
        biases.append(float(np.mean(r_om)))
    biases = np.array(biases)
    assert biases.std() > 0, "各轨迹量测偏置应不同"


def test_config_has_meas_bias_ratio():
    """meas_bias_ratio 必须走 config, 禁止硬编码。"""
    dist = load_config()["sim"]["disturbance"]
    assert "meas_bias_ratio" in dist, "config sim.disturbance 缺 meas_bias_ratio"
    assert 0.0 <= float(dist["meas_bias_ratio"]) < 1.0
