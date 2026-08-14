"""飞轮 HI 构造测试 (plan P4 验收 + S1 自校准改造)。

直接调 build_hi 函数 (内存), 依赖 data/simulated/wheel/sim_v1/seed_42/wheel_all.h5; 未生成则 skip。

S1 起 build_features(df, sim_cfg) **不再接收 params**: 只吃遥测列 + config,
从接口层面保证不可能用到仿真真值 Kt/Tc/b0/omega0。
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config                         # noqa: E402
from src.sim.build_hi import (                            # noqa: E402
    build_features, build_hi_Tf, calibrate_healthy, identify_b_hat, rolling_std,
    failure_threshold, XT_COLS, TELEMETRY_COLS,
)

SIM_H5 = ROOT / "data" / "simulated" / "wheel" / "sim_v1" / "seed_42" / "wheel_all.h5"

# ---- 容差标定 (2026-08-07 实测 100 条轨迹, 有噪逐点口径, 见 docs/progress.md S1) ----
# b̂ vs b_true 逐点相对 MAE: mean=0.0212 median=0.0167 p95=0.0535 max=0.0912
# 阈值取 0.15 = 实测 max 的 1.6 倍。注意这是**有噪逐点**口径, 比 S0 之前的
# "去噪平滑后" 口径 (阈值 0.25) 严格得多 —— 不是为过测试放宽。
B_HAT_REL_MAE_MAX = 0.15
# Kt_hat 相对偏差 |·|: mean=0.0132 median=0.0091 p95=0.0373 max=0.0934 -> 阈值 0.15
KT_REL_DEV_MAX = 0.15

# ---- S2' 改动 5: HI 与失效判据同源性判据 ----
# HI 首达 1.0 的 index 与 EOL index 的相对偏移上限 (相对 EOL)。
# 实测 (100 轨迹 seed=42, hi_source=Tf + Tf_window=1 + Tf_fail_Kt_source=hat):
#   偏移恒为 0 (absmax=0.000000, 60/60 达标), 因阈值与分子同用 Kt_hat 与同一滑窗口径,
#   Kt_hat 作公因子严格约掉 → HI>=1 ⟺ rollmean(I_m)>=Im_rated, 与 wheel_sim 判据同源。
HI_EOL_REL_TOL = 0.02
HI_REACH_TOL = 1e-3          # HI 视为"达到 1.0"的容差 (与 build_hi/diag_split 一致)
N_TRAJ_CHECK = 100           # 同源性检查覆盖的轨迹条数上限


def _load_traj(idx: int = 0):
    """返回 (遥测 df, sim_cfg, truth)。truth 仅供测试断言, 绝不传入 build_features。"""
    if not SIM_H5.exists():
        pytest.skip(f"仿真数据未生成: {SIM_H5}")
    cfg = load_config()
    with h5py.File(SIM_H5, "r") as f:
        keys = sorted(f.keys())
        g = f[keys[idx % len(keys)]]
        df = pd.DataFrame({c: g[c][:] for c in TELEMETRY_COLS + ["b_true", "label_fail"]})
        truth = {k: float(g.attrs[k]) for k in ["Kt", "Tc", "b0", "omega0"]}
    return df, cfg["sim"], truth


def test_b_hat_tracks_b_true():
    """自校准 b̂ 应跟踪 b_true (有噪逐点相对 MAE, 非去噪口径)。"""
    df, sim, _ = _load_traj(0)
    _, b_hat, _, _, _ = build_features(df, sim)
    b_true = df["b_true"].values
    rel = np.mean(np.abs(b_hat - b_true)) / (np.mean(np.abs(b_true)) + 1e-30)
    assert rel < B_HAT_REL_MAE_MAX, \
        f"b̂ 有噪逐点相对 MAE {rel:.4f} 应 < {B_HAT_REL_MAE_MAX}"


def test_b_hat_tracks_b_true_multi_traj():
    """多条轨迹一致跟踪 (防某条侥幸通过)。"""
    rels = []
    for i in range(8):
        df, sim, _ = _load_traj(i)
        _, b_hat, _, _, _ = build_features(df, sim)
        b_true = df["b_true"].values
        rels.append(np.mean(np.abs(b_hat - b_true)) / (np.mean(np.abs(b_true)) + 1e-30))
    rels = np.array(rels)
    assert rels.max() < B_HAT_REL_MAE_MAX, \
        f"8 条轨迹 b̂ 相对 MAE 最大 {rels.max():.4f} 应 < {B_HAT_REL_MAE_MAX}"


def test_calibration_recovers_Kt():
    """健康段自校准应准确恢复 K_t (T_cmd 列独立支撑, 与真值对照仅在测试内)。"""
    devs = []
    for i in range(8):
        df, sim, truth = _load_traj(i)
        Kt_hat, _, _ = calibrate_healthy(
            df["I_m"].values, df["T_cmd"].values, df["omega_cmd"].values,
            float(sim["hi"]["healthy_frac"]))
        devs.append(abs(Kt_hat - truth["Kt"]) / truth["Kt"])
    devs = np.array(devs)
    assert devs.max() < KT_REL_DEV_MAX, \
        f"Kt_hat 相对偏差最大 {devs.max():.4f} 应 < {KT_REL_DEV_MAX}"


def test_calibration_regression_direction():
    """回归方向必须是 I_m 作因变量: 反向 (带噪量作自变量) 会因 EIV 系统性衰减 K_t。

    这条锁住 S1 的关键修复 —— 反向实测 K_t 偏 -75%, 若有人改回去应立即失败。
    """
    df, sim, truth = _load_traj(0)
    Im, tcmd = df["I_m"].values, df["T_cmd"].values
    omega, omega_cmd = df["omega"].values, df["omega_cmd"].values
    frac = float(sim["hi"]["healthy_frac"])
    Kt_fwd, _, _ = calibrate_healthy(Im, tcmd, omega_cmd, frac)
    # 反向: 把带噪 I_m / omega 当自变量解 T_cmd
    m = int(np.clip(round(frac * len(Im)), 8, len(Im)))
    A = np.stack([Im[:m], -np.sign(omega[:m]), -omega[:m]], axis=1).astype(np.float64)
    sc = np.maximum(np.linalg.norm(A, axis=0), 1e-30)
    coef, *_ = np.linalg.lstsq(A / sc, tcmd[:m].astype(np.float64), rcond=None)
    Kt_rev = float((coef / sc)[0])
    dev_fwd = abs(Kt_fwd - truth["Kt"]) / truth["Kt"]
    dev_rev = abs(Kt_rev - truth["Kt"]) / truth["Kt"]
    assert dev_fwd < dev_rev, \
        f"正向偏差 {dev_fwd:.4f} 应优于反向 {dev_rev:.4f} (EIV 衰减)"


def test_build_features_uses_no_truth():
    """build_features 调用路径不得出现真值 Kt/Tc/b0/omega0。

    三重保证:
      1. 签名只有 (df, sim_cfg) 两个参数, 无 params 入口;
      2. 传入 df 只含遥测列时仍能正常工作;
      3. 源码里 build_features 函数体不出现 params[...] / attrs 取真值的写法。
    """
    import inspect
    from src.sim import build_hi

    # 1. 签名不含 params
    sig = inspect.signature(build_features)
    assert list(sig.parameters) == ["df", "sim_cfg"], \
        f"build_features 签名应为 (df, sim_cfg), 实际 {list(sig.parameters)}"

    # 2. 只给遥测列 (剔除 b_true / label_fail) 也能跑通
    df, sim, _ = _load_traj(0)
    df_telemetry_only = df[TELEMETRY_COLS].copy()
    xT, b_hat, HI_A, HI_B, calib = build_features(df_telemetry_only, sim)
    assert xT.shape[1] == len(XT_COLS)
    # S2' 起 calib 额外携带 hi_source / Tf_fail (HI 口径元信息), 均非真值参数
    assert set(calib) == {"Kt_hat", "Tc_hat", "b0_hat", "b_fail", "hi_source", "Tf_fail"}

    # 3. 源码扫描: build_features + calibrate_healthy + identify_b_hat 函数体内
    #    不得出现真值参数名的取值写法
    forbidden = ['params["Kt"]', "params['Kt']", 'params["Tc"]', "params['Tc']",
                 'params["b0"]', "params['b0']", 'params["omega0"]', "params['omega0']",
                 'attrs["Kt"]', 'attrs["Tc"]', 'attrs["b0"]', 'attrs["omega0"]',
                 '"b_true"', "'b_true'"]
    for fn in (build_hi.build_features, build_hi.calibrate_healthy,
               build_hi.identify_b_hat, build_hi.failure_threshold):
        src = inspect.getsource(fn)
        hits = [t for t in forbidden if t in src]
        assert not hits, f"{fn.__name__} 出现真值取值 {hits}"


def test_failure_threshold_is_global_constant():
    """b_fail 全局固定, 不随轨迹变化, 且完全由 config 标称值决定。"""
    cfg = load_config()
    sim = cfg["sim"]
    bf = failure_threshold(sim)
    fc = sim["failure"]
    expect = (fc["Kt_nom"] * fc["Im_rated_A"] - fc["Tc_nom"]) / fc["omega_ref"]
    assert abs(bf - expect) < 1e-15, "b_fail 应严格等于 config 标称公式值"
    assert bf > 0, "b_fail 应为正"
    # 逐轨迹调 build_features 得到的 b_fail 必须完全相同
    vals = set()
    for i in range(5):
        df, s, _ = _load_traj(i)
        vals.add(round(build_features(df, s)[4]["b_fail"], 18))
    assert len(vals) == 1, f"b_fail 应对所有轨迹一致, 实际 {vals}"


def test_hi_B_in_range_and_monotone():
    """HI_B ∈ [0,1] 且单调非减 (causal accumulate 保证)。"""
    df, sim, _ = _load_traj(1)
    _, _, _, HI_B, _ = build_features(df, sim)
    assert -1e-6 <= HI_B.min() and HI_B.max() <= 1 + 1e-6, "HI_B 应 ∈ [0,1]"
    assert np.all(np.diff(HI_B) >= -1e-12), "HI_B 应单调非减"


def test_x_T_has_10_dims_and_new_col5():
    """x_T 10 维 (S5 追加 Tf_ratio/Tf_slope); 第 5 维 S1 起为 sigma_Im。"""
    df, sim, _ = _load_traj(2)
    xT, *_ = build_features(df, sim)
    assert xT.shape[1] == 10, f"x_T 维度 {xT.shape[1]} 应为 10"
    assert len(XT_COLS) == 10
    assert XT_COLS[4] == "sigma_Im", f"第 5 维应为 sigma_Im, 实际 {XT_COLS[4]}"
    assert XT_COLS[5] == "b_hat", f"第 6 维应为 b_hat, 实际 {XT_COLS[5]}"
    assert "Im_over_omega" not in XT_COLS, "Im_over_omega 应已被 sigma_Im 替换"
    # sigma_Im 列应等于 rolling_std(I_m, config 窗长)
    expect = rolling_std(df["I_m"].values, int(sim["hi"]["sigma_window"]))
    assert np.allclose(xT[:, 4], expect), "第 5 维应为 I_m 的 config 窗长滑窗标准差"


def test_sigma_Im_is_informative():
    """sigma_Im 非退化列: 应为正、有变化、且与 config 窗长一致 (非常数)。"""
    df, sim, _ = _load_traj(3)
    xT, *_ = build_features(df, sim)
    s = xT[:, 4]
    assert np.all(s >= 0), "标准差应非负"
    assert s.std() > 0, "sigma_Im 不应为常数列"
    assert np.all(np.isfinite(s))


def test_b_hat_is_sliding_window():
    """b̂ 应为滑窗 LSQ 而非逐点除法: 滑窗去趋势残差方差 < 逐点 (抗噪证据)。"""
    df, sim, _ = _load_traj(0)
    Im, tcmd = df["I_m"].values, df["T_cmd"].values
    om, om_cmd = df["omega"].values, df["omega_cmd"].values
    Kt_hat, Tc_hat, _ = calibrate_healthy(
        Im, tcmd, om_cmd, float(sim["hi"]["healthy_frac"]))
    win = int(sim["hi"]["lsq_window"])
    b_sliding = identify_b_hat(Im, tcmd, om, Kt_hat, Tc_hat, win)
    y = Kt_hat * Im - tcmd - Tc_hat * np.sign(om)
    b_pointwise = y / (om + 1e-20)
    seg = slice(len(b_sliding) // 4, 3 * len(b_sliding) // 4)

    def _resid_std(v):
        s = pd.Series(v[seg])
        trend = s.rolling(200, center=True, min_periods=1).mean()
        return float((s - trend).std())

    assert _resid_std(b_sliding) < _resid_std(b_pointwise), \
        "滑窗LSQ b̂ 去趋势残差应小于逐点除法 (抗噪)"


def test_no_nan_in_features():
    df, sim, _ = _load_traj(3)
    xT, b_hat, HI_A, HI_B, _ = build_features(df, sim)
    for name, arr in [("x_T", xT), ("b_hat", b_hat), ("HI_A", HI_A), ("HI_B", HI_B)]:
        assert np.all(np.isfinite(arr)), f"{name} 含 NaN/Inf"


def test_missing_telemetry_raises():
    """缺遥测列应显式报错, 而非静默走 fallback (旧版会自己合成 omega_cmd)。"""
    df, sim, _ = _load_traj(0)
    bad = df.drop(columns=["omega_cmd"])
    with pytest.raises(KeyError, match="omega_cmd"):
        build_features(bad, sim)


# ================= S2' 改动 5: HI 与失效判据同源性 =================
def _all_traj_hi_vs_eol():
    """逐条轨迹算 (HI 首达 1.0 的 index, EOL index, 是否失效)。

    EOL 与 event_observed 取自仿真 h5 的 attrs (wheel_sim 的 |I_m|>Im_rated 连续
    persistence 点判据写入), HI 由 build_features 重算 —— 两者独立来源, 才有对照意义。
    """
    if not SIM_H5.exists():
        pytest.skip(f"仿真数据未生成: {SIM_H5}")
    cfg = load_config()
    sim = cfg["sim"]
    out = []
    with h5py.File(SIM_H5, "r") as f:
        for k in sorted(f.keys())[:N_TRAJ_CHECK]:
            g = f[k]
            df = pd.DataFrame({c: g[c][:] for c in TELEMETRY_COLS})
            lf = np.asarray(g["label_fail"][:], dtype=int)
            observed = bool(g.attrs.get("event_observed", bool(lf.any())))
            # 仿真 h5 未写 eol_idx attr; EOL = label_fail 首个 1 的位置
            # (wheel_sim 里 label_fail[fi:]=1 且 eol_idx=fi, 二者等价)
            eol = int(np.argmax(lf)) if lf.any() else len(lf) - 1
            _, _, _, HI_B, _ = build_features(df, sim)
            reached = bool(HI_B.max() >= 1.0 - HI_REACH_TOL)
            first = int(np.argmax(HI_B >= 1.0 - HI_REACH_TOL)) if reached else -1
            out.append(dict(key=k, observed=observed, eol=eol,
                            reached=reached, first=first, n=len(HI_B)))
    return out


def test_hi_first_reach_aligns_with_eol():
    """全部失效轨迹: |HI 首达 1.0 的 index − EOL index| / EOL < HI_EOL_REL_TOL。

    这是 S2' 改动 1 的核心判据 —— HI 若不与失效判据同源, 监督标签 (HI) 与
    事件标签 (EOL/RUL) 会指向不同时刻, 学出来的 HI 动力学无法迁移。
    """
    rows = _all_traj_hi_vs_eol()
    failed = [r for r in rows if r["observed"]]
    assert failed, "仿真数据中无失效轨迹, 无法校核同源性"
    bad = []
    rels = []
    for r in failed:
        assert r["reached"], f"{r['key']} 已失效但 HI 未达 1.0 (max<1)"
        rel = abs(r["first"] - r["eol"]) / max(r["eol"], 1)
        rels.append(rel)
        if rel >= HI_EOL_REL_TOL:
            bad.append((r["key"], r["first"], r["eol"], rel))
    rels = np.array(rels)
    assert not bad, (
        f"{len(bad)}/{len(failed)} 条轨迹 HI 首达 1.0 与 EOL 相对偏移 >= "
        f"{HI_EOL_REL_TOL}; 实测 mean={rels.mean():.6f} max={rels.max():.6f}; "
        f"前 5 条 (key, HI首达, EOL, 相对偏移) = {bad[:5]}")


def test_hi_reach_count_equals_failed_count():
    """HI 达到 1.0 的轨迹数必须恰等于 event_observed==1 的轨迹数 (无多报无漏报)。"""
    rows = _all_traj_hi_vs_eol()
    n_reach = sum(1 for r in rows if r["reached"])
    n_obs = sum(1 for r in rows if r["observed"])
    extra = [r["key"] for r in rows if r["reached"] and not r["observed"]]
    miss = [r["key"] for r in rows if r["observed"] and not r["reached"]]
    assert n_reach == n_obs, (
        f"HI 达 1.0 轨迹数 {n_reach} != 失效轨迹数 {n_obs}; "
        f"多报 (HI达标但未失效) {extra[:5]}, 漏报 (失效但HI未达标) {miss[:5]}")


def test_hi_Tf_threshold_cancels_Kt():
    """同源性的代数根据: Tf 口径下 HI>=1 ⟺ 滑窗均值 I_m >= Im_rated, 与 Kt_hat 无关。

    把 Kt_hat 人为放大 2 倍, HI 首达 1.0 的 index 不应改变 —— 因阈值与分子同用
    Kt_hat, 该因子严格约掉。若有人把阈值改回标称 Kt_nom, 这条立即失败。
    """
    df, sim, _ = _load_traj(0)
    if str(sim["failure"]["hi_source"]) != "Tf":
        pytest.skip("当前 hi_source != 'Tf', 该代数性质不适用")
    Im, tcmd = df["I_m"].values, df["T_cmd"].values
    Kt_hat, _, _ = calibrate_healthy(
        Im, tcmd, df["omega_cmd"].values, float(sim["hi"]["healthy_frac"]))
    hi1, _, _ = build_hi_Tf(Im, tcmd, Kt_hat, sim)
    hi2, _, _ = build_hi_Tf(Im, tcmd, 2.0 * Kt_hat, sim)
    f1 = int(np.argmax(hi1 >= 1.0 - HI_REACH_TOL)) if hi1.max() >= 1.0 - HI_REACH_TOL else -1
    f2 = int(np.argmax(hi2 >= 1.0 - HI_REACH_TOL)) if hi2.max() >= 1.0 - HI_REACH_TOL else -1
    assert f1 == f2, f"Kt_hat 放大 2x 后 HI 首达点由 {f1} 变为 {f2}, 公因子未约掉"


def test_hi_source_config_switch():
    """hi_source 必须由 config 决定且只接受 'Tf'/'b'; 非法值应显式报错。"""
    df, sim, _ = _load_traj(0)
    import copy
    s = copy.deepcopy(sim)
    s["failure"]["hi_source"] = "Tf"
    hi_tf = build_features(df, s)[3]
    s["failure"]["hi_source"] = "b"
    hi_b = build_features(df, s)[3]
    assert not np.allclose(hi_tf, hi_b), "两种 hi_source 应给出不同 HI (否则分支未生效)"
    s["failure"]["hi_source"] = "bogus"
    with pytest.raises(ValueError, match="hi_source"):
        build_features(df, s)
