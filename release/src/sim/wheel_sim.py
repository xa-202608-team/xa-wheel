"""sim/wheel_sim.py

飞轮退化仿真器 — 物理 + 退化注入, 参数 YAML 化, 固定种子。

动力学 (plan §四 Phase 3):
  J·dω/dt = T_cmd - T_f
  T_f = T_c·sgn(ω) + b(t)·ω
  I_m = (T_cmd + T_f) / K_t        电机电流 (惯性需求 + 克服摩擦)
  b(t) = b_0·[1 + Δ·(1 - e^(-t/τ))]   黏性摩擦指数退化

注入: 温度 Arrhenius 调制 τ; 小概率润滑突变 (b 阶跃); 量测扰动分两路独立建模 ——
      逐点白噪声 (noise_ratio, 必须 size=N 逐点采样) + 每轨迹固定量测直流偏置
      (meas_bias_ratio, 传感器标定误差)。二者混为一谈会让 b̂ 携带逐轨迹随机常偏。
失效: |I_m| > Im_rated -> label_fail=1, 记录 EOL 时刻。
稳态跟踪偏差随退化增大 (omega 列)。

输出: data/simulated/wheel/sim_v1/seed_42/wheel_traj_{i:03d}.csv + wheel_all.h5
列: [t, omega, I_m, T, T_cmd, b_true, label_fail]

用法:
  python -m src.sim.wheel_sim --config configs/wheel.yaml --n_traj 100
  python -m src.sim.wheel_sim --config configs/wheel.yaml --n_traj 2 --seed 7 --hash
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import load_config, set_seed            # noqa: E402

SEC_PER_YEAR = 365.25 * 24 * 3600
K_B = 8.617e-5                     # eV/K 玻尔兹曼常数 (Arrhenius)


def sample_params(rng: np.random.Generator, sim_cfg: dict) -> dict:
    """从 config 区间随机采样一条飞轮的物理参数。"""
    U = lambda a, b: float(rng.uniform(a, b))
    p = sim_cfg["physics"]
    return dict(
        J=U(*p["J_range"]),
        Kt=U(*p["Kt_range"]),
        Tc=U(*p["Tc_range"]),
        b0=U(*p["b0_range"]),
        Delta=U(*p["Delta_range"]),
        tau_years=U(*p["tau_years_range"]),
        omega0=U(1000.0, 2000.0),     # 基准转速 rad/s (反作用轮高速段)
        T_base=U(265.0, 315.0),       # 基准温度 K
        T_amp=U(5.0, 25.0),           # 热摆幅 K
        seed_traj=int(rng.integers(0, 2**31)),
    )


def simulate(params: dict, sim_cfg: dict, traj_rng: np.random.Generator,
             duty: dict | None = None):
    """单条轨迹仿真 -> (DataFrame, eol_idx, failed)。

    duty=None (默认) -> **解析工况** (sim.profile.source: analytic)。行为与
        Basilisk-v1 之前逐位一致, 由 tests/basilisk/test_dataset_isolation.py::
        test_old_analytic_path_unchanged 钉死。
    duty=dict -> **外部工况注入** (sim.profile.source: basilisk)。只替换三条工况
        输入序列 omega_cmd / domega_dt / T_cmd (由 src/sim/basilisk_bridge.py 从
        1 s Basilisk 数据按 30 min 窗聚合而来), 下游的温度、Arrhenius、b(t) 退化、
        摩擦、电流、量测噪声、失效判据**全部走同一段代码, 公式一字未改**。
        必需键: omega_cmd, domega_dt, T_cmd (三者等长, 长度决定序列长度)。
    """
    dt = float(sim_cfg["sample_period_s"])
    T_years = float(sim_cfg["duration_years"])
    leo = float(sim_cfg["leo_period_s"])
    dist = sim_cfg["disturbance"]
    Im_rated = float(sim_cfg["failure"]["Im_rated_A"])

    if duty is None:
        t_s = np.arange(0, T_years * SEC_PER_YEAR, dt)
        # 指令转速: 基准 + LEO 周期姿态扰动 (2%)
        omega_cmd = params["omega0"] * (1.0 + 0.02 * np.sin(2 * np.pi * t_s / leo))
        domega_dt = (params["omega0"] * 0.02 * 2 * np.pi / leo
                     * np.cos(2 * np.pi * t_s / leo))
        T_cmd = params["J"] * domega_dt                     # 惯性指令力矩
    else:
        for k in ("omega_cmd", "domega_dt", "T_cmd"):
            if k not in duty:
                raise KeyError(f"duty 缺键 {k!r} (basilisk_bridge 契约)")
        omega_cmd = np.asarray(duty["omega_cmd"], dtype=np.float64)
        domega_dt = np.asarray(duty["domega_dt"], dtype=np.float64)
        T_cmd = np.asarray(duty["T_cmd"], dtype=np.float64)
        n = len(omega_cmd)
        if len(domega_dt) != n or len(T_cmd) != n:
            raise ValueError("duty 三序列长度必须一致")
        t_s = np.arange(n, dtype=np.float64) * dt
    t_years = t_s / SEC_PER_YEAR

    # 温度 (轨道热周期 ~90 min)
    T_kelvin = params["T_base"] + params["T_amp"] * np.sin(2 * np.pi * t_s / (90 * 60))

    # b(t): 指数退化, Arrhenius 温度加速
    Ea = float(dist["arrhenius_Ea_eV"])
    Tref = float(dist["arrhenius_T_ref_K"])
    accel = np.exp(-Ea / K_B * (1.0 / T_kelvin - 1.0 / Tref))
    integ = np.cumsum(accel) * dt / SEC_PER_YEAR          # 等效退化年数
    b = params["b0"] * (1.0 + params["Delta"] * (1.0 - np.exp(-integ / params["tau_years"])))

    # 小概率润滑突变 (b 阶跃上跳)
    if traj_rng.random() < float(dist["lube_spike_prob"]):
        spike_t = int(traj_rng.integers(1, len(t_s) - 1))
        b[spike_t:] *= float(traj_rng.uniform(1.3, 1.8))

    # 摩擦 / 力矩 / 电流
    sgn = np.sign(omega_cmd)
    Tf = params["Tc"] * sgn + b * omega_cmd
    Im = (T_cmd + Tf) / params["Kt"]

    # 稳态跟踪偏差 (退化越大, omega 偏离指令越多)
    err = 0.001 * params["omega0"] * (b / params["b0"] - 1.0)
    omega = omega_cmd - err

    # ---- 量测噪声 ----
    # 分两部分独立建模 (两路传感器彼此独立):
    #   (a) 逐点白噪声 noise_ratio: 必须传 size, 否则 normal 返回标量 ->
    #       整条轨迹被同一个常数平移, 等效于一个随机直流偏置而非噪声;
    #   (b) 每条轨迹固定的量测直流偏置 meas_bias_ratio: 传感器安装/标定误差,
    #       与时间无关, 单独显式建模, 不与 (a) 混淆。
    n_pts = len(t_s)
    nr = float(dist["noise_ratio"])
    br = float(dist["meas_bias_ratio"])
    omega_fs = float(np.max(np.abs(omega)))          # 满量程 (噪声/偏置的标定基准)
    Im_fs = float(np.max(np.abs(Im)))
    omega_bias = float(traj_rng.normal(0.0, br * omega_fs))
    Im_bias = float(traj_rng.normal(0.0, br * Im_fs))
    omega = omega + omega_bias + traj_rng.normal(0.0, nr * omega_fs, size=n_pts)
    Im = Im + Im_bias + traj_rng.normal(0.0, nr * Im_fs, size=n_pts)

    # 失效
    fail_mask = np.abs(Im) > Im_rated
    persistence = max(1, int(sim_cfg["failure"].get("persistence_samples", 1)))
    label_fail = np.zeros(len(t_s), dtype=np.int8)
    if persistence == 1:
        persistent_fail = fail_mask
    else:
        run = np.convolve(fail_mask.astype(np.int16),
                          np.ones(persistence, dtype=np.int16), mode="valid")
        persistent_fail = np.zeros_like(fail_mask)
        persistent_fail[persistence - 1:] = run >= persistence
    if persistent_fail.any():
        fi = int(np.argmax(persistent_fail)) - persistence + 1
        label_fail[fi:] = 1
        eol_idx = fi
        failed = True
    else:
        eol_idx = len(t_s) - 1
        failed = False

    df = pd.DataFrame({
        "t": t_s, "omega_cmd": omega_cmd, "omega": omega, "I_m": Im, "T": T_kelvin,
        "T_cmd": T_cmd, "b_true": b, "label_fail": label_fail,
    })
    return df, eol_idx, failed


def _hash_dfs(dfs) -> str:
    h = hashlib.sha256()
    for df in dfs:
        h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel.yaml")
    ap.add_argument("--n_traj", type=int, default=None)
    ap.add_argument("--out", default="data/simulated/wheel/sim_v1/seed_42")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--hash", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed, cfg["reproducibility"]["deterministic"],
             cfg["reproducibility"]["cudnn_benchmark"])
    sim_cfg = cfg["sim"]
    n_traj = args.n_traj or int(sim_cfg["n_traj"])
    rng = np.random.default_rng(seed)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs, summaries, params_list = [], [], []
    for i in range(n_traj):
        params = sample_params(rng, sim_cfg)
        traj_rng = np.random.default_rng(params["seed_traj"])
        df, eol, failed = simulate(params, sim_cfg, traj_rng)
        df.to_csv(out_dir / f"wheel_traj_{i:03d}.csv", index=False)
        dfs.append(df)
        params_list.append(params)
        summaries.append(dict(
            idx=i, failed=bool(failed), eol_idx=eol,
            b0=params["b0"], b_end=float(df["b_true"].iloc[-1]),
            Im_max=float(np.max(np.abs(df["I_m"].values))),
            omega0=params["omega0"], J=params["J"], Kt=params["Kt"], Tc=params["Tc"],
            tau=params["tau_years"], Delta=params["Delta"],
        ))

    import h5py
    with h5py.File(out_dir / "wheel_all.h5", "w") as f:
        for i, df in enumerate(dfs):
            g = f.create_group(f"traj_{i:03d}")
            for c in df.columns:
                g.create_dataset(c, data=df[c].values)
            # 参数元数据 (P4 build_hi 用真实 Kt/Tc 辨识 b̂ 验证跟踪 b_true)
            for k, v in params_list[i].items():
                g.attrs[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
    # 参数汇总 (备 build_hi / 复现审计)
    import json
    (out_dir / "params.json").write_text(json.dumps(
        [{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
          for k, v in p.items()} for p in params_list], indent=2), encoding="utf-8")

    n_fail = sum(s["failed"] for s in summaries)
    print(f">> 生成 {n_traj} 条轨迹 -> {out_dir}  (序列长 {len(dfs[0])})")
    print(f">> 失效轨迹 {n_fail}/{n_traj} ({100*n_fail/n_traj:.0f}%)")
    print(">> 参数分布:")
    for k in ["J", "tau", "Delta", "omega0", "b0"]:
        v = np.array([s[k] for s in summaries])
        print(f"   {k:8s}: [{v.min():.4g}, {v.max():.4g}]  mean={v.mean():.4g}")
    eols = np.array([s["eol_idx"] for s in summaries])
    print(f">> EOL index: min={eols.min()} max={eols.max()} mean={eols.mean():.0f}")
    if args.hash:
        print(f">> repro hash (seed={seed}, n_traj={n_traj}): {_hash_dfs(dfs)[:16]}")


if __name__ == "__main__":
    main()
