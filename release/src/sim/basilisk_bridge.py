"""sim/basilisk_bridge.py

Basilisk-v1 §10/§11 —— Basilisk 工况库 -> 现有飞轮退化模型的桥接层。

**这是整个任务最关键的边界。** 本模块的职责与禁令:

  职责: 从 profiles.h5 抽片段 -> 按 mode weights 拼长期任务序列 -> 在每个
        30 min 退化窗内对 1 s Basilisk 数据做统计聚合 -> 输出工况三序列
        (omega_cmd, domega_dt, T_cmd) 与 10 项工况统计。

  禁令: 本模块**不计算** HI / failure threshold / EOL / RUL / label_fail;
        **不改变**任何退化物理公式 (摩擦增长 / Arrhenius / 电流响应 / b_true 全部
        留在 src/sim/wheel_sim.py 里, 一字未改);
        **不 import** src.sim.build_hi / src.models / src.transfer。
        由 tests/basilisk/test_bridge.py::test_basilisk_never_generates_rul 钉死。

§11 铁律: Basilisk dt = 1 s -> 30 min window aggregation -> wheel degradation step。
  禁止只在 30 min 边界读一个瞬时值。实现方式: 每个退化窗对应
  n_per_window = sample_period_s / dt_s (= 1800) 个连续 Basilisk 点, 窗内做
  mean/std/rms/p95/穿零计数/机动占比 —— 机动与力矩峰值信息因此被保留。

工况 -> 退化模型的接口 (三序列, 与 wheel_sim.simulate(duty=...) 契约一致):
  omega_cmd[k] = 该窗的 rms 轮速 (符号取窗内均值符号)
      用 rms 而非 mean: 摩擦功耗 ∝ b·ω², rms 才是等效损耗转速; 用 mean 会把
      正负穿零的高损耗窗错算成低负荷 (desat 模式最明显)。
  domega_dt[k] = 该窗的 Basilisk 电机力矩均值 / J_wheel_ref, 即窗内平均角加速度
  T_cmd[k]     = params["J"] · domega_dt[k], 保持与 analytic 路径同一量纲关系
      (analytic 里 T_cmd = J·dω/dt; 这里 dω/dt 来自 Basilisk 实测力矩而非解析导数)
"""
from __future__ import annotations

import hashlib

import numpy as np

MODES = ("cruise", "nadir", "imaging", "desat", "safe")

# 每个退化窗输出的工况统计键 (§10 要求的 10 项, 顺序固定)
DUTY_KEYS = ("omega_mean", "omega_std", "omega_rms", "omega_p95",
             "torque_mean_abs", "torque_std", "torque_rms", "torque_p95",
             "zero_crossing_count", "maneuver_fraction")


def bridge_rng(seed: int, traj_id: int) -> np.random.Generator:
    """逐轨迹确定性 RNG (片段抽取 / 相位 / mode weights 全部由它派生)。"""
    h = hashlib.sha256(f"basilisk_bridge|{int(seed)}|{int(traj_id)}".encode())
    return np.random.default_rng(int.from_bytes(h.digest()[:8], "big") % (2 ** 63))


def load_profile_library(h5_path, wheel_index: int = 0) -> dict:
    """读 profiles.h5 -> {mode: [ {speed(N,), torque(N,), dt_s}, ... ]}。

    只取单轮口径 (wheel_index): 退化模型是**单轮**模型, 混多轮会让工况统计无物理对应。
    同时校验文件明确声明不含标签 (§0 边界的机器可读检查)。
    """
    import h5py
    lib: dict[str, list[dict]] = {m: [] for m in MODES}
    with h5py.File(h5_path, "r") as f:
        for flag in ("contains_rul_label", "contains_hi", "contains_eol"):
            if bool(f.attrs.get(flag, False)):
                raise ValueError(f"profiles.h5 声明 {flag}=True —— 违反 §0 分工边界")
        dt_file = float(f.attrs["dt_s"])
        if abs(dt_file - 1.0) > 1e-12:
            raise ValueError(f"profiles.h5 dt_s={dt_file} != 1.0 (§11)")
        for mode in MODES:
            if mode not in f:
                raise KeyError(f"profiles.h5 缺 mode 组 {mode}")
            for run_key in sorted(f[mode].keys()):
                g = f[f"{mode}/{run_key}"]
                sp = np.asarray(g["wheel_speed_rad_s"][:, wheel_index], dtype=np.float64)
                tq = np.asarray(g["motor_torque_Nm"][:, wheel_index], dtype=np.float64)
                lib[mode].append({"speed": sp, "torque": tq,
                                  "dt_s": float(g.attrs["dt_s"]),
                                  "key": f"{mode}/{run_key}"})
    empty = [m for m in MODES if not lib[m]]
    if empty:
        raise ValueError(f"profiles.h5 中以下 mode 无 run: {empty}")
    return lib


def sample_mode_weights(rng: np.random.Generator, wr: dict) -> dict[str, float]:
    """在 §12 冻结区间内各自均匀采样, 再归一化到和为 1。"""
    raw = {m: float(rng.uniform(*wr[m])) for m in MODES}
    s = sum(raw.values())
    if s <= 0:
        raise ValueError("mode weights 采样和为 0")
    return {m: v / s for m, v in raw.items()}


def build_mode_schedule(rng: np.random.Generator, weights: dict[str, float],
                        n_windows: int, seg_range: tuple[int, int]) -> np.ndarray:
    """生成逐窗 mode id 序列 (长度 n_windows)。

    按 weights 抽 mode, 每次连续占用 seg_windows 个窗 (从 seg_range 均匀抽),
    直到填满。这样得到的是**分段任务序列**而非逐窗独立抽样 —— 真实任务是成段的,
    逐窗独立抽会把工况打成白噪声, 30 min 尺度上的任务结构就消失了。

    强制保证每条轨迹至少用到 2 种 mode (§13 审计要求)。
    """
    lo, hi = int(seg_range[0]), int(seg_range[1])
    names = list(MODES)
    p = np.array([weights[m] for m in names], dtype=float)
    p = p / p.sum()
    out = np.empty(n_windows, dtype=np.int16)
    i = 0
    while i < n_windows:
        mi = int(rng.choice(len(names), p=p))
        seg = int(rng.integers(lo, hi + 1))
        j = min(i + seg, n_windows)
        out[i:j] = mi
        i = j
    # 至少 2 种 mode: 若退化为单一 mode, 把后半段换成次高权重的 mode
    if len(np.unique(out)) < 2 and n_windows >= 2:
        cur = int(out[0])
        alt = int(np.argsort(-p)[1]) if np.argsort(-p)[0] == cur else int(np.argsort(-p)[0])
        out[n_windows // 2:] = alt
    return out


def aggregate_window(speed: np.ndarray, torque: np.ndarray,
                     maneuver_torque_frac: float, u_max_Nm: float) -> dict:
    """把一个 30 min 窗内的 1 s 序列聚合成 10 项工况统计 (§10)。"""
    if speed.size == 0 or torque.size == 0:
        raise ValueError("aggregate_window 收到空窗")
    a_sp, a_tq = np.abs(speed), np.abs(torque)
    s = np.sign(speed)
    nz = s[s != 0]
    zc = int(np.count_nonzero(np.diff(nz) != 0)) if nz.size > 1 else 0
    return {
        "omega_mean": float(speed.mean()),
        "omega_std": float(speed.std()),
        "omega_rms": float(np.sqrt((speed ** 2).mean())),
        "omega_p95": float(np.percentile(a_sp, 95)),
        "torque_mean_abs": float(a_tq.mean()),
        "torque_std": float(torque.std()),
        "torque_rms": float(np.sqrt((torque ** 2).mean())),
        "torque_p95": float(np.percentile(a_tq, 95)),
        "zero_crossing_count": float(zc),
        "maneuver_fraction": float(np.mean(a_tq > maneuver_torque_frac * u_max_Nm)),
    }


def build_duty_series(lib: dict, cfg: dict, traj_id: int, seed: int,
                      n_windows: int) -> dict:
    """一条长寿命轨迹的完整工况序列。

    返回:
      mode_id            (n_windows,) int16  逐窗任务模式
      mode_weights       dict                本轨迹采样得到的归一化权重
      stats              {key: (n_windows,)} 10 项工况统计
      n_basilisk_samples int                 实际消耗的 1 s Basilisk 点数 (§11 证据)
      n_per_window       int                 每窗 1 s 点数 (=1800)
    """
    pcfg = cfg["sim"]["profile"]
    bcfg = pcfg["bridge"]
    dt_deg = float(cfg["sim"]["sample_period_s"])
    dt_bsk = float(pcfg["dt_s"])
    n_per_window = int(round(dt_deg / dt_bsk))
    if n_per_window < 2:
        # §11: 每窗必须聚合多个 1 s 点, 否则退化为瞬时采样
        raise ValueError(f"n_per_window={n_per_window} < 2, 违反 §11 聚合要求")

    rng = bridge_rng(seed, traj_id)
    weights = sample_mode_weights(rng, pcfg["mode_weights_range"])
    schedule = build_mode_schedule(rng, weights, n_windows,
                                   tuple(bcfg["segment_windows"]))
    mtf = float(pcfg["duty"]["maneuver_torque_frac"])
    umax = float(pcfg["duty"]["u_max_Nm"])
    random_phase = bool(bcfg["random_phase"])

    stats = {k: np.empty(n_windows, dtype=np.float64) for k in DUTY_KEYS}
    names = list(MODES)
    # 逐窗: 选中该窗的 mode -> 抽一条 run -> 抽起始相位 -> 取连续 n_per_window 点
    # (环形取用: profile run 长度 7200 s < 1800 s 窗? 不会; 但拼长序列时必须可循环)
    n_consumed = 0
    for w in range(n_windows):
        mode = names[int(schedule[w])]
        runs = lib[mode]
        r = runs[int(rng.integers(0, len(runs)))]
        sp_full, tq_full = r["speed"], r["torque"]
        n_full = sp_full.size
        if n_full < n_per_window:
            raise ValueError(
                f"profile run {r['key']} 只有 {n_full} 个 1 s 点, "
                f"不足一个退化窗 {n_per_window} 点; 请增大 sim.profile.run_duration_s")
        start = int(rng.integers(0, n_full)) if random_phase else 0
        idx = (start + np.arange(n_per_window)) % n_full      # 环形, 支持随机相位
        agg = aggregate_window(sp_full[idx], tq_full[idx], mtf, umax)
        for k in DUTY_KEYS:
            stats[k][w] = agg[k]
        n_consumed += n_per_window

    return {"mode_id": schedule, "mode_weights": weights, "stats": stats,
            "n_basilisk_samples": n_consumed, "n_per_window": n_per_window}


def duty_to_sim_inputs(duty: dict, params: dict, J_wheel_ref: float) -> dict:
    """工况统计 -> wheel_sim.simulate(duty=...) 需要的三序列。

    J_wheel_ref = 反作用轮转子惯量 (Basilisk Honeywell_HR16 的 Js), 用于把
    Basilisk 的**电机力矩**换算成**角加速度**: dω/dt = u / Js。这一步只是单位换算,
    不引入新的退化物理。

    omega_cmd 用 rms 并继承窗内均值符号 —— 理由见模块 docstring。
    """
    st = duty["stats"]
    sgn = np.sign(st["omega_mean"])
    sgn[sgn == 0] = 1.0
    omega_cmd = sgn * st["omega_rms"]
    domega_dt = st["torque_mean_abs"] * np.sign(st["torque_mean_abs"]) / max(
        float(J_wheel_ref), 1e-12)
    # torque_mean_abs 恒 >= 0, 上式等价于 /Js; 显式写 sign 是为了在换成有符号
    # 力矩统计时不改这行的语义 (若将来改用 torque_mean 则符号自动生效)。
    T_cmd = float(params["J"]) * domega_dt
    return {"omega_cmd": omega_cmd, "domega_dt": domega_dt, "T_cmd": T_cmd}
