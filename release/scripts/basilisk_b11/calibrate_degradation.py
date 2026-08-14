"""scripts/basilisk_b11/calibrate_degradation.py

Basilisk-B1.1 §3/§4 —— 峰值口径 (p95) 对齐的退化重标定。

本模块是 B1.1 的唯一标定逻辑来源, 被 audit_peak_reference / generate_candidates /
generate_dataset / 测试共同 import。

--------------------------------------------------------------------------------
与 B1 的唯一差异 (§3)
--------------------------------------------------------------------------------
B1 用 **window-RMS** 作为标定参考:
    speed_util  = |omega_rms|  / Omega_rated
    torque_util = |torque_rms| / u_max
而失效判据是 **pointwise peak**: |I_m(t)| > Im_rated 连续 persistence_samples 点。

Basilisk 五模式工况的波峰因数 (p95/rms ≈ 1.42) 远高于旧 analytic 正弦工况
(≈ 1.02), 于是 RMS 等效的 b0 标定在 peak 判据下过度消耗初始电流裕度 ——
B1 三个 candidate 的 failure_fraction 0.883/0.917/0.933 全部冲出 [0.35, 0.80]。

B1.1 只把**标定参考的口径**换成 p95:
    speed_util_p95  = p95(|omega|)  / Omega_rated
    torque_util_p95 = p95(|torque|) / u_max

关键事实: `omega_p95` / `torque_p95` 是 `src/sim/basilisk_bridge.aggregate_window`
的**既有输出** (DUTY_KEYS 成员), 因此换口径**不需要修改任何契约冻结代码** ——
basilisk_bridge.py / wheel_sim.py 的 hash 保持不变。

--------------------------------------------------------------------------------
明确不改 (由 tests/basilisk_b11 钉死)
--------------------------------------------------------------------------------
* wear 数学结构: `g_duty` / `wear_rate_multiplier` / `G_DUTY_MIN` / `b0_scale` /
  `apply_calibration` / `params_rng` **直接从 B1 模块 import**, 本文件不重新实现。
  这不是"风格选择"而是强制手段: 结构无法在 B1.1 侧被悄悄改写。
* g_duty 权重与指数: 来自 configs/wheel_basilisk_b11.yaml, 与 B1 config 逐值相同。
* mode weights / HR16 rated / failure criterion: 全部继承, b11 config 不出现。
* 仿真输入仍是 rms 口径 (`duty_to_sim_inputs` 的 omega_cmd = sign·omega_rms) ——
  §3 只换"标定参考尺度", 不换"喂给退化模型的工作点"。这一点很容易搞混:
  改仿真输入等于改 wear model, 是被禁止的。
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load_b1_module():
    """按路径加载 B1 的 calibrate_degradation (注册为独立模块名)。

    为什么按路径而不是 `import calibrate_degradation`: B1 与 B1.1 的模块同名,
    两个目录都会被 sys.path.insert, 直接 import 会命中先到的那一份。
    注册成 `b1_calibrate_degradation` 与本模块共存, 互不覆盖。
    """
    name = "b1_calibrate_degradation"
    if name in sys.modules:
        return sys.modules[name]
    p = ROOT / "scripts" / "basilisk_b1" / "calibrate_degradation.py"
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(mod)
    return mod


_B1 = _load_b1_module()

# ---- 从 B1 原样复用的部分 (退化结构, 一行都不改写) ----
DRIVE_KEYS = _B1.DRIVE_KEYS
G_DUTY_MIN = _B1.G_DUTY_MIN
rated_values_from_basilisk = _B1.rated_values_from_basilisk
check_rated_against_config = _B1.check_rated_against_config
b0_scale = _B1.b0_scale
g_duty = _B1.g_duty
wear_rate_multiplier = _B1.wear_rate_multiplier
apply_calibration = _B1.apply_calibration
base_n_windows = _B1.base_n_windows
calibration_protocol_hash = _B1.calibration_protocol_hash

# ---- B1.1 自己的部分 ----
# 逐轨迹 RNG 的命名空间刻意与 B1 **相同** ("basilisk_b1_params|...") ——
# 这样 B1.1 的第 i 条轨迹与 B1 的第 i 条拥有逐值相同的初始物理参数, B1 vs B1.1
# 成为严格配对比较, 口径切换的效应不被随机数差异污染 (§8 允许沿用 B1 seed)。
params_rng = _B1.params_rng


def load_b11_config(path: str | Path = "configs/wheel_basilisk_b11.yaml") -> dict:
    """递归解析 base_config 继承链 (b11 -> wheel_basilisk -> wheel)。

    复用 B1 的递归实现 (src/utils/basilisk_config.load_basilisk_config 只解析一层,
    会丢掉 wheel.yaml 的 sim.physics / sim.failure / sim.hi)。
    """
    return _B1.load_b1_config(path)


def _stat_keys(cfg: dict) -> tuple[str, str]:
    """从 config 读出本次标定使用的 duty 统计量名 (p95 口径)。"""
    rc = cfg["reference"]
    return str(rc["speed_stat"]), str(rc["torque_stat"])


def utilization_from_duty(duty: dict, rated: dict, n_per_window: int,
                          cfg: dict) -> dict:
    """duty stats -> 四个逐窗无量纲驱动量 (**p95 口径**)。

    与 B1 的差异只有两行: `omega_rms -> omega_p95`, `torque_rms -> torque_p95`,
    且统计量名从 config 读, 不硬编码 —— 想验证旧口径只需把
    reference.caliber 改成 rms, 无需改代码 (审计脚本正是这么做 A/B 对比的)。

    zero_crossing_rate / maneuver_fraction 不受口径影响 (非幅值量), 定义与 B1 同。
    """
    sk, tk = _stat_keys(cfg)
    st = duty["stats"]
    if sk not in st or tk not in st:
        raise KeyError(f"duty stats 缺少 {sk}/{tk}; 可用键 {sorted(st.keys())}")
    om_r = float(rated["omega_rated_rad_s"])
    tq_r = float(rated["torque_rated_Nm"])
    return {
        "speed_util": np.abs(st[sk]) / om_r,
        "torque_util": np.abs(st[tk]) / tq_r,
        "zero_crossing_rate": st["zero_crossing_count"] / float(n_per_window),
        "maneuver_fraction": st["maneuver_fraction"].astype(np.float64),
    }


def reference_duty(lib: dict, rated: dict, n_per_window: int,
                   maneuver_torque_frac: float, cfg: dict,
                   caliber: str | None = None) -> dict:
    """reference duty: 冻结 profiles.h5 上"全模式等权"的窗级驱动量均值。

    窗聚合规则与 B1 **完全一致** (非重叠整窗切分, 各 mode 等权平均), 只有幅值
    统计量的口径可选:
      caliber='p95' (默认, 由 config 给出) -> omega_p95 / torque_p95
      caliber='rms' (仅审计对比用)          -> omega_rms / torque_rms

    铁律: reference 完全由已冻结的 profile 库决定, 不含任何可调参数。
    **不得根据 candidate 的 failure fraction 反向调整** (§3)。
    """
    from src.sim.basilisk_bridge import MODES, aggregate_window

    rc = cfg["reference"]
    cal = str(caliber or rc["caliber"])
    if cal == "p95":
        sk, tk = str(rc["speed_stat"]), str(rc["torque_stat"])
    elif cal == "rms":
        sk, tk = str(rc["speed_stat_rms"]), str(rc["torque_stat_rms"])
    else:
        raise ValueError(f"未知 caliber={cal!r} (只允许 p95 / rms)")

    om_r = float(rated["omega_rated_rad_s"])
    tq_r = float(rated["torque_rated_Nm"])
    per_mode: dict[str, dict] = {}
    for m in MODES:
        acc: dict[str, list] = {k: [] for k in DRIVE_KEYS}
        for r in lib[m]:
            sp, tq = r["speed"], r["torque"]
            n_win = int(sp.size // n_per_window)
            if n_win < 1:
                raise ValueError(f"profile run {r['key']} 不足一个窗 {n_per_window}")
            for w in range(n_win):
                sl = slice(w * n_per_window, (w + 1) * n_per_window)
                agg = aggregate_window(sp[sl], tq[sl], maneuver_torque_frac, tq_r)
                acc["speed_util"].append(abs(agg[sk]) / om_r)
                acc["torque_util"].append(abs(agg[tk]) / tq_r)
                acc["zero_crossing_rate"].append(
                    agg["zero_crossing_count"] / float(n_per_window))
                acc["maneuver_fraction"].append(agg["maneuver_fraction"])
        per_mode[m] = {k: float(np.mean(v)) for k, v in acc.items()}
        per_mode[m]["n_windows"] = int(len(acc["speed_util"]))

    ref = {k: float(np.mean([per_mode[m][k] for m in MODES])) for k in DRIVE_KEYS}
    ref["per_mode"] = per_mode
    ref["n_per_window"] = int(n_per_window)
    ref["aggregate"] = str(cfg["reference"]["aggregate"])
    ref["caliber"] = cal
    ref["speed_stat"] = sk
    ref["torque_stat"] = tk
    return ref


# ---------------------------------------------------------------------------
# §8 common-random-number 对齐的轨迹构建
# ---------------------------------------------------------------------------
def prepare_context(cfg: dict, horizon_scales) -> dict:
    """一次性准备三 candidate 共用的上下文 (profile 库 / p95 reference / b0_scale)。

    同时算出 rms reference 供审计对比 (`ref_rms`) —— 只用于报告 ratio, 不参与标定。
    """
    from src.sim.basilisk_bridge import load_profile_library
    from src.utils.config import PROJECT_ROOT

    pcfg = cfg["sim"]["profile"]
    rated = rated_values_from_basilisk(
        float(pcfg["spacecraft"]["rw"]["max_momentum_Nms"]))
    checks = check_rated_against_config(rated, cfg)
    bad = [n for n, ok, _ in checks if not ok]
    if bad:
        raise SystemExit(f"!! rated 值与 config 不一致 {bad}; 禁止手工凑数, 停止")
    lib = load_profile_library(PROJECT_ROOT / cfg["paths"]["profile_h5"],
                               int(pcfg["bridge"]["wheel_index"]))
    n_per_window = int(round(float(cfg["sim"]["sample_period_s"])
                             / float(pcfg["dt_s"])))
    mtf = float(pcfg["duty"]["maneuver_torque_frac"])
    ref = reference_duty(lib, rated, n_per_window, mtf, cfg, caliber="p95")
    ref_rms = reference_duty(lib, rated, n_per_window, mtf, cfg, caliber="rms")
    # §4: b0_scale 由 p95 reference 推导, 不硬编码任何数值
    calib = b0_scale(cfg, rated, ref["speed_util"])
    calib_rms = b0_scale(cfg, rated, ref_rms["speed_util"])
    n_base = base_n_windows(cfg)
    # §8 CRN: 一次按**最长** horizon 生成 duty, 短 candidate 取前缀
    n_max = int(round(n_base * max(float(s) for s in horizon_scales)))
    return {"rated": rated, "rated_checks": checks, "lib": lib,
            "ref": ref, "ref_rms": ref_rms,
            "calib": calib, "calib_rms": calib_rms,
            "n_per_window": n_per_window, "n_base": n_base, "n_max": n_max}


def slice_duty(d: dict, n: int) -> dict:
    """取 duty 序列前 n 窗 (逐位相同的前缀 -> CRN 对齐)。"""
    return {"mode_id": d["mode_id"][:n],
            "mode_weights": d["mode_weights"],
            "stats": {k: v[:n] for k, v in d["stats"].items()},
            "n_basilisk_samples": int(n * d["n_per_window"]),
            "n_per_window": d["n_per_window"]}


def build_trajectory(traj_id: int, seed: int, cfg: dict, ctx: dict,
                     horizon_scale: float, duty_cache: dict | None = None):
    """构建一条 B1.1 轨迹 -> (df, eol_idx, failed, params, duty, drives)。

    CRN 契约 (§8): 给定 (seed, traj_id), 以下与 horizon_scale **无关**:
      * 原始物理参数 (params_rng)
      * duty 序列的前 n_base 个窗 (按 n_max 生成后截断)
      * 退化尺度 b0 / tau_years / omega0 (drives 一律在 base horizon 前缀上算)
    唯一差异 = 实际仿真窗数 n_win = round(n_base · horizon_scale)。
    """
    import numpy as _np
    from src.sim import basilisk_bridge as bb
    from src.sim.wheel_sim import sample_params, simulate

    n_base, n_max = ctx["n_base"], ctx["n_max"]
    n_win = int(round(n_base * float(horizon_scale)))
    if n_win > n_max:
        raise ValueError(f"n_win={n_win} > n_max={n_max}; prepare_context 的 "
                         "horizon_scales 未覆盖本 candidate")

    params_raw = sample_params(params_rng(seed, traj_id), cfg["sim"])

    key = int(traj_id)
    if duty_cache is not None and key in duty_cache:
        duty_max = duty_cache[key]
    else:
        duty_max = bb.build_duty_series(ctx["lib"], cfg, traj_id, seed, n_max)
        if duty_cache is not None:
            duty_cache[key] = duty_max

    duty = slice_duty(duty_max, n_win)

    # 退化尺度: drives 固定在 base horizon 前缀上算 -> A/B/C 完全同尺度
    drives_base = utilization_from_duty(slice_duty(duty_max, n_base),
                                        ctx["rated"], ctx["n_per_window"], cfg)
    params = apply_calibration(params_raw, drives_base, ctx["ref"], cfg,
                               ctx["rated"], ctx["calib"])

    # 仿真输入仍走 rms 口径的 omega_cmd (契约冻结的 duty_to_sim_inputs, 未改)
    sim_in = bb.duty_to_sim_inputs(duty, params,
                                   ctx["rated"]["rotor_inertia_Js_kgm2"])
    traj_rng = _np.random.default_rng(params["seed_traj"])
    sim_params = {k: v for k, v in params.items() if not k.startswith("_")}
    df, eol, failed = simulate(sim_params, cfg["sim"], traj_rng, duty=sim_in)

    drives = utilization_from_duty(duty, ctx["rated"], ctx["n_per_window"], cfg)
    return df, eol, failed, params, duty, drives


def paired_trajectory_hash(seed: int, cfg: dict, ctx: dict, n_traj: int) -> str:
    """§8 CRN 证据: 与 horizon 无关的共享输入 (params + base-horizon duty) 的 hash。

    三个 candidate 必须给出**完全相同**的值 —— 不同即 CRN 对齐被破坏。
    """
    h = hashlib.sha256()
    from src.sim import basilisk_bridge as bb
    from src.sim.wheel_sim import sample_params
    for i in range(int(n_traj)):
        p = sample_params(params_rng(seed, i), cfg["sim"])
        for k in sorted(p.keys()):
            h.update(f"{k}={p[k]!r};".encode())
        d = bb.build_duty_series(ctx["lib"], cfg, i, seed, ctx["n_base"])
        h.update(np.ascontiguousarray(d["mode_id"], dtype=np.int64).tobytes())
        for k in sorted(d["stats"].keys()):
            h.update(np.ascontiguousarray(d["stats"][k],
                                          dtype=np.float64).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# §6 Gate 12: 初始电流裕度
# ---------------------------------------------------------------------------
def initial_margin_ratio(df, cfg: dict) -> float:
    """单条轨迹的初始电流裕度比 = p95(|I_m| 在初始健康窗) / Im_rated。

    健康窗长 = gate.initial_margin_healthy_frac (必须等于 sim.hi.healthy_frac,
    由 test 钉死), 至少 1 个样本。

    用窗内 p95 而非单点极值 (§6 明令): 单点极值会被一次瞬态支配, 不反映群体裕度。
    """
    frac = float(cfg["gate"]["initial_margin_healthy_frac"])
    im_rated = float(cfg["sim"]["failure"]["Im_rated_A"])
    a = np.abs(np.asarray(df["I_m"].values, dtype=np.float64))
    n = max(1, int(round(a.size * frac)))
    return float(np.percentile(a[:n], 95) / im_rated)
