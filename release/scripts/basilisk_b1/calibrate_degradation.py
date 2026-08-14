"""scripts/basilisk_b1/calibrate_degradation.py

Basilisk-B1 §2/§4 —— 物理一致的量纲标定 (无量纲轮利用率驱动的磨损时钟)。

本模块是 B1 的**唯一**标定逻辑来源, 被 generate_candidates.py / generate_dataset.py /
audit_physics.py / 测试共同 import。不重复实现, 防两份数值漂移。

--------------------------------------------------------------------------------
为什么需要标定 (根因)
--------------------------------------------------------------------------------
旧 analytic 退化仿真隐含一个绝对轮速尺度: sample_params 里 omega0 ~ U(1000, 2000)
rad/s, 而 configs/wheel.yaml 的 sim.failure.omega_ref = 1500 (= 该区间均值) 正是
b0_range 的标定基准。失效判据是 |I_m| > Im_rated, 而

    I_m = (T_cmd + T_c·sgn(ω) + b(t)·ω) / K_t

主导项是黏性摩擦 b(t)·ω。Basilisk 官方 Honeywell_HR16 的 Omega_max 只有
628.32 rad/s, 实测五模式工况的窗 rms 轮速约 0.356·Omega_max ≈ 224 rad/s。
把 b0_range 原样搬过来, b(t)·ω 系统性缩小约 6.7 倍, 电流几乎到不了 Im_rated ——
这就是 BASILISK-V1 只有 5.33% failure fraction 的根因, 与"退化模型对不对"无关,
纯粹是**跨轮型量纲不一致**。

--------------------------------------------------------------------------------
标定原则 (§2/§4)
--------------------------------------------------------------------------------
1. 一切额定值来自 Basilisk rwFactory 的 Honeywell_HR16 定义 (Omega_max / u_max /
   Js), 不手工凑数。由 rated_values_from_basilisk() 运行时读取。
2. 退化**结构**一字不动: 仍是 wheel.yaml 的
        b(t) = b0·[1 + Δ·(1 - e^{-integ/τ})],  integ = ∫accel(T)dt
   B1 只改"尺度参数如何适配 HR16", 且全部改动发生在**调用侧参数**上
   (b0 / tau_years / omega0 三个 sample_params 输出), src/sim/wheel_sim.py
   一个字节都不改 (baseline_contract 已冻结它的 hash)。
3. 驱动量改为**无量纲轮利用率**:
        speed_util  = |ω_window_rms| / Omega_rated
        torque_util = torque_window_rms / u_max
   绝不再把 analytic 的 1000–2000 rad/s 当作 HR16 的绝对轮速尺度。

--------------------------------------------------------------------------------
两个标定量
--------------------------------------------------------------------------------
(A) b0 量纲重标定 —— 保持"额定电流预算中被黏性摩擦占用的份额"这一**无量纲**不变量。

    定义无量纲摩擦裕度消耗率
        beta(b, ω) = b·ω / (K_t,nom·I_m,rated - T_c,nom)
    即"当前轮速下黏性摩擦力矩占掉多少失效力矩预算"。这是唯一可跨轮型搬运的量:
    它同时含轮速与电流预算, 无量纲。

    旧设计的 beta 区间由 (b0_range, omega_ref) 读出 —— 注意 omega_ref **只用来
    反解旧设计的无量纲意图**, 不作为 HR16 的绝对轮速。再用 HR16 的实测工作点
    (speed_util_ref · Omega_rated) 重新有量纲化:

        b0_scale = omega_ref_design / (speed_util_ref · Omega_rated)

    speed_util_ref 由**已冻结的 profiles.h5** 实测算出 (reference_duty()),
    不是可调旋钮。三个 candidate 共用同一个 b0_scale。

(B) 磨损时钟 duty 调制 —— 退化速率随工况严酷度提高。

        g_duty = w_s·(su/su_ref)^p_s + w_t·(tu/tu_ref)^p_t
                 + w_z·(zcr/zcr_ref) + w_m·(mf/mf_ref)
        Σw = 1  =>  g_duty(reference duty) = 1.0
        tau_years_B1 = tau_years_raw / clip(g_duty, g_min, g_max)

    因为 db/dt = (b0·Δ/τ)·e^{-integ/τ}·accel(T), 有 db/dt ∝ 1/τ ∝ g_duty, 于是
    退化速率对 speed_util / torque_util / zero-crossing rate / maneuver fraction
    **单调递增**, 对温度经 accel(T) 单调递增, 且恒为正 (所有项非负, g_duty ≥ g_min > 0)。
    最恶劣工况下被 g_max 限成 nominal 的有限倍 (§4 要求)。

    指数取值的物理依据:
      p_s = 2.0  黏性摩擦功耗 P = b·ω², 磨损率随摩擦功耗增长;
      p_t = 1.0  Archard 磨损率随接触载荷线性增长;
      w_z / w_m 只给 0.10 —— 穿零与机动意味着边界润滑/启停工况, 已知加速轴承润滑
                 退化, 但缺公开定量依据, 故权重小 (见 docs/basilisk_b1/limitations.md)。

已知简化 (记录在 limitations, 不掩盖):
  g_duty 按**整条轨迹**取窗平均后调制 tau_years, 而非逐窗时变调制。原因: 逐窗调制
  需要改 wheel_sim.py 里的 integ 累加, 而该文件在 B1 被契约冻结 (hash 必须不变)。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def load_b1_config(path: str | Path = "configs/wheel_basilisk_b1.yaml") -> dict:
    """递归解析 base_config 继承链 -> 深度合并后的配置。

    为什么不用 src/utils/basilisk_config.load_basilisk_config: 它只解析**一层**
    base_config (wheel_basilisk.yaml -> wheel.yaml 够用)。B1 是两层链
    (wheel_basilisk_b1 -> wheel_basilisk -> wheel), 单层解析会丢掉 wheel.yaml 的
    sim.physics / sim.failure / sim.hi。该文件被 v1 脚本共用, 不去改它 (避免波及
    已冻结的 BASILISK-V1 产物), 在 B1 侧本地递归解析。
    """
    from src.utils.config import PROJECT_ROOT, load_config
    from src.utils.basilisk_config import deep_merge

    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    chain, cur = [], p
    seen = set()
    while True:
        rel = cur.resolve().as_posix()
        if rel in seen:
            raise ValueError(f"base_config 出现循环: {rel}")
        seen.add(rel)
        d = load_config(cur)
        base_rel = d.pop("base_config", None)
        chain.append((cur, d))
        if base_rel is None:
            break
        cur = PROJECT_ROOT / base_rel
    cfg: dict = {}
    for cur, d in reversed(chain):          # 从最底层 base 开始逐层覆盖
        cfg = deep_merge(cfg, d)
    cfg["_config_path"] = p.relative_to(PROJECT_ROOT).as_posix()
    cfg["_config_chain"] = [c.relative_to(PROJECT_ROOT).as_posix()
                            for c, _ in chain]
    return cfg


# 参与 g_duty 的四个无量纲驱动量 (顺序固定, 报告/测试共用)
DRIVE_KEYS = ("speed_util", "torque_util", "zero_crossing_rate", "maneuver_fraction")

# g_duty 下限: 空转轮不磨损是对的, 但 tau -> inf 会让退化完全消失, 数值上无意义。
G_DUTY_MIN = 0.05


def rated_values_from_basilisk(max_momentum_Nms: float) -> dict:
    """从 Basilisk 官方 rwFactory 读 Honeywell_HR16 额定值 (§2: 不手工凑数)。

    maxMomentum 必须与 sim.spacecraft.rw.max_momentum_Nms 一致 —— rwFactory 要求
    调用方显式给出, 且它决定 Js (Js = maxMomentum / Omega_max)。
    """
    from Basilisk.utilities import simIncludeRW

    f = simIncludeRW.rwFactory()
    o = f.create("Honeywell_HR16", [1.0, 0.0, 0.0],
                 maxMomentum=float(max_momentum_Nms), Omega=0.0)
    return {
        "source": "basilisk_rwFactory_Honeywell_HR16",
        "max_momentum_Nms": float(max_momentum_Nms),
        "omega_rated_rad_s": float(o.Omega_max),
        "torque_rated_Nm": float(o.u_max),
        "rotor_inertia_Js_kgm2": float(o.Js),
        "rotor_mass_kg": float(o.mass),
        "u_s_Nm": float(o.U_s),
        "u_d_Nm": float(o.U_d),
        "u_min_Nm": float(o.u_min),
    }


def check_rated_against_config(rated: dict, cfg: dict) -> list[tuple[str, bool, str]]:
    """config 里记录的额定值必须与运行时读数一致; 不一致即视为手工凑数。"""
    rc = cfg["rated"]
    tol = float(rc["tol_rel"])
    pairs = [("omega_rated_rad_s", "omega_rated_rad_s"),
             ("torque_rated_Nm", "torque_rated_Nm"),
             ("rotor_inertia_Js_kgm2", "rotor_inertia_Js_kgm2"),
             ("rotor_mass_kg", "rotor_mass_kg"),
             ("u_s_Nm", "u_s_Nm"), ("u_d_Nm", "u_d_Nm")]
    out = []
    for ck, rk in pairs:
        exp, got = float(rc[ck]), float(rated[rk])
        ok = abs(got - exp) <= tol * max(abs(exp), 1e-30)
        out.append((ck, ok, f"config={exp!r} basilisk={got!r}"))
    out.append(("rated.source", str(rc["source"]) == str(rated["source"]),
                f"{rc['source']} vs {rated['source']}"))
    return out


# ---------------------------------------------------------------------------
# 无量纲驱动量
# ---------------------------------------------------------------------------
def utilization_from_duty(duty: dict, rated: dict, n_per_window: int) -> dict:
    """duty stats -> 四个逐窗无量纲驱动量。

    speed_util 用窗 rms (与 basilisk_bridge.duty_to_sim_inputs 喂给退化模型的
    omega_cmd 同口径; 摩擦功耗 ∝ b·ω², rms 才是等效损耗转速)。
    """
    st = duty["stats"]
    om_r = float(rated["omega_rated_rad_s"])
    tq_r = float(rated["torque_rated_Nm"])
    return {
        "speed_util": np.abs(st["omega_rms"]) / om_r,
        "torque_util": np.abs(st["torque_rms"]) / tq_r,
        "zero_crossing_rate": st["zero_crossing_count"] / float(n_per_window),
        "maneuver_fraction": st["maneuver_fraction"].astype(np.float64),
    }


def utilization_from_arrays(speed: np.ndarray, torque: np.ndarray,
                            rated: dict) -> dict:
    """§3 审计用: 直接对 1 s 原始序列算逐点利用率 (不做窗聚合)。"""
    return {
        "speed_util": np.abs(speed) / float(rated["omega_rated_rad_s"]),
        "torque_util": np.abs(torque) / float(rated["torque_rated_Nm"]),
    }


def reference_duty(lib: dict, rated: dict, n_per_window: int,
                   maneuver_torque_frac: float) -> dict:
    """reference duty: 冻结 profiles.h5 上"全模式等权"的窗级驱动量均值。

    口径固定为对每个 mode 的每条 run 做**非重叠**整窗切分 (floor(N/n_per_window) 窗),
    各 mode 等权平均。这样 reference 完全由已冻结的 profile 库决定, 不含任何可调参数
    —— 它不是旋钮, 改它就等于改 profile 库, 而 profile 库 hash 已冻结。
    """
    from src.sim.basilisk_bridge import MODES, aggregate_window

    per_mode: dict[str, dict] = {}
    for m in MODES:
        acc = {k: [] for k in DRIVE_KEYS}
        for r in lib[m]:
            sp, tq = r["speed"], r["torque"]
            n_win = int(sp.size // n_per_window)
            if n_win < 1:
                raise ValueError(f"profile run {r['key']} 不足一个窗 {n_per_window}")
            for w in range(n_win):
                sl = slice(w * n_per_window, (w + 1) * n_per_window)
                agg = aggregate_window(sp[sl], tq[sl], maneuver_torque_frac,
                                       float(rated["torque_rated_Nm"]))
                acc["speed_util"].append(abs(agg["omega_rms"])
                                         / float(rated["omega_rated_rad_s"]))
                acc["torque_util"].append(abs(agg["torque_rms"])
                                          / float(rated["torque_rated_Nm"]))
                acc["zero_crossing_rate"].append(agg["zero_crossing_count"]
                                                 / float(n_per_window))
                acc["maneuver_fraction"].append(agg["maneuver_fraction"])
        per_mode[m] = {k: float(np.mean(v)) for k, v in acc.items()}
        per_mode[m]["n_windows"] = int(len(acc["speed_util"]))

    ref = {k: float(np.mean([per_mode[m][k] for m in MODES])) for k in DRIVE_KEYS}
    ref["per_mode"] = per_mode
    ref["n_per_window"] = int(n_per_window)
    ref["aggregate"] = "non_overlapping_windows_equal_weight_per_mode"
    return ref


# ---------------------------------------------------------------------------
# (A) b0 量纲重标定
# ---------------------------------------------------------------------------
def b0_scale(cfg: dict, rated: dict, speed_util_ref: float) -> dict:
    """b0 量纲重标定系数 (三 candidate 共用)。

    b0_scale = omega_ref_design / (speed_util_ref · Omega_rated)

    omega_ref_design = configs/wheel.yaml 的 sim.failure.omega_ref, 仅用于**反解旧
    设计的无量纲摩擦裕度意图**, 不作为 HR16 的绝对轮速 (§4 禁令)。
    """
    fc = cfg["sim"]["failure"]
    om_design = float(fc["omega_ref"])
    om_rated = float(rated["omega_rated_rad_s"])
    om_op = float(speed_util_ref) * om_rated
    if om_op <= 0:
        raise ValueError("speed_util_ref·Omega_rated <= 0, 无法标定")
    s = om_design / om_op
    # 无量纲摩擦裕度预算 (只做报告, 不参与计算)
    budget = float(fc["Kt_nom"]) * float(fc["Im_rated_A"]) - float(fc["Tc_nom"])
    b0_lo, b0_hi = [float(v) for v in cfg["sim"]["physics"]["b0_range"]]
    return {
        "b0_scale": float(s),
        "omega_ref_design_rad_s": om_design,
        "omega_operating_rad_s": om_op,
        "omega_rated_rad_s": om_rated,
        "speed_util_ref": float(speed_util_ref),
        "friction_budget_Nm": float(budget),
        "b0_range_original": [b0_lo, b0_hi],
        "b0_range_b1": [b0_lo * s, b0_hi * s],
        "beta_range_design": [b0_lo * om_design / budget, b0_hi * om_design / budget],
        "beta_range_b1": [b0_lo * s * om_op / budget, b0_hi * s * om_op / budget],
        "rationale": ("保持无量纲摩擦裕度消耗率 beta = b·ω/(Kt_nom·Im_rated-Tc_nom) "
                      "不变; omega_ref 仅用于反解旧设计意图, 不作 HR16 绝对尺度"),
    }


# ---------------------------------------------------------------------------
# (B) 磨损时钟 duty 调制
# ---------------------------------------------------------------------------
def g_duty(drives: dict, ref: dict, wd_cfg: dict) -> np.ndarray:
    """逐窗磨损驱动倍数 g_duty (在 reference duty 处 = 1.0)。

    恒非负 (四项均非负), 单调递增于每个驱动量, 上下限 clip 到 [G_DUTY_MIN, g_duty_max]。
    """
    floor = float(wd_cfg["ref_floor"])
    ws = float(wd_cfg["speed_util_weight"])
    wt = float(wd_cfg["torque_util_weight"])
    wz = float(wd_cfg["zero_crossing_weight"])
    wm = float(wd_cfg["maneuver_weight"])
    ps = float(wd_cfg["speed_util_exponent"])
    pt = float(wd_cfg["torque_util_exponent"])
    wsum = ws + wt + wz + wm
    if abs(wsum - 1.0) > 1e-9:
        raise ValueError(f"wear_drive 权重和必须为 1, 当前 {wsum}")
    for w in (ws, wt, wz, wm):
        if w < 0:
            raise ValueError("wear_drive 权重不得为负 (会产生负 wear rate)")

    def norm(key):
        return np.maximum(np.asarray(drives[key], dtype=np.float64), 0.0) \
            / max(float(ref[key]), floor)

    g = (ws * norm("speed_util") ** ps
         + wt * norm("torque_util") ** pt
         + wz * norm("zero_crossing_rate")
         + wm * norm("maneuver_fraction"))
    return np.clip(g, G_DUTY_MIN, float(wd_cfg["g_duty_max"]))


def wear_rate_multiplier(drives: dict, ref: dict, wd_cfg: dict) -> float:
    """整条轨迹的磨损速率倍数 = 窗平均 g_duty (见模块头"已知简化")。"""
    return float(np.mean(g_duty(drives, ref, wd_cfg)))


# ---------------------------------------------------------------------------
# 标定后的轨迹参数 (调用侧覆盖, 不改 wheel_sim.py)
# ---------------------------------------------------------------------------
def apply_calibration(params: dict, drives: dict, ref: dict, cfg: dict,
                      rated: dict, calib: dict) -> dict:
    """把 sample_params 的原始参数改写成 HR16 量纲一致的 B1 参数。

    只覆盖三个尺度参数, 其余 (J / Kt / Tc / Delta / T_base / T_amp / seed_traj) 原样保留:
      b0         <- b0 · b0_scale                     (量纲重标定, 与 duty 无关)
      tau_years  <- tau_years / g_duty_traj           (工况严酷度调制退化速率)
      omega0     <- speed_util_traj · Omega_rated     (取代 analytic 硬编码 1000-2000)

    omega0 在 duty 注入模式下只剩一个用处: wheel_sim 的稳态跟踪偏差项
    err = 0.001·omega0·(b/b0 - 1)。若继续用 analytic 的 ~1500, 就是把旧轮型的绝对
    轮速当 HR16 的尺度 (§4 明令禁止), 故按本轨迹实际工作点重设。
    """
    g = wear_rate_multiplier(drives, ref, cfg["wear_drive"])
    su_traj = float(np.mean(np.asarray(drives["speed_util"], dtype=np.float64)))
    out = dict(params)
    out["b0"] = float(params["b0"]) * float(calib["b0_scale"])
    out["tau_years"] = float(params["tau_years"]) / g
    out["omega0"] = su_traj * float(rated["omega_rated_rad_s"])
    out["_b1_g_duty"] = g
    out["_b1_speed_util_mean"] = su_traj
    return out


def params_rng(seed: int, traj_id: int) -> np.random.Generator:
    """逐轨迹物理参数 RNG —— 只依赖 (seed, traj_id), 与循环历史无关。

    这是 §7 common-random-number 对齐的前提: candidate A/B/C 的第 i 条轨迹必须拿到
    完全相同的初始物理参数, 不能因为 horizon 不同而错位。
    """
    h = hashlib.sha256(f"basilisk_b1_params|{int(seed)}|{int(traj_id)}".encode())
    return np.random.default_rng(int.from_bytes(h.digest()[:8], "big") % (2 ** 63))


def calibration_protocol_hash(payload: dict) -> str:
    """标定协议 hash (冻结后禁止再改)。"""
    import json
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


# ---------------------------------------------------------------------------
# §7 common-random-number 对齐的轨迹构建 (candidate / 正式数据集共用同一实现)
# ---------------------------------------------------------------------------
def base_n_windows(cfg: dict) -> int:
    """基准 horizon 的窗数 (= analytic lineage 的序列长, 保证时间轴同口径)。"""
    import numpy as _np
    from src.sim.wheel_sim import SEC_PER_YEAR
    dt = float(cfg["sim"]["sample_period_s"])
    return int(len(_np.arange(0, float(cfg["sim"]["duration_years"]) * SEC_PER_YEAR, dt)))


def prepare_context(cfg: dict, horizon_scales) -> dict:
    """一次性准备三 candidate 共用的上下文 (profile 库 / reference duty / b0_scale)。"""
    from src.sim.basilisk_bridge import load_profile_library
    from src.utils.config import PROJECT_ROOT

    pcfg = cfg["sim"]["profile"]
    rated = rated_values_from_basilisk(
        float(pcfg["spacecraft"]["rw"]["max_momentum_Nms"]))
    bad = [n for n, ok, _ in check_rated_against_config(rated, cfg) if not ok]
    if bad:
        raise SystemExit(f"!! rated 值与 config 不一致 {bad}; 禁止手工凑数, 停止")
    lib = load_profile_library(PROJECT_ROOT / cfg["paths"]["profile_h5"],
                               int(pcfg["bridge"]["wheel_index"]))
    n_per_window = int(round(float(cfg["sim"]["sample_period_s"]) / float(pcfg["dt_s"])))
    ref = reference_duty(lib, rated, n_per_window,
                         float(pcfg["duty"]["maneuver_torque_frac"]))
    calib = b0_scale(cfg, rated, ref["speed_util"])
    n_base = base_n_windows(cfg)
    # §7 CRN 关键: 一次按**最长** horizon 生成 duty, 短 candidate 取前缀 ->
    # 前缀逐位相同, 三 candidate 唯一差异真的只有 horizon。
    n_max = int(round(n_base * max(float(s) for s in horizon_scales)))
    return {"rated": rated, "lib": lib, "ref": ref, "calib": calib,
            "n_per_window": n_per_window, "n_base": n_base, "n_max": n_max}


def build_trajectory(traj_id: int, seed: int, cfg: dict, ctx: dict,
                     horizon_scale: float, duty_cache: dict | None = None):
    """构建一条 B1 轨迹 -> (df, eol_idx, failed, params_b1, duty, drives)。

    CRN 契约 (§7): 给定 (seed, traj_id), 以下三者与 horizon_scale **无关**:
      * 原始物理参数 (params_rng)
      * duty 序列的前 n_base 个窗 (按 n_max 生成后截断)
      * 退化尺度 b0 / tau_years / omega0 (drives 一律在 base horizon 前缀上算)
    唯一差异 = 实际仿真的窗数 n_win = round(n_base · horizon_scale)。
    """
    import numpy as _np
    from src.sim import basilisk_bridge as bb
    from src.sim.wheel_sim import sample_params, simulate

    n_base, n_max = ctx["n_base"], ctx["n_max"]
    n_win = int(round(n_base * float(horizon_scale)))
    if n_win > n_max:
        raise ValueError(f"n_win={n_win} > n_max={n_max}; prepare_context 的 "
                         "horizon_scales 未覆盖本 candidate")

    # 1) 原始物理参数: 只依赖 (seed, traj_id)
    params_raw = sample_params(params_rng(seed, traj_id), cfg["sim"])

    # 2) duty: 按 n_max 生成一次, 缓存复用 (三 candidate 共用同一条)
    key = int(traj_id)
    if duty_cache is not None and key in duty_cache:
        duty_max = duty_cache[key]
    else:
        duty_max = bb.build_duty_series(ctx["lib"], cfg, traj_id, seed, n_max)
        if duty_cache is not None:
            duty_cache[key] = duty_max

    def _slice(d, n):
        return {"mode_id": d["mode_id"][:n],
                "mode_weights": d["mode_weights"],
                "stats": {k: v[:n] for k, v in d["stats"].items()},
                "n_basilisk_samples": int(n * d["n_per_window"]),
                "n_per_window": d["n_per_window"]}

    duty = _slice(duty_max, n_win)

    # 3) 退化尺度: drives 固定在 base horizon 前缀上算 -> A/B/C 完全同尺度
    drives_base = utilization_from_duty(_slice(duty_max, n_base), ctx["rated"],
                                        ctx["n_per_window"])
    params_b1 = apply_calibration(params_raw, drives_base, ctx["ref"], cfg,
                                  ctx["rated"], ctx["calib"])

    # 4) 仿真 (退化物理走冻结的 src/sim/wheel_sim.simulate, 一字未改)
    sim_in = bb.duty_to_sim_inputs(duty, params_b1, ctx["rated"]["rotor_inertia_Js_kgm2"])
    traj_rng = _np.random.default_rng(params_b1["seed_traj"])
    sim_params = {k: v for k, v in params_b1.items() if not k.startswith("_")}
    df, eol, failed = simulate(sim_params, cfg["sim"], traj_rng, duty=sim_in)

    drives = utilization_from_duty(duty, ctx["rated"], ctx["n_per_window"])
    return df, eol, failed, params_b1, duty, drives


def paired_trajectory_hash(seed: int, cfg: dict, ctx: dict, n_traj: int) -> str:
    """§7 CRN 证据: 与 horizon 无关的共享输入 (params + base-horizon duty) 的 hash。

    三个 candidate 必须给出**完全相同**的值 —— 若不同, 说明 CRN 对齐被破坏。
    """
    h = hashlib.sha256()
    from src.sim.wheel_sim import sample_params
    from src.sim import basilisk_bridge as bb
    for i in range(int(n_traj)):
        p = sample_params(params_rng(seed, i), cfg["sim"])
        for k in sorted(p.keys()):
            h.update(f"{k}={p[k]!r};".encode())
        d = bb.build_duty_series(ctx["lib"], cfg, i, seed, ctx["n_base"])
        h.update(np.ascontiguousarray(d["mode_id"], dtype=np.int64).tobytes())
        for k in sorted(d["stats"].keys()):
            h.update(np.ascontiguousarray(d["stats"][k], dtype=np.float64).tobytes())
    return h.hexdigest()

