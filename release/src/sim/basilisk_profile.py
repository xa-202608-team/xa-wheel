"""sim/basilisk_profile.py

Basilisk-v1 §4–§7 —— 五种任务模式的工况库生成 (Basilisk 原生动力学)。

分工边界 (§0 铁律):
  Basilisk 只负责 —— 姿态动力学 / 姿态控制 / 反作用轮转速 / 控制力矩 / 任务模式 /
                     动量卸载工况
  Basilisk 绝不产生 —— HI / failure threshold / EOL / RUL / 任何监督标签
  本模块的输出只有: time_s, sigma_BN, omega_BN_B, wheel_speed_rad_s, motor_torque_Nm,
                   commanded_torque(body), 以及派生的 duty statistics。
  本模块**不 import** src.sim.build_hi / src.models / 任何 RUL 相关代码 (由
  tests/basilisk/test_bridge.py::test_basilisk_never_generates_rul 钉死)。

五模式的差异是**FSW 拓扑与工况机理层面的差异**, 不是同一正弦换幅值 (§5 禁令):
  cruise  : inertial3D 固定惯性指向 + 低控制增益 + 小扰动 -> 轮速近常值
  nadir   : hillPoint 轨道坐标系对地指向 + Earth 引力 + 真实 LEO 轨道
            -> 轮速随轨道周期起伏 (参考姿态本身在惯性系里以轨道角速度旋转)
  imaging : inertial3D 参考姿态**分段跳变** (多次 slew + settle) -> 明显加减速与力矩峰
  desat   : 持续单向外部扰动力矩使轮动量线性积累, 周期性 dump 事件施加反向
            body torque, 控制器用轮子对抗 -> 轮动量释放斜坡 (§5 允许 external
            torque abstraction, 不建磁力矩器硬件模型)
  safe    : 低控制带宽 (小 K / 大 P) + 更大扰动 + 不同指向 -> 轮速/力矩统计与 nominal 不同

所有模块均为 Basilisk 官方模块 (spacecraft / reactionWheelStateEffector /
inertial3D / hillPoint / simpleNav / attTrackingError / mrpFeedback /
rwMotorTorque / extForceTorque / gravBodyFactory / rwFactory), 未替换任何动力学。

随机数: 全部由 config seed 派生 (sha256("basilisk_profile|{seed}|{mode}|{run}")),
禁止全局 np.random.seed()。
"""
from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np

MODES = ("cruise", "nadir", "imaging", "desat", "safe")

# duty_stats 的键顺序 (§7); audit / bridge / 测试共用这一份定义
SPEED_STATS = ("mean_abs", "rms", "p95_abs", "max_abs")
TORQUE_STATS = ("mean_abs", "rms", "p95_abs", "max_abs")


def profile_rng(seed: int, mode: str, run: int) -> np.random.Generator:
    """逐 (mode, run) 确定性 RNG。与 wiener_pf.pf_rng 同一构造范式。"""
    h = hashlib.sha256(f"basilisk_profile|{int(seed)}|{mode}|{int(run)}".encode())
    return np.random.default_rng(int.from_bytes(h.digest()[:8], "big") % (2 ** 63))


# --------------------------------------------------------------------------
# Basilisk 场景搭建
# --------------------------------------------------------------------------
def _build_scenario(cfg_sc: dict, need_orbit: bool, dt_s: float,
                    rng: np.random.Generator, gains: dict, n_rw: int):
    """搭一个三轴姿态控制卫星。返回 (scSim, handles)。

    need_orbit=True 时挂 Earth 引力 + LEO 轨道 (nadir / desat 的对地指向必需);
    False 时纯姿态动力学 (cruise / imaging / safe 的惯性指向不需要轨道)。
    """
    from Basilisk.utilities import SimulationBaseClass, macros, simIncludeRW, \
        simIncludeGravBody, orbitalMotion, simHelpers
    from Basilisk.simulation import spacecraft, reactionWheelStateEffector, \
        simpleNav, extForceTorque
    from Basilisk.fswAlgorithms import inertial3D, hillPoint, attTrackingError, \
        mrpFeedback, rwMotorTorque
    from Basilisk.architecture import messaging

    scSim = SimulationBaseClass.SimBaseClass()
    step = macros.sec2nano(dt_s)
    proc = scSim.CreateNewProcess("bskProcess")
    proc.addTask(scSim.CreateNewTask("dynTask", step))
    proc.addTask(scSim.CreateNewTask("fswTask", step))

    # ---- hub ----
    I = list(np.asarray(cfg_sc["I_hub"], dtype=float).ravel())
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "leoSat"
    scObject.hub.mHub = float(cfg_sc["mass_kg"])
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    scObject.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(I)
    s0 = rng.uniform(-0.05, 0.05, 3)
    w0 = rng.uniform(-2e-4, 2e-4, 3)
    scObject.hub.sigma_BNInit = [[s0[0]], [s0[1]], [s0[2]]]
    scObject.hub.omega_BN_BInit = [[w0[0]], [w0[1]], [w0[2]]]
    scSim.AddModelToTask("dynTask", scObject)

    gravFactory = None
    if need_orbit:
        gravFactory = simIncludeGravBody.gravBodyFactory()
        earth = gravFactory.createEarth()
        earth.isCentralBody = True
        gravFactory.addBodiesTo(scObject)
        oe = orbitalMotion.ClassicElements()
        oe.a = float(cfg_sc["orbit"]["sma_m"])
        oe.e = float(cfg_sc["orbit"]["ecc"])
        oe.i = np.radians(float(cfg_sc["orbit"]["inc_deg"]))
        oe.Omega = np.radians(float(rng.uniform(0.0, 360.0)))
        oe.omega = np.radians(float(rng.uniform(0.0, 360.0)))
        oe.f = np.radians(float(rng.uniform(0.0, 360.0)))
        rN, vN = orbitalMotion.elem2rv(earth.mu, oe)
        scObject.hub.r_CN_NInit = rN
        scObject.hub.v_CN_NInit = vN

    # ---- 反作用轮 (官方 Honeywell_HR16, 3 正交 + 1 斜置) ----
    rwFactory = simIncludeRW.rwFactory()
    axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [1 / np.sqrt(3)] * 3][:n_rw]
    rw_objs = []
    om_bias = float(cfg_sc["rw"]["initial_speed_bias_rad_s"])
    om_spread = float(cfg_sc["rw"]["initial_speed_spread_rad_s"])
    for ax in axes:
        om = float(rng.normal(om_bias, om_spread)) * float(rng.choice([-1.0, 1.0]))
        rw_objs.append(rwFactory.create(
            "Honeywell_HR16", ax,
            maxMomentum=float(cfg_sc["rw"]["max_momentum_Nms"]),
            Omega=om * 60.0 / (2 * np.pi)))          # rwFactory 的 Omega 单位是 RPM
    rwEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwFactory.addToSpacecraft("RW", rwEffector, scObject)
    scSim.AddModelToTask("dynTask", rwEffector, 2)

    # ---- 外部扰动力矩 (常值 + 模式相关, 运行中可被 dump 事件改写) ----
    extFT = extForceTorque.ExtForceTorque()
    extFT.ModelTag = "extDisturbance"
    scObject.addDynamicEffector(extFT)
    scSim.AddModelToTask("dynTask", extFT)

    # ---- 导航 ----
    sNav = simpleNav.SimpleNav()
    sNav.ModelTag = "simpleNav"
    sNav.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    scSim.AddModelToTask("dynTask", sNav)

    # ---- 参考姿态生成器 ----
    if need_orbit:
        refMod = hillPoint.hillPoint()
        refMod.ModelTag = "hillPoint"
        refMod.transNavInMsg.subscribeTo(sNav.transOutMsg)
    else:
        refMod = inertial3D.inertial3D()
        refMod.ModelTag = "inertial3D"
        refMod.sigma_R0N = [0.0, 0.0, 0.0]
    scSim.AddModelToTask("fswTask", refMod)

    trkErr = attTrackingError.attTrackingError()
    trkErr.ModelTag = "attTrackingError"
    trkErr.attRefInMsg.subscribeTo(refMod.attRefOutMsg)
    trkErr.attNavInMsg.subscribeTo(sNav.attOutMsg)
    scSim.AddModelToTask("fswTask", trkErr)

    ctrl = mrpFeedback.mrpFeedback()
    ctrl.ModelTag = "mrpFeedback"
    ctrl.K = float(gains["K"])
    ctrl.P = float(gains["P"])
    ctrl.Ki = -1.0                     # <0 = 关积分 (官方写法)
    ctrl.integralLimit = 0.0
    ctrl.guidInMsg.subscribeTo(trkErr.attGuidOutMsg)
    ctrl.rwSpeedsInMsg.subscribeTo(rwEffector.rwSpeedOutMsg)
    scSim.AddModelToTask("fswTask", ctrl)

    alloc = rwMotorTorque.rwMotorTorque()
    alloc.ModelTag = "rwMotorTorque"
    alloc.controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    alloc.vehControlInMsg.subscribeTo(ctrl.cmdTorqueOutMsg)
    scSim.AddModelToTask("fswTask", alloc)
    rwEffector.rwMotorCmdInMsg.subscribeTo(alloc.rwMotorTorqueOutMsg)

    vehCfg = messaging.VehicleConfigMsgPayload()
    vehCfg.ISCPntB_B = I
    vcMsg = messaging.VehicleConfigMsg().write(vehCfg)
    ctrl.vehConfigInMsg.subscribeTo(vcMsg)
    rwParamMsg = rwFactory.getConfigMessage()
    ctrl.rwParamsInMsg.subscribeTo(rwParamMsg)
    alloc.rwParamsInMsg.subscribeTo(rwParamMsg)

    # ---- 记录器 (dt = 1 s, §11 要求 1 s 原始分辨率) ----
    # motor torque 取 rwOutMsgs[i].u_current —— RW 效应器**实际施加**的电机力矩
    # (已含 u_max 饱和与摩擦模型)。不取 rwMotorTorque 分配器的输出: 那是饱和前的
    # 期望值, 用它算 duty statistics 会高估真实力矩负荷。
    recs = {
        "rw": rwEffector.rwSpeedOutMsg.recorder(),
        "trq": alloc.rwMotorTorqueOutMsg.recorder(),
        "cmd": ctrl.cmdTorqueOutMsg.recorder(),
        "sc": scObject.scStateOutMsg.recorder(),
        "err": trkErr.attGuidOutMsg.recorder(),
    }
    rw_recs = [rwEffector.rwOutMsgs[i].recorder() for i in range(len(axes))]
    for r in list(recs.values()) + rw_recs:
        scSim.AddModelToTask("fswTask", r)

    # handles 必须持有 rwFactory / vcMsg / rwParamMsg / trkErr / sNav 的 Python 引用。
    # 它们是 SWIG 包装的 C++ 对象, Python 侧引用计数归零即析构, 而 Basilisk 的订阅端
    # 只存裸指针 -> 函数返回后消息内存被释放, GsMatrix_B 首行会读到野内存, 表现为
    # 第 0 号轮的 motor torque 恒为 ~5e-313 的非物理值 (实测已复现)。
    handles = dict(sc=scObject, rw=rwEffector, ext=extFT, ref=refMod, ctrl=ctrl,
                   alloc=alloc, recs=recs, n_rw=len(axes),
                   Js=[float(o.Js) for o in rw_objs],
                   gsHat=np.array(axes, dtype=float),
                   gravFactory=gravFactory, macros=macros, rw_recs=rw_recs,
                   _keepalive=(rwFactory, rw_objs, vcMsg, rwParamMsg, trkErr, sNav))
    return scSim, handles


def _wheel_momentum(h: dict) -> np.ndarray:
    """当前各轮角动量在 body 系的合矢量 h_w = Σ Js_i·Ω_i·gs_i (仅用于 desat 卸载律)。"""
    om = np.asarray(h["rw"].rwSpeedOutMsg.read().wheelSpeeds[: h["n_rw"]], dtype=float)
    return (np.asarray(h["Js"]) * om) @ h["gsHat"]


# --------------------------------------------------------------------------
# 五种模式的 run 函数
# --------------------------------------------------------------------------
def _run_segmented(scSim, h: dict, duration_s: float, dt_s: float,
                   seg_s: float, on_segment: Callable[[int, float], None]):
    """按段推进仿真; 每段开始前调用 on_segment(seg_idx, t_now) 改写参考姿态/外扰。

    分段执行 (ConfigureStopTime + ExecuteSimulation 反复调用) 是 Basilisk 官方
    多阶段 scenario 的标准写法, 不改动任何动力学。
    """
    macros = h["macros"]
    n_seg = max(1, int(round(duration_s / seg_s)))
    scSim.InitializeSimulation()
    for k in range(n_seg):
        t_now = k * seg_s
        on_segment(k, t_now)
        scSim.ConfigureStopTime(macros.sec2nano(min((k + 1) * seg_s, duration_s)))
        scSim.ExecuteSimulation()


def _mode_spec(mode: str, mcfg: dict) -> dict:
    """取该模式的配置块 (含增益 / 扰动 / 分段参数)。"""
    if mode not in mcfg:
        raise KeyError(f"configs 缺少 sim.profile.mode_params.{mode}")
    return mcfg[mode]


def generate_run(mode: str, run: int, cfg: dict, seed: int) -> dict:
    """跑一条 (mode, run) 工况, 返回原始 1 s 时序 dict。"""
    if mode not in MODES:
        raise ValueError(f"未知 mission mode: {mode}")
    pcfg = cfg["sim"]["profile"]
    dt_s = float(pcfg["dt_s"])
    if abs(dt_s - 1.0) > 1e-12:
        # §11: Basilisk 必须 1 s 分辨率, 30 min 聚合在 bridge 里做
        raise ValueError(f"sim.profile.dt_s 必须为 1.0 (§11), 当前 {dt_s}")
    duration_s = float(pcfg["run_duration_s"])
    spec = _mode_spec(mode, pcfg["mode_params"])
    rng = profile_rng(seed, mode, run)
    need_orbit = mode in ("nadir", "desat")

    scSim, h = _build_scenario(cfg["sim"]["profile"]["spacecraft"], need_orbit,
                              dt_s, rng, spec["gains"], int(pcfg["n_rw"]))
    ext, ref = h["ext"], h["ref"]
    tau_dist = float(spec["disturbance_torque_Nm"])
    base_dist = rng.normal(0.0, tau_dist, 3)

    # ---- 逐模式的分段回调 ----
    events: list[dict] = []

    if mode == "cruise":
        # 固定惯性指向; 扰动为缓变常值 (每段轻微漂移), 无机动
        seg_s = float(spec["segment_s"])

        def on_segment(k, t):
            d = base_dist + rng.normal(0.0, 0.1 * tau_dist, 3)
            ext.extTorquePntB_B = [[d[0]], [d[1]], [d[2]]]

    elif mode == "nadir":
        # hillPoint 对地指向; 参考姿态本身随轨道旋转 -> 轮速周期起伏
        seg_s = float(spec["segment_s"])

        def on_segment(k, t):
            d = base_dist + rng.normal(0.0, 0.2 * tau_dist, 3)
            ext.extTorquePntB_B = [[d[0]], [d[1]], [d[2]]]

    elif mode == "imaging":
        # 参考姿态分段跳变 (slew) + 保持 (settle); slew 幅角大 -> 力矩峰值高
        seg_s = float(spec["segment_s"])
        slew_deg = spec["slew_amplitude_deg"]
        settle_segs = int(spec["settle_segments"])

        def on_segment(k, t):
            d = base_dist + rng.normal(0.0, 0.1 * tau_dist, 3)
            ext.extTorquePntB_B = [[d[0]], [d[1]], [d[2]]]
            if k % (settle_segs + 1) == 0:
                ang = np.radians(rng.uniform(*slew_deg))
                axis = rng.normal(0.0, 1.0, 3)
                axis /= np.linalg.norm(axis)
                s = np.tan(ang / 4.0) * axis          # MRP <- 主旋转角/轴
                ref.sigma_R0N = [float(s[0]), float(s[1]), float(s[2])]
                events.append(dict(t=t, kind="slew", angle_deg=float(np.degrees(ang))))

    elif mode == "desat":
        # 单向常值扰动 -> 轮动量线性积累; 周期性 dump 事件施加反向 body torque,
        # 控制器用轮子对抗 -> 轮速被拉回 (明显的 momentum release 斜坡)
        seg_s = float(spec["segment_s"])
        secular = rng.normal(0.0, 1.0, 3)
        secular = secular / np.linalg.norm(secular) * float(spec["secular_torque_Nm"])
        dump_every = int(spec["dump_every_segments"])
        dump_segs = int(spec["dump_duration_segments"])
        dump_gain = float(spec["dump_gain"])
        state = {"in_dump": 0}

        def on_segment(k, t):
            if state["in_dump"] > 0:
                state["in_dump"] -= 1
                hw = _wheel_momentum(h)
                nrm = np.linalg.norm(hw)
                if nrm > 1e-12:
                    d = -dump_gain * hw / nrm * float(spec["dump_torque_Nm"])
                else:
                    d = np.zeros(3)
                ext.extTorquePntB_B = [[d[0]], [d[1]], [d[2]]]
                return
            if k > 0 and k % dump_every == 0:
                state["in_dump"] = dump_segs - 1
                events.append(dict(t=t, kind="dump_start"))
                hw = _wheel_momentum(h)
                nrm = np.linalg.norm(hw)
                d = (-dump_gain * hw / nrm * float(spec["dump_torque_Nm"])
                     if nrm > 1e-12 else np.zeros(3))
                ext.extTorquePntB_B = [[d[0]], [d[1]], [d[2]]]
                return
            d = secular + rng.normal(0.0, 0.05 * tau_dist, 3)
            ext.extTorquePntB_B = [[d[0]], [d[1]], [d[2]]]

    elif mode == "safe":
        # 低控制带宽 (小 K / 大 P, 见 config) + 更大扰动 + 固定安全指向
        seg_s = float(spec["segment_s"])
        ang = np.radians(float(spec["safe_pointing_deg"]))
        axis = rng.normal(0.0, 1.0, 3)
        axis /= np.linalg.norm(axis)
        s = np.tan(ang / 4.0) * axis
        ref.sigma_R0N = [float(s[0]), float(s[1]), float(s[2])]

        def on_segment(k, t):
            d = base_dist + rng.normal(0.0, 0.5 * tau_dist, 3)
            ext.extTorquePntB_B = [[d[0]], [d[1]], [d[2]]]
    else:                                                  # pragma: no cover
        raise AssertionError(mode)

    _run_segmented(scSim, h, duration_s, dt_s, seg_s, on_segment)

    recs, n_rw = h["recs"], h["n_rw"]
    n = len(recs["rw"].times())
    # 实际施加的电机力矩 (含饱和), 逐轮拼成 (N, n_rw)
    u_actual = np.column_stack([np.asarray(r.u_current, dtype=float).ravel()
                                for r in h["rw_recs"]])
    out = {
        "time_s": np.asarray(recs["rw"].times(), dtype=float) * 1e-9,
        "wheel_speed_rad_s": np.asarray(recs["rw"].wheelSpeeds)[:, :n_rw].astype(float),
        "motor_torque_Nm": u_actual,
        "commanded_rw_torque_Nm": np.asarray(recs["trq"].motorTorque)[:, :n_rw].astype(float),
        "commanded_torque": np.asarray(recs["cmd"].torqueRequestBody).astype(float),
        "sigma_BN": np.asarray(recs["sc"].sigma_BN).astype(float),
        "omega_BN_B": np.asarray(recs["sc"].omega_BN_B).astype(float),
        "sigma_BR": np.asarray(recs["err"].sigma_BR).astype(float),
    }
    if h["gravFactory"] is not None:
        h["gravFactory"].unloadSpiceKernels()
    out["_meta"] = dict(mode=mode, run=run, dt_s=dt_s, n_samples=n,
                        n_rw=n_rw, n_events=len(events))
    out["_events"] = events
    return out


# --------------------------------------------------------------------------
# duty statistics (§7)
# --------------------------------------------------------------------------
def duty_stats_of(speed: np.ndarray, torque: np.ndarray, dt_s: float,
                  maneuver_torque_frac: float, high_torque_frac: float,
                  u_max_Nm: float) -> dict:
    """单条 run 的工况统计。speed/torque 形状 (N,) —— 单轮口径。

    maneuver_fraction / high_torque_fraction 的阈值都以**电机额定力矩 u_max** 为
    基准并走 config, 不硬编码绝对数值 (否则换轮型号后统计口径失效)。
    """
    sp = np.asarray(speed, dtype=float).ravel()
    tq = np.asarray(torque, dtype=float).ravel()
    if sp.size == 0 or tq.size == 0:
        raise ValueError("duty_stats_of 收到空序列")
    a_sp, a_tq = np.abs(sp), np.abs(tq)
    # zero crossing: 轮速穿零次数 (动量方向反转, 轴承润滑膜最恶劣的工况)
    s = np.sign(sp)
    nz = s[s != 0]
    zc = int(np.count_nonzero(np.diff(nz) != 0)) if nz.size > 1 else 0
    return {
        "speed": {"mean_abs": float(a_sp.mean()), "rms": float(np.sqrt((sp ** 2).mean())),
                  "p95_abs": float(np.percentile(a_sp, 95)), "max_abs": float(a_sp.max())},
        "torque": {"mean_abs": float(a_tq.mean()), "rms": float(np.sqrt((tq ** 2).mean())),
                   "p95_abs": float(np.percentile(a_tq, 95)), "max_abs": float(a_tq.max())},
        "zero_crossing_count": zc,
        "maneuver_fraction": float(np.mean(a_tq > maneuver_torque_frac * u_max_Nm)),
        "high_torque_fraction": float(np.mean(a_tq > high_torque_frac * u_max_Nm)),
        # §7: 当前场景未建轨道遮蔽模型 -> 明确 null, 禁止伪造
        "eclipse_fraction": None,
        "n_samples": int(sp.size),
        "dt_s": float(dt_s),
    }
