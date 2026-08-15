"""src/sim/damage_model.py

Basilisk-B1.8 §5/§6/§9/§10/§11 —— 累计损伤退化模型。

================================================================================
这个模型是什么, 不是什么
================================================================================
**是**: 一个 `simulation degradation scenario` / `engineering-assumption
failure definition` —— 显式标注为 ASSUMED 的仿真退化场景, 用于构造透明、
可复现、物理方向正确的 RUL 研究数据。

**不是**: `manufacturer failure specification`。`D = 1` 是**项目定义的仿真
失效状态**, 不是 Honeywell / BCT / NanoAvionics / Basilisk 官方给出的硬件
失效阈值。依据 B1.6 (`B16_NO_DOCUMENTED_WHEEL`) 与 B1.7
(`B17_THERMAL_PROVENANCE_INSUFFICIENT`): 不存在可用的厂家失效阈值。

================================================================================
模型 (§5)
================================================================================
    dD/dt     = stress(t) / L_ref
    stress(t) = g_duty(t) * a_T(T(t))

  * `g_duty(t)`: 工况严酷度因子。**import** B1 的已验证实现
    (`scripts/basilisk_b1/calibrate_degradation.py::g_duty`), 权重与指数一律
    不改 (§5)。由 speed utilization / torque utilization / zero-crossing rate
    / maneuver fraction 四项加权而成, 权重和恒为 1。
  * `a_T(T)`: Arrhenius 温度加速。沿用 `src/sim/wheel_sim.py:103-105` 的
    **同一个**表达式, 不新建第二个温度加速函数 (§5)。

性质 (由 tests/basilisk_b18 钉死):
  * `stress >= 0` 处处成立 (`g_duty >= G_DUTY_MIN = 0.05`, `a_T > 0`)
  * `D(t)` 单调非降 (累加非负量)

================================================================================
reference condition (§6) —— 本模型最关键的可解释性质
================================================================================
`g_duty` 的四个权重和为 1, 每项按 `reference` 统计量归一化, 故在参考工况下
四项各为 1 -> `g_duty(reference) = 1.0` (结构恒等式, 非手填)。
`a_T(T_ref) = exp(0) = 1`。于是

    stress = 1   =>   D(t) = t / L_ref

即 **`L_ref` 就是参考工况下的设计寿命**。

================================================================================
与 B1 旧模型的关系 (§9)
================================================================================
旧式 (`wheel_sim.py:107`, 只读, 字节不变):
    b = b0 * (1 + Delta * (1 - exp(-integ / tau_years)))
新式:
    b(t) = b0 * (1 + Delta * D(t)),  D <= 1

`legacy_shape_used_as_damage_state = False`: `1 - exp(-integ/tau)` 不再作主
damage state。`tau_years` 仍被抽样以保持 CRN 逐位对齐, 但**不进入** `D(t)`。

**duty 只能接一次**: B1 把 `g_duty` 的**均值**用于缩放 `tau_years`
(`apply_calibration`); B1.8 把 `g_duty` 的**逐时刻数组**放进 `stress`。
两种接法是同一个严酷度因子的不同注入点, 同时用会让 duty 被双重计入。
因此本模块**不调用** `apply_calibration` 的 tau 缩放路径。

================================================================================
润滑突变 (§10) 与 温度角色 (§11)
================================================================================
§10: 润滑突变 `U(1.3, 1.8)` 保持原定义, 但**只乘到 `b(t)`**, 不跳变 `D`。
     这样 underlying damage 与 sudden observable friction excursion 可区分。
§11: `T(t)` 是 **environmental / telemetry condition**, **NOT** a
     physics-derived wheel thermal state。本模块不声称拥有 thermal model。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

SEC_PER_YEAR = 365.25 * 24 * 3600
K_B = 8.617e-5                     # eV/K 玻尔兹曼常数 (与 wheel_sim.py 同值)

# EOL 判据: 本阶段唯一的失效定义 (§4)
EOL_DAMAGE_THRESHOLD = 1.0

# 语义标签 —— 任何引用本模块的报告只能用这两个词
SEMANTICS = "engineering_assumption_failure_definition"
FORBIDDEN_NAMING = "manufacturer_failure_specification"
PROVENANCE = "NEW_B18_ASSUMPTION"


def _load_b1_calib():
    """按路径加载 B1 的 calibrate_degradation (g_duty 的唯一来源)。

    为什么 import 而不复制: §5 明令"不要重新调 g_duty 权重"。import 让权重
    结构在 B1.8 侧**无法**被悄悄改写 —— 这是强制手段, 不是风格选择。
    """
    name = "b18_damage_b1_calib"
    if name in sys.modules:
        return sys.modules[name]
    p = ROOT / "scripts" / "basilisk_b1" / "calibrate_degradation.py"
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {p}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(m)
    return m


_B1 = _load_b1_calib()

# 直接复用, 不重新实现 (§5)
g_duty = _B1.g_duty
G_DUTY_MIN = _B1.G_DUTY_MIN
DRIVE_KEYS = _B1.DRIVE_KEYS


# ---------------------------------------------------------------------------
# §5 温度加速: 与 wheel_sim.py 逐式相同
# ---------------------------------------------------------------------------
def arrhenius_accel(T_kelvin: np.ndarray, dist_cfg: dict) -> np.ndarray:
    """a_T(T) = exp(-Ea/k_B * (1/T - 1/T_ref))。

    与 `src/sim/wheel_sim.py:103-105` 的 `accel` 逐式相同 —— 同一个函数形式、
    同一组 config 键 (`arrhenius_Ea_eV` / `arrhenius_T_ref_K`)。
    §5 明令: 不新建第二个温度加速函数。

    恒 > 0, 因此不会让 stress 变负。
    """
    Ea = float(dist_cfg["arrhenius_Ea_eV"])
    Tref = float(dist_cfg["arrhenius_T_ref_K"])
    T = np.asarray(T_kelvin, dtype=np.float64)
    if np.any(T <= 0.0):
        raise ValueError("温度必须为正 (Kelvin); a_T 在 T<=0 处无定义")
    return np.exp(-Ea / K_B * (1.0 / T - 1.0 / Tref))


def reference_accel(dist_cfg: dict) -> float:
    """a_T 在参考温度处的值 —— 恒等于 1.0 (§6 的一半)。

    显式提供是为了让 §6 的 `stress(reference) = 1` 可以被数值断言, 而不是
    靠"读代码就知道 exp(0)=1"。
    """
    return float(arrhenius_accel(
        np.array([float(dist_cfg["arrhenius_T_ref_K"])]), dist_cfg)[0])


# ---------------------------------------------------------------------------
# §5 stress
# ---------------------------------------------------------------------------
def stress_series(drives: dict, ref: dict, wd_cfg: dict,
                  T_kelvin: np.ndarray, dist_cfg: dict) -> dict:
    """stress(t) = g_duty(t) * a_T(T(t))。

    参数
    ----
    drives : 逐窗四驱动量 (speed_util / torque_util / zero_crossing_rate /
             maneuver_fraction), 由 utilization_from_duty 给出, 长度 = 窗数。
    ref    : reference duty (冻结 profile 库决定, 无可调参数)。
    wd_cfg : wear_drive 配置 (权重/指数, §5 不改)。
    T_kelvin : 逐时刻温度序列, 长度 = 样本数 = 窗数 (本项目一窗一样本)。

    返回 dict: g / a_T / stress, 全部长度一致。

    stress >= 0 由构造保证并显式断言 —— 若断言失败说明 g_duty 或 a_T 被改坏,
    应立即停止而不是 clip 掉负值。
    """
    g = np.asarray(g_duty(drives, ref, wd_cfg), dtype=np.float64)
    aT = arrhenius_accel(T_kelvin, dist_cfg)
    if g.shape != aT.shape:
        raise ValueError(f"g_duty 长度 {g.shape} 与温度序列 {aT.shape} 不一致; "
                         "本项目一窗一样本, 二者必须等长")
    s = g * aT
    if not np.all(np.isfinite(s)):
        raise ValueError("stress 出现 NaN/Inf —— 拒绝继续 (§14 Gate 2)")
    if np.any(s < 0.0):
        raise ValueError("stress < 0 违反 §5; 不得 clip, 必须停止排查")
    return {"g_duty": g, "a_T": aT, "stress": s}


# ---------------------------------------------------------------------------
# §5/§6 damage
# ---------------------------------------------------------------------------
def damage_series(stress: np.ndarray, dt_s: float, L_ref_years: float
                  ) -> np.ndarray:
    """D(t) = cumsum(stress) * dt / SEC_PER_YEAR / L_ref。

    数值上是 `∫stress dt` 的右端点累加 (与 `wheel_sim.py:106` 的 `integ`
    同一种离散化 —— 保持一致, 不换成 trapezoid, 否则与旧产物不可比)。

    单调非降由 `stress >= 0` 保证; 这里显式断言 (§14 Gate 1)。
    """
    s = np.asarray(stress, dtype=np.float64)
    if np.any(s < 0.0):
        raise ValueError("stress < 0 -> D 不再单调, 拒绝")
    L = float(L_ref_years)
    if not (L > 0.0):
        raise ValueError(f"L_ref 必须为正, 得到 {L}")
    D = np.cumsum(s) * float(dt_s) / SEC_PER_YEAR / L
    if not np.all(np.isfinite(D)):
        raise ValueError("D 出现 NaN/Inf")
    if np.any(np.diff(D) < 0.0):
        raise ValueError("D 非单调 —— 违反 §5 要求")
    return D


def reference_damage(t_years: np.ndarray, L_ref_years: float) -> np.ndarray:
    """§6 参考工况下的解析解 D(t) = t / L_ref。

    仅供 `test_reference_damage_lifetime` 与文档使用, 不进入仿真主路径。
    """
    return np.asarray(t_years, dtype=np.float64) / float(L_ref_years)


def eol_from_damage(D: np.ndarray, threshold: float = EOL_DAMAGE_THRESHOLD
                    ) -> tuple[int, bool]:
    """EOL 只由 `D >= threshold` 产生 (§4/§14 Gate 5)。

    返回 (eol_idx, event_observed)。

    右删失 (`D_end < threshold`) 时 `event_observed = False`, `eol_idx` 取
    最后一个索引 —— 沿用 B1 口径, **不伪造 EOL**。下游 `derive_labels` 对
    删失轨迹只给 `rul_lower_bound`, 不给 `rul`。

    本函数**不接收**任何模型 prediction / test metric 参数 —— 这是结构性的
    §14 Gate 6/7 保证, 不靠约定。
    """
    d = np.asarray(D, dtype=np.float64)
    hit = d >= float(threshold)
    if hit.any():
        return int(np.argmax(hit)), True
    return int(d.size - 1), False


# ---------------------------------------------------------------------------
# §9 damage -> friction telemetry
# ---------------------------------------------------------------------------
def friction_from_damage(D: np.ndarray, b0: float, Delta: float,
                         threshold: float = EOL_DAMAGE_THRESHOLD
                         ) -> np.ndarray:
    """b(t) = b0 * (1 + Delta * D(t)), 在 D <= 1 区域。

    `D > 1` 之后该式无物理主张 (`D = 1` 已是失效态), 按 §9 的显式规则
    `post_eol_rule = b_frozen_at_D_eq_one` 把 `b` 冻结在 `D = 1` 处的值:

        b_post = b0 * (1 + Delta * 1) = b0 * (1 + Delta)

    实现即 `min(D, 1)`。这是一条**写进 config 与登记表的规则**, 不是顺手的
    线性外推 —— 区别在于它被显式记录且可被测试断言。
    """
    d = np.minimum(np.asarray(D, dtype=np.float64), float(threshold))
    return float(b0) * (1.0 + float(Delta) * d)


def apply_lube_spike(b: np.ndarray, traj_rng, dist_cfg: dict,
                     lo: float = 1.3, hi: float = 1.8) -> dict:
    """§10 润滑突变: 只乘到 `b`, **不动 `D`**。

    概率与倍数区间与 `wheel_sim.py:110-112` 逐值相同 (原定义不改)。
    抽随机数的**次数与顺序**也与旧实现相同 (先 random(), 命中才 integers +
    uniform), 以保持同一 `seed_traj` 下的随机流可比。

    返回 dict 记录是否发生、位置与倍数 —— 让"没突变"与"忘了做"可区分。
    """
    b = np.asarray(b, dtype=np.float64).copy()
    occurred, idx, mult = False, -1, 1.0
    if traj_rng.random() < float(dist_cfg["lube_spike_prob"]):
        idx = int(traj_rng.integers(1, b.size - 1))
        mult = float(traj_rng.uniform(lo, hi))
        b[idx:] *= mult
        occurred = True
    return {"b": b, "spike_occurred": occurred, "spike_idx": idx,
            "spike_multiplier": mult, "affects_damage_state": False}


# ---------------------------------------------------------------------------
# 元信息 (供报告与测试引用, 避免各处硬编码字符串)
# ---------------------------------------------------------------------------
MODEL_META = {
    "name": "B18_CUMULATIVE_DAMAGE",
    "equation": "dD/dt = stress(t) / L_ref",
    "stress_equation": "stress(t) = g_duty(t) * a_T(T(t))",
    "friction_equation": "b(t) = b0 * (1 + Delta * D(t))",
    "arrhenius_equation": "a_T(T) = exp(-Ea/k_B * (1/T - 1/T_ref))",
    "reference_identity": "D(t) = t / L_ref  when  g_duty=1 and T=T_ref",
    "eol_rule": "cumulative_damage_reaches_one",
    "eol_threshold_D": EOL_DAMAGE_THRESHOLD,
    "provenance": PROVENANCE,
    "semantics": SEMANTICS,
    "forbidden_naming": FORBIDDEN_NAMING,
    "second_temperature_acceleration_created": False,
    "g_duty_reweighted": False,
    "legacy_shape_used_as_damage_state": False,
    "tau_years_used_in_damage": False,
    "lube_spike_affects_damage_state": False,
    "T_is_physics_derived_thermal_state": False,
    "T_role": "environmental_or_telemetry_condition_used_as_arrhenius_input",
}
