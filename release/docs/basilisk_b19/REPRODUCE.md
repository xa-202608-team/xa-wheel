# BASILISK-B1.9 复现说明

**目标**：从零复现 `hi_damage_obs` 的构造、审计与冻结，并核验 14 个 Gate 与
两侧 content hash 一致。

---

## 1. 环境

| 项 | 值 |
|---|---|
| Python | 3.12.13（`/c/ProgramData/miniconda3/envs/DP_env/python.exe`） |
| Platform | Windows-11-10.0.26200-SP0 |
| 环境变量 | `PYTHONIOENCODING=utf-8`（**必需**，否则中文输出崩） |
| 工作目录 | 仓库根目录（脚本按 `ROOT` 相对路径寻址） |

B1.9 **不写入** `requirements.txt` / `Dockerfile` / `docker-compose.yml`
（长期约束）。

## 2. 配置继承陷阱（必读）

`configs/wheel_basilisk_b19.yaml` 通过 `base_config` 链继承。**不要**用
`yaml.safe_load` 直接读它 —— 会丢掉继承来的 `sim:` / `rated:` /
`wear_drive:` / `source:` / `model:` 块。必须用：

```python
from scripts.basilisk_b11 import calibrate_degradation as CAL
cfg = CAL.load_b11_config(ROOT / "configs/wheel_basilisk_b19.yaml")
```

关键键位（不要猜，已核实）：

| 量 | 键路径 | 值 |
|---|---|---|
| Arrhenius Ea | `cfg["sim"]["disturbance"]["arrhenius_Ea_eV"]` | 0.3 |
| Arrhenius T_ref | `cfg["sim"]["disturbance"]["arrhenius_T_ref_K"]` | 293.15 |
| 机动力矩比 | `cfg["sim"]["profile"]["duty"]["maneuver_torque_frac"]` | 0.02 |
| 窗内点数 | `round(sim.sample_period_s / sim.profile.dt_s)` | `round(1800/1.0)` = 1800 |
| 轮索引 | `cfg["sim"]["profile"]["bridge"]["wheel_index"]` | 0 |

## 3. 命名空间

| 类别 | 路径 |
|---|---|
| config | `configs/wheel_basilisk_b19.yaml` |
| scripts | `scripts/basilisk_b19/` |
| tests | `tests/basilisk_b19/` |
| checkpoints | `checkpoints/basilisk_b19/` |
| features | `data/features/wheel/basilisk_b19/` |
| docs | `docs/basilisk_b19/` |
| status | `STATUS_BASILISK_B19.md` |

**只读（改动即违约）**：`basilisk_v1`、`b1`/`b11`…`b18` 的全部产物、
analytic S2.5–S5B、`src/sim/wheel_sim.py`、`configs/wheel.yaml`，
以及 B1.8 的 `wheel_all.h5` / `params.json` / `target_features.h5`。

## 4. 复现序列（§11 八步，按序实跑）

```bash
export PYTHONIOENCODING=utf-8
PY=/c/ProgramData/miniconda3/envs/DP_env/python.exe

# 1) 冻结/核验前置 baseline
$PY scripts/basilisk_b19/verify_baseline.py --tag before

# 2) 可观测输入审计 (§3) —— 任一 g_duty 输入是 HIDDEN_TRUTH 即停
$PY scripts/basilisk_b19/audit_observable_inputs.py

# 3) 构造 HI_D_obs (§2/§4)
$PY scripts/basilisk_b19/derive_damage_proxy.py

# 4) 构建 feature 并自查 14 Gate (§6/§7/§8)
$PY scripts/basilisk_b19/build_features.py

# 5) 独立复审 —— 只读磁盘 h5 + 源数据 + 源码, 自己重算 14 Gate
$PY scripts/basilisk_b19/audit_features.py

# 6) 冻结定义 (仅当两侧一致且都 READY)
$PY scripts/basilisk_b19/freeze_feature_definition.py

# 7) 全量测试
$PY -m pytest tests/ -q

# 8) 核验前置产物逐字未变
$PY scripts/basilisk_b19/verify_baseline.py --tag after
```

**步 1 与步 8 必须报告同样的项数且全部不变**（290 文件 + 378 数值项 = 668 项）。

## 5. 为什么 build 与 audit 要写两遍

`build_features.py` 在内存里握有全部中间量时算 Gate，属于**自证**；
`audit_features.py` 只从磁盘 h5 + 源数据 + 源码重新推导，并**独立实现**
自己的 `content_hash`（不 import 构建侧函数）。两侧不一致 ⇒
`B19_AUDIT_MISMATCH`，其严重性高于任何单侧 FAIL。

本次两侧结果：

| 项 | 值 |
|---|---|
| build content sha256 | `510dd14e5f3f6ae214c2b2ed85f3eb0227504f4ccc0529ace233d5a26f378cba` |
| audit content sha256 | `510dd14e5f3f6ae214c2b2ed85f3eb0227504f4ccc0529ace233d5a26f378cba` |
| 不一致 Gate | `[]` |

## 6. 冻结哈希

| 项 | 值 |
|---|---|
| protocol sha256 | `e95bf867e083aa0d3095f31049c7587ab3acc21a463463eac640a427008e8a3e` |
| config sha256 | `95cdb74ad61fc9f7fc53031ec6f662d362aaa07cc977b374c762dc2fec7ae1bf` |
| feature content sha256 | `510dd14e5f3f6ae214c2b2ed85f3eb0227504f4ccc0529ace233d5a26f378cba` |
| feature 文件 sha256 | `575d708ae9330dfa37d407cf639ebbbd098c473ce1fc137ff1eef8feaab9c293` |
| 源数据 content sha256 | `61678e582f82bd136ffe509de7ec1274c07c39fe791a2a762fcf562a3d36a0f7` |

`content_hash` 与文件 sha256 分开记：HDF5 文件字节含时间戳/布局差异，
跨机复现看 **content hash**（按轨迹逐数组内容算），不看文件字节。

## 7. 方程只有一份实现（import 而非 copy）

| 方程 | 唯一出处 |
|---|---|
| `g_duty` | `scripts/basilisk_b1/calibrate_degradation.py::g_duty` |
| `a_T` (Arrhenius) | `src/sim/damage_model.py::arrhenius_accel` |
| `stress` / 累积 | `src/sim/damage_model.py::stress_series` / `damage_series` |
| 四个驱动量派生 | `scripts/basilisk_b11/calibrate_degradation.py::utilization_from_duty` |
| 参考工况 | `scripts/basilisk_b11/calibrate_degradation.py::reference_duty` |
| RUL 标签 | `scripts/basilisk_b18/build_features.py::derive_labels` |

因此"B1.9 偷偷改了权重 / Ea / L_ref / RUL 定义"在结构上不可能发生。
注意 `stress_series` 返回的键是 `"g_duty"`（不是 `"g"`）。

## 8. 跨平台容差

本阶段**纯 CPU、无训练**，不涉及 CUDA 非确定性。前缀不变性与
`x_T` 对比的容差都是**精确 0.0**（`atol == 0.0` 本身被测试断言，
防止有人事后放宽）。唯一的非零残差是 `mission_features` 以 float32
存储导致的 ~1e-9，只出现在离线 `IMPLEMENTATION_AUDIT` 里，不进任何 Gate 阈值。

## 9. 测试命名约定

`tests/` 下**没有** `__init__.py`，也**没有**根 `conftest.py`。因此：

- 测试**文件**名必须带 `b19_` 前缀（避免与其它阶段同名文件冲突）；
- `from conftest import ...` 依赖 pytest 的 rootdir 插入，可用；
- 协议 §10 指定的 12 个**函数**名保持逐字节一致，不加前缀。
