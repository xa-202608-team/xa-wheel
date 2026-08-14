# BASILISK-B2 协议 —— Trajectory Split + Target-only Generalization Gate

**本文件在任何训练发生之前冻结。** 判据（§8）在看到任何 test 数字之前写定，
之后不得修改阈值、不得增删判据、不得更换主指标。

---

## §0 前置

必须先读 `STATUS_BASILISK_B19.md`，仅当其中出现 `B19_FEATURE_READY`
才允许继续，否则立即停止并判 `B2_PRECONDITION_FAIL`。

**旧 analytic `S25_PASS` 不得继承。** 那是 schema_v1 数据 + friction-threshold
EOL 上的结论；本阶段的数据（Basilisk 工况）、失效定义（`D >= 1`）、
健康指标（`hi_damage_obs`）三者全部换过，S2.5 的通过与否对 B2 没有任何证明力。
B2 必须从零重新建立 Target-only 基线。

## §1 冻结输入

只用 `data/features/wheel/basilisk_b19/target_features.h5`。记录：

| 项 | 来源 |
|---|---|
| dataset content hash | B1.9 `feature_stats.json::feature_content_sha256` |
| feature 文件 hash | 现场重算 sha256 |
| 源 lifetime dataset hash | B1.9 记录的 `source_dataset_content_sha256` |
| B1.8 protocol hash | h5 attr `b18_protocol_sha256` |
| B1.9 protocol hash | h5 attr `b19_protocol_sha256` |
| B2 protocol hash | 本文件 sha256 |
| split hash | §2 定义 |

**不得重新生成 feature 文件，不得修改 B1.9 的任何产物。**

## §2 轨迹级划分

- 总体 150 条（event-observed 71 / censored 79）。
- 比例 **train 30% / val 20% / test 50%**。
- **轨迹级**：同一条轨迹的所有时间点必须整体落在同一划分，严禁按时间窗打散。
- **分层键 = `event_observed` / `censored`**。不用 EOL 分位数作主分层键，
  因为 79 条删失轨迹**没有 EOL**，对它们按 EOL 分层无从定义。
  event 组内再按 EOL 分位数细分 4 层，使三个划分的 event 难度分布可比。
- `split_seed = 20260810`（固定）。划分必须确定性：同 seed 同输入必得同结果。
- train / val / test 三集合**互不重叠**，并集 = 全部 150 条。
- 冻结轨迹 ID 列表与 split hash 到 `checkpoints/basilisk_b2/split.json`。

split hash 的 payload 只依赖 `{rule, split_seed, ratios, 三个已排序的 ID 列表}`，
与容器顺序无关。

## §3 不做 S3 truncation

第一阶段沿用**完整 train history**。`b2.target_truncate_hi = null`。

本阶段只回答："在完整目标域训练数据下，Target-only 是否能稳定学会新的 RUL 任务？"
截断场景（目标域未见失效）属于后续阶段，**本阶段不得引入**。

## §4 RUL 监督与删失契约

- event-observed 轨迹：用 `rul`（真实剩余寿命，已按 `rul_cap_ratio = 0.35` 截顶）。
- censored 轨迹：`rul` 为 NaN。**不得伪造完整 RUL**。使用已有的
  `rul_lower_bound`（B1.9 h5 内已存，定义 `lb(t) = min(n-1-t, cap_ratio·n)`，
  **不使用任何未来 EOL 信息**）。
- 主模型必须 **censor-aware**：
  ```
  L_rul = Huber_weighted(pred, rul | event_observed=True)
        + eta · mean[ relu(rul_lower_bound - pred) ]^2   (event_observed=False)
  ```
  删失样本**绝不进入 Huber**（下界当精确值监督会把预测往下拽）。
  `eta = 1.0`，实现复用 `run_groups::_censored_rul_loss`，不另发明。
- **评估侧**：删失轨迹不得伪造 EOL 类指标。任何依赖 EOL 的量（RUL RMSE / PH /
  alpha-lambda）在删失轨迹上必须是 `NaN` + `n = 0`，**严禁转 0**。

## §5 Target-only 输入

- `x_T` = B1.9 冻结的 10 列，逐字不变：
  `[I_m, omega, T, T_cmd, sigma_Im, b_hat, dT, omega_err, Tf_ratio, Tf_slope]`。
- **`hi_damage_obs` 不得作为普通输入列。** 理由（B1.9 §7 已述）：它是监督信号的
  构造式，若当普通输入喂进去，模型可以直接对它积分外推得到 RUL，构成**评价循环**。
- 允许 `hi_damage_obs` 进入 **health auxiliary loss**（`lambda_hi · MSE(hi_pred, hi)`）——
  它本身就是目标域的健康监督信号，作为辅助头目标是正当的。
- `mission_features` 本阶段**固定不进输入**（那是 B3 的消融变量）。

## §6 基线比较

| 代号 | 方法 | 可部署 | 说明 |
|---|---|---|---|
| A | `const_mean_info` | 是 | 输出 train 划分 info 区真实 RUL 均值；最强平凡常数基线 |
| B | `damage_extrapolation` | 是 | 由 `hi_damage_obs` 因果滑窗斜率线性外推到 `hi_fail = 1.0` |
| C | `target_only` | 是 | 本阶段主模型 |

**不跑 source transfer**（那是 B4）。
`persistence` 是 ORACLE-LIKE（读真值），**不进本阶段任何排名**。
A 的常数只由 train 划分求得，绝不看 val/test。

## §7 训练配置

沿用 S2.5 已验证的冻结组合，**不重新调参**：

| 项 | 值 |
|---|---|
| `early_stop_metric` | `info_macro_rmse`（**val** 侧） |
| `max_epochs` | 8 |
| `early_stop_patience` | 2 |
| `weight_decay` | 1e-3 |

这是 starting frozen baseline。**不得自动搜索超参。** 只有出现"训练根本不收敛 /
代码不兼容"才判 `B2_CONFIG_INCOMPATIBLE` 并停止，而不是转去调参。

checkpoint 选择只允许用 **val** 指标；test 在任何时刻都不得进入训练/早停/选择路径。

## §8 开发 Gate

5 个新 seed `[72, 73, 74, 75, 76]`（与 S2.5 的 42/43/44、52–56 及 S3 的 62–64 不相交）。
每 seed 跑 `target_only` + `const_mean_info`，在**完全相同的划分与取点**上比较。

主指标 = **test info trajectory-macro RMSE**。

`B2_GENERALIZATION_PASS` 要求以下 7 条**全部**满足：

| # | 判据 |
|---|---|
| 1 | `target_only` 至少 **4/5** seed 优于 `const_mean_info` |
| 2 | mean paired gain **> 0** |
| 3 | seed-level paired bootstrap **95% CI 下界 > 0** |
| 4 | info 口径 macro corr **> 0** 至少 4/5 seed |
| 5 | PSR **≥ 0.30** 至少 4/5 seed |
| 6 | warning / pre-EOL 指标**有效**（在 event-observed test 轨迹上可计算且非全 NaN） |
| 7 | checkpoint 选择**只用 val**（无 test 入口） |

否则 `B2_GENERALIZATION_FAIL`。

bootstrap 重采样单位 = **每 seed 一个配对差值**（5 个数里有放回抽 5 个求均值），
`bootstrap_seed = 20260810`，reps = 2000。
**绝不把时间点当独立样本** —— 那会把 n 从 5 放大到上万，人为制造置信度。

**禁止调阈值降门槛。** 上表阈值在本文件冻结后不得修改。

## §9 censor 分层报告

必须同时报告三档：

1. `event_observed_test` —— test 里的 event 轨迹（有真实 RUL，全指标可算）；
2. `censored_test` —— test 里的删失轨迹（RUL 真值不可见，只能报下界违反率
   `mean[pred < lb]` 与 HI 类指标；RUL RMSE 必须 `NaN` + `n = 0`）；
3. `common_evaluable_endpoints` —— 两类轨迹都可评的取点集合。

**不得把 censored 的 NaN EOL 指标转 0。** 空集合返回 `NaN` + `n = 0`。

## §10 输出

| 文件 | 内容 |
|---|---|
| `docs/basilisk_b2/protocol.md` | 本文件 |
| `docs/basilisk_b2/results.md` | 结果表 + 判据表 |
| `checkpoints/basilisk_b2/metrics.json` | 全部数值 |
| `checkpoints/basilisk_b2/split.json` | 冻结划分 |
| `STATUS_BASILISK_B2.md` | 唯一判词 |

最终只输出 `B2_GENERALIZATION_PASS` 或 `B2_GENERALIZATION_FAIL`
（前置不满足时 `B2_PRECONDITION_FAIL`；不收敛/不兼容时 `B2_CONFIG_INCOMPATIBLE`）。

**到此停止。不要进入迁移。**

## §11 只读边界

**不得修改**：`basilisk_v1`、`b1`/`b11`…`b19` 的全部产物、analytic S2.5–S5B
产物与协议、`src/sim/wheel_sim.py`、`configs/wheel.yaml`、
以及 B1.9 baseline_contract 里 `frozen_code` 组的 12 个源文件
（含 `src/transfer/train_transfer.py`、`src/experiments/run_groups.py`、
`src/baselines/trivial.py`、`src/experiments/metrics.py`）。

后果：旧栈把目标域 HI 列名**硬编码**为 `hi_b`，而 B1.9 的主 HI 叫
`hi_damage_obs`。因此 B2 **自带 loader**（`scripts/basilisk_b2/data_b2.py`），
从 `transfer.target_hi_key` 读列名，不改任何冻结代码。
训练循环、损失、指标、模型定义则**全部 import 复用**，不复制。

Basilisk 不得写入 `requirements.txt` / `Dockerfile` / `docker-compose.yml`。
