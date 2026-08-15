# BASILISK-B5 复现说明

本页只覆盖 B5 正式迁移评估。上游各阶段（S2.5–S5B、Basilisk V1–B4X、B2.1）的复现见
各自的 `docs/basilisk_*/REPRODUCE.md` 与 `docs/REPRODUCE.md`。

---

## 1. 环境

| 项 | 值 |
|----|----|
| python | 3.12.13 |
| platform | Windows-11-10.0.26200-SP0 |
| conda env | `DP_env` |
| 解释器 | `/c/ProgramData/miniconda3/envs/DP_env/python.exe` |

Windows 下运行前置 `PYTHONIOENCODING=utf-8`，否则中文日志在部分终端会报编码错。

```bash
export PYTHONIOENCODING=utf-8
PY=/c/ProgramData/miniconda3/envs/DP_env/python.exe
```

B5 全程 **CPU**，无 GPU 依赖，因此跨机器应当逐位一致（不适用 `docs/REPRODUCE.md` 里
GPU ±5% 的容差条款）。

**B5 未写入 `requirements.txt` / `Dockerfile` / `docker-compose.yml`** —— Basilisk 系列
是诊断/评估支线，不进主复现封装。依赖与主项目完全相同。

---

## 2. 输入前置（必须已存在，本阶段不重建）

| 产物 | 路径 | 校验 |
|------|------|------|
| B1.8 场景数据 | `data/processed/wheel/basilisk_b18/*` | 由 `baseline_contract.json` 逐文件哈希 |
| B1.9 特征 | `data/features/wheel/basilisk_b19/*` | 同上 |
| B2.1 划分 | `docs/basilisk_b21/split_manifest.json` | `split_sha256 = 23e2b944…` |
| 源域 checkpoint | `checkpoints/basilisk_b19/source_*.pt` | 同上 |
| S4/Basilisk evaluator | `src/eval/*` | `frozen_code` 组 |

若上述任一项缺失或哈希不符，第 4 节的第 1 步会直接判 **`B5_INVALID`** 并中止 ——
这是设计行为，不要绕过。

---

## 3. 冻结参数（全部走 config / protocol，无硬编码）

配置：`configs/wheel_basilisk_b5.yaml`
`config_sha256 = 09f02f4916c1eb115611916eb1e3235e255155523020abb48986f8eff018ee6f`

协议：`docs/basilisk_b5/protocol.md`
`protocol_sha256 = 592aeab38458bcec0673d54b7b8c1c1fb3f6cd1a58207cb73db49c2eb3786a57`

| 项 | 值 |
|----|----|
| 正式 seed | `[112, 113, 114, 115, 116]` |
| 禁用 seed | `[72,73,74,75,76, 92,93,94, 102,103,104,105,106]` |
| 输入 schema | `CORE_ONLY`（`decided_by_b3x_verdict = B3X_NO_STABILIZING_SIGNAL`） |
| 训练组 | `target_only` / `source_finetune` / `source_mmd_finetune` |
| 仅评估组 | `const_mean_info` / `damage_extrapolation` |
| 主指标 | `test_info_trajectory_macro_rmse` |
| bin 边界 | 取自 B2.1，`[26846.0, 37395.0]`，不重算 |
| bootstrap | seed 级配对差，n=5，5000 次，种子 `20260814` |

Gate 门槛（在出任何 B5 数字之前冻结，`frozen_before_any_b5_number = true`）：

```
min_improve_count = 4          n_seeds = 5
require_mean_positive = true   require_median_positive = true
require_ci_lower_positive = true
corr_tol = 0.02
require_catastrophic_not_worse = true
warning_miss_tol = 0.05
min_lifetime_bins_gain_positive = 2   n_lifetime_bins = 3
```

`summarize_b5.py` 每次运行都会把 config 里的活门槛与 `protocol_hash.json` 里的冻结值
逐项比对，并重新校验 `protocol.md` 自身的哈希；**任一不符即抛 `B5_INVALID`**。
这条自检是防止事后降门槛的机制，不要改。

---

## 4. 执行顺序（§22，必须按序）

```bash
export PYTHONIOENCODING=utf-8
PY=/c/ProgramData/miniconda3/envs/DP_env/python.exe

# 1. 冻结前基线核对 —— 任一冻结项变化立即 B5_INVALID
$PY scripts/basilisk_b5/verify_baseline.py --tag before

# 2. 冻结协议与门槛（在任何 B5 数字之前）
$PY scripts/basilisk_b5/freeze_protocol.py --config configs/wheel_basilisk_b5.yaml

# 3. 正式训练 + 评估：5 seed × 3 学习组 + 2 仅评估组（约 22 分钟 CPU）
$PY scripts/basilisk_b5/run_formal_transfer.py

# 4. 配对增益 + seed 级 bootstrap
$PY scripts/basilisk_b5/analyze_paired_gain.py

# 5. 寿命分 bin（bin 边界取自 B2.1）
$PY scripts/basilisk_b5/analyze_lifetime_bins.py

# 6. 报警 / 预后指标（冻结 evaluator）
$PY scripts/basilisk_b5/analyze_warning_metrics.py

# 7. 八条门槛判定 + 主表 + 工程推荐
$PY scripts/basilisk_b5/summarize_b5.py

# 8. 全量测试
$PY -m pytest tests/ -q

# 9. 冻结后基线复核 —— 确认全程未改动任何上游产物
$PY scripts/basilisk_b5/verify_baseline.py --tag after
```

第 1 步与第 9 步必须都输出 `B5_BASELINE_CONTRACT_OK`。

---

## 5. 预期产物

`checkpoints/basilisk_b5/`：

| 文件 | 内容 |
|------|------|
| `protocol_hash.json` | 冻结的协议哈希 + 门槛 + seed 列表 |
| `formal_metrics.json` | 逐 seed 逐组指标 + `{group}_meta`（选点/指纹） |
| `formal_raw.npz` | 25 组预测原始数组 |
| `paired_gain.json` | `gain_ft` / `gain_mmd` + bootstrap + gate 输入 |
| `lifetime_bins.json` | 分 bin 指标 + `gain_by_bin` + `short_life` + 删失退化 |
| `warning_metrics.json` | coverage/miss/FA/PH/α-λ/convergence |
| `summary.json` | 八条门槛逐条判定 + 主表 + 判定 + 工程推荐 |
| `{group}_s{seed}.pt` | 15 个 checkpoint（3 组 × 5 seed） |

`docs/basilisk_b5/`：`baseline_contract.json`、`protocol.md`、`results.md`、
`lifetime_bin_results.md`、`warning_results.md`、`limitations.md`、`REPRODUCE.md`。
根目录：`STATUS_BASILISK_B5.md`。

---

## 6. 预期数值（用于判断复现是否成功）

主指标（test info-zone trajectory-macro RMSE，5 seed 均值）：

| 组 | 期望值 |
|----|-------:|
| `target_only` | 0.241024 |
| `source_finetune` | 0.242659 |
| `source_mmd_finetune` | 0.241607 |
| `const_mean_info` | 0.288629 |
| `damage_extrapolation` | 0.054312 |

配对增益：

| 项 | 期望值 |
|----|-------:|
| `gain_ft` mean / improve | −0.001635 / 1-of-5 |
| `gain_mmd` mean / improve | −0.000583 / 2-of-5 |
| `gain_ft` CI95 | [−0.003828, +0.000704] |
| `gain_mmd` CI95 | [−0.006589, +0.005423] |

判定：

```
ft_verdict  = B5_FT_NO_POSITIVE_TRANSFER   (1/8 条通过)
mmd_verdict = B5_MMD_NO_POSITIVE_TRANSFER  (3/8 条通过)
verdict     = B5_NO_POSITIVE_TRANSFER      (§19 组合 D)
ENGINEERING_RECOMMENDATION = damage_extrapolation
```

CPU 全程确定，复现应逐位一致。若主指标出现任何偏离，先查第 2 节的输入哈希，
再查第 7 节的配对指纹 —— **不要通过调参去凑数字。**

---

## 7. 配对性自检（复现时最该看的两列）

§6 要求三个学习组在同一 seed 下严格配对。机器可验证的证据在
`formal_metrics.json` 的 `{group}_meta`：

| seed | `batch_order_signature`（前 10） | `init_weights_sha256`（前 10） |
|-----:|--------------------------------|------------------------------|
| 112 | `fa1fb7a7ad` | `8084dd553b` |
| 113 | `9586577f9f` | `ba586094b7` |
| 114 | `bb3ffe7de6` | `ba67d28330` |
| 115 | `1d4830141b` | `6a18344a9f` |
| 116 | `f125ef99c2` | `77a15c37ba` |

**同一 seed 内三组这两个值必须逐位相同；不同 seed 之间必须不同。**
两条都由 `tests/basilisk_b5/test_b5_training_fairness.py` 断言。

值得记下的一个坑：`DataLoader(shuffle=True)` 的 `torch.Generator` 是**有状态**的，
每个 epoch 采样都会推进它。三组共用同一个 loader 顺序训练时，若不在每组开训前
重置，第二组、第三组拿到的是"接着上一组走"的 batch 顺序 —— 那就不是严格配对，
RMSE 差里会混进 batch 顺序噪声。`data_b5.py::reset_target_batch_order()` 就是为此存在，
`run_formal_transfer.py` 在每组开训前调用它。**移除它会静默破坏配对性，
但所有数字看起来仍然"正常"。**

---

## 8. 不要做的事

- 不要重划 split、不要重算 bin 边界、不要改 `mmd_lambda`、不要加第四种迁移方法。
- 不要复用 seed 72–76 / 92–94 / 102–106（`forbidden_seeds`），也不要新增 seed 去改判定。
- 不要用 test 做选点或早停，不要跨 seed 挑最优 checkpoint。
- 不要因某组表现差而单独给它加 epoch。
- 不要把 NaN 转成 0。
- **不要自动跑低数据矩阵，不要自动执行 B6。** B5 完成后停止。
