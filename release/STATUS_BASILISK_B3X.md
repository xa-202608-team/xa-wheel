# STATUS — BASILISK-B3X (Mission-feature 消融诊断)

> **`POST_B2_FAIL_EXPLORATORY`** · **`NOT_FORMAL_EVIDENCE`**
>
> 本阶段**不是**正式 B3。**`B2_GENERALIZATION_FAIL` 保持终局**, 未被覆盖、
> 未被重新解释。本文件不宣称 mission features 有效, 不构成迁移证据。

```
B3X_BASELINE_CONTRACT_OK          (838 项, --tag before / after 一致)
B3X_SPLIT_REUSED_FROM_B2          (split_sha256 = 079296e5…4113ae)
B3X_NO_STABILIZING_SIGNAL         (4 条判据通过 3 条)
B2_GENERALIZATION_FAIL            (不变, 终局)
```

## 结论

Basilisk mission features **减小了** B2 在短寿命轨迹上的失控幅度,
但**没有把任何一个崩塌 seed 拉回正常量级**。缓解 ≠ 稳定化。

## 判据 (§9, 四条全为必要条件)

| # | 判据 | 阈值 | 实测 | |
|---|---|---|---|---|
| 1 | short-EOL 相对下降 ≥10% 的 seed 数 | ≥3/5 | 4 | PASS |
| 2 | overall mean gain > 0 | >0 | +0.130671 | PASS |
| 3 | 崩塌 seed 恢复数 (RMSE ≤ 0.2646356) | ≥2 | **0** | **FAIL** |
| 4 | warning miss 不恶化 (容差 0.02) | 无恶化 | 无 (4/5 改善) | PASS |

阈值全部在跑数之前写入 `docs/basilisk_b3x/protocol.md` §8 与
`configs/wheel_basilisk_b3x.yaml::b3x.verdict`。

## 关键数字

`gain_mission = RMSE_core_only − RMSE_core_plus_mission`

| seed | B2 崩塌? | CORE_ONLY | +MISSION | gain | 恢复? |
|---|---|---|---|---|---|
| 72 | 是 | 1.174494 | 0.987864 | +0.186630 | 否 |
| 73 | 否 | 0.310286 | 0.264705 | +0.045582 | 否 |
| 74 | 是 | 0.600155 | 0.375131 | +0.225024 | 否 |
| 75 | 否 | 0.476949 | 0.683991 | −0.207042 | 否 |
| 76 | 是 | 0.672135 | 0.268975 | +0.403160 | 否 (差 +0.0043) |

mean +0.130671 · median +0.186630 · std 0.227800 · improve 4/5
描述性 CI `[−0.0496, +0.2960]` (n=5, 跨 0, 不作显著性断言)

**收益的位置**: short-EOL mean gain **+1.4965** vs normal-EOL **+0.0065**
(相差 230 倍)。72 条正常轨迹上加 6 列输入等于没加。

**机制**: 主要是压住无界 `softplus` RUL 头的外推
(mean ΔPSR −1.4822, Δpred_std −0.4278), 兼有小幅方向改善
(mean Δmacro_corr +0.0591)。seed 72 的 short-EOL pred max 从 30.10 降到 22.77 ——
真值上界是 0.999, 即**同方向同量级失控, 只是幅度小些**。

## 两条必须一起读的警告

1. **CORE_ONLY 不是 B2 的复现。** 为满足 §6 (两臂 batch 顺序相同) 用了显式
   generator, 而 B2 用全局 RNG。seed 75 因此从 0.2646 变成 0.4769。
   → **gain 只在 B3X 内部两臂之间计算, 绝不减 B2 的表。**
   → 附带发现: B2 的"3 崩 2 正常"划分对 batch 顺序敏感, 失败面比 B2 表面更宽。
2. **mission features 是 `OBSERVABLE_DERIVED`。** 需要星上 1 Hz 窗内聚合能力;
   实测 `corr(|omega_telemetry|, omega_p95) ≈ 0.72–0.79` (判据 >0.999),
   低速下传遥测无法反推。详见 `docs/basilisk_b3x/limitations.md` §1。

## §10 → B4X

规则在 B3X 出数之前已写入 protocol §9:
`B3X_NO_STABILIZING_SIGNAL` → **B4X 三组统一使用 `core_only` (B2 冻结 10 列)**。
无论 B3X 结果如何都允许继续 B4X。

## 产物

| 文件 | 内容 |
|---|---|
| `configs/wheel_basilisk_b3x.yaml` | 两臂 / seed / 判据阈值 / 禁止列 |
| `docs/basilisk_b3x/protocol.md` | 跑数前冻结的协议 (含 §10 B4X 规则) |
| `docs/basilisk_b3x/baseline_contract.json` | 838 项契约 |
| `docs/basilisk_b3x/mission_ablation_results.md` | 主表 + §8 A–D |
| `docs/basilisk_b3x/short_eol_diagnostics.md` | 三子集 × 两臂 + 逐轨迹 |
| `docs/basilisk_b3x/limitations.md` | 可观测性边界 (必读) |
| `checkpoints/basilisk_b3x/mission_ablation_metrics.json` | 完整指标 |
| `checkpoints/basilisk_b3x/short_eol_diagnostics.json` | 分层指标 |
| `checkpoints/basilisk_b3x/summary.json` | 判据 + verdict |
| `checkpoints/basilisk_b3x/ablation_raw.npz` | 10 组 test 预测 |
| `scripts/basilisk_b3x/` | 4 个脚本 + `data_b3x.py` |
| `tests/basilisk_b3x/` | §19 的 7 个测试 + 扩展 |

## 未做的事

未做正式 B3; 未使用禁止标签; 未改 B2 任何产物; 未改冻结超参;
未加载 source checkpoint; 未跑 MMD; 未按 test 结果调参;
未声称 B2 判定需要修改。
