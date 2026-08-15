# STATUS · BASILISK-B4X (迁移诊断)

> **`POST_B2_FAIL_EXPLORATORY`** · **`NOT_FORMAL_EVIDENCE`** · **`EXPLORATORY_ONLY`**
>
> **`B2_GENERALIZATION_FAIL_REMAINS_FINAL`** · **`NOT_FORMAL_TRANSFER_EVIDENCE`**

```
B4X_BASELINE_CONTRACT_OK              (914 项, --tag before / after 一致)
B4X_SPLIT_REUSED_FROM_B2              (split_sha256 = 079296e5…4113ae)
B4X_INPUT_SCHEMA = core_only          (由 B3X_NO_STABILIZING_SIGNAL 唯一决定)
B4X_TRANSFER_STABILIZATION_SIGNAL     (source_finetune, A 组 4/4; 无方法过 B 组)
  └─ 归因 regularizes_extrapolation   (4/4 归因块; 不是迁移有效)
B2_GENERALIZATION_FAIL                (不变, 终局)
```

**读这个标签前必读一句**: A 组判据**只看 replay 套 (72/74/76)**, 而同一实验在
new 套 (92/93/94) 上符号完全相反 (overall mean gain **+0.2007 → −0.3001**,
improve **2/3 → 0/3**)。唯一允许的表述是

> source initialization regularizes extrapolation under the known
> B2 distribution gap

**不是** positive transfer proven。详见 `docs/basilisk_b4x/limitations.md` §1。

---

## §17 判据逐条

### `source_finetune` —— A **4/4**, B 1/4 ⇒ 标签 A

| 判据 | 值 | 阈值 | 结果 |
|---|---|---|---|
| A1 replay catastrophic rate 降低 ≥ 2 | 2 | 2 | **PASS** |
| A2 replay short-EOL mean gain > 0 | +2.342796 | 0 | **PASS** |
| A3 replay overall mean gain > 0 | +0.200681 | 0 | **PASS** |
| A4 replay normal-EOL 恶化 ≤ 5% | −2.065% (改善) | 5% | **PASS** |
| B1 new 中 gain>0 ≥ 2 | **0** | 2 | FAIL |
| B2 new normal-EOL mean gain > 0 | −0.010125 | 0 | FAIL |
| B3 macro corr 不恶化 | −0.06300 | ≥ −0.02 | FAIL |
| B4 warning miss 不恶化 | +0.01852 | ≤ 0.02 | PASS |

### `source_mmd_finetune` —— A 2/4, B 1/4

A2 (**−7.146157**) 与 A3 (**−0.585762**) 均 FAIL。short-EOL RMSE 最差 23.65
(tid 19 上 36.997), 真值上界 0.999。**MMD 无任何可主张的优势。**

## 两套 seed (§13, 分开报告, 不合并)

| 方法 | seed 套 | overall mean gain | improve | short-EOL | normal-EOL | catastrophic 降低 |
|---|---|---|---|---|---|---|
| source_finetune | **new** | **−0.300149** | 0/3 | −3.490416 | −0.010125 | 0/3 |
| source_finetune | **replay** | **+0.200681** | 2/3 | +2.342796 | +0.005943 | 2/3 |
| source_mmd | **new** | −0.784931 | 0/3 | −9.247111 | −0.015642 | 0/3 |
| source_mmd | **replay** | −0.585762 | 1/3 | −7.146157 | +0.010638 | 2/3 |

## 崩塌归因 (§18) —— 4/4 落在 `regularizes_extrapolation`

| 证据 | 数字 |
|---|---|
| short-EOL / normal-EOL gain 比 (replay, ft) | **394×** |
| PSR 平均变化 (replay, ft) | **−2.334** |
| pred std 平均变化 (replay, ft) | **−0.674** |
| normal-EOL RMSE 全 18 格区间 | [0.2471, 0.2926], 极差 0.0455 |
| seed 76 tid 19: target_only → ft | 6.789 → **0.459** (真值上界 0.999) |

overall 的全部组间差异由 **3 条 short-EOL 轨迹**驱动; 72 条普通轨迹上三组
基本无差别。source 组赢在"不失控", 不在"更准"。

## 已知不可用项 (不得隐藏)

- overall warning miss rate 18 格**全部 ≥ 0.6389**, 三组都不可用于在轨预警;
- `macro_ph` 18 格中 17 格恒为 0;
- tid 19 在 18 个格子中全部仍属 catastrophic, **从未被任何组救回**;
- 每套 seed n = 3, replay/ft 的 overall 描述性区间 **[−0.0073, +0.3511] 跨 0**,
  不作显著性断言。

## 产物清单 (§1)

| 路径 | 状态 |
|---|---|
| `configs/wheel_basilisk_b4x.yaml` | ✔ |
| `scripts/basilisk_b4x/verify_baseline.py` | ✔ 914 项契约 |
| `scripts/basilisk_b4x/data_b4x.py` | ✔ (schema 三方核验 + 双生成器) |
| `scripts/basilisk_b4x/run_transfer_diagnostic.py` | ✔ 6 seed × 3 组 |
| `scripts/basilisk_b4x/diagnose_transfer_stability.py` | ✔ |
| `scripts/basilisk_b4x/summarize_b4x.py` | ✔ |
| `tests/basilisk_b4x/` | ✔ 22 passed |
| `checkpoints/basilisk_b4x/{transfer_metrics,transfer_stability,summary,protocol_hash}.json` + `transfer_raw.npz` | ✔ |
| `docs/basilisk_b4x/{protocol,baseline_contract.json,transfer_results,short_eol_transfer_diagnostics,limitations}` | ✔ |
| `checkpoints/basilisk_b4/`, `docs/basilisk_b4/`, `STATUS_BASILISK_B4.md` | **恒缺席** (正式 B4 未做) |
| `checkpoints/basilisk_b4x/{formal_verdict,positive_transfer_proven}.json` | **恒缺席** |

## §22.18 是否值得做新的 B2.1 split-coverage robustness study

**值得, 且它比继续做迁移变体更有价值。** 理由是本阶段与 B3X 共同给出的两条
独立证据:

1. **B3X**: 仅改变 batch 顺序就把 B2 的"正常" seed 75 从 0.2646 推到 0.4769 ——
   B2 的 3-vs-2 双峰不是稳定的 seed 属性;
2. **B4X**: 同一 source encoder 在 replay 套 gain **+0.2007**, 在 new 套
   **−0.3001** —— 符号由 seed 决定。

两者指向同一个结构性问题: **test 含 4 条 EOL ≤ 22300 的 event 轨迹, 而 train
最小 EOL 22336 / val 最小 22426。失效区间在训练与模型选择信号中同时缺席。**
在这个缺口下, 任何方法对比都在测"谁的外推恰好更保守", 而不是"谁学到了退化
动力学"。

因此 B2.1 应做的是 **split 覆盖性的稳健性研究** (多次分层重划 + 检查每个划分
是否让 EOL 分布在 train/val/test 三侧可比), 而**不是**继续加迁移方法。
在覆盖问题解决之前, 迁移增益无法与外推保守性区分开。

**这只是下一步建议, 不是已执行的结论。本阶段不自动推进正式 S6。**
