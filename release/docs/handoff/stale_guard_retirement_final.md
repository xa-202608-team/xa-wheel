# stale_guard_retirement_final.md —— B8-HANDOFF 阶段生命周期守卫退役记录

**阶段**: `B8-HANDOFF-FINAL`
**授权来源**: B8-HANDOFF-FINAL §2「正式退役 lifecycle guard」
**改动性质**: **lifecycle governance（生命周期治理）**
**日期**: 2026-08-09

> **本次修改仅属 lifecycle governance，不改变任何科学结论或实验数字。**
> 任何实验数字未发生一比特变化；`data/` 未删除、未改动任何文件。

---

## 0. 退役前状态

```
python -m pytest tests/ -q
  1092 passed
  2 failed
  29 skipped
  1 xfailed
```

两个 failed：

| # | test |
|---|------|
| 1 | `tests/basilisk_b6/test_b6_baseline_contract.py::test_b6_old_artifacts_unchanged` |
| 2 | `tests/basilisk_b6/test_b6_baseline_contract.py::test_b6_b7_b8_artifacts_must_stay_absent` |

**唯一根因**：两条断言都在检查同一个事实 ——
`STATUS_BASILISK_B7.md` 在 B6 契约 `b6_must_stay_absent` 组中被登记为 `MISSING`，
但该文件已实际存在（mtime `2026-08-10 19:13`，3095 字节）。

全契约 457 个文件项中，「登记 MISSING 却实际存在」的项**只有这 1 项**：

```
[b6_must_stay_absent] STATUS_BASILISK_B7.md      exists=True   v=MISSING
```

`b6_must_stay_absent` 组其余 13 项全部仍然缺席（B8/S7/S8 产物、B7 的 metrics 等）。

---

## 1. 三条件证明（§1 要求）

### 条件 A — 只检查 downstream artifact absence / old hash

| 断言 | 检查内容 |
|------|---------|
| `test_b6_old_artifacts_unchanged` 中失败的那一行 | `assert not (ROOT / rel).exists()`，`rel` 来自 `exp == "MISSING"` 分支 —— 纯**存在性** |
| `test_b6_b7_b8_artifacts_must_stay_absent` | `assert v == "MISSING"`（契约登记值）+ `assert not (ROOT / rel).exists()` —— 纯**存在性** |

两条都**不读取任何文件内容**，不计算任何指标，不加载任何权重。
✅ 条件 A 满足。

### 条件 B — 失败原因仅为 B7/B8 已经人工授权存在

失败消息逐字：

```
AssertionError: STATUS_BASILISK_B7.md 本应缺席
assert not True
 +  where True = exists()
```

`STATUS_BASILISK_B7.md` 是 B7 阶段（`FINAL_FIGURES_REPORT_EVIDENCE_FREEZE`）
经显式人工授权执行后产出的状态文件。B7 的治理产物齐备：

```
docs/basilisk_b7/baseline_contract.json
docs/basilisk_b7/stale_guard_retirement.md
checkpoints/basilisk_b7/frozen_result_index.json
checkpoints/basilisk_b7/report_number_audit.json
```

✅ 条件 B 满足。

### 条件 C — 不涉及模型/指标/数据/split/leakage/统计结果/checkpoint/特征正确性

| 关注面 | 是否被这两条断言涉及 |
|--------|------------------|
| 模型架构 | ❌ 否 |
| 指标 / 统计结果 | ❌ 否 |
| 数据 / 特征 | ❌ 否 |
| split / leakage | ❌ 否 |
| checkpoint | ❌ 否 |
| 特征正确性 | ❌ 否 |

失败的那一行只在 `exp == "MISSING"` 分支内。该分支 `continue`，**永不进入**
下方的 `hashlib.sha256(...)` 比对。也即：这两条断言的失败与算法产物冻结机制
在代码路径上完全隔离。同文件内的哈希比对断言全部 PASS（`_n_hashed >= 20`）。

✅ 条件 C 满足。

**结论**: 2/2 failures = A 类 `LIFECYCLE_STALE`。B–G 类 = 0。无第三个 failure。
→ 未触发 `B8_HANDOFF_INVALID`。

---

## 2. 逐条退役记录（§3 要求字段）

### 退役条目 1

| 字段 | 值 |
|------|-----|
| `old_test` | `tests/basilisk_b6/test_b6_baseline_contract.py::test_b6_old_artifacts_unchanged` |
| `old_assertion` | `if exp == "MISSING": if p.exists(): bad.append((rel, "本应缺席却存在"))` |
| `old_lifecycle_stage` | `BASILISK_B6` (`FAILURE_LABEL_SCARCITY_FORMAL_MATRIX`) —— 当时 B7/B8 尚未授权，"下游永久缺席"成立 |
| `authorization_transition` | `B6 完成` → `B7 人工授权执行 (B7_REPORT_READY)` → `B8-PREP 人工授权 (B8_HANDOFF_READY)` → `B8-HANDOFF-FINAL §2 授权退役` |
| `replacement_guard` | `MISSING` 分支改为三态：缺席 → PASS；存在且在 `DOWNSTREAM_LIFECYCLE` 白名单 → 放行并计数（上限 = 白名单长度）；存在但无授权 → **仍然 FAIL**。哈希比对分支一条未减。 |
| `algorithm_numbers_affected` | **false** |
| `data_affected` | **false** |

### 退役条目 2

| 字段 | 值 |
|------|-----|
| `old_test` | `tests/basilisk_b6/test_b6_baseline_contract.py::test_b6_b7_b8_artifacts_must_stay_absent` |
| `old_assertion` | `for rel, v in grp.items(): assert v == "MISSING"; assert not (ROOT / rel).exists()` |
| `old_lifecycle_stage` | `BASILISK_B6` §22「接力防线」—— 意图是"B6 无论正负结论都不得自动运行 B7/B8" |
| `authorization_transition` | 同上。B7/B8 均已人工授权并正式执行，"永远缺席"前提消失 |
| `replacement_guard` | 测试**不删除**：保留"B6 冻结契约时确实登记了这些禁止项"的历史事实断言（证明 B6 阶段本身没有偷跑下游）。运行期防护移交下方三个新守卫。 |
| `algorithm_numbers_affected` | **false** |
| `data_affected` | **false** |

---

## 3. 替换守卫（§2 要求的五项核查）

新增 `tests/basilisk_b6/test_b6_baseline_contract.py` 中 4 个守卫：

### `test_b6_downstream_stage_transition_is_governed`
下游产物存在 → 其治理产物必须齐备。拦住"没有协议/契约/审计就先产出下游产物"。

### `test_b6_downstream_did_not_contaminate_upstream` ← **本次退役的核心补偿**

删掉「下游永远不能出现」，换成「下游出现了也绝不能污染上游」。
逐字节核查 §2 点名的全部五项：

| # | §2 要求 | 路径 | 冻结 sha256 |
|---|---------|------|-------------|
| 1 | B5 frozen metrics hash unchanged | `checkpoints/basilisk_b5/formal_metrics.json` | `6233e17139c1e606aae23f383ab20970cc48760a09ae38fa98cf2e7cfab8db82` |
| 2 | B6 frozen metrics hash unchanged | `checkpoints/basilisk_b6/all_metrics.json` | `3972059675cf911b60433dc327d36fd735ee5eb8fc4846e00ca06a6d81413b13` |
| 3 | B2.1 split hash unchanged | `docs/basilisk_b21/split_manifest.json` | `d8700e8a7bda7c45c1471c9039ad7e025507a8e0616553110c02226388917ebb` |
| 4 | source checkpoint hash unchanged | `checkpoints/source_tcn_pretrain.pt` | `d528c2d1b12b83e305bdc0809293e9b7ca8e84e07952dfb2c96bc46a5a8f7b66` |
| 5 | B1.9 feature hash unchanged | `data/features/wheel/basilisk_b19/target_features.h5` | `575d708ae9330dfa37d407cf639ebbbd098c473ce1fc137ff1eef8feaab9c293` |

哈希来源：`docs/basilisk_b6/baseline_contract.json` 与
`docs/basilisk_b7/baseline_contract.json` **两份契约取值一致**（第 2 项仅 B7 契约
覆盖，因为它是 B6 自身产物），且与当前磁盘实测值一致。任一项漂移 → `B6_INVALID`。

### `test_b6_undelegated_artifacts_still_absent`
未授权产物仍须缺席，**一条未放松**，共 16 项：
B7 的 metrics/summary/权重类（B7 是纯表达阶段，禁止训练）、
B8 的算法产物（B8 只允许打包 + Docker）、
星敏 S7/S8 全线（从未获得任何阶段授权）。

### `test_b6_stale_guard_retirement_is_documented`
强制本文件存在，且包含 `test_b6_old_artifacts_unchanged`、
`test_b6_b7_b8_artifacts_must_stay_absent`、`algorithm_numbers_affected`、
`data_affected` 四个关键字与「生命周期治理」声明。**禁止静默删除守卫。**

---

## 4. 反滥用

`tests/lifecycle_registry.py::LIFECYCLE_GUARD_RETIRED` 追加
`tests/basilisk_b6/test_b6_baseline_contract.py`（该文件被 B7 契约钉了哈希，
改造后哈希必然变化）。`assert_registry_is_not_widened()` 强制该豁免只能指向
`tests/` 下的测试文件 —— 任何试图把 `checkpoints/` / `data/` / `src/` /
`configs/` 或 `.h5` / `.pt` / `.py` 算法产物塞进豁免白名单的改动都会在测试期
被挡下。

三重数量上限防掏空：
```
_n_exempt            <= 3
_n_hashed            >= 20     # 逐字节实检项下限
_n_lifecycle_allowed <= len(DOWNSTREAM_LIFECYCLE 全部 artifacts)
```

---

## 5. 未修改清单（§0 绝对禁止事项核对）

| 项 | 状态 |
|----|------|
| `src/models/**` | 未改动 |
| `src/transfer/**` | 未改动 |
| `src/experiments/**` 算法逻辑 | 未改动 |
| source / B5 / B6 checkpoint | 未改动（逐字节核验 PASS） |
| B1.8 / B1.9 target feature | 未改动（逐字节核验 PASS） |
| B2.1 split | 未改动（逐字节核验 PASS） |
| MMD lambda / optimizer / loss / early stopping | 未改动 |
| HI / RUL / Basilisk profile / lifetime dataset / damage model | 未改动 |
| 任何正式 metrics JSON、B5/B6 正式结果数字 | 未改动 |
| 重新训练正式模型 | **未执行** |
| 重新跑 `--full` | **未执行** |
| 删除 `data/` 中任何现有数据文件 | **未执行（删除数 = 0）** |

**算法测试修改数 = 0。** 本次仅改 2 个文件：
- `tests/basilisk_b6/test_b6_baseline_contract.py`（生命周期治理测试）
- `tests/lifecycle_registry.py`（治理注册表）
