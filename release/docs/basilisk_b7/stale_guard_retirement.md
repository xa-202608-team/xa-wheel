# BASILISK-B7 —— stale lifecycle guard 正式退役记录

**阶段**：BASILISK-B7（结果表达与证据冻结阶段）
**判定**：`B7_REPORT_GOVERNANCE_READY`
**授权来源**：BASILISK-B7 §2「允许修改」+ §14「stale lifecycle guards 正式退役」+ 本阶段人工追加的治理授权（10 条）
**前序预告**：`STATUS_BASILISK_B6.md:197-203` 已明确记录「已知的一个 stale 生命周期 guard」，并写明「**本阶段未删除、未修改该测试**，在 B7/S8 退役。详见 `stale_guard_retirement.md`」。本文件即该预告的兑现。

---

## 0. 一句话结论

15 个失败测试**全部**属于 A 类 `LIFECYCLE_STALE`，零个属于 B–G 类；退役后全库 `1055 passed, 28 skipped, 1 xfailed, 0 failed`，**任何实验数字未发生一比特变化**，且被退役的哈希约束被一个**更强的语义约束**取代。

---

## 1. 失败分类表（先分类，后动手）

按治理授权第 1 条，先把全部失败逐条分类，只有 A 类允许退役。`guard_type` 取值域：
A. `LIFECYCLE_STALE` / B. `ALGORITHMIC` / C. `DATA_INTEGRITY` / D. `LEAKAGE` / E. `METRIC_CORRECTNESS` / F. `REPRODUCIBILITY` / G. `UNKNOWN`。

| # | test_name | test_file | failure_message | guarded_path | guard_type |
|---:|---|---|---|---|---|
| 1 | `test_old_results_hash_unchanged` | `tests/basilisk/test_dataset_isolation.py` | `docs/results.md: 0960790eeb17 -> 143be454d721` | `docs/results.md` | **A** |
| 2 | `test_old_basilisk_v1_unchanged` | `tests/basilisk_b1/test_old_basilisk_v1_unchanged.py` | 同上 | `docs/results.md` | **A** |
| 3 | `test_b11_old_b1_unchanged` | `tests/basilisk_b11/test_b11_baseline_and_protocol.py` | 同上 | `docs/results.md` | **A** |
| 4 | `test_b12_baseline_unchanged` | `tests/basilisk_b12/test_old_b11_unchanged.py` | 同上 | `docs/results.md` | **A** |
| 5 | `test_b13_old_artifacts_unchanged` | `tests/basilisk_b13/test_b13_baseline_contract.py` | 同上 | `docs/results.md` | **A** |
| 6 | `test_b14_old_artifacts_unchanged` | `tests/basilisk_b14/test_b14_baseline_contract.py` | 同上 | `docs/results.md` | **A** |
| 7 | `test_before_after_contract_verifies_clean` | `tests/basilisk_b14/test_b14_baseline_contract.py` | `契约项漂移: ['docs/results.md']` | `docs/results.md` | **A** |
| 8 | `test_old_artifacts_unchanged` | `tests/basilisk_b15/test_b15_baseline_contract.py` | `以下文件被修改: [('docs/results.md', ...)]` | `docs/results.md` | **A** |
| 9 | `test_b16_baseline_unchanged` | `tests/basilisk_b16/test_b16_baseline.py` | `既有产物被改动 (B1.6 禁止)` | `docs/results.md` | **A** |
| 10 | `test_b17_old_artifacts_unchanged` | `tests/basilisk_b17/test_b17_baseline_contract.py` | `既有产物被改动 (B1.7 禁止)` | `docs/results.md` | **A** |
| 11 | `test_b18_old_artifacts_unchanged` | `tests/basilisk_b18/test_b18_baseline_contract.py` | `旧产物被改动` | `docs/results.md` | **A** |
| 12 | `test_b18_readonly_sources_untouched` | `tests/basilisk_b18/test_b18_baseline_contract.py` | `docs/results.md 被修改` | `docs/results.md` | **A** |
| 13 | `test_b19_old_artifacts_unchanged` | `tests/basilisk_b19/test_b19_old_artifacts_unchanged.py` | `只读产物被改动` | `docs/results.md` | **A** |
| 14 | `test_b6_old_artifacts_unchanged` | `tests/basilisk_b6/test_b6_baseline_contract.py` | `冻结产物被改动 (B6_INVALID)` × 3 项 | `docs/results.md`、`tests/basilisk_b21/test_b21_discipline.py`、`tests/basilisk_b5/test_b5_baseline_contract.py` | **A** |
| 15 | `test_b6_prior_stages_untouched` | `tests/basilisk_b6/test_b6_baseline_contract.py` | `前序产物被 B6 改动: tests/basilisk_b21/test_b21_discipline.py` | `tests/basilisk_b21/test_b21_discipline.py` | **A** |

**分类统计**：A = 15，B = 0，C = 0，D = 0，E = 0，F = 0，G = 0。

**唯一 guarded_path 集合**（15 个失败仅由 3 个文件驱动）：

1. `docs/results.md` —— B7 授权改写的最终交付表达面
2. `tests/basilisk_b21/test_b21_discipline.py` —— B7 §14 授权改造的 lifecycle guard 测试
3. `tests/basilisk_b5/test_b5_baseline_contract.py` —— 同上

**零个失败涉及**：模型权重、数据集、特征、split、metrics JSON、verdict、bootstrap、RUL/HI 定义、训练公平性、checkpoint 选择。

---

## 2. A 类资格逐条核验（5 条严格定义）

治理授权第 2 条要求同时满足 5 条才算 `LIFECYCLE_STALE`。逐条核验：

| 条件 | 核验结果 |
|---|---|
| ① 保护的是历史阶段"未来产物必须不存在/不得改变"的生命周期约束 | ✅ 全部 15 条断言的语义都是"前阶段冻结时的磁盘状态不得改变"。 |
| ② 当前 B7 已获明确人工授权产生或修改该产物 | ✅ B7 §2 明确列出 `docs/results.md` 为允许修改；§14 明确授权退役 stale lifecycle guards；本阶段追加授权再次确认。 |
| ③ 失败仅由 `Path.exists()` / SHA256 changed / must_stay_absent 这类断言引起 | ✅ 15/15 均为 SHA256 比对失败，无一例外（见第 1 节 failure_message 列）。 |
| ④ **不检查** 模型数值 / split / labels / leakage / metrics / bootstrap / RUL / HI / training fairness / checkpoint selection | ✅ 这些断言只做字节比对，不读取任何数值内容。同文件内的数值断言（如 `test_contract_numeric_conclusions_unchanged`、`test_b6_source_checkpoint_and_encoder_frozen`）**全部未修改且全部通过**。 |
| ⑤ 去掉该 guard 不会改变任何实验数字 | ✅ `docs/results.md` 不被任何训练/评估/仿真脚本读取；两个测试文件是测试代码。改动后 124 项算法产物/源码/配置逐字节复核 → `DRIFT: NONE`。 |

---

## 3. 逐条退役记录（7 字段格式）

以下按治理授权第 7 条的 7 字段格式逐条记录。为避免重复，第 1–13 条共享同一模式，故按"guarded_path + 失败测试组"聚合为 3 条记录；每条都完整给出 7 个字段。

### 记录 R1 —— `docs/results.md` 的哈希冻结（覆盖失败 #1–#14）

| 字段 | 内容 |
|---|---|
| **old_test** | `test_old_results_hash_unchanged`、`test_old_basilisk_v1_unchanged`、`test_b11_old_b1_unchanged`、`test_b12_baseline_unchanged`、`test_b13_old_artifacts_unchanged`、`test_b14_old_artifacts_unchanged`、`test_before_after_contract_verifies_clean`、`test_old_artifacts_unchanged`(b15)、`test_b16_baseline_unchanged`、`test_b17_old_artifacts_unchanged`、`test_b18_old_artifacts_unchanged`、`test_b18_readonly_sources_untouched`、`test_b19_old_artifacts_unchanged`、`test_b6_old_artifacts_unchanged`（共 14 个测试，分布于 13 个文件） |
| **old_guard** | `sha256(docs/results.md) == "0960790eeb170d04febe83d2f1f4cb50d490ffe7fb95c40217fc9ddcd3983769"`，登记于 16 份 `docs/basilisk_*/baseline_contract.json` 的 `frozen_docs` / `b17_reconfirmed_readonly` / `b18_reconfirmed_readonly` / `b19_reconfirmed_readonly` 组 |
| **why_it_was_valid_before** | B1 → B6 期间 `docs/results.md` 的角色是**前阶段只读引用文档**。那些阶段的正当性建立在"只新增自己命名空间的产物、绝不回头改动既有结论文档"之上。若某阶段偷偷改写 `results.md` 去美化历史数字，整条证据链就失效。因此"哈希永久不变"在 B1–B6 期间是一条**必要且正确**的约束。 |
| **authorization_transition** | ① BASILISK-B7 §2 明确把 `docs/results.md` 列入「允许修改」清单；② B7 §5 要求在 `docs/results.md` 新章节写入四张主表（Table A–D）；③ B7 §18 要求最终化该文件（清理 smoke / 临时 gate 表 / S3 数字 / TODO / TBD）；④ 本阶段人工追加授权第 3–5 条明确规定 `docs/results.md` 由 `frozen numeric source` 转为 `final presentation artifact`，并要求用语义冻结取代哈希冻结；⑤ `STATUS_BASILISK_B6.md:197-203` 已预告该 guard 在 B7/S8 退役。 |
| **why_it_is_stale_now** | 该 guard 保护的命题是"未来阶段不得改动此文件"。B7 是**被显式授权改动此文件的阶段**，因此该命题的前提已被授权取代。继续保留它不再保护任何数值正确性，只会让一个已获授权的交付动作永久呈现为测试失败——即"用契约禁止契约自己授权的事"。 |
| **replacement_guard** | ① 新增共享注册表 `tests/lifecycle_registry.py`：`PRESENTATION_MUTABLE`（封闭白名单）+ `hash_exempt()` + `assert_registry_is_not_widened()` 反向护栏；② 新增 `scripts/basilisk_b7/audit_report_numbers.py`：**直接读取** `checkpoints/basilisk_b5/*` 与 `checkpoints/basilisk_b6/*`，对 86 个数字做精确字符串比对，输出 `REPORT_NUMERIC_AUDIT_PASS`；③ 五项语义冻结布尔值（见第 4 节）；④ `test_b18_readonly_sources_untouched` 中该路径的哈希断言被替换为**语义审计存在性 + 通过性断言**（不是删除检查，而是换成更强的检查）；⑤ 每个接入点新增三连防掏空断言：`assert_registry_is_not_widened()` + `_n_exempt <= 3` + `_n_hashed >= 20`。 |
| **does_not_affect_numbers** | **true** —— `docs/results.md` 不被任何训练 / 评估 / 仿真 / 特征工程脚本读取（它是纯 Markdown 交付文档）。改动后以 B6 契约为参照复核 124 项算法产物、源码与配置 → `DRIFT: NONE`。 |

### 记录 R2 —— `tests/basilisk_b21/test_b21_discipline.py` 的哈希冻结（覆盖失败 #14、#15）

| 字段 | 内容 |
|---|---|
| **old_test** | `test_b6_old_artifacts_unchanged`、`test_b6_prior_stages_untouched`（`tests/basilisk_b6/test_b6_baseline_contract.py`） |
| **old_guard** | `sha256(tests/basilisk_b21/test_b21_discipline.py) == "fb84202d..."`，登记于 `docs/basilisk_b5/baseline_contract.json` 与 `docs/basilisk_b6/baseline_contract.json` 的 `b21_tests` 组；并且该测试文件内部的 `test_b21_b5_absent_and_never_auto_run` 断言 B5 产物**必须永远不存在** |
| **why_it_was_valid_before** | B2.1 阶段无权产出迁移结论。当时若 B5 产物出现，说明有人越权跑了正式迁移评估并可能据此改写结论，因此"B5 产物必须缺席"是 B2.1 的核心纪律。B5/B6 把该测试文件本身钉哈希，是为了防止后来者悄悄放宽这条纪律。 |
| **authorization_transition** | ① B5 阶段获得显式授权成为「第一个允许产生正式迁移结论的阶段」（`checkpoints/basilisk_b5/summary.json: first_stage_allowed_to_conclude_transfer = true`）；② B6 获授权产出低标签矩阵；③ B7 §14 明确授权把"下游产物必须永远不存在"改造为"只有在获得显式阶段授权且存在正确 protocol / baseline contract / final verdict 时，下游产物才允许存在"；④ 本阶段追加授权第 8 条把修改权限限定为"实际被确认属于 LIFECYCLE_STALE 的 tests lifecycle guard 文件"。 |
| **why_it_is_stale_now** | B5 与 B6 都已在授权下完成并冻结，其产物**合法存在**。一条断言"B5 产物必须缺席"的测试在 B5 已合法完成后必然失败，且它保护的不再是纪律而是一个过期的阶段假设。 |
| **replacement_guard** | 该测试**未被删除**，改造为 lifecycle transition guard，拆为三部分：(1) 历史事实断言（契约冻结当时 B5 确实登记为 MISSING —— 历史证据保留）；(2) **仍然强断言**越权结论产物 `formal_transfer_verdict.json` / `positive_transfer_proven.json` 至今必须缺席（B2.1 越权下迁移结论永久禁止，与阶段推进无关）；(3) B2.1 自身行为记录（`b5_auto_run is False`、`transfer_run is False`、源码不含 `basilisk_b5` / `train_transfer`）。另新增 `test_b21_downstream_stages_are_governed_not_forbidden`：下游产物存在时必须同时具备 protocol / baseline contract / final verdict。 |
| **does_not_affect_numbers** | **true** —— 这是测试代码，不参与任何数值计算。B2.1 的 split (`23e2b944...`) 与全部 B2.1 结果 JSON 逐字节未变。 |

### 记录 R3 —— `tests/basilisk_b5/test_b5_baseline_contract.py` 的哈希冻结（覆盖失败 #14）

| 字段 | 内容 |
|---|---|
| **old_test** | `test_b6_old_artifacts_unchanged`（`tests/basilisk_b6/test_b6_baseline_contract.py`） |
| **old_guard** | `sha256(tests/basilisk_b5/test_b5_baseline_contract.py) == "d3f089b4..."`，登记于 `docs/basilisk_b6/baseline_contract.json` 的 `b5_tests` 组；该文件内部的 `test_b5_b6_artifacts_must_stay_absent` 断言 B6 产物必须永远不存在 |
| **why_it_was_valid_before** | B5 完成时明令「不自动跑低数据矩阵，不自动执行 B6」（`summary.json: exit_discipline`、`forbid_auto_run_b6 = true`）。若 B6 产物在 B5 阶段出现，说明流程越权自动推进，B5 的结论就不可信。 |
| **authorization_transition** | ① B6 已在显式授权下完成，产出 `protocol_hash.json`(`433ccaf2...`)、`label_subset_manifest.json`(`f7fe1aba...`)、`all_metrics.json`、`final_verdict.json`；② B7 §14 授权把 must_stay_absent 改造为"授权 + protocol + contract + verdict 齐备才允许存在"；③ 本阶段追加授权第 8 条。 |
| **why_it_is_stale_now** | B6 已合法完成并冻结，其产物合法存在，"B6 产物必须缺席"因此必然失败且不再保护任何东西。 |
| **replacement_guard** | 该测试**未被删除**，改为历史事实断言（契约当年确实登记 MISSING），运行期存在性判断移交三个新增守卫：`test_b5_downstream_stage_transition_is_governed`（下游存在必须治理齐备）、`test_b5_undelegated_artifacts_still_absent`（18 项**仍然禁止**的产物：B7 权重与新评估结果 / B8 算法产物 / B9 全部 / B5 低数据与截断旁路矩阵）、`test_b5_stale_guard_retirement_is_documented`（退役必须有文档）。`test_b5_old_artifacts_unchanged` 保留逐字节比对并加双重防掏空断言（`_n_hash_checked >= 350`、`n_hash_exempt <= len(HASH_EXEMPT)`）。 |
| **does_not_affect_numbers** | **true** —— 测试代码。B5 的 `summary.json` / `paired_gain.json` / `formal_metrics.json` / `warning_metrics.json` / `lifetime_bins.json` / 15 个 `.pt` 权重逐字节未变。 |

---

## 4. 语义冻结取代哈希冻结

按治理授权第 5 条，`docs/results.md` 的旧约束 `hash(...) == old_hash` 被替换为五项机器可核对的语义约束，记录在 `checkpoints/basilisk_b7/report_number_audit.json`：

| 语义冻结键 | 含义 | 实测 |
|---|---|---|
| `report_numbers_match_frozen_metrics` | 86 个被审计数字全部溯源到冻结的 B5/B6 machine-readable artifacts | ✅ PASS（未命中 0 项） |
| `report_contains_final_transfer_conclusion` | 含 `NO_POSITIVE_TRANSFER_SUPPORTED` | ✅ PASS |
| `report_contains_engineering_recommendation` | 含 `damage_extrapolation` 工程推荐 | ✅ PASS |
| `report_contains_negative_transfer_disclosure` | 含边界措辞「未能证明正向迁移」（而非"证明迁移无效"） | ✅ PASS |
| `report_contains_damage_baseline` | damage 基线出现在正表而非脚注 | ✅ PASS |

**为什么这比哈希更强**：哈希只能证明"文件没被人动过"；语义审计能证明"文件里每一个数字都与冻结产物对得上"。哈希对"一开始就写错的数字"完全无能——它会把错误一起冻结。语义审计则会当场失败。

**实证**：本次审计首轮即抓到一个真实缺口——`docs/results.md` 缺少 B5 全标签配对增益与其 CI95（6 个数字）。这正是旧哈希 guard **永远不可能发现**的问题。已按冻结值补入 §2.1（`-0.001635` / `-0.000583` 及两条跨零 CI95），非手敲近似值。

审计脚本同时执行 §17 禁止措辞检查（7 个短语），并按行判定：带 `不得` / `禁止` / `historical` / `rejected` 等标记的行是**在禁止**该措辞而非使用它，不计违规。

---

## 5. 反滥用设计（防止本次退役被当成先例滥用）

| 机制 | 位置 | 作用 |
|---|---|---|
| 封闭白名单 | `tests/lifecycle_registry.py: PRESENTATION_MUTABLE` | 只含 `docs/results.md`；目录级只含 `docs/figures/`、`docs/技术方案报告/`、`docs/basilisk_b7/`、`docs/数据集下载/` |
| 反向护栏 | `assert_registry_is_not_widened()` | 三层校验（不可变目录前缀 / 不可变后缀 `.h5 .npz .pt .yaml .py` / 关键产物文件名如 `split_manifest.json`、`protocol.json`、`target_features.h5`、`all_metrics.json`）。任何未来"顺手加一项"算法产物都会当场失败。 |
| 豁免上界 | 每个接入点 `_n_exempt <= 3` | 豁免数不得超过白名单规模 |
| 实检下界 | 每个接入点 `_n_hashed >= 20` | 绝大多数项仍逐字节比对，防止契约被掏空 |
| 测试文件限定 | `test_b6_prior_stages_untouched` 中 `assert rel.startswith("tests/")` | 前序产物豁免只能命中测试文件；B1.8/B1.9/B2/B2.1/B5 的数据、特征、权重、metrics、verdict 全部仍在逐字节比对 |
| 项数自洽保留 | b12 / b14 / b15 / b19 | 豁免项仍计入 `n_file` / `n_checked` 总账，维持 `n_file + n_absent == n_files` 等自洽断言不被绕过 |

**明确不在 mutable 名单内**（按治理授权第 3 条）：`checkpoints/**`、`target_features.h5`、`wheel_all.h5`、`split_manifest.json`、`protocol.json/md`、source checkpoints、metrics raw JSON、model weights、任何影响数字的 configs。

---

## 6. 未被修改的内容（治理授权第 6、8 条）

**历史证据完整保留**：B1.8 / B1.9 / B2 / B2.1 / B3X / B4X / B5 / B6 的 results JSON、protocols、hashes、STATUS、limitations、checkpoints **全部逐字节未变**；历史阶段 verdict 未被重写（`B2_GENERALIZATION_FAIL`、`B13_CALIBRATION_FAIL`、`B16_NO_DOCUMENTED_WHEEL`、`B5_NO_POSITIVE_TRANSFER`、`B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER` 等均原样保留）。

**零修改的测试类别**：model tests、metrics tests、leakage tests、split tests、transfer invariants、statistical tests、feature correctness tests —— 一个都没碰。

**未新增任何 xfail 来隐藏失败**：xfailed 数量 1（治理前 1，治理后 1）；skipped 数量 28（治理前 28，治理后 28）。

---

## 7. 退役后实测状态

| 项 | 治理前 | 治理后 |
|---|---|---|
| `pytest tests/ -q` | 15 failed, 1040 passed, 28 skipped, 1 xfailed | **0 failed, 1055 passed, 28 skipped, 1 xfailed** |
| `audit_report_numbers.py` | 不存在 | **`REPORT_NUMERIC_AUDIT_PASS`**（86 数字项，0 未命中） |
| 算法产物 / 源码 / 配置逐字节复核 | — | **124 项，`DRIFT: NONE`** |
| 契约内 `tests/` 文件变更 | — | **仅 2 项**（`test_b21_discipline.py`、`test_b5_baseline_contract.py`），19 项未变 |
| 守卫测试净数量 | — | **净增**（3 个 stale guard 改造为 transition guard，无一删除；另新增 6 个守卫测试） |

**判定**：`B7_REPORT_GOVERNANCE_READY`
