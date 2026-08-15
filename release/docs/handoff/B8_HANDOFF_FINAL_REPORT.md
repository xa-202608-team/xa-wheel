# B8-HANDOFF-FINAL 最终报告

**阶段**：B8-HANDOFF-FINAL（飞轮代码线终态冻结）
**时间戳**：2026-08-10T21:55:00+0800
**交付包**：`release/LEO_Bearings_Docker_Handoff/`（v1.0）
**结论**：`B8_HANDOFF_READY`

---

## 冻结声明

> 飞轮算法与实验线已冻结。后续只允许 Docker/环境/提交外裹层修改，不得再修改影响实验数字的代码。

---

## §16 逐项报告（18 项）

| # | 项目 | 结果 |
|---|------|------|
| 1 | 退役前 stale guards 数量 | **2**（同一根因：`STATUS_BASILISK_B7.md` 已获人工授权存在） |
| 2 | 已退役的 stale guards | **2**（均位于 `tests/basilisk_b6/test_b6_baseline_contract.py`：`test_b6_old_artifacts_unchanged` 的 `MISSING` 分支、`test_b6_b7_b8_artifacts_must_stay_absent`） |
| 3 | 被修改的算法测试数量 | **0**（`src/models/**`、`src/transfer/**`、`src/experiments/**` 算法逻辑、checkpoint、metrics JSON 全部未触碰） |
| 4 | 原仓库 pytest | **1098 passed, 29 skipped, 1 xfailed, 0 failed**（退役前 1092 passed + 2 failed；无新增 skip / 无新增 xfail） |
| 5 | release pytest | **488 passed, 185 skipped, 0 failed**（入口 `scripts/handoff/run_release_tests.py`，范围声明见 `RELEASE_TEST_SCOPE.txt`） |
| 6 | release `--fast` 输出 | `target_only 0.237267` / `source_finetune 0.241459` / `source_mmd 0.239710`；与原仓库 \|Δ\| = 4.54e-07 / 3.51e-07 / 1.39e-07，**全部 ≤ 1e-6** → `SMOKE_NUMERIC_MATCH_PASS` |
| 7 | source encoder 加载证据 | `source encoder loaded, missing = 8, unexpected = 0`（两次迁移路径各一次；encoder 全命中，heads/adapter 按设计不迁移） |
| 8 | 报告数字审计 | `REPORT_NUMERIC_AUDIT_PASS` —— 92 项 / 86 数值项 / 0 缺失 / 5 项语义冻结全 PASS |
| 9 | 冻结 hash 审计 | `FROZEN_EVIDENCE_AUDIT_PASS` —— 19 unchanged / 0 drift / 0 missing / 0 unpinned |
| 10 | `data/` 删除文件数 | **0**（原仓库 `data/` 文件数 9355，全程未变；`DATA_TREE_PROTECTED = true` guard 在任何删除前生效） |
| 11 | 清理文件数 | 38 目录 / **0 文件** / 11.11 MB；其中位于 `data/` 下 = **0**。额外移除 release 内 `src/__pycache__`（解释器再生字节码，非交付内容） |
| 12 | release 总文件数 | **471**（不含运行时再生的 `__pycache__`；其中 1 项为 manifest 自身） |
| 13 | release 总大小 | **1886.9 MB**（1.979 GB） |
| 14 | manifest 条目数 | **470**（`RELEASE_MANIFEST.sha256`；排除自引用 manifest 自身）→ `MANIFEST_VERIFY_PASS` 470/470，条目数 == 实际文件数，且**连续两次校验幂等** |
| 15 | 交接目录 | `release/LEO_Bearings_Docker_Handoff/`（版本标识 `HANDOFF_PACKAGE_VERSION.txt`，v1.0；`release/` 下无其它交接包，未覆盖任何已有交付） |
| 16 | HANDOFF_README | `release/LEO_Bearings_Docker_Handoff/HANDOFF_README.md` |
| 17 | DOCKER_CHECKLIST | `release/LEO_Bearings_Docker_Handoff/docs/handoff/DOCKER_CHECKLIST.md` |
| 18 | 最终判定 | **`B8_HANDOFF_READY`** |

---

## §1 判据满足性证明（为何不是 `B8_HANDOFF_INVALID`）

失败数恰为 **2**，未出现第三个失败。两条失败满足全部三条件：

- **A. 仅检查 downstream artifact absence / old hash** —— 失败断言位于 `exp == "MISSING"` 分支，
  该分支以 `continue` 结束，**代码路径上无法到达 sha256 比对**。这是结构性隔离，不是意图声明。
- **B. 失败原因仅为 B7/B8 已经人工授权存在** —— 唯一触发者 `STATUS_BASILISK_B7.md`，
  B7 阶段已获人工授权正式执行。
- **C. 不涉及**模型 / 指标 / 数据 / split / leakage / 统计结果 / checkpoint / 特征正确性 —— 同 A 的路径隔离。

---

## §2 退役方式（强化而非删除）

未删除任何测试文件，未注释任何 assert。移除的是「下游永远不能出现」这一已失效前提，
补入四条更强的约束：

1. `test_b6_downstream_stage_transition_is_governed` —— 下游出现则治理产物必须齐备
2. `test_b6_downstream_did_not_contaminate_upstream` —— **核心补偿**，逐字节校验 5 项上游冻结物
   （B5 formal_metrics / B6 all_metrics / B2.1 split_manifest / source checkpoint / B1.9 target_features）
3. `test_b6_undelegated_artifacts_still_absent` —— 16 条未授权路径仍须缺席（含 S7/S8 星敏线全部）
4. `test_b6_stale_guard_retirement_is_documented` —— 退役必须留有文档

反空心化断言保留：`_n_exempt <= 3`、`_n_hashed >= 20`、`_n_lifecycle_allowed <= len(whitelist)`、
`assert_registry_is_not_widened()`。

---

## §9 data/ 只读约定

`data/` 自本阶段起为 **READ ONLY**。原始公开数据 / processed source data / Basilisk profile /
target dataset / target feature / manifest·provenance 均不得删除。发现的旧产物仅登记于
`docs/handoff/DATA_CLEANUP_CANDIDATES.md`，**未产生任何删除动作**，是否清理由人类决定。

Docker 挂载建议：`./data:/workspace/data:ro`。不要把整个 data 树 COPY 进镜像层。

---

## 交给 Docker 负责人的边界

- **允许**：Dockerfile / docker-compose / 环境 pin / 入口脚本 / 镜像分层 / CI 外裹层
- **禁止**：任何会改变实验数字的改动（模型、迁移、损失、优化器、split、特征、checkpoint、metrics JSON）
- **验收用** `--verify` + `--fast`，**不要跑** `--full`（会覆写冻结证据且耗时数小时）
- 本阶段**未创建 Dockerfile**，按约定留给 Docker 负责人

---

`B8_HANDOFF_READY`
