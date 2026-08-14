# 上游生命周期守卫的退役说明（BASILISK-B6 §22）

本文件解释一件事：为什么全量测试里出现的**陈旧生命周期守卫失败**在 B6 阶段必然发生，
且为什么这些失败**不是**算法问题。

主对象是 §22 点名的
`tests/basilisk_b21/test_b21_discipline.py::test_b21_b5_absent_and_never_auto_run`；
B6 实测发现另外两个**同类**失败（同一机制，只是钉的是 B6 自己的产物），
一并在本文件定性归档，见末节《全量测试的三个同类失败》。

本阶段**不删除、不修改**这些测试（`stale_guard.forbid_deleting_in_this_stage = true`，
`forbid_editing_frozen_b21_test = true`）。三个测试所在的两个文件本身也都已被
B6 基线契约按哈希冻结（`tests/basilisk_b21/test_b21_discipline.py` →
`fb84202d…`，`tests/basilisk_b5/test_b5_baseline_contract.py` → `d3f089b4…`），
改动它们会直接触发 `B6_INVALID`。它们在 B7/S8 的代码冻结治理阶段一并退役。


---

## guard_purpose（守卫当初要防什么）

B2.1 阶段的授权范围只到"修 lifetime-support mismatch 并重新验证泛化"。
当时明确禁止顺手把正式迁移实验（B5）一起跑掉——否则会出现"划分刚改完、
还没被独立确认，就用它产出正式迁移结论"的既成事实。

该守卫因此断言两件事：

1. **契约层**：B5 产物在 B2.1 的基线契约里登记为 `MISSING`
   （`flat_sha256[rel] == "MISSING"`）。
2. **文件系统层**：这些路径当时在磁盘上确实不存在
   （`not (ROOT / rel).exists()`）。

守卫覆盖的路径：

| 路径 | 当前磁盘状态 |
|---|---|
| `checkpoints/basilisk_b5/metrics.json` | 仍然不存在（B5 的指标文件名是 `formal_metrics.json`） |
| `docs/basilisk_b5/results.md` | **存在** —— B5 经人工授权正式运行后产出 |
| `STATUS_BASILISK_B5.md` | **存在** —— 同上 |
| `checkpoints/basilisk_b21/formal_transfer_verdict.json` | 仍然不存在 |
| `checkpoints/basilisk_b21/positive_transfer_proven.json` | 仍然不存在 |

## authorization_transition（授权状态的迁移）

- B2.1 期间：B5 **未获授权**，守卫的两层断言都成立。
- B5 阶段：用户以正式 `/goal` 指令**明确授权**运行 B5 正式迁移实验，
  seeds `[112..116]`，结论 `B5_NO_POSITIVE_TRANSFER`。
- 于是"B5 产物必须缺席"这个前提被**人为授权**推翻，而不是被代码绕过。

## why_obsolete（为什么它现在是陈旧守卫）

失败发生在**文件系统存在性**断言上，不在契约哈希断言上。实测：

```
assert flat[rel] == "MISSING", rel          # ← 通过（契约未被改动）
assert not (ROOT / rel).exists(), rel       # ← 在 docs/basilisk_b5/results.md 处失败
```

据此逐条核对 §22 要求的三件事：

| 必须核实 | 结论 |
|---|---|
| `only_reason_is_authorized_b5_existence` | 成立。唯一失败原因是 B5 产物按人工授权存在。 |
| `not_an_algorithmic_error` | 成立。断言不涉及任何指标、划分、模型、统计量。 |
| `not_a_hash_change` | 成立。`flat_sha256` 的 `MISSING` 断言全部通过；没有任何冻结哈希发生变化。 |

换言之：该守卫记录的是**一个已经过期的生命周期前提**，而不是一个被破坏的不变量。
把它改绿的正确方式是在治理阶段整体退役它，而不是在结果阶段偷偷改断言——
后者会让"守卫失败"这件事失去信号价值。

## replacement_guard（接替它的守卫）

B6 已经建立等价但指向**下一个**未授权阶段的守卫：

- 契约组 `b6_must_stay_absent`（14 条 B7/S7/B8/S8 产物，全部登记为 `MISSING`）。
- 测试 `tests/basilisk_b6/test_b6_baseline_contract.py::test_b6_b7_b8_artifacts_must_stay_absent`。
- `checkpoints/basilisk_b6/final_verdict.json` 中的
  `b7_auto_run = false` / `b8_auto_run = false`。

即：守卫的语义没有丢失，只是把边界从"B5 未授权"前移到"B7/B8 未授权"。

## 本阶段的报告纪律

B6 的完整测试结果允许且仅允许出现**已知陈旧生命周期守卫**类别的失败
（`allowed_failure_kind = known_stale_lifecycle_guard`）。
出现任何**非该类别**的 failure 即判 `B6_INVALID` 并停止。

退役阶段：`B7_S8_CODE_FREEZE_GOVERNANCE`。

---

## 全量测试的三个同类失败（实测归档）

`python -m pytest tests/ -q` → **3 failed, 1048 passed, 28 skipped, 1 xfailed**。
三个失败全部是同一个机制：**上游阶段的契约把下游阶段产物钉为 `MISSING`，
而下游阶段现已获人工授权运行**。

| # | 测试 | 钉住的路径 | 授权来源 |
|---|---|---|---|
| 1 | `tests/basilisk_b21/…::test_b21_b5_absent_and_never_auto_run` | `docs/basilisk_b5/results.md`、`STATUS_BASILISK_B5.md` | B5 的正式 `/goal` 授权 |
| 2 | `tests/basilisk_b5/…::test_b5_old_artifacts_unchanged` | `docs/basilisk_b6/results.md`、`STATUS_BASILISK_B6.md` | **B6 的正式 `/goal` 授权** |
| 3 | `tests/basilisk_b5/…::test_b5_b6_artifacts_must_stay_absent` | `docs/basilisk_b6/results.md` | 同上 |

失败 #2 / #3 是 B5 契约在写的时候把 B6 产物钉为必须缺席——这正是 §22 描述的
同一个接力问题，只是方向从"B2.1 钉 B5"变成"B5 钉 B6"。B6 的
`verify_baseline.py` 已经为自己做过等价处理（把 `b5_must_stay_absent` 组里
指向 `checkpoints/basilisk_b6` / `docs/basilisk_b6` / `STATUS_BASILISK_B6`
的项剔除，并用 `b6_must_stay_absent` 接管），但 B5 的**测试文件**是冻结件，
不能同步修改，因此该失败必然浮现。

### 逐项取证（三条 §22 必核事项）

```
B5 契约 hash 不符:                 []      ← 零
B5 契约冻结文件消失:               []      ← 零
B5 契约中非-B6 的意外存在:         []      ← 零
b5_must_stay_absent 非 B6 两项:    checkpoints/basilisk_b5/low_data_matrix.json   exists=False
                                   checkpoints/basilisk_b5/truncation_matrix.json exists=False
```

- `only_reason_is_authorized_downstream_existence`：成立。三个失败的**唯一**原因
  都是"经人工授权的下游阶段产物现在存在"。
- `not_an_algorithmic_error`：成立。三处断言都只做 `Path.exists()`，
  不涉及任何指标、划分、模型、统计量。
- `not_a_hash_change`：成立。全部 hash 断言通过；B6 自己的权威校验
  `verify_baseline.py --tag after` 报 **契约项数 1199, 不变 1199 →
  `B6_BASELINE_CONTRACT_OK`**。
- B5 自己那两条"不得自动跑低数据/截断矩阵"的实质性禁令**仍然成立**
  （两个文件都不存在）——即 B5 的实质纪律没有被破坏，
  破的只是"B6 尚未运行"这个已过期的时间前提。

### 结论

**算法性测试全部通过。** 三个失败同属 `known_stale_lifecycle_guard`，
零 hash 变化、零冻结文件丢失、零实质禁令破坏。
本阶段不修改其中任何一个（两个文件均为冻结件），
统一在 `B7_S8_CODE_FREEZE_GOVERNANCE` 阶段退役。

