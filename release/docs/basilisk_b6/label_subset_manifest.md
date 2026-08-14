# BASILISK-B6 标签子集清单 (label subset manifest)

> 本文件由 `scripts/basilisk_b6/audit_label_subsets.py` 生成, 内容取自 `checkpoints/basilisk_b6/label_subset_manifest.json`。清单在**任何训练之前**生成并冻结 (§7), 写出后禁止更改。

## 0. 冻结标识

| 项 | 值 |
|---|---|
| subset_manifest_sha256 | `f7fe1ababe7b23db5d0ba41a7c644505267f684400a22aebb5144bf6ce041b1d` |
| subset_seed | `20260814` |
| protocol_sha256 | `433ccaf2515669c1f5afbc76029de068329afc7f67f0c10ae3372d14409b8412` |
| split_sha256 | `23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932` |
| lifetime bin edges | `[26846.0, 37395.0]` (取自 B2.1, 不从 B6 test 重算) |
| 选取方式 | 各 bin 内 sorted(tids) 后用 np.random.default_rng([subset_seed, bin_index]) 做一次确定性置换; 各档从同一置换取前 k 条 -> 各档嵌套。 |
| 审计判定 | **B6_LABEL_SUBSET_AUDIT_OK** (40 项, 失败 0) |

## 1. 稀缺轴的定义

- 稀缺的是: `event_observed_failure_labelled_trajectories`
- **不是**: `total_target_trajectories`
- 每档始终保留 B2.1 train 中全部 **24 条 censored 轨迹**; 未被选中的 event 轨迹**完全移除**, 不改标成 censored, 也不作为 unlabelled 输入带回。
- 理由: 若把稀缺定义成 `n_train` 总量下降, 就会重新制造 B2 的 coverage mismatch —— 那是在测另一个 (已知会失败的) 问题。

## 2. 各档 train 组成

| n_event_labeled | 角色 | event | censored | train 总数 | short | medium | long | 移除 event |
|---|---|---|---|---|---|---|---|---|
| 3 | SECONDARY | 3 | 24 | 27 | 1 | 1 | 1 | 18 |
| 5 | PRIMARY (唯一正式判定档) | 5 | 24 | 29 | 2 | 1 | 2 | 16 |
| 10 | SECONDARY | 10 | 24 | 34 | 3 | 3 | 4 | 11 |
| 21 | SECONDARY | 21 | 24 | 45 | 7 | 7 | 7 | 0 |

## 3. 逐档轨迹 ID

### n_event_labeled = 3 — SECONDARY

| bin | 配额 | 轨迹 ID (EOL) |
|---|---|---|
| short | 1 | `traj_047` (EOL=23623) |
| medium | 1 | `traj_092` (EOL=36955) |
| long | 1 | `traj_015` (EOL=50139) |

- event train IDs: `traj_015`, `traj_047`, `traj_092`
- 移除的 event (18 条): `traj_008`, `traj_012`, `traj_013`, `traj_025`, `traj_029`, `traj_036`, `traj_045`, `traj_079`, `traj_082`, `traj_103`, `traj_104`, `traj_109`, `traj_111`, `traj_114`, `traj_115`, `traj_128`, `traj_133`, `traj_149`
- censored (24 条): 全部保留, 与 B2.1 train 一致

### n_event_labeled = 5 — PRIMARY

| bin | 配额 | 轨迹 ID (EOL) |
|---|---|---|
| short | 2 | `traj_047` (EOL=23623)<br>`traj_115` (EOL=23554) |
| medium | 1 | `traj_092` (EOL=36955) |
| long | 2 | `traj_015` (EOL=50139)<br>`traj_133` (EOL=37610) |

- event train IDs: `traj_015`, `traj_047`, `traj_092`, `traj_115`, `traj_133`
- 移除的 event (16 条): `traj_008`, `traj_012`, `traj_013`, `traj_025`, `traj_029`, `traj_036`, `traj_045`, `traj_079`, `traj_082`, `traj_103`, `traj_104`, `traj_109`, `traj_111`, `traj_114`, `traj_128`, `traj_149`
- censored (24 条): 全部保留, 与 B2.1 train 一致

### n_event_labeled = 10 — SECONDARY

| bin | 配额 | 轨迹 ID (EOL) |
|---|---|---|
| short | 3 | `traj_047` (EOL=23623)<br>`traj_109` (EOL=22281)<br>`traj_115` (EOL=23554) |
| medium | 3 | `traj_079` (EOL=34435)<br>`traj_092` (EOL=36955)<br>`traj_103` (EOL=34195) |
| long | 4 | `traj_013` (EOL=41867)<br>`traj_015` (EOL=50139)<br>`traj_114` (EOL=38436)<br>`traj_133` (EOL=37610) |

- event train IDs: `traj_013`, `traj_015`, `traj_047`, `traj_079`, `traj_092`, `traj_103`, `traj_109`, `traj_114`, `traj_115`, `traj_133`
- 移除的 event (11 条): `traj_008`, `traj_012`, `traj_025`, `traj_029`, `traj_036`, `traj_045`, `traj_082`, `traj_104`, `traj_111`, `traj_128`, `traj_149`
- censored (24 条): 全部保留, 与 B2.1 train 一致

### n_event_labeled = 21 — SECONDARY

| bin | 配额 | 轨迹 ID (EOL) |
|---|---|---|
| short | 7 | `traj_025` (EOL=25316)<br>`traj_029` (EOL=23707)<br>`traj_036` (EOL=21658)<br>`traj_047` (EOL=23623)<br>`traj_109` (EOL=22281)<br>`traj_115` (EOL=23554)<br>`traj_149` (EOL=22336) |
| medium | 7 | `traj_008` (EOL=28558)<br>`traj_012` (EOL=34225)<br>`traj_079` (EOL=34435)<br>`traj_082` (EOL=37371)<br>`traj_092` (EOL=36955)<br>`traj_103` (EOL=34195)<br>`traj_111` (EOL=27708) |
| long | 7 | `traj_013` (EOL=41867)<br>`traj_015` (EOL=50139)<br>`traj_045` (EOL=41892)<br>`traj_104` (EOL=40384)<br>`traj_114` (EOL=38436)<br>`traj_128` (EOL=45886)<br>`traj_133` (EOL=37610) |

- event train IDs: `traj_008`, `traj_012`, `traj_013`, `traj_015`, `traj_025`, `traj_029`, `traj_036`, `traj_045`, `traj_047`, `traj_079`, `traj_082`, `traj_092`, `traj_103`, `traj_104`, `traj_109`, `traj_111`, `traj_114`, `traj_115`, `traj_128`, `traj_133`, `traj_149`
- 移除的 event (0 条): 无
- censored (24 条): 全部保留, 与 B2.1 train 一致

## 4. 嵌套性

各 bin 只洗牌一次, 各档从同一顺序取前 k 条 —— 故各档嵌套, 标签变多只新增轨迹。这样 §17 的"标签越少增益越大"趋势不会被"换了一批不同轨迹"混淆。

| 关系 | 是否子集 | 新增轨迹 |
|---|---|---|
| n=3 ⊂ n=5 | 是 | `traj_115`, `traj_133` |
| n=5 ⊂ n=10 | 是 | `traj_013`, `traj_079`, `traj_103`, `traj_109`, `traj_114` |
| n=10 ⊂ n=21 | 是 | `traj_008`, `traj_012`, `traj_025`, `traj_029`, `traj_036`, `traj_045`, `traj_082`, `traj_104`, `traj_111`, `traj_128`, `traj_149` |

## 5. val / test

- val 31 条 / test 74 条, 逐条沿用 B2.1, **B6 全程不改动** (§5)。
- 右删失轨迹无真实 EOL, 相关指标保持 NaN, 不转 0。

## 6. 审计明细

| 审计项 | 结果 | 说明 |
|---|---|---|
| `n3_event_count` | PASS | 选中 event 3 条, 去重后 3, 要求 3 |
| `n3_all_censored_retained` | PASS | censored 24/24, 与 B2.1 train censored 集合一致 |
| `n3_bin_coverage` | PASS | bin 计数 {'short': 1, 'medium': 1, 'long': 1} vs 配额 {'short': 1, 'medium': 1, 'long': 1} |
| `n3_bin_assignment_consistent` | PASS | 每条被选 event 的 bin 归属与 B2.1 edges 数字化结果一致 |
| `n3_train_total` | PASS | train = 选中 event ∪ 全部 censored, 共 27 条 |
| `n3_removed_not_in_train` | PASS | 被移除的 18 条 event 均不在该档 train 内 |
| `n3_removed_not_relabelled_censored` | PASS | 被移除的 event 未被改标进 censored 列表 (无伪造删失) |
| `n3_selected_event_has_finite_eol` | PASS | 所选 event 轨迹的 event_observed=True 且 EOL 有限 |
| `n3_censored_has_no_eol` | PASS | censored 轨迹无真实 EOL (指标保持 NaN, 不转 0) |
| `n5_event_count` | PASS | 选中 event 5 条, 去重后 5, 要求 5 |
| `n5_all_censored_retained` | PASS | censored 24/24, 与 B2.1 train censored 集合一致 |
| `n5_bin_coverage` | PASS | bin 计数 {'short': 2, 'medium': 1, 'long': 2} vs 配额 {'short': 2, 'medium': 1, 'long': 2} |
| `n5_bin_assignment_consistent` | PASS | 每条被选 event 的 bin 归属与 B2.1 edges 数字化结果一致 |
| `n5_train_total` | PASS | train = 选中 event ∪ 全部 censored, 共 29 条 |
| `n5_removed_not_in_train` | PASS | 被移除的 16 条 event 均不在该档 train 内 |
| `n5_removed_not_relabelled_censored` | PASS | 被移除的 event 未被改标进 censored 列表 (无伪造删失) |
| `n5_selected_event_has_finite_eol` | PASS | 所选 event 轨迹的 event_observed=True 且 EOL 有限 |
| `n5_censored_has_no_eol` | PASS | censored 轨迹无真实 EOL (指标保持 NaN, 不转 0) |
| `n10_event_count` | PASS | 选中 event 10 条, 去重后 10, 要求 10 |
| `n10_all_censored_retained` | PASS | censored 24/24, 与 B2.1 train censored 集合一致 |
| `n10_bin_coverage` | PASS | bin 计数 {'short': 3, 'medium': 3, 'long': 4} vs 配额 {'short': 3, 'medium': 3, 'long': 4} |
| `n10_bin_assignment_consistent` | PASS | 每条被选 event 的 bin 归属与 B2.1 edges 数字化结果一致 |
| `n10_train_total` | PASS | train = 选中 event ∪ 全部 censored, 共 34 条 |
| `n10_removed_not_in_train` | PASS | 被移除的 11 条 event 均不在该档 train 内 |
| `n10_removed_not_relabelled_censored` | PASS | 被移除的 event 未被改标进 censored 列表 (无伪造删失) |
| `n10_selected_event_has_finite_eol` | PASS | 所选 event 轨迹的 event_observed=True 且 EOL 有限 |
| `n10_censored_has_no_eol` | PASS | censored 轨迹无真实 EOL (指标保持 NaN, 不转 0) |
| `n21_event_count` | PASS | 选中 event 21 条, 去重后 21, 要求 21 |
| `n21_all_censored_retained` | PASS | censored 24/24, 与 B2.1 train censored 集合一致 |
| `n21_bin_coverage` | PASS | bin 计数 {'short': 7, 'medium': 7, 'long': 7} vs 配额 {'short': 7, 'medium': 7, 'long': 7} |
| `n21_bin_assignment_consistent` | PASS | 每条被选 event 的 bin 归属与 B2.1 edges 数字化结果一致 |
| `n21_train_total` | PASS | train = 选中 event ∪ 全部 censored, 共 45 条 |
| `n21_removed_not_in_train` | PASS | 被移除的 0 条 event 均不在该档 train 内 |
| `n21_removed_not_relabelled_censored` | PASS | 被移除的 event 未被改标进 censored 列表 (无伪造删失) |
| `n21_selected_event_has_finite_eol` | PASS | 所选 event 轨迹的 event_observed=True 且 EOL 有限 |
| `n21_censored_has_no_eol` | PASS | censored 轨迹无真实 EOL (指标保持 NaN, 不转 0) |
| `nested_levels` | PASS | 标签变多只新增轨迹, 不替换已有轨迹 |
| `n21_is_all_train_events` | PASS | n=21 覆盖 B2.1 train 全部 21 条 event |
| `val_untouched` | PASS | val 31 条 == 冻结值 31; B6 从不改动 |
| `test_untouched` | PASS | test 74 条 == 冻结值 74; B6 从不改动 |
