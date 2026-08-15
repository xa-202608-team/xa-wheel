# BASILISK-B6 §14 寿命分 bin 结果

protocol_sha256 = `433ccaf2515669c1f5afbc76029de068329afc7f67f0c10ae3372d14409b8412`  
subset_manifest_sha256 = `f7fe1ababe7b23db5d0ba41a7c644505267f684400a22aebb5144bf6ce041b1d`  
split_sha256 = `23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932` (B2.1, 未重划)  
输入 schema = `core_only`, n_features = 12  
正式 seed = [122, 123, 124, 125, 126]  
主指标 = `test_info_trajectory_macro_rmse`


> bin 边界取自 docs/basilisk_b21/split_manifest.json 的 lifetime_bins.event (quantile_basis = dataset_level_event_eol_before_training), 即在任何训练之前就已确定。**不得根据 B6 test 分布重算 bin**。test 侧 bin 成员在四个标签档之间完全相同 —— B6 只改 train, 从不改 test。

> NaN 一律保留 NaN, 并同时给出 n_evaluable / eligible_count。绝不把 NaN 转成 0 —— 不可评估与指标恰好为 0 是两件完全不同的事。右删失 test 轨迹没有真 EOL, PH / alpha-lambda 保持 NaN, 不伪造 EOL 指标。空 bin 返回 NaN + n = 0, 不伪造 0。

bin 边界 = [26846.0, 37395.0] ｜ test 侧计数 = {'short': 12, 'medium': 11, 'long': 12} ｜ 四档完全相同 (B6 只改 train)

右删失 bin 退化 = True (有效 bin 数 1) —— 如实报告, 不为凑三个 bin 编造边界。

## n_event_labeled = 3 (SECONDARY)

| 方法 | short RMSE (n_seed) | medium RMSE (n_seed) | long RMSE (n_seed) |
|---|---|---|---|
| `target_only` | 0.312593 (5) | 0.328278 (5) | 0.363590 (5) |
| `source_ft` | 0.334723 (5) | 0.350184 (5) | 0.387966 (5) |
| `source_mmd` | 0.309788 (5) | 0.320340 (5) | 0.322220 (5) |
| `const_mean` | 0.288671 (5) | 0.288611 (5) | 0.288605 (5) |
| `damage_extrap` | 0.048726 (5) | 0.051068 (5) | 0.062872 (5) |

逐 bin 配对增益 (正数 = source 方法更好):

| bin | n_traj | gain_ft | ≥0 | gain_mmd | ≥0 | damage RMSE |
|---|---|---|---|---|---|---|
| short | 12 | -0.022130 | 否 | +0.002806 | 是 | 0.048726 |
| medium | 11 | -0.021906 | 否 | +0.007939 | 是 | 0.051068 |
| long | 12 | -0.024376 | 否 | +0.041370 | 是 | 0.062872 |

- gain ≥ 0 的 bin 数: FT 0/3 ｜ MMD 3/3 (§15 条件 5 要求 ≥ 2)
- `localized_transfer_benefit` (仅短寿命 bin 获益): FT False ｜ MMD False
- catastrophic 计数 (逐 bin 逐 seed 之和): target_only 0/0/0 ｜ source_ft 0/0/0 ｜ source_mmd 0/0/0 ｜ const_mean 0/0/0 ｜ damage_extrap 0/0/0

## n_event_labeled = 5 (**PRIMARY**)

| 方法 | short RMSE (n_seed) | medium RMSE (n_seed) | long RMSE (n_seed) |
|---|---|---|---|
| `target_only` | 0.324676 (5) | 0.334724 (5) | 0.359067 (5) |
| `source_ft` | 0.341569 (5) | 0.350776 (5) | 0.371489 (5) |
| `source_mmd` | 0.315279 (5) | 0.304321 (5) | 0.310355 (5) |
| `const_mean` | 0.288670 (5) | 0.288611 (5) | 0.288605 (5) |
| `damage_extrap` | 0.048726 (5) | 0.051068 (5) | 0.062872 (5) |

逐 bin 配对增益 (正数 = source 方法更好):

| bin | n_traj | gain_ft | ≥0 | gain_mmd | ≥0 | damage RMSE |
|---|---|---|---|---|---|---|
| short | 12 | -0.016893 | 否 | +0.009397 | 是 | 0.048726 |
| medium | 11 | -0.016052 | 否 | +0.030403 | 是 | 0.051068 |
| long | 12 | -0.012422 | 否 | +0.048712 | 是 | 0.062872 |

- gain ≥ 0 的 bin 数: FT 0/3 ｜ MMD 3/3 (§15 条件 5 要求 ≥ 2)
- `localized_transfer_benefit` (仅短寿命 bin 获益): FT False ｜ MMD False
- catastrophic 计数 (逐 bin 逐 seed 之和): target_only 0/0/0 ｜ source_ft 0/0/1 ｜ source_mmd 0/1/0 ｜ const_mean 0/0/0 ｜ damage_extrap 0/0/0

## n_event_labeled = 10 (SECONDARY)

| 方法 | short RMSE (n_seed) | medium RMSE (n_seed) | long RMSE (n_seed) |
|---|---|---|---|
| `target_only` | 0.296493 (5) | 0.286826 (5) | 0.309841 (5) |
| `source_ft` | 0.331326 (5) | 0.310263 (5) | 0.338104 (5) |
| `source_mmd` | 0.298553 (5) | 0.282991 (5) | 0.295487 (5) |
| `const_mean` | 0.288671 (5) | 0.288611 (5) | 0.288605 (5) |
| `damage_extrap` | 0.048726 (5) | 0.051068 (5) | 0.062872 (5) |

逐 bin 配对增益 (正数 = source 方法更好):

| bin | n_traj | gain_ft | ≥0 | gain_mmd | ≥0 | damage RMSE |
|---|---|---|---|---|---|---|
| short | 12 | -0.034833 | 否 | -0.002061 | 否 | 0.048726 |
| medium | 11 | -0.023437 | 否 | +0.003835 | 是 | 0.051068 |
| long | 12 | -0.028263 | 否 | +0.014353 | 是 | 0.062872 |

- gain ≥ 0 的 bin 数: FT 0/3 ｜ MMD 2/3 (§15 条件 5 要求 ≥ 2)
- `localized_transfer_benefit` (仅短寿命 bin 获益): FT False ｜ MMD False
- catastrophic 计数 (逐 bin 逐 seed 之和): target_only 0/0/0 ｜ source_ft 0/0/0 ｜ source_mmd 0/0/0 ｜ const_mean 0/0/0 ｜ damage_extrap 0/0/0

## n_event_labeled = 21 (SECONDARY)

| 方法 | short RMSE (n_seed) | medium RMSE (n_seed) | long RMSE (n_seed) |
|---|---|---|---|
| `target_only` | 0.223050 (5) | 0.246148 (5) | 0.245753 (5) |
| `source_ft` | 0.228106 (5) | 0.255587 (5) | 0.241957 (5) |
| `source_mmd` | 0.229407 (5) | 0.253524 (5) | 0.251394 (5) |
| `const_mean` | 0.288670 (5) | 0.288611 (5) | 0.288605 (5) |
| `damage_extrap` | 0.048726 (5) | 0.051068 (5) | 0.062872 (5) |

逐 bin 配对增益 (正数 = source 方法更好):

| bin | n_traj | gain_ft | ≥0 | gain_mmd | ≥0 | damage RMSE |
|---|---|---|---|---|---|---|
| short | 12 | -0.005056 | 否 | -0.006358 | 否 | 0.048726 |
| medium | 11 | -0.009439 | 否 | -0.007376 | 否 | 0.051068 |
| long | 12 | +0.003796 | 是 | -0.005641 | 否 | 0.062872 |

- gain ≥ 0 的 bin 数: FT 1/3 ｜ MMD 0/3 (§15 条件 5 要求 ≥ 2)
- `localized_transfer_benefit` (仅短寿命 bin 获益): FT False ｜ MMD False
- catastrophic 计数 (逐 bin 逐 seed 之和): target_only 0/0/0 ｜ source_ft 0/0/0 ｜ source_mmd 0/0/0 ｜ const_mean 0/0/0 ｜ damage_extrap 0/0/0

