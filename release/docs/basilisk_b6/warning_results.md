# BASILISK-B6 §14 报警 / 预测视界结果

protocol_sha256 = `433ccaf2515669c1f5afbc76029de068329afc7f67f0c10ae3372d14409b8412`  
subset_manifest_sha256 = `f7fe1ababe7b23db5d0ba41a7c644505267f684400a22aebb5144bf6ce041b1d`  
split_sha256 = `23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932` (B2.1, 未重划)  
输入 schema = `core_only`, n_features = 12  
正式 seed = [122, 123, 124, 125, 126]  
主指标 = `test_info_trajectory_macro_rmse`


> coverage 与 miss rate 必须成对出现, 不得只报成功检测而隐藏 miss rate。warning valid 只说明口径自洽, **不等于 warning 可用**。

> NaN 一律保留 NaN, 并同时给出 n_evaluable / eligible_count。绝不把 NaN 转成 0 —— 不可评估与指标恰好为 0 是两件完全不同的事。右删失 test 轨迹没有真 EOL, PH / alpha-lambda 保持 NaN, 不伪造 EOL 指标。空 bin 返回 NaN + n = 0, 不伪造 0。

evaluator = frozen S4/Basilisk (warning_lead_time / prognostic_horizon / alpha_lambda_accuracy / convergence_metric)  
rul_threshold = 0.2 ｜ persistence = 3 ｜ alpha = 0.2 ｜ lambdas = [0.3, 0.5, 0.7] ｜ late_from = 0.5

## n_event_labeled = 3 (SECONDARY)

| 方法 | warn coverage | miss rate | coverage before EOL | miss before EOL | false alarm | PH (n_seed) | convergence | late conv | warning_valid |
|---|---|---|---|---|---|---|---|---|---|
| `target_only` | 0.4229 | 0.5771 | 0.3771 | 0.6229 | 0.0205 | -0.02 (5) | 0.2728 | 0.3539 | 5/5 |
| `source_ft` | 0.4343 | 0.5657 | 0.3829 | 0.6171 | 0.0205 | -0.02 (5) | 0.2916 | 0.3860 | 5/5 |
| `source_mmd` | 0.4229 | 0.5771 | 0.3600 | 0.6400 | 0.0256 | -0.02 (5) | 0.2738 | 0.3508 | 5/5 |
| `const_mean` | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.00 (5) | 0.4123 | 0.3826 | 5/5 |
| `damage_extrap` | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0256 | 0.00 (5) | NaN | NaN | 5/5 |

- §15 条件 8 (miss rate 不高于 target_only + 0.05): target 0.5771 ｜ FT 0.5657 (PASS) ｜ MMD 0.5771 (PASS)
- PH / alpha-lambda 的分母只含 event-observed 轨迹; 右删失轨迹保持 NaN, 不伪造 EOL。

## n_event_labeled = 5 (**PRIMARY**)

| 方法 | warn coverage | miss rate | coverage before EOL | miss before EOL | false alarm | PH (n_seed) | convergence | late conv | warning_valid |
|---|---|---|---|---|---|---|---|---|---|
| `target_only` | 0.4286 | 0.5714 | 0.3886 | 0.6114 | 0.0154 | -0.04 (5) | 0.2698 | 0.3392 | 5/5 |
| `source_ft` | 0.4914 | 0.5086 | 0.4800 | 0.5200 | 0.0205 | -0.07 (5) | 0.2684 | 0.3462 | 5/5 |
| `source_mmd` | 0.3943 | 0.6057 | 0.3486 | 0.6514 | 0.0154 | -0.02 (5) | 0.2760 | 0.3377 | 5/5 |
| `const_mean` | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.00 (5) | 0.4124 | 0.3829 | 5/5 |
| `damage_extrap` | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0256 | 0.00 (5) | NaN | NaN | 5/5 |

- §15 条件 8 (miss rate 不高于 target_only + 0.05): target 0.5714 ｜ FT 0.5086 (PASS) ｜ MMD 0.6057 (PASS)
- PH / alpha-lambda 的分母只含 event-observed 轨迹; 右删失轨迹保持 NaN, 不伪造 EOL。

## n_event_labeled = 10 (SECONDARY)

| 方法 | warn coverage | miss rate | coverage before EOL | miss before EOL | false alarm | PH (n_seed) | convergence | late conv | warning_valid |
|---|---|---|---|---|---|---|---|---|---|
| `target_only` | 0.3143 | 0.6857 | 0.2857 | 0.7143 | 0.0000 | -0.02 (5) | 0.2748 | 0.3487 | 5/5 |
| `source_ft` | 0.3829 | 0.6171 | 0.3429 | 0.6571 | 0.0000 | -0.04 (5) | 0.2885 | 0.3745 | 5/5 |
| `source_mmd` | 0.2914 | 0.7086 | 0.2571 | 0.7429 | 0.0000 | -0.01 (5) | 0.2762 | 0.3480 | 5/5 |
| `const_mean` | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.00 (5) | 0.4123 | 0.3827 | 5/5 |
| `damage_extrap` | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0256 | 0.00 (5) | NaN | NaN | 5/5 |

- §15 条件 8 (miss rate 不高于 target_only + 0.05): target 0.6857 ｜ FT 0.6171 (PASS) ｜ MMD 0.7086 (PASS)
- PH / alpha-lambda 的分母只含 event-observed 轨迹; 右删失轨迹保持 NaN, 不伪造 EOL。

## n_event_labeled = 21 (SECONDARY)

| 方法 | warn coverage | miss rate | coverage before EOL | miss before EOL | false alarm | PH (n_seed) | convergence | late conv | warning_valid |
|---|---|---|---|---|---|---|---|---|---|
| `target_only` | 0.3086 | 0.6914 | 0.2743 | 0.7257 | 0.0103 | -0.01 (5) | 0.2280 | 0.2754 | 5/5 |
| `source_ft` | 0.3543 | 0.6457 | 0.3029 | 0.6971 | 0.0256 | -0.02 (5) | 0.2140 | 0.2607 | 5/5 |
| `source_mmd` | 0.4057 | 0.5943 | 0.3429 | 0.6571 | 0.0410 | -0.03 (5) | 0.2182 | 0.2567 | 5/5 |
| `const_mean` | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.00 (5) | 0.4123 | 0.3828 | 5/5 |
| `damage_extrap` | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0256 | 0.00 (5) | NaN | NaN | 5/5 |

- §15 条件 8 (miss rate 不高于 target_only + 0.05): target 0.6914 ｜ FT 0.6457 (PASS) ｜ MMD 0.5943 (PASS)
- PH / alpha-lambda 的分母只含 event-observed 轨迹; 右删失轨迹保持 NaN, 不伪造 EOL。

