# BASILISK-B6 正式结果: 失效标签稀缺矩阵

protocol_sha256 = `433ccaf2515669c1f5afbc76029de068329afc7f67f0c10ae3372d14409b8412`  
subset_manifest_sha256 = `f7fe1ababe7b23db5d0ba41a7c644505267f684400a22aebb5144bf6ce041b1d`  
split_sha256 = `23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932` (B2.1, 未重划)  
输入 schema = `core_only`, n_features = 12  
正式 seed = [122, 123, 124, 125, 126]  
主指标 = `test_info_trajectory_macro_rmse`


> B6 §14: 在 B2.1 冻结划分上, 沿 **失效标签稀缺** 轴 (n_event_labeled ∈ {3,5,10,21}, 每档始终保留全部 24 条 censored) 给出四档正式主结果表, 并描述 paired gain 随标签数量的变化。稀缺轴是 event-observed failure label 数量, **不是 target 轨迹总数** —— 真实航天约束是失效标签稀缺, 退化 / 未失效运行数据仍可大量存在。

> 四档标签子集是**嵌套**的 (n=3 ⊂ n=5 ⊂ n=10 ⊂ n=21): 每个寿命 bin 只洗牌一次, 各档取前 k 条。这样趋势不会被"换了一组不同轨迹"混淆。

> 移除 event 轨迹会同时改变该档的 rul_scale 与特征 z-score (二者都由 train 行统计得出) —— 这是标签预算的直接后果, 不是公平性破坏: 同一 cell 内三种方法共享同一份 rul_scale / mu / sd。

> §18: damage_extrapolation 留在主表内, 不降级为脚注。不得因为赛题主题叫"迁移学习"就隐藏物理外推基线领先这一事实。

> const_mean_info 是标尺不是候选方法: 它只回答"这个模型是否学到了任何东西"。它被列入 excluded_oracle_methods, 不参与工程推荐排名。

## n_event_labeled = 3 (SECONDARY)

train = 27 (event 3 + censored 24) ｜ val = 31 ｜ test = 74 (val/test 未动)  
event 分 bin = short 1, medium 1, long 1 ｜ role = `SECONDARY`  
rul_scale (逐 seed) = [18408.6, 18408.6, 18408.6, 18408.6, 18408.6]  
catastrophic 阈值 (逐 seed, 该档 target_only validation 中位轨迹 RMSE × 2.0) = [1.024152, 1.051919, 1.022264, 0.939556, 1.096084]

| 方法 | info macro RMSE | ±std | info pooled | full macro | MAE | macro corr | PSR | catastrophic | warn cov | miss | false alarm | PH | convergence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `target_only` | 0.335007 | 0.021825 | 0.360554 | NaN | 0.287590 | 0.8632 | 1.095 | 0.0000 | 0.4229 | 0.5771 | 0.0205 | -0.02 | 0.2728 |
| `source_ft` | 0.357837 | 0.042342 | 0.385398 | NaN | 0.307826 | 0.7820 | 1.133 | 0.0000 | 0.4343 | 0.5657 | 0.0205 | -0.02 | 0.2916 |
| `source_mmd` | 0.317366 | 0.037528 | 0.341744 | NaN | 0.272042 | 0.8100 | 1.026 | 0.0000 | 0.4229 | 0.5771 | 0.0256 | -0.02 | 0.2738 |
| `const_mean` | 0.288629 | 0.000000 | 0.288630 | NaN | 0.249961 | NaN | 0.000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.00 | 0.4123 |
| `damage_extrap` | 0.054312 | 0.000000 | 0.056606 | NaN | 0.038461 | 0.9858 | 1.035 | 0.0000 | 1.0000 | 0.0000 | 0.0256 | 0.00 | NaN |

alpha-lambda accuracy (右删失轨迹不参与, 保持 NaN):

| 方法 | λ=0.3 | λ=0.5 | λ=0.7 |
|---|---|---|---|
| `target_only` | 0.8686 | 0.5943 | 0.1486 |
| `source_ft` | 0.8743 | 0.5543 | 0.1429 |
| `source_mmd` | 0.8457 | 0.5429 | 0.2114 |
| `const_mean` | 0.0000 | 0.0857 | 0.4571 |
| `damage_extrap` | 0.5714 | 0.7429 | 0.9714 |

分 bin info macro RMSE (bin 取自 B2.1, 未按 B6 test 重算):

| 方法 | short | medium | long |
|---|---|---|---|
| `target_only` | 0.312593 | 0.328278 | 0.363590 |
| `source_ft` | 0.334723 | 0.350184 | 0.387966 |
| `source_mmd` | 0.309788 | 0.320340 | 0.322220 |
| `const_mean` | 0.288671 | 0.288611 | 0.288605 |
| `damage_extrap` | 0.048726 | 0.051068 | 0.062872 |

配对增益 (gain = target_only − source 方法, 正数 = source 更好):

| 方法 | mean | median | std | CI95 下界 | CI95 上界 | improve |
|---|---|---|---|---|---|---|
| `source_ft` | -0.022830 | -0.021927 | 0.041726 | -0.056178 | +0.010519 | 2/5 |
| `source_mmd` | +0.017641 | +0.012986 | 0.047209 | -0.017116 | +0.054113 | 3/5 |

与物理外推基线 / 常数标尺的比较 (§18, 主表内, 非脚注):

| 比较 | 值 | 含义 |
|---|---|---|
| target − damage | +0.280695 | target_only 不如物理外推 |
| ft − damage | +0.303525 | source_ft 不如物理外推 |
| mmd − damage | +0.263054 | source_mmd 不如物理外推 |
| ft − const | +0.069208 | 未打赢常数标尺 |
| mmd − const | +0.028737 | 未打赢常数标尺 |

公平性 (同 cell 内 train/val/test IDs、batch 顺序、初始权重同一): **PASS** ｜ checkpoint 选择 = validation_only

## n_event_labeled = 5 (**PRIMARY**)

train = 29 (event 5 + censored 24) ｜ val = 31 ｜ test = 74 (val/test 未动)  
event 分 bin = short 2, medium 1, long 2 ｜ role = `PRIMARY`  
rul_scale (逐 seed) = [18408.6, 18408.6, 18408.6, 18408.6, 18408.6]  
catastrophic 阈值 (逐 seed, 该档 target_only validation 中位轨迹 RMSE × 2.0) = [0.84327, 0.550821, 0.97917, 0.661101, 0.956117]

| 方法 | info macro RMSE | ±std | info pooled | full macro | MAE | macro corr | PSR | catastrophic | warn cov | miss | false alarm | PH | convergence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `target_only` | 0.339625 | 0.021203 | 0.359653 | NaN | 0.292245 | 0.8576 | 1.149 | 0.0000 | 0.4286 | 0.5714 | 0.0154 | -0.04 | 0.2698 |
| `source_ft` | 0.354721 | 0.017159 | 0.375022 | NaN | 0.306295 | 0.8641 | 1.213 | 0.0057 | 0.4914 | 0.5086 | 0.0205 | -0.07 | 0.2684 |
| `source_mmd` | 0.310147 | 0.032811 | 0.331104 | NaN | 0.265965 | 0.8161 | 1.022 | 0.0057 | 0.3943 | 0.6057 | 0.0154 | -0.02 | 0.2760 |
| `const_mean` | 0.288629 | 0.000000 | 0.288630 | NaN | 0.249961 | NaN | 0.000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.00 | 0.4124 |
| `damage_extrap` | 0.054312 | 0.000000 | 0.056606 | NaN | 0.038461 | 0.9858 | 1.035 | 0.0000 | 1.0000 | 0.0000 | 0.0256 | 0.00 | NaN |

alpha-lambda accuracy (右删失轨迹不参与, 保持 NaN):

| 方法 | λ=0.3 | λ=0.5 | λ=0.7 |
|---|---|---|---|
| `target_only` | 0.8286 | 0.4514 | 0.1600 |
| `source_ft` | 0.8857 | 0.4514 | 0.1200 |
| `source_mmd` | 0.7429 | 0.4286 | 0.2171 |
| `const_mean` | 0.0000 | 0.0857 | 0.4571 |
| `damage_extrap` | 0.5714 | 0.7429 | 0.9714 |

分 bin info macro RMSE (bin 取自 B2.1, 未按 B6 test 重算):

| 方法 | short | medium | long |
|---|---|---|---|
| `target_only` | 0.324676 | 0.334724 | 0.359067 |
| `source_ft` | 0.341569 | 0.350776 | 0.371489 |
| `source_mmd` | 0.315279 | 0.304321 | 0.310355 |
| `const_mean` | 0.288670 | 0.288611 | 0.288605 |
| `damage_extrap` | 0.048726 | 0.051068 | 0.062872 |

配对增益 (gain = target_only − source 方法, 正数 = source 更好):

| 方法 | mean | median | std | CI95 下界 | CI95 上界 | improve |
|---|---|---|---|---|---|---|
| `source_ft` | -0.015096 | -0.010302 | 0.018139 | -0.029398 | -0.001197 | 2/5 |
| `source_mmd` | +0.029478 | +0.043665 | 0.036520 | +0.001531 | +0.057866 | 3/5 |

与物理外推基线 / 常数标尺的比较 (§18, 主表内, 非脚注):

| 比较 | 值 | 含义 |
|---|---|---|
| target − damage | +0.285313 | target_only 不如物理外推 |
| ft − damage | +0.300409 | source_ft 不如物理外推 |
| mmd − damage | +0.255835 | source_mmd 不如物理外推 |
| ft − const | +0.066091 | 未打赢常数标尺 |
| mmd − const | +0.021517 | 未打赢常数标尺 |

公平性 (同 cell 内 train/val/test IDs、batch 顺序、初始权重同一): **PASS** ｜ checkpoint 选择 = validation_only

## n_event_labeled = 10 (SECONDARY)

train = 34 (event 10 + censored 24) ｜ val = 31 ｜ test = 74 (val/test 未动)  
event 分 bin = short 3, medium 3, long 4 ｜ role = `SECONDARY`  
rul_scale (逐 seed) = [18408.6, 18408.6, 18408.6, 18408.6, 18408.6]  
catastrophic 阈值 (逐 seed, 该档 target_only validation 中位轨迹 RMSE × 2.0) = [0.816935, 1.007049, 0.773235, 0.791471, 0.691516]

| 方法 | info macro RMSE | ±std | info pooled | full macro | MAE | macro corr | PSR | catastrophic | warn cov | miss | false alarm | PH | convergence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `target_only` | 0.298031 | 0.019782 | 0.320631 | NaN | 0.254649 | 0.8960 | 0.940 | 0.0000 | 0.3143 | 0.6857 | 0.0000 | -0.02 | 0.2748 |
| `source_ft` | 0.327030 | 0.011156 | 0.356345 | NaN | 0.283300 | 0.8982 | 1.032 | 0.0000 | 0.3829 | 0.6171 | 0.0000 | -0.04 | 0.2885 |
| `source_mmd` | 0.292611 | 0.028754 | 0.312975 | NaN | 0.249012 | 0.9073 | 0.893 | 0.0000 | 0.2914 | 0.7086 | 0.0000 | -0.01 | 0.2762 |
| `const_mean` | 0.288629 | 0.000000 | 0.288630 | NaN | 0.249961 | NaN | 0.000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.00 | 0.4123 |
| `damage_extrap` | 0.054312 | 0.000000 | 0.056606 | NaN | 0.038461 | 0.9858 | 1.035 | 0.0000 | 1.0000 | 0.0000 | 0.0256 | 0.00 | NaN |

alpha-lambda accuracy (右删失轨迹不参与, 保持 NaN):

| 方法 | λ=0.3 | λ=0.5 | λ=0.7 |
|---|---|---|---|
| `target_only` | 0.8571 | 0.4686 | 0.2171 |
| `source_ft` | 0.8971 | 0.4514 | 0.1143 |
| `source_mmd` | 0.8114 | 0.4800 | 0.2057 |
| `const_mean` | 0.0000 | 0.0857 | 0.4571 |
| `damage_extrap` | 0.5714 | 0.7429 | 0.9714 |

分 bin info macro RMSE (bin 取自 B2.1, 未按 B6 test 重算):

| 方法 | short | medium | long |
|---|---|---|---|
| `target_only` | 0.296493 | 0.286826 | 0.309841 |
| `source_ft` | 0.331326 | 0.310263 | 0.338104 |
| `source_mmd` | 0.298553 | 0.282991 | 0.295487 |
| `const_mean` | 0.288671 | 0.288611 | 0.288605 |
| `damage_extrap` | 0.048726 | 0.051068 | 0.062872 |

配对增益 (gain = target_only − source 方法, 正数 = source 更好):

| 方法 | mean | median | std | CI95 下界 | CI95 上界 | improve |
|---|---|---|---|---|---|---|
| `source_ft` | -0.028999 | -0.021720 | 0.020813 | -0.046837 | -0.015607 | 0/5 |
| `source_mmd` | +0.005420 | +0.010316 | 0.044824 | -0.033723 | +0.037254 | 4/5 |

与物理外推基线 / 常数标尺的比较 (§18, 主表内, 非脚注):

| 比较 | 值 | 含义 |
|---|---|---|
| target − damage | +0.243719 | target_only 不如物理外推 |
| ft − damage | +0.272718 | source_ft 不如物理外推 |
| mmd − damage | +0.238299 | source_mmd 不如物理外推 |
| ft − const | +0.038401 | 未打赢常数标尺 |
| mmd − const | +0.003982 | 未打赢常数标尺 |

公平性 (同 cell 内 train/val/test IDs、batch 顺序、初始权重同一): **PASS** ｜ checkpoint 选择 = validation_only

## n_event_labeled = 21 (SECONDARY)

train = 45 (event 21 + censored 24) ｜ val = 31 ｜ test = 74 (val/test 未动)  
event 分 bin = short 7, medium 7, long 7 ｜ role = `SECONDARY`  
rul_scale (逐 seed) = [18408.6, 18408.6, 18408.6, 18408.6, 18408.6]  
catastrophic 阈值 (逐 seed, 该档 target_only validation 中位轨迹 RMSE × 2.0) = [0.581779, 0.515114, 0.56901, 0.53927, 0.570764]

| 方法 | info macro RMSE | ±std | info pooled | full macro | MAE | macro corr | PSR | catastrophic | warn cov | miss | false alarm | PH | convergence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `target_only` | 0.238093 | 0.003216 | 0.248370 | NaN | 0.199794 | 0.9212 | 0.763 | 0.0000 | 0.3086 | 0.6914 | 0.0103 | -0.01 | 0.2280 |
| `source_ft` | 0.241492 | 0.004549 | 0.255416 | NaN | 0.203279 | 0.8938 | 0.844 | 0.0000 | 0.3543 | 0.6457 | 0.0256 | -0.02 | 0.2140 |
| `source_mmd` | 0.244525 | 0.010367 | 0.257172 | NaN | 0.207440 | 0.9120 | 0.832 | 0.0000 | 0.4057 | 0.5943 | 0.0410 | -0.03 | 0.2182 |
| `const_mean` | 0.288629 | 0.000000 | 0.288630 | NaN | 0.249961 | NaN | 0.000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.00 | 0.4123 |
| `damage_extrap` | 0.054312 | 0.000000 | 0.056606 | NaN | 0.038461 | 0.9858 | 1.035 | 0.0000 | 1.0000 | 0.0000 | 0.0256 | 0.00 | NaN |

alpha-lambda accuracy (右删失轨迹不参与, 保持 NaN):

| 方法 | λ=0.3 | λ=0.5 | λ=0.7 |
|---|---|---|---|
| `target_only` | 0.7657 | 0.4857 | 0.2800 |
| `source_ft` | 0.7886 | 0.5086 | 0.2971 |
| `source_mmd` | 0.7143 | 0.4286 | 0.2571 |
| `const_mean` | 0.0000 | 0.0857 | 0.4571 |
| `damage_extrap` | 0.5714 | 0.7429 | 0.9714 |

分 bin info macro RMSE (bin 取自 B2.1, 未按 B6 test 重算):

| 方法 | short | medium | long |
|---|---|---|---|
| `target_only` | 0.223050 | 0.246148 | 0.245753 |
| `source_ft` | 0.228106 | 0.255587 | 0.241957 |
| `source_mmd` | 0.229407 | 0.253524 | 0.251394 |
| `const_mean` | 0.288670 | 0.288611 | 0.288605 |
| `damage_extrap` | 0.048726 | 0.051068 | 0.062872 |

配对增益 (gain = target_only − source 方法, 正数 = source 更好):

| 方法 | mean | median | std | CI95 下界 | CI95 上界 | improve |
|---|---|---|---|---|---|---|
| `source_ft` | -0.003399 | -0.003983 | 0.003473 | -0.005659 | -0.000353 | 1/5 |
| `source_mmd` | -0.006432 | -0.001376 | 0.012755 | -0.017535 | +0.001088 | 1/5 |

与物理外推基线 / 常数标尺的比较 (§18, 主表内, 非脚注):

| 比较 | 值 | 含义 |
|---|---|---|
| target − damage | +0.183781 | target_only 不如物理外推 |
| ft − damage | +0.187180 | source_ft 不如物理外推 |
| mmd − damage | +0.190213 | source_mmd 不如物理外推 |
| ft − const | -0.047137 | 打赢常数标尺 |
| mmd − const | -0.044104 | 打赢常数标尺 |

公平性 (同 cell 内 train/val/test IDs、batch 顺序、初始权重同一): **PASS** ｜ checkpoint 选择 = validation_only

## §18 必须原样写出的结论

> **学习模型未超过基于已知累计损伤结构的物理外推基线。**

四个标签档下 target_only / source_ft / source_mmd 的 test information-zone trajectory-macro RMSE 均高于 `damage_extrapolation`。这是本阶段的事实陈述, 不因赛题主题是迁移学习而被隐藏或弱化。

