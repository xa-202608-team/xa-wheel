# B5 —— formal transfer on the frozen B2.1 split (计划, 未执行)

> **状态: `PLANNED_NOT_RUN`**
>
> 本文件是 B2.1 §14 出口要求写下的**下一阶段计划**。
> **B5 未运行, 不得自动运行。** 需要显式的新指令才能开始。
>
> **`B2_GENERALIZATION_FAIL` 保持终局** · **`B3X_NO_STABILIZING_SIGNAL`** ·
> **`B4X_TRANSFER_STABILIZATION_SIGNAL` 仅 `EXPLORATORY_ONLY`** ·
> **当前不存在任何正式的 positive-transfer 结论。**

---

## 1. 为什么现在才能谈迁移

在 B2 划分上做正式迁移是没有意义的: test 含有比全部 train/val 更短命的
event 轨迹 (min EOL 21658 < train 22336), Target-only 本身 3/5 seed 崩溃到
0.7–1.0。在那种底座上, 迁移带来的任何数值变化都无法与"划分缺陷造成的方差"
区分 —— 这正是 B3X 得到 `NO_STABILIZING_SIGNAL`、B4X 只能停在
`EXPLORATORY_ONLY` 的原因。

B2.1 给出了一个 Target-only **五 seed 全部落在 [0.2379, 0.2488]**、
short-life 无 catastrophic 的稳定底座。只有在这种底座上,
"迁移是否有效"才是一个可被证伪的问题。

## 2. 冻结面 (B5 一律不得修改)

* **B2.1 划分**: `split_sha256 = 23e2b944...5c932`, 逐字复用
  `docs/basilisk_b21/split_manifest.json`, **不重算三分位、不换 seed、不重划**。
* **B2 冻结训练配置**: `max_epochs=8`, `early_stop_patience=2`,
  `weight_decay=1e-3`, `early_stop_metric=info_macro_rmse`,
  `finetune_lr=1e-4`, `batch_size=128`, `post_eol_weight=0.1`,
  `capped_weight=0.1`, `target_hi_key=hi_damage_obs`, `censor_hinge_eta=1.0`。
* 模型结构 / optimizer / loss / HI / RUL 定义 / 源域 checkpoint /
  MMD lambda / mission features —— 全部沿用, 不调。
* B2 / B3X / B4X / B2.1 的既有产物只读。

## 3. 对比设置

同一 B2.1 划分、同一五个 seed `[102,103,104,105,106]`, 配对比较:

| 方法 | 角色 |
|---|---|
| `target_only` | **B5 的主基线** (B2.1 已测定, 数值可直接复用并重跑核对) |
| `transfer_mmd` | 待检验对象 (源域预训练 + adapter + MMD/CORAL 三阶段) |
| `const_mean_info` | 平凡下界 (沿用) |
| `damage_extrapolation` | 物理外推参照 (沿用) |

* 主指标同 B2.1: **test info trajectory-macro RMSE**。
* 迁移必须发生在 **HI / 退化动力学层**, 不在原始波形层。
* `persistence` 不进入可部署方法排名; `nPHM` 不作为模型选择指标;
  oracle 类方法留在 `EXCLUDED_ORACLE_METHODS`。

## 4. 判定条件 (须在跑之前预登记, 不得事后调整)

`B5_TRANSFER_PASS` 拟定要求 —— **基线是 `target_only`, 不是 `const_mean_info`**:

1. `transfer_mmd` 优于 `target_only` 于 ≥ 4/5 seed;
2. mean paired gain > 0;
3. seed 级配对 bootstrap 95% CI lower > 0 (n = 5, 单位 = 一个 seed 一个配对差值);
4. short-life 子集不退化 (short-life catastrophic 计数 ≤ target_only, 且 ≥ 4/5 seed 为 0);
5. warning 指标保持有效 (coverage/miss 互补、miss 显式报告);
6. checkpoint 只按 validation 选;
7. 删失下界违反率不高于 target_only。

任一不满足 → `B5_TRANSFER_FAIL`, 并且**不得**改写为"趋势向好""部分有效"。
catastrophic 阈值仍只由 **validation** 派生, 禁止 test 派生。

## 5. 必须同时回答的问题 (否则 PASS 也没有说服力)

B2.1 §4 记录了一个未解事实: `damage_extrapolation` (0.0543) 全面优于
`target_only` (0.2441) 约 4.5 倍。因此 B5 即使拿到
`B5_TRANSFER_PASS`, 也只证明"迁移优于 Target-only", **不等于**
"学习路线优于物理路线"。B5 报告须同时给出 `transfer_mmd` vs
`damage_extrapolation` 的对照, 并如实说明差距方向。

## 6. 纪律 (沿用)

禁止全局 `np.random.seed()`; 空 bin 返回 NaN + n=0 不伪造 0;
无 NaN 悄悄转 0; 右删失轨迹不得伪造 EOL; 阈值走 config 不硬编码;
不得用时间点数量当独立样本; 不得只统计成功检测而隐藏 miss rate;
LLM 不参与数值寿命预测; 不写入 `requirements.txt` / `Dockerfile` /
`docker-compose.yml`; S5B 已关闭不得重开。

## 7. 起跑前置

* 新建隔离命名空间 `b5`, 冻结 B2.1 及以前全部哈希 (契约起点 = 974 项 + B2.1 自身产物);
* 本阶段判定条件先落盘并哈希, 再产生任何数字;
* 得到显式指令后才执行 —— **当前状态 `PLANNED_NOT_RUN`**。
