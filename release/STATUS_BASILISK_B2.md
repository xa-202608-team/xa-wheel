# STATUS — BASILISK-B2

## 判定

```
B2_SPLIT_FROZEN               (§2, 150 条 = 44 / 31 / 75)
B2_BASELINE_CONTRACT_OK       (§1, 743/743)
B2_GENERALIZATION_FAIL        (§8, 4/7)
```

**结论：`B2_GENERALIZATION_FAIL`** —— Target-only 模型未能稳定优于最强平凡
基线，未取得 `B2_GENERALIZATION_PASS`。

判据在 `docs/basilisk_b2/protocol.md` §8 中于任何 test 数字产生之前写定，
运行后未增删判据、未更换主指标、未重新调参。

## 本阶段回答的唯一问题

> 在 B1.9 冻结的 Basilisk 数据（D-based EOL + `hi_damage_obs`）上，
> Target-only 模型能否**稳定**优于最强平凡可部署基线 `const_mean_info`？

不能。旧 analytic `S25_PASS` 未被继承（数据、失效定义、健康指标三者全换），
B2 从零重建的 Target-only 基线未通过。

## 判据结果

| # | 判据 | 需要 | 实得 | |
|---|---|---|---|---|
| 1 | 优于 const 的 seed 数 | ≥ 4/5 | 2/5 | 未达标 |
| 2 | mean paired gain > 0 | > 0 | −0.317628 | 未达标 |
| 3 | bootstrap 95% CI 下界 > 0 | > 0 | −0.574034 | 未达标 |
| 4 | macro corr > 0 的 seed 数 | ≥ 4/5 | 5/5 | PASS |
| 5 | PSR ≥ 0.30 的 seed 数 | ≥ 4/5 | 5/5 | PASS |
| 6 | warning / pre-EOL 指标有效 | 5/5 | 5/5 | PASS |
| 7 | checkpoint 只按 val 选 | 5/5 | 5/5 | PASS |

mean paired gain = −0.317628，95% CI [−0.574034, −0.062896]
（n = 5 个 seed 级配对差，2000 次重采样；重采样单位是 seed，非时间点）。

## 关键数字

| 项 | 值 |
|---|---|
| 主指标 | test info trajectory-macro RMSE |
| target_only（5 seed） | 1.0142 / 0.2563 / 0.7887 / 0.2646 / 0.7073 |
| const_mean_info | 0.2886 |
| damage_extrapolation | 0.0544（与失效判据同源，不作公平上界） |
| test 中位轨迹 RMSE | 0.2404–0.2548（全 5 seed 均优于 const） |
| best val info_macro | 0.2361–0.2435（全 5 seed 落在 0.007 宽带内） |
| split sha256 | `079296e5fba206da6ba9dab1078a8eaa6957e37ea97d0782b7a3fa01164113ae` |

## 失败模式

双峰，不是普遍变差。seed 73/75 正常（gain +0.032 / +0.024，PSR 0.74 / 0.82），
seed 72/74/76 崩塌（gain −0.73 / −0.50 / −0.42，PSR 9.81 / 6.90 / 5.88）。
三个崩塌 seed 炸的是**同一组轨迹（tid 19 / 109 / 111）**。

结构原因：这三条是 test 中 EOL 最短的一批（21848 / 22281 / 27708），
而 train 最短 EOL 22336、val 最短 22426 —— test 含 4 条比 train 与 val 中
任何一条都短命的轨迹，模型在该区段无训练信号，**val 也无从发现失控**。
5 个 seed 的 best val 几乎相同而 test 相差 4 倍，即判据 7 成立
（确实没有偷看 test）但 val 作为选择信号无效，两者是不同的事。

误差是全局尺度失控而非末期发散：seed 72 在 τ ∈ [0,0.5) / [0.5,0.9) / [0.9,1.0)
三段 RMSE 均 ≈3，pred 最大 27.90（真值 ≤ 1）。

判据 5 的 PSR 门槛是**下界**（排除"预测几乎不动"），崩塌 seed 的 PSR 远高于
下界却方向相反（预测标准差为真值 6–10 倍），故**判据 5 本次未起筛选作用**；
按 §8 不得事后增删判据，仅作记录。

## §9 删失分层（未伪造任何 EOL 指标）

39 条右删失 test 轨迹的 `info_macro_rmse` 恒为 `NaN` + `n_evaluable = 0`，
附 `rul_metrics_reason = "right_censored_no_true_rul_never_fabricated"`；
其 τ 轴带 `tau_semantics = "observation_progress_not_life_fraction"`。
**无 NaN 被转成 0。**

删失轨迹上唯一合法可算量 —— 下界违反率：`target_only` 12–17%，
`const_mean_info` **82.5%**，`damage_extrapolation` 100%。
即两个基线在删失轨迹上系统性提前报死，而学习模型尊重下界。
该量不参与 §8 判据，但说明 const 的 0.2886 并非全面更优。

warning 指标虽结构有效，漏报率为 **81–94%**（`rul_threshold = 0.2`,
`persistence = 3`），显式记录，不因只统计成功检测而隐藏。

## 后续阶段

**不进入 B3、不进入 B4。** 两者前置条件均为 `B2_GENERALIZATION_PASS`，
该条件未取得。在 Target-only 基线尚未稳定成立时做迁移增益对比无法归因 ——
基线本身有 3/5 概率崩塌，任何"迁移有效"的结论都不可信。

按 §7，本次结果不触发超参搜索；按 §10，B2 到此停止，不进入迁移。

## 产物

| 文件 | 内容 |
|---|---|
| `docs/basilisk_b2/protocol.md` | §0–§10 协议（训练前冻结） |
| `docs/basilisk_b2/results.md` | 5 seed 结果、判据逐条、失败模式诊断、§9 分层 |
| `docs/basilisk_b2/baseline_contract.json` | 743 项基线契约（319 文件 + 424 数值） |
| `checkpoints/basilisk_b2/split.json` | 冻结划分 + split sha256 |
| `checkpoints/basilisk_b2/metrics.json` | 逐 seed 全量指标 + 判定块 |
| `checkpoints/basilisk_b2/protocol_hash.json` | protocol / split / feature 哈希 |
| `scripts/basilisk_b2/` | `build_split` `data_b2` `eval_b2` `baselines_b2` `train_b2` `run_gate` `verify_baseline` |
| `tests/basilisk_b2/` | 6 个测试文件（划分 / 删失契约 / 输入契约 / 闸门纪律 / 基线契约 / evaluator 一致性） |

上游未被改动：B1.9 契约 `--verify` 743/743 不变。
Basilisk 未写入 `requirements.txt` / `Dockerfile` / `docker-compose.yml`。
