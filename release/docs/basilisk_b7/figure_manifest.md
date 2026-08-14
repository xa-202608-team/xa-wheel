# B7 figure manifest（§6–§11）

全部图由 `scripts/basilisk_b7/make_figures.py` 生成，数据只读自冻结产物，未训练、未重新评估、未重新 bootstrap。

## 风格约定（§11）

- dpi = 320（≥ 300），PNG + SVG 成对输出
- 坐标轴 / 图例 / 图注全英文，论文风格，无花哨主题、无 3D、无渐变背景
- y 轴不截断（Figure 3 主图与辅图都从 0 起）
- 置信区间一律标明 95%，且来自 B6 冻结 bootstrap
- exploratory（B3X / B4X）结果不进入任何最终图

## 产物与哈希

| file | sha256 |
|---|---|
| `docs/figures/basilisk_final/fig1_basilisk_health_trajectory.png` | `d8b374e87196b996…` |
| `docs/figures/basilisk_final/fig1_basilisk_health_trajectory.svg` | `4fb3b0b5dc9dc641…` |
| `docs/figures/basilisk_final/fig2_transfer_gain_vs_labels.png` | `a99e54bc65ff451c…` |
| `docs/figures/basilisk_final/fig2_transfer_gain_vs_labels.svg` | `1109fe75ffd789b8…` |
| `docs/figures/basilisk_final/fig3_primary_method_comparison.png` | `8724c2a3d9e60022…` |
| `docs/figures/basilisk_final/fig3_primary_method_comparison.svg` | `4f3a7dc6cf4f6088…` |
| `docs/figures/basilisk_final/fig4_health_management_flow.png` | `d40cf6ff3be88206…` |
| `docs/figures/basilisk_final/fig4_health_management_flow.svg` | `f171b5c900e64a1d…` |
| `docs/figures/basilisk_final/fig5_split_coverage_diagnostic.png` | `6b9f3312bb5b5135…` |
| `docs/figures/basilisk_final/fig5_split_coverage_diagnostic.svg` | `2cbad03ef10b1c92…` |

## Figure 1 —— 轨迹与 seed 的确定性选择规则（§6）

- seed 规则：median seed by PRIMARY-level target_only info macro RMSE (not the best seed) → seed = **124**
- 轨迹规则：primary test set, event-observed only, EOL closest to the median event EOL of the test set, ties broken by smallest traj_id → **traj_042**
- test 中 event-observed 候选数 = 35，其 EOL 中位数 = 29928 samples，所选轨迹 EOL = 29928 samples （1.707 年）
- `manual_selection = false`
- 绘出的方法：['damage_extrapolation', 'source_mmd_finetune', 'target_only']
- 因无冻结逐点预测而略去的方法：['source_finetune']（**不为出图而重训**）

## Figure 2（§7）

- x = [3, 5, 10, 21]（event-observed 失效标签轨迹数）
- y = paired RMSE gain = target_only − transfer，图注写明 “Positive value favors transfer.”
- 误差棒来源：frozen B6 bootstrap，`bootstrap_rerun = false`
- `trend_line_fitted = false`
- 画 y=0 参考线，并用垂直线 + 注释突出 n=5 PRIMARY
- 图注明确写出 “no primary positive-transfer verdict was obtained.”

## Figure 3（§8）

- 方法：['damage_extrapolation', 'target_only', 'source_mmd_finetune', 'source_finetune']；分箱：['short', 'medium', 'long']
- `y_axis_truncated = false` ——主图保留完整比例，诚实显示 physics baseline 与学习方法的差距
- 辅图（learning methods only）同样从 0 起：true

## Figure 4（§9）

- 目标域主链：Basilisk mission modes → wheel speed / command torque → 30-min aggregation → telemetry → self-calibrated features → observable HI → censor-aware RUL → warning metrics → maintenance decision
- 源域为**独立逻辑框**：XJTU-SY → encoder pretraining → fine-tuning / MMD adaptation
- `claims_transfer_improves_accuracy = false`；输出处标注 “The formal study did not establish positive transfer.”
- 标注部署不需要 Basilisk runtime，使用已提交的 mission-profile library

## Figure 5（§10，可选但已实现）

- B2 各 split 最小 event EOL（年）：{'train': 1.2740132329454712, 'val': 1.2791467031713437, 'test': 1.2353410905772302}
- B2.1 各 split 最小 event EOL（年）：{'train': 1.2353410905772302, 'val': 1.2461784166096281, 'test': 1.262035135751768}
- `rmse_compared_across_splits = false` —— B2 与 B2.1 的 test split 不同，图中不做 RMSE 公平比较
