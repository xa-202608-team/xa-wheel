# DATA_CLEANUP_CANDIDATES — 仅记录，不删除

**`DATA_TREE_PROTECTED = true`** —— 本文件列出的任何路径在 B8-PREP 阶段
**均未被删除、移动、改写**。是否清理由人类决定。

---

## 判定依据

以下路径是**旧 analytic 阶段**的遗留产物。它们在 basilisk 生成脚本里被显式列为
禁止输出目标，说明项目已明确与其切割：

```python
# scripts/basilisk/generate_wheel_dataset.py
FORBIDDEN_OUT = ("data/simulated/wheel/sim_v1", "data/features/wheel/schema_v1")

# scripts/basilisk/build_features.py
forbidden = "data/features/wheel/schema_v1"
```

---

## 候选清单

| # | 路径 | 体积 | 判定 | 仍被引用？ |
|---|---|---|---|---|
| 1 | `data/simulated/wheel/sim_v1/seed_42/wheel_all.h5` | 287 MB | 旧 analytic 仿真产物 | 被 `configs/wheel.yaml` 等旧链路默认值引用 |
| 2 | `data/features/wheel/schema_v1/target_features.h5` | 327 MB | 旧 analytic 目标域特征 | 同上；**B5/B6 不读它** |
| 3 | `data/features/wheel/basilisk_b1/` | 空目录 | B1 早期特征 | 无 |
| 4 | `data/features/wheel/basilisk_b14/` | 空目录 | B14 中间特征 | 无 |
| 5 | `data/features/wheel/basilisk_b18/` | 807 MB | B18 中间特征（被 B19 取代） | `configs/wheel_basilisk_b19.yaml` 的 `b18_feature_h5` |
| 6 | `data/features/wheel/basilisk_v1/` | 807 MB | 早期 basilisk 特征 | 无 |
| 7 | `data/sim/wheel_basilisk_b11/candidates/` | — | B1.1 候选场景（已 freeze 出 A） | 审计脚本 |
| 8 | `data/sim/wheel_basilisk_b1/`、`b12/`、`b13/`、`b14/`、`v1/` | — | 各阶段中间仿真 | 各阶段 verify_baseline |

---

## 为什么不建议现在删

1. **#1 / #2 仍被旧链路默认值引用**：`configs/wheel.yaml`、
   `src/transfer/train_transfer.py`、`src/experiments/run_groups.py`、
   `src/baselines/physical_extrap.py` 里均有 `schema_v1` 硬编码默认路径。
   删掉会让这些入口在无 `--config` 覆盖时直接崩。
2. **#5 是 B19 的声明上游**：`configs/wheel_basilisk_b19.yaml` 的
   `paths.b18_feature_h5` 指向它。删掉会破坏 B19 的可追溯链。
3. **#3–#8 是各阶段 `verify_baseline.py` 的核查对象**：这些脚本按哈希核对历史产物，
   删掉会让对应阶段的基线契约测试失败。原仓库的定位是**留证**。
4. 交付包**不含**这些文件 —— release 用白名单构建，体积已从 41.8 GB 压到 1.89 GB。
   清理原仓库对交付没有收益。

---

## 若确实要清理

建议顺序：先跑 `python -m pytest tests/ -q` 记下基线，再逐项删除并重跑，
确认失败项只增加"文件消失"类且可接受。**删除前请先备份 `data/`**——
本阶段全程的前提就是它尚未备份。

---

`DATA_TREE_PROTECTED = true`
