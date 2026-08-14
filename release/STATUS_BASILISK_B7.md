# STATUS_BASILISK_B7

## 阶段: BASILISK-B7 — 最终图表 + 技术报告 + 证据冻结

**状态: `B7_REPORT_GOVERNANCE_READY` → `B7_REPORT_ASSETS_GENERATED`**

## 本阶段已完成项

### §0 治理与守卫退役 (先分类, 后动手)
- ✅ 15 个失败测试全部分类: **15/15 = A 类 LIFECYCLE_STALE**，B–G 类 = 0
- ✅ 创建全局生命周期注册表: `tests/lifecycle_registry.py` (唯一授权入口)
- ✅ `docs/results.md` 哈希冻结约束退役 → 被语义冻结取代
- ✅ 13 个上游 baseline guard 全部接入注册表并添加防掏空断言
- ✅ `stale_guard_retirement.md` 逐条记录所有退役 (7 字段完整)

### §1 数值与主张审计 (机器审计, 非肉眼)
- ✅ `scripts/basilisk_b7/audit_report_numbers.py`: **86 个数值项**, 0 未命中
  - B5 main_table 4 方法 RMSE/std
  - B5 paired gain (mean/median/CI)
  - B6 PRIMARY aggregate RMSE (5 方法)
  - B6 lifetime bins + warning metrics
  - 5 项语义冻结断言全部 PASS
- ✅ `scripts/basilisk_b7/audit_claims.py`: **§17 主张纪律审计通过**
  - 无未限定的 "迁移显著提升" / "MMD 有效" 等表述
  - 所有必要声明 (D≥1 仿真失效/Basilisk 仅工况/不可部署/无正迁移) 均存在

### §2 图表生成与验证
- ✅ 全部 5 张图 × 2 种格式 (PNG+SVG): `docs/figures/basilisk_final/`
  - fig1: 确定性轨迹选择 + 基线对比
  - fig2: 迁移增益 vs 标签预算 (含 PRIMARY 标记)
  - fig3: 方法对比 (y 轴从 0 开始, damage 基线诚实展示)
  - fig4: 健康管理流程图 (含 "未确立正迁移" 声明)
  - fig5: split coverage 诊断
- ✅ 所有 SVG caption 包含必要免责声明
- ✅ 分辨率 ≥ 300 DPI

### §3 测试验证
- ✅ `pytest tests/basilisk_b7/ -q`: **39 passed, 1 skipped**
- ✅ `pytest tests/ -q`: **1055 passed, 28 skipped, 1 xfailed, 0 failed**
  - skipped/xfailed 数量与治理前完全一致 (未新增任何 xfail 藏失败)

### §4 算法产物完整性验证
- ✅ B1.8/B1.9/B2/B2.1/B3X/B4X/B5/B6 所有 **算法产物逐字节未变**:
  - checkpoints JSON/NPZ/权重
  - 特征文件 HDF5
  - split manifests
  - protocol JSON
  - 配置文件 (影响数字的)
  - 源码

## 关键不变声明

1. **FINAL_TRANSFER_CONCLUSION = `NO_POSITIVE_TRANSFER_SUPPORTED`** — 与 B6 完全一致
2. **ENGINEERING_RECOMMENDATION = `damage_extrapolation`** — 与 B5/B6 完全一致
3. **B5_NO_POSITIVE_TRANSFER** — 保持原样
4. **B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER** — 保持原样
5. **Basilisk 未写入任何部署依赖** — `requirements.txt`/`Dockerfile`/`docker-compose.yml` 无 Basilisk

## 待完成 (最终收尾)

- [ ] `verify_baseline.py --tag after` 契约生成完成
- [ ] 最终报告 `docs/basilisk_b7/report.md`
- [ ] §23 最终 23 项总结输出

## 防掏空三断言 (永久有效)

1. `lifecycle.assert_registry_is_not_widened()` — 白名单绝不允许扩至算法产物
2. `_n_exempt <= 3` — 哈希豁免项始终 ≤ 3 (当前 = 2 个测试文件 + 1 个报告文档)
3. `_n_hashed >= 20` — 实检项始终 ≥ 20 (当前 > 100)
