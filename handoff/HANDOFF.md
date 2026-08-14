# 飞轮线下装配与两阶段校验手册（wheel HANDOFF）

本仓库只承载飞轮**代码基线**（`release/` 层级 + 容器封装）。
数据、checkpoint、参照结果等线下工件不入 Git，按下表装配后再做完整性校验。

## 线下工件槽位映射（artifact-map）

见同目录 `artifact-map.yaml`（`schema_version: 1.1.0`，逐槽位映射）：

| 本地槽位（仓库根） | 交付包 payload（相对 XA-202608_最终交付） | role |
|---|---|---|
| `data/` | `04_数据/wheel` | dataset |
| `checkpoints/` | `03_代码/components/wheel/release/checkpoints` | checkpoint |
| `results/reference/` | `05_结果/reference/wheel` | reference_result |

装配方式（示例）：

```bash
# 1) 克隆本仓库后创建槽位目录
mkdir -p data checkpoints results/reference

# 2) 从线下交付包复制 payload（以实际交付根 $PKG 为准）
cp -r "$PKG/04_数据/wheel/." data/
cp -r "$PKG/03_代码/components/wheel/release/checkpoints/." checkpoints/
cp -r "$PKG/05_结果/reference/wheel/." results/reference/

# 3) 容器场景: data 以只读卷挂载到 /app/release/data (见 Dockerfile 注释)
```

`data/`、`checkpoints/`、`results/` 为 `.gitignore` 白名单式忽略的本地槽位：
工件本体不入库，仅 7 个槽位描述文件（各槽位 `README.md` + `data_manifest.json` /
`checkpoint_manifest.json` / `public_summary.json` / `expected_metrics.json`）入库
（与相控阵组件同法）；公开的槽位语义以本文件与 `artifact-map.yaml` 为准。

## 两阶段完整性校验

```bash
# 阶段一（无 --artifact-root）：只校验 Git 代码基线
python release/scripts/handoff/verify_manifest.py

# 阶段二（有 --artifact-root）：先校验代码基线，
#        再校验 $ARTIFACT_ROOT/HANDOFF_MANIFEST.json 及其登记文件哈希
python release/scripts/handoff/verify_manifest.py --artifact-root "$ARTIFACT_ROOT"
```

- 阶段一真值 = `release/CODE_MANIFEST.sha256`（仅列受跟踪的 release/ 代码、配置、测试、文档）。
- 阶段二真值 = `$ARTIFACT_ROOT/HANDOFF_MANIFEST.json`（契约
  `schemas/handoff-manifest.schema.json` 的 `files[]`：逐文件 `path/role/size/sha256`）。
- 原交付 `RELEASE_MANIFEST.sha256`（470 条，含 data/checkpoints 等线下工件）**不进入 Git**，
  仅作线下装配 `HANDOFF_MANIFEST.json` 的参考输入，不是 Git 代码校验真值。

> **旧清单失配警示**：入库的 `release/RELEASE_SHA256SUMS.txt` / `release/RELEASE_MANIFEST.json`
> 是原交付的**历史快照**，覆盖线下工件，且对个别代码文件已过时
> （如授权重写的 `release/scripts/handoff/verify_manifest.py` 不在其列、个别测试文件
> 哈希与交付源字节不符），与当前仓库比对出现缺失/失配属预期现象。
> Git 代码校验真值是 `release/CODE_MANIFEST.sha256`，以它为准。

## 冻结科学结论（不可覆盖）

```
FINAL_TRANSFER_CONCLUSION = NO_POSITIVE_TRANSFER_SUPPORTED
ENGINEERING_RECOMMENDATION = damage_extrapolation
```

详见 `release/FINAL_RESULTS.md`。线下工件的 size/sha256 在阶段二逐条核对，
冻结结论对应的 checkpoint/结果哈希校验未随拆分降低。
