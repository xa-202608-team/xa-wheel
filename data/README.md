# data/ — 本地数据工件槽位

本目录是飞轮组件的**本地工件槽位**，存放线下装配的数据本体（约 1.65 GB），**不进入 Git**
（`.gitignore` 白名单式忽略 `/data/**`，仅 README 与清单描述文件入库；与相控阵组件同法）。

## 约定

- 目录内容、来源与哈希登记在 `data_manifest.json`（`schema_version=1.1.0`，
  `component=wheel`；当前 `status=LOCAL_ARTIFACT_REQUIRED`、`entries=[]`，
  待线下装配后逐项补登 sha256/size，不得预填编造值）。
- 与交付 payload 的映射见仓库根 `handoff/artifact-map.yaml`（`04_数据/wheel`）；
  容器内以 `/app/release/data:ro` 只读挂载。
- 装配与两阶段校验步骤见仓库根 `handoff/HANDOFF.md`；
  校验命令：`python release/scripts/handoff/verify_manifest.py --artifact-root "$ARTIFACT_ROOT"`。
