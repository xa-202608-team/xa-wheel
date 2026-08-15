# checkpoints/ — 本地权重工件槽位

本目录是飞轮组件的**本地工件槽位**，存放线下装配的 B 系列冻结权重与协议哈希，
**不进入 Git**（`.gitignore` 白名单式忽略 `/checkpoints/**`，仅 README 与清单描述文件入库；
权重另受 `*.pt/*.pth/*.ckpt` 规则双重忽略；与相控阵组件同法）。

## 约定

- 权重清单登记在 `checkpoint_manifest.json`（`schema_version=1.1.0`，
  `component=wheel`；当前 `status=LOCAL_ARTIFACT_REQUIRED`、`entries=[]`，
  待线下装配后逐项补登 sha256/size，不得预填编造值）。
- 与交付 payload 的映射见仓库根 `handoff/artifact-map.yaml`
  （`03_代码/components/wheel/release/checkpoints`）。
- 装配与两阶段校验步骤见仓库根 `handoff/HANDOFF.md`；
  冻结证据核验另见 `release/scripts/handoff/audit_frozen_evidence.py`。
