# results/ — 本地结果工件槽位

本目录是飞轮组件的**本地工件槽位**，存放线下装配的参照结果（`results/reference/`），
**不进入 Git**（`.gitignore` 白名单式忽略 `/results/**`，仅 README 与指标摘要描述文件入库；
与相控阵组件同法）。

## 约定

- `public_summary.json`：公开冻结摘要（`schema_version=1.1.0`，`component=wheel`）。
  当前 `status=NOT_YET_VERIFIED`、`metrics=[]`；冻结数字真值在
  `release/FINAL_RESULTS.md` 与 B6/B7 冻结契约（线下工件），逐项核实前不得填入。
- `expected_metrics.json`：复现验证的期望指标门槛（reproduce-smoke 的
  target_only/source_finetune/source_mmd_finetune RMSE 见根 `README.md`），
  同样在逐项核实前保持 `NOT_YET_VERIFIED` + 空 `metrics`。
- 参照结果映射：`05_结果/reference/wheel` → `results/reference/`（见
  `handoff/artifact-map.yaml`）；装配后用
  `python release/scripts/handoff/verify_manifest.py --artifact-root "$ARTIFACT_ROOT"` 校验。
