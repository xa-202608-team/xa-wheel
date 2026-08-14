# DOCKER_HANDOFF_CHECKLIST — 逐项签收

签收人：____________  日期：____________

每项请**实际执行**后勾选，不要凭文档勾选。

---

## A. 环境

- [ ] A1 Python 3.11 环境就绪
- [ ] A2 `pip install -r requirements.txt` 无报错
- [ ] A3 `python -c "import torch; print(torch.__version__)"` 正常
- [ ] A4 确认**无需**安装 Basilisk（见 README §4）

## B. 包完整性

- [ ] B1 `python scripts/handoff/verify_release.py` → `RESULTS: 25/25` + `RELEASE_VERIFY_PASS`
- [ ] B2 第 11 项「B5/B6 dependency closure」显示 `依赖闭包完整 (28 项)`
- [ ] B3 `RELEASE_SHA256SUMS.txt` 与 `RELEASE_MANIFEST.json` 存在
- [ ] B4 `RELEASE_VERSION.txt` 已读

## C. 数据

- [ ] C1 `data/MANIFEST_DOCKER_HANDOFF.md` 已读，理解 A/B/C/D 四类
- [ ] C2 `data/features/wheel/basilisk_b19/target_features.h5` 存在（**训练实际输入**，~837 MB）
- [ ] C3 `data/features/wheel/schema_v1/source_features.h5` 存在（源域派生特征）
- [ ] C4 `data/sim/wheel_basilisk_b18/final/wheel_all.h5` 存在（~807 MB，上游追溯用）
- [ ] C5 `data/mission_profile/basilisk_v1/profiles.h5` 存在（冻结任务剖面）
- [ ] C6 已确认**不会**在容器内重新生成或覆写 `data/**`

## D. 运行时验证

- [ ] D1 `python scripts/run_all.py --verify` 全 5 步通过
- [ ] D2 `python scripts/run_all.py --fast` 输出 `✅ FAST PASSED`
- [ ] D3 `--fast` 日志中出现 `已加载 source encoder` ×2（证明源域权重真的被加载）
- [ ] D4 `--fast` 日志中三组训练均出结果（target_only / source_finetune / source_mmd_finetune）
- [ ] D5 `--fast` 末尾出现 `✅ 冻结证据 checkpoints/basilisk_b5 未被改动 (5 seed 完整)`
- [ ] D6 `checkpoints/_smoke_b5/formal_metrics.json` 已生成（冒烟产物，可随时删）
- [ ] D7 确认 `checkpoints/basilisk_b5/` 与 `basilisk_b6/` 修改时间**未变**

## E. 结果与文档

- [ ] E1 `FINAL_RESULTS.md` 已读，理解 `NO_POSITIVE_TRANSFER_SUPPORTED`
- [ ] E2 理解「迁移判定为负」与「推荐 damage_extrapolation」是**两个独立结论**
- [ ] E3 `docs/figures/basilisk_final/` 有 5 PNG + 5 SVG
- [ ] E4 `docs/basilisk_b5/REPRODUCE.md`、`docs/basilisk_b6/REPRODUCE.md` 已读
- [ ] E5 `docs/results.md` 存在
- [ ] E6 `docs/技术方案报告/` 存在

## F. Docker 封装

- [ ] F1 基础镜像选定（建议 `python:3.11-slim` + torch CPU 轮子）
- [ ] F2 `WORKDIR` 设为包根，`ENV PYTHONPATH=/app`
- [ ] F3 镜像内跑通 `--verify`
- [ ] F4 镜像内跑通 `--fast`
- [ ] F5 大数据文件（~1.65 GB）挂载策略确定：建议 volume 挂载而非烤进镜像层
- [ ] F6 `data/` 挂载为**只读**（`:ro`）
- [ ] F7 `--full` **未**在封装验收流程中执行

## G. 安全

- [ ] G1 `docs/handoff/local_path_audit.md` 已读（0 secrets）
- [ ] G2 确认无 `.env` / `*.key` / `*.pem` / credentials 类文件
- [ ] G3 `configs/**` 内无绝对路径泄漏（`verify_release.py` 第 9 项自动校验）

---

## 附注：为什么有「依赖闭包」这一项检查

首版打包的排除规则用**子串匹配**判断历史阶段脚本：模式 `basilisk_b1` 连坐命中了
`basilisk_b11` / `b18` / `b19`，`basilisk_b2` 连坐命中了 `b21`。结果 B5/B6 运行时
真正需要的模块、划分文件、特征文件被**静默剔除**，而当时的 `--fast` 是一个只打印
"logic accessible" 的桩，于是照样"通过"。

两个教训已固化进本包：

1. 排除规则改为**按路径段精确比对**，并显式维护 `REQUIRED_PHASE_SCRIPTS` 依赖闭包白名单
   （含 `scripts/basilisk_b1/calibrate_degradation.py` —— 它是被 `importlib` 按**文件路径**
   动态加载的，不出现在任何 import 语句里）。
2. `--fast` 改为**真跑正式 B5 脚本**。桩式冒烟等于没有冒烟。

签收时请特别确认 **B2** 与 **D2–D5**——它们正是当初漏掉的那类问题。
