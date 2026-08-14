"""tests/basilisk_b5/test_b5_baseline_contract.py —— §2 基线契约测试。

契约必须在 B5 产出任何数字之前冻结, 并在阶段结束后原值复核。
任何被冻结项发生变化 → B5_INVALID。

契约文件的实际形状 (承自 B1.8 起的同一族): 顶层 `groups` 是
{组名: {相对路径: sha256 或 "MISSING"}}, 各 `*_results` / `*_expected` 是
数值/字符串块。项数 = 文件项 + 数值项, 由脚本打印, 不写进 JSON。
"""
from __future__ import annotations

import hashlib
import json

from conftest import ROOT


def _flat_files(c: dict) -> dict:
    out = {}
    for g, d in c["groups"].items():
        for rel, sha in d.items():
            out[rel] = (g, sha)
    return out


def _flat_numeric(c: dict) -> dict:
    out = {}
    for k, v in c.items():
        if k.endswith("_results") and isinstance(v, dict):
            out.update(v)
    return out


# ---------------------------------------------------------------------------
# 生命周期迁移守卫 (B7 §14 改造)
# ---------------------------------------------------------------------------
# 本文件原有两个守卫断言"下游阶段 (B6) 产物必须永远不存在", 以及"B5 契约里所有
# MISSING 项必须一直 MISSING"。这个前提在 B6 获得显式人工阶段授权并完成之后
# 必然失效 —— 它们会永久失败, 且失败原因与 B5 自身的正确性无关。
#
# B7 (FINAL_FIGURES_REPORT_EVIDENCE_FREEZE) 阶段按 §2 / §14 的授权把它们改造为
# 生命周期迁移守卫: 下游产物**允许存在**, 但只有在其 protocol / baseline
# contract / final verdict 齐备时才允许。这比原断言更强 —— 原断言只能拦住
# "阶段推进", 新断言拦住的是"没有协议就先跑数字"这一真正的越界。
#
# 退役理由 / 授权来源 / 新守卫语义详见 docs/basilisk_b7/stale_guard_retirement.md。

# 允许在 B5 之后合法出现的下游阶段。每项 = (阶段名, 触发产物, 必需治理产物)。
DOWNSTREAM_LIFECYCLE: dict[str, dict[str, tuple[str, ...]]] = {
    "BASILISK_B6": {
        "artifacts": (
            "docs/basilisk_b6/results.md",
            "STATUS_BASILISK_B6.md",
            "checkpoints/basilisk_b6/all_metrics.json",
        ),
        "governance": (
            "docs/basilisk_b6/protocol.md",
            "docs/basilisk_b6/baseline_contract.json",
            "checkpoints/basilisk_b6/protocol_hash.json",
            "checkpoints/basilisk_b6/label_subset_manifest.json",
            "checkpoints/basilisk_b6/final_verdict.json",
        ),
    },
    "BASILISK_B7": {
        "artifacts": (
            "checkpoints/basilisk_b7/final_tables.json",
            "STATUS_BASILISK_B7.md",
        ),
        "governance": (
            "docs/basilisk_b7/baseline_contract.json",
            "checkpoints/basilisk_b7/frozen_result_index.json",
        ),
    },
}

# B7 是纯结果表达阶段 —— 它不训练, 因此这些产物**仍然**必须缺席。
# 这部分不是"陈旧守卫", 而是本阶段及以后都必须成立的真实约束, 保留不动。
STILL_FORBIDDEN: tuple[str, ...] = (
    # B7 绝不训练 -> 不得出现任何 B7 权重或新评估结果
    "checkpoints/basilisk_b7/target_only_s122.pt",
    "checkpoints/basilisk_b7/source_finetune_s122.pt",
    "checkpoints/basilisk_b7/source_mmd_finetune_s122.pt",
    "checkpoints/basilisk_b7/all_raw.npz",
    "checkpoints/basilisk_b7/all_metrics.json",
    "checkpoints/basilisk_b7/formal_metrics.json",
    "checkpoints/basilisk_b7/paired_statistics.json",
    # B8 只允许打包 + Docker, 不允许算法产物
    "checkpoints/basilisk_b8/all_metrics.json",
    "checkpoints/basilisk_b8/formal_metrics.json",
    "checkpoints/basilisk_b8/paired_statistics.json",
    "checkpoints/basilisk_b8/final_verdict.json",
    # B9 的算法实验尚未获得任何授权
    "checkpoints/basilisk_b9/all_metrics.json",
    "checkpoints/basilisk_b9/final_verdict.json",
    "docs/basilisk_b9/results.md",
    "STATUS_BASILISK_B9.md",
    # B5 自身当年就禁止的旁路矩阵, 与阶段推进无关, 永久有效
    "checkpoints/basilisk_b5/low_data_matrix.json",
    "checkpoints/basilisk_b5/truncation_matrix.json",
)

# §2 授权 B7 重写的报告表达面。它们不是数值来源 (B7 全部数字读
# checkpoints/basilisk_b7/frozen_result_index.json), 数值正确性由
# scripts/basilisk_b7/audit_report_numbers.py 对种保证, 不靠哈希冻结。
REPORT_SURFACE_EXEMPT: tuple[str, ...] = (
    "docs/results.md",
)

# B7 §2 / §14 显式授权本阶段为生命周期治理改造这两个测试文件, 因此它们的哈希
# 在 B7 内**预期会变**。若继续按旧哈希比对, 会把"授权的治理改动"误报成"越界
# 篡改" —— 那等于用契约禁止指令明确授权的动作。
#
# 豁免不等于放任: 这两个文件的正确性由
#   - docs/basilisk_b7/stale_guard_retirement.md (书面依据, 且被
#     test_b5_stale_guard_retirement_is_documented 强制存在)
#   - test_b5_downstream_stage_transition_is_governed (授权 + 治理完整性)
#   - test_b5_undelegated_artifacts_still_absent   (未授权项仍须缺席)
#   - tests/basilisk_b7/test_b7_stale_guard_retirement.py
# 共同保证, 比单纯哈希冻结更强。
#
# 严格限于这 2 个测试治理文件: 不豁免任何结果 JSON / 训练权重 / 算法源码 / config。
LIFECYCLE_GUARD_RETIREMENT_EXEMPT: tuple[str, ...] = (
    "tests/basilisk_b21/test_b21_discipline.py",
    "tests/basilisk_b5/test_b5_baseline_contract.py",
)

# 允许跳过哈希比对但**必须仍然存在**的全部路径。
HASH_EXEMPT: tuple[str, ...] = (
    REPORT_SURFACE_EXEMPT + LIFECYCLE_GUARD_RETIREMENT_EXEMPT
)


def _authorized_to_exist(rel: str) -> bool:
    """该路径是否属于"已获显式阶段授权、允许在 B5 之后出现"的下游产物。"""
    for spec in DOWNSTREAM_LIFECYCLE.values():
        if rel in spec["artifacts"]:
            return True
    return False


def test_b5_contract_identity(b5_contract):
    assert b5_contract["stage"] == "BASILISK_B5"
    assert b5_contract["label"] == "FORMAL_TRANSFER_EVALUATION"
    assert b5_contract["invalid_label"] == "B5_INVALID"
    assert int(b5_contract["n_files"]) >= 400, "契约文件项不应少于继承的 B2.1"
    assert len(_flat_numeric(b5_contract)) >= 300, "契约数值项覆盖不足"


def test_b5_contract_no_placeholder_no_unreadable(b5_contract):
    """契约里不允许残留占位符, 也不允许有读不出来的冻结文件。"""
    blob = json.dumps(b5_contract, ensure_ascii=False)
    assert "PLACEHOLDER_FILLED_ON_FIRST_WRITE" not in blob
    assert "UNREADABLE" not in blob


def test_b5_contract_missing_only_in_absent_groups(b5_contract):
    """MISSING 只允许出现在 *must_stay_absent / *deliberately_absent 组里。"""
    bad = [(g, rel) for rel, (g, sha) in _flat_files(b5_contract).items()
           if sha == "MISSING" and "absent" not in g]
    assert not bad, f"非缺席组里出现 MISSING: {bad[:5]}"


def test_b5_contract_records_all_required_items(b5_contract):
    """§2 逐项: 数据/特征/划分/ID/协议/指标/源权重/结构/lambda/训练配置/评估器。"""
    num = _flat_numeric(b5_contract)
    need = ("b21_split_sha256", "b21_train_ids_sha256", "b21_val_ids_sha256",
            "b21_test_ids_sha256", "b21_protocol_hash_sha256",
            "b21_metrics_hash_sha256", "source_checkpoint_sha256",
            "encoder_architecture_sha256", "evaluator_metrics_sha256",
            "mmd_lambda")
    for k in need:
        assert k in num, f"契约缺少 §2 必录项 {k}"
    # B1.8 数据集哈希与 B1.9 特征哈希经由继承的 b18/b19 数值块记录
    assert any("dataset" in k or "scenario" in k for k in num), "缺 B1.8 数据集记录"
    assert any("feature" in k for k in num), "缺 B1.9 特征记录"


def test_b5_contract_train_config_frozen(b5_contract):
    """训练配置必须逐项记录 —— 事后改超参会被这条抓住。"""
    num = _flat_numeric(b5_contract)
    for k in ("train_config_max_epochs", "train_config_early_stop_patience",
              "train_config_weight_decay", "train_config_finetune_lr",
              "train_config_early_stop_metric"):
        assert k in num, f"训练配置缺 {k}"
    assert num["train_config_max_epochs"] == "8"
    assert num["train_config_early_stop_patience"] == "2"
    assert num["train_config_early_stop_metric"] == "info_macro_rmse"
    assert num["mmd_lambda"] == "1.0"


def test_b5_contract_reconfirms_b21_split_hash(b5_contract, b21_split_ro):
    num = _flat_numeric(b5_contract)
    assert num["b21_split_sha256"] == str(b21_split_ro["split_sha256"])
    assert num["b21_split_verdict"] == "B21_SPLIT_VALID"
    assert num["b21_verdict"] == "B21_GENERALIZATION_PASS"


def test_b5_contract_records_b4x_as_exploratory(b5_contract):
    """B4X 的信号必须以"探索性"身份入契约, 防止被当成正式结论继承。"""
    num = _flat_numeric(b5_contract)
    assert num["b4x_exploratory_only"] == "True"
    assert num["b4x_formal_stage"] == "False"
    assert num["b4x_verdict_via_b21"] == "B4X_TRANSFER_STABILIZATION_SIGNAL"


def test_b5_old_artifacts_unchanged(b5_contract):
    """test_b5_old_artifacts_unchanged (§21) —— 冻结文件按契约逐一重算哈希。

    B7 §14 改造: 原版把契约里所有 `MISSING` 项一律断言"必须一直缺席", 因而在
    B6 / B7 获得显式阶段授权并推进后永久失败。现在区分三类:

      1. 有哈希的冻结项 -> 仍然逐字节比对 (**这部分是本测试的核心, 未削弱**);
      2. `MISSING` 且属于已授权下游阶段的产物 -> 允许存在, 但转由
         test_b5_downstream_stage_transition_is_governed 检查其治理完整性;
      3. `MISSING` 且不属于任何授权阶段 -> 仍然必须缺席 (真正的越界防护)。

    §2 授权 B7 重写的报告表达面 (docs/results.md) 单独豁免哈希比对, 其数值
    正确性由 scripts/basilisk_b7/audit_report_numbers.py 与冻结索引对种保证。
    """
    bad = []
    n_hash_checked = 0
    n_hash_exempt = 0
    n_lifecycle_allowed = 0
    for rel, (g, exp) in _flat_files(b5_contract).items():
        p = ROOT / rel
        if rel in REPORT_SURFACE_EXEMPT:
            # 不比哈希, 但必须仍然存在 —— 报告表达面被删掉也是问题。
            n_hash_exempt += 1
            if not p.exists():
                bad.append((rel, "报告表达面消失"))
            continue
        if rel in LIFECYCLE_GUARD_RETIREMENT_EXEMPT:
            # §2 / §14 授权的生命周期治理改动 —— 哈希预期会变, 但文件不得消失。
            n_hash_exempt += 1
            if not p.exists():
                bad.append((rel, "生命周期守卫文件消失 (不得静默删除)"))
            continue
        if exp == "MISSING":
            if not p.exists():
                continue
            if _authorized_to_exist(rel):
                n_lifecycle_allowed += 1
                continue
            bad.append((rel, "本应缺席却存在 (无阶段授权)"))
            continue
        if not p.exists():
            bad.append((rel, "已消失"))
            continue
        n_hash_checked += 1
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != exp:
            bad.append((rel, f"{exp[:8]}->{got[:8]}"))
    assert not bad, f"冻结产物被改动 (B5_INVALID): {bad[:8]}"
    # 防止豁免逻辑把整个测试掏空: 绝大多数项必须仍然走真实哈希比对。
    # 350 的来源 (实测, 非猜测): B5 契约 415 个文件项 = 355 个带哈希 + 60 个
    # MISSING; 355 减去 2 个落在契约内的豁免项 (docs/results.md 与
    # tests/basilisk_b21/test_b21_discipline.py) = 353 项仍做逐字节比对。
    # 阈值取 350 留极小余量; 若未来豁免名单被扩大, 这里会先失败。
    assert n_hash_checked >= 350, \
        f"哈希实检项过少 ({n_hash_checked}), 豁免逻辑可能被滥用"
    # 豁免只允许极少数几项 —— 数量本身也要设上限。
    assert n_hash_exempt <= len(HASH_EXEMPT), \
        f"豁免项数 {n_hash_exempt} 超出白名单长度 {len(HASH_EXEMPT)}"


def test_b5_downstream_stage_transition_is_governed():
    """B7 §14: 下游阶段产物只有在治理产物齐备时才允许存在。

    取代旧的 test_b5_b6_artifacts_must_stay_absent。旧断言("B6 必须永远不存在")
    在 B6 获得显式人工阶段授权后必然过期; 新断言拦住的是真正的越界 ——
    没有 protocol / baseline contract / final verdict 就先产出结果数字。
    """
    for stage, spec in DOWNSTREAM_LIFECYCLE.items():
        present = [a for a in spec["artifacts"] if (ROOT / a).exists()]
        if not present:
            continue
        for gov in spec["governance"]:
            assert (ROOT / gov).exists(), (
                f"{stage} 已产出 {present[0]} 但缺治理产物 {gov} —— "
                f"未经协议冻结即产生结果数字, 属越界")


def test_b5_undelegated_artifacts_still_absent():
    """B7 §14: 未获授权的产物仍然必须缺席 —— 新守卫不得放松这一条。

    含: B7 自己的训练权重与新评估结果 (B7 是结果表达阶段, 禁止训练)、
    B8 的算法产物 (B8 只允许打包 + Docker)、B9 的算法实验 (尚未授权)、
    以及 B5 当年就禁止的低数据/截断旁路矩阵。
    """
    present = [rel for rel in STILL_FORBIDDEN if (ROOT / rel).exists()]
    assert not present, f"以下产物未获阶段授权, 本应缺席: {present}"


def test_b5_stale_guard_retirement_is_documented():
    """退役必须留下书面依据, 不允许静默删除守卫。"""
    doc = ROOT / "docs" / "basilisk_b7" / "stale_guard_retirement.md"
    assert doc.exists(), "缺 docs/basilisk_b7/stale_guard_retirement.md"
    txt = doc.read_text(encoding="utf-8")
    for key in ("test_b5_old_artifacts_unchanged",
                "test_b5_b6_artifacts_must_stay_absent",
                "test_b21_b5_absent_and_never_auto_run"):
        assert key in txt, f"退役文档未记录 {key}"


def test_b5_prior_stages_untouched(b5_contract):
    """B5 不得修改 B1.8 / B1.9 / B2 / B2.1 / B3X / B4X 的任何产物 (§1)。

    B7 §14 例外: tests/basilisk_b21/test_b21_discipline.py 按 §2 显式授权做
    生命周期守卫退役改造, 其哈希预期变化 (见 HASH_EXEMPT 注释)。该文件是**测试
    治理文件, 不是数据/结果/算法产物**, 改它动不了任何数字。
    """
    tags = ("basilisk_b18", "basilisk_b19", "basilisk_b2/", "basilisk_b21",
            "basilisk_b3x", "basilisk_b4x")
    protected = {rel: sha for rel, (g, sha) in _flat_files(b5_contract).items()
                 if any(t in rel for t in tags) and sha != "MISSING"
                 and rel not in HASH_EXEMPT}
    assert len(protected) >= 20, "受保护前序产物数量异常, 契约覆盖不足"
    for rel, exp in protected.items():
        p = ROOT / rel
        assert p.exists(), f"前序产物缺失: {rel}"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == exp, \
            f"前序产物被 B5 改动: {rel}"
    # 豁免项必须仍然存在 —— 允许改造, 不允许删除。
    for rel in HASH_EXEMPT:
        if any(t in rel for t in tags):
            assert (ROOT / rel).exists(), f"豁免项不得消失: {rel}"


def test_b5_source_checkpoint_and_encoder_frozen(b5_contract):
    """源域权重与编码器结构必须与契约一致 —— 迁移起点不能变。"""
    num = _flat_numeric(b5_contract)
    for rel, key in (("checkpoints/source_tcn_pretrain.pt",
                      "source_checkpoint_sha256"),
                     ("src/models/tcn_encoder.py",
                      "encoder_architecture_sha256"),
                     ("src/experiments/metrics.py",
                      "evaluator_metrics_sha256")):
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert got == num[key], f"{rel} 与契约记录不符"


def test_b5_not_written_into_requirements_or_docker():
    """Basilisk 不得写入 requirements.txt / Dockerfile / docker-compose.yml。"""
    for rel in ("requirements.txt", "Dockerfile", "docker-compose.yml"):
        p = ROOT / rel
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore").lower()
        assert "basilisk" not in txt, f"{rel} 被写入 basilisk 内容"


def test_b5_contract_chain_requires_b21_pass(b5_contract):
    assert b5_contract["frozen_chain"]["BASILISK_B21"] == \
        "B21_GENERALIZATION_PASS"
    assert b5_contract["frozen_chain"]["BASILISK_B2"] == "B2_GENERALIZATION_FAIL"
    assert b5_contract["frozen_chain"]["BASILISK_B3X"] == \
        "B3X_NO_STABILIZING_SIGNAL"


def test_b5_b6_artifacts_must_stay_absent(b5_contract):
    """§20 原版 —— **已由 B7 §14 退役为生命周期迁移守卫, 保留为可追溯记录。**

    旧语义: "B5 不得自动跑 B6" -> 断言 b5_must_stay_absent 组内所有路径永远缺席。
    为何 obsolete: B6 已获显式人工阶段授权并完成 (`B6_NO_PRIMARY_LOW_LABEL_
    POSITIVE_TRANSFER`), "永远缺席"的前提不再成立, 该断言只会永久失败。

    本测试**不删除**, 而是改为检查旧守卫真正想保护的不变量 —— 契约当年确实
    登记了这些禁止项 (即 B5 阶段本身没有偷跑 B6), 这一历史事实永久可查。
    运行期的越界防护移交:
      - test_b5_downstream_stage_transition_is_governed  (授权 + 治理完整性)
      - test_b5_undelegated_artifacts_still_absent       (未授权项仍须缺席)
    退役依据: docs/basilisk_b7/stale_guard_retirement.md
    """
    grp = b5_contract["groups"]["b5_must_stay_absent"]
    # 历史事实: B5 冻结契约时, B6 产物确实被登记为 MISSING (B5 没有偷跑下游)。
    assert any("basilisk_b6" in k for k in grp)
    assert "checkpoints/basilisk_b5/low_data_matrix.json" in grp
    assert "checkpoints/basilisk_b5/truncation_matrix.json" in grp
    for rel, v in grp.items():
        assert v == "MISSING", f"{rel} 在 B5 契约中本应登记为 MISSING"
    # 当前时刻的存在性判断已移交上述两个新守卫, 此处不再断言。
