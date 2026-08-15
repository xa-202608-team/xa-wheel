"""tests/basilisk_b6/test_b6_baseline_contract.py —— §3 基线契约测试。

契约必须在 B6 产出任何数字之前冻结, 并在阶段结束后原值复核。
任何被冻结项发生变化 → B6_INVALID。

§3 要求至少逐项冻结: B1.8 dataset hash / B1.9 feature hash / B2.1 split hash /
B2.1 train·val·test IDs / **B5 protocol hash · config hash · results hash** /
source checkpoint hash / encoder architecture / MMD lambda / metrics
implementation / damage baseline implementation / target-only architecture。
"""
from __future__ import annotations

import hashlib
import json

from conftest import ROOT

# --------------------------------------------------------------------------
# 共享生命周期治理注册表 —— 全库唯一来源。
# B7 授权改写 docs/results.md 作为最终交付表达面, 因此该文件的"哈希永久不变"
# 约束退役, 改由语义审计 (scripts/basilisk_b7/audit_report_numbers.py) 看守。
# 算法产物 (数据/特征/split/权重/metrics/verdict) 仍然逐字节冻结, 不受影响。
# 退役理由逐条记录: docs/basilisk_b7/stale_guard_retirement.md
# --------------------------------------------------------------------------
import importlib.util as _ilu   # noqa: E402

_lr_spec = _ilu.spec_from_file_location(
    "lifecycle_registry", ROOT / "tests" / "lifecycle_registry.py")
lifecycle = _ilu.module_from_spec(_lr_spec)
_lr_spec.loader.exec_module(lifecycle)


# --------------------------------------------------------------------------
# B8 §2 生命周期治理: B7/B8 下游产物的"永久缺席"约束退役
# --------------------------------------------------------------------------
# 旧语义 (B6 §22 接力防线): b6_must_stay_absent 组内所有 B7/B8 路径**永远**缺席。
# 该断言在 B6 阶段成立 —— 它拦住的是"B6 无论正负结论都不得自动运行 B7/B8"。
#
# 到 B8-HANDOFF 阶段, B7 (最终图表+报告+证据冻结) 与 B8 (打包交付) 都已获得
# 显式人工阶段授权并正式执行。"永远缺席"的前提消失, 该断言只会永久失败 ——
# 它保护的不是任何数值正确性, 而是一个已被人工授权取代的生命周期假设。
#
# 替换语义 (更强, 不是放松):
#   旧: 下游产物永远不能出现              -> 只能拦住"阶段推进"
#   新: 下游产物可以出现, 但 (a) 必须有齐备的治理产物,
#       (b) 必须逐字节证明**没有污染任何上游冻结产物**,
#       (c) 未获授权的算法/数字产物仍然必须缺席
# 新守卫拦住的是真正的越界: 没有协议就先产数字 / 打包顺手改了实验结果。
#
# 退役理由逐条记录: docs/handoff/stale_guard_retirement_final.md
# 本次改动仅属 lifecycle governance, 不改动任何算法数字或数据。

# 允许在 B6 之后合法出现的下游阶段。每项 = (触发产物, 必需治理产物)。
DOWNSTREAM_LIFECYCLE: dict[str, dict[str, tuple[str, ...]]] = {
    "BASILISK_B7": {
        "artifacts": (
            "STATUS_BASILISK_B7.md",
            "docs/basilisk_b7/results.md",
        ),
        "governance": (
            "docs/basilisk_b7/baseline_contract.json",
            "docs/basilisk_b7/stale_guard_retirement.md",
            "checkpoints/basilisk_b7/frozen_result_index.json",
            "checkpoints/basilisk_b7/report_number_audit.json",
        ),
    },
    "BASILISK_B8": {
        "artifacts": (
            "STATUS_BASILISK_B8.md",
            "docs/basilisk_b8/packaging.md",
            "checkpoints/basilisk_b8/package_manifest.json",
        ),
        "governance": (
            "docs/handoff/B8_COMPLETION_REPORT.md",
            "docs/handoff/stale_guard_retirement_final.md",
        ),
    },
}

# B7 是纯表达阶段、B8 是纯打包阶段 —— 两者都不训练、不产生新实验数字。
# 因此这些产物**仍然**必须缺席。这不是陈旧守卫, 而是永久有效的真实约束。
STILL_FORBIDDEN: tuple[str, ...] = (
    # B7 只表达冻结数字, 禁止重算
    "checkpoints/basilisk_b7/formal_metrics.json",
    "checkpoints/basilisk_b7/metrics.json",
    "checkpoints/basilisk_b7/summary.json",
    "checkpoints/basilisk_b7/all_metrics.json",
    "checkpoints/basilisk_b7/paired_statistics.json",
    "checkpoints/basilisk_b7/final_verdict.json",
    # B8 只打包, 禁止任何算法产物
    "checkpoints/basilisk_b8/all_metrics.json",
    "checkpoints/basilisk_b8/formal_metrics.json",
    "checkpoints/basilisk_b8/paired_statistics.json",
    "checkpoints/basilisk_b8/final_verdict.json",
    # 星敏 (S7/S8) 线从未获得任何阶段授权
    "STATUS_BASILISK_S7.md",
    "STATUS_BASILISK_S8.md",
    "checkpoints/basilisk_s7/metrics.json",
    "checkpoints/basilisk_s8/package_manifest.json",
    "docs/basilisk_s7/results.md",
    "docs/basilisk_s8/packaging.md",
)

# 无论下游走到哪一阶段, 这五类产物必须逐字节不变 —— 这是"下游出现也不许
# 污染上游"的硬核心。哈希取自 B6/B7 冻结契约 (两份一致, 见 B8 报告 §9)。
FROZEN_UPSTREAM: dict[str, str] = {
    # B5 正式迁移评估结果 (NO_POSITIVE_TRANSFER 的数字依据)
    "checkpoints/basilisk_b5/formal_metrics.json":
        "6233e17139c1e606aae23f383ab20970cc48760a09ae38fa98cf2e7cfab8db82",
    # B6 失效标签稀缺矩阵结果
    "checkpoints/basilisk_b6/all_metrics.json":
        "3972059675cf911b60433dc327d36fd735ee5eb8fc4846e00ca06a6d81413b13",
    # B2.1 冻结划分 (个体级 train/val/test, 防泄漏的根据)
    "docs/basilisk_b21/split_manifest.json":
        "d8700e8a7bda7c45c1471c9039ad7e025507a8e0616553110c02226388917ebb",
    # 源域预训练权重 (迁移起点)
    "checkpoints/source_tcn_pretrain.pt":
        "d528c2d1b12b83e305bdc0809293e9b7ca8e84e07952dfb2c96bc46a5a8f7b66",
    # B1.9 目标域特征 (正式输入 schema CORE_ONLY)
    "data/features/wheel/basilisk_b19/target_features.h5":
        "575d708ae9330dfa37d407cf639ebbbd098c473ce1fc137ff1eef8feaab9c293",
}


def _authorized_to_exist(rel: str) -> bool:
    """该路径是否属于"已获显式阶段授权、允许在 B6 之后出现"的下游产物。"""
    for spec in DOWNSTREAM_LIFECYCLE.values():
        if rel in spec["artifacts"]:
            return True
    return False


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


def test_b6_contract_identity(b6_contract):
    assert b6_contract["stage"] == "BASILISK_B6"
    assert b6_contract["label"] == "FAILURE_LABEL_SCARCITY_FORMAL_MATRIX"
    assert b6_contract["invalid_label"] == "B6_INVALID"
    assert int(b6_contract["n_files"]) >= 400, "契约文件项不应少于继承的 B5"
    assert len(_flat_numeric(b6_contract)) >= 300, "契约数值项覆盖不足"


def test_b6_contract_no_placeholder_no_unreadable(b6_contract):
    blob = json.dumps(b6_contract, ensure_ascii=False)
    assert "PLACEHOLDER_FILLED_ON_FIRST_WRITE" not in blob
    assert "UNREADABLE" not in blob


def test_b6_contract_missing_only_in_absent_groups(b6_contract):
    bad = [(g, rel) for rel, (g, sha) in _flat_files(b6_contract).items()
           if sha == "MISSING" and "absent" not in g]
    assert not bad, f"非缺席组里出现 MISSING: {bad[:5]}"


def test_b6_contract_records_all_required_items(b6_contract):
    """§3 逐项: 数据/特征/划分/ID/B5 三哈希/源权重/结构/lambda/指标/damage 基线。"""
    num = _flat_numeric(b6_contract)
    need = ("b21_split_sha256", "b21_train_ids_sha256", "b21_val_ids_sha256",
            "b21_test_ids_sha256", "source_checkpoint_sha256",
            "encoder_architecture_sha256", "evaluator_metrics_sha256",
            "mmd_lambda")
    for k in need:
        assert k in num, f"契约缺少 §3 必录项 {k}"
    # §3: B5 的 protocol / config / results 三哈希必须逐项进契约
    for tag in ("protocol", "config", "results"):
        assert any(k.startswith("b5_") and tag in k for k in num), \
            f"契约缺 B5 {tag} hash"
    assert any("dataset" in k or "scenario" in k for k in num), "缺 B1.8 数据集记录"
    assert any("feature" in k for k in num), "缺 B1.9 特征记录"
    # damage 基线实现与 target-only 结构
    files = _flat_files(b6_contract)
    assert any("physical_extrap" in rel for rel in files), "缺 damage 基线实现"
    assert any("tcn_encoder" in rel or "heads" in rel for rel in files), \
        "缺 target-only 结构文件"


def test_b6_contract_train_config_frozen(b6_contract):
    """训练配置逐项记录 —— 事后改超参会被这条抓住 (§2 forbidden_changes)。"""
    num = _flat_numeric(b6_contract)
    for k in ("train_config_max_epochs", "train_config_early_stop_patience",
              "train_config_weight_decay", "train_config_finetune_lr",
              "train_config_early_stop_metric"):
        assert k in num, f"训练配置缺 {k}"
    assert num["train_config_max_epochs"] == "8"
    assert num["train_config_early_stop_patience"] == "2"
    assert num["train_config_early_stop_metric"] == "info_macro_rmse"
    assert num["mmd_lambda"] == "1.0", "§2 禁止改 MMD lambda"


def test_b6_contract_reconfirms_b21_split_hash(b6_contract, b21_split_ro):
    num = _flat_numeric(b6_contract)
    assert num["b21_split_sha256"] == str(b21_split_ro["split_sha256"])
    assert num["b21_split_sha256"] == \
        "23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932"
    assert num["b21_split_verdict"] == "B21_SPLIT_VALID"
    assert num["b21_verdict"] == "B21_GENERALIZATION_PASS"


def test_b6_contract_records_b5_as_final(b6_contract):
    """§0: B5_NO_POSITIVE_TRANSFER 是终局, 必须以终局身份入契约。"""
    num = _flat_numeric(b6_contract)
    assert num["b5_verdict"] == "B5_NO_POSITIVE_TRANSFER"
    assert num["b5_ft_positive_transfer"] == "False"
    assert num["b5_mmd_positive_transfer"] == "False"
    assert num["b5_engineering_recommendation"] == "damage_extrapolation"
    assert num["b5_target_only_mean_rmse"] == "0.241024"
    assert num["b5_source_finetune_mean_rmse"] == "0.242659"
    assert num["b5_source_mmd_finetune_mean_rmse"] == "0.241607"
    assert num["b5_damage_extrapolation_mean_rmse"] == "0.054312"


def test_b6_contract_records_b4x_as_exploratory(b6_contract):
    """B4X 的信号必须以"探索性"身份入契约, 不得被当成正式结论继承。"""
    num = _flat_numeric(b6_contract)
    assert num["b4x_exploratory_only"] == "True"
    assert num["b4x_formal_stage"] == "False"


def test_b6_old_artifacts_unchanged(b6_contract):
    """test_b6_old_artifacts_unchanged (§23) —— 冻结文件按契约逐一重算哈希。

    B8 §2 生命周期治理: `MISSING` 项的判定从"永远缺席"改为"缺席, 或已获显式
    阶段授权 (DOWNSTREAM_LIFECYCLE)"。未获授权的 MISSING 项仍然必须缺席。
    算法产物的逐字节冻结不受影响 —— 下面的哈希比对一条未减。
    """
    bad = []
    _n_hashed = _n_exempt = _n_lifecycle_allowed = 0
    for rel, (g, exp) in _flat_files(b6_contract).items():
        p = ROOT / rel
        if exp == "MISSING":
            if not p.exists():
                continue
            if _authorized_to_exist(rel):
                # B7/B8 已获人工阶段授权 —— 允许存在。治理完整性与
                # "不得污染上游"分别由下面两个新守卫看守。
                _n_lifecycle_allowed += 1
                continue
            bad.append((rel, "本应缺席却存在 (无阶段授权)"))
            continue
        if lifecycle.hash_exempt(rel):
            # B7 §14 授权: 最终表达面 + 已改造为 lifecycle transition
            # guard 的两个测试文件。算法产物不在此列。
            assert p.exists(), f"{rel} 已豁免哈希但文件消失"
            _n_exempt += 1
            continue
        if not p.exists():
            bad.append((rel, "已消失"))
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        _n_hashed += 1
        if got != exp:
            bad.append((rel, f"{exp[:8]}->{got[:8]}"))
    assert not bad, f"冻结产物被改动 (B6_INVALID): {bad[:8]}"

    # 防掏空 (B7 生命周期治理): 豁免只允许命中封闭白名单, 且实检项须占绝大多数。
    lifecycle.assert_registry_is_not_widened()
    assert _n_exempt <= 3, f"哈希豁免项过多 ({_n_exempt}) —— 疑似滥用"
    assert _n_hashed >= 20, f"逐字节实检项过少 ({_n_hashed}) —— 契约可能被掏空"
    # 生命周期放行项同样设上限: 只允许 DOWNSTREAM_LIFECYCLE 白名单内的少数产物。
    _n_allowed_max = sum(len(s["artifacts"]) for s in DOWNSTREAM_LIFECYCLE.values())
    assert _n_lifecycle_allowed <= _n_allowed_max, \
        f"生命周期放行项 {_n_lifecycle_allowed} 超出授权白名单 {_n_allowed_max}"


def test_b6_prior_stages_untouched(b6_contract):
    """§2: B6 不得修改 B1.8/B1.9/B2/B2.1/B3X/B4X/**B5** 的任何产物。"""
    tags = ("basilisk_b18", "basilisk_b19", "basilisk_b2/", "basilisk_b21",
            "basilisk_b3x", "basilisk_b4x", "basilisk_b5")
    protected = {rel: sha for rel, (g, sha) in _flat_files(b6_contract).items()
                 if any(t in rel for t in tags) and sha != "MISSING"}
    assert len(protected) >= 20, "受保护前序产物数量异常, 契约覆盖不足"
    n_ex = 0
    for rel, exp in protected.items():
        p = ROOT / rel
        assert p.exists(), f"前序产物缺失: {rel}"
        if lifecycle.hash_exempt(rel):
            # 唯一命中项是 B2.1/B5 的 lifecycle guard **测试文件**
            # (B7 §14 授权改造)。B1.8/B1.9/B2/B2.1/B5 的任何
            # 数据/特征/权重/metrics/verdict 产物都不在豁免名单内,
            # 仍在下面逐字节比对。
            assert rel.startswith("tests/"), \
                f"只有测试文件可豁免, 但命中了 {rel}"
            n_ex += 1
            continue
        assert hashlib.sha256(p.read_bytes()).hexdigest() == exp, \
            f"前序产物被 B6 改动: {rel}"
    lifecycle.assert_registry_is_not_widened()
    assert n_ex <= 2, f"前序产物豁免项过多 ({n_ex})"


def test_b6_source_checkpoint_and_encoder_frozen(b6_contract):
    """源域权重、编码器结构、评估器实现必须与契约一致 —— 迁移起点不能变。"""
    num = _flat_numeric(b6_contract)
    for rel, key in (("checkpoints/source_tcn_pretrain.pt",
                      "source_checkpoint_sha256"),
                     ("src/models/tcn_encoder.py",
                      "encoder_architecture_sha256"),
                     ("src/experiments/metrics.py",
                      "evaluator_metrics_sha256")):
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert got == num[key], f"{rel} 与契约记录不符"


def test_b6_not_written_into_requirements_or_docker():
    """Basilisk 不得写入 requirements.txt / Dockerfile / docker-compose.yml。"""
    for rel in ("requirements.txt", "Dockerfile", "docker-compose.yml"):
        p = ROOT / rel
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore").lower()
        assert "basilisk" not in txt, f"{rel} 被写入 basilisk 内容"


def test_b6_contract_chain_requires_b5_negative(b6_contract):
    ch = b6_contract["frozen_chain"]
    assert ch["BASILISK_B21"] == "B21_GENERALIZATION_PASS"
    assert ch["BASILISK_B2"] == "B2_GENERALIZATION_FAIL"
    assert ch["BASILISK_B3X"] == "B3X_NO_STABILIZING_SIGNAL"
    assert ch["BASILISK_B5"] == "B5_NO_POSITIVE_TRANSFER"


def test_b6_b7_b8_artifacts_must_stay_absent(b6_contract):
    """§22 原版 —— **已由 B8 §2 退役为生命周期迁移守卫, 保留为可追溯记录。**

    旧语义: b6_must_stay_absent 组内 B7/B8 路径**永远**缺席。
    为何 obsolete: B7 (最终图表+报告+证据冻结) 与 B8 (打包交付) 均已获显式
    人工阶段授权并正式执行, "永远缺席"的前提不再成立, 该断言只会永久失败。
    它保护的不是数值正确性, 而是"下游不得推进"这一已被授权取代的假设。

    本测试**不删除**, 而是改为检查旧守卫真正想保护的不变量 —— B6 冻结契约时
    确实把 B7/B8 产物登记为 MISSING (即 B6 阶段本身没有偷跑下游), 这一历史
    事实永久可查。运行期的越界防护移交:
      - test_b6_downstream_stage_transition_is_governed (授权 + 治理完整性)
      - test_b6_downstream_did_not_contaminate_upstream (上游逐字节未污染)
      - test_b6_undelegated_artifacts_still_absent      (未授权项仍须缺席)
    退役依据: docs/handoff/stale_guard_retirement_final.md
    """
    grp = b6_contract["groups"]["b6_must_stay_absent"]
    assert grp, "b6_must_stay_absent 组为空 —— 接力防线失效"
    assert any("basilisk_b7" in k or "basilisk_b8" in k for k in grp)
    # 历史事实: B6 冻结契约时, B7/B8 产物确实被登记为 MISSING。
    for rel, v in grp.items():
        assert v == "MISSING", f"{rel} 在 B6 契约中本应登记为 MISSING"
    # 当前时刻的存在性判断已移交上述三个新守卫, 此处不再断言。


def test_b6_downstream_stage_transition_is_governed():
    """B8 §2 替换守卫 (1/3): 下游产物只有在治理产物齐备时才允许存在。

    拦住的是真正的越界 —— 没有协议/契约/审计就先产出下游产物。
    比旧断言更强: 旧断言只能拦"阶段推进", 新断言拦"无治理的阶段推进"。
    """
    for stage, spec in DOWNSTREAM_LIFECYCLE.items():
        present = [a for a in spec["artifacts"] if (ROOT / a).exists()]
        if not present:
            continue
        for gov in spec["governance"]:
            assert (ROOT / gov).exists(), (
                f"{stage} 已产出 {present[0]} 但缺治理产物 {gov} —— "
                f"未经协议/审计冻结即推进阶段, 属越界")


def test_b6_downstream_did_not_contaminate_upstream():
    """B8 §2 替换守卫 (2/3): 下游阶段出现后, 上游冻结产物必须逐字节不变。

    这是本次退役的**核心补偿条件**: 删掉"下游永远不能出现", 换成
    "下游出现了也绝不能污染上游"。覆盖 §2 要求的五项:
      B5 frozen metrics / B6 frozen metrics / B2.1 split /
      source checkpoint / B1.9 target feature
    任何一项漂移 -> B6_INVALID (等价于 B8_HANDOFF_INVALID)。
    """
    drift = []
    for rel, exp in FROZEN_UPSTREAM.items():
        p = ROOT / rel
        assert p.exists(), f"上游冻结产物消失: {rel}"
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != exp:
            drift.append(f"{rel}: {exp[:12]} -> {got[:12]}")
    assert not drift, (
        "下游阶段污染了上游冻结产物 (B6_INVALID / B8_HANDOFF_INVALID):\n  "
        + "\n  ".join(drift))
    # 防掏空: 五项一个都不许少。
    assert len(FROZEN_UPSTREAM) >= 5, "上游冻结核查项被削减"


def test_b6_undelegated_artifacts_still_absent():
    """B8 §2 替换守卫 (3/3): 未获授权的产物仍然必须缺席 —— 不得放松。

    B7 是纯表达阶段、B8 是纯打包阶段, 两者都不训练、不产生新实验数字;
    星敏 S7/S8 线从未获得任何阶段授权。
    """
    present = [rel for rel in STILL_FORBIDDEN if (ROOT / rel).exists()]
    assert not present, f"以下产物未获阶段授权, 本应缺席: {present}"


def test_b6_stale_guard_retirement_is_documented():
    """B8 §3: 退役必须留下书面依据, 不允许静默删除守卫。"""
    doc = ROOT / "docs" / "handoff" / "stale_guard_retirement_final.md"
    assert doc.exists(), "缺 docs/handoff/stale_guard_retirement_final.md"
    txt = doc.read_text(encoding="utf-8")
    for key in ("test_b6_old_artifacts_unchanged",
                "test_b6_b7_b8_artifacts_must_stay_absent",
                "algorithm_numbers_affected",
                "data_affected"):
        assert key in txt, f"退役文档未记录 {key}"
    # 必须显式声明本次改动不触碰科学结论
    assert ("lifecycle governance" in txt or "生命周期治理" in txt), \
        "退役文档未声明改动性质为 lifecycle governance"
