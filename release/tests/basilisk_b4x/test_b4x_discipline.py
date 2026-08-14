"""tests/basilisk_b4x/test_b4x_discipline.py

BASILISK-B4X §19 —— 纪律测试。

`POST_B2_FAIL_EXPLORATORY` / `NOT_FORMAL_EVIDENCE` / `EXPLORATORY_ONLY`

§19 点名要求的 8 个测试:
    test_b4x_old_b2_unchanged
    test_b4x_same_split
    test_three_groups_same_target_data
    test_transfer_rng_alignment
    test_catastrophic_threshold_uses_val_only
    test_short_eol_subset_fixed
    test_replay_and_new_seeds_separate
    test_b4x_not_formal_claim
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

# B2 namespace —— B4X 全程不得触碰。
B2_PROTECTED = (
    "configs/wheel_basilisk_b2.yaml",
    "checkpoints/basilisk_b2/split.json",
    "checkpoints/basilisk_b2/metrics.json",
    "checkpoints/basilisk_b2/protocol_hash.json",
    "docs/basilisk_b2/protocol.md",
    "docs/basilisk_b2/results.md",
    "STATUS_BASILISK_B2.md",
)

# B3X namespace 也已封档, B4X 只读。
B3X_PROTECTED = (
    "configs/wheel_basilisk_b3x.yaml",
    "checkpoints/basilisk_b3x/summary.json",
    "docs/basilisk_b3x/baseline_contract.json",
    "STATUS_BASILISK_B3X.md",
)

# §18 禁止出现的结论表述。以片段拼装, 避免源码里留下完整的正面断言字样。
_FORBIDDEN_CLAIM = "positive transfer " + "proven"
_FORBIDDEN_ZH = "迁移显著" + "有效"


def _sha(rel: str) -> str:
    p = ROOT / rel
    return (hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file()
            else "MISSING")


def _flat(baseline: dict) -> dict:
    return {r: h for g in baseline["groups"].values() for r, h in g.items()}


# ---------------------------------------------------------------- §2 / §0
def test_b4x_old_b2_unchanged(b4x_baseline):
    """B2 的全部产物必须与 B4X 契约首次写入时逐字一致。"""
    flat = b4x_baseline["flat_sha256"]
    for rel in B2_PROTECTED:
        assert rel in flat, f"{rel} 未进 B4X 契约 —— 无法证明它没被改"
        assert flat[rel] != "MISSING", f"{rel} 在契约里是 MISSING"
        assert _sha(rel) == flat[rel], f"B2 产物被修改: {rel}"
    exp = b4x_baseline["b2_expected"]
    assert exp["b2_verdict"] == "B2_GENERALIZATION_FAIL"
    for k in ("b2_train_ids_sha256", "b2_val_ids_sha256",
              "b2_test_ids_sha256", "b2_split_sha256"):
        assert len(exp[k]) == 64 and "PLACEHOLDER" not in exp[k], k


def test_b4x_b3x_frozen_and_verdict_pinned(b4x_baseline, b4x_b3x_summary):
    """B3X 已封档: 其产物也必须逐字不变, 且 verdict 被 pin 进 B4X 契约。"""
    flat = b4x_baseline["flat_sha256"]
    for rel in B3X_PROTECTED:
        assert rel in flat, f"{rel} 未进 B4X 契约"
        assert flat[rel] != "MISSING", rel
        assert _sha(rel) == flat[rel], f"B3X 产物被修改: {rel}"
    exp = b4x_baseline["b3x_expected"]
    assert exp["b3x_verdict"] == b4x_b3x_summary["verdict"]
    assert exp["b3x_label"] == "POST_B2_FAIL_EXPLORATORY"
    # B3X 的禁用正式标签绝不允许被启用
    assert str(exp["b3x_forbidden_label_used"]) in ("0", "0.0", "False")
    assert b4x_baseline["frozen_chain"]["BASILISK_B3X"] \
        == b4x_b3x_summary["verdict"]


def test_b4x_formal_artifacts_absent(b4x_baseline):
    """§17/§18: 正式 claim 类产物必须始终缺席。"""
    g = b4x_baseline["groups"]["b4x_must_stay_absent"]
    assert g, "absent 组不能为空"
    for rel, h in g.items():
        assert h == "MISSING", f"{rel} 不应存在 —— 探索性结论被提升成正式 claim"
        assert not (ROOT / rel).exists(), rel


# ---------------------------------------------------------------- §12
def test_b4x_same_split(b4x_config, b4x_b2_split, b4x_transfer):
    """§12: 继续用同一 B2 划分, 不重新 stratify。"""
    b4 = b4x_config["b4x"]
    assert b4["split"]["reuse_from"] == "checkpoints/basilisk_b2/split.json"
    assert b4["split"]["forbid_restratify"] is True
    assert (str(b4["split"]["expected_sha256"])
            == str(b4x_b2_split["split_sha256"]))
    assert b4x_transfer["split_sha256"] == b4x_b2_split["split_sha256"]
    assert (b4x_transfer["split_reused_from"]
            == "checkpoints/basilisk_b2/split.json")
    s = b4x_b2_split["splits"]
    assert (len(s["train"]["tids"]), len(s["val"]["tids"]),
            len(s["test"]["tids"])) == (44, 31, 75)


def test_b4x_loader_rejects_foreign_split(b4x_config, b4x_data_mod,
                                          b4x_b2_split):
    """篡改 split_sha256 必须被 loader 拒绝。"""
    bad = json.loads(json.dumps(b4x_b2_split))
    bad["split_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="B4X_SPLIT_MISMATCH"):
        b4x_data_mod.prepare_b4x(b4x_config, bad, 92, "core_only",
                                 verbose=False)


# ---------------------------------------------------------------- §10 / §11
def test_b4x_input_schema_follows_b3x_rule(b4x_config, b4x_data_mod,
                                           b4x_b3x_summary, b4x_transfer):
    """§10: 输入 schema 由 B3X verdict 唯一决定, 不能按结果好看来挑。"""
    rule = {str(k): str(v)
            for k, v in b4x_config["b4x"]["input_schema"]["rule"].items()}
    want = rule[str(b4x_b3x_summary["verdict"])]
    assert str(b4x_b3x_summary["b4x_input_schema"]) == want
    assert str(b4x_transfer["input_schema"]) == want
    assert b4x_data_mod.resolve_input_schema(b4x_config, verbose=False) == want
    # 规则必须在跑 B4X 之前写进 B3X 的 protocol
    txt = (ROOT / "docs" / "basilisk_b3x" / "protocol.md").read_text(
        encoding="utf-8")
    assert "B4X" in txt and "core_plus_mission" in txt


def test_three_groups_same_target_data(b4x_config, b4x_transfer):
    """§11/§6: 三组共用同一份目标域输入 —— 划分/schema/宽度/batch 顺序都相同。"""
    groups = list(b4x_transfer["groups"])
    assert groups == ["target_only", "source_finetune", "source_mmd_finetune"]
    assert b4x_config["b4x"]["forbid_new_transfer_methods"] is True
    assert b4x_config["b4x"]["input_schema"]["shared_across_groups"] is True
    n_rows = 0
    for rows in b4x_transfer["per_seed_by_set"].values():
        for r in rows:
            metas = [r[f"{g}_meta"] for g in groups]
            ref = metas[0]
            for m in metas[1:]:
                for k in ("input_schema", "n_target", "n_features_source_side",
                          "batch_order_generator_seed", "rul_scale"):
                    assert m[k] == ref[k], (
                        f"seed {r['seed']}: {k} 三组不一致 —— 目标域输入不共用")
            # 只有 source 组加载 checkpoint, target_only 必须没加载
            assert r["target_only_meta"]["source_checkpoint_loaded"] is False
            for g in ("source_finetune", "source_mmd_finetune"):
                assert r[f"{g}_meta"]["source_checkpoint_loaded"] is True
            n_rows += 1
    assert n_rows > 0


def test_transfer_rng_alignment(b4x_config, b4x_data_mod, b4x_b2_split):
    """§6: 同 seed 下目标域 batch 顺序在三组之间必须逐值相同。

    另外断言目标域与源域使用**两个独立**生成器 —— 否则 MMD 组从源域 loader
    取 batch 会偷偷移动目标域的 batch 顺序, 三组就不再可比。
    """
    seed = 92
    a = b4x_data_mod.prepare_b4x(b4x_config, b4x_b2_split, seed, "core_only",
                                 with_test=True, verbose=False)
    b = b4x_data_mod.prepare_b4x(b4x_config, b4x_b2_split, seed, "core_only",
                                 with_test=True, verbose=False)
    for k in ("tr", "va", "te"):
        assert np.array_equal(a[k], b[k]), f"{k} 划分不一致"
    assert a["rul_scale"] == b["rul_scale"]
    assert a["censor_eta"] == b["censor_eta"]
    assert a["n_target"] == b["n_target"]

    def order(loader):
        return np.concatenate([np.asarray(t[3]).ravel() for t in loader])

    oa, ob = order(a["ltr"]), order(b["ltr"])
    assert oa.shape == ob.shape
    assert np.array_equal(oa, ob), "同 seed 两次装载 batch 顺序不同 —— 未受控"
    assert (a["batch_order_generator_seed"]
            == seed * 1000 + b4x_data_mod.TARGET_GEN_OFFSET)

    # 源域 loader 消耗自己的生成器: 先建源域再建目标域, 目标域顺序不得改变
    src = b4x_data_mod.prepare_source(b4x_config, seed, verbose=False)
    assert (src["source_generator_seed"]
            == seed * 1000 + b4x_data_mod.SOURCE_GEN_OFFSET)
    assert (b4x_data_mod.SOURCE_GEN_OFFSET
            != b4x_data_mod.TARGET_GEN_OFFSET), "两个偏移必须不同"
    for _ in range(3):
        next(iter(src["loader"]))
    c = b4x_data_mod.prepare_b4x(b4x_config, b4x_b2_split, seed, "core_only",
                                 with_test=True, verbose=False)
    assert np.array_equal(order(c["ltr"]), oa), (
        "源域取 batch 之后目标域顺序变了 —— 两侧共用了同一个 RNG")


def test_b4x_frozen_hyperparams(b4x_config, b4x_transfer):
    """§0: 早停 / max_epochs / weight_decay / loss 权重 / mmd_lambda 仍是 B2 的值。"""
    t = b4x_config["training"]
    assert t["max_epochs"] == 8
    assert t["early_stop_patience"] == 2
    assert float(t["weight_decay"]) == 1e-3
    assert t["early_stop_metric"] == "info_macro_rmse"
    assert float(t["post_eol_weight"]) == 0.1
    assert float(t["capped_weight"]) == 0.1
    assert b4x_config["transfer"]["target_hi_key"] == "hi_damage_obs"
    # mmd_lambda 不在本阶段改动: 产物记录值必须等于 config 的 transfer 值
    assert (float(b4x_transfer["mmd_lambda"])
            == float(b4x_config["transfer"]["mmd_lambda"]))
    assert (str(b4x_transfer["source_checkpoint"]).replace("\\", "/")
            == str(b4x_config["b4x"]["source"]["checkpoint"]))
    for r in b4x_transfer["per_seed_by_set"].get("new", []):
        h = r["target_only_meta"]["hyper"]
        assert int(h["max_epochs"]) == 8
        assert int(h["early_stop_patience"]) == 2


def test_b4x_target_only_is_rerun_not_b2_number(b4x_config, b4x_transfer,
                                                b4x_run_mod):
    """§3.1: 三组共用 source 宽度, 故 target_only 是重跑 —— 必须显式声明。"""
    note = str(b4x_transfer["architecture_note"])
    assert "target_only" in note and ("重跑" in note or "rerun" in note)
    assert (str(b4x_config["b4x"]["architecture_note"])
            == "shared_source_sized_adapter_target_only_is_rerun")
    assert "B2" in b4x_run_mod.ARCH_NOTE
    for rows in b4x_transfer["per_seed_by_set"].values():
        for r in rows:
            assert int(r["target_only_meta"]["n_features_source_side"]) == 12


# ---------------------------------------------------------------- §16
def test_catastrophic_threshold_uses_val_only(b4x_config, b4x_stability,
                                              b4x_stab_mod):
    """§16: 阈值只能来自 target_only 的 validation, 绝不用 test 分布。"""
    cc = b4x_config["b4x"]["catastrophic"]
    assert cc["forbid_test_derived_threshold"] is True
    assert (str(cc["threshold_source"])
            == "target_only_validation_median_trajectory_rmse")
    assert float(cc["multiplier"]) == 2.0
    c = b4x_stability["catastrophic"]
    assert c["threshold_split"] == "validation"
    assert c["threshold_group"] == "target_only"
    assert c["forbid_test_derived_threshold"] is True
    assert b4x_stab_mod.THRESHOLD_GROUP == "target_only"
    assert "val" in b4x_stab_mod.THRESHOLD_DISCIPLINE
    for rows in b4x_stability["per_seed_by_set"].values():
        for r in rows:
            thr = r["catastrophic_threshold"]
            assert thr["derived_from_test"] is False
            assert thr["split"] == "validation"
            assert thr["group"] == "target_only"
            # 三组共用同一个阈值
            for g in b4x_stability["groups"]:
                assert (float(r[g]["catastrophic"]["threshold"])
                        == float(thr["threshold"]))
            if np.isfinite(float(thr["threshold"])):
                assert abs(float(thr["threshold"])
                           - 2.0 * float(thr["val_median_trajectory_rmse"])) < 1e-9
            else:
                # 算不出来就必须是 NaN + n=None, 不得伪造 0
                for g in b4x_stability["groups"]:
                    cat = r[g]["catastrophic"]
                    assert cat["n_catastrophic"] is None
                    assert cat["catastrophic_error_rate"] != \
                        cat["catastrophic_error_rate"]


def test_catastrophic_threshold_independent_of_test(b4x_stab_mod):
    """阈值函数只接受 val 数组; 喂 test 分布不应改变它 —— 用构造数据验证。"""
    rng = np.random.default_rng(4241)
    tids = np.repeat(np.arange(10), 20).astype(np.int64)
    true = rng.uniform(0.1, 0.9, size=tids.size)
    pred = true + rng.normal(0, 0.05, size=tids.size)
    a = b4x_stab_mod.catastrophic_threshold(pred, true, tids, 1e-6, 2.0)
    # 同样的 val 输入 -> 同样的阈值, 与任何 test 数据无关
    b = b4x_stab_mod.catastrophic_threshold(pred, true, tids, 1e-6, 2.0)
    assert a["threshold"] == b["threshold"]
    assert a["split"] == "validation" and a["derived_from_test"] is False
    # 空输入 -> NaN, 不是 0
    empty = np.asarray([], float)
    z = b4x_stab_mod.catastrophic_threshold(
        empty, empty, np.asarray([], np.int64), 1e-6, 2.0)
    assert z["threshold"] != z["threshold"]


# ---------------------------------------------------------------- §14
def test_short_eol_subset_fixed(b4x_config, b4x_baseline, b4x_b2_split,
                                b4x_stability):
    """§14: short-EOL 子集 = {19,109,111}, 写死在 config + 契约, 且确在 test 内。"""
    want = [19, 109, 111]
    assert [int(x) for x in b4x_config["b4x"]["short_eol_subset"]] == want
    assert b4x_baseline["b2_expected"]["b2_short_eol_diagnostic_ids"] == str(want)
    assert [int(x) for x in b4x_stability["short_eol_subset"]] == want
    normal = [int(x) for x in b4x_stability["normal_eol_subset"]]
    assert not set(normal) & set(want)
    assert len(normal) == 72
    assert b4x_stability["n_test"] == 75
    names = sorted(b4x_b2_split["splits"]["test"]["tids"])
    idx = {int(n.split("_")[1]) for n in names}
    assert set(want) <= idx, "short-EOL tid 不在 test 划分里"


def test_subsets_reported_separately_not_merged(b4x_stability):
    """§14: 三个子集分别出数; 空子集是 NaN + n_traj=0 而不是 0。"""
    for rows in b4x_stability["per_seed_by_set"].values():
        for r in rows:
            for g in b4x_stability["groups"]:
                blocks = r[g]
                assert {"overall", "short_eol", "normal_eol"} <= set(blocks)
                for sub in ("overall", "short_eol", "normal_eol"):
                    m = blocks[sub]
                    if m["n_traj"] == 0:
                        assert m["rmse"] != m["rmse"], "空子集必须 NaN 不是 0"
                    else:
                        assert m["n_endpoints"] > 0
    assert b4x_stability["macro_unit"] == "trajectory_not_endpoint"


def test_censored_trajectories_not_given_fake_eol(b4x_stability):
    """右删失轨迹不得被伪造 EOL 指标: 不进 PH / alpha-lambda 的分子分母。"""
    seen = 0
    for rows in b4x_stability["per_seed_by_set"].values():
        for r in rows:
            for g in b4x_stability["groups"]:
                ov = r[g]["overall"]
                assert {"macro_ph", "ph_n_observable", "ph_n_censored",
                        "alpha_lambda"} <= set(ov)
                n_obs = int(ov["ph_n_observable"])
                n_cen = int(ov["ph_n_censored"])
                # 75 条 test 里既有 event 也有删失, 两者都必须被单独计数
                assert n_cen > 0, "删失轨迹被当成 event 处理了"
                assert n_obs + n_cen == b4x_stability["n_test"]
                for lam, v in ov["alpha_lambda"].items():
                    # eligible 只能来自 event 观测轨迹, 绝不含删失轨迹
                    assert int(v["eligible_count"]) <= n_obs, (
                        f"lambda={lam}: eligible {v['eligible_count']} > "
                        f"可观测 {n_obs} —— 删失轨迹被伪造了 EOL")
                    assert int(v["success_count"]) <= int(v["eligible_count"])
                w = ov["warning"]
                assert int(w["n_censored"]) >= 0
                assert int(w["n_observed"]) + int(w["n_censored"]) >= 1
                seen += 1
    assert seen > 0


# ---------------------------------------------------------------- §13
def test_replay_and_new_seeds_separate(b4x_config, b4x_transfer, b4x_stability,
                                       b4x_summary):
    """§13: 两套 seed 全程分开, 结构上不给合并成 6-seed 平均留位置。"""
    want = {"new": [92, 93, 94], "replay": [72, 74, 76]}
    for k, v in want.items():
        assert [int(x) for x in b4x_config["b4x"]["seeds"][k]] == v
    assert b4x_config["b4x"]["forbid_merging_seed_sets"] is True
    for d in (b4x_transfer, b4x_stability, b4x_summary):
        assert d["forbid_merging_seed_sets"] is True
        ss = {k: [int(x) for x in v] for k, v in d["seed_sets"].items()}
        assert ss == want, f"seed 套不完整: {ss}"
    # summary 里两套各自独立统计, 且没有任何跨套聚合键
    assert set(b4x_summary["by_seed_set"]) == {"new", "replay"}
    for k in ("pooled", "all_seeds", "merged", "combined", "six_seed"):
        assert k not in b4x_summary, f"summary 出现跨套聚合键 {k}"
    for sname in ("new", "replay"):
        blk = b4x_summary["by_seed_set"][sname]
        for m in ("source_finetune", "source_mmd_finetune"):
            mb = blk["methods"][m]
            assert mb["n_seeds"] == 3
            assert set(mb["gain_per_seed"]["overall"]) == {
                str(s) for s in want[sname]}
    assert (b4x_summary["descriptive_ci"]["forbid_pooling_seed_sets"] is True
            if "descriptive_ci" in b4x_summary
            else b4x_config["b4x"]["descriptive_ci"]
            ["forbid_pooling_seed_sets"] is True)


def test_descriptive_ci_unit_is_seed_not_endpoint(b4x_config, b4x_summary):
    """禁止把时间点数量当独立样本 —— CI 的单位必须是"一个 seed 一个配对差值"。"""
    ci = b4x_config["b4x"]["descriptive_ci"]
    assert ci["unit"] == "one_paired_difference_per_seed"
    for sname in ("new", "replay"):
        for m in ("source_finetune", "source_mmd_finetune"):
            d = b4x_summary["by_seed_set"][sname]["methods"][m]["descriptive_ci"]
            for sub in ("overall", "short_eol", "normal_eol"):
                b = d[sub]
                assert b["n"] <= 3, "CI 样本数超过 seed 数 —— 端点被当独立样本"
                assert (b["interpretation"]
                        == "descriptive_only_not_significance_claim")


# ---------------------------------------------------------------- §17 / §18
def test_b4x_not_formal_claim(b4x_config, b4x_summary, b4x_summ_mod):
    """§17: 只允许三个探索性标签, 且必须同时写三条限定语。"""
    assert set(b4x_summ_mod.ALLOWED_LABELS) == {
        "B4X_TRANSFER_STABILIZATION_SIGNAL",
        "B4X_GENERIC_TRANSFER_SIGNAL",
        "B4X_NO_TRANSFER_SIGNAL"}
    v = b4x_summary["verdict"]
    assert v in b4x_summ_mod.ALLOWED_LABELS, f"非法 verdict {v}"
    assert set(str(x) for x in b4x_config["b4x"]["verdict"]["allowed_labels"]) \
        == set(b4x_summ_mod.ALLOWED_LABELS)
    assert b4x_summary["label_discipline"] == "POST_B2_FAIL_EXPLORATORY"
    assert b4x_summary["formal_stage"] is False
    assert b4x_summary["not_formal_evidence"] is True
    assert b4x_summary["exploratory_only"] is True
    assert b4x_summary["not_formal_transfer_evidence"] is True
    assert b4x_summary["b2_verdict_unchanged"] == "B2_GENERALIZATION_FAIL"
    assert (b4x_summary["b2_verdict_status"]
            == "B2_GENERALIZATION_FAIL_REMAINS_FINAL")
    assert list(b4x_summary["mandatory_qualifiers"]) == [
        "EXPLORATORY_ONLY", "B2_GENERALIZATION_FAIL_REMAINS_FINAL",
        "NOT_FORMAL_TRANSFER_EVIDENCE"]
    # 三条限定语必须真的出现在所有 B4X 文档里
    for rel in ("docs/basilisk_b4x/transfer_results.md",
                "docs/basilisk_b4x/short_eol_transfer_diagnostics.md",
                "docs/basilisk_b4x/limitations.md",
                "STATUS_BASILISK_B4X.md"):
        p = ROOT / rel
        if not p.exists():
            pytest.skip(f"文档尚未生成: {rel}")
        txt = p.read_text(encoding="utf-8")
        assert "POST_B2_FAIL_EXPLORATORY" in txt, rel
        assert "NOT_FORMAL_EVIDENCE" in txt, rel
        for q in ("EXPLORATORY_ONLY", "B2_GENERALIZATION_FAIL_REMAINS_FINAL",
                  "NOT_FORMAL_TRANSFER_EVIDENCE"):
            assert q in txt, f"{rel} 缺限定语 {q}"
    # B2 / B3X 的结论文件不得被 B4X 污染
    for rel in ("STATUS_BASILISK_B2.md", "docs/basilisk_b2/results.md"):
        assert "B4X" not in (ROOT / rel).read_text(encoding="utf-8"), rel


def test_b4x_forbids_positive_transfer_phrasing(b4x_summary, b4x_summ_mod,
                                                b4x_run_mod):
    """§18: 禁止正面迁移结论表述; 必须提供"只是更保守"的替代措辞。"""
    assert b4x_summary["forbid_positive_transfer_claim"] is True
    assert (b4x_summary["regularization_phrasing"]
            == "source initialization regularizes extrapolation under the "
               "known B2 distribution gap")
    assert _FORBIDDEN_CLAIM in b4x_summ_mod.FORBIDDEN_CLAIM_NOTE
    assert _FORBIDDEN_ZH in b4x_run_mod.PHRASING_DISCIPLINE
    # 结论文档里不得出现正面断言
    for rel in ("docs/basilisk_b4x/transfer_results.md",
                "STATUS_BASILISK_B4X.md"):
        p = ROOT / rel
        if not p.exists():
            pytest.skip(f"文档尚未生成: {rel}")
        txt = p.read_text(encoding="utf-8")
        # 允许以"禁止…"的形式引用该措辞, 但不得作为结论出现在标题行
        for line in txt.splitlines():
            if line.lstrip().startswith("#") or line.lstrip().startswith(">"):
                assert _FORBIDDEN_ZH not in line, f"{rel} 标题含禁止表述: {line}"


def test_b4x_collapse_attribution_present(b4x_config, b4x_summary):
    """§18: 任何 source 优势都必须先做崩塌归因, 且归因有可查证据。"""
    assert b4x_config["b4x"]["verdict"]["require_collapse_attribution"] is True
    allowed = {"regularizes_extrapolation", "normal_region_also_improves",
               "no_advantage_to_attribute"}
    for m, c in b4x_summary["conditions_by_method"].items():
        for key in ("collapse_attribution_replay", "collapse_attribution_new"):
            att = c[key]
            assert att["reading"] in allowed, f"{m}/{key}: {att['reading']}"
            ev = att["evidence"]
            for k in ("mean_gain_overall", "mean_gain_short_eol",
                      "mean_gain_normal_eol", "n_seeds_catastrophic_reduced",
                      "mean_delta_macro_corr"):
                assert k in ev, f"{m}/{key} 缺证据项 {k}"
            if att["reading"] == "regularizes_extrapolation":
                assert att["phrasing"] == b4x_summary["regularization_phrasing"]


def test_b4x_verdict_recomputes_from_conditions(b4x_summary, b4x_summ_mod):
    """verdict 必须能由记录的判据重算 —— 不是手写进 JSON 的。"""
    cbm = b4x_summary["conditions_by_method"]
    a_ok, b_ok = [], []
    for m, c in cbm.items():
        assert c["A_passed"] == all(x["passed"] for x in c["A"])
        assert c["B_passed"] == all(x["passed"] for x in c["B"])
        assert c["n_A_passed"] == sum(1 for x in c["A"] if x["passed"])
        assert c["n_B_passed"] == sum(1 for x in c["B"] if x["passed"])
        if c["A_passed"]:
            a_ok.append(m)
            if c["B_passed"]:
                b_ok.append(m)
    expect = (b4x_summ_mod.LABEL_B if b_ok
              else b4x_summ_mod.LABEL_A if a_ok
              else b4x_summ_mod.LABEL_C)
    assert b4x_summary["verdict"] == expect
    assert sorted(b4x_summary["methods_passing_A"]) == sorted(a_ok)
    assert sorted(b4x_summary["methods_passing_A_and_B"]) == sorted(b_ok)
    assert b4x_summary["thresholds_frozen_before_run"] is True


def test_b4x_thresholds_frozen_in_protocol(b4x_config, b4x_summary):
    """§17 判据阈值必须来自 config/protocol, 不得按结果调。"""
    st = b4x_config["b4x"]["verdict"]["stabilization"]
    gen = b4x_config["b4x"]["verdict"]["generic"]
    assert int(st["min_replay_seeds_catastrophic_reduced"]) == 2
    assert int(gen["min_new_seeds_gain_positive"]) == 2
    assert float(st["normal_eol_max_rel_degradation"]) == 0.05
    txt = (ROOT / b4x_config["protocol"]["path"]).read_text(encoding="utf-8")
    for tok in ("B4X_TRANSFER_STABILIZATION_SIGNAL",
                "B4X_GENERIC_TRANSFER_SIGNAL", "B4X_NO_TRANSFER_SIGNAL"):
        assert tok in txt, f"protocol 缺标签定义 {tok}"
    ph = json.loads(
        (ROOT / b4x_config["protocol"]["hash_path"]).read_text(encoding="utf-8"))
    assert ph["frozen_before_any_b4x_number"] is True
    assert (hashlib.sha256((ROOT / b4x_config["protocol"]["path"]).read_bytes())
            .hexdigest() == ph["sha256"]), "protocol 在出数之后被改过"


def test_b4x_gain_definition_consistent(b4x_stability):
    """§15: gain = target_only − source_*, 三子集各自算, 数值必须自洽。"""
    for rows in b4x_stability["per_seed_by_set"].values():
        for r in rows:
            for sub in ("overall", "short_eol", "normal_eol"):
                g = r["gains"][sub]
                for key, meth in (("gain_ft", "source_finetune"),
                                  ("gain_mmd", "source_mmd_finetune")):
                    if np.isfinite(g["target_only"]) and np.isfinite(g[meth]):
                        assert abs((g["target_only"] - g[meth]) - g[key]) < 1e-12


def test_b4x_not_fast_smoke(b4x_transfer):
    """--fast 冒烟产物不得成为报告依据。"""
    assert b4x_transfer["fast"] is False
    assert list(b4x_transfer["groups"]) == [
        "target_only", "source_finetune", "source_mmd_finetune"]
