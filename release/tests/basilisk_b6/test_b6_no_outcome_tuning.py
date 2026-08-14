"""tests/basilisk_b6/test_b6_no_outcome_tuning.py —— §10/§12/§15 反"看结果调参"纪律。

B6 是收口, 不是扩方法。本文件守三件事:
- 不得搜索 / 新增 MMD lambda, 不得加 Wiener PF / rate model / 新迁移方法 (§10)
- 不得在看到 test 之后降低门槛 (§15 forbid_lowering_bar)
- 不得把 oracle 类方法 (const / persistence / true-rul) 混入可部署排名 (§19)
"""
from __future__ import annotations

from conftest import ROOT, code_nospace, code_only

B6_SCRIPTS = ("scripts/basilisk_b6/build_label_subsets.py",
              "scripts/basilisk_b6/audit_label_subsets.py",
              "scripts/basilisk_b6/data_b6.py",
              "scripts/basilisk_b6/run_formal_matrix.py",
              "scripts/basilisk_b6/paired_statistics.py",
              "scripts/basilisk_b6/summarize_matrix.py",
              "scripts/basilisk_b6/final_transfer_verdict.py",
              "scripts/basilisk_b6/freeze_protocol.py")
TRAINED = ("target_only", "source_finetune", "source_mmd_finetune")


def test_b6_no_mmd_search(b6_config, b6_protocol_hash):
    """test_b6_no_mmd_search (§23/§10) —— MMD lambda 单一冻结值, 无搜索。"""
    assert float(b6_config["transfer"]["mmd_lambda"]) == 1.0
    assert float(b6_protocol_hash["mmd_lambda_frozen"]) == 1.0
    assert bool(b6_config["b6"]["forbid_new_mmd_lambda"]) is True
    assert bool(b6_config["b6"]["exit_rules"]["forbid_tuning_mmd"]) is True
    for rel in B6_SCRIPTS:
        src = code_nospace(rel)
        for bad in ("mmd_lambdas", "lambda_grid", "lambda_sweep",
                    "for lam in", "mmd_lambda_candidates", "itertools.product"):
            assert bad.replace(" ", "") not in src, f"{rel} 出现 MMD 搜索: {bad}"


def test_b6_single_mmd_lambda_in_metrics(b6_metrics):
    """产物侧确认: 整个矩阵里 MMD lambda 只出现一个取值。"""
    seen = set()
    for lv in b6_metrics["by_level"].values():
        for row in lv["per_seed"]:
            for st in row["source_mmd_finetune_meta"]["stages"]:
                if bool(st.get("use_mmd")):
                    seen.add(float(st["mmd_lambda"]))
    assert seen == {1.0}, f"MMD lambda 出现多个取值: {seen}"


def test_b6_no_new_methods(b6_config, b6_metrics):
    """§10: 方法矩阵只有 3 训练组 + 2 评估组, 不得扩方法。"""
    b6 = b6_config["b6"]
    assert list(b6["trained_groups"]) == list(TRAINED)
    assert list(b6["eval_only_groups"]) == ["damage_extrapolation",
                                            "const_mean_info"]
    for k in ("forbid_new_transfer_methods", "forbid_wiener_pf",
              "forbid_rate_model"):
        assert bool(b6[k]) is True, f"b6.{k} 必须为 true"
    allowed = set(TRAINED) | {"damage_extrapolation", "const_mean_info"}
    for lv in b6_metrics["by_level"].values():
        for row in lv["per_seed"]:
            got = {k for k in row if k in allowed}
            extra = {k for k in row
                     if k.endswith(("_finetune", "_only", "_extrapolation"))
                     and k not in allowed}
            assert extra == set(), f"出现未授权方法组: {extra}"
            assert got == allowed, f"方法组不全: 缺 {allowed - got}"


def test_b6_no_forbidden_method_names_in_code():
    """§10 静态禁令: 代码里不得出现 Wiener PF / rate model 的实现痕迹。"""
    for rel in B6_SCRIPTS:
        src = code_nospace(rel).lower()
        for bad in ("wiener", "particlefilter", "particle_filter",
                    "ratemodel", "rate_model"):
            assert bad not in src, f"{rel} 出现被禁方法: {bad}"


def test_b6_forbid_lowering_bar(b6_config, b6_protocol_hash):
    """§15: 门槛在任何 B6 数字之前冻结, 且 gate 判定读冻结件而非实时 config。"""
    g = b6_config["b6"]["gate"]
    assert bool(g["forbid_lowering_bar"]) is True
    assert bool(b6_protocol_hash["forbid_lowering_bar"]) is True
    assert bool(b6_protocol_hash["frozen_before_any_b6_number"]) is True
    thr = b6_protocol_hash["gate_thresholds"]
    # 冻结阈值必须与 §15 的十条逐字一致
    assert int(thr["n_conditions"]) == 10
    assert int(thr["n_seeds"]) == 5
    assert bool(g["all_required"]) is True
    assert int(thr["min_improve_count"]) == 4
    assert int(thr["min_lifetime_bins_gain_nonneg"]) == 2
    assert int(thr["n_lifetime_bins"]) == 3
    assert float(thr["corr_tol"]) == 0.02
    assert float(thr["warning_miss_tol"]) == 0.05
    for k in ("require_mean_positive", "require_median_positive",
              "require_ci_lower_positive", "require_catastrophic_not_worse",
              "require_fairness_pass", "require_beat_const_mean_info"):
        assert bool(thr[k]) is True, f"gate_thresholds.{k} 必须为 true"
    # 冻结件与实时 config 必须一致 —— 否则说明 config 在跑完后被改过
    for k, v in thr.items():
        if k in g:
            assert type(v) is type(g[k]) or float(v) == float(g[k]), \
                f"gate.{k} 与冻结件不符: {g[k]} vs {v}"


def test_b6_gate_labels_frozen(b6_protocol_hash, b6_config):
    """§15: 判定标签冻结, 不得改写措辞来软化结论。"""
    lab = b6_protocol_hash["gate_labels"]
    assert str(lab["primary_pass"]) == "B6_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER"
    assert str(lab["primary_fail"]) == \
        "B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER"
    assert dict(b6_config["b6"]["gate"]["labels"]) == dict(lab)


def test_b6_verdict_reads_frozen_thresholds():
    """§15: 判定脚本必须从 protocol_hash.json 取阈值, 而不是直接吃实时 config。"""
    rel = "scripts/basilisk_b6/final_transfer_verdict.py"
    raw = (ROOT / rel).read_text(encoding="utf-8")
    # 这些键名都是字符串字面量, code_only 会剥掉, 故按原文扫
    assert 'proto["gate_thresholds"]' in raw, "判定脚本未读取冻结的 gate_thresholds"
    assert 'proto["gate_labels"]' in raw, "判定脚本未读取冻结的 gate_labels"
    assert 'cfg["protocol"]["hash_path"]' in raw, \
        "判定脚本未从 protocol.hash_path 打开冻结件"
    assert "protocol_sha256" in raw, "判定脚本未校验 protocol 哈希"
    # 且必须真的在读文件 (不是把阈值抄进脚本)
    assert "read_text" in code_only(rel), "判定脚本没有读取冻结件的实际 IO"


def test_b6_bootstrap_unit_is_seed_level(b6_config, b6_protocol_hash):
    """标准约束: 禁止用时间点 / 轨迹条数当独立样本做显著性。"""
    cb = b6_config["b6"]["bootstrap"]
    for b in (cb, b6_protocol_hash["bootstrap"]):
        assert str(b["unit"]) == "seed_level_paired_gain"
        assert int(b["n"]) == 5
        assert int(b["samples"]) >= 5000
        assert int(b["seed"]) == 20260814
    # 两条禁令只在 config 侧声明 (冻结件只留可复算参数)
    assert bool(cb["forbid_endpoint_bootstrap"]) is True
    assert bool(cb["forbid_trajectory_count_significance"]) is True


def test_b6_no_global_seed_anywhere():
    """标准约束: 禁止全局 np.random.seed()。"""
    for rel in B6_SCRIPTS:
        src = code_nospace(rel)
        assert "np.random.seed(" not in src, f"{rel} 使用了全局 np.random.seed"
        # torch.manual_seed 用于模型初始化 (§11 要求同 cell 三组初始权重同一),
        # 与"禁止全局 np.random.seed"是两回事; 但 DataLoader 取样必须走局部
        # generator, 否则三组的 batch 顺序无法对齐。
        if "torch.manual_seed(" in src:
            assert ("generator" in src.lower()
                    or "reset_target_batch_order" in src), \
                f"{rel} 用了全局 torch.manual_seed 却没有局部 batch generator"


def test_b6_oracle_methods_excluded_from_recommendation(b6_config):
    """§19: const / persistence / true-rul 不得进入可部署方法排名。"""
    er = b6_config["b6"]["engineering_recommendation"]
    assert bool(er["single_recommendation"]) is True
    assert list(er["candidates"]) == list(TRAINED) + ["damage_extrapolation"]
    ex = set(er["excluded_oracle_methods"])
    assert {"const_mean_info", "persistence", "true_rul_lookup"} <= ex
    assert not (set(er["candidates"]) & ex), "候选里混入了 oracle 方法"
    assert bool(er["priority_frozen"]) is True
    assert list(er["priority_order"]) == [
        "primary_test_info_macro_rmse", "warning_miss",
        "catastrophic_rate", "simplicity_deployability"]


def test_b6_no_nan_to_zero_in_scripts():
    """标准约束: 无 NaN 被悄悄转成 0。"""
    for rel in B6_SCRIPTS:
        src = code_nospace(rel)
        for bad in ("np.nan_to_num", "nan_to_num(", "fillna(0"):
            assert bad.replace(" ", "") not in src, f"{rel} 出现 NaN->0: {bad}"


def test_b6_no_hardcoded_gate_numbers_in_verdict():
    """标准约束: 阈值走 config/冻结件, 判定脚本里不得硬编码。"""
    src = code_nospace("scripts/basilisk_b6/final_transfer_verdict.py")
    for bad in ("0.02", "0.05"):
        assert bad not in src, f"判定脚本硬编码了阈值 {bad} —— 必须走冻结件"


def test_b6_exit_rules_block_b7_b8(b6_config):
    """§26: B6 到此停止 —— 不得自动跑 B7 / B8, 不得改 split / HI / RUL。"""
    ex = b6_config["b6"]["exit_rules"]
    for k in ("stop_after_b6", "forbid_auto_run_b7", "forbid_auto_run_b8",
              "forbid_new_algorithms", "forbid_tuning_mmd",
              "forbid_changing_split", "forbid_changing_hi_rul"):
        assert bool(ex[k]) is True, f"exit_rules.{k} 必须为 true"
    assert list(ex["next_stage_allowed_only"]) == [
        "B7: figures + report", "B8: packaging + Docker"]
    for rel in B6_SCRIPTS:
        src = code_nospace(rel)
        for bad in ("basilisk_b7", "basilisk_b8", "run_b7", "run_b8"):
            assert bad not in src or "must_stay_absent" in src, \
                f"{rel} 触碰了 B7/B8"


def test_b6_last_stage_for_core_numbers(b6_protocol_hash):
    """§0: B6 是飞轮线最后一个允许产生新核心实验数字的阶段。"""
    assert bool(
        b6_protocol_hash["last_stage_allowed_to_produce_core_numbers"]) is True
    b5 = b6_protocol_hash["b5_frozen_conclusion"]
    assert str(b5["verdict"]) == "B5_NO_POSITIVE_TRANSFER"
    assert str(b5["engineering_recommendation"]) == "damage_extrapolation"
    assert bool(b5["must_not_be_overwritten"]) is True
