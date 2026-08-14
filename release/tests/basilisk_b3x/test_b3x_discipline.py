"""tests/basilisk_b3x/test_b3x_discipline.py

BASILISK-B3X §19 —— 纪律测试。

`POST_B2_FAIL_EXPLORATORY` / `NOT_FORMAL_EVIDENCE`

§19 点名要求的 7 个测试:
    test_b3x_old_b2_unchanged
    test_b3x_same_split
    test_mission_features_prefix_safe
    test_no_future_mission_features
    test_same_rng_between_arms
    test_short_eol_subset_fixed
    test_b3x_not_formal_verdict
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

# B2 namespace —— B3X 全程不得触碰。
B2_PROTECTED = (
    "configs/wheel_basilisk_b2.yaml",
    "checkpoints/basilisk_b2/split.json",
    "checkpoints/basilisk_b2/metrics.json",
    "checkpoints/basilisk_b2/protocol_hash.json",
    "docs/basilisk_b2/protocol.md",
    "docs/basilisk_b2/results.md",
    "STATUS_BASILISK_B2.md",
)

# §9 禁止使用的正式标签。以字符片段拼装, 避免在源码里留下可被误读为
# "本阶段给出了该结论"的完整符号。
_FORBIDDEN_FORMAL = "B3_MISSION_" + "FEATURES_USEFUL"


def _sha(rel: str) -> str:
    p = ROOT / rel
    return (hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file()
            else "MISSING")


# ---------------------------------------------------------------- §2 / §0
def test_b3x_old_b2_unchanged(b3x_baseline):
    """B2 的全部产物必须与 B3X 契约首次写入时逐字一致。"""
    flat = b3x_baseline["flat_sha256"]
    checked = 0
    for rel in B2_PROTECTED:
        assert rel in flat, f"{rel} 未进 B3X 契约 —— 无法证明它没被改"
        assert flat[rel] != "MISSING", f"{rel} 在契约里是 MISSING"
        assert _sha(rel) == flat[rel], f"B2 产物被修改: {rel}"
        checked += 1
    assert checked == len(B2_PROTECTED)


def test_b3x_old_b2_verdict_and_ids_pinned(b3x_baseline, b3x_b2_split):
    """§2: B2 verdict / split hash / train·val·test ID 哈希都在契约里。"""
    exp = b3x_baseline["b2_expected"]
    assert exp["b2_verdict"] == "B2_GENERALIZATION_FAIL"
    assert exp["b2_split_sha256"] == b3x_b2_split["split_sha256"]
    for k in ("b2_train_ids_sha256", "b2_val_ids_sha256",
              "b2_test_ids_sha256"):
        assert len(exp[k]) == 64, f"{k} 不是 sha256"
        assert "PLACEHOLDER" not in exp[k]
    # 断言具体路径被 pin, 而不是断言组名 —— 组名可以重排, 路径不能漏。
    flat = {r: h for g in b3x_baseline["groups"].values() for r, h in g.items()}
    for rel in ("configs/wheel_basilisk_b2.yaml",
                "checkpoints/basilisk_b2/split.json",
                "checkpoints/basilisk_b2/metrics.json",
                "checkpoints/basilisk_b2/protocol_hash.json",
                "docs/basilisk_b2/protocol.md",
                "scripts/basilisk_b2/run_gate.py"):
        assert rel in flat, f"{rel} 未进契约"
        assert flat[rel] != "MISSING", rel


def test_b3x_formal_b3_artifacts_absent(b3x_baseline):
    """§9: 正式 B3 标签类产物必须始终缺席。"""
    g = b3x_baseline["groups"]["b3x_must_stay_absent"]
    assert g, "absent 组不能为空"
    for rel, h in g.items():
        assert h == "MISSING", f"{rel} 不应存在 —— 探索性结论被提升成正式 claim"
        assert not (ROOT / rel).exists(), rel


# ---------------------------------------------------------------- §4
def test_b3x_same_split(b3x_config, b3x_b2_split, b3x_ablation):
    """§4: 完全复用 B2 划分, 不重新 stratify, test 分布不变。"""
    b3 = b3x_config["b3x"]
    assert b3["split"]["reuse_from"] == "checkpoints/basilisk_b2/split.json"
    assert b3["split"]["forbid_restratify"] is True
    assert (str(b3["split"]["expected_sha256"])
            == str(b3x_b2_split["split_sha256"]))
    # 跑出来的产物记录的划分必须是同一个
    assert b3x_ablation["split_sha256"] == b3x_b2_split["split_sha256"]
    assert (b3x_ablation["split_reused_from"]
            == "checkpoints/basilisk_b2/split.json")
    s = b3x_b2_split["splits"]
    assert (len(s["train"]["tids"]), len(s["val"]["tids"]),
            len(s["test"]["tids"])) == (44, 31, 75)


def test_b3x_loader_rejects_foreign_split(b3x_config, b3x_data_mod,
                                          b3x_b2_split):
    """篡改 split_sha256 必须被 loader 拒绝, 不是靠口头承诺。"""
    bad = json.loads(json.dumps(b3x_b2_split))
    bad["split_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="B3X_SPLIT_MISMATCH"):
        b3x_data_mod.prepare_b3x(b3x_config, bad, 72, "core_only",
                                 verbose=False)


# ---------------------------------------------------------------- §3
def test_mission_features_prefix_safe(b3x_config, b3x_data_mod,
                                      b3x_feature_h5):
    """§3: mission features 必须 causal —— 截断轨迹重算, 前缀逐值相等。

    这不是"看代码相信它是 causal", 而是真的把轨迹砍短再算一遍。
    """
    cfg = b3x_config
    npw = b3x_data_mod.resolve_n_per_window(cfg)
    cols = b3x_data_mod.arm_mission_cols(cfg, "core_plus_mission")
    assert cols, "B 臂 mission 列不能为空"

    import h5py
    with h5py.File(b3x_feature_h5, "r") as f:
        names = sorted(f.keys())[:3]
        n_rows = {t: int(f[t]["x_T"].shape[0]) for t in names}

    full = b3x_data_mod.mission_matrix(b3x_feature_h5, cols, npw,
                                       tid_names=names)
    # 每条砍掉后 40%
    lim = {t: max(8, int(n * 0.6)) for t, n in n_rows.items()}
    trunc = b3x_data_mod.mission_matrix(b3x_feature_h5, cols, npw,
                                        tid_names=names, n_rows_limit=lim)
    # 逐轨迹比较前缀
    off_f = off_t = 0
    for t in names:
        n, k = n_rows[t], lim[t]
        a = full[off_f:off_f + n][:k]
        b = trunc[off_t:off_t + k]
        assert a.shape == b.shape
        assert np.array_equal(a, b), (
            f"{t}: 截断后重算的 mission 特征与全长前缀不一致 —— 非 causal")
        off_f += n
        off_t += k
    assert off_t == sum(lim.values())


def test_no_future_mission_features(b3x_config, b3x_data_mod, b3x_config_raw):
    """§3: 禁止 future / lifetime / EOL / 标签派生列进入任何一臂。"""
    cfg = b3x_config
    banned = set(b3x_data_mod.FORBIDDEN_MISSION_COLS) | set(
        b3x_data_mod.FORBIDDEN_PLAIN_INPUT_COLS)
    for arm in cfg["b3x"]["arms"]:
        cols = b3x_data_mod.arm_mission_cols(cfg, arm)
        assert not (set(cols) & banned), f"{arm} 含禁止列"
        for c in cols:
            assert "future" not in c and "lifetime" not in c
            assert "eol" not in c.lower() and "rul" not in c.lower()
    # config 里必须显式列出禁止项 (不是只靠代码常量)
    fb = set(str(x) for x in cfg["b3x"]["forbidden_features"])
    for must in ("mode_id", "future_mode_schedule", "lifetime_mode_fraction",
                 "eol_idx", "rul", "hi_damage_obs"):
        assert must in fb, f"config forbidden_features 缺 {must}"
    # 篡改配置塞入禁止列必须被拒
    import copy
    bad = copy.deepcopy(cfg)
    bad["b3x"]["arms"]["core_plus_mission"]["mission_features"] = ["mode_id"]
    with pytest.raises(AssertionError, match="禁止输入列"):
        b3x_data_mod.arm_mission_cols(bad, "core_plus_mission")


def test_zero_crossing_rate_is_derived_not_fabricated(b3x_config,
                                                      b3x_data_mod,
                                                      b3x_feature_h5):
    """§3: h5 里只有 count; rate 必须 = count / n_per_window, 量纲不能瞎凑。"""
    import h5py
    npw = b3x_data_mod.resolve_n_per_window(b3x_config)
    assert npw == int(round(
        float(b3x_config["sim"]["sample_period_s"])
        / float(b3x_config["sim"]["profile"]["dt_s"])))
    with h5py.File(b3x_feature_h5, "r") as f:
        mg = f[sorted(f.keys())[0]]["mission_features"]
        assert b3x_data_mod.DERIVED_RATE_COL not in mg, (
            "h5 里居然有 rate 列 —— 派生逻辑需重新确认")
        assert b3x_data_mod.DERIVED_RATE_FROM in mg
        cnt = np.asarray(mg[b3x_data_mod.DERIVED_RATE_FROM][:32], np.float64)
    got = b3x_data_mod.mission_matrix(
        b3x_feature_h5, [b3x_data_mod.DERIVED_RATE_COL], npw,
        tid_names=[sorted(h5py.File(b3x_feature_h5, "r").keys())[0]])[:32, 0]
    assert np.allclose(got, cnt / npw, rtol=1e-6, atol=1e-12)


# ---------------------------------------------------------------- §6
def test_same_rng_between_arms(b3x_config, b3x_data_mod, b3x_b2_split):
    """§6: 两臂在同一 seed 下 batch 顺序 / 划分 / rul_scale 必须相同。

    唯一允许的差异是输入列数。**不断言权重逐值相同** —— 改变输入维度
    会平移 RNG 的消耗位置, 那是输入消融的固有后果, 断言它相同就是
    在断言一件假的事。
    """
    seed = 72
    a = b3x_data_mod.prepare_b3x(b3x_config, b3x_b2_split, seed,
                                 "core_only", with_test=True, verbose=False)
    b = b3x_data_mod.prepare_b3x(b3x_config, b3x_b2_split, seed,
                                 "core_plus_mission", with_test=True,
                                 verbose=False)
    assert a["rul_scale"] == b["rul_scale"]
    assert a["censor_eta"] == b["censor_eta"]
    assert a["cap_eps"] == b["cap_eps"]
    assert a["hi_key"] == b["hi_key"]
    for k in ("tr", "va", "te"):
        assert np.array_equal(a[k], b[k]), f"{k} 划分不一致"
    assert (a["batch_order_generator_seed"]
            == b["batch_order_generator_seed"])
    assert len(a["ltr"].dataset) == len(b["ltr"].dataset)
    assert len(a["lte"].dataset) == len(b["lte"].dataset)

    def order(loader):
        return np.concatenate([np.asarray(t[3]).ravel() for t in loader])

    oa, ob = order(a["ltr"]), order(b["ltr"])
    assert oa.shape == ob.shape
    assert np.array_equal(oa, ob), "两臂 batch 顺序不一致 —— §6 公平性被破坏"
    # 唯一差异
    assert b["n_target"] - a["n_target"] == len(b["mission_cols"]) == 6
    assert a["core_cols"] == b["core_cols"]


def test_arms_share_frozen_b2_hyperparams(b3x_config):
    """§0: 早停 / max_epochs / weight_decay / loss 权重必须仍是 B2 的值。"""
    t = b3x_config["training"]
    assert t["max_epochs"] == 8
    assert t["early_stop_patience"] == 2
    assert float(t["weight_decay"]) == 1e-3
    assert t["early_stop_metric"] == "info_macro_rmse"
    assert float(t["post_eol_weight"]) == 0.1
    assert float(t["capped_weight"]) == 0.1
    assert b3x_config["transfer"]["target_hi_key"] == "hi_damage_obs"
    f = b3x_config["b3x"]["fairness"]
    for k in ("same_split", "same_batch_order", "same_init_rng",
              "same_optimizer", "same_epochs", "same_early_stopping",
              "same_evaluation", "same_censoring_contract"):
        assert f[k] is True, k
    assert f["only_difference"] == "input_features"


# ---------------------------------------------------------------- §7
def test_short_eol_subset_fixed(b3x_config, b3x_baseline, b3x_b2_split,
                                b3x_short_eol):
    """§7: short-EOL 子集 = {19,109,111}, 写死在 config + 契约, 且确在 test 内。"""
    want = [19, 109, 111]
    assert [int(x) for x in b3x_config["b3x"]["short_eol_subset"]] == want
    assert b3x_baseline["b2_expected"]["b2_short_eol_diagnostic_ids"] == str(want)
    assert [int(x) for x in b3x_short_eol["short_eol_subset"]] == want
    # 与 normal 子集不重叠, 并集 = 全部 75 条 test
    normal = [int(x) for x in b3x_short_eol["normal_eol_subset"]]
    assert not set(normal) & set(want)
    assert len(normal) == 72
    assert b3x_short_eol["n_test"] == 75
    # tid 空间 = sorted(h5 keys) 下标, 三条都必须真在 test 里
    names = sorted(b3x_b2_split["splits"]["test"]["tids"])
    idx = {int(n.split("_")[1]) for n in names}
    assert set(want) <= idx, "short-EOL tid 不在 test 划分里"


def test_subsets_reported_separately_not_merged(b3x_short_eol):
    """§7: 两个子集必须分别出数, 空子集是 NaN + n_traj=0 而不是 0。"""
    for r in b3x_short_eol["per_seed"]:
        for arm in ("core_only", "core_plus_mission"):
            blocks = r[arm]
            assert set(("overall", "short_eol", "normal_eol")) <= set(blocks)
            for sub in ("overall", "short_eol", "normal_eol"):
                m = blocks[sub]
                if m["n_traj"] == 0:
                    assert m["rmse"] != m["rmse"], "空子集必须是 NaN, 不是 0"
                else:
                    assert m["n_endpoints"] > 0
            # short + normal 的轨迹数不应超过 overall
            assert (blocks["short_eol"]["n_traj"]
                    + blocks["normal_eol"]["n_traj"]
                    <= blocks["overall"]["n_traj"] + 1)
    assert b3x_short_eol["macro_unit"] == "trajectory_not_endpoint"


# ---------------------------------------------------------------- §9
def test_b3x_not_formal_verdict(b3x_config, b3x_summary, b3x_summ_mod):
    """§9: 只允许两个探索性标签; 正式标签禁用; B2 判定不被覆盖。"""
    assert set(b3x_summ_mod.ALLOWED_LABELS) == {
        "B3X_STABILIZING_SIGNAL", "B3X_NO_STABILIZING_SIGNAL"}
    v = b3x_summary["verdict"]
    assert v in b3x_summ_mod.ALLOWED_LABELS, f"非法 verdict {v}"
    assert b3x_summary["label_discipline"] == "POST_B2_FAIL_EXPLORATORY"
    assert b3x_summary["formal_stage"] is False
    assert b3x_summary["not_formal_evidence"] is True
    assert b3x_summary["b2_verdict_unchanged"] == "B2_GENERALIZATION_FAIL"
    assert (b3x_summary["b2_verdict_status"]
            == "B2_GENERALIZATION_FAIL_REMAINS_FINAL")
    assert _FORBIDDEN_FORMAL in b3x_summary["forbidden_formal_label"]
    assert b3x_config["b3x"]["verdict"]["forbid_formal_label"] is True
    assert b3x_config["b3x"]["formal_stage"] is False
    # 探索性标签不得出现在任何正式结论文件里
    for rel in ("STATUS_BASILISK_B2.md", "docs/basilisk_b2/results.md"):
        txt = (ROOT / rel).read_text(encoding="utf-8")
        assert "B3X" not in txt, f"{rel} 被 B3X 污染"


def test_b3x_thresholds_frozen_before_run(b3x_config, b3x_summary,
                                          b3x_b2_metrics):
    """§9: 四条判据的阈值必须来自 protocol/config, 且"恢复"阈值取自 B2 既有事实。"""
    vc = b3x_config["b3x"]["verdict"]
    d = b3x_summary["decision"]
    assert d["thresholds_frozen_before_run"] is True
    assert d["n_conditions"] == 4 and d["all_required"] is True
    assert int(vc["min_short_eol_improved_seeds"]) == 3
    assert int(vc["min_collapsed_recovered"]) == 2
    # 恢复阈值 = B2 中最差正常 seed 的 overall RMSE (seed 75), 逐值核对
    per = {int(r["seed"]): float(r["paired"]["target_only"])
           for r in b3x_b2_metrics["per_seed"]}
    normal = [int(s) for s in b3x_config["b3x"]["normal_seeds"]]
    assert (abs(float(vc["recovered_threshold_value"])
                - max(per[s] for s in normal)) < 1e-12), (
        "recovered_threshold_value 必须等于 B2 正常 seed 的最差值")
    # 该阈值绝不能取自 B3X 自己的结果
    b3x_rmse = [r["overall"]["rmse_mission"] for r in b3x_summary["per_seed"]]
    assert float(vc["recovered_threshold_value"]) not in set(b3x_rmse)


def test_b3x_verdict_recomputes_from_metrics(b3x_summary):
    """verdict 必须能由记录的判据重算出来 —— 不是手写进 JSON 的。"""
    conds = b3x_summary["decision"]["conditions"]
    n_pass = sum(1 for c in conds if c["passed"])
    assert n_pass == b3x_summary["decision"]["n_passed"]
    expect = ("B3X_STABILIZING_SIGNAL" if n_pass == len(conds)
              else "B3X_NO_STABILIZING_SIGNAL")
    assert b3x_summary["verdict"] == expect


def test_b3x_b4x_schema_rule_written_before_b4x(b3x_summary):
    """§10: B4X 输入 schema 规则必须由 verdict 唯一决定, 不能挑好看的。"""
    rule = b3x_summary["b4x_input_schema_rule"]
    assert rule == {"B3X_STABILIZING_SIGNAL": "core_plus_mission",
                    "B3X_NO_STABILIZING_SIGNAL": "core_only"}
    assert b3x_summary["b4x_input_schema"] == rule[b3x_summary["verdict"]]
    assert b3x_summary["b4x_allowed_regardless"] is True
    # 规则必须写在 protocol 里 (跑 B4X 之前)
    txt = (ROOT / "docs" / "basilisk_b3x" / "protocol.md").read_text(
        encoding="utf-8")
    assert "B4X" in txt and "core_plus_mission" in txt


def test_b3x_gain_not_compared_against_b2_table(b3x_ablation, b3x_summary):
    """CORE_ONLY 不是 B2 的复现; gain 只在 B3X 两臂之间算。必须写明, 不得含糊。"""
    for d in (b3x_ablation, b3x_summary):
        note = str(d["comparability_note"])
        assert "B2" in note and "batch" in note
    for r in b3x_ablation["per_seed"]:
        g = r["gain_mission"]
        assert (g["definition"]
                == "RMSE_core_only - RMSE_core_plus_mission")
        assert abs((g["core_only"] - g["core_plus_mission"]) - g["gain"]) < 1e-12


def test_b3x_not_fast_smoke(b3x_ablation):
    """--fast 冒烟产物不得成为报告依据。"""
    assert b3x_ablation["fast"] is False
    assert b3x_ablation["seeds"] == [72, 73, 74, 75, 76]
    assert list(b3x_ablation["arms"]) == ["core_only", "core_plus_mission"]
