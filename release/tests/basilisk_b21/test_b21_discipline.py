"""tests/basilisk_b21/test_b21_discipline.py —— B2.1 纪律不变量。

`SPLIT_COVERAGE_ROBUSTNESS`

本阶段只改**划分协议**。这些测试守住三件事:
  1. 划分协议本身满足 §2..§5 的全部要求 (含 min-EOL 序关系与删失分层);
  2. §7 的"不得用 test 表现挑划分"是结构性的, 不是口头承诺;
  3. B2 / B3X / B4X 的旧结论与旧产物不被本阶段改写或重新解释,
     且 B5 绝不被自动运行。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

SPLITTER_REL = "scripts/basilisk_b21/build_b21_split.py"
GATE_REL = "scripts/basilisk_b21/run_b21_gate.py"
SUM_REL = "scripts/basilisk_b21/summarize_b21.py"


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _code_only(rel: str) -> str:
    """去掉注释与字符串字面量后的源码。

    禁止性检查必须看**真实代码**, 不能看文本: 明令禁止某个 token 的注释里
    必然出现那个 token, 按原文 grep 会把"写明禁止"误判成"违反禁止"。
    """
    import io
    import tokenize
    out = []
    with io.open(ROOT / rel, encoding="utf-8") as f:
        for tok in tokenize.generate_tokens(f.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# §2..§5 划分协议
# ---------------------------------------------------------------------------
def test_b21_split_is_valid_and_trajectory_level(b21_split):
    assert b21_split["verdict"] == "B21_SPLIT_VALID"
    assert b21_split["level"] == "trajectory"
    assert b21_split["coverage"]["n_passed"] == b21_split["coverage"]["n_checks"]
    assert b21_split["coverage"]["all_passed"] is True


def test_b21_no_trajectory_leakage(b21_split):
    """§5 要求 1: 无重叠 + 并集覆盖全部轨迹。轨迹级, 杜绝同一退化轨迹泄漏。"""
    s = {k: set(b21_split["splits"][k]["tids"]) for k in ("train", "val", "test")}
    assert not (s["train"] & s["val"])
    assert not (s["train"] & s["test"])
    assert not (s["val"] & s["test"])
    assert sum(len(v) for v in s.values()) == b21_split["n_traj"]
    assert len(s["train"] | s["val"] | s["test"]) == b21_split["n_traj"]


def test_b21_every_event_lifetime_bin_in_all_splits(b21_split):
    """§5 要求 2: short/medium/long 三档在 train/val/test 都有代表。"""
    names = set(b21_split["lifetime_bins"]["event"]["names"])
    for k in ("train", "val", "test"):
        assert names <= set(b21_split["splits"][k]["event_bins_present"]), k


def test_b21_min_event_eol_ordering(b21_split):
    """§5 要求 3/4 —— 本阶段存在的核心理由。

    train / val 的最短 event 寿命必须 <= test 的最短 event 寿命, 否则失效区
    又一次落在训练信号与模型选择信号之外, lifetime-support mismatch 没修好。
    """
    m = b21_split["coverage"]["min_event_eol"]
    assert m["train"] <= m["test"], m
    assert m["val"] <= m["test"], m


def test_b21_fixes_the_b2_mismatch(b21_split, b2_split_ro):
    """B2 的缺陷确实存在, 且 B2.1 确实修好了它 (对照, 不改 B2 任何产物)。"""
    import h5py
    h5 = ROOT / b21_split["feature_h5"]
    eol = {}
    with h5py.File(h5, "r") as f:
        for k in sorted(f.keys()):
            if bool(int(f[k].attrs["event_observed"])):
                eol[k] = int(f[k].attrs["eol_idx"])

    def _min(sp, name):
        v = [eol[t] for t in sp["splits"][name]["tids"] if t in eol]
        return min(v)

    # B2: 缺陷存在
    assert _min(b2_split_ro, "test") < _min(b2_split_ro, "train")
    assert _min(b2_split_ro, "test") < _min(b2_split_ro, "val")
    # B2.1: 缺陷消失
    assert _min(b21_split, "test") >= _min(b21_split, "train")
    assert _min(b21_split, "test") >= _min(b21_split, "val")


def test_b21_train_val_contain_short_life(b21_split):
    """§5 要求 5: train 与 val 都必须含 short-life event 轨迹。"""
    short = b21_split["lifetime_bins"]["event"]["names"][0]
    for k in ("train", "val"):
        assert short in b21_split["splits"][k]["event_bins_present"], k
        assert any(s.startswith(f"event/{short}")
                   for s in b21_split["splits"][k]["strata"]), k


def test_b21_censored_present_in_all_splits(b21_split):
    """§5 要求 6: 删失比例在三个 split 都有体现。"""
    for k in ("train", "val", "test"):
        assert b21_split["splits"][k]["n_censored"] > 0, k
        assert b21_split["splits"][k]["censored_frac"] > 0.0, k


def test_b21_censored_never_given_eol(b21_split):
    """§3: 右删失轨迹绝不被赋予 EOL。

    结构性检查: 删失侧的分层键必须是 observed_duration, 且 manifest 里
    删失轨迹只有 obs-dur 范围, 没有任何 EOL 字段被写到删失身上。
    """
    cen = b21_split["lifetime_bins"]["censored"]
    assert cen["key"] == "observed_duration"
    assert cen["forbid_fake_eol"] is True
    assert "eol" not in cen["key"]
    for k in ("train", "val", "test"):
        s = b21_split["splits"][k]
        # event EOL 范围只在 event 轨迹上定义; 删失只有 obs-dur 范围
        assert s["censored_obs_dur_min"] is not None
        assert "censored_eol_min" not in s
        assert "censored_eol_max" not in s
    # 源码层面: 删失分箱用的是 obs_dur, 不是 eol
    src = _src(SPLITTER_REL)
    assert 'np.where(~ev, meta["obs_dur"], np.nan)' in src
    assert 'np.where(ev, meta["eol"], np.nan)' in src


def test_b21_censored_degenerate_reported_honestly(b21_split):
    """删失观测时长若全部相同, 三分位退化 —— 必须如实报告, 不得编造边界。"""
    cen = b21_split["lifetime_bins"]["censored"]
    n_eff = int(cen["n_effective_bins"])
    assert cen["degenerate"] is bool(n_eff < len(cen["names"]))
    # 非空 bin 的计数之和必须等于删失总数 (没有轨迹被丢掉或重复计数)
    assert sum(cen["counts"].values()) == b21_split["n_censored"]
    if cen["degenerate"]:
        assert "degenerate_note" in cen and cen["degenerate_note"]


def test_b21_bins_are_preregistered_dataset_level_terciles(b21_split,
                                                           b21_config):
    """§2: 分位是数据集级、在训练之前算定的三分位。edges 必须可复算。"""
    import h5py
    sc = b21_config["b21"]["split"]
    assert sc["eol_use"] == "split_stratification_only"
    q = list(sc["event"]["quantiles"])
    assert len(q) == 2 and abs(q[0] - 1 / 3) < 1e-9 and abs(q[1] - 2 / 3) < 1e-9
    h5 = ROOT / b21_split["feature_h5"]
    ev_eol = []
    with h5py.File(h5, "r") as f:
        for k in sorted(f.keys()):
            if bool(int(f[k].attrs["event_observed"])):
                ev_eol.append(float(f[k].attrs["eol_idx"]))
    want = [float(np.quantile(ev_eol, x)) for x in q]
    assert np.allclose(b21_split["lifetime_bins"]["event"]["edges"], want)
    assert (b21_split["lifetime_bins"]["event"]["quantile_basis"]
            == "dataset_level_event_eol_before_training")


def test_b21_joint_stratification_recorded(b21_split, b21_config):
    """§4: 联合分层 = event/censored x lifetime bin, 且逐 split 有层格计数。"""
    assert list(b21_split["joint_stratify_by"]) == ["event_censored",
                                                    "lifetime_bin"]
    for k in ("train", "val", "test"):
        strata = b21_split["splits"][k]["strata"]
        assert strata, k
        assert sum(strata.values()) == b21_split["splits"][k]["n"]
        assert any(s.startswith("event/") for s in strata), k
        assert any(s.startswith("censored/") for s in strata), k


def test_b21_split_ratios_close_to_30_20_50(b21_split):
    n = b21_split["n_traj"]
    got = {k: b21_split["splits"][k]["n"] / n for k in ("train", "val", "test")}
    for k, want in (("train", 0.30), ("val", 0.20), ("test", 0.50)):
        assert abs(got[k] - want) <= 0.03, (k, got)


def test_b21_split_hash_covers_ids_and_bins(b21_split):
    """split hash 必须覆盖 IDs 与分箱定义, 否则改了分箱哈希不变。"""
    import hashlib
    sc_edges = b21_split["lifetime_bins"]
    payload = json.dumps({
        "ratios": [b21_split["ratios"]["train"], b21_split["ratios"]["val"],
                   b21_split["ratios"]["test"]],
        "split_seed": b21_split["split_seed"],
        "level": b21_split["level"],
        "joint_stratify_by": list(b21_split["joint_stratify_by"]),
        "event_quantiles": list(sc_edges["event"]["quantiles"]),
        "censored_quantiles": list(sc_edges["censored"]["quantiles"]),
        "event_bin_edges": sc_edges["event"]["edges"],
        "censored_bin_edges": sc_edges["censored"]["edges"],
        "short_bin_anchor_order": list(b21_split["short_bin_anchor_order"]),
        "train": b21_split["splits"]["train"]["tids"],
        "val": b21_split["splits"]["val"]["tids"],
        "test": b21_split["splits"]["test"]["tids"],
    }, sort_keys=True, ensure_ascii=False)
    assert (hashlib.sha256(payload.encode("utf-8")).hexdigest()
            == b21_split["split_sha256"])


def test_b21_manifest_has_required_fields(b21_split):
    """§6: manifest 必须含 IDs / strata / counts / EOL ranges / split hash。"""
    assert b21_split["split_sha256"]
    for k in ("train", "val", "test"):
        s = b21_split["splits"][k]
        for key in ("tids", "tid_indices", "strata", "n", "n_event",
                    "n_censored", "event_eol_min", "event_eol_max"):
            assert key in s, (k, key)
        assert len(s["tids"]) == s["n"] == len(s["tid_indices"])


def test_b21_split_differs_from_b2_split(b21_split, b2_split_ro):
    """新划分必须是新的; 若哈希与 B2 相同说明什么都没改。"""
    assert b21_split["split_sha256"] != b2_split_ro["split_sha256"]
    assert b21_split["split_seed"] != b2_split_ro["split_seed"]


# ---------------------------------------------------------------------------
# §7 不得用 test 表现挑划分 —— 结构性保证
# ---------------------------------------------------------------------------
def test_b21_splitter_cannot_see_test_performance():
    """划分脚本不得 import 模型/评估模块, 不得读 metrics 文件。"""
    src = _src(SPLITTER_REL)
    tree = ast.parse(src)
    mods = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods += [x.name for x in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.append(n.module)
    bad = [m for m in mods
           if m.startswith(("torch", "src.models", "src.transfer",
                            "src.experiments", "src.baselines"))
           or "eval" in m or "metrics" in m or "train" in m]
    assert not bad, f"划分脚本不得依赖 {bad}"
    for token in ("metrics.json", "gate_metrics", "info_macro_rmse"):
        assert token not in src, token


def test_b21_splitter_has_no_seed_search_cli():
    """CLI 结构上不提供多 seed 择优入口 (按 argparse 实参检查, 不按原文 grep)。"""
    tree = ast.parse(_src(SPLITTER_REL))
    flags = [n.args[0].value for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "add_argument"
             and n.args and isinstance(n.args[0], ast.Constant)]
    assert flags == ["--config"], flags
    assert "forbid_seed_search" in _src(SPLITTER_REL)


def test_b21_invalid_split_reported_not_reseeded(b21_split, b21_config,
                                                 b21_splitter):
    """§7: 覆盖失败 -> B21_SPLIT_INVALID, 不是换 seed 重试。

    用一个刻意破坏 min-EOL 序关系的假划分喂进 coverage_checks, 必须判 FAIL,
    且 build_split 里没有任何重试循环。
    """
    assert b21_splitter.INVALID_LABEL == "B21_SPLIT_INVALID"
    assert b21_config["b21"]["coverage_requirements"]["invalid_label"] \
        == "B21_SPLIT_INVALID"
    assert b21_config["b21"]["coverage_requirements"]["forbid_seed_search"] \
        is True
    # build_split 的函数体内不得出现 while / for-retry 形式的重试
    fn = next(n for n in ast.parse(_src(SPLITTER_REL)).body
              if isinstance(n, ast.FunctionDef) and n.name == "build_split")
    assert not [n for n in ast.walk(fn) if isinstance(n, ast.While)], \
        "build_split 不得有重试循环"

    # 破坏性构造: 把 test 里最短的 event 轨迹与 train 交换, 使序关系失效
    meta = b21_splitter.read_traj_meta(ROOT / b21_split["feature_h5"])
    sc = b21_config["b21"]["split"]
    built = b21_splitter.build_split(meta, sc)
    ok = b21_splitter.coverage_checks(meta, built, sc,
                                      b21_config["b21"]
                                      ["coverage_requirements"])
    assert ok["all_passed"] is True
    bad = {k: list(v) for k, v in built["assign"].items()}
    ev, eol = meta["event"], meta["eol"]
    tr_ev = [i for i in bad["train"] if ev[i]]
    lo = min(tr_ev, key=lambda i: eol[i])
    bad["train"].remove(lo)
    bad["test"].append(lo)
    res = b21_splitter.coverage_checks(
        meta, {**built, "assign": bad}, sc,
        b21_config["b21"]["coverage_requirements"])
    assert res["all_passed"] is False
    assert any(c["id"] == 3 and not c["pass"] for c in res["checks"])


def test_b21_short_bin_anchor_rule_is_coverage_driven(b21_split, b21_config):
    """§5.1 锚定规则必须按 EOL 升序, 且与任何性能指标无关。"""
    order = list(b21_config["b21"]["split"]["short_bin_anchor_order"])
    assert order == ["train", "val", "test"]
    anchors = b21_split["short_bin_anchors"]
    assert set(anchors) == {"train", "val", "test"}
    import h5py
    with h5py.File(ROOT / b21_split["feature_h5"], "r") as f:
        e = {k: int(f[anchors[k]].attrs["eol_idx"]) for k in order}
    assert e["train"] <= e["val"] <= e["test"], e
    # 每个锚定轨迹确实落在对应 split
    for k in order:
        assert anchors[k] in b21_split["splits"][k]["tids"]


def test_b21_no_global_numpy_seed():
    for rel in (SPLITTER_REL, GATE_REL, SUM_REL):
        code = _code_only(rel)
        assert "np.random.seed" not in code, rel
        assert "torch.manual_seed" not in code, rel
    assert "default_rng" in _code_only(SPLITTER_REL)


# ---------------------------------------------------------------------------
# §8/§9 训练配置与方法范围
# ---------------------------------------------------------------------------
def test_b21_reuses_frozen_b2_training(b21_gate):
    """§8: 逐项复用 B2 冻结的 Target-only 训练配置。"""
    ft = b21_gate["frozen_training"]
    assert ft["early_stop_metric"] == "info_macro_rmse"
    assert int(ft["max_epochs"]) == 8
    assert int(ft["early_stop_patience"]) == 2
    assert abs(float(ft["weight_decay"]) - 1e-3) < 1e-12
    for r in b21_gate["per_seed"]:
        h = r["hyper"]
        assert h["early_stop_metric"] == "info_macro_rmse"
        assert int(h["max_epochs"]) == 8
        assert int(h["early_stop_patience"]) == 2
        assert abs(float(h["weight_decay"]) - 1e-3) < 1e-12


def test_b21_only_three_methods_no_transfer(b21_gate, b21_config):
    """§9: 只比三个方法, 不跑迁移。"""
    assert set(b21_gate["methods"]) == {"target_only", "const_mean_info",
                                        "damage_extrapolation"}
    assert b21_gate["transfer_run"] is False
    assert b21_config["b21"]["forbid_transfer"] is True
    src = _src(GATE_REL)
    for token in ("train_transfer", "source_finetune", "source_mmd",
                  "mmd_lambda", "source_checkpoint"):
        assert token not in src, token


def test_b21_seeds_are_new(b21_gate):
    """§9: 五个新 seed, 与 B2/B3X/B4X 的 seed 不重叠。"""
    assert list(b21_gate["seeds"]) == [102, 103, 104, 105, 106]
    old = {72, 73, 74, 75, 76, 92, 93, 94}
    assert not (set(b21_gate["seeds"]) & old)


def test_b21_not_fast_smoke(b21_gate):
    assert b21_gate["fast_smoke"] is False


def test_b21_checkpoint_selection_val_only(b21_gate):
    """§12 条件 7: checkpoint 只按 validation 选 (结构性)。"""
    for r in b21_gate["per_seed"]:
        assert r["checkpoint_selection"] == "val_only"
    src = _src(GATE_REL)
    # 训练调用里不得出现 test loader
    assert 'train_target_only(\n        cfg, data, seed' in src
    assert '"lte"' in src  # test 只交给 evaluator
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_seed")
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "train_target_only"]
    assert len(calls) == 1
    txt = ast.dump(calls[0])
    assert "lte" not in txt


# ---------------------------------------------------------------------------
# §10/§11 指标纪律
# ---------------------------------------------------------------------------
def test_b21_primary_metric_is_trajectory_macro(b21_gate):
    assert (b21_gate["decision"]["primary_metric"]
            == "test_info_trajectory_macro_rmse")
    for r in b21_gate["per_seed"]:
        assert r["paired"]["metric"] == "test_info_macro_rmse"


def test_b21_all_required_metrics_reported(b21_gate):
    """§11 指标集必须齐全, 逐 seed 都在。"""
    bins = ("short", "medium", "long")
    for r in b21_gate["per_seed"]:
        assert "macro_corr" in r["shape"] and "psr" in r["shape"]
        w = r["target_only"]["warning"]
        for k in ("warning_coverage", "miss_rate", "coverage_before_eol",
                  "miss_rate_before_eol", "false_alarm_rate"):
            assert k in w, k
        for b in bins:
            assert b in r["lifetime_subsets"]["target_only"], b
            assert "macro_rmse" in r["lifetime_subsets"]["target_only"][b]
        assert "violation_rate" in r["censored_lower_bound"]


def test_b21_empty_subset_is_nan_not_zero(b21_gate_mod):
    """空子集必须 NaN + n=0, 绝不写 0。"""
    fake = {"_raw": {"pred": np.array([0.5, 0.6]), "true": np.array([0.4, 0.3]),
                     "tids": np.array([1, 1])}}
    r = b21_gate_mod.subset_rmse(fake, [999], cap_eps=1e-6)
    assert np.isnan(r["macro_rmse"])
    assert r["n_traj_evaluable"] == 0
    assert r["macro_rmse"] != 0
    assert "no_evaluable" in r["reason"]


def test_b21_miss_rate_reported_not_hidden(b21_gate):
    """不得只统计成功检测而隐藏 miss rate; coverage 与 miss 必须互补。"""
    for r in b21_gate["per_seed"]:
        w = r["target_only"]["warning"]
        assert np.isfinite(float(w["miss_rate"]))
        assert abs(float(w["coverage_before_eol"])
                   + float(w["miss_rate_before_eol"]) - 1.0) < 1e-9
        assert r["warning_validity"]["complementary"] is True


def test_b21_censored_rul_metrics_stay_nan(b21_gate):
    """右删失轨迹没有真实 RUL -> RUL 指标恒为 NaN + n=0, 只报下界违反率。"""
    for r in b21_gate["per_seed"]:
        cen = r["target_only"]["censor_report"]["censored_test"]
        assert cen["info_macro_rmse"] is None or np.isnan(
            float(cen["info_macro_rmse"]))
        assert int(cen["n_evaluable_rul_points"]) == 0
        assert "never_fabricated" in cen["rul_metrics_reason"]
        assert int(cen["n_traj"]) > 0


def test_b21_catastrophic_threshold_from_validation_only(b21_gate):
    """§11.1: 阈值只来自 target_only 的 validation, 禁止 test 派生。"""
    assert (b21_gate["catastrophic_definition"]["threshold_source"]
            == "target_only_validation_median_trajectory_rmse")
    assert b21_gate["catastrophic_definition"][
        "forbid_test_derived_threshold"] is True
    for r in b21_gate["per_seed"]:
        t = r["catastrophic_threshold"]
        assert t["split"] == "validation"
        assert t["group"] == "target_only"
        assert t["derived_from_test"] is False
        assert int(t["n_val_traj_evaluable"]) > 0


def test_b21_catastrophic_count_none_when_unevaluable(b21_gate_mod):
    """不可评估就是不可评估: rate=NaN, count=None, 绝不写 0。"""
    r = b21_gate_mod.catastrophic_rate(np.array([]), np.array([]),
                                       np.array([], dtype=np.int64),
                                       1e-6, float("nan"))
    assert r["n_catastrophic"] is None
    assert np.isnan(r["catastrophic_error_rate"])


def test_b21_no_silent_nan_to_zero_in_source():
    for rel in (GATE_REL, SUM_REL):
        src = _src(rel)
        assert "nan_to_num" not in src, rel
        assert "fillna(0" not in src, rel


def test_b21_thresholds_from_config_not_hardcoded():
    """不得硬编码阈值 —— 全部走 config。"""
    src = _src(GATE_REL)
    for key in ("min_improve_count", "min_corr_count",
                "min_no_catastrophic_short_count", "bootstrap_samples",
                "bootstrap_seed", "multiplier",
                "max_short_life_catastrophic_per_seed"):
        assert key in src, key
    # 判定里不得出现裸阈值字面量
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "decide")
    nums = [n.value for n in ast.walk(fn)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool)]
    assert all(v in (0, 1, 2, 3, 4, 5, 6, 7) for v in nums), nums


# ---------------------------------------------------------------------------
# §12 判定纪律
# ---------------------------------------------------------------------------
def test_b21_bootstrap_unit_is_seed_not_endpoint(b21_gate):
    bs = b21_gate["decision"]["bootstrap"]
    assert bs["unit"] == "one_paired_difference_per_seed"
    assert int(bs["n"]) == len(b21_gate["seeds"])
    assert int(bs["n_rep"]) == 2000


def test_b21_verdict_recomputes_from_conditions(b21_gate, b21_summary):
    conds = b21_gate["decision"]["conditions"]
    n_pass = sum(bool(c["pass"]) for c in conds)
    want = ("B21_GENERALIZATION_PASS" if n_pass == len(conds)
            else "B21_GENERALIZATION_FAIL")
    assert b21_gate["verdict"] == want
    assert b21_summary["verdict"] == want
    assert b21_summary["decision"]["recomputed_from_conditions"] is True
    assert len(conds) == 7


def test_b21_verdict_label_allowed(b21_summary):
    assert b21_summary["verdict"] in ("B21_GENERALIZATION_PASS",
                                     "B21_GENERALIZATION_FAIL")


def test_b21_nphm_not_a_selection_metric():
    """nPHM / PH 不得成为模型选择指标。"""
    src = _src(GATE_REL)
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "decide")
    body = ast.dump(fn)
    for token in ("nphm", "macro_ph", "prognostic_horizon", "alpha_lambda"):
        assert token not in body.lower(), token


def test_b21_persistence_not_ranked(b21_gate):
    """persistence 不进入可部署方法排名。"""
    assert "persistence" not in [m.lower() for m in b21_gate["methods"]]
    ex = [m.lower() for m in b21_gate["decision"]["excluded_oracle_methods"]]
    assert ex, "必须显式记录被排除的 oracle 类方法"


# ---------------------------------------------------------------------------
# 旧结论冻结 + §13/§14 出口
# ---------------------------------------------------------------------------
def test_b21_old_verdicts_pinned_in_contract(b21_contract):
    r = b21_contract["b21_results"]
    assert r["b4x_verdict"] == "B4X_TRANSFER_STABILIZATION_SIGNAL"
    assert r["b4x_exploratory_only"] == "True"
    assert r["b4x_not_formal_transfer_evidence"] == "True"
    assert r["b4x_b2_verdict_unchanged"] == "B2_GENERALIZATION_FAIL"
    assert r["b3x_verdict_via_b4x"] == "B3X_NO_STABILIZING_SIGNAL"
    assert r["b4x_n_methods_passing_A_and_B"] == "0"
    ch = b21_contract["frozen_chain"]
    assert ch["BASILISK_B2"] == "B2_GENERALIZATION_FAIL"
    assert ch["BASILISK_B3X"] == "B3X_NO_STABILIZING_SIGNAL"
    assert ch["BASILISK_B4X"] == "B4X_TRANSFER_STABILIZATION_SIGNAL"


def test_b21_mismatch_evidence_pinned(b21_contract):
    """lifetime-support mismatch 的量化证据必须钉死, 不能被事后否认。"""
    r = b21_contract["b21_results"]
    assert r["b2_lifetime_support_mismatch"] == "True"
    assert int(r["b2_train_min_event_eol"]) > int(r["b2_test_min_event_eol"])
    assert int(r["b2_val_min_event_eol"]) > int(r["b2_test_min_event_eol"])
    assert int(r["b2_n_test_event_below_train_min"]) > 0


def test_b21_b2_b3x_b4x_artifacts_in_contract(b21_contract):
    """旧产物进契约 = 必须逐字不变。"""
    flat = b21_contract["flat_sha256"]
    for rel in ("checkpoints/basilisk_b2/split.json",
                "checkpoints/basilisk_b2/metrics.json",
                "checkpoints/basilisk_b3x/summary.json",
                "checkpoints/basilisk_b4x/summary.json",
                "checkpoints/basilisk_b4x/transfer_metrics.json",
                "configs/wheel_basilisk_b2.yaml",
                "STATUS_BASILISK_B4X.md"):
        assert rel in flat, rel
        assert flat[rel] != "MISSING", rel


def test_b21_own_artifacts_not_in_contract(b21_contract):
    """B2.1 自己的产物不进契约 (它们会变)。"""
    flat = b21_contract["flat_sha256"]
    for rel in ("configs/wheel_basilisk_b21.yaml",
                "docs/basilisk_b21/split_manifest.json",
                "checkpoints/basilisk_b21/gate_metrics.json",
                "checkpoints/basilisk_b21/summary.json"):
        assert rel not in flat, rel


def test_b21_b5_absent_and_never_auto_run(b21_contract, b21_summary):
    """§13/§14 原版 —— **已由 B7 §14 退役为生命周期迁移守卫, 保留为可追溯记录。**

    旧语义: "B2.1 不自动跑 B5" -> 断言 B5 产物永远缺席。
    为何 obsolete: B5 已获显式人工阶段授权并完成 (`B5_NO_POSITIVE_TRANSFER`),
    "永远缺席"的前提不再成立, 该断言只会永久失败, 且失败与 B2.1 自身正确性无关。

    本测试**不删除**。旧守卫真正要保护的不变量是"**B2.1 阶段自己没有偷跑 B5**"
    —— 这是一个关于 B2.1 的历史事实, 永久可查, 现改为直接断言它:
      1. B2.1 契约冻结时, B5 产物确实登记为 MISSING;
      2. B2.1 的 summary 明确记录未自动跑 B5、未跑迁移;
      3. B2.1 的汇总脚本源码里不含 basilisk_b5 / train_transfer 调用。
    此外, B2.1 阶段**绝不允许**产生的迁移结论产物 (formal_transfer_verdict /
    positive_transfer_proven) 仍然必须缺席 —— 那不是"下游阶段", 而是 B2.1
    越权自己下迁移结论, 永久禁止。

    运行期的下游阶段防护移交:
      tests/basilisk_b5/test_b5_downstream_stage_transition_is_governed
      tests/basilisk_b5/test_b5_undelegated_artifacts_still_absent
      tests/basilisk_b7/test_b7_stale_guard_retirement.py
    退役依据: docs/basilisk_b7/stale_guard_retirement.md
    """
    flat = b21_contract["flat_sha256"]

    # (1) 历史事实: 冻结 B2.1 契约时, 下游 B5 产物确实登记为 MISSING。
    for rel in ("checkpoints/basilisk_b5/metrics.json",
                "docs/basilisk_b5/results.md",
                "STATUS_BASILISK_B5.md"):
        assert flat[rel] == "MISSING", f"{rel} 在 B2.1 契约中本应登记为 MISSING"

    # (2) B2.1 越权下迁移结论 —— 与阶段推进无关, 永久禁止, 现在仍须缺席。
    for rel in ("checkpoints/basilisk_b21/formal_transfer_verdict.json",
                "checkpoints/basilisk_b21/positive_transfer_proven.json"):
        assert flat[rel] == "MISSING", rel
        assert not (ROOT / rel).exists(), \
            f"{rel} 存在 —— B2.1 不得自行产出迁移结论"

    # (3) B2.1 自身的行为记录: 没有自动跑 B5, 没有跑迁移。
    assert b21_summary["b5_auto_run"] is False
    assert b21_summary["forbid_auto_run_b5"] is True
    assert b21_summary["transfer_run"] is False
    code = _code_only(SUM_REL)
    assert "basilisk_b5" not in code
    assert "train_transfer" not in code


def test_b21_downstream_stages_are_governed_not_forbidden():
    """B7 §14: 下游阶段允许存在, 但必须治理齐备。

    这是替代"下游必须永远不存在"的新守卫。它比旧守卫更强: 旧守卫只能拦住
    "阶段推进"这件本身合法的事, 新守卫拦住的是真正的越界 —— 没有 protocol /
    baseline contract / final verdict 就先产出结果数字。
    """
    transitions = {
        "BASILISK_B5": (
            ("docs/basilisk_b5/results.md", "STATUS_BASILISK_B5.md"),
            ("docs/basilisk_b5/protocol.md",
             "docs/basilisk_b5/baseline_contract.json",
             "checkpoints/basilisk_b5/protocol_hash.json",
             "checkpoints/basilisk_b5/summary.json"),
        ),
        "BASILISK_B6": (
            ("docs/basilisk_b6/results.md", "STATUS_BASILISK_B6.md"),
            ("docs/basilisk_b6/protocol.md",
             "docs/basilisk_b6/baseline_contract.json",
             "checkpoints/basilisk_b6/protocol_hash.json",
             "checkpoints/basilisk_b6/final_verdict.json"),
        ),
    }
    for stage, (artifacts, governance) in transitions.items():
        present = [a for a in artifacts if (ROOT / a).exists()]
        if not present:
            continue
        for gov in governance:
            assert (ROOT / gov).exists(), (
                f"{stage} 已产出 {present[0]} 但缺治理产物 {gov}")


def test_b21_exit_rule_matches_verdict(b21_summary):
    v = b21_summary["verdict"]
    nxt = b21_summary["next_step"]
    if v == "B21_GENERALIZATION_PASS":
        assert nxt == "B5 formal transfer on the frozen B2.1 split"
    else:
        assert nxt == "stop_do_not_run_transfer"


def test_b21_no_positive_transfer_claim(b21_summary):
    """本阶段不产生任何迁移结论。"""
    txt = json.dumps(b21_summary, ensure_ascii=False)
    for bad in ("positive transfer proven", "迁移显著有效",
                "positive_transfer_proven"):
        assert bad not in txt, bad
    assert b21_summary["frozen_facts"][
        "no_formal_positive_transfer_claim"] is True
    assert b21_summary["frozen_facts"]["b4x_status"] == "EXPLORATORY_ONLY"


def test_b21_pass_does_not_overturn_b2(b21_summary):
    """PASS 不得被写成'B2 判错了'。"""
    assert b21_summary["frozen_facts"]["b2_verdict"] == "B2_GENERALIZATION_FAIL"
    assert (b21_summary["frozen_facts"]["b2_verdict_status"]
            == "B2_GENERALIZATION_FAIL_REMAINS_FINAL")
    assert "不追溯修改" in b21_summary["verdict_meaning"] or \
           "失败原因在别处" in b21_summary["verdict_meaning"]


def test_b21_s5b_not_reopened():
    """S5B 阶段已关闭, 本阶段不得重开或重新解释。"""
    for rel in (SPLITTER_REL, GATE_REL, SUM_REL,
                "configs/wheel_basilisk_b21.yaml",
                "docs/basilisk_b21/protocol.md"):
        src = _src(rel)
        assert "S5B" not in src, rel


def test_b21_not_in_requirements_or_docker():
    """Basilisk 不得写入 requirements.txt / Dockerfile / docker-compose.yml。"""
    for rel in ("requirements.txt", "Dockerfile", "docker-compose.yml"):
        p = ROOT / rel
        if not p.exists():
            continue
        s = p.read_text(encoding="utf-8").lower()
        for token in ("basilisk", "b21"):
            assert token not in s, (rel, token)


def test_b21_protocol_frozen_before_numbers():
    """protocol 必须在任何 B2.1 数字之前冻结, 且 hash 被记录。"""
    ph = ROOT / "checkpoints/basilisk_b21/protocol_hash.json"
    if not ph.exists():
        pytest.skip("需先跑 summarize_b21.py")
    import hashlib
    d = json.loads(ph.read_text(encoding="utf-8"))
    assert d["frozen_before_any_b21_number"] is True
    assert d["b2_verdict_unchanged"] == "B2_GENERALIZATION_FAIL"
    assert d["b3x_verdict_unchanged"] == "B3X_NO_STABILIZING_SIGNAL"
    real = hashlib.sha256(
        (ROOT / "docs/basilisk_b21/protocol.md").read_bytes()).hexdigest()
    assert d["b21_protocol"] == real


def test_b21_protocol_states_key_rules():
    """protocol 必须写明分箱定义 / 六条覆盖要求 / B21_SPLIT_INVALID 规则。"""
    p = (ROOT / "docs/basilisk_b21/protocol.md").read_text(encoding="utf-8")
    for token in ("B21_SPLIT_INVALID", "lifetime-support mismatch",
                  "B21_GENERALIZATION_PASS", "B21_GENERALIZATION_FAIL",
                  "B5 formal transfer on the frozen B2.1 split",
                  "split_stratification_only", "observed_duration",
                  "one_paired_difference_per_seed"):
        assert token in p, token
    assert "B2_GENERALIZATION_FAIL` 保持终局" in p
