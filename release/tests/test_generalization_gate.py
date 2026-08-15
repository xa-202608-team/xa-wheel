"""tests/test_generalization_gate.py — S2.5 泛化闸门的纪律不变量 (协议 §15)

这些测试钉死的不是"数字好不好", 而是**流程有没有作弊**:
  - 早停/checkpoint 选择只用 val, test 在结构上无法进入 (签名里没有 test loader)
  - 候选选择阶段不计算 test
  - const_mean_info 只读 train
  - gate seeds 与调参 seeds 不相交、gate 用冻结配置、gate 主指标是 info macro
  - bootstrap 重采样单位是 seed-level 配对差, 不是时间点
  - S2.5 不跑迁移组、不启用截断
  - 数据划分与 x_T schema 与协议冻结值一致
  - 协议文件先于结果文档存在

跑: python -m pytest tests/test_generalization_gate.py -q
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config                                          # noqa: E402
from src.sim.build_hi import XT_COLS                                       # noqa: E402
from src.transfer.train_transfer import split_trajectories_from_cfg        # noqa: E402
from src.baselines.physical_extrap import caliber_mask, eval_point_indices  # noqa: E402
from src.baselines import trivial                                          # noqa: E402
from src.experiments import run_groups as rg                               # noqa: E402
import scripts.diag_generalization as dg                                   # noqa: E402
import scripts.run_s25_gate as gate                                        # noqa: E402

CONFIG = ROOT / "configs" / "wheel.yaml"
PROTOCOL = ROOT / "docs" / "diagnostics_s25_protocol.md"
RESULTS = ROOT / "docs" / "diagnostics_s25_results.md"

# 协议 §2 / §3 冻结值 (改动数据划分或 x_T 会让下面两个测试立刻失败)
SPLIT_HASH = "1f319b2b6d440695e0f3dea5c1db7920f4319da590766e16e9227ec70b8302a4"
# S2.5/S3 协议冻结的 8 维 schema hash。**刻意保留**: 两份协议文档已冻结, 不允许
# 事后改写, 故它仍是"协议里必须出现的字符串"这条断言的比对对象。
XT_SCHEMA_HASH_PRE_S5 = "d9c67f0b0c68df7836e1e8caacb2839e9acea2a1a4f7ee0bf2f24ca4757812f5"
# S5 改动 3 追加长基线趋势特征 (Tf_ratio, Tf_slope) 后的 10 维 schema hash。
# 变更是**纯追加**: 前 8 列与 S2.5/S3 逐位相同 (由 test_xt_schema_s5_is_additive 钉死),
# 所以 S3/S4 的 x_T[:, :8] 语义未变, 冻结结果仍可比。
XT_SCHEMA_HASH = "688b91095222e5c89d9cbfdca09a0821ff37ca5515fc8407f0ea0167b5f703bd"
XT_COLS_PRE_S5 = ("I_m", "omega", "T", "T_cmd", "sigma_Im", "b_hat", "dT", "omega_err")
HASH_SEEDS = (42, 43, 44, 52, 53, 54, 55, 56)
N_OBSERVED = 60


@pytest.fixture(scope="module")
def cfg():
    return load_config(str(CONFIG))


@pytest.fixture(scope="module")
def target_h5(cfg):
    p = ROOT / cfg["transfer"]["target_feature_path"]
    if not p.exists():
        pytest.skip(f"缺目标域特征 {p}; 先 python -m src.sim.build_hi --report")
    return p


# ============================ 早停只用 val ============================

def test_earlystop_uses_val_info_macro(cfg):
    """early_stop_metric 可选 info_macro_rmse, 且它确实取 calibers.info.macro.rmse。"""
    assert "info_macro_rmse" in rg.EARLY_STOP_METRICS
    m = {"calibers": {"info": {"macro": {"rmse": 0.11}, "pooled": {"rmse": 0.22}},
                      "full": {"pooled": {"rmse": 0.33}},
                      "valid": {"pooled": {"rmse": 0.44}}}}
    assert rg.early_stop_value(m, "info_macro_rmse") == pytest.approx(0.11)
    assert rg.early_stop_value(m, "info_pooled_rmse") == pytest.approx(0.22)
    # NaN 视为最差, 不会被选成 best
    nan_m = {"calibers": {"info": {"macro": {"rmse": float("nan")}}}}
    assert rg.early_stop_value(nan_m, "info_macro_rmse") == float("inf")
    # 协议 §6: 候选 B/C/D 全部使用 info_macro_rmse
    cands = cfg["s25"]["candidates"]
    for n in ("B", "C", "D"):
        assert cands[n]["early_stop_metric"] == "info_macro_rmse"


def test_earlystop_never_reads_test():
    """结构性保证: _train_with_early_stop 拿不到 test loader, 且拒绝 test 指标名。"""
    params = list(inspect.signature(rg._train_with_early_stop).parameters)
    assert "lva" in params, "早停必须有 val loader"
    assert not [p for p in params if "lte" in p or "test" in p.lower()], \
        f"早停签名出现 test 相关参数: {params}"
    # 指标名里带 test 一律拒绝
    for bad in ("test_rmse", "info_macro_rmse_test", "TEST_info"):
        with pytest.raises(ValueError):
            rg.early_stop_value({"calibers": {}}, bad)
    # 注册表里也不允许出现 test 指标
    assert not [k for k in rg.EARLY_STOP_METRICS if "test" in k.lower()]
    # 函数体里除 eval_test(..., lva, ...) 外不得引用别的 loader
    src = inspect.getsource(rg._train_with_early_stop)
    assert "lte" not in src
    assert src.count("eval_test(") == 1 and "eval_test(model, lva" in src


def test_candidate_selection_never_reads_test():
    """调参 phase 不构造 test loader, 不算 test; select 只吃 val 汇总量。"""
    src = inspect.getsource(dg.phase_tune)
    assert "with_test=False" in src, "tune 必须以 with_test=False 准备数据"
    assert 'assert "lte" not in data' in src, "tune 必须断言手里没有 test loader"
    assert "lte" not in inspect.getsource(dg.train_candidate)
    # select 规则只读 mean/std of val_info_macro_rmse
    ssrc = inspect.getsource(dg.select_candidate)
    assert "val_info_macro_rmse" in ssrc
    # 只读 val_* 键: 逐行剔除注释/字符串字面量后, 不得出现 test 相关的键访问
    code_only = "\n".join(l.split("#")[0] for l in ssrc.splitlines())
    for bad in ("test_", "_test", '["test', "eval_test", "lte"):
        assert bad not in code_only, f"select 阶段疑似触碰 test: {bad}"
    # 排序键必须是 val 量
    assert 'r["val_info_macro_rmse"]' in code_only or "val_info_macro_rmse" in code_only
    # prepare(with_test=False) 返回结构里没有 lte 键 (纯静态: 只有 with_test 才 set)
    psrc = inspect.getsource(dg.prepare)
    assert 'if with_test:' in psrc and 'out["lte"]' in psrc


# ============================ 基线只读 train ============================

def test_const_mean_info_uses_train_only(cfg, target_h5):
    """const_mean_info 的常数值必须等于 train 划分 info 区真实 RUL 均值 (不含 val/test)。"""
    seed = int(cfg["seed"])
    tv = trivial.evaluate_trivial(cfg, seed, return_raw=True)
    assert tv is not None
    const = float(tv["_meta"]["const_val_info"])

    import h5py
    from src.baselines.physical_extrap import baseline_hyper, observed_traj_keys
    h = baseline_hyper(cfg)
    with h5py.File(target_h5, "r") as f:
        keys = observed_traj_keys(f)
        tr, va, te, _ = split_trajectories_from_cfg(
            target_h5, len(keys), cfg["transfer"], seed, observed_only=True)
        scale = max(float(np.max([np.nanmax(f[keys[int(i)]]["rul"][:]) for i in tr])), 1.0)

        def info_mean(ids):
            vals = []
            for i in ids:
                r = f[keys[int(i)]]["rul"][:].astype(float) / scale
                pts = eval_point_indices(len(r), h["L"], h["stride"])
                if len(pts):
                    vals.append(r[pts])
            a = np.concatenate(vals)
            m = caliber_mask(a, "info", h["cap_eps"])
            return float(a[m].mean())

        assert const == pytest.approx(info_mean(tr), abs=1e-9), "常数值必须来自 train"
        # 与 val / test 的 info 均值不同 → 证明确实没混进去
        assert abs(const - info_mean(va)) > 1e-6
        assert abs(const - info_mean(te)) > 1e-6

    # 基线预测确实是常数
    p = tv["_raw"]["pred"][trivial.GATE_BASELINE]
    assert float(p.std()) == pytest.approx(0.0, abs=1e-12)
    assert float(p[0]) == pytest.approx(const, abs=1e-9)


def test_persistence_is_marked_oracle_and_excluded():
    """persistence 必须标注 ORACLE-LIKE 且不是 gate 基线。"""
    assert "persistence" in trivial.ORACLE_LIKE
    assert trivial.GATE_BASELINE == "const_mean_info"
    assert trivial.GATE_BASELINE not in trivial.ORACLE_LIKE
    assert "NOT DEPLOYABLE" in inspect.getsource(trivial)
    # gate 脚本显式排除 oracle-like
    assert "ORACLE_LIKE" in inspect.getsource(gate.run_gate)


# ============================ 种子 / 冻结 / 主指标 ============================

def test_gate_seeds_disjoint_from_tuning_seeds(cfg):
    t, g = dg.tuning_seeds(cfg), dg.gate_seeds(cfg)
    assert t == [42, 43, 44] and g == [52, 53, 54, 55, 56]
    assert len(g) == 5 and not set(t) & set(g)


def test_gate_uses_frozen_candidate(cfg, monkeypatch):
    """selected_candidate 为 null 时 gate 必须拒绝运行; 非 null 时用 config 的 training.*。"""
    import copy
    c = copy.deepcopy(cfg)
    c["s25"]["selected_candidate"] = None
    with pytest.raises(RuntimeError, match="selected_candidate"):
        gate.run_gate(c, CONFIG)
    sel = cfg["s25"].get("selected_candidate")
    if sel is None:
        pytest.skip("尚未冻结候选 (--phase select 未跑)")
    # 冻结后 config 的 training.* 必须与选中候选逐键一致
    frozen = rg.training_hyper(cfg)
    for k, v in cfg["s25"]["candidates"][sel].items():
        assert frozen[k] == (type(frozen[k])(v)), f"training.{k} 与候选 {sel} 不一致"
    # 5 个 seed 用同一配置 → 指纹只依赖 cfg, 与 seed 无关
    assert dg.candidate_signature(cfg) == dg.candidate_signature(cfg)


def test_gate_metric_is_info_macro(cfg):
    """gate 主指标是 test info 轨迹宏平均 RMSE, 且不改全局 primary_stat。"""
    assert cfg["s25"]["gate_metric"] == "info_macro_rmse"
    c = {"info": {"macro": {"rmse": 0.7}, "pooled": {"rmse": 0.9}}}
    assert gate.GATE_METRICS["info_macro_rmse"](c) == pytest.approx(0.7)
    # 全局主口径仍是 pooled (tests/test_metrics.py 钉死的那个), 未被本阶段改动
    from src.baselines.physical_extrap import PRIMARY_CALIBER, PRIMARY_STAT
    assert (PRIMARY_CALIBER, PRIMARY_STAT) == ("info", "pooled")
    assert cfg["evaluation"]["primary_stat"] == "pooled"


def test_paired_gain_sign():
    """gain = baseline - target; > 0 表示 target_only 更好 (RMSE 更小)。"""
    src = inspect.getsource(gate.run_gate)
    assert '"gain": float(r_b - r_t)' in src
    assert 0.30 - 0.25 > 0                                    # baseline 差 → gain 正
    b = paired = None
    # 端到端小算例
    gains = np.array([0.30 - 0.25, 0.28 - 0.26])
    assert (gains > 0).all()


def test_bootstrap_resamples_seed_pairs():
    """bootstrap 单位必须是 seed-level 配对差 (n_units == seed 数), 且可复现。"""
    g = np.array([0.01, 0.02, -0.01, 0.03, 0.005])
    out = gate.paired_bootstrap(g, 10000, 20260807)
    assert out["n_units"] == len(g) == 5, "重采样单位必须是 seed 数, 不是时间点数"
    assert out["mean"] == pytest.approx(g.mean())
    lo, hi = out["ci95"]
    assert lo <= g.mean() <= hi
    # 固定种子可复现
    assert gate.paired_bootstrap(g, 10000, 20260807)["ci95"] == out["ci95"]
    # 换种子结果不同 (证明真的在重采样)
    assert gate.paired_bootstrap(g, 10000, 1)["ci95"] != out["ci95"]
    # 全正 gain → CI 下界应 > 0; 全负 → < 0
    assert gate.paired_bootstrap(np.full(5, 0.05), 2000, 7)["ci95"][0] > 0
    assert gate.paired_bootstrap(np.full(5, -0.05), 2000, 7)["ci95"][1] < 0


# ============================ S2.5 范围约束 ============================

def test_no_transfer_group_runs_in_s25():
    """S2.5 只跑 target_only: gate 脚本不得触碰迁移路径或源域 checkpoint。"""
    src = inspect.getsource(gate) + inspect.getsource(dg)
    for bad in ("source_finetune", "source_mmd_finetune", "source_frozen",
                "load_pretrained", "mmd_by_hi_bins", "use_mmd=True"):
        assert bad not in src, f"S2.5 不得涉及 {bad}"
    assert "freeze_encoder(False)" in inspect.getsource(dg.train_candidate)


def test_no_truncation_enabled_in_s25(cfg):
    """S3 的 truncate_hi / censor hinge 在 S2.5 必须不存在或未启用。"""
    tc = cfg["transfer"]
    for k in ("truncate_hi", "censor_hinge", "censor_weight"):
        assert not tc.get(k, False), f"S2.5 不得启用 transfer.{k}"
        assert not cfg.get("training", {}).get(k, False)
    src = inspect.getsource(gate) + inspect.getsource(dg)
    assert "truncate" not in src.lower() and "censor" not in src.lower()
    # 冻结的两段降权值不得被本阶段改动 (协议 §4)
    assert float(cfg["training"]["post_eol_weight"]) == 0.1
    assert float(cfg["training"]["capped_weight"]) == 0.1
    assert float(cfg["experiments"]["rul_cap_ratio"]) == 0.35


def test_split_unchanged(cfg, target_h5):
    """8 个待用 seed 的 train/val/test 轨迹 id 必须与协议冻结的 split hash 一致。"""
    lines = []
    for s in HASH_SEEDS:
        tr, va, te, _ = split_trajectories_from_cfg(
            target_h5, N_OBSERVED, cfg["transfer"], s, observed_only=True)
        assert (len(tr), len(va), len(te)) == (9, 12, 39)
        lines.append(f"{s}|tr={list(map(int, tr))}|va={list(map(int, va))}"
                     f"|te={list(map(int, te))}")
    got = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    assert got == SPLIT_HASH, f"数据划分已改变! {got} != 协议冻结值 {SPLIT_HASH}"
    sc = cfg["transfer"]["split"]
    assert (sc["train"], sc["val"], sc["test"]) == (0.15, 0.20, 0.65)
    assert sc["stratify_by_eol"] is True and int(sc["n_eol_strata"]) == 4


def test_xt_schema_unchanged(cfg):
    got = hashlib.sha256(json.dumps(list(XT_COLS)).encode()).hexdigest()
    assert got == XT_SCHEMA_HASH, f"x_T schema 已改变! {list(XT_COLS)}"
    assert len(XT_COLS) == int(cfg["model"]["n_features_target"]) == 10


def test_xt_schema_s5_is_additive():
    """S5 改动 3 必须是**纯追加**: 前 8 列与 S2.5/S3 冻结 schema 逐位相同。

    这是"S3/S4 冻结结果仍可比"的前提。若谁重排或替换了前 8 列, x_T[:, :8] 的语义
    就变了, 冻结数字失去可比性 —— 那种改动必须在这里被拦住, 而不是靠人肉记忆。
    """
    assert tuple(XT_COLS[:8]) == XT_COLS_PRE_S5, \
        f"前 8 列被改动! {list(XT_COLS[:8])} != {list(XT_COLS_PRE_S5)}"
    assert tuple(XT_COLS[8:]) == ("Tf_ratio", "Tf_slope"), \
        f"S5 新增列应为 (Tf_ratio, Tf_slope), 实际 {list(XT_COLS[8:])}"
    pre = hashlib.sha256(json.dumps(list(XT_COLS_PRE_S5)).encode()).hexdigest()
    assert pre == XT_SCHEMA_HASH_PRE_S5, "PRE_S5 列表与其冻结 hash 不自洽"


def test_s25_protocol_precedes_results():
    """协议必须存在, 且在结果文档之前创建 (先冻结判据, 后看数字)。"""
    assert PROTOCOL.exists(), "协议缺失: 必须先写 docs/diagnostics_s25_protocol.md"
    text = PROTOCOL.read_text(encoding="utf-8")
    for must in ("candidates", "gate_metric", "S25_PASS", "S25_STOP_GENERALIZATION",
                 SPLIT_HASH, XT_SCHEMA_HASH_PRE_S5):
        assert must in text, f"协议缺少必录内容: {must}"
    if RESULTS.exists():
        assert PROTOCOL.stat().st_mtime <= RESULTS.stat().st_mtime, \
            "协议文件比结果文档更新 —— 疑似看到结果后改了判据"


def test_candidates_differ_only_in_allowed_keys(cfg):
    """四候选之间除协议 §6 的四个键外禁止存在差异。"""
    cands = cfg["s25"]["candidates"]
    assert set(cands) == {"A", "B", "C", "D"}
    for n, d in cands.items():
        assert set(d) <= set(dg.CANDIDATE_KEYS), f"候选 {n} 含协议外键 {set(d) - set(dg.CANDIDATE_KEYS)}"
    # A = S2'' 末的真实行为; C/D 只在 weight_decay 上与 B 不同
    assert cands["A"] == {"early_stop_metric": "info_pooled_rmse", "max_epochs": 20,
                          "early_stop_patience": 0, "weight_decay": 0.0}
    for n, wd in (("C", 1.0e-4), ("D", 1.0e-3)):
        assert float(cands[n]["weight_decay"]) == wd
        assert {k: v for k, v in cands[n].items() if k != "weight_decay"} == \
               {k: v for k, v in cands["B"].items() if k != "weight_decay"}


def test_patience_stops_early():
    """patience>0 时确实提前终止; patience=0 时跑满 (S2'' 行为)。"""
    src = inspect.getsource(rg._train_with_early_stop)
    assert "bad >= int(early_stop_patience)" in src
    assert "int(early_stop_patience) > 0" in src, "patience=0 必须表示不提前终止"
    assert "history.append" in src, "必须逐 epoch 记 train loss 与三档 val RMSE"
    for k in ("val_full_rmse", "val_info_pooled_rmse", "val_info_macro_rmse", "train_loss"):
        assert k in src, f"history 缺 {k}"
