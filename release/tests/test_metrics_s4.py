"""S4 指标构造测试 (协议 docs/diagnostics_s4_protocol.md §16/§17)。

场景 A–H 全部用**手工构造**的轨迹, 不依赖真实数据里恰好存在某种情形 ——
删失语义、空 bin、macro/pooled 差异、重复 endpoint 都必须被主动证明。

另含 §17: S4 只增加 evaluator, `checkpoints/s3_gate_metrics.json` 的 SHA256 不得变。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.experiments.metrics import (NAN, alpha_lambda_accuracy, convergence_metric,
                                     dedup_endpoints, in_alpha_band,
                                     paired_lead_difference, prognostic_horizon,
                                     prognostics_cfg, staged_metrics,
                                     tolerance_band, warning_lead_time)

ROOT = Path(__file__).resolve().parents[1]
S3_JSON = ROOT / "checkpoints" / "s3_gate_metrics.json"
S4_PROTOCOL = ROOT / "docs" / "diagnostics_s4_protocol.md"
# S4 协议 §1 记录的 S3 指标指纹 (S4 全程只读, 必须保持)
FROZEN_S3_METRICS_SHA = ("f11f46b107168fe0d21e16aa5c286789580a0eec073f8cd879c4"
                         "ac6f9528f68e")

ALPHA, FLOOR = 0.20, 0.02
BINS = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.000001)]
THR, PERSIST = 0.20, 3


# ============================== 构造夹具 ==============================

def _traj(tid, n=11, observed=True):
    """一条线性退化轨迹: tau 0→1 均匀, true RUL 1→0 线性, HI 0→1 线性。"""
    tau = np.linspace(0.0, 1.0, n)
    true = 1.0 - tau
    hi = tau.copy()
    tids = np.full(n, tid, dtype=np.int64)
    return tids, tau, true, hi, observed


def _stack(*trajs):
    tids = np.concatenate([t[0] for t in trajs])
    tau = np.concatenate([t[1] for t in trajs])
    true = np.concatenate([t[2] for t in trajs])
    hi = np.concatenate([t[3] for t in trajs])
    ev = {int(t[0][0]): bool(t[4]) for t in trajs}
    return tids, tau, true, hi, ev


# ============================== A. 完美预测器 ==============================

def test_A_perfect_predictor():
    """perfect: 分阶段 RMSE=0 / PH=完整可观测区间 / alpha-lambda=1 / convergence=0。"""
    tids, tau, true, hi, ev = _stack(_traj(0), _traj(1))
    pred = true.copy()

    st = staged_metrics(pred, true, hi, tids, BINS)
    for b in st["bins"]:
        if b["n_points"]:
            assert b["macro_rmse"] == pytest.approx(0.0, abs=1e-12)
            assert b["pooled_rmse"] == pytest.approx(0.0, abs=1e-12)
            assert b["macro_mae"] == pytest.approx(0.0, abs=1e-12)

    ph = prognostic_horizon(pred, true, tau, tids, ev, ALPHA, FLOOR)
    # 完美预测器从 tau=0 起就在带内且从不脱靶 → PH = 1 - 0 = 1
    assert ph["macro_ph"] == pytest.approx(1.0)
    assert ph["n_observable"] == 2 and ph["n_censored"] == 0
    assert all(v["observable"] for v in ph["per_trajectory"].values())

    al = alpha_lambda_accuracy(pred, true, tau, tids, ev, ALPHA, FLOOR,
                               [0.3, 0.5, 0.7])
    for lam in ("0.3", "0.5", "0.7"):
        assert al["by_lambda"][lam]["accuracy"] == pytest.approx(1.0)
        assert al["by_lambda"][lam]["eligible_count"] == 2

    cv = convergence_metric(pred, true, tau, tids)
    assert cv["macro_convergence"] == pytest.approx(0.0, abs=1e-12)
    assert cv["macro_late_convergence"] == pytest.approx(0.0, abs=1e-12)

    wl = warning_lead_time(pred, tau, tids, ev, THR, PERSIST)
    # true RUL 在 tau>=0.8 处 <= 0.2, 连续 3 点 → 首次满足在 tau=0.8
    assert wl["warning_coverage"] == pytest.approx(1.0)
    assert wl["miss_rate"] == pytest.approx(0.0)
    assert wl["conditional_lead"] == pytest.approx(0.2, abs=1e-9)
    assert np.isfinite(wl["conditional_lead"])


# ============================== B. 常数预测器 ==============================

def test_B_constant_predictor_finite_nonzero_error():
    """常数预测器: 误差有限且非零; PSR=0 (无动态范围); corr=NaN 而非 0。"""
    tids, tau, true, hi, ev = _stack(_traj(0), _traj(1))
    pred = np.full_like(true, 0.5)

    st = staged_metrics(pred, true, hi, tids, BINS)
    for b in st["bins"]:
        if b["n_points"]:
            assert np.isfinite(b["macro_rmse"]) and b["macro_rmse"] > 0
            assert b["pred_std"] == pytest.approx(0.0)
            assert b["pred_std_true_std_ratio"] == pytest.approx(0.0)
            # 常数输出 ⇒ 相关系数无定义, 必须是 NaN, **不得静默返回 0**
            assert np.isnan(b["macro_corr"])

    cv = convergence_metric(pred, true, tau, tids)
    assert np.isfinite(cv["macro_convergence"]) and cv["macro_convergence"] > 0


# ============================== C. 延迟预测器 ==============================

def test_C_delayed_predictor_worse_ph():
    """延迟命中的预测器 PH 必须严格差于完美预测器。"""
    tids, tau, true, hi, ev = _stack(_traj(0, n=11))
    perfect = true.copy()
    # 前 6 点 (tau<=0.5) 严重偏离, 之后精确 → 稳定进入带内的时刻推迟到 tau=0.6
    delayed = true.copy()
    delayed[:6] += 0.6

    ph_p = prognostic_horizon(perfect, true, tau, tids, ev, ALPHA, FLOOR)
    ph_d = prognostic_horizon(delayed, true, tau, tids, ev, ALPHA, FLOOR)
    assert ph_d["macro_ph"] < ph_p["macro_ph"]
    assert ph_d["per_trajectory"][0]["ph_tau_hit"] == pytest.approx(0.6)
    assert ph_d["macro_ph"] == pytest.approx(0.4)

    # convergence 同向: 早期偏离让面积变大
    assert (convergence_metric(delayed, true, tau, tids)["macro_convergence"]
            > convergence_metric(perfect, true, tau, tids)["macro_convergence"])


# ============================== D. 永不预警的预测器 ==============================

def test_D_never_warning_raises_miss_rate():
    """永不报警: coverage 降到 0, miss_rate 升到 1, conditional_lead 为 NaN。"""
    tids, tau, true, hi, ev = _stack(_traj(0), _traj(1))
    never = np.full_like(true, 0.9)          # 恒 > 阈值 0.20, 永不触发

    wl = warning_lead_time(never, tau, tids, ev, THR, PERSIST)
    assert wl["warning_coverage"] == pytest.approx(0.0)
    assert wl["miss_rate"] == pytest.approx(1.0)
    assert wl["n_warned_observed"] == 0
    # 只报 conditional_lead 会把漏报藏起来 ⇒ 此处必须是 NaN, 不能是"很好的 0"
    assert np.isnan(wl["conditional_lead"])
    assert all(not v["warning_issued"] for v in wl["per_trajectory"].values())


def test_D2_whole_risk_and_conditional_are_separate():
    """一条报警、一条不报警: whole-risk 与 conditional 必须给出不同的图景。"""
    t0 = _traj(0)
    t1 = _traj(1)
    tids, tau, true, hi, ev = _stack(t0, t1)
    pred = np.where(tids == 0, 1.0 - tau, 0.9)     # tid0 完美报警, tid1 永不报警

    wl = warning_lead_time(pred, tau, tids, ev, THR, PERSIST)
    assert wl["warning_coverage"] == pytest.approx(0.5)
    assert wl["miss_rate"] == pytest.approx(0.5)
    # conditional_lead 只看报警的那条 ⇒ 看起来"完美", 必须与 miss_rate 同表读
    assert wl["conditional_lead"] == pytest.approx(0.2, abs=1e-9)
    assert wl["conditional_lead_n"] == 1


def test_D3_paired_lead_difference_exposes_small_common_set():
    """配对提前量只在两方法都报警的公共轨迹上算, 且必须暴露 n_common。"""
    tids, tau, true, hi, ev = _stack(_traj(0), _traj(1))
    a = 1.0 - tau                                  # 两条都报警
    b = np.where(tids == 0, 1.0 - tau, 0.9)        # 只有 tid0 报警
    wa = warning_lead_time(a, tau, tids, ev, THR, PERSIST)
    wb = warning_lead_time(b, tau, tids, ev, THR, PERSIST)
    d = paired_lead_difference(wa, wb)
    assert d["n_common"] == 1
    assert d["n_warned_a"] == 2 and d["n_warned_b"] == 1
    assert d["mean_paired_lead_diff"] == pytest.approx(0.0, abs=1e-12)


# ============================== E. 早期误报 ==============================

def test_E_early_false_alarm_counted():
    """在已知未失效 (删失) 的轨迹上报警 = false alarm, 必须被计入。"""
    obs = _traj(0, observed=True)
    cen = _traj(1, observed=False)
    tids, tau, true, hi, ev = _stack(obs, cen)
    # 删失轨迹上恒低于阈值 → 早期即触发报警 (误报); observed 轨迹正常
    pred = np.where(tids == 1, 0.05, 1.0 - tau)

    wl = warning_lead_time(pred, tau, tids, ev, THR, PERSIST)
    assert wl["n_observed"] == 1 and wl["n_censored"] == 1
    assert wl["false_alarm_rate"] == pytest.approx(1.0)
    assert wl["per_trajectory"][1]["false_warning"] is True
    # 误报轨迹没有 EOL ⇒ 不得凭空生成提前量
    assert np.isnan(wl["per_trajectory"][1]["normalized_lead"])
    assert wl["per_trajectory"][0]["false_warning"] is False


def test_E2_no_false_alarm_when_censored_stays_high():
    tids, tau, true, hi, ev = _stack(_traj(0, observed=True),
                                     _traj(1, observed=False))
    pred = np.where(tids == 1, 0.9, 1.0 - tau)
    wl = warning_lead_time(pred, tau, tids, ev, THR, PERSIST)
    assert wl["false_alarm_rate"] == pytest.approx(0.0)


def test_E3_post_eol_warning_not_counted_as_coverage():
    """tau>1 (EOL 之后) 才报出的警报没有预警价值, 不得计入 coverage<EOL。

    真实数据的 tau 可达中位 4.85 (EOL 后仍有大量采样点), 若只看 warning_issued,
    "失效后才报警"会被算成命中 —— 那是把漏报伪装成命中 (协议 §9)。
    """
    # 一条轨迹: tau 0→3 (EOL 在 tau=1); 预测只在 tau>=2 之后才降到阈值以下
    tau = np.linspace(0.0, 3.0, 13)
    true = np.clip(1.0 - tau, 0.0, None)
    tids = np.zeros(len(tau), dtype=np.int64)
    ev = {0: True}
    pred = np.where(tau >= 2.0, 0.05, 0.9)

    wl = warning_lead_time(pred, tau, tids, ev, THR, PERSIST)
    v = wl["per_trajectory"][0]
    assert v["warning_issued"] is True                      # 确实报了
    assert v["tau_warning"] > 1.0                           # 但是在 EOL 之后
    assert v["detected_before_eol"] is False
    assert v["normalized_lead"] < 0                         # 负提前量 = 事后报警
    # 宽口径把它算成命中; 严口径 (可部署性) 必须判为漏报
    assert wl["warning_coverage"] == pytest.approx(1.0)
    assert wl["coverage_before_eol"] == pytest.approx(0.0)
    assert wl["miss_rate_before_eol"] == pytest.approx(1.0)
    assert wl["n_warned_observed"] == 1 and wl["n_warned_before_eol"] == 0
    assert np.isnan(wl["conditional_lead_before_eol"])


def test_E4_pre_eol_warning_counted_in_both_calibers():
    """EOL 之前报出的警报在宽/严两个口径下都算命中, 且提前量为正。"""
    tau = np.linspace(0.0, 3.0, 13)
    true = np.clip(1.0 - tau, 0.0, None)
    tids = np.zeros(len(tau), dtype=np.int64)
    ev = {0: True}
    pred = np.where(tau >= 0.5, 0.05, 0.9)                  # 早于 EOL 报出
    wl = warning_lead_time(pred, tau, tids, ev, THR, PERSIST)
    assert wl["per_trajectory"][0]["detected_before_eol"] is True
    assert wl["warning_coverage"] == pytest.approx(1.0)
    assert wl["coverage_before_eol"] == pytest.approx(1.0)
    assert wl["conditional_lead_before_eol"] > 0


# ============================== F. 删失轨迹 ==============================

def test_F_censored_trajectory_no_fabricated_eol():
    """删失轨迹绝不生成 EOL / PH / lead time —— 一律 NaN + observable=False。"""
    tids, tau, true, hi, ev = _stack(_traj(0, observed=True),
                                     _traj(1, observed=False))
    pred = true.copy()                              # 即使预测完美也不许伪造

    ph = prognostic_horizon(pred, true, tau, tids, ev, ALPHA, FLOOR)
    assert ph["n_observable"] == 1 and ph["n_censored"] == 1
    c = ph["per_trajectory"][1]
    assert c["observable"] is False
    assert np.isnan(c["ph_normalized"]) and np.isnan(c["ph_tau_hit"])
    assert c["reason"] == "right_censored_no_true_eol"
    # macro 只在可观测轨迹上聚合 ⇒ 不被 NaN 污染, 也不被删失轨迹"顶满"
    assert ph["macro_ph"] == pytest.approx(1.0)

    al = alpha_lambda_accuracy(pred, true, tau, tids, ev, ALPHA, FLOOR, [0.5])
    assert al["by_lambda"]["0.5"]["eligible_count"] == 1       # 删失不入分母
    assert al["per_trajectory"]["0.5"][1]["eligible"] is False
    assert al["per_trajectory"]["0.5"][1]["reason"] == "right_censored"

    wl = warning_lead_time(pred, tau, tids, ev, THR, PERSIST)
    assert np.isnan(wl["per_trajectory"][1]["normalized_lead"])
    assert wl["per_trajectory"][1]["detected_before_eol"] is False


def test_F2_all_censored_gives_nan_not_zero():
    """全删失时 coverage / miss_rate 必须是 NaN (无 observed 分母), 不是 0/1。"""
    tids, tau, true, hi, ev = _stack(_traj(0, observed=False))
    pred = true.copy()
    wl = warning_lead_time(pred, tau, tids, ev, THR, PERSIST)
    assert np.isnan(wl["warning_coverage"]) and np.isnan(wl["miss_rate"])
    ph = prognostic_horizon(pred, true, tau, tids, ev, ALPHA, FLOOR)
    assert np.isnan(ph["macro_ph"])


# ============================== G. macro vs pooled ==============================

def test_G_macro_differs_from_pooled_with_unequal_lengths():
    """长短两轨迹: 长轨迹绝不因点多而支配 macro ⇒ macro != pooled。"""
    # 短轨迹 (3 点) 误差大; 长轨迹 (61 点) 误差小
    s_tau = np.linspace(0.0, 1.0, 3)
    l_tau = np.linspace(0.0, 1.0, 61)
    tau = np.concatenate([s_tau, l_tau])
    true = np.concatenate([1 - s_tau, 1 - l_tau])
    hi = np.concatenate([s_tau, l_tau])          # 全 bin 覆盖
    tids = np.concatenate([np.zeros(3, np.int64), np.ones(61, np.int64)])
    err = np.concatenate([np.full(3, 0.40), np.full(61, 0.01)])
    pred = true + err

    st = staged_metrics(pred, true, hi, tids, [(0.0, 1.000001)])
    b = st["bins"][0]
    assert b["n_points"] == 64 and b["n_trajectories"] == 2
    # pooled 被 61 个小误差点拉低; macro 两条轨迹等权 ⇒ 明显更高
    assert b["pooled_rmse"] == pytest.approx(
        float(np.sqrt(np.mean(err ** 2))), rel=1e-12)
    assert b["macro_rmse"] == pytest.approx((0.40 + 0.01) / 2, rel=1e-12)
    assert b["macro_rmse"] > b["pooled_rmse"] * 2


def test_G2_empty_bin_is_nan_not_zero():
    """空 bin: n_points=0 且所有指标 NaN, **不伪造 0**。"""
    tids, tau, true, hi, ev = _stack(_traj(0))
    st = staged_metrics(true, true, hi, tids, [(0.0, 0.2), (5.0, 6.0)])
    empty = st["bins"][1]
    assert empty["n_points"] == 0 and empty["n_trajectories"] == 0
    for k in ("pooled_rmse", "pooled_mae", "macro_rmse", "macro_mae",
              "macro_corr", "pred_std_true_std_ratio",
              "pred_mean", "pred_std", "true_mean", "true_std"):
        assert np.isnan(empty[k]), f"空 bin 的 {k} 被伪造成 {empty[k]}"


# ============================== H. 重复 endpoint ==============================

def test_H_duplicate_endpoint_dedup_behaviour():
    """同一 (traj, endpoint) 重复出现时 evaluator 必须去重, 且报告丢弃数量。"""
    tids, tau, true, hi, ev = _stack(_traj(0, n=5))
    # 把第 2 个点整段复制一遍 (同 tid 同 tau) —— 模拟上游取点重复
    dup = np.array([1])
    tids2 = np.concatenate([tids, tids[dup]])
    tau2 = np.concatenate([tau, tau[dup]])
    true2 = np.concatenate([true, true[dup]])
    hi2 = np.concatenate([hi, hi[dup]])
    pred2 = true2 + 0.05

    keep, kt, ktau = dedup_endpoints(tids2, tau2)[:3]
    assert keep.sum() == 5 and (~keep).sum() == 1
    assert keep[-1] == False       # noqa: E712 —— 重复项被丢弃的是后出现的那个

    st = staged_metrics(pred2, true2, hi2, tids2, [(0.0, 1.000001)])
    assert st["n_dropped_duplicate_endpoints"] == 1
    assert st["bins"][0]["n_points"] == 5      # 去重后点数, 不是 6

    # 去重后的指标必须与"本来就没有重复"的输入完全一致
    st_clean = staged_metrics(true + 0.05, true, hi, tids, [(0.0, 1.000001)])
    assert st["bins"][0]["pooled_rmse"] == pytest.approx(
        st_clean["bins"][0]["pooled_rmse"], rel=1e-12)


# ============================== alpha 带 / floor ==============================

def test_absolute_floor_prevents_band_collapse_near_eol():
    """true_rul→0 时若无 absolute_floor, 任何有限误差都判脱靶 —— floor 必须生效。"""
    true = np.array([0.0, 0.01, 0.5])
    band_with = tolerance_band(true, ALPHA, FLOOR)
    band_without = tolerance_band(true, ALPHA, 0.0)
    assert band_with[0] == pytest.approx(FLOOR) and band_without[0] == 0.0
    assert band_with[2] == pytest.approx(0.10)     # 0.2*0.5 > floor ⇒ 相对带主导
    pred = np.array([0.01, 0.0, 0.5])
    assert in_alpha_band(pred, true, ALPHA, FLOOR).all()
    assert not in_alpha_band(pred, true, ALPHA, 0.0)[0]


def test_alpha_lambda_one_vote_per_trajectory():
    """禁止用时间点数量当独立样本: eligible_count 恒等于合格轨迹数。"""
    short = _traj(0, n=5)
    long_ = _traj(1, n=101)
    tids, tau, true, hi, ev = _stack(short, long_)
    pred = true.copy()
    al = alpha_lambda_accuracy(pred, true, tau, tids, ev, ALPHA, FLOOR, [0.5])
    assert al["by_lambda"]["0.5"]["eligible_count"] == 2      # 不是 5+101
    assert al["by_lambda"]["0.5"]["success_count"] == 2


# ============================== convergence 不是 RMSE ==============================

def test_convergence_distinguishes_early_vs_late_failure():
    """同一 RMSE 下'早期差晚期好'与'早期好晚期崩'的 late_convergence 必须不同。"""
    tids, tau, true, hi, ev = _stack(_traj(0, n=21))
    e = np.zeros(21)
    e[:10] = 0.3                                   # 早期差
    early_bad = true + e
    late_bad = true + e[::-1].copy()               # 晚期差 (同一组误差幅值)

    ca = convergence_metric(early_bad, true, tau, tids)
    cb = convergence_metric(late_bad, true, tau, tids)
    # 总面积接近 (同一误差集合), 但 late 区面积截然不同
    assert ca["macro_convergence"] == pytest.approx(cb["macro_convergence"], rel=0.2)
    assert cb["macro_late_convergence"] > ca["macro_late_convergence"] * 3


def test_convergence_single_point_trajectory_is_nan():
    """单点轨迹无法积分 ⇒ NaN, 不退化成绝对误差。"""
    cv = convergence_metric(np.array([0.5]), np.array([0.9]), np.array([0.5]),
                            np.array([0]))
    assert np.isnan(cv["per_trajectory"][0]["convergence"])
    assert np.isnan(cv["macro_convergence"])


# ============================== config 口径 ==============================

def test_prognostics_cfg_frozen_values():
    """S4 阈值必须来自 config (§5), 且与协议记录一致; 缺键必须报错而非用默认值。"""
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "wheel.yaml").read_text(encoding="utf-8"))
    pc = prognostics_cfg(cfg)
    assert pc["hi_bins"] == [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.000001)]
    assert pc["alpha"] == pytest.approx(0.20)
    assert pc["absolute_floor"] == pytest.approx(0.02)     # 不得硬编码
    assert pc["lambdas"] == [0.3, 0.5, 0.7]
    assert pc["rul_threshold"] == pytest.approx(0.20)
    assert pc["persistence"] == 3
    assert pc["normalize_time"] is True
    assert pc["late_from"] == pytest.approx(0.5)
    with pytest.raises(KeyError):
        prognostics_cfg({"evaluation": {}})


def test_eval_test_backward_compatible_without_cfg():
    """§14: 不传 cfg 时 eval_test 一个 S4 key 都不多 —— 旧 S2.5/S3 调用不会崩。"""
    import inspect

    from src.experiments import run_groups as rg
    sig = inspect.signature(rg.eval_test)
    assert sig.parameters["cfg"].default is None
    src = inspect.getsource(rg.eval_test)
    assert "if cfg is not None" in src, "S4 指标必须挂在 cfg 非空分支内"
    # 旧 4 参调用形式仍然合法
    assert list(sig.parameters)[:4] == ["model", "loader", "device", "cap_eps"]


# ============================== §17 S3 数字不变 ==============================

def test_s4_does_not_mutate_s3_metrics():
    """S4 只增加 evaluator: checkpoints/s3_gate_metrics.json 的 SHA256 必须不变。"""
    assert S3_JSON.exists(), "缺 S3 冻结指标文件"
    got = hashlib.sha256(S3_JSON.read_bytes()).hexdigest()
    assert got == FROZEN_S3_METRICS_SHA, (
        f"S3 指标文件被改动! 期望 {FROZEN_S3_METRICS_SHA} 实得 {got}")
    d = json.loads(S3_JSON.read_text(encoding="utf-8"))
    assert d["verdict"] == "S3_NO_TRANSFER_SIGNAL", "S4 不得改变 S3 判词"


def test_s4_protocol_frozen_and_records_required_items():
    """§4: 协议在出数前存在, 且记录了全部必需项。"""
    assert S4_PROTOCOL.exists()
    txt = S4_PROTOCOL.read_text(encoding="utf-8")
    for k in (FROZEN_S3_METRICS_SHA,                 # S3 metrics hash
              "0ed0c105bd68729400f455e12e38451961441a537986dc2f6ab2499acccf4b28",
              "43ab6580e2dde8a4053a420ee3c4a842caac6f9ee0131971addae8da0a8cd547",
              "target_stride", "唯一 endpoint 规则", "hi_bins",
              "alpha", "absolute_floor", "rul_threshold", "persistence",
              "Prognostic Horizon", "Convergence", "nPHM",
              "S4_METRICS_READY", "S4_INVALID", "S3_NO_TRANSFER_SIGNAL"):
        assert k in txt, f"S4 协议缺记录项: {k}"


def test_nphm_overflow_guard_unchanged():
    """§11: phm_score 的历史定义与溢出保护不得被改动, 且不作模型选择指标。"""
    from src.baselines import physical_extrap as pe
    import inspect
    assert pe._EXP_CLIP == 700.0
    sig = inspect.signature(pe.phm_score)
    assert sig.parameters["early_scale"].default == pytest.approx(0.13)
    assert sig.parameters["late_scale"].default == pytest.approx(0.10)
    # 极端输入不得溢出 / 不得返回 inf
    v = pe.phm_score(np.array([1e6, -1e6]), np.array([0.0, 0.0]))
    assert np.isfinite(v)


def test_persistence_stays_not_deployable():
    """§15: persistence 保持 ORACLE_LIKE, 不进入可部署方法排名。"""
    from src.baselines import trivial as tv
    assert "persistence" in tv.ORACLE_LIKE
    assert tv.GATE_BASELINE == "const_mean_info"
    assert tv.GATE_BASELINE not in tv.ORACLE_LIKE
