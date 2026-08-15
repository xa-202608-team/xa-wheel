"""tests/basilisk_b12/test_failure_candidate_registry.py

§19 —— 判据登记表与窗级判定的测试:
  * test_candidate_registry_exactly_three
  * test_f2_uses_friction_torque
  * test_window_failure_not_point_failure
  * test_persistence_equal_all_candidates
  * test_candidate_priority_f2_f3_f1
"""
from __future__ import annotations

import numpy as np
import pytest

F1, F2, F3 = "CURRENT_WINDOW_P95", "FRICTION_TORQUE_P95", "VISCOUS_FRICTION_STATE"


def test_candidate_registry_exactly_three(b12_cfg, cand_rec, b12_gen):
    """§7/§14: 恰好三个变体, 不得新增 F4/F5。"""
    reg = list(b12_cfg["failure_candidates"]["registry"])
    assert len(reg) == 3, f"registry 必须恰好 3 个, 实际 {len(reg)}: {reg}"
    assert set(reg) == {F1, F2, F3}, f"registry 内容不符: {reg}"
    assert set(b12_cfg["failure_candidates"]["spec"].keys()) == set(reg)

    # candidate_statistic 只认这三个名字, 第四个必须直接拒绝
    with pytest.raises(SystemExit):
        b12_gen.candidate_statistic(None, "F4_SOMETHING", {}, b12_cfg)

    if cand_rec is not None:
        assert len(cand_rec["registry_frozen"]) == 3
        assert set(cand_rec["per_candidate"].keys()) == {F1, F2, F3}


def test_f2_uses_friction_torque(b12_gen, b12_cfg, cand_rec):
    """§7: F2 统计量必须是 T_f = b*|omega| + Tc 的窗级 q95, 阈值 = eta_margin*u_max。"""
    import pandas as pd
    om = np.array([100.0, -200.0, 300.0, -50.0])
    b = np.array([1e-5, 2e-5, 3e-5, 4e-5])
    Tc = 1.25e-3
    df = pd.DataFrame({"omega_cmd": om, "b_true": b, "I_m": np.zeros(4),
                       "T_cmd": np.zeros(4)})
    stat = b12_gen.candidate_statistic(df, F2, {"Tc": Tc, "Kt": 0.025}, b12_cfg)
    assert np.allclose(stat, b * np.abs(om) + Tc), \
        "F2 统计量不是 b*|omega_cmd| + Tc"
    assert np.all(stat >= Tc), "F2 统计量应始终 >= Tc (摩擦力矩非负下界)"

    spec = b12_cfg["failure_candidates"]["spec"][F2]
    assert spec["statistic"] == "friction_torque_q95"
    assert spec["threshold_rule"] == "eta_margin_times_umax"
    assert spec["eta_margin_rule"] == "one_minus_torque_util_ref_p95"

    if cand_rec is None:
        pytest.skip("candidates.json 尚未生成")
    d = cand_rec["per_candidate"][F2]["threshold_derivation"]
    # 阈值必须能由登记规则复算 —— 不许是硬编码数字
    eta = 1.0 - float(d["torque_util_ref_p95"])
    assert abs(eta - float(d["eta_margin"])) < 1e-15
    assert abs(eta * float(d["u_max_Nm"])
               - float(cand_rec["per_candidate"][F2]["threshold"])) < 1e-15, \
        "F2 阈值无法由 (1 - torque_util_ref_p95) * u_max 复算"
    assert float(d["u_max_Nm"]) == 0.2, "u_max 被改动 (§2 禁止改 torque_rated)"


def test_window_failure_not_point_failure(b12_gen, b12_cfg):
    """§9: 禁止 single-sample failure —— 单点尖峰不得触发, 且 persistence<2 必须拒绝。

    注意: 单点抗扰性来自 **窗宽与分位数的组合**, 不是 persistence。
    np.percentile 线性插值在 win_n 个样本上取秩 i = q*(win_n-1), 结果由 sorted[floor(i)]
    与 sorted[ceil(i)] 插值而来。若窗内有 k 个远超阈值的样本 (其余为 0), 则 q95 > 0
    当且仅当 ceil(i) >= win_n - k, 即

        k_min = win_n - 1 - floor(q * (win_n - 1))

    win_n=48, q=0.95: i = 44.65, k_min = 48 - 1 - 44 = **3**。
    win_n=8: i = 6.65, k_min = 8 - 1 - 6 = **1** —— 小窗下单点即可拉过 q95, 会得出
    "代码违反 §9"的伪结论。**所以本测试必须用冻结的 win_n=48。**
    """
    fc = b12_cfg["failure_candidates"]
    win_n = int(fc["window_samples"])
    win_q = float(fc["window_quantile"])
    pers = int(fc["persistence_windows"])
    assert win_n == 48 and win_q == 0.95, "本测试的秩推导绑定冻结的 48/0.95 口径"
    n = win_n * 12
    thr = 10.0

    # (a) 单点巨大尖峰: 窗内 q95 不应被单点拉过阈值 -> 不判失效
    stat = np.zeros(n); stat[3] = 1e6
    dec = b12_gen.redecide_eol(stat, thr, win_n, win_q, pers, n)
    assert dec["failed"] is False, "单个样本尖峰触发了失效 —— 违反 §9"

    # (b) 每窗都有一个尖峰 (仍是逐点现象): 同样不应触发
    stat = np.zeros(n); stat[::win_n] = 1e6
    dec = b12_gen.redecide_eol(stat, thr, win_n, win_q, pers, n)
    assert dec["failed"] is False, "逐窗单点尖峰触发了失效 —— 违反 §9"

    # (b2) 秩边界: k_min-1 个越限样本仍不足, 恰好 k_min 个才够 (钉死单点抗扰强度)
    k_min = win_n - 1 - int(np.floor(win_q * (win_n - 1)))
    assert k_min == 3, f"48/0.95 下 k_min 应为 3, 实际 {k_min}"
    for k, want_fail in ((k_min - 1, False), (k_min, True)):
        stat = np.zeros(n)
        for wi in range(n // win_n):                  # 每窗放 k 个越限样本
            stat[wi * win_n:wi * win_n + k] = 1e6
        dec = b12_gen.redecide_eol(stat, thr, win_n, win_q, pers, n)
        assert dec["failed"] is want_fail, \
            f"每窗 {k} 个越限样本 -> failed={dec['failed']}, 期望 {want_fail} (k_min={k_min})"

    # (c) 连续 pers-1 个窗整体越限: 仍不足
    stat = np.zeros(n); stat[2 * win_n:(2 + pers - 1) * win_n] = 100.0
    dec = b12_gen.redecide_eol(stat, thr, win_n, win_q, pers, n)
    assert dec["failed"] is False, f"仅 {pers - 1} 窗越限就触发了失效"

    # (d) 连续 pers 个窗整体越限: 触发, EOL = 首窗起始样本
    stat = np.zeros(n); stat[2 * win_n:(2 + pers) * win_n] = 100.0
    dec = b12_gen.redecide_eol(stat, thr, win_n, win_q, pers, n)
    assert dec["failed"] is True
    assert dec["eol_window"] == 2, f"EOL 窗应为 2, 实际 {dec['eol_window']}"
    assert dec["eol_idx"] == 2 * win_n, "EOL 应落在首个越限窗的起始样本"
    assert dec["label"][dec["eol_idx"]] == 1 and dec["label"][dec["eol_idx"] - 1] == 0
    assert np.all(np.diff(dec["label"]) >= 0), "label_fail 必须单调"

    # (e) 未触发 -> 右删失, eol_idx = n-1
    dec = b12_gen.redecide_eol(np.zeros(n), thr, win_n, win_q, pers, n)
    assert dec["failed"] is False and dec["eol_idx"] == n - 1
    assert dec["label"].sum() == 0, "删失轨迹不得有任何 label_fail=1"

    # (f) persistence < 2 必须被拒绝
    with pytest.raises(SystemExit):
        b12_gen.redecide_eol(np.zeros(n), thr, win_n, win_q, 1, n)


def test_persistence_equal_all_candidates(b12_cfg, cand_rec):
    """§9: 三变体 persistence / 窗口 / 分位完全相同, 且沿用已冻结 simulator 值。"""
    import yaml
    from pathlib import Path
    fc = b12_cfg["failure_candidates"]
    pers = int(fc["persistence_windows"])
    base = yaml.safe_load(Path("configs/wheel.yaml").read_text(encoding="utf-8"))
    assert pers == int(base["sim"]["failure"]["persistence_samples"]), \
        "§9: persistence_windows 应沿用 wheel.yaml 已冻结的 persistence_samples"
    assert pers >= 2, "§9 禁止单窗失效"

    if cand_rec is None:
        pytest.skip("candidates.json 尚未生成")
    assert int(cand_rec["persistence_windows"]) == pers
    assert cand_rec["single_sample_failure_allowed"] is False
    assert cand_rec["eol_rule"] == "window_level_persistence"

    # 三变体共用同一组窗参数 (记录在顶层, 故不可能逐变体不同) —— 再从 h5 属性核验
    import h5py
    seen = {}
    for name, cd in cand_rec["per_candidate"].items():
        with h5py.File(Path(cd["h5"]), "r") as f:
            seen[name] = (int(f.attrs["persistence_windows"]),
                          int(f.attrs["window_samples"]),
                          float(f.attrs["window_quantile"]))
    assert len(set(seen.values())) == 1, f"三变体窗级参数不一致: {seen}"
    assert seen[F2] == (pers, int(fc["window_samples"]), float(fc["window_quantile"]))


def test_candidate_priority_f2_f3_f1(b12_cfg, cand_rec, audit_rec, sel_rec):
    """§13: 优先级 F2 > F3 > F1 冻结, 不得依据 failure fraction 重排。"""
    pri = list(b12_cfg["failure_candidates"]["priority"])
    assert pri == [F2, F3, F1], f"优先级被改动: {pri}"

    for rec, tag in ((cand_rec, "candidates.json"),
                     (audit_rec, "candidate_audit.json"),
                     (sel_rec, "selection_result.json")):
        if rec is None:
            continue
        assert list(rec["priority_frozen"]) == [F2, F3, F1], \
            f"{tag} 中的优先级与冻结值不符"

    if sel_rec is None:
        pytest.skip("selection_result.json 尚未生成")
    # 选择必须严格按优先级: 被选中者之前的所有 candidate 都必须 not selectable
    tbl = sel_rec["selection_table"]
    assert [x["candidate"] for x in tbl] == [F2, F3, F1], \
        "选择表未按冻结优先级排列"
    sel = sel_rec["selected_candidate"]
    if sel is not None:
        idx = [x["candidate"] for x in tbl].index(sel)
        for x in tbl[:idx]:
            assert x["selectable"] is False, \
                f"{x['candidate']} 可选却被跳过 —— 违反冻结优先级"
        assert tbl[idx]["selectable"] is True
    else:
        assert all(x["selectable"] is False for x in tbl), \
            "verdict=FAIL 但存在可选 candidate"
