#!/usr/bin/env python
"""scripts/basilisk_b11/audit_candidates.py

Basilisk-B1.1 §9/§10 —— candidate 物理 Gate (**12 条**) + 选择规则 (A11 > B11 > C11)。

Gate 1–10 与 B1 逐值相同 (不放宽); 11/12 为 B1.1 新增:
  Gate 11 early-EOL concentration : fraction(EOL < 0.1 × horizon) < 0.30
  Gate 12 initial current margin  : initial_margin_ratio_p95 <= 0.50

**本脚本只读物理量。** 允许读: failure fraction / EOL 分布 / friction-growth 分布 /
温度与电流物理范围 / mode-conditioned wear ordering / censored fraction /
early-EOL 占比 / 初始电流裕度 / 数值稳定性。

**绝对禁止** (§17): RUL RMSE、correlation、PH、warning lead、transfer gain、
任何神经网络指标。实现层面的硬保证:
  * 本脚本不 import src.models / src.transfer / src.experiments / torch;
  * 也不读取 checkpoints/ 下任何模型指标 json;
  * 由 tests/basilisk_b11/test_candidate_selection_no_rul.py 与
    test_candidate_selection_no_model_metric.py 用 AST 钉死。

Gate 判据全部走 config (configs/wheel_basilisk_b11.yaml gate:), 不硬编码阈值。
失效率带宽是 §9 预先登记的 35%–80% —— **不为靠近 65% 微调**。

mode 严酷度排序由 **B1.1 自己的 p95 reference** 算出, 不沿用 B1 的 rms 版本 ——
口径变了, 排序必须重新推导才自洽 (排序本身仍是纯物理量, 不涉任何模型)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "basilisk_b11"))

import calibrate_degradation as cal                     # noqa: E402
from src.sim.basilisk_bridge import MODES               # noqa: E402

# §9 的 12 条 gate 名称 (顺序固定, 报告/测试共用)
GATE_NAMES = (
    "1 failure_fraction 落在预先登记带宽内",
    "2 censored_fraction >= 下限",
    "3 event-observed 轨迹数 >= 下限",
    "4 EOL 未集中在最后一个 index",
    "5 EOL IQR > 0",
    "6 b_true 随时间总体递增",
    "7 无 NaN / Inf",
    "8 温度 / 电流不超物理硬界",
    "9 mission mode 高负荷排序可映射到 wear drive",
    "10 未读取任何 RUL / 模型指标",
    "11 early-EOL 未过度集中在 horizon 起始段",
    "12 初始健康段电流裕度充足",
)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相关 (只用 numpy, 避免为一个数引入 scipy 依赖)。"""
    def rank(a):
        return np.argsort(np.argsort(a)).astype(np.float64)
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def mode_severity_from_p95_reference(rec: dict, cfg: dict) -> tuple[list[str], dict]:
    """从 B1.1 的 p95 reference 逐模式算 g_duty -> 严酷度排序 (纯物理量)。

    为什么不沿用 B1 的 `physics_audit.json` 排序: 那是 rms 口径下的 g_duty。
    B1.1 换了 reference 口径, 逐模式 g_duty 随之变化, 沿用旧排序会让 Gate 9
    检验一个与本阶段标定无关的假设。
    """
    ref = rec["reference_p95"]
    per_mode = rec["reference_p95_per_mode"]
    wd = cfg["wear_drive"]
    gd = {}
    for m in MODES:
        drives = {k: np.array([float(per_mode[m][k])]) for k in cal.DRIVE_KEYS}
        gd[m] = float(cal.g_duty(drives, ref, wd)[0])
    order = sorted(MODES, key=lambda m: -gd[m])
    return order, gd


def eval_gates(res: dict, cfg: dict, mode_severity_order: list[str]) -> dict:
    """对一个 candidate 逐条评 §9 的 12 项。返回 {gate: (pass, detail)}。"""
    g = cfg["gate"]
    rows = res["per_traj"]
    n = len(rows)
    ff = res["failure_fraction"]
    cf = res["censored_fraction"]
    n_obs = res["n_event_observed"]
    out = {}

    lo, hi = float(g["failure_fraction_min"]), float(g["failure_fraction_max"])
    out[GATE_NAMES[0]] = (bool(lo <= ff <= hi),
                          f"ff={ff:.4f}, 带宽 [{lo}, {hi}] (§9 预先登记)")

    out[GATE_NAMES[1]] = (bool(cf >= float(g["censored_fraction_min"])),
                          f"censored_fraction={cf:.4f} >= {g['censored_fraction_min']}")

    out[GATE_NAMES[2]] = (bool(n_obs >= int(g["min_event_observed"])),
                          f"n_event_observed={n_obs} >= {g['min_event_observed']}")

    # gate 4: 删失轨迹的 eol_idx = n-1 是**约定**(不是观测到的失效), 只统计
    # event-observed 轨迹, 否则删失比例会被误算成"EOL 堆积"。
    obs_rows = [r for r in rows if r["failed"]]
    if obs_rows:
        frac_end = float(np.mean([r["at_horizon_end"] for r in obs_rows]))
        ok4 = bool(frac_end < float(g["eol_at_horizon_end_frac_max"]))
        d4 = (f"event-observed 中 EOL==horizon_end 占比 {frac_end:.4f} < "
              f"{g['eol_at_horizon_end_frac_max']} (n_obs={len(obs_rows)})")
    else:
        frac_end, ok4 = float("nan"), False
        d4 = "无 event-observed 轨迹, 无法评估 (n=0, 不伪造 0)"
    out[GATE_NAMES[3]] = (ok4, d4)

    if obs_rows:
        e_obs = np.array([r["eol_idx"] for r in obs_rows], dtype=float)
        iqr = float(np.percentile(e_obs, 75) - np.percentile(e_obs, 25))
        ok5 = bool(iqr > float(g["eol_iqr_min"]))
        d5 = f"event-observed EOL IQR = {iqr:.1f} > {g['eol_iqr_min']}"
    else:
        iqr, ok5, d5 = float("nan"), False, "无 event-observed 轨迹 (n=0)"
    out[GATE_NAMES[4]] = (ok5, d5)

    bmono = float(min(r["b_monotone_frac"] for r in rows))
    bratio = np.array([r["b_ratio"] for r in rows])
    ok6 = bool(bmono >= float(g["b_monotone_frac_min"]) and bratio.min() > 1.0)
    out[GATE_NAMES[5]] = (ok6, f"最小单调步占比 {bmono:.6f} >= "
                               f"{g['b_monotone_frac_min']}, b_final/b_init "
                               f"min={bratio.min():.3f} p50={np.median(bratio):.3f} "
                               f"max={bratio.max():.3f}")

    n_nan = sum(r["n_nan"] for r in rows)
    ok7 = bool(n_nan == 0 and all(r["all_finite"] for r in rows))
    out[GATE_NAMES[6]] = (ok7, f"NaN 总数 {n_nan}, all_finite "
                               f"{sum(r['all_finite'] for r in rows)}/{n}")

    Tmin = min(r["T_min"] for r in rows); Tmax = max(r["T_max"] for r in rows)
    Imax = max(r["Im_absmax"] for r in rows)
    ok8 = bool(Tmin > float(g["temperature_min_K"])
               and Tmax < float(g["temperature_max_K"])
               and Imax < float(g["current_absmax_A"]))
    out[GATE_NAMES[7]] = (ok8, f"T ∈ [{Tmin:.1f}, {Tmax:.1f}] K "
                               f"(界 [{g['temperature_min_K']}, "
                               f"{g['temperature_max_K']}]), |I_m|max = {Imax:.4f} A "
                               f"(界 {g['current_absmax_A']})")

    mcw = res["mode_conditioned_wear"]
    present = [m for m in mode_severity_order if mcw[m]["n_windows"] > 0]
    if len(present) >= 3:
        rank_sev = np.array([mode_severity_order.index(m) for m in present], float)
        meas = np.array([mcw[m]["mean_g_duty"] for m in present], float)
        rho = spearman(-rank_sev, meas)
        ok9 = bool(np.isfinite(rho) and rho > float(g["wear_ordering_min_spearman"]))
        d9 = (f"spearman(严酷度排序, 实测 mean g_duty) = {rho:.4f} > "
              f"{g['wear_ordering_min_spearman']}; 覆盖 mode {present}")
    else:
        rho, ok9 = float("nan"), False
        d9 = f"仅 {len(present)} 个 mode 有窗, 不足以评排序 (n={len(present)})"
    out[GATE_NAMES[8]] = (ok9, d9)

    out[GATE_NAMES[9]] = (bool(not res.get("used_rul_metric", False)
                               and not res.get("used_model_metric", False)),
                          "used_rul_metric=False, used_model_metric=False "
                          "(§17: gate 只看物理量)")

    # ---- Gate 11 (新增): early-EOL concentration ----
    # 分子只可能是 event-observed (censored 的 eol_idx = n-1 恒不满足 < 0.1·horizon),
    # 分母是全部 n 条 —— 这正是 §6 定义的 fraction(EOL < 0.1 * horizon)。
    ef = float(g["early_eol_horizon_frac"])
    emax = float(g["early_eol_frac_max"])
    n_early = sum(r["early_eol"] for r in rows)
    frac_early = float(n_early / n)
    ok11 = bool(frac_early < emax)
    out[GATE_NAMES[10]] = (ok11, f"fraction(EOL < {ef}×horizon) = {frac_early:.4f} "
                                 f"< {emax}  ({n_early}/{n} 条; 全部来自 "
                                 f"event-observed)")

    # ---- Gate 12 (新增): initial current margin ----
    # 逐轨迹先取初始健康窗内 p95(|I_m|)/Im_rated, 再对 trajectory-level ratio 取 p95。
    # §6 明令: 不使用单个极端时间点。
    imr = np.array([r["initial_margin_ratio"] for r in rows], dtype=float)
    mmax = float(g["initial_margin_ratio_max"])
    imr_p95 = float(np.percentile(imr, 95))
    ok12 = bool(imr_p95 <= mmax)
    out[GATE_NAMES[11]] = (ok12, f"initial_margin_ratio_p95 = {imr_p95:.4f} <= {mmax} "
                                 f"(逐轨迹健康窗 p95 -> 轨迹间 p95; "
                                 f"median={np.median(imr):.4f}, "
                                 f"max={imr.max():.4f}, "
                                 f"healthy_frac={g['initial_margin_healthy_frac']})")

    return {
        "gates": out,
        "eligible": all(v[0] for v in out.values()),
        "derived": {
            "failure_fraction": ff, "censored_fraction": cf,
            "n_event_observed": n_obs, "n_censored": res["n_censored"],
            "eol_frac_at_horizon_end_observed": frac_end,
            "eol_iqr_observed": iqr,
            "early_eol_fraction": frac_early, "n_early_eol": int(n_early),
            "initial_margin_ratio_p95": imr_p95,
            "initial_margin_ratio_median": float(np.median(imr)),
            "initial_margin_ratio_max": float(imr.max()),
            "b_monotone_frac_min": bmono,
            "b_ratio_p5_p50_p95": [float(np.percentile(bratio, q))
                                   for q in (5, 50, 95)],
            "T_min_K": Tmin, "T_max_K": Tmax,
            "T_p95_K": float(np.percentile([r["T_p95"] for r in rows], 95)),
            "Im_absmax_A": Imax,
            "Im_p95_A": float(np.percentile([r["Im_p95"] for r in rows], 95)),
            "n_nan_total": n_nan,
            "post_eol_rows_total": int(sum(r["n_post_eol_rows"] for r in rows)),
            "wear_ordering_spearman": rho,
            "hi_proxy_note": ("HI 动态范围在 §15 build_features 后正式评估; "
                              "此处用 b_final/b_init 作 friction-growth 分布代理"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel_basilisk_b11.yaml")
    ap.add_argument("--candidates",
                    default="checkpoints/basilisk_b11/candidates.json")
    ap.add_argument("--protocol",
                    default="checkpoints/basilisk_b11/protocol_hash.json")
    ap.add_argument("--json",
                    default="checkpoints/basilisk_b11/candidate_audit.json")
    ap.add_argument("--md", default="docs/basilisk_b11/candidate_report.md")
    a = ap.parse_args()

    cfg = cal.load_b11_config(a.config)
    cand = json.loads((ROOT / a.candidates).read_text(encoding="utf-8"))
    rec = json.loads((ROOT / a.protocol).read_text(encoding="utf-8"))
    order, mode_gd = mode_severity_from_p95_reference(rec, cfg)
    priority = list(cfg["candidates"]["priority"])

    print("=" * 78)
    print("Basilisk-B1.1 §9/§10 candidate 物理 Gate (12 条; 不使用任何 RUL / 模型指标)")
    print(f"  reference 口径 = {cand['reference_caliber']}  "
          f"derived b0_scale = {cand['b0_calibration']['b0_scale']:.9f}")
    print(f"  失效率带宽 (§9 预先登记) = [{cfg['gate']['failure_fraction_min']}, "
          f"{cfg['gate']['failure_fraction_max']}]  —— 不追求 65%")
    print(f"  新增 Gate 11 early-EOL < {cfg['gate']['early_eol_frac_max']}, "
          f"Gate 12 initial margin p95 <= {cfg['gate']['initial_margin_ratio_max']}")
    print(f"  优先顺序 = {' > '.join(priority)}")
    print(f"  mode 严酷度 (p95 口径重新推导) = {' > '.join(order)}")
    print("   " + "  ".join(f"{m}:{mode_gd[m]:.3f}" for m in order))
    print("=" * 78)

    audits = {}
    for cname in priority:
        res = dict(cand["candidates"][cname])
        res["used_rul_metric"] = cand["used_rul_metric"]
        res["used_model_metric"] = cand["used_model_metric"]
        audits[cname] = eval_gates(res, cfg, order)

    print(f"\n{'candidate':30s} {'h×':>4s} {'obs':>4s} {'cens':>5s} {'ff':>7s} "
          f"{'cf':>7s} {'early':>7s} {'margin':>7s} {'IQR':>9s} {'elig':>6s}")
    for cname in priority:
        c = cand["candidates"][cname]; d = audits[cname]["derived"]
        print(f"{cname:30s} {c['horizon_scale']:4.1f} {c['n_event_observed']:4d} "
              f"{c['n_censored']:5d} {c['failure_fraction']:7.4f} "
              f"{c['censored_fraction']:7.4f} {d['early_eol_fraction']:7.4f} "
              f"{d['initial_margin_ratio_p95']:7.4f} "
              f"{d['eol_iqr_observed']:9.1f} "
              f"{'YES' if audits[cname]['eligible'] else 'NO':>6s}")

    for cname in priority:
        print(f"\n-- {cname} 逐条 Gate --")
        for gname, (ok, detail) in audits[cname]["gates"].items():
            print(f"   [{'PASS' if ok else 'FAIL'}] {gname}\n          {detail}")

    # ---- §10 选择规则: 严格按 priority 取第一个 eligible ----
    selected = next((c for c in priority if audits[c]["eligible"]), None)
    print()
    if selected is None:
        verdict = "B11_CALIBRATION_FAIL"
        print(">> 三个 candidate 全部不通过 -> B11_CALIBRATION_FAIL, 停止")
        print("   §11 要求: 禁止改 b0_scale / Im_rated / threshold / 加 candidate / 重跑")
    else:
        verdict = "B11_CANDIDATE_SELECTED"
        print(f">> selected candidate = {selected}  "
              f"(优先顺序 {' > '.join(priority)} 中第一个 eligible)")
        skipped = priority[:priority.index(selected)]
        if skipped:
            print(f"   跳过 {skipped} 因其 gate 未全过 (见上); "
                  "选择规则纯粹按预先登记顺序, 不看任何模型表现, 也不看谁的 ff 更接近 65%")

    payload = {
        "stage": "BASILISK_B11_CANDIDATE_AUDIT",
        "config": a.config,
        "protocol_sha256": rec["protocol_sha256"],
        "calibration_core_hash": rec["calibration_core_hash"],
        "reference_caliber": cand["reference_caliber"],
        "b0_scale": float(cand["b0_calibration"]["b0_scale"]),
        "failure_fraction_band": [float(cfg["gate"]["failure_fraction_min"]),
                                  float(cfg["gate"]["failure_fraction_max"])],
        "band_rationale": ("§9 预先登记 35%-80%: 既保留足够 event-observed 轨迹, "
                           "又保留 >=15% censoring; 明确不追求 65%"),
        "new_gates_rationale": {
            "gate_11": ("B1 的 Gate 4/5 只防 EOL 右侧堆积, 不防左侧。B1 实测 "
                        "event-observed EOL 中位数 0.281 年 / 69.8% < 0.5 年 却通过了 "
                        "10 条中的 8 条 —— Gate 11 补上左侧约束。"),
            "gate_12": ("B1 失败最直接的指标 (p95(I_m at t=0)/Im_rated = 0.621 vs "
                        "analytic 0.289) 当时没有任何 Gate 约束。逐轨迹健康窗 p95 -> "
                        "轨迹间 p95, 不用单点极值。"),
        },
        "priority": priority,
        "mode_severity_order": order,
        "mode_severity_source": ("由 B1.1 p95 reference 逐模式 g_duty 重新推导 "
                                 "(不沿用 B1 的 rms 口径排序)"),
        "mode_reference_g_duty": mode_gd,
        "paired_trajectory_hash": cand["paired_trajectory_hash"],
        "crn_ok": cand["crn_ok"],
        "per_candidate": {
            c: {
                "horizon_scale": cand["candidates"][c]["horizon_scale"],
                "n_windows": cand["candidates"][c]["n_windows"],
                "horizon_years": cand["candidates"][c]["horizon_years"],
                "content_sha256": cand["candidates"][c]["content_sha256"],
                "eol_quantiles_all": cand["candidates"][c]["eol_quantiles_all"],
                "eol_quantiles_observed":
                    cand["candidates"][c]["eol_quantiles_observed"],
                "mode_conditioned_wear":
                    cand["candidates"][c]["mode_conditioned_wear"],
                "wall_clock_s": cand["candidates"][c]["wall_clock_s"],
                "eligible": audits[c]["eligible"],
                "gates": [{"name": g, "pass": bool(ok), "detail": d}
                          for g, (ok, d) in audits[c]["gates"].items()],
                "derived": audits[c]["derived"],
            } for c in priority},
        "selected_candidate": selected,
        "selection_rule": "按 config candidates.priority 取第一个全部 12 gate PASS 者",
        "used_rul_metric": False, "used_model_metric": False,
        "selected_before_any_RUL_training": True,
        "verdict": verdict,
    }
    jp = ROOT / a.json
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                  encoding="utf-8")
    print(f">> 写入 {jp.relative_to(ROOT)}")
    write_md(ROOT / a.md, payload, cand, cfg)
    print(f">> 报告写入 {(ROOT / a.md).relative_to(ROOT)}")
    print(f">> {verdict}")
    return 0 if selected is not None else 1


def write_md(path: Path, p: dict, cand: dict, cfg: dict) -> None:
    pr = p["priority"]
    d0 = cand["b0_calibration"]
    dr = cand["reference_duty_p95"]
    drr = cand["reference_duty_rms_for_comparison"]
    L = ["# Basilisk-B1.1 §9/§10 —— Candidate 物理 Gate 报告 (12 条)", "",
         "本文件由 `scripts/basilisk_b11/audit_candidates.py` 自动生成。", "",
         "> **§17 纪律**: 本报告的每一个数字都是物理量。全程未使用 RUL RMSE / "
         "correlation / PH / warning lead / transfer gain / 任何神经网络指标",
         f"> (`used_rul_metric = {p['used_rul_metric']}`, "
         f"`used_model_metric = {p['used_model_metric']}`)。", "",
         f"协议 hash `{p['protocol_sha256']}`; "
         f"`calibration_core_hash` `{p['calibration_core_hash']}`。", "",
         "## 1. 标定口径 (B1.1 的唯一改动)", "",
         "| 量 | B1 (rms) | B1.1 (p95) |", "|---|---|---|",
         f"| `speed_util_ref` | {drr['speed_util']:.9f} | "
         f"**{dr['speed_util']:.9f}** |",
         f"| `torque_util_ref` | {drr['torque_util']:.9f} | "
         f"**{dr['torque_util']:.9f}** |",
         f"| `b0_scale` | "
         f"{cand['b0_calibration_rms_for_comparison']['b0_scale']:.9f} | "
         f"**{d0['b0_scale']:.9f}** |", "",
         "`zero_crossing_rate` / `maneuver_fraction` 两个 reference 分量两口径下相同",
         "(非幅值量)。wear 数学结构、权重、指数、`Im_rated`、failure criterion 全部未动。",
         "", "## 2. 预先登记的 candidate 参数表 (§7, 冻结, 无 D/E)", "",
         "| candidate | horizon_scale | n_windows | 任务期 (年) | "
         "退化参数 | content sha256 |", "|---|---|---|---|---|---|"]
    for c in pr:
        d = p["per_candidate"][c]
        L.append(f"| {c} | ×{d['horizon_scale']} | {d['n_windows']} | "
                 f"{d['horizon_years']:.2f} | 三者完全相同 | "
                 f"`{d['content_sha256'][:16]}` |")
    L += ["", f"共用退化尺度: `b0_scale` = {d0['b0_scale']:.9f}, "
          f"`speed_util_ref(p95)` = {d0['speed_util_ref']:.9f}, "
          f"`tau_years <- tau_years_raw / g_duty`, "
          f"`omega0 <- speed_util_traj · {d0['omega_rated_rad_s']:.2f}`。", "",
          f"§8 CRN: `paired_trajectory_hash` = `{p['paired_trajectory_hash']}` "
          f"(三 candidate 共享), 前缀逐位比对 `crn_ok = {p['crn_ok']}`。", "",
          "## 3. 60-traj failure / EOL / 新增两指标", "",
          "| candidate | observed | censored | ff | censored_frac | "
          "early-EOL frac | initial margin p95 | IQR(obs) |",
          "|---|---|---|---|---|---|---|---|"]
    for c in pr:
        dv = p["per_candidate"][c]["derived"]
        L.append(f"| {c} | {dv['n_event_observed']} | {dv['n_censored']} | "
                 f"{dv['failure_fraction']:.4f} | {dv['censored_fraction']:.4f} | "
                 f"{dv['early_eol_fraction']:.4f} | "
                 f"{dv['initial_margin_ratio_p95']:.4f} | "
                 f"{dv['eol_iqr_observed']:.1f} |")
    L += ["", "EOL 分位数 (**只在 event-observed 上算** —— 删失轨迹的 `eol_idx = n-1` ",
          "是约定而非观测, 混入会伪造 EOL 指标):", "",
          "| candidate | p5 | p25 | p50 | p75 | p95 |", "|---|---|---|---|---|---|"]
    for c in pr:
        q = p["per_candidate"][c]["eol_quantiles_observed"]
        if q is None:
            L.append(f"| {c} | 无 event-observed 轨迹 (n=0, 不伪造数值) | | | | |")
        else:
            L.append(f"| {c} | " + " | ".join(
                f"{q[f'p{v}']:.0f}" for v in (5, 25, 50, 75, 95)) + " |")
    L += ["", "## 4. 物理范围 / friction growth / mode-conditioned wear", "",
          "| candidate | b_final/b_init p5/p50/p95 | T [min,max] K | "
          "T p95 | \\|I_m\\|max A | I_m p95 | NaN | post-EOL rows |",
          "|---|---|---|---|---|---|---|---|"]
    for c in pr:
        dv = p["per_candidate"][c]["derived"]
        br = "/".join(f"{v:.2f}" for v in dv["b_ratio_p5_p50_p95"])
        L.append(f"| {c} | {br} | [{dv['T_min_K']:.1f}, {dv['T_max_K']:.1f}] | "
                 f"{dv['T_p95_K']:.1f} | {dv['Im_absmax_A']:.4f} | "
                 f"{dv['Im_p95_A']:.4f} | {dv['n_nan_total']} | "
                 f"{dv['post_eol_rows_total']} |")
    L += ["", "mode-conditioned wear rate (窗平均 `g_duty`; 空组记 NaN + n=0, 不伪造 0):",
          "", "| candidate | " + " | ".join(MODES) + " |",
          "|---|" + "---|" * len(MODES)]
    for c in pr:
        mcw = p["per_candidate"][c]["mode_conditioned_wear"]
        L.append(f"| {c} | " + " | ".join(
            f"{mcw[m]['mean_g_duty']:.4f} (n={mcw[m]['n_windows']})"
            for m in MODES) + " |")
    L += ["", f"严酷度排序: **{' > '.join(p['mode_severity_order'])}**",
          f"({p['mode_severity_source']}; reference g_duty: " +
          ", ".join(f"`{m}`={p['mode_reference_g_duty'][m]:.4f}"
                    for m in p["mode_severity_order"]) + ")", "",
          "## 5. Gate 逐条结果 (§9, **12 条**全过才 eligible)", ""]
    for c in pr:
        d = p["per_candidate"][c]
        L += [f"### {c} —— eligible = **{d['eligible']}**", "",
              "| # | gate | 结果 | 细节 |", "|---|---|---|---|"]
        for i, g in enumerate(d["gates"], 1):
            L.append(f"| {i} | {g['name']} | {'PASS' if g['pass'] else 'FAIL'} | "
                     f"{g['detail']} |")
        L.append("")
    ng = p["new_gates_rationale"]
    L += ["### 新增两条 Gate 的动机", "",
          f"- **Gate 11**: {ng['gate_11']}",
          f"- **Gate 12**: {ng['gate_12']}", "",
          "## 6. 选择结果", "",
          f"- 失效率带宽 (§9 预先登记): **{p['failure_fraction_band']}**",
          f"- 依据: {p['band_rationale']}",
          f"- 优先顺序 (§10): **{' > '.join(pr)}**",
          f"- 选择规则: {p['selection_rule']}",
          f"- **selected_candidate = `{p['selected_candidate']}`**",
          f"- `selected_before_any_RUL_training = "
          f"{p['selected_before_any_RUL_training']}`",
          f"- 结论: **{p['verdict']}**", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
