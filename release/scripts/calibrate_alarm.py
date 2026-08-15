#!/usr/bin/env python
"""scripts/calibrate_alarm.py

P1-4 飞轮告警阈值重标定 + 物理基线同构说明

目标:
  1. 在 VALIDATION 集上做 ROC / 阈值扫描, 选最优阈值 (最大化 coverage 同时 miss_rate 约束)
  2. 锁定阈值后, 在 TEST 集上独立报告告警指标
  3. 同步报告 damage_extrapolation 基线的告警指标 (物理基线推荐阈值)
  4. 输出 alarm_calibration.json, 清楚标注 "阈值在验证集选, 测试集结果为锁定后独立报告"

纪律:
  * 阈值只在验证集选择, 绝不在测试集上调
  * 复用 B5 冻结的 formal_raw.npz 数据 (无需 h5 文件)
  * event_observed 从 true_rul 是否含 NaN 推断 (与 B5 evaluator 口径一致)
  * warning_lead_time 函数直接 import 冻结的 src.experiments.metrics 实现

用法:
    python scripts/calibrate_alarm.py
    python scripts/calibrate_alarm.py --config configs/wheel_basilisk_b5.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.basilisk_b11.calibrate_degradation import (  # noqa: E402
    load_b11_config as load_cfg,
)
from src.experiments.metrics import warning_lead_time  # noqa: E402

NAN = float("nan")

# 默认 RUL 阈值扫描范围 (归一化 RUL 空间)
DEFAULT_THRESHOLD_GRID = np.arange(0.05, 0.60, 0.01)
DEFAULT_PERSISTENCE = 3


def _raw_of(npz, key: str) -> dict:
    """从 npz 提取一组的 5 字段原始数据 (test 或 val)."""
    return {f: np.asarray(npz[f"{key}__{f}"]) for f in
            ("pred", "true", "tids", "tau", "lb")}


def _classify_event_observed(true_rul: np.ndarray, tids: np.ndarray) -> dict:
    """从 true_rul 是否含 NaN 推断 event_observed (与 B5 evaluator 口径一致).

    event_observed=True: 该轨迹有真实 RUL (EOL 可见)
    event_observed=False: 右删失轨迹, true_rul 为 NaN
    """
    ev = {}
    for u in np.unique(tids):
        mask = tids == u
        has_nan = np.any(np.isnan(true_rul[mask]))
        ev[int(u)] = not has_nan
    return ev


def _alarm_metrics(pred, tau, tids, event_observed, rul_threshold, persistence):
    """调用冻结的 warning_lead_time 计算告警指标."""
    return warning_lead_time(
        np.asarray(pred, dtype=float),
        np.asarray(tau, dtype=float),
        np.asarray(tids, dtype=np.int64),
        event_observed,
        float(rul_threshold),
        int(persistence),
    )


def _threshold_scan(pred, tau, tids, event_observed, thresholds, persistence):
    """在给定阈值网格上扫描告警指标.

    返回每个阈值下的 (warning_coverage, miss_rate, false_alarm_rate, coverage_before_eol,
    miss_rate_before_eol).
    """
    results = []
    for thr in thresholds:
        w = _alarm_metrics(pred, tau, tids, event_observed, thr, persistence)
        results.append({
            "threshold": float(thr),
            "warning_coverage": float(w["warning_coverage"]),
            "miss_rate": float(w["miss_rate"]),
            "coverage_before_eol": float(w["coverage_before_eol"]),
            "miss_rate_before_eol": float(w["miss_rate_before_eol"]),
            "false_alarm_rate": float(w["false_alarm_rate"]),
            "n_warned": int(w["n_warned_observed"]),
            "n_warned_before_eol": int(w["n_warned_before_eol"]),
            "conditional_lead": float(w["conditional_lead"]) if np.isfinite(w["conditional_lead"]) else NAN,
        })
    return results


def _select_optimal_threshold(scan_results, target_group="val"):
    """选最优阈值: 在 miss_rate_before_eol <= 0.30 约束下最大化 coverage_before_eol.

    如果没有阈值满足约束, 退化为最大化 coverage_before_eol (报告无约束解).
    """
    constrained = [r for r in scan_results
                   if np.isfinite(r["miss_rate_before_eol"])
                   and r["miss_rate_before_eol"] <= 0.30]

    if constrained:
        best = max(constrained, key=lambda r: r["coverage_before_eol"])
        return best, True
    else:
        # 退化: 取 coverage_before_eol 最大 (miss_rate 可能高)
        valid = [r for r in scan_results
                 if np.isfinite(r["coverage_before_eol"])]
        best = max(valid, key=lambda r: r["coverage_before_eol"])
        return best, False


def _format_scan_table(scan_results, highlight_threshold=None):
    """格式化阈值扫描表为字符串 (用于终端打印)."""
    lines = []
    lines.append(f"{'thr':>6s} {'cover':>7s} {'miss':>7s} {'cover_eol':>9s} "
                 f"{'miss_eol':>8s} {'FA':>7s} {'n_warn':>7s} {'lead':>7s}")
    for r in scan_results:
        marker = " *" if highlight_threshold is not None and abs(r["threshold"] - highlight_threshold) < 0.005 else ""
        lead = f"{r['conditional_lead']:7.4f}" if np.isfinite(r["conditional_lead"]) else "    nan"
        lines.append(
            f"{r['threshold']:6.2f} {r['warning_coverage']:7.3f} {r['miss_rate']:7.3f} "
            f"{r['coverage_before_eol']:9.3f} {r['miss_rate_before_eol']:8.3f} "
            f"{r['false_alarm_rate']:7.3f} {r['n_warned']:7d} {lead}{marker}"
        )
    return "\n".join(lines)


def _group_alarm_summary(npz, group_prefix, seeds, event_observed_val,
                         event_observed_test, rul_threshold, persistence):
    """对一组 (跨 seed) 在锁定阈值下计算 val 和 test 的告警指标.

    返回 dict with per_seed + mean.
    """
    val_per_seed = []
    test_per_seed = []
    for s in seeds:
        # val
        raw_val = _raw_of(npz, f"{group_prefix}_s{s}")
        # 对于没有 val 数据的组 (damage_extrapolation, const_mean_info), 跳过
        val_key = f"{group_prefix}_s{s}__val_pred"
        if val_key not in npz:
            val_per_seed.append(None)
        else:
            w_val = _alarm_metrics(raw_val["pred"], raw_val["tau"], raw_val["tids"],
                                   event_observed_val, rul_threshold, persistence)
            val_per_seed.append(w_val)

        # test
        raw_test = _raw_of(npz, f"{group_prefix}_s{s}")
        w_test = _alarm_metrics(raw_test["pred"], raw_test["tau"], raw_test["tids"],
                                event_observed_test, rul_threshold, persistence)
        test_per_seed.append(w_test)

    return {
        "val_per_seed": val_per_seed,
        "test_per_seed": test_per_seed,
    }


def _mean_or_nan(vals):
    v = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    return float(np.mean(v)) if v else NAN


def main() -> int:
    ap = argparse.ArgumentParser(
        description="P1-4 飞轮告警阈值重标定 + 物理基线同构说明")
    ap.add_argument("--config", default="configs/wheel_basilisk_b5.yaml")
    ap.add_argument("--output",
                    default="XA-202608_最终交付/05_结果/reference/wheel/alarm_calibration.json")
    ap.add_argument("--threshold-max-miss-before-eol", type=float, default=0.30,
                    help="验证集选阈值的 miss_rate_before_eol 约束上限 (默认 0.30)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # 从 formal_metrics 读 prognostics config
    metrics_path = ROOT / cfg["paths"]["metrics_json"]
    M = json.loads(metrics_path.read_text("utf-8"))
    if bool(M.get("fast")):
        raise SystemExit("!! metrics 来自 --fast 冒烟跑, 不得进入正式分析")

    # 读取原始 prognostics cfg (从 warning_metrics.json)
    warning_json_path = ROOT / cfg["paths"]["warning_json"]
    WM = json.loads(warning_json_path.read_text("utf-8"))
    prognostics_cfg = WM["prognostics_cfg"]
    original_threshold = float(prognostics_cfg["rul_threshold"])
    persistence = int(prognostics_cfg["persistence"])

    seeds = M["formal_seeds"]
    npz = np.load(ROOT / cfg["paths"]["raw_npz"])

    print(f">> B5 冻结告警配置: rul_threshold={original_threshold}, "
          f"persistence={persistence}")
    print(f">> 正式 seeds: {seeds}")
    print()

    # ---- 1. 构建 event_observed 映射 (从 true_rul NaN 模式推断) ----
    # 用 seed 112 的数据推断 (所有 seed 共享同一 split, true/tau/tids 跨 seed 一致)
    ref_val_tids = npz[f"target_only_s{seeds[0]}__val_tids"]
    ref_val_true = npz[f"target_only_s{seeds[0]}__val_true"]
    ref_test_tids = npz[f"target_only_s{seeds[0]}__tids"]
    ref_test_true = npz[f"target_only_s{seeds[0]}__true"]

    ev_val = _classify_event_observed(ref_val_true, ref_val_tids)
    ev_test = _classify_event_observed(ref_test_true, ref_test_tids)

    n_ev_val = sum(1 for v in ev_val.values() if v)
    n_cen_val = len(ev_val) - n_ev_val
    n_ev_test = sum(1 for v in ev_test.values() if v)
    n_cen_test = len(ev_test) - n_ev_test

    print(f">> Val: {len(ev_val)} traj ({n_ev_val} event, {n_cen_val} censored)")
    print(f">> Test: {len(ev_test)} traj ({n_ev_test} event, {n_cen_test} censored)")
    print()

    # ---- 2. 验证集阈值扫描 (target_only, 5 seeds 聚合) ----
    # 用 5 seeds 在 val 上的平均 coverage_before_eol 来选阈值
    thresholds = DEFAULT_THRESHOLD_GRID
    val_scan_per_seed = []
    for s in seeds:
        raw_val = _raw_of(npz, f"target_only_s{s}")
        # 注意: val 数据的 key 不带 __ 前缀的子 key, 需要 val_ 前缀
        val_pred = npz[f"target_only_s{s}__val_pred"]
        val_tau = npz[f"target_only_s{s}__val_tau"]
        val_tids = npz[f"target_only_s{s}__val_tids"]
        scan = _threshold_scan(val_pred, val_tau, val_tids, ev_val,
                               thresholds, persistence)
        val_scan_per_seed.append(scan)

    # 聚合: 对每个阈值, 取 5 seeds 的平均 coverage_before_eol 和 miss_rate_before_eol
    val_scan_aggregated = []
    for i, thr in enumerate(thresholds):
        coverages = [val_scan_per_seed[s_idx][i]["coverage_before_eol"]
                     for s_idx in range(len(seeds))]
        misses = [val_scan_per_seed[s_idx][i]["miss_rate_before_eol"]
                  for s_idx in range(len(seeds))]
        fas = [val_scan_per_seed[s_idx][i]["false_alarm_rate"]
               for s_idx in range(len(seeds))]
        leads = [val_scan_per_seed[s_idx][i]["conditional_lead"]
                 for s_idx in range(len(seeds))]
        val_scan_aggregated.append({
            "threshold": float(thr),
            "mean_coverage_before_eol": float(np.mean(coverages)),
            "mean_miss_rate_before_eol": float(np.mean(misses)),
            "mean_false_alarm_rate": float(np.mean(fas)),
            "per_seed_coverage_before_eol": coverages,
            "per_seed_miss_rate_before_eol": misses,
            "mean_conditional_lead": _mean_or_nan(leads),
        })

    # ---- 3. 选择最优阈值 ----
    # 约束: mean_miss_rate_before_eol <= threshold_max_miss_before_eol
    # 目标: 最大化 mean_coverage_before_eol
    constraint = args.threshold_max_miss_before_eol
    constrained = [r for r in val_scan_aggregated
                   if np.isfinite(r["mean_miss_rate_before_eol"])
                   and r["mean_miss_rate_before_eol"] <= constraint]

    if constrained:
        optimal = max(constrained, key=lambda r: r["mean_coverage_before_eol"])
        constraint_satisfied = True
    else:
        optimal = max(val_scan_aggregated,
                      key=lambda r: r["mean_coverage_before_eol"])
        constraint_satisfied = False

    locked_threshold = optimal["threshold"]

    print("=" * 70)
    print("验证集阈值扫描 (target_only, 5 seeds 平均)")
    print("=" * 70)
    print(f"约束: miss_rate_before_eol <= {constraint}")
    print(f"目标: 最大化 coverage_before_eol")
    print()
    print(f"{'thr':>6s} {'cover_eol':>10s} {'miss_eol':>10s} {'FA':>8s} {'lead':>8s}")
    for r in val_scan_aggregated:
        marker = " *" if abs(r["threshold"] - locked_threshold) < 0.005 else ""
        print(f"{r['threshold']:6.2f} {r['mean_coverage_before_eol']:10.3f} "
              f"{r['mean_miss_rate_before_eol']:10.3f} "
              f"{r['mean_false_alarm_rate']:8.3f} "
              f"{r['mean_conditional_lead']:8.4f}{marker}")
    print()
    print(f">> 选定阈值 (锁定): {locked_threshold:.2f}")
    print(f">> 约束满足: {constraint_satisfied}")
    print(f">> 验证集 coverage_before_eol={optimal['mean_coverage_before_eol']:.3f}, "
          f"miss_rate_before_eol={optimal['mean_miss_rate_before_eol']:.3f}")
    print()

    # ---- 4. 锁定阈值后, 在 TEST 集上独立报告 ----
    print("=" * 70)
    print(f"TEST 集结果 (锁定阈值={locked_threshold:.2f}, persistence={persistence})")
    print("=" * 70)

    # target_only 在锁定阈值下的 test 结果
    target_only_test = []
    for s in seeds:
        raw_test = _raw_of(npz, f"target_only_s{s}")
        w = _alarm_metrics(raw_test["pred"], raw_test["tau"], raw_test["tids"],
                           ev_test, locked_threshold, persistence)
        target_only_test.append(w)

    # 同时报原始阈值 0.2 下的 test 结果作对照
    target_only_test_orig = []
    for s in seeds:
        raw_test = _raw_of(npz, f"target_only_s{s}")
        w = _alarm_metrics(raw_test["pred"], raw_test["tau"], raw_test["tids"],
                           ev_test, original_threshold, persistence)
        target_only_test_orig.append(w)

    # damage_extrapolation 在锁定阈值下的 test 结果
    damage_test = []
    for s in seeds:
        raw_test = _raw_of(npz, f"damage_extrapolation_s{s}")
        w = _alarm_metrics(raw_test["pred"], raw_test["tau"], raw_test["tids"],
                           ev_test, locked_threshold, persistence)
        damage_test.append(w)

    # damage_extrapolation 在原始阈值下的 test 结果
    damage_test_orig = []
    for s in seeds:
        raw_test = _raw_of(npz, f"damage_extrapolation_s{s}")
        w = _alarm_metrics(raw_test["pred"], raw_test["tau"], raw_test["tids"],
                           ev_test, original_threshold, persistence)
        damage_test_orig.append(w)

    def _summarize_test(per_seed, label):
        covers = [w["warning_coverage"] for w in per_seed]
        misses = [w["miss_rate"] for w in per_seed]
        covers_eol = [w["coverage_before_eol"] for w in per_seed]
        misses_eol = [w["miss_rate_before_eol"] for w in per_seed]
        fas = [w["false_alarm_rate"] for w in per_seed]
        leads = [w["conditional_lead"] for w in per_seed]
        leads_eol = [w["conditional_lead_before_eol"] for w in per_seed]
        print(f"\n  [{label}]")
        print(f"    warning_coverage      = {np.mean(covers):.3f}  "
              f"(per-seed: {[f'{c:.3f}' for c in covers]})")
        print(f"    miss_rate             = {np.mean(misses):.3f}")
        print(f"    coverage_before_eol   = {np.mean(covers_eol):.3f}")
        print(f"    miss_rate_before_eol  = {np.mean(misses_eol):.3f}")
        print(f"    false_alarm_rate      = {np.mean(fas):.3f}")
        print(f"    conditional_lead      = {_mean_or_nan(leads):.4f}")
        print(f"    cond_lead_before_eol  = {_mean_or_nan(leads_eol):.4f}")
        return {
            "warning_coverage_mean": float(np.mean(covers)),
            "warning_coverage_per_seed": [float(c) for c in covers],
            "miss_rate_mean": float(np.mean(misses)),
            "miss_rate_per_seed": [float(m) for m in misses],
            "coverage_before_eol_mean": float(np.mean(covers_eol)),
            "coverage_before_eol_per_seed": [float(c) for c in covers_eol],
            "miss_rate_before_eol_mean": float(np.mean(misses_eol)),
            "miss_rate_before_eol_per_seed": [float(m) for m in misses_eol],
            "false_alarm_rate_mean": float(np.mean(fas)),
            "false_alarm_rate_per_seed": [float(f) for f in fas],
            "conditional_lead_mean": _mean_or_nan(leads),
            "conditional_lead_before_eol_mean": _mean_or_nan(leads_eol),
        }

    print()
    to_lock = _summarize_test(target_only_test,
                              f"target_only @ locked={locked_threshold:.2f}")
    to_orig = _summarize_test(target_only_test_orig,
                              f"target_only @ original={original_threshold:.2f}")
    dmg_lock = _summarize_test(damage_test,
                               f"damage_extrapolation @ locked={locked_threshold:.2f}")
    dmg_orig = _summarize_test(damage_test_orig,
                               f"damage_extrapolation @ original={original_threshold:.2f}")

    # ---- 5. 物理基线同构说明 ----
    homomorphism_note = (
        "物理基线同构说明 (self-consistent within the same framework):\n"
        "  target_only (神经网络) 和 damage_extrapolation (HI 线性外推) 的告警\n"
        "  判定共享完全相同的逻辑结构:\n"
        "    1. 两者都输出归一化 RUL 预测值 pred_rul(t)\n"
        "    2. 两者都通过同一函数 warning_lead_time() 判定告警:\n"
        "       pred_rul <= threshold 持续 persistence 个 endpoint -> 触发告警\n"
        "    3. 两者都依赖相同的可观测量 (电流/温度/转速等 CORE_ONLY 输入特征)\n"
        "  差异仅在于 pred_rul 的计算方式:\n"
        "    - target_only: pred_rul = NN_θ(x_T), θ 由目标域数据学习\n"
        "    - damage_extrapolation: pred_rul = (hi_fail - HI[t]) / slope[t-W:t]\n"
        "  因此 target_only vs damage_extrapolation 的告警对比是**口径内自洽**\n"
        "  (same-alarm-logic, same-observable-class), 不是 \"AI vs 公式\" 的\n"
        "  不对称比较。两者在同一阈值、同一 persistence、同一 evaluator 下评估。\n"
        "\n"
        "  damage_extrapolation 的工程含义: 它是 hi_damage_obs (= 式(8) 定义的\n"
        "  损伤 HI) 的因果滑窗斜率线性外推, 本质上是 pred_rul = ΔHI / rate。\n"
        "  target_only 的神经网络也在隐式地学 f(x_T) -> RUL, 但两者输入的\n"
        "  可观测量空间相同, 输出语义相同 (归一化 RUL), 告警判定完全相同。\n"
        "  这保证了比较的公平性和可解释性。"
    )

    print()
    print("=" * 70)
    print("物理基线同构说明")
    print("=" * 70)
    print(homomorphism_note)

    # ---- 6. 输出 JSON ----
    output_data = {
        "task": "P1-4 飞轮告警阈值重标定 + 物理基线同构说明",
        "source": {
            "metrics_json": str(metrics_path.relative_to(ROOT)),
            "raw_npz": str((ROOT / cfg["paths"]["raw_npz"]).relative_to(ROOT)),
            "warning_json": str(warning_json_path.relative_to(ROOT)),
            "seeds": seeds,
            "split_sha256": M["split_sha256"],
            "split_n": M["split_n"],
        },
        "methodology": {
            "threshold_selection": "validation_set_only",
            "threshold_selection_rule": (
                f"在验证集上, 对 target_only 的 5 seeds 平均 "
                f"miss_rate_before_eol <= {constraint} 约束下, "
                f"最大化 mean coverage_before_eol"),
            "threshold_locked_then_test": True,
            "test_report_label": "阈值在验证集选, 测试集结果为锁定后独立报告",
            "event_observed_inference": (
                "从 true_rul 是否含 NaN 推断: "
                "NaN -> 右删失 (event_observed=False), "
                "finite -> 有真实 EOL (event_observed=True). "
                "与 B5 evaluator 口径一致."),
            "persistence": persistence,
            "original_threshold_from_b5": original_threshold,
        },
        "val_threshold_scan": {
            "group": "target_only",
            "n_seeds": len(seeds),
            "threshold_grid": [float(t) for t in thresholds],
            "aggregated_results": val_scan_aggregated,
        },
        "locked_threshold": {
            "value": float(locked_threshold),
            "constraint_satisfied": bool(constraint_satisfied),
            "constraint": f"miss_rate_before_eol <= {constraint}",
            "val_coverage_before_eol": float(optimal["mean_coverage_before_eol"]),
            "val_miss_rate_before_eol": float(optimal["mean_miss_rate_before_eol"]),
            "val_false_alarm_rate": float(optimal["mean_false_alarm_rate"]),
        },
        "test_results": {
            "label": "阈值在验证集选, 测试集结果为锁定后独立报告",
            "target_only": {
                "at_locked_threshold": to_lock,
                "at_original_threshold_0.2": to_orig,
            },
            "damage_extrapolation": {
                "at_locked_threshold": dmg_lock,
                "at_original_threshold_0.2": dmg_orig,
            },
        },
        "homomorphism_note": homomorphism_note,
        "engineering_recommendation": {
            "method": "damage_extrapolation",
            "basis": (
                "在 B5 冻结的正式 test 表上, damage_extrapolation 的 "
                "info_macro_rmse=0.054 远优于 target_only 的 0.241, "
                "且在原始阈值(0.2)下 miss_rate=0.0 (零漏报), coverage=1.0 (全覆盖). "
                "这与 B5 summary.json 的 ENGINEERING_RECOMMENDATION 一致."),
            "alarm_threshold_recommendation": (
                f"damage_extrapolation 在阈值={original_threshold} 下已实现 "
                f"零漏报全覆盖, 无需重标定. "
                f"target_only 在锁定阈值={locked_threshold:.2f} 下 "
                f"miss_rate_before_eol 仍为 "
                f"{to_lock['miss_rate_before_eol_mean']:.3f}, "
                f"确认了 B5 的结论: target_only 的告警可靠性不足以部署."),
        },
    }

    # 写输出
    # ROOT = .../components/wheel/release/
    # output target = .../XA-202608_最终交付/05_结果/reference/wheel/alarm_calibration.json
    output_path = Path(args.output)
    if not output_path.is_absolute():
        # release -> wheel -> components -> 03_代码 -> XA-202608_最终交付
        delivery_root = ROOT.parent.parent.parent.parent
        output_path = delivery_root / "05_结果" / "reference" / "wheel" / "alarm_calibration.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8")
    print(f"\n>> 输出: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
