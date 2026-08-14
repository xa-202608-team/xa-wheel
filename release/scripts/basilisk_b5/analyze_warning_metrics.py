#!/usr/bin/env python
"""scripts/basilisk_b5/analyze_warning_metrics.py

BASILISK-B5 §13 —— warning / prognostics 指标。

复用已冻结的 S4 / Basilisk evaluator (`warning_lead_time` / `prognostic_horizon` /
`alpha_lambda_accuracy` / `convergence_metric`), 一行不改。

§13 纪律:
  * **NaN 必须保留 NaN, 并同时给出 n_evaluable。绝不转成 0。**
  * 不得只统计成功检测而隐藏 miss rate —— coverage 与 miss 成对出现。
  * 右删失轨迹没有真 EOL, PH / alpha-lambda 保持 NaN, 绝不伪造 EOL 指标。

用法:
    python scripts/basilisk_b5/analyze_warning_metrics.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.basilisk_b11.calibrate_degradation import (  # noqa: E402
    load_b11_config as load_cfg,
)
from scripts.basilisk_b2.eval_b2 import b2_endpoint_axis  # noqa: E402
from scripts.basilisk_b2.run_gate import warning_valid  # noqa: E402
from scripts.basilisk_b4x.diagnose_transfer_stability import (  # noqa: E402
    eol_metrics,
)
from src.experiments.metrics import (  # noqa: E402
    convergence_metric,
    prognostics_cfg,
    warning_lead_time,
)

NAN = float("nan")

ALL_GROUPS = ("target_only", "source_finetune", "source_mmd_finetune",
              "const_mean_info", "damage_extrapolation")

NAN_DISCIPLINE = (
    "NaN 一律保留 NaN, 并同时给出 n_evaluable / eligible_count。"
    "绝不把 NaN 转成 0 —— 不可评估与指标恰好为 0 是两件完全不同的事, "
    "把前者写成后者会让读表的人以为 PH = 0 (预警毫无提前量), "
    "而真相是这条轨迹右删失、根本没有真 EOL 可比。"
)

MISS_DISCIPLINE = (
    "coverage 与 miss rate 必须成对出现。"
    "不得只报成功检测数而隐藏 miss rate —— 那会把 warning 的可用性夸大。"
    "warning valid 只说明这组指标口径自洽 (coverage + miss = 1 等), "
    "**不等于 warning 可用**: B2.1 上 miss rate 高达 0.514–0.686 时指标依然 valid。"
)

CENSORED_DISCIPLINE = (
    "右删失 test 轨迹不得伪造 EOL 指标。"
    "PH / alpha-lambda 只在 event-observed 轨迹上计算, 删失轨迹计入 n_censored, "
    "不进分子也不进分母。"
)


def _raw_of(npz, key: str) -> dict:
    return {f: np.asarray(npz[f"{key}__{f}"]) for f in
            ("pred", "true", "tids", "tau", "lb")}


def _warning_block(raw: dict, axis: dict, pc: dict) -> dict:
    """一组的完整 warning + prognostics 块。"""
    tids = np.asarray(raw["tids"], np.int64)
    ev = {int(u): bool(axis[int(u)]["event_observed"]) for u in np.unique(tids)}
    w = warning_lead_time(np.asarray(raw["pred"], float),
                          np.asarray(raw["tau"], float), tids, ev,
                          pc["rul_threshold"], pc["persistence"])
    e = eol_metrics(raw["pred"], raw["true"], raw["tau"], tids, axis, pc)
    cv = convergence_metric(raw["pred"], raw["true"], raw["tau"], tids,
                            late_from=float(pc["late_from"]))
    ok, det = warning_valid(w)
    n_ev = int(sum(1 for v in ev.values() if v))
    return {
        "warning": dict(w),
        "warning_valid": bool(ok),
        "warning_valid_detail": det,
        "prognostic_horizon": {
            "macro_ph": e["macro_ph"],
            "n_evaluable": int(e["ph_n_observable"]),
            "n_censored_excluded": int(e["ph_n_censored"]),
        },
        "alpha_lambda": e["alpha_lambda"],
        "convergence": {"macro": cv.get("macro_convergence"),
                        "late_convergence": cv.get("macro_late_convergence"),
                        "n_evaluable": len(cv.get("per_trajectory", {}))},
        "n_event_observed": n_ev,
        "n_censored": int(len(ev) - n_ev),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel_basilisk_b5.yaml")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b5 = cfg["b5"]
    wc = b5["warning"]
    if not bool(wc["forbid_nan_to_zero"]):
        raise SystemExit("!! forbid_nan_to_zero 必须为 true (§13)")
    if not bool(wc["forbid_hiding_miss_rate"]):
        raise SystemExit("!! forbid_hiding_miss_rate 必须为 true (§13)")

    M = json.loads((ROOT / cfg["paths"]["metrics_json"]).read_text("utf-8"))
    if bool(M.get("fast")):
        raise SystemExit("!! metrics 来自 --fast 冒烟跑, 不得进入正式分析")
    npz = np.load(ROOT / cfg["paths"]["raw_npz"])
    axis = b2_endpoint_axis(cfg)
    pc = prognostics_cfg(cfg)

    rows = M["per_seed"]
    seeds = [int(r["seed"]) for r in rows]
    per_seed: list[dict] = []
    for r in rows:
        seed = int(r["seed"])
        blk: dict = {"seed": seed}
        for g in ALL_GROUPS:
            blk[g] = _warning_block(_raw_of(npz, f"{g}_s{seed}"), axis, pc)
        per_seed.append(blk)

    def _mean(vals) -> tuple[float, int]:
        """均值 + 参与均值的 seed 数。全 NaN -> NaN + 0, 不返回 0.0。"""
        v = [float(x) for x in vals
             if x is not None and np.isfinite(float(x))]
        return (float(np.mean(v)) if v else NAN), len(v)

    by_group: dict[str, dict] = {}
    for g in ALL_GROUPS:
        cells = [p[g] for p in per_seed]
        blk: dict = {}
        for key in ("warning_coverage", "miss_rate", "coverage_before_eol",
                    "miss_rate_before_eol", "false_alarm_rate"):
            m, n = _mean(c["warning"].get(key) for c in cells)
            blk[key] = {"mean": m, "n_seeds_evaluable": n,
                        "per_seed": [c["warning"].get(key) for c in cells]}
        ph_m, ph_n = _mean(c["prognostic_horizon"]["macro_ph"] for c in cells)
        blk["prognostic_horizon"] = {
            "mean": ph_m, "n_seeds_evaluable": ph_n,
            "per_seed": [c["prognostic_horizon"]["macro_ph"] for c in cells],
            "n_evaluable_traj":
                [c["prognostic_horizon"]["n_evaluable"] for c in cells],
            "n_censored_excluded":
                [c["prognostic_horizon"]["n_censored_excluded"] for c in cells],
        }
        lams = sorted({k for c in cells for k in c["alpha_lambda"]})
        blk["alpha_lambda"] = {}
        for lk in lams:
            m, n = _mean(c["alpha_lambda"].get(lk, {}).get("accuracy")
                         for c in cells)
            blk["alpha_lambda"][lk] = {
                "accuracy_mean": m, "n_seeds_evaluable": n,
                "eligible_count":
                    [c["alpha_lambda"].get(lk, {}).get("eligible_count")
                     for c in cells],
                "success_count":
                    [c["alpha_lambda"].get(lk, {}).get("success_count")
                     for c in cells],
            }
        cm, cn = _mean(c["convergence"]["macro"] for c in cells)
        lm, ln = _mean(c["convergence"]["late_convergence"] for c in cells)
        blk["convergence"] = {"macro_mean": cm, "n_seeds_evaluable": cn,
                              "late_mean": lm, "late_n_seeds_evaluable": ln,
                              "per_seed_macro":
                                  [c["convergence"]["macro"] for c in cells]}
        blk["warning_valid_count"] = int(sum(1 for c in cells
                                             if c["warning_valid"]))
        blk["n_seeds"] = len(cells)
        by_group[g] = blk

    # §9 条件 7 的输入: miss rate 不得高于 target_only + tol
    tol = float(b5["gate"]["warning_miss_tol"])
    t_miss = by_group["target_only"]["miss_rate"]["mean"]
    gate_inputs = {
        "warning_miss_tol": tol,
        "target_only_miss_mean": t_miss,
        "ft_miss_mean": by_group["source_finetune"]["miss_rate"]["mean"],
        "mmd_miss_mean": by_group["source_mmd_finetune"]["miss_rate"]["mean"],
        "ft_within_tol": bool(
            np.isfinite(t_miss)
            and np.isfinite(by_group["source_finetune"]["miss_rate"]["mean"])
            and by_group["source_finetune"]["miss_rate"]["mean"]
            <= t_miss + tol),
        "mmd_within_tol": bool(
            np.isfinite(t_miss)
            and np.isfinite(by_group["source_mmd_finetune"]["miss_rate"]["mean"])
            and by_group["source_mmd_finetune"]["miss_rate"]["mean"]
            <= t_miss + tol),
    }

    out = {
        "stage": "BASILISK_B5",
        "section": "§13 warning / prognostics",
        "nan_discipline": NAN_DISCIPLINE,
        "miss_discipline": MISS_DISCIPLINE,
        "censored_discipline": CENSORED_DISCIPLINE,
        "evaluator": "frozen S4/Basilisk (warning_lead_time / "
                     "prognostic_horizon / alpha_lambda_accuracy / "
                     "convergence_metric)",
        "prognostics_cfg": {"rul_threshold": pc["rul_threshold"],
                            "persistence": pc["persistence"],
                            "alpha": pc["alpha"],
                            "lambdas": list(pc["lambdas"]),
                            "absolute_floor": pc["absolute_floor"],
                            "late_from": pc["late_from"]},
        "formal_seeds": seeds,
        "split_sha256": str(M["split_sha256"]),
        "by_group": by_group,
        "gate_inputs": gate_inputs,
        "per_seed": per_seed,
        "verdict_not_decided_here":
            "Gate 条件 7 的判定在 summarize_b5.py。",
    }
    op = ROOT / cfg["paths"]["warning_json"]
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                             default=float) + "\n", encoding="utf-8")
    print(f">> 已写 {op.relative_to(ROOT)}")
    print(f"{'group':22s} {'cover':>7s} {'miss':>7s} {'FA':>7s} {'PH':>8s} "
          f"{'conv':>8s} {'valid':>6s}")
    for g in ALL_GROUPS:
        b = by_group[g]
        ph = b["prognostic_horizon"]["mean"]
        print(f"{g:22s} {b['warning_coverage']['mean']:7.4f} "
              f"{b['miss_rate']['mean']:7.4f} "
              f"{b['false_alarm_rate']['mean']:7.4f} "
              f"{ph:8.4f} {b['convergence']['macro_mean']:8.4f} "
              f"{b['warning_valid_count']}/{b['n_seeds']:d}")
    print(f">> §9 条件 7 (miss ≤ target + {tol}): "
          f"ft={gate_inputs['ft_within_tol']}, "
          f"mmd={gate_inputs['mmd_within_tol']}")
    print(">> miss rate 与 coverage 成对报告; NaN 保留 NaN + n_evaluable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
