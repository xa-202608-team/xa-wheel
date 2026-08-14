#!/usr/bin/env python
"""scripts/basilisk_b4x/diagnose_transfer_stability.py

BASILISK-B4X §14/§15/§16 —— 分层诊断 + catastrophic_error_rate。

`POST_B2_FAIL_EXPLORATORY` / `NOT_FORMAL_EVIDENCE` / `EXPLORATORY_ONLY`

§14 三个子集各自单独报告, 不合并:
    overall            test 全 75 条 (含 PH / alpha-lambda)
    short_eol_subset   {19, 109, 111}
    normal_eol_subset  test \\ short_eol (72 条)

§16 `catastrophic_error_rate` (纯诊断指标):
    threshold = multiplier x median(trajectory RMSE of **target_only 的
                validation** 轨迹)
    对每条 test trajectory: RMSE > threshold 即记为 catastrophic。
    **阈值只使用 target_only 的 validation 分布, 绝不使用 test 分布来选。**
    三组共用同一个阈值 (来自 target_only 的 val), 否则三组不可比。

纪律 (与 B3X / B2 同):
  * macro 以轨迹为单位, 绝不把 endpoint 当独立样本;
  * 空子集 -> NaN + n_traj=0, **不伪造 0**;
  * 右删失轨迹没有真实 EOL, 不伪造 EOL 指标 (PH / alpha-lambda 一律 NaN);
  * warning miss 用 whole-risk 分母, 不用"只统计成功检测"的条件口径。

用法:
    python scripts/basilisk_b4x/diagnose_transfer_stability.py --config configs/wheel_basilisk_b4x.yaml
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
from scripts.basilisk_b3x.diagnose_short_eol import (  # noqa: E402
    _subset_metrics,
    _subset_warning,
)
from src.baselines.physical_extrap import caliber_eps, caliber_mask  # noqa: E402
from src.experiments.metrics import (  # noqa: E402
    alpha_lambda_accuracy,
    prognostic_horizon,
    prognostics_cfg,
)

CFG_PATH = "configs/wheel_basilisk_b4x.yaml"
NAN = float("nan")

DIAG_PURPOSE = (
    "B4X §14/§16: 把三组的 test 预测按 short-EOL / normal-EOL 拆开, 并计算"
    "catastrophic_error_rate, 用于判断 source 系方法的优势究竟来自"
    "避免了 target_only 的 catastrophic failure, 还是正常区域也有持续改善。"
    "这是探索性诊断, 不是正式 B4, 不构成迁移证据。"
    "B2_GENERALIZATION_FAIL 保持终局。"
)

# §16 阈值来源纪律。抽成模块级常量的理由同前几阶段 (扫描器豁免常量声明块)。
THRESHOLD_DISCIPLINE = (
    "catastrophic 阈值只允许由 target_only 的 validation 轨迹 RMSE 中位数"
    "乘 multiplier 得到。禁止使用 test 分布来选阈值, 禁止按哪个阈值让结论"
    "更好看来挑。三组共用同一个阈值, 否则三组不可比。"
)

GROUPS = ("target_only", "source_finetune", "source_mmd_finetune")
THRESHOLD_GROUP = "target_only"


def per_trajectory_rmse(pred, true, tids, cap_eps: float) -> dict[int, float]:
    """逐轨迹 info 口径 RMSE。无可评估点的轨迹**不进** dict (而不是记 0)。"""
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    tids = np.asarray(tids, np.int64)
    m = caliber_mask(true, "info", cap_eps) & np.isfinite(pred) & np.isfinite(true)
    out: dict[int, float] = {}
    for u in np.unique(tids):
        s = (tids == u) & m
        if not s.any():
            continue
        out[int(u)] = float(np.sqrt(np.mean((pred[s] - true[s]) ** 2)))
    return out


def catastrophic_threshold(pred_val, true_val, tids_val, cap_eps: float,
                           multiplier: float) -> dict:
    """§16 阈值: multiplier x median(target_only val 逐轨迹 RMSE)。"""
    per = per_trajectory_rmse(pred_val, true_val, tids_val, cap_eps)
    if not per:
        # 没有可评估的 val 轨迹 -> 阈值无定义。绝不退化成一个默认数字。
        return {"threshold": NAN, "val_median_trajectory_rmse": NAN,
                "n_val_traj_evaluable": 0, "multiplier": float(multiplier),
                "split": "validation", "group": THRESHOLD_GROUP,
                "derived_from_test": False,
                "source": "target_only_validation_median_trajectory_rmse",
                "reason": "no_evaluable_validation_trajectories"}
    vals = np.array(sorted(per.values()), float)
    med = float(np.median(vals))
    return {
        "threshold": float(multiplier) * med,
        "val_median_trajectory_rmse": med,
        "n_val_traj_evaluable": int(len(vals)),
        "multiplier": float(multiplier),
        "split": "validation",
        "group": THRESHOLD_GROUP,
        "source": "target_only_validation_median_trajectory_rmse",
        "val_traj_rmse_min": float(vals.min()),
        "val_traj_rmse_max": float(vals.max()),
        "derived_from_test": False,
    }


def catastrophic_rate(pred, true, tids, cap_eps: float,
                      threshold: float) -> dict:
    """test 上的 catastrophic count / rate。分母 = 可评估的 test 轨迹数。"""
    per = per_trajectory_rmse(pred, true, tids, cap_eps)
    if not per or not np.isfinite(threshold):
        # 不可评估就是不可评估: rate = NaN, count = None。绝不写 0
        # (0 会被读成"没有 catastrophic 轨迹", 与"算不出来"是两件事)。
        return {"n_traj_evaluable": len(per), "n_catastrophic": None,
                "catastrophic_error_rate": NAN, "catastrophic_tids": [],
                "threshold": float(threshold),
                "reason": ("no_evaluable_trajectories" if not per
                           else "threshold_undefined")}
    bad = sorted(t for t, v in per.items() if v > threshold)
    return {
        "n_traj_evaluable": len(per),
        "n_catastrophic": len(bad),
        "catastrophic_error_rate": len(bad) / len(per),
        "catastrophic_tids": bad,
        "threshold": float(threshold),
        "worst_traj_rmse": float(max(per.values())),
        "median_traj_rmse": float(np.median(list(per.values()))),
    }


def eol_metrics(pred, true, tau, tids, axis, pc: dict) -> dict:
    """§14 的 PH / alpha-lambda。右删失轨迹一律 NaN, 不伪造 EOL。"""
    tids = np.asarray(tids, np.int64)
    ev = {int(u): bool(axis[int(u)]["event_observed"]) for u in np.unique(tids)}
    ph = prognostic_horizon(pred, true, tau, tids, ev,
                            pc["alpha"], pc["absolute_floor"])
    al = alpha_lambda_accuracy(pred, true, tau, tids, ev,
                               pc["alpha"], pc["absolute_floor"],
                               pc["lambdas"])
    return {
        "macro_ph": ph["macro_ph"],
        "ph_n_observable": ph["n_observable"],
        "ph_n_censored": ph["n_censored"],
        "alpha_lambda": {k: {"accuracy": v["accuracy"],
                             "eligible_count": v["eligible_count"],
                             "success_count": v["success_count"]}
                         for k, v in al["by_lambda"].items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG_PATH)
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b4 = cfg["b4x"]
    axis = b2_endpoint_axis(cfg)
    cap_eps = caliber_eps(cfg)
    pc = prognostics_cfg(cfg)

    mp = ROOT / cfg["paths"]["metrics_json"]
    if not mp.exists():
        raise SystemExit(f"!! 先跑 run_transfer_diagnostic.py: 缺 {mp}")
    tm = json.loads(mp.read_text(encoding="utf-8"))
    if tm.get("fast"):
        raise SystemExit("!! transfer metrics 是 --fast 冒烟产物, 不得进诊断")
    npz = np.load(mp.parent / "transfer_raw.npz")

    split = json.loads(
        (ROOT / b4["split"]["reuse_from"]).read_text(encoding="utf-8"))
    key2tid = {a_["key"]: t for t, a_ in axis.items()}
    test_tids = sorted(key2tid[k] for k in split["splits"]["test"]["tids"])

    short = sorted(int(x) for x in b4["short_eol_subset"])
    missing = [t for t in short if t not in test_tids]
    if missing:
        raise SystemExit(f"!! short-EOL tid {missing} 不在 test 中 —— "
                         f"子集定义与 B2 划分不一致")
    normal = [t for t in test_tids if t not in set(short)]
    print(f">> test {len(test_tids)} 条: short-EOL {short} / "
          f"normal-EOL {len(normal)} 条")

    cc = b4["catastrophic"]
    if not bool(cc.get("forbid_test_derived_threshold", False)):
        raise SystemExit("!! config 必须声明 forbid_test_derived_threshold: true")
    mult = float(cc["multiplier"])
    groups = list(tm["groups"])

    by_set: dict[str, list[dict]] = {}
    for sname, rows_in in tm["per_seed_by_set"].items():
        rows: list[dict] = []
        for r in rows_in:
            seed = int(r["seed"])
            # ---- §16: 阈值先从 target_only 的 **val** 算出, 三组共用 ----
            kt = f"{THRESHOLD_GROUP}_s{seed}"
            thr = catastrophic_threshold(
                npz[f"{kt}__val_pred"], npz[f"{kt}__val_true"],
                npz[f"{kt}__val_tids"], cap_eps, mult)
            row: dict = {"seed": seed, "seed_set": sname,
                         "catastrophic_threshold": thr}
            for g in groups:
                k = f"{g}_s{seed}"
                pred, true = npz[f"{k}__pred"], npz[f"{k}__true"]
                tids, tau = npz[f"{k}__tids"], npz[f"{k}__tau"]
                row[g] = {
                    "overall": {
                        **_subset_metrics(pred, true, tids, test_tids, cap_eps),
                        "warning": _subset_warning(pred, tau, tids, axis,
                                                   test_tids, pc),
                        **eol_metrics(pred, true, tau, tids, axis, pc),
                    },
                    "short_eol": {
                        **_subset_metrics(pred, true, tids, short, cap_eps),
                        "warning": _subset_warning(pred, tau, tids, axis,
                                                   short, pc),
                    },
                    "normal_eol": {
                        **_subset_metrics(pred, true, tids, normal, cap_eps),
                        "warning": _subset_warning(pred, tau, tids, axis,
                                                   normal, pc),
                    },
                    "catastrophic": catastrophic_rate(pred, true, tids, cap_eps,
                                                      thr["threshold"]),
                    "per_short_traj": {
                        str(t): _subset_metrics(pred, true, tids, [t], cap_eps)
                        for t in short},
                }
            # ---- §15 gains, 三个子集分别算 ----
            if all(g in row for g in GROUPS):
                gains = {}
                for sub in ("overall", "short_eol", "normal_eol"):
                    t0 = row["target_only"][sub]["rmse"]
                    ft = row["source_finetune"][sub]["rmse"]
                    md = row["source_mmd_finetune"][sub]["rmse"]
                    gains[sub] = {
                        "target_only": t0, "source_finetune": ft,
                        "source_mmd_finetune": md,
                        "gain_ft": t0 - ft, "gain_mmd": t0 - md,
                        "rel_gain_ft": ((t0 - ft) / t0
                                        if np.isfinite(t0) and t0 > 0 else NAN),
                        "rel_gain_mmd": ((t0 - md) / t0
                                         if np.isfinite(t0) and t0 > 0 else NAN),
                    }
                row["gains"] = gains
                cr = {g: row[g]["catastrophic"]["catastrophic_error_rate"]
                      for g in GROUPS}
                row["catastrophic_rates"] = cr
                print(f"   seed {seed} [{sname}]  overall gain_ft "
                      f"{gains['overall']['gain_ft']:+.4f} / gain_mmd "
                      f"{gains['overall']['gain_mmd']:+.4f} | short-EOL "
                      f"{gains['short_eol']['gain_ft']:+.4f} / "
                      f"{gains['short_eol']['gain_mmd']:+.4f} | normal "
                      f"{gains['normal_eol']['gain_ft']:+.4f} / "
                      f"{gains['normal_eol']['gain_mmd']:+.4f}")
                print(f"      catastrophic rate (thr "
                      f"{thr['threshold']:.4f}): "
                      + "  ".join(f"{g}={cr[g]:.3f}" for g in GROUPS))
            rows.append(row)
        by_set[sname] = rows

    out = {
        "stage": "BASILISK_B4X",
        "label": str(b4["label"]),
        "not_formal_evidence": True,
        "exploratory_only": True,
        "b2_verdict_unchanged": "B2_GENERALIZATION_FAIL",
        "b2_verdict_status": "B2_GENERALIZATION_FAIL_REMAINS_FINAL",
        "not_formal_transfer_evidence": True,
        "purpose": DIAG_PURPOSE,
        "threshold_discipline": THRESHOLD_DISCIPLINE,
        "caliber": "info",
        "cap_eps": float(cap_eps),
        "macro_unit": "trajectory_not_endpoint",
        "warning_denominator": "whole_risk_eligible_observed",
        "catastrophic": {
            "multiplier": mult,
            "threshold_source": str(cc["threshold_source"]),
            "threshold_group": THRESHOLD_GROUP,
            "threshold_split": "validation",
            "forbid_test_derived_threshold": True,
            "definition": str(cc["definition"]),
        },
        "short_eol_subset": short,
        "normal_eol_subset": normal,
        "n_test": len(test_tids),
        "groups": groups,
        "seed_sets": {k: [int(r["seed"]) for r in v] for k, v in by_set.items()},
        "forbid_merging_seed_sets": True,
        "per_seed_by_set": by_set,
    }
    op = ROOT / cfg["paths"]["stability_json"]
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                             default=float) + "\n", encoding="utf-8")
    print(f">> 已写 {op.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
