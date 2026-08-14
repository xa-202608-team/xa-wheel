#!/usr/bin/env python
"""scripts/basilisk_b3x/diagnose_short_eol.py

BASILISK-B3X §7/§8 —— short-EOL 与 normal-EOL 分层诊断。

`POST_B2_FAIL_EXPLORATORY` / `NOT_FORMAL_EVIDENCE`

§7 要求两个子集**分别**报告, 不合并:
    short_eol_subset  = {19, 109, 111}   (B2 中反复失控的 test 轨迹)
    normal_eol_subset = test \\ short_eol (72 条)
各算 RMSE / MAE / corr / pred std / true std / PSR / warning miss。

纪律:
  * macro 以**轨迹**为单位, 绝不把 endpoint 数当独立样本;
  * 空子集 -> NaN + n_traj=0, **不伪造 0**;
  * 右删失轨迹没有真实 RUL, 落不进 info 掩码, 其 RUL 指标是 NaN 而非 0;
  * warning miss 用 whole-risk 分母 (eligible observed 轨迹), 不用
    "只统计成功检测"的条件口径 —— 那会把漏报藏起来。

§8 C 的判据材料: pred mean / pred std / true std / PSR / corr 联合变化。
    只降方差  -> pred std 降、corr 基本不动;
    改变方向  -> corr 明显上升 (或 pred mean 朝真值方向移动)。

用法:
    python scripts/basilisk_b3x/diagnose_short_eol.py --config configs/wheel_basilisk_b3x.yaml
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
from src.baselines.physical_extrap import caliber_eps, caliber_mask  # noqa: E402
from src.experiments.metrics import (  # noqa: E402
    prognostics_cfg,
    warning_lead_time,
)

CFG_PATH = "configs/wheel_basilisk_b3x.yaml"
NAN = float("nan")

DIAG_PURPOSE = (
    "B3X §7: 把两臂的 test 预测按 short-EOL / normal-EOL 拆开, 判断 mission "
    "features 的作用究竟落在 B2 失控的那 3 条短寿命轨迹上, 还是普通轨迹上。"
    "这是探索性诊断, 不是正式 B3。B2_GENERALIZATION_FAIL 保持终局。"
)


def _subset_metrics(pred, true, tids, keep, cap_eps: float) -> dict:
    """一个 tid 子集上的 info 口径指标。空子集 -> NaN + n_traj=0。"""
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    tids = np.asarray(tids, np.int64)
    sel = np.isin(tids, np.asarray(sorted(keep), np.int64))
    m = sel & caliber_mask(true, "info", cap_eps)
    n_pts = int(m.sum())
    if n_pts == 0:
        # 绝不返回 0 —— 没有可评估点就是不可评估
        return {"n_traj": 0, "n_endpoints": 0, "rmse": NAN, "mae": NAN,
                "corr": NAN, "pred_mean": NAN, "pred_std": NAN,
                "true_std": NAN, "psr": NAN,
                "reason": "no_evaluable_info_endpoints"}
    p, t, u = pred[m], true[m], tids[m]
    rmses, maes, corrs = [], [], []
    for k in np.unique(u):
        s = u == k
        pk, tk = p[s], t[s]
        ok = np.isfinite(pk) & np.isfinite(tk)
        if not ok.any():
            continue
        rmses.append(float(np.sqrt(np.mean((pk[ok] - tk[ok]) ** 2))))
        maes.append(float(np.mean(np.abs(pk[ok] - tk[ok]))))
        # 单点或常数序列的相关系数无定义 -> NaN, 不填 0
        if ok.sum() >= 2 and pk[ok].std() > 0 and tk[ok].std() > 0:
            corrs.append(float(np.corrcoef(pk[ok], tk[ok])[0, 1]))
    ts, ps = float(t.std()), float(p.std())
    return {
        "n_traj": len(rmses),
        "n_endpoints": n_pts,
        "rmse": float(np.mean(rmses)) if rmses else NAN,
        "mae": float(np.mean(maes)) if maes else NAN,
        # macro corr: 逐轨迹 corr 再平均; 无定义的轨迹不计入分母
        "corr": float(np.mean(corrs)) if corrs else NAN,
        "n_traj_corr_defined": len(corrs),
        "pred_mean": float(p.mean()),
        "pred_std": ps,
        "true_std": ts,
        "pred_max": float(p.max()),
        "true_max": float(t.max()),
        "psr": (ps / ts) if ts > 0 else NAN,
    }


def _subset_warning(pred, tau, tids, axis, keep, pc: dict) -> dict:
    """子集上的 whole-risk warning miss。分母 = 子集内 eligible observed 轨迹。"""
    tids = np.asarray(tids, np.int64)
    sel = np.isin(tids, np.asarray(sorted(keep), np.int64))
    if not sel.any():
        return {"n_observed": 0, "n_censored": 0, "miss_rate_before_eol": NAN,
                "coverage_before_eol": NAN, "false_alarm_rate": NAN,
                "reason": "empty_subset"}
    ev = {int(u): bool(axis[int(u)]["event_observed"])
          for u in np.unique(tids[sel])}
    w = warning_lead_time(np.asarray(pred, float)[sel],
                          np.asarray(tau, float)[sel], tids[sel], ev,
                          pc["rul_threshold"], pc["persistence"])
    return {k: w[k] for k in ("n_observed", "n_censored", "warning_coverage",
                              "miss_rate", "coverage_before_eol",
                              "miss_rate_before_eol", "false_alarm_rate")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG_PATH)
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b3 = cfg["b3x"]
    axis = b2_endpoint_axis(cfg)
    cap_eps = caliber_eps(cfg)
    pc = prognostics_cfg(cfg)

    mp = ROOT / cfg["paths"]["metrics_json"]
    if not mp.exists():
        raise SystemExit(f"!! 先跑 run_mission_ablation.py: 缺 {mp}")
    abl = json.loads(mp.read_text(encoding="utf-8"))
    if abl.get("fast"):
        raise SystemExit("!! ablation metrics 是 --fast 冒烟产物, 不得进诊断")
    npz = np.load(mp.parent / "ablation_raw.npz")

    split = json.loads(
        (ROOT / b3["split"]["reuse_from"]).read_text(encoding="utf-8"))
    key2tid = {a_["key"]: t for t, a_ in axis.items()}
    test_tids = sorted(key2tid[k] for k in split["splits"]["test"]["tids"])

    short = sorted(int(x) for x in b3["short_eol_subset"])
    missing = [t for t in short if t not in test_tids]
    if missing:
        raise SystemExit(f"!! short-EOL tid {missing} 不在 test 中 —— "
                         f"子集定义与 B2 划分不一致")
    normal = [t for t in test_tids if t not in set(short)]
    print(f">> test {len(test_tids)} 条: short-EOL {short} / "
          f"normal-EOL {len(normal)} 条")

    arms = list(abl["arms"])
    rows: list[dict] = []
    for r in abl["per_seed"]:
        seed = int(r["seed"])
        row: dict = {"seed": seed}
        for arm in arms:
            k = f"{arm}_s{seed}"
            pred, true = npz[f"{k}__pred"], npz[f"{k}__true"]
            tids, tau = npz[f"{k}__tids"], npz[f"{k}__tau"]
            row[arm] = {
                "overall": {
                    **_subset_metrics(pred, true, tids, test_tids, cap_eps),
                    "warning": _subset_warning(pred, tau, tids, axis,
                                               test_tids, pc),
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
                "per_short_traj": {
                    str(t): _subset_metrics(pred, true, tids, [t], cap_eps)
                    for t in short},
            }
        if len(arms) == 2:
            A, B = row["core_only"], row["core_plus_mission"]
            row["gain"] = {
                sub: {
                    "rmse_core": A[sub]["rmse"],
                    "rmse_mission": B[sub]["rmse"],
                    "gain": (A[sub]["rmse"] - B[sub]["rmse"]),
                    "rel_gain": ((A[sub]["rmse"] - B[sub]["rmse"])
                                 / A[sub]["rmse"]
                                 if np.isfinite(A[sub]["rmse"])
                                 and A[sub]["rmse"] > 0 else NAN),
                } for sub in ("overall", "short_eol", "normal_eol")}
            print(f"   seed {seed}: overall gain "
                  f"{row['gain']['overall']['gain']:+.4f} | short-EOL gain "
                  f"{row['gain']['short_eol']['gain']:+.4f} "
                  f"(rel {row['gain']['short_eol']['rel_gain']:+.1%}) | "
                  f"normal gain {row['gain']['normal_eol']['gain']:+.4f}")
        rows.append(row)

    out = {
        "stage": "BASILISK_B3X",
        "label": str(b3["label"]),
        "not_formal_evidence": True,
        "b2_verdict_unchanged": "B2_GENERALIZATION_FAIL",
        "purpose": DIAG_PURPOSE,
        "caliber": "info",
        "cap_eps": float(cap_eps),
        "macro_unit": "trajectory_not_endpoint",
        "warning_denominator": "whole_risk_eligible_observed",
        "short_eol_subset": short,
        "normal_eol_subset": normal,
        "n_test": len(test_tids),
        "arms": arms,
        "seeds": [int(r["seed"]) for r in abl["per_seed"]],
        "per_seed": rows,
    }
    op = ROOT / cfg["paths"]["short_eol_json"]
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                             default=float) + "\n", encoding="utf-8")
    print(f">> 已写 {op.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
