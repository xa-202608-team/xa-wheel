#!/usr/bin/env python
"""scripts/basilisk_b5/analyze_lifetime_bins.py

BASILISK-B5 §11/§12 —— 寿命分 bin 分析 + 短寿命鲁棒性。

§11 bin 边界**直接取 B2.1 冻结的 short/medium/long**, 用 np.digitize 按预登记
edges 重建成员关系。**不得根据 B5 test 分布重算 bin** —— 那等于按结果挑分组。
分箱函数 lifetime_bin_map / _strata_of 直接 import B2.1 的实现, 不抄一份。

逐组逐 bin: RMSE / MAE / corr / warning coverage / catastrophic count。
逐 bin 计算 target - FT 与 target - MMD。空 bin -> NaN + n = 0, 不伪造 0。

§12 短寿命 (本阶段起可正式使用): 逐组报 mean RMSE / per-seed RMSE /
catastrophic count / pred_std / true_std。若迁移优势**只**出现在 short bin,
判 localized_transfer_benefit, 明确写成局部收益, 不泛化成全周期提升。

用法:
    python scripts/basilisk_b5/analyze_lifetime_bins.py
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
from scripts.basilisk_b21.run_b21_gate import lifetime_bin_map  # noqa: E402
from scripts.basilisk_b3x.diagnose_short_eol import (  # noqa: E402
    _subset_metrics,
    _subset_warning,
)
from scripts.basilisk_b4x.diagnose_transfer_stability import (  # noqa: E402
    per_trajectory_rmse,
)
from src.baselines.physical_extrap import caliber_eps  # noqa: E402
from src.experiments.metrics import prognostics_cfg  # noqa: E402

NAN = float("nan")

ALL_GROUPS = ("target_only", "source_finetune", "source_mmd_finetune",
              "const_mean_info", "damage_extrapolation")
TRAINED = ("target_only", "source_finetune", "source_mmd_finetune")

BIN_SOURCE_NOTE = (
    "bin 边界取自 docs/basilisk_b21/split_manifest.json 的 lifetime_bins.event, "
    "quantile_basis = dataset_level_event_eol_before_training, "
    "即在任何训练之前就已确定。"
    "**不得根据 B5 test 分布重算 bin** —— 按结果重新分组等于挑一个好看的切法。"
    "右删失轨迹的 bin 轴在 B2.1 已确认退化 (79 条观测时长相同, 有效 bin 只有 1 个), "
    "本阶段照旧如实报告, 绝不为了凑三个 bin 而发明 EOL。"
)

LOCALIZED_NOTE = (
    "若迁移增益只在 short bin 为正、其余 bin 不为正, 判 localized_transfer_benefit。"
    "此时必须写成局部收益, **不要泛化成全周期提升**。"
)

EMPTY_BIN_NOTE = (
    "空 bin 返回 NaN + n = 0, 绝不写 0 —— "
    "没有可评估轨迹与 RMSE 恰好为 0 是两件完全不同的事。"
)


def _raw_of(npz, key: str) -> dict:
    """从 formal_raw.npz 取一组的 test 预测。"""
    return {f: np.asarray(npz[f"{key}__{f}"]) for f in
            ("pred", "true", "tids", "tau", "lb")}


def _bin_block(raw: dict, ids: set, cap_eps: float, thr: float,
               axis: dict, pc: dict) -> dict:
    """一个 (组, bin) 的完整指标块。空集 -> NaN + n=0。"""
    m = _subset_metrics(raw["pred"], raw["true"], raw["tids"], ids, cap_eps)
    w = (_subset_warning(raw["pred"], raw["tau"], raw["tids"], axis, ids, pc)
         if ids else None)
    per = per_trajectory_rmse(raw["pred"], raw["true"], raw["tids"], cap_eps)
    vals = {t: v for t, v in per.items() if t in ids}
    if not vals or not np.isfinite(thr):
        cat = {"n_traj_evaluable": len(vals), "n_catastrophic": None,
               "catastrophic_error_rate": NAN,
               "reason": ("no_evaluable_trajectories" if not vals
                          else "threshold_undefined")}
    else:
        bad = sorted(t for t, v in vals.items() if v > thr)
        cat = {"n_traj_evaluable": len(vals), "n_catastrophic": len(bad),
               "catastrophic_error_rate": len(bad) / len(vals),
               "catastrophic_tids": bad,
               "worst_traj_rmse": float(max(vals.values()))}
    return {**m, "warning": w, "catastrophic": cat,
            "n_traj_in_bin": len(ids)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel_basilisk_b5.yaml")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b5 = cfg["b5"]
    lb = b5["lifetime_bins"]
    if not bool(lb["forbid_recompute_from_b5_test"]):
        raise SystemExit("!! forbid_recompute_from_b5_test 必须为 true (§11)")

    M = json.loads((ROOT / cfg["paths"]["metrics_json"]).read_text("utf-8"))
    if bool(M.get("fast")):
        raise SystemExit("!! metrics 来自 --fast 冒烟跑, 不得进入正式分析")
    split = json.loads((ROOT / cfg["paths"]["b21_split_json"]).read_text("utf-8"))
    if str(split["split_sha256"]) != str(M["split_sha256"]):
        raise SystemExit("!! 划分哈希与 metrics 记录不一致 —— B5_INVALID")

    names = list(lb["names"])
    if names != list(split["lifetime_bins"]["event"]["names"]):
        raise SystemExit(f"!! bin 名与 B2.1 manifest 不一致: {names}")
    edges = [float(x) for x in split["lifetime_bins"]["event"]["edges"]]

    # 直接 import B2.1 的分箱实现: bin 归属只有一处定义。
    bmap = lifetime_bin_map(split)
    by_bin = {n: {int(t) for t, b in bmap.items() if b == n} for n in names}
    print(f">> bin 边界 (来自 B2.1, 未重算): {edges}")
    for n in names:
        print(f"   {n}: {len(by_bin[n])} 条 event 轨迹 (test)")

    npz = np.load(ROOT / cfg["paths"]["raw_npz"])
    rows = M["per_seed"]
    seeds = [int(r["seed"]) for r in rows]
    axis = b2_endpoint_axis(cfg)
    cap_eps = caliber_eps(cfg)
    pc = prognostics_cfg(cfg)

    # per (seed, group, bin)
    per_seed: list[dict] = []
    for r in rows:
        seed = int(r["seed"])
        thr = float(r["catastrophic_threshold"]["threshold"])
        blk: dict = {"seed": seed, "catastrophic_threshold": thr,
                     "threshold_split": str(r["catastrophic_threshold"]
                                            .get("split", "validation"))}
        for g in ALL_GROUPS:
            raw = _raw_of(npz, f"{g}_s{seed}")
            blk[g] = {n: _bin_block(raw, by_bin[n], cap_eps, thr, axis, pc)
                      for n in names}
        per_seed.append(blk)

    def _mean(vals) -> float:
        v = [float(x) for x in vals
             if x is not None and np.isfinite(float(x))]
        return float(np.mean(v)) if v else NAN

    # 跨 seed 聚合 + 逐 bin gain
    by_group: dict[str, dict] = {}
    for g in ALL_GROUPS:
        by_group[g] = {}
        for n in names:
            cells = [p[g][n] for p in per_seed]
            n_cat = [c["catastrophic"]["n_catastrophic"] for c in cells]
            by_group[g][n] = {
                "rmse_mean": _mean(c["rmse"] for c in cells),
                "rmse_per_seed": [round(float(c["rmse"]), 6)
                                  if np.isfinite(float(c["rmse"])) else None
                                  for c in cells],
                "mae_mean": _mean(c["mae"] for c in cells),
                "corr_mean": _mean(c["corr"] for c in cells),
                "pred_std_mean": _mean(c["pred_std"] for c in cells),
                "true_std_mean": _mean(c["true_std"] for c in cells),
                "psr_mean": _mean(c["psr"] for c in cells),
                "warning_coverage_mean": _mean(
                    (c["warning"] or {}).get("warning_coverage")
                    for c in cells),
                "warning_miss_mean": _mean(
                    (c["warning"] or {}).get("miss_rate") for c in cells),
                "catastrophic_count_total": (int(sum(x for x in n_cat
                                                     if x is not None))
                                             if any(x is not None
                                                    for x in n_cat) else None),
                "catastrophic_count_per_seed": n_cat,
                "n_traj_in_bin": int(cells[0]["n_traj_in_bin"]),
                "n_traj_evaluable_mean": _mean(c["n_traj"] for c in cells),
            }

    gain_by_bin: dict[str, dict] = {}
    for n in names:
        t0 = by_group["target_only"][n]["rmse_mean"]
        ft = by_group["source_finetune"][n]["rmse_mean"]
        md = by_group["source_mmd_finetune"][n]["rmse_mean"]
        dm = by_group["damage_extrapolation"][n]["rmse_mean"]
        gain_by_bin[n] = {
            "target_only": t0, "source_finetune": ft,
            "source_mmd_finetune": md, "damage_extrapolation": dm,
            "gain_ft": t0 - ft, "gain_mmd": t0 - md,
            "gain_ft_positive": bool(np.isfinite(t0 - ft) and (t0 - ft) > 0),
            "gain_mmd_positive": bool(np.isfinite(t0 - md) and (t0 - md) > 0),
            "transfer_vs_damage_ft": dm - ft,
            "transfer_vs_damage_mmd": dm - md,
        }

    n_pos_ft = sum(1 for n in names if gain_by_bin[n]["gain_ft_positive"])
    n_pos_mmd = sum(1 for n in names if gain_by_bin[n]["gain_mmd_positive"])
    short = names[0]

    # §12 localized benefit: 只有 short bin 为正
    loc_ft = (gain_by_bin[short]["gain_ft_positive"]
              and not any(gain_by_bin[n]["gain_ft_positive"]
                          for n in names[1:]))
    loc_mmd = (gain_by_bin[short]["gain_mmd_positive"]
               and not any(gain_by_bin[n]["gain_mmd_positive"]
                           for n in names[1:]))

    short_block = {g: {
        "mean_rmse": by_group[g][short]["rmse_mean"],
        "per_seed_rmse": by_group[g][short]["rmse_per_seed"],
        "catastrophic_count_per_seed":
            by_group[g][short]["catastrophic_count_per_seed"],
        "catastrophic_count_total":
            by_group[g][short]["catastrophic_count_total"],
        "pred_std": by_group[g][short]["pred_std_mean"],
        "true_std": by_group[g][short]["true_std_mean"],
        "psr": by_group[g][short]["psr_mean"],
    } for g in ALL_GROUPS}

    out = {
        "stage": "BASILISK_B5",
        "section": "§11/§12 lifetime bins",
        "bin_source_note": BIN_SOURCE_NOTE,
        "localized_note": LOCALIZED_NOTE,
        "empty_bin_note": EMPTY_BIN_NOTE,
        "bins": {"names": names, "edges": edges,
                 "source": str(lb["source"]),
                 "quantile_basis":
                     str(split["lifetime_bins"]["event"]["quantile_basis"]),
                 "recomputed_from_b5_test": False,
                 "counts_test": {n: len(by_bin[n]) for n in names}},
        "censored_bins": {
            "degenerate": bool(split["lifetime_bins"]["censored"]["degenerate"]),
            "n_effective_bins":
                int(split["lifetime_bins"]["censored"]["n_effective_bins"]),
            "note": str(split["lifetime_bins"]["censored"]["degenerate_note"]),
        },
        "formal_seeds": seeds,
        "split_sha256": str(M["split_sha256"]),
        "by_group": by_group,
        "gain_by_bin": gain_by_bin,
        "n_bins_gain_positive": {"ft": n_pos_ft, "mmd": n_pos_mmd,
                                 "n_bins": len(names)},
        "short_life": {"bin": short, "by_group": short_block},
        "localized_transfer_benefit": {
            "ft": bool(loc_ft), "mmd": bool(loc_mmd),
            "label": str(lb["localized_benefit_label"]),
        },
        "per_seed": per_seed,
        "verdict_not_decided_here":
            "Gate 条件 8 (至少 2/3 bin gain > 0) 的判定在 summarize_b5.py。",
    }
    op = ROOT / cfg["paths"]["lifetime_json"]
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                             default=float) + "\n", encoding="utf-8")
    print(f">> 已写 {op.relative_to(ROOT)}")
    print(f"{'bin':8s} {'target':>9s} {'ft':>9s} {'mmd':>9s} {'damage':>9s} "
          f"{'gain_ft':>10s} {'gain_mmd':>10s}")
    for n in names:
        b = gain_by_bin[n]
        print(f"{n:8s} {b['target_only']:9.4f} {b['source_finetune']:9.4f} "
              f"{b['source_mmd_finetune']:9.4f} "
              f"{b['damage_extrapolation']:9.4f} "
              f"{b['gain_ft']:+10.6f} {b['gain_mmd']:+10.6f}")
    print(f">> gain > 0 的 bin 数: ft {n_pos_ft}/3, mmd {n_pos_mmd}/3 "
          f"(§9 条件 8 要求 ≥ 2)")
    print(f">> localized_transfer_benefit: ft={loc_ft}, mmd={loc_mmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
