#!/usr/bin/env python
"""scripts/basilisk_b6/paired_statistics.py

BASILISK-B6 §13/§14/§17/§24-6 —— 逐标签档的 seed 级配对统计。

对每个标签档 n_event_labeled ∈ {3,5,10,21} 独立计算:
  * gain_ft / gain_mmd 的 per-seed 值、mean、median、std、improve_count;
  * seed 级 paired bootstrap 95% CI (单位 = seed 级配对差值, n=5, ≥5000 次);
  * §14 要求的全部报告量 (info/full RMSE、MAE、corr、PSR、catastrophic、
    warning coverage/miss/false alarm/PH/alpha-lambda/convergence);
  * §14 的 short/medium/long 寿命 bin RMSE 与逐 bin gain;
  * §18 与 damage_extrapolation 的差 (target−damage / ft−damage / mmd−damage);
  * §15 条件 10 所需的 "是否打赢 const_mean_info"。

§13 符号: gain = RMSE_target_only − RMSE_source_method, **正数 = source 方法更好**。

bootstrap 纪律 (沿用 B5): 重采样单位是 seed 级配对差值, 一个 seed 一个数;
绝不把时间点当独立样本, 绝不用轨迹数量抬高显著性。n=5 的 CI 只描述训练随机性下
差值方向是否稳定, **不是泛化误差的置信区间** —— 这点写进 limitations。

写出三个产物: paired_statistics.json / lifetime_bins.json / warning_metrics.json。
判定留给 final_transfer_verdict.py —— 本脚本只出数字, 不下结论。

用法:
    python scripts/basilisk_b6/paired_statistics.py --config configs/wheel_basilisk_b6.yaml
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
from scripts.basilisk_b2.run_gate import paired_bootstrap, warning_valid  # noqa: E402
from scripts.basilisk_b21.run_b21_gate import lifetime_bin_map  # noqa: E402
from scripts.basilisk_b3x.diagnose_short_eol import (  # noqa: E402
    _subset_metrics,
    _subset_warning,
)
from scripts.basilisk_b3x.summarize_b3x import _stat  # noqa: E402
from scripts.basilisk_b4x.diagnose_transfer_stability import (  # noqa: E402
    eol_metrics,
    per_trajectory_rmse,
)
from src.baselines.physical_extrap import caliber_eps  # noqa: E402
from src.experiments.metrics import (  # noqa: E402
    convergence_metric,
    prognostics_cfg,
    warning_lead_time,
)

NAN = float("nan")
INVALID = "B6_INVALID"

ALL_GROUPS = ("target_only", "source_finetune", "source_mmd_finetune",
              "const_mean_info", "damage_extrapolation")
TRAINED = ("target_only", "source_finetune", "source_mmd_finetune")

GAIN_DEFINITION = (
    "gain_ft(n, seed) = RMSE_target_only(n, seed) − RMSE_source_finetune(n, seed); "
    "gain_mmd(n, seed) = RMSE_target_only(n, seed) − "
    "RMSE_source_mmd_finetune(n, seed); "
    "主指标 = test information-zone trajectory-macro RMSE。"
    "**正数 = source 方法更好**。符号在 protocol 冻结时已固定, 不得反转。"
    "配对只在**同一标签档、同一 seed**内进行 —— 跨档相减是不同训练集, "
    "跨阶段相减 (拿 B6 减 B5/B2.1) 是不同尺子。"
)

BOOTSTRAP_DISCIPLINE = (
    "重采样单位是 seed 级配对差值, n = 5, 一个 seed 一个数。"
    "绝不把时间点当独立样本; 绝不用 endpoint bootstrap 替代 seed bootstrap; "
    "绝不用轨迹数量抬高显著性。"
    "n = 5 的 bootstrap 只描述训练随机性下差值方向是否稳定, "
    "**不是泛化误差的置信区间**。"
)

BIN_SOURCE_NOTE = (
    "bin 边界取自 docs/basilisk_b21/split_manifest.json 的 lifetime_bins.event "
    "(quantile_basis = dataset_level_event_eol_before_training), "
    "即在任何训练之前就已确定。**不得根据 B6 test 分布重算 bin**。"
    "test 侧 bin 成员在四个标签档之间完全相同 —— B6 只改 train, 从不改 test。"
)

NAN_DISCIPLINE = (
    "NaN 一律保留 NaN, 并同时给出 n_evaluable / eligible_count。"
    "绝不把 NaN 转成 0 —— 不可评估与指标恰好为 0 是两件完全不同的事。"
    "右删失 test 轨迹没有真 EOL, PH / alpha-lambda 保持 NaN, 不伪造 EOL 指标。"
    "空 bin 返回 NaN + n = 0, 不伪造 0。"
)

MISS_DISCIPLINE = (
    "coverage 与 miss rate 必须成对出现, 不得只报成功检测而隐藏 miss rate。"
    "warning valid 只说明口径自洽, **不等于 warning 可用**。"
)


def _raw_of(npz, key: str) -> dict:
    return {f: np.asarray(npz[f"{key}__{f}"]) for f in
            ("pred", "true", "tids", "tau", "lb")}


def _macro_corr(m: dict) -> float:
    v = m.get("shape", {}).get("macro_corr")
    if v is None:
        return NAN
    v = float(v)
    return v if np.isfinite(v) else NAN


def _cal(m: dict, caliber: str, scope: str, stat: str) -> float:
    """从 evaluate_arrays_b2 的 calibers 块取值。缺失 / 非有限 -> NaN, 绝不转 0。"""
    v = m.get("calibers", {}).get(caliber, {}).get(scope, {}).get(stat)
    if v is None:
        return NAN
    v = float(v)
    return v if np.isfinite(v) else NAN


def _psr(m: dict) -> dict:
    sh = m.get("shape", {})
    out = {}
    for k in ("pred_std", "true_std", "psr"):
        v = sh.get(k)
        out[k] = (float(v) if v is not None and np.isfinite(float(v)) else NAN)
    return out


def _fin_mean(vals) -> float:
    v = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    return float(np.mean(v)) if v else NAN


def _mean_n(vals) -> tuple[float, int]:
    """均值 + 参与均值的项数。全 NaN -> (NaN, 0), 绝不返回 0.0。"""
    v = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    return (float(np.mean(v)) if v else NAN), len(v)


def _bin_block(raw: dict, ids: set, cap_eps: float, thr: float,
               axis: dict, pc: dict) -> dict:
    """一个 (组, bin) 的指标块。空集 -> NaN + n=0。"""
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
               "worst_traj_rmse": float(max(vals.values()))}
    return {**m, "warning": w, "catastrophic": cat, "n_traj_in_bin": len(ids)}


def _warning_block(raw: dict, axis: dict, pc: dict) -> dict:
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
        "prognostic_horizon": {"macro_ph": e["macro_ph"],
                               "n_evaluable": int(e["ph_n_observable"]),
                               "n_censored_excluded": int(e["ph_n_censored"])},
        "alpha_lambda": e["alpha_lambda"],
        "convergence": {"macro": cv.get("macro_convergence"),
                        "late_convergence": cv.get("macro_late_convergence"),
                        "n_evaluable": len(cv.get("per_trajectory", {}))},
        "n_event_observed": n_ev,
        "n_censored": int(len(ev) - n_ev),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel_basilisk_b6.yaml")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b6 = cfg["b6"]
    ph = json.loads((ROOT / cfg["protocol"]["hash_path"]).read_text("utf-8"))

    mp = ROOT / cfg["paths"]["metrics_json"]
    if not mp.exists():
        raise SystemExit(f"!! 缺 {mp.relative_to(ROOT)} —— 先跑 run_formal_matrix.py")
    M = json.loads(mp.read_text(encoding="utf-8"))
    if bool(M.get("fast")):
        raise SystemExit("!! metrics 来自 --fast 冒烟跑, 不得进入正式分析")
    if str(M["protocol_sha256"]) != str(ph["protocol_sha256"]):
        raise SystemExit(f"!! metrics 的 protocol_sha256 与冻结件不符 —— {INVALID}")
    if str(M["subset_manifest_sha256"]) != \
            str(json.loads((ROOT / cfg["paths"]["subset_manifest_json"])
                           .read_text("utf-8"))["subset_manifest_sha256"]):
        raise SystemExit(f"!! metrics 的 subset_manifest_sha256 不符 —— {INVALID}")

    bc = b6["bootstrap"]
    if str(bc["unit"]) != "seed_level_paired_gain":
        raise SystemExit("!! §13 bootstrap 单位必须是 seed_level_paired_gain")
    n_rep = int(bc["samples"])
    if n_rep < 5000:
        raise SystemExit(f"!! 要求重采样 ≥ 5000, config 写 {n_rep}")
    bseed = int(bc["seed"])
    if not bool(bc["forbid_endpoint_bootstrap"]) \
            or not bool(bc["forbid_trajectory_count_significance"]):
        raise SystemExit("!! bootstrap 的两条禁止项必须为 true")

    levels = [int(x) for x in M["label_levels"]]
    primary = int(M["primary_level"])
    split = json.loads((ROOT / cfg["paths"]["b21_split_json"]).read_text("utf-8"))
    if str(split["split_sha256"]) != str(M["split_sha256"]):
        raise SystemExit(f"!! 划分哈希与 metrics 记录不一致 —— {INVALID}")

    lbc = cfg["b5"]["lifetime_bins"]      # bin 名/来源沿用 B2.1 -> B5 的同一份声明
    names = list(lbc["names"])
    if names != list(split["lifetime_bins"]["event"]["names"]):
        raise SystemExit(f"!! bin 名与 B2.1 manifest 不一致: {names}")
    if names != list(b6["report_lifetime_bins"]):
        raise SystemExit(f"!! §14 报告 bin 与 B2.1 bin 名不一致")
    edges = [float(x) for x in split["lifetime_bins"]["event"]["edges"]]
    bmap = lifetime_bin_map(split)
    by_bin_ids = {n: {int(t) for t, b in bmap.items() if b == n} for n in names}

    npz = np.load(ROOT / cfg["paths"]["raw_npz"])
    axis = b2_endpoint_axis(cfg)
    cap_eps = caliber_eps(cfg)
    pc = prognostics_cfg(cfg)
    tol = float(b6["gate"]["warning_miss_tol"])
    hv = float(cfg["b5"]["output_stability"]["high_variance_psr"])

    stats_levels: dict[str, dict] = {}
    life_levels: dict[str, dict] = {}
    warn_levels: dict[str, dict] = {}

    for n_event in levels:
        L = M["by_level"][str(n_event)]
        rows = L["per_seed"]
        seeds = [int(r["seed"]) for r in rows]
        if len(rows) != int(bc["n"]):
            raise SystemExit(f"!! n={n_event} 的 seed 数 {len(rows)} != {bc['n']}")

        # ---------- §13 配对统计 ----------
        per_seed: list[dict] = []
        for r in rows:
            g = r["gains"]
            per_seed.append({
                "seed": int(r["seed"]),
                "n_train": int(r["n_train"]),
                "rul_scale": float(r["rul_scale"]),
                **{k: float(g[k]) for k in ALL_GROUPS},
                "gain_ft": float(g["gain_ft"]),
                "gain_mmd": float(g["gain_mmd"]),
                "gain_ft_positive": bool(float(g["gain_ft"]) > 0),
                "gain_mmd_positive": bool(float(g["gain_mmd"]) > 0),
                "target_minus_damage": float(g["target_minus_damage"]),
                "ft_minus_damage": float(g["ft_minus_damage"]),
                "mmd_minus_damage": float(g["mmd_minus_damage"]),
                "ft_minus_const": float(g["ft_minus_const"]),
                "mmd_minus_const": float(g["mmd_minus_const"]),
                "info_pooled_rmse": {k: _cal(r[k], "info", "pooled", "rmse")
                                     for k in ALL_GROUPS},
                # full 口径含右删失点 -> 多为 NaN, 保留 NaN 不转 0 (§14)
                "full_macro_rmse": {k: _cal(r[k], "full", "macro", "rmse")
                                    for k in ALL_GROUPS},
                "mae": {k: _cal(r[k], "info", "macro", "mae")
                        for k in ALL_GROUPS},
                "corr": {k: _macro_corr(r[k]) for k in ALL_GROUPS},
                "stability": {k: _psr(r[k]) for k in ALL_GROUPS},
                "catastrophic_rate": {
                    k: (float(v["catastrophic_error_rate"])
                        if v.get("catastrophic_error_rate") is not None
                        and np.isfinite(float(v["catastrophic_error_rate"]))
                        else NAN) for k, v in r["catastrophic"].items()},
                "catastrophic_n": {k: v.get("n_catastrophic")
                                   for k, v in r["catastrophic"].items()},
                "catastrophic_threshold":
                    float(r["catastrophic_threshold"]["threshold"]),
                "warning_miss": {k: (float(r[k]["warning"].get("miss_rate", NAN))
                                     if r[k].get("warning") else NAN)
                                 for k in TRAINED},
                "warning_valid_target_only":
                    bool(warning_valid(r["target_only"]["warning"])[0]),
                "fairness": dict(r["fairness"]),
                "best_epoch": {k: int(r[f"{k}_meta"]["best_epoch"])
                               for k in TRAINED},
                "val_metric": {k: float(r[f"{k}_meta"]["val_metric"]["value"])
                               for k in TRAINED},
                "checkpoint_selection_basis": sorted({
                    str(r[f"{k}_meta"]["val_metric"]["selection_basis"])
                    for k in TRAINED}),
                "batch_order_signature":
                    str(r["target_only_meta"]["batch_order_signature"]),
                "init_weights_sha256":
                    str(r["target_only_meta"]["init_weights_sha256"]),
                "train_ids_signature":
                    str(r["target_only_meta"]["train_ids_signature"]),
            })

        def _col(key: str) -> list[float]:
            return [float(p[key]) for p in per_seed]

        boots: dict[str, dict] = {}
        for nm in ("gain_ft", "gain_mmd"):
            d = _col(nm)
            bs = paired_bootstrap(d, n_rep, bseed)
            boots[nm] = {
                "per_seed": [round(x, 8) for x in d],
                "mean": float(np.mean(d)),
                "median": float(np.median(d)),
                "std": float(np.std(d, ddof=1)) if len(d) > 1 else NAN,
                "improve_count": int(sum(1 for x in d if x > 0)),
                "n_seeds": len(d),
                "bootstrap": bs,
                "ci95_lower": float(bs["ci_lower"]),
                "ci95_upper": float(bs["ci_upper"]),
                "ci_lower_positive": bool(float(bs["ci_lower"]) > 0),
            }

        corr_means = {g: _fin_mean(p["corr"][g] for p in per_seed)
                      for g in ALL_GROUPS}
        cat_means = {g: _fin_mean(p["catastrophic_rate"].get(g)
                                  for p in per_seed) for g in ALL_GROUPS}
        agg = {g: _stat(_col(g)) for g in ALL_GROUPS}
        for k in ("target_minus_damage", "ft_minus_damage", "mmd_minus_damage",
                  "ft_minus_const", "mmd_minus_const"):
            agg[k] = _stat(_col(k))
        stability = {}
        for g in ALL_GROUPS:
            psr_v = [p["stability"][g]["psr"] for p in per_seed]
            fin = [x for x in psr_v if np.isfinite(x)]
            stability[g] = {
                "psr_per_seed": [round(x, 6) if np.isfinite(x) else None
                                 for x in psr_v],
                "psr_mean": float(np.mean(fin)) if fin else NAN,
                "pred_std_mean": _fin_mean(p["stability"][g]["pred_std"]
                                           for p in per_seed),
                "true_std_mean": _fin_mean(p["stability"][g]["true_std"]
                                           for p in per_seed),
                "n_seeds_psr_above_threshold": int(sum(1 for x in fin if x > hv)),
                "high_variance_warning": bool(fin and np.mean(fin) > hv),
            }
        fairness_all = bool(all(p["fairness"]["pass"] for p in per_seed))
        val_only = bool(all(p["checkpoint_selection_basis"] == ["validation_only"]
                            for p in per_seed))

        stats_levels[str(n_event)] = {
            "n_event_labeled": int(n_event),
            "role": str(L["role"]),
            "is_primary": bool(n_event == primary),
            "train_composition": dict(L["train_composition"]),
            "event_bin_counts": dict(L["event_bin_counts"]),
            "formal_seeds": seeds,
            "per_seed": per_seed,
            "aggregate_rmse": agg,
            "gain": boots,
            "corr_mean_by_group": corr_means,
            "catastrophic_rate_mean_by_group": cat_means,
            "output_stability": stability,
            "fairness_pass": fairness_all,
            "checkpoint_validation_only": val_only,
            "mean_info_macro_rmse": {g: agg[g]["mean"] for g in ALL_GROUPS},
        }

        # ---------- §14 寿命 bin ----------
        life_ps: list[dict] = []
        for r in rows:
            seed = int(r["seed"])
            thr = float(r["catastrophic_threshold"]["threshold"])
            blk: dict = {"seed": seed, "catastrophic_threshold": thr}
            for g in ALL_GROUPS:
                raw = _raw_of(npz, f"n{n_event}_{g}_s{seed}")
                blk[g] = {nm: _bin_block(raw, by_bin_ids[nm], cap_eps, thr,
                                         axis, pc) for nm in names}
            life_ps.append(blk)

        life_bg: dict[str, dict] = {}
        for g in ALL_GROUPS:
            life_bg[g] = {}
            for nm in names:
                cells = [p[g][nm] for p in life_ps]
                n_cat = [c["catastrophic"]["n_catastrophic"] for c in cells]
                rm, rn = _mean_n(c["rmse"] for c in cells)
                life_bg[g][nm] = {
                    "rmse_mean": rm,
                    "n_seeds_evaluable": rn,
                    "rmse_per_seed": [round(float(c["rmse"]), 6)
                                      if np.isfinite(float(c["rmse"])) else None
                                      for c in cells],
                    "mae_mean": _fin_mean(c["mae"] for c in cells),
                    "corr_mean": _fin_mean(c["corr"] for c in cells),
                    "warning_coverage_mean": _fin_mean(
                        (c["warning"] or {}).get("warning_coverage")
                        for c in cells),
                    "warning_miss_mean": _fin_mean(
                        (c["warning"] or {}).get("miss_rate") for c in cells),
                    "catastrophic_count_total":
                        (int(sum(x for x in n_cat if x is not None))
                         if any(x is not None for x in n_cat) else None),
                    "catastrophic_count_per_seed": n_cat,
                    "n_traj_in_bin": int(cells[0]["n_traj_in_bin"]),
                }

        gain_by_bin: dict[str, dict] = {}
        for nm in names:
            t0 = life_bg["target_only"][nm]["rmse_mean"]
            ft = life_bg["source_finetune"][nm]["rmse_mean"]
            md = life_bg["source_mmd_finetune"][nm]["rmse_mean"]
            dm = life_bg["damage_extrapolation"][nm]["rmse_mean"]
            gain_by_bin[nm] = {
                "target_only": t0, "source_finetune": ft,
                "source_mmd_finetune": md, "damage_extrapolation": dm,
                "gain_ft": t0 - ft, "gain_mmd": t0 - md,
                # §15 条件 5 要求 gain >= 0, 故此处用 >= 0 判定 (NaN 一律 False)
                "gain_ft_nonneg": bool(np.isfinite(t0 - ft) and (t0 - ft) >= 0),
                "gain_mmd_nonneg": bool(np.isfinite(t0 - md) and (t0 - md) >= 0),
                "gain_ft_positive": bool(np.isfinite(t0 - ft) and (t0 - ft) > 0),
                "gain_mmd_positive": bool(np.isfinite(t0 - md) and (t0 - md) > 0),
                "n_traj_in_bin": life_bg["target_only"][nm]["n_traj_in_bin"],
            }
        n_nonneg_ft = sum(1 for nm in names if gain_by_bin[nm]["gain_ft_nonneg"])
        n_nonneg_mmd = sum(1 for nm in names if gain_by_bin[nm]["gain_mmd_nonneg"])
        short = names[0]
        life_levels[str(n_event)] = {
            "n_event_labeled": int(n_event),
            "is_primary": bool(n_event == primary),
            "by_group": life_bg,
            "gain_by_bin": gain_by_bin,
            "n_bins_gain_nonneg": {"ft": n_nonneg_ft, "mmd": n_nonneg_mmd,
                                   "n_bins": len(names)},
            "localized_transfer_benefit": {
                "ft": bool(gain_by_bin[short]["gain_ft_positive"]
                           and not any(gain_by_bin[nm]["gain_ft_positive"]
                                       for nm in names[1:])),
                "mmd": bool(gain_by_bin[short]["gain_mmd_positive"]
                            and not any(gain_by_bin[nm]["gain_mmd_positive"]
                                        for nm in names[1:])),
                "label": str(lbc["localized_benefit_label"]),
            },
            "per_seed": life_ps,
        }

        # ---------- §14 warning / prognostics ----------
        warn_ps: list[dict] = []
        for r in rows:
            seed = int(r["seed"])
            blk = {"seed": seed}
            for g in ALL_GROUPS:
                blk[g] = _warning_block(_raw_of(npz, f"n{n_event}_{g}_s{seed}"),
                                        axis, pc)
            warn_ps.append(blk)

        warn_bg: dict[str, dict] = {}
        for g in ALL_GROUPS:
            cells = [p[g] for p in warn_ps]
            blk: dict = {}
            for key in ("warning_coverage", "miss_rate", "coverage_before_eol",
                        "miss_rate_before_eol", "false_alarm_rate"):
                m_, n_ = _mean_n(c["warning"].get(key) for c in cells)
                blk[key] = {"mean": m_, "n_seeds_evaluable": n_,
                            "per_seed": [c["warning"].get(key) for c in cells]}
            phm, phn = _mean_n(c["prognostic_horizon"]["macro_ph"] for c in cells)
            blk["prognostic_horizon"] = {
                "mean": phm, "n_seeds_evaluable": phn,
                "per_seed": [c["prognostic_horizon"]["macro_ph"] for c in cells],
                "n_evaluable_traj":
                    [c["prognostic_horizon"]["n_evaluable"] for c in cells],
                "n_censored_excluded":
                    [c["prognostic_horizon"]["n_censored_excluded"]
                     for c in cells]}
            lams = sorted({k for c in cells for k in c["alpha_lambda"]})
            blk["alpha_lambda"] = {}
            for lk in lams:
                m_, n_ = _mean_n(c["alpha_lambda"].get(lk, {}).get("accuracy")
                                 for c in cells)
                blk["alpha_lambda"][lk] = {
                    "accuracy_mean": m_, "n_seeds_evaluable": n_,
                    "eligible_count":
                        [c["alpha_lambda"].get(lk, {}).get("eligible_count")
                         for c in cells],
                    "success_count":
                        [c["alpha_lambda"].get(lk, {}).get("success_count")
                         for c in cells]}
            cm, cn = _mean_n(c["convergence"]["macro"] for c in cells)
            lm, ln = _mean_n(c["convergence"]["late_convergence"] for c in cells)
            blk["convergence"] = {"macro_mean": cm, "n_seeds_evaluable": cn,
                                  "late_mean": lm,
                                  "late_n_seeds_evaluable": ln,
                                  "per_seed_macro":
                                      [c["convergence"]["macro"] for c in cells]}
            blk["warning_valid_count"] = int(sum(1 for c in cells
                                                 if c["warning_valid"]))
            blk["n_seeds"] = len(cells)
            warn_bg[g] = blk

        t_miss = warn_bg["target_only"]["miss_rate"]["mean"]
        ft_miss = warn_bg["source_finetune"]["miss_rate"]["mean"]
        md_miss = warn_bg["source_mmd_finetune"]["miss_rate"]["mean"]
        warn_levels[str(n_event)] = {
            "n_event_labeled": int(n_event),
            "is_primary": bool(n_event == primary),
            "by_group": warn_bg,
            "gate_inputs": {
                "warning_miss_tol": tol,
                "target_only_miss_mean": t_miss,
                "ft_miss_mean": ft_miss,
                "mmd_miss_mean": md_miss,
                "ft_within_tol": bool(np.isfinite(t_miss) and np.isfinite(ft_miss)
                                      and ft_miss <= t_miss + tol),
                "mmd_within_tol": bool(np.isfinite(t_miss) and np.isfinite(md_miss)
                                       and md_miss <= t_miss + tol),
            },
            "per_seed": warn_ps,
        }

        b = stats_levels[str(n_event)]["gain"]
        print(f">> n={n_event:<2} [{L['role']}] "
              f"gain_ft mean {b['gain_ft']['mean']:+.6f} "
              f"med {b['gain_ft']['median']:+.6f} "
              f"imp {b['gain_ft']['improve_count']}/5 "
              f"CI[{b['gain_ft']['ci95_lower']:+.6f},{b['gain_ft']['ci95_upper']:+.6f}]"
              f" | gain_mmd mean {b['gain_mmd']['mean']:+.6f} "
              f"med {b['gain_mmd']['median']:+.6f} "
              f"imp {b['gain_mmd']['improve_count']}/5 "
              f"CI[{b['gain_mmd']['ci95_lower']:+.6f},"
              f"{b['gain_mmd']['ci95_upper']:+.6f}]")

    # ---------- §17 敏感性趋势 (仅描述, 不拟合趋势线) ----------
    sens = {"x_axis": "n_event_labeled", "x_values": levels,
            "y_axis": "paired_gain",
            "forbid_trend_line_fit": bool(b6["sensitivity"]["forbid_trend_line_fit"]),
            "forbid_reason": str(b6["sensitivity"]["forbid_reason"]),
            "by_method": {}}
    for nm, key in (("source_finetune", "gain_ft"),
                    ("source_mmd_finetune", "gain_mmd")):
        sens["by_method"][nm] = [{
            "n_event_labeled": n,
            "role": stats_levels[str(n)]["role"],
            "is_primary": stats_levels[str(n)]["is_primary"],
            "mean": stats_levels[str(n)]["gain"][key]["mean"],
            "median": stats_levels[str(n)]["gain"][key]["median"],
            "std": stats_levels[str(n)]["gain"][key]["std"],
            "ci95_lower": stats_levels[str(n)]["gain"][key]["ci95_lower"],
            "ci95_upper": stats_levels[str(n)]["gain"][key]["ci95_upper"],
            "improve_count": stats_levels[str(n)]["gain"][key]["improve_count"],
        } for n in levels]

    common = {
        "stage": "BASILISK_B6",
        "protocol_sha256": str(M["protocol_sha256"]),
        "config_sha256": str(M["config_sha256"]),
        "subset_manifest_sha256": str(M["subset_manifest_sha256"]),
        "split_sha256": str(M["split_sha256"]),
        "label_levels": levels,
        "primary_level": primary,
        "secondary_levels": [int(x) for x in M["secondary_levels"]],
        "formal_seeds": [int(s) for s in M["formal_seeds"]],
    }

    out = {
        **common,
        "section": "§13/§17 paired statistics",
        "gain_definition": GAIN_DEFINITION,
        "bootstrap_discipline": BOOTSTRAP_DISCIPLINE,
        "nan_discipline": NAN_DISCIPLINE,
        "primary_metric": str(b6["primary_metric"]),
        "bootstrap_config": {"unit": str(bc["unit"]), "n": int(bc["n"]),
                             "samples": n_rep, "seed": bseed,
                             "forbid_endpoint_bootstrap": True,
                             "forbid_trajectory_count_significance": True},
        "by_level": stats_levels,
        "sensitivity": sens,
        "secondary_cannot_override_primary": True,
        "verdict_not_decided_here":
            "Gate 判定在 final_transfer_verdict.py; 本脚本只出数字, 不下结论。",
    }
    op = ROOT / cfg["paths"]["paired_json"]
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                             default=float) + "\n", encoding="utf-8")
    print(f">> 已写 {op.relative_to(ROOT)}")

    lout = {
        **common,
        "section": "§14 lifetime bins",
        "bin_source_note": BIN_SOURCE_NOTE,
        "nan_discipline": NAN_DISCIPLINE,
        "bins": {"names": names, "edges": edges, "source": str(lbc["source"]),
                 "quantile_basis":
                     str(split["lifetime_bins"]["event"]["quantile_basis"]),
                 "recomputed_from_b6_test": False,
                 "counts_test": {nm: len(by_bin_ids[nm]) for nm in names},
                 "identical_across_levels": True},
        "censored_bins": {
            "degenerate": bool(split["lifetime_bins"]["censored"]["degenerate"]),
            "n_effective_bins":
                int(split["lifetime_bins"]["censored"]["n_effective_bins"])},
        "by_level": life_levels,
    }
    lp = ROOT / cfg["paths"]["lifetime_json"]
    lp.write_text(json.dumps(lout, indent=2, ensure_ascii=False,
                             default=float) + "\n", encoding="utf-8")
    print(f">> 已写 {lp.relative_to(ROOT)}")

    wout = {
        **common,
        "section": "§14 warning / prognostics",
        "nan_discipline": NAN_DISCIPLINE,
        "miss_discipline": MISS_DISCIPLINE,
        "evaluator": "frozen S4/Basilisk (warning_lead_time / "
                     "prognostic_horizon / alpha_lambda_accuracy / "
                     "convergence_metric)",
        "prognostics_cfg": {"rul_threshold": pc["rul_threshold"],
                            "persistence": pc["persistence"],
                            "alpha": pc["alpha"],
                            "lambdas": list(pc["lambdas"]),
                            "absolute_floor": pc["absolute_floor"],
                            "late_from": pc["late_from"]},
        "by_level": warn_levels,
    }
    wp = ROOT / cfg["paths"]["warning_json"]
    wp.write_text(json.dumps(wout, indent=2, ensure_ascii=False,
                             default=float) + "\n", encoding="utf-8")
    print(f">> 已写 {wp.relative_to(ROOT)}")
    print(">> Gate 判定在 final_transfer_verdict.py, 本脚本只出数字, 不下结论")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
