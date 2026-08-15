#!/usr/bin/env python
"""scripts/basilisk_b21/run_b21_gate.py

BASILISK-B2.1 §8..§12 —— 在新划分上重跑 Target-only 泛化 Gate。

与 B2 §8 的 run_gate.py 的差别**只有两处**:
  1. 读的划分是 docs/basilisk_b21/split_manifest.json (新协议), 不是 B2 的;
  2. 追加 §11 的分层报告 (short/medium/long-life RMSE + catastrophic short-life)。

训练配置 / 损失 / 模型 / evaluator / 基线**全部 import 复用**, 一行不改:
  prepare_b2 / train_target_only / evaluate_b2 / const_mean_info /
  damage_extrapolation。§8 要求"复用冻结的 B2 Target-only 训练配置", 复用的
  最强形式就是直接调用同一个函数, 而不是抄一份。

§9: 只比 target_only / const_mean_info / damage_extrapolation。
    不跑任何迁移方法 —— 本模块不 import 任何 src.transfer 的训练入口。
§12 bootstrap 单位 = 一个 seed 一个配对差值 (5 个数)。绝不把 endpoint
    当独立样本。

用法:
    python scripts/basilisk_b21/run_b21_gate.py
    python scripts/basilisk_b21/run_b21_gate.py --seeds 102 --fast   # 冒烟
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.basilisk_b11.calibrate_degradation import (  # noqa: E402
    load_b11_config as load_cfg,
)
from scripts.basilisk_b2.baselines_b2 import (  # noqa: E402
    EXCLUDED_ORACLE_METHODS,
    const_mean_info,
    const_pred_on,
    damage_extrapolation,
)
from scripts.basilisk_b2.data_b2 import prepare_b2  # noqa: E402
from scripts.basilisk_b2.eval_b2 import (  # noqa: E402
    b2_endpoint_axis,
    evaluate_arrays_b2,
    evaluate_b2,
    strip_private,
)
from scripts.basilisk_b2.run_gate import (  # noqa: E402
    paired_bootstrap,
    warning_valid,
)
from scripts.basilisk_b2.train_b2 import train_target_only  # noqa: E402
from scripts.basilisk_b4x.diagnose_transfer_stability import (  # noqa: E402
    catastrophic_rate,
    catastrophic_threshold,
    per_trajectory_rmse,
)
from scripts.diag_generalization import summarize_history  # noqa: E402

CFG_PATH = "configs/wheel_basilisk_b21.yaml"
NAN = float("nan")

GATE_PURPOSE = (
    "B2.1 §8..§12: 在修复了 lifetime-support mismatch 的新划分上重问 B2 的"
    "泛化问题 —— target_only 是否稳定优于最强可部署平凡基线 const_mean_info。"
    "本阶段只改划分协议, 训练配置逐项复用 B2 冻结值, 不跑任何迁移方法。"
    "B2_GENERALIZATION_FAIL 保持终局: 本阶段 PASS 不追溯修改 B2 的判定, "
    "只说明'划分覆盖不足'确实是 B2 失败的一个实质因素; 本阶段 FAIL 则说明"
    "修好覆盖之后问题依旧, 失败原因在别处。"
)

LIFETIME_SUBSET_NOTE = (
    "short/medium/long-life 子集只由 **event-observed** 的 test 轨迹构成, "
    "bin 归属来自 split_manifest 预登记的数据集级 EOL 三分位。右删失轨迹没有"
    "真实 RUL, 因此不属于任何 lifetime 子集, 其 RUL 类指标恒为 NaN + n=0 —— "
    "不伪造 EOL, 不把 NaN 悄悄转成 0。空子集返回 NaN + n=0。"
)

CENSORED_LB_NOTE = (
    "censored lower-bound violation rate = frac(pred < rul_lower_bound), "
    "两侧同在归一化 RUL 单位下比较 (lb / rul_scale)。用的是已知下界, "
    "不涉及任何伪造 EOL。"
)

METHODS = ("target_only", "const_mean_info", "damage_extrapolation")


def lifetime_bin_map(split: dict) -> dict[int, str]:
    """从 manifest 读 test 侧 event 轨迹的 lifetime bin 归属 (tid -> bin 名)。

    只读 manifest 的预登记分箱, 不在此处重算分位 —— 分位必须只有一处定义。
    """
    out: dict[int, str] = {}
    names = list(split["lifetime_bins"]["event"]["names"])
    for i, s in zip(split["splits"]["test"]["tid_indices"],
                    _strata_of(split, "test")):
        kind, _, bn = str(s).partition("/")
        if kind == "event" and bn in names:
            out[int(i)] = bn
    return out


def _strata_of(split: dict, name: str) -> list[str]:
    """manifest 只存了 strata 计数, 逐条归属需按 tid 顺序重建。

    重建方式: event 轨迹用 manifest 的 event bin edges 对其 EOL 数字化。
    edges 来自 manifest (预登记), 不重新算分位。
    """
    import h5py
    edges = list(split["lifetime_bins"]["event"]["edges"])
    ev_names = list(split["lifetime_bins"]["event"]["names"])
    cen_names = list(split["lifetime_bins"]["censored"]["names"])
    c_edges = list(split["lifetime_bins"]["censored"]["edges"])
    h5 = ROOT / split["feature_h5"]
    out: list[str] = []
    with h5py.File(h5, "r") as f:
        for tid in split["splits"][name]["tids"]:
            g = f[tid]
            if bool(int(g.attrs["event_observed"])):
                b = int(np.digitize(float(g.attrs["eol_idx"]), edges))
                out.append(f"event/{ev_names[b]}")
            else:
                b = int(np.digitize(float(g["x_T"].shape[0]), c_edges))
                out.append(f"censored/{cen_names[b]}")
    return out


def subset_rmse(m: dict, tids_wanted, cap_eps: float) -> dict:
    """一个 lifetime 子集上的 trajectory-macro RMSE。空子集 -> NaN + n=0。"""
    raw = m["_raw"]
    per = per_trajectory_rmse(raw["pred"], raw["true"], raw["tids"], cap_eps)
    vals = [v for t, v in per.items() if t in set(int(x) for x in tids_wanted)]
    if not vals:
        # 绝不写 0: 没有可评估轨迹与"RMSE 恰好为 0"是两件完全不同的事
        return {"macro_rmse": NAN, "n_traj_evaluable": 0,
                "n_traj_in_subset": len(list(tids_wanted)),
                "reason": "no_evaluable_trajectories"}
    return {"macro_rmse": float(np.mean(vals)),
            "n_traj_evaluable": len(vals),
            "n_traj_in_subset": len(list(tids_wanted)),
            "worst_traj_rmse": float(max(vals))}


def run_seed(cfg: dict, split: dict, axis: dict, seed: int, bins: dict,
             dmg_cache: dict | None = None) -> dict:
    """一个 seed: target_only 训练 + 三方法在同一 test 取点上评估 + §11 分层。"""
    print(f"\n{'=' * 66}\n[B2.1 §9] seed {seed}\n{'=' * 66}")
    t0 = time.time()
    data = prepare_b2(cfg, split, seed, with_test=True, verbose=True)
    cap_eps = float(data["cap_eps"])
    mult = float(cfg["b21"]["catastrophic"]["multiplier"])

    # --- C. target_only (复用 B2 冻结训练配置) ---
    model, history, th, budget = train_target_only(
        cfg, data, seed, f"b21_target_only_s{seed}")
    m_te = evaluate_b2(model, data["lte"], cfg, axis, data,
                       tag=f"target_only_s{seed}")
    # catastrophic 阈值只来自 **validation**, 禁止 test 派生
    m_va = evaluate_b2(model, data["lva"], cfg, axis, data,
                       tag=f"target_only_val_s{seed}")
    thr = catastrophic_threshold(m_va["_raw"]["pred"], m_va["_raw"]["true"],
                                 m_va["_raw"]["tids"], cap_eps, mult)

    # --- A. const_mean_info (常数只由 train 算) ---
    cval = const_mean_info(cfg, axis, data)
    pa, ta, ia = const_pred_on(axis, data["te"], cval, cfg, data)
    m_const = evaluate_arrays_b2(pa, ta, ia, cfg, axis, data,
                                 tag=f"const_mean_info_s{seed}")

    # --- B. damage_extrapolation (非学习, 与 seed 无关, 但同 test 取点) ---
    if dmg_cache is not None and "pred" in dmg_cache:
        pb, tb, ib = dmg_cache["pred"], dmg_cache["true"], dmg_cache["tids"]
    else:
        pb, tb, ib = damage_extrapolation(cfg, axis, data, data["te"])
        if dmg_cache is not None:
            dmg_cache.update({"pred": pb, "true": tb, "tids": ib})
    m_dmg = evaluate_arrays_b2(pb, tb, ib, cfg, axis, data,
                               tag=f"damage_extrapolation_s{seed}")

    # 取点必须逐点一致, 否则指标不可比 —— 直接报错, 不静默对齐
    for nm, arr in (("const", ia), ("damage", ib)):
        if not np.array_equal(arr, m_te["_raw"]["tids"]):
            raise SystemExit(f"!! {nm} 与模型 test 取点不一致, 拒绝比较")

    # --- §11 lifetime 分层 (三方法都算, 口径完全相同) ---
    by_bin = {n: [t for t, b in bins.items() if b == n]
              for n in cfg["b21"]["split"]["event"]["bins"]}
    lifetime: dict[str, dict] = {}
    for meth, mm in (("target_only", m_te), ("const_mean_info", m_const),
                     ("damage_extrapolation", m_dmg)):
        lifetime[meth] = {n: subset_rmse(mm, ids, cap_eps)
                          for n, ids in by_bin.items()}

    # short-life catastrophic: 阈值来自 val, 只统计 short-life test 轨迹
    short_name = cfg["b21"]["split"]["event"]["bins"][0]
    per_te = per_trajectory_rmse(m_te["_raw"]["pred"], m_te["_raw"]["true"],
                                 m_te["_raw"]["tids"], cap_eps)
    sh_ids = set(int(x) for x in by_bin[short_name])
    sh_eval = {t: v for t, v in per_te.items() if t in sh_ids}
    if not sh_eval or not np.isfinite(thr["threshold"]):
        sh_cat = {"n_traj_evaluable": len(sh_eval), "n_catastrophic": None,
                  "catastrophic_error_rate": NAN, "catastrophic_tids": [],
                  "reason": ("no_evaluable_trajectories" if not sh_eval
                             else "threshold_undefined")}
    else:
        bad = sorted(t for t, v in sh_eval.items() if v > thr["threshold"])
        sh_cat = {"n_traj_evaluable": len(sh_eval), "n_catastrophic": len(bad),
                  "catastrophic_error_rate": len(bad) / len(sh_eval),
                  "catastrophic_tids": bad,
                  "worst_traj_rmse": float(max(sh_eval.values()))}
    cat_all = catastrophic_rate(m_te["_raw"]["pred"], m_te["_raw"]["true"],
                                m_te["_raw"]["tids"], cap_eps,
                                thr["threshold"])
    max_sh = int(cfg["b21"]["catastrophic"]
                 ["max_short_life_catastrophic_per_seed"])
    no_cat_short = (sh_cat["n_catastrophic"] is not None
                    and sh_cat["n_catastrophic"] <= max_sh)

    gain = float(m_const["info_macro_rmse"]) - float(m_te["info_macro_rmse"])
    wok, wdet = warning_valid(m_te["warning"])

    row = {
        "seed": int(seed),
        "const_mean_info_value": float(cval),
        "target_only": {k: v for k, v in strip_private(m_te).items()
                        if k != "calibers"},
        "const_mean_info": {k: v for k, v in strip_private(m_const).items()
                            if k != "calibers"},
        "damage_extrapolation": {k: v for k, v
                                 in strip_private(m_dmg).items()
                                 if k != "calibers"},
        "paired": {
            "metric": "test_info_macro_rmse",
            "target_only": float(m_te["info_macro_rmse"]),
            "const_mean_info": float(m_const["info_macro_rmse"]),
            "damage_extrapolation": float(m_dmg["info_macro_rmse"]),
            "gain_vs_const": gain,
            "better_than_const": bool(gain > 0),
        },
        "shape": m_te["shape"],
        "lifetime_subsets": lifetime,
        "catastrophic_threshold": thr,
        "catastrophic_overall": cat_all,
        "catastrophic_short_life": sh_cat,
        "no_catastrophic_short_life": bool(no_cat_short),
        "censored_lower_bound": {
            "violation_rate": m_te["censor_report"]["censored_test"][
                "lower_bound_violation_rate"],
            "n_evaluable": m_te["censor_report"]["censored_test"][
                "n_lower_bound_evaluable"],
            "note": CENSORED_LB_NOTE,
        },
        "warning_validity": wdet,
        "warning_valid": wok,
        "history": history,
        "history_summary": summarize_history(history, th["early_stop_metric"]),
        "hyper": th,
        "budget": budget,
        "checkpoint_selection": "val_only",
        "elapsed_s": round(time.time() - t0, 1),
    }
    sh = lifetime["target_only"][short_name]["macro_rmse"]
    print(f"  [seed {seed}] target_only {m_te['info_macro_rmse']:.4f} vs "
          f"const {m_const['info_macro_rmse']:.4f} (gain {gain:+.4f})  "
          f"corr {m_te['shape']['macro_corr']:.3f}  PSR "
          f"{m_te['shape']['psr']:.3f}  short-life RMSE {sh:.4f}  "
          f"short-cat {sh_cat['n_catastrophic']}  [{row['elapsed_s']}s]")
    return row


def decide(rows: list, gc: dict) -> dict:
    """§12 七条件判定。任一不满足 -> B21_GENERALIZATION_FAIL。"""
    n = len(rows)
    diffs = [r["paired"]["gain_vs_const"] for r in rows]
    n_better = sum(bool(r["paired"]["better_than_const"]) for r in rows)
    corrs = [r["shape"]["macro_corr"] for r in rows]
    n_corr = sum(bool(np.isfinite(c) and c > 0) for c in corrs)
    n_nocat = sum(bool(r["no_catastrophic_short_life"]) for r in rows)
    n_warn = sum(bool(r["warning_valid"]) for r in rows)
    n_valonly = sum(r["checkpoint_selection"] == "val_only" for r in rows)
    bs = paired_bootstrap(diffs, int(gc["bootstrap_samples"]),
                          int(gc["bootstrap_seed"]))
    fin = [d for d in diffs if np.isfinite(d)]
    mean_gain = float(np.mean(fin)) if fin else NAN

    conds = [
        {"id": 1, "name": "target_only 优于 const_mean_info 的 seed 数",
         "need": f">= {gc['min_improve_count']}/{n}", "got": f"{n_better}/{n}",
         "pass": bool(n_better >= int(gc["min_improve_count"]))},
        {"id": 2, "name": "mean paired gain > 0",
         "need": "> 0", "got": f"{mean_gain:.6f}",
         "pass": bool(np.isfinite(mean_gain) and mean_gain > 0)},
        {"id": 3, "name": "seed 级配对 bootstrap 95% CI lower > 0",
         "need": "> 0", "got": f"{bs['ci_lower']:.6f}",
         "pass": bool(np.isfinite(bs["ci_lower"]) and bs["ci_lower"] > 0)},
        {"id": 4, "name": "macro corr > 0 的 seed 数",
         "need": f">= {gc['min_corr_count']}/{n}", "got": f"{n_corr}/{n}",
         "pass": bool(n_corr >= int(gc["min_corr_count"]))},
        {"id": 5, "name": "无 catastrophic short-life failure 的 seed 数",
         "need": f">= {gc['min_no_catastrophic_short_count']}/{n}",
         "got": f"{n_nocat}/{n}",
         "pass": bool(n_nocat >= int(gc["min_no_catastrophic_short_count"]))},
        {"id": 6, "name": "warning 指标有效", "need": f"{n}/{n}",
         "got": f"{n_warn}/{n}", "pass": bool(n_warn == n)},
        {"id": 7, "name": "checkpoint 只按 validation 选", "need": f"{n}/{n}",
         "got": f"{n_valonly}/{n}", "pass": bool(n_valonly == n)},
    ]
    n_pass = sum(c["pass"] for c in conds)
    verdict = (str(gc["pass_label"]) if n_pass == len(conds)
               else str(gc["fail_label"]))
    return {
        "conditions": conds, "n_conditions": len(conds), "n_passed": n_pass,
        "verdict": verdict,
        "primary_metric": "test_info_trajectory_macro_rmse",
        "per_seed_gain": diffs, "mean_paired_gain": mean_gain,
        "bootstrap": bs,
        "macro_corr_per_seed": corrs,
        "psr_per_seed": [r["shape"]["psr"] for r in rows],
        "n_no_catastrophic_short": n_nocat,
        "excluded_oracle_methods": list(EXCLUDED_ORACLE_METHODS),
        "transfer_run": False,
        "thresholds_frozen_before_run": True,
        "threshold_source": ("docs/basilisk_b21/protocol.md §12 + "
                             "configs/wheel_basilisk_b21.yaml::b21.gate"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG_PATH)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--fast", action="store_true",
                    help="仅冒烟: max_epochs=1 (结果不得用于判定)")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b21 = cfg["b21"]
    gc = b21["gate"]
    seeds = a.seeds if a.seeds else [int(s) for s in b21["seeds"]]
    if a.fast:
        cfg["training"]["max_epochs"] = 1
        cfg["b2_frozen_training_expected"]["max_epochs"] = 1

    sp = ROOT / cfg["paths"]["split_json"]
    split = json.loads(sp.read_text(encoding="utf-8"))
    if str(split.get("verdict")) != "B21_SPLIT_VALID":
        raise SystemExit(f"!! 划分未通过覆盖校验 ({split.get('verdict')}) —— "
                         f"按 §7/§13 停止, 不得在无效划分上跑 Gate")
    axis = b2_endpoint_axis(cfg)
    bins = lifetime_bin_map(split)

    print(f"[B2.1 §9] Gate: seeds={seeds}  主指标={gc['metric'] if 'metric' in gc else b21['primary_metric']}"
          f"  基线={gc['baseline']}")
    print(f"        split_sha256={split['split_sha256']}  "
          f"(B2.1 新划分, 非 B2 划分)")
    print(f"        test lifetime bins: "
          f"{ {n: sum(1 for b in bins.values() if b == n) for n in b21['split']['event']['bins']} }")

    dmg_cache: dict = {}
    rows = [run_seed(cfg, split, axis, s, bins, dmg_cache) for s in seeds]
    dec = decide(rows, gc)

    out = {
        "stage": "BASILISK_B21",
        "section": "§8..§12 split-coverage generalization gate",
        "label": str(b21["label"]),
        "gate_purpose": GATE_PURPOSE,
        "frozen_facts": dict(b21["frozen_facts"]),
        "transfer_run": False,
        "feature_h5": cfg["transfer"]["target_feature_path"],
        "hi_key": cfg["transfer"]["target_hi_key"],
        "split_sha256": split["split_sha256"],
        "split_seed": split["split_seed"],
        "split_manifest": cfg["paths"]["split_json"],
        "split_verdict": str(split["verdict"]),
        "methods": list(METHODS),
        "seeds": seeds,
        "fast_smoke": bool(a.fast),
        "frozen_training": cfg["b2_frozen_training_expected"],
        "frozen_training_source": "configs/wheel_basilisk_b2.yaml (§8 复用)",
        "lifetime_subset_note": LIFETIME_SUBSET_NOTE,
        "censored_lower_bound_note": CENSORED_LB_NOTE,
        "catastrophic_definition": dict(b21["catastrophic"]),
        "test_lifetime_bin_counts": {
            n: sum(1 for b in bins.values() if b == n)
            for n in b21["split"]["event"]["bins"]},
        "per_seed": rows,
        "decision": dec,
        "verdict": dec["verdict"],
    }
    p = ROOT / cfg["paths"]["metrics_json"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                            default=float) + "\n", encoding="utf-8")

    print(f"\n{'=' * 66}\n[B2.1 §12] 判定\n{'=' * 66}")
    for c in dec["conditions"]:
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  条件{c['id']} "
              f"{c['name']}: 需 {c['need']}, 实得 {c['got']}")
    bs = dec["bootstrap"]
    print(f"  mean paired gain = {dec['mean_paired_gain']:+.6f}  "
          f"95% CI [{bs['ci_lower']:+.6f}, {bs['ci_upper']:+.6f}]  "
          f"(n={bs['n']} seed 级配对, {bs['n_rep']} 次重采样)")
    print(f"  -> {dec['verdict']}  ({dec['n_passed']}/{dec['n_conditions']})")
    print(f"  metrics -> {p.relative_to(ROOT)}")
    if a.fast:
        print("  !! --fast 冒烟结果, 不得用于正式判定")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
